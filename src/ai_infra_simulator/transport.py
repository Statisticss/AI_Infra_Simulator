"""Packet-event loss-recovery model; all simulation times are in nanoseconds.

This is a transport-mechanism model, not a wire-compatible RoCE/UET stack.
The forward and feedback directions each have a finite-rate serializer.
There are no switch queues, congestion drops, PFC, or congestion controllers.
"""

# 分组级传输底座：对比按序接收的 GBN 与允许乱序的选择性重传。
# 内核统一使用 ns；输入 RTT 使用 us，链路速率使用 Gbit/s。
# 本文件将正向数据和反向反馈分别序列化；实验 004 在子类中接入共享物理 NIC。
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import heapq
import math
import random
from typing import Callable


@dataclass(frozen=True)
class TransportConfig:
    # 参数对象不可变，便于多个试验共享配置而不意外改变彼此的协议行为。
    # overhead_bytes 是统一的线速开销预算，不代表某个协议的精确头部长度。
    protocol: str = "uet"
    bandwidth_gbps: float = 400.0
    rtt_us: float = 10.0  # propagation + fixed NIC processing, excluding serialization
    payload_bytes: int = 4096
    overhead_bytes: int = 128  # equalized total on-wire overhead, not a UET header claim
    ack_bytes: int = 96
    ack_every: int = 4
    ack_delay_us: float = 0.5
    # BDP 是带宽与 RTT 的乘积；window_packets 可显式覆盖按 BDP 推导的窗口。
    window_bdp: float = 4.0
    window_packets: int | None = None
    max_window_packets: int = 32640
    # 数据故障、反馈故障和 Trim 分开注入；重传包同样经历数据故障过程。
    data_loss: float = 0.05
    ack_loss: float = 0.05
    loss_model: str = "iid"
    burst_length: float = 8.0
    trim_probability: float = 0.0  # conditional on surviving physical loss
    # 多路径只引入确定性的逐包喷洒和传播时延差，不额外增加端口带宽。
    path_count: int = 4
    path_spread_us: float = 2.0  # total spread of one-way propagation times
    early_recovery: bool = True
    early_factor: float = 1.5
    rto_factor: float = 4.0
    rto_us: float | None = None
    nack_interval_rtt: float = 1.0
    max_events: int = 100_000_000
    max_time_s: float = 3600.0

    def __post_init__(self):
        # 先验证概率、时间、包数等输入，避免无效参数生成貌似合理的结果。
        if self.protocol not in {"gbn", "uet"}:
            raise ValueError("protocol must be gbn or uet")
        if self.loss_model not in {"iid", "burst"}:
            raise ValueError("loss_model must be iid or burst")
        for name in ("data_loss", "ack_loss", "trim_probability"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("bandwidth_gbps", "rtt_us", "payload_bytes", "ack_bytes", "ack_every",
                     "window_bdp", "max_window_packets", "path_count", "early_factor",
                     "rto_factor", "max_events", "max_time_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("payload_bytes", "ack_bytes", "ack_every", "max_window_packets", "path_count", "max_events"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.overhead_bytes < 0 or self.ack_delay_us < 0 or self.nack_interval_rtt < 0:
            raise ValueError("overhead and delay parameters must be nonnegative")
        if not 0 <= self.path_spread_us < self.rtt_us:
            raise ValueError("path_spread_us must be nonnegative and smaller than RTT")
        if self.window_packets is not None and (not isinstance(self.window_packets, int) or self.window_packets < 1):
            raise ValueError("window_packets must be a positive integer")
        if self.rto_us is not None and self.rto_us <= 0:
            raise ValueError("rto_us must be positive")
        if self.burst_length < 1:
            raise ValueError("burst_length must be at least one packet")
        if self.loss_model == "burst" and self.data_loss / ((1 - self.data_loss) * self.burst_length) > 1:
            raise ValueError("burst probability and length imply an invalid transition probability")

    @property
    def window(self) -> int:
        if self.window_packets is not None:
            return self.window_packets
        # 一个在途包包含 payload 和线速开销；向上取整后再受单连接跨度上限约束。
        bdp_packets = self.bandwidth_gbps * self.rtt_us * 1000 / (8 * (self.payload_bytes + self.overhead_bytes))
        return min(self.max_window_packets, max(1, math.ceil(self.window_bdp * bdp_packets)))

    @property
    def rto_ns(self) -> float:
        return (self.rto_us if self.rto_us is not None else self.rtt_us * self.rto_factor) * 1000

    @property
    def early_ns(self) -> float:
        # 早期重传的等待必须覆盖路径差和 ACK 合并等待，避免把正常乱序当作丢包。
        # Only positive ACK evidence plus an elapsed guard may trigger early retransmission.
        return max(self.rtt_us * self.early_factor,
                   self.rtt_us + self.path_spread_us + self.ack_delay_us) * 1000


class PacketLoss:
    """Independent erasure, or a stationary two-state all-good/all-bad process.

    Burst length is measured in attempted data packets, NOT elapsed time.
    This burst model is not a model of a persistent physical link outage.
    """

    def __init__(self, probability: float, seed: int, mode: str = "iid", burst_length: float = 8.0):
        # 突发模型以“发送尝试次数”为步长；它不等价于持续若干微秒的链路中断。
        # 两个转移概率使稳态坏状态比例等于指定丢包率。
        self.p = probability
        self.rng = random.Random(seed)
        self.mode = mode
        self.bad = self.rng.random() < probability
        self.bad_to_good = 1 / burst_length
        self.good_to_bad = probability / ((1 - probability) * burst_length) if probability else 0

    def drop(self) -> bool:
        if self.p == 0:
            return False
        if self.mode == "iid":
            return self.rng.random() < self.p
        dropped = self.bad
        if self.bad:
            self.bad = not (self.rng.random() < self.bad_to_good)
        else:
            self.bad = self.rng.random() < self.good_to_bad
        return dropped


@dataclass
class FlowResult:
    # duration_ns 包含发送端最终确认；receiver_complete_ns 仅表示接收端收齐。
    # 唯一有效字节与实际线速字节分别保存，计算 Goodput 时不能把重传加入分子。
    duration_ns: float
    receiver_complete_ns: float
    payload_bytes: int
    packets: int
    data_transmissions: int
    retransmissions: int
    physical_drops: int
    trim_events: int
    out_of_order_discards: int
    duplicate_deliveries: int
    ack_transmissions: int
    ack_drops: int
    nack_transmissions: int
    early_retransmissions: int
    timeout_retransmissions: int
    forward_wire_bytes: int
    reverse_wire_bytes: int
    max_receiver_ooo_packets: int
    event_count: int
    window_packets: int

    def as_dict(self) -> dict:
        return asdict(self)


class PacketSimulation:
    """One finite transfer, including the final sender acknowledgement.

    Loss hooks exist for deterministic validation and are never consulted by the
    sender. The sender learns only from received feedback and its own timers.
    """

    def __init__(self, config: TransportConfig, size_bytes: int, seed: int = 1,
                 data_drop_hook: Callable[[int, int], bool] | None = None,
                 ack_drop_hook: Callable[[int], bool] | None = None,
                 trace: bool = False):
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ValueError("size_bytes must be a positive integer")
        self.c = config
        self.window = config.window
        self.size = size_bytes
        self.n = math.ceil(size_bytes / config.payload_bytes)
        self.now = 0.0
        self.events: list[tuple] = []
        self.order = 0
        self.event_count = 0
        self.tx_free = 0.0
        self.ack_free = 0.0
        self.tx_pending = False
        self.next_seq = 0
        # 三种边界不能混用：base 是本地正确认前缀，cack 是接收方报告的连续前缀，
        # expected 是接收端实际缺少的第一个 PSN；边界均采用右端不包含的表示。
        self.high_sent = 0  # exclusive
        self.base = 0  # first packet not positively acknowledged by sender
        self.cack = 0  # receiver cumulative ACK reported to sender, exclusive
        self.expected = 0  # first missing at receiver
        # 第 i 位对应 PSN=i。接收位图和发送端已知位图分开，避免发送端预知丢包。
        self.rx_bits = 0
        self.ack_bits = 0
        self.rx_count = 0
        self.max_ooo = 0
        self.max_seen = -1
        self.sack_track = 0
        self.highest_ack = -1
        self.acked_count = 0
        self.attempt = [0] * self.n
        # 每包尝试序号用于识别过期定时器；新一次发送不能被上一次的超时事件误伤。
        self.sent_at = [0.0] * self.n
        self.rto_backoff = [0] * self.n
        self.retx_queue: deque[int] = deque()
        self.retx_queued: set[int] = set()
        self.rx_complete = 0.0
        self.complete = False
        self.ack_count = 0
        self.ack_timer_generation = 0
        self.ack_trigger = -1
        self.last_nack_seq = -1
        self.last_nack_time = -math.inf
        self.last_recovery_seq = -1
        self.last_recovery_time = -math.inf
        self.gbn_timer_generation = 0
        self.data_loss = PacketLoss(config.data_loss, seed * 17 + 1, config.loss_model, config.burst_length)
        # 数据与 ACK 使用不同随机流；测试钩子只用于信道注入，不传给恢复算法。
        self.ack_loss = PacketLoss(config.ack_loss, seed * 17 + 2)
        self.trim_rng = random.Random(seed * 17 + 3)
        self.data_drop_hook = data_drop_hook
        self.ack_drop_hook = ack_drop_hook
        self.trace = [] if trace else None
        self.stats = dict(data_transmissions=0, retransmissions=0, physical_drops=0,
                          trim_events=0, out_of_order_discards=0, duplicate_deliveries=0,
                          ack_transmissions=0, ack_drops=0, nack_transmissions=0,
                          early_retransmissions=0, timeout_retransmissions=0,
                          forward_wire_bytes=0, reverse_wire_bytes=0)

    def log(self, kind: str, seq: int, when: float | None = None):
        if self.trace is not None:
            self.trace.append({"time_ns": self.now if when is None else when, "event": kind, "seq": seq})

    def schedule(self, when: float, callback, *args):
        # 递增序号为同一时刻的事件确定稳定次序，保证固定种子的结果可复现。
        self.order += 1
        heapq.heappush(self.events, (when, self.order, callback, args))

    def wake_tx(self):
        # 同一连接最多挂起一个发送事件，避免多个 ACK 同时唤醒导致重复占用链路。
        if not self.tx_pending and not self.complete:
            self.tx_pending = True
            self.schedule(max(self.now, self.tx_free), self.send)

    def is_acked(self, seq: int) -> bool:
        return bool((self.ack_bits >> seq) & 1)

    def send(self):
        self.tx_pending = False
        if self.complete:
            return
        if self.c.protocol == "gbn":
            seq = max(self.next_seq, self.base)
            if seq >= self.n or seq >= self.base + self.window:
                return
            self.next_seq = seq + 1
        else:
            # 优先修复仍未被确认的包；队列中已被后来 ACK 覆盖的项可以直接跳过。
            while self.retx_queue and self.is_acked(self.retx_queue[0]):
                self.retx_queued.discard(self.retx_queue.popleft())
            if self.retx_queue:
                seq = self.retx_queue.popleft()
                self.retx_queued.discard(seq)
            else:
                seq = self.next_seq
                # 限制的是从 CACK 开始的 PSN 跨度，不能只数未确认包而无限向前发送。
                # Bound the PSN SPAN, not merely the number of unacknowledged packets.
                if seq >= self.n or seq >= self.cack + self.window:
                    return
                self.next_seq += 1
        self.high_sent = max(self.high_sent, seq + 1)
        self.attempt[seq] += 1
        attempt = self.attempt[seq]
        self.sent_at[seq] = self.now
        self.stats["data_transmissions"] += 1
        self.stats["retransmissions"] += int(attempt > 1)
        # 包括尾部不足一个 payload 的情况；所有发送尝试先支付完整线速字节。
        length = min(self.c.payload_bytes, self.size - seq * self.c.payload_bytes)
        wire = length + self.c.overhead_bytes
        self.stats["forward_wire_bytes"] += wire
        self.tx_free = self.now + wire * 8 / self.c.bandwidth_gbps
        self.log("retransmit" if attempt > 1 else "send", seq)
        if self.c.protocol == "uet" and self.c.path_count > 1:
            path = (self.stats["data_transmissions"] - 1) % self.c.path_count
            offset = (path / (self.c.path_count - 1) - 0.5) * self.c.path_spread_us * 1000
        else:
            offset = 0.0
        arrival = self.tx_free + self.c.rtt_us * 500 + offset
        # 先判定物理擦除，只有存活的数据帧才有机会被 Trim；擦除不产生免费通知。
        dropped = self.data_drop_hook(seq, attempt) if self.data_drop_hook else self.data_loss.drop()
        if dropped:
            self.stats["physical_drops"] += 1
            self.log("physical_drop", seq, arrival)
        elif self.c.trim_probability and self.trim_rng.random() < self.c.trim_probability:
            self.stats["trim_events"] += 1
            self.log("trim", seq, arrival)
            if self.c.protocol == "uet":
                self.schedule(arrival, self.receive_trim, seq)
        elif self.c.protocol == "gbn":
            # Exact event coalescing: a fixed, ordered path preserves transmission
            # order, so its receiver can be advanced to this future arrival now.
            # No receiver state is exposed to the sender; feedback is still delayed.
            self.receive_data(seq, attempt, arrival)
        else:
            self.schedule(arrival, self.receive_data, seq, attempt)
        if self.c.protocol == "uet":
            # 定时器随每次尝试生成；回调会核对尝试序号及 ACK 状态后再决定恢复。
            if self.c.early_recovery:
                self.schedule(self.now + self.c.early_ns, self.early_timeout, seq, attempt)
            self.schedule(self.now + self.c.rto_ns * (2 ** min(self.rto_backoff[seq], 3)),
                          self.uet_timeout, seq, attempt)
        elif seq == self.base:
            self.arm_gbn_timer()
        self.wake_tx()

    def receive_trim(self, seq: int):
        # A physical erasure never calls this method.
        self.send_feedback(self.now, seq, nack=seq)

    def receive_data(self, seq: int, attempt: int, arrival: float | None = None):
        when = self.now if arrival is None else arrival
        if self.c.protocol == "gbn":
            # 传统按序接收丢弃缺口之后的有效包，因而一次丢包可能引起后缀重传。
            if seq > self.expected:
                self.stats["out_of_order_discards"] += 1
                self.log("ooo_discard", seq, when)
                if (self.last_nack_seq != self.expected or
                    when - self.last_nack_time >= self.c.rtt_us * 1000 * self.c.nack_interval_rtt):
                    self.last_nack_seq, self.last_nack_time = self.expected, when
                    self.send_feedback(when, seq, nack=self.expected)
                return
            if seq < self.expected:
                self.stats["duplicate_deliveries"] += 1
                self.send_feedback(when, seq)
                return
            self.expected += 1
            self.rx_count += 1
        else:
            # 选择性接收保留乱序包；重复到达只触发反馈，不重复累加有效载荷。
            bit = 1 << seq
            if self.rx_bits & bit:
                self.stats["duplicate_deliveries"] += 1
                self.send_feedback(when, seq)
                return
            self.rx_bits |= bit
            self.rx_count += 1
            self.max_seen = max(self.max_seen, seq)
            # 位运算求最低连续 1 的个数，也就是接收端第一个缺口的位置。
            # Count trailing one bits: all lower PSNs have arrived.
            self.expected = (self.rx_bits ^ (self.rx_bits + 1)).bit_length() - 1
            self.max_ooo = max(self.max_ooo, self.rx_count - self.expected)
            if self.sack_track < self.expected:
                self.sack_track = self.expected
            elif self.expected <= seq < self.sack_track:
                self.sack_track = seq
        self.log("deliver", seq, when)
        if self.rx_count == self.n and not self.rx_complete:
            self.rx_complete = when
        self.ack_count += 1
        self.ack_trigger = seq
        # 平时合并 ACK；尾包、重传及全部收齐时立即反馈，缩短操作收尾时间。
        if self.ack_count >= self.c.ack_every or seq == self.n - 1 or attempt > 1 or self.rx_count == self.n:
            self.send_feedback(when, seq)
        elif self.c.protocol == "uet" and self.ack_count == 1:
            # GBN uses an explicit AR on window boundary below, avoiding a
            # speculative delayed-ACK timer in its coalesced receiver path.
            generation = self.ack_timer_generation
            self.schedule(when + self.c.ack_delay_us * 1000, self.delayed_ack, generation)
        elif self.c.protocol == "gbn" and (seq + 1) % self.window == 0:
            self.send_feedback(when, seq)

    def delayed_ack(self, generation: int):
        # 已有即时 ACK 发出后，旧的合并计时器应失效，避免额外重复控制包。
        if generation == self.ack_timer_generation and self.ack_count:
            self.send_feedback(self.now, self.ack_trigger)

    def send_feedback(self, when: float, trigger: int, nack: int | None = None):
        # CACK 确认连续前缀，ACK_PSN 确认触发包，64 位 SACK 补充乱序接收信息。
        # 反馈本身需要序列化且可能丢失；接收端收齐不意味着发送端已经知道。
        self.ack_timer_generation += 1
        self.ack_count = 0
        cack = 0 if self.c.protocol == "uet" and nack is not None else self.expected
        if self.c.protocol == "uet" and nack is None:
            sack_base = max(0, self.sack_track // 8 * 8)
            bitmap = (self.rx_bits >> sack_base) & ((1 << 64) - 1)
            if self.sack_track + 63 < self.max_seen:
                self.sack_track += 64
            ack_psn = trigger
        else:
            sack_base = bitmap = 0
            ack_psn = -1
        self.stats["ack_transmissions"] += 1
        self.stats["nack_transmissions"] += int(nack is not None)
        self.stats["reverse_wire_bytes"] += self.c.ack_bytes
        self.ack_free = max(when, self.ack_free) + self.c.ack_bytes * 8 / self.c.bandwidth_gbps
        dropped = (self.ack_drop_hook(self.stats["ack_transmissions"])
                   if self.ack_drop_hook else self.ack_loss.drop())
        if dropped:
            self.stats["ack_drops"] += 1
            return
        self.schedule(self.ack_free + self.c.rtt_us * 500, self.receive_feedback,
                      cack, ack_psn, sack_base, bitmap, nack)

    def receive_feedback(self, cack: int, ack_psn: int, sack_base: int, bitmap: int, nack: int | None):
        if self.complete:
            return
        old_base = self.base
        self.cack = max(self.cack, cack)
        new_bits = (1 << cack) - 1
        if self.c.protocol == "uet" and nack is None:
            if ack_psn >= 0:
                new_bits |= 1 << ack_psn
            new_bits |= bitmap << sack_base
        newly_acked = new_bits & ~self.ack_bits
        # 只合并正确认；旧 ACK 的零位不撤销新 ACK 已确认的包。
        self.ack_bits |= new_bits  # zero bits MUST NOT clear previous SACK knowledge
        self.acked_count += newly_acked.bit_count()
        if newly_acked:
            self.highest_ack = max(self.highest_ack, newly_acked.bit_length() - 1)
        self.base = (self.ack_bits ^ (self.ack_bits + 1)).bit_length() - 1
        if self.acked_count == self.n:
            self.complete = True
            return
        if self.c.protocol == "gbn":
            if nack is not None and nack >= self.base:
                if (self.last_recovery_seq != self.base or
                    self.now - self.last_recovery_time >= self.c.rtt_us * 1000):
                    self.last_recovery_seq, self.last_recovery_time = self.base, self.now
                    self.next_seq = self.base
            if self.base != old_base:
                self.arm_gbn_timer()
        elif nack is not None and 0 <= nack < self.high_sent and not self.is_acked(nack):
            # Ignore repeated NACKs for a retransmission still in flight.
            if self.attempt[nack] == 1 or self.now - self.sent_at[nack] >= self.c.rtt_us * 1000 - 1e-6:
                self.queue_retransmission(nack, "early_retransmissions")
        self.wake_tx()

    def queue_retransmission(self, seq: int, reason: str):
        if not self.is_acked(seq) and seq not in self.retx_queued:
            self.retx_queued.add(seq)
            self.retx_queue.append(seq)
            self.stats[reason] += 1
            self.wake_tx()

    def early_timeout(self, seq: int, attempt: int):
        # 有更高 PSN 的正确认、等待已到期、当前包仍未确认，才能触发早期修复。
        if not self.is_acked(seq) and self.attempt[seq] == attempt and self.highest_ack > seq:
            self.queue_retransmission(seq, "early_retransmissions")

    def uet_timeout(self, seq: int, attempt: int):
        # 尾包可能没有后续 ACK 作为证据，必须保留超时兜底和有限指数退避。
        if not self.is_acked(seq) and self.attempt[seq] == attempt:
            self.rto_backoff[seq] += 1
            self.queue_retransmission(seq, "timeout_retransmissions")

    def arm_gbn_timer(self):
        self.gbn_timer_generation += 1
        if self.base < self.high_sent:
            when = max(self.now + 1e-6, self.sent_at[self.base] + self.c.rto_ns)
            self.schedule(when, self.gbn_timeout, self.gbn_timer_generation)

    def gbn_timeout(self, generation: int):
        if not self.complete and generation == self.gbn_timer_generation:
            self.stats["timeout_retransmissions"] += 1
            self.next_seq = self.base
            self.wake_tx()

    def run(self) -> FlowResult:
        # 按事件时间推进到发送端最终确认；超出预算时显式报错，不输出部分 Goodput。
        self.wake_tx()
        while self.events and not self.complete:
            when, _, callback, args = heapq.heappop(self.events)
            self.now = when
            self.event_count += 1
            if self.event_count > self.c.max_events or self.now > self.c.max_time_s * 1e9:
                raise RuntimeError("Simulation budget exceeded; do not report an incomplete transfer as goodput")
            callback(*args)
        if not self.complete or self.rx_count != self.n:
            raise RuntimeError("Transfer failed to complete")
        return FlowResult(duration_ns=self.now, receiver_complete_ns=self.rx_complete,
                          payload_bytes=self.size, packets=self.n,
                          max_receiver_ooo_packets=self.max_ooo, event_count=self.event_count,
                          window_packets=self.c.window, **self.stats)


def simulate_flow(config: TransportConfig, size_bytes: int, seed: int = 1) -> FlowResult:
    return PacketSimulation(config, size_bytes, seed).run()
