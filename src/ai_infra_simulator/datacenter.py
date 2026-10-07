"""A nonblocking fabric with shared physical NIC transmit AND receive budgets.

Multiple endpoints run concurrently on one event clock. Each NIC sends its own
data and ACKs for incoming data through the same serializer. All received frames
also traverse a finite-rate ingress serializer and a bounded queue. This closes
the independent-reverse-link approximation of experiments 001--003.
"""
# 实验 004：所有 rank 和 collective 阶段运行在同一个事件时钟上。
# 每个 NIC 只有一个 TX 和一个 RX 串行服务资源；数据与 ACK 都占用这两个资源。
# fabric 内部假定非阻塞，仅显式模拟目的端口的有限接收队列。
from __future__ import annotations

from collections import deque
import hashlib
import heapq
import math

from .advanced_transport import PortConnection, RecoveryPolicy
from .transport import TransportConfig


def flow_seed(seed, phase, rank):
    # 试验种子、阶段、rank 共同确定独立随机流，避免不同连接复用丢包序列。
    key = f"dc:{seed}:{phase}:{rank}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), "big")


class NIC:
    # 一个物理全双工端口：发送和接收各自可并行，但同一方向不能重叠服务两帧。
    def __init__(self, fabric, index):
        self.fabric, self.index = fabric, index
        self.tx_free = self.rx_free = 0.0
        self.data_ready, self.control_ready = deque(), deque()
        # pending 防止重复排入发送事件；计数器区分有效数据方向与控制反馈方向。
        self.pending = False
        self.data_tx_bytes = self.control_tx_bytes = self.rx_bytes = 0
        self.rx_queued_bytes = self.max_rx_queued_bytes = 0
        self.max_control_queue = 0
        self.ooo_packets = self.max_ooo_packets = 0
        self.trace = [] if fabric.trace else None

    def request_data(self, conn):
        # 多次 ACK/定时器唤醒同一连接时，只保留一个待服务请求。
        if not conn.tx_pending and not conn.complete:
            conn.tx_pending = True
            self.data_ready.append(conn)
        self.arm()

    def request_control(self, conn, snapshot):
        # ACK 快照在产生时固定，排队期间不能补入后来才收到的数据确认。
        self.control_ready.append((conn, snapshot))
        self.max_control_queue = max(self.max_control_queue, len(self.control_ready))
        self.arm()

    def arm(self):
        # 发送事件不得早于当前时钟或上一帧发送结束；这就是共享带宽约束。
        if not self.pending and (self.data_ready or self.control_ready):
            self.pending = True
            self.fabric.schedule(max(self.fabric.now, self.tx_free), self.send)

    def send(self):
        self.pending = False
        start = self.fabric.now
        if start < self.tx_free - 1e-6:
            raise AssertionError("NIC transmissions overlap")
        if self.control_ready:
            # ACK 在包边界优先，不能抢占正在发送的数据帧；仍支付完整发送时间。
            conn, snapshot = self.control_ready.popleft()
            size = conn.c.ack_bytes
            self.tx_free = start + size * 8 / conn.c.bandwidth_gbps
            self.control_tx_bytes += size
            conn.emit_control(self.tx_free, snapshot)
            kind = "ack"
        else:
            conn = self.data_ready.popleft()
            conn.tx_pending = False
            # 父类发送函数处理丢包、重传及计时器；字节差额就是本次真实发送成本。
            before = conn.stats["forward_wire_bytes"]
            conn.send()
            size = conn.stats["forward_wire_bytes"] - before
            self.data_tx_bytes += size
            kind = "data"
        if size and self.trace is not None:
            self.trace.append(dict(start_ns=start, end_ns=self.tx_free, kind=kind,
                                   bytes=size, phase=conn.phase, source=self.index,
                                   destination=conn.source if kind == "ack" else conn.destination))
        self.arm()

    def receive_frame(self, size, callback, args):
        # 信道内已经丢失的帧不会到这里；存活帧再次支付最后一跳 RX 序列化。
        # 队列字节包含正在服务的帧；容量溢出使整个场景失败，不静默丢弃或免计费。
        # Loss has already occurred inside the fabric. Surviving data and ACKs
        # both pay this last-hop serialization and compete for the same buffer.
        self.rx_queued_bytes += size
        self.max_rx_queued_bytes = max(self.max_rx_queued_bytes, self.rx_queued_bytes)
        if self.rx_queued_bytes > self.fabric.queue_capacity_bytes:
            raise RuntimeError("Ingress buffer exceeded; this non-congested scenario is invalid")
        start = max(self.fabric.now, self.rx_free)
        self.rx_free = start + size * 8 / self.fabric.c.bandwidth_gbps
        self.rx_bytes += size
        self.fabric.schedule(self.rx_free, self.finish_receive, size, callback, args)

    def finish_receive(self, size, callback, args):
        # 只有整帧接收服务完成，传输状态机才能消费数据或 ACK。
        self.rx_queued_bytes -= size
        callback(*args)


class FabricConnection(PortConnection):
    # 对端点恢复策略的适配层：复用协议，替换时钟、链路资源和反馈发送路径。
    def __init__(self, fabric, policy, config, size, seed, source, destination, phase,
                 data_drop_hook=None, ack_drop_hook=None):
        self.source, self.destination, self.phase = source, destination, phase
        self.initializing = True
        super().__init__(fabric, policy, config, size, seed, data_drop_hook, ack_drop_hook, fabric.trace)
        self.initializing = False

    @property
    def now(self):
        return self.port.now

    @now.setter
    def now(self, value):
        # 基类初始化会写 now=0；新 collective 阶段不能因此重置共享网络时间。
        # Creating a later collective phase must not rewind the shared clock.
        if not self.initializing:
            self.port.now = value

    @property
    def tx_free(self):
        return self.port.nics[self.source].tx_free

    @tx_free.setter
    def tx_free(self, value):
        # 同理，初始化连接不能清空 NIC 中已有的晚到 ACK 或在途发送占用。
        if not self.initializing:
            self.port.nics[self.source].tx_free = value

    def schedule(self, when, callback, *args):
        if callback == self.receive_data:
            # 数据到达接收端口后先进入有限 RX 队列，再回调协议的 receive_data。
            seq = args[0]
            size = min(self.c.payload_bytes, self.size-seq*self.c.payload_bytes) + self.c.overhead_bytes
            self.port.schedule(when, self.port.nics[self.destination].receive_frame, size, callback, args)
        else:
            self.port.schedule(when, callback, *args)

    def wake_tx(self):
        self.port.nics[self.source].request_data(self)

    def receive_data(self, *args):
        # 按接收端汇总尚未连续到达的 payload，衡量乱序状态而不是所有接收字节。
        before = self.rx_count - self.expected
        super().receive_data(*args)
        nic = self.port.nics[self.destination]
        nic.ooo_packets += self.rx_count - self.expected - before
        nic.max_ooo_packets = max(nic.max_ooo_packets, nic.ooo_packets)

    def send_feedback(self, when, trigger, nack=None):
        # 复现两种反馈策略的位图构造，但统一排入真实 NIC，取消独享 ACK 通道。
        if self.policy.name == "coverage" and nack is None:
            self.send_coverage_feedback(when, trigger, max(0, trigger-56)//8*8)
            return
        self.ack_timer_generation += 1
        self.ack_count = 0
        if nack is None:
            base = max(0, self.sack_track//8*8)
            bitmap = (self.rx_bits >> base) & ((1 << 64)-1)
            if self.sack_track+63 < self.max_seen:
                self.sack_track += 64
            snapshot = (self.expected, trigger, base, bitmap, None)
        else:
            snapshot = (0, -1, 0, 0, nack)
        self.port.nics[self.destination].request_control(self, snapshot)

    def send_coverage_feedback(self, when, trigger, sack_base):
        # 较旧区间由 CACK 覆盖；校验 SACK 相对连续前缀的偏移仍可由字段表示。
        self.ack_timer_generation += 1
        self.ack_count = 0
        sack_base = max(sack_base, self.expected//64*64)
        offset = sack_base - self.expected
        if not -32768 <= offset <= 32767:
            raise AssertionError("SACK offset is not encodable")
        self.max_sack_offset = max(self.max_sack_offset, abs(offset))
        snapshot = (self.expected, trigger, sack_base, (self.rx_bits >> sack_base) & ((1 << 64)-1), None)
        self.port.nics[self.destination].request_control(self, snapshot)

    def emit_control(self, finish_ns, snapshot):
        # 到实际发送时才累计 ACK 字节、抽取反馈丢包结果，随后进入对端 RX 队列。
        # Snapshot state was captured when the ACK was generated, not when it
        # left a queue or arrived at the sender. The sender has no receive oracle.
        self.stats["ack_transmissions"] += 1
        self.stats["nack_transmissions"] += int(snapshot[-1] is not None)
        self.stats["reverse_wire_bytes"] += self.c.ack_bytes
        dropped = (self.ack_drop_hook(self.stats["ack_transmissions"])
                   if self.ack_drop_hook else self.ack_loss.drop())
        if dropped:
            self.stats["ack_drops"] += 1
        else:
            self.port.schedule(finish_ns+self.c.rtt_us*500,
                               self.port.nics[self.source].receive_frame,
                               self.c.ack_bytes, self.receive_feedback, snapshot)

    def receive_feedback(self, *args):
        # 只在未完成→完成的状态转移时通知阶段调度器，重复 ACK 不重复完成计数。
        old = self.complete
        super().receive_feedback(*args)
        if self.complete and not old:
            self.port.connection_finished(self)


class Fabric:
    # 非阻塞 fabric 的事件内核及业务调度器；不模拟 GPU 运算或交换机拥塞控制。
    def __init__(self, config, policy, size_bytes, ranks, workload, seed, trace=False,
                 queue_capacity_bytes=256*1024, data_drop_hook=None, ack_drop_hook=None):
        if config.protocol != "uet" or config.trim_probability:
            # 此实验专门观察物理擦除；不能把故障损坏伪装成带包头的 Trim 通知。
            raise ValueError("This study models selective recovery of physical erasures; Trim must be zero")
        if config.window > 32640:
            raise ValueError("A connection's PSN span must fit the modeled field")
        if not isinstance(ranks, int) or ranks < 2:
            raise ValueError("ranks must be an integer >= 2")
        if workload not in {"permutation", "ring_allreduce"}:
            raise ValueError("unknown workload")
        if not isinstance(size_bytes, int) or size_bytes <= 0 or (workload == "ring_allreduce" and size_bytes % ranks):
            raise ValueError("invalid payload or indivisible AllReduce tensor")
        if not isinstance(queue_capacity_bytes, int) or queue_capacity_bytes < config.payload_bytes+config.overhead_bytes:
            raise ValueError("the ingress buffer must fit a frame")
        self.c, self.policy = config, policy
        self.size, self.ranks, self.workload, self.seed = size_bytes, ranks, workload, seed
        self.trace, self.queue_capacity_bytes = trace, queue_capacity_bytes
        self.data_drop_hook, self.ack_drop_hook = data_drop_hook, ack_drop_hook
        self.now = self.order = self.event_count = self.completed = 0
        self.ack_free = 0.0  # unused inherited constructor attribute; ACKs use real NIC queues
        self.events = []
        self.nics = [NIC(self, i) for i in range(ranks)]
        self.connections = []
        self.ooo_packets = self.max_ooo_packets = 0
        self.steps = 1 if workload == "permutation" else 2*(ranks-1)
        # permutation 每 rank 发一份完整消息；Ring 每阶段发张量的 1/N。
        self.chunk = size_bytes if workload == "permutation" else size_bytes//ranks
        self.phase = -1
        self.phase_start = 0.0
        self.phase_completions = 0
        self.phase_durations_ns = []
        self.done = False

    def schedule(self, when, callback, *args):
        # 堆键为时间与稳定序号，禁止逆时序事件；所有 NIC 共用这一个事件堆。
        if when < self.now-1e-5:
            raise AssertionError("event scheduled in the past")
        self.order += 1
        heapq.heappush(self.events, (when, self.order, callback, args))

    def start_phase(self):
        # 阶段启动创建各 rank 的逻辑连接，但 NIC、旧事件和队列持续存在。
        self.phase += 1
        self.phase_start = self.now
        self.phase_completions = 0
        for rank in range(self.ranks):
            conn = FabricConnection(self, self.policy, self.c, self.chunk,
                                    flow_seed(self.seed, self.phase, rank), rank, (rank+1)%self.ranks,
                                    self.phase, self.data_drop_hook, self.ack_drop_hook)
            self.connections.append(conn)
            conn.wake_tx()

    def connection_finished(self, conn):
        assert conn.phase == self.phase
        self.phase_completions += 1
        if self.phase_completions == self.ranks:
            # 全部发送端最终确认后才进入下一阶段；阶段时长由最慢 rank 决定。
            self.phase_durations_ns.append(self.now-self.phase_start)
            if self.phase+1 == self.steps:
                self.done = True
            else:
                # New traffic shares NICs and queues with any late control frames
                # from preceding phases. No serializer or event state is reset.
                self.start_phase()

    def run(self):
        self.start_phase()
        while self.events and not self.done:
            self.now, _, callback, args = heapq.heappop(self.events)
            self.event_count += 1
            if self.event_count > self.c.max_events or self.now > self.c.max_time_s*1e9:
                raise RuntimeError("Simulation budget exceeded; no partial result")
            callback(*args)
        if not self.done or not all(c.complete and c.rx_count == c.n for c in self.connections):
            raise RuntimeError("Transfer incomplete")
        duration = max(self.now, *(n.tx_free for n in self.nics), *(n.rx_free for n in self.nics))
        # 最终 ACK 之后已预留的尾部序列化也计入分母，不截去收尾成本。
        counters = {key: sum(c.stats[key] for c in self.connections) for key in self.connections[0].stats}
        packets = sum(c.n for c in self.connections)
        per_rank_bytes = self.chunk*self.steps
        data_wire = counters["forward_wire_bytes"]
        ack_wire = counters["reverse_wire_bytes"]
        total_wire = data_wire+ack_wire
        nic_rows = []
        for nic in self.nics:
            tx_wire = nic.data_tx_bytes+nic.control_tx_bytes
            # 用每端口字节账本验证发送/接收速率上限，避免多个流“各得一条 400G”。
            assert tx_wire*8/self.c.bandwidth_gbps <= duration+1e-3
            assert nic.rx_bytes*8/self.c.bandwidth_gbps <= duration+1e-3
            nic_rows.append(dict(rank=nic.index, data_tx_bytes=nic.data_tx_bytes,
                                 ack_tx_bytes=nic.control_tx_bytes, rx_bytes=nic.rx_bytes,
                                 tx_utilization_pct=tx_wire*8/duration/self.c.bandwidth_gbps*100,
                                 max_ingress_queue_bytes=nic.max_rx_queued_bytes,
                                 max_control_queue_frames=nic.max_control_queue,
                                 max_ooo_payload_bytes=nic.max_ooo_packets*self.c.payload_bytes))
        goodput = per_rank_bytes*8/duration/self.c.bandwidth_gbps*100
        # 恒等式：Goodput = payload/全部 TX 字节 × 平均 NIC TX 利用率。
        # ACK、包头、重传和尾部副本均进入 TX 字节；唯一 payload 不重复计数。
        wire_efficiency = per_rank_bytes*self.ranks/total_wire*100
        mean_utilization = sum(n["tx_utilization_pct"] for n in nic_rows)/self.ranks
        assert math.isclose(goodput, wire_efficiency*mean_utilization/100, rel_tol=1e-12)
        assert sum(n["data_tx_bytes"] for n in nic_rows) == data_wire
        assert sum(n["ack_tx_bytes"] for n in nic_rows) == ack_wire
        assert packets+counters["retransmissions"] == counters["data_transmissions"]
        assert packets+counters["physical_drops"] <= counters["data_transmissions"]
        result = dict(workload=self.workload, ranks=self.ranks, steps=self.steps,
                      tensor_bytes_per_rank=self.size, payload_bytes_per_rank=per_rank_bytes,
                      total_unique_payload_bytes=per_rank_bytes*self.ranks,
                      duration_ms=duration/1e6, final_confirmation_ms=self.now/1e6,
                      goodput_pct=goodput, bus_gbps=goodput*self.c.bandwidth_gbps/100,
                      algorithm_gbps=self.size*8/duration,
                      data_amplification=counters["data_transmissions"]/packets,
                      observed_data_loss_pct=100*counters["physical_drops"]/counters["data_transmissions"],
                      observed_ack_loss_pct=100*counters["ack_drops"]/counters["ack_transmissions"],
                      feedback_to_data_pct=100*ack_wire/data_wire,
                      wire_payload_efficiency_pct=wire_efficiency,
                      mean_nic_tx_utilization_pct=mean_utilization,
                      window_packets_per_connection=self.c.window,
                      window_payload_mib_per_connection=self.c.window*self.c.payload_bytes/2**20,
                      max_receiver_ooo_mib=max(n["max_ooo_payload_bytes"] for n in nic_rows)/2**20,
                      max_ingress_queue_bytes=max(n["max_ingress_queue_bytes"] for n in nic_rows),
                      tail_redundant_transmissions=sum(c.tail_redundant_transmissions for c in self.connections),
                      closure_ack_transmissions=sum(c.closure_ack_transmissions for c in self.connections),
                      packets=packets, event_count=self.event_count, **counters)
        flows = [dict(phase=c.phase, rank=c.source, destination=c.destination,
                      # 保存逐流最终状态，允许报告审计每个包收齐、确认及阶段时间。
                      seed=flow_seed(self.seed, c.phase, c.source), completion_ns=c.completion_ns,
                      receiver_complete_ns=c.rx_complete, received_packets=c.rx_count,
                      acknowledged_packets=c.acked_count, max_sack_offset=c.max_sack_offset,
                      max_receiver_ooo_packets=c.max_ooo, window_packets=c.window,
                      payload_bytes=c.size, packets=c.n, **c.stats) for c in self.connections]
        output = dict(metrics=result, nics=nic_rows, flows=flows, phase_durations_ns=self.phase_durations_ns)
        if self.trace:
            output["nic_traces"] = [n.trace for n in self.nics]
            output["flow_traces"] = [c.trace for c in self.connections]
        return output


def simulate_datacenter(config, size_bytes, ranks=8, workload="ring_allreduce", seed=1,
                        policy=None, **kwargs):
    # 对外入口；默认策略只保护末尾需要恢复的包，不给整条数据流做三倍复制。
    return Fabric(config, policy or RecoveryPolicy(tail_copies=3, tail_packets=32640),
                  size_bytes, ranks, workload, seed, **kwargs).run()
