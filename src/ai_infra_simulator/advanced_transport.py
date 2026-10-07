"""Endpoint recovery experiments on one shared, full-duplex physical port.

This module preserves the experiment-001 transport as its baseline.  The new
policy is a research candidate, NOT a reconstruction of proprietary CIPU logic
and NOT a complete UET implementation.  A PDC always has a bounded PSN span;
adding PDCs buys state, never additional port bandwidth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import heapq
import hashlib
import math
from typing import Callable

from .transport import PacketSimulation, TransportConfig


@dataclass(frozen=True)
class RecoveryPolicy:
    name: str = "coverage"
    closure_copies: int = 2
    tail_copies: int = 2
    tail_packets: int = 64

    def __post_init__(self):
        if self.name not in {"baseline", "coverage"}:
            raise ValueError("unknown recovery policy")
        for key in ("closure_copies", "tail_copies", "tail_packets"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")


def pdc_seed(flow_seed: int, index: int) -> int:
    # Preserve the original one-PDC stream. Additional PDCs use a separate hash
    # namespace: adding rank_stride again would alias rank 0/PDC 1 and rank 1/PDC 0.
    if index == 0:
        return flow_seed
    return int.from_bytes(hashlib.blake2b(f"pdc:{flow_seed}:{index}".encode(), digest_size=16).digest(), "big")


class SharedPort:
    """An exact packet serializer and a fair, work-conserving PDC arbiter."""

    def __init__(self):
        self.now = self.tx_free = self.ack_free = 0.0
        self.events = []
        self.order = self.event_count = 0
        self.ready = deque()
        self.tx_pending = False
        self.connections: list[PortConnection] = []
        self.completed = 0
        self.ooo_packets = self.max_ooo_packets = 0

    def schedule(self, when, callback, *args):
        if when < self.now - 1e-5:
            raise AssertionError("an event cannot be scheduled in the past")
        self.order += 1
        heapq.heappush(self.events, (when, self.order, callback, args))

    def request_tx(self, conn):
        if not conn.tx_pending and not conn.complete:
            conn.tx_pending = True
            self.ready.append(conn)
        self.arm_tx()

    def arm_tx(self):
        if self.ready and not self.tx_pending:
            self.tx_pending = True
            self.schedule(max(self.now, self.tx_free), self.send)

    def send(self):
        self.tx_pending = False
        conn = self.ready.popleft()
        conn.tx_pending = False
        conn.send()
        self.arm_tx()

    def run(self, config):
        for conn in self.connections:
            conn.wake_tx()
        while self.events and self.completed < len(self.connections):
            self.now, _, callback, args = heapq.heappop(self.events)
            self.event_count += 1
            if self.event_count > config.max_events or self.now > config.max_time_s * 1e9:
                raise RuntimeError("Simulation budget exceeded; no partial goodput is reported")
            callback(*args)
        if not all(c.complete and c.rx_count == c.n for c in self.connections):
            raise RuntimeError("Transfer did not complete")


class PortConnection(PacketSimulation):
    def __init__(self, port, policy, *args, **kwargs):
        self.port, self.policy = port, policy
        super().__init__(*args, **kwargs)
        self.last_sent = -1
        self.closed_through = -1
        self.tail_copy_left = {}
        self.tail_redundant_transmissions = 0
        self.closure_ack_transmissions = 0
        self.max_sack_offset = 0
        self.completion_ns = 0.0

    @property
    def now(self):
        return self.port.now

    @now.setter
    def now(self, value):
        self.port.now = value

    @property
    def tx_free(self):
        return self.port.tx_free

    @tx_free.setter
    def tx_free(self, value):
        self.port.tx_free = value

    @property
    def ack_free(self):
        return self.port.ack_free

    @ack_free.setter
    def ack_free(self, value):
        self.port.ack_free = value

    def schedule(self, when, callback, *args):
        self.port.schedule(when, callback, *args)

    def wake_tx(self):
        self.port.request_tx(self)

    def log(self, kind, seq, when=None):
        if kind in {"send", "retransmit"}:
            self.last_sent = seq
        super().log(kind, seq, when)

    def send(self):
        before = self.stats["data_transmissions"]
        super().send()
        if self.policy.name == "baseline" or self.stats["data_transmissions"] == before:
            return
        seq = self.last_sent
        remaining = self.tail_copy_left.get(seq, 0)
        if remaining:
            # This copy and every other copy pay serialization and can be lost.
            self.tail_redundant_transmissions += 1
            remaining -= 1
        elif (self.policy.tail_copies > 1 and self.attempt[seq] > 1
              and self.next_seq == self.n
              and self.n - self.acked_count <= self.policy.tail_packets):
            remaining = self.policy.tail_copies - 1
        self.tail_copy_left[seq] = remaining
        if remaining and not self.is_acked(seq):
            self.retx_queued.add(seq)
            self.retx_queue.append(seq)
            self.wake_tx()

    def receive_data(self, seq, attempt, arrival=None):
        old_ooo = self.rx_count - self.expected
        super().receive_data(seq, attempt, arrival)
        self.port.ooo_packets += self.rx_count - self.expected - old_ooo
        self.port.max_ooo_packets = max(self.port.max_ooo_packets, self.port.ooo_packets)
        if self.policy.name == "baseline":
            return
        # Ordinary ACKs overlap the preceding 56 PSNs. Once a higher block
        # arrives, explicitly repeat the earlier block after the reordering guard.
        # This is positive SACK coverage, not an oracle declaring zero bits lost.
        last_closed = self.max_seen // 64 - 1
        if seq == self.n - 1:
            last_closed = seq // 64
        if last_closed > self.closed_through:
            guard = (self.c.path_spread_us + self.c.ack_delay_us) * 1000
            for block in range(self.closed_through + 1, last_closed + 1):
                for copy in range(self.policy.closure_copies):
                    self.schedule(self.now + guard * (copy + 1), self.closure_ack, block * 64)
            self.closed_through = last_closed

    def closure_ack(self, sack_base):
        if self.complete:
            return
        self.closure_ack_transmissions += 1
        self.send_coverage_feedback(self.now, self.max_seen, sack_base)

    def send_feedback(self, when, trigger, nack=None):
        if self.policy.name == "baseline" or nack is not None:
            super().send_feedback(when, trigger, nack)
        else:
            self.send_coverage_feedback(when, trigger, max(0, trigger - 56) // 8 * 8)

    def send_coverage_feedback(self, when, trigger, sack_base):
        self.ack_timer_generation += 1
        self.ack_count = 0
        # Old blocks are already represented by CACK. Avoid unbounded negative
        # offsets when a delayed closure ACK outlives a cumulative advance.
        sack_base = max(sack_base, self.expected // 64 * 64)
        offset = sack_base - self.expected
        if not -32768 <= offset <= 32767:
            raise AssertionError("SACK offset exceeds the modeled UET field")
        self.max_sack_offset = max(self.max_sack_offset, abs(offset))
        bitmap = (self.rx_bits >> sack_base) & ((1 << 64) - 1)
        self.stats["ack_transmissions"] += 1
        self.stats["reverse_wire_bytes"] += self.c.ack_bytes
        self.ack_free = max(when, self.ack_free) + self.c.ack_bytes * 8 / self.c.bandwidth_gbps
        dropped = (self.ack_drop_hook(self.stats["ack_transmissions"])
                   if self.ack_drop_hook else self.ack_loss.drop())
        if dropped:
            self.stats["ack_drops"] += 1
        else:
            self.schedule(self.ack_free + self.c.rtt_us * 500, self.receive_feedback,
                          self.expected, trigger, sack_base, bitmap, None)

    def receive_feedback(self, *args):
        was_complete = self.complete
        super().receive_feedback(*args)
        if self.complete and not was_complete:
            self.completion_ns = self.now
            self.port.completed += 1


def simulate_port(config: TransportConfig, size_bytes: int, seed: int = 1,
                  policy: RecoveryPolicy | None = None, pdcs: int = 1,
                  total_window_packets: int | None = None,
                  data_drop_hook: Callable[[int, int], bool] | None = None,
                  ack_drop_hook: Callable[[int], bool] | None = None,
                  trace: bool = False) -> dict:
    """Stripe a finite message over bounded PDCs sharing ONE physical port.

    Both policy variants use the same arbiter, wire costs, loss rules and state
    allocation. Test hooks are local PSNs / local ACK ordinals in each PDC.
    """
    if config.protocol != "uet":
        raise ValueError("shared-port experiments model UET RUD mechanisms only")
    if not isinstance(size_bytes, int) or size_bytes < 1:
        raise ValueError("size_bytes must be a positive integer")
    packets = math.ceil(size_bytes / config.payload_bytes)
    if not isinstance(pdcs, int) or pdcs < 1 or pdcs > packets:
        raise ValueError("pdcs must be between one and the message packet count")
    budget = total_window_packets if total_window_packets is not None else config.window * pdcs
    if not isinstance(budget, int) or not pdcs <= budget <= pdcs * 32640:
        raise ValueError("each PDC must receive 1..32640 PSN slots")
    policy = policy or RecoveryPolicy()
    port = SharedPort()
    for index in range(pdcs):
        n = packets // pdcs + int(index < packets % pdcs)
        part_bytes = n * config.payload_bytes
        if index == pdcs - 1:
            part_bytes -= packets * config.payload_bytes - size_bytes
        window = budget // pdcs + int(index < budget % pdcs)
        conn = PortConnection(port, policy, replace(config, window_packets=window),
                              part_bytes, pdc_seed(seed, index), data_drop_hook,
                              ack_drop_hook, trace)
        port.connections.append(conn)
    port.run(config)
    # Drain any already reserved final serialization, including redundant copies.
    duration = max(port.now, port.tx_free)
    stats = {key: sum(c.stats[key] for c in port.connections) for key in port.connections[0].stats}
    goodput = size_bytes * 8 / duration / config.bandwidth_gbps * 100
    result = dict(duration_ns=duration, receiver_complete_ns=max(c.rx_complete for c in port.connections),
                  payload_bytes=size_bytes, packets=packets, goodput_pct=goodput,
                  goodput_gbps=goodput * config.bandwidth_gbps / 100,
                  pdcs=pdcs, window_packets=budget, window_payload_mib=budget * config.payload_bytes / 2**20,
                  max_receiver_ooo_packets=port.max_ooo_packets,
                  max_receiver_ooo_mib=port.max_ooo_packets * config.payload_bytes / 2**20,
                  max_sack_offset=max(c.max_sack_offset for c in port.connections),
                  event_count=port.event_count,
                  tail_redundant_transmissions=sum(c.tail_redundant_transmissions for c in port.connections),
                  closure_ack_transmissions=sum(c.closure_ack_transmissions for c in port.connections),
                  data_amplification=stats["data_transmissions"] / packets,
                  realized_loss_pct=stats["physical_drops"] / stats["data_transmissions"] * 100,
                  feedback_to_data_pct=stats["reverse_wire_bytes"] / stats["forward_wire_bytes"] * 100,
                  asymptotic_ceiling_pct=100 * (1-config.data_loss) * config.payload_bytes /
                                         (config.payload_bytes+config.overhead_bytes),
                  **stats)
    # Conservation checks run in every trial, not only in unit tests.
    assert result["forward_wire_bytes"] * 8 / config.bandwidth_gbps <= duration + 1e-3
    assert result["data_transmissions"] == packets + result["retransmissions"]
    assert result["max_receiver_ooo_packets"] <= budget
    assert result["physical_drops"] + sum(c.rx_count for c in port.connections) <= result["data_transmissions"]
    if trace:
        result["traces"] = [c.trace for c in port.connections]
    return result


def window_plan(config: TransportConfig, bdp_multiplier: float, memory_mib: int = 2048,
                max_pdcs: int = 64) -> tuple[int, int]:
    """Choose minimum PDC count to cover the requested aggregate PSN span."""
    if bdp_multiplier <= 0 or memory_mib <= 0 or max_pdcs < 1:
        raise ValueError("resource limits must be positive")
    bdp = config.bandwidth_gbps * config.rtt_us * 1000 / (8 * (config.payload_bytes + config.overhead_bytes))
    budget = min(math.ceil(bdp_multiplier * bdp), memory_mib * 2**20 // config.payload_bytes,
                 max_pdcs * 32640)
    return math.ceil(budget / 32640), budget
