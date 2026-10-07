"""Communication-only, step-synchronous Ring AllReduce on a nonblocking fabric.

Each of 2*(N-1) steps sends S/N bytes concurrently on N independent full-duplex
logical links. The next step waits for all sender completions. This conservative
barrier model is not NCCL's chunk-pipelined kernel schedule or a topology model.
"""

from dataclasses import asdict
from typing import Any

from .transport import TransportConfig, simulate_flow


COUNTERS = ("packets", "data_transmissions", "retransmissions", "physical_drops", "trim_events",
            "out_of_order_discards", "duplicate_deliveries", "ack_transmissions", "ack_drops",
            "nack_transmissions", "early_retransmissions", "timeout_retransmissions",
            "forward_wire_bytes", "reverse_wire_bytes", "event_count")


def simulate_workload(config: TransportConfig, size_bytes: int, seed: int,
                      workload: str = "bulk", ranks: int = 8) -> dict[str, Any]:
    if workload not in {"bulk", "ring_allreduce"}:
        raise ValueError("workload must be bulk or ring_allreduce")
    if workload == "ring_allreduce" and (ranks < 2 or size_bytes % ranks):
        raise ValueError("Ring AllReduce needs >=2 ranks and a tensor size divisible by ranks")
    steps = 1 if workload == "bulk" else 2 * (ranks - 1)
    links = 1 if workload == "bulk" else ranks
    chunk_bytes = size_bytes // links
    counters = dict.fromkeys(COUNTERS, 0)
    phases = []
    flows = []
    max_ooo = 0
    # With zero loss, paths and ACK generation are deterministic. Reusing an
    # identical flow saves work without sampling or extrapolating a lossy run.
    deterministic = config.data_loss == config.ack_loss == config.trim_probability == 0
    cached = None
    for step in range(steps):
        durations = []
        for rank in range(links):
            flow_seed = seed * 1_000_003 + step * 1009 + rank * 7919
            result = cached if deterministic and cached is not None else simulate_flow(config, chunk_bytes, flow_seed)
            if deterministic:
                cached = result
            row = asdict(result)
            row.update(step=step, rank=rank, seed=flow_seed)
            flows.append(row)
            durations.append(result.duration_ns)
            max_ooo = max(max_ooo, result.max_receiver_ooo_packets)
            for name in COUNTERS:
                counters[name] += getattr(result, name)
        phases.append(max(durations))
    duration = sum(phases)
    per_rank_bytes = chunk_bytes * steps
    # bytes * 8 / ns is Gbit/s; a bit/ns is exactly a Gbit/s.
    algorithm_gbps = size_bytes * 8 / duration
    bus_gbps = per_rank_bytes * 8 / duration
    result = dict(duration_ms=duration / 1e6, algorithm_gbps=algorithm_gbps, bus_gbps=bus_gbps,
                  goodput_pct=bus_gbps / config.bandwidth_gbps * 100,
                  payload_per_rank_bytes=per_rank_bytes, total_unique_payload_bytes=per_rank_bytes * links,
                  payload_size_bytes=size_bytes, chunk_bytes=chunk_bytes, ranks=links, steps=steps,
                  window_packets=config.window, max_receiver_ooo_packets=max_ooo,
                  max_receiver_ooo_mib=max_ooo * config.payload_bytes / 1024**2,
                  phase_durations_ms=[t / 1e6 for t in phases], **counters)
    result["observed_data_loss_pct"] = 100 * counters["physical_drops"] / counters["data_transmissions"]
    result["observed_ack_loss_pct"] = 100 * counters["ack_drops"] / counters["ack_transmissions"]
    result["transmission_amplification"] = counters["data_transmissions"] / counters["packets"]
    result["wire_payload_efficiency_pct"] = 100 * result["total_unique_payload_bytes"] / counters["forward_wire_bytes"]
    result["feedback_to_forward_pct"] = 100 * counters["reverse_wire_bytes"] / counters["forward_wire_bytes"]
    result["asymptotic_erasure_ceiling_pct"] = (100 * (1-config.data_loss) * (1-config.trim_probability)
                                             * config.payload_bytes / (config.payload_bytes + config.overhead_bytes))
    return {"metrics": result, "flows": flows}
