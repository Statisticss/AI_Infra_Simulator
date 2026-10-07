"""Communication-only, step-synchronous Ring AllReduce on a nonblocking fabric.

Each of 2*(N-1) steps sends S/N bytes concurrently on N independent full-duplex
logical links. The next step waits for all sender completions. This conservative
barrier model is not NCCL's chunk-pipelined kernel schedule or a topology model.
"""

# 实验 001 的 collective 封装，只模拟通信阶段依赖，不执行 GPU 张量归约。
# 各逻辑流独立求解、每步取最慢流；实验 004 改用整个 fabric 的持续共享事件时钟。
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
    # Ring 包含 reduce-scatter 和 all-gather，各 N-1 步，每步每 rank 发送 S/N。
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
    # 仅在信道和路径完全确定时复用相同流；有损试验必须实际运行每个随机样本。
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
        # 全局步障碍：下一阶段等待本阶段所有 rank 完成，不能使用平均流时长。
        phases.append(max(durations))
    duration = sum(phases)
    per_rank_bytes = chunk_bytes * steps
    # bytes * 8 / ns is Gbit/s; a bit/ns is exactly a Gbit/s.
    algorithm_gbps = size_bytes * 8 / duration
    # bus 带宽按必要网络 payload 计量；8 rank Ring 的 busbw = algbw × 1.75。
    bus_gbps = per_rank_bytes * 8 / duration
    result = dict(duration_ms=duration / 1e6, algorithm_gbps=algorithm_gbps, bus_gbps=bus_gbps,
                  goodput_pct=bus_gbps / config.bandwidth_gbps * 100,
                  payload_per_rank_bytes=per_rank_bytes, total_unique_payload_bytes=per_rank_bytes * links,
                  payload_size_bytes=size_bytes, chunk_bytes=chunk_bytes, ranks=links, steps=steps,
                  window_packets=config.window, max_receiver_ooo_packets=max_ooo,
                  max_receiver_ooo_mib=max_ooo * config.payload_bytes / 1024**2,
                  phase_durations_ms=[t / 1e6 for t in phases], **counters)
    result["observed_data_loss_pct"] = 100 * counters["physical_drops"] / counters["data_transmissions"]
    # 实际丢包率以全部发送尝试为分母；有限样本会围绕设定概率波动。
    result["observed_ack_loss_pct"] = 100 * counters["ack_drops"] / counters["ack_transmissions"]
    result["transmission_amplification"] = counters["data_transmissions"] / counters["packets"]
    result["wire_payload_efficiency_pct"] = 100 * result["total_unique_payload_bytes"] / counters["forward_wire_bytes"]
    result["feedback_to_forward_pct"] = 100 * counters["reverse_wire_bytes"] / counters["forward_wire_bytes"]
    result["asymptotic_erasure_ceiling_pct"] = (100 * (1-config.data_loss) * (1-config.trim_probability)
                                             * config.payload_bytes / (config.payload_bytes + config.overhead_bytes))
    return {"metrics": result, "flows": flows}
