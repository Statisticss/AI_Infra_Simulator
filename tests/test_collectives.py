from dataclasses import replace

import pytest

from ai_infra_simulator.collectives import simulate_workload
from ai_infra_simulator.transport import PacketSimulation, TransportConfig


def test_ring_counts_both_phases_and_all_ranks():
    config = TransportConfig(data_loss=0, ack_loss=0, path_count=1, path_spread_us=0, window_packets=1024)
    result = simulate_workload(config, 8 * 32 * 4096, seed=1, workload="ring_allreduce", ranks=8)
    metrics = result["metrics"]
    assert len(result["flows"]) == 8 * 14
    assert metrics["total_unique_payload_bytes"] == 2 * 7 * (8 * 32 * 4096)
    assert metrics["bus_gbps"] == pytest.approx(metrics["algorithm_gbps"] * 1.75)
    assert metrics["duration_ms"] == pytest.approx(sum(metrics["phase_durations_ms"]))
    assert metrics["goodput_pct"] < metrics["asymptotic_erasure_ceiling_pct"]


def test_ring_lossy_step_waits_for_slowest_rank():
    result = simulate_workload(TransportConfig(), 8 * 64 * 4096, seed=29,
                               workload="ring_allreduce", ranks=8)
    for step, time_ms in enumerate(result["metrics"]["phase_durations_ms"]):
        assert time_ms == max(flow["duration_ns"] for flow in result["flows"] if flow["step"] == step) / 1e6


class UncoalescedGBN(PacketSimulation):
    """Reference implementation with an explicit event for every packet arrival."""
    def receive_data(self, seq, attempt, arrival=None):
        if arrival is not None:
            self.schedule(arrival, self.receive_data, seq, attempt)
        else:
            super().receive_data(seq, attempt)


@pytest.mark.parametrize("seed", [11, 29, 47])
def test_gbn_event_coalescing_is_equivalent_to_explicit_packet_events(seed):
    config = TransportConfig(protocol="gbn", window_packets=256)
    fast = PacketSimulation(config, 2048 * 4096, seed).run().as_dict()
    reference = UncoalescedGBN(config, 2048 * 4096, seed).run().as_dict()
    fast.pop("event_count")
    reference.pop("event_count")
    assert fast == reference


def test_packet_size_sets_nontrivial_goodput_bound():
    result = simulate_workload(TransportConfig(payload_bytes=1024), 256 * 1024, seed=47)
    assert result["metrics"]["asymptotic_erasure_ceiling_pct"] == pytest.approx(95 * 1024 / 1152)
    assert result["metrics"]["asymptotic_erasure_ceiling_pct"] < 90
