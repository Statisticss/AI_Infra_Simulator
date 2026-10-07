from dataclasses import replace
import math

import pytest

from ai_infra_simulator.advanced_transport import RecoveryPolicy
from ai_infra_simulator.datacenter import Fabric, flow_seed, simulate_datacenter
from ai_infra_simulator.transport import TransportConfig


def lossless(**kwargs):
    return TransportConfig(data_loss=0, ack_loss=0, path_spread_us=0, window_bdp=8, **kwargs)


def test_single_packet_per_rank_matches_two_serializers_per_direction():
    c = lossless()
    r = simulate_datacenter(c, 4096, ranks=4, workload="permutation")
    # Forward: TX then RX; return ACK: TX then RX. Propagation adds one RTT.
    expected = 2*(4096+128+96)*8/400 + 10_000
    assert r["metrics"]["duration_ms"]*1e6 == pytest.approx(expected)
    assert r["metrics"]["retransmissions"] == 0
    assert r["metrics"]["forward_wire_bytes"] == 4*(4096+128)
    assert r["metrics"]["reverse_wire_bytes"] >= 4*96


def test_data_and_control_never_overlap_on_a_physical_tx_port():
    c = lossless(ack_every=1)
    r = simulate_datacenter(c, 256*4096, ranks=4, workload="permutation", trace=True)
    for trace, nic in zip(r["nic_traces"], r["nics"]):
        assert {f["kind"] for f in trace} == {"ack", "data"}
        assert all(a["end_ns"] <= b["start_ns"]+1e-6 for a, b in zip(trace, trace[1:]))
        assert sum(f["bytes"] for f in trace) == nic["data_tx_bytes"]+nic["ack_tx_bytes"]
        for frame in trace:
            assert frame["end_ns"]-frame["start_ns"] == pytest.approx(frame["bytes"]*8/c.bandwidth_gbps)
            assert frame["source"] == nic["rank"]
            assert frame["destination"] == (nic["rank"]+(1 if frame["kind"] == "data" else -1)) % 4


def test_ingress_fifo_is_finite_rate_and_bounded():
    c = lossless()
    f = Fabric(c, RecoveryPolicy(), 4096, 2, "permutation", 1, queue_capacity_bytes=9000)
    delivered = []
    f.nics[0].receive_frame(4224, lambda: delivered.append(f.now), ())
    f.nics[0].receive_frame(4224, lambda: delivered.append(f.now), ())
    while f.events:
        import heapq
        f.now, _, callback, args = heapq.heappop(f.events)
        callback(*args)
    assert delivered == pytest.approx([4224*8/400, 2*4224*8/400])
    assert f.nics[0].rx_queued_bytes == 0
    assert f.nics[0].max_rx_queued_bytes == 8448
    f.nics[0].receive_frame(4224, lambda: None, ())
    f.nics[0].receive_frame(4224, lambda: None, ())
    with pytest.raises(RuntimeError, match="buffer exceeded"):
        f.nics[0].receive_frame(4224, lambda: None, ())


def test_ring_keeps_global_clock_and_nic_state_between_all_phases():
    c = lossless()
    r = simulate_datacenter(c, 2**18, ranks=4, trace=True)
    m = r["metrics"]
    assert m["steps"] == 6 and len(r["flows"]) == 24
    assert m["total_unique_payload_bytes"] == 2**18*6
    assert sum(r["phase_durations_ns"]) == pytest.approx(m["final_confirmation_ms"]*1e6)
    ends = [max(f["completion_ns"] for f in r["flows"] if f["phase"] == phase) for phase in range(6)]
    assert all(a < b for a, b in zip(ends, ends[1:]))
    for trace in r["nic_traces"]:
        assert all(a["end_ns"] <= b["start_ns"]+1e-6 for a, b in zip(trace, trace[1:]))
        for frame in trace:
            if frame["kind"] == "data" and frame["phase"] > 0:
                assert frame["start_ns"] >= ends[frame["phase"]-1]
    assert m["bus_gbps"] == pytest.approx(m["algorithm_gbps"]*1.5)


def test_tail_and_redundant_repair_losses_recover_without_an_oracle():
    c = lossless(early_factor=1.1, rto_factor=1.25)
    r = simulate_datacenter(c, 32*4096, ranks=2, workload="permutation", trace=True,
                            data_drop_hook=lambda seq, attempt: seq == 31 and attempt <= 4,
                            ack_drop_hook=lambda ordinal: ordinal <= 3)
    assert r["metrics"]["physical_drops"] == 8
    assert r["metrics"]["tail_redundant_transmissions"] >= 4
    assert r["metrics"]["ack_drops"] == 6
    for trace in r["flow_traces"]:
        assert min(x["time_ns"] for x in trace if x["event"] == "retransmit") >= c.rtt_us*1000


def test_every_byte_contributes_to_the_goodput_budget():
    c = TransportConfig(window_bdp=8, early_factor=1.1, rto_factor=1.25)
    r = simulate_datacenter(c, 512*4096+19, ranks=4, workload="permutation", seed=11)
    m = r["metrics"]
    assert m["total_unique_payload_bytes"] == 4*(512*4096+19)
    assert math.isclose(m["goodput_pct"], m["wire_payload_efficiency_pct"]*m["mean_nic_tx_utilization_pct"]/100)
    assert m["physical_drops"] > 0 and m["ack_drops"] > 0
    assert all(n["tx_utilization_pct"] <= 100 for n in r["nics"])


def test_seeds_and_reproducibility():
    seeds = [flow_seed(s, phase, rank) for s in (3,7,11,29) for phase in range(14) for rank in range(8)]
    assert len(seeds) == len(set(seeds))
    c = lossless()
    assert simulate_datacenter(c, 4096, ranks=2) == simulate_datacenter(c, 4096, ranks=2)


def test_input_validation_and_budget_exhaustion():
    with pytest.raises(ValueError):
        simulate_datacenter(lossless(), 999, ranks=4)
    with pytest.raises(ValueError):
        simulate_datacenter(replace(lossless(), window_packets=32641), 4096)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        simulate_datacenter(replace(lossless(), max_events=1), 4096)
