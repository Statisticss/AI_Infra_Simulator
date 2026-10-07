# 端点扩展验证：多 PDC 共用带宽、尾部副本仍计费、资源边界及随机流独立。
from dataclasses import replace

import pytest

from ai_infra_simulator.advanced_transport import RecoveryPolicy, simulate_port, window_plan, pdc_seed
from ai_infra_simulator.transport import PacketSimulation, TransportConfig


@pytest.mark.parametrize("seed", [3, 29, 47])
def test_one_pdc_baseline_reproduces_original_engine(seed):
    c = TransportConfig(window_bdp=8)
    old = PacketSimulation(c, 2**20, seed).run().as_dict()
    new = simulate_port(c, 2**20, seed, RecoveryPolicy("baseline"))
    for key, value in old.items():
        if key != "event_count":
            assert new[key] == pytest.approx(value), key


@pytest.mark.parametrize("pdcs", [1, 2, 8])
@pytest.mark.parametrize("mode", ["baseline", "coverage"])
def test_stripes_share_one_serializer_and_conserve_partial_payload(pdcs, mode):
    c = TransportConfig(data_loss=0, ack_loss=0, path_spread_us=0)
    size = 1024 * 1024 + 19
    r = simulate_port(c, size, policy=RecoveryPolicy(mode), pdcs=pdcs,
                      total_window_packets=1000, trace=True)
    assert r["payload_bytes"] == size
    assert r["retransmissions"] == 0
    assert r["forward_wire_bytes"] == size + r["packets"] * c.overhead_bytes
    ideal = r["forward_wire_bytes"] * 8 / c.bandwidth_gbps + c.rtt_us * 1000
    assert r["duration_ns"] >= ideal
    assert r["duration_ns"] < ideal + pdcs * c.ack_bytes * 8 / c.bandwidth_gbps + 1e-6
    sends = sorted(x["time_ns"] for t in r["traces"] for x in t if x["event"] == "send")
    assert all(b > a for a, b in zip(sends, sends[1:]))


def test_copies_pay_bytes_and_both_can_be_lost():
    # 故意让首次发送和最初副本连续丢失，确认后续恢复仍能完成且字节全计入。
    c = TransportConfig(data_loss=0, ack_loss=0, path_spread_us=0, rto_factor=1.5)
    # Last original and both initial repairs vanish; next repair round must run.
    r = simulate_port(c, 32 * 4096, policy=RecoveryPolicy(tail_copies=2),
                      data_drop_hook=lambda seq, attempt: seq == 31 and attempt <= 3)
    assert r["physical_drops"] == 3
    assert r["tail_redundant_transmissions"] >= 1
    assert r["data_transmissions"] >= 36
    assert r["duration_ns"] > c.rto_ns * 2


def test_feedback_erasure_and_tail_erasure_both_recover():
    c = TransportConfig(data_loss=0, ack_loss=0)
    r = simulate_port(c, 128 * 4096, policy=RecoveryPolicy(tail_copies=3),
                      data_drop_hook=lambda seq, attempt: seq == 127 and attempt == 1,
                      ack_drop_hook=lambda ordinal: ordinal <= 12)
    assert r["physical_drops"] == 1
    assert r["ack_drops"] == 12
    assert r["goodput_pct"] > 0


def test_loss_feedback_never_precedes_a_round_trip():
    c = TransportConfig(data_loss=0, ack_loss=0, path_spread_us=0)
    r = simulate_port(c, 256 * 4096, trace=True,
                      data_drop_hook=lambda seq, attempt: seq == 0 and attempt == 1)
    repairs = [x for x in r["traces"][0] if x["event"] == "retransmit"]
    assert min(x["time_ns"] for x in repairs) >= c.rtt_us * 1000


def test_reordering_without_loss_has_no_replay_and_sack_offset_is_bounded():
    c = TransportConfig(data_loss=0, ack_loss=0, rtt_us=1000, path_spread_us=20,
                        early_factor=1.1, rto_factor=1.25)
    r = simulate_port(c, 8 * 2**20, pdcs=4, total_window_packets=20000)
    assert r["retransmissions"] == 0
    assert r["max_sack_offset"] <= 32640


def test_window_budget_and_single_pdc_field_limit():
    c = TransportConfig(rtt_us=10000)
    pdcs, budget = window_plan(c, 8)
    assert budget == 2048 * 2**20 // 4096
    assert pdcs == 17
    with pytest.raises(ValueError):
        simulate_port(c, 2**20, total_window_packets=32641)
    with pytest.raises(ValueError):
        simulate_port(c, 2**20, pdcs=2, total_window_packets=1)


def test_seed_reproducibility_and_input_validation():
    c = TransportConfig()
    assert simulate_port(c, 2**18, 31) == simulate_port(c, 2**18, 31)
    for bad in (0, 1.5):
        with pytest.raises(ValueError):
            RecoveryPolicy(tail_copies=bad)


def test_pdcs_and_ranks_do_not_reuse_random_streams():
    seeds = [pdc_seed(11 * 1_000_003 + step * 1009 + rank * 7919, pdc)
             for step in range(14) for rank in range(8) for pdc in range(64)]
    assert len(set(seeds)) == len(seeds)
    assert pdc_seed(11, 0) == 11  # unchanged experiment-001 baseline
