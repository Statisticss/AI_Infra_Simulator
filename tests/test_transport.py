# 底座协议验证：解析时延、GBN 后缀丢弃、SACK 正确认、反馈丢失和无预知恢复。
from dataclasses import replace
import math

import pytest

from ai_infra_simulator.transport import PacketLoss, PacketSimulation, TransportConfig, simulate_flow


def clean(protocol="uet", **kwargs):
    return replace(TransportConfig(protocol=protocol, data_loss=0, ack_loss=0,
                                   path_count=1, path_spread_us=0, window_packets=256), **kwargs)


@pytest.mark.parametrize("protocol", ["gbn", "uet"])
def test_lossless_matches_serialization_plus_rtt(protocol):
    config = clean(protocol)
    size = 64 * config.payload_bytes + 123
    result = simulate_flow(config, size)
    expected = ((size + math.ceil(size / config.payload_bytes) * config.overhead_bytes
                 + config.ack_bytes) * 8 / config.bandwidth_gbps + config.rtt_us * 1000)
    assert result.duration_ns == pytest.approx(expected)
    assert result.retransmissions == result.physical_drops == 0
    assert result.forward_wire_bytes == size + result.packets * config.overhead_bytes


def test_single_loss_discards_suffix_only_for_gbn():
    results = {}
    for protocol in ["gbn", "uet"]:
        sim = PacketSimulation(clean(protocol), 100 * 4096,
                               data_drop_hook=lambda seq, attempt: seq == 4 and attempt == 1)
        results[protocol] = sim.run()
    assert results["gbn"].out_of_order_discards > 0
    assert results["gbn"].retransmissions > 1
    assert results["uet"].retransmissions == 1
    assert results["uet"].out_of_order_discards == 0
    assert results["uet"].max_receiver_ooo_packets > 0
    assert results["uet"].early_retransmissions == 1


@pytest.mark.parametrize("protocol", ["gbn", "uet"])
def test_lost_last_packet_requires_timeout(protocol):
    result = PacketSimulation(clean(protocol), 16 * 4096,
                              data_drop_hook=lambda seq, attempt: seq == 15 and attempt == 1).run()
    assert result.physical_drops == 1
    assert result.timeout_retransmissions >= 1
    assert result.duration_ns > clean(protocol).rto_ns


@pytest.mark.parametrize("protocol", ["gbn", "uet"])
def test_single_packet_ack_loss_does_not_count_duplicate_payload(protocol):
    result = PacketSimulation(clean(protocol), 4096, ack_drop_hook=lambda ordinal: ordinal == 1).run()
    assert result.ack_drops == 1
    assert result.retransmissions == 1
    assert result.payload_bytes == 4096
    assert result.duplicate_deliveries == 1


def test_retransmissions_can_also_be_lost():
    result = PacketSimulation(clean(), 64 * 4096,
                              data_drop_hook=lambda seq, attempt: seq == 10 and attempt <= 2).run()
    assert result.physical_drops == result.retransmissions == 2


def test_spraying_reorders_without_retransmitting_lossless_data():
    result = simulate_flow(clean(path_count=4, path_spread_us=2, window_packets=512), 2048 * 4096)
    assert result.max_receiver_ooo_packets > 0
    assert result.retransmissions == 0


def test_zero_sack_bits_never_revoke_prior_acks():
    # 较旧位图中的零位不能撤销较新反馈已经确认的数据。
    sim = PacketSimulation(clean(), 128 * 4096)
    sim.receive_feedback(0, 5, 0, 1 << 5, None)
    sim.receive_feedback(0, 6, 0, 0, None)
    assert sim.is_acked(5) and sim.is_acked(6)
    assert sim.acked_count == 2


def test_sender_has_no_oracle_for_tail_loss():
    sim = PacketSimulation(clean(), 4096, data_drop_hook=lambda seq, attempt: attempt == 1, trace=True)
    sim.run()
    retransmissions = [e for e in sim.trace if e["event"] == "retransmit"]
    assert retransmissions[0]["time_ns"] >= sim.c.rto_ns


def test_physical_loss_never_creates_trim_notification():
    # 故障擦除后没有可转发包头，不能凭空触发 Trim 快速通知。
    result = simulate_flow(clean(data_loss=0.05), 8192 * 4096, seed=45)
    assert result.physical_drops > 0
    assert result.trim_events == result.nack_transmissions == 0


def test_trim_is_a_distinct_notification_mechanism():
    result = simulate_flow(clean(trim_probability=0.05), 2048 * 4096, seed=45)
    assert result.physical_drops == 0
    assert result.trim_events > 0
    assert result.nack_transmissions == result.trim_events


@pytest.mark.parametrize("protocol", ["gbn", "uet"])
def test_small_window_and_lost_control_packets_make_progress(protocol):
    result = simulate_flow(clean(protocol, window_packets=3, ack_every=4, data_loss=0.05, ack_loss=0.05),
                           128 * 4096, seed=8)
    assert result.packets == 128
    assert result.duration_ns > 0


@pytest.mark.parametrize("mode", ["iid", "burst"])
def test_loss_process_has_expected_stationary_rate(mode):
    loss = PacketLoss(0.05, seed=45, mode=mode, burst_length=8)
    fraction = sum(loss.drop() for _ in range(200_000)) / 200_000
    assert fraction == pytest.approx(0.05, abs=0.005)


def test_identical_seed_reproduces_every_counter():
    config = clean(data_loss=0.05, ack_loss=0.05)
    assert simulate_flow(config, 256 * 4096, 123) == simulate_flow(config, 256 * 4096, 123)


@pytest.mark.parametrize("parameters", [{"data_loss": 1}, {"ack_loss": -0.1}, {"window_packets": 0},
                                      {"path_spread_us": 10}, {"payload_bytes": 4096.5}])
def test_invalid_configuration_is_rejected(parameters):
    with pytest.raises(ValueError):
        clean(**parameters)


def test_incomplete_transfer_is_an_error_not_zero_goodput():
    with pytest.raises(RuntimeError, match="budget"):
        PacketSimulation(clean(max_events=100), 4096, data_drop_hook=lambda seq, attempt: True).run()
