#!/usr/bin/env python3
"""Recompute study 004 coverage, timing, bytes, port budgets and acceptance."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from ai_infra_simulator.datacenter import flow_seed
from ai_infra_simulator.experiments.datacenter_study import acceptance, expand_jobs, source_hash, summarize


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6), (a, b)


def read_results(directory):
    if (directory/"jobs").is_dir():
        return [json.loads(p.read_text()) for p in sorted((directory/"jobs").glob("*.json"))]
    with gzip.open(directory/"raw_trials.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]


def verify(directory, config):
    raw = config.read_bytes()
    document = yaml.safe_load(raw)
    manifest = json.loads((directory/"manifest.json").read_text())
    freeze = json.loads((directory/"freeze.json").read_text())
    digest = source_hash()
    assert manifest["source_sha256"] == freeze["source_sha256"] == digest
    assert manifest["config_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["document"] == freeze["document"] == document
    assert manifest["python"].startswith("3.10.")
    expected = {j["job_id"]: j for j in expand_jobs(document)}
    results = read_results(directory)
    assert len(results) == len(expected) == manifest["jobs"]
    assert {r["job"]["job_id"] for r in results} == set(expected)
    assert sorted(expected) == freeze["selected_job_ids"]
    total_flows = total_packets = total_attempts = total_acks = total_events = 0
    for result in results:
        job, m, nics, flows = (result[k] for k in ("job", "metrics", "nics", "flows"))
        c = job["transport"]
        assert job == expected[job["job_id"]] and result["source_sha256"] == digest
        ranks = job["ranks"]
        size = round(job["size_mib"]*2**20)
        steps = 1 if job["workload"] == "permutation" else 2*(ranks-1)
        chunk = size if steps == 1 else size//ranks
        assert chunk % c["payload_bytes"] == 0
        assert len(nics) == ranks and len(flows) == ranks*steps
        assert {(f["phase"], f["rank"]) for f in flows} == {(s, r) for s in range(steps) for r in range(ranks)}
        assert len({f["seed"] for f in flows}) == len(flows)
        duration = m["duration_ms"]*1e6
        phase_end = 0.
        for phase, dt in enumerate(result["phase_durations_ns"]):
            current = [f for f in flows if f["phase"] == phase]
            assert all(phase_end < f["receiver_complete_ns"] <= f["completion_ns"] <= duration for f in current)
            phase_end += dt
            close(max(f["completion_ns"] for f in current), phase_end)
        close(phase_end, m["final_confirmation_ms"]*1e6)
        assert phase_end <= duration
        for f in flows:
            assert f["seed"] == flow_seed(job["seed"], f["phase"], f["rank"])
            assert f["destination"] == (f["rank"]+1) % ranks
            assert f["payload_bytes"] == chunk
            assert f["packets"] == f["received_packets"] == f["acknowledged_packets"] == chunk//c["payload_bytes"]
            assert f["data_transmissions"] == f["packets"]+f["retransmissions"]
            assert f["physical_drops"]+f["packets"] <= f["data_transmissions"]
            assert f["max_receiver_ooo_packets"] <= f["window_packets"] <= 32640
            assert f["max_sack_offset"] <= 32767
            assert f["forward_wire_bytes"] == f["data_transmissions"]*(c["payload_bytes"]+c["overhead_bytes"])
            assert f["reverse_wire_bytes"] == f["ack_transmissions"]*c["ack_bytes"]
            assert f["trim_events"] == f["out_of_order_discards"] == 0
        for key in ("data_transmissions", "retransmissions", "physical_drops", "ack_transmissions", "ack_drops",
                    "forward_wire_bytes", "reverse_wire_bytes", "packets"):
            assert sum(f[key] for f in flows) == m[key]
        for nic in nics:
            assert nic["data_tx_bytes"] == sum(f["forward_wire_bytes"] for f in flows if f["rank"] == nic["rank"])
            assert nic["ack_tx_bytes"] == sum(f["reverse_wire_bytes"] for f in flows if f["destination"] == nic["rank"])
            tx = nic["data_tx_bytes"]+nic["ack_tx_bytes"]
            close(nic["tx_utilization_pct"], tx*8/duration/c["bandwidth_gbps"]*100)
            assert tx*8/c["bandwidth_gbps"] <= duration+1e-6
            assert nic["rx_bytes"]*8/c["bandwidth_gbps"] <= duration+1e-6
            assert nic["rx_bytes"] >= chunk*steps+chunk//c["payload_bytes"]*steps*c["overhead_bytes"]
            assert nic["max_ingress_queue_bytes"] <= job["queue_capacity_bytes"]
        close(m["goodput_pct"], chunk*steps*8/duration/c["bandwidth_gbps"]*100)
        close(m["bus_gbps"], chunk*steps*8/duration)
        close(m["algorithm_gbps"], size*8/duration)
        close(m["goodput_pct"], m["wire_payload_efficiency_pct"]*m["mean_nic_tx_utilization_pct"]/100)
        close(m["wire_payload_efficiency_pct"], chunk*steps*ranks/(m["forward_wire_bytes"]+m["reverse_wire_bytes"])*100)
        close(m["observed_data_loss_pct"], 100*m["physical_drops"]/m["data_transmissions"])
        close(m["observed_ack_loss_pct"], 100*m["ack_drops"]/m["ack_transmissions"])
        assert m["total_unique_payload_bytes"] == chunk*steps*ranks
        if c["data_loss"] == c["ack_loss"] == 0:
            assert m["retransmissions"] == m["physical_drops"] == m["ack_drops"] == 0
        total_flows += len(flows)
        total_packets += m["packets"]
        total_attempts += m["data_transmissions"]
        total_acks += m["ack_transmissions"]
        total_events += m["event_count"]
    trials, summary = summarize(results)
    for name, frame in (("trials", trials), ("summary", summary)):
        pd.testing.assert_frame_equal(pd.read_csv(directory/f"{name}.csv").reset_index(drop=True),
                                      frame.reset_index(drop=True), check_dtype=False, check_like=True,
                                      rtol=1e-10, atol=1e-9)
    decision = acceptance(summary, document["target_pct"])
    assert decision == manifest["acceptance"]
    info = dict(status="PASS", source_sha256=digest, jobs=len(results), flows=total_flows,
                unique_packets=total_packets, data_transmissions=total_attempts,
                ack_transmissions=total_acks, events=total_events, acceptance=decision,
                checks=["exact frozen job coverage", "model/config hashes", "independent seed namespace",
                        "every packet received and acknowledged", "persistent collective phase clock",
                        "data and ACK byte conservation", "shared per-NIC TX/RX budget", "bounded ingress buffers",
                        "bounded PSN span", "Goodput/algorithm/bus definitions", "CSV recomputation", "frozen acceptance rule"])
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent/"results/004_datacenter_90pct")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/004_datacenter_90pct.yaml")
    args = parser.parse_args()
    result = verify(args.input, args.config)
    (args.input/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
