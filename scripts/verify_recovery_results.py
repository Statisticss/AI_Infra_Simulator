#!/usr/bin/env python3
"""Audit exact job coverage, provenance and conservation of completed studies."""
# 实验 002/003 审计入口：覆盖训练与验证任务、逐流状态和共享端口资源预算。
# 原始报告对应 f59f05c；新增中文注释后需用匹配源码审计，或在新目录重新运行。
from pathlib import Path
import json
import math
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ai_infra_simulator.experiments.loss_recovery import source_hash
from ai_infra_simulator.experiments.recovery_study import expand_jobs, choose_policy


def verify(name):
    output = ROOT.parent / "results" / name
    document = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    selection = json.loads((output / "selection.json").read_text()) if "tuning" in document else None
    jobs = expand_jobs(document, "eval", selection)
    if selection:
        # 用训练结果重新选择策略，并检查正式评估种子从未参与选参。
        jobs += expand_jobs(document, "tune")
        computed = choose_policy(pd.read_csv(output / "tuning_trials.csv"), document)
        assert (computed["window_bdp"], computed["tail_copies"]) == (selection["window_bdp"], selection["tail_copies"])
        assert not set(selection["training_seeds"]) & set(selection["evaluation_seeds"])
    saved = {p.stem: json.loads(p.read_text()) for p in (output / "jobs").glob("*.json")}
    # 精确比较任务集，缺少任何任务或混入别的配置均不能通过。
    assert set(saved) == {j["job_id"] for j in jobs}, "Missing or unexpected trials"
    code_hash = source_hash()
    total_flows = total_transmissions = 0
    for job in jobs:
        r = saved[job["job_id"]]
        assert r["job"] == job and r["source_sha256"] == code_hash
        metrics, flows = r["metrics"], r["flows"]
        c = job["transport"]
        rank_count = 1 if job["workload"] == "bulk" else job["ranks"]
        steps = 1 if rank_count == 1 else 2 * (rank_count - 1)
        assert len(flows) == rank_count * steps
        duration = sum(max(f["duration_ns"] for f in flows if f["step"] == step) for step in range(steps))
        # 每步取最慢 rank，再累加全部步骤；不能用流平均时间高估 collective 性能。
        assert math.isclose(metrics["duration_ms"] * 1e6, duration)
        total_flows += len(flows)
        total_transmissions += metrics["data_transmissions"]
        for f in flows:
            assert f["payload_bytes"] == round(job["size_mib"] * 2**20) // rank_count
            assert f["data_transmissions"] == f["packets"] + f["retransmissions"]
            assert f["physical_drops"] + f["packets"] <= f["data_transmissions"]
            assert f["max_receiver_ooo_packets"] <= f["window_packets"]
            assert f["forward_wire_bytes"] * 8 / c["bandwidth_gbps"] <= f["duration_ns"] + 1e-3
            assert f["reverse_wire_bytes"] == c["ack_bytes"] * f["ack_transmissions"]
            assert f["max_sack_offset"] <= 32767
            assert f["receiver_complete_ns"] <= f["duration_ns"]
            assert 0 < f["goodput_pct"] <= 100
            if c["data_loss"] == c["ack_loss"] == 0:
                assert f["retransmissions"] == 0
    info = dict(study=name, jobs=len(jobs), flows=total_flows, data_transmissions=total_transmissions,
                source_sha256=code_hash, checks="PASS")
    (output / "verification.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    for study in sys.argv[1:] or ["002_cipu_inspired", "003_scale_across"]:
        verify(study)
