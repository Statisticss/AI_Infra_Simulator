# 002/003 实验编排验证：训练/评估隔离、逻辑流计量和资源优先的选参规则。
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ai_infra_simulator.experiments.recovery_study import choose_policy, expand_jobs, run_job, summarize


ROOT = Path(__file__).resolve().parents[1]


def test_training_and_evaluation_are_disjoint_and_job_ids_unique():
    d = yaml.safe_load((ROOT / "configs/003_scale_across.yaml").read_text())
    training = expand_jobs(d, "tune")
    evaluation = expand_jobs(d, "eval", {"window_bdp": 4, "tail_copies": 3})
    assert len(training) == 36
    assert not set(j["seed"] for j in training) & set(j["seed"] for j in evaluation)
    assert len({j["job_id"] for j in training + evaluation}) == len(training + evaluation)
    assert all(j["total_window_packets"] <= 2**19 for j in evaluation)


def test_ring_uses_max_of_ranks_and_correct_busbandwidth():
    d = yaml.safe_load((ROOT / "configs/002_cipu_inspired.yaml").read_text())
    d["experiments"] = [dict(id="small", workload="ring_allreduce", size_mib=[1],
                              rtt_us=[10], data_loss=[.05], strategies=["cipu_candidate"])]
    d["seeds"] = [11, 29]
    results = [run_job(j) for j in expand_jobs(d, "eval")]
    for result in results:
        assert len(result["flows"]) == 112
        duration = sum(max(f["duration_ns"] for f in result["flows"] if f["step"] == step)
                       for step in range(14))
        assert result["metrics"]["duration_ms"] == duration / 1e6
        assert result["metrics"]["goodput_gbps"] == pytest.approx(2**20 * 1.75 * 8 / duration)
    trials, summary = summarize(results)
    assert len(trials) == 2 and len(summary) == 1
    assert summary.iloc[0].goodput_pct_ci95 > 0


def test_search_one_percent_tie_prefers_resources_not_test_data():
    d = yaml.safe_load((ROOT / "configs/003_scale_across.yaml").read_text())
    rows = pd.DataFrame([
        dict(search_window_bdp=2, tail_copies=2, goodput_pct=50, window_payload_mib=200),
        dict(search_window_bdp=4, tail_copies=3, goodput_pct=50.4, window_payload_mib=400),
    ])
    selected = choose_policy(rows, d)
    assert selected["window_bdp"] == 2
    assert selected["tail_copies"] == 2
