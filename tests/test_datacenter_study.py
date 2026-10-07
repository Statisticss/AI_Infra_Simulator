# 004 验收验证：正式种子与试跑隔离，且均值置信下界和最低样本都必须达标。
import copy

import pandas as pd
import pytest
import yaml

from ai_infra_simulator.experiments.datacenter_study import ROOT, acceptance, expand_jobs, source_files


def test_frozen_matrix_has_independent_seeds_and_explicit_parameters():
    document = yaml.safe_load((ROOT/"configs/004_datacenter_90pct.yaml").read_text())
    jobs = expand_jobs(document)
    assert len(jobs) == 36
    primary = [j for j in jobs if j["condition"] == "acceptance"]
    assert len(primary) == 10 and {j["seed"] for j in primary}.isdisjoint(document["pilot_seeds"])
    assert all(j["transport"]["data_loss"] == j["transport"]["ack_loss"] == .05 for j in primary)
    assert all(j["transport"]["trim_probability"] == 0 for j in jobs)
    for invalid_seeds in ([3, 11], [11, 11]):
        changed = copy.deepcopy(document)
        changed["seeds"] = invalid_seeds
        with pytest.raises(ValueError):
            expand_jobs(changed)
    assert len(source_files()) == 6


def test_acceptance_requires_all_samples_and_confidence_lower_bound():
    def verdict(mean=90.7, ci=.02, minimum=90.6, count=10):
        row = dict(condition="acceptance", goodput_pct=mean, goodput_pct_ci95=ci,
                   goodput_min_pct=minimum, trials=count)
        return acceptance(pd.DataFrame([row]), 90)["status"]
    assert verdict() == "PASS"
    assert verdict(ci=1) == "FAIL"
    assert verdict(minimum=89.9) == "FAIL"
    assert verdict(count=9) == "FAIL"
    assert acceptance(pd.DataFrame([dict(condition="small_tensor")]), 90)["status"] == "NOT_EVALUATED"
