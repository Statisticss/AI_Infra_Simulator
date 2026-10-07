"""Frozen-scenario, independent-seed evaluation of shared-NIC loss recovery."""
# 实验 004 批量入口：先冻结模型和配置，再执行独立随机种子并统一验收。
# 仿真时钟单位为 ns；wall_seconds 只记录 Python 程序运行耗时，不是通信时延。
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import pandas as pd
from scipy.stats import t as student_t
import yaml

from ai_infra_simulator.advanced_transport import RecoveryPolicy
from ai_infra_simulator.datacenter import simulate_datacenter
from ai_infra_simulator.transport import TransportConfig

ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = ("src/ai_infra_simulator/transport.py", "src/ai_infra_simulator/advanced_transport.py",
                "src/ai_infra_simulator/datacenter.py",
                "src/ai_infra_simulator/experiments/datacenter_study.py", "run_datacenter_study.py", "uv.lock")


def source_files():
    # 对实际依赖的源码和 uv.lock 按原始字节取摘要；注释变化也会改变摘要。
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def source_hash():
    return hashlib.sha256(json.dumps(source_files(), sort_keys=True).encode()).hexdigest()


def expand_jobs(document):
    # 正式种子不允许重复，也不能与试跑种子重合，防止把选参样本用于验收。
    seeds = document["seeds"]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(document["pilot_seeds"]):
        raise ValueError("evaluation seeds must be unique and disjoint from pilot seeds")
    jobs = []
    for experiment in document["experiments"]:
        if not 1 <= experiment["seed_count"] <= len(seeds):
            raise ValueError("invalid number of seeds")
        config = asdict(TransportConfig(**(document["transport"] | experiment.get("transport", {}))))
        # 显式展开默认值；每个任务的完整参数随原始结果一起保存。
        policy = asdict(RecoveryPolicy(**(document["policy"] | experiment.get("policy", {}))))
        for seed in seeds[:experiment["seed_count"]]:
            job = dict(condition=experiment["id"], workload=experiment["workload"],
                       size_mib=experiment["size_mib"], ranks=document["ranks"], seed=seed,
                       queue_capacity_bytes=document["queue_capacity_bytes"], transport=config, policy=policy)
            job["job_id"] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
            jobs.append(job)
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate jobs")
    return jobs


def run_job(job):
    # 每个 worker 真正执行全部分组事件，并在前后校验源码未被改动。
    started = perf_counter()
    code_hash = source_hash()
    output = simulate_datacenter(TransportConfig(**job["transport"]), round(job["size_mib"]*2**20),
                                 ranks=job["ranks"], workload=job["workload"], seed=job["seed"],
                                 policy=RecoveryPolicy(**job["policy"]),
                                 queue_capacity_bytes=job["queue_capacity_bytes"])
    if source_hash() != code_hash:
        raise RuntimeError("Source changed while executing a trial")
    return dict(job=job, source_sha256=code_hash, wall_seconds=perf_counter()-started, **output)


def summarize(results):
    # trials 保存每个样本，summary 按场景汇总；不能先合并不同 RTT/消息大小。
    rows = []
    for result in results:
        job = result["job"]
        rows.append(dict(condition=job["condition"], seed=job["seed"], job_id=job["job_id"],
                         size_mib=job["size_mib"], rtt_us=job["transport"]["rtt_us"],
                         data_loss=job["transport"]["data_loss"], ack_loss=job["transport"]["ack_loss"],
                         policy=job["policy"]["name"], **result["metrics"]))
    trials = pd.DataFrame(rows).sort_values(["condition", "seed"])
    metadata = ["condition", "workload", "ranks", "steps", "size_mib", "rtt_us", "data_loss", "ack_loss", "policy"]
    summaries = []
    for condition, samples in trials.groupby("condition", sort=True):
        row = {name: samples.iloc[0][name] for name in metadata}
        row["trials"] = len(samples)
        for metric in results[0]["metrics"]:
            if metric in metadata:
                continue
            row[metric] = samples[metric].mean()
            row[metric+"_ci95"] = (student_t.ppf(.975, len(samples)-1)*samples[metric].std(ddof=1)/len(samples)**.5
                                    if len(samples) > 1 else 0.)
        row["goodput_min_pct"] = samples.goodput_pct.min()
        row["goodput_max_pct"] = samples.goodput_pct.max()
        summaries.append(row)
    return trials, pd.DataFrame(summaries)


def acceptance(summary, target):
    # 同时检查样本数、最差单次和均值置信下界；不是只挑一个超过 90% 的种子。
    rows = summary[summary.condition == "acceptance"]
    if rows.empty:
        return dict(status="NOT_EVALUATED", target_pct=target)
    row = rows.iloc[0]
    lower = float(row.goodput_pct-row.goodput_pct_ci95)
    # CI 半宽只量化随机擦除的样本波动，不代表拓扑或协议简化带来的模型误差。
    passed = len(rows) == 1 and row.trials >= 10 and lower >= target and row.goodput_min_pct >= target
    return dict(status="PASS" if passed else "FAIL", target_pct=target,
                trials=int(row.trials), mean_goodput_pct=float(row.goodput_pct),
                ci95_half_width=float(row.goodput_pct_ci95), ci95_lower_pct=lower,
                minimum_goodput_pct=float(row.goodput_min_pct),
                criterion="10+ frozen-policy trials: mean 95% Student-t CI lower bound AND every trial >= target")


def main(argv=None):
    # 运行器强制 Python 3.10；选择 --groups 时应使用独立输出目录。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/004_datacenter_90pct.yaml")
    parser.add_argument("--output", type=Path, default=ROOT.parent/"results/004_datacenter_90pct")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("This experiment requires Python 3.10")
    if args.workers < 1:
        parser.error("workers must be positive")
    config_bytes = args.config.read_bytes()
    document = yaml.safe_load(config_bytes)
    jobs = expand_jobs(document)
    if args.groups:
        if set(args.groups)-{j["condition"] for j in jobs}:
            parser.error("unknown group")
        jobs = [j for j in jobs if j["condition"] in args.groups]
    if not jobs:
        parser.error("no jobs selected")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output/"jobs"
    checkpoints.mkdir(exist_ok=True)
    started = perf_counter()
    digest = source_hash()
    freeze = dict(study=document["study"], source_sha256=digest, source_files=source_files(),
                  config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                  document=document, selected_job_ids=sorted(j["job_id"] for j in jobs),
                  python=platform.python_version(), frozen_at=datetime.now(timezone.utc).isoformat(),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)))
    freeze_path = args.output/"freeze.json"
    if freeze_path.exists():
        # 输出目录绑定源码、配置、任务集和解释器版本，不允许混装不同批次结果。
        saved = json.loads(freeze_path.read_text())
        for key in ("source_sha256", "config_sha256", "selected_job_ids", "python"):
            if saved[key] != freeze[key]:
                raise SystemExit("Frozen scenario/source mismatch; use a new output directory")
        freeze = saved
    else:
        # 验收目标与完整参数必须先落盘，再运行任何正式样本。
        # Persist policy, scope and acceptance rule BEFORE any evaluation result.
        freeze_path.write_text(json.dumps(freeze, indent=2)+"\n")
    results, pending = [], []
    for job in jobs:
        file = checkpoints/(job["job_id"]+".json")
        if args.resume and file.exists():
            # 恢复只接受逐字段匹配的检查点；增加注释后也要换新输出目录。
            result = json.loads(file.read_text())
            if result["job"] != job or result["source_sha256"] != digest:
                raise SystemExit(f"Stale checkpoint: {file.name}")
            results.append(result)
        else:
            pending.append(job)
    print(f"Python {platform.python_version()}; {len(jobs)} jobs; {len(results)} reused", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # worker 各自维护完整 fabric，进程并发只加速独立试验，不影响仿真带宽。
        futures = [pool.submit(run_job, job) for job in pending]
        for future in as_completed(futures):
            result = future.result()
            if result["source_sha256"] != digest:
                raise RuntimeError("Source changed during evaluation")
            file = checkpoints/(result["job"]["job_id"]+".json")
            file.write_text(json.dumps(result, indent=2)+"\n")
            results.append(result)
            print(f"{len(results)}/{len(jobs)} {result['job']['condition']} seed={result['job']['seed']} "
                  f"goodput={result['metrics']['goodput_pct']:.4f}% elapsed={perf_counter()-started:.1f}s", flush=True)
    if args.config.read_bytes() != config_bytes or source_hash() != digest:
        raise RuntimeError("Configuration or source changed; results not accepted")
    trials, summary = summarize(results)
    trials.to_csv(args.output/"trials.csv", index=False)
    summary.to_csv(args.output/"summary.csv", index=False)
    decision = acceptance(summary, document["target_pct"])
    # 结果即使不达标仍完整保存，并用退出码 2 表明验收失败，不能筛掉低值重算。
    manifest = dict(**freeze, jobs=len(results), reused=len(results)-len(pending),
                    wall_seconds=perf_counter()-started, platform=platform.platform(),
                    finished_at=datetime.now(timezone.utc).isoformat(), acceptance=decision)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(decision, indent=2), flush=True)
    if decision["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
