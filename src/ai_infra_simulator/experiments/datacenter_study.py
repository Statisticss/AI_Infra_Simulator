"""Frozen-scenario, independent-seed evaluation of shared-NIC loss recovery."""
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
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def source_hash():
    return hashlib.sha256(json.dumps(source_files(), sort_keys=True).encode()).hexdigest()


def expand_jobs(document):
    seeds = document["seeds"]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(document["pilot_seeds"]):
        raise ValueError("evaluation seeds must be unique and disjoint from pilot seeds")
    jobs = []
    for experiment in document["experiments"]:
        if not 1 <= experiment["seed_count"] <= len(seeds):
            raise ValueError("invalid number of seeds")
        config = asdict(TransportConfig(**(document["transport"] | experiment.get("transport", {}))))
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
    rows = summary[summary.condition == "acceptance"]
    if rows.empty:
        return dict(status="NOT_EVALUATED", target_pct=target)
    row = rows.iloc[0]
    lower = float(row.goodput_pct-row.goodput_pct_ci95)
    passed = len(rows) == 1 and row.trials >= 10 and lower >= target and row.goodput_min_pct >= target
    return dict(status="PASS" if passed else "FAIL", target_pct=target,
                trials=int(row.trials), mean_goodput_pct=float(row.goodput_pct),
                ci95_half_width=float(row.goodput_pct_ci95), ci95_lower_pct=lower,
                minimum_goodput_pct=float(row.goodput_min_pct),
                criterion="10+ frozen-policy trials: mean 95% Student-t CI lower bound AND every trial >= target")


def main(argv=None):
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
        saved = json.loads(freeze_path.read_text())
        for key in ("source_sha256", "config_sha256", "selected_job_ids", "python"):
            if saved[key] != freeze[key]:
                raise SystemExit("Frozen scenario/source mismatch; use a new output directory")
        freeze = saved
    else:
        # Persist policy, scope and acceptance rule BEFORE any evaluation result.
        freeze_path.write_text(json.dumps(freeze, indent=2)+"\n")
    results, pending = [], []
    for job in jobs:
        file = checkpoints/(job["job_id"]+".json")
        if args.resume and file.exists():
            result = json.loads(file.read_text())
            if result["job"] != job or result["source_sha256"] != digest:
                raise SystemExit(f"Stale checkpoint: {file.name}")
            results.append(result)
        else:
            pending.append(job)
    print(f"Python {platform.python_version()}; {len(jobs)} jobs; {len(results)} reused", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
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
    manifest = dict(**freeze, jobs=len(results), reused=len(results)-len(pending),
                    wall_seconds=perf_counter()-started, platform=platform.platform(),
                    finished_at=datetime.now(timezone.utc).isoformat(), acceptance=decision)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(decision, indent=2), flush=True)
    if decision["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
