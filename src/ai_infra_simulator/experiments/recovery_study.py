"""Experiments 002/003: auditable strategy search and held-out evaluation."""
# 实验 002/003：将端点恢复策略、窗口/PDC 资源及 RTT/丢包率组合展开为可复现任务。
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import yaml

from ai_infra_simulator.advanced_transport import RecoveryPolicy, simulate_port, window_plan
from ai_infra_simulator.transport import TransportConfig
from .loss_recovery import source_hash


COUNTERS = ("packets", "data_transmissions", "retransmissions", "physical_drops", "ack_transmissions",
            "ack_drops", "forward_wire_bytes", "reverse_wire_bytes", "event_count",
            "tail_redundant_transmissions", "closure_ack_transmissions", "duplicate_deliveries",
            "early_retransmissions", "timeout_retransmissions")


def search_hash(document):
    # 选参结果绑定训练矩阵、传输参数和资源预算，不能拿旧搜索结果套新场景。
    return hashlib.sha256(json.dumps({k: document[k] for k in ("tuning", "transport", "resources")},
                                     sort_keys=True).encode()).hexdigest()


def strategy(document, name, rtt, loss, selection=None, overrides=None):
    # 根据实验名选择基线、只扩窗口、只改恢复或二者联合；数据和 ACK 默认同丢包率。
    cfg = dict(document["transport"])
    cfg.update(rtt_us=rtt, data_loss=loss, ack_loss=loss)
    cfg.update(overrides or {})
    c = TransportConfig(**cfg)
    baseline = name in {"uet", "window_only"}
    chosen = selection or {"window_bdp": 8, "tail_copies": 3}
    if not baseline:
        c = replace(c, early_factor=1.1, rto_factor=1.25)
    policy = RecoveryPolicy("baseline" if baseline else "coverage", closure_copies=2,
                            tail_copies=chosen["tail_copies"], tail_packets=32640)
    if name in {"optimized", "window_only", "tuning"}:
        pdcs, budget = window_plan(c, chosen["window_bdp"], **document["resources"])
    else:
        pdcs, budget = 1, c.window
    return c, policy, pdcs, budget


def add_job(jobs, document, group, name, workload, size, rtt, loss, seed,
            selection=None, overrides=None):
    c, policy, pdcs, budget = strategy(document, name, rtt, loss, selection, overrides)
    job = dict(group=group, strategy=name, workload=workload, size_mib=size, ranks=8,
               rtt_us=rtt, data_loss=loss, seed=seed, transport=asdict(c), policy=asdict(policy),
               pdcs=pdcs, total_window_packets=budget,
               search_window_bdp=selection["window_bdp"] if selection else 8,
               tail_copies=policy.tail_copies)
    # condition_id 聚合同参数的随机样本；job_id 还包括种子，用来唯一标识检查点。
    job["condition_id"] = hashlib.sha256(json.dumps({k: v for k, v in job.items() if k != "seed"},
                                                   sort_keys=True).encode()).hexdigest()[:16]
    job["job_id"] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]
    jobs.append(job)


def expand_jobs(document, phase, selection=None):
    # tune 只展开训练种子；eval 使用冻结策略和独立评估种子，避免验证数据泄漏。
    jobs = []
    if phase == "tune":
        tuning = document["tuning"]
        for w, copies, rtt, loss, seed in itertools.product(tuning["window_bdp"], tuning["tail_copies"],
                tuning["rtt_us"], tuning["data_loss"], tuning["seeds"]):
            add_job(jobs, document, "tuning", "tuning", "bulk", tuning["size_mib"], rtt, loss,
                    seed, {"window_bdp": w, "tail_copies": copies})
    else:
        for exp in document["experiments"]:
            for name, size, rtt, loss in itertools.product(exp["strategies"], exp["size_mib"],
                                                         exp["rtt_us"], exp["data_loss"]):
                seeds = exp.get("seeds", document["seeds"])
                if loss == 0 and not exp.get("transport", {}).get("ack_loss", 0):
                    seeds = seeds[:1]  # deterministic paths, zero data/control loss
                for seed in seeds:
                    add_job(jobs, document, exp["id"], name, exp["workload"], size, rtt, loss, seed,
                            selection, exp.get("transport"))
    return jobs


def run_job(job):
    # 本版逐流模拟，再逐阶段取最慢完成时间；多端点持续时钟模型另见实验 004。
    started = perf_counter()
    c, policy = TransportConfig(**job["transport"]), RecoveryPolicy(**job["policy"])
    ranks = 1 if job["workload"] == "bulk" else job["ranks"]
    steps = 1 if ranks == 1 else 2 * (ranks - 1)
    size = round(job["size_mib"] * 2**20)
    if job["workload"] not in {"bulk", "ring_allreduce"} or size % ranks:
        raise ValueError("unsupported workload or non-divisible ring tensor")
    chunk = size // ranks
    flows, phases = [], []
    cached = None
    for step in range(steps):
        times = []
        for rank in range(ranks):
            # Matches experiment 001's per-flow seed derivation.
            seed = job["seed"] * 1_000_003 + step * 1009 + rank * 7919
            flow = cached if cached else simulate_port(c, chunk, seed, policy, job["pdcs"],
                                                        job["total_window_packets"])
            if c.data_loss == c.ack_loss == c.trim_probability == 0:
                cached = flow
            flows.append(dict(flow, step=step, rank=rank, seed=seed))
            times.append(flow["duration_ns"])
        phases.append(max(times))
    duration = sum(phases)
    counters = {k: sum(f[k] for f in flows) for k in COUNTERS}
    goodput = chunk * steps * 8 / duration / c.bandwidth_gbps * 100
    # reference 只是理想流体参考；报告中的实测 Goodput 来自上面的分组事件结果。
    wire = chunk + ((chunk + c.payload_bytes - 1) // c.payload_bytes) * c.overhead_bytes
    reference = chunk * 8 / (wire * 8 / (c.bandwidth_gbps * (1-c.data_loss)) + c.rtt_us * 1000)
    metrics = dict(duration_ms=duration / 1e6, goodput_pct=goodput,
                   goodput_gbps=goodput * c.bandwidth_gbps / 100,
                   algorithm_gbps=size * 8 / duration,
                   data_amplification=counters["data_transmissions"] / counters["packets"],
                   realized_loss_pct=counters["physical_drops"] / counters["data_transmissions"] * 100,
                   feedback_to_data_pct=counters["reverse_wire_bytes"] / counters["forward_wire_bytes"] * 100,
                   max_receiver_ooo_mib=max(f["max_receiver_ooo_mib"] for f in flows),
                   window_payload_mib=flows[0]["window_payload_mib"],
                   asymptotic_ceiling_pct=flows[0]["asymptotic_ceiling_pct"],
                   ideal_fluid_reference_pct=reference / c.bandwidth_gbps * 100,
                   pdcs=job["pdcs"], window_packets=job["total_window_packets"],
                   phase_durations_ms=[t/1e6 for t in phases], **counters)
    return dict(job=job, metrics=metrics, flows=flows, wall_seconds=perf_counter()-started,
                source_sha256=source_hash())


def summarize(results):
    # 各计数器先保留每个种子的原值，再计算跨种子均值和 Student-t 置信区间。
    rows = []
    for result in results:
        job = result["job"]
        row = {k: v for k, v in job.items() if k not in {"policy", "transport"}}
        row.update({k: v for k, v in result["metrics"].items() if not isinstance(v, list)})
        row["wall_seconds"] = result["wall_seconds"]
        row["ack_loss"] = job["transport"]["ack_loss"]
        row["loss_model"] = job["transport"]["loss_model"]
        row["path_spread_us"] = job["transport"]["path_spread_us"]
        rows.append(row)
    trials = pd.DataFrame(rows).sort_values(["group", "strategy", "rtt_us", "data_loss", "size_mib", "seed"])
    metadata = ["condition_id", "group", "strategy", "workload", "size_mib", "ranks", "rtt_us", "data_loss",
                "ack_loss", "loss_model", "path_spread_us", "pdcs", "total_window_packets", "search_window_bdp",
                "tail_copies", "window_payload_mib"]
    aggregate = []
    for _, samples in trials.groupby("condition_id"):
        row = {k: samples.iloc[0][k] for k in metadata}
        row["trials"] = len(samples)
        for metric in result["metrics"]:
            if metric == "phase_durations_ms" or metric in metadata:
                continue
            row[metric] = samples[metric].mean()
            row[metric + "_ci95"] = (student_t.ppf(.975, len(samples)-1) * samples[metric].std(ddof=1)
                                        / len(samples)**.5 if len(samples) > 1 else 0.)
        aggregate.append(row)
    summary = pd.DataFrame(aggregate).sort_values(["group", "strategy", "rtt_us", "data_loss", "size_mib"])
    return trials, summary


def choose_policy(trials, document):
    # 用训练场景 Goodput 的几何均值排序；差距在 1% 内时优先省内存、少副本。
    # 此函数只接收训练结果，不读取评估种子表现。
    # Primary objective: geometric mean full-transfer goodput across training
    # scenarios/seeds. Within 1% of the best, prefer less provisioned memory,
    # then fewer proactive tail copies. Evaluation seeds are never read here.
    ranking = []
    for (w, copies), samples in trials.groupby(["search_window_bdp", "tail_copies"]):
        ranking.append(dict(window_bdp=int(w), tail_copies=int(copies),
                            score=float(np.exp(np.log(samples.goodput_pct).mean())),
                            average_window_mib=float(samples.window_payload_mib.mean())))
    best = max(r["score"] for r in ranking)
    eligible = [r for r in ranking if r["score"] >= best * .99]
    winner = min(eligible, key=lambda r: (r["average_window_mib"], r["tail_copies"], -r["score"]))
    return dict(winner, ranking=sorted(ranking, key=lambda r: -r["score"]),
                objective="geometric mean full-transfer goodput; 1% tie favors memory then copies",
                training_seeds=document["tuning"]["seeds"], evaluation_seeds=document["seeds"],
                search_config_sha256=search_hash(document),
                source_sha256=source_hash())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["tune", "eval"], default="eval")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Use Python 3.10: code/.venv/bin/python")
    if args.workers < 1:
        parser.error("workers must be positive")
    document = yaml.safe_load(args.config.read_text())
    if "tuning" in document and set(document["seeds"]) & set(document["tuning"]["seeds"]):
        raise SystemExit("Training and evaluation seeds must be disjoint")
    selection = None
    args.output.mkdir(parents=True, exist_ok=True)
    selected_file = args.output / "selection.json"
    code_hash = source_hash()
    if args.phase == "eval" and "tuning" in document:
        # 003 必须先 tune；源码或搜索配置变化后，旧 selection.json 会被拒绝。
        selection = json.loads(selected_file.read_text())
        if selection["source_sha256"] != code_hash or selection["search_config_sha256"] != search_hash(document):
            raise SystemExit("Model or search configuration changed after tuning; rerun tuning")
    jobs = expand_jobs(document, args.phase, selection)
    if args.groups:
        jobs = [j for j in jobs if j["group"] in args.groups]
    if not jobs:
        parser.error("no jobs selected")
    checkpoints = args.output / "jobs"
    checkpoints.mkdir(exist_ok=True)
    results, pending = [], []
    for job in jobs:
        checkpoint = checkpoints / (job["job_id"] + ".json")
        # 检查点校验按源码原始字节执行，因此只改注释也需要独立结果目录。
        if args.resume and checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["source_sha256"] != code_hash or saved["job"] != job:
                raise SystemExit(f"Stale checkpoint: {checkpoint}")
            results.append(saved)
        else:
            pending.append(job)
    started = perf_counter()
    print(f"Python {platform.python_version()}; {len(jobs)} jobs, {len(results)} reused; {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job): job for job in pending}
        for future in as_completed(futures):
            result = future.result()
            if result["source_sha256"] != code_hash:
                raise RuntimeError("Source changed while jobs were executing")
            (checkpoints / (result["job"]["job_id"] + ".json")).write_text(json.dumps(result, indent=2))
            results.append(result)
            if len(results) % 10 == 0 or len(results) == len(jobs):
                print(f"{len(results)}/{len(jobs)} complete; elapsed {perf_counter()-started:.1f}s", flush=True)
    trials, summary = summarize(results)
    prefix = "tuning_" if args.phase == "tune" else ""
    trials.to_csv(args.output / f"{prefix}trials.csv", index=False)
    summary.to_csv(args.output / f"{prefix}summary.csv", index=False)
    if args.phase == "tune":
        selection = choose_policy(trials, document)
        selected_file.write_text(json.dumps(selection, indent=2))
        print(json.dumps(selection, indent=2), flush=True)
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    phase=args.phase, source_sha256=code_hash,
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                    jobs=len(jobs), reused=len(jobs)-len(pending),
                    wall_seconds=perf_counter()-started, selection=selection,
                    training_seeds=document.get("tuning", {}).get("seeds", []), evaluation_seeds=document["seeds"])
    (args.output / f"{prefix}manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
