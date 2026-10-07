"""Configuration-driven runner. No networking or compilation is used by simulations."""

# 实验 001 运行器：读取 YAML 参数扫描，执行分组仿真，汇总统计并生成图表。
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import pandas as pd
from scipy.stats import t as student_t
import yaml

from ai_infra_simulator.collectives import simulate_workload
from ai_infra_simulator.transport import TransportConfig


def source_hash() -> str:
    # 001–003 对整个模型包的 Python 文件取原始字节摘要；注释和新增模块也会改变它。
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def expand_jobs(document: dict) -> list[dict]:
    # 用笛卡尔积展开扫描轴，生成完整配置和稳定任务编号，便于恢复及审计。
    jobs = []
    for experiment in document["experiments"]:
        config = dict(document["transport_defaults"])
        config.update(experiment.get("transport", {}))
        sweep = experiment.get("sweep", {})
        axes = list(sweep)
        combinations = itertools.product(*(sweep[axis] for axis in axes)) if axes else [()]
        for values in combinations:
            current = config | dict(zip(axes, values))
            # ACK loss tracks data loss unless the experiment fixes it explicitly.
            if experiment.get("ack_loss_tracks_data", True):
                current["ack_loss"] = current["data_loss"]
            transport = TransportConfig(**current)
            is_zero = transport.data_loss == transport.ack_loss == transport.trim_probability == 0
            # 无损且路径确定时只运行一次；随机有损条件使用全部指定种子。
            seeds = [document["seeds"][0]] if is_zero else document["seeds"]
            for seed in seeds:
                job = {"experiment": experiment["id"], "group": experiment["group"],
                       "workload": experiment["workload"], "size_mib": experiment["size_mib"],
                       "ranks": experiment.get("ranks", 8), "seed": seed, "transport": asdict(transport)}
                job["job_id"] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]
                jobs.append(job)
    return jobs


def run_job(job: dict) -> dict:
    # MiB 转换为真实字节数后执行，Python 运行耗时与模拟通信时延分开记录。
    started = perf_counter()
    result = simulate_workload(TransportConfig(**job["transport"]),
                               round(job["size_mib"] * 1024**2), job["seed"],
                               job["workload"], job["ranks"])
    return {"job": job, "metrics": result["metrics"], "flows": result["flows"],
            "wall_seconds": perf_counter() - started, "source_sha256": source_hash()}


def summarize(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 统计组键包含协议、时延、窗口等全部关键条件，避免把不同配置混成一个均值。
    rows = []
    for result in results:
        job = result["job"]
        row = {key: value for key, value in job.items() if key != "transport"}
        row.update(job["transport"])
        row.update({k: v for k, v in result["metrics"].items() if not isinstance(v, list)})
        row["wall_seconds"] = result["wall_seconds"]
        rows.append(row)
    raw = pd.DataFrame(rows)
    keys = ["experiment", "group", "workload", "size_mib", "ranks", "protocol", "bandwidth_gbps",
            "rtt_us", "payload_bytes", "overhead_bytes", "data_loss", "ack_loss", "loss_model",
            "burst_length", "trim_probability", "early_recovery", "rto_factor", "rto_us", "window_bdp",
            "window_packets", "path_count", "path_spread_us"]
    metrics = ["goodput_pct", "duration_ms", "algorithm_gbps", "bus_gbps", "transmission_amplification",
               "observed_data_loss_pct", "observed_ack_loss_pct", "wire_payload_efficiency_pct",
               "max_receiver_ooo_mib", "feedback_to_forward_pct", "asymptotic_erasure_ceiling_pct"]
    summary = []
    for key, samples in raw.groupby(keys, dropna=False, sort=True):
        entry = dict(zip(keys, key))
        count = len(samples)
        entry["trials"] = count
        for metric in metrics:
            entry[metric] = samples[metric].mean()
            halfwidth = (student_t.ppf(0.975, count - 1) * samples[metric].std(ddof=1) / count**0.5
                         if count > 1 else 0.0)
            entry[metric + "_ci95"] = halfwidth
        summary.append(entry)
    aggregate = pd.DataFrame(summary)
    raw["retained_vs_zero_pct"] = float("nan")
    # 相对无损保留率另算；不能把“保留无损性能的 90%”称作“线速 Goodput 90%”。
    aggregate["retained_vs_zero_pct"] = float("nan")
    baseline_keys = ["experiment", "protocol", "rtt_us", "size_mib", "payload_bytes", "window_packets",
                     "early_recovery", "loss_model", "path_spread_us"]
    for index, row in aggregate.iterrows():
        baseline = aggregate[(aggregate.data_loss == 0) & (aggregate.ack_loss == 0) & (aggregate.trim_probability == 0)]
        for field in baseline_keys:
            baseline = baseline[baseline[field] == row[field]]
        if len(baseline) == 1:
            aggregate.loc[index, "retained_vs_zero_pct"] = row.goodput_pct / baseline.iloc[0].goodput_pct * 100
            mask = raw.experiment == row.experiment
            for field in keys:
                mask &= raw[field].isna() if pd.isna(row[field]) else raw[field] == row[field]
            raw.loc[mask, "retained_vs_zero_pct"] = raw.loc[mask, "goodput_pct"] / baseline.iloc[0].goodput_pct * 100
    return raw, aggregate


def plot_results(summary: pd.DataFrame, output: Path):
    # 图表只读取已完成试验的汇总数据，不用理论公式补齐未运行的样本。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    colors = {"gbn": "#c45d45", "uet": "#177e89"}
    labels = {"gbn": "Traditional RoCE / GBN model", "uet": "UET RUD mechanism model"}
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    main = summary[(summary.group == "main") & (summary.data_loss == 0.05)]
    if not main.empty:
        order = main.sort_values(["rtt_us", "size_mib"]).experiment.drop_duplicates().tolist()
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
        x = np.arange(len(order))
        for i, protocol in enumerate(["gbn", "uet"]):
            rows = main[main.protocol == protocol].set_index("experiment").reindex(order)
            offset = (i - 0.5) * 0.36
            bars = axes[0].bar(x + offset, rows.goodput_pct, width=0.35, color=colors[protocol],
                              label=labels[protocol], yerr=rows.goodput_pct_ci95, capsize=3)
            for bar, value in zip(bars, rows.goodput_pct):
                if np.isfinite(value):
                    axes[0].text(bar.get_x()+bar.get_width()/2, value+1.4, f"{value:.1f}%",
                                 ha="center", fontsize=9)
        ticks = []
        for name in order:
            row = main[main.experiment == name].iloc[0]
            ticks.append(f"{row.rtt_us:g} us RTT\n{row.size_mib:g} MiB/rank")
        axes[0].set_xticks(x, ticks, fontsize=9)
        axes[0].set_ylim(0, 105)
        axes[0].axhline(90, ls="--", color="#697585", lw=1, label="90% target")
        axes[0].axhline(95 * 4096 / 4224, ls=":", color="#334155", lw=1, label="92.12% asymptotic bound")
        axes[0].set_ylabel("AllReduce bus bandwidth / 400 Gbit/s (%)")
        axes[0].set_title("8-rank step-synchronous Ring AllReduce")
        axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        axes[0].grid(axis="y", alpha=0.15)
        for protocol in ["gbn", "uet"]:
            rows = main[main.protocol == protocol].set_index("experiment").reindex(order)
            axes[1].plot(x, rows.duration_ms, "o-", color=colors[protocol], label=labels[protocol])
        axes[1].set_yscale("log")
        axes[1].set_xticks(x, ticks, fontsize=9)
        axes[1].set_ylabel("AllReduce completion time (ms, log scale)")
        axes[1].set_title("The slowest rank gates each step")
        axes[1].grid(axis="y", which="both", alpha=0.15)
        fig.suptitle("5% independent data and feedback loss | 4 KiB payload | 128 B equalized overhead", fontsize=12)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(figures / "allreduce.png", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for protocol in ["gbn", "uet"]:
        rows = summary[(summary.group == "loss_sweep") & (summary.protocol == protocol)].sort_values("data_loss")
        if not rows.empty:
            axes[0, 0].errorbar(rows.data_loss * 100, rows.goodput_pct, yerr=rows.goodput_pct_ci95,
                               marker="o", capsize=3, color=colors[protocol], label=labels[protocol])
    axes[0, 0].set(xlabel="Independent data + feedback loss (%)", ylabel="Bulk goodput / line rate (%)",
                   title="Loss-rate sweep: 32 MiB, 10 us RTT", ylim=(0, 102))
    axes[0, 0].legend(fontsize=8)
    rows = summary[summary.group == "window"].sort_values("window_bdp")
    if not rows.empty:
        axes[0, 1].errorbar(rows.window_bdp, rows.goodput_pct, yerr=rows.goodput_pct_ci95,
                           marker="o", color=colors["uet"], capsize=3)
    axes[0, 1].set(xlabel="PSN window / base BDP", ylabel="Bulk goodput / line rate (%)",
                   title="Window sensitivity: UET, 128 MiB, 10 us RTT", ylim=(0, 102))
    axes[0, 1].axhline(90, ls="--", color="#697585", lw=1)
    rows = summary[summary.group == "recovery"].sort_values("experiment")
    if not rows.empty:
        names = {"fault_fast": "Physical loss\nEarly + RTO", "fault_rto": "Physical loss\nRTO only",
                 "trim_fast": "Synthetic trim\nNACK + SACK", "burst_fast": "Burst loss\nMean run 8"}
        axes[1, 0].bar([names.get(x, x) for x in rows.experiment], rows.goodput_pct,
                       yerr=rows.goodput_pct_ci95, capsize=3, color=colors["uet"])
    axes[1, 0].set(ylabel="Bulk goodput / line rate (%)", ylim=(0, 102), title="Recovery / loss-shape sensitivity, UET")
    rows = summary[summary.group == "mtu"].sort_values("payload_bytes")
    if not rows.empty:
        axes[1, 1].errorbar(rows.payload_bytes, rows.goodput_pct, yerr=rows.goodput_pct_ci95,
                           marker="o", color=colors["uet"], label="Simulation", capsize=3)
        axes[1, 1].plot(rows.payload_bytes, rows.asymptotic_erasure_ceiling_pct, "--", color="#334155",
                        label="(1-p) * payload / wire size")
    axes[1, 1].set(xlabel="Payload bytes", ylabel="Bulk goodput / line rate (%)", ylim=(50, 102),
                   title="Header overhead matters: 5% loss, 32 MiB")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.15)
    fig.suptitle("UET is modeled as selected mechanisms, not a complete protocol implementation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/001_loss_recovery.yaml"))
    parser.add_argument("--output", type=Path, default=Path("../results/001_loss_recovery"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--groups", nargs="+", help="Optional experiment groups to run")
    parser.add_argument("--resume", action="store_true", help="Reuse matching per-job checkpoints")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Run this experiment with Python 3.10 (code/.venv/bin/python).")
    if args.workers < 1:
        parser.error("--workers must be positive")
    document = yaml.safe_load(args.config.read_text())
    jobs = expand_jobs(document)
    if args.groups:
        jobs = [job for job in jobs if job["group"] in args.groups]
    if not jobs:
        parser.error("No matching experiments")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "jobs"
    checkpoints.mkdir(exist_ok=True)
    results = []
    pending = []
    code_hash = source_hash()
    for job in jobs:
        saved = checkpoints / (job["job_id"] + ".json")
        if args.resume and saved.exists():
            result = json.loads(saved.read_text())
            if result["source_sha256"] != code_hash or result["job"] != job:
                raise SystemExit(f"Stale checkpoint {saved}; choose a new output directory or omit --resume")
            results.append(result)
        else:
            pending.append(job)
    started = perf_counter()
    print(f"Python {platform.python_version()}; {len(jobs)} jobs, {len(results)} reused; {args.workers} workers", flush=True)

    def save(result):
        job = result["job"]
        destination = checkpoints / (job["job_id"] + ".json")
        temporary = destination.with_suffix(".tmp")
        # 先写临时文件再替换，减少中断时留下半份检查点的风险。
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(destination)
        results.append(result)
        print(f"[{len(results)}/{len(jobs)}] {job['experiment']} {job['transport']['protocol']} "
              f"p={job['transport']['data_loss']:g} seed={job['seed']} "
              f"goodput={result['metrics']['goodput_pct']:.3f}% wall={result['wall_seconds']:.2f}s", flush=True)

    if args.workers == 1:
        for job in pending:
            save(run_job(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_job, job) for job in pending]
            for future in as_completed(futures):
                save(future.result())
    results.sort(key=lambda result: result["job"]["job_id"])
    raw, summary = summarize(results)
    raw.to_csv(args.output / "trials.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    plot_results(summary, args.output)
    manifest = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                "source_sha256": code_hash, "config": document, "selected_groups": args.groups,
                "jobs": len(jobs), "wall_seconds_this_run": perf_counter() - started,
                "workers": args.workers,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                "metric": "AllReduce: [2(N-1)/N * S * 8 / completion_time] / link_rate; bulk: [S*8/T]/link_rate",
                "confidence_intervals": "95% Student-t intervals across independent seeded trials; simulation uncertainty only"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {args.output.resolve()} ({perf_counter()-started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
