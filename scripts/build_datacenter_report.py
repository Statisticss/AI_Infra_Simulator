#!/usr/bin/env python3
"""Build study 004 report from audited raw trials, including a portable archive."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from verify_datacenter_results import read_results, verify

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "acceptance": "候选：1 GiB/rank AllReduce，10 us",
    "baseline_recovery": "机制基线：1 GiB/rank AllReduce，10 us",
    "medium_tensor": "候选：256 MiB/rank AllReduce，10 us",
    "small_tensor": "候选：32 MiB/rank AllReduce，10 us",
    "higher_rtt": "候选：1 GiB/rank AllReduce，50 us",
    "permutation": "候选：128 MiB/rank permutation，10 us",
    "lossless_reference": "候选：1 GiB/rank AllReduce，无损参考",
}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def savefig(fig, directory, name):
    fig.savefig(directory/f"{name}.png", dpi=180, bbox_inches="tight")
    output = directory/f"{name}.svg"
    fig.savefig(output, bbox_inches="tight", metadata={"Date": None})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines())+"\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent/"results/004_datacenter_90pct")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Use Python 3.10")
    audit = verify(args.input, ROOT/"configs/004_datacenter_90pct.yaml")
    target = ROOT/"reports/004_datacenter_90pct"
    figures = target/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name in ("trials.csv", "summary.csv", "manifest.json", "freeze.json"):
        if (args.input/name).resolve() != (target/name).resolve():
            shutil.copy2(args.input/name, target/name)
    (target/"verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    # Read before opening the destination, so the committed archive is also a
    # valid --input for re-rendering without rerunning the simulation.
    results = sorted(read_results(args.input), key=lambda r: r["job"]["job_id"])
    with (target/"raw_trials.jsonl.gz").open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as archive:
            for result in results:
                archive.write((json.dumps(result, sort_keys=True, separators=(",", ":"))+"\n").encode())
    trials = pd.read_csv(target/"trials.csv")
    summary = pd.read_csv(target/"summary.csv").set_index("condition")
    manifest = json.loads((target/"manifest.json").read_text())
    main_row = summary.loc["acceptance"]
    main_trials = trials[trials.condition == "acceptance"].sort_values("seed")
    decision = manifest["acceptance"]
    baseline, zero, permutation = (summary.loc[k] for k in ("baseline_recovery", "lossless_reference", "permutation"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "ai-infra-simulator-datacenter-004"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})
    axes[0].plot(np.arange(len(main_trials)), main_trials.goodput_pct, "o", color="#087f8c", markersize=7)
    axes[0].axhspan(main_row.goodput_pct-main_row.goodput_pct_ci95,
                   main_row.goodput_pct+main_row.goodput_pct_ci95, color="#087f8c", alpha=.14, label="95% CI of mean")
    axes[0].axhline(main_row.goodput_pct, color="#087f8c", lw=1, label=f"Mean {main_row.goodput_pct:.3f}%")
    axes[0].axhline(90, ls="--", color="#b76524", label="90% target")
    axes[0].set_xticks(np.arange(len(main_trials)), main_trials.seed)
    axes[0].set(xlabel="Independent evaluation seed", ylabel="Goodput / 400 Gbit/s (%)",
                title="1 GiB/rank Ring AllReduce · 10 us RTT", ylim=(89.9, 90.9))
    axes[0].legend(loc="lower right", fontsize=9)
    keys = ["acceptance", "medium_tensor", "small_tensor", "higher_rtt", "permutation"]
    labels = ["Ring\n1 GiB\n10 us", "Ring\n256 MiB\n10 us", "Ring\n32 MiB\n10 us",
              "Ring\n1 GiB\n50 us", "Permutation\n128 MiB\n10 us"]
    bars = axes[1].bar(np.arange(len(keys)), summary.loc[keys].goodput_pct,
                       yerr=summary.loc[keys].goodput_pct_ci95, capsize=3,
                       color=["#087f8c", "#7993a5", "#7993a5", "#7993a5", "#087f8c"])
    for bar, key in zip(bars, keys):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.8,
                     f"{summary.loc[key].goodput_pct:.2f}%", ha="center", fontsize=9)
    axes[1].axhline(90, ls="--", color="#b76524", lw=1)
    axes[1].set_xticks(np.arange(len(keys)), labels)
    axes[1].set(ylabel="Goodput / 400 Gbit/s (%)", title="Scope and limits · same candidate", ylim=(0, 103))
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("5% IID data + ACK loss | 8 shared full-duplex NICs | 4 KiB payload | complete operation")
    fig.tight_layout()
    savefig(fig, figures, "goodput")

    # Fractions of aggregate port capacity, recomputed separately for every trial.
    factor = 8/(main_trials.duration_ms*1e6*400*8)*100
    parts = {
        "Unique network payload": main_trials.goodput_pct.mean(),
        "Repair / duplicate payload": ((main_trials.data_transmissions-main_trials.packets)*4096*factor).mean(),
        "All data-frame overhead": (main_trials.data_transmissions*128*factor).mean(),
        "ACK frames": (main_trials.reverse_wire_bytes*factor).mean(),
        "Idle / completion waiting": (100-main_trials.mean_nic_tx_utilization_pct).mean(),
    }
    assert abs(sum(parts.values())-100) < 1e-8
    fig, ax = plt.subplots(figsize=(12, 2.7))
    colors = ["#087f8c", "#d99b46", "#8ba0b4", "#ad7ca4", "#d9dee5"]
    left = 0.
    for (label, value), color in zip(parts.items(), colors):
        ax.barh([0], [value], left=left, color=color, label=f"{label}: {value:.3f}%")
        left += value
    ax.set(xlim=(0, 100), yticks=[], xlabel="Fraction of total NIC TX capacity over full completion time (%)",
           title="AllReduce capacity budget · all original, repair and feedback frames charged")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.38), ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    savefig(fig, figures, "capacity_budget")

    rows = []
    for key in NAMES:
        row = summary.loc[key]
        value = (f"{row.goodput_pct:.3f}（确定性）" if row.trials == 1
                 else f"{row.goodput_pct:.3f} ± {row.goodput_pct_ci95:.3f}")
        rows.append([NAMES[key], int(row.trials), value,
                     f"{row.goodput_min_pct:.3f}", f"{row.duration_ms:.3f}"])
    detail = table(["场景", "种子数", "Goodput %（均值 ± 95% CI 半宽）", "最低 %", "完成时间 ms"], rows)
    by_seed = table(["种子", "Goodput %", "完成时间 ms", "实际数据丢失 %", "实际 ACK 丢失 %"],
                    [[int(row.seed), f"{row.goodput_pct:.5f}", f"{row.duration_ms:.5f}",
                      f"{row.observed_data_loss_pct:.5f}", f"{row.observed_ack_loss_pct:.5f}"]
                     for _, row in main_trials.iterrows()])
    budget = table(["容量用途", "占端口容量 %"],
                   [[label, f"{value:.4f}"] for label, value in zip(
                       ["唯一有效网络载荷", "重传/副本载荷", "所有数据帧开销", "所有 ACK", "空闲及完成等待"], parts.values())])
    report = f"""# 实验 004：5% 丢包下超过 90% Goodput

**验收结果：{decision['status']}。** 在 8 rank、每 rank 1 GiB、400 Gbit/s、10 us 名义 RTT 的 Ring AllReduce 通信仿真中，
数据及 ACK 均有 5% 独立丢失时，完整操作 Goodput 为 **{main_row.goodput_pct:.3f}% ± {main_row.goodput_pct_ci95:.3f} 个百分点**（均值的 95% CI）。
10 个独立正式种子全部超过 90%，最小值 **{main_row.goodput_min_pct:.3f}%**，置信区间下界 **{decision['ci95_lower_pct']:.3f}%**。

这对应每 rank **{main_row.bus_gbps:.2f} Gbit/s** 的有效网络带宽和 **{main_row.duration_ms:.3f} ms** 的平均完成时间。
所有数据、ACK、重传、冗余副本均计入物理端口预算，并计入起步和最终确认时间。
独立的 128 MiB/rank permutation 对照达到 **{permutation.goodput_pct:.3f}% ± {permutation.goodput_pct_ci95:.3f}**，该指标不含 AllReduce 换算因子。

本实验在 Python **{manifest['python']}** 中真实执行；结果来自分组事件仿真，不是解析公式生成的样本。
它验证了一个合理数据中心场景中的可达性；并非 CIPU 实测、完整 UET 实现或跨场景最优性证明。

## 场景和策略

| 项目 | 设定 |
|---|---|
| 网络 | 8 个端点，各一个全双工 400 Gbit/s NIC，非阻塞内部 fabric |
| 业务 | 每 rank 1 GiB 张量；Ring 14 阶段，每阶段每 rank 128 MiB |
| 时延 | 10 us 传播/固定处理 RTT，另加收发序列化与队列；四路径总偏差 2 us |
| 数据/反馈 | payload 4096 B + 总开销 128 B；ACK 96 B；数据与 ACK 共享本端 TX 和目的端 RX |
| 故障 | 每次数据发送、每个 ACK 均独立以 5% 概率丢失；重传和副本无豁免；Trim=0 |
| 恢复 | 乱序接收、重叠 SACK、区间反馈补充、短 RTT 恢复定时器、尾部重传总副本数 3 |
| 资源 | 每连接窗口 947 包，约 3.699 MiB payload span；每目的端口 RX 队列上限 256 KiB |
| 原则 | 参数在正式运行前固定；试跑 seed 3 与正式种子分开 |

400 GbE 和 4 KiB 包长的公开依据、协议步骤及完整简化条件见 [方法与依据](METHOD.md)。
核心原因是选择性恢复把平均发送放大控制在 **{main_row.data_amplification:.5f} 倍**，接近理想的 1/0.95=1.05263；
较大窗口和大块通信使平均 NIC TX 利用率达到 **{main_row.mean_nic_tx_utilization_pct:.3f}%**。
反馈占数据字节的 **{main_row.feedback_to_data_pct:.4f}%**；这些反馈使用同一物理 NIC 的发送带宽。

## 结果与边界

{detail}

除无损参考行外，表中数据和 ACK 丢失概率均为 5%。CI 是跨种子的双侧 Student-t 区间；无损参考不估计随机区间。
机制基线使用相同网络、包长和 8 BDP 窗口，保留滚动 SACK、较长定时器，禁用候选的覆盖反馈与尾部副本。
基线在这个大块、短 RTT 场景也达到约 {baseline.goodput_pct:.2f}%；候选平均增加 **{main_row.goodput_pct-baseline.goodput_pct:.3f} 个百分点**。
因此，达到 90% 不应被解释成只有这一种算法才能实现；场景选择和开销预算同样关键。

![Goodput and limits](figures/goodput.png)

每 rank 256 MiB 和 32 MiB 的 AllReduce 均未达到 90%；提高到 50 us RTT 后也下降。
各阶段等待最慢 rank、最终丢包的恢复以及小消息起停成本会降低有效利用率。
本次结果不扩展到 1–10 ms 的跨数据中心 AllReduce；[实验 003](../003_scale_across/REPORT.md)的结论继续保留。

## 口径和逐项预算

本报告 Goodput = 每 rank 必需且不重复的网络 payload × 8 /（完整操作时间 × 400 Gbit/s）。
Ring AllReduce 的必要网络 payload 为张量大小的 `2(N−1)/N = 1.75` 倍。
按照 [NCCL tests 的 bus bandwidth 定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)，
本结果的 bus bandwidth 为 **{main_row.bus_gbps:.2f} Gbit/s**，algorithm bandwidth 为 **{main_row.algorithm_gbps:.2f} Gbit/s**。
相对无损 Goodput 的保留率另为 **{main_row.goodput_pct/zero.goodput_pct*100:.2f}%**；这不是用于验收的“90%”。

{budget}

上述五项逐 trial 精确相加为 100%。有效载荷/全部 TX 字节为 **{main_row.wire_payload_efficiency_pct:.3f}%**，
乘以平均端口利用率即可得到约 {main_row.goodput_pct:.3f}% 的 Goodput。
主实验观测的数据丢失比例为 **{main_row.observed_data_loss_pct:.4f}%**，ACK 为 **{main_row.observed_ack_loss_pct:.4f}%**。
十次试验中接收端乱序 payload 峰值最大 **{main_trials.max_receiver_ooo_mib.max():.3f} MiB**；
目的端口 ingress 队列峰值最大 **{int(main_trials.max_ingress_queue_bytes.max())} B**，低于设置的 256 KiB。

![Capacity budget](figures/capacity_budget.png)

## 十次主实验

{by_seed}

## 复现与审计

```bash
uv sync --python 3.10 --locked
source .venv/bin/activate
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_datacenter_study.py --workers 4
python scripts/verify_datacenter_results.py
python scripts/build_datacenter_report.py
```

默认输出在仓库同级 `results/004_datacenter_90pct/`；可用 `--output` 指定独立目录。
完整实验本次墙钟耗时约 **{manifest['wall_seconds']/60:.1f} 分钟**，取决于机器与负载。
同一输出目录可加 `--resume`，会严格核对配置及源码摘要；单组试验用 `--groups acceptance` 并指定另一个输出目录。

不重跑仿真，直接核验 GitHub 随附的压缩原始结果：

```bash
python scripts/verify_datacenter_results.py --input reports/004_datacenter_90pct
```

审计覆盖 **{audit['jobs']} 个任务、{audit['flows']} 条逻辑流、{audit['data_transmissions']:,} 次数据发送、{audit['ack_transmissions']:,} 次 ACK 发送**，全部通过。
59 个测试覆盖解析时延、数据/ACK 不重叠、有限 RX、阶段时钟保持、尾部多次丢失恢复、字节预算和统计验收等。
源码/依赖锁摘要：`{manifest['source_sha256']}`。
运行从上一版提交 `{manifest['git_head'][:7]}` 的工作区开始，新模型文件当时尚未提交，因而 manifest 的 dirty 标志为 true；
逐文件 SHA-256 固定了本次实际运行代码，并可与本报告一起提交的源码直接核对。

文件：[配置](../../configs/004_datacenter_90pct.yaml) · [全部 trial CSV](trials.csv) · [汇总 CSV](summary.csv) ·
[完整原始 trial 压缩包](raw_trials.jsonl.gz) · [运行信息](manifest.json) · [运行前冻结记录](freeze.json) · [审计结果](verification.json)。
旧版代码及结果已先上传至 [f59f05c](https://github.com/Statisticss/AI_Infra_Simulator/commit/f59f05cea8a29a989ef97dec4e5cdf818a603dca)，保持可追溯。
"""
    (target/"REPORT.md").write_text(report)
    print(f"Report built: {target/'REPORT.md'}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
