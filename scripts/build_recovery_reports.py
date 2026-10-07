#!/usr/bin/env python3
"""Render reports from completed 002/003 trials; never synthesizes measurements."""
# 读取已完成的实验 002/003 汇总与 manifest，输出报告及图表，不在此重新运行协议。
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"uet": "#a95343", "cipu_candidate": "#117d89", "coverage_fixed": "#d29830",
          "window_only": "#75869d", "optimized": "#117d89"}
LABELS = {"uet": "UET mechanism baseline", "cipu_candidate": "CIPU-inspired candidate",
          "coverage_fixed": "Recovery only, same window", "window_only": "Window only",
          "optimized": "Recovery + shared-port PDCs"}
CN = {"uet": "UET 机制基线", "cipu_candidate": "CIPU 启发候选", "coverage_fixed": "仅优化恢复",
      "window_only": "仅扩展窗口", "optimized": "联合优化"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def ci(row, key="goodput_pct"):
    # 表中的 ± 是跨独立种子均值的 95% 置信区间半宽，不是标准差。
    return f"{row[key]:.2f} ± {row[key+'_ci95']:.2f}"


def savefig(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    svg = directory / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def load(name, base):
    # 只把摘要、运行信息、选参结果等小型文件放入仓库，逐流大数据留在输出目录。
    source, target = base / name, ROOT / "reports" / name
    summary = pd.read_csv(source / "summary.csv")
    metadata = json.loads((source / "manifest.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    figures = target / "figures"
    figures.mkdir(exist_ok=True)
    for filename in ("trials.csv", "summary.csv", "manifest.json", "selection.json", "verification.json",
                     "tuning_trials.csv", "tuning_summary.csv", "tuning_manifest.json"):
        if (source / filename).exists():
            shutil.copy2(source / filename, target / filename)
    return summary, metadata, target, figures


def cipu(base):
    # 展示消息大小、AllReduce 和突发丢包边界，候选结果不等同于厂商芯片实测。
    s, meta, target, figures = load("002_cipu_inspired", base)
    lossy = s[s.data_loss == .05]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for ax, group, title in zip(axes, ["cipu_bulk", "cipu_allreduce"],
                                ["One complete bulk transfer", "8-rank step-synchronous Ring AllReduce"]):
        subset = lossy[lossy.group == group]
        sizes = sorted(subset.size_mib.unique())
        for i, method in enumerate(["uet", "cipu_candidate"]):
            rows = subset[subset.strategy == method].set_index("size_mib").reindex(sizes)
            bars = ax.bar(np.arange(len(sizes)) + (i-.5)*.36, rows.goodput_pct, width=.35,
                          yerr=rows.goodput_pct_ci95, capsize=3, color=COLORS[method], label=LABELS[method])
            for bar, value in zip(bars, rows.goodput_pct):
                ax.text(bar.get_x()+bar.get_width()/2, value+1.5, f"{value:.1f}", ha="center", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.5))
        ax.set_xticks(np.arange(len(sizes)), [f"{x:g} MiB" for x in sizes])
        ax.set(xlabel="Message size" if group == "cipu_bulk" else "Tensor size per rank",
               ylabel="Full-completion goodput / 400 Gbit/s (%)", title=title, ylim=(0, 104))
        ax.axhline(90, ls="--", lw=1, color="#64748b", label="90% target")
        ax.axhline(95*4096/4224, ls=":", lw=1, color="#334155", label="92.12% asymptotic reference")
        ax.grid(axis="y", alpha=.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("5% data + ACK loss | 10 us RTT | 4 KiB payload | identical 947-packet window")
    fig.tight_layout(rect=(0, .12, 1, .94))
    savefig(fig, figures, "cipu_goodput")
    rows = []
    for group in ("cipu_bulk", "cipu_allreduce"):
        for size in sorted(s[s.group == group].size_mib.unique()):
            pair = lossy[(lossy.group == group) & (lossy.size_mib == size)].set_index("strategy")
            b, o = pair.loc["uet"], pair.loc["cipu_candidate"]
            zero = s[(s.group == group) & (s.size_mib == size) & (s.strategy == "cipu_candidate") & (s.data_loss == 0)].iloc[0]
            rows.append(["Bulk" if group == "cipu_bulk" else "Ring AllReduce", f"{size:g}", ci(b), ci(o),
                         f"{o.duration_ms:.3f}", f"{o.goodput_pct/zero.goodput_pct*100:.2f}%"])
    main = lossy[(lossy.group == "cipu_bulk") & (lossy.size_mib == 128) & (lossy.strategy == "cipu_candidate")].iloc[0]
    large = lossy[(lossy.group == "cipu_allreduce") & (lossy.size_mib == 1024) & (lossy.strategy == "cipu_candidate")].iloc[0]
    burst = lossy[lossy.group == "cipu_burst"]
    details = table(["场景", "大小 MiB（Ring 为每 rank）", "UET 基线 %", "候选 %", "候选完成 ms", "候选相对无损保留率"], rows)
    burst_table = table(["策略", "Goodput %", "实际丢包 %", "发送尝试/有效包"],
                        [[CN[r.strategy], ci(r), f"{r.realized_loss_pct:.3f}", f"{r.data_amplification:.4f}"]
                         for _, r in burst.iterrows()])
    text = f"""# 实验 002：CIPU 2.0 启发的恢复策略

在 Python {meta['python']} 中，候选方案在 **5% 数据及反馈独立丢失**时，
128 MiB Bulk 的完整完成 Goodput 为 **{ci(main)}%**（95% CI），
即 **{main.goodput_gbps:.2f} Gbit/s**；8 rank、每 rank 1 GiB 的 Ring 结果为 **{ci(large)}%**。
这证明在下述模型条件下可以超过 90%，不构成 CIPU 芯片性能复现。

## 公开资料与设计

[原文 §2.4](https://zartbot.github.io/blog/arch/cipu/index.html) 提到 CIPU 2.0 的 5%/90% 结果，
但本次可读取的正文没有完整的测试参数。公开方向包括有损、多路径、SACK 和 RC 接口兼容；
本次设计的重叠反馈、区间反馈补充、短计时器和尾部副本不能归为厂商已公开算法。
详见 [搜索分析与候选说明](RESEARCH.md)。

候选与 UET 机制基线均为 400 Gbit/s、10 us RTT、4 KiB 有效载荷、128 B 统一开销、
四路径、单逻辑连接、947 包窗口（约 8 BDP）。区别仅在反馈与恢复策略。
候选尾部重传总副本数为 3；所有副本、控制包都序列化并可能丢失。物理故障不产生 Trim。

## 完整结果

表中 `±` 为 95% CI 半宽，5 个独立种子；Goodput 以端口线速为分母。
最后一列另外报告相对无损性能，避免把两种“90%”混为一谈。

{details}

![CIPU-inspired result](figures/cipu_goodput.png)

128 MiB 候选发送尝试/有效包为 **{main.data_amplification:.5f}**，
理想纯选择重传为 1/0.95 = 1.05263；实际数据丢失比例为 **{main.realized_loss_pct:.3f}%**。
反向反馈字节/正向数据字节为 **{main.feedback_to_data_pct:.3f}%**，
观察到的乱序有效载荷峰值均值为 **{main.max_receiver_ooo_mib:.3f} MiB**。

90% 不是所有消息都能达到。短消息、每步等待最慢 rank、尾部连续丢失都会放大额外 RTT。
4 KiB+128 B 和 5% 丢失给出的渐近参考只有 92.12%；高 Goodput 要求恢复接近最少必要重传。
此场景中已有 UET 风格基线也能超过 90%，因此 90% 本身不足以证明某种专有实现或标准身份。

## 突发损失对照

下表为平均坏状态持续 8 个发送尝试的两态模型，仍标定长期损失率 5%；128 MiB、10 us。
各次有限样本实际丢包率并不完全一致，不能仅据均值推断突发损失更容易处理。

{burst_table}

在这个突发模型中，候选反而慢于基线，说明短计时器和相邻的冗余重传并非对所有故障分布都有利。
因此本实验的超过 90% 结论限定在所报告的独立丢包条件，不能推及连续闪断或黑洞。

## 复现与边界

```bash
source .venv/bin/activate
OPENBLAS_NUM_THREADS=1 python run_recovery_study.py --config configs/002_cipu_inspired.yaml --output ../results/002_cipu_inspired --workers 4
python scripts/build_recovery_reports.py --study 002
```

{meta['jobs']} 个任务，Python {meta['python']}。参数和所有样本见
[配置](../../configs/002_cipu_inspired.yaml)、[summary.csv](summary.csv)、[trials.csv](trials.csv)、[manifest.json](manifest.json)。
完整任务集、来源和守恒核验见 [verification.json](verification.json)。
源码 SHA-256：`{meta['source_sha256']}`。

AllReduce 是通信专用的分步 Ring，不含计算和 NCCL 流水，也未模拟邻居数据与反向 ACK 的物理 NIC 争用。
无交换机拥塞、在线 RTT 估计、持续链路中断或生产硬件校准。
完整假设、控制包成本、资源定义和协议依据见 [公共方法](../003_scale_across/METHOD.md)。
"""
    (target / "REPORT.md").write_text(text)
    print(target / "REPORT.md")


def wan(base):
    # 并列展示 RTT×丢包矩阵、窗口/恢复消融及长流和 collective 对照。
    s, meta, target, figures = load("003_scale_across", base)
    chosen = meta["selection"]
    matrix = s[s.group == "wan_matrix"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharey=True)
    for ax, rtt in zip(axes.flat, sorted(matrix.rtt_us.unique())):
        for method in ("uet", "coverage_fixed", "optimized"):
            rows = matrix[(matrix.rtt_us == rtt) & (matrix.strategy == method)].sort_values("data_loss")
            ax.errorbar(rows.data_loss*100, rows.goodput_pct, yerr=rows.goodput_pct_ci95,
                        color=COLORS[method], marker="o", capsize=2, lw=1.7, ms=4, label=LABELS[method])
        ref = matrix[(matrix.rtt_us == rtt) & (matrix.strategy == "optimized")].sort_values("data_loss")
        ax.plot(ref.data_loss*100, ref.ideal_fluid_reference_pct, ":", color="#4a5568",
                 label="Ideal fluid reference, includes 1 RTT")
        ax.set_xscale("symlog", linthresh=.1)
        ax.set_xticks([0, .1, 1, 3, 5, 10], ["0", "0.1", "1", "3", "5", "10"])
        ax.set(title=f"RTT = {rtt/1000:g} ms", xlabel="Independent data + feedback loss (%)",
               ylabel="Full 1 GiB transfer goodput (%)", ylim=(0, 101))
        ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Scale-across | one shared 400 Gbit/s port | finite completion, including the final ACK")
    fig.tight_layout(rect=(0, .10, 1, .95))
    savefig(fig, figures, "wan_loss_sweep")
    table_rows = []
    for rtt in sorted(matrix.rtt_us.unique()):
        for loss in sorted(matrix.data_loss.unique()):
            rows = matrix[(matrix.rtt_us == rtt) & (matrix.data_loss == loss)].set_index("strategy")
            table_rows.append([f"{rtt/1000:g}", f"{loss*100:g}", ci(rows.loc["uet"]),
                               ci(rows.loc["coverage_fixed"]), ci(rows.loc["optimized"])])
    result_table = table(["RTT ms", "丢包 %", "UET 基线 %", "仅优化恢复 %", "联合优化 %"], table_rows)
    five = matrix[matrix.data_loss == .05]
    resource_table = table(["RTT ms", "PDC 数", "配置窗口 MiB", "实测乱序峰值均值 MiB", "完成 ms", "提速比"],
        [[f"{r.rtt_us/1000:g}", int(r.pdcs), f"{r.window_payload_mib:.2f}", f"{r.max_receiver_ooo_mib:.2f}",
          f"{r.duration_ms:.2f}", f"{r.goodput_pct/five[(five.strategy=='uet') & (five.rtt_us==r.rtt_us)].iloc[0].goodput_pct:.2f}×"]
         for _, r in five[five.strategy == "optimized"].iterrows()])
    ablation = pd.concat([five[five.rtt_us.isin([1000, 10000])], s[s.group == "wan_window_only"]])
    ablation_table = table(["RTT ms", "策略", "Goodput %", "发送尝试/有效包", "反馈/数据字节 %"],
        [[f"{r.rtt_us/1000:g}", CN[r.strategy], ci(r), f"{r.data_amplification:.4f}", f"{r.feedback_to_data_pct:.3f}"]
         for _, r in ablation.sort_values(["rtt_us", "strategy"]).iterrows()])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for i, method in enumerate(("uet", "coverage_fixed", "window_only", "optimized")):
        rows = ablation[ablation.strategy == method].sort_values("rtt_us")
        bars = axes[0].bar(np.arange(2)+(i-1.5)*.19, rows.goodput_pct, width=.18,
                          yerr=rows.goodput_pct_ci95, capsize=2, color=COLORS[method], label=LABELS[method])
    axes[0].set_xticks([0, 1], ["1 ms", "10 ms"])
    axes[0].set(title="5% loss: separate algorithm and memory effects", ylabel="1 GiB full-completion goodput (%)", ylim=(0, 100))
    for size, style in ((1024, "o-"), (4096, "s--")):
        rows = (five[(five.strategy == "optimized") & five.rtt_us.isin([1000, 5000, 10000])] if size == 1024
                else s[s.group == "wan_long_flow"]).sort_values("rtt_us")
        axes[1].errorbar(rows.rtt_us/1000, rows.goodput_pct, yerr=rows.goodput_pct_ci95, fmt=style,
                        capsize=3, label=f"{size/1024:g} GiB, optimized")
    axes[1].axhline(95*4096/4224, color="#64748b", ls=":", label="92.12% asymptotic reference")
    axes[1].set(title="5% loss: message length amortizes recovery", xlabel="RTT (ms)", ylabel="Full-completion goodput (%)", ylim=(0, 100))
    axes[1].legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, 1))
    savefig(fig, figures, "wan_ablation")
    extras = s[s.group.isin(["wan_long_flow", "wan_allreduce", "wan_path_skew", "wan_burst"])]
    extra_table = table(["实验组", "RTT ms", "大小 MiB", "策略", "Goodput %", "完成 ms", "重复数"],
        [[r.group, f"{r.rtt_us/1000:g}", f"{r.size_mib:g}", CN[r.strategy], ci(r), f"{r.duration_ms:.2f}", int(r.trials)]
         for _, r in extras.iterrows()])
    ranking = table(["目标 BDP 倍数", "尾部总副本", "训练几何平均 Goodput %", "训练平均窗口 MiB"],
        [[r["window_bdp"], r["tail_copies"], f"{r['score']:.3f}", f"{r['average_window_mib']:.2f}"] for r in chosen["ranking"]])
    key_rows = five.set_index(["rtt_us", "strategy"])
    a, b = key_rows.loc[(1000, "optimized")], key_rows.loc[(10000, "optimized")]
    text = f"""# 实验 003：长 RTT、不同丢包率与受资源约束的恢复优化

在 400 Gbit/s、1 GiB 完整消息、数据和反馈均独立丢失 5% 的条件下，
RTT 1 ms 的 UET 机制基线为 **{ci(key_rows.loc[(1000, 'uet')])}%**，联合优化为 **{ci(a)}%**；
RTT 10 ms 时分别为 **{ci(key_rows.loc[(10000, 'uet')])}%** 和 **{ci(b)}%**。
这些是包含启动、等待及最终确认的完成 Goodput，不是无限长流的吞吐率。

## 策略与选参

使用重叠 SACK、区间收尾反馈、更短恢复计时器和尾部选择性冗余，
再把消息条带化到多个有界 PDC；所有 PDC 共享同一 400 Gbit/s 端口。
用独立训练种子搜索 9 组参数，选择 **{chosen['window_bdp']} BDP 目标窗口、尾部总副本 {chosen['tail_copies']}**，
聚合窗口有效载荷预算不超过 **2 GiB**，最多 **64 PDC**。
结果中的“最优”仅指此候选集合及训练目标，不是所有协议、拓扑和负载的全局最优。
实现、依据、资源定义和未模拟因素详见 [METHOD.md](METHOD.md)。

## 完整 RTT × 丢包率矩阵

每点 1 GiB；每包 4096 B 有效载荷 + 128 B 统一开销；四路径单程跨度 2 us。
`±` 为 95% CI 半宽，主矩阵有损点 n=5；无损点确定性 n=1。
“仅优化恢复”维持基线的单 PDC 与相同窗口，便于识别算法本身的收益。

{result_table}

![WAN loss sweep](figures/wan_loss_sweep.png)

单 PDC 常规 32640 包窗口只有 127.5 MiB 有效载荷；400 Gbit/s、10 ms 的线速 BDP 已达 500 MB。
窗口耗尽、累计确认缺口以及有限 SACK 的覆盖延迟会造成停顿或误重传。
扩大窗口和改善反馈解决的是不同问题，且尾部仍可能需要多个 RTT。
UET 标准允许不同实现，这个基线的数字不能代表所有 UET 产品。

## 5% 丢包：资源与消融

下表给出联合优化所用的实际 PDC 数和配置窗口；乱序峰值是逐次运行峰值的均值。
窗口有效载荷预算不是 NIC SRAM 需求，重放描述符、bitmap、宿主/GPU 存储需另行设计。

{resource_table}

{ablation_table}

![WAN ablation](figures/wan_ablation.png)

反向反馈独立付出序列化及丢失代价。尾部副本也占用数据端口，
因此优化可能用更高的发送放大换取更短收尾，不能称作免费纠错。
5% 丢失下的渐近参考为 92.12%；10% 丢失下只有 87.27%，后者无法靠纯重传达到 90% 线速 Goodput。

## 长流、AllReduce 和鲁棒性

`wan_long_flow` 为 4 GiB 完整消息（不是截取稳态区间）；`wan_allreduce` 为 8 rank、每 rank 1 GiB，
所有逻辑邻居都使用表内长 RTT；`wan_path_skew` 为 100 us 路径时延跨度；
`wan_burst` 为平均连续丢 8 个发送尝试的两态模型。四组均为 5% 标定丢包率。
WAN 的两态过程按 PDC 独立生成，改变条带数也会改变端口聚合损失相关性；
这组不能用来推断面对同一物理链路突发轨迹的公平性能优势。002 的突发对照双方均为单 PDC。

{extra_table}

Ring 每一步等待最慢 rank，共 14 步，长 RTT 的收尾代价被重复放大。
更长消息能摊薄固定开销，但无法消除它。真实跨机房训练还应研究机房内归约、机房间交换、
跨阶段流水和更好的站点拓扑；这些不在本次模拟范围内，未将预期收益写成已实现性能。

## 全部搜索结果与独立验证

训练场景：1 GiB、RTT 1/10 ms、5% 丢包，种子 3/7；9 组参数共 36 次。
目标为完整完成 Goodput 的几何平均；距最好 1% 内优先节省窗口，再减少副本。
随后才用种子 11/29/47/71/101 做主评估（长流和 Ring 使用前三个）。

{ranking}

保留 [selection.json](selection.json)、[tuning_summary.csv](tuning_summary.csv)、
[tuning_trials.csv](tuning_trials.csv)、[summary.csv](summary.csv)、[trials.csv](trials.csv)
与 [manifest.json](manifest.json)。没有根据评估结果重新选择或隐去较差点。
完整任务集、来源和守恒核验见 [verification.json](verification.json)。

## 复现

```bash
source .venv/bin/activate
OPENBLAS_NUM_THREADS=1 python run_recovery_study.py --config configs/003_scale_across.yaml --output ../results/003_scale_across --phase tune --workers 4
OPENBLAS_NUM_THREADS=1 python run_recovery_study.py --config configs/003_scale_across.yaml --output ../results/003_scale_across --phase eval --workers 4
python scripts/build_recovery_reports.py --study 003
```

本次 {meta['jobs']} 个评估任务，另有 36 个训练任务；Python {meta['python']}。
源码 SHA-256：`{meta['source_sha256']}`。相同源码和配置可用 `--resume`；不匹配会拒绝复用。

模型不含交换机排队/拥塞控制、GPU 计算、在线 RTT 估计或链路黑洞。
AllReduce 延续独立逻辑链路近似，未模拟邻居数据与反向 ACK 在同一物理 NIC 上的争用。
因此这些结果可比较恢复机制及资源瓶颈，不能作为商用硬件或真实多机房 NCCL 性能保证。
"""
    (target / "REPORT.md").write_text(text)
    print(target / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "results")
    parser.add_argument("--study", choices=["002", "003", "all"], default="all")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "ai-infra-simulator-recovery-studies"})
    if args.study in {"002", "all"}:
        cipu(args.results)
    if args.study in {"003", "all"}:
        wan(args.results)


if __name__ == "__main__":
    main()
