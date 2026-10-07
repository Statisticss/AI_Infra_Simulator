# AI_Infra_Simulator

使用 **Python 3.10** 进行 AI 基础设施传输研究。从可独立运行、可验证的实验开始，
逐步积累共同的事件、链路、传输和 collective 模型。

研究范围包括 Scale-out（RoCEv2、InfiniBand）、Scale-up（NVLink、UALink）、
Scale-across（长距离 RDMA），以及 PCIe、CXL 等总线传输。
当前实现范围见下表；其余协议尚未实现。

## 已完成实验

| 编号 | 问题 | 模型与状态 |
|---|---|---|
| 001 | 5% 故障丢包下，GBN 与选择性重传的 Goodput 差别有多大？能否达到 90%？ | 分组级传输 + 分步 Ring AllReduce；192 个任务、5 个有损随机种子，结果和参数已保存 |
| 002 | 根据 CIPU 2.0 的公开方向设计可达 90% 的候选策略 | 重叠 SACK、区间反馈、尾部冗余；Bulk 与 8 rank Ring，详见独立报告 |
| 003 | RTT 1–10 ms 下不同丢包率的性能与优化 | 共享 400 Gbit/s 端口的多个 PDC、2 GiB 窗口预算、独立选参与评估、资源消融 |
| 004 | 在合理数据中心场景中，严格按物理 NIC 预算达到 5% 丢包 / 90% Goodput | 全部 rank 共用事件时钟，数据和 ACK 共享实际 NIC 收发端口；36 次试验及完整压缩账本 |

**[实验 004：5% 丢包下超过 90% Goodput](reports/004_datacenter_90pct/REPORT.md)** ·
[场景、协议及指标定义](reports/004_datacenter_90pct/METHOD.md)

最新结果：8 rank、每 rank 1 GiB、400 Gbit/s、10 us RTT 的 Ring AllReduce 通信模型中，
数据和 ACK 都独立丢失 5% 时，候选策略的完整操作 Goodput 为 **90.685% ± 0.013 个百分点（95% CI）**。
10 次独立试验最低为 **90.656%**，对应平均网络带宽 **362.74 Gbit/s/rank**。
计入全部报文开销、反馈、重传、尾部副本和最终确认等待，按 AllReduce bus bandwidth / 端口速率定义。
这是大块、短 RTT、非阻塞 fabric 场景的仿真；256 MiB/rank、32 MiB/rank 和 50 us RTT 对照均未达到 90%。

![实验 004：正式种子及边界](reports/004_datacenter_90pct/figures/goodput.png)

**[实验 002：CIPU 启发策略](reports/002_cipu_inspired/REPORT.md)** ·
[CIPU 公开资料分析](reports/002_cipu_inspired/RESEARCH.md) ·
**[实验 003：Scale-across](reports/003_scale_across/REPORT.md)** ·
[新增模型的方法与局限](reports/003_scale_across/METHOD.md)

**[阅读实验 001 报告](reports/001_loss_recovery/REPORT.md)** ·
[模型假设与原始依据](reports/001_loss_recovery/METHOD.md) ·
[全部配置](configs/001_loss_recovery.yaml)

实验 001 的原机制对照：8 rank、每 rank 1 GiB、400 Gbit/s、10 us RTT、数据及反馈均独立丢失 5% 时，
传统 RC/GBN 模型的 Goodput 为 **12.95%**，UET RUD 风格机制模型为 **90.55%**。
这是指定参数下的机制仿真结果，**不是完整 UET 实现、CIPU 实测或对真实 NCCL 系统的性能承诺**。
报告同时展示小消息、长 RTT、窗口、RTO、Trim 和包长的敏感性。
001–003 的精确源码及结果保留在提交 `f59f05c`；004 补齐了多端点的数据/ACK 共享物理 NIC 模型。

## 安装与运行

使用 uv 恢复 Python 3.10 和锁定依赖：

```bash
git clone git@github.com:Statisticss/AI_Infra_Simulator.git
cd AI_Infra_Simulator
uv sync --python 3.10 --locked
source .venv/bin/activate
python --version
python -m pytest -q
```

运行完整实验（默认结果写入仓库同级 `results/001_loss_recovery/`）：

```bash
OPENBLAS_NUM_THREADS=1 python run_experiment.py --workers 4
python scripts/build_loss_report.py
```

完整实验在初次运行的本机上约需 4 分钟，机器与系统负载不同会影响耗时。
运行器会检查 Python 版本，拒绝使用非 3.10 环境。

可以先运行更小的实验组：

```bash
python run_experiment.py --groups window --output ../results/001_window --workers 4
```

也可选择 `main`、`loss_sweep`、`recovery`、`mtu`、`wan_window`、`controls`。
加 `--resume` 可恢复同一源码与配置下已完成的任务；源码变化时应使用新结果目录。
结果包含 `trials.csv`、`summary.csv`、图表、运行元数据与逐流 JSON 检查点。

新增实验 002 / 003 分别运行；003 必须先用训练种子选参，再用独立种子评估：

```bash
OPENBLAS_NUM_THREADS=1 python run_recovery_study.py --config configs/002_cipu_inspired.yaml --output ../results/002_cipu_inspired --workers 4
OPENBLAS_NUM_THREADS=1 python run_recovery_study.py --config configs/003_scale_across.yaml --output ../results/003_scale_across --phase tune --workers 4
OPENBLAS_NUM_THREADS=1 python run_recovery_study.py --config configs/003_scale_across.yaml --output ../results/003_scale_across --phase eval --workers 4
python scripts/verify_recovery_results.py
python scripts/build_recovery_reports.py
```

003 的评估组包括 `wan_matrix`、`wan_window_only`、`wan_long_flow`、`wan_allreduce`、
`wan_path_skew`、`wan_burst`，可通过 `--groups` 选择，并为局部实验使用独立的输出目录。
评估目录须先包含对应源码/搜索配置的 `selection.json`（先在该目录运行 `--phase tune`）。
`--resume` 只复用严格匹配的检查点。完整批量实验可能需数分钟至数十分钟，取决于机器与并发。
小型汇总、置信区间、全部候选得分和 PNG/SVG 科研图随代码保存。

实验 004 固定参数后执行 10 次主场景验证及 26 次对照，本机完整运行约 6 分钟：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_datacenter_study.py --workers 4
python scripts/verify_datacenter_results.py
python scripts/build_datacenter_report.py
```

原始结果默认写入 `../results/004_datacenter_90pct/`。加 `--resume` 可恢复严格匹配的运行；
单场景可用 `--groups acceptance --output ../results/004_acceptance_only`。
报告目录附带约 324 KiB 的全部 trial 压缩账本，下载后可直接审计而无需再运行仿真：

```bash
python scripts/verify_datacenter_results.py --input reports/004_datacenter_90pct
```

Notebook 使用：

```bash
python -m jupyterlab
```

## 代码结构

```text
configs/001_loss_recovery.yaml        # 参数矩阵、随机种子
src/ai_infra_simulator/
  transport.py                       # 传输、故障、反馈、重传、事件内核
  collectives.py                     # 通信量和 Ring 步骤依赖
  advanced_transport.py              # 共享端口、多 PDC、反馈与尾部候选策略
  datacenter.py                      # 多端点共享时钟、真实 NIC TX/RX、持续 collective 阶段
  experiments/loss_recovery.py        # 批量运行、统计、绘图
  experiments/recovery_study.py       # 002/003 选参、独立评估及逐流保存
  experiments/datacenter_study.py     # 004 冻结场景、独立种子与统计验收
tests/                               # 解析边界与协议状态验证
scripts/build_loss_report.py         # 从结果生成报告
scripts/build_recovery_reports.py    # 002/003 报告与科学图
scripts/verify_recovery_results.py   # 完整任务集、来源、字节及状态预算核验
scripts/build_datacenter_report.py   # 004 报告、容量预算图及原始账本归档
scripts/verify_datacenter_results.py # 004 逐连接/端口守恒及压缩归档核验
reports/001_loss_recovery/            # 可分享的报告、统计表与图
run_experiment.py                    # 无需安装本项目包的入口
run_recovery_study.py                # 002/003 入口
run_datacenter_study.py              # 004 入口
pyproject.toml / uv.lock             # Python 3.10 环境与依赖
```

当前事件内核使用 Python 标准库 `heapq`，仿真过程不依赖 C++、ns-3 或外部服务。
模型参考 UET 规范、IRN 论文与 HPCC 的 ns-3 实现，具体对应关系及简化见实验方法文档。
环境同时提供 SimPy、NumPy、SciPy、pandas、Matplotlib、NetworkX、PyYAML、pytest 和 JupyterLab，
后续独立任务可以选择合适的建模工具。

## 实验约定

- 仿真必须使用 Python 3.10；新增依赖同步记录在 `pyproject.toml` 和 `uv.lock`。
- 每项实验有独立配置与报告，固定种子，记录字节、带宽、时间和 Goodput 的口径。
- 区分规范要求、实现选择、建模假设和实测参数；不将机制模型称为完整协议实现。
- 真实故障丢包、拥塞丢包和 Trim 通知分别建模；重传和控制反馈的成本显式记录。
- 对关键状态机与解析边界做验证，在跨实验稳定后再提取公共模块。
- `.venv/`、原始资料、原始数据和大体量逐流结果不提交；小型结果摘要与图表随实验保存。

原始本地工作区将本仓库置于 `code/`，在同级保留 `docs/`、`data/`、`results/`。
