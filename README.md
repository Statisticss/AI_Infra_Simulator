# AI_Infra_Simulator

使用 **Python 3.10** 进行 AI 基础设施传输研究。从可独立运行、可验证的实验开始，
逐步积累共同的事件、链路、传输和 collective 模型。

研究范围包括 Scale-out（RoCEv2、InfiniBand）、Scale-up（NVLink、UALink）、
Scale-across（长距离 RDMA），以及 PCIe、CXL 等总线传输。
当前实现范围见下表；其余协议尚未实现。

## 版本与结果对应关系

| 版本 | 内容 | 使用方式 |
|---|---|---|
| [sim-004-results](https://github.com/Statisticss/AI_Infra_Simulator/tree/sim-004-results)（提交 `4807b56`） | 已验证的仿真代码、报告、图表、36 个任务的完整压缩原始结果 | 对已有结果做严格源码审计，或复现原始结果 |
| [sim-004-zh-comments](https://github.com/Statisticss/AI_Infra_Simulator/tree/sim-004-zh-comments) | 为全部 Python 文件补充中文注释，并更新本 README | 阅读实现、继续研究、在新输出目录运行实验 |

中文注释版保留原始实验参数、依赖锁和全部已发表结果。23 个 Python 文件与结果版进行了
抽象语法树及去注释后的 token 比对，可执行代码一致；已有 59 项测试通过。
协议状态机、定时器、随机种子、带宽计量及验收规则均与结果版相同。

源码摘要按文件原始字节计算，**新增注释也会改变 SHA-256**。
已有 `manifest.json`、`freeze.json` 和检查点继续记录实际运行时的原始摘要；
历史结果使用结果版审计，注释版使用新输出目录。下面分别给出命令。

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

### 用中文注释版运行实验 004

完整矩阵包含 10 次主场景验证和 26 次对照，本机约需 6 分钟。
使用独立目录保存注释版的新结果，避免与原始源码摘要混用：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_datacenter_study.py \
  --output ../results/004_datacenter_90pct_zh --workers 4
python scripts/verify_datacenter_results.py --input ../results/004_datacenter_90pct_zh
```

运行器会检查 Python 版本，拒绝非 3.10 环境。
同一源码、配置、任务范围及 Python 版本下，加 `--resume` 可续跑。
小型试跑可以只运行 5 个并发消息传输样本：

```bash
OPENBLAS_NUM_THREADS=1 python run_datacenter_study.py --groups permutation \
  --output ../results/004_permutation_zh --workers 2
```

`--groups` 还支持 `acceptance`、`lossless_reference`、`baseline_recovery`、
`medium_tensor`、`small_tensor`、`higher_rtt`。单组输出使用独立目录；
`verify_datacenter_results.py` 审核的是配置中的完整矩阵，不接受缺少其他组的部分结果。
只有主场景组参与 90% 目标验收，对照组不达标不会被剔除。

完整运行后，若需要根据新结果重新发布报告：

```bash
python scripts/build_datacenter_report.py --input ../results/004_datacenter_90pct_zh
```

该命令更新 `reports/004_datacenter_90pct/` 内的报告、图表和归档。
本次中文注释提交保留了结果版原有报告，没有重新生成或改写测量记录。

### 直接审计已经发表的结果

报告目录附带约 324 KiB 的 `raw_trials.jsonl.gz`，无需重跑仿真即可检查全部样本。
在已经激活 Python 3.10 环境的仓库根目录执行以下命令，建立结果版的独立检出目录：

```bash
git fetch origin --tags
git worktree add --detach ../AI_Infra_Simulator-results sim-004-results
python ../AI_Infra_Simulator-results/scripts/verify_datacenter_results.py \
  --input ../AI_Infra_Simulator-results/reports/004_datacenter_90pct
```

如果该检出目录已经存在，直接执行最后一条审计命令即可。
审计核验全部任务、源码摘要、每个包的接收与确认状态、字节守恒、端口容量、CSV 和置信区间。
不要将旧记录中的摘要替换为当前摘要来绕过检查；只改注释后的摘要不匹配是正常的来源保护。

### 运行实验 001–003

001 比较 GBN 和选择性恢复，可运行完整矩阵或用 `--groups window` 等组名缩小范围：

```bash
OPENBLAS_NUM_THREADS=1 python run_experiment.py --output ../results/001_loss_recovery_zh --workers 4
python scripts/build_loss_report.py --input ../results/001_loss_recovery_zh --output ../results/001_loss_recovery_zh/report
```

002 评估 CIPU 公开方向启发的端点恢复候选；003 必须先用训练种子选参，再用独立种子评估：

```bash
OPENBLAS_NUM_THREADS=1 python run_recovery_study.py --config configs/002_cipu_inspired.yaml \
  --output ../results/002_cipu_inspired_zh --workers 4
OPENBLAS_NUM_THREADS=1 python run_recovery_study.py --config configs/003_scale_across.yaml \
  --output ../results/003_scale_across_zh --phase tune --workers 4
OPENBLAS_NUM_THREADS=1 python run_recovery_study.py --config configs/003_scale_across.yaml \
  --output ../results/003_scale_across_zh --phase eval --workers 4
```

输出包含 `trials.csv`、`summary.csv`、`manifest.json` 和逐流检查点。
003 的 `selection.json` 与源码和搜索配置绑定；修改参数或注释后应在新目录重新选参。
001–003 的历史报告及专用审计按原报告的目录约定使用，对应精确源码提交 `f59f05c`。

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

## 中文代码阅读指南

建议先阅读实验 004 的完整调用链，再回看底座和早期实验：

1. [配置](configs/004_datacenter_90pct.yaml)：明确业务大小、网络参数、恢复策略和种子。
2. [运行器](src/ai_infra_simulator/experiments/datacenter_study.py)：`expand_jobs` 展开任务，`run_job` 执行一次仿真，`summarize` 和 `acceptance` 汇总验收。
3. [共享网络](src/ai_infra_simulator/datacenter.py)：`Fabric` 管理持续事件时钟与阶段，`NIC` 负责发送/接收排队，`FabricConnection` 把协议接到物理资源。
4. [恢复策略](src/ai_infra_simulator/advanced_transport.py)：`PortConnection` 实现重叠 SACK、区间补充反馈和尾部副本。
5. [协议底座](src/ai_infra_simulator/transport.py)：`PacketSimulation` 实现发包、信道丢失、接收、确认和超时；`TransportConfig` 定义参数和单位。
6. [审计脚本](scripts/verify_datacenter_results.py)及[测试](tests/test_datacenter.py)：理解每个字节、时延和资源上限如何核验。

一次包传输的主要调用顺序是：

```text
Fabric.start_phase → 连接请求发送 → NIC.send → 信道判定丢失/到达
→ 对端 NIC.receive_frame/finish_receive → 连接更新接收位图
→ ACK 快照进入对端发送队列 → 原发送端收到 ACK 并更新确认状态
→ 继续发送/重传 → 本阶段全部 rank 确认后启动下一阶段
```

注释特别说明了以下容易混淆的量：

| 名称 | 含义与单位 |
|---|---|
| PSN、`next_seq` | 报文序号、下一份尚未首次发送的报文序号 |
| `expected` / `cack` / `base` | 接收端实际首个缺口 / 已报告连续确认前缀 / 发送端合并正确认后的连续前缀 |
| SACK | 选择确认位图；只合并正确认，零位不撤销已经确认的包 |
| BDP、`window_bdp` | 带宽时延积及其窗口倍率；窗口限制的是 PSN 跨度 |
| `now`、`tx_free`、`rx_free` | 当前事件时间、发送资源释放时间、接收资源释放时间，单位 ns |
| `rtt_us`、`ack_delay_us` | 名义 RTT、ACK 合并等待，输入单位 us |
| `bandwidth_gbps` | 端口 MAC 服务速率，单位 Gbit/s；1 bit/ns = 1 Gbit/s |
| `data_loss`、`ack_loss` | 每次发送尝试的丢失概率；重传和反馈自身也可能丢失 |
| `tail_copies` | 每轮尾部修复总副本数，包含该轮第一次重传，不是整条流的复制次数 |
| `duration_ms` / `wall_seconds` | 模拟的完整通信时间 / Python 程序实际执行耗时，不能混用 |

对于 N 个 rank、每 rank 张量大小 S 的 Ring AllReduce：

```text
必要网络字节/rank = S × 2(N−1)/N
algorithm bandwidth = S × 8 / 完成时间
bus bandwidth = 必要网络字节/rank × 8 / 完成时间
Goodput = bus bandwidth / 端口速率
```

重传、ACK 和协议头不增加 Goodput 分子，但全部消耗时间及物理带宽。
实验 004 的验收要求至少 10 个独立样本，每次都 ≥90%，且均值 95% 置信区间下界也 ≥90%。

## 实验约定

- 仿真必须使用 Python 3.10；新增依赖同步记录在 `pyproject.toml` 和 `uv.lock`。
- 每项实验有独立配置与报告，固定种子，记录字节、带宽、时间和 Goodput 的口径。
- 区分规范要求、实现选择、建模假设和实测参数；不将机制模型称为完整协议实现。
- 真实故障丢包、拥塞丢包和 Trim 通知分别建模；重传和控制反馈的成本显式记录。
- 对关键状态机与解析边界做验证，在跨实验稳定后再提取公共模块。
- `.venv/`、原始资料、原始数据和大体量逐流结果不提交；小型结果摘要与图表随实验保存。

原始本地工作区将本仓库置于 `code/`，在同级保留 `docs/`、`data/`、`results/`。
