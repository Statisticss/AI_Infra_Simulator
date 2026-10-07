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

**[阅读实验 001 报告](reports/001_loss_recovery/REPORT.md)** ·
[模型假设与原始依据](reports/001_loss_recovery/METHOD.md) ·
[全部配置](configs/001_loss_recovery.yaml)

代表性结果：8 rank、每 rank 1 GiB、400 Gbit/s、10 us RTT、数据及反馈均独立丢失 5% 时，
传统 RC/GBN 模型的 Goodput 为 **12.95%**，UET RUD 风格机制模型为 **90.55%**。
这是指定参数下的机制仿真结果，**不是完整 UET 实现、CIPU 实测或对真实 NCCL 系统的性能承诺**。
报告同时展示小消息、长 RTT、窗口、RTO、Trim 和包长的敏感性。

![实验 001 AllReduce 结果](reports/001_loss_recovery/figures/allreduce.png)

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
  experiments/loss_recovery.py        # 批量运行、统计、绘图
tests/                               # 解析边界与协议状态验证
scripts/build_loss_report.py         # 从结果生成报告
reports/001_loss_recovery/            # 可分享的报告、统计表与图
run_experiment.py                    # 无需安装本项目包的入口
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
