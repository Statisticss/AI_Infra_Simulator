# AI_Infra_Simulator

用于 AI Infrastructure（AI 基础设施）相关研究与实验的 Python 仿真项目。

当前阶段：完成本地 Python 3.10 基础环境搭建，并通过更新此 README 验证 GitHub 同步流程。

## 研究方向

后续可逐步添加以下仿真模块：

- GPU / 加速器资源分配、任务队列与调度。
- 分布式训练和推理中的通信开销与网络拓扑。
- 计算、网络和存储资源的性能建模。
- 实验指标统计、结果分析与可视化。

以上为规划方向，目前尚未实现具体仿真模型。

## 基础环境

- Python：3.10 系列，本地环境使用 3.10.19。
- 环境与依赖管理：uv、项目独立的 `.venv`。
- 离散事件仿真：SimPy。
- 数值计算与数据分析：NumPy、SciPy、pandas。
- 网络拓扑与绘图：NetworkX、Matplotlib。
- 实验配置：PyYAML。
- 开发工具：JupyterLab、ipykernel、pytest。

依赖版本在本地 `uv.lock` 中锁定，Python 版本由 `.python-version` 和 `pyproject.toml` 约束。

## 本地工作区

```text
AI-Infra-Sim/
├── code/            # 本 GitHub 仓库和 Python 仿真代码
│   ├── src/         # 模型与公共模块（预留）
│   ├── examples/    # 实验示例（预留）
│   ├── notebooks/   # 交互式分析（预留）
│   ├── configs/     # 实验参数（预留）
│   └── .venv/       # Python 3.10 虚拟环境
├── docs/            # 本地资料与笔记
├── data/            # 输入数据
└── results/         # 仿真输出
```

在已配置的本地工作区中，从 `AI-Infra-Sim` 目录启动：

```bash
cd code
source .venv/bin/activate
python --version
python -m jupyterlab
```

Notebook 内核选择 `Python 3.10 (AI Infra)`。

退出虚拟环境：

```bash
deactivate
```

## 同步状态

此次仅上传 README 进行同步测试。依赖配置、锁文件和预留目录已在本地准备，
后续随仿真代码逐步提交；当前仅克隆远程仓库还不能重建本地环境。

虚拟环境、原始数据和仿真结果不纳入代码仓库。
