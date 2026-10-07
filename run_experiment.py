#!/usr/bin/env python3
"""Run the independent loss-recovery experiment using the local Python 3.10 env."""
# 实验 001 命令行入口；将本地 src 加入导入路径，无需安装项目包即可运行。
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_infra_simulator.experiments.loss_recovery import main

if __name__ == "__main__":
    main()
