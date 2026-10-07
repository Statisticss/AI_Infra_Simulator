#!/usr/bin/env python3
"""Run independent CIPU-inspired and scale-across studies in Python 3.10."""
# 实验 002/003 入口；--config 选择实验，003 用 --phase tune/eval 区分选参与验证。
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ai_infra_simulator.experiments.recovery_study import main

if __name__ == "__main__":
    main()
