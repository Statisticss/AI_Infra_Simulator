#!/usr/bin/env python3
"""Run study 004 using Python 3.10."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ai_infra_simulator.experiments.datacenter_study import main

if __name__ == "__main__":
    main()
