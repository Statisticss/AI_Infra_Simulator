#!/usr/bin/env python3
"""Run the independent loss-recovery experiment using the local Python 3.10 env."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_infra_simulator.experiments.loss_recovery import main

if __name__ == "__main__":
    main()
