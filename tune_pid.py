#!/usr/bin/env python3
"""Quadcopter Inner-Loop Rate PID Finetuning Runner.

ONE-CLICK FINETUNING WORKFLOW:
  * In VS Code: Open this file and click "Run Python File" (Play button ▶) in the top-right.
  * In Terminal: Run `.venv/bin/python tune_pid.py`
  * Flags:
      --timeout 600            : Target optimization time budget in seconds (default: 600s / 10 minutes)
      --dr 0.70                : Domain Randomization intensity (0.0=nominal, 0.7=realistic, 1.0=full stress)
      --workers 8              : Number of parallel CPU workers (default: auto, utilizes available cores)
      --apply                  : Automatically update default gains in rate_pid.py when tuning completes
      --quick                  : Quick 60-second test run to verify pipeline
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure repository root and Simulation directories are on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_ROOT, "Simulation")
for _p in [_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.tune_rate_pid import tune_rate_pid, DEFAULT_TIMEOUT_SECONDS, DEFAULT_DR_LEVEL

# ======================================================================================
# TUNING CONFIGURATION (Edit options directly below, then click Run ▶ in VS Code)
# ======================================================================================
DEFAULT_MODEL: str = "latest"          # "latest" auto-selects newest checkpoint in logs/
TIMEOUT_SECONDS: float = 600.0         # 10 minutes (600s) default runtime budget
DR_LEVEL: float = 0.70                 # Domain Randomization level [0.0, 1.0]
EPISODES_PER_EVAL: int = 4             # Evaluation flights per candidate gain vector
BENCHMARK_SEEDS: int = 25              # Test flights for final paired comparison report
PARALLEL_WORKERS: int | None = None    # None = auto-detect CPU cores (up to 8)
APPLY_TUNED_GAINS: bool = False        # Set True to overwrite default gains in rate_pid.py
QUICK_SMOKE_TEST: bool = False         # Set True for a rapid 60-second trial
# ======================================================================================


def main():
    parser = argparse.ArgumentParser(description="Quadcopter Inner-Loop Rate PID Finetuning")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model checkpoint path or 'latest'")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="Time budget in seconds (default: 600s / 10 mins)")
    parser.add_argument("--dr", type=float, default=DR_LEVEL, help="Domain Randomization intensity [0.0, 1.0]")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_EVAL, help="Episodes per candidate during search")
    parser.add_argument("--benchmark-seeds", type=int, default=BENCHMARK_SEEDS, help="Episodes for final paired comparison")
    parser.add_argument("--workers", type=int, default=PARALLEL_WORKERS, help="Parallel worker processes (default: auto)")
    parser.add_argument("--apply", action="store_true", default=APPLY_TUNED_GAINS, help="Apply best gains to rate_pid.py default values")
    parser.add_argument("--quick", action="store_true", default=QUICK_SMOKE_TEST, help="Run rapid 60s smoke test")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save logs, JSON report, and plots")
    args = parser.parse_args()

    tune_rate_pid(
        model_name=args.model,
        timeout=args.timeout,
        dr_level=args.dr,
        episodes=args.episodes,
        benchmark_seeds=args.benchmark_seeds,
        workers=args.workers,
        quick=args.quick,
        apply_gains=args.apply,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
