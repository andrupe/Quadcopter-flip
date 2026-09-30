"""
Root entrypoint for the vectorized MuJoCo MJX quadcopter PPO training run.

The implementation lives in ``Simulation/train_mjx.py``; this is only a convenience shim so
the run can be started from the repository root.

Usage:
  .venv/bin/python train_mjx.py --smoke
  .venv/bin/python train_mjx.py --total-timesteps 30000000
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_THIS_DIR, "Simulation")
for _p in [_THIS_DIR, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.train_mjx import main

if __name__ == "__main__":
    raise SystemExit(main())
