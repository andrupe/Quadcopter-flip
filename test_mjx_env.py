"""
Root entrypoint for the MJX environment sanity + throughput benchmark.

Usage:
  .venv/bin/python test_mjx_env.py
  .venv/bin/python test_mjx_env.py --batches 256,1024
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_THIS_DIR, "Simulation")
for _p in [_THIS_DIR, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.test_mjx_env import main

if __name__ == "__main__":
    sys.exit(main())
