"""
Compile and run the host check for `policy_guard.c` - the app's failsafe descent and its
estimator plausibility gate.

WHY THIS IS A PYTHON DRIVER AND NOT A `make test`
-------------------------------------------------
Same reason as `policy_host_check.py` / `stedgeai_host_check.py`: the app itself cannot be
host-compiled (FreeRTOS, the STM32 HAL, the stock controller), so the safety logic that
decides whether a real flight is abandoned lives in `src/policy_guard.c`, which has no
firmware dependencies at all, and `test/guard_check.c` drives it directly. There is no
hardware in the loop and no firmware build.

Run:  .venv/bin/python -u Simulation/deploy/guard_host_check.py
      CC=gcc .venv/bin/python -u Simulation/deploy/guard_host_check.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

_THIS = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.join(_THIS, "app_policy_controller")


def main(argv: list[str]) -> int:
    cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
    if cc is None:
        print("no C compiler found - set CC=/path/to/clang")
        return 2

    sources = [os.path.join(_APP, "src", "policy_guard.c"),
               os.path.join(_APP, "test", "guard_check.c")]
    for path in sources:
        if not os.path.isfile(path):
            print(f"missing {path}")
            return 2

    out = os.path.join(tempfile.gettempdir(), "quad_guard_check.bin")
    cmd = [cc, "-O1", "-std=c11", "-Wall", "-Wextra", "-Werror",
           "-I", os.path.join(_APP, "src"),
           "-o", out, *sources, "-lm"]
    print(" ".join(cmd))
    build = subprocess.run(cmd, capture_output=True, text=True)
    if build.stdout.strip():
        print(build.stdout)
    if build.stderr.strip():
        print(build.stderr, file=sys.stderr)
    if build.returncode != 0:
        print("COMPILE FAILED")
        return 1

    return subprocess.run([out], text=True).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
