#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify the GENERATED ST network on this Mac, against torch, over real frames.

    .venv/bin/python Simulation/deploy/stedgeai_host_check.py             # 3000 frames
    .venv/bin/python Simulation/deploy/stedgeai_host_check.py --frames 500

This is the ST-backend twin of `policy_host_check.py`: it compiles the SAME
`policy_net_stedgeai.c` + the SAME generated `network.c`/`network_data.c` that go into
the firmware image, runs them over real frames from the encoder corpus, and diffs the
output against torch. Nothing is re-implemented here - the file that flies is the file
that is checked.

WHY IT IS NOT `stedgeai validate --mode host`
---------------------------------------------
That would be the natural tool, and it gets all the way to compiling the model, but its
final step cannot run on an Apple-silicon Mac:

  * ST ships the host runtime as **x86_64-only** (`libruntime.a`, `libst_cmsis_nn.a`,
    `libcmsis-nn.a` in the pack's inspector workspace);
  * an x86_64 compiler cannot run here either - the Command Line Tools' `libxcrun.dylib`
    has lost its x86_64 slice, so `arch -x86_64 clang` dies inside `xcrun`
    (and the CLI itself runs under Rosetta, which passes that preference down to its
    children - that is the "E103: unable to build the shared library" error).

A NATIVE clang can, however, CROSS-compile to x86_64 (`-arch x86_64`), and the resulting
binary runs happily under Rosetta. So the harness is built that way and the runtime libs
link as-is. If you later install full Xcode, `stedgeai validate --mode host` should work
too - `Simulation/deploy/stedgeai_validate.py` drives exactly that and is kept for it.

WHAT IS CHECKED
---------------
  * ST generated code vs the ONNX graph it was generated from -> conversion fidelity
    (expected at float32 noise level)
  * ST generated code vs torch                                -> end to end, which
    INCLUDES the known cost of the `--compat` export (tanh GELU: z 5.26e-04,
    action 1.58e-04). Checked against those measured numbers, not against zero.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Dict, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR, _THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_policy import _frame_stream, ref_ff_stream, resolve_model  # noqa: E402
from policy_host_check import torch_reference  # noqa: E402
from stedgeai_validate import build_dataset  # noqa: E402

APP = os.path.join(_THIS_DIR, "app_policy_controller")
SRC = os.path.join(APP, "src")
GEN = os.path.join(SRC, "generated", "stedgeai")
TEST = os.path.join(APP, "test")
WORK = os.path.join(_THIS_DIR, "st_ai_validate")

DEFAULT_STEDGEAI_DIR = "/Applications/ST/STEdgeAI/4.0"
HOST_LIBS = ("Utilities/mac/targets/common/EmbedNets/tools/inspector/workspace/lib/static")

CONV_Z_TOL = 1.0e-4
CONV_ACTION_TOL = 1.0e-4
COMPAT_Z_TOL = 2.0e-3
COMPAT_ACTION_TOL = 5.0e-4


def _find_libs(stedgeai_dir: str) -> str:
    lib = os.path.join(stedgeai_dir, HOST_LIBS)
    for name in ("libruntime.a", "libst_cmsis_nn.a", "libcmsis-nn.a"):
        if not os.path.isfile(os.path.join(lib, name)):
            raise SystemExit(
                f"ERROR: {name} not found in {lib}\n"
                "       the ST pack installs its host runtime there; pass --stedgeai-dir")
    return lib


def build_harness(stedgeai_dir: str, verbose: bool = False) -> str:
    clang = shutil.which("clang") or "/usr/bin/clang"
    lib = _find_libs(stedgeai_dir)
    inc = os.path.join(stedgeai_dir, "Middlewares", "ST", "AI", "Inc")
    binary = os.path.join(WORK, "stedgeai_host_check")

    missing = [p for p in (os.path.join(GEN, "network.c"), os.path.join(GEN, "network_data.c"),
                           os.path.join(SRC, "policy_net_stedgeai.c"),
                           os.path.join(TEST, "stedgeai_host_check.c")) if not os.path.isfile(p)]
    if missing:
        raise SystemExit("ERROR: missing sources:\n  " + "\n  ".join(missing)
                         + "\n(run Simulation/deploy/install_stedgeai.py first)")

    # -arch x86_64: the ST host runtime is x86_64-only (see the module docstring). A native
    # clang cross-compiles happily; the result runs under Rosetta.
    cmd = [clang, "-std=c99", "-O2", "-arch", "x86_64", "-DHAS_LOG=6",
           f"-I{SRC}", f"-I{GEN}", f"-I{inc}",
           os.path.join(SRC, "policy_net_stedgeai.c"),
           os.path.join(GEN, "network.c"),
           os.path.join(GEN, "network_data.c"),
           os.path.join(TEST, "stedgeai_host_check.c"),
           "-L", lib, "-lruntime", "-lst_cmsis_nn", "-lcmsis-nn", "-lm",
           "-o", binary]
    if verbose:
        print("cc      :", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("ERROR: harness build failed\n" + proc.stderr[-4000:])
    if proc.stderr.strip():
        print("compiler notes:", proc.stderr.strip()[:500])
    return binary


def main() -> int:
    ap = argparse.ArgumentParser(description="Host check of the ST generated policy net")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--model", default="latest")
    ap.add_argument("--onnx", default=os.path.join(_THIS_DIR, "models",
                                                   "policy_step_stedgeai.onnx"))
    # Same flag and default as policy_host_check.py, so the two host checks are driven the
    # same way. This used to be hardcoded, which silently broke the check the moment the
    # encoder checkpoint was not at that exact path.
    ap.add_argument("--encoder", default=os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt"))
    ap.add_argument("--stedgeai-dir", default=DEFAULT_STEDGEAI_DIR)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)

    frames, source = _frame_stream(args.frames)
    n = len(frames)
    ff = ref_ff_stream(n)
    fin = os.path.join(WORK, "frames.bin")
    fout = os.path.join(WORK, "out.bin")
    # ROW FORMAT: [frame(33) | ref_ff(3)], the same layout host_check.c reads - one row is a
    # complete policy step, and the harness splits it back apart exactly as the firmware does.
    np.concatenate([frames, ff], axis=1).astype(np.float32).tofile(fin)

    binary = build_harness(args.stedgeai_dir, args.verbose)
    if args.verbose:
        print("binary  :", binary, "(x86_64, runs under Rosetta)")
    proc = subprocess.run([binary, fin, fout, str(n)], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("ERROR: the harness failed")
    print(f"frames  : {n} from {source}")
    print(f"harness : {proc.stderr.strip()}")

    raw = np.fromfile(fout, dtype=np.float32)
    if raw.size != n * 20:
        raise SystemExit(f"ERROR: expected {n * 20} floats from the harness, got {raw.size}")
    raw = raw.reshape(n, 20)
    st_action, st_z = raw[:, :4], raw[:, 4:]

    ref = build_dataset(frames, args.onnx, ref_ff=ff)             # the ONNX graph it came from
    tz, ta = torch_reference(frames, resolve_model(args.model),
                             args.encoder, ref_ff=ff)

    def diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        d = np.abs(a - b)
        return float(np.max(d)), float(np.mean(d))

    conv_z = diff(st_z, ref["z"])
    conv_a = diff(st_action, ref["action"])
    end_z = diff(st_z, tz)
    end_a = diff(st_action, ta)

    print("\n--- conversion fidelity (generated code vs the ONNX it came from) ---")
    print(f"  z      : max {conv_z[0]:.3e}  mean {conv_z[1]:.3e}   (tol {CONV_Z_TOL:.0e})")
    print(f"  action : max {conv_a[0]:.3e}  mean {conv_a[1]:.3e}   (tol {CONV_ACTION_TOL:.0e})")
    print("--- end to end (generated code vs torch, includes the --compat GELU) ---")
    print(f"  z      : max {end_z[0]:.3e}  mean {end_z[1]:.3e}   (tol {COMPAT_Z_TOL:.0e})")
    print(f"  action : max {end_a[0]:.3e}  mean {end_a[1]:.3e}   (tol {COMPAT_ACTION_TOL:.0e})")
    sat_st = 100.0 * float(np.mean(np.abs(st_action) >= 0.999))
    sat_to = 100.0 * float(np.mean(np.abs(ta) >= 0.999))
    print(f"  clipped actions: {sat_st:.1f}%  (torch {sat_to:.1f}%)")

    ok_conv = conv_z[0] <= CONV_Z_TOL and conv_a[0] <= CONV_ACTION_TOL
    ok_end = end_z[0] <= COMPAT_Z_TOL and end_a[0] <= COMPAT_ACTION_TOL
    print()
    print("VERDICT:", "PASS" if (ok_conv and ok_end) else "FAIL",
          f"(conversion {'ok' if ok_conv else 'OUT OF TOLERANCE'}, "
          f"end to end {'ok' if ok_end else 'OUT OF TOLERANCE'})")
    return 0 if (ok_conv and ok_end) else 1


if __name__ == "__main__":
    raise SystemExit(main())
