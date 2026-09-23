#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the ST-generated network on THIS MACHINE and diff it against torch.

    .venv/bin/python Simulation/deploy/stedgeai_validate.py                # 3000 frames
    .venv/bin/python Simulation/deploy/stedgeai_validate.py --frames 500

`stedgeai validate --mode host` compiles the generated C-model for the host and runs it
with the reference network-runtime library - i.e. it executes the very code that went into
the firmware image, no board required. This script feeds it REAL frames from the encoder
corpus and the expected outputs of the whole deployed step, so the report is a per-sample
check of the pipeline that will actually fly:

    o_t(29) | aux(4) | h_in(48)  ->  action(4) | h_out(48) | z(16)

h_in is not a free parameter: it is the GRU state the same graph produced on the previous
frame (the firmware feeds h_out straight back into h_in), so the dataset carries the
sequence rather than resetting the state every sample.

TWO NUMBERS, TWO DIFFERENT THINGS - both are reported:
  * ST generated code vs the ONNX graph it came from  -> conversion fidelity. This should
    be at float32 noise level (~1e-06); anything larger means the tool and the firmware
    copy disagree.
  * ST generated code vs torch                        -> what the vehicle does vs what was
    trained. This INCLUDES the known cost of the `--compat` export (tanh GELU instead of
    erf: z 5.26e-04, action 1.58e-04), so it is checked against those measured numbers,
    not against 0.

The verdict is a PASS only if both hold; the numbers are printed either way.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_policy import _frame_stream, resolve_model  # noqa: E402
from policy_host_check import torch_reference  # noqa: E402

STEDGEAI = os.environ.get(
    "STEDGEAI", "/Applications/ST/STEdgeAI/4.0/Utilities/mac/stedgeai")
DEFAULT_ONNX = os.path.join(_THIS_DIR, "models", "policy_step_stedgeai.onnx")
DATASET_DIR = os.path.join(_THIS_DIR, "st_ai_validate")
OUT_DIR = os.path.join(DATASET_DIR, "out")
WS_DIR = os.path.join(_THIS_DIR, "st_ai_ws")

# conversion fidelity: the tool's C-model vs the ONNX it was generated from
CONV_Z_TOL = 1.0e-4
CONV_ACTION_TOL = 1.0e-4
# end-to-end: the tool's C-model vs torch, i.e. the measured cost of the --compat export
COMPAT_Z_TOL = 2.0e-3
COMPAT_ACTION_TOL = 5.0e-4


def build_dataset(frames: np.ndarray, onnx_path: str,
                  ref_ff: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Sequential run of the deployed ONNX graph: the h_in the firmware would feed.

    `ref_ff` is the actor's feed-forward block (a separate graph input, so it reaches the
    actor without entering the GRU). Defaults to the shared synthetic stream so every
    consumer of this helper feeds the graph the SAME block.
    """
    import onnxruntime as ort

    from export_policy import ref_ff_stream

    if ref_ff is None:
        ref_ff = ref_ff_stream(len(frames))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    n = len(frames)
    h = np.zeros((1, 48), dtype=np.float32)
    h_in = np.zeros((n, 48), dtype=np.float32)
    action = np.zeros((n, 4), dtype=np.float32)
    h_out = np.zeros((n, 48), dtype=np.float32)
    z = np.zeros((n, 16), dtype=np.float32)
    for i in range(n):
        h_in[i] = h[0]
        a, h, zz = sess.run(["action", "h_out", "z"],
                            {"o_t": frames[i:i + 1, :29], "aux": frames[i:i + 1, 29:],
                             "ref_ff": ref_ff[i:i + 1], "h_in": h})
        action[i] = a[0]
        h_out[i] = h[0]
        z[i] = zz[0]
    return {"h_in": h_in, "action": action, "h_out": h_out, "z": z}


def _metrics_from_report(text: str) -> Dict[str, float]:
    """Pull the headline error metrics out of the tool's report (best effort)."""
    out: Dict[str, float] = {}
    for name in ("rmse", "mae", "l2r", "snr", "cos"):
        m = re.search(rf"^\s*{name}\s*[:=]\s*([-+0-9.eE]+)", text, re.MULTILINE)
        if m:
            try:
                out[name] = float(m.group(1))
            except ValueError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Host-validate the ST generated policy net")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--model", default="latest")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--stedgeai", default=STEDGEAI)
    ap.add_argument("--keep", action="store_true", help="keep the dataset/report files")
    args = ap.parse_args()

    if not os.path.isfile(args.stedgeai):
        print(f"ERROR: stedgeai not found at {args.stedgeai}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.onnx):
        print(f"ERROR: {args.onnx} missing - run export_onnx.py --compat", file=sys.stderr)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    # -- dataset: real frames + the reference outputs for the whole deployed step --------
    frames, source = _frame_stream(args.frames)
    ref = build_dataset(frames, args.onnx)
    npz = os.path.join(DATASET_DIR, "policy_val_dataset.npz")
    np.savez(npz,
             m_inputs_1=frames[:, :29], m_inputs_2=frames[:, 29:], m_inputs_3=ref["h_in"],
             m_outputs_1=ref["action"], m_outputs_2=ref["h_out"], m_outputs_3=ref["z"])
    print(f"frames  : {len(frames)} from {source}")
    print(f"dataset : {os.path.relpath(npz, _PROJECT_ROOT)}")

    # -- the tool ------------------------------------------------------------------------
    cmd = [args.stedgeai, "validate", "--model", args.onnx, "--target", "stm32f4",
           "--mode", "host", "-vi", npz, "--output", OUT_DIR, "--workspace", WS_DIR]
    print(f"run     : {' '.join(cmd[-9:])}\n")
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-25:]
    print("--- stedgeai validate (tail) ---")
    print("\n".join(tail))
    if proc.returncode != 0:
        print("\nERROR: validate failed", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return 1

    # -- the predictions the tool made, versus torch -------------------------------------
    io_npz = os.path.join(OUT_DIR, "network_val_io.npz")
    if not os.path.isfile(io_npz):
        candidates = [f for f in os.listdir(OUT_DIR) if f.endswith(".npz")]
        if not candidates:
            print("\nERROR: no prediction file was written by validate", file=sys.stderr)
            return 1
        io_npz = os.path.join(OUT_DIR, sorted(candidates)[0])

    with np.load(io_npz) as d:
        keys = set(d.files)
        got = {k: np.asarray(d[k], dtype=np.float32) for k in ("c_outputs_1", "c_outputs_2",
                                                               "c_outputs_3") if k in keys}
        got_m = {k: np.asarray(d[k], dtype=np.float32) for k in ("m_outputs_1", "m_outputs_2",
                                                                 "m_outputs_3") if k in keys}
    if len(got) != 3:
        print(f"\nERROR: {os.path.basename(io_npz)} has {sorted(keys)}; "
              "expected c_outputs_1..3", file=sys.stderr)
        return 1

    tz, ta = torch_reference(frames, resolve_model(args.model), os.path.join(
        _PROJECT_ROOT, "logs", "encoder_gru.pt"))

    def _diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        d = np.abs(a - b)
        return float(np.max(d)), float(np.mean(d))

    conv_z = _diff(got["c_outputs_3"], ref["z"])
    conv_a = _diff(got["c_outputs_1"], ref["action"])
    end_z = _diff(got["c_outputs_3"], tz)
    end_a = _diff(got["c_outputs_1"], ta)

    print("\n--- conversion fidelity (ST generated code vs the ONNX it came from) ---")
    print(f"  z      : max {conv_z[0]:.3e}  mean {conv_z[1]:.3e}   (tol {CONV_Z_TOL:.0e})")
    print(f"  action : max {conv_a[0]:.3e}  mean {conv_a[1]:.3e}   (tol {CONV_ACTION_TOL:.0e})")
    print("--- end to end (ST generated code vs torch, includes the --compat GELU) ---")
    print(f"  z      : max {end_z[0]:.3e}  mean {end_z[1]:.3e}   (tol {COMPAT_Z_TOL:.0e})")
    print(f"  action : max {end_a[0]:.3e}  mean {end_a[1]:.3e}   (tol {COMPAT_ACTION_TOL:.0e})")
    sat = float(np.mean(np.abs(got["c_outputs_1"]) >= 0.999))
    print(f"  clipped actions: {100.0 * sat:.1f}%  (torch: {100.0 * float(np.mean(np.abs(ta) >= 0.999)):.1f}%)")

    report = os.path.join(OUT_DIR, "network_validate_report.txt")
    if os.path.isfile(report):
        with open(report, "r", encoding="utf-8", errors="replace") as fh:
            m = _metrics_from_report(fh.read())
        if m:
            print("--- tool metrics ---")
            print("  " + "  ".join(f"{k}={v:.3e}" for k, v in sorted(m.items())))

    ok_conv = conv_z[0] <= CONV_Z_TOL and conv_a[0] <= CONV_ACTION_TOL
    ok_end = end_z[0] <= COMPAT_Z_TOL and end_a[0] <= COMPAT_ACTION_TOL
    print()
    print("VERDICT:", "PASS" if (ok_conv and ok_end) else "FAIL",
          f"(conversion {'ok' if ok_conv else 'OUT OF TOLERANCE'}, "
          f"end to end {'ok' if ok_end else 'OUT OF TOLERANCE'})")
    if not args.keep:
        pass  # the dataset and the report are cheap to regenerate and gitignored
    return 0 if (ok_conv and ok_end) else 1


if __name__ == "__main__":
    raise SystemExit(main())
