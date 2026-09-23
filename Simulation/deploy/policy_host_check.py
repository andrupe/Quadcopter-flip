# -*- coding: utf-8 -*-
"""
Compile the deployed C network on the Mac and diff it against the torch deployment path.

WHY
---
`export_policy.py --verify` proves the EXPORT (weights, layout, standardization, gate
order, action clipping) is right by reconstructing it in numpy. This script closes the
remaining gap: it builds `app_policy_controller/src/policy_net.c` with clang - the same
portable source the firmware compiles - runs it over a real frame stream and compares
BOTH the latent z and the action against torch's deployment path (`encoder.step` +
`model.predict`). If these agree, the only remaining differences on the drone are the
OBSERVATION ASSEMBLY and the units, which is exactly where the debugging effort belongs.

USAGE
-----
    .venv/bin/python Simulation/deploy/policy_host_check.py
    .venv/bin/python Simulation/deploy/policy_host_check.py --frames 6000 --keep
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from typing import Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_policy import (  # noqa: E402
    ACTOR_FRAME_DIM, REF_FF_DIM, _frame_stream, actor_forward, encoder_head, extract_actor,
    extract_encoder, gru_step, ref_ff_stream, resolve_model, standardize_frame,
)

APP_DIR = os.path.join(_SIM_DIR, "deploy", "app_policy_controller")
SRC = os.path.join(APP_DIR, "src")
TEST = os.path.join(APP_DIR, "test")

Z_TOL = 2.0e-5
ACTION_TOL = 2.0e-5


def compile_host_check(out_dir: str) -> str:
    exe = os.path.join(out_dir, "host_check")
    cmd = [
        "clang", "-O2", "-std=c11", "-Wall", "-Wextra",
        f"-I{SRC}",
        "-o", exe,
        os.path.join(TEST, "host_check.c"),
        os.path.join(SRC, "policy_net.c"),
        os.path.join(SRC, "generated", "policy_weights.c"),
        "-lm",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("clang FAILED:\n" + res.stderr)
        raise SystemExit(2)
    if res.stderr.strip():
        print("clang warnings:\n" + res.stderr)
    return exe


def torch_reference(frames: np.ndarray, model_path: str, encoder_path: str,
                    ref_ff: Optional[np.ndarray] = None):
    """z and action from the torch deployment path, step by step over the stream.

    `ref_ff` is the reference feed-forward block the ACTOR takes as its last 3 inputs. It is
    generated from the same helper the C side is fed from when not supplied, so the two can
    never be compared against different blocks - which would look like a numerical mismatch.
    """
    import torch

    from actor_input import load_checkpoint
    from encoder.history_encoder import load_encoder_checkpoint

    if ref_ff is None:
        ref_ff = ref_ff_stream(len(frames))
    model = load_checkpoint(model_path)
    encoder, _norm, _ckpt = load_encoder_checkpoint(encoder_path)
    h = encoder.init_state(1)

    zs = np.zeros((len(frames), 16), dtype=np.float32)
    acts = np.zeros((len(frames), 4), dtype=np.float32)
    with torch.no_grad():
        for i, raw in enumerate(frames):
            # The encoder is fed the STANDARDIZED frame; the C code does its own
            # standardization, so feed the raw frame through the same standardizer here.
            _s = standardize_frame(raw, _norm.frame_mean, _norm.frame_std, _norm.clip)
            x = torch.from_numpy(_s[None, :])
            z, h = encoder.step(x, h)
            zs[i] = z.numpy()[0]
            actor_in = np.concatenate(
                [raw[:ACTOR_FRAME_DIM], zs[i], ref_ff[i]]).astype(np.float32)
            a, _ = model.predict(actor_in, deterministic=True)
            acts[i] = np.asarray(a, dtype=np.float32).reshape(-1)
    return zs, acts


def main() -> int:
    ap = argparse.ArgumentParser(description="C vs torch check of the deployed network")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--model", default="latest")
    ap.add_argument("--encoder", default=os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt"))
    ap.add_argument("--keep", action="store_true", help="keep the temp dir with the binaries")
    args = ap.parse_args()

    model_path = resolve_model(args.model)
    frames, source = _frame_stream(args.frames)
    ff = ref_ff_stream(len(frames))
    print(f"frames  : {len(frames)} from {source}")
    print(f"ref_ff  : synthetic, {REF_FF_DIM} dims "
          f"(norm {float(np.linalg.norm(ff, axis=1).mean()):.2f} m/s^2 mean, "
          f"{int((np.linalg.norm(ff, axis=1) == 0).sum())} zero-collective frames)")
    print(f"model   : {os.path.basename(model_path)}")

    tmp = tempfile.mkdtemp(prefix="policy_host_check_")
    try:
        exe = compile_host_check(tmp)
        frames_bin = os.path.join(tmp, "frames.bin")
        out_bin = os.path.join(tmp, "out.bin")
        # ROW FORMAT: [frame(ENC_IN) | ref_ff(REF_FF_DIM)] - one file, and the split the C
        # side has to make explicit (policy_step takes the two as separate arguments).
        rows = np.concatenate([frames, ff], axis=1)
        np.ascontiguousarray(rows, dtype=np.float32).tofile(frames_bin)

        res = subprocess.run([exe, frames_bin, out_bin], capture_output=True, text=True)
        if res.returncode != 0:
            print("host_check FAILED:\n" + res.stderr)
            return 2
        print("C       : " + res.stderr.strip())

        c_out = np.fromfile(out_bin, dtype=np.float32).reshape(-1, 20)
        assert len(c_out) == len(frames), f"expected {len(frames)} rows, got {len(c_out)}"
        c_act = c_out[:, :4]
        c_z = c_out[:, 4:]

        # numpy-side export as a third opinion (already verified against torch in export)
        act = extract_actor(model_path)
        enc = extract_encoder(args.encoder)
        np_z = np.zeros((len(frames), 16), dtype=np.float32)
        np_act = np.zeros((len(frames), 4), dtype=np.float32)
        h = np.zeros(enc["hidden"], dtype=np.float32)
        for i, raw in enumerate(frames):
            h = gru_step(enc, standardize_frame(raw, enc["norm_mean"], enc["norm_std"],
                                                enc["norm_clip"]), h)
            np_z[i] = encoder_head(enc, h)
            np_act[i] = actor_forward(
                act, np.concatenate([raw[:ACTOR_FRAME_DIM], np_z[i],
                                     ff[i]]).astype(np.float32))

        tz, ta = torch_reference(frames, model_path, args.encoder, ref_ff=ff)

        def report(name: str, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
            d = np.abs(a - b)
            ok = float(d.max()) <= tol
            print(f"  {name:22s} max {d.max():.3g}  mean {d.mean():.3g}  "
                  f"{'PASS' if ok else 'FAIL'} (tol {tol:g})")
            return ok

        print("compare :")
        ok = True
        ok &= report("z: C vs torch", c_z, tz, Z_TOL)
        ok &= report("action: C vs torch", c_act, ta, ACTION_TOL)
        ok &= report("z: C vs numpy", c_z, np_z, Z_TOL)
        ok &= report("action: C vs numpy", c_act, np_act, ACTION_TOL)

        print("RESULT  : " + ("ALL GREEN" if ok else "MISMATCH"))
        return 0 if ok else 3
    finally:
        if not args.keep:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"kept    : {tmp}")


if __name__ == "__main__":
    raise SystemExit(main())
