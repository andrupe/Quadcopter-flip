# -*- coding: utf-8 -*-
"""
Pinpoint WHERE the compiled C network first diverges from the numpy/torch reference.

`policy_host_check.py` says C != torch; this says at which step and by how much, and
prints the first few components of both latent vectors so the cause is visible instead of
guessed at. Run it after any change to policy_net.c / the export.

    .venv/bin/python scratch/check_policy_net_c.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_SIM = os.path.join(_ROOT, "Simulation")
_DEPLOY = os.path.join(_SIM, "deploy")
for _p in [_ROOT, _SIM, _DEPLOY]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_policy import (  # noqa: E402
    ACTOR_FRAME_DIM, _frame_stream, actor_forward, encoder_head, extract_actor,
    extract_encoder, gru_step, ref_ff_stream, resolve_model, standardize_frame,
)
from policy_host_check import compile_host_check  # noqa: E402


def main() -> int:
    n = 6
    frames, source = _frame_stream(n)
    ff = ref_ff_stream(n)
    # Optional argv[1]: a model path. These probes compile the C from the CURRENT
    # generated weights, so they must be pointed at the same checkpoint that was exported -
    # "latest" is only correct immediately after an export of the latest run.
    model_path = resolve_model(sys.argv[1] if len(sys.argv) > 1 else "latest")
    act = extract_actor(model_path)
    enc = extract_encoder(os.path.join(_ROOT, "logs", "encoder_gru.pt"))
    print(f"frames from {source}")

    # -- reference (numpy) ---------------------------------------------------------
    np_z = np.zeros((n, 16), dtype=np.float32)
    np_a = np.zeros((n, 4), dtype=np.float32)
    h = np.zeros(enc["hidden"], dtype=np.float32)
    for i, raw in enumerate(frames):
        x = standardize_frame(raw, enc["norm_mean"], enc["norm_std"], enc["norm_clip"])
        h = gru_step(enc, x, h)
        np_z[i] = encoder_head(enc, h)
        np_a[i] = actor_forward(
            act, np.concatenate([raw[:ACTOR_FRAME_DIM], np_z[i], ff[i]]).astype(np.float32))

    # -- C -------------------------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="net_debug_")
    exe = compile_host_check(tmp)
    fb = os.path.join(tmp, "f.bin")
    ob = os.path.join(tmp, "o.bin")
    np.ascontiguousarray(np.concatenate([frames, ref_ff_stream(n)], axis=1),
                         dtype=np.float32).tofile(fb)
    import subprocess

    subprocess.run([exe, fb, ob], check=True, capture_output=True)
    c = np.fromfile(ob, dtype=np.float32).reshape(-1, 20)
    c_z, c_a = c[:, 4:], c[:, :4]

    # -- compare -------------------------------------------------------------------
    for i in range(n):
        dz = float(np.max(np.abs(c_z[i] - np_z[i])))
        da = float(np.max(np.abs(c_a[i] - np_a[i])))
        flag = "  <-- FIRST DIVERGENCE" if dz > 1e-5 and i == 0 else ""
        print(f"step {i}: |dz| {dz:.3e}  |da| {da:.3e}{flag}")
        if dz > 1e-5 and i <= 1:
            print(f"   raw[:6]     {np.array2string(frames[i][:6], precision=4)}")
            x = standardize_frame(frames[i], enc["norm_mean"], enc["norm_std"], enc["norm_clip"])
            print(f"   x_std[:6]   {np.array2string(x[:6], precision=4)}")
            print(f"   z np  {np.array2string(np_z[i][:5], precision=5)}")
            print(f"   z C   {np.array2string(c_z[i][:5], precision=5)}")
            print(f"   a np  {np.array2string(np_a[i], precision=5)}")
            print(f"   a C   {np.array2string(c_a[i], precision=5)}")

    bad = np.where((np.abs(c_z - np_z) > 1e-5).any(axis=1))[0]
    print("first bad z step:", int(bad[0]) if len(bad) else "none")
    print("kept:", tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
