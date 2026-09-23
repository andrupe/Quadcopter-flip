# -*- coding: utf-8 -*-
"""
Stage-by-stage probe of the compiled C forward pass vs the numpy reference.

Companion to `scratch/check_policy_net_c.py` (which says IF they differ): this says WHERE.
It compiles `test/net_probe.c`, runs it on the first frame of a real shard, and diffs each
stage of the pipeline against numpy:

    1. standardized input   x_std   (33)   - the encoder's frozen normalization
    2. gate pre-activations gg      (144)  - (W_ih x + b_ih) + (W_hh h + b_hh), h = 0
    3. hidden state         h       (48)   - one GRU step
    4. GELU output          head_a  (48)   - head Linear -> exact GELU
    5. latent               z       (16)   - head Linear -> Tanh

    .venv/bin/python scratch/probe_policy_net_c.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.special import erf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_SIM = os.path.join(_ROOT, "Simulation")
_DEPLOY = os.path.join(_SIM, "deploy")
for _p in [_ROOT, _SIM, _DEPLOY]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_policy import (  # noqa: E402
    _frame_stream, extract_encoder, ref_ff_stream, resolve_model, standardize_frame,
)

APP = os.path.join(_DEPLOY, "app_policy_controller")
SRC = os.path.join(APP, "src")
GEN = os.path.join(SRC, "generated")
TEST = os.path.join(APP, "test")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> int:
    frames, source = _frame_stream(1)
    raw = frames[0]
    # net_probe.c reads the same row host_check.c does: [frame(33) | ref_ff(3)].
    ff = ref_ff_stream(1)[0]
    model_path = resolve_model(sys.argv[1] if len(sys.argv) > 1 else "latest")
    enc = extract_encoder(os.path.join(_ROOT, "logs", "encoder_gru.pt"))
    H = enc["hidden"]

    tmp = tempfile.mkdtemp(prefix="net_probe_")
    exe = os.path.join(tmp, "net_probe")
    res = subprocess.run(
        ["clang", "-O2", "-std=c11", "-Wall", f"-I{SRC}", "-o", exe,
         os.path.join(TEST, "net_probe.c"),
         os.path.join(SRC, "policy_net.c"),
         os.path.join(GEN, "policy_weights.c"), "-lm"],
        capture_output=True, text=True)
    if res.returncode != 0:
        print("clang FAILED:\n" + res.stderr)
        return 2

    fb = os.path.join(tmp, "frame.bin")
    ob = os.path.join(tmp, "stages.bin")
    np.ascontiguousarray(np.concatenate([raw, ff]), dtype=np.float32).tofile(fb)
    subprocess.run([exe, fb, ob], check=True, capture_output=True)

    blob = np.fromfile(ob, dtype=np.float32)
    o1 = 0
    c_xstd = blob[o1:o1 + 33]; o1 += 33
    c_gg = blob[o1:o1 + 3 * H]; o1 += 3 * H
    c_h = blob[o1:o1 + H]; o1 += H
    c_head = blob[o1:o1 + H]; o1 += H
    c_z = blob[o1:o1 + 16]

    # -- numpy reference of the same stages (h_prev = 0, as at the start of a flight) --
    # Kept deliberately faithful to torch's GRU decomposition: b_ih in the input part,
    # b_hh in the hidden part (so the new gate multiplies r through the biased hidden
    # term). A probe that "simplifies" this into one biased sum re-creates the C bug.
    x = standardize_frame(raw, enc["norm_mean"], enc["norm_std"], enc["norm_clip"])
    h_prev = np.zeros(H, dtype=np.float32)
    gi = enc["w_ih"] @ x + enc["b_ih"]
    gh = enc["w_hh"] @ h_prev + enc["b_hh"]
    gg = gi + gh
    r = sigmoid(gi[:H] + gh[:H])
    zt = sigmoid(gi[H:2 * H] + gh[H:2 * H])
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    h = ((1.0 - zt) * n + zt * h_prev).astype(np.float32)
    head = np.asarray(0.5 * (enc["head_w1"] @ h + enc["head_b1"])
                      * (1.0 + erf((enc["head_w1"] @ h + enc["head_b1"]) / np.sqrt(2.0))),
                      dtype=np.float32)
    z = np.tanh(enc["head_w2"] @ head + enc["head_b2"]).astype(np.float32)

    def diff(name: str, c: np.ndarray, p: np.ndarray, worst: int = 5) -> float:
        d = np.abs(c - p)
        bad = int(np.argmax(d))
        print(f"{name:28s} max {d.max():.3e} at {bad:3d}   "
              f"C {c[bad]:+.5f} vs np {p[bad]:+.5f}")
        if d.max() > 1e-5:
            idx = np.argsort(d)[-worst:][::-1]
            print(f"    worst idx {idx.tolist()}  d={[float(f'{d[i]:.2e}') for i in idx]}")
        return float(d.max())

    print(f"frame from {source}")
    print("stage diffs (C vs numpy):")
    e1 = diff("1. x_std (standardize)", c_xstd, x)
    e2 = diff("2. gg (gate pre-act)", c_gg, np.asarray(gg, dtype=np.float32))
    e3 = diff("3. h (GRU step)", c_h, h)
    e4 = diff("4. head GELU", c_head, head)
    e5 = diff("5. z (latent)", c_z, z)

    ok = max(e1, e2, e3, e4, e5) < 1e-5
    print("RESULT: " + ("ALL STAGES MATCH" if ok else "STAGE MISMATCH"))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
