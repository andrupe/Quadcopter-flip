"""
Profile the encoder training step to locate the throughput bottleneck.

Motivation: a 512-window/512-batch training step measured ~0.68 s/step, roughly 8x the
~83 ms implied by the network's FLOP count, and the process only drew ~2.2 of 10 cores.
Before committing to a long full-data run, find out where the time actually goes.

Run:  .venv/bin/python scratch/profile_encoder_step.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import PRIV_TARGET_DIM  # noqa: E402
from encoder.history_encoder import EncoderWithHead, gaussian_nll  # noqa: E402
from encoder.observation_spec import ENCODER_IN_DIM  # noqa: E402

B, T = 512, 100
print("=" * 74)
print("encoder training step profile")
print("=" * 74)
print(f"  os.cpu_count() = {os.cpu_count()}")
print(f"  torch intra-op threads (default) = {torch.get_num_threads()}")
print(f"  torch inter-op threads (default) = {torch.get_num_interop_threads()}")
print(f"  batch={B}  T={T}  in={ENCODER_IN_DIM}  targets={PRIV_TARGET_DIM}")

# ---------------------------------------------------------------------------------
print()
print("A. network fwd+bwd+optimizer vs thread count")
print("-" * 74)
x = torch.randn(B, T, ENCODER_IN_DIM)
y = torch.randn(B, PRIV_TARGET_DIM)

fwd_bwd = {}
for nt in (1, 2, 4, 6, 8, 10):
    torch.set_num_threads(nt)
    model = EncoderWithHead(n_targets=PRIV_TARGET_DIM, f_in=ENCODER_IN_DIM)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def train_step():
        _, mu, lv = model(x)
        loss = gaussian_nll(mu, model.clamp_logvar(lv), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(2):
        train_step()
    t0 = time.perf_counter()
    n = 5
    for _ in range(n):
        train_step()
    dt = (time.perf_counter() - t0) / n
    fwd_bwd[nt] = dt

    # forward only, for comparison
    with torch.inference_mode():
        model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        fdt = (time.perf_counter() - t0) / n
    print(f"  threads={nt:>2}   train step {dt * 1e3:8.1f} ms   forward {fdt * 1e3:8.1f} ms")

best_nt = min(fwd_bwd, key=fwd_bwd.get)
print(f"  -> fastest at threads={best_nt} ({fwd_bwd[best_nt] * 1e3:.1f} ms)")

# ---------------------------------------------------------------------------------
print()
print("B. data preparation cost (the python window-sampling loop)")
print("-" * 74)
rng = np.random.default_rng(0)
fake = [
    {
        "sframe": rng.normal(size=(800, ENCODER_IN_DIM)).astype(np.float32),
        "starget": rng.normal(size=(800, PRIV_TARGET_DIM)).astype(np.float32),
    }
    for _ in range(40)
]


def prep(batch: int) -> float:
    X = np.empty((batch, T, ENCODER_IN_DIM), dtype=np.float32)
    Y = np.empty((batch, PRIV_TARGET_DIM), dtype=np.float32)
    t0 = time.perf_counter()
    for b in range(batch):
        ep = fake[rng.integers(0, len(fake))]
        std = ep["sframe"]
        t = int(rng.integers(0, std.shape[0]))
        lo = t - T + 1
        if lo < 0:
            X[b, : -lo] = std[0]
            X[b, -lo:] = std[0 : t + 1]
        else:
            X[b] = std[lo : t + 1]
        Y[b] = ep["starget"][t]
    return (time.perf_counter() - t0) * 1e3


prep(64)
for batch in (128, 512, 2048):
    ms = prep(batch)
    print(f"  batch={batch:>5}   window sampling {ms:8.1f} ms   ({ms / batch * 1000:.1f} us/sample)")

# ---------------------------------------------------------------------------------
print()
print("C. augmentation cost (adds a second pass over the batch)")
print("-" * 74)
X = np.zeros((B, T, ENCODER_IN_DIM), dtype=np.float32)
t0 = time.perf_counter()
for _ in range(5):
    sigma = rng.uniform(0.02, 0.15, size=(B, 1, ENCODER_IN_DIM)).astype(np.float32)
    X += rng.normal(0.0, 1.0, size=X.shape).astype(np.float32) * sigma
print(f"  gaussian noise stream on ({B},{T},{ENCODER_IN_DIM}) : {(time.perf_counter() - t0) / 5 * 1e3:.1f} ms")

print()
print("=" * 74)
print("interpretation")
print("-" * 74)
print("  forward on batch 10 was measured at 0.54 ms, so 512 should be ~28 ms.")
print("  If the train step is far above ~3x that, the cost is framework/thread overhead,")
print("  not FLOPs, and reducing threads will help rather than hurt.")
print("=" * 74)
