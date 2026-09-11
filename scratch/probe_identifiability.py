"""
Identifiability probe: what is the best achievable R^2 per target, ignoring the encoder?

This answers the question the gate table cannot: when a target misses its gate, is the
encoder underfitting, or is the target simply not recoverable from the observation?

Three numbers per target group, in increasing order of squeeze:

    A. ridge probe on the RAW window      (N, T*C) -> target       <- upper-ish bound
       No compression at all: 100 steps x 21 channels = 2100 features. If a linear map
       from the whole window cannot recover a target, no 16-dim encoder will either, and
       the gate is unachievable and should be revised rather than retrained against.
    B. ridge probe on z                   (N, z_dim) -> target     <- what the actor can use
       Reported by train_encoder.py (linear_probe).
    C. the encoder's own head             (N, z_dim) -> target     <- gated
       The actual gate metric.

Reading of the three:
    A high, B low, C low   -> the information exists but the 16-dim z bottleneck discards
                              it: raise z_dim or drop unidentifiable targets.
    A high, B high, C low  -> z carries it but the head is underfit: more steps.
    A low                  -> the target is not recoverable from this observation set at
                              all. No amount of training helps. Revise the gate.

A caveat on the meaning of A: a LINEAR probe on a raw window is a lower bound on what is
achievable (the true relationship is nonlinear), but a high A proves identifiability. A
low A is suggestive, not conclusive.

Run:  .venv/bin/python scratch/probe_identifiability.py --data logs/encoder_data --samples 15000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import PRIV_TARGET_GROUPS, PRIV_TARGET_DIM  # noqa: E402

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "Simulation", "encoder"))
from encoder.observation_spec import ENCODER_IN_DIM  # noqa: E402
from encoder.train_encoder import (  # noqa: E402
    HISTORY_LEN,
    compute_norm,
    load_shards,
    split_episodes,
    standardize_episodes,
)


def sample_windows(episodes, rng, n: int, T: int):
    """Sample n window end-indices, cold-start padded exactly as training does."""
    ids = rng.integers(0, len(episodes), size=n)
    X = np.empty((n, T, ENCODER_IN_DIM), dtype=np.float32)
    Y = np.empty((n, PRIV_TARGET_DIM), dtype=np.float32)
    for b, ep_i in enumerate(ids):
        ep = episodes[ep_i]
        std = ep["sframe"]
        t = int(rng.integers(0, std.shape[0]))
        lo = t - T + 1
        if lo < 0:
            X[b, : -lo] = std[0]
            X[b, -lo :] = std[0 : t + 1]
        else:
            X[b] = std[lo : t + 1]
        Y[b] = ep["starget"][t]
    return X, Y


def ridge_r2(X: np.ndarray, Y: np.ndarray, lam: float, n_val: int):
    """Fit ridge on the first len-n_val rows, score on the rest. Returns per-col R^2."""
    Xtr, Ytr = X[:-n_val], Y[:-n_val]
    Xva, Yva = X[-n_val:], Y[-n_val:]

    # Bias term, so the fit is affine.
    Xtr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1), dtype=Xtr.dtype)], axis=1)
    Xva = np.concatenate([Xva, np.ones((Xva.shape[0], 1), dtype=Xva.dtype)], axis=1)

    d = Xtr.shape[1]
    XtX = Xtr.T.astype(np.float64) @ Xtr.astype(np.float64)
    XtY = Xtr.T.astype(np.float64) @ Ytr.astype(np.float64)
    W = np.linalg.solve(XtX + lam * np.eye(d), XtY)

    pred = Xva.astype(np.float64) @ W
    ss_res = ((pred - Yva) ** 2).sum(axis=0)
    ss_tot = ((Yva - Yva.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)
    return r2, pred, Yva


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_PROJECT_ROOT, "logs", "encoder_data"))
    ap.add_argument("--samples", type=int, default=15000)
    ap.add_argument("--raw-lam", type=float, default=1e4, help="ridge strength for the raw-window probe")
    ap.add_argument("--z-lam", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("=" * 78)
    print("target identifiability probe (no encoder bottleneck)")
    print("=" * 78)

    t_load = time.perf_counter()
    episodes = load_shards(args.data)
    train, val = split_episodes(episodes, 0.05, args.seed)
    norm = compute_norm(train)
    standardize_episodes(episodes, norm)
    print(f"  loaded + standardized in {time.perf_counter() - t_load:.0f}s")

    n_targets = sum(d for _, d in PRIV_TARGET_GROUPS)
    assert n_targets == PRIV_TARGET_DIM, (n_targets, PRIV_TARGET_DIM)
    X, Y = sample_windows(episodes, rng, args.samples, HISTORY_LEN)
    X = X.reshape(len(X), -1)                     # flatten the window: (N, T*C)
    n_val = max(500, len(X) // 5)
    print(f"  sampled {len(X)} windows -> raw feature dim {X.shape[1]} (T={HISTORY_LEN} x C={ENCODER_IN_DIM})")
    print(f"  ridge lambda={args.raw_lam:g}, val rows={n_val}")

    t0 = time.perf_counter()
    r2_raw, _, _ = ridge_r2(X, Y, args.raw_lam, n_val)
    print(f"  fit in {time.perf_counter() - t0:.1f}s")

    # Group-level R^2, computed on the pooled squared error (not the mean of per-dim
    # R^2), so a group with one high-variance dim is not hidden by low-variance dims.
    start = 0
    print()
    print(f"  {'target group':<20}{'dims':>5}{'raw-window R2':>15}   {'per-dim':>8}")
    print("  " + "-" * 62)
    results = {}
    for name, dim in PRIV_TARGET_GROUPS:
        sl = slice(start, start + dim)
        sub = r2_raw[sl]
        results[name] = float(np.mean(sub))
        print(f"  {name:<20}{dim:>5}{np.mean(sub):>15.3f}   {np.round(sub, 2)}")
        start += dim

    print()
    print("=" * 78)
    print("how to read this")
    print("-" * 78)
    print("  A target whose raw-window linear R^2 is LOW cannot be recovered by any")
    print("  16-dim encoder, and its gate should be revised rather than retrained against.")
    print("  A target with HIGH raw-window R^2 but a low encoder R^2 means the information")
    print("  is present and the bottleneck is the encoder (z_dim, steps, or the loss).")
    print("  Caveat: a linear probe inside a 100-step window is a LOWER bound on what is")
    print("  achievable, so high values prove identifiability while low values only")
    print("  suggest its absence.")
    print("=" * 78)


if __name__ == "__main__":
    main()
