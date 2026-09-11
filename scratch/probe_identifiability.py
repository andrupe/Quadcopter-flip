"""
Identifiability probe: which privileged targets are recoverable from the observation
stream, and how much of that survives the 16-dim z bottleneck?

The question the gates cannot answer on their own. When a target misses its gate, either
the encoder is underfit, or the target is simply not represented in the observations.

Two numbers per target group, in increasing order of squeeze:

    A. ridge probe on the RAW standardized frame   (33 dims) -> target
       No history, no bottleneck: if a linear map from ONE frame cannot recover a target,
       no 16-dim recurrent encoder will either, and the gate should be revised rather than
       retrained against.
    B. ridge probe on z                            (16 dims) -> target
       The information the ACTOR can actually use.
    C. the encoder's own mu head                   (16 dims) -> target   <- the gated one
       Reported with the gate stored in the checkpoint, for reference.

Reading of the three:
    A high, B low, C low   -> the information exists but z discards it: raise z_dim or drop
                              the target.
    A high, B high, C low  -> z carries it but the head is underfit: train longer.
    A low                  -> not recoverable from this observation set at all; no amount
                              of training helps - revise the gate, not the model.

Caveats. A is a LINEAR probe, so a low value is suggestive rather than conclusive (the true
map is nonlinear) while a high value DOES prove recoverability. Both probes are trained on
standardized targets, which is free here because R^2 is invariant under an affine rescaling
of the target. The probe is memoryless by construction (single frame): recurrent context
could in principle recover more, and that difference is exactly what the GRU gate measures.

Run:  .venv/bin/python scratch/probe_identifiability.py --episodes 150 --samples 20000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from quad_flip_env import PRIV_TARGET_GROUPS  # noqa: E402
from encoder.history_encoder import load_encoder_checkpoint  # noqa: E402
from encoder.observation_spec import ENCODER_IN_DIM  # noqa: E402
from encoder.train_encoder import load_episodes, r2_per_group  # noqa: E402

DEFAULT_DATA = os.path.join(_PROJECT_ROOT, "logs", "encoder_data")
DEFAULT_CKPT = os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt")


def ridge(X: np.ndarray, Y: np.ndarray, Xv: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression (features are already standardized)."""
    A = X.T @ X + alpha * np.eye(X.shape[1], dtype=np.float64)
    W = np.linalg.solve(A, X.T @ Y)
    return Xv @ W


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-target identifiability on the current corpus.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--episodes", type=int, default=150, help="episodes to load (subsample)")
    ap.add_argument("--samples", type=int, default=20_000, help="timesteps to probe on")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        print(f"No corpus at {args.data}. Collect it first:\n"
              "  .venv/bin/python Simulation/encoder/collect_data.py --frames 1200000")
        return 1
    if not os.path.isfile(args.checkpoint):
        print(f"No encoder checkpoint at {args.checkpoint}. Train it first:\n"
              "  .venv/bin/python Simulation/encoder/train_encoder.py")
        return 1

    rng = np.random.default_rng(args.seed)
    encoder, norm, ckpt = load_encoder_checkpoint(args.checkpoint)
    gates = dict(ckpt.get("extra", {}).get("gates", {}))

    episodes = load_episodes(args.data)
    order = rng.permutation(len(episodes))[: max(1, args.episodes)]
    episodes = [episodes[i] for i in order]
    n_frames = sum(len(f) for f, _ in episodes)
    print(f"Loaded {len(episodes)} episodes / {n_frames:,} frames from {args.data}")
    print(f"Encoder: {os.path.basename(args.checkpoint)} "
          f"(f_in={encoder.f_in}, z={encoder.z_dim}, targets={ckpt['config']['n_targets']})")

    # --- z for every frame of the sampled episodes (causal, from the true episode start) --
    Z_all, X_all, Y_all = [], [], []
    for frames, targets in episodes:
        x = torch.from_numpy(norm.standardize_frame(frames))[None, :, :]
        with torch.no_grad():
            z = encoder.forward_sequence(x)[0].numpy()          # [T, z]
        Z_all.append(z)
        X_all.append(norm.standardize_frame(frames))
        Y_all.append(norm.standardize_targets(targets))
    X = np.concatenate(X_all).astype(np.float64)
    Z = np.concatenate(Z_all).astype(np.float64)
    Y = np.concatenate(Y_all).astype(np.float64)

    # Train/test split by TIMESTEP here, not by episode: this is a probe of what is
    # extractable from a given frame, not of generalisation to new flights, so leakage
    # through neighbouring frames is not the failure mode (the encoder gate itself does
    # split by episode).
    n = min(args.samples, X.shape[0])
    idx = rng.permutation(X.shape[0])[:n]
    n_tr = int(0.7 * n)
    tr, va = idx[:n_tr], idx[n_tr:]
    if len(va) < 100:
        print("Too few validation frames; raise --samples.")
        return 1

    pred_raw = ridge(X[tr], Y[tr], X[va])
    pred_z = ridge(Z[tr], Y[tr], Z[va])

    xa = torch.from_numpy(norm.standardize_frame(
        np.concatenate([f for f, _ in episodes])[va].astype(np.float32)))
    # The mu head was trained on whole sequences; single frames at arbitrary points are
    # fine for a memoryless linear comparison, but z from forward_sequence is the honest
    # input, so the head is fed the sequence-z instead.
    mu_head = ckpt["config"]["n_targets"]
    encoder_with_head_state = None
    try:
        from encoder.history_encoder import EncoderWithHead
        ewh = EncoderWithHead(
            n_targets=mu_head, f_in=encoder.f_in, width=encoder.width, z_dim=encoder.z_dim
        )
        ewh.load_state_dict({
            **{f"encoder.{k}": v for k, v in ckpt["encoder_state"].items()},
            **{f"mu_head.{k}": v for k, v in ckpt["mu_head_state"].items()},
            **{f"logvar_head.{k}": v for k, v in ckpt["logvar_head_state"].items()},
        })
        ewh.eval()
        with torch.no_grad():
            pred_head = ewh.mu_head(torch.from_numpy(Z[va].astype(np.float32))).numpy()
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"Note: could not rebuild the mu head ({exc}); skipping column C.")
        pred_head = None

    r2_raw = r2_per_group(pred_raw, Y[va], PRIV_TARGET_GROUPS)
    r2_z = r2_per_group(pred_z, Y[va], PRIV_TARGET_GROUPS)
    r2_head = r2_per_group(pred_head, Y[va], PRIV_TARGET_GROUPS) if pred_head is not None else {}

    print(f"\nProbed on {len(va):,} held-out frames "
          f"({len(tr):,} train). Targets are standardized; R^2 is scale-invariant.\n")
    print(f"{'target':>20} {'A raw(33)':>10} {'B z(16)':>9} {'C head':>8} {'gate':>7}  reading")
    print("-" * 78)
    for name, _dim in PRIV_TARGET_GROUPS:
        a = r2_raw.get(name, float("nan"))
        b = r2_z.get(name, float("nan"))
        c = r2_head.get(name, float("nan"))
        g = gates.get(name, float("nan"))
        if np.isnan(a):
            reading = "-"
        elif a < 0.10:
            reading = "not in the observations"
        elif a < 0.30:
            reading = "weak signal only"
        elif b < 0.40:
            reading = "lost in the z bottleneck"
        elif not np.isnan(c) and c < b - 0.10:
            reading = "head underfit vs its own z"
        else:
            reading = "identifiable"
        gate_s = f"{g:7.2f}" if not np.isnan(g) else "      -"
        c_s = f"{c:8.3f}" if not np.isnan(c) else "       -"
        print(f"{name:>20} {a:>10.3f} {b:>9.3f} {c_s} {gate_s}  {reading}")
    print("-" * 78)
    print("A = memoryless ridge probe on one standardized frame (upper-ish bound without")
    print("    any bottleneck). B = ridge probe on the encoder's z. C = the trained head.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
