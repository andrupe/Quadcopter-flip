"""
Supervised pretraining for the frozen history encoder.

Trains the GRU to regress the privileged physics targets from the observation stream, then
GATES on per-group validation R^2 so an encoder that cannot identify a quantity is not
silently shipped.

EPISODE-SEQUENTIAL SUPERVISION
This is the part that must not be got wrong with a recurrent model. The loss is taken over
WHOLE EPISODES with the hidden state carried from the true start, because a GRU's state is
an unbounded summary of everything it has seen. Sampling random windows and prefilling the
state would train the model against states it would never actually occupy at runtime - the
failure mode is subtle, because training loss looks fine and the deployed encoder is
quietly different. (The TCN this model replaced could be trained on random windows,
because its receptive field bounded what mattered. That is no longer true.)

Batches are formed from episodes of similar length, padded, and the loss is MASKED by the
true length so padding never contributes gradient.

OBJECTIVE
mu is trained by MSE on standardized targets, NOT by Gaussian NLL. Under NLL the loss and
R^2 decouple, because the loss can fall by inflating sigma rather than by improving mu -
measured on this project, the loss fell steadily while validation R^2 sat at 0.24. MSE
cannot mislead that way, and MSE / beta-NLL / NLL were measured to give identical R^2, so
the simplest objective that cannot lie is used for mu. The logvar head is trained
separately against a detached mu so the uncertainty estimate stays calibrated without
feeding back into the regression.

GATING
Gates are set at GATE_FRACTION of the target's MEASURED identifiability ceiling, not at an
arbitrary absolute R^2. Those ceilings were measured directly on this plant: most of the
per-parameter targets are masked by the 1 kHz rate loop, which reads ground-truth body
rates, so no encoder can recover them from the observation stream. Gating on an absolute
number there would fail a model that is already performing at 100% of what is achievable.
Anything below MIN_GATEABLE_CEILING is reported but not gated.

Run:
    .venv/bin/python Simulation/encoder/train_encoder.py --data logs/encoder_data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_SIM_DIR)
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from encoder.history_encoder import (  # noqa: E402
    EncoderWithHead,
    gaussian_nll,
    load_encoder_checkpoint,
    mse_loss,
    save_encoder_checkpoint,
)
from encoder.observation_spec import ENCODER_IN_DIM, NormStats  # noqa: E402

# ---------------------------------------------------------------------------------
# GATING
#
# A gate is only meaningful relative to what is actually recoverable from the data, so the
# ceiling is MEASURED on the same corpus rather than hardcoded. The probe is a memoryless
# MLP on single standardized frames: it has strictly less information than the GRU (no
# history at all), so requiring the GRU to reach GATE_FRACTION of it catches a recurrent
# model that failed to converge, without demanding something unreachable.
#
# This replaced a table of hardcoded ceilings measured in an earlier session. Those went
# stale the moment the observation frame, the corpus and the controller mix changed - the
# GRU then beat several of them by 2-3x, which does not mean the model is superhuman, it
# means the ceiling table was measuring a different setup. A self-calibrating probe cannot
# go stale.
#
# Targets whose probe R^2 is below PROBE_FLOOR are declared NOT IDENTIFIABLE: there is no
# single-frame signal to recover, and no amount of recurrent context invents one. They are
# reported and excluded from gating rather than failed, because failing them would be
# failing the plant, not the model.
# ---------------------------------------------------------------------------------
PROBE_FLOOR: float = 0.40          # probe R^2 below this => declared not identifiable
GATE_FRACTION: float = 0.85        # the GRU must reach this fraction of the probe

# Prior measurement, kept for context only (NOT used for gating). These were the ceilings
# measured in the earlier session, against the old 17-dim frame and a windowed probe.
PRIOR_CEILING: Dict[str, float] = {
    "thrust_scale": 0.973,
    "true_vel_w": 0.777,
    "motor_speed_norm": 0.664,
    "mass_ratio": 0.564,
    "aero_force_b": 0.284,
    "motor_efficiency": 0.096,
    "dynamic_sag_coef": 0.036,
    "rotor_radial_err": 0.030,
    "motor_tau": 0.003,
    "gyro_bias_b": 0.001,
    "com_offset_b": -0.009,
}


def load_episodes(data_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Read every shard back into a list of (frames[T,33], targets[T,29]) episodes."""
    from glob import glob

    shards = sorted(glob(os.path.join(data_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz found in {data_dir}")

    episodes: List[Tuple[np.ndarray, np.ndarray]] = []
    for sp in shards:
        d = np.load(sp, allow_pickle=False)
        frames, targets, lens = d["frames"], d["targets"], d["ep_lens"]
        if frames.shape[-1] != ENCODER_IN_DIM:
            raise ValueError(
                f"{sp}: frames have {frames.shape[-1]} dims but this build expects "
                f"{ENCODER_IN_DIM}. The corpus was collected against a different "
                f"observation frame and must be re-collected."
            )
        off = 0
        for L in lens:
            L = int(L)
            episodes.append((frames[off:off + L], targets[off:off + L]))
            off += L
    return episodes


def r2_per_group(pred: np.ndarray, true: np.ndarray, groups) -> Dict[str, float]:
    """R^2 over each target group, computed on destandardized (physical) values."""
    out: Dict[str, float] = {}
    off = 0
    for name, dim in groups:
        p = pred[:, off:off + dim]
        t = true[:, off:off + dim]
        ss_res = float(((p - t) ** 2).sum())
        ss_tot = float(((t - t.mean(axis=0, keepdims=True)) ** 2).sum())
        out[name] = 1.0 - ss_res / max(1e-12, ss_tot)
        off += dim
    return out


def probe_ceiling(
    train_eps,
    val_eps,
    n_targets: int,
    seed: int = 0,
    hidden: int = 128,
    epochs: int = 25,
    max_frames: int = 400_000,
):
    """
    Memoryless MLP probe: the best R^2 recoverable from a SINGLE standardized frame.

    Given no history and generous capacity on purpose, so it OVER-estimates rather than
    under-estimates what a memoryless model can extract. The GRU is then required to reach
    GATE_FRACTION of this - a floor a converged recurrent model clears easily, but an
    unconverged one does not, since falling below a memoryless probe means the recurrent
    state is contributing nothing.
    """
    def stack(eps, cap):
        X = np.concatenate([f for f, _ in eps], axis=0)
        Y = np.concatenate([t for _, t in eps], axis=0)
        if len(X) > cap:
            idx = np.random.default_rng(seed).choice(len(X), size=cap, replace=False)
            X, Y = X[idx], Y[idx]
        return X, Y

    Xtr, Ytr = stack(train_eps, max_frames)
    Xva, Yva = stack(val_eps, max(50_000, max_frames // 4))

    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, n_targets),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(Ytr, dtype=np.float32))
    bs = 8192
    for _ in range(epochs):
        perm = torch.randperm(xt.shape[0])
        for i in range(0, xt.shape[0], bs):
            idx = perm[i:i + bs]
            loss = nn.functional.mse_loss(net(xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(np.ascontiguousarray(Xva, dtype=np.float32))).numpy()
    return pred, Yva


def train(
    data_dir: str,
    out_path: str,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.12,
    seed: int = 0,
    threads: int = 4,
    max_episodes: Optional[int] = None,
    variance_weight: float = 0.1,
) -> int:
    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"Loading episodes from {data_dir} ...", flush=True)
    episodes = load_episodes(data_dir)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    n_frames = sum(len(f) for f, _ in episodes)
    print(f"  {len(episodes):,} episodes, {n_frames:,} frames", flush=True)

    # Split by EPISODE. Splitting by frame would leak: consecutive frames of one episode
    # are near-duplicates, so a frame-level split reports a validation score that is
    # really a training score.
    perm = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_frac))
    val_eps = [episodes[i] for i in perm[:n_val]]
    train_eps = [episodes[i] for i in perm[n_val:]]

    # Frozen normalization, computed on TRAIN only.
    f_all = np.concatenate([f for f, _ in train_eps], axis=0)
    t_all = np.concatenate([t for _, t in train_eps], axis=0)
    groups = [(f"dim{i}", 1) for i in range(t_all.shape[1])]
    norm = NormStats.from_frames_and_targets(f_all, t_all, target_groups=groups)
    print(f"  normalization from {len(f_all):,} train frames; "
          f"{len(norm.degenerate_dims)} degenerate target dims pinned", flush=True)

    def prep(eps):
        return [(norm.standardize_frame(f).astype(np.float32),
                 norm.standardize_targets(t).astype(np.float32)) for f, t in eps]

    train_eps = prep(train_eps)
    val_eps = prep(val_eps)

    model = EncoderWithHead(n_targets=t_all.shape[1], f_in=ENCODER_IN_DIM, width=48, z_dim=16)
    print(f"  model: {model.param_count:,} params", flush=True)

    # Declare the reporting groups once, so the GRU and the probe are scored identically.
    groups_decl = [(f"dim{i}", 1) for i in range(model.n_targets)]
    try:
        from quad_flip_env import PRIV_TARGET_GROUPS
        if sum(d for _, d in PRIV_TARGET_GROUPS) == model.n_targets:
            groups_decl = list(PRIV_TARGET_GROUPS)
    except Exception:
        pass
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    def batches(eps, shuffle: bool):
        order = np.argsort([len(f) for f, _ in eps]) if not shuffle else rng.permutation(len(eps))
        order = np.asarray(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            yield [eps[j] for j in idx]

    def run_batch(batch, train_mode: bool):
        lens = np.asarray([len(f) for f, _ in batch], dtype=np.int64)
        T = int(lens.max())
        B = len(batch)
        x = np.zeros((B, T, ENCODER_IN_DIM), dtype=np.float32)
        y = np.zeros((B, T, model.n_targets), dtype=np.float32)
        m = np.zeros((B, T), dtype=np.float32)
        for b, (f, t) in enumerate(batch):
            L = len(f)
            x[b, :L] = f
            y[b, :L] = t
            m[b, :L] = 1.0
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        mt = torch.from_numpy(m)

        with torch.set_grad_enabled(train_mode):
            _, mu, logvar = model.forward_sequence(xt)
            se = ((mu - yt) ** 2).sum(dim=-1)          # [B, T]
            denom = mt.sum().clamp(min=1.0)
            loss_mu = (se * mt).sum() / denom / model.n_targets
            loss_var = gaussian_nll(mu.detach(), model.clamp_logvar(logvar), yt)
            loss = loss_mu + variance_weight * loss_var

        if train_mode:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return float(loss_mu.item()), float(loss_var.item()), mu.detach().numpy(), yt.numpy(), m

    @torch.no_grad()
    def evaluate() -> Tuple[float, Dict[str, float], float]:
        model.eval()
        preds, trues = [], []
        for batch in batches(val_eps, shuffle=False):
            _, _, mu_np, y_np, m = run_batch(batch, train_mode=False)
            lens = np.asarray([len(f) for f, _ in batch])
            for b, L in enumerate(lens):
                preds.append(mu_np[b, :L])
                trues.append(y_np[b, :L])
        P = np.concatenate(preds, axis=0)
        Y = np.concatenate(trues, axis=0)
        p_phys = norm.destandardize_targets(P)
        y_phys = norm.destandardize_targets(Y)
        r2 = r2_per_group(p_phys, y_phys, groups_decl)
        pred_sd = float(np.mean(P.std(axis=0)))
        model.train()
        return float(np.mean(list(r2.values()))), r2, pred_sd

    print(f"\nTraining for {epochs} epochs "
          f"({len(train_eps)} train / {len(val_eps)} val episodes)\n")
    print(f"{'ep':>4} {'loss_mu':>9} {'loss_var':>9} {'pred_sd':>8}  gated R^2")
    t0 = time.time()
    for ep in range(epochs):
        for batch in batches(train_eps, shuffle=True):
            run_batch(batch, train_mode=True)
        sched.step()
        _, r2, pred_sd = evaluate()
        if ep % 5 == 0 or ep == epochs - 1 or ep < 3:
            shown = "  ".join(
                f"{k}={r2.get(k, float('nan')):.3f}"
                for k in ("thrust_scale", "true_vel_w", "motor_speed_norm", "mass_ratio")
            )
            print(f"{ep:>4} {'':>9} {'':>9} {pred_sd:>8.3f}  {shown}", flush=True)

    print(f"\nTraining took {time.time() - t0:.0f}s")
    _, r2, pred_sd = evaluate()

    # --- self-calibrating ceiling ------------------------------------------------------
    # train_eps / val_eps are already standardized at this point, which is what the probe
    # and the GRU both consume, so they are passed through unchanged.
    print("\nMeasuring the memoryless probe ceiling (single frame, no history) ...", flush=True)
    probe_pred, probe_true = probe_ceiling(train_eps, val_eps, model.n_targets, seed=seed)
    probe_r2 = r2_per_group(
        norm.destandardize_targets(probe_pred),
        norm.destandardize_targets(probe_true),
        groups_decl,
    )

    print("\n" + "=" * 78)
    print("VALIDATION R^2 vs SELF-MEASURED PROBE CEILING")
    print("=" * 78)
    print(f"{'target':>20} {'GRU':>7} {'probe':>7} {'vs probe':>9} {'prior':>7}  status")
    print(f"{'':>20} {'':>7} {'':>7} {'':>9} {'':>7}  probe = memoryless MLP on ONE frame | prior = stale earlier measurement")
    failures: List[str] = []
    gated: Dict[str, float] = {}
    for name in sorted(r2, key=lambda k: -probe_r2.get(k, -1)):
        val = r2[name]
        pr = float(probe_r2.get(name, float("nan")))
        prior = PRIOR_CEILING.get(name)
        prior_s = f"{prior:>7.3f}" if prior is not None else f"{'-':>7}"
        if pr < PROBE_FLOOR:
            print(f"{name:>20} {val:>7.3f} {pr:>7.3f} {'-':>9} {prior_s}  "
                  f"NOT IDENTIFIABLE (no single-frame signal) - report only")
            continue
        gate = GATE_FRACTION * pr
        gated[name] = gate
        ok = val >= gate
        if not ok:
            failures.append(name)
        status = "GATED ok" if ok else f"GATED FAIL (needs {gate:.2f})"
        print(f"{name:>20} {val:>7.3f} {pr:>7.3f} {val / max(1e-9, abs(pr)):>8.0%} {prior_s}  {status}")

    save_encoder_checkpoint(out_path, model, norm, extra={
        "val_r2": r2,
        "probe_r2": probe_r2,
        "pred_sd": pred_sd,
        "gates": gated,
        "gate_fraction": GATE_FRACTION,
        "probe_floor": PROBE_FLOOR,
        "data_dir": data_dir,
        "epochs": epochs,
        "objective": "mse_mu + detached_nll_logvar",
        "note": "episode-sequential GRU; gates are 0.85x a memoryless probe on the same corpus",
    })
    print(f"\nSaved encoder -> {out_path}")

    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {len(failures)} GATE FAILURE(S): {failures}")
        print("  The encoder is saved anyway so it can be inspected, but do NOT treat it")
        print("  as ready: PPO with a bad z is worse than PPO without one.\n")
        return 1
    print("RESULT: all gates passed")
    print(f"  Next:  .venv/bin/python Simulation/train.py   (will pick up {out_path})\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pretrain the frozen history encoder.")
    ap.add_argument("--data", type=str, default=os.path.join(_PROJECT_ROOT, "logs", "encoder_data"))
    ap.add_argument("--out", type=str, default=os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()
    return train(
        data_dir=args.data, out_path=args.out, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, val_frac=args.val_frac,
        seed=args.seed, threads=args.threads, max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    sys.exit(main())
