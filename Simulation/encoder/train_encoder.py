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

BATCHING: LENGTH-BUCKETED, NOT GLOBALLY SHUFFLED
Batches used to be drawn from a global random permutation of episodes. Episode lengths
span 213-800 steps, so padding to the longest member of a random batch wasted ~48% of
every forward/backward pass (measured on the real corpus: 51.6% of batch slots carried real
frames). The order is now LENGTH-SORTED, cut into blocks of `batch_size * bucket_factor`
episodes, and only the BLOCK order (and the order within a block) is shuffled each epoch:
batches stay length-homogeneous while the epoch still visits every episode exactly once and
the block boundaries move every epoch. Measured on the real corpus:

    global shuffle      51.6% real    (legacy, --bucket-factor 0)
    bucket factor 1     87.2% real
    bucket factor 2     88.9% real    (default -> 1.72x fewer padded steps)
    bucket factor 4     86.1% real
    bucket factor 8     79.8% real

The trade-off is diversity inside one batch: a batch now draws from a 128-episode length
window instead of the whole corpus. That is a modest loss (the epoch-level objective is
unchanged, and `bucket_factor` exposes the knob in the safe direction), and it is
preferable to the failure mode it replaces, which is 1.7x the compute for nothing.

CORRUPTED-ESTIMATE AUGMENTATION
Half the training episodes have their pose/velocity estimate deliberately broken (stale
samples, teleports, collapses, slow divergence, IMU-side bias) by encoder/corruption.py,
and the drift targets are moved with it. The decoder's job - recovering the plant
parameters and the estimator's own error - must survive an estimate that is lying, because
that is what the deployed estimator does. Validation is measured on BOTH a clean and a
fixed corrupted fold, so hardening cannot be mistaken for degradation.

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

from encoder.corruption import CorruptionConfig, corrupt_episode, verify_layout  # noqa: E402
from encoder.history_encoder import (  # noqa: E402
    DynamicsPredictorHead,
    EncoderWithDynamicsHead,
    EncoderWithHead,
    gaussian_nll,
    load_encoder_checkpoint,
    mse_loss,
    save_encoder_checkpoint,
)
from encoder.observation_spec import (  # noqa: E402
    ACTION_DIM,
    ACTION_INDICES,
    ENCODER_IN_DIM,
    EST_DRIFT_GROUPS,
    NormStats,
    PHYS_STATE_DIM,
    PHYS_STATE_GROUPS,
    PHYS_STATE_INDICES,
    group_slice,
)

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

# Length-bucketing width, in batches. 0 = the legacy global shuffle. See the module
# docstring for the measured real-frame fractions. 2 IS THE MEASURED OPTIMUM - do not raise
# it. Re-measured 2026-09-16 over the actual corpus (3,246 episodes / 1.2M frames, batch 64):
#
#     factor  0     1     2     3     4     6   (0 = global shuffle)
#     real%  51.5  87.7  91.0  87.3  85.8  83.7
#
# The intuition that "wider buckets are more homogeneous" is backwards here: a bigger block
# spans a wider slice of the length distribution, so the longest episode in a block sets an
# ever-larger pad for everything else in it. The optimum is narrow and already taken.
BUCKET_FACTOR: int = 2

# Default robustness budget: the worst relative R^2 loss, over the gated target groups,
# that the corrupted validation fold is allowed to cost. Only enforced under
# --require-robust; otherwise it is reported. 25% is a deliberately loose starting point -
# a model that loses a quarter of its accuracy across the board on a 20 m teleport is not
# robust, it is merely surviving.
ROBUST_GAP_MAX: float = 0.25

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


def current_frame_mode() -> Optional[str]:
    """
    The environment's actor-frame convention, or None if the plant is not importable.

    None means "cannot check" and every consumer must degrade to *not* enforcing it, so
    the encoder package stays usable without MuJoCo on the path.
    """
    try:
        from quad_flip_env import ACTOR_FRAME_MODE
        return str(ACTOR_FRAME_MODE)
    except Exception:
        return None


def load_episodes(data_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Read every shard back into a list of (frames[T,33], targets[T,n]) episodes."""
    from glob import glob

    shards = sorted(glob(os.path.join(data_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz found in {data_dir}")

    want_mode = current_frame_mode()
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
        # The x,y content of the frame changed meaning with ACTOR_FRAME_MODE (anchored to
        # the launch pose vs raw room coordinates). The two are NOT interchangeable, and a
        # shape check cannot see the difference, so the mode is recorded per shard and
        # enforced here. A corpus from before this key existed is the absolute-frame one.
        if want_mode is not None:
            got = str(d["frame_mode"]) if "frame_mode" in d.files else "absolute_xy (pre-2026-09-16)"
            if got != want_mode:
                raise ValueError(
                    f"{sp}: collected under actor-frame mode {got!r}, but this build uses "
                    f"{want_mode!r}.\n"
                    f"  o_t's x,y channels changed meaning, so this corpus cannot be used "
                    f"as-is. Re-collect it:\n"
                    f"    .venv/bin/python Simulation/encoder/collect_data.py"
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
        var_t = float(((t - t.mean(axis=0, keepdims=True)) ** 2).mean())
        if var_t < 1e-5:
            # Channel is near-constant (e.g. battery voltage on short flights); R^2 is undefined
            out[name] = 0.0
        else:
            ss_tot = var_t * t.size
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


def probe_ceiling_dynamics(
    train_eps,
    val_eps,
    seed: int = 0,
    hidden: int = 128,
    epochs: int = 25,
    max_frames: int = 400_000,
):
    """
    Memoryless MLP probe for dynamics: predicts Δs_t from (s_t, a_t) with NO history.
    """
    def stack(eps, cap):
        X_list, Y_list = [], []
        for f, _ in eps:
            if len(f) > 1:
                s = f[:, PHYS_STATE_INDICES]
                a = f[1:, ACTION_INDICES]
                X_list.append(np.concatenate([s[:-1], a], axis=-1))
                Y_list.append(s[1:] - s[:-1])
        X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, PHYS_STATE_DIM + ACTION_DIM), dtype=np.float32)
        Y = np.concatenate(Y_list, axis=0) if Y_list else np.zeros((0, PHYS_STATE_DIM), dtype=np.float32)
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
        nn.Linear(hidden, PHYS_STATE_DIM),
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
    mode: str = "self_supervised",
    horizon: int = 5,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.12,
    seed: int = 0,
    threads: int = 4,
    max_episodes: Optional[int] = None,
    variance_weight: float = 0.1,
    bucket_factor: int = BUCKET_FACTOR,
    eval_every: int = 4,
    corrupt: bool = True,
    corrupt_episode_prob: Optional[float] = None,
    require_robust: bool = False,
    robust_gap_max: float = ROBUST_GAP_MAX,
) -> int:
    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Augmentation draws from its OWN stream. Sharing `rng` would mean that changing a
    # corruption parameter silently changes the train/val split and the batch order, which
    # makes an A/B comparison between two runs impossible to interpret.
    aug_rng = np.random.default_rng(seed + 1000)

    print(f"Loading episodes from {data_dir} ...", flush=True)
    episodes = load_episodes(data_dir)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    n_frames = sum(len(f) for f, _ in episodes)
    print(f"  {len(episodes):,} episodes, {n_frames:,} frames", flush=True)

    # Splitting by EPISODE. Splitting by frame would leak: consecutive frames of one
    # episode are near-duplicates, so a frame-level split reports a validation score that
    # is really a training score.
    perm = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_frac))
    val_eps = [episodes[i] for i in perm[:n_val]]
    train_eps = [episodes[i] for i in perm[n_val:]]

    # Frozen normalization, computed on TRAIN only.
    f_all = np.concatenate([f for f, _ in train_eps], axis=0)
    t_all = np.concatenate([t for _, t in train_eps], axis=0)
    n_targets = int(t_all.shape[1])
    norm = NormStats.from_frames_and_targets(
        f_all, t_all, target_groups=[(f"dim{i}", 1) for i in range(n_targets)])
    print(f"  normalization from {len(f_all):,} train frames; "
          f"{len(norm.degenerate_dims)} degenerate target dims pinned", flush=True)
    del f_all, t_all

    # Mode configuration
    if mode == "self_supervised":
        groups_decl = list(PHYS_STATE_GROUPS)
        deltas = []
        for f, _ in train_eps:
            if len(f) > 1:
                s_ep = f[:, PHYS_STATE_INDICES]
                deltas.append(s_ep[1:] - s_ep[:-1])
        delta_std = np.concatenate(deltas, axis=0).std(axis=0) if deltas else np.ones(PHYS_STATE_DIM, dtype=np.float32)
        delta_std = np.where(delta_std < 1e-3, 1.0, delta_std).astype(np.float32)
        delta_std_t = torch.from_numpy(delta_std)
        drift_slice = None
        model = EncoderWithDynamicsHead(
            f_in=ENCODER_IN_DIM, width=48, z_dim=16, state_dim=PHYS_STATE_DIM, action_dim=ACTION_DIM
        )
        print(f"  model: {model.param_count:,} params, mode: self_supervised (horizon={horizon})", flush=True)
    else:
        delta_std = None
        delta_std_t = None
        groups_decl = [(f"dim{i}", 1) for i in range(n_targets)]
        drift_slice = None
        try:
            from quad_flip_env import PRIV_TARGET_GROUPS, PRIV_TARGET_DIM
            priv_groups = list(PRIV_TARGET_GROUPS)
            if int(PRIV_TARGET_DIM) == n_targets:
                groups_decl = list(priv_groups)
                drift_slice = group_slice(priv_groups, EST_DRIFT_GROUPS)
                if drift_slice is None:
                    print(f"  WARNING: {EST_DRIFT_GROUPS} not found in PRIV_TARGET_GROUPS; "
                          f"corruption will not shift the drift targets", flush=True)
            else:
                print(f"  WARNING: the corpus has {n_targets} targets but quad_flip_env declares "
                      f"{PRIV_TARGET_DIM}. Re-collect with collect_data.py for drift-target supervision.",
                      flush=True)
        except Exception as exc:
            print(f"  WARNING: quad_flip_env not importable ({type(exc).__name__}); using generic groups",
                  flush=True)

        model = EncoderWithHead(n_targets=n_targets, f_in=ENCODER_IN_DIM, width=48, z_dim=16)
        print(f"  model: {model.param_count:,} params, {model.n_targets} privileged targets", flush=True)

    if corrupt:
        cfg = CorruptionConfig()
        if corrupt_episode_prob is not None:
            cfg.episode_prob = float(corrupt_episode_prob)
        print(f"  corruption: {verify_layout()}", flush=True)
        print(f"    episode_prob={cfg.episode_prob:.2f} "
              f"families={','.join(cfg.normalised_probs()[0])} "
              f"blocks={cfg.n_blocks_range} len={cfg.block_len_range}", flush=True)
    else:
        cfg = CorruptionConfig(enabled=False)

    # A FIXED corrupted validation fold (drawn once from `aug_rng`), never re-drawn per epoch
    if cfg.enabled:
        val_eps_corrupt = []
        marks = []
        for i, (f, t) in enumerate(val_eps):
            f_c, t_c, mask = corrupt_episode(f, t, aug_rng, cfg, drift_slice)
            val_eps_corrupt.append((f_c, t_c))
            if i < 200:
                marks.append(float((mask > 0).mean()))
        print(f"    corrupted val fold: {np.mean(marks) * 100:.1f}% of steps marked", flush=True)
    else:
        val_eps_corrupt = val_eps

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    def batches(eps, shuffle: bool):
        n = len(eps)
        if n == 0:
            return
        lens = np.asarray([len(f) for f, _ in eps], dtype=np.int64)
        order = np.argsort(lens, kind="stable")
        if not shuffle:
            for i in range(0, n, batch_size):
                yield [eps[j] for j in order[i:i + batch_size]]
            return
        if int(bucket_factor) <= 0:
            for i in range(0, n, batch_size):
                yield [eps[j] for j in rng.permutation(n)[i:i + batch_size]]
            return
        off = int(rng.integers(0, n)) if n > 1 else 0
        rot = np.concatenate([order[off:], order[:off]])
        block = max(int(batch_size), int(batch_size) * int(bucket_factor))
        blocks = [rot[i:i + block] for i in range(0, n, block)]
        rng.shuffle(blocks)
        for blk in blocks:
            blk = np.asarray(blk)[rng.permutation(len(blk))]
            for i in range(0, len(blk), batch_size):
                yield [eps[j] for j in blk[i:i + batch_size]]

    def run_batch(batch, train_mode: bool):
        lens = np.asarray([len(f) for f, _ in batch], dtype=np.int64)
        T = int(lens.max())
        B = len(batch)

        if mode == "self_supervised":
            x = np.zeros((B, T, ENCODER_IN_DIM), dtype=np.float32)
            s = np.zeros((B, T, PHYS_STATE_DIM), dtype=np.float32)
            a = np.zeros((B, max(1, T - 1), ACTION_DIM), dtype=np.float32)
            m = np.zeros((B, T), dtype=np.float32)
            cm = np.zeros((B, T), dtype=np.float32)
            for b, (f, t) in enumerate(batch):
                L = len(f)
                f_use = f
                if train_mode and cfg.enabled:
                    f_corr, _, mask = corrupt_episode(f, t, aug_rng, cfg, drift_slice)
                    f_use = f_corr
                    cm[b, :L] = mask
                x[b, :L] = norm.standardize_frame(f_use)
                s[b, :L] = f[:, PHYS_STATE_INDICES]
                if L > 1:
                    a[b, :L - 1] = f[1:, ACTION_INDICES]
                m[b, :L] = 1.0

            xt = torch.from_numpy(x)
            st = torch.from_numpy(s)
            at = torch.from_numpy(a)
            mt = torch.from_numpy(m)

            with torch.set_grad_enabled(train_mode):
                z, preds, targets = model.rollout_sequence(xt, st, at, horizon=horizon)
                total_loss = torch.tensor(0.0)
                if preds:
                    K_eff = len(preds)
                    T_eff = T - K_eff
                    for k in range(K_eff):
                        p_k = preds[k]
                        t_k = targets[k]
                        m_k = mt[:, k + 1:k + 1 + T_eff]
                        denom = m_k.sum().clamp(min=1.0) * PHYS_STATE_DIM
                        norm_diff = (p_k - t_k) / delta_std_t
                        loss_k = ((norm_diff ** 2) * m_k.unsqueeze(-1)).sum() / denom
                        total_loss = total_loss + (0.9 ** k) * loss_k
                    total_loss = total_loss / K_eff

            if train_mode:
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            real = float(mt.sum().item())
            corrupt = float((cm * m).sum().item())
            if preds:
                K_eff = len(preds)
                pred_1 = (preds[0] - st[:, :T - K_eff]).detach().cpu().numpy()
                true_1 = (targets[0] - st[:, :T - K_eff]).numpy()
            else:
                pred_1 = np.zeros((B, 0, PHYS_STATE_DIM), dtype=np.float32)
                true_1 = np.zeros((B, 0, PHYS_STATE_DIM), dtype=np.float32)
            return float(total_loss.item()), 0.0, pred_1, true_1, m, real, corrupt

        else:
            x = np.zeros((B, T, ENCODER_IN_DIM), dtype=np.float32)
            y = np.zeros((B, T, model.n_targets), dtype=np.float32)
            m = np.zeros((B, T), dtype=np.float32)
            cm = np.zeros((B, T), dtype=np.float32)
            for b, (f, t) in enumerate(batch):
                L = len(f)
                f_use = f
                if train_mode and cfg.enabled:
                    f_corr, _, mask = corrupt_episode(f, t, aug_rng, cfg, drift_slice)
                    f_use = f_corr
                    cm[b, :L] = mask
                x[b, :L] = norm.standardize_frame(f_use)
                y[b, :L] = norm.standardize_targets(t)
                m[b, :L] = 1.0
            xt = torch.from_numpy(x)
            yt = torch.from_numpy(y)
            mt = torch.from_numpy(m)

            with torch.set_grad_enabled(train_mode):
                _, mu, logvar = model.forward_sequence(xt)
                se = ((mu - yt) ** 2).sum(dim=-1)          # [B, T]
                denom = mt.sum().clamp(min=1.0)
                loss_mu = (se * mt).sum() / denom / model.n_targets
                loss_var = gaussian_nll(mu.detach(), model.clamp_logvar(logvar), yt, mask=mt)
                loss = loss_mu + variance_weight * loss_var

            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            real = float(mt.sum().item())
            corrupt = float((cm * m).sum().item())
            return (float(loss_mu.item()), float(loss_var.item()), mu.detach().numpy(),
                    yt.numpy(), m, real, corrupt)

    @torch.no_grad()
    def evaluate(eps) -> Tuple[Dict[str, float], float]:
        """Per-group R^2 on eps."""
        model.eval()
        preds, trues = [], []
        for batch in batches(eps, shuffle=False):
            _, _, p_np, t_np, _m, _r, _c = run_batch(batch, train_mode=False)
            lens = np.asarray([len(f) for f, _ in batch])
            for b, L in enumerate(lens):
                if mode == "self_supervised":
                    L_eff = max(0, min(L - 1, p_np.shape[1]))
                    if L_eff > 0:
                        preds.append(p_np[b, :L_eff])
                        trues.append(t_np[b, :L_eff])
                else:
                    preds.append(p_np[b, :L])
                    trues.append(t_np[b, :L])
        P = np.concatenate(preds, axis=0) if preds else np.zeros((0, len(groups_decl)))
        Y = np.concatenate(trues, axis=0) if trues else np.zeros((0, len(groups_decl)))
        if mode == "self_supervised":
            r2 = r2_per_group(P, Y, groups_decl)
        else:
            r2 = r2_per_group(norm.destandardize_targets(P), norm.destandardize_targets(Y), groups_decl)
        pred_sd = float(np.mean(P.std(axis=0))) if len(P) > 0 else 0.0
        model.train()
        return r2, pred_sd

    print(f"\nTraining for {epochs} epochs "
          f"({len(train_eps)} train / {len(val_eps)} val episodes, "
          f"bucket_factor={bucket_factor})\n")
    print(f"{'ep':>4} {'real%':>6} {'corr%':>6} {'pred_sd':>8}  "
          f"clean R^2 / corrupted R^2")
    t0 = time.time()
    r2 = {}
    r2_corrupt = {}
    pred_sd = 0.0
    best_score = -np.inf
    best_state: Optional[dict] = None
    best_epoch = -1
    eval_every = max(1, int(eval_every))
    for ep in range(epochs):
        n_real = n_slots = n_corrupt = 0.0
        for batch in batches(train_eps, shuffle=True):
            _lm, _lv, _mu, _yn, _m, real, corrupt = run_batch(batch, train_mode=True)
            n_real += real
            n_corrupt += corrupt
            n_slots += float(len(batch) * max(len(f) for f, _ in batch))
        sched.step()

        due = (ep % eval_every == 0) or (ep >= epochs - eval_every) or ep < 3
        if due:
            r2, pred_sd = evaluate(val_eps)
            r2_corrupt, _ = evaluate(val_eps_corrupt)
            finite = [float(v) for v in r2.values() if np.isfinite(v)]
            score = float(np.mean(finite)) if finite else -np.inf
            if score > best_score:
                best_score, best_epoch = score, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if mode == "self_supervised":
                preferred = ("delta_v", "delta_w", "delta_p", "delta_q", "delta_f_b", "delta_v_batt")
            else:
                preferred = ("thrust_scale", "true_vel_w", "motor_speed_norm", "mass_ratio",
                             "est_drift_p", "est_drift_v")
            keys = [k for k in preferred if k in r2] or [k for k, _ in groups_decl[:4]]
            shown = "  ".join(
                f"{k}={r2.get(k, float('nan')):.3f}/{r2_corrupt.get(k, float('nan')):.3f}"
                for k in keys
            )
            print(f"{ep:>4} {100.0 * n_real / max(1.0, n_slots):>6.1f} "
                  f"{100.0 * n_corrupt / max(1.0, n_real):>6.1f} {pred_sd:>8.3f}  {shown}"
                  f"{'  *best' if best_epoch == ep else ''}",
                  flush=True)

    print(f"\nTraining took {time.time() - t0:.0f}s")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored the best epoch: {best_epoch} (mean clean R^2 = {best_score:.4f})")
    r2, pred_sd = evaluate(val_eps)
    r2_corrupt, _ = evaluate(val_eps_corrupt)

    # --- self-calibrating ceiling ------------------------------------------------------
    if mode == "self_supervised":
        print("\nMeasuring memoryless dynamics probe ceiling (single frame + action, no history) ...", flush=True)
        probe_pred, probe_true = probe_ceiling_dynamics(train_eps, val_eps, seed=seed)
        probe_r2 = r2_per_group(probe_pred, probe_true, groups_decl)
    else:
        def prep(eps):
            return [(norm.standardize_frame(f).astype(np.float32),
                     norm.standardize_targets(t).astype(np.float32)) for f, t in eps]
        train_std = prep(train_eps)
        val_std = prep(val_eps)
        print("\nMeasuring the memoryless probe ceiling (single frame, no history) ...", flush=True)
        probe_pred, probe_true = probe_ceiling(train_std, val_std, model.n_targets, seed=seed)
        probe_r2 = r2_per_group(
            norm.destandardize_targets(probe_pred),
            norm.destandardize_targets(probe_true),
            groups_decl,
        )
        del train_std

    print("\n" + "=" * 78)
    print("VALIDATION R^2 vs SELF-MEASURED PROBE CEILING")
    print("=" * 78)
    print(f"{'target':>20} {'GRU':>7} {'corrupt':>7} {'probe':>7} {'vs probe':>9} {'prior':>7}  status")
    print(f"{'':>20} {'':>7} {'':>7} {'':>7} {'':>9} {'':>7}  "
          f"probe = memoryless MLP on ONE frame | corrupt = same GRU on the corrupted fold")
    failures: List[str] = []
    gated: Dict[str, float] = {}
    gaps: Dict[str, float] = {}
    drift_gaps: Dict[str, float] = {}
    for name in sorted(r2, key=lambda k: -probe_r2.get(k, -1)):
        val = r2[name]
        cor = float(r2_corrupt.get(name, float("nan")))
        if val > 0.15 and np.isfinite(cor):
            bucket = drift_gaps if name in EST_DRIFT_GROUPS else gaps
            bucket[name] = max(0.0, 1.0 - cor / max(1e-9, val))
        pr = float(probe_r2.get(name, float("nan")))
        prior = PRIOR_CEILING.get(name)
        prior_s = f"{prior:>7.3f}" if prior is not None else f"{'-':>7}"
        if pr < PROBE_FLOOR:
            print(f"{name:>20} {val:>7.3f} {cor:>7.3f} {pr:>7.3f} {'-':>9} {prior_s}  "
                  f"NOT IDENTIFIABLE (no single-frame signal) - report only")
            continue
        gate = GATE_FRACTION * pr
        gated[name] = gate
        ok = val >= gate
        if not ok:
            failures.append(name)
        status = "GATED ok" if ok else f"GATED FAIL (needs {gate:.2f})"
        print(f"{name:>20} {val:>7.3f} {cor:>7.3f} {pr:>7.3f} "
              f"{val / max(1e-9, abs(pr)):>8.0%} {prior_s}  {status}")

    worst_gap = max(gaps.values()) if gaps else 0.0
    worst_name = max(gaps, key=lambda k: gaps[k]) if gaps else "-"
    measurable = bool(gaps)
    print("\n" + "-" * 78)
    if not measurable:
        print("ROBUSTNESS: INCONCLUSIVE - no target group reached clean R^2 > 0.15")
    else:
        print(f"ROBUSTNESS: worst relative R^2 loss on the corrupted fold "
              f"= {worst_gap:.1%} ({worst_name}), budget {robust_gap_max:.0%}")
        if not corrupt:
            print("  (augmentation was DISABLED: this measures the un-hardened model)")
        elif worst_gap <= robust_gap_max:
            print("  ok - the estimate can be broken without losing the identification")
        else:
            print(f"  WARNING: {worst_name} leans on the pose estimate; raise corruption strength")
    print("-" * 78)
    robust_ok = bool(corrupt and measurable and worst_gap <= robust_gap_max)

    save_encoder_checkpoint(out_path, model, norm, extra={
        "mode": mode,
        "horizon": int(horizon) if mode == "self_supervised" else None,
        "delta_std": delta_std.tolist() if delta_std is not None else None,
        "val_r2": r2,
        "val_r2_corrupt": r2_corrupt,
        "robust_gap": gaps,
        "robust_gap_drift": drift_gaps,
        "robust_gap_worst": worst_gap,
        "probe_r2": probe_r2,
        "pred_sd": pred_sd,
        "gates": gated,
        "gate_fraction": GATE_FRACTION,
        "probe_floor": PROBE_FLOOR,
        "data_dir": data_dir,
        "epochs": epochs,
        "frame_mode": current_frame_mode(),
        "bucket_factor": int(bucket_factor),
        "corruption": {
            "enabled": bool(cfg.enabled),
            "episode_prob": float(cfg.episode_prob),
            "families": list(cfg.normalised_probs()[0]),
            "block_len_range": list(cfg.block_len_range),
            "jump_pos_range": list(cfg.jump_pos_range),
            "drift_slice": list(drift_slice) if drift_slice else None,
        },
        "target_groups": [list(g) for g in groups_decl],
        "objective": f"dynamics_rollout_{horizon}" if mode == "self_supervised" else "mse_mu + detached_nll_logvar",
    })
    print(f"\nSaved encoder -> {out_path}")

    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {len(failures)} GATE FAILURE(S): {failures}")
        print("  The encoder is saved anyway so it can be inspected.\n")
        return 1
    if require_robust and not robust_ok:
        print("RESULT: clean gates passed, but --require-robust was set and the corrupted "
              f"budget is not met ({worst_gap:.1%} vs {robust_gap_max:.0%})\n")
        return 1
    print("RESULT: all gates passed")
    print(f"  Next:  .venv/bin/python Simulation/train.py   (will pick up {out_path})\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pretrain the frozen history encoder.")
    ap.add_argument("--data", type=str, default=os.path.join(_PROJECT_ROOT, "logs", "encoder_data"))
    ap.add_argument("--out", type=str, default=os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt"))
    ap.add_argument("--mode", type=str, default="self_supervised", choices=["self_supervised", "privileged"],
                    help="Pretraining mode: self_supervised (forward dynamics deltas) or privileged (regression)")
    ap.add_argument("--horizon", type=int, default=5,
                    help="Rollout horizon K for multi-step dynamics prediction (default 5)")
    ap.add_argument("--epochs", type=int, default=200,
                    help="cosine-annealed epochs; the schedule stretches with this value, "
                         "so a longer run also gets a longer anneal")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bucket-factor", type=int, default=BUCKET_FACTOR,
                    help="length-bucketing width in batches (0 = legacy global shuffle). "
                         "Measured real-frame fraction: 0 -> 0.52, 2 -> 0.89, 8 -> 0.80")
    ap.add_argument("--no-corrupt", action="store_true",
                    help="disable the corrupted-estimate augmentation (for A/B runs)")
    ap.add_argument("--corrupt-episode-prob", type=float, default=None,
                    help="fraction of training episodes corrupted (default 0.5)")
    ap.add_argument("--require-robust", action="store_true",
                    help="fail the run if the corrupted-fold R^2 budget is exceeded")
    ap.add_argument("--robust-gap-max", type=float, default=ROBUST_GAP_MAX,
                    help="worst allowed relative R^2 loss on the corrupted fold")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--eval-every", type=int, default=4,
                    help="run the two validation passes every N epochs (and always over "
                         "the last N and the first 3); the best epoch is restored before "
                         "the gates are measured and the checkpoint is written")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()
    return train(
        data_dir=args.data, out_path=args.out, mode=args.mode, horizon=args.horizon,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, val_frac=args.val_frac,
        seed=args.seed, threads=args.threads, max_episodes=args.max_episodes,
        bucket_factor=args.bucket_factor, corrupt=not args.no_corrupt,
        eval_every=args.eval_every,
        corrupt_episode_prob=args.corrupt_episode_prob,
        require_robust=args.require_robust, robust_gap_max=args.robust_gap_max,
    )


if __name__ == "__main__":
    sys.exit(main())
