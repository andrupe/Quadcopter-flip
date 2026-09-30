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
    state_increment,
)
from encoder.observation_spec import (  # noqa: E402
    ACTION_DIM,
    ACTION_INDICES,
    DEFAULT_CLIP,
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
#
# *** PROBE_FLOOR IS PAIRED WITH THE OBJECTIVE - IT IS NOT UNIVERSAL. *** It is only
# meaningful when the target is a STATIC property of the plant (privileged mode), because
# then "is there a single-frame signal?" is the right question. It is WRONG in
# self_supervised mode, where every target is a FUTURE delta and the probe is a memoryless
# model competing against a recurrent one. On the 2026-09-23 run all six probes sat below
# 0.40 (max 0.294) while the GRU reached 0.455 on delta_q - 41x the probe - so every group
# was excluded, `gated` came back EMPTY, and the run printed "clean gates passed" having
# asserted nothing. self_supervised therefore gates on an ABSOLUTE clean-R^2 floor instead;
# see GATE_MIN_CLEAN_R2 and the mode branch in the gate table.
# ---------------------------------------------------------------------------------
PROBE_FLOOR: float = 0.40          # privileged ONLY: probe R^2 below this => excluded
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

# ---------------------------------------------------------------------------------
# ROBUSTNESS STATISTIC
#
# The budget is the CORRUPTED/CLEAN RMSE RATIO on a gated group: "under the corruption,
# how many times worse is the prediction?". Bounded below by 0, monotone in the degradation,
# and independent of the target's own variance.
#
# WHY NOT A RELATIVE R^2 LOSS. It was `1 - R2_corrupt / R2_clean`, which is unbounded
# whenever the clean R^2 is small: on the 2026-09-23 run that formula reported 129558.5%
# because R2_clean was 0.347 and R2_corrupt -448.7. R^2 on a near-zero-variance 1-step delta
# is ill-conditioned - a corruption-scaled error divided by a minuscule natural variance - so
# the ratio was measuring the CONDITIONING, not the robustness. A 25% budget on a
# denominator of 0.347 is also a hidden ABSOLUTE bar (R2_corrupt >= 0.26) that means
# something different for every target, so the numbers were not comparable across groups.
#
# Calibrated on the 2026-09-24 measurement (seed 42, exact fold): a model that is perfect on
# clean steps and unbounded on corrupted ones scores 6.6x - 113x here, so 2.0x is still a
# loose bar. It fails a model that does not survive corruption, and it does so legibly.
# ---------------------------------------------------------------------------------
ROBUST_RATIO_MAX: float = 2.0

# Minimum clean R^2 for a target to be GATED at all, and therefore to enter the budget.
# Below this there is nothing to measure, so the group is reported and excluded. The SAME
# floor decides gating and budget inclusion on purpose: two different thresholds is exactly
# how a target ends up budgeted while being declared unidentifiable.
GATE_MIN_CLEAN_R2: float = 0.15

# Weight applied to CORRUPTED steps in the self-supervised loss. 1.0 = uniform, i.e. exactly
# the pre-2026-09-24 behaviour.
#
# *** MEASURED NULL - DO NOT EXPECT THIS TO FIX THE CORRUPTED REGIME. ***
# The hypothesis was: the loss is a weighted MEAN over all steps and only ~19% are corrupted,
# so the loss-optimal solution is to fit the clean majority and ignore the corrupted regime;
# weighting should make the corrupted steps carry their share of the gradient.
# A/B, 2026-09-24, 30 epochs, 5000 episodes, seed 42, ONLY this knob changed (numbers are
# normalized corrupted/clean error, then clean mean R^2):
#     delta_w 3.63 -> 3.40   delta_q 5.87 -> 6.41   delta_p 1.05 -> 1.04
#     delta_v 1.72 -> 1.72   delta_f_b 1.17 -> 1.12   clean mean R^2 0.0724 -> 0.0659
# 5x more gradient on the corrupted steps moved NOTHING (delta_q got slightly worse).
# So the binding constraint is UPSTREAM of the loss weighting. Leading candidate: the
# corrupted input is clipped into a plateau and no longer CARRIES the information.
# `standardize_frame` clips at +-10 and 31.85% of channels are pinned under corruption vs
# 1.42% clean, so many distinct corrupted states collapse onto the same input - and no
# weighting scheme can recover information the input no longer carries.
# Kept as a knob with the default OFF (1.0) so this dead end is not re-run.
#
# The weighting is a weighted mean (sum(w*err^2)/sum(w)), so it does NOT rescale the loss.
CORRUPT_STEP_WEIGHT: float = 1.0

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


def error_stats(pred: np.ndarray, true: np.ndarray, groups) -> Dict[str, Dict[str, float]]:
    """
    Per-group RMSE, target sigma and R^2, in the TARGET's own units.

    This is what the robustness statistic is built from. A ratio of RMSEs is bounded below
    by 0, monotone in the error, and carries the target's units, so "the corrupted fold
    costs 3.4x the clean RMSE" is readable without knowing anything about the target's
    variance - which is precisely what a relative R^2 loss fails to provide.
    """
    out: Dict[str, Dict[str, float]] = {}
    off = 0
    for name, dim in groups:
        p = pred[:, off:off + dim]
        t = true[:, off:off + dim]
        err2 = (p - t) ** 2
        var_t = float(((t - t.mean(axis=0, keepdims=True)) ** 2).mean())
        sigma = float(np.sqrt(var_t))
        rmse = float(np.sqrt(err2.mean())) if err2.size else float("nan")
        r2 = (1.0 - float(err2.sum()) / max(1e-12, var_t * t.size)) if var_t >= 1e-5 else 0.0
        out[name] = {
            "rmse": rmse,
            "sigma": sigma,
            "r2": r2,
            "nrmse": (rmse / sigma) if sigma > 0 else float("nan"),
        }
        off += dim
    return out


def robustness_ratio(stats_clean: Dict[str, float], stats_corrupt: Dict[str, float]) -> float:
    """
    Bounded, SCALE-FREE degradation statistic: normalized corrupted error / normalized clean
    error, where "normalized" means divided by THAT fold's own target sigma.

    BOTH sides are normalized on purpose. A raw RMSE ratio is still denominator-dominated when
    the corruption changes the TARGET distribution instead of only the input: a `jump`
    injection puts an unpredictable displacement into the pose targets, so on the corrupted
    fold delta_p's target sigma rises 100x. Measured 2026-09-24 on a 30-epoch model, delta_p's
    raw ratio read 102x while the model was sitting at the no-change baseline (nrmse 0.94 ->
    1.01) - i.e. it was being charged for a target that merely got harder. Normalizing each
    side by its own sigma asks the question that matters: how much worse than your own clean
    skill are you under corruption?

    1.0 == no degradation. Takes the per-group dicts produced by `error_stats`.
    """
    nc = stats_clean.get("nrmse")
    nk = stats_corrupt.get("nrmse")
    if nc is None or nk is None or not np.isfinite(nc) or not np.isfinite(nk):
        return float("nan")
    return float(nk / max(1e-12, nc))


def self_test_robustness_metric() -> None:
    """
    Deterministic, sub-millisecond regression test for the robustness statistic.

    It runs on EVERY training invocation, --smoke included. The gate it guards was only ever
    exercised on the production path (`--require-robust` is added by the pipeline only when
    SMOKE=false), so the 2026-09-23 failure surfaced 93 minutes into a run. This check is
    cheap enough to run unconditionally, and it targets the DEFECT CLASS rather than one
    instance of it.

    The property asserted is VARIANCE-INVARIANCE on BOTH sides: the reported degradation
    depends only on how much worse the predictions get relative to the FOLD'S OWN target
    scale, never on the target's variance. Three folds, all a genuine 10.0x degradation:

        A  baseline                             clean 0.80 sigma, corrupted 8.0 sigma
        B  worse clean skill                    clean 0.95 sigma, corrupted 9.5 sigma
        C  corrupted target sigma x10            clean 0.80 sigma, corrupted 80 sigma
           (the delta_p `jump` shape)

    A relative R^2 loss reports 176x for A and 916x for B (it divides by the clean R^2). A raw
    RMSE ratio reports 10x for A and 100x for C (it charges the model for a harder target).
    Both are the same defect seen from two sides: the number moves with the target's variance
    instead of with the model. The errors are constructed so sigma and the RMSEs are exact,
    which makes the expected values equalities rather than estimates.
    """
    groups = (("t", 1),)
    n = 4096
    truth = np.tile(np.array([-1.0, 1.0], dtype=np.float64), n // 2).reshape(-1, 1)
    # (label, clean rmse in sigma units, corrupted target sigma multiplier,
    #  corrupted rmse in that fold's OWN sigma units)
    cases = (
        ("A baseline", 0.80, 1.0, 8.0),
        ("B worse clean skill", 0.95, 1.0, 9.5),
        ("C corrupted target sigma x10", 0.80, 10.0, 8.0),
    )
    ratios: List[float] = []
    for label, rmse_c, sigma_k, nrmse_k in cases:
        truth_k = truth * sigma_k
        err_c = np.tile(np.array([rmse_c, -rmse_c]), n // 2).reshape(-1, 1)
        err_k_abs = nrmse_k * sigma_k
        err_k = np.tile(np.array([err_k_abs, -err_k_abs]), n // 2).reshape(-1, 1)
        sc = error_stats(truth + err_c, truth, groups)["t"]
        sk = error_stats(truth_k + err_k, truth_k, groups)["t"]
        if abs(sc["rmse"] - rmse_c) > 1e-9 or abs(sk["rmse"] - err_k_abs) > 1e-9:
            raise RuntimeError(
                f"robustness self-test [{label}]: the constructed RMSE is not exact "
                f"(clean {sc['rmse']!r} vs {rmse_c}, corrupt {sk['rmse']!r} vs {err_k_abs})")
        ratio = robustness_ratio(sc, sk)
        if not np.isfinite(ratio) or abs(ratio - nrmse_k / rmse_c) > 1e-9:
            raise RuntimeError(
                f"robustness self-test [{label}]: the statistic is not "
                f"normalized-corrupted / normalized-clean error ({ratio!r} vs "
                f"{nrmse_k / rmse_c!r})")
        ratios.append(ratio)
    if max(ratios) - min(ratios) > 1e-9:
        raise RuntimeError(
            f"robustness self-test: the statistic is NOT variance-invariant ({ratios!r} for "
            "the same 10.0x degradation). It must ignore BOTH a small target variance (a "
            "relative R^2 loss explodes) and a corrupted fold whose target got harder (a raw "
            "RMSE ratio inflates).")


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
                # Same increment definition as the GRU targets (attitude = relative
                # rotation), otherwise the "ceiling" column would be graded on a different
                # target than the group it is meant to bound.
                with torch.no_grad():
                    inc = state_increment(
                        torch.from_numpy(np.ascontiguousarray(s[:-1])),
                        torch.from_numpy(np.ascontiguousarray(s[1:])),
                    )
                Y_list.append(inc.numpy())
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
    width: int = 48,
    clip: float = DEFAULT_CLIP,
    variance_weight: float = 0.1,
    bucket_factor: int = BUCKET_FACTOR,
    eval_every: int = 4,
    corrupt: bool = True,
    corrupt_episode_prob: Optional[float] = None,
    require_robust: bool = False,
    robust_ratio_max: float = ROBUST_RATIO_MAX,
    corrupt_step_weight: float = CORRUPT_STEP_WEIGHT,
) -> int:
    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Augmentation draws from its OWN stream. Sharing `rng` would mean that changing a
    # corruption parameter silently changes the train/val split and the batch order, which
    # makes an A/B comparison between two runs impossible to interpret.
    aug_rng = np.random.default_rng(seed + 1000)

    # Fail fast on a broken MEASUREMENT, before spending an hour on training. This is the one
    # assertion that can run under --smoke, where the production-only robustness budget
    # cannot: an undertrained model has no signal yet, so that budget would always trip.
    self_test_robustness_metric()

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
        f_all, t_all, target_groups=[(f"dim{i}", 1) for i in range(n_targets)], clip=float(clip))
    print(f"  normalization from {len(f_all):,} train frames; "
          f"{len(norm.degenerate_dims)} degenerate target dims pinned", flush=True)
    del f_all, t_all

    # Mode configuration
    if mode == "self_supervised":
        groups_decl = list(PHYS_STATE_GROUPS)
        deltas = []
        with torch.no_grad():
            for f, _ in train_eps:
                if len(f) > 1:
                    s_ep = f[:, PHYS_STATE_INDICES]
                    # Same increment definition the targets use: the attitude channel is the
                    # relative rotation, NOT a component-wise quaternion difference. The two
                    # differ by ~20x in scale (sigma(rotvec) ~ 0.004 vs sigma(delta_q_euclid)
                    # ~ 0.094), so this scaling must follow the target definition.
                    inc = state_increment(
                        torch.from_numpy(np.ascontiguousarray(s_ep[:-1])),
                        torch.from_numpy(np.ascontiguousarray(s_ep[1:])),
                    )
                    deltas.append(inc.numpy())
        delta_std = np.concatenate(deltas, axis=0).std(axis=0) if deltas else np.ones(PHYS_STATE_DIM, dtype=np.float32)
        delta_std = np.where(delta_std < 1e-3, 1.0, delta_std).astype(np.float32)
        delta_std_t = torch.from_numpy(delta_std)
        drift_slice = None
        model = EncoderWithDynamicsHead(
            f_in=ENCODER_IN_DIM, width=int(width), z_dim=16, state_dim=PHYS_STATE_DIM, action_dim=ACTION_DIM
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

        model = EncoderWithHead(n_targets=n_targets, f_in=ENCODER_IN_DIM, width=int(width), z_dim=16)
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

    # A FIXED corrupted validation fold, INDEPENDENT of the training corruption settings.
    #
    # THE FOLD IS THE INSTRUMENT; the training cfg is the TREATMENT. They used to share `cfg`
    # and `aug_rng`, which silently confounded every A/B: raising --corrupt-episode-prob both
    # trained on more corruption AND made the measuring instrument gentler, and the two effects
    # cancel into a flat, meaningless curve. The fold therefore gets its OWN stream
    # (seed + 2000, so it cannot be perturbed by any training knob) and the canonical
    # CorruptionConfig defaults, and it is built even when training corruption is DISABLED -
    # that is what makes `--no-corrupt` a usable control (train clean, still measure the same
    # instrument) instead of a run whose "corrupted" fold is secretly the clean one.
    eval_cfg = CorruptionConfig()
    eval_rng = np.random.default_rng(seed + 2000)
    val_eps_corrupt = []
    marks = []
    for i, (f, t) in enumerate(val_eps):
        f_c, t_c, mask = corrupt_episode(f, t, eval_rng, eval_cfg, drift_slice)
        val_eps_corrupt.append((f_c, t_c))
        if i < 200:
            marks.append(float((mask > 0).mean()))
    print(f"    corrupted val fold (FIXED instrument, episode_prob="
          f"{eval_cfg.episode_prob:.2f}): {np.mean(marks) * 100:.1f}% of steps marked", flush=True)

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
            cmt = torch.from_numpy(cm)

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
                        # CORRUPTED-STEP WEIGHTING (see CORRUPT_STEP_WEIGHT).
                        # A step counts as corrupted if EITHER the frame the prediction
                        # starts from or the frame it predicts was corrupted - the damage
                        # arrives by both routes: the corrupted input poisons z, and for the
                        # pose channels the injected jump is in the target itself (measured:
                        # the delta_p target sigma is 100x larger on the corrupted fold,
                        # while delta_w's is unchanged at 1.000 - there it is purely input).
                        # weight 1.0 leaves w_k == m_k, i.e. bit-identical to before.
                        w_k = 1.0 + (corrupt_step_weight - 1.0) * torch.maximum(
                            cmt[:, :T_eff], cmt[:, k + 1:k + 1 + T_eff])
                        w_k = w_k * m_k
                        denom = w_k.sum().clamp(min=1.0) * PHYS_STATE_DIM
                        norm_diff = (p_k - t_k) / delta_std_t
                        loss_k = ((norm_diff ** 2) * w_k.unsqueeze(-1)).sum() / denom
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
                # preds/targets are ALREADY increments from the window start, with the
                # attitude as a relative rotation. The old `- st[:, :T - K_eff]` was the
                # component-wise quaternion difference that the double cover broke, so it
                # must not be reapplied here.
                pred_1 = preds[0].detach().cpu().numpy()
                true_1 = targets[0].detach().cpu().numpy()
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
    def evaluate(eps) -> Tuple[Dict[str, float], float, np.ndarray, np.ndarray]:
        """Per-group R^2 on eps, plus the destandardized (physical) predictions/targets.

        The arrays are returned because the robustness statistic is a ratio of RMSEs, and an
        R^2 alone cannot be turned back into one once the target variance is unknown.
        """
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
            Pd, Yd = P, Y
        else:
            Pd, Yd = norm.destandardize_targets(P), norm.destandardize_targets(Y)
        r2 = r2_per_group(Pd, Yd, groups_decl)
        pred_sd = float(np.mean(P.std(axis=0))) if len(P) > 0 else 0.0
        model.train()
        return r2, pred_sd, Pd, Yd

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
            r2, pred_sd, _Pc, _Yc = evaluate(val_eps)
            r2_corrupt, _, _Pk, _Yk = evaluate(val_eps_corrupt)
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
    r2, pred_sd, P_clean, Y_clean = evaluate(val_eps)
    r2_corrupt, _, P_corrupt, Y_corrupt = evaluate(val_eps_corrupt)

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

    stats_clean = error_stats(P_clean, Y_clean, groups_decl)
    stats_corrupt = error_stats(P_corrupt, Y_corrupt, groups_decl)

    print("\n" + "=" * 78)
    print("VALIDATION R^2 vs PROBE CEILING, and CORRUPTED-FOLD DEGRADATION")
    print("=" * 78)
    print(f"{'target':>20} {'R2clean':>8} {'R2corr':>9} {'probe':>7} "
          f"{'rmse_c':>10} {'rmse_k':>10} {'nr_r':>7} {'raw_r':>7} {'nr_k':>7}  status")
    print(f"{'':>20} probe = memoryless MLP on ONE frame | rmse in the target's own units | "
          f"nr_r = NORMALIZED corrupted/clean error (the budgeted ratio, {robust_ratio_max:.1f}x) | "
          f"raw_r = raw rmse ratio (informational) | nr_k = nrmse on the corrupted fold")
    failures: List[str] = []
    gated: Dict[str, float] = {}
    ratios: Dict[str, float] = {}
    raw_ratios: Dict[str, float] = {}
    drift_ratios: Dict[str, float] = {}
    for name in sorted(r2, key=lambda k: -probe_r2.get(k, -1)):
        val = r2[name]
        cor = float(r2_corrupt.get(name, float("nan")))
        pr = float(probe_r2.get(name, float("nan")))
        sc_n, sk_n = stats_clean.get(name), stats_corrupt.get(name)
        ratio = (robustness_ratio(sc_n, sk_n)
                 if sc_n is not None and sk_n is not None else float("nan"))
        raw = (sk_n["rmse"] / max(1e-12, sc_n["rmse"])
               if sc_n is not None and sk_n is not None else float("nan"))

        # IDENTIFIABILITY IS MODE-SPECIFIC - see the PROBE_FLOOR comment block.
        if mode == "self_supervised":
            identifiable = bool(np.isfinite(val) and val >= GATE_MIN_CLEAN_R2)
            why = f"clean R^2 < {GATE_MIN_CLEAN_R2:.2f}"
        else:
            identifiable = bool(np.isfinite(pr) and pr >= PROBE_FLOOR)
            why = f"probe < {PROBE_FLOOR:.2f}"

        cells = (f"{name:>20} {val:>8.3f} {cor:>9.3f} {pr:>7.3f} "
                 f"{sc_n['rmse']:>10.5f} {sk_n['rmse']:>10.5f} {ratio:>7.2f} {raw:>7.2f} "
                 f"{sk_n['nrmse']:>7.2f}")
        if not identifiable:
            print(f"{cells}  report only ({why})")
            continue
        gate = GATE_FRACTION * max(pr, 0.0)
        gated[name] = gate
        ok = val >= gate
        if not ok:
            failures.append(name)
        if np.isfinite(ratio):
            (drift_ratios if name in EST_DRIFT_GROUPS else ratios)[name] = ratio
        if np.isfinite(raw):
            raw_ratios[name] = raw
        print(f"{cells}  {'GATED ok' if ok else f'GATED FAIL (needs {gate:.2f})'}")

    worst_ratio = max(ratios.values()) if ratios else 0.0
    worst_name = max(ratios, key=lambda k: ratios[k]) if ratios else "-"
    measurable = bool(ratios)
    gated_any = bool(gated)
    print("\n" + "-" * 78)
    if not gated_any:
        print("ROBUSTNESS: NO GATED TARGET - the clean gate table is EMPTY, so 'clean gates "
              "passed' would assert nothing.")
    elif not measurable:
        print("ROBUSTNESS: INCONCLUSIVE - no gated group produced a finite RMSE ratio")
    else:
        print(f"ROBUSTNESS: worst NORMALIZED corrupted/clean error = {worst_ratio:.2f}x "
              f"({worst_name}), budget {robust_ratio_max:.1f}x")
        if not corrupt:
            print("  (augmentation DISABLED: this model never saw a corrupted input - it is the "
                  "CLEAN-TRAINING control, measured on the same fixed instrument)")
        elif worst_ratio <= robust_ratio_max:
            print("  ok - the estimate can be broken without losing the prediction")
        else:
            print(f"  WARNING: {worst_name} leans on the corrupted channels - the prediction "
                  f"error grows {worst_ratio:.1f}x. NOTE loss weighting is a MEASURED NULL "
                  f"(see CORRUPT_STEP_WEIGHT): the constraint is upstream of the loss, most "
                  f"likely that the corrupted input is CLIPPED into a plateau (31.85% of "
                  f"channels pinned vs 1.42% clean), so distinct corrupted states collapse "
                  f"onto the same input. Cheapest discriminating test: shrink the corruption "
                  f"envelope and see whether the ratio responds at all.")
        print("  NOTE: the corrupted fold scores its clean and corrupted steps TOGETHER, so this "
              "ratio is diluted by the clean steps inside it - a LOWER bound on the degradation.")
        print("        Per-step and per-family split: "
              "Simulation/scratch/diag_encoder_robustness.py")
    print("-" * 78)
    robust_ok = bool(corrupt and measurable and gated_any and worst_ratio <= robust_ratio_max)

    save_encoder_checkpoint(out_path, model, norm, extra={
        "mode": mode,
        "horizon": int(horizon) if mode == "self_supervised" else None,
        "delta_std": delta_std.tolist() if delta_std is not None else None,
        "val_r2": r2,
        "val_r2_corrupt": r2_corrupt,
        # The robustness statistic, in the units it is budgeted in: corrupted/clean RMSE.
        # The old `robust_gap*` (relative R^2 loss) keys are GONE on purpose - that form was
        # unbounded (129558.5% on the 2026-09-23 run) and not comparable across targets.
        "robust_ratio": ratios,
        "robust_ratio_raw": raw_ratios,
        "robust_ratio_drift": drift_ratios,
        "robust_ratio_worst": worst_ratio,
        "robust_ratio_max": robust_ratio_max,
        "rmse_clean": {k: v["rmse"] for k, v in stats_clean.items()},
        "rmse_corrupt": {k: v["rmse"] for k, v in stats_corrupt.items()},
        "sigma_clean": {k: v["sigma"] for k, v in stats_clean.items()},
        "sigma_corrupt": {k: v["sigma"] for k, v in stats_corrupt.items()},
        "nrmse_corrupt": {k: v["nrmse"] for k, v in stats_corrupt.items()},
        "probe_r2": probe_r2,
        "pred_sd": pred_sd,
        "gates": gated,
        "gate_fraction": GATE_FRACTION,
        "probe_floor": PROBE_FLOOR,
        "gate_min_clean_r2": GATE_MIN_CLEAN_R2,
        "data_dir": data_dir,
        "epochs": epochs,
        "frame_mode": current_frame_mode(),
        "bucket_factor": int(bucket_factor),
        "corruption": {
            "enabled": bool(cfg.enabled),
            "episode_prob": float(cfg.episode_prob),
            "step_weight": float(corrupt_step_weight),
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
        if not gated_any:
            print("RESULT: NO TARGET WAS GATED, so the clean gate table is EMPTY and 'clean "
                  "gates passed' would assert nothing. Treated as a FAILURE, not a pass.")
            print("  (Not enforced under --smoke: an undertrained model has no signal yet.)\n")
        elif not measurable:
            print("RESULT: no gated target produced a finite RMSE ratio - INCONCLUSIVE.\n")
        else:
            print("RESULT: clean gates passed, but --require-robust was set and the "
                  f"corrupted-fold budget is not met ({worst_ratio:.2f}x vs "
                  f"{robust_ratio_max:.1f}x)\n")
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
    ap.add_argument("--robust-ratio-max", type=float, default=ROBUST_RATIO_MAX,
                    help="worst allowed corrupted/clean RMSE ratio over the gated target "
                         "groups (1.0 = no degradation). Replaces the old unbounded "
                         "relative-R^2-loss budget.")
    ap.add_argument("--corrupt-step-weight", type=float, default=CORRUPT_STEP_WEIGHT,
                    help="weight applied to corrupted steps in the self-supervised loss "
                         "(1.0 = uniform, the pre-2026-09-24 behaviour)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--eval-every", type=int, default=4,
                    help="run the two validation passes every N epochs (and always over "
                         "the last N and the first 3); the best epoch is restored before "
                         "the gates are measured and the checkpoint is written")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--width", type=int, default=48,
                    help="GRU hidden width (z_dim stays 16 so the actor input width is fixed)")
    ap.add_argument("--clip", type=float, default=DEFAULT_CLIP,
                    help="standardisation clip in sigma. Lower = more saturation; the corrupted "
                         "fold pins ~32%% of channels at the default "
                         "(measured 2026-09-24)")
    args = ap.parse_args()
    return train(
        data_dir=args.data, out_path=args.out, mode=args.mode, horizon=args.horizon,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, val_frac=args.val_frac,
        seed=args.seed, threads=args.threads, max_episodes=args.max_episodes,
        bucket_factor=args.bucket_factor, corrupt=not args.no_corrupt,
        eval_every=args.eval_every,
        corrupt_episode_prob=args.corrupt_episode_prob,
        require_robust=args.require_robust, robust_ratio_max=args.robust_ratio_max,
        corrupt_step_weight=args.corrupt_step_weight,
        width=args.width, clip=args.clip,
    )


if __name__ == "__main__":
    sys.exit(main())
