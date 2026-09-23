"""
Corruption augmentation for the history encoder's input stream.

WHY THIS EXISTS
The encoder's job is to identify unmodelled plant parameters from the recent sensor
stream. Its input frame is dominated by the Lighthouse-fused ESTIMATE of position and
velocity, and that estimate is the single most fragile channel on the vehicle:

  * During a blackout the deck produces no fix at all, so the estimator dead-reckons and
    the estimate DIVERGES from truth with no self-correction (see lighthouse.py).
  * On the real airframe the estimator can also fail LOUDLY rather than gracefully: the
    bring-up logs recorded `stateEstimate.x` walking to +98 m at 5-9 m/s on a STATIC
    vehicle while the KF repeatedly reset itself. That is a "blatantly wrong value", not
    a slow drift, and every downstream consumer of the estimate inherits it.

A corpus collected with only the nominal (well-behaved, mm-level) sensor model teaches the
encoder to trust those two channels completely. This module corrupts them during training
so the encoder learns to fall back on what is actually trustworthy - the gyro, the
specific force, the previous action and the recurrent history - instead of hallucinating a
plant parameter from a state estimate that is currently lying.

WHAT IS CORRUPTED, AND WHY IT MUST BE COHERENT
The actor frame is not 29 independent numbers: eight of its channels are DERIVED from the
estimate, and the environment computes them from the same sample it exposes as
`pos`/`vel`/`omega`:

    p_err = ref.p  - p_est        v_err = ref.v  - v_est
    w_err = ref.omega - omega     att_err = rotvec(R_ref^T R(quat))

So a corruption that only touches `pos` produces a frame that CANNOT OCCUR on the
vehicle: an estimate that is 5 m off with a p_err that still reflects the old estimate.
Training on that pairing teaches the model about a signal it will never see, and (worse)
leaves the easy tell "p_err is inconsistent with pos" available as a shortcut. Every
family below therefore applies the EXACT first-order correction to the derived channels:

    pos    += dp   ->  p_err   -= dp
    vel    += dv   ->  v_err   -= dv
    omega  += dw   ->  w_err   -= dw
    quat    <- quat * exp(dtheta/2)  ->  att_err += dtheta     (BCH to first order)

and the drift TARGETS move with the estimate, because a target of "how wrong is the
estimate" must stay true of the frame the model is actually handed:

    est_drift_p += dp,  est_drift_v += dv

WHICH CHANNELS ARE LEFT ALONE
`stale` deliberately does NOT hold the gyro, the specific force or the battery. Those are
separate sensors that keep working when the pose estimate stops updating, and preserving
them is the entire point of the augmentation: the model has to learn that they are the
reliable half of its input. Only the estimator-provided channels go stale.

NON-GOALS
This is a DATA transform, not a loss change. mu is still trained with MSE (see
train_encoder.py: under NLL the loss and R^2 decouple, and MSE / beta-NLL / NLL were
measured to give identical R^2 on this project). Robustness comes from the input
distribution, not from a robust loss kernel.

Standalone and numpy-only, so it can be unit-tested without torch or the plant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------------
# ACTOR FRAME LAYOUT
#
# MIRRORED from quad_flip_env.py (O_POS ... O_W_ERR). Duplicated here rather than
# imported so this module stays free of the MuJoCo import chain, and protected by
# `verify_layout()`, which train_encoder.py calls before touching a single frame. Drift
# between this copy and the environment is therefore a hard error, not a silent
# mis-application of every corruption below.
# ---------------------------------------------------------------------------------
O_POS, O_QUAT, O_OMEGA = 0, 3, 7
O_VELXY, O_VELZ = 10, 12
O_PREV_ACTION = 13
O_P_ERR, O_V_ERR, O_ATT_ERR, O_W_ERR = 17, 20, 23, 26
ACTOR_FRAME_DIM: int = 29

AUX_OFFSET: int = ACTOR_FRAME_DIM          # specific force (3) then v_batt_norm (1)
AUX_FORCE_DIM: int = 3
FRAME_DIM: int = ACTOR_FRAME_DIM + 4       # 33


def verify_layout() -> str:
    """
    Cross-check the mirrored offsets against the environment, if it is importable.

    Returns a short description of what was checked. Raises RuntimeError on a mismatch -
    a wrong offset here corrupts the corpus silently, which is exactly the failure class
    this project has been bitten by before (the stale `obs[:51]` slices).
    """
    import sys
    import os

    _here = os.path.dirname(os.path.abspath(__file__))
    _sim = os.path.dirname(_here)
    if _sim not in sys.path:
        sys.path.insert(0, _sim)
    try:
        import quad_flip_env as _env  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - encoder must work without the plant
        return f"quad_flip_env not importable ({type(exc).__name__}); layout NOT cross-checked"

    pairs = [
        ("O_POS", O_POS, _env.O_POS),
        ("O_QUAT", O_QUAT, _env.O_QUAT),
        ("O_OMEGA", O_OMEGA, _env.O_OMEGA),
        ("O_VELXY", O_VELXY, _env.O_VELXY),
        ("O_VELZ", O_VELZ, _env.O_VELZ),
        ("O_PREV_ACTION", O_PREV_ACTION, _env.O_PREV_ACTION),
        ("O_P_ERR", O_P_ERR, _env.O_P_ERR),
        ("O_V_ERR", O_V_ERR, _env.O_V_ERR),
        ("O_ATT_ERR", O_ATT_ERR, _env.O_ATT_ERR),
        ("O_W_ERR", O_W_ERR, _env.O_W_ERR),
        ("ACTOR_FRAME_DIM", ACTOR_FRAME_DIM, _env.ACTOR_SINGLE_OBS_DIM),
    ]
    bad = [(n, a, b) for n, a, b in pairs if int(a) != int(b)]
    if bad:
        raise RuntimeError(
            "corruption.py's mirrored actor-frame layout disagrees with quad_flip_env: "
            + ", ".join(f"{n}: corruption={a} env={b}" for n, a, b in bad)
        )
    return f"layout matches quad_flip_env ({len(pairs)} offsets)"


# ---------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------
@dataclass
class CorruptionConfig:
    """
    Envelope for the corrupted-estimate augmentation.

    Defaults are deliberately WIDE and deliberately biased towards big errors. The
    nominal sensor model already produces mm-level noise on every frame of every episode,
    so there is nothing to be gained by re-adding small corruptions - what the corpus is
    missing is the tail: total staleness, and values so wrong that no filter would have
    produced them. Those are the ones the encoder must survive.
    """

    enabled: bool = True

    # Probability that an episode receives any corruption at all. Not 1.0: the clean
    # half of the corpus is what keeps the nominal-accuracy gates (and the 5 gated
    # physics targets) where they are.
    episode_prob: float = 0.5
    # Number of corrupted blocks drawn per corrupted episode, and their length in steps.
    # 3-50 steps = 30-500 ms, which spans a single-station dropout up to a long occlusion.
    n_blocks_range: Tuple[int, int] = (1, 3)
    block_len_range: Tuple[int, int] = (3, 50)
    # A FLIP IS A 2.2 s BLACKOUT (the deck is inverted, so every station is behind it) and
    # the dead-reckoned drift over that long is metre-scale. A corpus of half-second stales
    # only teaches the encoder about short gaps, which is the easy half of the problem, so
    # this fraction of blocks is drawn at full-blackout length instead.
    long_block_prob: float = 0.35
    long_len_range: Tuple[int, int] = (60, 250)

    # Family probabilities. Only names present in `families` are used, so the mapping is
    # by NAME and can never be silently misaligned by reordering.
    p_stale: float = 0.35      # zero-order hold of the estimator channels
    p_jump: float = 0.25       # gross outlier: teleport, or collapse towards the origin
    p_ramp: float = 0.25       # slow random-walk bias (the dead-reckoning shape)
    p_imu: float = 0.15        # IMU-side bias/scale + attitude jitter
    families: Tuple[str, ...] = ("stale", "jump", "ramp", "imu")

    # Gross outlier magnitudes. 20 m is a fraction of the 98 m runaway measured on
    # hardware, and far past anything the flight envelope produces (references peak
    # under 2 m).
    jump_pos_range: Tuple[float, float] = (0.5, 20.0)      # m
    jump_vel_range: Tuple[float, float] = (1.0, 20.0)      # m/s
    # Probability that a jump COLLAPSES the estimate towards the origin (reporting a pose
    # near zero) instead of offsetting it. Both are real estimator failure shapes.
    jump_collapse_prob: float = 0.25

    # Slow-bias magnitudes. The measured hardware divergence was ~0.5-1 m/s^2 of effective
    # horizontal acceleration, i.e. a few degrees of tilt bias leaking into gravity.
    ramp_walk_range: Tuple[float, float] = (0.02, 0.50)    # m/s^2 per sqrt(s)
    ramp_vel_walk_frac: float = 0.5                        # velocity walk vs the above

    # IMU-side: gyro bias (rad/s), specific-force bias (m/s^2), attitude jitter (deg).
    imu_gyro_bias_range: Tuple[float, float] = (0.02, 0.50)
    imu_accel_bias_range: Tuple[float, float] = (0.05, 0.80)
    imu_att_deg_range: Tuple[float, float] = (0.5, 5.0)
    v_batt_scale_range: Tuple[float, float] = (0.85, 1.15)

    # Hard magnitude guard, applied after every family: a draw can never inject an
    # inf/NaN into the corpus, and the standardization clip (±10 sigma) is never the
    # thing that decides what the model sees.
    max_abs: float = 1.0e4

    def normalised_probs(self) -> Tuple[Tuple[str, ...], np.ndarray]:
        """Active family names and their normalised probabilities (empty if all zero)."""
        by_name: Dict[str, float] = {
            "stale": float(self.p_stale),
            "jump": float(self.p_jump),
            "ramp": float(self.p_ramp),
            "imu": float(self.p_imu),
        }
        names = tuple(f for f in self.families if by_name.get(f, 0.0) > 0.0)
        if not names:
            return (), np.zeros(0, dtype=np.float64)
        p = np.asarray([by_name[f] for f in names], dtype=np.float64)
        return names, p / p.sum()


# ---------------------------------------------------------------------------------
# the transform
# ---------------------------------------------------------------------------------
def _apply_estimate_delta(
    frames: np.ndarray,
    sl: slice,
    dp: Optional[np.ndarray] = None,
    dv: Optional[np.ndarray] = None,
    dw: Optional[np.ndarray] = None,
    dtheta: Optional[np.ndarray] = None,
) -> None:
    """
    Apply a COHERENT estimate corruption in place on frames[sl].

    Exact identities, first-order in dtheta (see the module docstring): the derived error
    channels move by the negative of the estimate delta, so the frame stays one a real
    estimator could have emitted. Deltas may be [3] (constant over the block) or [L, 3].
    """
    if dp is not None:
        frames[sl, O_POS:O_POS + 3] += dp
        frames[sl, O_P_ERR:O_P_ERR + 3] -= dp
    if dv is not None:
        frames[sl, O_VELXY:O_VELXY + 2] += dv[..., :2]
        frames[sl, O_VELZ] += dv[..., 2]
        frames[sl, O_V_ERR:O_V_ERR + 3] -= dv
    if dw is not None:
        frames[sl, O_OMEGA:O_OMEGA + 3] += dw
        frames[sl, O_W_ERR:O_W_ERR + 3] -= dw
    if dtheta is not None:
        # quat <- quat * exp(dtheta/2): a BODY-frame rotation, matching the attitude
        # jitter the environment itself applies in _compute_actor_obs.
        q = frames[sl, O_QUAT:O_QUAT + 4].astype(np.float64)
        dth = np.broadcast_to(np.asarray(dtheta, dtype=np.float64), q[:, :3].shape)
        half = 0.5 * dth
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        dx, dy, dz = half[:, 0], half[:, 1], half[:, 2]
        out = np.column_stack([
            qw - qx * dx - qy * dy - qz * dz,
            qx + qw * dx + qy * dz - qz * dy,
            qy - qx * dz + qw * dy + qz * dx,
            qz + qx * dy - qy * dx + qw * dz,
        ])
        out /= np.maximum(1e-12, np.linalg.norm(out, axis=1, keepdims=True))
        out = np.where(out[:, :1] < 0.0, -out, out)   # env keeps w >= 0 per row
        frames[sl, O_QUAT:O_QUAT + 4] = out.astype(frames.dtype)
        # BCH to first order: log(exp(att_err) exp(dtheta)) ~= att_err + dtheta.
        frames[sl, O_ATT_ERR:O_ATT_ERR + 3] += dtheta


def _shift_drift_targets(
    targets: np.ndarray,
    sl: slice,
    drift_slice: Optional[Tuple[int, int]],
    dp: Optional[np.ndarray],
    dv: Optional[np.ndarray],
) -> None:
    """Move the drift targets with the estimate so the supervision stays true."""
    if drift_slice is None:
        return
    a, b = drift_slice
    if dp is not None:
        targets[sl, a:a + 3] += dp
    if dv is not None:
        targets[sl, a + 3:b] += dv


def corrupt_episode(
    frames: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
    cfg: CorruptionConfig,
    drift_slice: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Corrupt one episode in the RAW (physical) frame domain.

    :param frames: [T, 33] float32, the encoder frame as the env produced it.
    :param targets: [T, n_targets] float32. Modified only on the drift slice.
    :param drift_slice: (start, stop) of [est_drift_p(3), est_drift_v(3)] in `targets`,
        or None when the corpus predates those targets (then targets are left untouched).
    :returns: (frames_out, targets_out, mask[T] float32) - the mask is 1.0 on corrupted
        steps. The mask is for REPORTING (clean vs corrupted R^2), never for masking the
        loss: training on the corrupted frames is the entire point.

    The inputs are copied; the caller's arrays are never modified.
    """
    frames = np.array(frames, dtype=np.float32, copy=True)
    targets = np.array(targets, dtype=np.float32, copy=True)
    T = int(frames.shape[0])
    mask = np.zeros(T, dtype=np.float32)

    if not cfg.enabled or T == 0 or rng.random() >= cfg.episode_prob:
        return frames, targets, mask

    names, probs = cfg.normalised_probs()
    if not names:
        return frames, targets, mask

    n_lo, n_hi = int(cfg.n_blocks_range[0]), int(cfg.n_blocks_range[1])
    n_lo = max(1, n_lo)
    n_hi = max(n_lo, n_hi)

    for _ in range(int(rng.integers(n_lo, n_hi + 1))):
        fam = str(rng.choice(np.asarray(names, dtype=object), p=probs))
        lo_b, hi_b = (cfg.long_len_range if rng.random() < cfg.long_block_prob
                      else cfg.block_len_range)
        lo_b, hi_b = max(1, int(lo_b)), max(1, int(hi_b))
        if hi_b < lo_b:
            lo_b, hi_b = hi_b, lo_b
        L = int(rng.integers(lo_b, hi_b + 1))
        if L >= T:                             # short episode: leave at least one clean step
            L = max(1, T - 1)
        start = int(rng.integers(0, T - L + 1)) if T > L else 0
        sl = slice(start, start + L)
        mask[sl] = 1.0
        _apply_family(frames, targets, sl, fam, rng, cfg, drift_slice)

    np.nan_to_num(frames, copy=False, nan=0.0, posinf=cfg.max_abs, neginf=-cfg.max_abs)
    np.nan_to_num(targets, copy=False, nan=0.0, posinf=cfg.max_abs, neginf=-cfg.max_abs)
    np.clip(frames, -cfg.max_abs, cfg.max_abs, out=frames)
    np.clip(targets, -cfg.max_abs, cfg.max_abs, out=targets)
    return frames, targets, mask


def _apply_family(
    frames: np.ndarray,
    targets: np.ndarray,
    sl: slice,
    fam: str,
    rng: np.random.Generator,
    cfg: CorruptionConfig,
    drift_slice: Optional[Tuple[int, int]],
) -> None:
    """Apply one corruption family over frames[sl] (and the matching drift targets)."""
    if fam == "stale":
        # Zero-order hold of the ESTIMATOR channels: pos, vel and their derived errors are
        # frozen at the last clean sample. The gyro, specific force and battery keep
        # updating - that asymmetry is the teaching signal.
        i = int(sl.start)
        if i == 0:
            return                             # nothing before the episode to hold
        for a, b in ((O_POS, O_POS + 3), (O_VELXY, O_VELXY + 3),
                     (O_P_ERR, O_P_ERR + 3), (O_V_ERR, O_V_ERR + 3)):
            frames[sl, a:b] = frames[i - 1, a:b]
        if drift_slice is not None:
            d0, d1 = drift_slice
            targets[sl, d0:d1] = targets[i - 1, d0:d1]
        return

    if fam == "jump":
        if rng.random() < cfg.jump_collapse_prob:
            # COLLAPSE: the estimator reports a pose near the origin instead of the
            # vehicle's - a constant, large, structured error.
            here = frames[int(sl.start), O_POS:O_POS + 3].astype(np.float64)
            dp = rng.normal(0.0, 0.25, size=3) - here
            dv = rng.normal(0.0, 1.0, size=3)
        else:
            # TELEPORT: a signed offset in a random direction. z glitches less than xy,
            # because the barometer fusion usually survives what breaks the xy fix.
            dp = rng.uniform(*cfg.jump_pos_range) * rng.normal(size=3)
            dp[2] *= 0.25
            dv = (rng.uniform(*cfg.jump_vel_range) * rng.normal(size=3)
                  if rng.random() > 0.35 else np.zeros(3))
        _apply_estimate_delta(frames, sl, dp, dv)
        _shift_drift_targets(targets, sl, drift_slice, dp, dv)
        return

    if fam == "ramp":
        # A random walk on the estimate, with the position integral of the velocity walk
        # on top: the shape a dead-reckoned estimate actually takes.
        walk = float(rng.uniform(*cfg.ramp_walk_range))
        vwalk = walk * cfg.ramp_vel_walk_frac
        n = sl.stop - sl.start
        dv = np.cumsum(rng.normal(0.0, max(vwalk, 1e-6), size=(n, 3)), axis=0)
        dp = np.cumsum(dv, axis=0)
        _apply_estimate_delta(frames, sl, dp, dv)
        _shift_drift_targets(targets, sl, drift_slice, dp, dv)
        return

    if fam == "imu":
        gw = rng.uniform(*cfg.imu_gyro_bias_range) * rng.normal(size=3)
        _apply_estimate_delta(frames, sl, dw=gw)
        dtheta = np.radians(rng.uniform(*cfg.imu_att_deg_range)) * rng.normal(size=3)
        _apply_estimate_delta(frames, sl, dtheta=dtheta)
        f0 = AUX_OFFSET
        frames[sl, f0:f0 + AUX_FORCE_DIM] += (
            rng.uniform(*cfg.imu_accel_bias_range) * rng.normal(size=3)
        ).astype(frames.dtype)
        vb = AUX_OFFSET + AUX_FORCE_DIM
        frames[sl, vb] *= float(rng.uniform(*cfg.v_batt_scale_range))
        return
