"""
JAX port of ``Simulation/lighthouse.py`` -- the Crazyflie Lighthouse deck model.

Ported faithfully: station geometry drawn INSIDE the deck's own acceptance cone, cone
occlusion about body +z, per-station dropout, a minimum station count for a fix, dead
reckoning with a tilt-error / accelerometer-bias random walk, and the four-guard fix
plausibility gate with its streak breaker.

WHY IT MATTERS (from the baseline's own header): the deck is the only absolute
position/velocity source, a FLIP is a modelled blackout, and a policy trained on truth
learns a control law that assumes a sensor the real vehicle does not have during the exact
manoeuvre this project exists for.

DIAGNOSTIC STRINGS.  ``_fix_is_plausible`` returns a reason string in numpy; strings are
not JAX-friendly, so this port returns an integer code and ``REJECT_*`` maps it back on the
host for logging.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

MAX_STATIONS: int = 4                   # max(LighthouseConfig.station_counts)
GRAVITY: float = 9.81

# Fix-plausibility verdicts (mirrors the numpy reason strings).
REJECT_NONE, REJECT_NONFINITE, REJECT_JUMP, REJECT_DV, REJECT_RANGE, REJECT_FORCED = range(6)
REJECT_NAMES: tuple = (
    "accepted", "non-finite fix", "jump", "velocity step", "range", "forced",
)


@struct.dataclass
class LighthouseConfig:
    """Per-episode installation envelope.  Defaults are the baseline's."""
    station_counts: tuple = (2, 3, 4)
    cone_frac_min: float = 0.30
    cone_frac_max: float = 0.80
    dist_lo: float = 1.5
    dist_hi: float = 4.0
    azimuth_jitter: float = 0.25
    half_angle_deg_lo: float = 55.0
    half_angle_deg_hi: float = 80.0
    max_range: float = 6.0
    min_stations_for_fix: int = 2
    pos_noise_lo: float = 0.003
    pos_noise_hi: float = 0.010
    vel_noise_lo: float = 0.020
    vel_noise_hi: float = 0.060
    z_noise_scale: float = 2.0
    dropout_lo: float = 0.0
    dropout_hi: float = 0.03
    tilt_walk_lo: float = 0.010
    tilt_walk_hi: float = 0.040
    accel_walk_lo: float = 0.02
    accel_walk_hi: float = 0.10
    fix_decimation: int = 1
    max_fix_jump: float = 5.0
    max_fix_dv: float = 4.0
    max_reject_streak: int = 15
    max_fix_range: float = 0.0          # 0 disables; the env sets 4 * FLIGHT_RADIUS


@struct.dataclass
class LighthouseState:
    """Everything the sensor model owns, including the estimate itself."""
    stations: jax.Array          # (MAX_STATIONS, 3)
    active: jax.Array            # (MAX_STATIONS,) bool
    n_stations: jax.Array        # int32
    half_angle: jax.Array        # scalar
    anchor: jax.Array            # (3,)
    p_est: jax.Array             # (3,)
    v_est: jax.Array             # (3,)
    tilt_err: jax.Array          # (2,)   random-walk gravity-projection error
    accel_bias: jax.Array        # (3,)
    outage_t: jax.Array          # scalar seconds since last accepted fix
    has_fix: jax.Array           # bool
    n_visible: jax.Array         # int32, after dropout
    step_count: jax.Array        # int32
    # per-episode noise scales (dr-interpolated once at reset)
    s_pos: jax.Array
    s_vel: jax.Array
    s_drop: jax.Array
    s_tilt: jax.Array
    s_accel: jax.Array
    # gate bookkeeping
    reject_streak: jax.Array     # int32
    n_fix_rejected: jax.Array    # int32
    n_fix_forced: jax.Array      # int32
    n_nonfinite: jax.Array       # int32
    last_reject: jax.Array       # int32 code


def _lerp(lo: float, hi: float, dr):
    return lo + jnp.clip(dr, 0.0, 1.0) * (hi - lo)


def reset(cfg: LighthouseConfig, st: LighthouseState, key, p0, v0, dr) -> LighthouseState:
    """
    Draw a fresh installation and seat the estimate at truth.

    Station count / geometry are ALWAYS randomised (an installation property); only the
    noise magnitudes scale with ``dr``.
    """
    k_count, k_geom, k_alpha, k_dist = jax.random.split(key, 4)

    counts = jnp.asarray(cfg.station_counts, dtype=jnp.int32)
    n_stations = counts[jax.random.randint(k_count, (), 0, counts.shape[0])]

    half_angle = jnp.radians(jax.random.uniform(
        k_geom, (), minval=cfg.half_angle_deg_lo, maxval=cfg.half_angle_deg_hi))

    # Even nominal bearings about a random base, plus jitter.  Slot i is valid only for
    # i < n_stations; the spacing itself depends on n_stations (a 2-station and a
    # 4-station room are different installations, not the same one sampled twice).
    k_base, k_jit = jax.random.split(k_geom)
    base = jax.random.uniform(k_base, (), minval=0.0, maxval=2.0 * jnp.pi)
    idx = jnp.arange(MAX_STATIONS, dtype=jnp.float32)
    bearing = base + 2.0 * jnp.pi * idx / jnp.maximum(n_stations, 1)
    bearing = bearing + jax.random.uniform(
        k_jit, (MAX_STATIONS,), minval=-cfg.azimuth_jitter, maxval=cfg.azimuth_jitter)

    # Placement angle as a FRACTION of the very cone the visibility test uses, which is
    # what guarantees an upright deck sees every station by construction.
    frac = jax.random.uniform(k_alpha, (MAX_STATIONS,),
                              minval=0.0, maxval=cfg.cone_frac_max)
    alpha = jnp.maximum(frac, cfg.cone_frac_min) * half_angle
    dist = jax.random.uniform(k_dist, (MAX_STATIONS,), minval=cfg.dist_lo, maxval=cfg.dist_hi)

    anchor = jnp.asarray(p0, dtype=jnp.float32)
    stations = anchor[None, :] + jnp.stack([
        dist * jnp.sin(alpha) * jnp.cos(bearing),
        dist * jnp.sin(alpha) * jnp.sin(bearing),
        dist * jnp.cos(alpha),
    ], axis=-1)
    active = idx < n_stations

    return st.replace(
        stations=stations,
        active=active,
        n_stations=n_stations,
        half_angle=half_angle,
        anchor=anchor,
        p_est=jnp.asarray(p0, dtype=jnp.float32),
        v_est=jnp.asarray(v0, dtype=jnp.float32),
        tilt_err=jnp.zeros(2),
        accel_bias=jnp.zeros(3),
        outage_t=jnp.array(0.0),
        has_fix=jnp.array(True),
        n_visible=n_stations,
        step_count=jnp.array(0, dtype=jnp.int32),
        s_pos=_lerp(cfg.pos_noise_lo, cfg.pos_noise_hi, dr),
        s_vel=_lerp(cfg.vel_noise_lo, cfg.vel_noise_hi, dr),
        s_drop=_lerp(cfg.dropout_lo, cfg.dropout_hi, dr),
        s_tilt=_lerp(cfg.tilt_walk_lo, cfg.tilt_walk_hi, dr),
        s_accel=_lerp(cfg.accel_walk_lo, cfg.accel_walk_hi, dr),
        reject_streak=jnp.array(0, dtype=jnp.int32),
        n_fix_rejected=jnp.array(0, dtype=jnp.int32),
        n_fix_forced=jnp.array(0, dtype=jnp.int32),
        n_nonfinite=jnp.array(0, dtype=jnp.int32),
        last_reject=jnp.array(REJECT_NONE, dtype=jnp.int32),
    )


def visible_mask(cfg: LighthouseConfig, st: LighthouseState, p, R_world):
    """
    Which stations can illuminate the deck: inside the acceptance cone about body +z, in
    range, and present.  Inverted => body +z points at the floor => every station drops.
    """
    to_station = st.stations - p[None, :]
    dist = jnp.linalg.norm(to_station, axis=1)
    ok_range = dist <= cfg.max_range
    unit = to_station / jnp.maximum(dist, 1e-9)[:, None]
    deck_normal = R_world[:, 2]
    cos_ang = unit @ deck_normal
    return st.active & ok_range & (cos_ang >= jnp.cos(st.half_angle))


def _fix_is_plausible(cfg: LighthouseConfig, st: LighthouseState, p_raw, v_raw, dt):
    """Returns (ok, code).  Same four guards, same order, as the numpy original."""
    finite = jnp.all(jnp.isfinite(p_raw)) & jnp.all(jnp.isfinite(v_raw))

    p_pred = st.p_est + st.v_est * dt
    jump = jnp.linalg.norm(p_raw - p_pred)
    ok_jump = jump <= cfg.max_fix_jump

    max_dv = cfg.max_fix_dv + GRAVITY * st.outage_t
    dv = jnp.linalg.norm(v_raw - st.v_est)
    ok_dv = dv <= max_dv

    span = jnp.linalg.norm(p_raw - st.anchor)
    ok_range = (cfg.max_fix_range <= 0.0) | (span <= cfg.max_fix_range)

    ok = finite & ok_jump & ok_dv & ok_range

    code = jnp.where(
        ~finite, REJECT_NONFINITE,
        jnp.where(~ok_jump, REJECT_JUMP,
                  jnp.where(~ok_dv, REJECT_DV,
                            jnp.where(~ok_range, REJECT_RANGE, REJECT_NONE))))
    return ok, code.astype(jnp.int32)


def observe(cfg: LighthouseConfig, st: LighthouseState, p_true, v_true, R_world, dt, key,
            min_stations=None):
    """
    One sensor + estimator step.  Returns (new_state, out) where ``out`` carries the
    estimate the controller must use, plus the diagnostics the baseline reports.

    ``min_stations`` overrides ``cfg.min_stations_for_fix`` for this step.  The failure
    injector uses it to starve fixes through the model's OWN path (has_fix simply can
    never be true and the estimate falls through to dead reckoning), which is exactly how
    the numpy LighthouseFailure implementation does it.
    """
    k_vis, k_fix = jax.random.split(key)
    step = st.step_count + 1

    # -- geometry, then per-station dropout on top --------------------------------
    vis = visible_mask(cfg, st, p_true, R_world)
    k_drop, k_fix = jax.random.split(k_fix)
    keep = jax.random.uniform(k_drop, (MAX_STATIONS,)) >= st.s_drop
    vis = vis & keep
    n_vis = jnp.sum(vis).astype(jnp.int32)

    need = cfg.min_stations_for_fix if min_stations is None else min_stations
    decimated = (step % jnp.maximum(cfg.fix_decimation, 1)) == 0
    has_fix = (n_vis >= need) & decimated

    # -- raw fix, drawn then TESTED (a rejected fix falls through to dead reckoning) --
    sigma = jnp.array([st.s_pos, st.s_pos, st.s_pos * cfg.z_noise_scale])
    k_p, k_v = jax.random.split(k_fix)
    p_raw = p_true + jax.random.normal(k_p, (3,)) * sigma
    v_raw = v_true + jax.random.normal(k_v, (3,)) * st.s_vel

    ok, code = _fix_is_plausible(cfg, st, p_raw, v_raw, dt)

    # The gate may veto, but not forever: after `max_reject_streak` consecutive
    # rejections it stands down and takes the fix, because a persistent disagreement
    # means the DEAD-RECKONED estimate is the wrong one.
    standdown = st.reject_streak >= cfg.max_reject_streak
    accept = has_fix & (ok | standdown)

    rejected_now = has_fix & (~ok) & (~standdown)
    forced_now = has_fix & (~ok) & standdown

    # -- update the estimate -------------------------------------------------------
    def dead_reckon(s):
        s_tilt = s.tilt_err + jax.random.normal(
            jax.random.fold_in(k_fix, 1), (2,)) * s.s_tilt * jnp.sqrt(dt)
        s_acc = s.accel_bias + jax.random.normal(
            jax.random.fold_in(k_fix, 2), (3,)) * s.s_accel * jnp.sqrt(dt)
        # A tilt error mis-projects gravity into ~g*theta of spurious HORIZONTAL
        # acceleration; it integrates twice, so position error grows ~quadratically.
        a_drift = jnp.array([GRAVITY * s_tilt[0], GRAVITY * s_tilt[1], s_acc[2]])
        v_new = s.v_est + a_drift * dt
        p_new = s.p_est + v_new * dt
        return p_new, v_new, s_tilt, s_acc

    p_dr, v_dr, tilt_dr, acc_dr = dead_reckon(st)

    p_est = jnp.where(accept, p_raw, p_dr)
    v_est = jnp.where(accept, v_raw, v_dr)
    # A fix re-anchors attitude and velocity, so the accumulated error states are mostly
    # cancelled -- retained at 25%, as a real EKF would not fully trust one update.
    tilt_err = jnp.where(accept, st.tilt_err * 0.25, tilt_dr)
    accel_bias = jnp.where(accept, st.accel_bias * 0.25, acc_dr)

    outage_t = jnp.where(accept, 0.0, st.outage_t + dt)

    # Last line of defence: np.clip does NOT remove NaN, and one NaN here would poison
    # every observation and the GRU state for the rest of the episode.
    finite = jnp.all(jnp.isfinite(p_est)) & jnp.all(jnp.isfinite(v_est))
    p_est = jnp.where(finite, p_est, jnp.nan_to_num(p_est))
    v_est = jnp.where(finite, v_est, jnp.nan_to_num(v_est))

    new = st.replace(
        p_est=p_est,
        v_est=v_est,
        tilt_err=tilt_err,
        accel_bias=accel_bias,
        outage_t=outage_t,
        has_fix=accept,
        n_visible=n_vis,
        step_count=step,
        reject_streak=jnp.where(accept, 0,
                                jnp.where(rejected_now, st.reject_streak + 1, st.reject_streak)),
        n_fix_rejected=st.n_fix_rejected + rejected_now.astype(jnp.int32),
        n_fix_forced=st.n_fix_forced + forced_now.astype(jnp.int32),
        n_nonfinite=st.n_nonfinite + (~finite).astype(jnp.int32),
        last_reject=jnp.where(
            accept, jnp.array(REJECT_FORCED, dtype=jnp.int32) * forced_now
                    + jnp.array(REJECT_NONE, dtype=jnp.int32),
            jnp.where(rejected_now, code, st.last_reject)),
    )

    out = {
        "p_est": p_est,
        "v_est": v_est,
        "has_fix": accept,
        "n_visible": n_vis,
        "outage_t": outage_t,
        "drifted": jnp.linalg.norm(p_est - p_true),
        "n_fix_rejected": new.n_fix_rejected,
        "n_fix_forced": new.n_fix_forced,
        "last_reject": new.last_reject,
    }
    return new, out


def empty_state(cfg: LighthouseConfig) -> LighthouseState:
    """A zeroed state so the pytree has the right shapes before the first reset."""
    return LighthouseState(
        stations=jnp.zeros((MAX_STATIONS, 3)),
        active=jnp.zeros(MAX_STATIONS, dtype=bool),
        n_stations=jnp.array(0, dtype=jnp.int32),
        half_angle=jnp.array(0.0),
        anchor=jnp.zeros(3),
        p_est=jnp.zeros(3),
        v_est=jnp.zeros(3),
        tilt_err=jnp.zeros(2),
        accel_bias=jnp.zeros(3),
        outage_t=jnp.array(0.0),
        has_fix=jnp.array(False),
        n_visible=jnp.array(0, dtype=jnp.int32),
        step_count=jnp.array(0, dtype=jnp.int32),
        s_pos=jnp.array(0.0),
        s_vel=jnp.array(0.0),
        s_drop=jnp.array(0.0),
        s_tilt=jnp.array(0.0),
        s_accel=jnp.array(0.0),
        reject_streak=jnp.array(0, dtype=jnp.int32),
        n_fix_rejected=jnp.array(0, dtype=jnp.int32),
        n_fix_forced=jnp.array(0, dtype=jnp.int32),
        n_nonfinite=jnp.array(0, dtype=jnp.int32),
        last_reject=jnp.array(REJECT_NONE, dtype=jnp.int32),
    )
