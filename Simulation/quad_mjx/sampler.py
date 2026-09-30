"""
Trajectory sampler: port of ``TrajectorySampler`` (draws, screens, rejection loop).

The numpy sampler draws a candidate, screens it at 96 (256 for a chain) points and redraws
up to ``max_resample`` times, falling back to a hover.  JAX cannot branch per draw on
traced values, so the same algorithm runs as a fixed-size ``lax.fori_loop``: every attempt
draws its candidate, screens it, and the FIRST accepted candidate is kept.  Same semantics,
fixed shapes.

A single manoeuvre and a chain share one code path: a single manoeuvre is just
``n_parts == 1`` with an identity relocation (see ``trajectories.TrajSpec``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import spec
from .trajectories import (
    IDX_CHAIN, IDX_FLIP, IDX_HOVER, IDX_V8, MAX_DEG, MAX_PARTS, MAX_SEG, PAR_DIM,
    S_A, S_AX, S_B, S_C0, S_CLIMB_T, S_COAST, S_CZ, S_DOWN, S_DUR, S_EASE,
    S_HA, S_HB, S_HC, S_HEAD, S_HOLD_T, S_K, S_NWP, S_P0, S_P1, S_PHI, S_RFRAC,
    S_SETTLE, S_SEGT, S_THR_DN, S_THR_UP, S_UP, S_W, S_WPEAK, S_YAW, S_YAW_R, S_Z0,
    GRAVITY, MASS, MAX_THRUST_TOTAL, TRAJECTORY_TAIL, TrajSpec,
    _pose_part, rotz, sample as traj_sample, yaw_of,
)

DU = 48            # uniforms per segment slot
NG = 8             # global uniforms per attempt

UI_KIND, UI_P0X, UI_P0Y, UI_YAW, UI_YAWR = 0, 1, 2, 3, 4
UI_DUR, UI_A, UI_B, UI_W, UI_CLIMB = 5, 6, 7, 8, 9
UI_EASE, UI_SETTLE = 10, 11
UI_FLIP_K, UI_FLIP_OM, UI_FLIP_RF, UI_FLIP_AX = 12, 13, 14, 15
UI_LISS_A, UI_LISS_B, UI_LISS_PHI, UI_LISS_C = 16, 17, 18, 19
UI_HEAD, UI_DIST, UI_LAT, UI_CYC = 20, 21, 22, 23
UI_TO_ZSTART, UI_TO_ZTGT, UI_TO_CLIMB, UI_TO_HOLD = 24, 25, 26, 27
UI_WP = 28          # 28..36 = three waypoint targets (x, y, z each)
UI_SEGT = 37
UI_SPARE0, UI_SPARE1, UI_SPARE2 = 38, 39, 40

# CHAIN_KINDS / CHAIN_KIND_WEIGHTS from TrajectorySampler.
CHAIN_KIND_INDEX = jnp.array([IDX_FLIP, IDX_V8, 4, 3, 5, 2], dtype=jnp.int32)  # flip,v8,orbit,f8,liss,wp
CHAIN_KIND_WEIGHTS = jnp.array([0.26, 0.24, 0.14, 0.12, 0.10, 0.14])

N_SCREEN_MAX = 256


def _u(un, lo, hi):
    return lo + un * (hi - lo)


def _choice(un, vals):
    vals = jnp.asarray(vals)
    i = jnp.clip((un * vals.shape[0]).astype(jnp.int32), 0, vals.shape[0] - 1)
    return vals[i]


def _weighted_choice(un, n, weights):
    c = jnp.cumsum(weights)
    return jnp.clip(jnp.searchsorted(c, un, side="right"), 0, n - 1)


def _pp(k, d, u):
    """d-th derivative of u**k wrt u, k a Python int (build time only)."""
    if k < d:
        return 0.0
    c = 1.0
    for j in range(d):
        c *= (k - j)
    return c * u ** (k - d)


def _wp_fit(wps):
    """
    Hermite fit, port of ``WaypointTrajectory._fit``: position at each knot plus zero
    velocity, acceleration AND jerk at both ends -> K + 6 conditions, degree K + 5.
    """
    K = wps.shape[0]
    n = K + 6
    u_knots = jnp.linspace(0.0, 1.0, K)
    rows = [jnp.stack([jnp.asarray(_pp(k, 0, u_knots[i])) for k in range(n)]) for i in range(K)]
    for (u, d) in ((0.0, 1), (1.0, 1), (0.0, 2), (1.0, 2), (0.0, 3), (1.0, 3)):
        rows.append(jnp.stack([jnp.asarray(_pp(k, d, u)) for k in range(n)]))
    A = jnp.stack(rows)
    b = jnp.concatenate([wps, jnp.zeros((6, 3))], axis=0)
    coef = jnp.linalg.solve(A, b)
    return jnp.concatenate([coef, jnp.zeros((MAX_DEG - n, 3))], axis=0)


def _empty_par():
    return jnp.zeros(PAR_DIM)


# ======================================================================================
# one builder per family; every `chain` decision is a jnp.where on a TRACED flag
# ======================================================================================
def _par_hover(u, cfg, chain):
    p0 = jnp.array([_u(u[UI_P0X], -cfg["bounds_xy"], cfg["bounds_xy"]),
                    _u(u[UI_P0Y], -cfg["bounds_xy"], cfg["bounds_xy"]), cfg["spawn_z"]])
    p0 = jnp.where(chain, jnp.array([0.0, 0.0, cfg["spawn_z"]]), p0)
    return (_empty_par().at[S_P0:S_P0 + 3].set(p0)
            .at[S_YAW].set(_u(u[UI_YAW], -jnp.pi, jnp.pi))
            .at[S_YAW_R].set(_u(u[UI_YAWR], -1.5, 1.5))
            .at[S_DUR].set(_u(u[UI_DUR], 2.5, 6.0)))


def _par_takeoff(u, cfg, chain):
    z_start = jnp.where(u[UI_TO_ZSTART] < 0.50, 0.025, _u(u[UI_TO_ZSTART], 0.025, 1.20))
    p0 = jnp.array([_u(u[UI_P0X], -0.35, 0.35), _u(u[UI_P0Y], -0.35, 0.35), z_start])
    p1 = jnp.array([_u(u[UI_SPARE1], -0.40, 0.40), _u(u[UI_LAT], -0.40, 0.40),
                    _u(u[UI_TO_ZTGT], 0.90, 1.50)])
    climb = jnp.maximum(1.2, _u(u[UI_TO_CLIMB], 1.6, 2.8))
    hold = _u(u[UI_TO_HOLD], 2.5, 5.0)
    return (_empty_par().at[S_P0:S_P0 + 3].set(p0).at[S_P1:S_P1 + 3].set(p1)
            .at[S_CLIMB_T].set(climb).at[S_HOLD_T].set(hold)
            .at[S_YAW].set(_u(u[UI_YAW], -jnp.pi, jnp.pi))
            .at[S_YAW_R].set(_u(u[UI_YAWR], -1.5, 1.5))
            .at[S_DUR].set(climb + hold))


def _par_waypoints(u, cfg, chain):
    """Only the geometry-independent slots; the knots are fitted in _build_slot."""
    seg_t = jnp.where(chain, _u(u[UI_SEGT], 0.8, 1.1), _u(u[UI_SEGT], 0.9, 1.6))
    n_knots = jnp.where(chain, 3.0, 4.0)          # chain: 3 targets; standalone: p0 + 3
    return (_empty_par().at[S_NWP].set(n_knots).at[S_SEGT].set(seg_t)
            .at[S_YAW].set(_u(u[UI_YAW], -jnp.pi, jnp.pi))
            .at[S_DUR].set(seg_t * (n_knots - 1.0)))


def _par_fig8(u, cfg, chain):
    z0 = cfg["spawn_z"]
    w = jnp.where(chain, _u(u[UI_W], 1.5, 2.0),
                  jnp.maximum(_u(u[UI_W], 0.8, 1.8), cfg["min_w"]))
    A = jnp.where(chain, _u(u[UI_A], 0.20, 0.32), _u(u[UI_A], 0.20, 0.45))
    B = jnp.where(chain, _u(u[UI_B], 0.20, 0.32), _u(u[UI_B], 0.15, 0.35))
    T = 2.0 * jnp.pi / jnp.maximum(w, 1e-6)
    settle_c = _u(u[UI_SETTLE], 0.6, 0.9)
    settle_s = jnp.minimum(0.35 * T, 1.2)
    return (_empty_par().at[S_A].set(A).at[S_B].set(B).at[S_W].set(w).at[S_Z0].set(z0)
            .at[S_EASE].set(jnp.where(chain, settle_c, 0.0))
            .at[S_SETTLE].set(jnp.where(chain, settle_c, settle_s))
            .at[S_DUR].set(T).at[S_YAW].set(0.0)
            .at[S_YAW_R].set(_u(u[UI_YAWR], -1.5, 1.5)))


def _par_orbit(u, cfg, chain):
    T_c = _u(u[UI_DUR], 2.8, 3.6)
    T_s = _u(u[UI_DUR], 3.0, 6.0)
    R = jnp.where(chain, _u(u[UI_A], 0.20, 0.45), _u(u[UI_A], 0.20, 0.50))
    climb = jnp.where(chain, _u(u[UI_CLIMB], -0.15, 0.20), _u(u[UI_CLIMB], -0.20, 0.30))
    turns = jnp.where(chain, 1.0, _choice(u[UI_CYC], jnp.array([1.0, 1.0, 2.0])))
    T = jnp.where(chain, T_c, T_s)
    w = jnp.where(chain, 1.2, turns * 2.0 * jnp.pi / jnp.maximum(T, 1e-6))
    ease_c = _u(u[UI_EASE], 0.6, 0.9)
    return (_empty_par().at[S_P1:S_P1 + 3].set(jnp.array([0.0, 0.0, cfg["spawn_z"]]))
            .at[S_A].set(R).at[S_W].set(w).at[S_Z0].set(climb)
            .at[S_EASE].set(jnp.where(chain, ease_c, 0.0))
            .at[S_SETTLE].set(jnp.where(chain, ease_c, jnp.minimum(0.35 * T, 1.2)))
            .at[S_DUR].set(T)
            .at[S_YAW].set(jnp.where(chain, 0.0, _u(u[UI_YAW], -jnp.pi, jnp.pi)))
            .at[S_YAW_R].set(jnp.where(chain, 0.0, _u(u[UI_YAWR], -1.5, 1.5))))


def _par_lissajous(u, cfg, chain):
    z0 = cfg["spawn_z"]
    w = jnp.where(chain, _u(u[UI_W], 1.5, 2.0),
                  jnp.maximum(_u(u[UI_W], 0.7, 1.7), cfg["min_w"]))
    A = jnp.where(chain, _u(u[UI_A], 0.15, 0.28), _u(u[UI_A], 0.15, 0.40))
    B = jnp.where(chain, _u(u[UI_B], 0.15, 0.28), _u(u[UI_B], 0.15, 0.40))
    C = jnp.where(chain, _u(u[UI_LISS_C], 0.0, 0.10), _u(u[UI_LISS_C], 0.0, 0.12))
    ha = _choice(u[UI_LISS_A], jnp.array([1.0, 1.0, 2.0]))
    hb = _choice(u[UI_LISS_B], jnp.array([2.0, 3.0]))
    T = 2.0 * jnp.pi / jnp.maximum(w, 1e-6)
    settle_c = _u(u[UI_SETTLE], 0.6, 0.9)
    return (_empty_par().at[S_A].set(A).at[S_B].set(B).at[S_W].set(w).at[S_Z0].set(z0)
            .at[S_HA].set(ha).at[S_HB].set(hb).at[S_HC].set(2.0).at[S_CZ].set(C)
            .at[S_PHI].set(_u(u[UI_LISS_PHI], 0.0, jnp.pi))
            .at[S_EASE].set(jnp.where(chain, settle_c, 0.0))
            .at[S_SETTLE].set(jnp.where(chain, settle_c, jnp.minimum(0.35 * T, 1.2)))
            .at[S_DUR].set(T)
            .at[S_YAW].set(jnp.where(chain, 0.0, _u(u[UI_YAW], -jnp.pi, jnp.pi)))
            .at[S_YAW_R].set(jnp.where(chain, 0.0, _u(u[UI_YAWR], -1.5, 1.5))))


def _par_slalom(u, cfg, chain):
    z0 = cfg["spawn_z"]
    heading = _u(u[UI_HEAD], -jnp.pi, jnp.pi)
    dist = _u(u[UI_DIST], 0.5, 1.3)
    u_vec = jnp.array([jnp.cos(heading), jnp.sin(heading), 0.0])
    start = -u_vec * (dist * 0.5)
    A = _u(u[UI_LAT], 0.10, 0.35)
    cycles = _choice(u[UI_CYC], jnp.array([2.0, 2.5, 3.0]))
    T = _u(u[UI_DUR], 2.2, 4.0)
    w = cycles * 2.0 * jnp.pi / jnp.maximum(T, 1e-6)
    return (_empty_par().at[S_P0:S_P0 + 3].set(start).at[S_HEAD].set(heading)
            .at[S_A].set(dist).at[S_B].set(A).at[S_W].set(w).at[S_Z0].set(z0)
            .at[S_EASE].set(0.0).at[S_SETTLE].set(jnp.minimum(0.35 * T, 1.2))
            .at[S_DUR].set(T)
            .at[S_YAW].set(_u(u[UI_YAW], -jnp.pi, jnp.pi))
            .at[S_YAW_R].set(_u(u[UI_YAWR], -1.5, 1.5)))


def _par_v8(u, cfg, chain):
    T = _u(u[UI_DUR], 2.0, 2.8)
    w = 2.0 * jnp.pi / jnp.maximum(T, 1e-6)
    return (_empty_par()
            .at[S_A].set(_u(u[UI_A], 0.22, 0.38))
            .at[S_B].set(_u(u[UI_B], 0.16, 0.30)).at[S_W].set(w)
            .at[S_HEAD].set(_u(u[UI_HEAD], -jnp.pi, jnp.pi))
            .at[S_EASE].set(_u(u[UI_EASE], 0.40, 0.50) * T)
            .at[S_SETTLE].set(_u(u[UI_SETTLE], 0.40, 0.50) * T)
            .at[S_DUR].set(T)
            .at[S_YAW].set(jnp.where(chain, 0.0, _u(u[UI_YAW], -jnp.pi, jnp.pi)))
            .at[S_YAW_R].set(jnp.where(chain, 0.0, _u(u[UI_YAWR], -1.5, 1.5))))


def _par_flip(u, cfg, chain):
    """Port of ``_make_flip`` + ``Flip.__init__`` (saturating powered phases)."""
    use_pitch = u[UI_FLIP_AX] < 0.75
    axis = jnp.where(use_pitch, jnp.array([0.0, 1.0, 0.0]), jnp.array([1.0, 0.0, 0.0]))
    limit = jnp.where(use_pitch, cfg["rate_pitch"], cfg["rate_roll"])
    k = 1.0
    rate_frac = jnp.clip(_u(u[UI_FLIP_RF], 0.28, 0.35), 0.0, 0.9)
    omega_target = _u(u[UI_FLIP_OM], 0.60, 0.70) * limit
    coast = 2.0 * jnp.pi * k / jnp.maximum(1e-6, omega_target * (1.0 - rate_frac))

    v0 = GRAVITY * coast / 2.0
    a_max_net = 0.90 * MAX_THRUST_TOTAL / MASS - GRAVITY
    uu = jnp.maximum(1e-3, a_max_net)
    c0 = v0 / uu
    omega_peak = 2.0 * jnp.pi * k / jnp.maximum(1e-6, coast * (1.0 - rate_frac))
    thrust = MASS * (uu + GRAVITY)
    yaw = jnp.where(chain, 0.0, _u(u[UI_YAW], -jnp.pi, jnp.pi))
    return (_empty_par().at[S_AX:S_AX + 3].set(axis)
            .at[S_K].set(k).at[S_COAST].set(coast).at[S_C0].set(c0)
            .at[S_UP].set(uu).at[S_DOWN].set(uu).at[S_RFRAC].set(rate_frac)
            .at[S_WPEAK].set(omega_peak).at[S_THR_UP].set(thrust).at[S_THR_DN].set(thrust)
            .at[S_YAW].set(yaw).at[S_DUR].set(c0 + coast + c0))


def _flip_pre_ok(par, cfg):
    """Port of ``Flip.is_feasible`` with the sampler's z_max / cage ceiling."""
    D, c0, uu = par[S_COAST], par[S_C0], par[S_UP]
    v0 = GRAVITY * D / 2.0
    exc = v0 ** 2 / (2.0 * uu) + v0 ** 2 / (2.0 * GRAVITY)
    lo = par[S_P0 + 2]
    cage_z = cfg.get("cage_ceiling_z", cfg["z_max"])
    return ((par[S_THR_UP] <= 0.95 * MAX_THRUST_TOTAL) & (par[S_THR_UP] >= 0.0)
            & (par[S_WPEAK] <= 0.95 * cfg["max_rate"])
            & (c0 >= 0.08) & (D >= 0.08)
            & (lo >= 0.30) & (lo + exc <= cage_z)
            & jnp.isfinite(c0 + D + uu))


FAMILY_BUILDERS = (
    _par_hover, _par_takeoff, _par_waypoints, _par_fig8, _par_orbit,
    _par_lissajous, _par_slalom, _par_v8, _par_flip,
)


def _build_slot(u, cfg, chain, kind):
    """Build par + waypoint coefficients + the draw's own screen for one slot."""
    p0_std = jnp.array([_u(u[UI_P0X], -cfg["bounds_xy"], cfg["bounds_xy"]),
                        _u(u[UI_P0Y], -cfg["bounds_xy"], cfg["bounds_xy"]), cfg["spawn_z"]])
    p0_chain = jnp.array([0.0, 0.0, cfg["spawn_z"]])
    p0_v8 = jnp.array([_u(u[UI_P0X], -0.30, 0.30), _u(u[UI_P0Y], -0.30, 0.30),
                       cfg["spawn_z"]])
    p0 = jnp.where(kind == IDX_V8, p0_v8, p0_std)
    p0 = jnp.where(chain, p0_chain, p0)

    par = _empty_par()
    # DO NOT put this behind `lax.switch`.  It looks like free work saved (only one family
    # can be the answer) but under `vmap` -- which is how the training pool draws -- a
    # BATCHED predicate makes XLA compute EVERY branch and select, so a switch changes
    # nothing and only adds dispatch.  Measured 2026-09-23, vmap(cond) with one heavy lane
    # in 256: 18.1 us vs 4.2 us for the light-only path = both branches run.  See
    # `scratch/bench_mjx.py` section `sampler`.
    for i, f in enumerate(FAMILY_BUILDERS):
        par = jnp.where(kind == i, f(u, cfg, chain), par)

    # Flip pop excursion clamping to fit 2.0 m cage ceiling
    v0_flip = GRAVITY * par[S_COAST] / 2.0
    exc_flip = v0_flip ** 2 / (2.0 * jnp.maximum(1e-3, par[S_UP])) + v0_flip ** 2 / (2.0 * GRAVITY)
    cage_z = cfg.get("cage_ceiling_z", cfg["z_max"])
    p0_flip_z = jnp.clip(cfg["spawn_z"], 1.10, jnp.maximum(1.10, cage_z - exc_flip - 0.05))
    p0_flip_std = jnp.array([p0_std[0], p0_std[1], p0_flip_z])
    p0_flip_chain = jnp.array([0.0, 0.0, p0_flip_z])

    p0_slot = jnp.where(
        kind == IDX_FLIP,
        jnp.where(chain, p0_flip_chain, p0_flip_std),
        jnp.where(kind == IDX_V8,
                  jnp.where(chain, p0_chain, p0_v8),
                  jnp.where(chain, p0_chain, p0_std))
    )
    par = par.at[S_P0:S_P0 + 3].set(p0_slot)

    # waypoint knots: chain = 3 targets from the canonical start; standalone = p0 + 3
    bb = cfg["waypoint_bounds_xy"]
    zr = cfg["z_range"]
    tgts = []
    for i in range(3):
        b = UI_WP + 3 * i
        tgts.append(jnp.array([_u(u[b], -bb, bb), _u(u[b + 1], -bb, bb),
                               _u(u[b + 2], zr[0], zr[1])]))
    tgt_stack = jnp.stack(tgts)                       # (3,3)
    wps_std = jnp.concatenate([p0_std[None, :], tgt_stack], axis=0)       # K = 4
    wps_chain = tgt_stack                                                 # K = 3
    # both fits run on purpose: see the note above `FAMILY_BUILDERS` -- a `lax.cond` here
    # would be computed on both sides anyway under vmap.  The 9x9/10x10 solves are not the
    # sampler's bottleneck; the rejection loop's SCREEN is.
    coef = jnp.where(chain, _wp_fit(wps_chain), _wp_fit(wps_std))

    pre_ok = jnp.where(kind == IDX_FLIP, _flip_pre_ok(par, cfg), jnp.asarray(True))
    return par, coef, pre_ok


# ======================================================================================
# assembly
# ======================================================================================
def _assemble(kinds, pars, coefs, durs, n_parts, is_chain, reloc_R, reloc_shift, starts,
              maneuver_duration, pre_ok, n_screen):
    # pin int32: under jax_enable_x64 a python int literal becomes int64, and a scan carry
    # whose dtype changes between iterations is a hard error
    n_parts = jnp.asarray(n_parts, jnp.int32)
    wp_deg = jnp.zeros(MAX_PARTS)
    i_last = jnp.clip(n_parts - 1, 0, MAX_PARTS - 1)
    p, _v, _a, R, spin, _t = _pose_part(
        kinds[i_last], pars[i_last], coefs[i_last], wp_deg[i_last], durs[i_last])
    Rz, sh = reloc_R[i_last], reloc_shift[i_last]
    return TrajSpec(
        kind=kinds, par=pars, wp_coef=coefs, wp_deg=wp_deg, starts=starts,
        reloc_R=reloc_R, reloc_shift=reloc_shift, n_parts=n_parts, is_chain=is_chain,
        duration=maneuver_duration, total=maneuver_duration + TRAJECTORY_TAIL,
        hold_p=Rz @ p + sh, hold_R=Rz @ R, hold_spin=spin,
        hold_thrust=jnp.asarray(MASS * GRAVITY), n_screen=n_screen, pre_ok=pre_ok,
    )


def _screen(spec_, cfg):
    """
    Thrust / rate / volume / footprint / episode-length screens.

    The grid is rebuilt PER RESOLUTION over the WHOLE trajectory (96 points normally, 256
    for a chain, exactly as the numpy sampler does).  A single grid with a truncated
    validity mask would only screen the first fraction of the trace -- measured: a chain
    passed the screen with |xy| = 1.31 m against a 0.75 m footprint.
    """
    offset = jnp.array([0.0, 0.0, cfg["spawn_z"]])

    def metrics(n_pts):
        ts = jnp.linspace(0.0, spec_.total, n_pts)
        refs = jax.vmap(lambda t: traj_sample(spec_, t))(ts)
        return (refs.thrust_ff.max(), refs.thrust_ff.min(),
                jnp.linalg.norm(refs.omega, axis=-1).max(),
                jnp.linalg.norm(refs.p - offset, axis=-1).max(),
                jnp.max(jnp.abs(refs.p[:, :2])).max(),
                refs.p[:, 2].max())

    # Both grids are computed and selected ON PURPOSE -- see the note above
    # `FAMILY_BUILDERS`.  A `lax.cond` here is not a saving: the predicate is a BATCHED
    # value under vmap, so XLA evaluates both branches regardless.  Measured neutral
    # (1.03x, i.e. noise) in `scratch/bench_mjx.py` section `sampler`.
    fine = metrics(cfg["n_screen_chain"])
    coarse = metrics(cfg["n_screen"])
    thr_max, thr_min, rate, dist, foot, z_peak = jax.tree_util.tree_map(
        lambda a, b: jnp.where(spec_.is_chain, a, b), fine, coarse)

    cage_z = cfg.get("cage_ceiling_z", cfg["z_max"])
    ok = ((thr_max <= 0.95 * MAX_THRUST_TOTAL) & (thr_min >= 0.0)
          & (rate <= 0.95 * cfg["max_rate"])
          & (dist <= 0.85 * cfg["flight_radius"])
          & (foot <= cfg["bounds_xy"])
          & (z_peak <= cage_z)
          & (spec_.total <= cfg["episode_seconds"]))
    return spec_.pre_ok & ok


def _fallback_spec(cfg):
    """``Trajectory(Hover([0, 0, 1.2], 0.0, duration=2.0))``."""
    par = (_empty_par().at[S_P0:S_P0 + 3].set(jnp.array([0.0, 0.0, 1.2]))
           .at[S_DUR].set(2.0))
    pars = jnp.tile(par, (MAX_PARTS, 1))
    kinds = jnp.zeros(MAX_PARTS, dtype=jnp.int32)
    coefs = jnp.zeros((MAX_PARTS, MAX_DEG, 3))
    durs = jnp.where(jnp.arange(MAX_PARTS) == 0, 2.0, 0.0)
    starts = jnp.cumsum(jnp.concatenate([jnp.zeros(1), durs[:-1]]))
    reloc_R = jnp.tile(jnp.eye(3), (MAX_PARTS, 1, 1))
    reloc_shift = jnp.zeros((MAX_PARTS, 3))
    return _assemble(kinds, pars, coefs, durs, jnp.array(1, dtype=jnp.int32),
                     jnp.asarray(False), reloc_R, reloc_shift, starts,
                     jnp.array(2.0), jnp.asarray(True), jnp.asarray(cfg["n_screen"]))


# ======================================================================================
# the sampler
# ======================================================================================
def sample(key, cfg, weights, mass=MASS, kind_pin=-1, max_resample=40):
    """
    Draw a feasible trajectory (port of ``TrajectorySampler.sample``).

    ``weights`` is the live per-family draw-weight vector, so the training-time mixture
    curriculum (raising ``chain``) works exactly as it does in the numpy env.
    """
    keys = jax.random.split(key, max_resample)

    def attempt(carry, k):
        found, best = carry
        ku, kg = jax.random.split(k)
        g = jax.random.uniform(kg, (NG,))

        chosen = jnp.where(kind_pin >= 0, jnp.array(kind_pin, dtype=jnp.int32),
                           _weighted_choice(g[0], spec.N_KINDS, weights))
        is_chain = chosen == IDX_CHAIN
        n_seg = _choice(g[1], jnp.array([2, 3], jnp.int32)).astype(jnp.int32)
        hold = _u(g[2], 0.30, 0.55)
        station = jnp.array([_u(g[3], -0.15, 0.15), _u(g[4], -0.15, 0.15),
                             cfg["spawn_z"]])
        yaw0 = _u(g[5], -jnp.pi, jnp.pi)

        U = jax.random.uniform(ku, (MAX_SEG, DU))
        chain_sel = _weighted_choice(U[:, UI_KIND], CHAIN_KIND_INDEX.shape[0],
                                     CHAIN_KIND_WEIGHTS)
        chain_kind = CHAIN_KIND_INDEX[chain_sel]

        kinds = jnp.zeros(MAX_PARTS, dtype=jnp.int32)
        pars = jnp.zeros((MAX_PARTS, PAR_DIM))
        coefs = jnp.zeros((MAX_PARTS, MAX_DEG, 3))
        durs = jnp.zeros(MAX_PARTS)
        pre_ok = jnp.ones(MAX_PARTS, dtype=bool)

        for s in range(MAX_SEG):
            in_chain = is_chain | (s > 0)
            kind_s = jnp.where(in_chain, chain_kind[s], chosen)
            par_s, coef_s, ok_s = _build_slot(U[s], cfg, in_chain, kind_s)
            kinds = kinds.at[2 * s].set(kind_s)
            pars = pars.at[2 * s].set(par_s)
            coefs = coefs.at[2 * s].set(coef_s)
            durs = durs.at[2 * s].set(par_s[S_DUR])
            pre_ok = pre_ok.at[2 * s].set(ok_s)

        hpar = (_empty_par().at[S_P0:S_P0 + 3].set(jnp.zeros(3))
                .at[S_YAW].set(0.0).at[S_YAW_R].set(0.0).at[S_DUR].set(hold))
        for s in range(MAX_SEG - 1):
            nxt = chain_kind[s + 1]
            hpar_s = hpar.at[S_DUR].set(
                jnp.where(is_chain & (s + 1 < n_seg), hold, 0.0))
            kinds = kinds.at[2 * s + 1].set(jnp.where(is_chain, IDX_HOVER, IDX_HOVER))
            pars = pars.at[2 * s + 1].set(hpar_s)
            durs = durs.at[2 * s + 1].set(jnp.where(is_chain & (s + 1 < n_seg), hold, 0.0))
            _ = nxt

        n_parts = jnp.where(is_chain, 2 * n_seg - 1, jnp.asarray(1, jnp.int32))
        active = jnp.arange(MAX_PARTS) < n_parts
        durs = jnp.where(active, durs, 0.0)
        starts = jnp.cumsum(jnp.concatenate([jnp.zeros(1), durs[:-1]]))
        maneuver_duration = jnp.sum(durs)

        def reloc_step(carry, i):
            st_p, st_yaw = carry

            def canon(t):
                return _pose_part(kinds[i], pars[i], coefs[i], jnp.zeros(()), t)

            p_s, _v, _a, R_s, _s1, _t1 = canon(0.0)
            p_e, _v2, _a2, R_e, _s2, _t2 = canon(durs[i])
            dyaw = jnp.where(is_chain, st_yaw - yaw_of(R_s), 0.0)
            Rz = rotz(dyaw)
            shift = jnp.where(is_chain, st_p - Rz @ p_s, jnp.zeros(3))
            return (Rz @ p_e + shift, yaw_of(Rz @ R_e)), (Rz, shift)

        (_, _), (reloc_R, reloc_shift) = jax.lax.scan(
            reloc_step, (station, yaw0), jnp.arange(MAX_PARTS), length=MAX_PARTS)

        live_ok = jnp.all(jnp.where(active, pre_ok, True))
        cand = _assemble(kinds, pars, coefs, durs, n_parts, is_chain,
                         reloc_R, reloc_shift, starts, maneuver_duration, live_ok,
                         jnp.where(is_chain, cfg["n_screen_chain"], cfg["n_screen"]))
        ok = _screen(cand, cfg)
        take = ok & (~found)
        merged = jax.tree_util.tree_map(lambda a, b: jnp.where(take, a, b), cand, best)
        return (found | ok, merged), None

    # `max_resample` is a CAP, not a cost.  A `lax.scan` over it pays for all 40 attempts
    # on every draw even though the numpy `while` loop almost always stops in the first
    # couple -- measured at 9.5 ms per trajectory draw, 94% of an env reset.  A
    # `while_loop` keeps the exact same draw ORDER and first-success semantics, and under
    # vmap it runs until the slowest lane is satisfied (typically a handful of attempts).
    init = (jnp.asarray(False), _fallback_spec(cfg), jnp.asarray(0, jnp.int32))

    def cond(carry):
        found, _best, i = carry
        return (~found) & (i < max_resample)

    def body(carry):
        found, best, i = carry
        k_i = jax.lax.dynamic_index_in_dim(keys, i, axis=0, keepdims=False)
        (found, best), _ = attempt((found, best), k_i)
        return (found, best, i + 1)

    _, best, _ = jax.lax.while_loop(cond, body, init)
    return best


# ======================================================================================
# config helpers (mirror of trajectories.TrajectoryConfig + its default weights)
# ======================================================================================
def sample_batch(keys, cfg, weights, mass=MASS):
    """
    ``sample`` vmapped over a batch of keys -- ONE batched draw.

    This is the training path (TIP 1).  The rejection loop is a `while_loop` whose predicate
    is data-dependent, so under vmap it runs until the SLOWEST lane has found a feasible
    candidate -- which is why the batch should be sized to the number of resets a rollout
    will actually need, not to the environment count.
    """
    return jax.vmap(lambda k: sample(k, cfg, weights, mass=mass, kind_pin=-1))(keys)


def default_weights():
    """TrajectoryConfig.weights in spec.KIND_NAMES order."""
    return jnp.array([0.15, 0.10, 0.10, 0.06, 0.09, 0.08, 0.08, 0.09, 0.20, 0.05])


def traj_cfg(episode_seconds=spec.EPISODE_SECONDS, n_screen=96, n_screen_chain=256):
    """The sampler's configuration, as a dict of host constants (ranges are tuples)."""
    return {
        "spawn_z": spec.SPAWN_Z,
        "flight_radius": spec.FLIGHT_RADIUS,
        "bounds_xy": 0.75,
        "waypoint_bounds_xy": 0.50,
        "z_range": (1.2, 2.0),
        "cage_ceiling_z": spec.CAGE_CEILING_Z,
        "z_max": spec.CAGE_CEILING_Z,
        "max_resample": 40,
        "episode_seconds": float(episode_seconds),
        "rate_roll": 20.0,
        "rate_pitch": 20.0,
        "max_rate": 20.0,
        "n_screen": int(n_screen),
        "n_screen_chain": int(n_screen_chain),
        "min_w": 2.0 * 3.141592653589793 / max(1e-6, float(episode_seconds) - TRAJECTORY_TAIL),
    }
