"""
JAX port of ``Simulation/trajectories.py`` -- the reference repertoire.

All 10 families are ported (hover, takeoff, waypoints, figure8, orbit, lissajous, slalom,
v8, flip, chain), together with the sampler's feasibility / volume / footprint /
episode-length screens and its rejection loop.

HOW THE PORT IS STRUCTURED
A trajectory is a fixed-shape ``TrajSpec`` pytree:

    parts[0..n_parts)   a manoeuvre (or, for a chain, the segments and the hover beats
                        between them)
    starts[.]           absolute start time of each part
    reloc_R/reloc_shift the rigid relocation applied to each part

A single manoeuvre is simply ``n_parts == 1`` with an IDENTITY relocation, which makes the
whole thing uniform: ``Chain``'s exact junctions and ``ShiftedManeuver``'s rigid motion are
the same mechanism, computed once at draw time by a scan that walks the parts and threads
the station pose through them (``_relocate_parts``).

FAMILY PARAMETERS live in a flat ``par`` vector of ``PAR_DIM`` floats, with the slot layout
documented in ``_pose_part``.  Slots are reused between families because exactly one family
is active per part.

DEVIATION FROM THE NUMPY ORIGINAL: ``WaypointTrajectory`` is always built with 4 or 5
knots exactly as the sampler draws it, but the polynomial coefficients are SOLVED at draw
time and stored in the spec (``wp_coef``), because solving an 11x11 system per sample would
be wasteful inside the inner loop.  The fit itself is the same Hermite system.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from . import spec

GRAVITY = spec.GRAVITY
MASS = spec.MASS
MAX_THRUST_TOTAL = spec.MAX_THRUST
TRAJECTORY_TAIL = 1.0

MAX_PARTS = 7                  # 2 * MAX_SEG - 1
MAX_SEG = 4
MAX_DEG = 12                   # waypoint polynomial degree cap (n = K + 6 <= 11)

# --------------------------------------------------------------------------------------
# par-slot layout (see _pose_part)
# --------------------------------------------------------------------------------------
S_P0, S_P1, S_WP2, S_WP3, S_WP4 = 0, 3, 6, 9, 12
S_A, S_B, S_W, S_Z0 = 15, 16, 17, 18
S_EASE, S_SETTLE, S_DUR, S_YAW, S_YAW_R = 19, 20, 21, 22, 23
S_HA, S_HB, S_HC, S_CZ, S_PHI = 24, 25, 26, 27, 28
S_HEAD = 29
S_AX, S_K, S_COAST, S_C0, S_UP, S_DOWN, S_RFRAC, S_WPEAK = 30, 33, 34, 35, 36, 37, 38, 39
S_THR_UP, S_THR_DN, S_CLIMB_T, S_HOLD_T, S_NWP, S_SEGT = 40, 41, 42, 43, 44, 45
PAR_DIM = 48

KH = spec.KIND_INDEX          # name -> index
IDX_HOVER = KH["hover"]
IDX_TAKEOFF = KH["takeoff"]
IDX_WAYPOINTS = KH["waypoints"]
IDX_FIG8 = KH["figure8"]
IDX_ORBIT = KH["orbit"]
IDX_LISSAJOUS = KH["lissajous"]
IDX_SLALOM = KH["slalom"]
IDX_V8 = KH["v8"]
IDX_FLIP = KH["flip"]
IDX_CHAIN = KH["chain"]


# ======================================================================================
# rotation helpers (port of _normalize / axis_angle_rotation /
# dcm_from_thrust_dir_and_yaw / omega_from_dcm)
# ======================================================================================
def normalize(v, fallback=(0.0, 0.0, 1.0)):
    n = jnp.linalg.norm(v)
    fb = jnp.asarray(fallback)
    return jnp.where(n < 1e-9, fb, v / jnp.maximum(n, 1e-12))


def axis_angle_rotation(axis, angle):
    k = normalize(axis)
    K = jnp.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return jnp.eye(3) + jnp.sin(angle) * K + (1.0 - jnp.cos(angle)) * (K @ K)


def rotz(angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def dcm_from_thrust_dir_and_yaw(z_b, yaw):
    """
    R (body->world) with body z along z_b and heading pinned to yaw.  Same construction
    as the numpy original including the heading-degenerate quarter-turn fallback.
    """
    z_b = normalize(z_b)
    zx, zy, zz = z_b[0], z_b[1], z_b[2]

    def build(yaw_eff):
        cy, sy = jnp.cos(yaw_eff), jnp.sin(yaw_eff)
        y = jnp.array([zy * 0.0 - zz * sy, zz * cy - zx * 0.0, zx * sy - zy * cy])
        return y

    y = build(yaw)
    y = jnp.where(jnp.linalg.norm(y) < 1e-6, build(yaw + 0.5 * jnp.pi), y)
    y = normalize(y)
    x = jnp.cross(y, z_b) * 0.0 + jnp.array([
        y[1] * z_b[2] - y[2] * z_b[1],
        y[2] * z_b[0] - y[0] * z_b[2],
        y[0] * z_b[1] - y[1] * z_b[0],
    ])
    return jnp.stack([x, y, z_b], axis=1)


def omega_from_dcm(R_prev, R_mid, R_next, h):
    Rdot = (R_next - R_prev) / (2.0 * h)
    W = R_mid.T @ Rdot
    return 0.5 * jnp.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]])


def yaw_of(R):
    return jnp.arctan2(R[1, 0], R[0, 0])


def smoothstep(t):
    t = jnp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ======================================================================================
# envelopes (port of _settle_envelope / _rest_envelope)
# ======================================================================================
def settle_envelope(t, T, L):
    inside = (L <= 1e-9) | (t <= T - L)
    Ls = jnp.maximum(L, 1e-9)
    s = jnp.clip((t - (T - L)) / Ls, 0.0, 1.0)
    ds = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / Ls
    dds = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (Ls * Ls)
    e = 1.0 - (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)
    return (jnp.where(inside, 1.0, e), jnp.where(inside, 0.0, -ds),
            jnp.where(inside, 0.0, -dds))


def rest_envelope(t, T, L0, L1):
    L0s = jnp.maximum(L0, 1e-9)
    L1s = jnp.maximum(L1, 1e-9)
    s0 = jnp.clip(t / L0s, 0.0, 1.0)
    d0 = (30.0 * s0**2 - 60.0 * s0**3 + 30.0 * s0**4) / L0s
    dd0 = (60.0 * s0 - 180.0 * s0**2 + 120.0 * s0**3) / (L0s * L0s)
    e0 = 10.0 * s0**3 - 15.0 * s0**4 + 6.0 * s0**5

    s1 = jnp.clip((t - (T - L1)) / L1s, 0.0, 1.0)
    d1 = (30.0 * s1**2 - 60.0 * s1**3 + 30.0 * s1**4) / L1s
    dd1 = (60.0 * s1 - 180.0 * s1**2 + 120.0 * s1**3) / (L1s * L1s)
    e1 = 1.0 - (10.0 * s1**3 - 15.0 * s1**4 + 6.0 * s1**5)

    at_start = (L0 > 1e-9) & (t < L0)
    at_end = (~at_start) & (L1 > 1e-9) & (t > T - L1)
    e = jnp.where(at_start, e0, jnp.where(at_end, e1, 1.0))
    de = jnp.where(at_start, d0, jnp.where(at_end, -d1, 0.0))
    dde = jnp.where(at_start, dd0, jnp.where(at_end, -dd1, 0.0))
    return e, de, dde


def _yaw_eased(t, duration, yaw, yaw_rate):
    """Maneuver._yaw: quintic-eased heading with zero rate at both ends."""
    u = jnp.clip(t / jnp.maximum(duration, 1e-9), 0.0, 1.0)
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    eased = yaw + yaw_rate * duration * s
    return jnp.where(jnp.abs(yaw_rate) < 1e-9, yaw, eased)


def _flat_attitude(a, yaw):
    return dcm_from_thrust_dir_and_yaw(a + jnp.array([0.0, 0.0, GRAVITY]), yaw)


def _required_thrust(a):
    return MASS * jnp.linalg.norm(a + jnp.array([0.0, 0.0, GRAVITY]))


def _quintic(t, T):
    """s, ds, dds for the 0->1 quintic on [0, T]."""
    u = jnp.clip(t / jnp.maximum(T, 1e-9), 0.0, 1.0)
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / jnp.maximum(T, 1e-9)
    dds = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / jnp.maximum(T, 1e-9) ** 2
    return s, ds, dds


# ======================================================================================
# per-family pose
# ======================================================================================
def _pose_hover(par, t):
    p0 = par[S_P0:S_P0 + 3]
    duration, yaw, yaw_rate = par[S_DUR], par[S_YAW], par[S_YAW_R]
    a = jnp.zeros(3)
    return (p0, jnp.zeros(3), a,
            _flat_attitude(a, _yaw_eased(t, duration, yaw, yaw_rate)),
            0.0, _required_thrust(a))


def _pose_takeoff(par, t):
    p0, p1 = par[S_P0:S_P0 + 3], par[S_P1:S_P1 + 3]
    tc = jnp.maximum(par[S_CLIMB_T], 0.5)
    delta = p1 - p0
    u = jnp.clip(t / tc, 0.0, 1.0)
    u2, u3, u4, u5, u6, u7 = u**2, u**3, u**4, u**5, u**6, u**7
    s = 35.0 * u4 - 84.0 * u5 + 70.0 * u6 - 20.0 * u7
    ds = (140.0 * u3 - 420.0 * u4 + 420.0 * u5 - 140.0 * u6) / tc
    d2s = (420.0 * u2 - 1680.0 * u3 + 2100.0 * u4 - 840.0 * u5) / (tc * tc)
    climbing = (t > 0.0) & (t < tc)
    p = jnp.where(climbing, p0 + s * delta, jnp.where(t <= 0.0, p0, p1))
    v = jnp.where(climbing, ds * delta, jnp.zeros(3))
    a = jnp.where(climbing, d2s * delta, jnp.zeros(3))
    return (p, v, a, _flat_attitude(a, _yaw_eased(t, par[S_DUR], par[S_YAW], par[S_YAW_R])),
            0.0, _required_thrust(a))


def _poly_pow(k, d, u):
    """
    The d-th derivative of u**k with respect to u:  k*(k-1)*...*(k-d+1) * u**(k-d),
    and exactly 0 when k < d.  ``d`` is a Python int so the loop is unrolled statically
    (multiplying all four factors regardless of d would silently give the 4th derivative
    for every d).
    """
    fac = jnp.ones_like(k)
    for j in range(d):
        fac = fac * jnp.where(k >= j, k - j, 1.0)
    fac = jnp.where(k < d, 0.0, fac)
    return fac * jnp.power(u, jnp.maximum(k - d, 0.0))


def _pose_waypoints(par, wp_coef, wp_deg, t):
    duration = jnp.maximum(par[S_DUR], 1e-9)
    u = t / duration
    ks = jnp.arange(MAX_DEG, dtype=jnp.float32)

    def deriv(d):
        pw = jax.vmap(lambda k: _poly_pow(k, d, u))(ks)
        return (pw @ wp_coef) / duration ** d

    p, v, a = deriv(0), deriv(1), deriv(2)
    return p, v, a, _flat_attitude(a, par[S_YAW]), 0.0, _required_thrust(a)


def _pose_fig8(par, t):
    A, B, w, z0 = par[S_A], par[S_B], par[S_W], par[S_Z0]
    T, ease, settle = par[S_DUR], par[S_EASE], par[S_SETTLE]
    ph = w * t
    lem = jnp.array([A * jnp.sin(ph), 0.5 * B * jnp.sin(2.0 * ph), 0.0])
    lem_v = jnp.array([A * w * jnp.cos(ph), B * w * jnp.cos(2.0 * ph), 0.0])
    lem_a = jnp.array([-A * w * w * jnp.sin(ph), -2.0 * B * w * w * jnp.sin(2.0 * ph), 0.0])
    e_s, de_s, dde_s = settle_envelope(t, T, settle)
    e_r, de_r, dde_r = rest_envelope(t, T, ease, settle)
    use_rest = ease > 0.0
    e = jnp.where(use_rest, e_r, e_s)
    de = jnp.where(use_rest, de_r, de_s)
    dde = jnp.where(use_rest, dde_r, dde_s)
    p = jnp.array([0.0, 0.0, z0]) + e * lem
    v = de * lem + e * lem_v
    a = dde * lem + 2.0 * de * lem_v + e * lem_a
    yaw = _yaw_eased(t, T, par[S_YAW], par[S_YAW_R])
    return p, v, a, _flat_attitude(a, yaw), 0.0, _required_thrust(a)


def _pose_orbit(par, t):
    center, R, w, climb = par[S_P1:S_P1 + 3], par[S_A], par[S_W], par[S_Z0]
    T, ease, settle = par[S_DUR], par[S_EASE], par[S_SETTLE]
    base0 = jnp.array([R, 0.0, 0.0])
    e_s, de_s, dde_s = settle_envelope(t, T, settle)
    e_r, de_r, dde_r = rest_envelope(t, T, ease, settle)
    use_rest = ease > 0.0
    e = jnp.where(use_rest, e_r, e_s)
    de = jnp.where(use_rest, de_r, de_s)
    dde = jnp.where(use_rest, dde_r, dde_s)
    off = jnp.where(use_rest, base0, jnp.zeros(3))
    c, s = jnp.cos(w * t), jnp.sin(w * t)
    base = jnp.array([R * c, R * s, climb * t / T]) - off
    base_v = jnp.array([-R * w * s, R * w * c, climb / T])
    base_a = jnp.array([-R * w * w * c, -R * w * w * s, 0.0])
    p = center + e * base
    v = de * base + e * base_v
    a = dde * base + 2.0 * de * base_v + e * base_a
    yaw = _yaw_eased(t, T, par[S_YAW], par[S_YAW_R])
    return p, v, a, _flat_attitude(a, yaw), 0.0, _required_thrust(a)


def _pose_lissajous(par, t):
    A, B, w, z0 = par[S_A], par[S_B], par[S_W], par[S_Z0]
    ha, hb, hc, C, phi = par[S_HA], par[S_HB], par[S_HC], par[S_CZ], par[S_PHI]
    T, ease, settle = par[S_DUR], par[S_EASE], par[S_SETTLE]
    base0 = jnp.array([A * jnp.sin(phi), 0.0, 0.0])
    e_s, de_s, dde_s = settle_envelope(t, T, settle)
    e_r, de_r, dde_r = rest_envelope(t, T, ease, settle)
    use_rest = ease > 0.0
    e = jnp.where(use_rest, e_r, e_s)
    de = jnp.where(use_rest, de_r, de_s)
    dde = jnp.where(use_rest, dde_r, dde_s)
    off = jnp.where(use_rest, base0, jnp.zeros(3))
    args = jnp.array([ha * w * t + phi, hb * w * t, hc * w * t])
    amp = jnp.array([A, B, C])
    freqs = jnp.array([ha, hb, hc])
    base = amp * jnp.sin(args) - off
    base_v = amp * freqs * w * jnp.cos(args)
    base_a = -amp * (freqs * w) ** 2 * jnp.sin(args)
    p = jnp.array([0.0, 0.0, z0]) + e * base
    v = de * base + e * base_v
    a = dde * base + 2.0 * de * base_v + e * base_a
    yaw = _yaw_eased(t, T, par[S_YAW], par[S_YAW_R])
    return p, v, a, _flat_attitude(a, yaw), 0.0, _required_thrust(a)


def _pose_slalom(par, t):
    p0, heading, dist = par[S_P0:S_P0 + 3], par[S_HEAD], par[S_A]
    A, w, z0 = par[S_B], par[S_W], par[S_Z0]
    T, ease, settle = par[S_DUR], par[S_EASE], par[S_SETTLE]
    s, ds, dds = _quintic(t, T)
    e_s, de_s, dde_s = settle_envelope(t, T, settle)
    e_r, de_r, dde_r = rest_envelope(t, T, ease, settle)
    use_rest = ease > 0.0
    e = jnp.where(use_rest, e_r, e_s)
    de = jnp.where(use_rest, de_r, de_s)
    dde = jnp.where(use_rest, dde_r, dde_s)
    u_vec = jnp.array([jnp.cos(heading), jnp.sin(heading), 0.0])
    n_vec = jnp.array([-jnp.sin(heading), jnp.cos(heading), 0.0])
    ph = w * t
    sp, cp = jnp.sin(ph), jnp.cos(ph)
    off = jnp.array([p0[0], p0[1], z0])
    p = off + u_vec * (dist * s) + e * (n_vec * (A * sp))
    v = u_vec * (dist * ds) + de * (n_vec * (A * sp)) + e * (n_vec * (A * w * cp))
    a = (u_vec * (dist * dds) + dde * (n_vec * (A * sp))
         + 2.0 * de * (n_vec * (A * w * cp)) + e * (n_vec * (-A * w * w * sp)))
    yaw = _yaw_eased(t, T, par[S_YAW], par[S_YAW_R])
    return p, v, a, _flat_attitude(a, yaw), 0.0, _required_thrust(a)


def _pose_v8(par, t):
    p0, Az, Au = par[S_P0:S_P0 + 3], par[S_A], par[S_B]
    w, heading = par[S_W], par[S_HEAD]
    T, ease, settle = par[S_DUR], par[S_EASE], par[S_SETTLE]
    e, de, dde = rest_envelope(t, T, ease, settle)
    ph = w * t
    s1, c1 = jnp.sin(ph), jnp.cos(ph)
    s2, c2 = jnp.sin(2.0 * ph), jnp.cos(2.0 * ph)
    u_vec = jnp.array([jnp.cos(heading), jnp.sin(heading), 0.0])
    z_vec = jnp.array([0.0, 0.0, 1.0])
    base = u_vec * (Au * s2) + z_vec * (Az * s1)
    base_v = u_vec * (2.0 * Au * w * c2) + z_vec * (Az * w * c1)
    base_a = u_vec * (-4.0 * Au * w * w * s2) + z_vec * (-Az * w * w * s1)
    p = p0 + e * base
    v = de * base + e * base_v
    a = dde * base + 2.0 * de * base_v + e * base_a
    yaw = _yaw_eased(t, T, par[S_YAW], par[S_YAW_R])
    return p, v, a, _flat_attitude(a, yaw), 0.0, _required_thrust(a)


def _pose_flip(par, t):
    p0 = par[S_P0:S_P0 + 3]
    axis = par[S_AX:S_AX + 3]
    k, D, c0, u, w, rfrac, wpeak = (par[S_K], par[S_COAST], par[S_C0], par[S_UP],
                                    par[S_DOWN], par[S_RFRAC], par[S_WPEAK])
    thr_up, thr_dn, yaw = par[S_THR_UP], par[S_THR_DN], par[S_YAW]
    z0 = p0[2]
    c1 = c0 + D
    total = 2.0 * jnp.pi * k

    # vertical profile: climb / coast / arrest (piecewise constant acceleration)
    climb = t <= c0
    coast = (t > c0) & (t <= c1)
    z_climb = z0 + 0.5 * u * t * t
    v_climb = u * t
    dt_c = t - c0
    v0 = u * c0
    z_coast = z0 + 0.5 * u * c0**2 + v0 * dt_c - 0.5 * GRAVITY * dt_c**2
    v_coast = v0 - GRAVITY * dt_c
    v1 = u * c0 - GRAVITY * D
    z1 = z0 + 0.5 * u * c0**2 + u * c0 * D - 0.5 * GRAVITY * D**2
    dt_a = t - c1
    z_arrest = z1 + v1 * dt_a + 0.5 * w * dt_a**2
    v_arrest = v1 + w * dt_a

    z = jnp.where(climb, z_climb, jnp.where(coast, z_coast, z_arrest))
    vz = jnp.where(climb, v_climb, jnp.where(coast, v_coast, v_arrest))
    az = jnp.where(climb, u, jnp.where(coast, -GRAVITY, w))

    p = jnp.array([p0[0], p0[1], z])
    v = jnp.array([0.0, 0.0, vz])
    a = jnp.array([0.0, 0.0, az])
    thrust = jnp.maximum(0.0, MASS * (az + GRAVITY))

    # spin schedule: trapezoid with flat top, entirely inside the coast window
    ramp = rfrac * D
    dt = t - c0
    after = t >= c1
    before = t <= c0
    spin_ramp_up = jnp.where(
        ramp <= 1e-9, total * dt / jnp.maximum(D, 1e-9),
        0.5 * wpeak * dt * dt / jnp.maximum(ramp, 1e-9))
    spin_flat = 0.5 * wpeak * ramp + wpeak * (dt - ramp)
    spin_ramp_dn = total - 0.5 * wpeak * (c1 - t) ** 2 / jnp.maximum(ramp, 1e-9)
    spin_mid = jnp.where(dt <= ramp, spin_ramp_up, jnp.where(dt <= D - ramp, spin_flat, spin_ramp_dn))
    phi = jnp.where(before, 0.0, jnp.where(after, total, spin_mid))

    R_base = dcm_from_thrust_dir_and_yaw(jnp.array([0.0, 0.0, 1.0]), yaw)
    R = axis_angle_rotation(axis, phi) @ R_base
    return p, v, a, R, phi, thrust


def _pose_part(kind, par, wp_coef, wp_deg, t):
    """Dispatch one part (any of the 9 non-chain families)."""
    branches = [
        lambda: _pose_hover(par, t),
        lambda: _pose_takeoff(par, t),
        lambda: _pose_waypoints(par, wp_coef, wp_deg, t),
        lambda: _pose_fig8(par, t),
        lambda: _pose_orbit(par, t),
        lambda: _pose_lissajous(par, t),
        lambda: _pose_slalom(par, t),
        lambda: _pose_v8(par, t),
        lambda: _pose_flip(par, t),
        lambda: _pose_hover(par, t),          # chain index never reaches here
    ]
    return jax.lax.switch(kind, branches)


# ======================================================================================
# TrajSpec
# ======================================================================================
@struct.dataclass
class TrajSpec:
    """Fixed-shape description of a sampled trajectory (a pytree, so it can be vmapped)."""
    kind: jax.Array                    # (MAX_PARTS,) int32
    par: jax.Array                     # (MAX_PARTS, PAR_DIM)
    wp_coef: jax.Array                 # (MAX_PARTS, MAX_DEG, 3)
    wp_deg: jax.Array                  # (MAX_PARTS,)
    starts: jax.Array                  # (MAX_PARTS,)
    reloc_R: jax.Array                 # (MAX_PARTS, 3, 3)
    reloc_shift: jax.Array             # (MAX_PARTS, 3)
    n_parts: jax.Array                 # int32
    is_chain: jax.Array                # bool
    duration: jax.Array                # manoeuvre duration (excludes the tail)
    total: jax.Array                   # duration + TRAJECTORY_TAIL
    hold_p: jax.Array                  # (3,) terminal hold
    hold_R: jax.Array                  # (3,3)
    hold_spin: jax.Array
    hold_thrust: jax.Array
    # bookkeeping for diagnostics
    n_screen: jax.Array                # int32 (96 or 256)
    pre_ok: jax.Array                  # bool: the draw's own exact screen (flip) passed


@struct.dataclass
class Reference:
    """One reference sample.  Mirrors trajectories.Reference (minus `t`/`kind`)."""
    p: jax.Array
    v: jax.Array
    a: jax.Array
    R: jax.Array
    omega: jax.Array
    thrust_ff: jax.Array
    spin: jax.Array
    kind: jax.Array                    # int32 index into spec.KIND_NAMES


def _active_part(spec_: TrajSpec, t):
    """Index of the part active at t (junctions belong to the LATER part)."""
    ge = (t >= spec_.starts - 1e-12).astype(jnp.int32)
    idx = jnp.sum(ge) - 1
    return jnp.clip(idx, 0, spec_.n_parts - 1)


def _relocated_pose(spec_: TrajSpec, i, local_t):
    p, v, a, R, spin, thr = _pose_part(spec_.kind[i], spec_.par[i], spec_.wp_coef[i],
                                       spec_.wp_deg[i], local_t)
    Rz, sh = spec_.reloc_R[i], spec_.reloc_shift[i]
    return (Rz @ p + sh, Rz @ v, Rz @ a, Rz @ R, spin, thr)


def sample(spec_: TrajSpec, t):
    """
    Reference at absolute time t.  Port of ``Trajectory.sample``: the manoeuvre is frozen
    into its terminal hover past ``duration``, and omega is a CENTRAL DIFFERENCE with an
    UNCLAMPED stencil (clamping makes it one-sided exactly where the reference is most
    dynamic).
    """
    t = jnp.clip(t, 0.0, spec_.total)
    i = _active_part(spec_, t)
    local_t = t - spec_.starts[i]
    p, v, a, R, spin, thr = _relocated_pose(spec_, i, local_t)

    over = t > spec_.duration
    p = jnp.where(over, spec_.hold_p, p)
    v = jnp.where(over, jnp.zeros(3), v)
    a = jnp.where(over, jnp.zeros(3), a)
    R = jnp.where(over, spec_.hold_R, R)
    spin = jnp.where(over, spec_.hold_spin, spin)
    thr = jnp.where(over, spec_.hold_thrust, thr)

    h = 0.001
    R_m = _freeze(_relocated_pose(spec_, _active_part(spec_, t - h), (t - h) - spec_.starts[_active_part(spec_, t - h)])[3],
                  spec_, t - h)
    R_p = _freeze(_relocated_pose(spec_, _active_part(spec_, t + h), (t + h) - spec_.starts[_active_part(spec_, t + h)])[3],
                  spec_, t + h)
    omega = omega_from_dcm(R_m, R, R_p, h)

    kind = jnp.where(
        spec_.is_chain & (t >= spec_.duration),
        jnp.array(IDX_HOVER, dtype=jnp.int32),
        spec_.kind[i],
    )
    return Reference(p=p, v=v, a=a, R=R, omega=omega, thrust_ff=thr, spin=spin, kind=kind)


def _freeze(R, spec_: TrajSpec, t):
    """Apply the terminal-hover freeze to a bare R (used by the omega stencil)."""
    return jnp.where(t > spec_.duration, spec_.hold_R, R)
