"""
Dynamically feasible reference trajectories for the tracking task.

Replaces the hand-built two-phase (flip -> hover) reward with a single tracking
formulation: a reference of (p, v, a, R, omega) is produced at every step and the policy
is rewarded for following it. One policy then covers waypoint tracking, figure-8s and
acrobatic flips without a bespoke reward per manoeuvre.

WHY FEASIBILITY IS CONSTRUCTED, NOT ASSUMED
For a quadrotor the total thrust acts along the body z axis, so
    T * z_b = m * (a - g_vec),  g_vec = (0, 0, -g)
which means the reference attitude is DETERMINED by the reference acceleration:
    z_b = normalize(a + g * e_z)
The consequence that matters for acrobatics: while inverted, z_b points down, so a
positive collective thrust would push the vehicle into the floor. A 360 deg flip
therefore CANNOT hold altitude - it must coast ballistically with T = 0. Rather than
drawing an arbitrary smooth curve and hoping, each manoeuvre here constructs an
ACCELERATION profile whose coast window is exactly a_z = -g, integrates it for v and p,
and then verifies T_req = m * rho * a_hi <= T_max plus altitude bounds. Infeasible
parameter draws are rejection-sampled.

ATTITUDE PARAMETERISATION
    R_ref(t) = R_axis(phi(t)) @ R_base(t)
where R_base is the flatness-implied attitude (small, near-upright outside the coast
window) and phi is an explicit rotation angle about a chosen WORLD axis. Parameterising
the ANGLE rather than a quaternion avoids the double-cover problem for multi-rotation
manouevres: phi simply runs 0 -> 2*pi*k analytically. At phi = 0 or 2*pi*k the spin is
the identity, so it is consistent with the flatness-implied attitude at the phase
boundaries where the thrust authority returns.

Body rate is obtained by central differences of R_ref rather than by hand-derived
algebra. That is what a real trajectory generator does numerically, and it removes a
whole class of sign errors from the cross-product terms.

Standalone and dependency-free apart from numpy so it can be unit-tested and reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

GRAVITY: float = 9.81
MASS_NOMINAL: float = 0.028
MAX_THRUST_TOTAL: float = 0.60          # matches QuadcopterMuJoCo params["maxThr"]
MIN_THRUST_TOTAL: float = 0.0


# =====================================================================================
# small rotation helpers
# =====================================================================================
def _normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0]) if fallback is None else np.asarray(fallback, dtype=np.float64)
    return v / n


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula."""
    k = _normalize(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def dcm_from_thrust_dir_and_yaw(z_b: np.ndarray, yaw: float) -> np.ndarray:
    """
    Build R (body->world) with body z along z_b and heading pinned to yaw.

    Identical construction to the one already used by the PD controller in
    collect_data.py, so reference and controller agree on what 'heading' means.
    """
    z_b = _normalize(z_b)
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_b = np.cross(z_b, x_c)
    if np.linalg.norm(y_b) < 1e-6:
        x_c = np.array([np.cos(yaw + np.pi / 2.0), np.sin(yaw + np.pi / 2.0), 0.0])
        y_b = np.cross(z_b, x_c)
    y_b = _normalize(y_b)
    x_b = np.cross(y_b, z_b)
    return np.column_stack([x_b, y_b, z_b])


def omega_from_dcm(R_prev: np.ndarray, R_mid: np.ndarray, R_next: np.ndarray, h: float) -> np.ndarray:
    """
    Body rate from central differences: [omega]_x = R^T Rdot.

    Returns omega in the BODY frame (which is the frame the rate PID and the gyro use).
    """
    Rdot = (R_next - R_prev) / (2.0 * h)
    W = R_mid.T @ Rdot
    return 0.5 * np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]])


def smoothstep(t: float) -> float:
    """C1 ramp on [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _window(t: float, t0: float, t1: float, ramp: float) -> float:
    """Trapezoidal thrust-authority window, C1, in [0, 1]."""
    if ramp <= 0.0:
        return 1.0 if t0 <= t <= t1 else 0.0
    return smoothstep((t - t0) / ramp) * (1.0 - smoothstep((t - (t1 - ramp)) / ramp))


def _poly_deriv(k: int, d: int, t: float) -> float:
    """d-th derivative of t**k, evaluated at t (used to build the Hermite system)."""
    if k < d:
        return 0.0
    c = 1.0
    for j in range(d):
        c *= (k - j)
    return c * (t ** (k - d))


# =====================================================================================
# reference
# =====================================================================================
@dataclass
class Reference:
    """One sample of the reference at a point in time."""

    t: float
    p: np.ndarray            # (3) world position
    v: np.ndarray            # (3) world velocity
    a: np.ndarray            # (3) world acceleration
    R: np.ndarray            # (3,3) body->world rotation
    omega: np.ndarray        # (3) body-frame angular rate
    thrust_ff: float         # feedforward collective thrust (N)
    spin: float = 0.0        # rotation angle phi, for logging / diagnostics
    kind: str = ""

    def __post_init__(self) -> None:
        for name in ("p", "v", "a", "omega"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        self.R = np.asarray(self.R, dtype=np.float64)


# =====================================================================================
# manoeuvres
# =====================================================================================
class Maneuver:
    """
    A finite-duration reference. `pose(t)` returns (p, v, a, R_base, spin, thrust_ff).

    Subclasses must set `duration` and implement `pose`. The trajectory wrapper handles
    the body-rate differentiation and the spin composition.
    """

    kind: str = "base"

    duration: float = 1.0

    def __init__(self) -> None:
        self.duration = float(self.duration)

    def pose(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        raise NotImplementedError

    # -- shared algebra -----------------------------------------------------------
    @staticmethod
    def flat_attitude(a: np.ndarray, yaw: float) -> np.ndarray:
        """Reference attitude implied by the acceleration (differential flatness)."""
        z_b = a + np.array([0.0, 0.0, GRAVITY])
        return dcm_from_thrust_dir_and_yaw(z_b, yaw)

    @staticmethod
    def required_thrust(a: np.ndarray, mass: float = MASS_NOMINAL) -> float:
        """Collective thrust needed to realise acceleration a."""
        return float(mass * np.linalg.norm(a + np.array([0.0, 0.0, GRAVITY])))


class Hover(Maneuver):
    kind = "hover"

    def __init__(self, p0: Sequence[float], yaw: float = 0.0, duration: float = 2.0):
        super().__init__()
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.yaw = float(yaw)
        self.duration = float(duration)

    def pose(self, t):
        a = np.zeros(3)
        return self.p0.copy(), np.zeros(3), a, self.flat_attitude(a, self.yaw), 0.0, self.required_thrust(a)


class WaypointTrajectory(Maneuver):
    """
    Waypoint trajectory built from ONE global polynomial per axis.

    Each of the K waypoints imposes three conditions per axis - position, zero velocity,
    zero acceleration - so the polynomial carries 3K coefficients (degree 3K-1) and the
    system is solved exactly. This is Hermite interpolation with multiplicity 3 at each
    knot, which is non-singular by construction.

    WHY ONE GLOBAL POLYNOMIAL RATHER THAN A QUINTIC PER SEGMENT. The reference attitude
    is a function of the reference ACCELERATION, R = f(a), so the body rate depends on
    both a and its time derivative. Splicing independent per-segment quintics does keep
    a continuous - every segment starts and ends at a = 0 - but it leaves da/dt
    DISCONTINUOUS, because the jerk scale of each segment is set by that segment's own
    displacement. The result is a STEP in the reference body rate at every waypoint
    crossing (measured: 1.08 -> 1.72 rad/s across 0.2 ms for a two-segment example).
    That is a defect in the reference, not in the solver: the tracking reward would be
    asking for a rate the reference does not honour, and the finite-difference rate
    cannot reproduce R across the step - exactly what check_trajectories.py section D
    detects. One polynomial is C-infinity, so omega is smooth everywhere.

    The polynomial extrapolates smoothly but not meaningfully outside [0, duration];
    Trajectory.sample() clamps only the value it returns, so the +-h stencil used for
    omega is always evaluated on the smooth interior.
    """

    kind = "waypoints"

    def __init__(self, waypoints: Sequence[Sequence[float]], segment_time: float = 1.2, yaw: float = 0.0):
        super().__init__()
        self.wps = [np.asarray(w, dtype=np.float64) for w in waypoints]
        if len(self.wps) < 2:
            raise ValueError("WaypointTrajectory needs at least two waypoints")
        self.seg = float(segment_time)
        self.yaw = float(yaw)
        self.duration = self.seg * (len(self.wps) - 1)
        self._fit()

    def _fit(self) -> None:
        # Constraint count is kept deliberately low: position at each of the K waypoints,
        # plus zero velocity, acceleration AND jerk at the two ENDS. That is K + 6
        # conditions, so the polynomial has degree K + 5.
        #
        # Pinning the end JERK (not just v and a) is what makes the manoeuvre hand over
        # cleanly to the terminal hover. With only a(T) = 0 the polynomial can still be
        # passing through zero acceleration with nonzero slope, so the reference attitude
        # is still rotating at t = T - and the hover tail then freezes it, putting a step
        # in the reference body rate exactly at the junction (measured: 19 mrad of omega
        # inconsistency, all of it at the handover instant).
        #
        # Imposing zero velocity and acceleration at every INTERIOR waypoint too (degree
        # 3K - 1) was tried and rejected: the resulting high-degree polynomial rings
        # between knots, pushing the reference body rate to ~48 rad/s and making most
        # draws infeasible. Passing through the interior waypoints at speed is both
        # cheaper and more natural for tracking.
        K = len(self.wps)
        n = K + 6                                       # degree n - 1
        A = np.zeros((n, n))
        b = np.zeros((n, 3))
        u_knots = np.linspace(0.0, 1.0, K)
        for i, u in enumerate(u_knots):                 # position at every waypoint
            A[i] = [_poly_deriv(k, 0, u) for k in range(n)]
            b[i] = self.wps[i]
        # zero velocity / acceleration / jerk at both ends (d = 1, 2, 3)
        end_conds = [(0.0, 1), (1.0, 1), (0.0, 2), (1.0, 2), (0.0, 3), (1.0, 3)]
        for j, (u, d) in enumerate(end_conds):
            A[K + j] = [_poly_deriv(k, d, u) for k in range(n)]
        try:
            self.coef = np.linalg.solve(A, b)           # (n, 3), derivatives w.r.t. u
        except np.linalg.LinAlgError:                   # pragma: no cover - defensive
            self.coef = np.linalg.lstsq(A, b, rcond=None)[0]
        self._n = n

    def _deriv(self, t: float, d: int) -> np.ndarray:
        """d-th TIME derivative of position at time t (d = 0, 1, 2, ...)."""
        u = t / self.duration
        pw = np.array([_poly_deriv(k, d, u) for k in range(self._n)])
        return (pw @ self.coef) / (self.duration ** d)

    def pose(self, t):
        p = self._deriv(t, 0)
        v = self._deriv(t, 1)
        a = self._deriv(t, 2)
        return p, v, a, self.flat_attitude(a, self.yaw), 0.0, self.required_thrust(a)


class FigureEight(Maneuver):
    """
    Gerono lemniscate with analytic derivatives, damped to rest over a settle window.

        x = e(t) * A sin(w t),   y = e(t) * (B/2) sin(2 w t),   z = z0

    The raw lemniscate is periodic and is still moving at full speed when its duration
    expires, so it cannot on its own satisfy the requirement that every episode END in a
    hover. `e(t)` is a C2 amplitude envelope: 1 for the body of the manoeuvre, then a
    quintic ramp to zero over the last `settle` seconds.

    Because e, e' and e'' all reach zero at the end, the envelope makes p(t)->(0,0,z0)
    and, more importantly, v(t)->0 AND a(t)->0, so the reference attitude returns to
    level and the trajectory hands over to a genuine hover rather than a moving point.
    A plain amplitude cutoff would leave the attitude tilted by whatever acceleration
    the lemniscate still had.
    """

    kind = "figure8"

    def __init__(
        self,
        A: float,
        B: float,
        w: float,
        z0: float,
        cycles: float = 1.0,
        yaw: float = 0.0,
        settle: Optional[float] = None,
    ):
        super().__init__()
        self.A, self.B, self.w, self.z0, self.cycles, self.yaw = A, B, w, z0, cycles, yaw
        self.duration = float(cycles * 2.0 * np.pi / w)
        self.settle = float(min(0.35 * self.duration, 1.2) if settle is None else settle)

    def _envelope(self, t: float) -> Tuple[float, float, float]:
        """Amplitude envelope e(t) and its first two time derivatives."""
        T, L = self.duration, self.settle
        if L <= 1e-9 or t <= T - L:
            return 1.0, 0.0, 0.0
        s = float(np.clip((t - (T - L)) / L, 0.0, 1.0))
        ds = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / L
        dds = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (L * L)
        return 1.0 - (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5), -ds, -dds

    def pose(self, t):
        w, A, B = self.w, self.A, self.B
        ph = w * t
        lem = np.array([A * np.sin(ph), 0.5 * B * np.sin(2 * ph), 0.0])
        lem_v = np.array([A * w * np.cos(ph), B * w * np.cos(2 * ph), 0.0])
        lem_a = np.array([-A * w**2 * np.sin(ph), -2.0 * B * w**2 * np.sin(2 * ph), 0.0])

        e, de, dde = self._envelope(t)
        p = np.array([0.0, 0.0, self.z0]) + e * lem
        v = de * lem + e * lem_v
        a = dde * lem + 2.0 * de * lem_v + e * lem_a
        return p, v, a, self.flat_attitude(a, self.yaw), 0.0, self.required_thrust(a)


class Flip(Maneuver):
    """
    Vertical hop with a k-full-rotation spin, returning to EXACTLY its starting altitude
    with zero vertical velocity.

    The vertical acceleration is piecewise constant:

        climb  [0, c0]     a_z = u      thrust = m(u + g)
        coast  [c0, c0+D]  a_z = -g     thrust = 0        <- the rotation happens HERE
        arrest [c0+D, T]   a_z = w      thrust = m(w + g)

    Requiring v_z(T) = 0 and z(T) = z(0) gives two linear equations in (u, w), solved in
    closed form. Integrating twice is then unnecessary and the altitude excursion falls
    out of the dynamics instead of being tuned.

    WHY THE SPIN IS CONFINED TO THE COAST WINDOW
    Thrust acts along body z, so the spin rotates the thrust direction. Firing thrust
    while rotated ~90 deg would command a lateral acceleration the reference does not
    contain, and the vehicle would drift off the reference through no fault of the
    policy. Confining the rotation to the zero-thrust window makes the reference exactly
    self-consistent: thrust is only ever applied with the body upright.

    Body rate is a trapezoidal profile with flat top, peak
        omega_peak = 2*pi*k / (D * (1 - rate_frac))
    which is checked against the rate authority. That limit is why a 360 deg flip has a
    minimum coast duration: at the env's max_rate_pitch = 20 rad/s it cannot be done in
    less than 2*pi/20 = 0.31 s of rotation.
    """

    kind = "flip"

    def __init__(
        self,
        p0: Sequence[float],
        axis: Sequence[float] = (0.0, 1.0, 0.0),
        rotations: float = 1.0,
        coast: float = 0.55,
        yaw: float = 0.0,
        mass: float = MASS_NOMINAL,
        max_rate: float = 20.0,
        rate_frac: float = 0.35,
        accel_frac: float = 0.90,
    ):
        super().__init__()
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.axis = _normalize(axis)
        self.rotations = float(rotations)
        self.D = float(coast)
        self.yaw = float(yaw)
        self.mass = float(mass)
        self.max_rate = float(max_rate)
        self.rate_frac = float(np.clip(rate_frac, 0.0, 0.9))

        # SYMMETRIC, MINIMUM-EXCURSION construction.
        #
        # A ballistic coast of duration D that starts and ends at the same altitude with
        # opposite velocities requires entry/exit speed v0 = g*D/2. Requiring the climb
        # and the arrest to be mirror images (u = w) makes the altitude change cancel
        # exactly, and SATURATING the powered phases (u as large as authority allows)
        # MINIMISES the peak excursion, because the climb gain v0^2/(2u) falls as u rises.
        #
        # The earlier formulation solved for (u, w) given fixed phase durations, which let
        # the solver pick a small u and produced a 1.21 m peak - over the arena ceiling.
        self.v0 = GRAVITY * self.D / 2.0
        a_max_net = accel_frac * MAX_THRUST_TOTAL / self.mass - GRAVITY
        self.u = self.w = max(1e-3, a_max_net)
        self.c0 = self.P = self.v0 / self.u
        self.duration = self.c0 + self.D + self.P

        self.thrust_climb = self.mass * (self.u + GRAVITY)
        self.thrust_arrest = self.mass * (self.w + GRAVITY)
        self.omega_peak = 2.0 * np.pi * self.rotations / max(1e-6, self.D * (1.0 - self.rate_frac))

    # -- feasibility ---------------------------------------------------------------
    def peak_excursion(self) -> float:
        """Height gained above the start at the apex of the coast."""
        return self.v0**2 / (2.0 * self.u) + self.v0**2 / (2.0 * GRAVITY)

    def max_thrust_needed(self) -> float:
        return float(max(self.thrust_climb, self.thrust_arrest))

    def z_excursion(self) -> Tuple[float, float]:
        ts = np.linspace(0.0, self.duration, 200)
        z = np.array([self.pose(float(t))[0][2] for t in ts])
        return float(z.min()), float(z.max())

    def is_feasible(
        self,
        z_min: float = 0.30,
        z_max: float = 2.40,
        thrust_margin: float = 0.95,
        rate_margin: float = 0.95,
    ) -> bool:
        if not np.isfinite([self.u, self.w, self.thrust_climb, self.thrust_arrest]).all():
            return False
        if self.max_thrust_needed() > thrust_margin * MAX_THRUST_TOTAL:
            return False
        if min(self.thrust_climb, self.thrust_arrest) < MIN_THRUST_TOTAL:
            return False
        if self.omega_peak > rate_margin * self.max_rate:
            return False
        if self.c0 < 0.08 or self.P < 0.08:
            return False
        lo = self.p0[2]
        hi = lo + self.peak_excursion()
        return (lo >= z_min) and (hi <= z_max)

    # -- spin schedule -------------------------------------------------------------
    def _spin(self, t: float) -> float:
        """phi(t): 0 -> 2*pi*k, entirely inside the zero-thrust coast window."""
        c0, c1 = self.c0, self.c0 + self.D
        total = 2.0 * np.pi * self.rotations
        if t <= c0:
            return 0.0
        if t >= c1:
            return total
        a = self.rate_frac * self.D                       # ramp duration
        dt = t - c0
        if a <= 1e-9:
            return total * dt / self.D
        if dt <= a:
            return 0.5 * self.omega_peak * dt * dt / a
        if dt <= self.D - a:
            return 0.5 * self.omega_peak * a + self.omega_peak * (dt - a)
        rem = c1 - t
        return total - 0.5 * self.omega_peak * rem * rem / a

    # -- reference -----------------------------------------------------------------
    def _vertical(self, t: float) -> Tuple[float, float, float]:
        c0, c1 = self.c0, self.c0 + self.D
        z0 = self.p0[2]
        if t <= c0:
            return z0 + 0.5 * self.u * t * t, self.u * t, self.u
        if t <= c1:
            dt = t - c0
            v0 = self.u * c0
            return z0 + 0.5 * self.u * c0**2 + v0 * dt - 0.5 * GRAVITY * dt * dt, v0 - GRAVITY * dt, -GRAVITY
        v1 = self.u * c0 - GRAVITY * self.D
        z1 = z0 + 0.5 * self.u * self.c0**2 + self.u * self.c0 * self.D - 0.5 * GRAVITY * self.D**2
        dt = t - c1
        return z1 + v1 * dt + 0.5 * self.w * dt * dt, v1 + self.w * dt, self.w

    def pose(self, t):
        # No internal clamp - see the note in WaypointTrajectory.pose on why the
        # central-difference omega stencil requires a symmetric neighbourhood.
        z, v_z, a_z = self._vertical(t)
        p = np.array([self.p0[0], self.p0[1], z], dtype=np.float64)
        v = np.array([0.0, 0.0, v_z], dtype=np.float64)
        a = np.array([0.0, 0.0, a_z], dtype=np.float64)

        thrust = float(max(0.0, self.mass * (a_z + GRAVITY)))
        phi = self._spin(t)
        # Base attitude is upright (thrust is only ever applied upright), the spin is
        # applied about the world axis on top.
        R_base = dcm_from_thrust_dir_and_yaw(np.array([0.0, 0.0, 1.0]), self.yaw)
        R = axis_angle_rotation(self.axis, phi) @ R_base
        return p, v, a, R, phi, thrust


# =====================================================================================
# trajectory wrapper + sampling
# =====================================================================================
class Trajectory:
    """
    Wraps a Maneuver and turns it into a differentiable Reference stream.

    `omega_dt` sets the half-width of the central-difference stencil used to obtain the
    reference body rate from R(t). The stencil error scales as omega_dt^2 * |domega/dt|,
    and the quintic waypoint segments have a large jerk at each segment start (the
    acceleration ramps from 0 within a couple of samples), which used to show up as tens
    of mrad of rate error. At omega_dt = 1 ms that is down to ~2 mrad - small enough to
    ignore, and cheap (two extra pose evaluations per sample).
    """

    def __init__(self, maneuver: Maneuver, omega_dt: float = 0.001, tail: float = 0.6):
        self.maneuver = maneuver
        self.maneuver_duration = float(maneuver.duration)
        self.tail = float(max(0.0, tail))
        self.duration = self.maneuver_duration + self.tail
        self.h = float(omega_dt)

        # Terminal hold state - the "end in a hover" guarantee.
        #
        # Every manoeuvre finishes at rest and level: hover trivially, waypoints because
        # the fitted polynomial ends at zero v and a, figure-8 via its settle envelope,
        # and the flip because v_z(T) = 0 and its spin returns to the identity. The one
        # quantity that is NOT yet at its hover value when a flip ends is the vertical
        # acceleration: the arrest phase holds a_z = w right up to t = T, so the required
        # thrust at the terminal instant is m(w + g), not m*g. The tail snaps that to
        # hover trim so the episode genuinely concludes in a hold rather than at the end
        # of an arrest.
        p, _v, _a, R, spin, _thr = maneuver.pose(self.maneuver_duration)
        self._hold_p = np.asarray(p, dtype=np.float64).copy()
        self._hold_R = np.asarray(R, dtype=np.float64).copy()
        self._hold_spin = float(spin)
        self._hold_thrust = float(getattr(maneuver, "mass", MASS_NOMINAL) * GRAVITY)

    def _pose(self, t: float):
        """Manoeuvre pose, frozen into the terminal hover once the manoeuvre is over."""
        if t <= self.maneuver_duration:
            return self.maneuver.pose(t)
        return (self._hold_p, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64),
                self._hold_R, self._hold_spin, self._hold_thrust)
        self.h = float(omega_dt)

    def sample(self, t: float) -> Reference:
        t = float(np.clip(t, 0.0, self.duration))
        p, v, a, R, spin, thrust = self._pose(t)

        # Central difference for body rate. t_m / t_p are deliberately NOT clamped to the
        # horizon: clamping would make the stencil one-sided at the ends, dropping omega
        # to first-order accuracy exactly where the reference is most dynamic.
        h = self.h
        t_m, t_p = t - h, t + h
        R_m = self._pose(t_m)[3]
        R_p = self._pose(t_p)[3]
        omega = omega_from_dcm(R_m, R, R_p, h)

        return Reference(
            t=t, p=p, v=v, a=a, R=R, omega=omega,
            thrust_ff=thrust, spin=spin, kind=self.maneuver.kind,
        )

    def initial_state(self) -> Reference:
        return self.sample(0.0)


@dataclass
class TrajectoryConfig:
    """Sampling configuration for the manoeuvre mixture."""

    bounds_xy: float = 1.0
    z_range: Tuple[float, float] = (1.0, 1.8)
    weights: dict = field(default_factory=lambda: {
        "hover": 0.15, "waypoints": 0.35, "figure8": 0.25, "flip": 0.25,
    })
    max_resample: int = 40
    # Action-scaling limits of QuadFlipEnv, mirrored here so the sampler never emits a
    # reference the policy is structurally unable to follow. These are POLICY ACTION
    # SCALES, not physical limits; the physical ceilings are much higher (the rate loop
    # has ~700 rad/s^2 available, so 20 rad/s is reached in ~0.03 s).
    #
    # Roll and pitch are deliberately SYMMETRIC. The original 6 / 20 split was an artefact
    # of the flip task only ever rotating about pitch, and it silently made 360 deg ROLL
    # flips unsatisfiable - at 6 rad/s the rotation alone needs >= 1.05 s of ballistic
    # coast, which no 1.2 m hover can survive, so the sampler rejected every one.
    rate_limits: dict = field(default_factory=lambda: {"roll": 20.0, "pitch": 20.0})


class TrajectorySampler:
    """
    Draws a feasible manoeuvre. Rejection-samples until the feasibility check passes,
    falling back to a hover if it cannot (so the env never sees an impossible reference).
    """

    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.cfg = config or TrajectoryConfig()

    def _make_flip(self, rng, p0, yaw, mass) -> Optional[Flip]:
        use_pitch = bool(rng.random() < 0.75)
        axis_name = "pitch" if use_pitch else "roll"
        axis = np.array([0.0, 1.0, 0.0]) if use_pitch else np.array([1.0, 0.0, 0.0])
        limit = float(self.cfg.rate_limits[axis_name])

        k = float(rng.choice([1.0, 1.0, 1.0, 2.0]))
        rate_frac = float(rng.uniform(0.30, 0.45))
        # Derive the coast from the rate authority instead of guessing: the trapezoidal
        # profile needs  coast = 2*pi*k / (omega_target * (1 - rate_frac)).
        omega_target = float(rng.uniform(0.55, 0.80)) * limit
        coast = 2.0 * np.pi * k / max(1e-6, omega_target * (1.0 - rate_frac))

        fl = Flip(
            p0, axis=axis, rotations=k, coast=coast,
            yaw=yaw, mass=mass, max_rate=limit, rate_frac=rate_frac,
        )
        return fl if fl.is_feasible() else None

    def sample(self, rng: np.random.Generator, mass: float = MASS_NOMINAL,
               kind: Optional[str] = None) -> Trajectory:
        """
        Draw a feasible trajectory.

        `kind` pins the manoeuvre to one of the config weights ("hover", "waypoints",
        "figure8", "flip") for the high-level command interface; None samples from the
        mixture. In both cases the result is guaranteed feasible, and every manoeuvre is
        constructed to END IN A HOVER: hover trivially, waypoints and flips by their zero
        terminal velocity, figure-8 via its settle envelope.
        """
        kinds = list(self.cfg.weights.keys())
        probs = np.array([self.cfg.weights[k] for k in kinds], dtype=np.float64)
        probs /= probs.sum()
        if kind is not None and kind not in kinds:
            raise ValueError(f"unknown manoeuvre {kind!r}; expected one of {kinds}")

        for _ in range(self.cfg.max_resample):
            chosen = kind if kind is not None else str(rng.choice(kinds, p=probs))
            z0 = float(rng.uniform(*self.cfg.z_range))
            p0 = np.array([
                rng.uniform(-self.cfg.bounds_xy, self.cfg.bounds_xy),
                rng.uniform(-self.cfg.bounds_xy, self.cfg.bounds_xy),
                z0,
            ])
            yaw = float(rng.uniform(-np.pi, np.pi))

            if chosen == "hover":
                traj = Trajectory(Hover(p0, yaw, duration=float(rng.uniform(1.5, 3.0))))
            elif chosen == "waypoints":
                n = int(rng.integers(3, 5))
                wps = [p0] + [
                    np.array([
                        rng.uniform(-self.cfg.bounds_xy, self.cfg.bounds_xy),
                        rng.uniform(-self.cfg.bounds_xy, self.cfg.bounds_xy),
                        float(rng.uniform(*self.cfg.z_range)),
                    ])
                    for _ in range(n)
                ]
                traj = Trajectory(WaypointTrajectory(wps, segment_time=float(rng.uniform(0.9, 1.6)), yaw=yaw))
            elif chosen == "figure8":
                w = float(rng.uniform(0.8, 1.8))
                traj = Trajectory(FigureEight(
                    A=float(rng.uniform(0.2, 0.4)),
                    B=float(rng.uniform(0.2, 0.4)),
                    w=w, z0=z0, cycles=1.0, yaw=0.0,
                ))
            else:
                fl = self._make_flip(rng, p0, yaw, mass)
                if fl is None:
                    continue
                traj = Trajectory(fl)

            # Feasibility screen for non-flip manoeuvres. Flips are already screened
            # exactly by Flip.is_feasible().
            #
            # RATE IS CHECKED AS WELL AS THRUST. Bounding only the thrust was a real gap:
            # thrust depends on |a + g*e_z|, so a reference whose acceleration MAGNITUDE
            # stays near g while its DIRECTION whips around passes a thrust-only screen
            # completely, and a 47.9 rad/s reference did exactly that. The rate is
            # therefore the binding constraint for aggressive geometry and has to be part
            # of the screen.
            if chosen != "flip":
                ts = np.linspace(0.0, traj.duration, 96)
                refs = [traj.sample(float(t)) for t in ts]
                req = np.array([r.thrust_ff for r in refs])
                if req.max() > 0.95 * MAX_THRUST_TOTAL or req.min() < MIN_THRUST_TOTAL:
                    continue
                rate = max(float(np.linalg.norm(r.omega)) for r in refs)
                if rate > 0.95 * max(self.cfg.rate_limits.values()):
                    continue
            return traj

        # Fallback: a trivially feasible hold at the nominal altitude.
        return Trajectory(Hover([0.0, 0.0, 1.2], 0.0, duration=2.0))
