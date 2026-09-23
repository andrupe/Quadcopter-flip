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

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

GRAVITY: float = 9.81
MASS_NOMINAL: float = 0.033
MAX_THRUST_TOTAL: float = 0.60          # matches QuadcopterMuJoCo params["maxThr"]
MIN_THRUST_TOTAL: float = 0.0

# Seconds of terminal hover appended to every manoeuvre. Shared so the sampler can work
# out the highest manoeuvre frequency that still fits inside one episode (see
# TrajectoryConfig.episode_seconds) instead of duplicating the number.
TRAJECTORY_TAIL: float = 0.6


# =====================================================================================
# small rotation helpers
# =====================================================================================
def _normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit vector.

    Scalars rather than np.linalg.norm + broadcast division. At 3 elements numpy's
    dispatch overhead is two orders of magnitude larger than the arithmetic, and this
    is called several times per reference sample. Same formula, same float64
    operations, same result to the last ULP.
    """
    v = np.asarray(v, dtype=np.float64)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0]) if fallback is None else np.asarray(fallback, dtype=np.float64)
    return np.array([x / n, y / n, z / n], dtype=np.float64)


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula. skaei kathisterimeno telos pantwn brm"""
    k = _normalize(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def dcm_from_thrust_dir_and_yaw(z_b: np.ndarray, yaw: float) -> np.ndarray:
    """
    Build R (body->world) with body z along z_b and heading pinned to yaw.

    Identical construction to the one already used by the PD controller in
    collect_data.py, so reference and controller agree on what 'heading' means.

    WHY THIS IS WRITTEN WITH SCALARS. Measured on this machine, the numpy version cost
    ~41 us per call (np.cross dispatches through moveaxis/normalize_axis_tuple even for
    a 3-element vector) against ~3 us for the scalar twin, and it is called three times
    per reference sample - roughly a fifth of a whole environment step. The arithmetic
    is the same: y_b = z_b x [cos yaw, sin yaw, 0], x_b = y_b x z_b, both normalised.
    """
    z_b = _normalize(z_b)
    zx, zy, zz = float(z_b[0]), float(z_b[1]), float(z_b[2])
    yaw = float(yaw)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # y_b = z_b x x_c,  x_c = [cos(yaw), sin(yaw), 0]
    yx = zy * 0.0 - zz * sy
    yy = zz * cy - zx * 0.0
    yz = zx * sy - zy * cy
    if math.sqrt(yx * yx + yy * yy + yz * yz) < 1e-6:
        # z_b parallel to the heading axis: turn the heading reference a quarter turn.
        cy, sy = math.cos(yaw + 0.5 * math.pi), math.sin(yaw + 0.5 * math.pi)
        yx = zy * 0.0 - zz * sy
        yy = zz * cy - zx * 0.0
        yz = zx * sy - zy * cy
    ny = math.sqrt(yx * yx + yy * yy + yz * yz)
    yx, yy, yz = yx / ny, yy / ny, yz / ny
    # x_b = y_b x z_b
    xx = yy * zz - yz * zy
    xy = yz * zx - yx * zz
    xz = yx * zy - yy * zx
    return np.array([[xx, yx, zx], [xy, yy, zy], [xz, yz, zz]], dtype=np.float64)


def omega_from_dcm(R_prev: np.ndarray, R_mid: np.ndarray, R_next: np.ndarray, h: float) -> np.ndarray:
    """
    Body rate from central differences: [omega]_x = R^T Rdot.
    as poume oti bgazei noima nai nai
    Returns omega in the BODY frame (which is the frame the rate PID and the gyro use).
    """
    Rdot = (R_next - R_prev) / (2.0 * h)
    W = R_mid.T @ Rdot
    return 0.5 * np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]])


def smoothstep(t: float) -> float:
    """C1 ramp on [0, 1]. overengineered ala ok mpok"""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _window(t: float, t0: float, t1: float, ramp: float) -> float:
    """Trapezoidal thrust-authority window, C1, in [0, 1]."""
    if ramp <= 0.0:
        return 1.0 if t0 <= t <= t1 else 0.0
    return smoothstep((t - t0) / ramp) * (1.0 - smoothstep((t - (t1 - ramp)) / ramp))


def _yaw_of(R: np.ndarray) -> float:
    """Heading of a body->world rotation: the rotation about world z of its first column."""
    R = np.asarray(R, dtype=np.float64)
    return float(math.atan2(R[1, 0], R[0, 0]))


def _poly_deriv(k: int, d: int, t: float) -> float:
    """d-th derivative of t**k, evaluated at t (used to build the Hermite system)."""
    if k < d:
        return 0.0
    c = 1.0
    for j in range(d):
        c *= (k - j)
    return c * (t ** (k - d))


def _settle_envelope(t: float, T: float, L: float) -> Tuple[float, float, float]:
    """
    C2 amplitude envelope: 1 for the body of a manoeuvre, quintic ramp to 0 over the last
    `L` seconds. Returns (e, de/dt, d2e/dt2).

    Because e, e' and e'' all reach zero at T, multiplying a position profile by this makes
    p -> final, v -> 0 AND a -> 0. That last one matters: v -> 0 alone would leave the
    reference attitude tilted by whatever acceleration the profile still had, so the
    handover to the terminal hover would come with an attitude step.
    """
    if L <= 1e-9 or t <= T - L:
        return 1.0, 0.0, 0.0
    s = float(np.clip((t - (T - L)) / L, 0.0, 1.0))
    ds = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / L
    dds = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (L * L)
    return 1.0 - (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5), -ds, -dds


def _rest_envelope(t: float, T: float, L0: float, L1: float) -> Tuple[float, float, float]:
    """
    C2 REST envelope: a quintic ramp IN over the first `L0` seconds, 1 through the middle,
    a quintic ramp OUT over the last `L1` seconds - and FLAT outside [0, T].

    This is the two-sided twin of `_settle_envelope`, and it is what turns an oscillating
    shape into a COMMAND. Applied to a profile as
        p(t) = start + e(t) * (base(t) - base(0)),
    the zero value and zero first two derivatives of e at BOTH ends make v(0) = a(0) = 0
    and v(T) = a(T) = 0 for any base: the manoeuvre genuinely begins and ends parked, in
    a level attitude, which is what a `Chain` junction requires and what lets an
    acrobatic shape be spawned from rest without a velocity or attitude step.

    The flat extension outside [0, T] is not cosmetic. `Trajectory.sample` evaluates its
    +-h stencil across the horizon, so the profile has to stay smooth for t slightly
    below 0 and slightly above T; the quintic's first two derivatives are exactly zero at
    s = 0 and s = 1, so freezing the ramp value there is C2-continuous with it.
    """
    if L0 > 1e-9 and t < L0:
        s = float(np.clip(t / L0, 0.0, 1.0))
        ds = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / L0
        dds = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (L0 * L0)
        return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5, ds, dds
    if L1 > 1e-9 and t > T - L1:
        s = float(np.clip((t - (T - L1)) / L1, 0.0, 1.0))
        ds = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / L1
        dds = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (L1 * L1)
        return 1.0 - (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5), -ds, -dds
    return 1.0, 0.0, 0.0


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
        self.yaw: float = 0.0
        # Optional slow YAW manoeuvre. Without this the heading is pinned for the whole
        # episode, so nothing in the mixture ever asks the policy to track a yaw reference
        # - the yaw channel is trained purely as a disturbance-rejection problem.
        self.yaw_rate: float = 0.0

    def _yaw(self, t: float) -> float:
        """
        Heading at time t, eased so it starts AND ends at zero yaw rate.

        The easing is not cosmetic. omega is a central difference of R, and the terminal
        hover freezes R at its final value - so a heading still rotating at t = T would put
        a STEP in the reference body rate at the handover. That is the same defect the
        waypoint trajectory had at its knots. The quintic has zero 1st and 2nd derivatives
        at both ends, so the reference rate stays continuous.

        The clip is safe here (unlike elsewhere): the eased profile is C2-flat at u = 0 and
        u = 1, so freezing it outside [0, T] introduces no discontinuity.
        """
        if abs(self.yaw_rate) < 1e-9:
            return self.yaw
        u = float(np.clip(t / max(1e-9, self.duration), 0.0, 1.0))
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        return self.yaw + self.yaw_rate * self.duration * s

    def pose(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        raise NotImplementedError

    def kind_at(self, t: float) -> str:
        """
        Reference kind at time t - the key the reward's per-manoeuvre tolerances are
        looked up with, and the label telemetry carries.

        Constant for every single manoeuvre. `Chain` overrides it so that each of its
        segments reports its OWN kind: a 360 deg flip in the middle of a chain is graded
        with the flip tolerances and a hover pause with the hover tolerances, instead of
        the whole chain being graded on one compromise scale that would be sloppy in the
        pauses and over-strict in the acrobatics.
        """
        return self.kind

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

    def __init__(self, p0: Sequence[float], yaw: float = 0.0, duration: float = 2.0,
                 yaw_rate: float = 0.0):
        super().__init__()
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.yaw = float(yaw)
        self.yaw_rate = float(yaw_rate)
        self.duration = float(duration)

    def pose(self, t):
        a = np.zeros(3)
        return (self.p0.copy(), np.zeros(3), a,
                self.flat_attitude(a, self._yaw(t)), 0.0, self.required_thrust(a))


class Takeoff(Maneuver):
    """
    Smooth, dynamically feasible ascent trajectory from ground (or arbitrary altitude
    between 0 and 1.2m) to a stable hover station.

    Uses a degree-7 minimum-snap polynomial with zero velocity, acceleration, and jerk
    at both boundaries (u=0 and u=1), ensuring C^3-smooth handover from resting/ground
    trim to the target hover station.
    """

    kind = "takeoff"

    def __init__(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        climb_time: float = 2.0,
        hold_time: float = 3.0,
        yaw: float = 0.0,
        yaw_rate: float = 0.0,
    ):
        super().__init__()
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.climb_time = float(max(0.5, climb_time))
        self.hold_time = float(max(0.5, hold_time))
        self.duration = self.climb_time + self.hold_time
        self.yaw = float(yaw)
        self.yaw_rate = float(yaw_rate)

    def pose(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        tc = self.climb_time
        delta = self.p1 - self.p0
        if t <= 0.0:
            p = self.p0.copy()
            v = np.zeros(3, dtype=np.float64)
            a = np.zeros(3, dtype=np.float64)
        elif t < tc:
            u = float(t / tc)
            u2 = u * u
            u3 = u2 * u
            u4 = u3 * u
            u5 = u4 * u
            u6 = u5 * u
            u7 = u6 * u
            s = 35.0 * u4 - 84.0 * u5 + 70.0 * u6 - 20.0 * u7
            ds = (140.0 * u3 - 420.0 * u4 + 420.0 * u5 - 140.0 * u6) / tc
            d2s = (420.0 * u2 - 1680.0 * u3 + 2100.0 * u4 - 840.0 * u5) / (tc * tc)
            p = self.p0 + s * delta
            v = ds * delta
            a = d2s * delta
        else:
            p = self.p1.copy()
            v = np.zeros(3, dtype=np.float64)
            a = np.zeros(3, dtype=np.float64)

        R = self.flat_attitude(a, self._yaw(t))
        thrust = self.required_thrust(a)
        return p, v, a, R, 0.0, thrust


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
    kala ntaks
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
        ease: Optional[float] = None,
    ):
        super().__init__()
        self.A, self.B, self.w, self.z0, self.cycles, self.yaw = A, B, w, z0, cycles, yaw
        self.duration = float(cycles * 2.0 * np.pi / w)
        self.settle = float(min(0.35 * self.duration, 1.2) if settle is None else settle)
        # ease=0 (the default) leaves the mid-motion start the periodic families have
        # always had; a positive ease ramps the amplitude up from zero over that many
        # seconds, which turns the lemniscate into a rest-to-rest CLOSED LOOP (it starts
        # and ends at the offset point) that a Chain can link. See _rest_envelope.
        self.ease = float(0.0 if ease is None else max(0.0, ease))
        if self.ease + self.settle > self.duration + 1e-9:
            raise ValueError("FigureEight: ease + settle must not exceed the duration")

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

        # lem(0) = 0, so the offset subtraction the rest envelope needs is a no-op here:
        # the crossing point of the lemniscate is the manoeuvre's start/end point.
        if self.ease > 0.0:
            e, de, dde = _rest_envelope(t, self.duration, self.ease, self.settle)
        else:
            e, de, dde = self._envelope(t)
        p = np.array([0.0, 0.0, self.z0]) + e * lem
        v = de * lem + e * lem_v
        a = dde * lem + 2.0 * de * lem_v + e * lem_a
        return p, v, a, self.flat_attitude(a, self._yaw(t)), 0.0, self.required_thrust(a)


class Orbit(Maneuver):
    """
    Circular orbit, optionally climbing (a helix), with analytic derivatives.

        p(t) = c + e(t) * (R cos(w t), R sin(w t), climb * t / T)

    WHY THIS IS NOT JUST "ANOTHER PATH". A circle requires a constant inward centripetal
    acceleration, so the flatness relation banks the reference by atan(R w^2 / g) and HOLDS
    that bank for the whole manoeuvre. Every other trajectory here is level or only
    transiently tilted, so a sustained banked attitude is a genuinely new regime - and it is
    the one that loads the ROLL channel continuously rather than in bursts.

    The settle envelope is essential rather than cosmetic: a circle is still travelling at
    R*w when its period expires, so without damping it would hand an attitude and velocity
    step to the terminal hover.
    """

    kind = "orbit"

    def __init__(self, center: Sequence[float], radius: float, w: float, climb: float = 0.0,
                 turns: float = 1.0, yaw: float = 0.0, yaw_rate: float = 0.0,
                 settle: Optional[float] = None, duration: Optional[float] = None,
                 ease: Optional[float] = None):
        super().__init__()
        self.center = np.asarray(center, dtype=np.float64)
        self.R = float(radius)
        self.w = float(w)
        self.climb = float(climb)
        self.yaw = float(yaw)
        self.yaw_rate = float(yaw_rate)
        # `duration` lets the caller fix the length and derive the angular rate from it.
        # Sampling w directly instead forces a choice between very slow circles (a 2-turn
        # orbit at w=0.7 runs 18 s) and counting discrete turns.
        if duration is not None:
            self.duration = float(duration)
            self.w = float(turns * 2.0 * np.pi / max(1e-6, self.duration))
        else:
            self.duration = float(turns * 2.0 * np.pi / max(1e-6, abs(w)))
        self.settle = float(min(0.35 * self.duration, 1.2) if settle is None else settle)
        # With ease > 0 the circle is also ramped in from the CENTRE, which makes the
        # manoeuvre start and end at the same point (a rest-to-rest "lasso"): without it
        # the orbit begins mid-circle, already banked and already moving.
        self.ease = float(0.0 if ease is None else max(0.0, ease))
        if self.ease + self.settle > self.duration + 1e-9:
            raise ValueError("Orbit: ease + settle must not exceed the duration")
        self._base0 = np.array([self.R, 0.0, 0.0], dtype=np.float64)

    def pose(self, t):
        w, R, T = self.w, self.R, max(1e-9, self.duration)
        if self.ease > 0.0:
            e, de, dde = _rest_envelope(t, self.duration, self.ease, self.settle)
            off = self._base0
        else:
            e, de, dde = _settle_envelope(t, self.duration, self.settle)
            off = np.zeros(3)
        c, s = np.cos(w * t), np.sin(w * t)
        base = np.array([R * c, R * s, self.climb * t / T]) - off
        base_v = np.array([-R * w * s, R * w * c, self.climb / T])
        base_a = np.array([-R * w * w * c, -R * w * w * s, 0.0])
        p = self.center + e * base
        v = de * base + e * base_v
        a = dde * base + 2.0 * de * base_v + e * base_a
        return p, v, a, self.flat_attitude(a, self._yaw(t)), 0.0, self.required_thrust(a)


class Lissajous(Maneuver):
    """
    General Lissajous figure, damped to rest by the shared C2 settle envelope.

        x = e(t) A sin(a w t + phi),   y = e(t) B sin(b w t),   z = z0 + e(t) C sin(c w t)

    `FigureEight` is the (a=1, b=2, phi=0, C=0) special case. Generalising it buys genuinely
    different geometry - 1:3 loops, tilted and drifting patterns, a gentle altitude bob -
    from the same code path, so the mixture stops containing one memorisable fixed shape.
    """

    kind = "lissajous"

    def __init__(self, A: float, B: float, w: float, z0: float, a: float = 1.0, b: float = 2.0,
                 phi: float = 0.0, C: float = 0.0, c: float = 2.0, cycles: float = 1.0,
                 yaw: float = 0.0, yaw_rate: float = 0.0, settle: Optional[float] = None,
                 ease: Optional[float] = None):
        super().__init__()
        self.A, self.B, self.w, self.z0 = float(A), float(B), float(w), float(z0)
        self.a, self.b, self.phi, self.C, self.c = float(a), float(b), float(phi), float(C), float(c)
        self.yaw = float(yaw)
        self.yaw_rate = float(yaw_rate)
        self.duration = float(cycles * 2.0 * np.pi / max(1e-6, abs(w)))
        self.settle = float(min(0.35 * self.duration, 1.2) if settle is None else settle)
        # ease > 0 makes the pattern rest-to-rest (and closed, since the whole profile is
        # scaled about its t = 0 value) instead of the mid-motion start it shares with
        # the other periodic families.
        self.ease = float(0.0 if ease is None else max(0.0, ease))
        if self.ease + self.settle > self.duration + 1e-9:
            raise ValueError("Lissajous: ease + settle must not exceed the duration")
        self._base0 = np.array([self.A * math.sin(self.phi), 0.0, 0.0], dtype=np.float64)

    def pose(self, t):
        w = self.w
        if self.ease > 0.0:
            e, de, dde = _rest_envelope(t, self.duration, self.ease, self.settle)
            off = self._base0
        else:
            e, de, dde = _settle_envelope(t, self.duration, self.settle)
            off = np.zeros(3)
        args = np.array([self.a * w * t + self.phi, self.b * w * t, self.c * w * t])
        amp = np.array([self.A, self.B, self.C if self.C != 0.0 else 0.0])
        base = amp * np.sin(args) - off
        base_v = amp * self.__freqs() * w * np.cos(args)
        base_a = -amp * (self.__freqs() * w) ** 2 * np.sin(args)
        off_z = np.array([0.0, 0.0, self.z0])
        p = off_z + e * base
        v = de * base + e * base_v
        a = dde * base + 2.0 * de * base_v + e * base_a
        return p, v, a, self.flat_attitude(a, self._yaw(t)), 0.0, self.required_thrust(a)

    def __freqs(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c])


class Slalom(Maneuver):
    """
    Weaving traverse: a straight run with a sinusoidal lateral weave, along a heading that
    is deliberately NOT the direction of travel.

        travel = u * dist * s(t/T)          s = quintic, zero velocity and accel at both ends
        weave  = e(t) * n * A sin(w t)      e = settle envelope, C2 to zero

    Exercises continuous ROLL REVERSAL: the lateral acceleration flips sign every half
    period, so the reference bank rocks back and forth for the whole run. A figure-8 also
    reverses, but on a closed path at constant speed; here the heading and the direction of
    travel are independent, so the two channels must be tracked separately.

    The travel term carries its OWN zero-velocity profile rather than riding the settle
    envelope. Multiplying a traverse by a global envelope that decays to zero would drag it
    back to where it started - a weave in place, not a traverse.
    """

    kind = "slalom"

    def __init__(self, start: Sequence[float], heading: float, dist: float, A: float, w: float,
                 z0: float, yaw: float = 0.0, yaw_rate: float = 0.0,
                 settle: Optional[float] = None, cycles: float = 2.5,
                 duration: Optional[float] = None, ease: Optional[float] = None):
        super().__init__()
        self.p0 = np.asarray(start, dtype=np.float64)
        self.heading = float(heading)
        self.dist = float(dist)
        self.A = float(A)
        self.w = float(w)
        self.z0 = float(z0)
        self.yaw = float(yaw)
        self.yaw_rate = float(yaw_rate)
        # Duration-driven, like Orbit. Deriving w from an explicit duration keeps the weave
        # frequency tied to the episode length; clamping w instead collapsed every sample
        # onto the same duration and silently removed the variation.
        if duration is not None:
            self.duration = float(duration)
            self.w = float(cycles * 2.0 * np.pi / max(1e-6, self.duration))
        else:
            self.duration = float(np.clip(cycles * 2.0 * np.pi / max(1e-6, abs(w)), 2.5, 4.5))
        self.settle = float(min(0.35 * self.duration, 1.2) if settle is None else settle)
        # Only the WEAVE needs the rest envelope: the traverse already carries its own
        # zero-velocity quintic. Easing the traverse would drag the vehicle back to its
        # starting point instead of ending at the end of the run.
        self.ease = float(0.0 if ease is None else max(0.0, ease))
        if self.ease + self.settle > self.duration + 1e-9:
            raise ValueError("Slalom: ease + settle must not exceed the duration")

    def pose(self, t):
        T = max(1e-9, self.duration)
        u = float(t / T)
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / T
        dds = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (T * T)
        if self.ease > 0.0:
            e, de, dde = _rest_envelope(t, self.duration, self.ease, self.settle)
        else:
            e, de, dde = _settle_envelope(t, self.duration, self.settle)

        u_vec = np.array([np.cos(self.heading), np.sin(self.heading), 0.0])
        n_vec = np.array([-np.sin(self.heading), np.cos(self.heading), 0.0])
        ph = self.w * t

        off = np.array([self.p0[0], self.p0[1], self.z0])
        p = off + u_vec * (self.dist * s) + e * (n_vec * (self.A * np.sin(ph)))
        v = u_vec * (self.dist * ds) + de * (n_vec * (self.A * np.sin(ph))) \
            + e * (n_vec * (self.A * self.w * np.cos(ph)))
        a = u_vec * (self.dist * dds) + dde * (n_vec * (self.A * np.sin(ph))) \
            + 2.0 * de * (n_vec * (self.A * self.w * np.cos(ph))) \
            + e * (n_vec * (-self.A * self.w * self.w * np.sin(ph)))
        return p, v, a, self.flat_attitude(a, self._yaw(t)), 0.0, self.required_thrust(a)


class VerticalEight(Maneuver):
    """
    Fast figure-eight in a VERTICAL plane: the two leaves of the 8 are stacked in
    ALTITUDE, so the quad climbs over the top leaf, dives back through the crossing point
    and under the bottom leaf - the numeral 8, flying.

        u(t) = p0 + e(t) * Au * sin(2 phi)          horizontal, along `heading`
        z(t) = p0 + e(t) * Az * sin(phi)            vertical, phi = w t
        (the third axis is untouched: the whole path lives in the plane spanned by
         u_dir and e_z)

    WHY THE PATH IS 'AN 8' AND NOT A WEAVE. The horizontal weave runs at the SECOND
    harmonic of the vertical motion, which is exactly what closes the figure: at
    phi = 0, pi, 2pi BOTH coordinates return to the start, so the path crosses itself
    at its start/end point twice per cycle and the two leaves (z > start, z < start) are
    the two lobes. The 8 is a genuine lemniscate (Gerono), not a sine pair drawn twice.

    WHY A MODEST-LOOKING PATH IS AN ACROBATIC. Differentiating twice gives
        a_z = -Az w^2 sin(phi),        a_u = -4 Au w^2 sin(2 phi)
    The factor 4 on the weave is the whole story: at w ~ 3 rad/s a 0.15 m weave already
    demands ~5 m/s^2 of sideways acceleration, and it does so at the same time as the
    vertical acceleration swings towards -Az w^2. The flatness attitude
    z_b = normalize(a + g e_z) therefore pitches hard and snaps back twice per cycle.
    Measured over the sampled amplitudes: attitude excursions up to ~60-70 deg, peak
    reference body rate ~10-16 rad/s (a 360 deg flip's coast runs at 800-1000 deg/s =
    14-17 rad/s, so this is the same class of motion), and the collective thrust
    modulates from a light-load dip at the crossing towards the top of the band. It is
    the one manoeuvre in the mixture that reaches the attitude limits WITHOUT inverting.

    WHY IT DOES NOT GO OVER THE TOP. At the sampled amplitudes Az w^2 stays below g, so
    a_z + g > 0 for every t and the reference attitude never crosses the horizon. That is
    deliberate. As a_z approaches -g the flatness direction z_b = normalize(a + g e_z)
    approaches the origin of its argument, where it is DEGENERATE: the direction turns
    arbitrarily fast for an arbitrarily small change in a, so the reference body rate
    diverges. That is precisely the defect `Flip` avoids by confining its rotation to a
    zero-thrust coast. Draws that would push a_z below -g are rejected by the sampler's
    rate screen instead of being shipped as an unflyable reference.

    REST ENVELOPE. The base profile is multiplied by a C2 rest envelope (see
    `_rest_envelope`): a quintic ramp in over `ease` seconds, out over `settle` seconds,
    flat outside [0, T]. e, e' and e'' vanish at both ends, so the manoeuvre starts and
    ends in a genuine hover (v = a = 0, level attitude, thrust = m g) at exactly the point
    it began - which is what makes it chainable and what lets an episode spawn the
    vehicle directly on it. Passing `ease=0` restores the mid-motion start the other
    periodic families use.

    WHERE THE PEAK RATE COMES FROM. Most of it is in the entry/exit ramps: the weave
    velocity is multiplied by the envelope derivative while the amplitude is still small,
    so |a + g| is close to g and the attitude direction turns quickly. That is a real
    property of a fast rest-to-rest 8 (it has to get to full weave speed somehow), not a
    defect, and it is bounded by the same thrust/rate/volume screens as every family.
    """

    kind = "v8"

    def __init__(self, p0: Sequence[float], z_amp: float = 0.35, u_amp: float = 0.15,
                 w: float = 2.8, cycles: float = 1.0, heading: float = 0.0,
                 yaw: float = 0.0, yaw_rate: float = 0.0,
                 ease: Optional[float] = None, settle: Optional[float] = None):
        super().__init__()
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.Az = float(z_amp)
        self.Au = float(u_amp)
        self.w = float(max(1e-6, w))
        self.cycles = float(cycles)
        self.heading = float(heading)
        self.yaw = float(yaw)
        self.yaw_rate = float(yaw_rate)
        self.duration = float(self.cycles * 2.0 * np.pi / self.w)
        default = min(0.35 * self.duration, 0.9)
        self.ease = float(default if ease is None else max(0.0, ease))
        self.settle = float(default if settle is None else max(0.0, settle))
        if self.ease + self.settle > self.duration + 1e-9:
            raise ValueError("VerticalEight: ease + settle must not exceed the duration")

    def pose(self, t):
        e, de, dde = _rest_envelope(t, self.duration, self.ease, self.settle)
        w = self.w
        ph = w * t
        s1, c1 = math.sin(ph), math.cos(ph)
        s2, c2 = math.sin(2.0 * ph), math.cos(2.0 * ph)
        u_vec = np.array([math.cos(self.heading), math.sin(self.heading), 0.0])
        z_vec = np.array([0.0, 0.0, 1.0])
        # base(0) = 0, so the crossing point IS the manoeuvre's start/end point.
        base = u_vec * (self.Au * s2) + z_vec * (self.Az * s1)
        base_v = u_vec * (2.0 * self.Au * w * c2) + z_vec * (self.Az * w * c1)
        base_a = u_vec * (-4.0 * self.Au * w * w * s2) + z_vec * (-self.Az * w * w * s1)
        p = self.p0 + e * base
        v = de * base + e * base_v
        a = dde * base + 2.0 * de * base_v + e * base_a
        return p, v, a, self.flat_attitude(a, self._yaw(t)), 0.0, self.required_thrust(a)


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


class ShiftedManeuver(Maneuver):
    """
    A manoeuvre RELOCATED by a rigid motion: rotated about world z to a new heading, then
    translated so that its first sample lands on a requested point.

    WHY THIS IS THE RIGHT TRANSFORM TO MOVE A REFERENCE. A yaw rotation plus a translation
    is a rigid motion of the entire trajectory, so every dynamic property keeps its
    meaning: p, v, a and R rotate, |a| and |omega| are unchanged, and the required
    collective thrust is untouched. Relocation therefore preserves FEASIBILITY - a
    manoeuvre that fits the thrust and rate authority in one place fits it everywhere.

    OMEGA IS DELIBERATELY NOT ROTATED. The body frame turns with the vehicle, so for
    R' = Q R the body rate is [omega']_x = R'^T R'dot = R^T Q^T Q Rdot = R^T Rdot = [omega]_x:
    the same body-frame vector. (Rotating omega by Q, by analogy with v and a, would
    describe a different manoeuvre - a 90 deg yaw shift of a pitch flip would claim a
    roll rate. The wrapper's central difference over the RELOCATED R produces the right
    answer by construction.)
    """

    def __init__(self, inner: Maneuver, p0: Sequence[float], yaw0: float):
        super().__init__()
        self.inner = inner
        self.duration = float(inner.duration)
        p_s, _v, _a, R_s, _spin, _thr = inner.pose(0.0)
        dyaw = float(yaw0) - _yaw_of(R_s)
        c, s = math.cos(dyaw), math.sin(dyaw)
        self._Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._shift = np.asarray(p0, dtype=np.float64) - self._Rz @ np.asarray(p_s, dtype=np.float64)

    @property
    def kind(self) -> str:  # type: ignore[override]
        """The relocated manoeuvre is still that manoeuvre (reward tolerances, telemetry)."""
        return self.inner.kind

    def kind_at(self, t: float) -> str:
        return self.inner.kind_at(t)

    def pose(self, t):
        p, v, a, R, spin, thr = self.inner.pose(t)
        return (self._Rz @ p + self._shift, self._Rz @ v, self._Rz @ a,
                self._Rz @ R, spin, float(thr))


class Chain(Maneuver):
    """
    A CHAIN of manoeuvres flown back-to-back as ONE command - a longer mission than any
    single family (say: flip, recover, climb-and-weave, recover, loop), assembled from
    the manoeuvres this module already provides.

    THE CONTINUITY CONTRACT. A reference is only usable if it is CONTINUOUS in position,
    velocity, attitude and body rate. A step in any of them asks the policy for a move
    the reference itself does not make, and the tracking reward would then charge the
    policy for the reference's own defect. The chain therefore requires every segment to
    START AND END PARKED: v = 0, level attitude, zero body rate. That is the contract
    every manoeuvre here already satisfies at its END (it is what "every manoeuvre ends
    in a hover" means); `Hover`, `WaypointTrajectory`, `Flip` and `VerticalEight`
    satisfy it at BOTH ends, and the periodic families (`FigureEight`, `Lissajous`,
    `Orbit`, `Slalom`) become rest-to-rest by passing `ease=<seconds>`.

    THE JUNCTIONS. Each segment is RELOCATED with `ShiftedManeuver` - a world-frame yaw
    plus a translation - so that its start pose lands exactly on the previous segment's
    end pose. Since both sides are parked and level, the junction is then continuous in
    p, v, a, R and omega, and the thrust command is continuous too (both sides ask for
    m*g). Optional `hold` seconds of hover are inserted between segments, so a chain
    reads as a sequence of distinct commands with a beat between them.

    ONE DOCUMENTED EXCEPTION. `Flip` has a deliberately piecewise-constant vertical
    acceleration: its thrust steps from m*g up to m*(u+g) at take-off and snaps back to
    trim at the hand-off (see its docstring - the profile is what makes the flip close on
    altitude exactly). A junction involving a flip therefore steps |a| ALONG WORLD Z
    ONLY, with position, velocity, attitude and body rate still continuous - exactly how
    a flip already behaves when it opens an episode. The check below allows that single
    case and counts it in `self.thrust_steps`; any other acceleration step, in any
    direction, raises.

    WHAT IS VERIFIED AT CONSTRUCTION (`verify=True`, the default): every segment boundary
    must be parked (v, attitude tilt, body rate, and a purely vertical acceleration), and
    every junction must match in p, v, a, R and omega - measured with the same central
    difference the environment uses. The match is exact by construction, so the
    tolerances are tight (1e-6) and a future change to a manoeuvre's end state fails
    loudly here instead of quietly biasing the reward.

    FEASIBILITY. Relocation is a rigid motion, so a chain inherits the feasibility of its
    parts: |a|, |omega| and the required thrust are unchanged, while the sampler then
    screens the WHOLE chain - thrust, body rate, flight volume, the 1.5 m footprint and
    the episode horizon - exactly as it does for every other family. Chains are long
    (2-4 commands of 1.2-4 s plus pauses), which is one of the reasons the training
    episode is 15 s: a chain that outlived the episode would be truncated mid-manoeuvre
    and would never reach its terminal hover.
    """

    kind = "chain"

    # A junction may not be off by more than this. Everything except the flip's thrust
    # step is exact to machine precision; these exist to catch a REGRESSION in a
    # manoeuvre's end state, not to tolerate one. A_TOL and W_TOL are looser than p/v/R
    # because a manoeuvre whose acceleration ramps in over a finite window leaves a
    # boundary acceleration and body rate proportional to the ramp rather than zero
    # (measured on VerticalEight: a_xy 5e-5 m/s^2 and |omega| 4.1e-3 rad/s = 0.23 deg/s,
    # four orders below the tightest rate tolerance in TRACK_TOL). A real step - a
    # discontinuity in R, or a lateral acceleration jump - is O(1) and still fails.
    P_TOL: float = 1e-6
    V_TOL: float = 1e-6
    A_TOL: float = 1e-3
    R_TOL: float = 1e-6
    W_TOL: float = 0.08

    def __init__(self, maneuvers: Sequence[Maneuver], hold: float = 0.0, verify: bool = True):
        super().__init__()
        segs = list(maneuvers)
        if not segs:
            raise ValueError("Chain needs at least one manoeuvre")
        for m in segs:
            if not isinstance(m, Maneuver):
                raise TypeError(f"Chain segments must be Maneuver instances, got {type(m).__name__}")
        self.hold = float(max(0.0, hold))
        self.thrust_steps: int = 0
        self.segments: List[Maneuver] = []
        parts: List[Maneuver] = []
        station_p = np.zeros(3, dtype=np.float64)
        station_yaw = 0.0
        for i, m in enumerate(segs):
            seg = m if i == 0 else ShiftedManeuver(m, station_p, station_yaw)
            self.segments.append(seg)
            parts.append(seg)
            end = seg.pose(seg.duration)
            station_p = np.asarray(end[0], dtype=np.float64).copy()
            station_yaw = _yaw_of(end[3])
            if self.hold > 0.0 and i < len(segs) - 1:
                parts.append(Hover(station_p, station_yaw, duration=self.hold))
        self._parts = parts
        self._starts: List[float] = []
        t = 0.0
        for part in parts:
            self._starts.append(t)
            t += float(part.duration)
        self.duration = float(t)
        if verify:
            self._verify()

    # -- dispatch -------------------------------------------------------------------
    def _index(self, t: float) -> int:
        """Index of the part active at time t (junctions belong to the LATER part)."""
        for i in range(len(self._parts) - 1, -1, -1):
            if t >= self._starts[i] - 1e-12:
                return i
        return 0

    def kind_at(self, t: float) -> str:
        # Beyond the chain's own duration sits the Trajectory wrapper's terminal hover.
        if t >= self.duration:
            return "hover"
        i = self._index(t)
        return self._parts[i].kind_at(t - self._starts[i])

    def pose(self, t):
        i = self._index(t)
        return self._parts[i].pose(t - self._starts[i])

    # -- verification ---------------------------------------------------------------
    def _verify(self) -> None:
        """
        Enforce the continuity contract. Raises ValueError naming the offending segment.

        The two things it can catch are worth separating:
          * a segment that is not rest-to-rest - typically a periodic family built
            without `ease`, which starts mid-motion. The message names the class and the
            offending quantity, because silently relocating it would put a velocity step
            into the reference.
          * a junction whose states do not line up. Everything after relocation lines up
            to machine precision, so a failure here means a manoeuvre's end state is not
            what the chain assumed (a regression, not a tuning issue).
        """
        for idx, seg in enumerate(self.segments):
            for label, t in (("start", 0.0), ("end", float(seg.duration))):
                _p, v, a, R, _spin, _thr = seg.pose(t)
                tilt = float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
                if float(np.linalg.norm(v)) > self.V_TOL:
                    raise ValueError(
                        f"Chain segment {idx} ({type(seg).__name__}) does not {label} at rest: "
                        f"|v| = {float(np.linalg.norm(v)):.3e} m/s. A segment must begin and end "
                        f"parked; the periodic families need `ease=<seconds>` to do so.")
                if tilt > 1e-3:
                    raise ValueError(
                        f"Chain segment {idx} ({type(seg).__name__}) does not {label} level: "
                        f"tilt = {math.degrees(tilt):.3f} deg.")
                if abs(float(a[0])) > self.A_TOL or abs(float(a[1])) > self.A_TOL:
                    raise ValueError(
                        f"Chain segment {idx} ({type(seg).__name__}) does not {label} with a "
                        f"vertical acceleration: a_xy = {np.round(np.asarray(a)[:2], 6).tolist()}.")
                # Body rate at the boundary, with the same central difference the
                # Trajectory wrapper uses (h is the wrapper's stencil half-width).
                h = 1e-3
                Rm = seg.pose(t - h)[3]
                Rp = seg.pose(t + h)[3]
                w = float(np.linalg.norm(omega_from_dcm(Rm, R, Rp, h)))
                if w > self.W_TOL:
                    raise ValueError(
                        f"Chain segment {idx} ({type(seg).__name__}) does not {label} with zero "
                        f"body rate: |omega| = {w:.3e} rad/s.")

        for i in range(len(self._parts) - 1):
            a_end = self._parts[i].pose(float(self._parts[i].duration))
            b_start = self._parts[i + 1].pose(0.0)
            dp = float(np.linalg.norm(a_end[0] - b_start[0]))
            dv = float(np.linalg.norm(a_end[1] - b_start[1]))
            dR = float(np.max(np.abs(a_end[3] - b_start[3])))
            if dp > self.P_TOL or dv > self.V_TOL or dR > self.R_TOL:
                raise ValueError(
                    f"Chain junction {i} is discontinuous: |dp| = {dp:.3e} m, "
                    f"|dv| = {dv:.3e} m/s, |dR| = {dR:.3e}.")
            da = np.asarray(a_end[2], dtype=np.float64) - np.asarray(b_start[2], dtype=np.float64)
            if float(np.linalg.norm(da)) > self.A_TOL:
                # Legal ONLY as the flip's vertical thrust step; anything with a lateral
                # component - or a mismatch between two non-vertical accelerations - is a
                # real discontinuity in the reference attitude (R = f(a)) and is refused.
                if abs(float(da[0])) > self.A_TOL or abs(float(da[1])) > self.A_TOL:
                    raise ValueError(
                        f"Chain junction {i} steps the acceleration laterally: da = "
                        f"{np.round(da, 6).tolist()}.")
                self.thrust_steps += 1
            t_j = self._starts[i + 1]
            h = 1e-3
            Rj = self.pose(t_j)[3]
            w_j = float(np.linalg.norm(
                omega_from_dcm(self.pose(t_j - h)[3], Rj, self.pose(t_j + h)[3], h)))
            if w_j > self.W_TOL:
                raise ValueError(
                    f"Chain junction {i} steps the reference body rate: |omega| = {w_j:.3e} rad/s "
                    f"across the measurement stencil.")


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

    def __init__(self, maneuver: Maneuver, omega_dt: float = 0.001, tail: float = TRAJECTORY_TAIL):
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
            thrust_ff=thrust, spin=spin, kind=self.maneuver.kind_at(t),
        )

    def initial_state(self) -> Reference:
        return self.sample(0.0)


@dataclass
class TrajectoryConfig:
    """Sampling configuration for the manoeuvre mixture."""

    # FLIGHT VOLUME.
    #
    # A sphere of radius `flight_radius` centred on the spawn point (0, 0, spawn_z). The
    # sphere's natural bottom is spawn_z - flight_radius = -0.8 m, i.e. underground, so it
    # is effectively CLIPPED AT THE GROUND - which is exactly 1.2 m below the start. That
    # clip is the reason the spawn sits at the CENTRE of the volume rather than on its
    # floor: with the pad on the boundary, any downward component of the initial-velocity
    # randomisation would put the vehicle outside the volume on step 1.
    spawn_z: float = 1.2
    flight_radius: float = 2.0

    # TRAINING FOOTPRINT (hard limit). Every point of every sampled reference must satisfy
    # |x| <= bounds_xy and |y| <= bounds_xy, i.e. the whole manoeuvre fits inside a
    # 1.5 m x 1.5 m square centred on the world origin (bounds_xy = 0.75 = half of 1.5 m).
    # This is ENFORCED, not merely requested at draw time: the sampler screens 96 points
    # along each candidate and rejects any that leaves the square, so no episode is spent
    # chasing a reference the training volume does not contain. An earlier 1.0 draw bound
    # let references roam up to ~1.9 m out - far outside the square.
    # The flight sphere (flight_radius, 2.0 m) remains the VEHICLE's outer termination
    # guard; the square is deliberately much tighter and bounds the REFERENCES only.
    bounds_xy: float = 0.75

    # Waypoint targets are drawn from a square shrunk INSIDE the footprint, because the
    # fitted polynomial rings outside its knots. Measured over 1000 draws: targets at the
    # footprint edge (+-0.75) produce paths reaching +-1.51 and only 44% would pass the
    # footprint screen, while +-0.50 accepts 93%. The other families are geometric and
    # draw directly at their own limits: worst measured extents are figure-8 0.40,
    # lissajous 0.35, orbit 0.55 and slalom 0.63 m, all well inside the square.
    waypoint_bounds_xy: float = 0.50

    # Where reference TARGETS may go vertically. Every manoeuvre STARTS at spawn_z; only
    # waypoint targets are allowed to climb. At the top of this range the sphere's
    # horizontal allowance is still ~1.83 m, comfortably more than bounds_xy, so the
    # whole reference stays inside the volume.
    z_range: Tuple[float, float] = (1.2, 2.0)

    # Top of the sphere: spawn_z + flight_radius. Used to screen flip altitude excursions.
    z_max: float = 3.2

    # Weights are over DRAWS, not over accepted episodes: a manoeuvre that is rejected more
    # often is under-represented in the realised mixture. Flips still reject ~32% of draws
    # (almost all the k=2 doubles, whose altitude excursion is 4x the single and does not
    # fit inside a 2 m sphere), so the flip weight is set ABOVE its intended share to
    # compensate. Measured realised mixture at these weights is printed by
    # scratch/check_trajectories.py.
    #
    # `v8` (the vertical figure-eight) and `chain` are the 2026-09-15 additions. v8 is an
    # ACROBATIC family (attitude excursions up to ~60-70 deg, body rate up to ~16 rad/s,
    # heavy thrust modulation), so it is weighted like a sibling of the flip. A chain is
    # LONG - 2-4 commands plus hover pauses, so 6-12 s of a 15 s episode - which is why
    # its DRAW weight is small: weight is per draw, and draw share is not time share.
    # Set a weight to 0.0 to drop a family without touching the code.
    weights: dict = field(default_factory=lambda: {
        "hover": 0.07, "takeoff": 0.10, "waypoints": 0.11, "figure8": 0.06, "lissajous": 0.08,
        "orbit": 0.10, "slalom": 0.08, "flip": 0.25, "v8": 0.10, "chain": 0.05,
    })
    max_resample: int = 40

    # Episode length the reference must fit inside, in seconds. The env (train.py:
    # EPISODE_SECONDS) truncates at this horizon, so a manoeuvre that outlives it gets cut
    # mid-flight - which would break the "every episode ends in the terminal hover"
    # property the reward and the truncation bootstrap rely on. The env sets this from its
    # own episode_seconds at construction; the value here is only the standalone default.
    # FigureEight and Lissajous derive their duration from their angular frequency, so the
    # sampler floors that frequency at 2*pi / (episode_seconds - TRAJECTORY_TAIL).
    #
    # 8 -> 15 s on 2026-09-15: long enough to hold a full CHAIN (a couple of acrobatic
    # commands plus the hover beats between them) instead of truncating it, and it costs
    # nothing for the short families - they simply end earlier and the terminal hold runs
    # longer, which is a hover the policy already has to fly.
    episode_seconds: float = 15.0
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

    # -- mixture weights at runtime --------------------------------------------------
    def set_weight(self, name: str, weight: float) -> None:
        """Set one family's draw weight.

        Weights are RELATIVE: `TrajectorySampler.sample` normalises them by their sum on
        every draw, so raising one family automatically lowers every other family's share
        in proportion - no renormalisation is needed here, and adding a weight does not
        require knowing the others.

        This is the hook the training-time mixture curriculum drives (see
        `train.py`'s ManeuverMixCurriculumCallback): the sampler reads `cfg.weights` on
        every draw, so a change takes effect on the next environment reset, including in
        the SubprocVecEnv workers.
        """
        if name not in self.weights:
            raise ValueError(f"unknown manoeuvre {name!r}; expected one of {list(self.weights)}")
        w = float(weight)
        if not np.isfinite(w) or w < 0.0:
            raise ValueError(f"weight for {name!r} must be finite and >= 0, got {weight!r}")
        self.weights[name] = w

    def normalized_weights(self) -> dict:
        """The mixture probabilities as the sampler actually draws them (sums to 1)."""
        total = float(sum(self.weights.values()))
        if total <= 0.0:
            return {k: 0.0 for k in self.weights}
        return {k: float(v) / total for k, v in self.weights.items()}


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
        # Screen against the REAL top of the flight volume, not the old hardcoded 2.40 m.
        return fl if fl.is_feasible(z_max=self.cfg.z_max) else None

    # Chain recipes: the families a chain may be assembled from, and how often each is
    # drawn. Every one of them is rest-to-rest (the periodic members are built with
    # `ease` here, see Chain), which is what makes the junctions exact. `waypoints` is the
    # only member that MOVES the station; every other segment is a closed loop that
    # returns to where it started, so the chain does not wander out of the training
    # footprint simply by being long.
    CHAIN_KINDS: tuple = ("flip", "v8", "orbit", "figure8", "lissajous", "waypoints")
    CHAIN_KIND_WEIGHTS: tuple = (0.26, 0.24, 0.14, 0.12, 0.10, 0.14)

    def _make_v8(self, rng, p0, yaw) -> VerticalEight:
        """
        Draw the acrobatic vertical figure-eight.

        The ranges are the measured feasible band, not a guess. They were chosen by
        sweeping draws and keeping the worst case well inside the envelope: over 300 draws
        from these ranges the peak reference body rate was 15.5 rad/s (p90 10.5, median
        6.9 - the flip's coast runs at 14-17 rad/s), peak tilt 58 deg (median 44), peak
        thrust 0.465 N against the 0.60 N authority, and the vertical acceleration never
        came closer than 4.5 m/s^2 to -g, so the flatness direction never approaches the
        degenerate weightless point described in the class docstring.

        Sampling wider (e.g. z_amp up to 0.42 with a 2 s cycle) produces draws whose entry
        ramp drives a_z THROUGH -g for a few hundred milliseconds; the attitude direction
        then whips at >100 rad/s. Those draws are rejected by the rate screen, but there
        is no reason to generate them in the first place, so the band stops short of them.

        `ease` and `settle` are drawn as a FRACTION of the cycle, so the ramp always takes
        the same slice of the figure regardless of how fast the 8 is flown.
        """
        T = float(rng.uniform(2.2, 2.8))
        w = 2.0 * np.pi / T
        return VerticalEight(
            p0,
            z_amp=float(rng.uniform(0.26, 0.36)),
            u_amp=float(rng.uniform(0.18, 0.28)),
            w=w,
            heading=float(rng.uniform(-np.pi, np.pi)),
            yaw=yaw,
            yaw_rate=float(rng.uniform(-1.0, 1.0)),
            ease=float(rng.uniform(0.40, 0.50)) * T,
            settle=float(rng.uniform(0.40, 0.50)) * T,
        )

    def _make_chain(self, rng, mass) -> Optional[Maneuver]:
        """
        Assemble a multi-command chain: 2-3 rest-to-rest manoeuvres flown in sequence
        from a station near the spawn point, with a short hover beat between them.

        Each segment is built in CANONICAL coordinates (starting at the spawn altitude on
        a zero heading) and `Chain` relocates it onto the pose the previous segment left
        behind - that is what makes the junctions exact without the sampler knowing
        anything about the composition. Durations are drawn against a budget so the whole
        chain still fits inside one episode; a draw that cannot is abandoned and the
        sampler redraws, exactly like any other rejected candidate.
        """
        cfg = self.cfg
        # Small station box: every command loops back to the station, and the waypoint
        # segment rings slightly past its knots, so the whole chain has to fit inside the
        # 1.5 m footprint from a point that is not already near its edge.
        station = np.array([
            rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), float(cfg.spawn_z),
        ])
        yaw0 = float(rng.uniform(-np.pi, np.pi))
        hold = float(rng.uniform(0.30, 0.55))
        n = int(rng.integers(2, 4))
        budget = cfg.episode_seconds - TRAJECTORY_TAIL - (n - 1) * hold
        p_ref = np.array([0.0, 0.0, cfg.spawn_z])
        kinds = list(self.CHAIN_KINDS)
        probs = np.array(self.CHAIN_KIND_WEIGHTS, dtype=np.float64)
        probs /= probs.sum()
        parts: List[Maneuver] = []
        for _ in range(n):
            chosen = str(rng.choice(kinds, p=probs))
            if chosen == "flip":
                seg: Optional[Maneuver] = self._make_flip(rng, p_ref, 0.0, mass)
            elif chosen == "v8":
                seg = self._make_v8(rng, p_ref, 0.0)
            elif chosen == "orbit":
                settle = float(rng.uniform(0.6, 0.9))
                seg = Orbit(
                    center=[0.0, 0.0, cfg.spawn_z],
                    radius=float(rng.uniform(0.25, 0.45)), w=1.2,
                    climb=float(rng.uniform(-0.15, 0.20)), turns=1.0,
                    ease=settle, settle=settle,
                    duration=float(rng.uniform(2.8, 3.6)),
                )
            elif chosen == "figure8":
                settle = float(rng.uniform(0.6, 0.9))
                seg = FigureEight(
                    A=float(rng.uniform(0.20, 0.32)), B=float(rng.uniform(0.20, 0.32)),
                    w=float(rng.uniform(1.5, 2.0)), z0=cfg.spawn_z, cycles=1.0,
                    ease=settle, settle=settle,
                )
            elif chosen == "lissajous":
                settle = float(rng.uniform(0.6, 0.9))
                seg = Lissajous(
                    A=float(rng.uniform(0.15, 0.28)), B=float(rng.uniform(0.15, 0.28)),
                    w=float(rng.uniform(1.5, 2.0)), z0=cfg.spawn_z,
                    a=float(rng.choice([1.0, 1.0, 2.0])), b=float(rng.choice([2.0, 3.0])),
                    phi=float(rng.uniform(0.0, np.pi)), C=float(rng.uniform(0.0, 0.10)),
                    ease=settle, settle=settle,
                )
            else:
                # The one segment that translates: targets are drawn as small offsets
                # from the canonical start so that the station cannot run away, and the
                # z targets stay inside the band the flip and the v8 need around them.
                n_wp = 3
                seg = WaypointTrajectory(
                    waypoints=[
                        p_ref + np.array([
                            rng.uniform(-0.25, 0.25),
                            rng.uniform(-0.25, 0.25),
                            rng.uniform(-0.20, 0.30),
                        ])
                        for _ in range(n_wp)
                    ],
                    segment_time=float(rng.uniform(0.8, 1.1)),
                )
            if seg is None or float(seg.duration) > budget:
                return None
            budget -= float(seg.duration)
            parts.append(seg)
        try:
            chain = Chain(parts, hold=hold)
        except ValueError:
            return None
        # The chain is placed ON the drawn station with the same rigid relocation its
        # segments get internally, so every episode starts somewhere different while the
        # command stays relative to itself (and the whole thing stays well inside the
        # footprint whatever station was drawn).
        return ShiftedManeuver(chain, station, yaw0)

    def sample(self, rng: np.random.Generator, mass: float = MASS_NOMINAL,
               kind: Optional[str] = None) -> Trajectory:
        """
        Draw a feasible trajectory.

        `kind` pins the manoeuvre to one of the config weights (any key of cfg.weights)
        for the high-level command interface; None samples from the mixture. In both cases
        the result is guaranteed feasible AND guaranteed to finish inside one episode, so
        every manoeuvre ENDS IN A HOVER: hover trivially, waypoints and flips by their zero
        terminal velocity, figure-8 and lissajous via their settle envelope. It is also
        guaranteed to fit the training footprint: every point of the path satisfies
        |x|, |y| <= cfg.bounds_xy (a 1.5 m x 1.5 m square), enforced by a screen below.
        """
        kinds = list(self.cfg.weights.keys())
        probs = np.array([self.cfg.weights[k] for k in kinds], dtype=np.float64)
        probs /= probs.sum()
        if kind is not None and kind not in kinds:
            raise ValueError(f"unknown manoeuvre {kind!r}; expected one of {kinds}")

        # Highest angular frequency whose whole manoeuvre (plus the terminal-hold tail)
        # still fits the episode. Only the duration-from-w families use it, but the screen
        # at the bottom of the loop checks every family.
        min_w = 2.0 * np.pi / max(1e-6, self.cfg.episode_seconds - TRAJECTORY_TAIL)

        for _ in range(self.cfg.max_resample):
            chosen = kind if kind is not None else str(rng.choice(kinds, p=probs))
            # EVERY manoeuvre starts at the same altitude, which is also the CENTRE of the
            # flight sphere. Starting at the centre is what gives the initial-velocity
            # randomisation equal room in every direction; starting on the floor of the
            # volume would make any downward kick an immediate violation.
            z0 = float(self.cfg.spawn_z)
            p0 = np.array([
                rng.uniform(-self.cfg.bounds_xy, self.cfg.bounds_xy),
                rng.uniform(-self.cfg.bounds_xy, self.cfg.bounds_xy),
                z0,
            ])
            yaw = float(rng.uniform(-np.pi, np.pi))

            if chosen == "hover":
                traj = Trajectory(Hover(
                    p0, yaw, duration=float(rng.uniform(1.5, 3.0)),
                    yaw_rate=float(rng.uniform(-1.5, 1.5)),
                ))
            elif chosen == "takeoff":
                # Start from ground (z=0.025m) or anywhere between 0 and 1.2m
                if rng.random() < 0.50:
                    z_start = 0.025  # Ground resting level
                else:
                    z_start = float(rng.uniform(0.025, 1.20))
                p_start = np.array([
                    rng.uniform(-0.35, 0.35),
                    rng.uniform(-0.35, 0.35),
                    z_start,
                ], dtype=np.float64)
                z_target = float(rng.uniform(1.00, 1.30))
                p_target = np.array([
                    rng.uniform(-0.40, 0.40),
                    rng.uniform(-0.40, 0.40),
                    z_target,
                ], dtype=np.float64)
                climb_time = max(1.2, float(rng.uniform(1.6, 2.8)))
                hold_time = float(rng.uniform(2.5, 5.0))
                traj = Trajectory(Takeoff(
                    p0=p_start,
                    p1=p_target,
                    climb_time=climb_time,
                    hold_time=hold_time,
                    yaw=yaw,
                    yaw_rate=float(rng.uniform(-1.0, 1.0)),
                ))
            elif chosen == "waypoints":
                n = int(rng.integers(3, 5))
                # NOT bounds_xy: the interpolating polynomial rings past its knots, so
                # targets are drawn from the smaller square documented in
                # TrajectoryConfig.waypoint_bounds_xy.
                bb = self.cfg.waypoint_bounds_xy
                wps = [p0] + [
                    np.array([
                        rng.uniform(-bb, bb),
                        rng.uniform(-bb, bb),
                        float(rng.uniform(*self.cfg.z_range)),
                    ])
                    for _ in range(n)
                ]
                traj = Trajectory(WaypointTrajectory(wps, segment_time=float(rng.uniform(0.9, 1.6)), yaw=yaw))
            elif chosen == "figure8":
                w = max(float(rng.uniform(0.8, 1.8)), min_w)
                traj = Trajectory(FigureEight(
                    A=float(rng.uniform(0.2, 0.4)),
                    B=float(rng.uniform(0.2, 0.4)),
                    w=w, z0=z0, cycles=1.0, yaw=0.0,
                ))
            elif chosen == "orbit":
                # Centred on the world origin (not on the random p0) so the whole circle
                # stays well inside the flight sphere for ANY radius drawn.
                traj = Trajectory(Orbit(
                    center=[0.0, 0.0, z0],
                    radius=float(rng.uniform(0.25, 0.55)),
                    w=1.2,
                    climb=float(rng.uniform(-0.20, 0.30)),
                    turns=float(rng.choice([1.0, 1.0, 2.0])),
                    yaw=yaw, yaw_rate=float(rng.uniform(-1.2, 1.2)),
                    duration=float(rng.uniform(3.5, 7.0)),
                ))
            elif chosen == "lissajous":
                traj = Trajectory(Lissajous(
                    A=float(rng.uniform(0.15, 0.35)),
                    B=float(rng.uniform(0.15, 0.35)),
                    w=max(float(rng.uniform(0.7, 1.7)), min_w), z0=z0,
                    a=float(rng.choice([1.0, 1.0, 2.0])),
                    b=float(rng.choice([2.0, 3.0])),
                    phi=float(rng.uniform(0.0, np.pi)),
                    C=float(rng.uniform(0.0, 0.12)),
                    yaw=yaw, yaw_rate=float(rng.uniform(-1.0, 1.0)),
                ))
            elif chosen == "slalom":
                heading = float(rng.uniform(-np.pi, np.pi))
                dist = float(rng.uniform(0.5, 1.2))
                u_vec = np.array([np.cos(heading), np.sin(heading), 0.0])
                # Start offset so the traverse is CENTRED on the origin, which keeps the
                # whole path inside the flight sphere whatever heading was drawn.
                traj = Trajectory(Slalom(
                    start=-u_vec * (dist * 0.5), heading=heading, dist=dist,
                    A=float(rng.uniform(0.12, 0.30)), w=2.0, z0=z0,
                    yaw=yaw, yaw_rate=float(rng.uniform(-1.0, 1.0)),
                    cycles=float(rng.choice([2.0, 2.5, 3.0])),
                    duration=float(rng.uniform(2.5, 4.0)),
                ))
            elif chosen == "v8":
                # A tighter station box than the other families: the 8 WEAVES around its
                # start point by up to u_amp, and the start point is where the episode
                # spawns, so drawing it at the edge of the footprint would put the far
                # leaf straight through the footprint screen. (The families that roam -
                # figure-8, lissajous, orbit, slalom - are centred on the origin for the
                # same reason.)
                station = 0.30
                p0_v8 = np.array([
                    rng.uniform(-station, station), rng.uniform(-station, station), z0,
                ])
                traj = Trajectory(self._make_v8(rng, p0_v8, yaw))
            elif chosen == "chain":
                chain = self._make_chain(rng, mass)
                if chain is None:
                    continue
                traj = Trajectory(chain)
            else:
                fl = self._make_flip(rng, p0, yaw, mass)
                if fl is None:
                    continue
                # A flip is horizontal-stationary at p0 (Flip.pose writes p0's x/y
                # unchanged at every t), so its footprint is exactly the spawn point.
                # p0 is drawn from +-bounds_xy, so this passes by construction; the
                # guard is kept so the flip branch cannot silently leave the footprint
                # if that draw ever changes.
                if max(abs(p0[0]), abs(p0[1])) > self.cfg.bounds_xy:
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
                # A chain is long and carries several manoeuvres inside it, so its screen
                # is sampled four times as finely: at 96 points a 12 s chain is checked
                # every 125 ms, which is enough to walk past the peak of a flip's climb.
                n_screen = 256 if chosen == "chain" else 96
                ts = np.linspace(0.0, traj.duration, n_screen)
                refs = [traj.sample(float(t)) for t in ts]
                req = np.array([r.thrust_ff for r in refs])
                if req.max() > 0.95 * MAX_THRUST_TOTAL or req.min() < MIN_THRUST_TOTAL:
                    continue
                rate = max(float(np.linalg.norm(r.omega)) for r in refs)
                if rate > 0.95 * max(self.cfg.rate_limits.values()):
                    continue
                # VOLUME SCREEN. The reference must stay inside the flight sphere with
                # margin, or the policy would be asked to track a path that leaves the
                # arena - the drone would be terminated for doing exactly what it was told.
                offset = np.array([0.0, 0.0, self.cfg.spawn_z])
                furthest = max(float(np.linalg.norm(r.p - offset)) for r in refs)
                if furthest > 0.85 * self.cfg.flight_radius:
                    continue
                # FOOTPRINT SCREEN - this is what makes the 1.5 m x 1.5 m training
                # footprint a guarantee rather than a drawing convention. "Targets were
                # drawn inside" is not enough: the waypoint polynomial rings past its
                # knots, and this is the check that catches it (the other families pass
                # with 0.07-0.35 m of margin). Same 96 samples as the volume screen.
                if max(float(np.max(np.abs(r.p[:2]))) for r in refs) > self.cfg.bounds_xy:
                    continue

            # EPISODE-LENGTH SCREEN. Checked for every family (the flip branch returns
            # before the screen above), because a reference that outlives the episode is
            # truncated mid-manoeuvre and never reaches its terminal hover.
            if traj.duration > self.cfg.episode_seconds:
                continue
            return traj

        # Fallback: a trivially feasible hold at the nominal altitude.
        return Trajectory(Hover([0.0, 0.0, 1.2], 0.0, duration=2.0))
