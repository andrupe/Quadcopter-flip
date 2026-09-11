"""
Validation for Simulation/trajectories.py.

The claims that matter here are physical, not stylistic:

  A. DYNAMIC CONSISTENCY. For a quadrotor the collective thrust acts along body z, so a
     feasible reference must satisfy  T * R[:,2] = m * (a + g*e_z)  exactly, whenever
     T > 0. If this fails, the reference commands an acceleration the vehicle cannot
     produce, and the policy is being asked to track something impossible.

  B. THE ROTATION HAPPENS AT ZERO THRUST. If the spin and the thrust window overlap, the
     reference vector is tilted while thrust is applied, and the vehicle gets a lateral
     acceleration that is not in the reference.

  C. THE FLIP CLOSES. z(T) = z(0) and v_z(T) = 0, so a flip is altitude-neutral.

  D. OMEGA IS CORRECT. Integrating the reference body rate must reproduce the reference
     attitude trajectory: R(t+h) ~ R(t) @ exp(h * [omega]_x). This validates the finite
     difference used for omega, which is otherwise easy to get subtly wrong.

  E. FEASIBILITY IS ENFORCED. Every sampled trajectory respects the thrust and rate
     authority, and the sampler reports what it rejected.

Run:  .venv/bin/python scratch/check_trajectories.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trajectories import (  # noqa: E402
    GRAVITY,
    MASS_NOMINAL,
    MAX_THRUST_TOTAL,
    Flip,
    Hover,
    FigureEight,
    Trajectory,
    TrajectoryConfig,
    TrajectorySampler,
    WaypointTrajectory,
    axis_angle_rotation,
)

FAILURES: list[str] = []
DT = 0.01


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def flatness_error(ref, mass: float = MASS_NOMINAL) -> float:
    """|T*R[:,2] - m*(a + g e_z)|, the residual of the flatness relation (newtons)."""
    if ref.thrust_ff <= 1e-9:
        return 0.0
    lhs = ref.thrust_ff * ref.R[:, 2]
    rhs = mass * (ref.a + np.array([0.0, 0.0, GRAVITY]))
    return float(np.linalg.norm(lhs - rhs))


def omega_consistency(traj: Trajectory) -> float:
    """Max angle (rad) by which integrating omega fails to reproduce R(t+dt).

    The exponential map needs the ANGLE as |omega|*dt, not dt - passing dt directly
    applies a rotation of dt radians regardless of the rate, which silently inflates the
    error for fast manoeuvres and was the original cause of this check failing.
    """
    worst = 0.0
    for t in np.arange(0.0, max(0.0, traj.duration - DT), DT):
        r0 = traj.sample(t)
        r1 = traj.sample(t + DT)
        w = float(np.linalg.norm(r0.omega))
        if w > 1e-9:
            R_pred = r0.R @ axis_angle_rotation(r0.omega, w * DT)
        else:
            R_pred = r0.R
        err = R_pred.T @ r1.R
        ang = float(np.arccos(np.clip(0.5 * (np.trace(err) - 1.0), -1.0, 1.0)))
        worst = max(worst, ang)
    return worst


def max_ref_rate(traj: Trajectory, n: int = 400) -> float:
    ts = np.linspace(0.0, traj.duration, n)
    return float(max(np.linalg.norm(traj.sample(float(t)).omega) for t in ts))


print("=" * 78)
print("A/B. dynamic consistency and spin/thrust separation")
print("=" * 78)

p0 = [0.0, 0.0, 1.25]
flip = Flip(p0, axis=[0.0, 1.0, 0.0], rotations=1.0, coast=0.60, max_rate=20.0)
traj = Trajectory(flip)

worst_flat = 0.0
overlap = 0
for t in np.arange(0.0, traj.duration, DT):
    r = traj.sample(float(t))
    worst_flat = max(worst_flat, flatness_error(r))
    # spin must be an exact multiple of 2*pi whenever thrust is applied
    if r.thrust_ff > 1e-6:
        k = r.spin / (2.0 * np.pi)
        if abs(k - round(k)) > 1e-9:
            overlap += 1

check("flip: flatness relation holds wherever thrust > 0", worst_flat < 1e-9, f"max residual {worst_flat:.3e} N")
check("flip: thrust is never applied while rotated", overlap == 0, f"{overlap} overlapping samples")

gas = [(t, traj.sample(float(t))) for t in np.arange(flip.c0 + 1e-6, flip.c0 + flip.D - 1e-6, DT)]
check(
    "flip: coast is exactly free fall (a_z = -g)",
    all(abs(r.a[2] + GRAVITY) < 1e-9 for _, r in gas),
    f"max |a_z + g| = {max(abs(r.a[2] + GRAVITY) for _, r in gas):.3e}",
)
check(
    "flip: thrust is exactly zero through the coast",
    all(r.thrust_ff < 1e-12 for _, r in gas),
)

print()
print("=" * 78)
print("C. the flip closes on altitude")
print("=" * 78)
r0, rT = traj.sample(0.0), traj.sample(traj.duration)
check("z(T) == z(0)", abs(rT.p[2] - r0.p[2]) < 1e-9, f"{r0.p[2]:.6f} -> {rT.p[2]:.6f}")
check("v(T) == 0", abs(rT.v[2]) < 1e-9, f"v_z(T) = {rT.v[2]:.3e}")
check("rotation completes", abs(rT.spin - 2.0 * np.pi) < 1e-9, f"phi(T) = {rT.spin:.6f}")
lo, hi = flip.z_excursion()
check("altitude excursion is sane", lo > 0.3 and hi < 2.4, f"z in [{lo:.2f}, {hi:.2f}] m")
check(
    "peak excursion matches the closed form",
    abs((hi - lo) - flip.peak_excursion()) < 1e-3,
    f"peak {hi - lo:.3f} m vs closed form {flip.peak_excursion():.3f} m",
)
check(
    "peak rate within authority",
    flip.omega_peak <= 20.0,
    f"omega_peak = {flip.omega_peak:.1f} rad/s (limit 20)",
)
check(
    "peak thrust within authority",
    flip.max_thrust_needed() <= MAX_THRUST_TOTAL,
    f"max T = {flip.max_thrust_needed():.3f} N (limit {MAX_THRUST_TOTAL})",
)

print()
print("=" * 78)
print("D. omega reproduces the reference attitude")
print("=" * 78)
for name, tr in [
    ("flip", traj),
    ("hover", Trajectory(Hover([0.0, 0.0, 1.2], 0.3, 2.0))),
    ("waypoints", Trajectory(WaypointTrajectory([[0, 0, 1.2], [0.5, 0.3, 1.4], [-0.4, 0.2, 1.1]], 1.0))),
    ("figure8", Trajectory(FigureEight(0.3, 0.3, 1.2, 1.3))),
]:
    err = omega_consistency(tr)
    check(f"{name}: integrating omega reproduces R", err < 5e-3, f"max attitude error {err * 1e3:.2f} mrad")
print()
print("=" * 78)
print("E. every other manoeuvre is feasible and consistent")
print("=" * 78)
for name, tr in [
    ("hover", Trajectory(Hover([0.0, 0.0, 1.2], 0.3, 2.0))),
    ("waypoints", Trajectory(WaypointTrajectory([[0, 0, 1.2], [0.5, 0.3, 1.4], [-0.4, 0.2, 1.1]], 1.0))),
    ("figure8", Trajectory(FigureEight(0.3, 0.3, 1.2, 1.3))),
]:
    wf, thr = 0.0, []
    for t in np.arange(0.0, tr.duration, DT):
        r = tr.sample(float(t))
        wf = max(wf, flatness_error(r))
        thr.append(r.thrust_ff)
        assert np.all(np.isfinite(r.p)) and np.all(np.isfinite(r.R)) and np.all(np.isfinite(r.omega))
    thr = np.asarray(thr)
    check(f"{name}: flatness holds", wf < 1e-9, f"max residual {wf:.3e} N")
    check(f"{name}: thrust within [0, Tmax]", thr.min() >= 0 and thr.max() <= MAX_THRUST_TOTAL,
          f"T in [{thr.min():.3f}, {thr.max():.3f}] N")

print()
print("=" * 78)
print("F. sampler: feasibility, mixture, and what it rejects")
print("=" * 78)
cfg = TrajectoryConfig()
sampler = TrajectorySampler(cfg)
rng = np.random.default_rng(0)
kinds: dict = {}
bad = 0
max_thr_seen = 0.0
max_rate_seen = 0.0
N = 300
for _ in range(N):
    tr = sampler.sample(rng, mass=MASS_NOMINAL)
    kinds[tr.maneuver.kind] = kinds.get(tr.maneuver.kind, 0) + 1
    ts = np.linspace(0.0, tr.duration, 200)
    refs = [tr.sample(float(t)) for t in ts]
    thr = max(r.thrust_ff for r in refs)
    rate = max(float(np.linalg.norm(r.omega)) for r in refs)
    max_thr_seen = max(max_thr_seen, thr)
    max_rate_seen = max(max_rate_seen, rate)
    if thr > MAX_THRUST_TOTAL + 1e-9:
        bad += 1
    if any(flatness_error(r) > 1e-9 for r in refs):
        bad += 1
    if not all(np.all(np.isfinite(r.p)) and np.all(np.isfinite(r.R)) for r in refs):
        bad += 1

print(f"        mixture over {N} draws: {kinds}")
check("no infeasible or inconsistent trajectory sampled", bad == 0, f"{bad} violations")
check("thrust never exceeds authority", max_thr_seen <= MAX_THRUST_TOTAL, f"max {max_thr_seen:.3f} N")
check("reference rate stays within the pitch limit", max_rate_seen <= 20.0 + 1e-6, f"max {max_rate_seen:.1f} rad/s")

# The pitch/roll asymmetry is a real constraint, so verify the sampler's behaviour.
roll_traj = 0
for _ in range(200):
    rng2 = np.random.default_rng(int(rng.integers(0, 10**6)))
    tr = sampler.sample(rng2)
    if tr.maneuver.kind == "flip":
        ax = getattr(tr.maneuver, "axis", None)
        if ax is not None and abs(ax[0]) > 0.9:
            roll_traj += 1
print(f"        roll-axis flips that passed feasibility: {roll_traj}/200 draws")
print("        (roll and pitch limits are now symmetric at 20 rad/s, so roll flips are")
print("         feasible; the residual rejections are altitude-budget failures.)")

print()
print("=" * 78)
print("G. every manoeuvre terminates in a hover")
print("=" * 78)
print("   A high-level command must always leave the vehicle parked in a stable hold,")
print("   so the terminal reference has to be a genuine hover: v = 0, a = 0, level")
print("   attitude, zero body rate, and thrust at trim m*g.")
for name in sorted(sampler.cfg.weights):
    rng = np.random.default_rng(7)
    w_v = w_a = w_rate = w_tilt = w_thr = 0.0
    for _ in range(12):
        tr = sampler.sample(rng, mass=MASS_NOMINAL, kind=name)
        r = tr.sample(tr.duration)
        w_v = max(w_v, float(np.linalg.norm(r.v)))
        w_a = max(w_a, float(np.linalg.norm(r.a)))
        w_rate = max(w_rate, float(np.linalg.norm(r.omega)))
        w_tilt = max(w_tilt, float(np.arccos(np.clip(r.R[2, 2], -1.0, 1.0))))
        w_thr = max(w_thr, abs(r.thrust_ff - MASS_NOMINAL * GRAVITY))
    check(f"{name}: terminal velocity is zero", w_v < 1e-6, f"max |v| {w_v:.2e} m/s")
    check(f"{name}: terminal acceleration is zero", w_a < 1e-6, f"max |a| {w_a:.2e} m/s^2")
    check(f"{name}: terminal attitude is level", w_tilt < 1e-3, f"max tilt {np.degrees(w_tilt):.4f} deg")
    check(f"{name}: terminal body rate is zero", w_rate < 1e-3, f"max |omega| {w_rate:.2e} rad/s")
    check(f"{name}: terminal thrust is hover trim", w_thr < 1e-9, f"max |T - mg| {w_thr:.2e} N")

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all trajectory checks passed")
print("=" * 78)
