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
     authority AND fits the 1.5 m x 1.5 m training footprint (|x|, |y| <=
     TrajectoryConfig.bounds_xy), and the sampler reports what it rejected.
  F. MIXTURE. The realised family mix and the worst-case envelope over many draws.
  G. TERMINAL HOVER. Every manoeuvre ends parked, so an episode always finishes in a
     hold regardless of which family was drawn.
  H. VERTICAL FIGURE-EIGHT (v8). It is a genuine vertical lemniscate (it crosses itself
     halfway), it is acrobatic (large attitude swing, high body rate) and it is still
     exactly flyable.
  I. CHAINS. A chain must be continuous across its junctions (checked by DIFFERENTIATING
     the sampled reference, not by re-reading the constructor), feasible as a whole, and
     must report the kind of the segment being flown so the reward keeps its per-family
     tolerances.

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
    Chain,
    FigureEight,
    Flip,
    Hover,
    Orbit,
    ShiftedManeuver,
    Trajectory,
    TrajectoryConfig,
    TrajectorySampler,
    VerticalEight,
    WaypointTrajectory,
    axis_angle_rotation,
    omega_from_dcm,
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


def omega_consistency(traj: Trajectory, dt: float = DT) -> float:
    """Max angle (rad) by which integrating omega fails to reproduce R(t+dt).

    The exponential map needs the ANGLE as |omega|*dt, not dt - passing dt directly
    applies a rotation of dt radians regardless of the rate, which silently inflates the
    error for fast manoeuvres and was the original cause of this check failing.

    `dt` is the integration step, and it has to be chosen against the ANGULAR ACCELERATION
    of the reference, not just its rate: the check linearises the attitude over one step,
    so a manoeuvre that swings its thrust direction hard (v8, and chains that contain one)
    accumulates O(dt^2 * |domega/dt|) error at the same threshold the smooth families pass
    at dt = 10 ms. The claim being tested - "omega is the derivative of R" - is tested at
    2 ms for those families instead, and section H/I additionally compares the production
    stencil against a 10x finer one, which is the direct test of the rate itself.
    """
    worst = 0.0
    for t in np.arange(0.0, max(0.0, traj.duration - dt), dt):
        r0 = traj.sample(t)
        r1 = traj.sample(t + dt)
        w = float(np.linalg.norm(r0.omega))
        if w > 1e-9:
            R_pred = r0.R @ axis_angle_rotation(r0.omega, w * dt)
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
for name, tr, dt in [
    ("flip", traj, DT),
    ("hover", Trajectory(Hover([0.0, 0.0, 1.2], 0.3, 2.0)), DT),
    ("waypoints", Trajectory(WaypointTrajectory([[0, 0, 1.2], [0.5, 0.3, 1.4], [-0.4, 0.2, 1.1]], 1.0)), DT),
    ("figure8", Trajectory(FigureEight(0.3, 0.3, 1.2, 1.3)), DT),
    # The acrobatic families turn the attitude faster than the linearisation can follow at
    # 10 ms, so their rate consistency is integrated at 2 ms (see omega_consistency).
    ("v8", Trajectory(VerticalEight([0.0, 0.0, 1.2], 0.32, 0.22, 2.6)), 0.002),
]:
    err = omega_consistency(tr, dt)
    check(f"{name}: integrating omega reproduces R", err < 5e-3, f"max attitude error {err * 1e3:.2f} mrad")
print()
print("=" * 78)
print("E. every other manoeuvre is feasible and consistent")
print("=" * 78)
for name, tr in [
    ("hover", Trajectory(Hover([0.0, 0.0, 1.2], 0.3, 2.0))),
    ("waypoints", Trajectory(WaypointTrajectory([[0, 0, 1.2], [0.5, 0.3, 1.4], [-0.4, 0.2, 1.1]], 1.0))),
    ("figure8", Trajectory(FigureEight(0.3, 0.3, 1.2, 1.3))),
    ("v8", Trajectory(VerticalEight([0.0, 0.0, 1.2], 0.32, 0.22, 2.6))),
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
max_xy_seen = 0.0
N = 300
for _ in range(N):
    tr = sampler.sample(rng, mass=MASS_NOMINAL)
    kinds[tr.maneuver.kind] = kinds.get(tr.maneuver.kind, 0) + 1
    ts = np.linspace(0.0, tr.duration, 200)
    refs = [tr.sample(float(t)) for t in ts]
    thr = max(r.thrust_ff for r in refs)
    rate = max(float(np.linalg.norm(r.omega)) for r in refs)
    xy = max(float(np.max(np.abs(r.p[:2]))) for r in refs)
    max_thr_seen = max(max_thr_seen, thr)
    max_rate_seen = max(max_rate_seen, rate)
    max_xy_seen = max(max_xy_seen, xy)
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
check(
    "every path fits the 1.5 m x 1.5 m training footprint",
    max_xy_seen <= cfg.bounds_xy + 1e-9,
    f"max |x|,|y| = {max_xy_seen:.3f} m (limit {cfg.bounds_xy:.2f} m)",
)

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
print("H. the vertical figure-eight is an acrobatic AND is flyable")
print("=" * 78)
print("   v8 is a lemniscate in a VERTICAL plane: the 8's two leaves stack in altitude.")
print("   The claims to verify are that it really is an 8, that it really is acrobatic,")
print("   and that it is still exactly flyable - flatness, thrust, rate, altitude.")
v8 = VerticalEight([0.0, 0.0, 1.2], z_amp=0.32, u_amp=0.22, w=2.6, heading=0.0)
v8_traj = Trajectory(v8)
r0, r_mid, rT = (v8_traj.sample(x) for x in (0.0, v8.duration * 0.5, v8.duration))
check("v8: the path crosses itself at the halfway point",
      float(np.linalg.norm(r_mid.p - r0.p)) < 1e-9,
      f"|p(T/2) - p(0)| = {float(np.linalg.norm(r_mid.p - r0.p)):.2e} m")
ys = [float(v8_traj.sample(float(t)).p[1]) for t in np.linspace(0.0, v8.duration, 40)]
check("v8: the 8 lies in its vertical plane", max(abs(y) for y in ys) < 1e-12,
      f"max |y| = {max(abs(y) for y in ys):.2e} m (heading 0 -> x-z plane)")
zs = [float(v8_traj.sample(float(t)).p[2]) for t in np.linspace(0.0, v8.duration, 200)]
check("v8: climbs over one leaf and dives under the other",
      max(zs) > 1.2 + 0.15 and min(zs) < 1.2 - 0.15,
      f"z in [{min(zs):.2f}, {max(zs):.2f}] m around 1.20")

peak_tilt = peak_rate = wf = 0.0
t_lo, t_hi = 9e9, -9e9
for t in np.arange(0.0, v8_traj.duration, DT):
    r = v8_traj.sample(float(t))
    wf = max(wf, flatness_error(r))
    peak_tilt = max(peak_tilt, np.degrees(np.arccos(np.clip(float(r.R[2, 2]), -1.0, 1.0))))
    peak_rate = max(peak_rate, float(np.linalg.norm(r.omega)))
    t_lo = min(t_lo, float(r.thrust_ff))
    t_hi = max(t_hi, float(r.thrust_ff))
check("v8: flatness relation holds", wf < 1e-9, f"max residual {wf:.3e} N")
check("v8: it IS an acrobatic (attitude swings hard)", peak_tilt > 35.0,
      f"peak tilt {peak_tilt:.0f} deg")
check("v8: it IS an acrobatic (body rate beyond the smooth families)", peak_rate > 4.0,
      f"peak |omega| {peak_rate:.1f} rad/s")
check("v8: thrust stays inside [0, Tmax]", 0.0 <= t_lo and t_hi <= MAX_THRUST_TOTAL,
      f"T in [{t_lo:.3f}, {t_hi:.3f}] N")
check("v8: rate stays inside the policy's action scale", peak_rate <= 20.0,
      f"peak {peak_rate:.1f} rad/s (limit 20)")
check("v8: starts and ends parked",
      float(np.linalg.norm(r0.v)) < 1e-6 and float(np.linalg.norm(rT.v)) < 1e-6
      and float(np.linalg.norm(r0.a)) < 1e-6 and float(np.linalg.norm(rT.a)) < 1e-6,
      f"|v(0)| {float(np.linalg.norm(r0.v)):.1e}, |a(T)| {float(np.linalg.norm(rT.a)):.1e}")
check("v8: the flatness attitude never approaches weightlessness",
      all(float(v8_traj.sample(float(t)).a[2]) + GRAVITY > 1.0
          for t in np.linspace(0.0, v8_traj.duration, 400)),
      f"min (a_z + g) = {min(float(v8_traj.sample(float(t)).a[2]) + GRAVITY for t in np.linspace(0.0, v8_traj.duration, 400)):.2f} m/s^2")

# The sampled band must stay inside the same envelope, measured over many draws.
rng_v8 = np.random.default_rng(4)
worst_rate = worst_thrust = worst_tilt = 0.0
for _ in range(40):
    tr_v8 = Trajectory(sampler._make_v8(rng_v8, np.array([0.0, 0.0, 1.2]), 0.0))
    for t in np.linspace(0.0, tr_v8.duration, 150):
        r = tr_v8.sample(float(t))
        worst_rate = max(worst_rate, float(np.linalg.norm(r.omega)))
        worst_thrust = max(worst_thrust, float(r.thrust_ff))
        worst_tilt = max(worst_tilt, np.degrees(np.arccos(np.clip(float(r.R[2, 2]), -1.0, 1.0))))
check("v8: sampled band stays inside the rate and thrust envelope",
      worst_rate <= 0.95 * 20.0 and worst_thrust <= 0.95 * MAX_THRUST_TOTAL,
      f"worst of 40 draws: {worst_rate:.1f} rad/s, {worst_thrust:.3f} N, tilt {worst_tilt:.0f} deg")

print()
print("=" * 78)
print("I. a CHAIN is continuous, feasible, and reports per-segment kinds")
print("=" * 78)
print("   A chain links manoeuvres into one longer command. The junctions must not step")
print("   position, velocity, attitude or body rate; the relocation must preserve feasibility;")
print("   the whole thing must fit the volume/footprint/episode; and the reward must still be")
print("   able to look up the tolerance of the manoeuvre being flown.")
flip_for_chain = Flip([0.0, 0.0, 1.2], axis=[0.0, 1.0, 0.0], rotations=1.0, coast=0.6, max_rate=20.0)
chain_flip = Chain([
    FigureEight(0.25, 0.25, 1.7, 1.2, ease=0.7, settle=0.7),
    flip_for_chain,                                            # in the MIDDLE: 2 junctions
    VerticalEight([0.0, 0.0, 1.2], 0.30, 0.20, 2.6),           # relocated automatically
    Orbit([0.0, 0.0, 1.2], 0.3, 1.2, duration=3.2, ease=0.8, settle=0.8),
], hold=0.4)
chain_traj = Trajectory(chain_flip)
check("chain: reports per-segment kinds (not one blanket kind)",
      len({chain_traj.sample(float(t)).kind for t in np.linspace(0.0, chain_traj.duration, 300)}) >= 3,
      f"kinds seen = {sorted({chain_traj.sample(float(t)).kind for t in np.linspace(0.0, chain_traj.duration, 300)})}")

# Independent continuity check: differentiate the SAMPLED reference and compare against
# the reference's own v and a. A step at a junction would show up as a mismatch here.
h = 2e-3
worst_dv = worst_da_xy = 0.0
for t in np.arange(h, chain_traj.duration - h, 2.0 * DT):
    rm = chain_traj.sample(float(t - h))
    rp = chain_traj.sample(float(t + h))
    rc = chain_traj.sample(float(t))
    worst_dv = max(worst_dv, float(np.linalg.norm((rp.p - rm.p) / (2.0 * h) - rc.v)))
    da = (rp.v - rm.v) / (2.0 * h) - rc.a
    # Only the VERTICAL component may step (the flip's documented thrust step); a lateral
    # step would rotate the reference attitude without the reference asking for it.
    worst_da_xy = max(worst_da_xy, float(np.linalg.norm(da[:2])))
check("chain: position derivative matches the reference velocity (no v step)",
      worst_dv < 5e-3, f"max |dp/dt - v| = {worst_dv:.2e} m/s")
check("chain: no LATERAL acceleration step at any junction",
      worst_da_xy < 5e-2, f"max |da_xy/dt residual| = {worst_da_xy:.2e} m/s^2")
w_chain = omega_consistency(chain_traj, 0.002)
check("chain: integrating omega reproduces R across the junctions", w_chain < 5e-3,
      f"max attitude error {w_chain * 1e3:.2f} mrad")

# Direct test of the RATE itself: the production stencil (h = 1 ms) against a 10x finer
# one. A wrong or lagging stencil would show up here; the peak rates must agree to 1%.
rates_prod, rates_fine = [], []
for t in np.linspace(0.0, chain_traj.duration, 600):
    t = float(t)
    r = chain_traj.sample(t)
    Rm = chain_traj.maneuver.pose(t - 1e-4)[3]
    R0 = chain_traj.maneuver.pose(t)[3]
    Rp = chain_traj.maneuver.pose(t + 1e-4)[3]
    rates_prod.append(float(np.linalg.norm(r.omega)))
    rates_fine.append(float(np.linalg.norm(omega_from_dcm(Rm, R0, Rp, 1e-4))))
rates_prod = np.asarray(rates_prod); rates_fine = np.asarray(rates_fine)
check("chain: the production stencil resolves the same peak rate as a 10x finer one",
      abs(rates_prod.max() - rates_fine.max()) <= 0.01 * max(1e-9, rates_fine.max()),
      f"peak {rates_prod.max():.2f} vs {rates_fine.max():.2f} rad/s")

thr_chain = [chain_traj.sample(float(t)).thrust_ff for t in np.linspace(0.0, chain_traj.duration, 400)]
rate_chain = max(float(np.linalg.norm(chain_traj.sample(float(t)).omega))
                 for t in np.linspace(0.0, chain_traj.duration, 400))
xy_chain = max(float(np.max(np.abs(chain_traj.sample(float(t)).p[:2])))
               for t in np.linspace(0.0, chain_traj.duration, 400))
check("chain: thrust within authority", max(thr_chain) <= MAX_THRUST_TOTAL,
      f"max {max(thr_chain):.3f} N (flips step thrust by design, still inside authority)")
check("chain: rate within the policy's action scale", rate_chain <= 20.0,
      f"max {rate_chain:.1f} rad/s")
check("chain: fits the training footprint", xy_chain <= TrajectoryConfig().bounds_xy + 1e-9,
      f"max |x|,|y| = {xy_chain:.3f} m")
# The sampler's OWN chains must fit it too - they are relocated to a random station, so
# this is the guarantee that the screens, not the draw ranges, are what enforce the box.
sampled_ok = True
worst_sampled = 0.0
rng_ch = np.random.default_rng(8)
for _ in range(25):
    tr_ch = sampler.sample(rng_ch, kind="chain")
    for t in np.linspace(0.0, tr_ch.duration, 120):
        worst_sampled = max(worst_sampled,
                            float(np.max(np.abs(tr_ch.sample(float(t)).p[:2]))))
sampled_ok = worst_sampled <= TrajectoryConfig().bounds_xy + 1e-9
check("chain: 25 sampled chains all fit the footprint", sampled_ok,
      f"max |x|,|y| = {worst_sampled:.3f} m (limit {TrajectoryConfig().bounds_xy:.2f})")
check("chain: thrust steps exist where thrust steps (the documented flip exception)",
      chain_flip.thrust_steps == 2, f"{chain_flip.thrust_steps} thrust steps reported (flip has 2 junctions)")

# The contract is enforced, not assumed: a mid-motion segment must be refused.
try:
    Chain([FigureEight(0.3, 0.3, 1.2, 1.3)])
    check("chain: refuses a mid-motion segment", False, "no exception raised")
except ValueError as exc:
    check("chain: refuses a mid-motion segment (and says how to fix it)",
          "ease" in str(exc), str(exc)[:70] + "...")

# Relocation must preserve the dynamics, and must NOT rotate the body rate.
flip_ref = Trajectory(Flip([0.0, 0.0, 1.2], axis=[0.0, 1.0, 0.0], rotations=1.0, coast=0.6, max_rate=20.0))
flip_sh = Trajectory(ShiftedManeuver(
    Flip([0.0, 0.0, 1.2], axis=[0.0, 1.0, 0.0], rotations=1.0, coast=0.6, max_rate=20.0),
    np.array([0.55, -0.4, 1.35]), 1.1))
w_ref = max(float(np.linalg.norm(flip_ref.sample(float(t)).omega))
            for t in np.linspace(0.0, flip_ref.duration, 300))
w_sh = max(float(np.linalg.norm(flip_sh.sample(float(t)).omega))
           for t in np.linspace(0.0, flip_sh.duration, 300))
check("relocation preserves |omega| (so feasibility survives the move)",
      abs(w_ref - w_sh) < 1e-6, f"{w_ref:.6f} vs {w_sh:.6f} rad/s")
body_ok = True
for frac in (0.25, 0.5, 0.75):
    t = frac * flip_ref.duration
    a = flip_ref.sample(float(t)); b = flip_sh.sample(float(t))
    # A yaw-only relocation turns the BODY AXES with the vehicle, so the body-rate
    # VECTOR is unchanged - only its world-frame meaning rotated with R.
    body_ok = body_ok and float(np.linalg.norm(a.omega - b.omega)) < 1e-6
check("relocation leaves the body-frame rate vector unchanged", body_ok)

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all trajectory checks passed")
print("=" * 78)
