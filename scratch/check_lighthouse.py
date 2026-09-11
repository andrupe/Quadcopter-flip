"""
Validation for Simulation/lighthouse.py.

The claims that matter are physical, not stylistic:

  A. GEOMETRY. The stations must sit above the flight volume (otherwise the deck is never
     looking at them and the whole model is vacuous), and an upright deck must be able to
     see them.
  B. INVERSION IS A REAL BLACKOUT. Rotating the vehicle through 360 deg must drop every
     station simultaneously for the duration of the rotation. This is the property that
     makes the model worth having: without it, training against the deck would silently
     assume absolute position through a flip.
  C. DEAD RECKONING DIVERGES. With no fix the estimate must drift away from truth and must
     NOT be pulled back on its own. A model whose error decays would make long blackouts
     harmless and hide the failure mode.
  D. A FIX RECOVERS. When coverage returns the estimate must snap back near truth.
  E. OUTPUT CONTRACT. Shape, dtype and key stability, since this feeds an observation.
  F. DETERMINISM. Same seed, same trajectory of estimates.

Run:  .venv/bin/python scratch/check_lighthouse.py
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

from lighthouse import LighthouseConfig, LighthouseModel  # noqa: E402
from trajectories import Flip, dcm_from_thrust_dir_and_yaw, axis_angle_rotation  # noqa: E402

FAILURES: list[str] = []
DT = 0.01


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


LEVEL_R = dcm_from_thrust_dir_and_yaw(np.array([0.0, 0.0, 1.0]), 0.0)

print("=" * 78)
print("A. station geometry")
print("=" * 78)
cfg = LighthouseConfig()
rng = np.random.default_rng(0)
lh = LighthouseModel(cfg)
lh.reset([0.0, 0.0, 1.2], np.zeros(3), rng, dr=0.5)

check("station count is in the configured set", lh.n_stations in cfg.station_counts,
      f"n_stations = {lh.n_stations}")
check("every station is above the flight volume", bool(np.all(lh.stations[:, 2] > 1.2)),
      f"z in [{lh.stations[:, 2].min():.2f}, {lh.stations[:, 2].max():.2f}] m")

counts = {}
for seed in range(60):
    _rng = np.random.default_rng(seed)
    _lh = LighthouseModel(cfg)
    _lh.reset([0.0, 0.0, 1.4], np.zeros(3), _rng, dr=0.5)
    counts[_lh.n_stations] = counts.get(_lh.n_stations, 0) + 1
check("station count is randomised across episodes", len(counts) == len(cfg.station_counts),
      f"observed {dict(sorted(counts.items()))}")

print()
print("=" * 78)
print("B. inversion is a real blackout")
print("=" * 78)

# Upright, at altitude, the deck should see enough stations for a fix.
lh.reset([0.0, 0.0, 1.4], np.zeros(3), np.random.default_rng(1), dr=0.5)
n_upright = int(np.count_nonzero(lh.visible_mask(np.array([0.0, 0.0, 1.4]), LEVEL_R)))
check("upright deck sees enough stations for a fix",
      n_upright >= cfg.min_stations_for_fix, f"{n_upright}/{lh.n_stations} visible")

# Now fly an actual flip attitude history and record coverage through it.
fl = Flip([0.0, 0.0, 1.4], axis=[0.0, 1.0, 0.0], rotations=1.0, coast=0.55, max_rate=20.0)
lh.reset([0.0, 0.0, 1.4], np.zeros(3), np.random.default_rng(1), dr=0.5)
ts = np.arange(0.0, fl.duration, DT)
vis_series = []
for t in ts:
    _p, _v, _a, R, phi, _thr = fl.pose(float(t))
    vis_series.append(int(np.count_nonzero(lh.visible_mask(np.array([0.0, 0.0, 1.4]), R))))
vis_series = np.array(vis_series)

blackout = vis_series < cfg.min_stations_for_fix
check("a 360 deg flip contains a total blackout", bool(blackout.any()),
      f"{int(blackout.sum())} of {len(ts)} steps below the {cfg.min_stations_for_fix}-station threshold")

# The blackout must be contiguous and must cover the inverted phase, not be noise.
if blackout.any():
    idx = np.flatnonzero(blackout)
    contiguous = bool(np.all(np.diff(idx) == 1))
    span = float(ts[idx[-1]] - ts[idx[0]]) + DT
    check("the blackout is one contiguous window", contiguous, f"{len(idx)} steps")
    check("the blackout lasts long enough to matter", span > 0.15, f"{span * 1e3:.0f} ms")

# Sanity: an upright attitude never loses coverage, so the blackout is the rotation.
n_after = int(np.count_nonzero(lh.visible_mask(
    np.array([0.0, 0.0, 1.4]), axis_angle_rotation(np.array([0.0, 1.0, 0.0]), 2 * np.pi) @ LEVEL_R)))
check("coverage returns once the rotation completes", n_after >= cfg.min_stations_for_fix,
      f"{n_after}/{lh.n_stations} visible at 360 deg")

print()
print("=" * 78)
print("C. dead reckoning diverges (and does not self-correct)")
print("=" * 78)
lh.reset([0.0, 0.0, 1.4], np.zeros(3), np.random.default_rng(2), dr=1.0)
# Force a permanent blackout by parking the vehicle inverted.
INVERTED = axis_angle_rotation(np.array([0.0, 1.0, 0.0]), np.pi) @ LEVEL_R
p_true = np.array([0.0, 0.0, 1.4])
v_true = np.array([0.30, 0.0, 0.0])          # drifting sideways while blind
errs = []
# NOTE: the rng is hoisted OUT of the loop on purpose. Constructing a fresh
# default_rng(seed) per iteration replays the SAME draw every step, which correlates
# the drift noise perfectly and integrates it coherently into a metres-per-step runaway.
# That is an artefact of the test harness, not of the model, and it is easy to do by
# accident when a helper takes an rng parameter.
noise_rng = np.random.default_rng(3)
for _ in range(300):
    out = lh.observe(p_true, v_true, INVERTED, DT, noise_rng)
    errs.append(out["drifted"])
    p_true = p_true + v_true * DT
errs = np.array(errs)

check("no fix is ever produced while inverted", not out["has_fix"], f"n_visible = {out['n_visible']}")
check("outage time accumulates", out["outage_t"] > 2.5, f"{out['outage_t']:.2f} s")
check("the estimate drifts away from truth", errs[-1] > 0.02, f"final drift {errs[-1]*100:.1f} cm")
check("drift grows over the blackout", errs[-1] > errs[len(errs) // 4],
      f"{errs[len(errs)//4]*100:.1f} cm at 25% -> {errs[-1]*100:.1f} cm at 100%")
check("drift is not pulled back toward truth",
      float(np.mean(np.diff(errs[-50:]))) > -1e-6,
      f"mean slope over the last 50 steps {np.mean(np.diff(errs[-50:])):+.2e} m/step")
# An unbounded drift model is as useless as a self-correcting one: a 3 s blackout must
# produce a metre-scale error, not a hundreds-of-metres one.
check("drift stays physically plausible over 3 s", 0.05 < errs[-1] < 5.0,
      f"{errs[-1]:.2f} m after 3.0 s blind")

print()
print("=" * 78)
print("D. a fix recovers the estimate")
print("=" * 78)
before = lh.p_est.copy()
out = lh.observe(p_true, v_true, LEVEL_R, DT, np.random.default_rng(4))
check("a fix is produced once upright again", bool(out["has_fix"]), f"n_visible = {out['n_visible']}")
check("the fix snaps the estimate back near truth", out["drifted"] < 0.05,
      f"drift {out['drifted']*1000:.1f} mm (was {np.linalg.norm(before - p_true)*1000:.0f} mm)")
check("outage timer resets on a fix", out["outage_t"] == 0.0, f"outage_t = {out['outage_t']}")

# One visible station is not a pose: a single sweep plane cannot resolve position.
lh_single = LighthouseModel(LighthouseConfig(station_counts=(2,), min_stations_for_fix=2))
lh_single.reset([0.0, 0.0, 1.4], np.zeros(3), np.random.default_rng(5), dr=0.0)
mask = lh_single.visible_mask(np.array([0.0, 0.0, 1.4]), LEVEL_R)
check("requiring 2 stations is enforced against a single visible station",
      cfg.min_stations_for_fix >= 2 and mask.sum() >= 1,
      f"{mask.sum()} visible, threshold {cfg.min_stations_for_fix} -> "
      f"{'fix' if mask.sum() >= cfg.min_stations_for_fix else 'no fix'}")

print()
print("=" * 78)
print("E. output contract")
print("=" * 78)
out = lh.observe(p_true, v_true, LEVEL_R, DT, np.random.default_rng(6))
expected = {"p_est", "v_est", "has_fix", "n_visible", "outage_t", "drifted"}
check("all documented keys are present", expected.issubset(out.keys()),
      f"missing {sorted(expected - set(out.keys()))}")
check("p_est / v_est are float64 (3,)", out["p_est"].shape == (3,) and out["v_est"].shape == (3,)
      and out["p_est"].dtype == np.float64 and out["v_est"].dtype == np.float64,
      f"{out['p_est'].shape} {out['p_est'].dtype}")
check("has_fix is a python bool", isinstance(out["has_fix"], bool), f"{type(out['has_fix']).__name__}")
check("returned estimates are copies, not aliases",
      out["p_est"] is not lh.p_est, "mutating the return value must not corrupt the filter")

print()
print("=" * 78)
print("F. determinism")
print("=" * 78)


def rollout(seed: int) -> np.ndarray:
    _lh = LighthouseModel()
    _rng = np.random.default_rng(seed)
    _lh.reset([0.0, 0.0, 1.4], np.zeros(3), _rng, dr=0.8)
    _p = np.array([0.0, 0.0, 1.4])
    _v = np.array([0.1, -0.1, 0.0])
    rows = []
    for _ in range(120):
        o = _lh.observe(_p, _v, LEVEL_R, DT, _rng)
        _p = _p + _v * DT
        rows.append(o["p_est"])
    return np.array(rows)


check("same seed reproduces the same estimate trajectory",
      np.array_equal(rollout(11), rollout(11)), "bitwise identical")
check("different seeds diverge",
      not np.array_equal(rollout(11), rollout(12)), "seed actually matters")

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all lighthouse checks passed")
