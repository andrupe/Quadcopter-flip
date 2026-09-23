# -*- coding: utf-8 -*-
"""
Verify `evaluate.LighthouseFailure` - the evaluation-only Lighthouse / estimator failure
injection.

The flag exists to answer "what does the policy do when the estimator lies?", so the
checks are about the INJECTION being honest and coherent, not about the policy:

  A. "none" is a TRUE no-op. An env with the injector attached must produce a
     bit-identical observation sequence to one without it, or the instrument is
     perturbing the thing it is measuring.

  B. "loss" starves the estimator of fixes through the model's own path (n_visible is
     still reported truthfully; has_fix simply never becomes true), and the estimate
     then drifts off truth.

  C. "runaway" drives the estimate away at the documented accelerating rate, and the
     ACTOR FRAME MOVES WITH IT. This is the load-bearing property: `p_err` is derived
     from the same estimate inside `_compute_actor_obs`, so a corruption that moved
     `p_est` without moving `p_err` would hand the policy the disagreement as a tell.

  D. "outage" alternates between blind and fixed.

  E. "teleport" jumps by the configured distance per period.

  F. an unknown mode is refused, and two injectors cannot stack on one env.

Run:  .venv/bin/python scratch/check_lighthouse_failure.py
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "Simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quad_flip_env as qfe                      # noqa: E402
from quad_flip_env import QuadFlipEnv            # noqa: E402
from evaluate import LighthouseFailure           # noqa: E402

STEPS = 200          # 2 s at 100 Hz
DR = 0.0


def load_scripted_controller():
    """
    Exec only the imports + function defs of check_env_tracking.py (importing that module
    runs the whole suite, and copying `track_action` would let the two drift).

    WHY CLOSED LOOP MATTERS HERE: a fixed trim action is open-loop, so the vehicle drifts
    out of the flight sphere in ~2.6 s and the episode terminates on its own. That is fine
    for the outage checks but useless for the runaway, which needs
    sqrt(2*jump/a) = 3.16 s just to cross the fix gate's 5 m bar.
    """
    src = open(os.path.join(_ROOT, "scratch", "check_env_tracking.py"), encoding="utf-8").read()
    keep = [n for n in ast.parse(src).body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns: dict = {"__name__": "check_env_tracking_extract"}
    exec(compile(ast.Module(keep, type_ignores=[]), "check_env_tracking.py", "exec"), ns)
    for name in ("np", "GRAVITY", "MASS_NOMINAL", "dcm_from_thrust_dir_and_yaw", "QuadFlipEnv"):
        if name not in ns and hasattr(qfe, name):
            ns[name] = getattr(qfe, name)
    return ns["track_action"]


def build(mode="none", maneuver="hover", **kw):
    env = QuadFlipEnv(telemetry=False, maneuver=maneuver)
    env.set_dr_level(DR)
    env.reset(seed=0, options={"maneuver": maneuver})
    inj = None
    if mode is not None:
        inj = LighthouseFailure(env.lighthouse, mode=mode, **kw)
        inj.install()
    return env, inj


def run(env, inj, steps=STEPS, controller=None):
    """Step, recording what the actor was fed and what the estimate was."""
    fallback = np.asarray(env.hover_trim_action, dtype=np.float32).copy()
    rows = []
    for _ in range(steps):
        action = fallback if controller is None else \
            np.asarray(controller(env), dtype=np.float32)
        obs, _r, term, trunc, _info = env.step(action)
        rows.append({
            "obs": np.asarray(obs, dtype=np.float64).copy(),
            "roll": np.asarray(env.get_actor_obs(), dtype=np.float64).copy(),
            "p_est": env.lighthouse.p_est.copy(),
            "p_true": env.quad.pos.copy(),
            "ref_p": np.asarray(env.ref.p, dtype=np.float64).copy(),
            "outage": float(env.lighthouse.outage_t),
            "n_vis": int(env.lighthouse.n_visible),
            "rej": int(env.lighthouse.n_fix_rejected),
            "forced": int(env.lighthouse.n_fix_forced),
        })
        if term or trunc:
            break
    env.close()
    return rows


def fail(msg):
    print(f"  [FAIL] {msg}")
    return False


def ok(msg):
    print(f"  [ok  ] {msg}")
    return True


def main() -> int:
    good = True
    O_P_ERR = 17          # QuadFlipEnv actor-frame offset of p_err
    ctrl = load_scripted_controller()

    # -- A. "none" must be a true no-op ------------------------------------------------
    print("\n[A] mode='none' is a true no-op")
    e1, i1 = build(None)            # no injector at all
    r1 = run(e1, None, controller=ctrl)
    e2, i2 = build("none")          # injector attached but inert
    r2 = run(e2, i2, controller=ctrl)
    same = len(r1) == len(r2) and all(
        np.array_equal(a["obs"], b["obs"]) for a, b in zip(r1, r2))
    good &= ok("observation sequence is bit-identical") if same else \
        fail("attaching a 'none' injector changed the observation sequence")
    good &= ok(f"injector never fired (active={i2.active})") if not i2.active else \
        fail("the 'none' injector reported itself active")

    # -- B. "loss" ---------------------------------------------------------------------
    print("\n[B] mode='loss' starves the estimator without lying to it")
    e, inj = build("loss", at_s=0.5)
    rows = run(e, inj, controller=ctrl)
    before = [r for r in rows if r["outage"] < 0.01]
    after = rows[-1]
    blind = [r for r in rows if r["outage"] > 0.2]
    good &= ok(f"dead-reckoned for {after['outage']:.2f}s by the end") if after["outage"] > 1.0 else \
        fail(f"outage only reached {after['outage']:.2f}s")
    good &= ok("n_visible is still reported truthfully (not zeroed)") if after["n_vis"] > 0 else \
        fail("n_visible was zeroed - the injection is faking the geometry, not the fix")
    good &= ok("no fix was ever rejected (there was none to reject)") if after["rej"] == 0 else \
        fail(f"{after['rej']} rejections - a blind estimator should attempt no fixes")
    if blind:
        drift = float(np.linalg.norm(blind[-1]["p_est"] - blind[-1]["p_true"]))
        good &= ok(f"estimate drifted {drift:.3f} m off truth while blind") if drift > 0.01 else \
            fail(f"estimate did not drift ({drift:.4f} m) - dead reckoning is a no-op")

    # -- C. "runaway": the rate, and the COHERENCE of the actor frame -------------------
    print("\n[C] mode='runaway' drives the estimate, and the actor frame follows it")
    # ORBIT, not hover: a hover reference is only ~2.7 s, and the lie needs
    # sqrt(2*jump/a) = 3.16 s to cross the gate's 5 m bar at the hardware-matched severity
    # of 1.0. Bumping the severity to make it fit would be testing a different failure.
    e, inj = build("runaway", maneuver="orbit", at_s=0.0, severity=1.0)
    gate_m = float(e.lighthouse.cfg.max_fix_jump)
    rows = run(e, inj, steps=490, controller=ctrl)
    lie = np.array([np.linalg.norm(r["p_est"] - r["p_true"]) for r in rows])
    t_end = len(rows) / 100.0
    predicted = 0.5 * LighthouseFailure.RUNAWAY_ACCEL * t_end ** 2
    good &= ok(f"peak lie {lie.max():.3f} m vs 0.5*a*t^2 = {predicted:.3f} m "
               f"({100*lie.max()/max(1e-9,predicted):.1f}%)") \
        if abs(lie.max() - predicted) < 0.25 * predicted else \
        fail(f"peak lie {lie.max():.3f} m does not match the documented {predicted:.3f} m")
    good &= ok("the lie grew monotonically") if bool(np.all(np.diff(lie) > -1e-9)) else \
        fail("the lie was not monotone")

    crossed = np.argmax(lie > gate_m) / 100.0 if bool(np.any(lie > gate_m)) else None
    print(f"         (gate bar is {gate_m:.1f} m; crossed at t="
          f"{'%.2f s' % crossed if crossed is not None else 'never'})")

    # THE coherence check: p_err must equal ref.p - p_est, i.e. the error channel moved
    # with the estimate rather than disagreeing with it.
    worst = 0.0
    for r in rows:
        p_err = r["roll"][O_P_ERR:O_P_ERR + 3]
        worst = max(worst, float(np.max(np.abs(p_err - (r["ref_p"] - r["p_est"])))))
    good &= ok(f"p_err == ref.p - p_est to {worst:.2e} (coherent)") if worst < 1e-6 else \
        fail(f"p_err disagrees with ref.p - p_est by {worst:.2e} - the policy gets a tell")
    moved = float(np.max(np.abs(rows[-1]["roll"][O_P_ERR:O_P_ERR + 3]
                                - rows[0]["roll"][O_P_ERR:O_P_ERR + 3])))
    good &= ok(f"the p_err channel actually moved ({moved:.3f} m)") if moved > 0.5 else \
        fail(f"p_err barely moved ({moved:.3f} m) - the lie is not reaching the actor")
    good &= ok(f"the fix gate engaged ({rows[-1]['rej']} rejected, "
               f"{rows[-1]['forced']} forced)") if rows[-1]["rej"] > 0 else \
        fail(f"the gate never rejected despite a {lie.max():.2f} m lie "
             f"(bar {gate_m:.1f} m) - the persistent-lie path is not exercised")

    # -- D. "outage" alternates --------------------------------------------------------
    print("\n[D] mode='outage' alternates between blind and fixed")
    e, inj = build("outage", at_s=0.0, outage_s=0.5, period_s=1.0)
    rows = run(e, inj, controller=ctrl)
    blinded = sum(1 for r in rows if r["outage"] > 0.05)
    fixed = len(rows) - blinded
    good &= ok(f"{blinded} steps blind, {fixed} steps fixed") if blinded > 5 and fixed > 5 else \
        fail(f"outage did not alternate (blind={blinded}, fixed={fixed})")

    # -- E. "teleport" -----------------------------------------------------------------
    print("\n[E] mode='teleport' jumps by the configured distance")
    TP = 6.0
    e, inj = build("teleport", maneuver="orbit", at_s=0.0, period_s=1.0, teleport_m=TP)
    rows = run(e, inj, controller=ctrl)
    lie = np.array([np.linalg.norm(r["p_est"] - r["p_true"]) for r in rows])
    jumps = np.diff(lie)
    big = jumps[jumps > 1.0]
    good &= ok(f"{len(big)} jump(s) of ~{TP:.1f} m") if len(big) >= 1 and \
        abs(big[0] - TP) < 0.5 else fail(f"unexpected jumps: {np.round(big, 3)}")

    # -- F. guards ---------------------------------------------------------------------
    print("\n[F] guards")
    try:
        LighthouseFailure(None, mode="banana")
        good &= fail("an unknown mode was accepted")
    except ValueError:
        good &= ok("an unknown mode is refused")
    e, i1 = build("loss")
    try:
        LighthouseFailure(e.lighthouse, mode="loss").install()
        good &= fail("a second injector stacked onto the same env")
    except RuntimeError:
        good &= ok("a second injector on one env is refused")
    e.close()

    print("\n" + "=" * 74)
    if good:
        print("LIGHTHOUSE FAILURE CHECK PASSED")
    else:
        print("LIGHTHOUSE FAILURE CHECK FAILED")
    print("=" * 74)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
