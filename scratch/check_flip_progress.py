# -*- coding: utf-8 -*-
"""
A/B the flip rotation-progress term (`quad_flip_env.FLIP_PROGRESS`).

The term is only correct if it satisfies THREE properties at once, and each one is a
different failure mode:

  1. INERT ELSEWHERE. For the eight non-flip families the reward must be BIT-IDENTICAL
     with the term on and off. Checked by re-running the same seeded episode both ways.
     This is the property that keeps every recorded number comparable.

  2. IT STILL PAYS A REAL FLIP. A vehicle that actually rotates with the reference must
     not be penalised. Checked with the scripted geometric controller from
     `check_env_tracking.py`, which does execute the rotation (flip ~89-90% of the
     ceiling). The `max(rotvec_err, progress_err)` composition is what buys this: while
     the vehicle tracks, `|rotvec| = |ref.spin - spin_veh|` and the max is a no-op.

  3. IT WITHDRAWS THE FREE LUNCH. A vehicle that does NOT rotate must lose the attitude
     reward it used to collect for simply waiting for the reference to come back upright.
     Checked by flying the flip with the trim action held - a genuine hover - and showing
     the per-step reward collapse in the reference's arrest window.

Run:  .venv/bin/python scratch/check_flip_progress.py
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

import quad_flip_env as qfe  # noqa: E402
from quad_flip_env import QuadFlipEnv  # noqa: E402

SEED = 0
DR = 0.0
NON_FLIP = ("hover", "waypoints", "figure8", "lissajous", "orbit", "slalom", "v8")


def load_scripted_controller():
    """Exec only the imports + function defs of check_env_tracking.py (no suite side effects)."""
    src = open(os.path.join(_ROOT, "scratch", "check_env_tracking.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns: dict = {"__name__": "check_env_tracking_extract"}
    exec(compile(ast.Module(keep, type_ignores=[]), "check_env_tracking.py", "exec"), ns)
    for name in ("np", "GRAVITY", "MASS_NOMINAL", "dcm_from_thrust_dir_and_yaw", "QuadFlipEnv"):
        if name not in ns and hasattr(qfe, name):
            ns[name] = getattr(qfe, name)
    return ns


def run_episode(family, *, progress, controller=None, seed=SEED, dr=DR, trace=False):
    """
    One seeded episode. `controller(env) -> action`, or None to hold the hover trim.

    FLIP_PROGRESS is set on the MODULE (that is how `_compute_reward` reads it), and
    restored by the caller, so the two arms of the A/B differ only in that flag.
    """
    prev = qfe.FLIP_PROGRESS
    qfe.FLIP_PROGRESS = progress
    try:
        env = QuadFlipEnv(telemetry=False, maneuver=family)
        env.set_dr_level(dr)
        env.reset(seed=seed, options={"maneuver": family})
        trim = np.asarray(env.hover_trim_action, dtype=np.float32).copy()

        total = 0.0
        steps = 0
        min_dcm22 = 1.0
        rows = []
        while True:
            action = trim if controller is None else np.asarray(controller(env), dtype=np.float32)
            _obs, r, term, trunc, _info = env.step(action)
            total += float(r)
            steps += 1
            min_dcm22 = min(min_dcm22, float(env.quad.dcm[2, 2]))
            if trace:
                ref = env.ref
                rows.append((float(env.t), float(getattr(ref, "spin", 0.0)),
                             float(env.flip_spin_veh), float(env.flip_progress_err),
                             float(r), env.flip_axis,
                             float(np.linalg.norm(env._attitude_error_rotvec(ref.R, env.quad.dcm)))))
            if term or trunc:
                break
        out = {"reward": total, "steps": steps, "rew_per_step": total / max(1, steps),
               "min_dcm22": min_dcm22, "rows": rows, "axis": env.flip_axis}
        env.close()
        return out
    finally:
        qfe.FLIP_PROGRESS = prev


def run_chain_probe(controller, hold=0.0, seed=0):
    """
    Drive a chain of [pitch flip, roll flip] and summarise each flip SEGMENT.

    A hand-built chain rather than a sampled one, so the two segments are guaranteed to
    exist and to be on DIFFERENT axes - which is what proves the latched axis re-arms.
    `hold=0` puts the two flips back-to-back (the reset must fire on the reference's spin
    going backwards, with no kind change to help it); `hold>0` inserts a Hover between
    them (the reset fires on the kind change). Both paths are exercised.

    Segments are split on the RESET SIGNAL rather than on `kind`, because with `hold=0`
    the kind is "flip" continuously across the junction and a kind-based split would
    silently merge the two into one.
    """
    from trajectories import Chain, Flip, Trajectory, MASS_NOMINAL as TRAJ_MASS

    def mk(axis):
        return Flip([0.0, 0.0, 1.4], axis=axis, rotations=1.0, coast=0.55, yaw=0.0,
                    mass=TRAJ_MASS, max_rate=20.0)

    chain = Chain([mk((0.0, 1.0, 0.0)), mk((1.0, 0.0, 0.0))], hold=hold)
    env = QuadFlipEnv(telemetry=False, maneuver="chain")
    env.set_dr_level(DR)
    env.reset(seed=seed, options={"maneuver": "chain"})
    env.traj = Trajectory(chain)              # replace the sampled chain with ours
    env.ref = env.traj.sample(env.t)
    trim = np.asarray(env.hover_trim_action, dtype=np.float32).copy()

    steps = []
    while True:
        action = trim if controller is None else np.asarray(controller(env), dtype=np.float32)
        _obs, _r, term, trunc, _info = env.step(action)
        steps.append((float(env.t), str(getattr(env.ref, "kind", "?")),
                      None if env.flip_axis is None else np.asarray(env.flip_axis).copy(),
                      float(env.flip_spin_veh), float(getattr(env.ref, "spin", 0.0))))
        if term or trunc:
            break
    env.close()

    runs, cur, prev_kind, prev_spin = [], None, None, 0.0
    for t, kind, axis, veh, refspin in steps:
        is_flip = kind == "flip"
        # The reset signal itself, mirrored from `_update_flip_progress`.
        new_seg = is_flip and (prev_kind != "flip" or refspin + 1e-9 < prev_spin)
        if new_seg and cur is not None:
            runs.append(cur)
            cur = None
        if is_flip and cur is None:
            cur = {"t0": t, "t1": t, "axis": None, "spin0": veh, "spin1": veh,
                   "spinmax": veh, "refmax": refspin}
        if is_flip and cur is not None:
            cur["t1"] = t
            if axis is not None and cur["axis"] is None:
                cur["axis"] = axis          # the axis latched FOR THIS SEGMENT
            cur["spin1"] = veh
            cur["spinmax"] = max(cur["spinmax"], veh)
            cur["refmax"] = max(cur["refmax"], refspin)
        elif not is_flip and cur is not None:
            runs.append(cur)
            cur = None
        prev_kind, prev_spin = kind, refspin
    if cur is not None:
        runs.append(cur)
    return runs, steps


def main() -> int:
    sc = load_scripted_controller()
    track_action = sc["track_action"]
    failures = []

    print("=" * 78)
    print("FLIP PROGRESS CHECK")
    print(f"  FLIP_PROGRESS = {qfe.FLIP_PROGRESS}   ceiling/step = {qfe.REWARD_CEILING_PER_STEP}")
    print("=" * 78)

    # -- 1. inert for every non-flip family ---------------------------------------------
    print("\n[1] non-flip families: reward must be BIT-IDENTICAL on and off")
    print(f"    {'family':<11} {'off':>10} {'on':>10} {'delta':>12} {'steps':>7}")
    for fam in NON_FLIP:
        off = run_episode(fam, progress=False, controller=track_action)
        on = run_episode(fam, progress=True, controller=track_action)
        d = on["reward"] - off["reward"]
        flag = "OK" if abs(d) < 1e-9 else "CHANGED"
        if abs(d) >= 1e-9:
            failures.append(f"{fam}: reward changed by {d:+.6f} with the term enabled")
        print(f"    {fam:<11} {off['reward']:>10.3f} {on['reward']:>10.3f} {d:>+12.3e} {off['steps']:>7}  {flag}")

    # -- 2. a real flip must still be paid ----------------------------------------------
    print("\n[2] scripted flip (executes the rotation): must NOT be penalised")
    off = run_episode("flip", progress=False, controller=track_action)
    on = run_episode("flip", progress=True, controller=track_action)
    d = on["reward"] - off["reward"]
    pct_off = 100.0 * off["rew_per_step"] / qfe.REWARD_CEILING_PER_STEP
    pct_on = 100.0 * on["rew_per_step"] / qfe.REWARD_CEILING_PER_STEP
    print(f"    scripted flip : off {off['reward']:9.2f} ({pct_off:5.1f}% of ceiling)"
          f"   on {on['reward']:9.2f} ({pct_on:5.1f}%)   delta {d:+8.2f} ({100*d/off['reward']:+.1f}%)")
    print(f"    min dcm22     : off {off['min_dcm22']:+.3f}   on {on['min_dcm22']:+.3f}"
          f"   (negative = it actually went inverted)")
    if off["min_dcm22"] > -0.2:
        failures.append("the scripted controller did not invert, so [2] proves nothing")
    if d < -0.10 * off["reward"]:
        failures.append(f"a real flip lost {100*d/off['reward']:.1f}% of its reward")
    print(f"    -> {'OK' if d > -0.10 * off['reward'] else 'REGRESSION'}: "
          f"a rotating vehicle keeps {100*(1+d/off['reward']):.1f}% of its reward")

    # -- 3. hovering through the flip must lose the free lunch ---------------------------
    print("\n[3] trim held through the flip (never rotates): the free lunch must be gone")
    off = run_episode("flip", progress=False)
    on = run_episode("flip", progress=True)
    d = on["reward"] - off["reward"]
    print(f"    hover-through : off {off['reward']:9.2f} ({100*off['rew_per_step']/qfe.REWARD_CEILING_PER_STEP:5.1f}% of ceiling)"
          f"   on {on['reward']:9.2f} ({100*on['rew_per_step']/qfe.REWARD_CEILING_PER_STEP:5.1f}%)"
          f"   delta {d:+.2f} ({100*d/off['reward']:+.1f}%)")
    print(f"    min dcm22     : off {off['min_dcm22']:+.3f}   on {on['min_dcm22']:+.3f}")
    if d >= 0.0:
        failures.append("hovering through the flip was not penalised at all")
    print(f"    -> {'OK' if d < 0.0 else 'FAIL'}: refusing to rotate now costs "
          f"{abs(100*d/off['reward']):.1f}% of the episode reward")

    # -- mechanism: where the reward goes ------------------------------------------------
    print("\n[3b] where the loss lands (hover-through, on). The reference rotates during the")
    print("     zero-thrust coast; the vehicle never does, so the arrest window - where the")
    print("     reference is upright again and the rotvec error is back to ~0 - is where the")
    print("     old reward paid out for nothing.")
    trace = run_episode("flip", progress=True, trace=True)
    rows = trace["rows"]
    spins = [r[1] for r in rows]
    total_spin = max(spins) if spins else 0.0
    print(f"     axis latched at body {None if trace['axis'] is None else np.round(trace['axis'], 3)}"
          f"   reference spin total {total_spin:.2f} rad ({total_spin/(2*np.pi):.2f} turns)")
    print(f"     {'t':>6} {'ref.spin':>9} {'veh.spin':>9} {'prog.err':>9} {'rotvec':>7} {'reward':>7}")
    # Sample the rotation window: from where the reference starts turning to the end.
    turning = [i for i, r in enumerate(rows) if r[1] > 1e-6]
    if turning:
        idxs = np.unique(np.linspace(turning[0], len(rows) - 1, 9).astype(int))
        for i in idxs:
            t, sp, ve, pe, rw, _ax, rv = rows[i]
            print(f"     {t:6.2f} {sp:9.3f} {ve:9.3f} {pe:9.3f} {rv:7.3f} {rw:7.3f}")
    tail = [r[4] for r in rows if r[1] > 1e-6]
    if tail:
        print(f"     reward/step while the reference is past its rotation: "
              f"mean {np.mean(tail):.3f}  (ceiling {qfe.REWARD_CEILING_PER_STEP}, "
              f"attitude term is worth {qfe.TRACK_W_ATT})")

    # -- 4. chains: fire mid-chain, and RESET per segment --------------------------------
    # `kind_at` reports the SEGMENT's kind, so the term must apply to a flip inside a
    # chain. The reset is the part that needs proving: `accumulated_pitch` is cleared only
    # in `reset()`, so a design keyed on it would leave the SECOND flip of a chain
    # permanently credited against the first one's angle. The chain below puts a PITCH
    # flip and then a ROLL flip back-to-back, which also proves the latched axis re-arms.
    print("\n[4] chain of two flips (pitch then roll): must fire in each, from 0")
    for hold in (0.6, 0.0):
        label = "hover-separated" if hold > 0.0 else "back-to-back  "
        runs, _steps = run_chain_probe(track_action, hold=hold)
        print(f"    hold {hold:.1f}s ({label}): {len(runs)} flip segment(s)")
        for i, r in enumerate(runs):
            print(f"      segment {i + 1}: t {r['t0']:5.2f}..{r['t1']:5.2f}  "
                  f"axis {np.round(r['axis'], 3) if r['axis'] is not None else None}  "
                  f"veh.spin {r['spin0']:+.3f} -> {r['spin1']:+.3f} "
                  f"(max {r['spinmax']:+.3f}, ref {r['refmax']:.3f})")
            if r["axis"] is None:
                failures.append(f"chain (hold={hold}): segment {i + 1} never latched an axis")
            if abs(r["spin0"]) > 0.05:
                failures.append(f"chain (hold={hold}): segment {i + 1} started at veh.spin "
                                f"{r['spin0']:+.3f}, not 0 - per-segment reset did not fire")
            if r["spinmax"] < 0.9 * r["refmax"]:
                failures.append(f"chain (hold={hold}): segment {i + 1} reached only "
                                f"{r['spinmax']:.2f} of {r['refmax']:.2f} rad")
        if len(runs) != 2:
            failures.append(f"chain (hold={hold}): expected 2 flip segments, saw {len(runs)}")
        ok = len(runs) == 2 and all(abs(r["spin0"]) <= 0.05 for r in runs)
        print(f"      -> {'OK' if ok else 'FAIL'}: {len(runs)} segments, each starting from 0")

    print("\n" + "=" * 78)
    if failures:
        print("FLIP PROGRESS CHECK FAILED")
        for f in failures:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print("FLIP PROGRESS CHECK PASSED - inert elsewhere, still pays a real flip, and")
    print("refusing to rotate no longer collects the attitude reward.")
    print("=" * 78)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
