# -*- coding: utf-8 -*-
"""
Why does `live_flight` look worse than `evaluate`?

The two programs share the policy path (`env.step`, `ActorInput.prepare`,
`model.predict`), so the difference has to be either

  (a) the ACTOR FRAME the policy is fed, or
  (b) the CONDITIONS it is flown under (spawn, reference, termination).

This separates them. Section A compares the frame for a MATCHED state and reference,
which isolates the frame-building code. Section B flies both closed-loop and compares
per-channel frame statistics, which is where an out-of-distribution input shows up.

Run:  .venv/bin/python scratch/diag_live_vs_eval.py [model]
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "Simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import QuadFlipEnv                     # noqa: E402
from quad_flip_env import (                                # noqa: E402
    O_POS, O_QUAT, O_OMEGA, O_VELXY, O_VELZ, O_PREV_ACTION,
    O_P_ERR, O_V_ERR, O_ATT_ERR, O_W_ERR,
)
from live_policy_env import LiveFlightEnv          # noqa: E402
from live_target import HumanTarget, yaw_of        # noqa: E402
from actor_input import ActorInput, load_checkpoint, read_checkpoint_arch  # noqa: E402

BLOCKS = [(O_POS, 3, "pos"), (O_QUAT, 4, "quat"), (O_OMEGA, 3, "gyro"),
          (O_VELXY, 2, "vel_xy"), (O_VELZ, 1, "vel_z"), (O_PREV_ACTION, 4, "prev_act"),
          (O_P_ERR, 3, "p_err"), (O_V_ERR, 3, "v_err"), (O_ATT_ERR, 3, "att_err"),
          (O_W_ERR, 3, "w_err")]


def make(kind, spawn, seed=0, dr=0.0):
    """kind: 'eval' -> QuadFlipEnv, 'live' -> LiveFlightEnv, both as the programs build them."""
    cls = QuadFlipEnv if kind == "eval" else LiveFlightEnv
    env = cls(telemetry=False, maneuver="hover", random_initial_state=False,
              random_wind=True, random_battery=True)
    env.set_dr_level(dr)
    env.reset(seed=seed, options={"maneuver": "hover"})
    if kind == "live":
        # Reproduce `LiveFlight._place_at_start` EXACTLY, in its own order: the human is
        # synced BEFORE the vehicle is teleported, and the reference is re-installed in
        # between. Skipping sync_to leaves the reference at the reset pose and manufactures
        # a constant p_err that the real program never has.
        spawn = np.asarray(spawn, dtype=np.float64)
        env._human = HumanTarget(env.quad.pos, yaw0=yaw_of(env.quad.dcm))
        env._human.sync_to(spawn, v=None, yaw=0.0)
        env.set_external_reference(env._human)
        env.adopt_state(spawn, QuadFlipEnv._euler_to_quat(0.0, 0.0, 0.0),
                        vel=np.zeros(3), omega=np.zeros(3))
        env.arm_ground_start(float(spawn[2]) <= env.GROUND_RELEASE_Z)
    return env


def step_live(env, action):
    """Reproduce live_flight's policy step exactly (including the human update)."""
    env._human.update(env.dt, vehicle_p=env.lighthouse.p_est)
    return env.step(action)


def frame(env):
    return np.asarray(env.get_actor_obs(), dtype=np.float64)


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "latest"
    if model_path == "latest":
        import re
        logs = os.path.join(_ROOT, "logs")
        zips = [os.path.join(logs, f) for f in os.listdir(logs) if f.endswith(".zip")]
        zips = [z for z in zips if re.search(r"(\d+)_steps", os.path.basename(z))]
        model_path = max(zips, key=lambda p: int(re.search(r"(\d+)_steps", p).group(1)))
    print(f"model: {os.path.relpath(model_path, _ROOT)}  actor {read_checkpoint_arch(model_path)[0]} dims\n")

    ai = ActorInput(read_checkpoint_arch(model_path)[0],
                    encoder_path=os.path.join(_ROOT, "logs", "encoder_gru.pth".replace(".pth", ".pt")))
    model = load_checkpoint(model_path)

    # -- A: the frame for a MATCHED state + reference ------------------------------------
    print("[A] same pose, same hover reference - is the actor frame identical?")
    e = make("eval", (0, 0, 1.2))
    l = make("live", (0, 0, 1.2))
    fe, fl = frame(e), frame(l)
    print(f"    {'block':>10} {'max|eval|':>10} {'max|live|':>10} {'max|diff|':>10}")
    for off, n, name in BLOCKS:
        a, b = fe[off:off + n], fl[off:off + n]
        print(f"    {name:>10} {np.max(np.abs(a)):>10.4f} {np.max(np.abs(b)):>10.4f} "
              f"{np.max(np.abs(a - b)):>10.4e}")
    e.close(); l.close()

    # -- B: fly both closed-loop and watch the frame ------------------------------------
    print("\n[B] flown closed-loop (policy), 3 s, per-channel frame statistics")
    for label, kind, spawn in (("evaluate  (spawn 1.2 m)", "eval", (0, 0, 1.2)),
                               ("live      (spawn 1.2 m)", "live", (0, 0, 1.2)),
                               ("live      (FLOOR start)", "live", (0, 0, 0.015))):
        env = make(kind, spawn)
        ai.reset()
        obs = env._get_stacked_obs()
        rows, term = [], None
        for _ in range(300):
            act, _ = model.predict(ai.prepare(obs), deterministic=True)
            if kind == "live":
                obs, r, term, trunc, _ = step_live(env, act)
            else:
                obs, r, term, trunc, _ = env.step(act)
            rows.append(frame(env))
            if term or trunc:
                break
        F = np.array(rows)
        z = float(env.quad.pos[2])
        print(f"\n    {label}:  {len(rows)} steps, ended {term=}")
        print(f"      {'block':>10} {'mean|max|':>10} {'max|max|':>10}")
        for off, n, name in BLOCKS:
            seg = np.abs(F[:, off:off + n])
            print(f"      {name:>10} {seg.mean():>10.4f} {seg.max():>10.4f}")
        print(f"      final z {z:.3f} m   (trained band 1.2-2.0 m)")
        env.close()

    # -- C: where is the usable altitude band? ------------------------------------------
    # z is ABSOLUTE in the actor frame (only x,y are anchored), and the sampler builds every
    # reference at SPAWN_Z = 1.2 m with the ground as a terminal. So the interesting
    # question for a live flight is not "does it work" but "below what altitude does it
    # stop working" - a pilot takes off from the floor and hovers wherever they like.
    print("\n[C] handover altitude sweep (live env, policy flying, 3 s)")
    print(f"    {'start z':>8} {'mean|p_err|':>12} {'final z':>8} {'drift':>8} "
          f"{'mean|vel|':>9} {'peak tilt':>9}  verdict")
    for z0 in (0.30, 0.50, 0.70, 0.90, 1.10, 1.20, 1.50, 2.00):
        env = make("live", (0.0, 0.0, z0))
        ai.reset()
        obs = env._get_stacked_obs()
        perr, vel, tilt = [], [], []
        for _ in range(300):
            act, _ = model.predict(ai.prepare(obs), deterministic=True)
            obs, r, term, trunc, _ = step_live(env, act)
            perr.append(float(np.linalg.norm(np.asarray(env.ref.p) - env.quad.pos)))
            vel.append(float(np.linalg.norm(env.quad.vel)))
            tilt.append(np.degrees(np.arccos(np.clip(env.quad.dcm[2, 2], -1.0, 1.0))))
            if term or trunc:
                break
        zf = float(env.quad.pos[2])
        drift = zf - z0
        mpe, mvel, mtilt = float(np.mean(perr)), float(np.mean(vel)), float(np.max(tilt))
        if mpe < 0.15:
            verdict = "HOLDS"
        elif mpe < 0.35:
            verdict = "loose"
        else:
            verdict = "DIVERGES"
        print(f"    {z0:>8.2f} {mpe:>12.3f} {zf:>8.2f} {drift:>+8.2f} "
              f"{mvel:>9.2f} {mtilt:>9.1f}  {verdict}")
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
