"""Trace ONE 360-degree flip flown by the geometric tracker, to see how it fails.

The scripted flip controller this used to drive (collect_data.ScriptedFlipController) no
longer exists - the encoder corpus is collected with the same cascaded geometric tracker
the deployment would use, plus excitation (collect_data.TrackerExciter). This script uses
that tracker on a pinned flip reference and prints the quantities that show whether the
manoeuvre is feasible for the plant at all:

    t, z, tilt, accumulated pitch, inverted, flip completed, body pitch rate, commands

If the reference flip is infeasible (altitude budget, rate authority), it shows up here
BEFORE any PPO training: the tracker fails to hold the reference and the failure mode
(tumble, ground, out of volume) is printed when the episode ends.

Run:  .venv/bin/python scratch/debug_flip.py
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

from quad_flip_env import QuadFlipEnv  # noqa: E402
from encoder.collect_data import TrackerExciter  # noqa: E402

EPISODE_SECONDS = 6.0


def main() -> None:
    env = QuadFlipEnv(
        episode_seconds=EPISODE_SECONDS,
        obs_noise=False,
        random_wind=False,
        random_initial_state=False,
        random_battery=False,
    )
    env.set_dr_level(0.0)  # nominal plant: failures here are the reference's fault

    for trial, seed in enumerate([3, 17]):
        rng = np.random.default_rng(seed)
        env.reset(seed=seed, options={"maneuver": "flip"})
        ctrl = TrackerExciter(rng, "none")
        ctrl.reset()

        flip = env.traj.maneuver
        print("=" * 96)
        print(f"trial {trial}: kind={flip.kind} duration={env.traj.duration:.2f}s "
              f"peak_rate={getattr(flip, 'omega_peak', float('nan')):.1f} rad/s "
              f"coast={getattr(flip, 'D', float('nan')):.2f}s")
        print(f"{'t':>5}{'z':>7}{'tilt':>7}{'prog':>7}{'inv':>5}{'done':>6}"
              f"{'w_y':>8}{'a2':>7}{'a0':>7}")

        for _ in range(int(EPISODE_SECONDS / env.dt)):
            action = ctrl(env, env.dt)
            _, _, terminated, truncated, info = env.step(action)
            tilt = np.degrees(np.arccos(np.clip(env.quad.dcm[2, 2], -1.0, 1.0)))
            if env.steps % 5 == 0 or terminated or truncated:
                print(f"{env.t:>5.2f}{env.quad.pos[2]:>7.2f}{tilt:>7.0f}"
                      f"{env.accumulated_pitch:>7.2f}{int(env.has_inverted):>5}"
                      f"{int(env.flip_completed):>6}{env.quad.omega[1]:>8.1f}"
                      f"{action[2]:>7.2f}{action[0]:>7.2f}")
            if terminated or truncated:
                print(f"  -> ended at t={env.t:.2f}s reason={info['termination_reason']} "
                      f"flip_completed={env.flip_completed}")
                break
        print()


if __name__ == "__main__":
    main()
