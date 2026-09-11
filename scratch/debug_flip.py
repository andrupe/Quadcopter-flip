"""Trace the scripted flip controller to find why it tumbles instead of completing."""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation"), os.path.join(_PROJECT_ROOT, "Simulation", "encoder")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import QuadFlipEnv  # noqa: E402
from collect_data import ScriptedFlipController, MAX_THRUST  # noqa: E402

env = QuadFlipEnv(
    episode_seconds=6.0, obs_noise=False, random_wind=False,
    random_initial_state=False, random_battery=False,
    curriculum_arena=False, arena_radius_start=5.0, arena_radius_end=5.0,
)
env.set_dr_level(1.0)
env.set_dr_headroom(1.0)

for trial, seed in enumerate([3, 17]):
    rng = np.random.default_rng(seed)
    env.reset()
    ctrl = ScriptedFlipController()
    ctrl.reset(env, rng)
    print("=" * 96)
    print(
        f"trial {trial}: climb_time={ctrl.climb_time:.2f}s climb_thrust={ctrl.climb_thrust / (0.028 * 9.81):.2f}x hover "
        f"peak_rate={ctrl.peak_rate:.1f} cut={ctrl.cut:.2f} mass_ratio={env.active_disturbances['total_mass_g'] / 28.0:.3f}"
    )
    print(f"{'t':>5}{'z':>7}{'tilt':>7}{'prog':>7}{'inv':>5}{'done':>6}{'w_y':>8}{'a2':>7}{'a0':>7}  {'':>4}")
    for k in range(120):
        a = ctrl.act(env, rng)
        obs, r, term, trunc, info = env.step(a)
        tilt = np.degrees(np.arccos(np.clip(env.quad.dcm[2, 2], -1, 1)))
        if k % 5 == 0 or term:
            print(
                f"{env.t:>5.2f}{env.quad.pos[2]:>7.2f}{tilt:>7.0f}{env.accumulated_pitch:>7.2f}"
                f"{int(env.has_inverted):>5}{int(env.flip_completed):>6}"
                f"{env.quad.omega[1]:>8.1f}{a[2]:>7.2f}{a[0]:>7.2f}"
            )
        if term or trunc:
            print(f"  -> ended at t={env.t:.2f}s reason={info['termination_reason']}")
            break
    print()
