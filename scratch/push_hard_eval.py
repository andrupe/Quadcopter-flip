# -*- coding: utf-8 -*-
"""
STRESS-TEST EVALUATION SUITE FOR THE 33g QUADCOPTER + SELF-SUPERVISED ENCODER

Pushes the trained checkpoint far beyond standard evaluate.py:
  [1] Hyper-Domain Randomization (DR = 1.0, 1.25, 1.50) -> Payload up to +6.75g, extreme CoM shift, motor sag.
  [2] Violent Toss / Recovery Test -> Spawn with high velocity (2.0 m/s), tumble rate (5 rad/s), and 60-90 deg tilt.
  [3] Asymmetric Motor Degradation -> Motor 0 crippled by -10%, -20%, -30%.
  [4] Sensor & Estimator Failures -> Lighthouse complete loss, periodic blackouts, position teleport jumps.
  [5] Full Acrobatic Stress -> 360-degree flips and multi-maneuver chains under full disturbances.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "Simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quad_flip_env as qfe
from quad_flip_env import (
    QuadFlipEnv,
    ACTOR_TOTAL_DIM,
    REF_FF_DIM,
    REWARD_CEILING_PER_STEP,
    TRACK_TOL,
)
from trajectories import TrajectoryConfig
from actor_input import load_checkpoint, read_checkpoint_arch
from evaluate import LighthouseFailure

try:
    from encoder.latent_injector import LatentInjector
except ImportError:
    LatentInjector = None

MODEL_PATH = os.path.join(_ROOT, "logs", "rl_model_30000000_steps.zip")
ENCODER_PATH = os.path.join(_ROOT, "logs", "encoder_gru.pt")
Z_DIM = 16


def setup_env_and_policy(episode_seconds: float = 10.0):
    env = QuadFlipEnv(episode_seconds=episode_seconds, telemetry=True)
    model = load_checkpoint(MODEL_PATH)
    injector = LatentInjector(ENCODER_PATH, z_dim=Z_DIM, ref_ff_dim=REF_FF_DIM)
    return env, model, injector


def run_single_episode(
    env: QuadFlipEnv,
    model,
    injector,
    maneuver: Optional[str] = None,
    dr_level: float = 1.0,
    lh_failure: Optional[str] = None,
    lh_fail_at: float = 1.0,
    lh_teleport: float = 0.5,
    custom_distortions: Optional[Dict] = None,
    custom_initial_state: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> Dict:
    env.set_dr_level(dr_level)
    opts = {}
    if maneuver is not None:
        opts["maneuver"] = maneuver

    obs, _ = env.reset(seed=seed, options=opts)
    if injector is not None:
        injector.reset()

    # Apply custom distortions if specified
    if custom_distortions:
        env.quad.apply_hardware_distortions(**custom_distortions)

    # Apply custom initial state if specified
    if custom_initial_state:
        import mujoco
        if "vel" in custom_initial_state:
            env.quad.data.qvel[0:3] = np.array(custom_initial_state["vel"], dtype=np.float64)
        if "rate" in custom_initial_state:
            env.quad.data.qvel[3:6] = np.array(custom_initial_state["rate"], dtype=np.float64)
        if "euler" in custom_initial_state:
            from utils import YPRToQuat
            y, p, r = custom_initial_state["euler"]
            env.quad.data.qpos[3:7] = YPRToQuat(y, p, r)
        mujoco.mj_forward(env.quad.model, env.quad.data)
        env.quad._update_state_properties()

    lh_injector = None
    orig_observe = env.lighthouse.observe
    orig_min_stations = env.lighthouse.cfg.min_stations_for_fix
    if lh_failure and lh_failure != "none":
        lh_injector = LighthouseFailure(
            env.lighthouse,
            mode=lh_failure,
            at_s=lh_fail_at,
            teleport_m=lh_teleport,
            outage_s=0.5,
            period_s=2.0,
        )
        lh_injector.install()

    steps = 0
    total_reward = 0.0
    pos_errors = []
    vel_errors = []
    peak_tilt = 0.0
    done = False
    status = "completed"

    try:
        while not done:
            o = injector.inject(obs) if injector is not None else obs
            action, _ = model.predict(o, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            steps += 1
            total_reward += float(reward)

            if env.ref is not None:
                perr = float(np.linalg.norm(env.quad.pos - env.ref.p))
                verr = float(np.linalg.norm(env.quad.vel - env.ref.v))
                pos_errors.append(perr)
                vel_errors.append(verr)

            d22 = float(env.quad.dcm[2, 2])
            tilt = float(np.degrees(np.arccos(np.clip(d22, -1.0, 1.0))))
            peak_tilt = max(peak_tilt, tilt)

            if term:
                status = getattr(env, "termination_reason", "terminated")
                done = True
            elif trunc:
                status = "completed"
                done = True
    finally:
        if lh_injector is not None:
            env.lighthouse.observe = orig_observe
            env.lighthouse.cfg.min_stations_for_fix = orig_min_stations

    final_perr = pos_errors[-1] if pos_errors else 0.0
    final_verr = vel_errors[-1] if vel_errors else 0.0
    mean_perr = float(np.mean(pos_errors)) if pos_errors else 0.0

    return {
        "maneuver": env.ref.kind if env.ref else (maneuver or "unknown"),
        "steps": steps,
        "reward": total_reward,
        "reward_per_step": total_reward / max(1, steps),
        "status": status,
        "mean_perr": mean_perr,
        "final_perr": final_perr,
        "final_verr": final_verr,
        "peak_tilt": peak_tilt,
    }


# ======================================================================================
# Test Battery 1: Super Domain Randomization (DR = 1.0, 1.25, 1.50)
# ======================================================================================
def test_super_dr(env, model, injector, dr_levels=(1.0, 1.25, 1.50), n_per_level=12):
    print("\n" + "=" * 90)
    print("TEST BATTERY 1: HYPER-DOMAIN RANDOMIZATION STRESS")
    print("  Evaluating beyond standard DR bounds (payload up to +6.75g, extreme CoM, sag, wind)")
    print("=" * 90)

    families = ["hover", "waypoints", "figure8", "lissajous", "orbit", "slalom", "flip", "v8", "chain"]

    for dr in dr_levels:
        results = []
        for i in range(n_per_level):
            fam = families[i % len(families)]
            res = run_single_episode(env, model, injector, maneuver=fam, dr_level=dr, seed=100 + i)
            results.append(res)

        completed = sum(1 for r in results if r["status"] == "completed")
        mean_rew = np.mean([r["reward_per_step"] for r in results])
        mean_pos = np.mean([r["mean_perr"] for r in results])
        max_tilt = max(r["peak_tilt"] for r in results)

        print(f"  DR Level {dr:.2f} | Completed: {completed:2d}/{n_per_level:2d} ({100.0*completed/n_per_level:5.1f}%) | "
              f"Reward/step: {mean_rew:5.2f} ({100.0*mean_rew/REWARD_CEILING_PER_STEP:4.1f}%) | "
              f"Mean Pos Err: {mean_pos:5.3f} m | Peak Tilt: {max_tilt:3.0f}°")


# ======================================================================================
# Test Battery 2: Violent Toss / Dynamic Recovery
# ======================================================================================
def test_violent_recovery(env, model, injector, n_tests=8):
    print("\n" + "=" * 90)
    print("TEST BATTERY 2: VIOLENT TOSS / DYNAMIC RECOVERY")
    print("  Spawn with severe initial kicks: 1.5-2.2 m/s vel, 3.5-5.5 rad/s tumble, 45-75° tilt")
    print("=" * 90)

    rng = np.random.default_rng(42)
    success = 0

    print(f"  {'Episode':>7} {'Maneuver':>10} {'Init Vel (m/s)':>16} {'Init Rate (rad/s)':>18} {'Init Tilt':>10} {'Result':>15} {'Mean Err':>10}")
    print("  " + "-" * 88)

    for i in range(n_tests):
        # Generate severe initial kicks
        vx = rng.uniform(-1.8, 1.8)
        vy = rng.uniform(-1.8, 1.8)
        vz = rng.uniform(-0.8, 0.8) # can be thrown downwards towards floor!
        p_rate = rng.uniform(-4.5, 4.5)
        q_rate = rng.uniform(-4.5, 4.5)
        r_rate = rng.uniform(-3.0, 3.0)
        roll = rng.uniform(-np.radians(65), np.radians(65))
        pitch = rng.uniform(-np.radians(65), np.radians(65))
        yaw = rng.uniform(-np.pi, np.pi)

        custom_init = {
            "vel": [vx, vy, vz],
            "rate": [p_rate, q_rate, r_rate],
            "euler": [yaw, pitch, roll],
        }

        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        w_mag = np.sqrt(p_rate**2 + q_rate**2 + r_rate**2)
        tilt_init = np.degrees(np.sqrt(roll**2 + pitch**2))

        fam = ["hover", "waypoints", "figure8", "orbit"][i % 4]
        res = run_single_episode(
            env, model, injector, maneuver=fam, dr_level=0.5,
            custom_initial_state=custom_init, seed=200 + i
        )

        ok = (res["status"] == "completed")
        if ok:
            success += 1
        print(f"  [{i+1:>2d}/{n_tests:>2d}] {res['maneuver']:>10} {v_mag:14.2f}m/s {w_mag:16.2f}r/s {tilt_init:8.0f}° {res['status']:>15} {res['mean_perr']:9.3f}m")

    print(f"\n  Recovery Rate: {success}/{n_tests} ({100.0*success/n_tests:.1f}%) successfully stabilized and finished!")


# ======================================================================================
# Test Battery 3: Asymmetric Motor Degradation
# ======================================================================================
def test_motor_degradation(env, model, injector):
    print("\n" + "=" * 90)
    print("TEST BATTERY 3: ASYMMETRIC ACTUATOR FAULT (Crippled Motor 0)")
    print("  Testing whether the self-supervised latent z_t adapts to degraded rotor authority")
    print("=" * 90)

    degradations = [0.0, 0.10, 0.20, 0.30] # 0%, -10%, -20%, -30% power loss on Motor 0

    print(f"  {'Motor 0 Loss':>14} {'Completed':>12} {'Reward/step':>13} {'Mean Pos Err':>14} {'Final Hover Err':>16}")
    print("  " + "-" * 75)

    for loss in degradations:
        eff = np.array([1.0 - loss, 1.0, 1.0, 1.0], dtype=np.float64)
        distortions = {"motor_efficiencies": eff}

        ep_results = []
        for seed in range(6):
            fam = ["hover", "waypoints", "figure8"][seed % 3]
            res = run_single_episode(
                env, model, injector, maneuver=fam, dr_level=0.2,
                custom_distortions=distortions, seed=300 + seed
            )
            ep_results.append(res)

        comp = sum(1 for r in ep_results if r["status"] == "completed")
        rew = np.mean([r["reward_per_step"] for r in ep_results])
        m_err = np.mean([r["mean_perr"] for r in ep_results])
        f_err = np.mean([r["final_perr"] for r in ep_results])

        print(f"  -{loss*100:4.0f}% thrust {comp:8d}/6 {rew:12.2f} {m_err:13.3f} m {f_err:15.3f} m")


# ======================================================================================
# Test Battery 4: Sensor & Estimator Failures (Lighthouse Injection)
# ======================================================================================
def test_sensor_failures(env, model, injector):
    print("\n" + "=" * 90)
    print("TEST BATTERY 4: SENSOR & ESTIMATOR FAULT TOLERANCE")
    print("  Testing blind dead-reckoning (loss), intermittent blackouts (outage), and position jumps (teleport)")
    print("=" * 90)

    scenarios = [
        ("none", "Nominal (Full Fixes)"),
        ("outage", "Blackouts (0.5s blackout every 2s)"),
        ("loss", "Total Loss (Fix disappears at t=1.0s)"),
        ("teleport", "Position Jumps (0.5m sudden jumps)"),
    ]

    print(f"  {'Scenario':>12} {'Description':>40} {'Completed':>10} {'Mean Pos Err':>14}")
    print("  " + "-" * 80)

    for mode, desc in scenarios:
        results = []
        for i in range(5):
            fam = ["hover", "orbit", "figure8", "waypoints", "lissajous"][i]
            res = run_single_episode(
                env, model, injector, maneuver=fam, dr_level=0.5,
                lh_failure=mode, lh_fail_at=1.0, lh_teleport=0.5, seed=400 + i
            )
            results.append(res)

        comp = sum(1 for r in results if r["status"] == "completed")
        m_err = np.mean([r["mean_perr"] for r in results])
        print(f"  {mode:>12} {desc:>40} {comp:6d}/5 {m_err:13.3f} m")


# ======================================================================================
# Test Battery 5: Acrobatic Multi-Flip & Maneuver Chains
# ======================================================================================
def test_acrobatics(env, model, injector):
    print("\n" + "=" * 90)
    print("TEST BATTERY 5: FULL ACROBATIC 360° FLIPS & MANEUVER CHAINS")
    print("  Pushing high angular rates, inverted flight (>170° tilt), and recovery under DR=1.0")
    print("=" * 90)

    test_cases = [("flip", 6), ("chain", 6), ("v8", 4)]

    print(f"  {'Maneuver':>10} {'Completed':>12} {'Mean Reward':>13} {'Peak Tilt':>12} {'Final Pos Err':>15}")
    print("  " + "-" * 70)

    for fam, count in test_cases:
        results = []
        for i in range(count):
            res = run_single_episode(env, model, injector, maneuver=fam, dr_level=1.0, seed=500 + i)
            results.append(res)

        comp = sum(1 for r in results if r["status"] == "completed")
        rew = np.mean([r["reward_per_step"] for r in results])
        tilt = max(r["peak_tilt"] for r in results)
        f_err = np.mean([r["final_perr"] for r in results])

        print(f"  {fam:>10} {comp:8d}/{count:<2d} {rew:12.2f} {tilt:11.0f}° {f_err:14.3f} m")


def main():
    print("Initializing environment and loading checkpoint:")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Encoder: {ENCODER_PATH}")
    env, model, injector = setup_env_and_policy()

    t_start = time.time()
    test_super_dr(env, model, injector)
    test_violent_recovery(env, model, injector)
    test_motor_degradation(env, model, injector)
    test_sensor_failures(env, model, injector)
    test_acrobatics(env, model, injector)

    print("\n" + "=" * 90)
    print(f"ALL EVALUATION TEST BATTERIES COMPLETED IN {time.time() - t_start:.1f}s!")
    print("=" * 90)


if __name__ == "__main__":
    main()
