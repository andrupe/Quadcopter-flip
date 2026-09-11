# -*- coding: utf-8 -*-
"""
High-Performance Automated Finetuning for Quadcopter Inner-Loop Rate PID Controller.

Optimizes 3-axis discrete-time Rate PID gains (Kp, Ki, Kd) using parallel Differential
Evolution and closed-loop flight simulation with Domain Randomization (DR=0.70).

Features:
- Log-scale parameter representation for balanced multi-decade gain exploration.
- 2-Tier evaluation: Microsecond step-response stability filter + multi-seed flight sim.
- Time-budgeted global search (default: 10 minutes / 600s) with graceful SIGINT handling.
- Comprehensive paired benchmark (Default vs Tuned) across 25 identical test flights.
- Diagnostic export: JSON parameter ledger + comparative visualization plots.
- Optional --apply flag to automatically update default gains in rate_pid.py.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if sys.platform == "darwin":
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        if not libc.pthread_main_np():
            import matplotlib
            matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import scipy.optimize

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure project and Simulation roots are in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "Simulation" else _THIS_DIR
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quad_flip_env
from quadFiles.quad_mujoco import QuadcopterMuJoCo
from utils.rate_pid import RatePIDController
from asymmetric_policy import AsymmetricActorCriticPolicy


# ======================================================================================
# DEFAULT CONFIGURATION
# ======================================================================================
DEFAULT_TIMEOUT_SECONDS: float = 600.0   # 10 minutes
DEFAULT_DR_LEVEL: float = 0.70          # Realistic Sim-to-Real Domain Randomization
DEFAULT_EPISODES_PER_EVAL: int = 4      # Number of diverse flight seeds per candidate
DEFAULT_BENCHMARK_SEEDS: int = 25       # Final paired benchmark flight count
DEFAULT_MAX_STEPS: int = 800            # 8.0 seconds flight at 100 Hz RL
# ======================================================================================

# Log-10 Search Bounds: [kp_rp, ki_rp, kd_rp, kp_z, ki_z, kd_z]
LOG_BOUNDS_6D = [
    (-2.82, -2.10),   # log10(kp_rp) -> 0.00150 to 0.00794 (default: 0.0025 -> -2.60)
    (-4.00, -2.60),   # log10(ki_rp) -> 0.00010 to 0.00251 (default: 0.0005 -> -3.30)
    (-5.40, -4.00),   # log10(kd_rp) -> 0.000004 to 0.000100 (default: 0.000010 -> -5.00)
    (-2.82, -2.10),   # log10(kp_z)  -> 0.00150 to 0.00794 (default: 0.0035 -> -2.46)
    (-3.70, -2.52),   # log10(ki_z)  -> 0.00020 to 0.00302 (default: 0.0010 -> -3.00)
    (-5.40, -4.00),   # log10(kd_z)  -> 0.000004 to 0.000100 (default: 0.000010 -> -5.00)
]

# Module-level worker cache for multiprocessing efficiency
_WORKER_MODEL: Optional[PPO] = None
_WORKER_VEC_NORM: Optional[VecNormalize] = None
_WORKER_ENV: Optional[quad_flip_env.QuadFlipEnv] = None
_STOP_REQUESTED: bool = False


def _sigint_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n[TunePID] Termination signal received. Wrapping up current generation gracefully...")


def vector_to_gains(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert log10 parameter vector x to physical (kp, ki, kd) arrays."""
    linear = 10.0 ** np.asarray(x, dtype=np.float64)
    if len(linear) == 6:
        kp = np.array([linear[0], linear[0], linear[3]], dtype=np.float64)
        ki = np.array([linear[1], linear[1], linear[4]], dtype=np.float64)
        kd = np.array([linear[2], linear[2], linear[5]], dtype=np.float64)
    elif len(linear) == 9:
        kp = np.array([linear[0], linear[1], linear[2]], dtype=np.float64)
        ki = np.array([linear[3], linear[4], linear[5]], dtype=np.float64)
        kd = np.array([linear[6], linear[7], linear[8]], dtype=np.float64)
    else:
        raise ValueError(f"Invalid gain vector length: {len(linear)}")
    return kp, ki, kd


def gains_to_vector_6d(kp: np.ndarray, ki: np.ndarray, kd: np.ndarray) -> np.ndarray:
    """Convert physical (kp, ki, kd) to log10 6D parameter vector."""
    kp_rp = float(0.5 * (kp[0] + kp[1]))
    ki_rp = float(0.5 * (ki[0] + ki[1]))
    kd_rp = float(0.5 * (kd[0] + kd[1]))
    return np.log10([kp_rp, ki_rp, kd_rp, kp[2], ki[2], kd[2]])


def check_fast_stability(kp: np.ndarray, ki: np.ndarray, kd: np.ndarray) -> Tuple[bool, float]:
    """
    Tier 1 Fast Stability Filter (< 2 ms).
    Simulates short 0.15s step-responses on isolated QuadcopterMuJoCo dynamics.
    Returns (is_stable, penalty).
    """
    try:
        quad = QuadcopterMuJoCo()
        quad.reset()
        pid = RatePIDController(kp=kp, ki=ki, kd=kd)
        dt = 0.01
        throttle = float(quad.params["mB"] * 9.81)

        test_axes = [(0, 6.0), (1, 20.0), (2, 4.0)]
        total_overshoot = 0.0

        for ax, rate_tgt in test_axes:
            quad.reset()
            pid.reset()
            omega_cmd = np.zeros(3)
            omega_cmd[ax] = rate_tgt
            measured = []

            for _ in range(15):  # 150 ms
                quad.update(quad.t, dt, rate_cmd=(throttle, omega_cmd), rate_pid=pid)
                val = quad.data.qvel[3 + ax]
                if not np.isfinite(val) or abs(val) > 40.0:
                    return False, 1e6
                measured.append(val)

            measured = np.array(measured)
            peak = float(np.max(np.abs(measured)))
            if peak > abs(rate_tgt) * 1.6:  # > 60% overshoot
                total_overshoot += (peak - abs(rate_tgt)) / abs(rate_tgt)

        if total_overshoot > 1.2:
            return False, 5e4 + total_overshoot * 1e4

        return True, 0.0
    except Exception:
        return False, 1e6


def init_worker_process(model_path: str, norm_path: Optional[str], dr_level: float):
    """Worker process initializer to load model and environment once per core."""
    global _WORKER_MODEL, _WORKER_VEC_NORM, _WORKER_ENV
    torch.set_num_threads(1)

    _WORKER_ENV = quad_flip_env.QuadFlipEnv(
        episode_seconds=8.0,
        random_initial_state=True,
        random_initial_pos=True,
        random_initial_vel=True,
        random_initial_att=True,
        arena_radius=2.5,
        hover_gain=1.0,
    )
    _WORKER_ENV.set_dr_level(dr_level)

    dummy = DummyVecEnv([lambda: _WORKER_ENV])
    if norm_path and os.path.isfile(norm_path):
        _WORKER_VEC_NORM = VecNormalize.load(norm_path, dummy)
        _WORKER_VEC_NORM.training = False
    else:
        _WORKER_VEC_NORM = None

    _WORKER_MODEL = PPO.load(
        model_path,
        custom_objects=dict(
            policy_class=AsymmetricActorCriticPolicy,
            actor_obs_dim=51,
        ),
        device="cpu",
    )


def evaluate_single_flight(
    env: quad_flip_env.QuadFlipEnv,
    model: PPO,
    vec_norm: Optional[VecNormalize],
    kp: np.ndarray,
    ki: np.ndarray,
    kd: np.ndarray,
    seed: int,
    max_steps: int = 800,
) -> Dict[str, Any]:
    """Execute a single flight test episode with candidate PID gains."""
    env.set_rate_pid_gains(kp=kp, ki=ki, kd=kd)
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    omega_errors = []
    hover_drifts = []
    motor_chatter = 0.0
    prev_w = env.quad.wMotor.copy()

    for step in range(max_steps):
        obs_norm = vec_norm.normalize_obs(obs)[:51] if vec_norm else obs[:51]
        action, _ = model.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if "omega_des" in info:
            omega_errors.append(np.linalg.norm(info["omega_des"] - info["omega"]))

        motor_chatter += float(np.sum((env.quad.wMotor - prev_w) ** 2))
        prev_w = env.quad.wMotor.copy()

        if env.flip_completed:
            hover_drifts.append(float(np.linalg.norm(env.quad.pos[:2] - env.target_state[:2])))

        if terminated or truncated:
            break

    crashed = terminated or (env.quad.check_ground_contact() if hasattr(env.quad, "check_ground_contact") else False)
    flip_done = bool(env.flip_completed)

    mean_track_err = float(np.mean(omega_errors)) if omega_errors else 10.0
    mean_hover = float(np.mean(hover_drifts)) if hover_drifts else 0.5
    norm_chatter = float(motor_chatter / max(1, step + 1))

    return {
        "seed": seed,
        "steps": step + 1,
        "crashed": crashed,
        "flip": flip_done,
        "reward": total_reward,
        "track_err": mean_track_err,
        "hover_drift": mean_hover,
        "chatter": norm_chatter,
    }


def worker_eval_candidate(x: np.ndarray, seeds: List[int], dr_level: float) -> float:
    """Worker evaluation function called by multiprocessing pool."""
    global _WORKER_MODEL, _WORKER_VEC_NORM, _WORKER_ENV

    kp, ki, kd = vector_to_gains(x)

    # Tier 1: Fast stability filter
    is_stable, penalty = check_fast_stability(kp, ki, kd)
    if not is_stable:
        return penalty

    # Tier 2: Flight simulations across evaluation seeds
    if _WORKER_ENV is None or _WORKER_MODEL is None:
        raise RuntimeError("Worker process was not properly initialized!")

    _WORKER_ENV.set_dr_level(dr_level)

    total_loss = 0.0
    for s in seeds:
        res = evaluate_single_flight(_WORKER_ENV, _WORKER_MODEL, _WORKER_VEC_NORM, kp, ki, kd, seed=s)
        
        crash_cost = 500.0 * float(res["crashed"])
        flip_cost = 300.0 * float(not res["flip"])
        track_cost = 25.0 * res["track_err"]
        hover_cost = 80.0 * res["hover_drift"]
        chatter_cost = 2e-6 * res["chatter"]
        rew_cost = -0.01 * res["reward"]

        ep_loss = crash_cost + flip_cost + track_cost + hover_cost + chatter_cost + rew_cost
        total_loss += ep_loss

    return float(total_loss / len(seeds))


class RatePIDOptimizer:
    """Coordinates parallel tuning, budget management, telemetry, and reporting."""

    def __init__(
        self,
        model_path: str,
        norm_path: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        dr_level: float = DEFAULT_DR_LEVEL,
        episodes_per_eval: int = DEFAULT_EPISODES_PER_EVAL,
        workers: int = 4,
        quick_mode: bool = False,
    ):
        self.model_path = model_path
        self.norm_path = norm_path
        self.timeout_seconds = min(60.0, float(timeout_seconds)) if quick_mode else float(timeout_seconds)
        self.dr_level = float(dr_level)
        self.episodes_per_eval = 2 if quick_mode else int(episodes_per_eval)
        self.workers = max(1, int(workers))
        self.quick_mode = quick_mode

        self.start_time: float = 0.0
        self.eval_count: int = 0
        self.best_loss: float = float("inf")
        self.best_x: Optional[np.ndarray] = None
        self.history_loss: List[float] = []
        self.history_time: List[float] = []

        # Default gains baseline
        default_pid = RatePIDController()
        self.default_kp, self.default_ki, self.default_kd = default_pid.get_gains()
        self.default_x = gains_to_vector_6d(self.default_kp, self.default_ki, self.default_kd)

    def _objective(self, x: np.ndarray, seeds: List[int]) -> float:
        """Sequential fallback objective if workers == 1."""
        return worker_eval_candidate(x, seeds, self.dr_level)

    def _iteration_callback(self, xk: np.ndarray, convergence: float = 0.0) -> bool:
        """DE generation callback tracking progress, printing updates, and checking budget."""
        global _STOP_REQUESTED
        elapsed = time.time() - self.start_time
        remaining = max(0.0, self.timeout_seconds - elapsed)

        current_loss = self._objective(xk, seeds=[100 + i for i in range(self.episodes_per_eval)])
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_x = xk.copy()

        self.history_loss.append(self.best_loss)
        self.history_time.append(elapsed)

        kp, ki, kd = vector_to_gains(xk)
        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"Elapsed: {elapsed:5.1f}s ({int(elapsed/60)}m{int(elapsed%60):02d}s) | "
            f"Remaining: {remaining:5.1f}s | "
            f"Best Loss: {self.best_loss:7.2f} | "
            f"Kp=[{kp[0]:.4f}, {kp[2]:.4f}] Kd=[{kd[0]:.6f}, {kd[2]:.6f}]"
        )

        if elapsed >= self.timeout_seconds or _STOP_REQUESTED:
            print("\n[TunePID] Time budget reached or stop requested. Concluding optimization.")
            return True
        return False

    def optimize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run Differential Evolution optimization within time budget."""
        print("=" * 80)
        print("QUADCOPTER INNER-LOOP RATE PID CONTROLLER FINETUNING")
        print(f"Time Budget      : {self.timeout_seconds:.0f}s ({self.timeout_seconds/60:.1f} minutes)")
        print(f"Parallel Workers : {self.workers} CPU processes")
        print(f"DR Level         : {self.dr_level:.2f}")
        print(f"Episodes / Eval  : {self.episodes_per_eval} flight seeds")
        print(f"Model Checkpoint : {self.model_path}")
        print(f"Default Gains    : Kp={self.default_kp.tolist()} | Ki={self.default_ki.tolist()} | Kd={self.default_kd.tolist()}")
        print("=" * 80 + "\n")

        self.start_time = time.time()
        signal.signal(signal.SIGINT, _sigint_handler)

        seeds = [1000 + i for i in range(self.episodes_per_eval)]

        # Always initialize main process environment and model for callbacks and benchmarking
        init_worker_process(self.model_path, self.norm_path, self.dr_level)

        # Multiprocessing pool setup
        if self.workers > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(
                processes=self.workers,
                initializer=init_worker_process,
                initargs=(self.model_path, self.norm_path, self.dr_level),
            )
            default_loss = pool.apply(worker_eval_candidate, (self.default_x, seeds, self.dr_level))
        else:
            pool = None
            default_loss = self._objective(self.default_x, seeds=seeds)

        self.best_loss = default_loss
        self.best_x = self.default_x.copy()
        print(f"Baseline Default Gains Loss: {default_loss:.2f}\n")

        popsize = 6 if self.quick_mode else 8
        maxiter = 10 if self.quick_mode else 100

        try:
            res = scipy.optimize.differential_evolution(
                func=worker_eval_candidate if pool is not None else self._objective,
                bounds=LOG_BOUNDS_6D,
                args=(seeds, self.dr_level),
                popsize=popsize,
                maxiter=maxiter,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                workers=pool.map if pool is not None else 1,
                callback=self._iteration_callback,
                updating="deferred" if pool is not None else "immediate",
                seed=42,
            )
            if res.fun < self.best_loss:
                self.best_loss = float(res.fun)
                self.best_x = res.x.copy()
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Preserving best gains found.")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        best_kp, best_ki, best_kd = vector_to_gains(self.best_x)
        total_time = time.time() - self.start_time
        print(f"\nOptimization Finished in {total_time:.1f}s ({total_time/60:.2f} mins).")
        print(f"Best Gains Found: Kp={best_kp.tolist()} | Ki={best_ki.tolist()} | Kd={best_kd.tolist()}")
        return best_kp, best_ki, best_kd


def run_paired_benchmark(
    model_path: str,
    norm_path: Optional[str],
    default_gains: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tuned_gains: Tuple[np.ndarray, np.ndarray, np.ndarray],
    num_seeds: int = DEFAULT_BENCHMARK_SEEDS,
    dr_level: float = DEFAULT_DR_LEVEL,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run head-to-head comparison between default gains and tuned gains on identical flight seeds."""
    print("\n" + "=" * 80)
    print(f"HEAD-TO-HEAD BENCHMARK: DEFAULT vs TUNED PID ({num_seeds} identical paired test flights, DR={dr_level})")
    print("=" * 80)

    init_worker_process(model_path, norm_path, dr_level)
    global _WORKER_ENV, _WORKER_MODEL, _WORKER_VEC_NORM

    def _eval_gain_set(name: str, gains: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, Any]:
        kp, ki, kd = gains
        flips = 0
        survivals = 0
        returns = []
        track_errors_roll = []
        track_errors_pitch = []
        track_errors_yaw = []
        hover_xy_drifts = []
        hover_z_drifts = []
        chatter_list = []

        for ep in range(num_seeds):
            seed = 5000 + ep
            _WORKER_ENV.set_rate_pid_gains(kp=kp, ki=ki, kd=kd)
            obs, info = _WORKER_ENV.reset(seed=seed)

            ep_rew = 0.0
            prev_w = _WORKER_ENV.quad.wMotor.copy()
            ep_chatter = 0.0
            errs_r, errs_p, errs_y = [], [], []

            for step in range(DEFAULT_MAX_STEPS):
                obs_norm = _WORKER_VEC_NORM.normalize_obs(obs)[:51] if _WORKER_VEC_NORM else obs[:51]
                action, _ = _WORKER_MODEL.predict(obs_norm, deterministic=True)
                obs, rew, term, trunc, info = _WORKER_ENV.step(action)
                ep_rew += float(rew)

                if "omega_des" in info:
                    w_des = info["omega_des"]
                    w_meas = info["omega"]
                    errs_r.append(abs(w_des[0] - w_meas[0]))
                    errs_p.append(abs(w_des[1] - w_meas[1]))
                    errs_y.append(abs(w_des[2] - w_meas[2]))

                ep_chatter += float(np.sum((_WORKER_ENV.quad.wMotor - prev_w) ** 2))
                prev_w = _WORKER_ENV.quad.wMotor.copy()

                if term or trunc:
                    break

            crashed = term or (_WORKER_ENV.quad.check_ground_contact() if hasattr(_WORKER_ENV.quad, "check_ground_contact") else False)
            flips += int(_WORKER_ENV.flip_completed)
            survivals += int(not crashed)
            returns.append(ep_rew)

            if errs_r:
                track_errors_roll.append(float(np.mean(errs_r)))
                track_errors_pitch.append(float(np.mean(errs_p)))
                track_errors_yaw.append(float(np.mean(errs_y)))

            if _WORKER_ENV.flip_completed:
                hover_xy_drifts.append(float(np.linalg.norm(_WORKER_ENV.quad.pos[:2] - _WORKER_ENV.target_state[:2])))
                hover_z_drifts.append(float(abs(_WORKER_ENV.quad.pos[2] - _WORKER_ENV.target_state[2])))
            chatter_list.append(ep_chatter / max(1, step + 1))

        return {
            "name": name,
            "flips": flips,
            "flip_rate": flips / num_seeds,
            "survivals": survivals,
            "survival_rate": survivals / num_seeds,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "roll_mae": float(np.mean(track_errors_roll)) if track_errors_roll else 0.0,
            "pitch_mae": float(np.mean(track_errors_pitch)) if track_errors_pitch else 0.0,
            "yaw_mae": float(np.mean(track_errors_yaw)) if track_errors_yaw else 0.0,
            "hover_xy": float(np.mean(hover_xy_drifts)) if hover_xy_drifts else 0.0,
            "hover_z": float(np.mean(hover_z_drifts)) if hover_z_drifts else 0.0,
            "chatter": float(np.mean(chatter_list)) if chatter_list else 0.0,
        }

    metrics_def = _eval_gain_set("Default PID", default_gains)
    metrics_tun = _eval_gain_set("Tuned PID", tuned_gains)

    print(f"\n{'Metric':<30} | {'Default PID':<18} | {'Tuned PID':<18} | {'Improvement':<12}")
    print("-" * 86)
    fmt_pct = lambda d, t: f"{(t - d) / max(1e-6, abs(d)) * 100:+.1f}%" if d != 0 else "N/A"
    fmt_inv_pct = lambda d, t: f"{(d - t) / max(1e-6, abs(d)) * 100:+.1f}%" if d != 0 else "N/A"

    rows = [
        ("Flip Success Rate", f"{metrics_def['flip_rate']*100:.1f}%", f"{metrics_tun['flip_rate']*100:.1f}%", fmt_pct(metrics_def['flip_rate'], metrics_tun['flip_rate'])),
        ("Flight Survival Rate", f"{metrics_def['survival_rate']*100:.1f}%", f"{metrics_tun['survival_rate']*100:.1f}%", fmt_pct(metrics_def['survival_rate'], metrics_tun['survival_rate'])),
        ("Mean Episode Return", f"{metrics_def['mean_return']:.1f}", f"{metrics_tun['mean_return']:.1f}", fmt_pct(metrics_def['mean_return'], metrics_tun['mean_return'])),
        ("Pitch Rate Tracking MAE", f"{metrics_def['pitch_mae']:.3f} rad/s", f"{metrics_tun['pitch_mae']:.3f} rad/s", fmt_inv_pct(metrics_def['pitch_mae'], metrics_tun['pitch_mae'])),
        ("Roll Rate Tracking MAE", f"{metrics_def['roll_mae']:.3f} rad/s", f"{metrics_tun['roll_mae']:.3f} rad/s", fmt_inv_pct(metrics_def['roll_mae'], metrics_tun['roll_mae'])),
        ("Yaw Rate Tracking MAE", f"{metrics_def['yaw_mae']:.3f} rad/s", f"{metrics_tun['yaw_mae']:.3f} rad/s", fmt_inv_pct(metrics_def['yaw_mae'], metrics_tun['yaw_mae'])),
        ("Hover XY Position Drift", f"{metrics_def['hover_xy']:.3f} m", f"{metrics_tun['hover_xy']:.3f} m", fmt_inv_pct(metrics_def['hover_xy'], metrics_tun['hover_xy'])),
        ("Hover Z Altitude Error", f"{metrics_def['hover_z']:.3f} m", f"{metrics_tun['hover_z']:.3f} m", fmt_inv_pct(metrics_def['hover_z'], metrics_tun['hover_z'])),
        ("Motor Chatter Variance", f"{metrics_def['chatter']:.1f}", f"{metrics_tun['chatter']:.1f}", fmt_inv_pct(metrics_def['chatter'], metrics_tun['chatter'])),
    ]
    for m, d, t, imp in rows:
        print(f"{m:<30} | {d:<18} | {t:<18} | {imp:<12}")
    print("-" * 86 + "\n")

    return metrics_def, metrics_tun


def plot_tuning_diagnostics(
    default_gains: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tuned_gains: Tuple[np.ndarray, np.ndarray, np.ndarray],
    history_time: List[float],
    history_loss: List[float],
    save_path: str,
):
    """Generate 4-panel diagnostic plot comparing step response, flight tracking, and convergence."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Pitch Step Response (20 rad/s)
    ax_step = axes[0, 0]
    dt = 0.001
    sim_steps = 250
    t_arr = np.arange(sim_steps) * dt

    for name, gains, col, ls in [("Default", default_gains, "red", "--"), ("Tuned", tuned_gains, "blue", "-")]:
        kp, ki, kd = gains
        quad = QuadcopterMuJoCo()
        quad.reset()
        pid = RatePIDController(kp=kp, ki=ki, kd=kd)
        throttle = float(quad.params["mB"] * 9.81)
        rates_p = []
        for _ in range(sim_steps):
            quad.update(quad.t, dt, rate_cmd=(throttle, np.array([0.0, 20.0, 0.0])), rate_pid=pid)
            rates_p.append(quad.data.qvel[4])
        ax_step.plot(t_arr, rates_p, label=name, color=col, linestyle=ls, linewidth=1.8)

    ax_step.axhline(20.0, color="black", linestyle=":", label="Setpoint (20 rad/s)")
    ax_step.set_title("Pitch Step Response (Flip Bandwidth)")
    ax_step.set_xlabel("Time (s)")
    ax_step.set_ylabel("Pitch Rate ωy (rad/s)")
    ax_step.grid(True, alpha=0.3)
    ax_step.legend(loc="lower right")

    # Panel 2: Roll Step Response (6 rad/s)
    ax_roll = axes[0, 1]
    for name, gains, col, ls in [("Default", default_gains, "red", "--"), ("Tuned", tuned_gains, "blue", "-")]:
        kp, ki, kd = gains
        quad = QuadcopterMuJoCo()
        quad.reset()
        pid = RatePIDController(kp=kp, ki=ki, kd=kd)
        throttle = float(quad.params["mB"] * 9.81)
        rates_r = []
        for _ in range(sim_steps):
            quad.update(quad.t, dt, rate_cmd=(throttle, np.array([6.0, 0.0, 0.0])), rate_pid=pid)
            rates_r.append(quad.data.qvel[3])
        ax_roll.plot(t_arr, rates_r, label=name, color=col, linestyle=ls, linewidth=1.8)

    ax_roll.axhline(6.0, color="black", linestyle=":", label="Setpoint (6 rad/s)")
    ax_roll.set_title("Roll Step Response (Attitude Stabilization)")
    ax_roll.set_xlabel("Time (s)")
    ax_roll.set_ylabel("Roll Rate ωx (rad/s)")
    ax_roll.grid(True, alpha=0.3)
    ax_roll.legend(loc="lower right")

    # Panel 3: Yaw Step Response (4 rad/s)
    ax_yaw = axes[1, 0]
    for name, gains, col, ls in [("Default", default_gains, "red", "--"), ("Tuned", tuned_gains, "blue", "-")]:
        kp, ki, kd = gains
        quad = QuadcopterMuJoCo()
        quad.reset()
        pid = RatePIDController(kp=kp, ki=ki, kd=kd)
        throttle = float(quad.params["mB"] * 9.81)
        rates_y = []
        for _ in range(sim_steps):
            quad.update(quad.t, dt, rate_cmd=(throttle, np.array([0.0, 0.0, 4.0])), rate_pid=pid)
            rates_y.append(quad.data.qvel[5])
        ax_yaw.plot(t_arr, rates_y, label=name, color=col, linestyle=ls, linewidth=1.8)

    ax_yaw.axhline(4.0, color="black", linestyle=":", label="Setpoint (4 rad/s)")
    ax_yaw.set_title("Yaw Step Response (Heading Authority)")
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.set_ylabel("Yaw Rate ωz (rad/s)")
    ax_yaw.grid(True, alpha=0.3)
    ax_yaw.legend(loc="lower right")

    # Panel 4: Optimization Loss Convergence
    ax_opt = axes[1, 1]
    if history_time and history_loss:
        ax_opt.plot(history_time, history_loss, "g-o", linewidth=2.0, markersize=4, label="Best Composite Loss")
        ax_opt.set_title("Differential Evolution Convergence Over Time")
        ax_opt.set_xlabel("Elapsed Time (s)")
        ax_opt.set_ylabel("Objective Loss")
        ax_opt.grid(True, alpha=0.3)
        ax_opt.legend(loc="upper right")
    else:
        ax_opt.text(0.5, 0.5, "Single iteration test", horizontalalignment="center", verticalalignment="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Comparative diagnostic plot saved to: {save_path}")


def apply_tuned_gains_to_file(kp: np.ndarray, ki: np.ndarray, kd: np.ndarray):
    """Update default gain values in Simulation/utils/rate_pid.py."""
    rate_pid_file = os.path.join(_SIM_DIR, "utils", "rate_pid.py")
    if not os.path.isfile(rate_pid_file):
        print(f"[Error] Could not find rate_pid.py at: {rate_pid_file}")
        return

    with open(rate_pid_file, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    kp_str = f"self.kp = np.array([{kp[0]:.6f}, {kp[1]:.6f}, {kp[2]:.6f}], dtype=np.float64)"
    ki_str = f"self.ki = np.array([{ki[0]:.6f}, {ki[1]:.6f}, {ki[2]:.6f}], dtype=np.float64)"
    kd_str = f"self.kd = np.array([{kd[0]:.6f}, {kd[1]:.6f}, {kd[2]:.6f}], dtype=np.float64)"

    content = re.sub(r"self\.kp = np\.array\(\[.*?\], dtype=np\.float64\)", kp_str, content, count=1)
    content = re.sub(r"self\.ki = np\.array\(\[.*?\], dtype=np\.float64\)", ki_str, content, count=1)
    content = re.sub(r"self\.kd = np\.array\(\[.*?\], dtype=np\.float64\)", kd_str, content, count=1)

    with open(rate_pid_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Applied] Updated default RatePIDController gains in: {rate_pid_file}")


def tune_rate_pid(
    model_name: str = "latest",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    dr_level: float = DEFAULT_DR_LEVEL,
    episodes: int = DEFAULT_EPISODES_PER_EVAL,
    benchmark_seeds: int = DEFAULT_BENCHMARK_SEEDS,
    workers: Optional[int] = None,
    quick: bool = False,
    apply_gains: bool = False,
    output_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-level entry point to finetune Rate PID controller."""
    if os.path.isfile(model_name):
        model_path = os.path.abspath(model_name)
    elif os.path.isfile(os.path.join(_PROJECT_ROOT, model_name)):
        model_path = os.path.join(_PROJECT_ROOT, model_name)
    elif os.path.isfile(os.path.join(_PROJECT_ROOT, f"{model_name}.zip")):
        model_path = os.path.join(_PROJECT_ROOT, f"{model_name}.zip")
    else:
        logs_dir = os.path.join(_PROJECT_ROOT, "logs")
        zips = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".zip")]
        import re
        def _step_key(p: str):
            m = re.search(r"(\d+)_steps", os.path.basename(p))
            return int(m.group(1)) if m else os.path.getmtime(p)
        zips.sort(key=_step_key, reverse=True)
        model_path = zips[0] if zips else os.path.join(_PROJECT_ROOT, "quad_flip_model.zip")

    base_no_ext = os.path.splitext(model_path)[0]
    norm_candidates = [
        f"{base_no_ext}_vecnormalize.pkl",
        os.path.join(_PROJECT_ROOT, "quad_flip_model_vecnormalize.pkl"),
    ]
    norm_path = next((p for p in norm_candidates if os.path.isfile(p)), None)

    if workers is None or workers <= 0:
        cpu_total = os.cpu_count() or 4
        workers = max(1, min(8, cpu_total - 2))

    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, "logs")
    os.makedirs(output_dir, exist_ok=True)

    optimizer = RatePIDOptimizer(
        model_path=model_path,
        norm_path=norm_path,
        timeout_seconds=timeout,
        dr_level=dr_level,
        episodes_per_eval=episodes,
        workers=workers,
        quick_mode=quick,
    )

    tuned_kp, tuned_ki, tuned_kd = optimizer.optimize()

    default_gains = (optimizer.default_kp, optimizer.default_ki, optimizer.default_kd)
    tuned_gains = (tuned_kp, tuned_ki, tuned_kd)

    metrics_def, metrics_tun = run_paired_benchmark(
        model_path=model_path,
        norm_path=norm_path,
        default_gains=default_gains,
        tuned_gains=tuned_gains,
        num_seeds=benchmark_seeds if not quick else 5,
        dr_level=dr_level,
    )

    json_path = os.path.join(output_dir, "tuned_rate_pid.json")
    results_dict = {
        "timestamp": datetime.datetime.now().isoformat(),
        "timeout_seconds": timeout,
        "elapsed_seconds": time.time() - optimizer.start_time,
        "dr_level": dr_level,
        "model_checkpoint": model_path,
        "default_gains": {
            "kp": optimizer.default_kp.tolist(),
            "ki": optimizer.default_ki.tolist(),
            "kd": optimizer.default_kd.tolist(),
        },
        "tuned_gains": {
            "kp": tuned_kp.tolist(),
            "ki": tuned_ki.tolist(),
            "kd": tuned_kd.tolist(),
        },
        "benchmark_metrics": {
            "default": metrics_def,
            "tuned": metrics_tun,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Tuned PID parameters and benchmark report saved to: {json_path}")

    plot_path = os.path.join(output_dir, "pid_tuning_comparison.png")
    plot_tuning_diagnostics(
        default_gains=default_gains,
        tuned_gains=tuned_gains,
        history_time=optimizer.history_time,
        history_loss=optimizer.history_loss,
        save_path=plot_path,
    )

    if apply_gains:
        apply_tuned_gains_to_file(tuned_kp, tuned_ki, tuned_kd)

    return tuned_kp, tuned_ki, tuned_kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadcopter Inner-Loop Rate PID Finetuning")
    parser.add_argument("--model", type=str, default="latest", help="Path or name of model checkpoint")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Timeout budget in seconds (default: 600s / 10m)")
    parser.add_argument("--dr", type=float, default=DEFAULT_DR_LEVEL, help="Domain randomization level [0.0, 1.0]")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES_PER_EVAL, help="Episodes per candidate evaluation")
    parser.add_argument("--benchmark-seeds", type=int, default=DEFAULT_BENCHMARK_SEEDS, help="Test seeds for final paired benchmark")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (default: auto)")
    parser.add_argument("--quick", action="store_true", help="Quick 60-second smoke test run")
    parser.add_argument("--apply", action="store_true", help="Apply best gains directly to rate_pid.py default values")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for JSON and plots")
    args = parser.parse_args()

    tune_rate_pid(
        model_name=args.model,
        timeout=args.timeout,
        dr_level=args.dr,
        episodes=args.episodes,
        benchmark_seeds=args.benchmark_seeds,
        workers=args.workers,
        quick=args.quick,
        apply_gains=args.apply,
        output_dir=args.output_dir,
    )
