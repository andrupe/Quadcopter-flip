from __future__ import annotations

import os
import sys
import time
from typing import Optional

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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import mujoco
import mujoco.viewer

# Ensure paths are configured
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "Simulation" else _THIS_DIR
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import QuadFlipEnv
import utils


# ======================================================================================
# EVALUATION CONFIGURATION (Edit parameters directly here, then click Run in VS Code)
# ======================================================================================
MODEL_NAME: str = "quad_flip_model"  # Model name to evaluate (e.g. "quad_flip_model" or checkpoint)
EPISODE_SECONDS: float = 50.0         # Duration of each flight test (seconds)
SHOW_VIEWER: bool = True             # Launch interactive 3D MuJoCo viewer window
SHOW_PLOTS: bool = True             # Display 2D telemetry matplotlib plots after run
LOOP: bool = True                    # Loop replay continuously (set False for a single episode)
RANDOM_INITIAL_STATE: bool = True   # Set True to test policy robustness against random initial offsets
PLAYBACK_SPEED: float = 1.0         # Playback speed (0.25 = 4x slow-motion, 0.5 = 2x slow-mo, 1.0 = real-time)
DR_LEVEL: float = 1             # Domain Randomization intensity: 0.0 = nominal clean sim, 1.0 = full sim-to-real stress
# ======================================================================================


def evaluate(
    model_name: str = MODEL_NAME,
    episode_seconds: float = EPISODE_SECONDS,
    dr_level: float = DR_LEVEL,
    show_viewer: bool = SHOW_VIEWER,
    show_plots: bool = SHOW_PLOTS,
    loop: bool = LOOP,
    random_initial_state: bool = RANDOM_INITIAL_STATE,
    playback_speed: float = PLAYBACK_SPEED,
):
    """Run policy in MuJoCo with real-time 3D visualization and telemetry plotting."""
    model_path = os.path.join(_PROJECT_ROOT, f"{model_name}.zip")
    if not os.path.isfile(model_path):
        print(f"\n[Error] Model file not found at: {model_path}")
        print("Please train the model first by running train.py!\n")
        return

    print(f"\nLoading model: {model_path}")
    print(f"Evaluation DR Level: {dr_level:.2f} ({'Nominal clean sim' if dr_level == 0.0 else 'Sim-to-Real Hardened' if dr_level == 1.0 else 'Partial Randomization'})")
    model = PPO.load(model_path)
    env = QuadFlipEnv(episode_seconds=episode_seconds, random_initial_state=random_initial_state)
    env.set_dr_level(dr_level)
    obs, info = env.reset()

    # Load observation normalization statistics if available
    stats_path = os.path.join(_PROJECT_ROOT, f"{model_name}_vecnormalize.pkl")
    vec_norm = None
    if os.path.isfile(stats_path):
        print(f"Loaded VecNormalize statistics from: {stats_path}")
        dummy_vec = DummyVecEnv([lambda: env])
        vec_norm = VecNormalize.load(stats_path, dummy_vec)
        vec_norm.training = False

    viewer = None
    if show_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(env.quad.model, env.quad.data)
            viewer.cam.lookat = [0.0, 0.0, 1.2]
            viewer.cam.distance = 2.0
            print("Opened MuJoCo 3D Viewer. Press ESC or close the window to exit.\n")
        except Exception as e:
            print(f"Note: Could not launch interactive viewer window ({e}). Running headless.")

    # Telemetry storage (only populated if plotting is enabled)
    telemetry = {k: [] for k in ["t", "pos", "vel", "quat", "omega", "euler", "w_cmd", "wMotor", "thr", "tor"]} if show_plots else None

    episode_idx = 1
    total_reward = 0.0

    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break

            step_start = time.time()
            obs_input = vec_norm.normalize_obs(obs) if vec_norm else obs
            action, _ = model.predict(obs_input, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if telemetry is not None and episode_idx == 1:
                telemetry["t"].append(info["t"])
                telemetry["pos"].append(info["position"])
                telemetry["vel"].append(info["velocity"])
                telemetry["quat"].append(info["quat"])
                telemetry["omega"].append(info["omega"])
                telemetry["euler"].append(env.quad.euler.copy())
                telemetry["w_cmd"].append(info["motor_cmd"])
                telemetry["wMotor"].append(env.quad.wMotor.copy())
                telemetry["thr"].append(env.quad.thr.copy())
                telemetry["tor"].append(env.quad.tor.copy())

            if viewer is not None and viewer.is_running():
                viewer.sync()
                target_step_time = env.dt / max(0.01, playback_speed)
                sleep_time = target_step_time - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if terminated or truncated:
                if terminated:
                    if env.quad.check_ground_contact():
                        status = "Crashed (Ground Contact)"
                    elif env.quad.pos[2] > 2.5:
                        status = "Breached (Ceiling > 2.5m)"
                    elif float(np.linalg.norm(env.quad.pos[:2])) > 1.5:
                        status = "Breached (Arena XY > 1.5m)"
                    else:
                        status = "Terminated (Divergent State)"
                else:
                    status = "Completed (Stable Hover)"

                flip_deg = np.rad2deg(getattr(env, "accumulated_pitch", getattr(env, "accumulated_roll", 0.0)))
                tau_ms = env.quad.motor_tau * 1000.0
                lat = env.obs_latency
                batt = getattr(env.quad, "thrust_scale", 1.0)
                wind_spd = getattr(env.wind, "velW_max", 0.0) if env.random_wind else 0.0
                dr_str = f"tau={tau_ms:.1f}ms | lat={lat}st ({lat*10}ms) | batt={batt:.2f}x | wind={wind_spd:.2f}m/s"
                print(f"[Episode {episode_idx}] Steps: {env.steps:4d} ({info['t']:.2f}s) | Rew: {total_reward:7.1f} | Flip: {str(env.flip_completed):5s} ({flip_deg:3.0f}°) | {status}")
                print(f"             Disturbances: {dr_str}")

                if viewer is None or not loop:
                    break

                time.sleep(0.3)
                obs, info = env.reset()
                total_reward = 0.0
                episode_idx += 1
    finally:
        if viewer is not None and viewer.is_running():
            viewer.close()

    # Telemetry plotting
    if show_plots and telemetry and len(telemetry["t"]) > 0:
        print("\nDisplaying telemetry plots...")
        N = len(telemetry["t"])
        sDes = np.zeros([N, 16])
        sDes[:, 0:3] = env.target_state
        sDes[:, 9] = 1.0

        pdf_path = os.path.join(_PROJECT_ROOT, "telemetry_plots.pdf")
        utils.showFigures(
            env.quad.params,
            np.array(telemetry["t"]), np.array(telemetry["pos"]), np.array(telemetry["vel"]),
            np.array(telemetry["quat"]), np.array(telemetry["omega"]), np.array(telemetry["euler"]),
            np.array(telemetry["w_cmd"]), np.array(telemetry["wMotor"]), np.array(telemetry["thr"]), np.array(telemetry["tor"]),
            sDes, sDes,
            save_path=pdf_path,
        )


if __name__ == "__main__":
    # macOS GUI trampoline: launch_passive requires mjpython on macOS for interactive windowing
    if sys.platform == "darwin" and SHOW_VIEWER:
        is_mjpython = hasattr(mujoco.viewer, "_MJPYTHON") and mujoco.viewer._MJPYTHON is not None
        if not is_mjpython and os.environ.get("_MJP_TRAMPOLINED") != "1":
            import shutil
            mjpython_path = os.path.join(os.path.dirname(sys.executable), "mjpython")
            if not os.path.isfile(mjpython_path):
                mjpython_path = shutil.which("mjpython")
            if mjpython_path and os.path.isfile(mjpython_path):
                os.environ["_MJP_TRAMPOLINED"] = "1"
                os.execv(mjpython_path, [mjpython_path] + sys.argv)

    evaluate()
