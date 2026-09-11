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
import gymnasium as gym
from gymnasium import spaces
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
from actor_input import ActorInput, load_checkpoint, read_checkpoint_arch
import utils


class _StatsSpaceEnv(gym.Env):
    """Space-only stand-in used to load VecNormalize statistics.

    `VecNormalize.load` verifies that the venv it is handed has the SAME observation
    space as the saved statistics. Those statistics were collected INSIDE the training
    wrapper chain (LatentObsWrapper -> raw env), i.e. over the raw env observation plus
    the latent, so a DummyVecEnv around the raw env is rejected with a shape mismatch and
    the statistics would be skipped. This stub carries the right width and is never
    stepped or reset - it exists only to pass that check.
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(obs_dim),), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


# ======================================================================================
# EVALUATION CONFIGURATION (Edit parameters directly here, then click Run in VS Code)
# This module is the entry point: run it directly (VS Code Run button, or
# `.venv/bin/python Simulation/evaluate.py`).
# ======================================================================================
MODEL_NAME: str = "latest"            # "latest" auto-selects newest checkpoint in logs/
EPISODE_SECONDS: float = 8.0          # Must match the training horizon (quad_flip_env.EPISODE_SECONDS):
                                      # it caps the episode AND bounds the references the sampler may draw.
MANEUVER: Optional[str] = None        # Pin one family (hover/waypoints/figure8/lissajous/orbit/slalom/flip); None = sample the mixture
NUM_EPISODES: int = 1                 # Number of test episodes to run before showing plots
SHOW_VIEWER: bool = True              # Launch interactive 3D MuJoCo viewer window
SHOW_PLOTS: bool = True               # Display 2D telemetry matplotlib plots after run
LOOP: bool = True                    # Set True to loop continuously; False to show plots after 1 episode
EVAL_ACTOR_ONLY: bool = True          # True = pass ONLY the onboard sensor observation to model.predict

# The frozen encoder AND the [o_t | z] assembly now live in Simulation/actor_input.py and
# Simulation/encoder/latent_injector.py, so train-time, eval-time and tune-time cannot
# disagree about the observation layout. This is the SAME checkpoint train.py loads: a
# 45-dim [o_t | z] policy needs it, a 29-dim policy ignores it. Read at call time, so
# validation scripts can repoint it.
ENCODER_CHECKPOINT: str = os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt")

RANDOM_INITIAL_POS: bool = True      # False = ALWAYS spawn at fixed [0.0, 0.0, 1.2] meters
RANDOM_INITIAL_VEL: bool = True       # True = randomize initial linear and angular velocities
RANDOM_INITIAL_ATT: bool = True       # True = slight random orientation tilt (roll/pitch/yaw)
RANDOM_INITIAL_STATE: bool = True     # Master flag (used for compatibility)
PLAYBACK_SPEED: float = 1           # Playback speed (0.25 = 4x slow-motion, 0.5 = 2x slow-mo, 1.0 = real-time)
DR_LEVEL: float = 1                # Domain Randomization intensity: 0.0 = nominal clean sim, 1.0 = full sim-to-real stress
# ======================================================================================


def evaluate(
    model_name: str = MODEL_NAME,
    episode_seconds: float = EPISODE_SECONDS,
    num_episodes: int = NUM_EPISODES,
    dr_level: float = DR_LEVEL,
    maneuver: Optional[str] = MANEUVER,
    show_viewer: bool = SHOW_VIEWER,
    show_plots: bool = SHOW_PLOTS,
    loop: bool = LOOP,
    eval_actor_only: bool = EVAL_ACTOR_ONLY,
    random_initial_pos: bool = RANDOM_INITIAL_POS,
    random_initial_vel: bool = RANDOM_INITIAL_VEL,
    random_initial_att: bool = RANDOM_INITIAL_ATT,
    random_initial_state: bool = RANDOM_INITIAL_STATE,
    playback_speed: float = PLAYBACK_SPEED,
):
    """Run policy in MuJoCo with real-time 3D visualization and telemetry plotting."""
    if os.path.isfile(model_name):
        model_path = os.path.abspath(model_name)
    elif os.path.isfile(os.path.join(_PROJECT_ROOT, model_name)):
        model_path = os.path.join(_PROJECT_ROOT, model_name)
    elif os.path.isfile(os.path.join(_PROJECT_ROOT, f"{model_name}.zip")):
        model_path = os.path.join(_PROJECT_ROOT, f"{model_name}.zip")
    elif os.path.isfile(f"{model_name}.zip"):
        model_path = os.path.abspath(f"{model_name}.zip")
    else:
        model_path = os.path.join(_PROJECT_ROOT, f"{model_name}.zip")

    if model_name.lower() in ("latest", "auto") or not os.path.isfile(model_path):
        logs_dir = os.path.join(_PROJECT_ROOT, "logs")
        if os.path.isdir(logs_dir):
            zips = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".zip")]
            if zips:
                import re
                def _step_key(p: str):
                    m = re.search(r"(\d+)_steps", os.path.basename(p))
                    return int(m.group(1)) if m else os.path.getmtime(p)
                zips.sort(key=_step_key, reverse=True)
                model_path = zips[0]

    if not os.path.isfile(model_path):
        print(f"\n[Error] Model file not found at: {model_path}")
        print("Please train the model first by running train.py!\n")
        return

    print(f"\nLoading model: {model_path}")
    print(f"Evaluation DR Level: {dr_level:.2f} ({'Nominal clean sim' if dr_level == 0.0 else 'Sim-to-Real Hardened' if dr_level == 1.0 else 'Partial Randomization'})")
    print(f"Initial State      : Pos={'[0.0, 0.0, 1.2] (fixed)' if not random_initial_pos else 'Randomized'} | Vel={'Randomized' if random_initial_vel else 'Zero'}")
    detected_actor_dim, _ = read_checkpoint_arch(model_path)
    if detected_actor_dim is None:
        raise SystemExit(
            f"\n[Error] Could not read the actor input width from {model_path}.\n"
            "        Expected a policy.pth containing mlp_extractor.policy_net.0.weight.\n"
        )
    print(f"Inference Mode     : {'Actor Only (' + str(detected_actor_dim) + ' dims)' if eval_actor_only else 'Full Observation Vector'}")

    # The actor input is assembled by the shared adapter, which also enforces the layout:
    # 29 dims is plain o_t, 45 is [o_t | z] with the frozen history encoder, and anything
    # else is a pre-migration checkpoint whose weights cannot be evaluated against this
    # environment (slicing them would produce a wrong result with no error).
    try:
        actor_input = ActorInput(detected_actor_dim, encoder_path=ENCODER_CHECKPOINT)
    except ValueError as exc:
        raise SystemExit(f"\n[Error] {exc}\n")

    # --- observation path -----------------------------------------------------------
    # Everything needed to hand the policy exactly the input it was trained on lives in
    # Simulation/actor_input.py: it detects the width from the checkpoint, owns the frozen
    # encoder's recurrent state, and rejects a pre-migration checkpoint instead of
    # silently slicing it wrong.
    print(f"History Encoder    : {actor_input.describe()}")

    # Rebuild the policy from the WEIGHTS (read out of policy.pth and re-applied as
    # policy_kwargs), not from the saved policy_kwargs metadata, so a checkpoint stays
    # loadable even when the stored metadata is stale or incomplete.
    model = load_checkpoint(model_path)

    # Telemetry stays ON here: the evaluation loop reads the ~35-key info dict that the
    # training fast path skips (train.py sets telemetry=False in its workers only).
    env = QuadFlipEnv(
        episode_seconds=episode_seconds,
        random_initial_state=random_initial_state,
        random_initial_pos=random_initial_pos,
        random_initial_vel=random_initial_vel,
        random_initial_att=random_initial_att,
        maneuver=maneuver,
    )
    env.set_dr_level(dr_level)
    print(f"Manoeuvre          : "
          f"{maneuver if maneuver else 'sampled from the training mixture at every reset'}")

    obs, info = env.reset()
    actor_input.reset()

    # Load observation normalization statistics if available. Candidates are keyed on the
    # checkpoint's own stem first: train.py writes <model>_vecnormalize.pkl next to the
    # final model and one per checkpoint in logs/.
    stem = os.path.splitext(os.path.basename(model_path))[0]
    stats_candidates = [
        os.path.join(os.path.dirname(model_path), f"{stem}_vecnormalize.pkl"),
        os.path.join(_PROJECT_ROOT, f"{stem}_vecnormalize.pkl"),
        os.path.join(_PROJECT_ROOT, "quad_flip_model_vecnormalize.pkl"),
    ]
    vec_norm = None
    for sp in stats_candidates:
        if os.path.isfile(sp):
            try:
                # The dummy venv only has to carry the SPACE the statistics were collected
                # on (raw env observation + latent when the encoder is attached).
                dummy_vec = DummyVecEnv([lambda: _StatsSpaceEnv(actor_input.training_obs_dim)])
                vec_norm = VecNormalize.load(sp, dummy_vec)
                vec_norm.training = False
                print(f"Loaded VecNormalize statistics from: {sp} "
                      f"(obs normalisation {'ON' if getattr(vec_norm, 'norm_obs', False) else 'off'})")
                break
            except Exception as e:
                print(f"Note: Could not load {sp}: {e}")

    if vec_norm is None:
        # Not a problem by default: train.py runs with norm_obs=False / norm_reward=False,
        # so raw observations are exactly what the policy expects. (normalize_obs() would
        # have been a no-op even if the statistics had been found.)
        print("Note: no VecNormalize statistics found for this checkpoint; running on the "
              "raw observation (correct when training used norm_obs=False).")

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
    telemetry = {k: [] for k in [
        "t", "pos", "vel", "quat", "omega", "omega_des", "throttle", "euler",
        "w_cmd", "wMotor", "thr", "tor",
        # Reference setpoints, for the desired-state panels
        "ref_pos", "ref_vel", "ref_acc", "ref_quat", "ref_omega", "ref_yaw",
    ]} if show_plots else None

    episode_idx = 1
    total_reward = 0.0
    max_tilt_deg = 0.0
    batch_results = []

    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break

            step_start = time.time()
            # Actor-only: [o_t] (29) or [o_t | z] (45), assembled by the shared adapter so
            # evaluation cannot disagree with training about the layout. Full vector: the
            # [o_t | z | aux | privileged] the vectorised training path produced.
            if eval_actor_only:
                obs_input = actor_input.prepare(obs, vec_norm)
            else:
                obs_input = actor_input.prepare_full(obs, vec_norm)

            action, _ = model.predict(obs_input, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            tilt_deg = float(np.degrees(np.arccos(np.clip(env.quad.dcm[2, 2], -1.0, 1.0))))
            max_tilt_deg = max(max_tilt_deg, tilt_deg)

            if telemetry is not None and episode_idx == 1:
                ref = env.ref
                telemetry["t"].append(info["t"])
                telemetry["pos"].append(info["position"])
                telemetry["vel"].append(info["velocity"])
                telemetry["quat"].append(info["quat"])
                telemetry["omega"].append(info["omega"])
                telemetry["omega_des"].append(info.get("omega_des", np.zeros(3)))
                telemetry["throttle"].append(info.get("throttle", 0.0))
                telemetry["euler"].append(env.quad.euler.copy())
                telemetry["w_cmd"].append(info["motor_cmd"])
                telemetry["wMotor"].append(env.quad.wMotor.copy())
                telemetry["thr"].append(env.quad.thr.copy())
                telemetry["tor"].append(env.quad.tor.copy())
                if ref is not None:
                    telemetry["ref_pos"].append(np.asarray(ref.p, dtype=np.float64).copy())
                    telemetry["ref_vel"].append(np.asarray(ref.v, dtype=np.float64).copy())
                    telemetry["ref_acc"].append(np.asarray(ref.a, dtype=np.float64).copy())
                    telemetry["ref_quat"].append(env._dcm_to_quat(ref.R))
                    telemetry["ref_omega"].append(np.asarray(ref.omega, dtype=np.float64).copy())
                    telemetry["ref_yaw"].append(float(np.arctan2(ref.R[1, 0], ref.R[0, 0])))

            if viewer is not None and viewer.is_running():
                viewer.sync()
                target_step_time = env.dt / max(0.01, playback_speed)
                sleep_time = target_step_time - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if terminated or truncated:
                if terminated:
                    # These are the only reasons QuadFlipEnv can return (see
                    # _check_termination); anything else is reported verbatim rather than
                    # being mapped onto a stale label from the old two-phase task.
                    term_reason = info.get("termination_reason", getattr(env, "termination_reason", "none"))
                    if term_reason == "ground_crash":
                        status = "Crashed (Ground Contact)"
                    elif term_reason == "out_of_volume":
                        status = "Breached (Left the Flight Sphere)"
                    elif term_reason == "divergent_state":
                        status = "Terminated (Divergent State)"
                    else:
                        status = f"Terminated ({term_reason})"
                else:
                    # Truncation is the normal ending under tracking: the reference reaches
                    # its terminal hover (or, rarely, the step cap is hit first).
                    traj_done = env.traj is not None and env.t >= env.traj.duration
                    status = "Completed (Terminal Hover)" if traj_done else "Completed (Time Limit)"

                dist = info.get("active_disturbances", {})
                wind_spd = getattr(env.wind, "velW_max", 0.0) if env.random_wind else 0.0
                # Final tracking error against the REFERENCE actually flown (env.target_state
                # follows the reference sample; it is no longer a fixed setpoint).
                ref_pos = np.asarray(info.get("reference_position", env.target_state), dtype=np.float64)
                ref_vel = np.asarray(info.get("reference_velocity", env.quad.vel), dtype=np.float64)
                pos_err = float(np.linalg.norm(env.quad.pos - ref_pos))
                vel_err = float(np.linalg.norm(env.quad.vel - ref_vel))
                print(
                    f"[Episode {episode_idx}] {str(info.get('maneuver', 'n/a')):>9s} | "
                    f"Steps: {env.steps:4d} ({info['t']:.2f}s) | Rew: {total_reward:7.1f} | "
                    f"Final err: pos {pos_err:.2f}m vel {vel_err:.2f}m/s | "
                    f"PeakTilt: {max_tilt_deg:3.0f}° | {status}"
                )
                sp = getattr(env, "spawn_pos", env.quad.pos)
                sv = getattr(env, "spawn_vel", env.quad.vel)
                print(f"             Spawn State : pos=[{sp[0]:+.2f}, {sp[1]:+.2f}, {sp[2]:+.2f}]m | vel=[{sv[0]:+.2f}, {sv[1]:+.2f}, {sv[2]:+.2f}]m/s")
                if dist:
                    com = dist.get("com_offset", [0, 0, 0])
                    eff = dist.get("motor_efficiencies", [1, 1, 1, 1])
                    gb = dist.get("gyro_bias_rads", [0, 0, 0])
                    print(
                        f"             Disturbances: tau={dist.get('tau_up_ms', 25.0):.1f}/{dist.get('tau_down_ms', 25.0):.1f}ms "
                        f"| CoM=[{com[0]*1000:+.1f},{com[1]*1000:+.1f},{com[2]*1000:+.1f}]mm "
                        f"| payload={dist.get('payload_mass_g', 0.0):+.1f}g "
                        f"| sag={dist.get('dynamic_sag_coef', 0.0)*100:.0f}%"
                    )
                    print(
                        f"                           motors=[{eff[0]:.2f},{eff[1]:.2f},{eff[2]:.2f},{eff[3]:.2f}] "
                        f"| gyro_bias=[{gb[0]:+.3f},{gb[1]:+.3f},{gb[2]:+.3f}] "
                        f"| batt={dist.get('thrust_scale', 1.0):.2f}x "
                        f"| lat={dist.get('latency_steps', 0)}st ({dist.get('latency_steps', 0)*10}ms) "
                        f"| wind={wind_spd:.2f}m/s"
                    )
                else:
                    tau_ms = env.quad.motor_tau * 1000.0
                    lat = env.obs_latency
                    batt = getattr(env.quad, "thrust_scale", 1.0)
                    dr_str = f"tau={tau_ms:.1f}ms | lat={lat}st ({lat*10}ms) | batt={batt:.2f}x | wind={wind_spd:.2f}m/s"
                    print(f"             Disturbances: {dr_str}")

                batch_results.append({
                    "maneuver": info.get("maneuver", "n/a"),
                    "status": status,
                    "rew": total_reward,
                    "steps": env.steps,
                    "peak_tilt": max_tilt_deg,
                    "pos_err": pos_err,
                    "vel_err": vel_err,
                })

                if not loop and episode_idx >= num_episodes:
                    break

                time.sleep(0.3)
                obs, info = env.reset()
                actor_input.reset()
                total_reward = 0.0
                max_tilt_deg = 0.0
                episode_idx += 1
    finally:
        if viewer is not None and viewer.is_running():
            viewer.close()

    if not loop and len(batch_results) > 1:
        rews = np.array([r["rew"] for r in batch_results], dtype=np.float64)
        steps = np.array([r["steps"] for r in batch_results], dtype=np.float64)
        completed = sum(1 for r in batch_results if r["status"].startswith("Completed"))
        mean_pos = float(np.mean([r["pos_err"] for r in batch_results]))
        mean_vel = float(np.mean([r["vel_err"] for r in batch_results]))
        mean_tilt = float(np.mean([r["peak_tilt"] for r in batch_results]))
        print(f"\n{'='*70}")
        print(f"BATCH EVALUATION SUMMARY ({len(batch_results)} episodes | DR={dr_level:.2f}):")
        print(f"  Mean Return       : {rews.mean():.1f}  ({rews.sum() / steps.sum():.2f} reward/step)")
        print(f"  Completed         : {completed}/{len(batch_results)} finished the trajectory without a safety termination")
        print(f"  Mean Final Error  : pos = {mean_pos:.2f} m | vel = {mean_vel:.2f} m/s")
        print(f"  Mean Peak Tilt    : {mean_tilt:.0f}°")
        print(f"{'='*70}\n")

    # Telemetry plotting
    if show_plots and telemetry and len(telemetry["t"]) > 0:
        print("\nDisplaying telemetry plots...")
        N = len(telemetry["t"])
        # The setpoint panels are fed the REFERENCE the policy was asked to fly, step by
        # step, instead of a constant "target_state" (which made the desired-thrust panel a
        # flat zero and the trajectory panel look like a fixed point even during a flip).
        # Column layout expected by utils.display.makeFigures:
        #   sDes_calc: pos(0:3) vel(3:6) thrust(6:9) quat(9:13) omega(13:16)
        #   sDes_traj: pos(0:3) vel(3:6) accel(6:9) ... yaw(14)
        sDes_calc = np.zeros([N, 16])
        sDes_calc[:, 0:3] = np.array(telemetry["ref_pos"])
        sDes_calc[:, 3:6] = np.array(telemetry["ref_vel"])
        sDes_calc[:, 9:13] = np.array(telemetry["ref_quat"])
        sDes_calc[:, 13:16] = np.array(telemetry["ref_omega"])
        sDes_traj = np.zeros([N, 16])
        sDes_traj[:, 0:3] = np.array(telemetry["ref_pos"])
        sDes_traj[:, 3:6] = np.array(telemetry["ref_vel"])
        sDes_traj[:, 6:9] = np.array(telemetry["ref_acc"])
        sDes_traj[:, 14] = np.array(telemetry["ref_yaw"])

        pdf_path = os.path.join(_PROJECT_ROOT, "telemetry_plots.pdf")
        utils.showFigures(
            env.quad.params,
            np.array(telemetry["t"]), np.array(telemetry["pos"]), np.array(telemetry["vel"]),
            np.array(telemetry["quat"]), np.array(telemetry["omega"]), np.array(telemetry["euler"]),
            np.array(telemetry["w_cmd"]), np.array(telemetry["wMotor"]), np.array(telemetry["thr"]), np.array(telemetry["tor"]),
            sDes_traj, sDes_calc,
            save_path=pdf_path,
        )

        # Rate tracking diagnostic plot
        if "omega_des" in telemetry and len(telemetry["omega_des"]) > 0:
            omega_meas_arr = np.array(telemetry["omega"])
            omega_des_arr = np.array(telemetry["omega_des"])
            t_arr = np.array(telemetry["t"])
            fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
            axis_names = ["Roll Rate ωx (rad/s)", "Pitch Rate ωy (rad/s)", "Yaw Rate ωz (rad/s)"]
            for ax_idx in range(3):
                axes[ax_idx].plot(t_arr, omega_des_arr[:, ax_idx], "r--", label="Commanded (Policy)", linewidth=1.5)
                axes[ax_idx].plot(t_arr, omega_meas_arr[:, ax_idx], "b-", label="Measured (Gyro)", linewidth=1.0)
                axes[ax_idx].set_ylabel(axis_names[ax_idx])
                axes[ax_idx].grid(True, alpha=0.3)
                if ax_idx == 0:
                    axes[ax_idx].legend(loc="upper right")
            axes[-1].set_xlabel("Time (s)")
            fig.suptitle("Inner-Loop Rate PID Tracking Performance")
            fig.tight_layout()
            rate_plot_path = os.path.join(_PROJECT_ROOT, "rate_tracking_plot.png")
            fig.savefig(rate_plot_path, dpi=150)
            plt.close(fig)
            print(f"Rate tracking diagnostic plot saved to: {rate_plot_path}")


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
