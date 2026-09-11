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
import torch
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

from quad_flip_env import QuadFlipEnv, ACTOR_TOTAL_DIM, TOTAL_OBS_DIM, ENCODER_AUX_DIM
from asymmetric_policy import AsymmetricActorCriticPolicy
import utils


# ======================================================================================
# EVALUATION CONFIGURATION (Edit parameters directly here, then click Run in VS Code)
# NOTE: For the newly trained multi-phase Adaptive Architecture (Phase 1 PPO + Phase 2
# Estimator), run the top-level `evaluate.py` at the project root!
# ======================================================================================
MODEL_NAME: str = "latest"                          # "latest" auto-selects newest checkpoint in logs/
EPISODE_SECONDS: float = 10.0         # Duration of each flight test (seconds)
NUM_EPISODES: int = 1                 # Number of test episodes to run before showing plots
SHOW_VIEWER: bool = True              # Launch interactive 3D MuJoCo viewer window
SHOW_PLOTS: bool = True               # Display 2D telemetry matplotlib plots after run
LOOP: bool = True                    # Set True to loop continuously; False to show plots after 1 episode
EVAL_ACTOR_ONLY: bool = True          # True = pass ONLY the onboard sensor observation to model.predict

# Frozen history encoder produced by Simulation/encoder/train_encoder.py.
ENCODER_CHECKPOINT: str = os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt")
Z_DIM: int = 16


class _LatentInjector:
    """
    Owns the frozen encoder's recurrent state and appends z to a RAW env observation.

    Deliberately NOT a VecEnv wrapper. Wrapping the env in a DummyVecEnv to reach
    LatentObsWrapper looks equivalent, but SB3's DummyVecEnv AUTO-RESETS on done - so by
    the time the evaluation loop reads the terminal telemetry (env.steps, env.quad.pos,
    env.termination_reason) the environment has already been reset and the loop is reading
    a FRESH episode's state. That made every episode report "Steps: 0" and terminate
    immediately, hundreds of times in a row, with a stale termination reason.

    Injecting z directly keeps the raw env authoritative: the loop still calls env.step(),
    every env.* read stays correct, and the encoder is driven through its incremental
    `step()` - the same path the flight controller would use - rather than a batch path
    that only ever runs in training.

    Produces the identical layout to LatentObsWrapper:
        [o_t (actor_dim) | z (z_dim) | aux | privileged]
    """

    def __init__(self, encoder_path: str, actor_dim: int, aux_dim: int, z_dim: int):
        from encoder.history_encoder import load_encoder_checkpoint
        from encoder.observation_spec import frame_from_env_obs  # noqa: F401

        encoder, norm, _ = load_encoder_checkpoint(encoder_path)
        if int(encoder.z_dim) != int(z_dim):
            raise ValueError(
                f"encoder at {encoder_path} has z_dim={encoder.z_dim}, expected {z_dim}"
            )
        self.encoder = encoder
        self.norm = norm
        self.actor_dim = int(actor_dim)
        self.aux_dim = int(aux_dim)
        self.z_dim = int(z_dim)
        self.h = encoder.init_state(1)
        self.z = np.zeros((1, self.z_dim), dtype=np.float32)

    @torch.no_grad()
    def reset(self) -> None:
        """Zero the recurrent state. Must be called whenever the env resets."""
        self.h = self.encoder.init_state(1)
        self.z = np.zeros((1, self.z_dim), dtype=np.float32)

    @torch.no_grad()
    def inject(self, env_obs: np.ndarray) -> np.ndarray:
        from encoder.observation_spec import frame_from_env_obs

        raw = frame_from_env_obs(env_obs, self.actor_dim, self.aux_dim)
        frame = self.norm.standardize_frame(raw)[None, :]
        x = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        z, self.h = self.encoder.step(x, self.h)
        self.z = z.numpy()
        return np.concatenate(
            [env_obs[: self.actor_dim], self.z[0], env_obs[self.actor_dim:]]
        ).astype(np.float32)
RANDOM_INITIAL_POS: bool = True      # False = ALWAYS spawn at fixed [0.0, 0.0, 1.2] meters
RANDOM_INITIAL_VEL: bool = True       # True = randomize initial linear and angular velocities
RANDOM_INITIAL_ATT: bool = True       # True = slight random orientation tilt (roll/pitch/yaw)
RANDOM_INITIAL_STATE: bool = True     # Master flag (used for compatibility)
PLAYBACK_SPEED: float = 1           # Playback speed (0.25 = 4x slow-motion, 0.5 = 2x slow-mo, 1.0 = real-time)
DR_LEVEL: float = 1                # Domain Randomization intensity: 0.0 = nominal clean sim, 1.0 = full sim-to-real stress
HOVER_GAIN: float = 1.0               # Hover authority scale: 1.0 = unattenuated, 0.5 = 50% calm hover authority
# ======================================================================================


def evaluate(
    model_name: str = MODEL_NAME,
    episode_seconds: float = EPISODE_SECONDS,
    num_episodes: int = NUM_EPISODES,
    dr_level: float = DR_LEVEL,
    hover_gain: float = HOVER_GAIN,
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
    print(f"Hover Gain Scale   : {hover_gain:.2f} ({'Unattenuated (100% authority)' if hover_gain >= 1.0 else f'{int((1.0 - hover_gain)*100)}% attenuated'})")
    print(f"Initial State      : Pos={'[0.0, 0.0, 1.2] (fixed)' if not random_initial_pos else 'Randomized'} | Vel={'Randomized' if random_initial_vel else 'Zero'}")
    detected_actor_dim = ACTOR_TOTAL_DIM
    detected_net_arch = None
    try:
        import zipfile
        import io
        with zipfile.ZipFile(model_path, "r") as z:
            if "policy.pth" in z.namelist():
                with z.open("policy.pth") as f:
                    sd = torch.load(io.BytesIO(f.read()), map_location="cpu")
                    if "mlp_extractor.policy_net.0.weight" in sd:
                        detected_actor_dim = int(sd["mlp_extractor.policy_net.0.weight"].shape[1])
                    pi_dims = []
                    idx = 0
                    while f"mlp_extractor.policy_net.{idx}.weight" in sd:
                        pi_dims.append(int(sd[f"mlp_extractor.policy_net.{idx}.weight"].shape[0]))
                        idx += 2
                    vf_dims = []
                    idx = 0
                    while f"mlp_extractor.value_net.{idx}.weight" in sd:
                        vf_dims.append(int(sd[f"mlp_extractor.value_net.{idx}.weight"].shape[0]))
                        idx += 2
                    if pi_dims:
                        detected_net_arch = dict(pi=pi_dims, vf=vf_dims if vf_dims else [512, 256, 128])
    except Exception:
        pass

    coord_desc = ("trajectory-tracking frame" if detected_actor_dim in (ACTOR_TOTAL_DIM, ACTOR_TOTAL_DIM + 16)
                  else "PRE-MIGRATION layout")
    print(f"Inference Mode     : {'Actor Only (' + str(detected_actor_dim) + ' dims, ' + coord_desc + ')' if eval_actor_only else f'Full Observation Vector ({TOTAL_OBS_DIM} dims)'}")

    # GUARD against silently evaluating on the wrong input.
    #
    # `detected_actor_dim` is read back out of the checkpoint's first policy layer, and the
    # actor is then fed obs[..., :detected_actor_dim]. Two cases are valid here:
    #
    #   ACTOR_TOTAL_DIM (29)      current tracking frame, no encoder -> used as-is
    #   ACTOR_TOTAL_DIM + z (45)  trained WITH the encoder -> the wrapper is attached
    #                             below, which makes obs[:45] exactly [o_t | z]
    #
    # Anything else is a checkpoint from an EARLIER observation layout (the old 51-dim
    # 3-frame stack). Its weights are meaningless against this environment, and slicing it
    # would produce a wrong evaluation with no error, so it is refused.
    if detected_actor_dim not in (ACTOR_TOTAL_DIM, ACTOR_TOTAL_DIM + Z_DIM):
        raise SystemExit(
            f"\n[Error] Checkpoint expects a {detected_actor_dim}-dim actor input, but this "
            f"build produces {ACTOR_TOTAL_DIM} (no encoder) or {ACTOR_TOTAL_DIM + Z_DIM} "
            f"(with the encoder).\n"
            f"        This checkpoint PREDATES the trajectory-tracking migration - its "
            f"observation layout no longer exists, so its weights cannot be evaluated.\n"
            f"        Retrain with:  .venv/bin/python Simulation/train.py\n"
        )

    custom_objs = dict(
        policy_class=AsymmetricActorCriticPolicy,
        actor_obs_dim=detected_actor_dim,
    )
    if detected_net_arch:
        custom_objs["net_arch"] = detected_net_arch
        custom_objs["policy_kwargs"] = dict(
            actor_obs_dim=detected_actor_dim,
            activation_fn=torch.nn.Tanh,
            net_arch=detected_net_arch,
        )

    model = PPO.load(
        model_path,
        custom_objects=custom_objs,
    )
    env = QuadFlipEnv(
        episode_seconds=episode_seconds,
        random_initial_state=random_initial_state,
        random_initial_pos=random_initial_pos,
        random_initial_vel=random_initial_vel,
        random_initial_att=random_initial_att,
        arena_radius=2.5,
        hover_gain=hover_gain,
    )
    env.set_dr_level(dr_level)

    # --- observation path -----------------------------------------------------------
    # A checkpoint trained WITH the frozen history encoder expects its actor input to be
    # [o_t | z]. Only LatentObsWrapper can produce that. Slicing the raw env observation
    # instead would silently hand the actor [o_t | aux | privileged...] with no error and
    # no crash, so the wrapper is attached whenever the checkpoint's actor dim says a
    # latent is present - and the slicing logic below then needs no change at all, because
    # with the wrapper in place obs[:detected_actor_dim] IS exactly [o_t | z].
    if detected_actor_dim == ACTOR_TOTAL_DIM + Z_DIM:
        if not os.path.isfile(ENCODER_CHECKPOINT):
            raise SystemExit(
                f"\n[Error] Checkpoint expects a {Z_DIM}-dim latent, but no encoder was found at\n"
                f"        {ENCODER_CHECKPOINT}\n"
                f"        Rebuild it:  .venv/bin/python Simulation/encoder/collect_data.py\n"
                f"                     .venv/bin/python Simulation/encoder/train_encoder.py\n"
            )
        injector = _LatentInjector(ENCODER_CHECKPOINT, ACTOR_TOTAL_DIM, ENCODER_AUX_DIM, Z_DIM)
        print(f"History Encoder    : ATTACHED ({os.path.basename(ENCODER_CHECKPOINT)}); "
              f"actor input = [o_t({ACTOR_TOTAL_DIM}) | z({Z_DIM})]")
    else:
        injector = None
        print(f"History Encoder    : none; actor input = o_t({detected_actor_dim})")

    obs, info = env.reset()
    if injector is not None:
        injector.reset()
        obs = injector.inject(obs)

    # Load observation normalization statistics if available
    stats_candidates = [
        os.path.join(_PROJECT_ROOT, f"{model_name}_vecnormalize.pkl"),
        os.path.join(_PROJECT_ROOT, model_name.replace(".zip", "_vecnormalize.pkl")),
        os.path.join(os.path.dirname(model_path), f"{os.path.splitext(os.path.basename(model_path))[0]}_vecnormalize.pkl"),
        os.path.join(_PROJECT_ROOT, "quad_flip_model_vecnormalize.pkl"),
    ]
    vec_norm = None
    for sp in stats_candidates:
        if os.path.isfile(sp):
            try:
                dummy_vec = DummyVecEnv([lambda: env])
                vec_norm = VecNormalize.load(sp, dummy_vec)
                vec_norm.training = False
                print(f"Loaded VecNormalize statistics from: {sp}")
                break
            except Exception as e:
                print(f"Note: Could not load {sp} due to shape mismatch: {e}")

    if vec_norm is None:
        print("\n" + "!" * 70)
        print("⚠️  CRITICAL WARNING: No VecNormalize statistics found!")
        print("   The policy was trained with observation normalization.")
        print("   Running with raw observations will cause erratic behavior and crashes!")
        print("!" * 70 + "\n")

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
    telemetry = {k: [] for k in ["t", "pos", "vel", "quat", "omega", "omega_des", "throttle", "euler", "w_cmd", "wMotor", "thr", "tor"]} if show_plots else None

    episode_idx = 1
    total_reward = 0.0
    max_pitch_deg = 0.0
    max_tilt_deg = 0.0
    batch_results = []

    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break

            step_start = time.time()
            obs_normalized = vec_norm.normalize_obs(obs) if vec_norm else obs
            if eval_actor_only:
                # Pass strictly the actor observation slice matching model's expected dimension
                obs_input = obs_normalized[:detected_actor_dim]
                if len(obs_input) < detected_actor_dim:
                    obs_input = np.pad(obs_input, (0, detected_actor_dim - len(obs_input)))
            else:
                obs_input = obs_normalized

            action, _ = model.predict(obs_input, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if injector is not None:
                obs = injector.inject(obs)
            total_reward += reward
            max_pitch_deg = max(max_pitch_deg, abs(float(np.degrees(env.quad.euler[1]))))
            tilt_deg = float(np.degrees(np.arccos(np.clip(env.quad.dcm[2, 2], -1.0, 1.0))))
            max_tilt_deg = max(max_tilt_deg, tilt_deg)

            if telemetry is not None and episode_idx == 1:
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

            if viewer is not None and viewer.is_running():
                viewer.sync()
                target_step_time = env.dt / max(0.01, playback_speed)
                sleep_time = target_step_time - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if terminated or truncated:
                if terminated:
                    term_reason = info.get("termination_reason", getattr(env, "termination_reason", "none"))
                    if term_reason == "phase2_reinversion":
                        status = "Failed (Re-inverted in Phase 2)"
                    elif term_reason == "phase1_overrotation":
                        status = "Failed (Over-rotated in Phase 1)"
                    elif term_reason == "phase1_timeout":
                        status = "Failed (Phase 1 Timeout > 1.0s)"
                    elif term_reason == "ground_crash" or env.quad.check_ground_contact():
                        status = "Crashed (Ground Contact)"
                    elif term_reason == "ceiling_breach" or env.quad.pos[2] > 2.5:
                        status = "Breached (Ceiling > 2.5m)"
                    elif term_reason == "arena_breach" or float(np.linalg.norm(env.quad.pos[:2])) > env.arena_radius:
                        status = f"Breached (Arena XY > {env.arena_radius:.1f}m)"
                    else:
                        status = "Terminated (Divergent State)"
                else:
                    status = "Completed (Stable Hover)"

                dist = info.get("active_disturbances", {})
                flip_deg = np.rad2deg(getattr(env, "accumulated_pitch", getattr(env, "accumulated_roll", 0.0)))
                wind_spd = getattr(env.wind, "velW_max", 0.0) if env.random_wind else 0.0
                print(f"[Episode {episode_idx}] Steps: {env.steps:4d} ({info['t']:.2f}s) | Rew: {total_reward:7.1f} | Flip: {str(env.flip_completed):5s} (Progress: {flip_deg:3.0f}°, PeakTilt: {max_tilt_deg:3.0f}°) | {status}")
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
                    "flip": env.flip_completed,
                    "status": status,
                    "rew": total_reward,
                    "steps": env.steps,
                    "peak_tilt": max_tilt_deg,
                    "xy_drift": float(np.linalg.norm(env.quad.pos[:2] - env.target_state[:2])),
                    "z_err": float(abs(env.quad.pos[2] - env.target_state[2])),
                })

                if not loop and episode_idx >= num_episodes:
                    break

                time.sleep(0.3)
                obs, info = env.reset()
                if injector is not None:
                    injector.reset()
                    obs = injector.inject(obs)
                total_reward = 0.0
                max_pitch_deg = 0.0
                max_tilt_deg = 0.0
                episode_idx += 1
    finally:
        if viewer is not None and viewer.is_running():
            viewer.close()

    if not loop and len(batch_results) > 1:
        flips = sum(1 for r in batch_results if r["flip"])
        hovers = sum(1 for r in batch_results if "Stable Hover" in r["status"])
        mean_rew = np.mean([r["rew"] for r in batch_results])
        mean_xy = np.mean([r["xy_drift"] for r in batch_results])
        mean_z = np.mean([r["z_err"] for r in batch_results])
        mean_tilt = np.mean([r["peak_tilt"] for r in batch_results])
        print(f"\n{'='*70}")
        print(f"BATCH EVALUATION SUMMARY ({len(batch_results)} episodes | DR={dr_level:.2f} | HoverGain={hover_gain:.2f}):")
        print(f"  Flip Success Rate : {flips}/{len(batch_results)} ({flips/len(batch_results)*100:.0f}%)")
        print(f"  Hover Recovery    : {hovers}/{len(batch_results)} ({hovers/len(batch_results)*100:.0f}%)")
        print(f"  Mean Return       : {mean_rew:.1f}")
        print(f"  Mean Peak Tilt    : {mean_tilt:.0f}°")
        print(f"  Mean Hover Drift  : XY = {mean_xy:.2f}m | Z = {mean_z:.2f}m")
        print(f"{'='*70}\n")

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
