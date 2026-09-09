from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Ensure paths are configured
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "Simulation" else _THIS_DIR
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import QuadFlipEnv, ACTOR_TOTAL_DIM, TOTAL_OBS_DIM
from asymmetric_policy import AsymmetricActorCriticPolicy


# ======================================================================================
# TRAINING CONFIGURATION (Edit parameters directly here, then click Run in VS Code)
# ======================================================================================
TOTAL_TIMESTEPS: int = 20_000_000      # Target total steps for Asymmetric Actor-Critic run (~28-35 mins on 10 workers)
NUM_WORKERS: int = 10                # Parallel CPU worker environments
MODEL_NAME: str = "quad_flip_model"   # Model output name (.zip saved in root directory)
DEVICE: str = "cpu"                   # "cpu" (recommended for Apple Silicon / M4) or "mps"

# Training Mode:
LOAD_PREVIOUS_MODEL: bool = False     # False: Fresh training from scratch with clean slate
PREVIOUS_MODEL_PATH: Optional[str] = None # Model checkpoint to resume from (if LOAD_PREVIOUS_MODEL=True)

# Fresh Training Schedule (Piecewise Linear - Asymmetric Actor-Critic Aligned):
LR_START: float = 3e-4                # Phase 1 start: Initial high exploration learning rate
LR_MID: float = 1.5e-4                # Phase 2 start: Target learning rate at 3,000,000 steps (ADR ramp start)
LR_ADR_END: float = 5e-5              # Phase 3 start: Target learning rate at 12,000,000 steps (full DR reached)
LR_FLOOR: float = 3e-5                # Phase 3 end: Fine-tuning floor learning rate at 16,000,000 steps
LR_WARMUP_STEPS: int = 4_500_000      # Phase 1 duration: Nominal flip learning on clean sim before ADR (aligned with DR_START_STEPS)
LR_FINAL_STABLE: bool = False         # False: decay from LR_ADR_END to LR_FLOOR in Phase 3; True: hold at LR_FLOOR
ENT_COEF: float = 0.01                # Entropy coefficient (encourages exploration)
CHECKPOINT_FREQ: int = 50_000         # Checkpoint interval (timesteps per worker = 500,000 total steps)

# Automatic Domain Randomization (ADR):
DR_ENABLED: bool = True               # Enable progressive domain randomization
DR_START_STEPS: int = 4_500_000       # Start ADR after nominal flip is mastered (3.0M steps)
DR_END_STEPS: int = 14_500_000        # Reach full 100% DR at 12.0M steps (9.0M step ramp)

# Fine-Tuning Settings (Used only when LOAD_PREVIOUS_MODEL = True):
FT_LR_START: float = 3e-5             # Initial learning rate for fine-tuning
FT_LR_FLOOR: float = 1e-5             # Final floor learning rate for fine-tuning



# Environment options
EPISODE_SECONDS: float = 8.0         # Max flight time per episode (seconds)
TARGET_ALTITUDE: float = 1.2         # Target height for flip & recovery (meters)
SPAWN_ALTITUDE: float = 1.2          # Spawn height (meters)
ARENA_RADIUS: float = 2.5            # Arena radius during training (meters)
ACTION_MODE: str = "motor"           # "motor" or "thrust_moment"
RANDOM_INITIAL_STATE: bool = True    # Randomize spawn position, tilt, and velocity for robustness

# Empirically Validated Reward Tolerances & Weights (Synthesized from 10 Scientific Experiments):
# Exp 2: tol_xy=0.75m, tol_vel=0.20m/s, tol_so3=0.90, w_vel=3.0 -> Slashed nominal drift from 2.50m to 0.70m (flare braking)
# Exp 3: w_z=2.5, tol_z=0.08m, tol_z_vel_flip=0.55m/s -> Slashed altitude error to 0.11m under full DR (top score 72.2)
# Exp 6: w_action=0.70 -> Eliminated motor chatter and actuator saturation, cutting DR 1.0 drift to 2.07m
TOL_XY_HOVER: float = 0.75           # meters: expanded hover position basin
TOL_VEL_HOVER: float = 0.20          # m/s: tight velocity target for strong derivative damping
TOL_Z_HOVER: float = 0.08            # meters: tight vertical target to eliminate 5-7cm payload sag
TOL_SO3_ATTITUDE: float = 0.90       # SO(3) attitude tolerance (~55°: allows 15°-20° flare braking tilt)
TOL_Z_VEL_FLIP: float = 0.55         # m/s: climb velocity target during flip initiation
W_XY: float = 1.8                    # Planar XY lock weight
W_Z: float = 2.5                     # Vertical altitude lock weight (boosted from 1.5)
W_VEL: float = 3.0                   # Linear velocity damping weight (boosted from 1.6)
W_ACTION: float = 0.70               # Action rate-of-change regularizer (boosted from 0.35)
W_UPRIGHT: float = 0.8               # SO(3) upright attitude weight
W_HEADING: float = 1.0               # Yaw heading alignment weight
W_OMEGA: float = 1.2                 # Body rate damping weight
# ======================================================================================


class PiecewiseLinearSchedule:


    def __init__(
        self,
        total_timesteps: int,
        warmup_steps: int = LR_WARMUP_STEPS,
        anneal_steps: int = 10_000_000,
        lr_start: float = LR_START,
        lr_mid: float = LR_MID,
        lr_adr_end: float = LR_ADR_END,
        lr_floor: float = LR_FLOOR,
        final_stable: bool = LR_FINAL_STABLE,
    ):
        self.total_timesteps = int(total_timesteps)
        self.warmup_steps = int(warmup_steps)
        self.anneal_steps = int(anneal_steps)
        self.lr_start = float(lr_start)
        self.lr_mid = float(lr_mid)
        self.lr_adr_end = float(lr_adr_end)
        self.lr_floor = float(lr_floor)
        self.final_stable = bool(final_stable)

    def __call__(self, progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 (start) down to 0.0 (end)
        current_step = (1.0 - progress_remaining) * self.total_timesteps

        if current_step <= self.warmup_steps:
            # Phase 1 (0 -> warmup): lr_start -> lr_mid (nominal clean flip learning)
            p = max(0.0, current_step) / max(1, self.warmup_steps)
            return float(self.lr_start - p * (self.lr_start - self.lr_mid))
        elif current_step <= (self.warmup_steps + self.anneal_steps):
            # Phase 2 (warmup -> warmup+anneal): lr_mid -> lr_adr_end (active ADR adaptation)
            p = (current_step - self.warmup_steps) / max(1, self.anneal_steps)
            return float(self.lr_mid - p * (self.lr_mid - self.lr_adr_end))
        else:
            # Phase 3 (warmup+anneal -> total): lr_adr_end -> lr_floor (fine-tune & consolidate at 100% DR)
            if self.final_stable:
                return float(self.lr_floor)
            else:
                rem = max(1, self.total_timesteps - (self.warmup_steps + self.anneal_steps))
                p = min(1.0, (current_step - (self.warmup_steps + self.anneal_steps)) / rem)
                return float(self.lr_adr_end - p * (self.lr_adr_end - self.lr_floor))


def linear_schedule(initial_value: float, final_value: float = 3e-5):
    """Linearly decays learning rate from initial_value to final_value based on remaining progress."""
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


class DomainRandomizationCallback(BaseCallback):
    """
    Automatic Domain Randomization (ADR): linearly ramps dr_level from 0.0 to 1.0
    over training. Starts after start_steps, reaches 1.0 at end_steps.
    """

    def __init__(
        self,
        start_steps: int = DR_START_STEPS,
        end_steps: int = DR_END_STEPS,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.start_steps = int(start_steps)
        self.end_steps = int(end_steps)
        self.current_level: float = 0.0

    def _on_training_start(self) -> None:
        if self.start_steps == 0 and self.end_steps == 0:
            init_level = 1.0
        elif self.num_timesteps >= self.end_steps and self.end_steps > self.start_steps:
            init_level = 1.0
        elif self.num_timesteps <= self.start_steps:
            init_level = 0.0
        else:
            init_level = (self.num_timesteps - self.start_steps) / max(1, self.end_steps - self.start_steps)

        self.current_level = float(np.clip(init_level, 0.0, 1.0))
        self.training_env.env_method("set_dr_level", self.current_level)
        if self.verbose > 0:
            print(f"\n{'*'*65}")
            print(f"*** AUTOMATIC DOMAIN RANDOMIZATION (ADR) INITIALIZED ***")
            if self.start_steps == 0 and self.end_steps == 0:
                print(f"  DR Mode: Fixed 100% full domain randomization throughout (dr_level = 1.0)")
            else:
                print(f"  DR Ramp: 0.0 -> 1.0 over steps {self.start_steps:,} to {self.end_steps:,}")
                print(f"  Initial DR Level: {self.current_level:.2f} at start step {self.num_timesteps:,}")
            print(f"{'*'*65}\n")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        steps = self.num_timesteps
        if self.start_steps == 0 and self.end_steps == 0:
            level = 1.0
        elif steps < self.start_steps:
            level = 0.0
        elif steps >= self.end_steps:
            level = 1.0
        else:
            level = (steps - self.start_steps) / max(1, self.end_steps - self.start_steps)

        level = float(np.clip(level, 0.0, 1.0))

        # Only update workers when level actually changes (avoid overhead)
        if abs(level - self.current_level) > 0.005 or (level >= 1.0 and self.current_level < 1.0):
            self.current_level = level
            self.training_env.env_method("set_dr_level", level)

            if self.verbose > 0 and (int(level * 100) % 10 == 0 or level >= 1.0):
                print(f"  [ADR] DR Level = {level:.2f} (step {steps:,})")




class VecNormalizeCheckpointCallback(BaseCallback):
    """Saves VecNormalize statistics alongside each checkpoint and keeps root stats updated."""

    def __init__(self, save_freq: int, save_path: str, root_stats_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.root_stats_path = root_stats_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            ckpt_stats = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps_vecnormalize.pkl")
            if hasattr(self.training_env, "save"):
                self.training_env.save(ckpt_stats)
                self.training_env.save(self.root_stats_path)
        return True


def plot_training_curves(
    timesteps: list[int] | np.ndarray,
    rew_means: list[float] | np.ndarray,
    len_means: list[float] | np.ndarray,
    save_path: str,
    dr_start: Optional[int] = None,
    dr_end: Optional[int] = None,
) -> None:
    """Generates and saves a clean 2-panel plot of ep_rew_mean and ep_len_mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(timesteps)
    rew = np.asarray(rew_means)
    length = np.asarray(len_means)

    fig, (ax_rew, ax_len) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Quadcopter Flip - Training Progression", fontsize=14, fontweight="bold", y=0.98)

    # 1. Episode Reward Mean
    ax_rew.plot(t / 1e6, rew, color="#1f77b4", linewidth=2.2, label="ep_rew_mean")
    ax_rew.set_ylabel("Episode Return (Mean)", fontsize=11, fontweight="bold")
    ax_rew.grid(True, linestyle="--", alpha=0.5)
    if len(rew) > 0:
        max_idx = int(np.argmax(rew))
        ax_rew.annotate(
            f"Peak: {rew[max_idx]:.1f}",
            xy=(t[max_idx] / 1e6, rew[max_idx]),
            xytext=(15, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
            fontweight="bold",
            color="#1f77b4",
        )

    # 2. Episode Length Mean
    ax_len.plot(t / 1e6, length, color="#2ca02c", linewidth=2.2, label="ep_len_mean (steps)")
    ax_len.set_xlabel("Total Timesteps (Millions)", fontsize=11, fontweight="bold")
    ax_len.set_ylabel("Episode Length (Steps)", fontsize=11, fontweight="bold")
    ax_len.grid(True, linestyle="--", alpha=0.5)

    # Secondary y-axis for flight time in seconds (100 Hz -> 100 steps = 1.0s)
    ax_sec = ax_len.twinx()
    ax_sec.set_ylabel("Flight Time (Seconds)", fontsize=10, color="#555555")
    y_min, y_max = ax_len.get_ylim()
    ax_sec.set_ylim(y_min * 0.01, y_max * 0.01)

    # ADR vertical shading if applicable
    if dr_start is not None and dr_end is not None and dr_end > dr_start:
        for ax in [ax_rew, ax_len]:
            ax.axvspan(dr_start / 1e6, dr_end / 1e6, color="orange", alpha=0.12, label="ADR Ramp (0→100%)" if ax == ax_rew else "")
            ax.axvline(dr_start / 1e6, color="darkorange", linestyle=":", alpha=0.7)
            ax.axvline(dr_end / 1e6, color="darkorange", linestyle=":", alpha=0.7)

    ax_rew.legend(loc="upper left")
    ax_len.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Training curves updated: {save_path}")


class TrainingMetricsCallback(BaseCallback):
    """
    Records ep_rew_mean and ep_len_mean across rollouts,
    streams them to CSV, and automatically generates training curves.
    """

    def __init__(
        self,
        csv_path: str,
        plot_path: str,
        dr_start_steps: int = DR_START_STEPS,
        dr_end_steps: int = DR_END_STEPS,
        plot_freq: int = 100_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.plot_path = plot_path
        self.dr_start_steps = int(dr_start_steps)
        self.dr_end_steps = int(dr_end_steps)
        self.plot_freq = int(plot_freq)
        self.timesteps: list[int] = []
        self.ep_rew_means: list[float] = []
        self.ep_len_means: list[float] = []
        self.walltimes: list[float] = []
        self.start_time: float = time.time()
        self.last_plot_step: int = 0

        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("timestep,walltime_sec,ep_rew_mean,ep_len_mean\n")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not hasattr(self.model, "ep_info_buffer") or len(self.model.ep_info_buffer) == 0:
            return

        rew_mean = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
        len_mean = float(np.mean([ep["l"] for ep in self.model.ep_info_buffer]))
        step = int(self.num_timesteps)
        elapsed = float(time.time() - self.start_time)

        self.timesteps.append(step)
        self.ep_rew_means.append(rew_mean)
        self.ep_len_means.append(len_mean)
        self.walltimes.append(elapsed)

        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{step},{elapsed:.2f},{rew_mean:.4f},{len_mean:.2f}\n")

        if step - self.last_plot_step >= self.plot_freq:
            self.last_plot_step = step
            self.plot_metrics()

    def _on_training_end(self) -> None:
        self.plot_metrics()

    def plot_metrics(self) -> None:
        if len(self.timesteps) < 2:
            return
        plot_training_curves(
            timesteps=self.timesteps,
            rew_means=self.ep_rew_means,
            len_means=self.ep_len_means,
            save_path=self.plot_path,
            dr_start=self.dr_start_steps,
            dr_end=self.dr_end_steps,
        )


def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    num_workers: int = NUM_WORKERS,
    model_name: str = MODEL_NAME,
    device: str = DEVICE,
    learning_rate: float = LR_START,
    lr_mid: float = LR_MID,
    lr_adr_end: float = LR_ADR_END,
    lr_floor: float = LR_FLOOR,
    lr_final_stable: bool = LR_FINAL_STABLE,
    ft_lr_start: float = FT_LR_START,
    ft_lr_floor: float = FT_LR_FLOOR,
    ent_coef: float = ENT_COEF,
    checkpoint_freq: int = CHECKPOINT_FREQ,
    load_previous_model: bool = LOAD_PREVIOUS_MODEL,
    previous_model_path: Optional[str] = PREVIOUS_MODEL_PATH,
    random_initial_state: bool = RANDOM_INITIAL_STATE,
):
    """Main training loop using Stable-Baselines3 PPO."""
    torch.set_num_threads(1)

    # Resolve previous model candidate if requested
    model_to_load = None
    if load_previous_model:
        raw_candidate = previous_model_path if previous_model_path else f"{model_name}.zip"
        if not raw_candidate.endswith(".zip"):
            raw_candidate += ".zip"
        if os.path.isabs(raw_candidate) and os.path.isfile(raw_candidate):
            model_to_load = raw_candidate
        elif os.path.isfile(raw_candidate):
            model_to_load = os.path.abspath(raw_candidate)
        elif os.path.isfile(os.path.join(_PROJECT_ROOT, raw_candidate)):
            model_to_load = os.path.join(_PROJECT_ROOT, raw_candidate)
        else:
            print(f"⚠️ Warning: LOAD_PREVIOUS_MODEL is True, but '{raw_candidate}' was not found in cwd or project root. Starting from scratch.\n")

    anneal_steps = (DR_END_STEPS - DR_START_STEPS) if DR_ENABLED else 10_000_000

    def make_env():
        return QuadFlipEnv(
            action_mode=ACTION_MODE,
            episode_seconds=EPISODE_SECONDS,
            target_altitude=TARGET_ALTITUDE,
            spawn_altitude=SPAWN_ALTITUDE,
            random_initial_state=random_initial_state,
            arena_radius=ARENA_RADIUS,
            tol_xy_hover=TOL_XY_HOVER,
            tol_vel_hover=TOL_VEL_HOVER,
            tol_z_hover=TOL_Z_HOVER,
            tol_so3_attitude=TOL_SO3_ATTITUDE,
            tol_z_vel_flip=TOL_Z_VEL_FLIP,
            w_xy=W_XY,
            w_z=W_Z,
            w_vel=W_VEL,
            w_action=W_ACTION,
            w_upright=W_UPRIGHT,
            w_heading=W_HEADING,
            w_omega=W_OMEGA,
        )

    vec_env = make_vec_env(make_env, n_envs=num_workers, vec_env_cls=SubprocVecEnv)
    stats_path = os.path.join(_PROJECT_ROOT, f"{model_name}_vecnormalize.pkl")
    save_dir = os.path.join(_PROJECT_ROOT, "logs")
    os.makedirs(save_dir, exist_ok=True)

    initial_steps = 0
    if model_to_load:
        # Load saved VecNormalize statistics so observation scaling is perfectly preserved
        norm_candidate = model_to_load.replace(".zip", "_vecnormalize.pkl")
        if os.path.isfile(norm_candidate):
            print(f"Loaded VecNormalize statistics from: {norm_candidate}")
            vec_env = VecNormalize.load(norm_candidate, vec_env)
            vec_env.training = True
        elif os.path.isfile(stats_path):
            print(f"Loaded VecNormalize statistics from: {stats_path}")
            vec_env = VecNormalize.load(stats_path, vec_env)
            vec_env.training = True
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        print(f"Resuming model training from checkpoint: {model_to_load}")
        model = PPO.load(
            model_to_load,
            env=vec_env,
            device=device,
            custom_objects=dict(
                policy_class=AsymmetricActorCriticPolicy,
                actor_obs_dim=ACTOR_TOTAL_DIM,
            ),
        )
        initial_steps = int(getattr(model, "num_timesteps", 0))
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        model = None

    # Determine steps to train and lifetime steps
    if initial_steps > 0:
        if total_timesteps > initial_steps:
            steps_to_train = total_timesteps - initial_steps
            total_lifetime_steps = total_timesteps
        else:
            steps_to_train = total_timesteps
            total_lifetime_steps = initial_steps + steps_to_train
    else:
        steps_to_train = total_timesteps
        total_lifetime_steps = total_timesteps

    print(f"\n{'='*65}")
    print(f"=== Starting PPO Training: Quadcopter Acrobatic Flip (AAC) ===")
    print(f"  Architecture   : Asymmetric Actor-Critic (Actor: {ACTOR_TOTAL_DIM} dims | Critic: {TOTAL_OBS_DIM} dims)")
    print(f"  Workers        : {num_workers} parallel CPU processes")
    print(f"  Session Steps  : {steps_to_train:,} (Target lifetime: {total_lifetime_steps:,})")
    print(f"  Device         : {device.upper()}")
    print(f"  Output Model   : {model_name}.zip")
    print(f"  Random Spawn   : {'Enabled' if random_initial_state else 'Disabled'}")
    print(f"  Resume Mode    : {'Resuming from ' + model_to_load if model_to_load else 'Fresh Training (from scratch)'}")
    if model_to_load:
        print(f"  Fine-Tuning LR : {ft_lr_start:.1e} -> {ft_lr_floor:.1e} (Linear decay over {steps_to_train:,} steps)")
    else:
        print(f"  LR Schedule    : Phase 1 (0 -> {LR_WARMUP_STEPS:,}): {learning_rate:.1e} -> {lr_mid:.1e} (Nominal Flip)")
        print(f"                   Phase 2 ({LR_WARMUP_STEPS:,} -> {LR_WARMUP_STEPS + anneal_steps:,}): {lr_mid:.1e} -> {lr_adr_end:.1e} (ADR Adaptation)")
        print(f"                   Phase 3 ({LR_WARMUP_STEPS + anneal_steps:,} -> {total_timesteps:,}): {lr_adr_end:.1e} -> {lr_floor:.1e} (Hardened Polishing)")
    print(f"  Motor Dynamics : motor_tau = 25ms (realistic Crazyflie motor response)")
    print(f"  Flare Braking  : tol_xy={TOL_XY_HOVER}m, tol_vel={TOL_VEL_HOVER}m/s, tol_so3={TOL_SO3_ATTITUDE}, w_vel={W_VEL}")
    print(f"  Altitude Hold  : tol_z={TOL_Z_HOVER}m, tol_vz_flip={TOL_Z_VEL_FLIP}m/s, w_z={W_Z}")
    print(f"  Smoothness     : w_action={W_ACTION} (motor chatter regularized)")
    if DR_ENABLED:
        if DR_START_STEPS == 0 and DR_END_STEPS == 0:
            print(f"  ADR Mode       : Fixed 100% full domain randomization throughout (dr_level = 1.0)")
        else:
            print(f"  ADR Schedule   : dr_level 0.0 -> 1.0 over steps {DR_START_STEPS:,} to {DR_END_STEPS:,}")
    print(f"{'='*65}\n")

    if model is not None:
        # Fine-tuning mode: linear anneal from ft_lr_start down to ft_lr_floor over steps_to_train
        start_progress = steps_to_train / max(1, total_lifetime_steps)

        def ft_schedule(progress_remaining: float) -> float:
            p_norm = max(0.0, min(1.0, progress_remaining / start_progress))
            return ft_lr_floor + p_norm * (ft_lr_start - ft_lr_floor)

        model.lr_schedule = ft_schedule
        model._custom_objects = {"learning_rate": ft_schedule, "lr_schedule": ft_schedule}
    else:
        # Fresh training: 3-phase piecewise schedule synchronized with ADR curriculum
        lr_schedule = PiecewiseLinearSchedule(
            total_timesteps=total_lifetime_steps,
            warmup_steps=LR_WARMUP_STEPS,
            anneal_steps=anneal_steps,
            lr_start=learning_rate,
            lr_mid=lr_mid,
            lr_adr_end=lr_adr_end,
            lr_floor=lr_floor,
            final_stable=lr_final_stable,
        )
        model = PPO(
            policy=AsymmetricActorCriticPolicy,
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=2048,
            batch_size=512,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            policy_kwargs=dict(
                actor_obs_dim=ACTOR_TOTAL_DIM,
                activation_fn=nn.Tanh,
                net_arch=dict(pi=[128, 128], vf=[512, 256, 128]),
                log_std_init=-0.5,
            ),
            verbose=1,
            device=device,
        )

    checkpoint_cb = CheckpointCallback(save_freq=checkpoint_freq, save_path=save_dir)
    vecnorm_cb = VecNormalizeCheckpointCallback(
        save_freq=checkpoint_freq, save_path=save_dir, root_stats_path=stats_path
    )
    metrics_csv = os.path.join(save_dir, "training_metrics.csv")
    metrics_plot = os.path.join(_PROJECT_ROOT, "training_curves.png")
    metrics_cb = TrainingMetricsCallback(
        csv_path=metrics_csv,
        plot_path=metrics_plot,
        dr_start_steps=DR_START_STEPS,
        dr_end_steps=DR_END_STEPS,
        plot_freq=100_000,
    )
    callbacks = [checkpoint_cb, vecnorm_cb, metrics_cb]

    if DR_ENABLED:
        dr_cb = DomainRandomizationCallback(
            start_steps=DR_START_STEPS,
            end_steps=DR_END_STEPS,
        )
        callbacks.append(dr_cb)

    start_time = time.time()
    reset_timesteps = (model_to_load is None)
    try:
        model.learn(total_timesteps=steps_to_train, callback=callbacks, reset_num_timesteps=reset_timesteps)
    except KeyboardInterrupt:
        print("\n\n[Notice] Training interrupted by user (Ctrl+C). Gracefully saving current model and normalization stats...")

    elapsed = time.time() - start_time

    final_model_path = os.path.join(_PROJECT_ROOT, model_name)
    model.save(final_model_path)
    vec_env.save(stats_path)

    # Final metrics plot generation
    metrics_cb.plot_metrics()
    # Also save a copy inside logs/ directory
    metrics_cb.plot_path = os.path.join(save_dir, "training_curves.png")
    metrics_cb.plot_metrics()

    fps = max(1, model.num_timesteps - initial_steps) / max(1e-6, elapsed)
    print(f"\n{'='*65}")
    print(f"Training Session Ended in {elapsed:.1f}s ({fps:.0f} steps/s)")
    print(f"  Lifetime Steps : {model.num_timesteps:,}")
    print(f"  Saved Model    : {final_model_path}.zip")
    print(f"  Saved Stats    : {stats_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    train()
