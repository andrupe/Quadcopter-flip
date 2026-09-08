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

from quad_flip_env import QuadFlipEnv


# ======================================================================================
# TRAINING CONFIGURATION (Edit parameters directly here, then click Run in VS Code)
# ======================================================================================
TOTAL_TIMESTEPS: int = 20_000_000      # Target total steps for fresh run (~35-40 mins on 10 workers)
NUM_WORKERS: int = 10                 # Parallel CPU worker environments
MODEL_NAME: str = "quad_flip_model"   # Model output name (.zip saved in root directory)
DEVICE: str = "cpu"                   # "cpu" (recommended for Apple Silicon / M4) or "mps"

# Training Mode:
LOAD_PREVIOUS_MODEL: bool = False     # False: Fresh training from scratch with clean slate
PREVIOUS_MODEL_PATH: Optional[str] = None # Model checkpoint to resume from (if LOAD_PREVIOUS_MODEL=True)

# Fresh Training Schedule (Piecewise Linear - ETH Zurich Aligned):
LR_START: float = 3e-4                # Phase 1 start: Initial high exploration learning rate
LR_MID: float = 1e-4                  # Phase 2 start: Target learning rate at 3,500,000 steps (ADR ramp start)
LR_ADR_END: float = 4e-5              # Phase 3 start: Target learning rate at 13,500,000 steps (full DR reached)
LR_FLOOR: float = 3e-5                # Phase 3 end: Fine-tuning floor learning rate at 20,000,000 steps
LR_WARMUP_STEPS: int = 4_500_000      # Phase 1 duration: Nominal flip learning on clean sim before ADR (aligned with DR_START_STEPS)
LR_FINAL_STABLE: bool = False         # False: decay from LR_ADR_END to LR_FLOOR in Phase 3; True: hold at LR_FLOOR
ENT_COEF: float = 0.01                # Entropy coefficient (encourages exploration)
CHECKPOINT_FREQ: int = 50_000         # Checkpoint interval (timesteps per worker = 500,000 total steps)

# Automatic Domain Randomization (ADR):
DR_ENABLED: bool = True               # Enable progressive domain randomization
DR_START_STEPS: int = 4_500_000       # Start ADR after nominal flip is mastered (3.5M steps)
DR_END_STEPS: int = 14_500_000        # Reach full 100% DR at 13.5M steps (10M step ramp)

# Fine-Tuning Settings (Used only when LOAD_PREVIOUS_MODEL = True):
FT_LR_START: float = 3e-5             # Initial learning rate for fine-tuning
FT_LR_FLOOR: float = 1e-5             # Final floor learning rate for fine-tuning



# Environment options
EPISODE_SECONDS: float = 8.0         # Max flight time per episode (seconds)
TARGET_ALTITUDE: float = 1.2         # Target height for flip & recovery (meters)
SPAWN_ALTITUDE: float = 1.2          # Spawn height (meters)
ACTION_MODE: str = "motor"           # "motor" or "thrust_moment"
RANDOM_INITIAL_STATE: bool = True    # Randomize spawn position, tilt, and velocity for robustness
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
        model = PPO.load(model_to_load, env=vec_env, device=device)
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
    print(f"=== Starting PPO Training: Quadcopter Acrobatic Flip ===")
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
            policy="MlpPolicy",
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
    callbacks = [checkpoint_cb, vecnorm_cb]

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

    fps = max(1, model.num_timesteps - initial_steps) / max(1e-6, elapsed)
    print(f"\n{'='*65}")
    print(f"Training Session Ended in {elapsed:.1f}s ({fps:.0f} steps/s)")
    print(f"  Lifetime Steps : {model.num_timesteps:,}")
    print(f"  Saved Model    : {final_model_path}.zip")
    print(f"  Saved Stats    : {stats_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    train()
