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

from quad_flip_env import (
    ACTOR_TOTAL_DIM,
    TOTAL_OBS_DIM,
    FLIGHT_RADIUS,
    SPAWN_Z,
    INIT_RATE_RANGE,
    INIT_VEL_RANGE,
    QuadFlipEnv,
)
from asymmetric_policy import AsymmetricActorCriticPolicy

# The frozen history encoder lives in Simulation/encoder/. Imported defensively: the
# encoder is an ADD-ON, so a missing or incomplete encoder package must degrade to
# "train without the latent" rather than stopping training from starting at all. A hard
# import here means one absent file makes the whole trainer unimportable.
try:
    from encoder.latent_obs_wrapper import LatentObsWrapper
except ImportError:  # pragma: no cover - exercised by the no-encoder configuration
    LatentObsWrapper = None

# Frozen history encoder produced by Simulation/encoder/train_encoder.py. The wrapper
# injects its z output into the observation between the actor block and the aux block.
ENCODER_CHECKPOINT: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "encoder_gru.pt"
)
Z_DIM: int = 16


# ======================================================================================
# TRAINING CONFIGURATION (Edit parameters directly here, then click Run in VS Code)
# ======================================================================================
# ======================================================================================
# BASELINE RUN (trajectory tracking)
#
# These values replace a schedule tuned for the OLD two-phase flip task. Two properties of
# that task drove the old numbers, and neither one still holds:
#
#   1. It had a hard EXPLORATION problem. "Master the nominal flip before randomising
#      anything" is why the LR sat at 3e-4 for 7,000,000 steps. Tracking does not have that
#      shape: the reference tells the policy what to do at EVERY step and the reward is
#      dense and bounded, so there is no binary discovery event to wait for and no reason
#      to hold a high LR for 7M steps.
#   2. It was a SINGLE manoeuvre. This task is a mixture (hover / waypoints / figure-8 /
#      flip) present from step 0, so a "nominal only" phase has no meaning.
#
# The phases are therefore aligned with the ADR ramp instead of with a manoeuvre:
#   Phase 1 (no DR)      learn the mixture on clean dynamics
#   Phase 2 (= DR ramp)  adapt while the dynamics are being corrupted
#   Phase 3 (full DR)    polish
# LR_WARMUP_STEPS == DR_START_STEPS makes that alignment exact. Change one, change both.
#
# REVISION 2026-09-11 - measured, not guessed (previous run: 10M steps, DR 2M -> 7M):
#   2M was far too early. The per-step reward was still growing at +0.49/step per M steps
#   (window peak +0.71) when the ramp began, and net growth fell to ~0 once dr passed ~0.2.
#   A deterministic eval of the 2.5M checkpoint scored 3.48/step clean = 48% of the 7.3
#   maximum - only ~2/3 of the way to the 75% gate (5.475/step) that the scripted
#   controller clears at 5.72-6.57. Keep the mixture clean until the policy is CLOSE TO
#   THE GATE, then corrupt. Budget raised 10M -> 15M so the ramp still has a polish phase.
# ======================================================================================
TOTAL_TIMESTEPS: int = 15_000_000      # ~85-90 min at the measured ~2.8-3k steps/s on 10 workers
NUM_WORKERS: int = 10                  # Parallel CPU worker environments
MODEL_NAME: str = "quad_flip_model"    # Model output name (.zip saved in root directory)
DEVICE: str = "cpu"                    # "cpu" (recommended for Apple Silicon) or "mps"

# Training Mode:
LOAD_PREVIOUS_MODEL: bool = False      # False: Fresh training from scratch with clean slate
PREVIOUS_MODEL_PATH: Optional[str] = None  # Model checkpoint to resume from (if LOAD_PREVIOUS_MODEL=True)

LR_START: float = 3e-4                 # Phase 1: explore the mixture on clean dynamics
LR_MID: float = 1.5e-4                 # Phase 2 start, as the ADR ramp begins
LR_ADR_END: float = 5e-5               # Phase 3 start, at full DR
LR_FLOOR: float = 3e-5                 # Phase 3 end
LR_WARMUP_STEPS: int = 6_000_000       # Phase 1 duration. Keep equal to DR_START_STEPS.
LR_FINAL_STABLE: bool = False          # False: decay to LR_FLOOR in Phase 3; True: hold at LR_ADR_END
CHECKPOINT_FREQ: int = 250_000         # Timesteps per worker (2.5M total across 10 workers)

# ENTROPY. 0.01 -> 0.001 is the "classic" anneal and is deliberately conservative here.
# Andrychowicz et al. (2020), 'What Matters in On-Policy RL', find ent_coef = 0.0 is usually
# optimal for continuous control, and the RMA reference implementation (legged_gym) also
# uses 0.0 - with an adaptive-KL learning rate doing the work instead.
#
# The reason to keep a NONZERO coefficient on THIS task is the mixture: a policy that gives
# up on the flip can still collect decent average reward from hover and waypoints, which is
# exactly the kind of local optimum entropy pressure is the standard antidote to. Set
# ENT_COEF_END = 0.0 for the RMA-style configuration.
ENT_COEF: float = 0.01
ENT_COEF_END: float = 0.001

# Automatic Domain Randomization (ADR):
DR_ENABLED: bool = True                # Enable progressive domain randomization
DR_START_STEPS: int = 6_000_000        # Start corrupting only once CLEAN skill approaches the gate.
                                       # 2026-09-11: at the old 2M the policy was at 48% of max
                                       # and still improving ~+0.5/step per M - the ramp ate it.
DR_END_STEPS: int = 11_000_000         # Reach full 100% DR (5M ramp, unchanged)

# RMA-STYLE ADAPTIVE LEARNING RATE (Schulman 2017 KL controller, as used by legged_gym/RMA).
# Off by default so a baseline run has a PREDICTABLE schedule. Turn it on when you would
# rather the LR find its own level than trust the hand-tuned phases above - which is the
# right call precisely when the task has changed and the phases are what you least trust.
ADAPTIVE_KL_LR: bool = False
ADAPTIVE_KL_DESIRED: float = 0.01      # legged_gym default
ADAPTIVE_KL_LR_MIN: float = 1e-5
ADAPTIVE_KL_LR_MAX: float = 3e-3

# Fine-Tuning Settings (Used only when LOAD_PREVIOUS_MODEL = True):
FT_LR_START: float = 3e-5             # Initial learning rate for fine-tuning
FT_LR_FLOOR: float = 1e-5             # Final floor learning rate for fine-tuning



# Environment options
EPISODE_SECONDS: float = 8.0         # Max flight time per episode (seconds)
SPAWN_ALTITUDE: float = 1.2          # Spawn height (meters; must match SPAWN_Z in quad_flip_env.py)
# Horizontal footprint: the sampler keeps every suggested reference inside a 1.5 m x 1.5 m
# square (bounds_xy = 0.75 m in TrajectoryConfig, enforced by a screen at sample time). It
# is a fixed property of the task rather than a knob here on purpose - evaluation builds
# the same config, so training and evaluation cannot silently disagree about it.
ACTION_MODE: str = "rate_pid"        # "rate_pid" (thrust + body rate PID), "motor", or "thrust_moment"
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

    dr_level is the ONLY thing this callback drives. It used to also announce a shrinking
    "curriculum arena", but the env ignores that (the tracking task's difficulty comes
    from the commanded manoeuvre, and the flight volume is a fixed sphere), so the message
    was pure decoration and has been removed.
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


class StdFloorCallback(BaseCallback):
    """
    Scheduled exploration bounds: decays min_log_std from start_floor to end_floor
    over training. Early training forces broad exploration (std >= 0.37) to escape
    hover-only local optima; late training relaxes the floor (std >= 0.08) so the
    policy can converge to precise hover control.

    References:
      - Andrychowicz et al. (2020): free log_std learning is optimal for continuous control
      - Kaufmann et al. (2020) 'Deep Drone Acrobatics': no std clamping, relies on reward shaping
    We use a decaying floor as a compromise: prevents early collapse while allowing late precision.
    """

    def __init__(
        self,
        min_log_std_start: float = -1.0,
        min_log_std_end: float = -2.5,
        max_log_std: float = 0.0,
        total_timesteps: int = TOTAL_TIMESTEPS,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.min_log_std_start = float(min_log_std_start)
        self.min_log_std_end = float(min_log_std_end)
        self.max_log_std = float(max_log_std)
        self.total_timesteps = int(total_timesteps)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "log_std"):
            progress = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
            current_floor = self.min_log_std_start + progress * (self.min_log_std_end - self.min_log_std_start)
            with torch.no_grad():
                self.model.policy.log_std.data.clamp_(min=current_floor, max=self.max_log_std)


class EntCoefScheduleCallback(BaseCallback):
    """
    Linearly decays PPO entropy coefficient from start_value to end_value over training.

    Academic rationale:
      - Andrychowicz et al. (2020) 'What Matters in On-Policy RL': ent_coef = 0.0 is
        optimal for most continuous control, but our task has a strong local optimum
        (hover without flipping) that requires exploration pressure early.
      - PPO (Schulman 2017): used 0.01 for discrete Atari; continuous control uses lower.
      - Schedule: start at 0.01 (upper bound for continuous PPO) to escape hover trap,
        decay to 0.001 for precision hover convergence.
    """

    def __init__(
        self,
        start_value: float = 0.01,
        end_value: float = 0.001,
        total_timesteps: int = TOTAL_TIMESTEPS,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.total_timesteps = int(total_timesteps)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        progress = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
        new_ent = self.start_value + progress * (self.end_value - self.start_value)
        self.model.ent_coef = float(new_ent)


class AdaptiveKLScheduleCallback(BaseCallback):
    """
    RMA / legged_gym style adaptive learning rate: drive the PPO KL toward a target by
    scaling the LR multiplicatively once per rollout.

        kl > desired * 2   ->  lr /= 1.5     (steps too large, back off)
        kl < desired / 2   ->  lr *= 1.5     (steps too small, push harder)

    This is the one piece of the RMA reference implementation worth borrowing, and it is
    borrowed because of what actually went wrong here: the 3-phase schedule in this file
    was fitted to the OLD two-phase flip reward, so there is no reason its phase
    boundaries suit the tracking reward. An adaptive controller needs to know nothing
    about the task - only that the KL is measurable.

    The LR is clamped to [lr_min, lr_max] so one bad KL estimate cannot collapse or blow
    up the run. The 1.5x update is deliberately gentle: each rollout here is
    2048 x 10 = 20k steps, a much coarser control interval than legged_gym's, so a large
    multiplier would overshoot badly between corrections.
    """

    def __init__(
        self,
        desired_kl: float = ADAPTIVE_KL_DESIRED,
        lr_min: float = ADAPTIVE_KL_LR_MIN,
        lr_max: float = ADAPTIVE_KL_LR_MAX,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.desired_kl = float(desired_kl)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.lr_history: list[float] = []

    def _current_lr(self) -> float:
        """
        The learning rate the schedule is producing RIGHT NOW.

        SB3 stores `_current_progress_remaining` on the model and feeds it to
        `lr_schedule` on every update, so that is the argument to evaluate at. This used
        to call `lr_schedule(1.0)`, which asks for the value at the START of training -
        wrong for a decaying schedule, and silently so.
        """
        progress = getattr(self.model, "_current_progress_remaining", None)
        lr = getattr(self.model, "lr_schedule", None)
        if callable(lr):
            lr = lr(float(progress)) if progress is not None else lr(0.0)
        if lr is None:
            lr = self.model.learning_rate
        if callable(lr):
            lr = lr(0.0)
        lr = float(lr)
        return lr if np.isfinite(lr) and lr > 0.0 else self.lr_min

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is None:
            return
        kl = self.logger.name_to_value.get("train/approx_kl")
        if kl is None or not np.isfinite(kl):
            return

        lr = self._current_lr()
        if kl > self.desired_kl * 2.0:
            lr = max(self.lr_min, lr / 1.5)
        elif kl < self.desired_kl / 2.0:
            lr = min(self.lr_max, lr * 1.5)

        # SB3 reads lr_schedule(progress) every update, so a constant closure is what
        # actually pins the LR. Setting model.learning_rate alone would be overwritten on
        # the next _update_learning_rate() call.
        self.model.learning_rate = lr
        self.model.lr_schedule = lambda _progress, _lr=lr: _lr
        self.lr_history.append(lr)


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
        # APPEND, never truncate. This path is shared with the smoke tests and the
        # encoder-chain check (they call train() with their own model names), and opening
        # it with "w" used to wipe a live run's recorded history. The header is written
        # only for a new/empty file, so rows from separate runs simply accumulate.
        if not os.path.isfile(self.csv_path) or os.path.getsize(self.csv_path) == 0:
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
    ent_coef_end: float = ENT_COEF_END,
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

    # Schedule sanity. Both of these are silent when wrong - training just runs with a
    # schedule that does not do what the constants say it does - so they are checked.
    if not load_previous_model:
        if LR_WARMUP_STEPS != DR_START_STEPS:
            print(f"  [warn] LR_WARMUP_STEPS ({LR_WARMUP_STEPS:,}) != DR_START_STEPS "
                  f"({DR_START_STEPS:,}): the LR phases will not line up with the ADR ramp.")
        if LR_WARMUP_STEPS + anneal_steps > total_timesteps:
            print(f"  [warn] LR phases span {LR_WARMUP_STEPS + anneal_steps:,} steps but the run "
                  f"is {total_timesteps:,}: Phase 3 (full-DR polish) will never be reached.")

    def make_env():
        # NOTE: the old tol_*/w_* reward knobs are deliberately gone. The two-phase flip
        # reward they configured no longer exists; it was replaced by a single
        # trajectory-tracking objective whose tolerances are keyed per manoeuvre
        # (TRACK_TOL in quad_flip_env.py). Those tolerances are what let ONE reward serve
        # a hover, a figure-8 and a 360 deg flip, so exposing them as per-run tuning would
        # let a run silently re-weight the task back toward whichever manoeuvre it
        # happened to be failing.
        return QuadFlipEnv(
            action_mode=ACTION_MODE,
            episode_seconds=EPISODE_SECONDS,
            spawn_altitude=SPAWN_ALTITUDE,
            random_initial_state=random_initial_state,
            # Training reads nothing from the env's telemetry info dict (SB3's worker adds
            # what the PPO path needs), and building + pickling it across 10 workers costs
            # more than the physics does. Evaluation and the diagnostic scripts construct
            # QuadFlipEnv directly and still get the full dict.
            telemetry=False,
        )

    vec_env = make_vec_env(make_env, n_envs=num_workers, vec_env_cls=SubprocVecEnv)

    # Frozen history encoder -> z. Built HERE, in the trainer process, not inside the
    # workers: one batched forward serves all workers, and Simulation/quad_flip_env.py
    # stays torch-free (SubprocVecEnv uses fork/spawn, so a set_num_threads() call in this
    # process would not reach the workers anyway).
    actor_obs_dim = ACTOR_TOTAL_DIM
    if LatentObsWrapper is None:
        print("WARNING: the encoder package (Simulation/encoder/) is not importable.\n"
              f"         Training WITHOUT the history encoder; the actor sees {actor_obs_dim} dims.\n"
              f"         This is a valid configuration, just a less informed policy.\n")
    elif os.path.isfile(ENCODER_CHECKPOINT):
        vec_env = LatentObsWrapper(vec_env, encoder_path=ENCODER_CHECKPOINT, z_dim=Z_DIM)
        actor_obs_dim = ACTOR_TOTAL_DIM + Z_DIM
        print(f"Loaded history encoder from {ENCODER_CHECKPOINT}: actor sees {actor_obs_dim} dims "
              f"({ACTOR_TOTAL_DIM} obs + {Z_DIM} latent)")
    else:
        print(f"WARNING: encoder checkpoint not found at {ENCODER_CHECKPOINT}.\n"
              f"         Training WITHOUT the history encoder; the actor sees {actor_obs_dim} dims.")

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
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False, clip_obs=10.0)

        print(f"Resuming model training from checkpoint: {model_to_load}")
        model = PPO.load(
            model_to_load,
            env=vec_env,
            device=device,
            custom_objects=dict(
                policy_class=AsymmetricActorCriticPolicy,
                actor_obs_dim=actor_obs_dim,
            ),
        )
        initial_steps = int(getattr(model, "num_timesteps", 0))
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False, clip_obs=10.0)
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
    print(f"=== Starting PPO Training: Quadcopter Trajectory Tracking (AAC) ===")
    _critic_dim = int(vec_env.observation_space.shape[0])
    print(f"  Architecture   : Asymmetric Actor-Critic (Actor: {actor_obs_dim} dims | Critic: {_critic_dim} dims)")
    print(f"  Workers        : {num_workers} parallel CPU processes")
    print(f"  Session Steps  : {steps_to_train:,} (Target lifetime: {total_lifetime_steps:,})")
    print(f"  Device         : {device.upper()}")
    print(f"  Output Model   : {model_name}.zip")
    print(f"  Random Spawn   : {'Enabled' if random_initial_state else 'Disabled'}")
    print(f"  Resume Mode    : {'Resuming from ' + model_to_load if model_to_load else 'Fresh Training (from scratch)'}")
    if model_to_load:
        print(f"  Fine-Tuning LR : {ft_lr_start:.1e} -> {ft_lr_floor:.1e} (Linear decay over {steps_to_train:,} steps)")
    else:
        print(f"  Task           : TRAJECTORY TRACKING (hover / waypoints / figure-8 / lissajous / orbit / slalom / flip)")
        if ADAPTIVE_KL_LR:
            print(f"  LR Schedule    : ADAPTIVE KL (target {ADAPTIVE_KL_DESIRED}, "
                  f"clamped to [{ADAPTIVE_KL_LR_MIN:.0e}, {ADAPTIVE_KL_LR_MAX:.0e}]) - RMA/legged_gym style")
        else:
            print(f"  LR Schedule    : Phase 1 (0 -> {LR_WARMUP_STEPS:,}): {learning_rate:.1e} -> {lr_mid:.1e} (mixture, no DR)")
            print(f"                   Phase 2 ({LR_WARMUP_STEPS:,} -> {LR_WARMUP_STEPS + anneal_steps:,}): {lr_mid:.1e} -> {lr_adr_end:.1e} (DR ramp)")
            print(f"                   Phase 3 ({LR_WARMUP_STEPS + anneal_steps:,} -> {total_timesteps:,}): {lr_adr_end:.1e} -> {lr_floor:.1e} (full-DR polish)")
    print(f"  Entropy Coef   : {ent_coef:.4f} -> {ent_coef_end:.4f} (linear; set both to 0.0 for RMA-style)")
    print(f"  Tracker Tol.   : per-manoeuvre, see TRACK_TOL in quad_flip_env.py")
    print(f"  Motor Dynamics : motor_tau = 25ms (realistic Crazyflie motor response)")
    if DR_ENABLED:
        if DR_START_STEPS == 0 and DR_END_STEPS == 0:
            print(f"  ADR Mode       : Fixed 100% full domain randomization throughout (dr_level = 1.0)")
        else:
            print(f"  ADR Schedule   : dr_level 0.0 -> 1.0 over steps {DR_START_STEPS:,} to {DR_END_STEPS:,}")
    # The arena is no longer a shrinking curriculum: the task's difficulty comes from the
    # commanded manoeuvre, and the flight volume is the FIXED sphere the reference sampler
    # is screened against. Report that instead of an arena radius that no longer exists.
    print(f"  Flight Volume  : sphere r={FLIGHT_RADIUS:.1f}m centred (0,0,{SPAWN_Z:.1f}) "
          f"-> ceiling {SPAWN_Z + FLIGHT_RADIUS:.1f}m, floor = ground ({SPAWN_Z:.1f}m below start)")
    print(f"  Initial Kick   : {INIT_VEL_RANGE[0]:.2f}->{INIT_VEL_RANGE[1]:.2f} m/s per axis, "
          f"{INIT_RATE_RANGE[0]:.2f}->{INIT_RATE_RANGE[1]:.2f} rad/s per axis (scaled by DR)")
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
                actor_obs_dim=actor_obs_dim,
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
    # Default runs keep the canonical artifact names (tooling and the tracked CSV depend on
    # them); auxiliary runs - smoke tests, the encoder-chain check - get their own files so
    # they cannot overwrite the real run's history or plot.
    _run_suffix = "" if model_name == MODEL_NAME else f"_{model_name}"
    metrics_csv = os.path.join(save_dir, f"training_metrics{_run_suffix}.csv")
    metrics_plot = os.path.join(_PROJECT_ROOT, f"training_curves{_run_suffix}.png")
    metrics_cb = TrainingMetricsCallback(
        csv_path=metrics_csv,
        plot_path=metrics_plot,
        dr_start_steps=DR_START_STEPS,
        dr_end_steps=DR_END_STEPS,
        plot_freq=100_000,
    )
    std_floor_cb = StdFloorCallback(
        min_log_std_start=-1.0,   # Early: std >= 0.37 (broad exploration)
        min_log_std_end=-2.5,     # Late: std >= 0.08 (allows precision tracking)
        max_log_std=0.0,
        total_timesteps=total_lifetime_steps,
    )
    ent_schedule_cb = EntCoefScheduleCallback(
        start_value=ent_coef,     # escapes the "only hover well" local optimum
        end_value=ent_coef_end,   # 0.001 classic, 0.0 RMA-style
        total_timesteps=total_lifetime_steps,
    )
    callbacks = [checkpoint_cb, vecnorm_cb, metrics_cb, std_floor_cb, ent_schedule_cb]

    if ADAPTIVE_KL_LR:
        callbacks.append(AdaptiveKLScheduleCallback(
            desired_kl=ADAPTIVE_KL_DESIRED,
            lr_min=ADAPTIVE_KL_LR_MIN,
            lr_max=ADAPTIVE_KL_LR_MAX,
        ))

    if DR_ENABLED:
        dr_cb = DomainRandomizationCallback(
            start_steps=DR_START_STEPS,
            end_steps=DR_END_STEPS,
        )
        callbacks.append(dr_cb)

    start_time = time.time()
    reset_timesteps = (model_to_load is None)
    failure: Optional[BaseException] = None
    try:
        model.learn(total_timesteps=steps_to_train, callback=callbacks, reset_num_timesteps=reset_timesteps)
    except KeyboardInterrupt:
        print("\n\n[Notice] Training interrupted by user (Ctrl+C). Gracefully saving current model and normalization stats...")
    except BaseException as exc:  # noqa: BLE001 - save before propagating, then re-raise
        # A crash (dead worker, bad draw, OOM) must not also lose the run: on a 10M-step
        # job the checkpoint on disk can be hours old. Save first, report, then exit
        # non-zero so the failure is still visible to whatever launched this.
        failure = exc
        import traceback as _tb
        print("\n\n[Error] Training aborted by an exception. Saving the current model and stats first...")
        _tb.print_exc()

    elapsed = time.time() - start_time

    final_model_path = os.path.join(_PROJECT_ROOT, model_name)
    try:
        model.save(final_model_path)
        vec_env.save(stats_path)
    except Exception:
        import traceback as _tb
        print("[Error] Failed to save the model and/or stats:")
        _tb.print_exc()

    # Final metrics plot generation
    metrics_cb.plot_metrics()
    # Also save a copy inside logs/ directory
    metrics_cb.plot_path = os.path.join(save_dir, f"training_curves{_run_suffix}.png")
    metrics_cb.plot_metrics()

    fps = max(1, model.num_timesteps - initial_steps) / max(1e-6, elapsed)
    print(f"\n{'='*65}")
    print(f"Training Session Ended in {elapsed:.1f}s ({fps:.0f} steps/s)")
    print(f"  Lifetime Steps : {model.num_timesteps:,}")
    print(f"  Saved Model    : {final_model_path}.zip")
    print(f"  Saved Stats    : {stats_path}")
    print(f"{'='*65}\n")

    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    train()
