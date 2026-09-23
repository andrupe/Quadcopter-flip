from __future__ import annotations

import os
import sys
import time
from typing import Optional, Sequence, Union

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
    ACTOR_FRAME_MODE,
    ACTOR_TOTAL_DIM,
    REF_FF_DIM,
    TOTAL_OBS_DIM,
    FLIGHT_RADIUS,
    SPAWN_Z,
    INIT_RATE_RANGE,
    INIT_VEL_RANGE,
    REWARD_CEILING_PER_STEP,
    TRACK_W_ATT,
    TRACK_W_POS,
    TRACK_W_RATE,
    TRACK_W_VEL,
    QuadFlipEnv,
)
from asymmetric_policy import AsymmetricActorCriticPolicy
# Only for the family-name vocabulary of the mixture curriculum. Cheap (numpy-only) and
# importing it means a typo in the family name fails HERE, in the trainer, instead of
# killing a worker process at the first push (see ManeuverMixCurriculumCallback).
from trajectories import TrajectoryConfig

# The frozen history encoder lives in Simulation/encoder/. Imported defensively: the
# encoder is an ADD-ON, so a missing or incomplete encoder package must degrade to
# "train without the latent" rather than stopping training from starting at all. A hard
# import here means one absent file makes the whole trainer unimportable.
try:
    from encoder.latent_obs_wrapper import LatentObsWrapper
    from encoder.latent_injector import LatentInjector
except ImportError:  # pragma: no cover - exercised by the no-encoder configuration
    LatentObsWrapper = None
    LatentInjector = None

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
#   THE GATE, then corrupt. Budget 10M -> 15M (2026-09-11) -> 30M (2026-09-16).
#
# REVISION 2026-09-16 - 30M steps, phases stretched to roughly a third each:
#   The mixture gained a second acrobatic family (v8) and CHAINS on 2026-09-15, and the
#   chain share is now ramped up over the run (see CHAIN_MIX_* below), so the same 15M
#   budget covers a strictly harder task than it was fitted to. Rather than only
#   lengthening the tail, all three phases scale: clean 6M -> 10M (Phase 1 is gated on
#   CLEAN skill nearing the gate, and on the bigger mixture that takes longer), DR ramp
#   5M -> 10M (corruption introduced more gradually, which is what the 2026-09-11
#   measurement asked for), polish 4M -> 10M. The LR ENDPOINTS are unchanged - each phase
#   simply spans 10M steps - so the schedule keeps the shape that was actually measured.
#   The entropy and log_std schedules stretch with the horizon (both are driven by
#   total_timesteps), i.e. exploration pressure also decays more slowly: deliberate on a
#   longer run, but it does make the stochastic ep_rew_mean curve look noisier for longer.
# ======================================================================================
TOTAL_TIMESTEPS: int = 30_000_000      # ~2.9-3.0 h at the measured ~3k steps/s on 8 workers
# WORKER COUNT: 8, NOT 10. Measured on this fanless 4P+6E M4 Air, 6-8 workers beat 10 in
# every sweep (2959 @ 6 vs 2421 @ 10 steps/s). Ten processes oversubscribe the machine, add
# heat, and steal time from the SERIAL PPO update - the one stage that cannot be
# parallelised. `n_steps` is raised with it so the rollout stays 20480 samples
# (8 * 2560 == 10 * 2048); changing the worker count without it silently rescales the
# minibatch-per-rollout ratio that batch_size / n_epochs were tuned against.
NUM_WORKERS: int = 8                   # Parallel CPU worker environments
MODEL_NAME: str = "quad_flip_model"    # Model output name (.zip saved in root directory)
DEVICE: str = "cpu"                    # "cpu" (recommended for Apple Silicon) or "mps"

# Optional: pin EVERY episode to one manoeuvre family (None = sample the mixture, which is
# the production behaviour). This exists for SINGLE-MANOEUVRE PROBES - "can this policy
# learn a flip at all?" - where the mixture would otherwise spend most of its samples on
# families that already work. Read at call time, so a probe script sets `T.PIN_MANEUVER`
# before calling train() and the production run is untouched.
PIN_MANEUVER: Optional[str] = None

# Training Mode:
LOAD_PREVIOUS_MODEL: bool = False      # False: Fresh training from scratch with clean slate
PREVIOUS_MODEL_PATH: Optional[str] = None  # Model checkpoint to resume from (if LOAD_PREVIOUS_MODEL=True)

LR_START: float = 3e-4                 # Phase 1: explore the mixture on clean dynamics
LR_MID: float = 1.5e-4                 # Phase 2 start, as the ADR ramp begins
LR_ADR_END: float = 5e-5               # Phase 3 start, at full DR
LR_FLOOR: float = 3e-5                 # Phase 3 end
LR_WARMUP_STEPS: int = 10_000_000      # Phase 1 duration (10M of 30M). Keep equal to DR_START_STEPS.
LR_FINAL_STABLE: bool = False          # False: decay to LR_FLOOR in Phase 3; True: hold at LR_ADR_END
CHECKPOINT_FREQ: int = 250_000         # Timesteps per worker (2M total across 8 workers)

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

# PERIODIC DETERMINISTIC EVALUATION (see DeterministicEvalCallback).
# The policy this runs is the deterministic MEAN action, on the NOMINAL plant (dr = 0), so
# the number is directly comparable to the scripted-controller gate (5.475 reward/step =
# 75% of the 7.3 ceiling). Costs ~9 families x 15 s = 13,500 env steps per call, i.e. ~0.1%
# of the steps but ~4% of the WALL-CLOCK of a 30M-step run (the eval runs in the trainer
# process, where the workers cannot make progress). Set EVAL_FREQ_STEPS above
# TOTAL_TIMESTEPS to disable it.
EVAL_FREQ_STEPS: int = 500_000
# 1 -> 5 on 2026-09-16. The env is SEEDED (`reset(seed=k)`), so with ONE episode the sweep
# re-evaluates the SAME single reference draw per family at every evaluation: perfectly
# reproducible, but a BIASED estimate of the mixture mean. Measured per-family spread at
# 14M steps was 36% (chain) to 80% (waypoints) of the ceiling, so which reference you drew
# moves the aggregate by several points - and a separate unseeded run of the same
# checkpoint read 53.7% against this callback's 47.7%. Averaging 5 draws costs ~5 s per
# sweep (~4% of a 2 h run) and is what makes the number mean what it says.
EVAL_EPISODES_PER_FAMILY: int = 5
# The sweep used to be dr=0 ONLY, so the ADR ramp's BENEFIT was never measured - only its
# cost, and even that only indirectly. A second sweep at full DR, run every Nth evaluation
# (full DR costs ~1.4 reward/step, so it is not worth paying on every point), prices both
# sides of the trade. Both sweeps use the same seeds, so the comparison is
# reference-matched.
EVAL_ROBUST_DR: float = 1.0
EVAL_ROBUST_EVERY: int = 4

# PER-CHANNEL EXPLORATION FLOORS, log_std, for the action channels
# [thrust, roll, pitch, yaw].
#
# WHY PER-CHANNEL. The four channels do NOT have comparable authority: a unit action is
# 0.5*0.6 N of collective on channel 0, +/-20 rad/s on roll/pitch, and +/-4 rad/s on yaw.
# One scalar floor therefore means four different physical noise levels, and the quantity
# that actually matters is the noise relative to each channel's TRACKING TOLERANCE (1.2
# rad/s on hover, 6.0 through a flip) - not relative to its own range, which is what the
# old uniform floor implicitly assumed. Measured consequence: the old late floor of -2.5
# is std 0.08 = 1.6 rad/s on roll/pitch, i.e. ABOVE the 1.2 rad/s hover rate tolerance, so
# the final hover precision was noise-limited by exploration rather than by the policy.
#
# The late values below put the final rate noise at ~1.0 rad/s (roll/pitch) and ~0.2 rad/s
# (yaw), and leave thrust the widest relative band because it has no graded tolerance and
# the cheapest thing to be wrong about early. The START values are unchanged from the old
# uniform -1.0: broad exploration is wanted early, and it is the END of the schedule that
# sets the precision the run can reach.
LOG_STD_FLOOR_START: tuple = (-1.0, -1.0, -1.0, -1.0)
LOG_STD_FLOOR_END: tuple = (-2.0, -3.0, -3.0, -3.0)

# Automatic Domain Randomization (ADR):
DR_ENABLED: bool = True                # Enable progressive domain randomization
DR_START_STEPS: int = 10_000_000       # Start corrupting only once CLEAN skill approaches the gate.
                                       # 2026-09-11: at the old 2M the policy was at 48% of max
                                       # and still improving ~+0.5/step per M - the ramp ate it.
                                       # 2026-09-16: 6M -> 10M with the 30M budget (the clean
                                       # phase is a third of the run, not two fifths of 15M).
DR_END_STEPS: int = 20_000_000         # Reach full 100% DR across a 10M ramp (was 5M), leaving
                                       # 10M steps of full-DR polish instead of 4M.

# Adaptive Maneuver-Specific Flight Envelope Curriculum:
ENVELOPE_CURRICULUM_ENABLED: bool = True
ENVELOPE_FREE_STEPS: int = 5_000_000        # First 5M steps: distinctly free exploration (scale = 5.0)
ENVELOPE_ANNEAL_STEPS: int = 20_000_000     # Anneal down to scale = 1.5 by step 20M
ENVELOPE_START_SCALE: float = 5.0           # Free exploration multiplier (no premature boundary termination)
ENVELOPE_END_SCALE: float = 1.5             # Target tight envelope (within 50% extra margin over maneuver demand)

# MANOEUVRE MIXTURE CURRICULUM (chains arrive toward the END of the run).
#
# `chain` is the longest and hardest item in the set: 2-4 rest-to-rest commands flown back
# to back (6-12 s), with the periodic families inside it built as eased closed loops so
# the junctions are exact. At the base draw weight it is only ~2-4% of episodes - of the
# order of 2 chains per 20480-step rollout - so early on it is thin signal while the
# policy is still learning the single families, and it is the family most likely to be
# starved by a run that can collect decent average reward from hover/orbit/waypoints.
#
# So the base mix is HELD and the chain weight is ramped up toward the end of training, by
# which point the policy has the sub-manoeuvres and the remaining gap is composing them.
#
# MEASURED (400 draws each, current sampler): weight 0.05 -> 1.8-3.9% realized, 0.15 ->
# 12.0%, 0.25 -> 16.2%, 0.35 -> 19.5%. The realized share SATURATES because chain
# candidates are the ones most often rejected and the sampler's fallback is a Hover, so
# pushing past ~0.25 mostly buys reset time (2.6 -> 4.9 ms per draw). 0.25 also raises the
# mean episode length only 4.1 -> 4.8 s, which keeps ep_rew_mean comparable with earlier
# runs. Set CHAIN_MIX_ENABLED = False for the flat mixture.
#
# WINDOW (DECIDED 2026-09-16: tied to DR_END_STEPS, not LR_WARMUP_STEPS).
# Tying the ramp to LR_WARMUP_STEPS would grow the chain share WHILE the ADR ramp is
# corrupting the dynamics - two difficulty axes at once. That is the failure this project
# has already measured: on the 2026-09-11 run, net per-step reward growth under the DR ramp
# was ~0 (+0.42 / +0.15 / ~0 per M steps), and the fix was to DELAY difficulty rather than
# to change a hyperparameter. So the default defers chains into the full-DR polish phase
# (DR_END -> TOTAL = 20M -> 30M, a 10M ramp of its own). The cost is that chains are then
# only ever learned under full corruption; set CHAIN_MIX_START_STEPS = LR_WARMUP_STEPS to
# deliberately overlap the two axes instead.
CHAIN_MIX_ENABLED: bool = True
CHAIN_WEIGHT_START: float = 0.05       # must match TrajectoryConfig.weights["chain"]
CHAIN_WEIGHT_END: float = 0.25         # ~16% of DRAWS by the end (4-6x the base share)
CHAIN_MIX_START_STEPS: int = DR_END_STEPS      # defer chains until the DR ramp is finished
CHAIN_MIX_END_STEPS: int = TOTAL_TIMESTEPS     # reach the final share at the end of the run

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
# Episode length. 8 -> 15 s on 2026-09-15: the mixture now contains multi-command CHAINS
# (up to ~12 s) and every reference must be able to finish inside one episode, otherwise
# the terminal hover never happens and the truncation bootstrap is scored on a
# mid-manoeuvre reference. Must match quad_flip_env.EPISODE_SECONDS (the env syncs its
# sampler's horizon from this) and evaluate.py's copy.
EPISODE_SECONDS: float = 15.0        # Max flight time per episode (seconds)
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


class ManeuverMixCurriculumCallback(BaseCallback):
    """
    Ramps the `chain` family's draw weight up over training.

    The chain is the longest and hardest command in the mixture (2-4 rest-to-rest
    manoeuvres with hover beats between them, 6-12 s), and at the base weights it is a
    small fraction of draws, so it is introduced progressively: the base mix is held until
    `start_steps`, then the chain weight is interpolated linearly to `weight_end` at
    `end_steps` and stays there.

    Weights are RELATIVE - `TrajectorySampler.sample` normalises them per draw - so raising
    the chain weight lowers every other family's share in proportion and this callback
    never has to touch them. Only the weight itself crosses the worker RPC
    (`env_method("set_maneuver_weight", ...)`, the same path the ADR callback uses), and
    because the sampler reads its config on every draw the change lands at the next reset.
    """

    def __init__(
        self,
        start_steps: int = CHAIN_MIX_START_STEPS,
        end_steps: int = CHAIN_MIX_END_STEPS,
        weight_start: float = CHAIN_WEIGHT_START,
        weight_end: float = CHAIN_WEIGHT_END,
        family: str = "chain",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        # Validate the name HERE. `env_method` does not surface a worker exception as a
        # worker exception: SB3's worker loop dies and the trainer sees EOFError /
        # "Broken pipe" instead of the ValueError, which is a miserable thing to debug.
        # The env refuses an unknown name too, so a hand-written script is still safe.
        if family not in TrajectoryConfig().weights:
            raise ValueError(
                f"unknown manoeuvre {family!r}; expected one of {list(TrajectoryConfig().weights)}"
            )
        self.start_steps = int(start_steps)
        self.end_steps = int(end_steps)
        self.weight_start = float(weight_start)
        self.weight_end = float(weight_end)
        self.family = str(family)
        self.current_weight: Optional[float] = None

    def _weight_at(self, steps: int) -> float:
        """Linearly interpolated draw weight for the family at `steps`."""
        if self.end_steps <= self.start_steps:
            return self.weight_end
        u = (float(steps) - self.start_steps) / max(1, self.end_steps - self.start_steps)
        u = float(np.clip(u, 0.0, 1.0))
        return self.weight_start + u * (self.weight_end - self.weight_start)

    def _push(self, weight: float) -> None:
        self.current_weight = float(weight)
        self.training_env.env_method("set_maneuver_weight", self.family, float(weight))

    def _on_training_start(self) -> None:
        # Resumes must pick the schedule up where they left off, so the initial value is
        # evaluated at the CURRENT step count rather than assumed to be `weight_start`.
        w = self._weight_at(self.num_timesteps)
        self._push(w)
        if self.verbose > 0:
            print(f"\n{'*'*65}")
            print("*** MANOEUVRE MIXTURE CURRICULUM INITIALIZED ***")
            print(f"  '{self.family}' draw weight: {self.weight_start:.3f} -> {self.weight_end:.3f} "
                  f"over steps {self.start_steps:,} to {self.end_steps:,}")
            print(f"  Initial weight: {w:.3f} at step {self.num_timesteps:,}")
            print(f"{'*'*65}\n")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        steps = self.num_timesteps
        w = self._weight_at(steps)
        # Only push when the weight has actually moved (same threshold style as the ADR
        # callback): a 0.05 -> 0.25 ramp over millions of steps changes ~6e-4 per rollout,
        # so this reports roughly every 9 rollouts instead of every one.
        if self.current_weight is None or abs(w - self.current_weight) > 5e-3 or (
            w >= self.weight_end and self.current_weight < self.weight_end
        ):
            self._push(w)
            if self.verbose > 0:
                # The reachable REALIZED share is not the nominal one: chain candidates are
                # rejected more often than any other family and the sampler's fallback is a
                # Hover, so the measured share lags the normalized weight (0.25 -> ~16% of
                # draws). env.get_maneuver_weights() reports the exact probabilities that
                # are being drawn; print the weight, which is what the schedule drives.
                print(f"  [MIX] '{self.family}' draw weight = {w:.3f} (step {steps:,})")


class AdaptiveEnvelopeCurriculumCallback(BaseCallback):
    """
    Curriculum for the relative maneuver-specific flight envelope.

    Phase 1 (steps <= free_steps, e.g. first 4M steps):
        Envelope scale is set to start_scale (5.0), keeping exploration distinctly free
        to discover coarse dynamics, rotations, and acrobatic recovery without premature boundary
        termination.
    Phase 2 (free_steps < steps <= anneal_steps, e.g. 4M to 20M steps):
        Envelope scale linearly anneals from start_scale (5.0) down to end_scale (1.5).
        Maneuver-specific bounds progressively engage:
        - Flips are ceiling-clamped to prevent ballooning and drift.
        - Slaloms and traverses are bounded to their relative path tunnels.
    Phase 3 (steps > anneal_steps, e.g. 20M to 30M steps):
        Held at end_scale (1.5) for tight precision polish under full domain randomization.
    """

    def __init__(
        self,
        free_steps: int = ENVELOPE_FREE_STEPS,
        anneal_steps: int = ENVELOPE_ANNEAL_STEPS,
        start_scale: float = ENVELOPE_START_SCALE,
        end_scale: float = ENVELOPE_END_SCALE,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.free_steps = int(free_steps)
        self.anneal_steps = int(anneal_steps)
        self.start_scale = float(start_scale)
        self.end_scale = float(end_scale)
        self.current_scale: float = self.start_scale

    def _calc_scale(self, steps: int) -> float:
        if steps <= self.free_steps:
            return self.start_scale
        if steps >= self.anneal_steps:
            return self.end_scale
        frac = (steps - self.free_steps) / max(1, self.anneal_steps - self.free_steps)
        return float(self.start_scale + frac * (self.end_scale - self.start_scale))

    def _on_training_start(self) -> None:
        self.current_scale = self._calc_scale(self.num_timesteps)
        self.training_env.env_method("set_envelope_scale", self.current_scale)
        if self.verbose > 0:
            print(f"\n{'*'*65}")
            print(f"*** RELATIVE FLIGHT ENVELOPE CURRICULUM INITIALIZED ***")
            print(f"  Free Exploration: 0 -> {self.free_steps:,} steps (scale = {self.start_scale:.1f}, unconstrained bounds)")
            print(f"  Annealing Phase : {self.free_steps:,} -> {self.anneal_steps:,} steps ({self.start_scale:.1f} -> {self.end_scale:.1f})")
            print(f"  Precision Phase : {self.anneal_steps:,} -> End (scale = {self.end_scale:.1f}, within 50% maneuver demand)")
            print(f"  Initial Scale   : {self.current_scale:.2f} at step {self.num_timesteps:,}")
            print(f"{'*'*65}\n")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        steps = self.num_timesteps
        scale = self._calc_scale(steps)
        if abs(scale - self.current_scale) > 0.05:
            self.current_scale = scale
            self.training_env.env_method("set_envelope_scale", scale)
            if self.verbose > 0:
                print(f"  [Envelope] Relative Boundary Scale = {scale:.2f} (step {steps:,})")




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
    over training, PER ACTION CHANNEL. Early training forces broad exploration to escape
    hover-only local optima; late training relaxes the floor so the policy can converge to
    precise tracking.

    Both bounds accept either a scalar (uniform, the historical behaviour) or one value per
    action channel. The default is per-channel - see the LOG_STD_FLOOR_START/END comment for
    why a single scalar cannot be right across channels whose authority differs by 5x and
    whose tolerances differ by 5x in the other direction.

    References:
      - Andrychowicz et al. (2020): free log_std learning is optimal for continuous control
      - Kaufmann et al. (2020) 'Deep Drone Acrobatics': no std clamping, relies on reward shaping
    We use a decaying floor as a compromise: prevents early collapse while allowing late precision.
    """

    def __init__(
        self,
        min_log_std_start: Union[float, Sequence[float]] = LOG_STD_FLOOR_START,
        min_log_std_end: Union[float, Sequence[float]] = LOG_STD_FLOOR_END,
        max_log_std: float = 0.0,
        total_timesteps: int = TOTAL_TIMESTEPS,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.min_log_std_start = min_log_std_start
        self.min_log_std_end = min_log_std_end
        self.max_log_std = float(max_log_std)
        self.total_timesteps = int(total_timesteps)

    @staticmethod
    def _as_channel_vector(value: Union[float, Sequence[float]], n: int) -> np.ndarray:
        """Broadcast a scalar bound to `n` channels; a sequence must already have width n."""
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.full(n, float(arr[0]))
        if arr.size != n:
            raise ValueError(f"log_std bound has {arr.size} values but the action has {n} channels")
        return arr

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        policy = getattr(self.model, "policy", None)
        log_std = getattr(policy, "log_std", None)
        if log_std is None:
            return
        n = int(log_std.shape[0])
        progress = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
        start = self._as_channel_vector(self.min_log_std_start, n)
        end = self._as_channel_vector(self.min_log_std_end, n)
        floor = start + progress * (end - start)
        with torch.no_grad():
            # BOTH bounds must be tensors: clamp_() rejects a mixed signature
            # (min=Tensor, max=float) with "invalid combination of arguments", which is how
            # this was caught - the floor became per-channel and the ceiling was still a
            # scalar. Keep them the same kind.
            floor_t = torch.as_tensor(floor, dtype=log_std.dtype, device=log_std.device)
            ceil_t = torch.full_like(log_std, float(self.max_log_std))
            log_std.data.clamp_(min=floor_t, max=ceil_t)


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


class DeterministicEvalCallback(BaseCallback):
    """
    Periodic DETERMINISTIC, per-family evaluation on the nominal plant.

    WHY THIS EXISTS. `ep_rew_mean` is produced by the STOCHASTIC policy under a forced std
    floor, so it systematically understates skill (measured 0.5-1.3 reward/step below a
    deterministic rollout), and it is an average over the whole MIXTURE - so one family
    that is failing, or that the policy has quietly given up on, is invisible in it. Every
    "is the policy at the gate?" answer in this project's history had to be produced by a
    separate script run by hand against a checkpoint, which is how a schedule was once
    retuned on a number that had been misread (stochastic curve vs deterministic gate).

    GRADED ON THE NOMINAL PLANT (dr = 0). The ADR ramp changes the PLANT, not the policy,
    so a curve that mixes both is not comparable across a run. The scripted-controller gate
    it is compared against (5.475 reward/step = 75% of REWARD_CEILING_PER_STEP) is also a
    dr = 0 number.

    COST: n_families * n_episodes * episode_seconds at 100 Hz, all in the trainer process:
    with the defaults 9 * 5 * 15 s = 45 episodes, i.e. ~20k steps / ~5 s (~4% of a 2 h
    run), plus the dr=1 sweep on every EVAL_ROBUST_EVERY-th evaluation. The first
    evaluation is scheduled at freq_steps, so short smoke runs never pay it.

    Written to its OWN csv: the main metrics csv has a fixed 4-column schema that the
    plotting tooling and the tracked history depend on, so widening it would break both.
    """

    # The four tracking terms, in the order `_compute_reward` evaluates them. The order is
    # load-bearing: the decomposition assumes one kernel call per term per control step.
    TERM_NAMES = ("pos", "vel", "att", "rate")

    def __init__(
        self,
        freq_steps: int = EVAL_FREQ_STEPS,
        episodes_per_family: int = EVAL_EPISODES_PER_FAMILY,
        families: Optional[Sequence[str]] = None,
        episode_seconds: float = EPISODE_SECONDS,
        encoder_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        robust_dr: float = EVAL_ROBUST_DR,
        robust_every: int = EVAL_ROBUST_EVERY,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.freq_steps = int(freq_steps)
        self.episodes_per_family = int(episodes_per_family)
        self.families = list(families) if families is not None else list(TrajectoryConfig().weights)
        self.episode_seconds = float(episode_seconds)
        self.encoder_path = encoder_path
        self.csv_path = csv_path
        self.robust_dr = float(robust_dr)
        self.robust_every = max(1, int(robust_every))
        # Its own file: the nominal curve is the one compared against the scripted gate and
        # it must stay a clean dr=0 series. Nothing reads either file programmatically (the
        # columns are widened below), so a second file costs nothing and keeps them apart.
        self.robust_csv_path = (
            csv_path[:-4] + "_robust.csv" if csv_path and csv_path.endswith(".csv") else None
        )
        self.eval_count = 0
        # Start at freq_steps, NOT 0: otherwise the very first rollout boundary evaluates an
        # untrained policy and every smoke run pays for a full sweep it did not ask for.
        self.next_eval_step = int(freq_steps)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps < self.next_eval_step:
            return
        self.next_eval_step = self.num_timesteps + self.freq_steps
        try:
            self._evaluate_and_log()
        except Exception as exc:  # noqa: BLE001
            # A diagnostic must NEVER be able to end the run - same rule as the flight
            # telemetry log. Report and continue training.
            print(f"  [Eval] deterministic evaluation skipped: {type(exc).__name__}: {exc}")

    def _make_injector(self):
        """The z injector, or None when this run has no encoder attached."""
        if LatentInjector is None or not self.encoder_path or not os.path.isfile(self.encoder_path):
            return None
        # ref_ff_dim is REQUIRED here: without it the injector assembles [o_t | z | aux | priv]
        # and the policy's [:actor_obs_dim] slice silently picks up aux channels where it
        # expects the reference feed-forward. The widths are still "wide enough", so nothing
        # raises - the evaluation just measures a different observation than training uses.
        return LatentInjector(self.encoder_path, z_dim=Z_DIM, ref_ff_dim=REF_FF_DIM)

    def _evaluate_and_log(self) -> None:
        injector = self._make_injector()
        # Fresh env, never the training env: this must not touch the workers' state, and it
        # is constructed with telemetry=False so it shares the training hot path.
        env = QuadFlipEnv(episode_seconds=self.episode_seconds, telemetry=False)
        step = int(self.num_timesteps)

        per_family, total_steps, total_ret, terms = self._sweep(env, injector, 0.0, True)
        overall = total_ret / max(1, total_steps)
        pct = 100.0 * overall / REWARD_CEILING_PER_STEP

        if self.verbose > 0:
            print(f"\n  [Eval @ {step:,}] deterministic, dr=0, {total_steps} steps: "
                  f"{overall:.3f} reward/step = {pct:.1f}% of {REWARD_CEILING_PER_STEP:.2f} "
                  f"(gate 75%)")
            ranked = sorted(per_family.items(), key=lambda kv: kv[1])
            worst = "  ".join(f"{k}={v:.2f}" for k, v in ranked[:4])
            best = "  ".join(f"{k}={v:.2f}" for k, v in ranked[-3:])
            print(f"          lowest: {worst}")
            print(f"          highest: {best}", flush=True)
            if terms:
                w = {"pos": TRACK_W_POS, "vel": TRACK_W_VEL,
                     "att": TRACK_W_ATT, "rate": TRACK_W_RATE}
                parts = "  ".join(
                    f"{n}={w[n] * terms['kern'][i]:.2f}/{w[n]:.1f}"
                    for i, n in enumerate(self.TERM_NAMES))
                errs = "  ".join(f"{n}_err={terms['err'][i]:.3f}"
                                 for i, n in enumerate(self.TERM_NAMES))
                print(f"          terms: {parts}   (of their own weight)")
                print(f"          {errs}", flush=True)

        self._append_csv(self.csv_path, step, total_steps, overall, pct, per_family, terms)

        # --- robustness sweep: what the ADR ramp BUYS, not just what it costs ----------
        self.eval_count += 1
        if self.robust_dr > 0.0 and self.eval_count % self.robust_every == 0:
            r_fam, r_steps, r_ret, _ = self._sweep(env, injector, self.robust_dr, False)
            r_overall = r_ret / max(1, r_steps)
            r_pct = 100.0 * r_overall / REWARD_CEILING_PER_STEP
            if self.verbose > 0:
                print(f"  [Eval @ {step:,}] ROBUSTNESS dr={self.robust_dr:.2f}, {r_steps} steps: "
                      f"{r_overall:.3f} reward/step = {r_pct:.1f}%"
                      f"   (nominal cost {pct - r_pct:+.1f} pts)", flush=True)
            self._append_csv(self.robust_csv_path, step, r_steps, r_overall, r_pct,
                             r_fam, None)

    def _sweep(self, env, injector, dr: float, inject_terms: bool):
        """
        One deterministic pass over every family at a FIXED dr.

        Returns (per_family_mean_reward, total_steps, total_reward, terms) where `terms` is
        None unless instrumenting was on and any step was observed, else
        {"n", "err": [4], "kern": [4]} averaged over the observed steps.

        The per-term decomposition is the reason this exists at all: a single reward/step
        number cannot separate a POSITION deficit (weight 3.0) from an ATTITUDE one (2.0),
        nor say whether randomization costs accuracy or smoothness. Measured at 14M steps,
        position and attitude were 86% of the whole gap, and the entire dr=1 cost landed in
        the RATE and ACTION terms (w_err 1.56 -> 4.94 rad/s) with position unchanged.
        """
        env.set_dr_level(dr)
        rec: list = []
        orig = None
        if inject_terms:
            import quad_flip_env as _qfe

            orig = _qfe._tracking_kernel

            def _spy(err, tol):
                v = orig(err, tol)
                rec.append((float(err), float(tol), float(v)))
                return v

            _qfe._tracking_kernel = _spy

        per_family: dict = {}
        total_ret, total_steps = 0.0, 0
        err_sum, kern_sum, n_terms = [0.0] * 4, [0.0] * 4, 0
        try:
            for kind in self.families:
                fam_ret, fam_steps = 0.0, 0
                for k in range(self.episodes_per_family):
                    obs, _info = env.reset(seed=k, options={"maneuver": kind})
                    if injector is not None:
                        injector.reset()
                    done = False
                    while not done:
                        o = injector.inject(obs) if injector is not None else obs
                        action, _ = self.model.predict(o, deterministic=True)
                        rec.clear()
                        obs, reward, term, trunc, _info = env.step(action)
                        # `_compute_reward` calls _tracking_kernel exactly once per term, in
                        # the fixed order pos/vel/att/rate, so a chunk of four is one control
                        # step. Any other length is DISCARDED rather than mislabelled.
                        if inject_terms and len(rec) == 4:
                            for i in range(4):
                                err_sum[i] += rec[i][0]
                                kern_sum[i] += rec[i][2]
                            n_terms += 1
                        fam_ret += float(reward)
                        fam_steps += 1
                        done = bool(term or trunc)
                per_family[kind] = fam_ret / max(1, fam_steps)
                total_ret += fam_ret
                total_steps += fam_steps
        finally:
            # A diagnostic must never leak state into the training process.
            if orig is not None:
                import quad_flip_env as _qfe

                _qfe._tracking_kernel = orig

        terms = None
        if inject_terms and n_terms:
            terms = {"n": n_terms,
                     "err": [e / n_terms for e in err_sum],
                     "kern": [k / n_terms for k in kern_sum]}
        return per_family, total_steps, total_ret, terms

    def _append_csv(self, path, step, steps, overall, pct, per_family, terms) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cols = (["timestep", "steps", "overall_per_step", "pct_of_ceiling"]
                + list(self.families)
                + [f"err_{t}" for t in self.TERM_NAMES]
                + [f"kern_{t}" for t in self.TERM_NAMES])
        # SCHEMA GUARD. Writing the header only when the file is EMPTY is not enough once the
        # schema can change: `logs/eval_metrics.csv` from the 2026-09-16 run has 13 columns,
        # this callback writes 21, so the next run would have appended wider rows under a
        # narrower header - a file whose header and rows disagree, which breaks every reader
        # and does so SILENTLY, hours into a run. An existing file with a different header is
        # ROTATED instead of appended to.
        expected = ",".join(cols)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, encoding="utf-8") as fh:
                existing = fh.readline().strip()
            if existing != expected:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                rotated = f"{path[:-4]}.{stamp}.legacy.csv"
                os.replace(path, rotated)
                print(f"  [Eval] csv schema changed ({existing.count(',') + 1} -> "
                      f"{len(cols)} columns); rotated the old file to "
                      f"{os.path.basename(rotated)}")
        header_needed = not os.path.isfile(path) or os.path.getsize(path) == 0
        row = [str(step), str(steps), f"{overall:.4f}", f"{pct:.2f}"]
        row += [f"{per_family[k]:.4f}" for k in self.families]
        if terms:
            row += [f"{v:.4f}" for v in terms["err"]] + [f"{v:.4f}" for v in terms["kern"]]
        else:
            row += [""] * (2 * len(self.TERM_NAMES))
        with open(path, "a", encoding="utf-8") as f:
            if header_needed:
                f.write(expected + "\n")
            f.write(",".join(row) + "\n")


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
            # None in every production run: the mixture is the task. A probe sets this to
            # train a single family (see PIN_MANEUVER above).
            maneuver=PIN_MANEUVER,
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
    # The actor's width is ALWAYS [o_t | (z) | ref_ff]: the reference feed-forward block is
    # actor-facing and is present with or without the encoder, while z is inserted by the
    # wrapper between the two. Keep this in step with the wrapper, whose output prefix is
    # actor_total_dim + z_dim + ref_ff_dim.
    actor_obs_dim = ACTOR_TOTAL_DIM + REF_FF_DIM
    if LatentObsWrapper is None:
        print("WARNING: the encoder package (Simulation/encoder/) is not importable.\n"
              f"         Training WITHOUT the history encoder; the actor sees {actor_obs_dim} dims.\n"
              f"         This is a valid configuration, just a less informed policy.\n")
    elif os.path.isfile(ENCODER_CHECKPOINT):
        vec_env = LatentObsWrapper(
            vec_env, encoder_path=ENCODER_CHECKPOINT, z_dim=Z_DIM, ref_ff_dim=REF_FF_DIM
        )
        # The encoder's input is o_t, whose x,y channels changed meaning with
        # quad_flip_env.ACTOR_FRAME_MODE. A checkpoint from the other convention produces a
        # plausible-looking but wrong z, and PPO would happily train on it, so it is
        # refused here (ActorInput applies the same check on the evaluation path).
        extra = getattr(vec_env, "trained_meta", None) or {}
        got_mode = str(extra.get("frame_mode", "absolute_xy (pre-2026-09-16)"))
        if got_mode != str(ACTOR_FRAME_MODE):
            raise SystemExit(
                f"\n[Error] the encoder at {ENCODER_CHECKPOINT} was trained for actor-frame\n"
                f"        mode {got_mode!r}, but this build uses {ACTOR_FRAME_MODE!r}.\n"
                f"        Re-collect the corpus and retrain the encoder first:\n"
                f"          .venv/bin/python Simulation/encoder/collect_data.py\n"
                f"          .venv/bin/python Simulation/encoder/train_encoder.py\n"
            )
        actor_obs_dim = ACTOR_TOTAL_DIM + Z_DIM + REF_FF_DIM
        print(f"Loaded history encoder from {ENCODER_CHECKPOINT}: actor sees {actor_obs_dim} dims "
              f"({ACTOR_TOTAL_DIM} obs + {Z_DIM} latent + {REF_FF_DIM} ref feed-forward), "
              f"frame mode {got_mode}")
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
        print(f"  Task           : TRAJECTORY TRACKING (hover / waypoints / figure-8 / lissajous / orbit / slalom / flip / v8 / chain)")
        if ADAPTIVE_KL_LR:
            print(f"  LR Schedule    : ADAPTIVE KL (target {ADAPTIVE_KL_DESIRED}, "
                  f"clamped to [{ADAPTIVE_KL_LR_MIN:.0e}, {ADAPTIVE_KL_LR_MAX:.0e}]) - RMA/legged_gym style")
        else:
            print(f"  LR Schedule    : Phase 1 (0 -> {LR_WARMUP_STEPS:,}): {learning_rate:.1e} -> {lr_mid:.1e} (mixture, no DR)")
            print(f"                   Phase 2 ({LR_WARMUP_STEPS:,} -> {LR_WARMUP_STEPS + anneal_steps:,}): {lr_mid:.1e} -> {lr_adr_end:.1e} (DR ramp)")
            print(f"                   Phase 3 ({LR_WARMUP_STEPS + anneal_steps:,} -> {total_timesteps:,}): {lr_adr_end:.1e} -> {lr_floor:.1e} (full-DR polish)")
    print(f"  Entropy Coef   : {ent_coef:.4f} -> {ent_coef_end:.4f} (linear; set both to 0.0 for RMA-style)")
    print(f"  Tracker Tol.   : per-manoeuvre, see TRACK_TOL in quad_flip_env.py")
    if CHAIN_MIX_ENABLED:
        print(f"  Mix Curriculum : '{'chain'}' draw weight {CHAIN_WEIGHT_START:.2f} -> {CHAIN_WEIGHT_END:.2f} "
              f"over steps {CHAIN_MIX_START_STEPS:,} to {CHAIN_MIX_END_STEPS:,} "
              f"(base mix held before that; weights are relative, others scale down)")
    else:
        print(f"  Mix Curriculum : DISABLED (flat mixture from TrajectoryConfig.weights)")
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
    if ENVELOPE_CURRICULUM_ENABLED:
        print(f"  Envelope Curr. : Free exploration (scale {ENVELOPE_START_SCALE:.1f}) 0 -> {ENVELOPE_FREE_STEPS:,} steps, "
              f"annealing to {ENVELOPE_END_SCALE:.1f} at {ENVELOPE_ANNEAL_STEPS:,} steps")
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
            # n_steps * NUM_WORKERS must stay 20480 so the batch_size/n_epochs ratio is
            # unchanged; see the NUM_WORKERS comment.
            n_steps=2560,
            batch_size=512,
            n_epochs=5,
            # GAMMA vs THE EPISODE LENGTH. At 100 Hz, 0.995 gives an effective horizon of
            # 1/(1-gamma) = 200 steps = 2.0 s, but episodes became 15 s on 2026-09-15 and a
            # chain runs up to ~12 s: the terminal hover - where full reward lives - can be
            # >1400 steps from the decision that determines whether it is reached. 0.997 is
            # a 5 s horizon, which covers a single manoeuvre and the handover into its
            # terminal hover instead of only the last fifth of a second of it.
            gamma=0.997,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            policy_kwargs=dict(
                actor_obs_dim=actor_obs_dim,
                activation_fn=nn.Tanh,
                # Actor = ONE hidden layer of 32 (pi=[32]); keep in step with
                # AsymmetricActorCriticPolicy's default and actor_input.DEFAULT_NET_ARCH.
                net_arch=dict(pi=[32], vf=[512, 256, 128]),
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
        min_log_std_start=LOG_STD_FLOOR_START,   # per-channel; see the constant's comment
        min_log_std_end=LOG_STD_FLOOR_END,
        max_log_std=0.0,
        total_timesteps=total_lifetime_steps,
    )
    ent_schedule_cb = EntCoefScheduleCallback(
        start_value=ent_coef,     # escapes the "only hover well" local optimum
        end_value=ent_coef_end,   # 0.001 classic, 0.0 RMA-style
        total_timesteps=total_lifetime_steps,
    )
    eval_cb = DeterministicEvalCallback(
        freq_steps=EVAL_FREQ_STEPS,
        episodes_per_family=EVAL_EPISODES_PER_FAMILY,
        robust_dr=EVAL_ROBUST_DR,
        robust_every=EVAL_ROBUST_EVERY,
        episode_seconds=EPISODE_SECONDS,
        encoder_path=ENCODER_CHECKPOINT,
        csv_path=os.path.join(save_dir, f"eval_metrics{_run_suffix}.csv"),
    )
    callbacks = [checkpoint_cb, vecnorm_cb, metrics_cb, std_floor_cb, ent_schedule_cb, eval_cb]

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

    if CHAIN_MIX_ENABLED:
        # Chains arrive progressively - see the CHAIN_MIX_* constants above. Uses the
        # lifetime step count like the other schedules, so it also behaves on a resume.
        callbacks.append(ManeuverMixCurriculumCallback(
            family="chain",
            start_steps=CHAIN_MIX_START_STEPS,
            end_steps=CHAIN_MIX_END_STEPS,
            weight_start=CHAIN_WEIGHT_START,
            weight_end=CHAIN_WEIGHT_END,
        ))

    if ENVELOPE_CURRICULUM_ENABLED:
        callbacks.append(AdaptiveEnvelopeCurriculumCallback(
            free_steps=ENVELOPE_FREE_STEPS,
            anneal_steps=ENVELOPE_ANNEAL_STEPS,
            start_scale=ENVELOPE_START_SCALE,
            end_scale=ENVELOPE_END_SCALE,
        ))

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
    import argparse
    parser = argparse.ArgumentParser(description="PPO Trajectory Tracking Training")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total timesteps to train")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Number of parallel worker environments")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous model checkpoint")
    args = parser.parse_args()
    train(
        total_timesteps=args.total_timesteps,
        num_workers=args.num_workers,
        load_previous_model=args.resume or LOAD_PREVIOUS_MODEL,
    )
