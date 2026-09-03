from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
from gymnasium import spaces
# pyrefly: ignore [missing-import]
from stable_baselines3 import PPO
# pyrefly: ignore [missing-import]
from stable_baselines3.common.env_util import make_vec_env
# pyrefly: ignore [missing-import]
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# pyrefly: ignore [missing-import]
import mujoco
# pyrefly: ignore [missing-import]
import mujoco.viewer

# Ensure Simulation directory is in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from quadFiles.quad_mujoco import QuadcopterMuJoCo
from utils.windModel import Wind
from utils.mixer import mixerFM
import utils
import config


# ======================================================================================
# GLOBAL CONSTANTS & STAGE DEFINITIONS
# ======================================================================================

STAGE_NAMES: Dict[int, str] = {
    1: "Stage 1: Hover Mastery at [0, 0, 1.2m]",
    2: "Stage 2: 3D Waypoint Tracking",
    3: "Stage 3: Acrobatic 360° Roll Flip & Recovery",
    4: "Stage 4: Observation Noise (Sim-to-Real Robustness)",
    5: "Stage 5: Dynamic Wind Disturbance & Gust Rejection",
}

# Flip completion threshold: rotating through 180° inverted and righting past ~288°
FLIP_COMPLETE_THRESHOLD: float = 1.60 * np.pi


# ======================================================================================
# 1. CUSTOM GYMNASIUM ENVIRONMENT (MUJOCO BACKEND)
# ======================================================================================

class CustomQuadEnv(gym.Env):
    """
    Quadcopter Gymnasium Environment backed by the MuJoCo C-physics engine.
    Optimized for high-throughput parallel training on Apple Silicon (M4).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        action_mode: str = "motor",      # "thrust_moment" (Decoupled thrust + 3D moments) or "motor"
        episode_seconds: float = 15.0,    # Maximum episode length in seconds
        dt: float = 0.01,                # Physics integration step (100 Hz RL step)
        orient: str = "ENU",             # Standard East-North-Up (+Z is Up)
        random_wind: bool = False,       # Enable wind model
        wind_magnitude: float = 0.0,     # Wind speed in m/s
        spawn_altitude: float = 0.5,     # Spawn lower down (0.5m above floor)
        target_altitude: float = 1.2,    # Target hover & flip altitude (1.2m)
        curriculum_stage: int = 1,       # 1: Hover, 2: Waypoints, 3: Flip, 4: Noise, 5: Wind
    ):
        super().__init__()

        config.orient = orient
        self.action_mode = action_mode
        self.dt = float(dt)
        self.episode_seconds = float(episode_seconds)
        self.max_steps = int(np.ceil(self.episode_seconds / self.dt))
        self.random_wind = bool(random_wind)
        self.wind_magnitude = float(wind_magnitude)
        self.spawn_altitude = float(spawn_altitude)
        self.target_altitude = float(target_altitude)
        self.curriculum_stage = int(curriculum_stage)

        # Initialize Quadcopter MuJoCo C-backend once per worker
        # Re-using the C-structures across resets eliminates XML parsing overhead
        self.quad = QuadcopterMuJoCo()
        self.t: float = 0.0
        self.steps: int = 0
        self.prev_action: np.ndarray = np.zeros(4, dtype=np.float32)

        # Pre-allocated objects to eliminate heap allocations during simulation
        self.wind_random = Wind("RANDOMSINE", 2.0, 0.5, 180, -180, 20, -20)
        self.wind_none = Wind("NONE")
        self.wind: Wind = self.wind_none
        self._obs_buf = np.zeros(23, dtype=np.float32)
        self._last_delta_roll: float = 0.0
        self.gave_inverted_bonus: bool = False

        # Physical constants
        self.min_w = float(self.quad.params["minWmotor"])       # Min motor speed (0 rad/s)
        self.max_w = float(self.quad.params["maxWmotor"])       # Max motor speed (~2600 rad/s)
        self.hover_w = float(self.quad.params["w_hover"])       # Equilibrium hover speed (~1767 rad/s)
        self.mB = float(self.quad.params["mB"])                 # Quadcopter mass (~0.028 kg)
        self.g = float(self.quad.params["g"])                   # Gravitational acceleration (9.81 m/s^2)
        self.hover_thrust = self.mB * self.g                    # Equilibrium hover thrust (~0.275 N)

        # Torque limits for "thrust_moment" action mode (scaled for Crazyflie)
        self.max_torque_xy = 0.01  # Max roll/pitch moment (Nm)
        self.max_torque_z = 0.003  # Max yaw moment (Nm)

        # ------------------------------------------------------------------------------
        # ACTION SPACE: Normalized 4D vector in [-1.0, 1.0]
        # ------------------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------------------
        # OBSERVATION SPACE
        # ------------------------------------------------------------------------------
        obs_dim = self._get_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------------------
        # CUSTOM TASK VARIABLES (Acrobatic Flip & Waypoint Tracking)
        # ------------------------------------------------------------------------------
        self.boundary_radius: float = 1.2
        self.accumulated_roll = 0.0
        self.accumulated_pitch = 0.0
        self.accumulated_yaw = 0.0
        self.reached_altitude = False
        self.completed_roll = False
        self.gave_climb_bonus = False
        self.gave_completion_bonus = False

        # Sequential 3D waypoints for Stage 2 (Waypoint Tracking)
        self.waypoints = [
            np.array([0.0, 0.0, 1.2], dtype=np.float32),
            np.array([0.4, 0.3, 1.3], dtype=np.float32),
            np.array([-0.3, 0.4, 1.0], dtype=np.float32),
            np.array([0.3, -0.4, 1.2], dtype=np.float32),
            np.array([-0.4, -0.3, 1.1], dtype=np.float32),
        ]
        self.waypoint_idx = 0
        self.waypoints_completed = 0

        # In ENU coordinates, spawn at 0.5m, climb and flip at 1.2m target altitude
        self.initial_pos = np.array([0.0, 0.0, self.spawn_altitude], dtype=np.float32)
        self.target_state = np.array([0.0, 0.0, self.target_altitude], dtype=np.float32)

        # Configure initial stage dynamics
        self.set_curriculum_stage(self.curriculum_stage)

    def set_curriculum_stage(self, stage: int) -> None:
        """
        Dynamically update curriculum stage (callable across SubprocVecEnv).
        Stage 1: Hover Mastery at [0, 0, 1.2m]
        Stage 2: 3D Waypoint Tracking
        Stage 3: Acrobatic 360° Flip & Recovery
        Stage 4: Observation Noise (Sim-to-Real Robustness)
        Stage 5: Dynamic Wind Disturbance & Gust Rejection
        """
        self.curriculum_stage = int(stage)
        if self.curriculum_stage == 5 or self.random_wind:
            self.wind = Wind("RANDOMSINE", 2.0, 0.5, 180, -180, 20, -20)
        else:
            self.wind = Wind("NONE")

    # ----------------------------------------------------------------------------------
    # OBSERVATION DIMENSION
    # ----------------------------------------------------------------------------------
    def _get_observation_dim(self) -> int:
        """
        Return the total number of features in your observation vector.
        3 (rel_target) + 3 (pos) + 4 (quat) + 3 (vel) + 3 (omega) + 4 (prev_action)
        + 1 (flip_progress) + 1 (stage_reached) + 1 (stage_completed) = 23
        """
        return 23

    # ==================================================================================
    # HOOK 1: OBSERVATION FUNCTION
    # ==================================================================================
    def _compute_observation(self) -> np.ndarray:
        """
        Construct and return the observation array passed to the policy.

        AVAILABLE QUADCOPTER STATE ATTRIBUTES:
          - self.quad.pos   : [x, y, z] in meters (+Z is Up in ENU)
          - self.quad.vel   : [vx, vy, vz] in m/s (Linear velocity in world frame)
          - self.quad.quat  : [w, x, y, z] (Attitude quaternion)
          - self.quad.euler : [phi, theta, psi] in radians (Roll, Pitch, Yaw)
          - self.quad.omega : [p, q, r] in rad/s (Angular body rates)
          - self.quad.wMotor: [wM1, wM2, wM3, wM4] in rad/s (Current motor speeds)
          - self.prev_action: [a0, a1, a2, a3] (Action taken at previous step)
        """
        assert self.quad is not None

        # Rotation matrix from body to world frame (3x3)
        # R.T rotates world vectors into the quadcopter body frame (rotation-invariant)
        R = self.quad.dcm
        rel_target_body = R.T @ (self.target_state - self.quad.pos)
        vel_body = R.T @ self.quad.vel

        # Normalized flip progress from 0.0 (not started) to 1.0 (completed flip)
        flip_progress = np.clip(abs(self.accumulated_roll) / FLIP_COMPLETE_THRESHOLD, 0.0, 1.0)
        stage_reached = 1.0 if self.reached_altitude else 0.0
        stage_completed = 1.0 if self.completed_roll else 0.0

        # In-place buffer assignment eliminates 9 dynamic array allocations per step
        buf = self._obs_buf
        buf[0:3] = rel_target_body
        buf[3:6] = self.quad.pos
        buf[6:10] = self.quad.quat
        buf[10:13] = vel_body
        buf[13:16] = self.quad.omega
        buf[16:20] = self.prev_action
        buf[20] = flip_progress
        buf[21] = stage_reached
        buf[22] = stage_completed

        # Stages 4 & 5: Add sensor observation noise (Sim-to-Real Domain Randomization)
        if self.curriculum_stage >= 4:
            noise_scale = np.array([
                0.015, 0.015, 0.015,         # rel_target_body (1.5 cm)
                0.015, 0.015, 0.015,         # pos (1.5 cm)
                0.008, 0.008, 0.008, 0.008,  # quat
                0.04, 0.04, 0.04,            # vel_body (4 cm/s)
                0.08, 0.08, 0.08,            # omega (0.08 rad/s ~ 4.5 deg/s)
                0.0, 0.0, 0.0, 0.0,          # prev_action (exact)
                0.0, 0.0, 0.0,               # flags (exact)
            ], dtype=np.float32)
            return (buf + np.random.normal(0.0, noise_scale, size=23)).astype(np.float32)

        return buf.copy()

    def _compute_reward(self, action: np.ndarray) -> float:
        assert self.quad is not None

        pos = self.quad.pos
        vel = self.quad.vel
        omega = self.quad.omega
        roll, pitch, yaw = self.quad.euler
        dcm = self.quad.dcm

        target_roll = 2.0 * np.pi
        accum_roll = abs(self.accumulated_roll)

        # Position tracking relative to active target waypoint
        dist_xy = float(np.linalg.norm(pos[:2] - self.target_state[:2]))
        dist_z = float(abs(pos[2] - self.target_state[2]))
        dist_3d = float(np.linalg.norm(pos - self.target_state))

        if self.curriculum_stage == 1:
            # =========================================================================
            # CURRICULUM STAGE 1: HOVER MASTERY AT TARGET SETPOINT [0, 0, 1.2m]
            # =========================================================================
            r_alive = 2.0
            p_dist_linear = -3.5 * dist_3d
            r_center_lock = 3.0 * float(np.exp(-12.0 * (dist_3d ** 2)))
            r_z = 2.0 * float(np.exp(-25.0 * (dist_z ** 2)))  # Precision vertical lock
            r_upright = 2.0 * float(max(0.0, dcm[2, 2]))
            r_vel = float(np.exp(-0.8 * float(np.sum(np.square(vel)))))
            r_spin = float(np.exp(-0.2 * float(np.sum(np.square(omega)))))
            reward = r_alive + p_dist_linear + r_center_lock + r_z + r_upright + r_vel + r_spin

        elif self.curriculum_stage == 2:
            # =========================================================================
            # CURRICULUM STAGE 2: 3D WAYPOINT TRACKING
            # =========================================================================
            r_alive = 2.0
            p_dist_linear = -3.0 * dist_3d
            p_z = -4.0 * dist_z                               # Direct vertical pull to eliminate altitude droop
            r_center_lock = 3.5 * float(np.exp(-10.0 * (dist_3d ** 2)))
            r_z = 2.5 * float(np.exp(-25.0 * (dist_z ** 2)))  # Precision vertical lock
            r_upright = 2.0 * float(max(0.0, dcm[2, 2]))
            r_spin = float(np.exp(-0.2 * float(np.sum(np.square(omega)))))

            reward = r_alive + p_dist_linear + p_z + r_center_lock + r_z + r_upright + r_spin

        else:
            # =========================================================================
            # CURRICULUM STAGES 3, 4, 5: ACROBATIC 360° ROLL FLIP & RECOVERY
            # (Stage 3: Clean, Stage 4: +Sensor Noise, Stage 5: +Wind Turbulence)
            # =========================================================================
            if not self.reached_altitude:
                # Sub-phase 1: Climb to safe altitude (0.5m -> 1.0m)
                climb_progress = float(np.clip((pos[2] - self.spawn_altitude) / (self.target_altitude - self.spawn_altitude), 0.0, 1.0))
                r_climb = 3.0 * climb_progress
                r_vz = 1.0 * float(np.clip(vel[2], -0.5, 1.0))
                r_upright = 1.5 * float(max(0.0, dcm[2, 2]))
                p_drift_xy = -1.5 * dist_xy
                p_spin = -0.05 * float(np.clip(np.sum(np.square(omega)), 0.0, 40.0))
                reward = 1.5 + r_climb + r_vz + r_upright + p_drift_xy + p_spin

            elif not self.completed_roll:
                # Sub-phase 2: Execute 360° roll flip at altitude
                climb_bonus = 0.0
                if not self.gave_climb_bonus:
                    climb_bonus = 20.0
                    self.gave_climb_bonus = True

                # Inversion milestone bonus (+35.0) awarded when drone completes the first half (180°)
                inverted_bonus = 0.0
                if not self.gave_inverted_bonus and accum_roll >= np.pi:
                    inverted_bonus = 35.0
                    self.gave_inverted_bonus = True

                # Potential-based step progress: reward ONLY angular displacement achieved THIS step
                r_step_progress = 50.0 * (self._last_delta_roll / FLIP_COMPLETE_THRESHOLD)

                # Explosive snap-roll rate: strongly reward fast roll (> 15-25 rad/s)
                r_roll_rate = 12.0 * float(np.clip(omega[0] / 20.0, 0.0, 2.5))
                # Sluggish penalty: penalize slow rolling (< 10 rad/s) which causes gravity drop
                p_sluggish = -4.0 * float(max(0.0, 1.0 - omega[0] / 10.0))
                p_reverse = -4.0 * float(np.clip(-omega[0] / 10.0, 0.0, 2.0))
                p_parasitic = -0.02 * float(np.clip(omega[1]**2 + omega[2]**2, 0.0, 50.0))
                p_drift_xy = -0.5 * dist_xy
                reward = climb_bonus + inverted_bonus + r_step_progress + r_roll_rate + p_sluggish + p_reverse + p_parasitic + p_drift_xy

            else:
                # Sub-phase 3: Brake, return to target setpoint, and precision hover
                bonus = 0.0
                if not self.gave_completion_bonus:
                    bonus = 100.0  # Big jackpot for completing the full 360° rotation
                    self.gave_completion_bonus = True

                r_hover_alive = 2.0
                p_dist_linear = -4.0 * dist_3d
                r_center_lock = 3.0 * float(np.exp(-10.0 * (dist_3d ** 2)))
                r_upright = 2.0 * float(max(0.0, dcm[2, 2]))
                r_brake = float(np.exp(-0.15 * (np.linalg.norm(omega) ** 2)))
                r_vel = float(np.exp(-0.5 * (np.linalg.norm(vel) ** 2)))

                reward = (
                    bonus
                    + r_hover_alive
                    + p_dist_linear
                    + r_center_lock
                    + r_upright
                    + r_brake
                    + r_vel
                )

        # Action smoothness penalty
        d_action = action - self.prev_action
        p_act = -0.02 * float(np.sum(np.square(d_action)))
        reward += p_act

        return float(reward)

    def _check_termination(self) -> bool:
        """
        Check if episode should terminate early due to non-finite state,
        ground crash, or breaching the arena containment volume.
        """
        assert self.quad is not None

        # Check for NaN / non-finite state
        if not np.all(np.isfinite(self.quad.state)):
            return True

        # Horizontal containment:
        # In Stage 2, waypoints span out to 0.4m, so allow up to 2.2m from center
        max_dist_xy = 2.2 if self.curriculum_stage == 2 else 1.2
        dist_xy_origin = float(np.linalg.norm(self.quad.pos[:2]))
        if dist_xy_origin > max_dist_xy:
            return True

        # Vertical ceiling bound: must not exceed 2.2m
        if self.quad.pos[2] > 2.2:
            return True

        # Ground collision detection: MuJoCo contact with floor or low altitude (< 0.05m)
        if self.quad.check_ground_contact():
            return True

        # Anti-stalling and anti-hesitation timeouts only active in acrobatic flip stages (Stage 3+)
        if self.curriculum_stage >= 3:
            # Anti-camping timeout: drone must climb and reach safe altitude (0.95m) within 2.5s (250 steps)
            if not self.reached_altitude and self.steps > 250:
                return True

            # Anti-hesitation timeout: drone must execute flip within 3.0s (300 steps)
            if self.reached_altitude and not self.completed_roll and self.steps > 300:
                return True

        return False

    # =================================================================================
    # ENVIRONMENT RESET
    # =================================================================================
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets environment state at the beginning of each episode.
        """
        super().reset(seed=seed)
        options = options or {}

        # Reset simulation clock and step count
        self.t = 0.0
        self.steps = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

        # Wind dynamics (reusing pre-allocated wind objects to eliminate heap allocations)
        if self.curriculum_stage == 5 or self.random_wind:
            self.wind = self.wind_random
        else:
            self.wind = self.wind_none

        # Reset MuJoCo C physics state without reallocating
        # In Stage 2+, drone has already mastered hover/takeoff; spawn at 1.0m to begin flying immediately
        spawn_z = 1.2 if self.curriculum_stage >= 3 else self.spawn_altitude
        spawn_pos = np.array([0.0, 0.0, spawn_z], dtype=np.float64)
        self.quad.reset(pos=spawn_pos)

        # Set target state based on curriculum stage
        self.waypoint_idx = 0
        self.waypoints_completed = 0
        if self.curriculum_stage == 2:
            self.target_state = self.waypoints[0].copy()
        else:
            self.target_state = np.array([0.0, 0.0, self.target_altitude], dtype=np.float32)

        # Position target marker in MuJoCo viewer
        self.quad.set_target_marker(self.target_state)

        self.accumulated_roll = 0.0
        self.accumulated_pitch = 0.0
        self.accumulated_yaw = 0.0
        self._last_delta_roll = 0.0
        self.reached_altitude = bool(self.curriculum_stage >= 3 and spawn_z >= 0.95)
        self.completed_roll = False
        self.gave_climb_bonus = False
        self.gave_inverted_bonus = False
        self.gave_completion_bonus = False

        obs = self._compute_observation()
        info = {"target_state": self.target_state.copy()}
        return obs, info

    # ----------------------------------------------------------------------------------
    # ENVIRONMENT STEPPING (Converts action -> physics -> next state)
    # ----------------------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.quad is not None
        assert self.wind is not None

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Action mapping to physical motor commands (rad/s)
        if self.action_mode == "thrust_moment":
            throttle_pct = 0.5 * (action[0] + 1.0)
            thrust = throttle_pct * self.quad.params["maxThr"]
            moments = np.array([
                action[1] * self.max_torque_xy,
                action[2] * self.max_torque_xy,
                action[3] * self.max_torque_z,
            ])
            motor_cmd = mixerFM(self.quad, thrust, moments)
        else:
            motor_cmd = np.where(
                action >= 0.0,
                self.hover_w + action * (self.max_w - self.hover_w),
                self.hover_w + action * (self.hover_w - self.min_w),
            )

        # MuJoCo high-speed C physics integration step
        self.quad.update(self.t, self.dt, motor_cmd, self.wind)
        self.t += self.dt
        self.steps += 1

        # Stage 2: Waypoint sequencing & achievement bonus
        wp_bonus = 0.0
        if self.curriculum_stage == 2:
            dist_wp = float(np.linalg.norm(self.quad.pos - self.target_state))
            if dist_wp < 0.35:
                wp_bonus = 25.0
                self.waypoints_completed += 1
                self.waypoint_idx = (self.waypoint_idx + 1) % len(self.waypoints)
                self.target_state = self.waypoints[self.waypoint_idx].copy()
                self.quad.set_target_marker(self.target_state)

        # Unlock flip once drone has climbed to safe altitude (>= 0.95m) in Stages 3+
        self._last_delta_roll = 0.0
        if self.curriculum_stage >= 3:
            if not self.reached_altitude and self.quad.pos[2] >= 0.95:
                self.reached_altitude = True

            # Monotonic forward roll flip progress once drone has ascended to safe altitude
            if self.reached_altitude and not self.completed_roll:
                delta_roll = max(0.0, float(self.quad.omega[0] * self.dt))
                self.accumulated_roll += delta_roll
                self._last_delta_roll = delta_roll
                if self.accumulated_roll >= FLIP_COMPLETE_THRESHOLD:
                    self.completed_roll = True

        obs = self._compute_observation()
        reward = self._compute_reward(action) + wp_bonus
        terminated = self._check_termination()
        truncated = bool(self.steps >= self.max_steps)

        if terminated:
            reward -= 30.0

        info = {
            "t": self.t,
            "position": self.quad.pos.copy(),
            "velocity": self.quad.vel.copy(),
            "quat": self.quad.quat.copy(),
            "omega": self.quad.omega.copy(),
            "motor_cmd": motor_cmd.copy(),
            "accumulated_roll": self.accumulated_roll,
            "flip_completed": self.completed_roll,
            "waypoints_completed": self.waypoints_completed,
            "curriculum_stage": self.curriculum_stage,
        }

        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info


# ======================================================================================
# 2. CURRICULUM CALLBACK & TRAINING PIPELINE (OPTIMIZED FOR APPLE SILICON M4)
# ======================================================================================

class CurriculumCallback(BaseCallback):
    """
    Automated curriculum progression callback for multi-stage quadcopter flight.
    Evaluates rolling performance and automatically advances across stages:
      Stage 1 (Hover) -> Stage 2 (Waypoints) -> Stage 3 (Flip) -> Stage 4 (Noise) -> Stage 5 (Wind)
    """
    def __init__(
        self,
        check_freq: int = 10_240,
        verbose: int = 1,
        auto_advance: bool = True,
        stats_path: Optional[str] = None,
        initial_stage: int = 1,
        target_stage: int = 5,
        min_stage_steps: int = 40_000,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.auto_advance = auto_advance
        self.stats_path = stats_path
        self.current_stage = initial_stage
        self.target_stage = target_stage
        self.min_stage_steps = min_stage_steps
        self.stage_start_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_flips = []
        self.episode_waypoints = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                if "flip_completed" in info:
                    self.episode_flips.append(float(info["flip_completed"]))
                if "waypoints_completed" in info:
                    self.episode_waypoints.append(float(info["waypoints_completed"]))

        if self.n_calls % self.check_freq == 0 and len(self.episode_lengths) >= 20:
            mean_len = float(np.mean(self.episode_lengths[-50:]))
            mean_rew = float(np.mean(self.episode_rewards[-50:]))
            flip_rate = float(np.mean(self.episode_flips[-50:])) if self.episode_flips else 0.0
            mean_wps = float(np.mean(self.episode_waypoints[-50:])) if self.episode_waypoints else 0.0
            steps_in_stage = self.num_timesteps - self.stage_start_step

            if self.verbose:
                print(f"\n--- [Curriculum Monitor @ Step {self.num_timesteps:,} | Stage Step {steps_in_stage:,}] {STAGE_NAMES.get(self.current_stage, '')} ---")
                print(f"  EpLen Mean   : {mean_len:.1f} / 800")
                print(f"  EpRew Mean   : {mean_rew:.1f}")
                if self.current_stage >= 3:
                    print(f"  Flip Success : {flip_rate*100:.1f}%")
                if self.current_stage == 2:
                    print(f"  Waypoints/Ep : {mean_wps:.1f}")

            advance = False
            # Check mastery condition with safety minimum steps per stage
            if steps_in_stage >= self.min_stage_steps:
                if self.current_stage == 1 and mean_len >= 650 and mean_rew >= 2000:
                    advance = True
                elif self.current_stage == 2 and mean_len >= 450 and mean_wps >= 2.0:
                    advance = True
                elif self.current_stage == 3 and flip_rate >= 0.85 and mean_len >= 550:
                    advance = True
                elif self.current_stage == 4 and flip_rate >= 0.80 and mean_len >= 550:
                    advance = True
                elif self.current_stage == 5 and flip_rate >= 0.75 and mean_len >= 550:
                    advance = True

            if advance and self.auto_advance:
                if self.current_stage < self.target_stage:
                    old_stage = self.current_stage
                    self.current_stage += 1
                    next_title = STAGE_NAMES.get(self.current_stage, f"Stage {self.current_stage}").upper()

                    print(f"\n{'='*75}")
                    print(f"🎓 CURRICULUM UPGRADE: ADVANCING TO {next_title}!")
                    print(f"{'='*75}\n")

                    # Dynamically update all parallel environments
                    self.training_env.env_method("set_curriculum_stage", self.current_stage)

                    # Save milestone checkpoint
                    ckpt_name = f"curriculum_stage_{old_stage}_passed"
                    ckpt_path = os.path.join(_THIS_DIR, "..", ckpt_name)
                    self.model.save(ckpt_path)
                    print(f"Milestone checkpoint saved to: {ckpt_path}.zip")

                    # Reset stage timing and sliding windows
                    self.stage_start_step = self.num_timesteps
                    self.episode_rewards.clear()
                    self.episode_lengths.clear()
                    self.episode_flips.clear()
                    self.episode_waypoints.clear()
                else:
                    # Final target stage has been mastered! Stop training early!
                    print(f"\n{'='*75}")
                    print(f"🏆 ALL TARGET CURRICULUM STAGES COMPLETED & MASTERED!")
                    print(f"   Final Policy Converged at Step {self.num_timesteps:,}!")
                    print(f"   Stopping training early — optimal policy acquired.")
                    print(f"{'='*75}\n")
                    return False  # Signals SB3 to conclude training immediately

        return True


def train_custom_model(
    total_timesteps: int = 2_000_000,
    num_cpus: int = 8,
    model_save_name: str = "custom_quad_model",
    device: str = "cpu",
    ent_coef: float = 0.01,
    curriculum_stage: int = 1,
    curriculum_auto: bool = False,
    target_stage: int = 5,
    min_stage_steps: int = 40_000,
    load_model: Optional[str] = None,
):
    """
    Train using Stable-Baselines3 PPO with multi-core CPU parallelization.
    Leverages MuJoCo C-acceleration to maximize steps/second throughput on Mac M4.
    """
    print(f"\n{'='*70}")
    print(f"=== Starting MuJoCo Accelerated PPO Training on Mac M4 ===")
    print(f"  Curriculum     : {STAGE_NAMES.get(curriculum_stage, f'Stage {curriculum_stage}')}")
    print(f"  Target Stage   : {STAGE_NAMES.get(target_stage, f'Stage {target_stage}')}")
    print(f"  Auto-Advance   : {curriculum_auto} (Adaptive Early Stopping Enabled)")
    print(f"  Workers        : {num_cpus} parallel processes")
    print(f"  Max Timesteps  : {total_timesteps:,}")
    print(f"  Min Steps/Stage: {min_stage_steps:,}")
    print(f"  Device         : {device.upper()}")
    print(f"  Entropy Coef   : {ent_coef}")
    print(f"  Physics Engine : MuJoCo >= 3.0 (Compiled C)")
    print(f"{'='*70}\n")

    # Set single-thread torch inside worker processes to avoid CPU contention
    torch.set_num_threads(1)

    def make_env():
        return CustomQuadEnv(
            action_mode="motor",
            episode_seconds=8.0,
            dt=0.01,
            orient="ENU",
            random_wind=False,
            wind_magnitude=0.8,
            spawn_altitude=0.5,
            target_altitude=1.2,
            curriculum_stage=curriculum_stage,
        )

    def linear_schedule(initial_value: float, final_value: float = 3e-5):
        def func(progress_remaining: float) -> float:
            return final_value + progress_remaining * (initial_value - final_value)
        return func

    vec_env = make_vec_env(make_env, n_envs=num_cpus, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    stats_path = os.path.join(_THIS_DIR, "..", f"{model_save_name}_vecnormalize.pkl")

    if load_model and os.path.isfile(load_model):
        print(f"Resuming training from checkpoint: {load_model}")
        model = PPO.load(load_model, env=vec_env, device=device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=linear_schedule(3e-4, 3e-5),
            n_steps=2048,
            batch_size=512,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[256, 256]),
                log_std_init=-0.5,
            ),
            verbose=1,
            device=device,
        )

    save_dir = os.path.join(_THIS_DIR, "..", "logs")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=save_dir)

    callbacks = [checkpoint_callback]
    if curriculum_auto:
        curr_cb = CurriculumCallback(
            check_freq=10_240,
            auto_advance=True,
            stats_path=stats_path,
            initial_stage=curriculum_stage,
            target_stage=target_stage,
            min_stage_steps=min_stage_steps,
        )
        callbacks.append(curr_cb)

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    elapsed = time.time() - start_time

    final_path = os.path.join(_THIS_DIR, "..", model_save_name)
    model.save(final_path)
    vec_env.save(stats_path)

    fps = total_timesteps / max(1e-6, elapsed)
    print(f"\n{'='*70}")
    print(f"Training Complete! Total Time: {elapsed:.2f} s ({fps:.0f} steps/s)")
    print(f"Model saved to: {final_path}.zip")
    print(f"Normalization stats saved to: {stats_path}")
    print(f"{'='*70}\n")


# ======================================================================================
# 3. EVALUATION & 3D VISUALIZATION (MUJOCO PASSIVE VIEWER)
# ======================================================================================

def evaluate_and_plot(
    model_name: str = "custom_quad_model",
    episode_seconds: float = 15.0,
    show_viewer: bool = True,
    show_plots: bool = False,
    loop: bool = True,
    options: Optional[Dict[str, Any]] = None,
    curriculum_stage: int = 1,
):
    """
    Evaluate the trained policy with real-time 3D MuJoCo rendering and telemetry analysis.
    Supports continuous replay mode in the 3D viewer.
    """
    model_path = os.path.join(_THIS_DIR, "..", model_name + ".zip")
    if not os.path.isfile(model_path):
        print(f"Model '{model_name}.zip' not found.")
        print("Please train your model first:\n  python Simulation/custom_task_template.py --train\n")
        return

    print(f"Loading policy: {model_path}")
    model = PPO.load(model_path)

    print(f"Evaluation Curriculum Stage: {STAGE_NAMES.get(curriculum_stage, f'Stage {curriculum_stage}')}")

    env = CustomQuadEnv(
        action_mode="motor",
        episode_seconds=episode_seconds,
        dt=0.01,
        orient="ENU",
        random_wind=(curriculum_stage == 5),
        spawn_altitude=0.5,
        target_altitude=1.2,
        curriculum_stage=curriculum_stage,
    )

    obs, info = env.reset(options=options)

    # Load running observation normalization stats if available
    stats_path = os.path.join(_THIS_DIR, "..", f"{model_name}_vecnormalize.pkl")
    vec_norm = None
    if os.path.isfile(stats_path):
        print(f"Loading VecNormalize statistics: {stats_path}")
        dummy_vec = DummyVecEnv([lambda: env])
        vec_norm = VecNormalize.load(stats_path, dummy_vec)
        vec_norm.training = False
        vec_norm.norm_reward = False

    if model.observation_space.shape[0] != obs.shape[0]:
        print(f"\n[Error] Observation shape mismatch!")
        print(f"Model expects {model.observation_space.shape[0]} inputs, but env produces {obs.shape[0]}.")
        print("Please re-train with:\n  python Simulation/custom_task_template.py --train\n")
        return

    t_all, pos_all, vel_all, quat_all, omega_all, euler_all = [], [], [], [], [], []
    w_cmd_all, wMotor_all, thr_all, tor_all = [], [], [], []

    total_reward = 0.0

    print(f"Simulating evaluation ({episode_seconds}s episodes) with 3D visualization...")

    viewer = None
    if show_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(env.quad.model, env.quad.data)
            viewer.cam.lookat = [0.0, 0.0, 1.0]
            viewer.cam.distance = 1.6
            viewer.cam.elevation = -18.0
            viewer.cam.azimuth = 135.0
            print("Opened interactive MuJoCo 3D Viewer (Continuous Replay Mode).")
            print("  - Drag mouse to rotate / scroll to zoom")
            print("  - Press ESC or close window to exit\n")
        except Exception as e:
            print(f"Note: Could not open interactive OpenGL window ({e}). Proceeding headlessly.")
            viewer = None

    episode_idx = 1
    last_term = False
    last_trunc = False

    try:
        while True:
            # Stop if viewer was closed by user
            if viewer is not None and not viewer.is_running():
                break

            step_start = time.time()

            obs_input = vec_norm.normalize_obs(obs) if vec_norm is not None else obs
            action, _ = model.predict(obs_input, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Record telemetry for first episode (used if --plot is specified)
            if episode_idx == 1:
                t_all.append(info["t"])
                pos_all.append(info["position"])
                vel_all.append(info["velocity"])
                quat_all.append(info["quat"])
                omega_all.append(info["omega"])
                euler_all.append(env.quad.euler.copy())
                w_cmd_all.append(info["motor_cmd"])
                wMotor_all.append(env.quad.wMotor.copy())
                thr_all.append(env.quad.thr.copy())
                tor_all.append(env.quad.tor.copy())

            if viewer is not None and viewer.is_running():
                viewer.sync()
                # Maintain real-time playback speed
                elapsed_step = time.time() - step_start
                sleep_time = env.dt - elapsed_step
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if terminated or truncated:
                last_term = terminated
                last_trunc = truncated
                status = "Terminated (Crash/Boundary)" if terminated else "Completed (Time Limit)"
                print(f"[Episode {episode_idx}] Time: {info['t']:.2f}s ({env.steps} steps) | "
                      f"Reward: {total_reward:.1f} | Flip: {env.completed_roll} (Accum: {np.rad2deg(env.accumulated_roll):.1f}°) | {status}")

                # If headless or single episode requested, break
                if viewer is None or not loop:
                    break

                # 0.5s pause before looping so the viewer cleanly shows reset
                time.sleep(0.5)
                obs, info = env.reset(options=options)
                total_reward = 0.0
                episode_idx += 1
    finally:
        if viewer is not None and viewer.is_running():
            viewer.close()

    t_all = np.array(t_all)
    pos_all = np.array(pos_all)
    vel_all = np.array(vel_all)
    quat_all = np.array(quat_all)
    omega_all = np.array(omega_all)
    euler_all = np.array(euler_all)
    w_cmd_all = np.array(w_cmd_all)
    wMotor_all = np.array(wMotor_all)
    thr_all = np.array(thr_all)
    tor_all = np.array(tor_all)

    print(f"\n--- Evaluation Summary ---")
    if len(t_all) > 0:
        print(f"Simulated Time   : {t_all[-1]:.2f} s ({len(t_all)} steps)")
        print(f"Cumulative Reward: {total_reward:.2f}")
        print(f"Terminated       : {last_term}")
        print(f"Truncated        : {last_trunc}")

    if show_plots and len(t_all) > 0:
        print("\nDisplaying telemetry figures...")
        N = len(t_all)
        sDes_calc = np.zeros([N, 16])
        sDes_traj = np.zeros([N, 16])
        sDes_calc[:, 0:3] = env.target_state
        sDes_traj[:, 0:3] = env.target_state
        sDes_calc[:, 9] = 1.0

        utils.makeFigures(
            env.quad.params,
            t_all, pos_all, vel_all, quat_all, omega_all, euler_all,
            w_cmd_all, wMotor_all, thr_all, tor_all,
            sDes_traj, sDes_calc,
        )
        plt.show()


# ======================================================================================
# 4. CLI DISPATCHER
# ======================================================================================

if __name__ == "__main__":
    # macOS GUI trampoline: launch_passive requires execution under mjpython on macOS
    if (
        sys.platform == "darwin"
        and "--train" not in sys.argv
        and "--no-viewer" not in sys.argv
        and "-h" not in sys.argv
        and "--help" not in sys.argv
    ):
        is_mjpython = hasattr(mujoco.viewer, "_MJPYTHON") and mujoco.viewer._MJPYTHON is not None
        if not is_mjpython and os.environ.get("_MJP_TRAMPOLINED") != "1":
            mjpython_path = os.path.join(os.path.dirname(sys.executable), "mjpython")
            if not os.path.isfile(mjpython_path):
                import shutil
                mjpython_path = shutil.which("mjpython")
            if mjpython_path and os.path.isfile(mjpython_path):
                os.environ["_MJP_TRAMPOLINED"] = "1"
                os.execv(mjpython_path, [mjpython_path] + sys.argv)

    parser = argparse.ArgumentParser(description="Custom Quadcopter RL Task in MuJoCo")
    parser.add_argument("--train", action="store_true", help="Train the RL policy using MuJoCo")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained policy with 3D viewer")
    parser.add_argument("--plot", action="store_true", help="Display 2D telemetry matplotlib plots")
    parser.add_argument("--no-viewer", action="store_true", help="Disable 3D viewer window during eval")
    parser.add_argument("--single", action="store_true", help="Run only 1 episode instead of continuous replay")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps (default: 1,000,000)")
    parser.add_argument("--cpus", type=int, default=8, help="Number of CPU worker cores (default: 8)")
    parser.add_argument("--name", type=str, default="custom_quad_model", help="Model file save name")
    parser.add_argument("--device", type=str, default="cpu", help="Device for PPO training: 'cpu' or 'mps'")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient for PPO exploration (default: 0.01)")
    parser.add_argument("--curriculum", action="store_true", help="Enable automatic curriculum advancement across stages 1->5")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Curriculum stage to start from (1: Hover, 2: Waypoints, 3: Flip, 4: Noise, 5: Wind)")
    parser.add_argument("--target-stage", type=int, default=5, choices=[1, 2, 3, 4, 5], help="Final curriculum stage to reach before adaptive early stopping (default: 5)")
    parser.add_argument("--min-stage-steps", type=int, default=40_000, help="Minimum steps required per stage before graduating (default: 40,000)")
    parser.add_argument("--load", type=str, default=None, help="Path to checkpoint .zip to resume from")

    args = parser.parse_args()

    if args.train:
        train_custom_model(
            total_timesteps=args.steps,
            num_cpus=args.cpus,
            model_save_name=args.name,
            device=args.device,
            ent_coef=args.ent_coef,
            curriculum_stage=args.stage,
            curriculum_auto=args.curriculum,
            target_stage=args.target_stage,
            min_stage_steps=args.min_stage_steps,
            load_model=args.load,
        )
    else:
        evaluate_and_plot(
            model_name=args.name,
            episode_seconds=15.0,
            show_viewer=not args.no_viewer,
            show_plots=args.plot,
            loop=not args.single,
            curriculum_stage=args.stage,
        )
