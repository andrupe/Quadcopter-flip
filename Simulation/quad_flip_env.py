from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# Ensure Simulation and project root directories are in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "Simulation" else _THIS_DIR
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quadFiles.quad_mujoco import QuadcopterMuJoCo
from utils.windModel import Wind
from utils.mixer import mixerFM
from utils.rate_pid import RatePIDController
import utils
import config

# ======================================================================================
# ENVIRONMENT CONFIGURATION
# ======================================================================================
TARGET_ALTITUDE: float = 1.2         # Target hover altitude post-flip (meters)
TARGET_ALTITUDE_FLIP: float = 1.6   # Target pre-flip climb altitude (meters: +25cm upward punch)
SPAWN_ALTITUDE: float = 1.2          # Quadcopter spawn altitude (meters)
SIM_DT: float = 0.01                 # Timestep in seconds (0.01s = 10ms -> 100 Hz)

EPISODE_SECONDS: float = 8.0         # Episode duration in seconds
ACTION_MODE: str = "rate_pid"        # "rate_pid" (thrust + body rate PID), "motor", or "thrust_moment"
OBS_NOISE: bool = True               # Add Gaussian sensor noise (sim-to-real domain randomization)
RANDOM_WIND: bool = True             # Add dynamic wind disturbances
MAX_WIND_SPEED: float = 1.0          # Maximum wind speed in m/s
RANDOM_INITIAL_STATE: bool = True    # Randomize spawn position, attitude tilt, and velocity
MOTOR_TAU: float = 0.025             # 1st-order motor time constant (25ms for Crazyflie coreless DC)
MOTOR_TAU_RANGE: tuple = (0.020, 0.035) # Motor tau randomization range
RANDOM_BATTERY: bool = True          # Randomize battery voltage sag / thrust scaling
OBS_LATENCY_MAX_STEPS: int = 2       # Max observation delay in steps
PITCH_DIRECTION: float = 1.0         # +1.0 for front-flip, -1.0 for back-flip
FLIP_THRESHOLD: float = 2.0 * np.pi  # Rotation angle required for a full 360° pitch flip
OBS_HISTORY_LEN: int = 3             # Number of stacked observation frames (30ms temporal context)
ACTOR_SINGLE_OBS_DIM: int = 17       # Onboard sensor observation dimension
ACTOR_TOTAL_DIM: int = ACTOR_SINGLE_OBS_DIM * OBS_HISTORY_LEN  # 51 dims
PRIVILEGED_OBS_DIM: int = 44         # Privileged simulation truth
TOTAL_OBS_DIM: int = ACTOR_TOTAL_DIM + PRIVILEGED_OBS_DIM      # 95 dims
SINGLE_OBS_DIM: int = ACTOR_SINGLE_OBS_DIM

# Real-world Hardware Distortions & Physical Asymmetries (Domain Randomization)
COM_OFFSET_MAX_XY: float = 0.0025      # ±2.5 mm off-center Center of Mass
COM_OFFSET_MAX_Z: float = 0.0030       # ±3.0 mm vertical CoM offset
PAYLOAD_MASS_MAX: float = 0.0045       # +0 to 4.5g calibrated payload
ARM_LENGTH_JITTER_MAX: float = 0.0015  # ±1.5 mm independent rotor arm length variation
MOTOR_MISMATCH_MAX: float = 0.08       # Up to 8% independent motor efficiency degradation
DYNAMIC_SAG_COEF_MAX: float = 0.07     # Up to 7% 1S LiPo dynamic voltage sag
MOTOR_TAU_DOWN_FACTOR: float = 0.50    # tau_down slower than tau_up
GYRO_BIAS_MAX: float = 0.035          # ±0.035 rad/s (~2.0 deg/s) static gyro bias

# Sensor Observation Noise (1-sigma bounds)
OBS_NOISE_POS_RANGE: tuple = (0.005, 0.015)       # ±5mm to ±15mm
OBS_NOISE_VEL_RANGE: tuple = (0.020, 0.060)       # ±2cm/s to ±6cm/s
OBS_NOISE_OMEGA_RANGE: tuple = (0.030, 0.150)     # ±1.7°/s to ±8.6°/s
OBS_NOISE_ATT_DEG_RANGE: tuple = (0.5, 2.0)       # ±0.5° to ±2.0°

# Physical tolerances
TOL_PITCH_RATE: float = 20.0         # rad/s (pitch rate tracking bandwidth)
TOL_FLIP_ANGLE: float = 1.4          # rad (remaining angle tolerance for rotation completion progress)
TOL_ALT_DOWN: float = 0.40           # meters (tolerated dip during flip)
TOL_ALT_UP: float = 0.80             # meters (upward altitude tolerance)
ALT_PRE_CLIMB_BUFFER: float = 0.3   # meters (zero-penalty pre-climb ceiling buffer)
TOL_PARASITIC: float = 5.0           # rad/s (off-axis roll/yaw rate tolerance)
TOL_XY_DRIFT: float = 0.40           # meters (horizontal drift tolerance during flip)
TOL_POS_HOVER: float = 0.3          # meters (3D position error tolerance during hover)
TOL_XY_HOVER: float = 0.6          # meters (planar XY drift basin in hover)
TOL_Z_HOVER: float = 0.08            # meters (vertical altitude error tolerance in hover)
TOL_SO3_ATTITUDE: float = 0.70       # SO(3) attitude error (1 - R33) tolerance
TOL_HEADING: float = 0.5           # rad (~20° heading alignment tolerance)
TOL_VEL_HOVER: float = 0.25          # m/s (linear velocity damping tolerance)
TOL_Z_VEL_FLIP: float = 0.9       # m/s (target climb velocity during flip initiation)
TOL_OMEGA_HOVER: float = 3.0         # rad/s (angular velocity damping tolerance)
ARENA_RADIUS_START: float = 2.5      # Starting arena radius for curriculum learning (meters)
ARENA_RADIUS_END: float = 0.8        # Final arena radius at full domain randomization (meters)

# Smoothness and Deadband Parameters
TOL_ACTION_SMOOTH: float = 0.33      # 1st-order action rate norm tolerance (~1,300 RPM / step)
TOL_ACTION_JERK: float = 0.50        # 2nd-order action jerk norm tolerance (direction reversals)
POS_DEADBAND: float = 0.01           # 1 cm (0.01m) deadband: zero gradient on sensor noise floor
ACTION_EMA_ALPHA_FLIP: float = 0.9   # Fast response during acrobatic maneuver
ACTION_EMA_ALPHA_HOVER: float = 0.7  # Calm motor commands during hover
ACTION_HOVER_GAIN: float = 1.0       # Authority scale in steady hover
HOVER_BLEND_DURATION: float = 0.6    # Authority blend duration post-flip
# ======================================================================================


class QuadFlipEnv(gym.Env):
    """
    Quadcopter Gymnasium Environment backed by MuJoCo physics.
    Task: Execute an acrobatic 360° pitch flip and recover to precision hover.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        action_mode: str = ACTION_MODE,
        episode_seconds: float = EPISODE_SECONDS,
        dt: float = SIM_DT,
        target_altitude: float = TARGET_ALTITUDE,
        spawn_altitude: float = SPAWN_ALTITUDE,
        obs_noise: bool = OBS_NOISE,
        random_wind: bool = RANDOM_WIND,
        random_initial_state: bool = RANDOM_INITIAL_STATE,
        random_initial_pos: Optional[bool] = None,
        random_initial_vel: Optional[bool] = None,
        random_initial_att: Optional[bool] = None,
        random_battery: bool = RANDOM_BATTERY,
        motor_tau: float = MOTOR_TAU,
        pitch_direction: float = PITCH_DIRECTION,
        arena_radius: Optional[float] = None,
        arena_radius_start: float = ARENA_RADIUS_START,
        arena_radius_end: float = ARENA_RADIUS_END,
        curriculum_arena: bool = True,
        hover_gain: float = ACTION_HOVER_GAIN,
        tol_xy_hover: Optional[float] = None,
        tol_vel_hover: Optional[float] = None,
        tol_z_hover: Optional[float] = None,
        tol_so3_attitude: Optional[float] = None,
        tol_parasitic: Optional[float] = None,
        tol_heading: Optional[float] = None,
        tol_z_vel_flip: Optional[float] = None,
        tol_omega_hover: Optional[float] = None,
        w_xy: Optional[float] = None,
        w_z: Optional[float] = None,
        w_upright: Optional[float] = None,
        w_heading: Optional[float] = None,
        w_vel: Optional[float] = None,
        w_omega: Optional[float] = None,
        w_action: Optional[float] = None,
        w_progress: Optional[float] = None,
        rate_pid_kp: Optional[Union[np.ndarray, list, float]] = None,
        rate_pid_ki: Optional[Union[np.ndarray, list, float]] = None,
        rate_pid_kd: Optional[Union[np.ndarray, list, float]] = None,
    ):
        super().__init__()

        config.orient = "ENU"
        self.action_mode = action_mode
        self.dt = float(dt)
        self.episode_seconds = float(episode_seconds)
        self.max_steps = int(np.ceil(self.episode_seconds / self.dt))
        self.target_altitude = float(target_altitude)
        self.spawn_altitude = float(spawn_altitude)
        self.obs_noise = bool(obs_noise)
        self.random_initial_state = bool(random_initial_state)
        self.random_initial_pos = bool(random_initial_state if random_initial_pos is None else random_initial_pos)
        self.random_initial_vel = bool(random_initial_state if random_initial_vel is None else random_initial_vel)
        self.random_initial_att = bool(random_initial_state if random_initial_att is None else random_initial_att)
        self.random_battery = bool(random_battery)
        self.motor_tau = float(motor_tau)
        self.pitch_direction = float(pitch_direction)
        self.arena_radius_start = float(arena_radius_start)
        self.arena_radius_end = float(arena_radius_end)
        self.curriculum_arena = bool(curriculum_arena)
        if arena_radius is not None:
            self.arena_radius = float(arena_radius)
            if arena_radius_start == ARENA_RADIUS_START and arena_radius_end == ARENA_RADIUS_END and curriculum_arena:
                self.curriculum_arena = False
        else:
            self.arena_radius = self.arena_radius_start
        self.hover_gain = float(hover_gain)

        # Configurable reward tolerances
        self.tol_xy_hover = float(tol_xy_hover if tol_xy_hover is not None else TOL_XY_HOVER)
        self.tol_vel_hover = float(tol_vel_hover if tol_vel_hover is not None else TOL_VEL_HOVER)
        self.tol_z_hover = float(tol_z_hover if tol_z_hover is not None else TOL_Z_HOVER)
        self.tol_so3_attitude = float(tol_so3_attitude if tol_so3_attitude is not None else TOL_SO3_ATTITUDE)
        self.tol_parasitic = float(tol_parasitic if tol_parasitic is not None else TOL_PARASITIC)
        self.tol_heading = float(tol_heading if tol_heading is not None else TOL_HEADING)
        self.tol_z_vel_flip = float(tol_z_vel_flip if tol_z_vel_flip is not None else TOL_Z_VEL_FLIP)
        self.tol_omega_hover = float(tol_omega_hover if tol_omega_hover is not None else TOL_OMEGA_HOVER)

        # Configurable reward weights
        self.w_xy = float(w_xy if w_xy is not None else 1.8)
        self.w_z = float(w_z if w_z is not None else 2.5)
        self.w_upright = float(w_upright if w_upright is not None else 0.8)
        self.w_heading = float(w_heading if w_heading is not None else 1.0)
        self.w_vel = float(w_vel if w_vel is not None else 3.0)
        self.w_omega = float(w_omega if w_omega is not None else 1.2)
        self.w_action = float(w_action if w_action is not None else 0.70)
        self.w_progress = float(w_progress if w_progress is not None else 2.0)  # Legacy progress weight compatibility

        # MuJoCo physics model
        self.quad = QuadcopterMuJoCo(motor_tau=self.motor_tau)
        self.t: float = 0.0
        self.steps: int = 0
        self.prev_action: np.ndarray = np.zeros(4, dtype=np.float32)
        self.prev_prev_action: np.ndarray = np.zeros(4, dtype=np.float32)

        # Disturbances & latency tracking
        self.gyro_bias: np.ndarray = np.zeros(3, dtype=np.float32)
        self.active_disturbances: Dict[str, Any] = {}
        self.obs_latency: int = 0
        self.obs_buffer: list[np.ndarray] = []
        self.obs_history_len: int = OBS_HISTORY_LEN
        self.obs_history_buffer: list[np.ndarray] = []
        self.dr_level: float = 0.0

        # Dynamic wind disturbance
        self.wind = Wind("PERLIN", MAX_WIND_SPEED) if random_wind else Wind("NONE")
        self.random_wind = random_wind

        # Physical constants
        self.min_w = float(self.quad.params["minWmotor"])
        self.max_w = float(self.quad.params["maxWmotor"])
        self.hover_w = float(self.quad.params["w_hover"])
        self.max_torque_xy = 0.01
        self.max_torque_z = 0.003
        self.max_rate_xy: float = 6.0       # Max roll rate (rad/s)
        self.max_rate_pitch: float = 20.0   # Max pitch rate (rad/s) for flip
        self.max_rate_z: float = 4.0        # Max yaw rate (rad/s)
        self.rate_pid = RatePIDController(
            kp=rate_pid_kp,
            ki=rate_pid_ki,
            kd=rate_pid_kd,
            max_torque_xy=self.max_torque_xy,
            max_torque_z=self.max_torque_z,
        )

        # Task targets & flight progress tracking
        self.initial_pos = np.array([0.0, 0.0, self.spawn_altitude], dtype=np.float32)
        self.target_altitude_hover = float(self.target_altitude)
        self.target_altitude_flip = float(TARGET_ALTITUDE_FLIP)
        self.target_hover_state = np.array([0.0, 0.0, self.target_altitude_hover], dtype=np.float32)
        self.target_flip_state = np.array([0.0, 0.0, self.target_altitude_flip], dtype=np.float32)
        self.accumulated_pitch: float = 0.0
        self.total_pitch_rotated: float = 0.0
        self.reached_90: bool = False
        self.has_inverted: bool = False
        self.flip_completed: bool = False
        self.flip_completed_time: Optional[float] = None
        self.termination_reason: str = "none"

        # Action & observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.actor_single_obs_dim: int = ACTOR_SINGLE_OBS_DIM
        self.actor_total_dim: int = ACTOR_TOTAL_DIM
        self.privileged_obs_dim: int = PRIVILEGED_OBS_DIM
        self.total_obs_dim: int = TOTAL_OBS_DIM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32)

    @property
    def target_state(self) -> np.ndarray:
        return self.target_hover_state if self.flip_completed else self.target_flip_state


    def set_arena_radius(self, radius: float) -> None:
        """Manually sets the arena boundary radius."""
        self.arena_radius = float(radius)

    def set_dr_level(self, level: float) -> None:
        """Sets the Domain Randomization level [0.0, 1.0] and updates curriculum arena radius."""
        self.dr_level = float(np.clip(level, 0.0, 1.0))
        if self.curriculum_arena:
            self.arena_radius = float(
                self.arena_radius_start - self.dr_level * (self.arena_radius_start - self.arena_radius_end)
            )

    def set_rate_pid_gains(
        self,
        kp: Optional[Union[np.ndarray, list, float]] = None,
        ki: Optional[Union[np.ndarray, list, float]] = None,
        kd: Optional[Union[np.ndarray, list, float]] = None,
    ) -> None:
        """Dynamically update inner-loop Rate PID controller gains."""
        self.rate_pid.set_gains(kp=kp, ki=ki, kd=kd)

    def _compute_actor_obs(self) -> np.ndarray:
        pos = self.quad.pos.copy().astype(np.float32)

        quat = self.quad.quat.copy()
        if quat[0] < 0.0:
            quat = -quat

        raw_omega = getattr(self.quad, "omega_filtered", self.quad.omega)
        measured_omega = (raw_omega + self.gyro_bias).astype(np.float32)

        rel_z = float(self.target_state[2] - self.quad.pos[2])
        vel_z = float(self.quad.vel[2])
        flip_progress = float(np.clip(self.accumulated_pitch / FLIP_THRESHOLD, 0.0, 1.0))

        if self.obs_noise:
            dr = float(np.clip(self.dr_level, 0.0, 1.0))
            sigma_pos = OBS_NOISE_POS_RANGE[0] + dr * (OBS_NOISE_POS_RANGE[1] - OBS_NOISE_POS_RANGE[0])
            sigma_vel = OBS_NOISE_VEL_RANGE[0] + dr * (OBS_NOISE_VEL_RANGE[1] - OBS_NOISE_VEL_RANGE[0])
            sigma_omega = OBS_NOISE_OMEGA_RANGE[0] + dr * (OBS_NOISE_OMEGA_RANGE[1] - OBS_NOISE_OMEGA_RANGE[0])
            sigma_att_rad = np.radians(
                OBS_NOISE_ATT_DEG_RANGE[0] + dr * (OBS_NOISE_ATT_DEG_RANGE[1] - OBS_NOISE_ATT_DEG_RANGE[0])
            )

            pos += self.np_random.normal(0.0, sigma_pos, size=3).astype(np.float32)
            rel_z += float(self.np_random.normal(0.0, sigma_pos))
            vel_z += float(self.np_random.normal(0.0, sigma_vel))

            if sigma_att_rad > 1e-6:
                angle_jitter = self.np_random.normal(0.0, sigma_att_rad, size=3)
                dq = np.array([1.0, 0.5 * angle_jitter[0], 0.5 * angle_jitter[1], 0.5 * angle_jitter[2]], dtype=np.float64)
                dq = utils.vectNormalize(dq)
                quat = utils.quatMultiply(quat, dq)
                if quat[0] < 0.0:
                    quat = -quat

            measured_omega += self.np_random.normal(0.0, sigma_omega, size=3).astype(np.float32)

        return np.concatenate([
            pos.astype(np.float32),
            quat.astype(np.float32),
            measured_omega.astype(np.float32),
            np.array([rel_z, vel_z], dtype=np.float32),
            self.prev_action.astype(np.float32),
            np.array([flip_progress], dtype=np.float32),
        ], dtype=np.float32)

    def _compute_privileged_critic_obs(self) -> np.ndarray:
        R = self.quad.dcm
        rel_pos_body = R.T @ (self.target_state - self.quad.pos)
        vel_body = R.T @ self.quad.vel

        if self.random_wind:
            velW, qW1, qW2 = self.wind.randomWind(self.t)
            wind_world = np.array([
                velW * np.cos(qW1) * np.cos(qW2),
                velW * np.sin(qW1) * np.cos(qW2),
                velW * np.sin(qW2)
            ], dtype=np.float32)
        else:
            wind_world = np.zeros(3, dtype=np.float32)

        dist = self.active_disturbances
        com_offset = np.array(dist.get("com_offset", [0.0, 0.0, 0.0]), dtype=np.float32)
        payload_mass = np.array([dist.get("payload_mass_g", 0.0) / 1000.0], dtype=np.float32)
        total_mass = np.array([dist.get("total_mass_g", float(self.quad.base_mass * 1000.0)) / 1000.0], dtype=np.float32)
        motor_efficiencies = np.array(dist.get("motor_efficiencies", [1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        tau_up = np.array([dist.get("tau_up_ms", float(MOTOR_TAU * 1000.0)) / 1000.0], dtype=np.float32)
        tau_down = np.array([dist.get("tau_down_ms", float(MOTOR_TAU * 1000.0)) / 1000.0], dtype=np.float32)
        dynamic_sag = np.array([dist.get("dynamic_sag_coef", 0.0)], dtype=np.float32)
        thrust_scale = np.array([dist.get("thrust_scale", 1.0)], dtype=np.float32)
        gyro_bias = np.array(dist.get("gyro_bias_rads", [0.0, 0.0, 0.0]), dtype=np.float32)
        latency = np.array([float(self.obs_latency)], dtype=np.float32)
        dr_level = np.array([float(self.dr_level)], dtype=np.float32)
        w_motor_norm = (self.quad.wMotor / self.max_w).astype(np.float32)

        return np.concatenate([
            self.quad.pos.astype(np.float32),
            self.quad.vel.astype(np.float32),
            self.quad.quat.astype(np.float32),
            self.quad.omega.astype(np.float32),
            rel_pos_body.astype(np.float32),
            vel_body.astype(np.float32),
            w_motor_norm,
            wind_world,
            com_offset,
            payload_mass,
            total_mass,
            motor_efficiencies,
            tau_up,
            tau_down,
            dynamic_sag,
            thrust_scale,
            gyro_bias,
            latency,
            dr_level,
        ], dtype=np.float32)

    def _compute_observation(self) -> np.ndarray:
        return self._compute_actor_obs()

    def get_actor_obs(self) -> np.ndarray:
        return np.concatenate(self.obs_history_buffer, dtype=np.float32)

    def _get_stacked_obs(self) -> np.ndarray:
        stacked_actor = np.concatenate(self.obs_history_buffer, dtype=np.float32)
        privileged_critic = self._compute_privileged_critic_obs()
        return np.concatenate([stacked_actor, privileged_critic], dtype=np.float32)

    def _compute_stock_obs(self) -> np.ndarray:
        pos = self.quad.pos.copy().astype(np.float32)
        quat = self.quad.quat.copy()
        if quat[0] < 0.0:
            quat = -quat
        raw_omega = getattr(self.quad, "omega_filtered", self.quad.omega)
        omega = (raw_omega + self.gyro_bias).astype(np.float32)
        rel_z = float(self.target_state[2] - self.quad.pos[2])
        vel_z = float(self.quad.vel[2])
        prev_act = np.asarray(self.prev_action, dtype=np.float32)
        flip_progress = float(np.clip(self.accumulated_pitch / FLIP_THRESHOLD, 0.0, 1.0))
        return np.concatenate([
            pos,
            quat.astype(np.float32),
            omega,
            np.array([rel_z, vel_z], dtype=np.float32),
            prev_act,
            np.array([flip_progress], dtype=np.float32),
        ], dtype=np.float32)

    def _compute_reward(self, action: np.ndarray, delta_pitch: float) -> float:
        """
        - Phase 1 (Flip): Exponential pitch rate tracking, rotation progress, altitude lock, parasitic rate damping.
        - Phase 2 (Recovery): Exponential position lock, upright orientation (SO3), velocity & angular rate damping.
        """
        pos = self.quad.pos
        vel = self.quad.vel
        omega = self.quad.omega
        dcm = self.quad.dcm

        # Control smoothness exponential: peaks at 1.0 when motor changes remain within tolerance
        delta_action_norm = float(np.linalg.norm(action - self.prev_action))
        r_action = float(np.exp(-((delta_action_norm / TOL_ACTION_SMOOTH) ** 2)))

        if not self.flip_completed:
            # 1. Dynamic pitch rate tracking guidance (drives aggressive flip discovery)
            target_pitch_rate = 18.0 * self.pitch_direction
            if self.accumulated_pitch > 4.7:
                blend_rate = float(np.clip((FLIP_THRESHOLD - self.accumulated_pitch) / (FLIP_THRESHOLD - 4.7), 0.0, 1.0))
                target_pitch_rate *= blend_rate
            pitch_rate_err = float(abs(omega[1] - target_pitch_rate))
            r_pitch_rate = float(np.exp(-((pitch_rate_err / 5.0) ** 2)))

            # 2. Rotation completion progress: strictly monotonic linear progress [0.0, 1.0]
            r_progress = float(np.clip(self.accumulated_pitch / FLIP_THRESHOLD, 0.0, 1.0))

            # 3. Asymmetric Altitude: allow pre-climb up to +0.30m (1.50m) during flip with zero penalty.
            # When inverted (dcm[2, 2] < 0), naturally allow ballistic drop without panicking the policy.
            inversion = float(max(0.0, -dcm[2, 2]))
            tol_down_eff = TOL_ALT_DOWN * (1.0 + 1.5 * inversion)
            alt_drop = max(0.0, self.target_state[2] - pos[2])
            alt_climb = max(0.0, pos[2] - (self.target_state[2] + ALT_PRE_CLIMB_BUFFER))
            r_alt_down = float(np.exp(-((alt_drop / tol_down_eff) ** 2)))
            r_alt_up = float(np.exp(-((alt_climb / TOL_ALT_UP) ** 2)))
            r_altitude = 0.5 * r_alt_down + 0.5 * r_alt_up

            # 4. Parasitic off-axis damping (roll omega[0] and yaw omega[2])
            parasitic_rate = float(np.sqrt(omega[0] ** 2 + omega[2] ** 2))
            r_parasitic = float(np.exp(-((parasitic_rate / TOL_PARASITIC) ** 2)))

            # 5. Planar containment relative to target (stay near commanded XY)
            xy_dist = float(np.linalg.norm(pos[:2] - self.target_state[:2]))
            r_xy = float(np.exp(-((xy_dist / TOL_XY_DRIFT) ** 2)))

            # 6. Gentle upward pop during flip (+Z in ENU frame):
            # Target is ~+0.4 m/s climb to counteract inverted altitude drop.
            # Downward velocity (falling, vz < 0) is steeply penalized to prevent ground contact.
            vz = float(vel[2])
            if vz >= TOL_Z_VEL_FLIP:
                # Climbing at or above target (+0.4 m/s): gentle decay to avoid ceiling breach
                r_z_vel = float(np.exp(-(((vz - TOL_Z_VEL_FLIP) / 0.35) ** 2)))
            else:
                # Slower climb or falling: tight tolerance drops sharply to ~0 for vz <= 0
                r_z_vel = float(np.exp(-(((TOL_Z_VEL_FLIP - vz) / 0.20) ** 2)))

            # 7. Inverted Throttle Management (Acrobatic Ballistic Flip):
            # When upside down (inversion > 0, dcm[2, 2] < 0), rotor thrust points straight at the ground.
            # Reward cutting throttle to near-zero (< 20%) while inverted so the drone doesn't blast into the floor.
            norm_throttle = float(0.5 * (action[0] + 1.0))  # [0.0 = zero thrust, 1.0 = max thrust]
            r_cut_throttle = float(np.exp(-((norm_throttle / 0.25) ** 2)))
            r_inverted_throttle = inversion * r_cut_throttle

            reward = (
                0.5 * r_pitch_rate
                + 2.0 * r_progress
                + 1.0 * r_altitude
                + 1.0 * r_z_vel
                + 0.5 * r_parasitic
                + 0.5 * r_xy
                + 0.5 * r_action
            )
        else:
            # --- PHASE 2: RECOVER & PRECISION HOVER ---
            # Post-flip living bonus: reduced to 1.0 to balance Phase 2 vs Phase 1 economics
            r_alive = 2.0

            # 1. 3D Position lock to target setpoint
            pos_error = float(np.linalg.norm(pos - self.target_state))
            r_pos = float(np.exp(-((pos_error / TOL_POS_HOVER) ** 2)))

            # 2. Upright attitude on SO(3): (1 - R33) is 0 when upright, 2 when upside down
            so3_error = float(1.0 - max(0.0, dcm[2, 2]))
            r_upright = float(np.exp(-(so3_error / TOL_SO3_ATTITUDE)))

            # 3. Heading lock (yaw alignment): x_body projected onto world forward x-axis
            heading_error = float(np.abs(np.arctan2(dcm[1, 0], dcm[0, 0])))
            r_heading = float(np.exp(-((heading_error / TOL_HEADING) ** 2)))

            # 4. Linear velocity damping (peaks at zero velocity)
            vel_norm = float(np.linalg.norm(vel))
            r_vel = float(np.exp(-((vel_norm / TOL_VEL_HOVER) ** 2)))

            # 5. Angular velocity damping (peaks at zero body rates)
            omega_norm = float(np.linalg.norm(omega))
            r_omega = float(np.exp(-((omega_norm / TOL_OMEGA_HOVER) ** 2)))

            reward = (
                r_alive
                + 2.5 * r_pos 
                + 2.0 * r_upright
                + 1.3 * r_heading
                + 0.5 * r_vel
                + 1.2 * r_omega
                + 0.5 * r_action
            )

        return float(reward)

    def _check_termination(self) -> bool:
        if not np.all(np.isfinite(self.quad.state)):
            self.termination_reason = "divergent_state"
            return True
        if float(np.linalg.norm(self.quad.pos[:2])) > self.arena_radius:
            self.termination_reason = "arena_breach"
            return True
        if self.quad.pos[2] > 2.5:
            self.termination_reason = "ceiling_breach"
            return True
        if self.quad.check_ground_contact():
            self.termination_reason = "ground_crash"
            return True
        return False

    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        w = float(cr * cp * cy + sr * sp * sy)
        x = float(sr * cp * cy - cr * sp * sy)
        y = float(cr * sp * cy + sr * cp * sy)
        z = float(cr * cp * sy - sr * sp * cy)
        return np.array([w, x, y, z], dtype=np.float64)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.t = 0.0
        self.steps = 0
        self.rate_pid.reset()
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_prev_action = np.zeros(4, dtype=np.float32)
        self.accumulated_pitch = 0.0
        self.total_pitch_rotated = 0.0
        self.reached_90 = False
        self.has_inverted = False
        self.flip_completed = False
        self.flip_completed_time = None
        self.termination_reason = "none"

        dr = self.dr_level
        max_lat = int(round(dr * OBS_LATENCY_MAX_STEPS))
        self.obs_latency = int(self.np_random.integers(0, max_lat + 1)) if max_lat > 0 else 0
        self.obs_buffer = []
        self.obs_history_buffer = []

        if self.random_wind:
            self.wind.velW_max = MAX_WIND_SPEED * max(0.1, dr)
            self.wind.reseed()

        if self.random_battery and dr > 0.0:
            batt_lo = 1.0 - dr * 0.15
            batt_hi = 1.0 + dr * 0.10
            thrust_scale = float(self.np_random.uniform(batt_lo, batt_hi))
        else:
            thrust_scale = 1.0

        if dr > 0.0:
            com_dx = float(self.np_random.uniform(-dr * COM_OFFSET_MAX_XY, dr * COM_OFFSET_MAX_XY))
            com_dy = float(self.np_random.uniform(-dr * COM_OFFSET_MAX_XY, dr * COM_OFFSET_MAX_XY))
            com_dz = float(self.np_random.uniform(-dr * COM_OFFSET_MAX_Z, dr * COM_OFFSET_MAX_Z))
            com_offset = np.array([com_dx, com_dy, com_dz], dtype=np.float64)

            m_payload = float(self.np_random.uniform(0.0, dr * PAYLOAD_MASS_MAX))
            total_mass = self.quad.base_mass + m_payload
            inertia_scale = total_mass / self.quad.base_mass
            total_inertia = self.quad.base_inertia * inertia_scale

            site_offsets = [
                np.array([
                    float(self.np_random.uniform(-dr * ARM_LENGTH_JITTER_MAX, dr * ARM_LENGTH_JITTER_MAX)),
                    float(self.np_random.uniform(-dr * ARM_LENGTH_JITTER_MAX, dr * ARM_LENGTH_JITTER_MAX)),
                    0.0,
                ], dtype=np.float64)
                for _ in range(4)
            ]

            motor_efficiencies = np.array([
                1.0 - float(self.np_random.uniform(0.0, dr * MOTOR_MISMATCH_MAX))
                for _ in range(4)
            ], dtype=np.float64)

            dynamic_sag_coef = float(self.np_random.uniform(0.0, dr * DYNAMIC_SAG_COEF_MAX))

            tau_lo = MOTOR_TAU - dr * (MOTOR_TAU - MOTOR_TAU_RANGE[0])
            tau_hi = MOTOR_TAU + dr * (MOTOR_TAU_RANGE[1] - MOTOR_TAU)
            tau_up = float(self.np_random.uniform(tau_lo, tau_hi))
            tau_down_mult = 1.0 + float(self.np_random.uniform(0.0, dr * MOTOR_TAU_DOWN_FACTOR))
            tau_down = float(tau_up * tau_down_mult)
            self.quad.motor_tau = tau_up

            self.gyro_bias = self.np_random.uniform(
                -dr * GYRO_BIAS_MAX, dr * GYRO_BIAS_MAX, size=3
            ).astype(np.float32)

            self.quad.apply_hardware_distortions(
                mass=total_mass,
                com_offset=com_offset,
                inertia=total_inertia,
                site_offsets=site_offsets,
                motor_efficiencies=motor_efficiencies,
                tau_up=tau_up,
                tau_down=tau_down,
                dynamic_sag_coef=dynamic_sag_coef,
            )

            self.active_disturbances = {
                "dr_level": dr,
                "com_offset": com_offset.tolist(),
                "payload_mass_g": float(m_payload * 1000.0),
                "total_mass_g": float(total_mass * 1000.0),
                "site_offsets_mm": (np.array(site_offsets) * 1000.0).tolist(),
                "motor_efficiencies": motor_efficiencies.tolist(),
                "tau_up_ms": float(tau_up * 1000.0),
                "tau_down_ms": float(tau_down * 1000.0),
                "dynamic_sag_coef": dynamic_sag_coef,
                "gyro_bias_rads": self.gyro_bias.tolist(),
                "thrust_scale": float(thrust_scale),
                "latency_steps": int(self.obs_latency),
            }
        else:
            self.quad.motor_tau = MOTOR_TAU
            self.gyro_bias = np.zeros(3, dtype=np.float32)
            self.quad.apply_hardware_distortions()
            self.active_disturbances = {
                "dr_level": 0.0,
                "com_offset": [0.0, 0.0, 0.0],
                "payload_mass_g": 0.0,
                "total_mass_g": float(self.quad.base_mass * 1000.0),
                "site_offsets_mm": [[0.0, 0.0, 0.0]] * 4,
                "motor_efficiencies": [1.0, 1.0, 1.0, 1.0],
                "tau_up_ms": float(MOTOR_TAU * 1000.0),
                "tau_down_ms": float(MOTOR_TAU * 1000.0),
                "dynamic_sag_coef": 0.0,
                "gyro_bias_rads": [0.0, 0.0, 0.0],
                "thrust_scale": float(thrust_scale),
                "latency_steps": int(self.obs_latency),
            }

        if self.random_initial_pos:
            xy_jitter = 0.05 + dr * 0.15
            z_jitter = 0.03 + dr * 0.09
            pos_jitter = self.np_random.uniform(
                [-xy_jitter, -xy_jitter, -z_jitter],
                [xy_jitter, xy_jitter, z_jitter],
                size=3,
            ).astype(np.float64)
            spawn_pos = self.initial_pos + pos_jitter
        else:
            spawn_pos = self.initial_pos.copy()

        if self.random_initial_att:
            att_rp = 0.05 + dr * 0.12
            att_y = 0.035 + dr * 0.10
            r_init = float(self.np_random.uniform(-att_rp, att_rp))
            p_init = float(self.np_random.uniform(-att_rp, att_rp))
            y_init = float(self.np_random.uniform(-att_y, att_y))
            spawn_quat = self._euler_to_quat(r_init, p_init, y_init)
        else:
            spawn_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.quad.reset(pos=spawn_pos, quat=spawn_quat, thrust_scale=thrust_scale)

        if self.random_initial_vel:
            vel_lin = max(0.08, 0.05 + dr * 0.15)
            vel_ang = max(0.12, 0.10 + dr * 0.35)
            self.quad.data.qvel[0:3] = self.np_random.uniform(-vel_lin, vel_lin, size=3)
            self.quad.data.qvel[3:6] = self.np_random.uniform(-vel_ang, vel_ang, size=3)
            mujoco.mj_forward(self.quad.model, self.quad.data)
            self.quad._update_state_properties()

        self.spawn_pos = spawn_pos.copy()
        self.spawn_vel = self.quad.vel.copy()
        self.quad.set_target_marker(self.target_state)

        initial_obs = self._compute_actor_obs()
        self.obs_buffer = [initial_obs.copy() for _ in range(self.obs_latency + 1)]
        delayed_obs = self.obs_buffer[0]
        self.obs_history_buffer = [delayed_obs.copy() for _ in range(self.obs_history_len)]

        stacked_obs = self._get_stacked_obs()
        return stacked_obs, {
            "t": float(self.t),
            "target_state": self.target_state.copy(),
            "position": self.quad.pos.copy(),
            "velocity": self.quad.vel.copy(),
            "quat": self.quad.quat.copy(),
            "omega": self.quad.omega.copy(),
            "omega_des": np.zeros(3, dtype=np.float32),
            "throttle": 0.0,
            "motor_cmd": np.zeros(4, dtype=np.float32),
            "spawn_pos": self.spawn_pos.copy(),
            "spawn_vel": self.spawn_vel.copy(),
            "accumulated_pitch": self.accumulated_pitch,
            "accumulated_roll": self.accumulated_pitch,
            "total_pitch_rotated": self.total_pitch_rotated,
            "termination_reason": self.termination_reason,
            "has_inverted": self.has_inverted,
            "flip_completed": self.flip_completed,
            "actor_obs": self.get_actor_obs(),
            "privileged_obs": self._compute_privileged_critic_obs(),
            "stock_obs": self._compute_stock_obs(),
            "single_obs": delayed_obs.copy(),
            "active_disturbances": self.active_disturbances,
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Smooth hover authority attenuation
        if self.hover_gain < 1.0 and self.flip_completed and self.flip_completed_time is not None:
            t_since_flip = self.t - self.flip_completed_time
            blend = float(np.clip(t_since_flip / HOVER_BLEND_DURATION, 0.0, 1.0))
            hover_scale = 1.0 - (1.0 - self.hover_gain) * blend
            action = hover_scale * action

        # Phase-dependent EMA filtering
        ema_alpha = ACTION_EMA_ALPHA_HOVER if self.flip_completed else ACTION_EMA_ALPHA_FLIP
        action = ema_alpha * action + (1.0 - ema_alpha) * self.prev_action

        # Action execution mapping
        omega_des_cmd = np.zeros(3, dtype=np.float32)
        throttle_cmd = 0.0

        if self.action_mode == "rate_pid":
            throttle_cmd = float(0.5 * (action[0] + 1.0) * self.quad.params["maxThr"])
            omega_des_cmd = np.array([
                action[1] * self.max_rate_xy,
                action[2] * self.max_rate_pitch,
                action[3] * self.max_rate_z,
            ], dtype=np.float64)
            # Step MuJoCo physics with sub-stepping Rate PID
            self.quad.update(
                t=self.t,
                dt=self.dt,
                wind=self.wind,
                rate_cmd=(throttle_cmd, omega_des_cmd),
                rate_pid=self.rate_pid,
                gyro_bias=self.gyro_bias if self.obs_noise else None,
            )
            motor_cmd = self.quad.last_motor_cmd.copy()
        elif self.action_mode == "thrust_moment":
            throttle_cmd = float(0.5 * (action[0] + 1.0) * self.quad.params["maxThr"])
            moments = np.array([
                action[1] * self.max_torque_xy,
                action[2] * self.max_torque_xy,
                action[3] * self.max_torque_z,
            ])
            motor_cmd = mixerFM(self.quad, throttle_cmd, moments)
            self.quad.update(self.t, self.dt, motor_cmd, self.wind)
        else:
            motor_cmd = np.where(
                action >= 0.0,
                self.hover_w + action * (self.max_w - self.hover_w),
                self.hover_w + action * (self.hover_w - self.min_w),
            )
            self.quad.update(self.t, self.dt, motor_cmd, self.wind)

        self.t += self.dt
        self.steps += 1

        # Net rotation along pitch axis (signed delta: forward pitching adds, backward pitching subtracts)
        delta_pitch = float(self.pitch_direction * self.quad.omega[1] * self.dt)
        milestone_bonus = 0.0

        if not self.flip_completed:
            self.total_pitch_rotated += max(0.0, delta_pitch)
            # Accumulate net pitch progress (signed, bounded in [0, FLIP_THRESHOLD])
            self.accumulated_pitch = min(FLIP_THRESHOLD, max(0.0, self.accumulated_pitch + delta_pitch))

            # Milestone 1: Reaching vertical (90 deg / 1.57 rad)
            if not self.reached_90 and self.accumulated_pitch >= 0.5 * np.pi:
                self.reached_90 = True
                milestone_bonus += 5.0

            # Inversion milestone check (R33 < -0.2)
            if not self.has_inverted:
                # Cap rotation progress at 180° until drone physically crosses into inverted flight
                self.accumulated_pitch = min(float(np.pi), self.accumulated_pitch)
                # Anti-ratchet: if the drone is upright (R33 > 0.7), accumulated pitch cannot exceed physical tilt.
                # This prevents normal hover wobbles from ratcheting false progress up to 180°.
                if self.quad.dcm[2, 2] > 0.7:
                    phys_tilt = float(np.arccos(np.clip(self.quad.dcm[2, 2], -1.0, 1.0)))
                    self.accumulated_pitch = min(self.accumulated_pitch, phys_tilt)

                if self.quad.dcm[2, 2] < -0.2:
                    self.has_inverted = True
                    milestone_bonus += 10.0
            else:
                # Completion milestone check (360° rotation + upright attitude R33 > 0.6)
                if self.accumulated_pitch >= FLIP_THRESHOLD and self.quad.dcm[2, 2] > 0.6:
                    self.flip_completed = True
                    self.flip_completed_time = float(self.t)
                    self.accumulated_pitch = FLIP_THRESHOLD
                    self.quad.set_target_marker(self.target_state)
                    milestone_bonus += 20.0

        obs_current = self._compute_actor_obs()
        self.obs_buffer.append(obs_current)
        if len(self.obs_buffer) > self.obs_latency + 1:
            self.obs_buffer.pop(0)
        delayed_obs = self.obs_buffer[0]

        self.obs_history_buffer.append(delayed_obs)
        if len(self.obs_history_buffer) > self.obs_history_len:
            self.obs_history_buffer.pop(0)
        stacked_obs = self._get_stacked_obs()

        reward = self._compute_reward(action, delta_pitch) + milestone_bonus
        terminated = self._check_termination()
        truncated = bool(self.steps >= self.max_steps)

        if terminated:
            reward -= 30.0

        info = {
            "t": float(self.t),
            "target_state": self.target_state.copy(),
            "position": self.quad.pos.copy(),
            "velocity": self.quad.vel.copy(),
            "quat": self.quad.quat.copy(),
            "omega": self.quad.omega.copy(),
            "omega_des": omega_des_cmd.copy(),
            "throttle": float(throttle_cmd),
            "motor_cmd": motor_cmd.copy(),
            "spawn_pos": self.spawn_pos.copy(),
            "spawn_vel": self.spawn_vel.copy(),
            "accumulated_pitch": self.accumulated_pitch,
            "accumulated_roll": self.accumulated_pitch,
            "total_pitch_rotated": self.total_pitch_rotated,
            "termination_reason": self.termination_reason,
            "arena_radius": float(self.arena_radius),
            "reached_90": self.reached_90,
            "has_inverted": self.has_inverted,
            "flip_completed": self.flip_completed,
            "milestone_bonus": float(milestone_bonus),
            "actor_obs": self.get_actor_obs(),
            "privileged_obs": self._compute_privileged_critic_obs(),
            "stock_obs": self._compute_stock_obs(),
            "single_obs": delayed_obs.copy(),
            "active_disturbances": self.active_disturbances,
        }

        # Shift action history for 2nd-order difference
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()

        return stacked_obs, reward, terminated, truncated, info


# Backwards compatibility alias
CustomQuadEnv = QuadFlipEnv


if __name__ == "__main__":
    env = QuadFlipEnv()
    obs, info = env.reset()
    print("✓ QuadFlipEnv (Pitch Flip with Two-Phase Exponential Reward) initialized!")
    print(f"  Observation shape : {obs.shape}")
    print(f"  Action shape      : {env.action_space.shape}")
    print(f"  Target state      : {info['target_state']}")
    print(f"  Pitch direction   : {'Front-flip (+Y)' if env.pitch_direction > 0 else 'Back-flip (-Y)'}")
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    print(f"  Sample step reward: {rew:.2f}")
    print("All checks passed.")