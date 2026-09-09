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
import utils
import config

# ======================================================================================
# ENVIRONMENT CONFIGURATION
# ======================================================================================
TARGET_ALTITUDE: float = 1.2         # Target hover altitude post-flip (meters)
TARGET_ALTITUDE_FLIP: float = 1.45   # Target pre-flip climb altitude (meters: +25cm upward punch)
SPAWN_ALTITUDE: float = 1.2          # Quadcopter spawn altitude (meters)
SIM_DT: float = 0.01                 # Timestep in seconds (0.01s = 10ms -> 100 Hz)

EPISODE_SECONDS: float = 8.0         # Episode duration in seconds
ACTION_MODE: str = "motor"           # "motor" (direct rotor rad/s) or "thrust_moment"
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

# Physical tolerances for Cauchy kernels: 1 / (1 + (error / tol)^2)
TOL_PITCH_RATE: float = 14.0         # rad/s (pitch rate tracking bandwidth)
TOL_ALT_DOWN: float = 0.30           # meters (tolerated dip during flip)
TOL_ALT_UP: float = 0.70             # meters (upward altitude tolerance)
ALT_PRE_CLIMB_BUFFER: float = 0.15   # meters (zero-penalty pre-climb ceiling buffer)
TOL_PARASITIC: float = 5.0           # rad/s (off-axis roll/yaw rate tolerance)
TOL_XY_DRIFT: float = 0.40           # meters (horizontal drift tolerance during flip)
TOL_XY_HOVER: float = 0.50           # meters (planar XY drift basin in hover)
TOL_Z_HOVER: float = 0.08            # meters (vertical altitude error tolerance in hover)
TOL_SO3_ATTITUDE: float = 0.70       # SO(3) attitude error (1 - R33) tolerance
TOL_HEADING: float = 0.80            # rad (~20° heading alignment tolerance)
TOL_VEL_HOVER: float = 0.25          # m/s (linear velocity damping tolerance)
TOL_Z_VEL_FLIP: float = 0.55         # m/s (target climb velocity during flip initiation)
TOL_OMEGA_HOVER: float = 3.0         # rad/s (angular velocity damping tolerance)

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
        arena_radius: float = 2.5,
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
        self.arena_radius = float(arena_radius)
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
        self.w_progress = float(w_progress if w_progress is not None else 20.0)  # PBRS full potential scale

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

        # Task targets & flight progress tracking
        self.initial_pos = np.array([0.0, 0.0, self.spawn_altitude], dtype=np.float32)
        self.target_altitude_hover = float(self.target_altitude)
        self.target_altitude_flip = float(TARGET_ALTITUDE_FLIP)
        self.target_hover_state = np.array([0.0, 0.0, self.target_altitude_hover], dtype=np.float32)
        self.target_flip_state = np.array([0.0, 0.0, self.target_altitude_flip], dtype=np.float32)
        self.accumulated_pitch: float = 0.0
        self.total_pitch_rotated: float = 0.0
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

    @staticmethod
    def _cauchy(error: float, tol: float) -> float:
        """Heavy-tailed Lorentzian/Cauchy kernel: 1 / (1 + (e / tol)^2)."""
        return float(1.0 / (1.0 + (error / tol) ** 2))

    def set_dr_level(self, level: float) -> None:
        self.dr_level = float(np.clip(level, 0.0, 1.0))

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

    def _compute_flip_envelope(self) -> float:
        """Evaluates physical containment during the dynamic flip maneuver."""
        pos = self.quad.pos
        vel = self.quad.vel
        omega = self.quad.omega

        # 1. Dynamic pitch rate tracking guidance
        target_pitch_rate = 18.0 * self.pitch_direction
        if self.accumulated_pitch > 4.7:
            blend_rate = float(np.clip((FLIP_THRESHOLD - self.accumulated_pitch) / (FLIP_THRESHOLD - 4.7), 0.0, 1.0))
            target_pitch_rate *= blend_rate
        pitch_rate_err = float(abs(omega[1] - target_pitch_rate))
        r_pitch_rate = self._cauchy(pitch_rate_err, TOL_PITCH_RATE)

        # 2. Asymmetric altitude preservation
        alt_drop = max(0.0, self.target_flip_state[2] - pos[2])
        alt_climb = max(0.0, pos[2] - (self.target_flip_state[2] + ALT_PRE_CLIMB_BUFFER))
        r_alt_down = self._cauchy(alt_drop, TOL_ALT_DOWN)
        r_alt_up = self._cauchy(alt_climb, TOL_ALT_UP)
        r_altitude = 0.5 * r_alt_down + 0.5 * r_alt_up

        # 3. Off-axis parasitic rate damping & yaw suppression (prevents coning precession)
        parasitic_rate = float(np.sqrt(omega[0] ** 2 + omega[2] ** 2))
        r_parasitic = self._cauchy(parasitic_rate, 3.0)
        p_yaw = -0.5 * float((abs(omega[2]) - 4.0) / 4.0) if abs(omega[2]) > 4.0 else 0.0

        # 4. Planar XY drift containment
        xy_dist = float(np.linalg.norm(pos[:2] - self.target_flip_state[:2]))
        r_xy = self._cauchy(xy_dist, TOL_XY_DRIFT)

        # 5. Upward velocity pop
        vz = float(vel[2])
        if vz >= self.tol_z_vel_flip:
            r_z_vel = self._cauchy(vz - self.tol_z_vel_flip, 0.35)
        else:
            r_z_vel = self._cauchy(self.tol_z_vel_flip - vz, 0.20)

        return (
            1.2 * r_pitch_rate
            + 1.0 * r_altitude
            + 0.8 * r_z_vel
            + 1.2 * r_parasitic
            + 0.5 * r_xy
            + p_yaw
        )

    def _compute_hover_envelope(self) -> float:
        """Evaluates steady-state station keeping with Cauchy fat-tails and sensor deadbands."""
        pos = self.quad.pos
        vel = self.quad.vel
        omega = self.quad.omega
        dcm = self.quad.dcm

        # 1. Upright attitude on SO(3)
        so3_error = float(1.0 - dcm[2, 2])
        r_upright = self._cauchy(so3_error, self.tol_so3_attitude) if dcm[2, 2] > 0.0 else 0.0

        # Upright gate: position, velocity, and alive rewards require right-side up flight
        upright_gate = float(np.clip((dcm[2, 2] - 0.2) / 0.6, 0.0, 1.0))
        r_alive = 1.0 * upright_gate

        # 2. Planar XY position lock with 1 cm deadband
        xy_error = float(np.linalg.norm(pos[:2] - self.target_hover_state[:2]))
        xy_err_eff = max(0.0, xy_error - POS_DEADBAND)
        r_xy = self._cauchy(xy_err_eff, self.tol_xy_hover)

        # 3. Vertical altitude lock with 1 cm deadband
        z_error = float(abs(pos[2] - self.target_hover_state[2]))
        z_err_eff = max(0.0, z_error - POS_DEADBAND)
        r_z = self._cauchy(z_err_eff, self.tol_z_hover)

        # 4. Heading alignment
        heading_error = float(abs(np.arctan2(dcm[1, 0], dcm[0, 0])))
        r_heading = self._cauchy(heading_error, self.tol_heading)

        # 5. Linear and angular velocity damping
        vel_norm = float(np.linalg.norm(vel))
        r_vel = self._cauchy(vel_norm, self.tol_vel_hover)

        omega_norm = float(np.linalg.norm(omega))
        r_omega = self._cauchy(omega_norm, self.tol_omega_hover)

        # Residual spin penalty
        p_spin = -0.5 * float((omega_norm - 3.0) / 3.0) if omega_norm > 3.0 else 0.0

        return (
            r_alive
            + upright_gate * (
                self.w_xy * r_xy
                + self.w_z * r_z
                + self.w_heading * r_heading
                + self.w_vel * r_vel
            )
            + self.w_upright * r_upright
            + self.w_omega * r_omega
            + p_spin
        )

    def _compute_reward(self, action: np.ndarray, delta_pitch_potential: float) -> float:
        """
        Computes the blended reward combining:
        - Telescoping potential-based progress (PBRS)
        - Sigmoidally blended flip & hover task envelopes
        - First-order rate and second-order jerk actuator regularization
        """
        # 1. High-order actuator regularization (rate + jerk)
        delta_action = action - self.prev_action
        delta2_action = action - 2.0 * self.prev_action + self.prev_prev_action
        norm_delta1 = float(np.linalg.norm(delta_action))
        norm_delta2 = float(np.linalg.norm(delta2_action))

        r_act_rate = self._cauchy(norm_delta1, TOL_ACTION_SMOOTH)
        r_act_jerk = self._cauchy(norm_delta2, TOL_ACTION_JERK)
        r_action_smooth = 0.6 * r_act_rate + 0.4 * r_act_jerk

        # 2. Potential-Based Pitch Progress (strictly bounded telescoping sum)
        r_potential = self.w_progress * delta_pitch_potential

        # 3. Continuous Sigmoid Phase Blending
        # Centered at 88% completion (~315°) with width k=16 (smooth over [0.75, 1.0])
        progress_ratio = float(np.clip(self.accumulated_pitch / FLIP_THRESHOLD, 0.0, 1.0))
        raw_blend = float(1.0 / (1.0 + np.exp(-16.0 * (progress_ratio - 0.88))))

        if self.flip_completed and self.flip_completed_time is not None:
            # Glides seamlessly to 1.0 over 200ms post-flip to eliminate any transition bump
            t_post = self.t - self.flip_completed_time
            blend = float(np.clip(raw_blend + (1.0 - raw_blend) * (t_post / 0.20), 0.0, 1.0))
        else:
            blend = raw_blend

        r_flip_envelope = self._compute_flip_envelope()
        r_hover_envelope = self._compute_hover_envelope()

        reward = (
            r_potential
            + (1.0 - blend) * r_flip_envelope
            + blend * r_hover_envelope
            + self.w_action * r_action_smooth
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
        if not self.flip_completed and self.accumulated_pitch > FLIP_THRESHOLD + 1.2:
            self.termination_reason = "phase1_overrotation"
            return True
        if not self.flip_completed and self.t > 1.0:
            self.termination_reason = "phase1_timeout"
            return True
        if self.flip_completed and self.quad.dcm[2, 2] < -0.1:
            self.termination_reason = "phase2_reinversion"
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
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_prev_action = np.zeros(4, dtype=np.float32)
        self.accumulated_pitch = 0.0
        self.total_pitch_rotated = 0.0
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

        # Motor mapping
        if self.action_mode == "thrust_moment":
            throttle = 0.5 * (action[0] + 1.0) * self.quad.params["maxThr"]
            moments = np.array([
                action[1] * self.max_torque_xy,
                action[2] * self.max_torque_xy,
                action[3] * self.max_torque_z,
            ])
            motor_cmd = mixerFM(self.quad, throttle, moments)
        else:
            motor_cmd = np.where(
                action >= 0.0,
                self.hover_w + action * (self.max_w - self.hover_w),
                self.hover_w + action * (self.hover_w - self.min_w),
            )

        # MuJoCo physics step
        self.quad.update(self.t, self.dt, motor_cmd, self.wind)
        self.t += self.dt
        self.steps += 1

        # Track PBRS potential delta: record pre-step potential
        prev_pitch_norm = float(np.clip(self.accumulated_pitch / FLIP_THRESHOLD, 0.0, 1.0))

        # Net rotation along pitch axis
        net_pitch_delta = float(self.pitch_direction * self.quad.omega[1] * self.dt)

        if not self.flip_completed:
            self.accumulated_pitch = max(0.0, self.accumulated_pitch + net_pitch_delta)
            self.total_pitch_rotated += max(0.0, net_pitch_delta)

            # Inversion milestone check (R33 < -0.2)
            if not self.has_inverted:
                self.accumulated_pitch = min(float(np.pi), self.accumulated_pitch)
                if self.quad.dcm[2, 2] < -0.2:
                    self.has_inverted = True
            else:
                # Completion milestone check (360° rotation + upright attitude R33 > 0.6)
                if self.accumulated_pitch >= FLIP_THRESHOLD and self.quad.dcm[2, 2] > 0.6:
                    self.flip_completed = True
                    self.flip_completed_time = float(self.t)
                    self.accumulated_pitch = FLIP_THRESHOLD
                    self.quad.set_target_marker(self.target_state)

        # True Potential Difference: Phi(s') - Phi(s)
        curr_pitch_norm = float(np.clip(self.accumulated_pitch / FLIP_THRESHOLD, 0.0, 1.0))
        delta_pitch_potential = curr_pitch_norm - prev_pitch_norm

        obs_current = self._compute_actor_obs()
        self.obs_buffer.append(obs_current)
        if len(self.obs_buffer) > self.obs_latency + 1:
            self.obs_buffer.pop(0)
        delayed_obs = self.obs_buffer[0]

        self.obs_history_buffer.append(delayed_obs)
        if len(self.obs_history_buffer) > self.obs_history_len:
            self.obs_history_buffer.pop(0)
        stacked_obs = self._get_stacked_obs()

        reward = self._compute_reward(action, delta_pitch_potential)
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
            "motor_cmd": motor_cmd.copy(),
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

        # Shift action history for 2nd-order difference
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()

        return stacked_obs, reward, terminated, truncated, info


# Backwards compatibility alias
CustomQuadEnv = QuadFlipEnv


if __name__ == "__main__":
    env = QuadFlipEnv()
    obs, info = env.reset()
    print("✓ QuadFlipEnv (Pitch Flip with PBRS & Cauchy Kernels) initialized!")
    print(f"  Observation shape : {obs.shape}")
    print(f"  Action shape      : {env.action_space.shape}")
    print(f"  Target state      : {info['target_state']}")
    print(f"  Pitch direction   : {'Front-flip (+Y)' if env.pitch_direction > 0 else 'Back-flip (-Y)'}")
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    print(f"  Sample step reward: {rew:.2f}")
    print("All checks passed.")