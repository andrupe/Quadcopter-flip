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
import config

# ======================================================================================
# ENVIRONMENT CONFIGURATION (Edit default task parameters directly here)
# ======================================================================================
TARGET_ALTITUDE: float = 1.2         # Target hover altitude post-flip (meters)
TARGET_ALTITUDE_FLIP: float = 1.45   # Target pre-flip climb altitude (meters: +25cm upward punch)
SPAWN_ALTITUDE: float = 1.2          # Quadcopter spawn altitude (meters)
SIM_DT: float = 0.01                 # Simulation and control timestep in seconds (0.01s = 10ms -> 100 Hz)

EPISODE_SECONDS: float = 8.0         # Episode duration in seconds
ACTION_MODE: str = "motor"           # "motor" (direct rotor rad/s) or "thrust_moment"
OBS_NOISE: bool = True              # Add Gaussian sensor noise (sim-to-real domain randomization)
RANDOM_WIND: bool = True            # Add dynamic wind disturbances
MAX_WIND_SPEED: float = 5.0          # Maximum wind speed in m/s (Crazyflie realistic limit)
RANDOM_INITIAL_STATE: bool = True    # Randomize spawn position, attitude tilt, and velocity for robustness
MOTOR_TAU: float = 0.025             # 1st-order motor time constant (seconds: 25ms for Crazyflie coreless DC)
MOTOR_TAU_RANGE: tuple = (0.020, 0.035) # Per-episode motor tau randomization range (real Crazyflie: 20-30ms + margin)
RANDOM_BATTERY: bool = True          # Randomize battery voltage sag / thrust scaling
OBS_LATENCY_MAX_STEPS: int = 2       # Max observation delay in steps (models sensor->compute->actuator pipeline)
PITCH_DIRECTION: float = 1.0         # +1.0 for front-flip (nose down, +omega_y), -1.0 for back-flip (nose up, -omega_y)
FLIP_THRESHOLD: float = 2.0 * np.pi  # Rotation angle required for a full 360° pitch flip

# Physical tolerances for reward exponentials (error scale where reward drops to exp(-1) ~ 0.37)
# Tune these directly in real physical units (meters, m/s, rad/s)
TOL_PITCH_RATE: float = 14.0         # rad/s (pitch rate tracking bandwidth around 20 rad/s target)
TOL_FLIP_ANGLE: float = 1.4          # rad (remaining angle tolerance for rotation completion progress)
TOL_ALT_DOWN: float = 0.20         # meters (tight tolerance: steep penalty for dropping below target)
TOL_ALT_UP: float = 0.60           # meters (increased tolerance: lenient upward ceiling penalty)
ALT_PRE_CLIMB_BUFFER: float = 0.15 # meters (free upward altitude buffer above 1.45m: zero penalty up to 1.60m)
TOL_PARASITIC: float = 5.0           # rad/s (off-axis roll/yaw rate tolerance)
TOL_XY_DRIFT: float = 0.4           # meters (horizontal drift tolerance)
TOL_POS_HOVER: float = 0.3          # meters (3D position error tolerance during hover)
TOL_SO3_ATTITUDE: float = 0.70       # SO(3) attitude error (1 - R33) tolerance (~50° tilt)
TOL_HEADING: float = 0.8         # rad (~20° heading alignment tolerance during hover)
TOL_VEL_HOVER: float = 0.8       # m/s (linear velocity tolerance during hover)
TOL_Z_VEL_FLIP: float = 0.6          # m/s (target climb velocity during flip initiation in ENU frame: +Z is up)
TOL_OMEGA_HOVER: float = 9        # rad/s (angular body rate tolerance during hover)
TOL_ACTION_SMOOTH: float = 0.33      # action delta norm tolerance (~1,300 RPM change per 10ms step) 1389 is the max from crazyflie
ACTION_EMA_ALPHA_FLIP: float = 0.9   # EMA during flip: 90% new + 10% prev (fast response for acrobatic maneuver)
ACTION_EMA_ALPHA_HOVER: float = 0.7  # EMA during hover: 70% new + 30% prev (smooth, calm motor commands)
ACTION_HOVER_GAIN: float = 0.55       # Control authority scale in steady hover (eliminates motor chatter/wobble)
HOVER_BLEND_DURATION: float = 0.6     # Duration (seconds) to transition from full flip recovery to calm hover
# ======================================================================================


class QuadFlipEnv(gym.Env):
    """
    Quadcopter Gymnasium Environment backed by the MuJoCo C-physics engine.
    Task: Execute an acrobatic 360° pitch flip (front-flip / back-flip) and recover to hover.
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
        random_battery: bool = RANDOM_BATTERY,
        motor_tau: float = MOTOR_TAU,
        pitch_direction: float = PITCH_DIRECTION,
        arena_radius: float = 1.5,
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
        self.random_battery = bool(random_battery)
        self.motor_tau = float(motor_tau)
        self.pitch_direction = float(pitch_direction)
        self.arena_radius = float(arena_radius)

        # MuJoCo quadcopter simulation model with 1st-order motor dynamics
        self.quad = QuadcopterMuJoCo(motor_tau=self.motor_tau)
        self.t: float = 0.0
        self.steps: int = 0
        self.prev_action: np.ndarray = np.zeros(4, dtype=np.float32)

        # Observation latency buffer (sim-to-real: models sensor->compute->actuator delay)
        self.obs_latency: int = 0  # Current episode's latency in steps (randomized per reset)
        self.obs_buffer: list = []  # Ring buffer of recent observations

        # Domain randomization level: 0.0 = easy (learn the flip), 1.0 = full DR (sim-to-real hardened)
        self.dr_level: float = 0.0

        # Dynamic wind disturbance (Perlin = aperiodic, realistic turbulence)
        self.wind = Wind("PERLIN", MAX_WIND_SPEED) if random_wind else Wind("NONE")
        self.random_wind = random_wind

        # Physical constants
        self.min_w = float(self.quad.params["minWmotor"])
        self.max_w = float(self.quad.params["maxWmotor"])
        self.hover_w = float(self.quad.params["w_hover"])
        self.max_torque_xy = 0.01
        self.max_torque_z = 0.003

        # Task targets & flip tracking
        self.initial_pos = np.array([0.0, 0.0, self.spawn_altitude], dtype=np.float32)
        self.target_altitude_hover = float(self.target_altitude)
        self.target_altitude_flip = float(TARGET_ALTITUDE_FLIP)
        self.target_hover_state = np.array([0.0, 0.0, self.target_altitude_hover], dtype=np.float32)
        self.target_flip_state = np.array([0.0, 0.0, self.target_altitude_flip], dtype=np.float32)
        self.accumulated_pitch: float = 0.0
        self.has_inverted: bool = False
        self.flip_completed: bool = False
        self.flip_completed_time: Optional[float] = None

        # Action: 4 normalized motor commands in [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: 18 dimensions (Purely coordinate-independent & ego-centric)
        # rel_pos_body (3) + quat (4) + vel_body (3) + omega (3) + prev_action (4) + flip_progress (1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

    @property
    def target_state(self) -> np.ndarray:
        return self.target_hover_state if self.flip_completed else self.target_flip_state


    def set_dr_level(self, level: float) -> None:
        """
        Set domain randomization intensity. Called by ADR callback during training.
        0.0 = minimal randomization (easy, let policy discover the flip)
        1.0 = full randomization (hardened for sim-to-real transfer)
        """
        self.dr_level = float(np.clip(level, 0.0, 1.0))

    def _compute_observation(self) -> np.ndarray:
        """
        Constructs and returns the 18-dim coordinate-independent observation vector.
        All features are relative/ego-centric, allowing the policy to deploy at any altitude or position.
        """
        R = self.quad.dcm  # Body to world rotation matrix (3x3)
        rel_pos_body = R.T @ (self.target_state - self.quad.pos)
        vel_body = R.T @ self.quad.vel
        flip_progress = np.clip(self.accumulated_pitch / FLIP_THRESHOLD, 0.0, 1.0)

        # Canonicalize quaternion to positive hemisphere (w >= 0) to eliminate SO(3) antipodal ambiguity
        quat = self.quad.quat.copy()
        if quat[0] < 0.0:
            quat = -quat

        obs = np.concatenate([
            rel_pos_body,               # 3: Target position relative to quad in body frame
            quat,                       # 4: Canonicalized attitude quaternion [w, x, y, z] with w >= 0
            vel_body,                   # 3: Linear velocity in body frame
            self.quad.omega,            # 3: Angular velocity [p, q, r] in rad/s
            self.prev_action,           # 4: Action from previous step
            np.array([flip_progress]),  # 1: Flip completion ratio [0.0, 1.0]
        ], dtype=np.float32)

        if self.obs_noise:
            noise_std = np.full(18, 0.02, dtype=np.float32)
            noise_std[13:] = 0.0  # Keep action and progress deterministic
            obs += np.random.normal(0.0, noise_std, size=18).astype(np.float32)

        return obs

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

            # 2. Rotation completion progress (exponential as remaining angle approaches 0)
            remaining_flip = max(0.0, FLIP_THRESHOLD - self.accumulated_pitch)
            r_progress = float(np.exp(-(remaining_flip / TOL_FLIP_ANGLE)))

            # 3. Asymmetric Altitude: allow pre-climb up to +0.30m (1.50m) during flip with zero penalty
            alt_drop = max(0.0, self.target_state[2] - pos[2])
            alt_climb = max(0.0, pos[2] - (self.target_state[2] + ALT_PRE_CLIMB_BUFFER))
            r_alt_down = float(np.exp(-((alt_drop / TOL_ALT_DOWN) ** 2)))
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

            reward = (
                2.0 * r_progress
                + 1.0 * r_altitude
                + 0.8 * r_z_vel
                + 0.5 * r_parasitic
                + 0.5 * r_xy
                + 0.2 * r_action
            )
        else:
            # --- PHASE 2: RECOVER & PRECISION HOVER ---
            # Post-flip living bonus: reduced to 1.0 to balance Phase 2 vs Phase 1 economics
            r_alive = 1.0

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
                + 0.6 * r_action
            )

        return float(reward)

    def _check_termination(self) -> bool:
        """Terminates on non-finite state, arena breach, ceiling breach, or ground crash."""
        if not np.all(np.isfinite(self.quad.state)):
            return True
        if float(np.linalg.norm(self.quad.pos[:2])) > self.arena_radius:  # Arena boundary
            return True
        if self.quad.pos[2] > 2.5:                         # Ceiling limit
            return True
        if self.quad.check_ground_contact():                # Ground crash
            return True
        return False

    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Converts Euler roll, pitch, yaw (radians) to MuJoCo scalar-first quaternion [w, x, y, z]."""
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

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
        self.accumulated_pitch = 0.0
        self.has_inverted = False
        self.flip_completed = False
        self.flip_completed_time = None

        # Per-episode domain randomization scaled by dr_level (0.0=easy, 1.0=full)
        dr = self.dr_level

        # 1. Motor tau: fixed at nominal when dr=0, randomized 20-35ms when dr=1
        if self.random_initial_state and dr > 0.0:
            tau_lo = MOTOR_TAU - dr * (MOTOR_TAU - MOTOR_TAU_RANGE[0])  # 0.025 → 0.020
            tau_hi = MOTOR_TAU + dr * (MOTOR_TAU_RANGE[1] - MOTOR_TAU)  # 0.025 → 0.035
            self.quad.motor_tau = float(self.np_random.uniform(tau_lo, tau_hi))
        else:
            self.quad.motor_tau = MOTOR_TAU

        # 2. Observation latency: 0 steps when dr=0, up to OBS_LATENCY_MAX_STEPS when dr=1
        max_lat = int(round(dr * OBS_LATENCY_MAX_STEPS))
        self.obs_latency = int(self.np_random.integers(0, max_lat + 1)) if max_lat > 0 else 0
        self.obs_buffer = []

        # 3. Wind: scale max wind speed with dr_level (gentle breeze → full gusts)
        if self.random_wind:
            self.wind.velW_max = MAX_WIND_SPEED * max(0.1, dr)  # Always some minimal wind
            self.wind.reseed()

        # 4. Battery / thrust variation: narrow range at dr=0, wide at dr=1
        if self.random_battery:
            batt_lo = 1.0 - dr * 0.15   # 1.0 → 0.85
            batt_hi = 1.0 + dr * 0.10   # 1.0 → 1.10
            thrust_scale = float(self.np_random.uniform(batt_lo, batt_hi))
        else:
            thrust_scale = 1.0

        if self.random_initial_state:
            # 5. Position jitter: scale with dr_level (±5cm → ±20cm horizontal, ±3cm → ±12cm vertical)
            xy_jitter = 0.05 + dr * 0.15   # 5cm → 20cm
            z_jitter = 0.03 + dr * 0.09    # 3cm → 12cm
            pos_jitter = self.np_random.uniform(
                [-xy_jitter, -xy_jitter, -z_jitter],
                [xy_jitter, xy_jitter, z_jitter],
                size=3,
            ).astype(np.float64)
            spawn_pos = self.initial_pos + pos_jitter

            # 6. Attitude jitter: scale with dr_level (±3° → ±10° roll/pitch, ±2° → ±8° yaw)
            att_rp = 0.05 + dr * 0.12   # ~3° → ~10°
            att_y = 0.035 + dr * 0.10   # ~2° → ~8°
            r_init = float(self.np_random.uniform(-att_rp, att_rp))
            p_init = float(self.np_random.uniform(-att_rp, att_rp))
            y_init = float(self.np_random.uniform(-att_y, att_y))
            spawn_quat = self._euler_to_quat(r_init, p_init, y_init)

            self.quad.reset(pos=spawn_pos, quat=spawn_quat, thrust_scale=thrust_scale)

            # 7. Velocity jitter: scale with dr_level
            vel_lin = 0.05 + dr * 0.15   # ±0.05 → ±0.20 m/s
            vel_ang = 0.10 + dr * 0.35   # ±0.10 → ±0.45 rad/s
            self.quad.data.qvel[0:3] = self.np_random.uniform(-vel_lin, vel_lin, size=3)
            self.quad.data.qvel[3:6] = self.np_random.uniform(-vel_ang, vel_ang, size=3)
            mujoco.mj_forward(self.quad.model, self.quad.data)
        else:
            self.quad.reset(pos=self.initial_pos, thrust_scale=thrust_scale)

        self.quad.set_target_marker(self.target_state)

        return self._compute_observation(), {
            "target_state": self.target_state.copy(),
            "stock_obs": self._compute_stock_obs(),
        }

    def _compute_stock_obs(self) -> np.ndarray:
        """
        Extracts the 14-dimensional observation measurable by a stock Crazyflie 2.1:
        [rel_z (1), vel_z (1), quat (4), omega (3), prev_action (4), flip_progress (1)].
        """
        rel_z = float(self.target_state[2] - self.quad.pos[2])
        vel_z = float(self.quad.vel[2])
        quat = np.asarray(self.quad.quat, dtype=np.float32)
        omega = np.asarray(self.quad.omega, dtype=np.float32)
        prev_act = np.asarray(self.prev_action, dtype=np.float32)
        progress = float(self.accumulated_pitch)
        return np.concatenate([
            np.array([rel_z, vel_z], dtype=np.float32),
            quat,
            omega,
            prev_act,
            np.array([progress], dtype=np.float32),
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Smooth hover gain attenuation: transition from full authority (1.0x) during flip recovery to calm hover (0.55x)
        if self.flip_completed and self.flip_completed_time is not None:
            t_since_flip = self.t - self.flip_completed_time
            blend = float(np.clip(t_since_flip / HOVER_BLEND_DURATION, 0.0, 1.0))
            hover_scale = 1.0 - (1.0 - ACTION_HOVER_GAIN) * blend
            action = hover_scale * action

        # Phase-dependent EMA: light smoothing during flip (fast response), strong during hover (calm)
        ema_alpha = ACTION_EMA_ALPHA_HOVER if self.flip_completed else ACTION_EMA_ALPHA_FLIP
        action = ema_alpha * action + (1.0 - ema_alpha) * self.prev_action

        # Map normalized action to physical motor commands (rad/s)
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

        # MuJoCo physics step (motor_tau=25ms in the physics backend provides realistic motor lag)
        self.quad.update(self.t, self.dt, motor_cmd, self.wind)
        self.t += self.dt
        self.steps += 1

        # Track pitch flip progress along body y-axis (omega[1])
        delta_pitch = max(0.0, float(self.pitch_direction * self.quad.omega[1] * self.dt))
        if not self.flip_completed:
            self.accumulated_pitch += delta_pitch
            # Milestone 1: Physical inversion check (must reach inverted attitude R33 < -0.2)
            if not self.has_inverted:
                # Cap rotation progress at 180° until the drone physically crosses into inverted flight
                self.accumulated_pitch = min(float(np.pi), self.accumulated_pitch)
                if self.quad.dcm[2, 2] < -0.2:
                    self.has_inverted = True
            else:
                # Milestone 2: 360° completion requires full rotation AND returning upright (R33 > 0.6)
                if self.accumulated_pitch >= FLIP_THRESHOLD and self.quad.dcm[2, 2] > 0.6:
                    self.flip_completed = True
                    self.flip_completed_time = float(self.t)
                    self.quad.set_target_marker(self.target_state)

        obs_current = self._compute_observation()

        # Observation latency: return a delayed observation to model real sensor pipeline
        self.obs_buffer.append(obs_current)
        if len(self.obs_buffer) > self.obs_latency + 1:
            self.obs_buffer.pop(0)
        obs = self.obs_buffer[0]  # Oldest buffered obs (delayed by self.obs_latency steps)

        reward = self._compute_reward(action, delta_pitch)
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
            "accumulated_pitch": self.accumulated_pitch,
            "accumulated_roll": self.accumulated_pitch,  # Backwards compatibility
            "has_inverted": self.has_inverted,
            "flip_completed": self.flip_completed,
            "stock_obs": self._compute_stock_obs(),
        }

        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info


# Backwards compatibility alias
CustomQuadEnv = QuadFlipEnv


if __name__ == "__main__":
    # Quick sanity test if run directly
    env = QuadFlipEnv()
    obs, info = env.reset()
    print("✓ QuadFlipEnv (Pitch Flip) initialized successfully!")
    print(f"  Observation shape : {obs.shape}")
    print(f"  Action shape      : {env.action_space.shape}")
    print(f"  Target state      : {info['target_state']}")
    print(f"  Pitch direction   : {'Front-flip (+Y)' if env.pitch_direction > 0 else 'Back-flip (-Y)'}")
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    print(f"  Sample step reward: {rew:.2f}")
    print("All checks passed.")
