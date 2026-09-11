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
from lighthouse import LighthouseConfig, LighthouseModel
from trajectories import Reference, Trajectory, TrajectoryConfig, TrajectorySampler

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
# ======================================================================================
# OBSERVATION LAYOUT
#   env obs vector : [ o_t (17) | aux (4) | privileged (44) ]                  = 65 dims
#   vec obs vector : [ o_t (17) | z (16) | aux (4) | privileged (44) ]         = 81 dims
#                    (the z block is injected by LatentObsWrapper, in the trainer process)
# The actor consumes only the first ACTOR_SINGLE_OBS_DIM + z dims = 33; the critic
# consumes the whole vector. `aux` carries the encoder-only sensors (specific force,
# battery voltage) and must NOT be visible to the actor.
# ======================================================================================
OBS_HISTORY_LEN: int = 1             # Stacked actor frames. 1 = single frame; the causal history
                                     # encoder subsumes the old 3-frame (30ms) stack.
ACTOR_SINGLE_OBS_DIM: int = 29       # Onboard sensor observation dimension (actor-facing frame)
ENCODER_AUX_DIM: int = 4             # 3-axis specific force + normalised battery voltage
ACTOR_TOTAL_DIM: int = ACTOR_SINGLE_OBS_DIM * OBS_HISTORY_LEN  # 29 dims
PRIVILEGED_OBS_DIM: int = 44         # Privileged simulation truth
TOTAL_OBS_DIM: int = ACTOR_TOTAL_DIM + ENCODER_AUX_DIM + PRIVILEGED_OBS_DIM  # 77 dims
SINGLE_OBS_DIM: int = ACTOR_SINGLE_OBS_DIM

# --------------------------------------------------------------------------------------
# ACTOR FRAME LAYOUT (29 dims).
#
# Everything here is genuinely onboard telemetry, and for the pose/velocity entries it is
# the Lighthouse-fused ESTIMATE, not simulation truth - so the policy sees the same sensor
# degradation a real airframe sees when the deck loses coverage mid-flip.
#
# The reference is presented as ERRORS rather than as an absolute target. Errors keep the
# scale of every channel roughly comparable across a hover, a figure-8 and a 360 deg flip,
# which is what lets ONE policy and ONE reward formula cover the whole repertoire with no
# mode input. An absolute target would make the position channel span 2 m on a waypoint
# run and 0 m on a hover, and the policy would have to learn to renormalise per manoeuvre.
#
# There is no explicit manoeuvre/mode channel by design: the reference errors already
# determine the task, and adding a mode bit would only give the policy a way to condition
# on something the reference does not need.
# --------------------------------------------------------------------------------------
O_POS, O_QUAT, O_OMEGA = 0, 3, 7                        # world position est (3), quat (4), body rate (3)
O_VELXY, O_VELZ = 10, 12                                # horizontal vel est (2), vertical vel est (1)
O_PREV_ACTION = 13                                      # previous applied action (4)
O_P_ERR, O_V_ERR, O_ATT_ERR, O_W_ERR = 17, 20, 23, 26   # reference errors (3 each)
assert O_W_ERR + 3 == ACTOR_SINGLE_OBS_DIM, "actor frame layout does not fill the frame"

# Layout offsets inside the env observation vector
AUX_OFFSET: int = ACTOR_TOTAL_DIM                      # 29
PRIV_OFFSET: int = ACTOR_TOTAL_DIM + ENCODER_AUX_DIM   # 33

# ======================================================================================
# PHYSICS REGRESSION TARGETS for the frozen history encoder
# The quantities z must encode. Deliberately EXCLUDES the ADR curriculum level and the
# injected observation latency: those are training artifacts, not physics.
#
# `rel_wind_b` is intentionally NOT in this list. Measured on this plant, the fluid
# force from a 1.0 m/s wind (the full ADR envelope) is 2.6e-4 N = 0.009 m/s^2, i.e.
# ~0.09% of hover thrust - far below the identification floor. `aero_force_b` captures
# the same physics in the form that actually matters and is large during fast flight.
# Use get_wind_state() to log rel_wind_b as a diagnostic only.
#
# The fluid TORQUE (qfrc_passive[3:6]) is deliberately excluded: measured std is ~1e-6
# N.m against ~2e-3 N for the force, i.e. 1000x smaller, so standardizing it would
# amplify numerical noise by three orders of magnitude for no physical signal.
# ======================================================================================
PRIV_TARGET_GROUPS: tuple = (
    ("mass_ratio", 1),          # m / m_nom
    ("com_offset_b", 3),        # CoM offset, body frame (m)
    ("rotor_radial_err", 4),    # per-rotor radius error vs nominal (m)
    ("motor_efficiency", 4),    # per-motor thrust scale, 1.0 = nominal
    ("motor_tau", 2),           # [tau_up, tau_down] (s)
    ("thrust_scale", 1),        # battery / kTh scale
    ("dynamic_sag_coef", 1),    # burst voltage sag coefficient
    ("gyro_bias_b", 3),         # rad/s
    ("true_vel_w", 3),          # ground-truth velocity (denoising target)
    ("motor_speed_norm", 4),    # actual Omega_i / maxW (denoising target)
    ("aero_force_b", 3),        # fluid (wind/drag) force, body frame
)
PRIV_TARGET_DIM: int = 29       # sum of the group dims above

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

# Encoder-only auxiliary observation noise (1-sigma). Specific force is the dominant
# identification signal for mass / thrust scale / CoM, so it is given a realistic
# BMI088-class noise floor plus vibration.
OBS_NOISE_ACCEL_RANGE: tuple = (0.02, 0.12)       # m/s^2

# ======================================================================================
# TRAJECTORY TRACKING OBJECTIVE
#
# A single exponential-kernel tracking reward replaces the old two-phase flip reward. The
# tolerances are keyed on the reference's manoeuvre kind because one fixed tolerance
# cannot serve the whole repertoire: 0.12 m of position error is a near-miss in a hover
# and a rounding error during a 360 deg flip. Keying them on kind keeps one formula while
# letting the flip be graded on the scale the flip actually operates at.
# ======================================================================================
TRACK_TOL: dict = {
    "hover":     {"pos": 0.12, "vel": 0.25, "att": 0.20, "rate": 1.2},
    "waypoints": {"pos": 0.25, "vel": 0.60, "att": 0.30, "rate": 2.5},
    "figure8":   {"pos": 0.25, "vel": 0.60, "att": 0.30, "rate": 2.5},
    "flip":      {"pos": 0.35, "vel": 1.20, "att": 0.55, "rate": 6.0},
}
TRACK_W_POS: float = 3.0
TRACK_W_VEL: float = 1.0
TRACK_W_ATT: float = 2.0
TRACK_W_RATE: float = 0.8

# ARENA. The tracking task is bounded by the trajectories the sampler can build, so the
# arena only needs to be a generous outer guard against the policy flying away, not a
# curriculum. It is no longer shrunk over training.
ARENA_RADIUS: float = 3.0            # meters, hard outer guard
Z_MIN_SAFE: float = 0.25             # meters, ground-crash threshold for termination
Z_MAX_SAFE: float = 2.45             # meters, ceiling-breach threshold for termination

# Smoothness and Deadband Parameters
TOL_ACTION_SMOOTH: float = 0.33      # 1st-order action rate norm tolerance (~1,300 RPM / step)
ACTION_EMA_ALPHA: float = 0.8        # single EMA constant: no flip/hover phase distinction any more
TERMINATION_PENALTY: float = 30.0    # reward subtracted on a crash / divergence / breach
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
        arena_radius_start: float = ARENA_RADIUS,
        arena_radius_end: float = ARENA_RADIUS,
        curriculum_arena: bool = True,
        hover_gain: float = 1.0,
        trajectory_config: Optional[TrajectoryConfig] = None,
        lighthouse_config: Optional[LighthouseConfig] = None,
        maneuver: Optional[str] = None,
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
        # The tracking task is bounded by the trajectories the sampler can build, so the
        # arena is now a fixed outer guard rather than a shrinking curriculum. The
        # curriculum parameters are still accepted so existing callers keep working, but
        # they no longer drive anything.
        self.arena_radius_start = float(arena_radius_start)
        self.arena_radius_end = float(arena_radius_end)
        self.curriculum_arena = False
        self.arena_radius = float(arena_radius) if arena_radius is not None else ARENA_RADIUS
        self.hover_gain = float(hover_gain)

        # Trajectory tracking: the reference generator, the Lighthouse sensor model, and
        # the current reference sample. `maneuver` pins a fixed high-level command; None
        # samples from the mixture every episode, which is what pretraining wants.
        self.traj_cfg = trajectory_config or TrajectoryConfig()
        self.sampler = TrajectorySampler(self.traj_cfg)
        self.lighthouse = LighthouseModel(lighthouse_config)
        self._maneuver_command: Optional[str] = maneuver
        self.traj: Optional[Trajectory] = None
        self.ref: Optional[Reference] = None
        self._last_actor_frame: Optional[np.ndarray] = None
        self.w_action_smooth: float = 0.5

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
        self.aux_buffer: list[np.ndarray] = []
        self._aux_delayed: np.ndarray = np.zeros(ENCODER_AUX_DIM, dtype=np.float32)
        self.obs_history_len: int = OBS_HISTORY_LEN
        self.obs_history_buffer: list[np.ndarray] = []
        self.dr_level: float = 0.0
        # Multiplies dr_level for the *randomization samplings only*, so the frozen
        # history encoder can be pretrained on a wider envelope than PPO will ever
        # visit. set_dr_level() clips to [0, 1], so the headroom cannot be expressed
        # through dr_level alone. Curriculum (arena radius) stays on dr_level.
        self.dr_headroom: float = 1.0

        # Dynamic wind disturbance
        self.wind = Wind("PERLIN", MAX_WIND_SPEED) if random_wind else Wind("NONE")
        self.random_wind = random_wind

        # Physical constants
        self.min_w = float(self.quad.params["minWmotor"])
        self.max_w = float(self.quad.params["maxWmotor"])
        self.hover_w = float(self.quad.params["w_hover"])
        self.max_torque_xy = 0.01
        self.max_torque_z = 0.003

        # RATE ACTION SCALES (rad/s).
        #
        # These map the normalised policy action in [-1, 1] onto a body-rate setpoint, so
        # they ARE the authority the policy can command. They must stay in step with the
        # trajectory sampler's limits (synced below) or the sampler will emit references
        # the policy is structurally unable to follow.
        #
        # 20 rad/s = 1146 deg/s, chosen to be acrobatic but conservative:
        #   - the BMI088 gyro runs its +-2000 deg/s full scale, so this is 57% of range and
        #     leaves the rate loop real headroom before the signal saturates;
        #   - available angular acceleration is torque/inertia = 0.01 / 1.43e-5 ~ 700
        #     rad/s^2, so 20 rad/s is reached in ~0.03 s - comfortably inside the 1 kHz
        #     inner loop, which is therefore never asked to track something it cannot;
        #   - a 360 deg flip needs >= 2*pi/20 = 0.31 s of rotation, which fits inside the
        #     ballistic coast the altitude budget allows.
        #
        # ROLL and PITCH are deliberately symmetric. The old 6 / 20 split was an artefact of
        # the flip task only ever rotating about pitch, and it made every 360 deg ROLL
        # rotation infeasible - the sampler rejected the entire roll family, silently.
        self.max_rate_xy: float = 20.0      # Max roll rate (rad/s)
        self.max_rate_pitch: float = 20.0   # Max pitch rate (rad/s)
        self.max_rate_z: float = 4.0        # Max yaw rate (rad/s); yaw authority is far lower

        # Keep the sampler's notion of authority in step with the command scales above.
        self.traj_cfg.rate_limits = {"roll": self.max_rate_xy, "pitch": self.max_rate_pitch}

        # Normalized hover-trim throttle command: a0 = 2*m_nom*g/maxThr - 1 (~ -0.0844).
        # Used as the "prior action" fiction at reset so that pretraining cold-start
        # padding and the environment's first frame agree exactly.
        self.hover_trim_action: np.ndarray = np.array([
            2.0 * self.quad.params["mB"] * self.quad.params["g"] / self.quad.params["maxThr"] - 1.0,
            0.0,
            0.0,
            0.0,
        ], dtype=np.float32)
        self.rate_pid = RatePIDController(
            kp=rate_pid_kp,
            ki=rate_pid_ki,
            kd=rate_pid_kd,
            max_torque_xy=self.max_torque_xy,
            max_torque_z=self.max_torque_z,
        )

        # Flight-progress diagnostics. `accumulated_pitch` is retained because several
        # diagnostic scripts report how far the vehicle ACTUALLY rotated; it is no longer
        # part of the objective, which now scores tracking error only.
        self.initial_pos = np.array([0.0, 0.0, self.spawn_altitude], dtype=np.float32)
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
        """Current reference position, as a float32 array.

        Retained because several diagnostics and the encoder data driver read it. It used
        to switch between a pre-flip and a post-flip setpoint; it is now simply wherever
        the reference trajectory currently is, which is the only meaningful notion of
        "target" under trajectory tracking.
        """
        if self.ref is not None:
            return self.ref.p.astype(np.float32)
        return self.quad.pos.astype(np.float32)

    def set_command(self, maneuver: Optional[str]) -> None:
        """High-level command: pin the manoeuvre flown on the NEXT reset.

        This is the interface the ground station drives. Passing None restores mixture
        sampling. Note that no manoeuvre channel is added to the observation - the
        reference errors in the actor frame already determine the task, so the policy
        needs no mode input and a commanded manoeuvre is indistinguishable, to the
        policy, from having sampled it.
        """
        if maneuver is not None and maneuver not in self.traj_cfg.weights:
            raise ValueError(f"unknown manoeuvre {maneuver!r}; expected one of {list(self.traj_cfg.weights)}")
        self._maneuver_command = maneuver

    def set_arena_radius(self, radius: float) -> None:
        """Manually sets the arena boundary radius."""
        self.arena_radius = float(radius)

    def set_dr_level(self, level: float) -> None:
        """Sets the Domain Randomization level in [0, 1].

        No longer touches the arena: the tracking task's difficulty comes from the
        commanded manoeuvre, not from shrinking the flight volume.
        """
        self.dr_level = float(np.clip(level, 0.0, 1.0))

    def set_dr_headroom(self, headroom: float) -> None:
        """
        Scales the randomization ENVELOPE without touching the curriculum level.

        Used only for pretraining the history encoder: set dr_headroom > 1 so the
        frozen encoder has already seen everything the policy's expanding ADR will
        later visit. Has no effect on the arena radius or the ADR schedule.
        """
        self.dr_headroom = float(max(0.0, headroom))

    @property
    def dr_eff(self) -> float:
        """Effective randomization level used for parameter, wind and noise sampling."""
        return float(self.dr_level * self.dr_headroom)

    def set_rate_pid_gains(
        self,
        kp: Optional[Union[np.ndarray, list, float]] = None,
        ki: Optional[Union[np.ndarray, list, float]] = None,
        kd: Optional[Union[np.ndarray, list, float]] = None,
    ) -> None:
        """Dynamically update inner-loop Rate PID controller gains."""
        self.rate_pid.set_gains(kp=kp, ki=ki, kd=kd)

    # -- rotation helpers ----------------------------------------------------------
    @staticmethod
    def _quat_to_dcm(q: np.ndarray) -> np.ndarray:
        """Quaternion (w, x, y, z) -> body-to-world rotation matrix."""
        q = np.asarray(q, dtype=np.float64)
        w, x, y, z = q / max(1e-9, np.linalg.norm(q))
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @staticmethod
    def _dcm_to_quat(R: np.ndarray) -> np.ndarray:
        """Body-to-world rotation matrix -> quaternion (w, x, y, z), Shepperd's method."""
        R = np.asarray(R, dtype=np.float64)
        tr = float(np.trace(R))
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
        q = q / max(1e-9, float(np.linalg.norm(q)))
        return -q if q[0] < 0.0 else q

    @staticmethod
    def _attitude_error_rotvec(R_ref: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Rotation vector (axis * angle) of R_ref^T R, i.e. the attitude tracking error
        expressed in the CURRENT body frame.

        A rotation vector is used rather than a quaternion because it is 3-dimensional
        and has no double-cover sign ambiguity, which a flip would otherwise run straight
        into. The near-pi branch matters for the same reason: the usual
        vee(R - R^T) / (2 sin theta) formula divides by sin(theta) ~ 0 exactly at the
        half-flip instant, so the axis is recovered from the symmetric part instead.
        """
        R_err = np.asarray(R_ref, dtype=np.float64).T @ np.asarray(R, dtype=np.float64)
        th = float(np.arccos(np.clip(0.5 * (np.trace(R_err) - 1.0), -1.0, 1.0)))
        if th < 1e-8:
            return np.zeros(3, dtype=np.float64)
        if th > np.pi - 1e-6:
            A = 0.5 * (R_err + np.eye(3))
            diag = np.sqrt(np.clip(np.diag(A), 0.0, None))
            k = int(np.argmax(diag))
            axis = A[:, k] / diag[k] if diag[k] > 1e-9 else np.array([1.0, 0.0, 0.0])
            return th * (axis / max(1e-9, float(np.linalg.norm(axis))))
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]) / (2.0 * np.sin(th))
        return th * axis

    def _reference_or_default(self) -> Reference:
        """Current reference, or a level hover at the true state before the first reset."""
        if self.ref is not None:
            return self.ref
        return Reference(
            t=self.t, p=self.quad.pos.copy(), v=np.zeros(3), a=np.zeros(3),
            R=self.quad.dcm.copy(), omega=np.zeros(3),
            thrust_ff=float(self.quad.base_mass * self.quad.params["g"]),
            spin=0.0, kind="hover",
        )

    def _compute_actor_obs(self) -> np.ndarray:
        """
        Actor-facing frame (29 dims). See the layout note at the top of the file.

        Pose and velocity come from the Lighthouse-fused ESTIMATE, not from truth, so
        during a blackout the policy sees a dead-reckoned state that is drifting - which
        is exactly the situation it will be in on the real airframe mid-flip. Also note
        the reference errors are computed against that same estimate: grading the policy
        against an error it cannot observe would leak truth into the observation and
        quietly undo the point of the sensor model.

        Position and velocity noise are NOT added here - the Lighthouse model already
        applies its own, and double-counting them would inflate the apparent sensor
        quality that the encoder is supposed to learn to distrust.
        """
        ref = self._reference_or_default()
        est = np.asarray(self.lighthouse.p_est, dtype=np.float64)
        est_v = np.asarray(self.lighthouse.v_est, dtype=np.float64)

        quat = self.quad.quat.copy()
        if quat[0] < 0.0:
            quat = -quat

        raw_omega = getattr(self.quad, "omega_filtered", self.quad.omega)
        measured_omega = (raw_omega + self.gyro_bias).astype(np.float64)

        if self.obs_noise:
            dr = self.dr_eff
            sigma_omega = OBS_NOISE_OMEGA_RANGE[0] + dr * (OBS_NOISE_OMEGA_RANGE[1] - OBS_NOISE_OMEGA_RANGE[0])
            sigma_att_rad = np.radians(
                OBS_NOISE_ATT_DEG_RANGE[0] + dr * (OBS_NOISE_ATT_DEG_RANGE[1] - OBS_NOISE_ATT_DEG_RANGE[0])
            )
            if sigma_att_rad > 1e-6:
                angle_jitter = self.np_random.normal(0.0, sigma_att_rad, size=3)
                dq = np.array([1.0, 0.5 * angle_jitter[0], 0.5 * angle_jitter[1],
                               0.5 * angle_jitter[2]], dtype=np.float64)
                dq = utils.vectNormalize(dq)
                quat = utils.quatMultiply(quat, dq)
                if quat[0] < 0.0:
                    quat = -quat
            measured_omega = measured_omega + self.np_random.normal(0.0, sigma_omega, size=3)

        att_err = self._attitude_error_rotvec(ref.R, self._quat_to_dcm(quat))

        frame = np.zeros(ACTOR_SINGLE_OBS_DIM, dtype=np.float32)
        frame[O_POS:O_POS + 3] = est
        frame[O_QUAT:O_QUAT + 4] = quat
        frame[O_OMEGA:O_OMEGA + 3] = measured_omega
        frame[O_VELXY:O_VELXY + 2] = est_v[:2]
        frame[O_VELZ] = est_v[2]
        frame[O_PREV_ACTION:O_PREV_ACTION + 4] = self.prev_action
        frame[O_P_ERR:O_P_ERR + 3] = ref.p - est
        frame[O_V_ERR:O_V_ERR + 3] = ref.v - est_v
        frame[O_ATT_ERR:O_ATT_ERR + 3] = att_err
        frame[O_W_ERR:O_W_ERR + 3] = ref.omega - measured_omega
        return frame

    def _compute_encoder_aux(self) -> np.ndarray:
        """
        Encoder-only auxiliary observations (ENCODER_AUX_DIM = 4):

            [0:3]  specific force, body frame (m/s^2) - the accelerometer. This is the
                   dominant identification signal for mass, thrust scale, motor
                   mismatch and CoM offset, and it is genuine on-board telemetry.
            [3]    normalised battery voltage V/V_nom (1S LiPo).

        Deliberately withheld from the actor: the actor keeps exactly its legacy
        17-dim observable so the history encoder's contribution can be measured
        against the existing baseline instead of hiding behind an already-improved
        observation. See LatentObsWrapper for how z is injected.
        """
        sf = np.asarray(self.quad.specific_force_b, dtype=np.float32).copy()

        if self.obs_noise:
            dr = self.dr_eff
            sigma_accel = OBS_NOISE_ACCEL_RANGE[0] + dr * (OBS_NOISE_ACCEL_RANGE[1] - OBS_NOISE_ACCEL_RANGE[0])
            sf += self.np_random.normal(0.0, sigma_accel, size=3).astype(np.float32)

        v_batt = np.array([float(self.quad.v_batt_norm)], dtype=np.float32)
        return np.concatenate([sf, v_batt], dtype=np.float32)

    def _compute_privileged_critic_obs(self) -> np.ndarray:
        R = self.quad.dcm
        ref = self._reference_or_default()
        # Reference position expressed in the body frame. This is the critic's view of
        # where it is being asked to go; the actor gets the same information only as an
        # error against its (possibly dead-reckoned) estimate.
        rel_pos_body = R.T @ (ref.p - self.quad.pos)
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

    def get_priv_targets(self) -> np.ndarray:
        """
        Physics regression targets for history-encoder supervision, in SI units and in
        the order given by PRIV_TARGET_GROUPS (PRIV_TARGET_DIM = 32 dims).

        All quantities are ground truth read straight off the plant, so this must never
        be called on the real vehicle - it exists only for pretraining and for on-policy
        probing during PPO (see the verification ladder).
        """
        dist = self.active_disturbances

        total_mass = float(dist.get("total_mass_g", self.quad.base_mass * 1000.0)) / 1000.0
        mass_ratio = total_mass / self.quad.base_mass

        com_offset_b = np.asarray(dist.get("com_offset", [0.0, 0.0, 0.0]), dtype=np.float64)

        nominal_sites = np.asarray(self.quad.base_site_pos, dtype=np.float64)          # (4, 3)
        nominal_r = np.sqrt(np.sum(nominal_sites[:, :2] ** 2, axis=1))
        site_off = np.asarray(dist.get("site_offsets_mm", np.zeros((4, 3))), dtype=np.float64) / 1000.0
        true_r = np.sqrt(np.sum((nominal_sites[:, :2] + site_off[:, :2]) ** 2, axis=1))
        rotor_radial_err = true_r - nominal_r

        motor_efficiency = np.asarray(dist.get("motor_efficiencies", [1.0] * 4), dtype=np.float64)
        motor_tau = np.array([
            float(dist.get("tau_up_ms", self.quad.motor_tau * 1000.0)) / 1000.0,
            float(dist.get("tau_down_ms", self.quad.motor_tau * 1000.0)) / 1000.0,
        ])
        thrust_scale = float(dist.get("thrust_scale", 1.0))
        dynamic_sag_coef = float(dist.get("dynamic_sag_coef", 0.0))
        gyro_bias_b = np.asarray(self.gyro_bias, dtype=np.float64)

        true_vel_w = np.asarray(self.quad.vel, dtype=np.float64)
        motor_speed_norm = np.asarray(self.quad.wMotor, dtype=np.float64) / self.max_w

        # qfrc_passive for this model contains ONLY fluid forces (no springs/dampers
        # are defined), so its translational part is exactly the wind/drag force in the
        # world frame. The rotational part is omitted - see PRIV_TARGET_GROUPS.
        passive = np.asarray(self.quad.data.qfrc_passive[:3], dtype=np.float64)
        aero_force_b = self.quad.dcm.T @ passive

        return np.concatenate([
            np.array([mass_ratio]),
            com_offset_b,
            rotor_radial_err,
            motor_efficiency,
            motor_tau,
            np.array([thrust_scale]),
            np.array([dynamic_sag_coef]),
            gyro_bias_b,
            true_vel_w,
            motor_speed_norm,
            aero_force_b,
        ]).astype(np.float32)

    def get_wind_state(self) -> np.ndarray:
        """
        Wind diagnostics, NOT a regression target. Returns
        [wind_world (3), relative air velocity in body frame (3)].

        Reported for analysis only: at the ADR envelope (<= 1.0 m/s) the resulting
        force is ~0.009 m/s^2, an order of magnitude or more below what the noisy
        17-dim observable can resolve.
        """
        if self.random_wind:
            velW, qW1, qW2 = self.wind.randomWind(self.t)
            wind_world = np.array([
                velW * np.cos(qW1) * np.cos(qW2),
                velW * np.sin(qW1) * np.cos(qW2),
                velW * np.sin(qW2),
            ])
        else:
            wind_world = np.zeros(3, dtype=np.float64)

        rel_air_vel_b = self.quad.dcm.T @ (wind_world - self.quad.vel)
        return np.concatenate([wind_world, rel_air_vel_b]).astype(np.float32)

    def _compute_observation(self) -> np.ndarray:
        return self._compute_actor_obs()

    def get_actor_obs(self) -> np.ndarray:
        return np.concatenate(self.obs_history_buffer, dtype=np.float32)

    def _get_stacked_obs(self) -> np.ndarray:
        """
        Env observation vector: [ stacked actor frames | delayed encoder aux | privileged ].

        The aux block is kept as a SEPARATE, single-frame block rather than being
        interleaved into the stack or appended per frame, so that the actor
        observation stays a clean PREFIX of the vector for any OBS_HISTORY_LEN
        (which keeps AsymmetricActorCriticPolicy's [:actor_obs_dim] slice valid).
        The aux is latency-delayed to match the actor frames - an undelayed
        accelerometer would give the encoder an unrealistic peek at the current
        state that no real airframe provides.
        """
        stacked_actor = np.concatenate(self.obs_history_buffer, dtype=np.float32)
        privileged_critic = self._compute_privileged_critic_obs()
        return np.concatenate([stacked_actor, self._aux_delayed, privileged_critic], dtype=np.float32)

    def get_encoder_frame(self) -> np.ndarray:
        """
        The single encoder input frame at the current step: [o_t (17) | aux_t (4)].

        o_t already carries the POST-EMA action applied at t-1 in its prev_action
        slots, so no separate action channel is needed - adding one would be a
        provable duplicate of obs[12:16] one step earlier.
        """
        return np.concatenate([
            self.obs_history_buffer[-1],
            self._aux_delayed,
        ]).astype(np.float32)

    def _compute_stock_obs(self) -> np.ndarray:
        """The most recent undelayed actor frame.

        This used to be a separate legacy 17-dim layout. Now that the actor frame is the
        only observation there is nothing to reconstruct, so this simply returns the frame
        that was actually produced last, cached rather than recomputed - recomputing would
        draw a fresh set of sensor noise and silently disagree with the observation the
        policy was handed.
        """
        if getattr(self, "_last_actor_frame", None) is not None:
            return self._last_actor_frame.copy()
        return self._compute_actor_obs()

    def _compute_reward(self, action: np.ndarray, delta_pitch: float) -> float:
        """
        Trajectory tracking reward: exponential kernels on position, velocity, attitude and
        body-rate tracking error, plus an action-smoothness term.

        WHICH STATE IS GRADED. The reward is scored against SIMULATION TRUTH, not against
        the Lighthouse estimate. That is deliberate, and it is not the same thing as
        leaking truth into the observation: the reward is a training signal, and grading
        it on a drifting estimate would REWARD the policy for flying to where it merely
        believes it is, which is precisely how an estimator error becomes a silent,
        self-consistent flight error on the real vehicle. The policy never sees truth; only
        the reward function does.

        WHY EXPONENTIAL KERNELS. Each term is bounded in [0, 1], so the achievable return
        per step is comparable across a hover, a figure-8 and a 360 deg flip and PPO's
        single value head does not have to span wildly different magnitudes. They also
        saturate to zero rather than growing without bound, so an early catastrophic error
        cannot dominate the gradient before the policy can fly at all.

        The manoeuvre's terminal hover is not special-cased. Every trajectory ENDS in a
        hover (see trajectories.py), so the tracking objective already assigns full reward
        for a clean finish - there is no separate terminal bonus to tune, and no way for
        the policy to collect the bonus without actually being in a hover.
        """
        ref = self._reference_or_default()
        pos = self.quad.pos
        vel = self.quad.vel
        dcm = self.quad.dcm
        # Grade against the same rate signal the inner loop tracks, so the policy is not
        # penalised for filter lag it cannot remove.
        omega = np.asarray(getattr(self.quad, "omega_filtered", self.quad.omega), dtype=np.float64)

        tol = TRACK_TOL.get(ref.kind, TRACK_TOL["hover"])

        p_err = float(np.linalg.norm(ref.p - pos))
        v_err = float(np.linalg.norm(ref.v - vel))
        att_err = float(np.linalg.norm(self._attitude_error_rotvec(ref.R, dcm)))
        w_err = float(np.linalg.norm(ref.omega - omega))

        r_pos = float(np.exp(-((p_err / tol["pos"]) ** 2)))
        r_vel = float(np.exp(-((v_err / tol["vel"]) ** 2)))
        r_att = float(np.exp(-((att_err / tol["att"]) ** 2)))
        r_rate = float(np.exp(-((w_err / tol["rate"]) ** 2)))

        delta_action_norm = float(np.linalg.norm(action - self.prev_action))
        r_action = float(np.exp(-((delta_action_norm / TOL_ACTION_SMOOTH) ** 2)))

        return float(
            TRACK_W_POS * r_pos
            + TRACK_W_VEL * r_vel
            + TRACK_W_ATT * r_att
            + TRACK_W_RATE * r_rate
            + self.w_action_smooth * r_action
        )

    def _check_termination(self) -> bool:
        if not np.all(np.isfinite(self.quad.state)):
            self.termination_reason = "divergent_state"
            return True
        if float(np.linalg.norm(self.quad.pos[:2])) > self.arena_radius:
            self.termination_reason = "arena_breach"
            return True
        if self.quad.pos[2] > Z_MAX_SAFE:
            self.termination_reason = "ceiling_breach"
            return True
        if self.quad.pos[2] < Z_MIN_SAFE:
            self.termination_reason = "ground_crash"
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
        # Bind the wind model to THIS environment's seeded generator before anything draws
        # from it. windModel defaults to the module-level `random`, which is global and
        # shared process-wide, so without this the wind an episode sees depends on how
        # many other Wind objects exist - and reset(seed=...) would not be reproducible.
        self.wind.set_rng(self.np_random)
        self.t = 0.0
        self.steps = 0
        self.rate_pid.reset()
        # "Prior action" fiction at t=0: assume the vehicle was holding hover trim
        # before the episode began. This makes the environment's first frame identical
        # to the cold-start padding used when sampling pretraining windows.
        self.prev_action = self.hover_trim_action.copy()
        self.prev_prev_action = self.hover_trim_action.copy()
        self.accumulated_pitch = 0.0
        self.total_pitch_rotated = 0.0
        self.reached_90 = False
        self.has_inverted = False
        self.flip_completed = False
        self.flip_completed_time = None
        self.termination_reason = "none"

        dr = self.dr_eff
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
                "dr_eff": float(dr),
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
                "dr_eff": 0.0,
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

        # --- reference trajectory -----------------------------------------------------
        # Sampled BEFORE the spawn is chosen, because the spawn is derived from it: the
        # episode has to begin ON the reference, otherwise the policy spends the first
        # second of every episode simply catching up and the tracking reward measures the
        # catch-up rather than the tracking.
        requested = (options or {}).get("maneuver", self._maneuver_command)
        self.traj = self.sampler.sample(self.np_random, mass=float(self.quad.base_mass), kind=requested)
        self.ref = self.traj.sample(0.0)
        ref0 = self.ref

        # The reference attitude at t = 0 is level for every manoeuvre by construction
        # (hover trivially; waypoints and figure-8 because their acceleration and its
        # derivative are zero there; the flip because its spin starts at 0), so the spawn
        # attitude is level plus a small perturbation.
        if self.random_initial_pos:
            xy_jitter = 0.05 + dr * 0.15
            z_jitter = 0.03 + dr * 0.09
            pos_jitter = self.np_random.uniform(
                [-xy_jitter, -xy_jitter, -z_jitter],
                [xy_jitter, xy_jitter, z_jitter],
                size=3,
            ).astype(np.float64)
        else:
            pos_jitter = np.zeros(3, dtype=np.float64)
        spawn_pos = ref0.p + pos_jitter

        # Spawn at the REFERENCE attitude, not at level.
        #
        # The reference is level in TILT at t = 0 for every manoeuvre, but its YAW is
        # randomised per episode. Spawning level therefore starts the episode with a
        # heading error equal to that yaw - up to 180 deg of attitude error the policy can
        # do nothing about, charged against the attitude kernel and blamed on the policy.
        base_quat = self._dcm_to_quat(ref0.R)
        if self.random_initial_att:
            att_rp = 0.05 + dr * 0.12
            att_y = 0.035 + dr * 0.10
            jit = np.array([
                float(self.np_random.uniform(-att_rp, att_rp)),
                float(self.np_random.uniform(-att_rp, att_rp)),
                float(self.np_random.uniform(-att_y, att_y)),
            ])
            # Applied on the RIGHT (body frame), so this is a small perturbation about the
            # reference attitude rather than a world-frame offset.
            dq = utils.vectNormalize(np.array([1.0, 0.5 * jit[0], 0.5 * jit[1], 0.5 * jit[2]]))
            spawn_quat = utils.quatMultiply(base_quat, dq)
        else:
            spawn_quat = base_quat

        self.quad.reset(pos=spawn_pos, quat=spawn_quat, thrust_scale=thrust_scale)

        if self.random_initial_vel:
            vel_lin = max(0.08, 0.05 + dr * 0.15)
            vel_ang = max(0.12, 0.10 + dr * 0.35)
            # Spawn on the reference VELOCITY too, not at rest. Figure-8 starts at
            # v = (A*w, B*w, 0) rather than zero, so a rest spawn would inject a large
            # initial error that has nothing to do with the policy's behaviour.
            self.quad.data.qvel[0:3] = ref0.v + self.np_random.uniform(-vel_lin, vel_lin, size=3)
            self.quad.data.qvel[3:6] = ref0.omega + self.np_random.uniform(-vel_ang, vel_ang, size=3)
            mujoco.mj_forward(self.quad.model, self.quad.data)
            self.quad._update_state_properties()
        elif np.linalg.norm(ref0.v) > 1e-9:
            # Even with jitter disabled the reference velocity must be honoured, or the
            # episode starts with an error the policy cannot avoid.
            self.quad.data.qvel[0:3] = ref0.v
            mujoco.mj_forward(self.quad.model, self.quad.data)
            self.quad._update_state_properties()

        # Lighthouse installation for this episode, anchored on the true spawn.
        self.lighthouse.reset(spawn_pos, self.quad.vel, self.np_random, dr=self.dr_eff)

        self.spawn_pos = spawn_pos.copy()
        self.spawn_vel = self.quad.vel.copy()
        self.quad.set_target_marker(ref0.p)

        initial_obs = self._compute_actor_obs()
        self._last_actor_frame = initial_obs.copy()
        initial_aux = self._compute_encoder_aux()
        self.obs_buffer = [initial_obs.copy() for _ in range(self.obs_latency + 1)]
        self.aux_buffer = [initial_aux.copy() for _ in range(self.obs_latency + 1)]
        delayed_obs = self.obs_buffer[0]
        self._aux_delayed = self.aux_buffer[0].copy()
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
            "encoder_frame": self.get_encoder_frame(),
            "dr_eff": float(self.dr_eff),
            "active_disturbances": self.active_disturbances,
            "maneuver": (self.traj.maneuver.kind if self.traj is not None else "none"),
            "reference_position": (self.ref.p.copy() if self.ref is not None else self.quad.pos.copy()),
            "reference_velocity": (self.ref.v.copy() if self.ref is not None else self.quad.vel.copy()),
            "lighthouse_fix": bool(self.lighthouse.fix_available),
            "lighthouse_visible": int(self.lighthouse.n_visible),
            "lighthouse_outage_s": float(self.lighthouse.outage_t),
            "lighthouse_drift_m": float(np.linalg.norm(self.lighthouse.p_est - self.quad.pos)),
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Actuator-command smoothing. A single constant replaces the old flip/hover phase
        # switch: under tracking there is no phase to switch on, and the reference already
        # supplies the smoothness the policy is graded against.
        action = ACTION_EMA_ALPHA * action + (1.0 - ACTION_EMA_ALPHA) * self.prev_action

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

        # Advance the reference, then the sensor. The order matters: the observation built
        # below must describe the same instant as the reference it will be compared
        # against, and the Lighthouse update must happen before the actor frame is read or
        # the policy would see the previous step's estimate against this step's reference.
        if self.traj is not None:
            self.ref = self.traj.sample(self.t)
        self.lighthouse.observe(self.quad.pos, self.quad.vel, self.quad.dcm, self.dt, self.np_random)

        # Rotation diagnostics only - these no longer influence the reward. They are kept
        # because the evaluation and benchmark scripts report them as a measure of what
        # the aircraft physically did, which is still the fastest way to see a flip fail.
        delta_pitch = float(self.pitch_direction * self.quad.omega[1] * self.dt)
        self.total_pitch_rotated += max(0.0, delta_pitch)
        self.accumulated_pitch = min(FLIP_THRESHOLD, max(0.0, self.accumulated_pitch + delta_pitch))
        if not self.flip_completed and self.accumulated_pitch >= FLIP_THRESHOLD:
            self.flip_completed = True
            self.flip_completed_time = float(self.t)
        if self.quad.dcm[2, 2] < -0.2:
            self.has_inverted = True
        if self.accumulated_pitch >= 0.5 * np.pi:
            self.reached_90 = True

        obs_current = self._compute_actor_obs()
        self._last_actor_frame = obs_current.copy()
        self.obs_buffer.append(obs_current)
        if len(self.obs_buffer) > self.obs_latency + 1:
            self.obs_buffer.pop(0)
        delayed_obs = self.obs_buffer[0]

        aux_current = self._compute_encoder_aux()
        self.aux_buffer.append(aux_current)
        if len(self.aux_buffer) > self.obs_latency + 1:
            self.aux_buffer.pop(0)
        self._aux_delayed = self.aux_buffer[0]

        self.obs_history_buffer.append(delayed_obs)
        if len(self.obs_history_buffer) > self.obs_history_len:
            self.obs_history_buffer.pop(0)
        stacked_obs = self._get_stacked_obs()

        reward = self._compute_reward(action, delta_pitch)
        terminated = self._check_termination()
        # The episode ends when the TRAJECTORY does, not at a fixed time limit, so every
        # episode finishes in the manoeuvre's terminal hover. That is what makes "always
        # end in a hover state" a property of the task rather than something the reward
        # has to bribe the policy into.
        traj_done = self.traj is not None and self.t >= self.traj.duration
        truncated = bool(self.steps >= self.max_steps or traj_done)

        if terminated:
            reward -= TERMINATION_PENALTY

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
            "maneuver": (self.traj.maneuver.kind if self.traj is not None else "none"),
            "reference_position": (self.ref.p.copy() if self.ref is not None else self.quad.pos.copy()),
            "reference_velocity": (self.ref.v.copy() if self.ref is not None else self.quad.vel.copy()),
            "lighthouse_fix": bool(self.lighthouse.fix_available),
            "lighthouse_visible": int(self.lighthouse.n_visible),
            "lighthouse_outage_s": float(self.lighthouse.outage_t),
            "lighthouse_drift_m": float(np.linalg.norm(self.lighthouse.p_est - self.quad.pos)),
            "actor_obs": self.get_actor_obs(),
            "privileged_obs": self._compute_privileged_critic_obs(),
            "stock_obs": self._compute_stock_obs(),
            "single_obs": delayed_obs.copy(),
            "encoder_frame": self.get_encoder_frame(),
            "dr_eff": float(self.dr_eff),
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