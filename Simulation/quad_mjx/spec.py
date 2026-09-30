"""
Ported environment specification (JAX/MJX).

SINGLE SOURCE OF TRUTH: ``Simulation/quad_flip_env.py`` and ``Simulation/trajectories.py``.
Per the 2026-09-23 decision the numpy baseline is FROZEN and is not imported here; the
constants below are a faithful copy.  Anything that changes in the baseline must change
here too -- ``check_mjx_tracking.py`` prints a diff of the scalar constants it can see so a
drift is visible rather than silent.

WHY THIS FILE EXISTS AT ALL
The previous MJX attempt re-specified the problem instead of porting it (different reward
weights, exponential kernels, no per-family tolerances, 3 manoeuvres, no sensor model),
which is why its numbers are not comparable to the baseline's.  This package ports the
objective exactly; the only deliberate deviations are the ones listed in ``PORT_NOTES``.
"""

from __future__ import annotations

import os

import jax.numpy as jnp

# ======================================================================================
# TIMING / FLIGHT VOLUME
# ======================================================================================
GRAVITY: float = 9.81
SIM_DT: float = 0.01                    # control period (100 Hz)
PHYSICS_DT: float = 0.001               # MuJoCo <option timestep>, must match scene.xml
SUBSTEPS: int = int(round(SIM_DT / PHYSICS_DT))   # 10  <-- the previous port used 5 and so
                                                  #     advanced only 5 ms of physics per
                                                  #     10 ms of control time (half speed).
EPISODE_SECONDS: float = 15.0
MAX_STEPS: int = int(round(EPISODE_SECONDS / SIM_DT))   # 1500

SPAWN_Z: float = 1.2                    # metres; every manoeuvre starts at the sphere centre
FLIGHT_RADIUS: float = 2.0              # metres; VEHiCLE's hard outer guard
VOLUME_CENTER_Z: float = SPAWN_Z

# ======================================================================================
# ACTION MAPPING  (quad_flip_env.step, action_mode="rate_pid")
#   a0 -> thrust = 0.5 (a0 + 1) * MAX_THRUST
#   a1, a2 -> +-MAX_RATE_XY   a3 -> +-MAX_RATE_Z
#   then one EMA pass:  action = ALPHA * action + (1 - ALPHA) * prev_action
# ======================================================================================
ACTION_EMA_ALPHA: float = 0.8
# T1-A mirror of `quad_flip_env.ACTION_MAX_DELTA`: bound the per-step move of the raw action
# before the EMA. The port copies the baseline's constants on purpose (it does not import
# them), so a change THERE has to be made HERE too or the port silently trains a different
# control law.
ACTION_MAX_DELTA: float = float(os.environ.get("QUAD_ACTION_MAX_DELTA", 0.5))
ACTION_DIM: int = 4                     # [thrust, roll rate, pitch rate, yaw rate]
MAX_THRUST: float = 0.60                # N, total collective
MAX_RATE_XY: float = 20.0               # rad/s (roll and pitch are symmetric)
MAX_RATE_Z: float = 4.0                 # rad/s; yaw authority is deliberately far lower
PITCH_DIRECTION: float = 1.0            # +1 front flip, -1 back flip

# ======================================================================================
# MANOEUVRE KINDS  (trajectories.py)
# The reward is keyed on the ACTIVE SEGMENT's kind, so a Chain reports the kind it is
# currently flying (Maneuver.kind_at) rather than "chain".
# ======================================================================================
KIND_NAMES: tuple = (
    "hover", "takeoff", "waypoints", "figure8", "orbit",
    "lissajous", "slalom", "v8", "flip", "chain",
)
N_KINDS: int = len(KIND_NAMES)
KIND_INDEX: dict = {name: i for i, name in enumerate(KIND_NAMES)}

# Per-family tracking tolerances (rows in KIND_NAMES order; columns pos/vel/att/rate).
# Copied verbatim from quad_flip_env.TRACK_TOL.
TRACK_TOL_NAMES: tuple = ("pos", "vel", "att", "rate")
TRACK_TOL = jnp.array([
    [0.12, 0.25, 0.20, 1.2],   # hover
    [0.15, 0.35, 0.20, 1.5],   # takeoff
    [0.25, 0.60, 0.30, 2.5],   # waypoints
    [0.25, 0.60, 0.30, 2.5],   # figure8
    [0.30, 0.60, 0.30, 2.5],   # orbit
    [0.30, 0.60, 0.30, 2.5],   # lissajous
    [0.30, 0.60, 0.30, 2.5],   # slalom
    [0.30, 0.80, 0.45, 4.5],   # v8
    [0.35, 1.20, 0.55, 6.0],   # flip
    [0.30, 0.80, 0.45, 4.0],   # chain (fallback for an unlabelled segment)
], dtype=jnp.float32)

# ======================================================================================
# REWARD  (quad_flip_env._compute_reward)
#   r = 3.0 r_pos + 1.0 r_vel + 2.0 r_att + 0.8 r_rate + 0.5 r_action
#   r_* = cauchy(err / tol) = 1 / (1 + x^2), EXCEPT r_action which keeps the gaussian
#   shape on purpose (it is a regulariser, measured 0.97-1.00).
# ======================================================================================
TRACK_W_POS: float = 3.0
TRACK_W_VEL: float = 1.0
TRACK_W_ATT: float = 2.0
TRACK_W_RATE: float = 0.8
W_ACTION_SMOOTH: float = 0.5
TOL_ACTION_SMOOTH: float = 0.33
REWARD_CEILING_PER_STEP: float = (
    TRACK_W_POS + TRACK_W_VEL + TRACK_W_ATT + TRACK_W_RATE + W_ACTION_SMOOTH
)   # 7.3
REWARD_KERNEL: str = "cauchy"           # "gaussian" reproduces pre-2026-09-16 runs
# Overridable so the penalty can be A/B'd without editing this file (same pattern as
# ACTION_MAX_DELTA above).  MEASURED 2026-09-24: 30.0 is far too small to offset the
# "reset into a fresh, easy episode" gain, so the trainer PAYS the policy to terminate -
# the trained policy commanded ~1.4x hover thrust, climbed at ~6 m/s^2, and broke the 0.5 m
# tunnel at ~50 steps while its stochastic training reward still ROSE (4.543 vs 3.840 for
# an untrained net).  Terminating refreshes the episode; the ~4-steps-of-reward penalty
# does not cover it.
TERMINATION_PENALTY: float = float(os.environ.get("QUAD_TERM_PENALTY", 30.0))

# FLIP_PROGRESS: through a flip the rotvec WRAPS (a full turn returns the body to the
# attitude it started in), so the attitude kernel alone cannot see rotation and the
# cheapest way to collect it is to hover through the manoeuvre.  During a flip the
# attitude error is therefore max(rotvec_err, |ref.spin - veh.spin|).  It rides the
# existing TRACK_W_ATT so the ceiling stays 7.3 and the reward is bit-identical for the
# nine non-flip families.
FLIP_PROGRESS: bool = True
FLIP_SPIN_AXIS_MIN_RATE: float = 1.0    # rad/s at which the flip axis is latched
FLIP_THRESHOLD: float = 2.0 * 3.141592653589793

# ======================================================================================
# OBSERVATION LAYOUT (unchanged from the baseline)
#   env obs : [ o_t 29 | ref_ff 3 | aux 4 | privileged 44 ]                 = 80
#   vec obs : [ o_t 29 | z 16 | ref_ff 3 | aux 4 | privileged 44 ]          = 96
#   ACTOR   : [ o_t 29 | z 16 | ref_ff 3 ]                                  = 48
# ======================================================================================
OBS_HISTORY_LEN: int = 1
ACTOR_SINGLE_OBS_DIM: int = 29
REF_FF_DIM: int = 3
ENCODER_AUX_DIM: int = 4
PRIVILEGED_OBS_DIM: int = 44
ACTOR_TOTAL_DIM: int = ACTOR_SINGLE_OBS_DIM * OBS_HISTORY_LEN        # 29
TOTAL_OBS_DIM: int = ACTOR_TOTAL_DIM + REF_FF_DIM + ENCODER_AUX_DIM + PRIVILEGED_OBS_DIM  # 80
Z_DIM: int = 16
ACTOR_DIM_NO_ENCODER: int = ACTOR_TOTAL_DIM + REF_FF_DIM             # 32
ACTOR_DIM_WITH_ENCODER: int = ACTOR_TOTAL_DIM + Z_DIM + REF_FF_DIM   # 48
# The CRITIC consumes the WRAPPED observation -- the raw env obs with z spliced in after
# o_t -- i.e. exactly what SB3's vec env exposes (baseline: LatentInjector, and
# train.py's `critic_dim = vec_env.observation_space.shape[0]` = 80 + 16).
# It is NOT TOTAL_OBS_DIM (80).  This mistake is invisible until you build the critic.
CRITIC_OBS_DIM: int = (ACTOR_TOTAL_DIM + Z_DIM + REF_FF_DIM
                       + ENCODER_AUX_DIM + PRIVILEGED_OBS_DIM)          # 96
assert CRITIC_OBS_DIM == TOTAL_OBS_DIM + Z_DIM

O_POS, O_QUAT, O_OMEGA = 0, 3, 7
O_VELXY, O_VELZ = 10, 12
O_PREV_ACTION = 13
O_P_ERR, O_V_ERR, O_ATT_ERR, O_W_ERR = 17, 20, 23, 26
assert O_W_ERR + 3 == ACTOR_SINGLE_OBS_DIM

REF_FF_OFFSET: int = ACTOR_TOTAL_DIM                       # 29
AUX_OFFSET: int = ACTOR_TOTAL_DIM + REF_FF_DIM             # 32
PRIV_OFFSET: int = AUX_OFFSET + ENCODER_AUX_DIM            # 36

GRAVITY_VEC = jnp.array([0.0, 0.0, GRAVITY])

ACTOR_FRAME_MODE: str = "anchored_xy"
ANCHOR_ACTOR_XY: bool = True            # False reproduces the pre-2026-09-16 layout

# ======================================================================================
# INITIAL KICK  (reset)
# ======================================================================================
INIT_VEL_RANGE: tuple = (0.10, 0.60)        # m/s per axis
INIT_RATE_RANGE: tuple = (0.15, 1.50)       # rad/s per axis
INIT_POS_JITTER_XY: tuple = (0.05, 0.15)    # base + dr * span
INIT_POS_JITTER_Z: tuple = (0.03, 0.09)
INIT_ATT_JITTER_RP: tuple = (0.05, 0.12)
INIT_ATT_JITTER_YAW: tuple = (0.035, 0.10)
OBS_LATENCY_MAX_STEPS: int = 3          # was 2; T1-A widened the rate-domain DR

# ======================================================================================
# SENSOR NOISE  (1-sigma ranges, linearly interpolated on the DR level)
# ======================================================================================
OBS_NOISE_POS_RANGE: tuple = (0.005, 0.015)
OBS_NOISE_VEL_RANGE: tuple = (0.020, 0.060)
OBS_NOISE_OMEGA_RANGE: tuple = (0.030, 0.250)     # was (0.030, 0.150)
OBS_NOISE_ATT_DEG_RANGE: tuple = (0.5, 2.0)
OBS_NOISE_ACCEL_RANGE: tuple = (0.02, 0.12)

# ======================================================================================
# DOMAIN RANDOMISATION ENVELOPES  (scaled by dr in [0, 1])
# ======================================================================================
MOTOR_TAU: float = 0.025
MOTOR_TAU_RANGE: tuple = (0.020, 0.035)
MOTOR_TAU_DOWN_FACTOR: float = 0.50
COM_OFFSET_MAX_XY: float = 0.0025
COM_OFFSET_MAX_Z: float = 0.0030
PAYLOAD_MASS_MAX: float = 0.0050
ARM_LENGTH_JITTER_MAX: float = 0.0015
MOTOR_MISMATCH_MAX: float = 0.08
DYNAMIC_SAG_COEF_MAX: float = 0.25     # Up to 25% sag (matches real pack ~23% thrust drop)
GYRO_BIAS_MAX: float = 0.055            # rad/s; was 0.035 (T1-A widened the rate-domain DR)
BATTERY_THRUST_SCALE_RANGE: tuple = (0.70, 1.05)   # 1 - dr*0.30 .. 1 + dr*0.05 (covers 0.437 N authority)
THRUST_NL_MAX: float = 0.10             # thrust-curve nonlinearity
RATE_PID_NOISE_MAX: float = 0.04        # inner loop gyro noise injected into rate PID
CAGE_CEILING_Z: float = 2.8             # 2.8 m screening ceiling for gentler flip profile
CAGE_FLOOR_Z: float = 0.05
SPAWN_Z_RANGE: tuple = (0.80, 1.20)     # DR-able spawn altitude fitting inside cage

# Trained (clean) plant values -- also the values the sim-hover trim was computed from.
MASS: float = 0.033                     # kg
DXM: float = 0.0325
DYM: float = 0.0325
K_TH: float = 2.2e-8                    # N / (rad/s)^2
K_TO: float = 7.94e-10                  # N*m / (rad/s)^2
MIN_W: float = 0.0
MAX_W: float = 2600.0

# ======================================================================================
# FLIGHT-ENVELOPE TERMINATION  (quad_flip_env._check_termination)
#
# Two nested guards.  Inside `envelope_scale < 4.0` the reference-relative envelope is
# active and tightens as the curriculum progresses; the outer sphere is always live as a
# divergence guard.  `eff_scale = scale / 1.5` reaches 1.0 at the curriculum end (1.5).
# ======================================================================================
ENVELOPE_SCALE_START: float = 5.0       # >= 4.0 = free exploration, inner envelope OFF
ENVELOPE_SCALE_END: float = 1.5
ENVELOPE_FREE_THRESHOLD: float = 4.0
ENVELOPE_EFF_DIVISOR: float = 1.5
FLIP_MAX_CEILING_EXCURSION: float = 0.60    # * eff_scale
FLIP_MAX_XY_DRIFT: float = 0.80             # * eff_scale
TUNNEL_MIN_ERROR: float = 0.50              # max(0.50, 3.5 * tol_pos) * eff_scale
TUNNEL_TOL_MULT: float = 3.5
OUTER_RADIUS_FREE: float = 3.5              # scale >= 4.0
OUTER_RADIUS_TIGHT: float = 2.5
GROUND_TERMINATE_Z: float = 0.03            # chassis/legs touching the floor
LIGHTHOUSE_FIX_RANGE_MULT: float = 4.0      # env enables max_fix_range = 4 * FLIGHT_RADIUS
TAKEOFF_GRACE_Z: float = 0.20
TAKEOFF_GRACE_SECONDS: float = 1.2
TAKEOFF_UPRIGHT_DCM22: float = 0.50

# Wind: the baseline uses a Perlin wind model, rescaled to MAX_WIND_SPEED * max(0.1, dr).
MAX_WIND_SPEED: float = 1.0

# Failure-injection modes (Simulation/evaluate.py > LighthouseFailure), ported so the
# runaway/teleport/outage/loss behaviour is available in the JAX env as well.
LHF_NONE, LHF_LOSS, LHF_OUTAGE, LHF_RUNAWAY, LHF_TELEPORT, LHF_LATCH = 0, 1, 2, 3, 4, 5
LHF_MODE_NAMES: tuple = ("none", "loss", "outage", "runaway", "teleport", "latch")

PORT_NOTES: tuple = (
    "The inner rate PID is the SAME loop as quad_mujoco.update (gains, torque limits, "
    "ground-truth qvel + gyro bias). It is deliberately NOT replaced by direct moment "
    "control: that would be a different closed loop and a domain shift.",
    "The PID integrator persists across the whole episode (the previous port rebuilt it "
    "every control step, which removed integral action entirely).",
    "Anti-windup clamps ki * integral(error dt) to +-max_integral, in torque units, "
    "exactly as RatePIDController does.",
    "Physicstime == control time: SUBSTEPS * PHYSICS_DT == SIM_DT.",
)


def float_scalar(x):
    """
    A STRONG-typed float scalar in the ambient precision (float32 normally, float64 with x64).

    WHY THIS EXISTS.  `jnp.array(0.0)` is WEAK-typed, and the env's arithmetic promotes such
    leaves to STRONG within a single step.  The state `reset` returns must therefore match the
    state `step` returns leaf for leaf, or XLA compiles the whole rollout TWICE -- once for the
    state produced by the batched reset and once for the state that came back out of the
    rollout.  MEASURED at the production shape: 6 of 300 `EnvState` leaves were
    weak-vs-strong, and that cost a second ~30 s compilation (61 s of startup instead of 30 s).

    `jnp.zeros(())` is strong AND follows x64, which matters because `check_mjx_parity.py`
    runs the whole port in float64: pinning `jnp.float32` here breaks that path with an
    "input carry has type float32[] but output carry has type float64[]" error in `step_plant`.
    """
    return jnp.asarray(x, jnp.zeros(()).dtype)


def tracking_kernel(err, tol):
    """cauchy 1/(1 + (err/tol)^2), or the legacy gaussian.  Mirrors _tracking_kernel."""
    x = err / jnp.maximum(tol, 1e-9)
    if REWARD_KERNEL == "gaussian":
        return jnp.exp(-(x * x))
    return 1.0 / (1.0 + x * x)


def action_kernel(delta_action_norm):
    """Gaussian action-smoothness kernel (kept gaussian on purpose)."""
    x = delta_action_norm / TOL_ACTION_SMOOTH
    return jnp.exp(-(x * x))
