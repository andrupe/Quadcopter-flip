from __future__ import annotations

import os
import sys
from dataclasses import replace
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
from trajectories import GRAVITY, Reference, Trajectory, TrajectoryConfig, TrajectorySampler

# ======================================================================================
# ENVIRONMENT CONFIGURATION
# ======================================================================================
# Nominal spawn altitude. This is the same quantity as SPAWN_Z below (the centre of the
# flight volume and the altitude every sampled manoeuvre starts from); keep them equal.
SPAWN_ALTITUDE: float = 1.2          # Quadcopter spawn altitude (meters)
SIM_DT: float = 0.01                 # Timestep in seconds (0.01s = 10ms -> 100 Hz)

EPISODE_SECONDS: float = 15.0        # Episode duration in seconds (must match train.py)
                                     # 8 -> 15 s on 2026-09-15: the mixture now contains
                                     # multi-command CHAINS (up to ~12 s) and every family
                                     # must be able to finish inside one episode, or its
                                     # terminal hover never happens and the truncation
                                     # bootstrap is graded on a mid-manoeuvre reference.
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
#   env obs vector : [ o_t (29) | ref_ff (3) | aux (4) | privileged (44) ]          = 80 dims
#   vec obs vector : [ o_t (29) | z (16) | ref_ff (3) | aux (4) | privileged (44) ] = 96 dims
#                    (the z block is injected by LatentObsWrapper, in the trainer process)
#
# The actor consumes the PREFIX actor_single (29) + z (16, when attached) + ref_ff (3) = 48
# dims; the critic consumes the whole vector. `aux` carries the encoder-only sensors
# (specific force, battery voltage) and must NOT be visible to the actor.
#
# WHY ref_ff SITS BEFORE aux. The actor's view is a clean PREFIX of the vector, which is
# what both the policy (obs[..., :actor_obs_dim]) and the wrapper rely on, so anything the
# actor must see has to come before the encoder-only and privileged blocks. Placing it
# after the actor frame and before aux is what makes "actor = prefix" hold with z inserted
# in between.
# ======================================================================================
OBS_HISTORY_LEN: int = 1             # Stacked actor frames. 1 = single frame; the causal history
                                     # encoder subsumes the old 3-frame (30ms) stack.
ACTOR_SINGLE_OBS_DIM: int = 29       # Onboard sensor observation dimension (actor-facing frame)
REF_FF_DIM: int = 3                  # Reference feed-forward specific force command, WORLD frame
ENCODER_AUX_DIM: int = 4             # 3-axis specific force + normalised battery voltage
ACTOR_TOTAL_DIM: int = ACTOR_SINGLE_OBS_DIM * OBS_HISTORY_LEN  # 29 dims
PRIVILEGED_OBS_DIM: int = 44         # Privileged simulation truth
TOTAL_OBS_DIM: int = ACTOR_TOTAL_DIM + REF_FF_DIM + ENCODER_AUX_DIM + PRIVILEGED_OBS_DIM  # 80
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

# --------------------------------------------------------------------------------------
# REFERENCE FEED-FORWARD (3 dims).  a_ff = a_ref + g*e_z, WORLD frame, m/s^2.
#
# WHY THIS EXISTS. The actor frame above carries the reference only as ERRORS, and the
# policy is memoryless (the GRU's z summarises the recent sensor stream, not the plan), so
# nothing the policy sees encodes the reference's ACCELERATION - it would have to
# differentiate p_err/v_err across steps to recover it, and it cannot. The feed-forward is
# the dominant term in any tracking controller: the feedback terms are corrections to it.
# Without it the best a policy can do is learn a high-gain lagged proportional law, which is
# exactly the failure this project measured live: the trained policy sat ~1.4 m behind a
# continuously translating reference.
#
# WHY a+g AND NOT a. The flatness map is thrust = m*|a + g| and z_b = normalize(a + g), so
# the NORM of this vector is the collective feed-forward and its DIRECTION is the desired
# thrust direction. Giving `a` alone would make the policy add g and take the norm itself.
#
# WHY WORLD FRAME. It keeps the channel's scale independent of attitude (a body-frame
# version spins with the airframe, and through a flip that is the difference between a
# smooth channel and one that sweeps the whole sphere in half a second). The policy already
# receives the attitude quaternion, so the body-frame form is one rotation away.
#
# A Hover has a = 0, so this is [0, 0, g] and |a_ff| = g (1 g of collective). A Flip's
# zero-thrust coast has a = -g, so this is [0, 0, 0]: the channel says "zero collective"
# exactly where the reference commands zero collective.
# --------------------------------------------------------------------------------------
GRAVITY_VEC: np.ndarray = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)

# Layout offsets inside the env observation vector
REF_FF_OFFSET: int = ACTOR_TOTAL_DIM                      # 29
AUX_OFFSET: int = ACTOR_TOTAL_DIM + REF_FF_DIM            # 32
PRIV_OFFSET: int = AUX_OFFSET + ENCODER_AUX_DIM           # 36

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
    # SELF-SUPERVISED ESTIMATOR-DRIFT TARGETS (added 2026-09-16).
    #
    # These are not plant parameters: they are the estimator's OWN ERROR, and they exist
    # so the encoder has to say out loud how wrong the state estimate currently is rather
    # than silently mis-attribute a corrupted pose to a plant fault. z then carries a
    # usable "discount the estimate" signal to the policy, and the logvar head has a
    # genuinely heteroscedastic target (mm-level with a fix, growing without one).
    #
    # `est_drift_p` is what a disappearing fix looks like from the inside, and it is the
    # quantity the corrupted-estimate augmentation in encoder/corruption.py perturbs in
    # lockstep with the frame, so the two mechanisms stay consistent by construction.
    ("est_drift_p", 3),         # p_est - p_true (world, m)
    ("est_drift_v", 3),         # v_est - v_true (world, m/s)
)
PRIV_TARGET_DIM: int = 35       # sum of the group dims above

# Real-world Hardware Distortions & Physical Asymmetries (Domain Randomization)
COM_OFFSET_MAX_XY: float = 0.0025      # ±2.5 mm off-center Center of Mass
COM_OFFSET_MAX_Z: float = 0.0030       # ±3.0 mm vertical Center of Mass offset
PAYLOAD_MASS_MAX: float = 0.0050       # +0 to 5.0g calibrated payload (covers 33g-38g total mass)
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
#
# EVERY kind emitted by a manoeuvre at some time must appear here - a missing key
# silently falls back to the hover set, which for a flying manoeuvre is the tightest
# scale in the table. orbit / lissajous / slalom did exactly that until 2026-09-11: the
# scripted geometric controller scored 60-64% of the per-step maximum on them against the
# solvability floor, with the position kernel sitting near its saturated flat region for
# realistic errors. They now take the waypoints/figure8 velocity, attitude and rate
# tolerances (same dynamic class) with position one notch looser (0.30 m), because they
# roam over the largest extent of the smooth set (orbit radius up to 0.55 m, slalom
# traverse up to 1.2 m). Measured with the scripted controller: 77-84%, in band with the
# other families, vs 74-75% at 0.25 m and 60-64% under the hover fallback.
# `v8` and `chain` were added 2026-09-15; check_env_tracking.py section B asserts this
# table covers every kind the sampler can emit and re-scores all of them.
# ======================================================================================
TRACK_TOL: dict = {
    "hover":     {"pos": 0.12, "vel": 0.25, "att": 0.20, "rate": 1.2},
    "takeoff":   {"pos": 0.15, "vel": 0.35, "att": 0.20, "rate": 1.5},
    "waypoints": {"pos": 0.25, "vel": 0.60, "att": 0.30, "rate": 2.5},
    "figure8":   {"pos": 0.25, "vel": 0.60, "att": 0.30, "rate": 2.5},
    "orbit":     {"pos": 0.30, "vel": 0.60, "att": 0.30, "rate": 2.5},
    "lissajous": {"pos": 0.30, "vel": 0.60, "att": 0.30, "rate": 2.5},
    "slalom":    {"pos": 0.30, "vel": 0.60, "att": 0.30, "rate": 2.5},
    "flip":      {"pos": 0.35, "vel": 1.20, "att": 0.55, "rate": 6.0},
    # v8 (the vertical figure-eight) is the second acrobatic family: fast but NOT inverted,
    # so it holds attitude and velocity closer than a flip while still swinging ~45-60 deg
    # of tilt at up to ~15 rad/s. The tolerances sit between the flip's and the smooth
    # families' - measured with the scripted controller in check_env_tracking.py section B.
    "v8":        {"pos": 0.30, "vel": 0.80, "att": 0.45, "rate": 4.5},
    # `chain` is a FALLBACK only: a Chain reports the kind of the segment it is currently
    # flying (`Maneuver.kind_at`), so mid-chain the reward uses the segment's own scale -
    # hover tolerances during a pause, flip tolerances through the rotation. This entry
    # exists so the coverage assertion in check_env_tracking holds and so a chain built
    # from an unlabelled custom manoeuvre still has a sane scale (waypoints-class).
    "chain":     {"pos": 0.30, "vel": 0.80, "att": 0.45, "rate": 4.0},
}
TRACK_W_POS: float = 3.0
TRACK_W_VEL: float = 1.0
TRACK_W_ATT: float = 2.0
TRACK_W_RATE: float = 0.8
W_ACTION_SMOOTH: float = 0.5

# ======================================================================================
# FLIP ROTATION PROGRESS (flip-scoped: numerically inert for every other family)
#
# WHY THE ATTITUDE TERM ALONE CANNOT SEE A FLIP. The attitude term scores
# `||rotvec(R_ref^T R)||`, and a rotvec WRAPS: a full 360 deg turn returns the body to
# the attitude it started in, so the error is ~0 at BOTH ends of the rotation and maximal
# in the MIDDLE. Maximising it therefore never requires rotating - the cheapest way to
# collect the attitude reward on a flip is to sit still and wait for the reference to
# come back upright. Measured: with the flip selected and the reward unchanged, the
# vehicle's dcm22 never went below +0.60 over 10/10 pinned-flip runs (peak tilt 18-53 deg
# against a reference reaching -1.000), while `flip` still read as one of the BEST
# families (~79%, and 4.07 in a flip-only probe). The score did not measure the flip.
#
# THE FIX. During a flip, also require the vehicle to have accumulated the rotation the
# reference has accumulated:
#
#     progress_err = |ref.spin - spin_veh|        spin_veh = integral of the body rate
#                                                 along the reference's own flip axis
#     att_err      = max(rotvec_err, progress_err)
#
# The `max` (rather than a replacement) is what makes this surgical. When the vehicle
# IS rotating with the reference the two errors agree to within the rotvec's own wrap -
# for a body turning with the reference, |rotvec| = |ref.spin - spin_veh| while both
# angles are under pi - so a tracking vehicle's reward is UNCHANGED. They separate only
# when the vehicle does NOT rotate: at the end of the flip the rotvec error is 0 but
# progress_err is a full 2*pi*k, so the free attitude reward for hovering through the
# manoeuvre is withdrawn and rotating strictly dominates. Measured cost of the refusal:
# the flip's arrest phase drops from r_att ~ 1.0 to r_att ~ 0.008.
#
# SCOPING. Gated on `ref.kind == "flip"`, which `Maneuver.kind_at` reports for the
# flip SEGMENT of a chain too, and which no other family emits - so the reward for the
# eight non-flip families is bit-identical. `ref.spin` defaults to 0.0 and only `Flip`
# defines a non-trivial schedule, so the term is inert by construction elsewhere.
#
# NO NEW WEIGHT. The term rides the existing TRACK_W_ATT (2.0), so
# REWARD_CEILING_PER_STEP stays 7.3 and every recorded percentage stays comparable. It
# also keeps `_tracking_kernel` at exactly four calls per step, which is what the CSV
# term decomposition and `scratch/check_eval_callback.py` spy on.
#
# TARGET. `ref.spin` runs 0 -> 2*pi*rotations analytically, so using it (rather than the
# FLIP_THRESHOLD constant) handles the k=2 double flips the sampler draws without a
# special case.
#
# Ablate without an edit:  QUAD_FLIP_PROGRESS=0 .venv/bin/python scratch/check_env_tracking.py
# ======================================================================================
FLIP_PROGRESS: bool = os.environ.get("QUAD_FLIP_PROGRESS", "1").strip().lower() not in (
    "0", "false", "no", "off")

# The reference's body rate at which the flip's rotation axis is latched. Zero while the
# reference climbs (the spin is confined to the zero-thrust coast window) and 11-16 rad/s
# once it rotates, so any threshold in between is unambiguous.
FLIP_SPIN_AXIS_MIN_RATE: float = 1.0

# ======================================================================================
# REWARD KERNEL SHAPE
#
# Each tracking term is a bounded, monotone, maximum-at-zero kernel of err/tol. WHICH
# kernel is not cosmetic: a GAUSSIAN kernel exp(-(e/tol)^2) collapses to ~0 within a
# factor of two of its own tolerance and takes its gradient with it.
#
# Measured, on the hover tolerance (pos = 0.12 m) with a weight of 3.0 (41% of the
# ceiling):
#
#     p_err     gaussian    cauchy
#     0.12 m      0.368      0.500
#     0.24 m      0.018      0.200
#     0.35 m      0.0002     0.105
#     0.60 m      2.7e-08    0.038
#
# So under the gaussian a policy flying 2x-5x the tolerance - which is where a policy
# spends its entire early training, and where the live point-hold sits at 0.2-0.7 m -
# collects essentially zero position signal, and the scripted geometric controller loses
# 27% of the per-step maximum on HOVER for exactly this reason (measured: hover 73% of max
# against a 70% floor). The kernel, not the controller, is what is being measured there.
#
# CAUCHY keeps the [0,1] bound, is smooth and monotone, is FLATTER near the target than
# the gaussian (so it does not distort the precision end of the task) and decays like
# 1/e^2 instead of e^(-e^2), which keeps a usable gradient out to several tolerances.
# "gaussian" remains selectable for reproducing pre-2026-09-16 runs.
# Overridable at import time so an A/B of the two kernels needs no edit:
#     QUAD_REWARD_KERNEL=gaussian .venv/bin/python scratch/check_env_tracking.py
REWARD_KERNEL: str = os.environ.get("QUAD_REWARD_KERNEL", "cauchy").strip().lower()

# Ceiling of the per-step reward, i.e. what a perfect tracker collects. Single source of
# truth: the value used to be quoted as a literal 7.3 in several comments, which is exactly
# the kind of number that silently goes stale when a weight is retuned.
REWARD_CEILING_PER_STEP: float = TRACK_W_POS + TRACK_W_VEL + TRACK_W_ATT + TRACK_W_RATE + W_ACTION_SMOOTH


def _tracking_kernel(err: float, tol: float) -> float:
    """
    Map a tracking error to a bounded [0, 1] reward, 1.0 at zero error.

    `tol` sets the scale (kernel(tol) = 0.37 gaussian / 0.50 cauchy), not a hard cutoff:
    the kernel is smooth and strictly monotone everywhere, so there is no error at which
    the gradient discontinuously vanishes. See REWARD_KERNEL for why the tail shape
    matters more than the scale.
    """
    if tol <= 0.0:                                     # guard: a zero tolerance would divide
        return 1.0 if err == 0.0 else 0.0
    x = err / tol
    if REWARD_KERNEL == "gaussian":
        return float(np.exp(-(x * x)))
    return float(1.0 / (1.0 + x * x))                  # cauchy

# ======================================================================================
# FLIGHT VOLUME  (mirror of TrajectoryConfig in trajectories.py - keep in step)
#
# A sphere of radius FLIGHT_RADIUS centred on the FIXED world point (0, 0, SPAWN_Z).
#
# Everything starts at SPAWN_Z, which is the sphere's CENTRE. That is deliberate and is
# what makes the initial-state randomisation survivable: the sphere's natural bottom is
# SPAWN_Z - FLIGHT_RADIUS = -0.8 m, i.e. underground, so it is effectively clipped by the
# ground at exactly 1.2 m below the start point. Spawning on the floor of the volume
# instead would make ANY downward component of the initial kick an immediate violation.
#
# The REFERENCES are held to a much tighter bound than the sphere: TrajectorySampler
# enforces a 1.5 m x 1.5 m square footprint on every path it emits
# (|x|, |y| <= TrajectoryConfig.bounds_xy = 0.75 m), so training never proposes a
# reference outside that square - it screens 96 points along every candidate rather than
# trusting the draw ranges. The sphere is the VEHICLE's outer termination guard.
# ======================================================================================
SPAWN_Z: float = 1.2                 # metres; every manoeuvre starts here
FLIGHT_RADIUS: float = 2.0           # metres; hard outer boundary, centred on the spawn
VOLUME_CENTER: tuple = (0.0, 0.0, SPAWN_Z)

# ======================================================================================
# ACTOR FRAME ANCHORING (2026-09-16): the x,y channels are RELATIVE, z stays absolute.
#
# `o_t[O_POS:O_POS+3]` used to be the raw Lighthouse estimate in ROOM coordinates. That
# made the actor input depend on WHERE the vehicle happened to be: a translation of the
# whole episode changes that one channel and nothing else (everything else in the frame is
# a difference, a body-frame quantity, an acceleration or an action). Measured on the 15M
# checkpoint, a 0.5 m translation moved the rate command by 54% of its own magnitude, and
# the dependence is roughly LINEAR with no saturation - so an episode flown 2-4 m from
# whatever the estimator calls the origin (i.e. any real room) is far outside the trained
# span. The task itself is already relative: the reference is relocated onto the launch
# pose, and every error channel is a difference. Only this channel was not.
#
# So the actor frame's x,y are now measured from the pose the episode started at:
#
#     x_rel = est.x - anchor.x        y_rel = est.y - anchor.y        z = est.z
#
# At launch the pair reads exactly (0, 0) wherever the vehicle is, and moving the whole
# room leaves the observation BIT-IDENTICAL. That is what makes the policy truly
# position-agnostic ("it does not matter where we start").
#
# WHY x,y AND NOT z. Horizontally there is no landmark: the task is a relative reference
# plus a guard, both of which move with the anchor. Vertically there IS one - gravity, the
# floor and the ceiling are absolute - so an absolute z is an exact, launch-height-
# independent ground cue. Anchoring z too would make a launch at 0.5 m indistinguishable
# from one at 1.2 m, which is how a policy is flown into the floor.
#
# The ANCHOR MOVES WHENEVER THE EPISODE DOES: `reset()` and the live layer's
# `adopt_state()` handover, which is exactly where the GRU state and the observation
# buffers are also reseated. A manoeuvre launched from inside a live policy segment does
# NOT re-anchor: the reference is already relocated onto the current pose, so the frame
# stays continuous for the encoder's history.
#
# KEEP IN STEP: `collect_data.py` records ACTOR_FRAME_MODE into every shard and
# `export_policy.py` bakes it into the firmware as POLICY_FRAME_ANCHORED_XY, so a corpus,
# an encoder checkpoint or an image built for the other mode fails loudly rather than
# silently feeding the policy a frame it was not trained on.
# ======================================================================================
ACTOR_FRAME_MODE: str = "anchored_xy"
ANCHOR_ACTOR_XY: bool = True         # False reproduces the pre-2026-09-16 layout (ablation only)

# INITIAL KICK. The vehicle starts at the centre of the volume with an arbitrary
# TRANSLATIONAL and ROTATIONAL velocity, so the policy has to recover from a disturbed
# start rather than from the trim condition. Both are (dr=0, dr=1) envelopes and are
# applied per axis, so the kick can point in any direction - including straight down.
INIT_VEL_RANGE: tuple = (0.10, 0.60)     # m/s per axis
INIT_RATE_RANGE: tuple = (0.15, 1.50)    # rad/s per axis

# Smoothness and Deadband Parameters
TOL_ACTION_SMOOTH: float = 0.33      # 1st-order action rate norm tolerance (~1,300 RPM / step)
ACTION_EMA_ALPHA: float = 0.8        # single EMA constant: no flip/hover phase distinction any more
TERMINATION_PENALTY: float = 30.0    # reward subtracted on a crash / divergence / breach
# ======================================================================================


class QuadFlipEnv(gym.Env):
    """
    Quadcopter Gymnasium Environment backed by MuJoCo physics.
    Task: track a sampled reference trajectory (hover / waypoints / figure-8 / orbit /
    lissajous / slalom / 360 deg flip / vertical figure-eight / multi-command chain) and
    recover to the terminal hover.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        action_mode: str = ACTION_MODE,
        episode_seconds: float = EPISODE_SECONDS,
        dt: float = SIM_DT,
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
        arena_radius_start: float = FLIGHT_RADIUS,
        arena_radius_end: float = FLIGHT_RADIUS,
        curriculum_arena: bool = True,
        trajectory_config: Optional[TrajectoryConfig] = None,
        lighthouse_config: Optional[LighthouseConfig] = None,
        maneuver: Optional[str] = None,
        rate_pid_kp: Optional[Union[np.ndarray, list, float]] = None,
        rate_pid_ki: Optional[Union[np.ndarray, list, float]] = None,
        rate_pid_kd: Optional[Union[np.ndarray, list, float]] = None,
        telemetry: bool = True,
    ):
        super().__init__()

        config.orient = "ENU"
        self.action_mode = action_mode
        self.dt = float(dt)
        self.episode_seconds = float(episode_seconds)
        self.max_steps = int(np.ceil(self.episode_seconds / self.dt))
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
        # they no longer drive anything, and `arena_radius` is diagnostic only - the
        # termination test below uses the FIXED flight sphere (flight_radius).
        self.arena_radius_start = float(arena_radius_start)
        self.arena_radius_end = float(arena_radius_end)
        self.curriculum_arena = False
        self.arena_radius = float(arena_radius) if arena_radius is not None else FLIGHT_RADIUS
        self.flight_radius = FLIGHT_RADIUS
        self.envelope_scale: float = 5.0
        self.volume_center = np.array(VOLUME_CENTER, dtype=np.float64)
        # Origin of the actor frame's x,y channels (see ACTOR_FRAME_MODE). Numeric z here
        # is unused by the frame - only x,y are subtracted - but the guard uses it.
        self.anchor_pos = np.zeros(3, dtype=np.float64)

        # Trajectory tracking: the reference generator, the Lighthouse sensor model, and
        # the current reference sample. `maneuver` pins a fixed high-level command; None
        # samples from the mixture every episode, which is what pretraining wants.
        self.traj_cfg = trajectory_config or TrajectoryConfig()
        self.sampler = TrajectorySampler(self.traj_cfg)
        # The flight volume is the plausibility bound for a fix: a sample that places the
        # vehicle metres outside the sphere the task lives in is not a pose, it is a
        # divergence. 4x the sphere radius (8 m) is loose enough to never fire on any
        # legitimate manoeuvre, including the metre-scale re-acquisition after a flip
        # blackout, while still rejecting the metre-to-hundred-metre walk-away observed
        # on the real estimator. Set explicitly here because the model cannot know the
        # volume; free-flight tools leave it at 0 (disabled) by not constructing the env.
        if lighthouse_config is None:
            lighthouse_config = LighthouseConfig(max_fix_range=4.0 * FLIGHT_RADIUS)
        elif float(lighthouse_config.max_fix_range) <= 0.0:
            lighthouse_config = replace(lighthouse_config, max_fix_range=4.0 * FLIGHT_RADIUS)
        self.lighthouse = LighthouseModel(lighthouse_config)
        self._maneuver_command: Optional[str] = maneuver
        self.traj: Optional[Trajectory] = None
        self.ref: Optional[Reference] = None
        self._last_actor_frame: Optional[np.ndarray] = None
        self.w_action_smooth: float = W_ACTION_SMOOTH

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
        # TRAINING FAST PATH. When False, `step()` returns an empty info dict instead of
        # the ~35-key telemetry dict. Nothing in the PPO path reads those keys - SB3's
        # worker adds `terminal_observation` and `TimeLimit.truncated` itself and the
        # Monitor wrapper adds the `episode` statistics - but building them costs ~40 us
        # of worker time per step and a 2.7 KB pickle through the worker pipe. Evaluation
        # and all diagnostics scripts construct the env with the default (True), so no
        # consumer changes. Set False in train.py's env factory only.
        self.telemetry = bool(telemetry)
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
        #     inner loop, which is therefore never asked to track something it cannot.
        #     That figure is only true while the COMPILED body inertia equals the intended
        #     diagonal: MuJoCo silently replaces a tensor that violates the rigid-body
        #     triangle inequality (Ixx+Iyy >= Izz) with an isotropic average, which cost
        #     ~35% here once. quadFiles/quad_mujoco.py now checks it at construction;
        #   - a 360 deg flip needs >= 2*pi/20 = 0.31 s of rotation, which fits inside the
        #     ballistic coast the altitude budget allows.
        #
        # ROLL and PITCH are deliberately symmetric. The old 6 / 20 split was an artefact of
        # the flip task only ever rotating about pitch, and it made every 360 deg ROLL
        # rotation infeasible - the sampler rejected the entire roll family, silently.
        self.max_rate_xy: float = 20.0      # Max roll rate (rad/s)
        self.max_rate_pitch: float = 20.0   # Max pitch rate (rad/s)
        self.max_rate_z: float = 4.0        # Max yaw rate (rad/s); yaw authority is far lower

        # Keep the sampler's notion of authority in step with the command scales above,
        # and its notion of an episode with this env's: a reference that outlives the
        # episode would be truncated mid-manoeuvre, breaking the "every episode ends in
        # the terminal hover" property the reward and the truncation bootstrap rely on.
        self.traj_cfg.rate_limits = {"roll": self.max_rate_xy, "pitch": self.max_rate_pitch}
        self.traj_cfg.episode_seconds = float(self.episode_seconds)

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
        # part of the objective, which now scores tracking error only. It is also NOT what
        # FLIP_PROGRESS integrates: it is clamped to [0, FLIP_THRESHOLD] and only cleared
        # in `reset()`, so a second flip inside one chain would be credited against the
        # first one's angle. `flip_spin_veh` below is the per-segment, signed, uncapped
        # integral instead.
        self.initial_pos = np.array([0.0, 0.0, self.spawn_altitude], dtype=np.float32)
        self.accumulated_pitch: float = 0.0
        self.accumulated_roll: float = 0.0
        self.total_pitch_rotated: float = 0.0
        self.reached_90: bool = False
        self.has_inverted: bool = False
        self.flip_completed: bool = False
        self.flip_completed_time: Optional[float] = None
        self.termination_reason: str = "none"

        # Flip rotation progress (see FLIP_PROGRESS). `flip_spin_veh` is the signed net
        # rotation the vehicle has actually executed within the CURRENT flip segment;
        # `flip_axis` is the unit BODY-frame axis the reference is rotating about, latched
        # once per segment (see `_update_flip_progress`).
        self.flip_spin_veh: float = 0.0
        self.flip_axis: Optional[np.ndarray] = None
        self.flip_progress_err: float = 0.0
        self._flip_last_kind: str = ""
        self._flip_last_spin: float = 0.0

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

    def set_maneuver_weight(self, name: str, weight: float) -> None:
        """Set one family's draw weight in the mixture (the curriculum hook).

        Weights are relative - the sampler normalises them on every draw - so raising the
        `chain` weight lowers every other family's share in proportion. The sampler holds
        THIS config object, so the change is live at the next reset and needs no rebuild.

        Driven from the trainer process over the vec-env RPC channel
        (`env_method("set_maneuver_weight", ...)`, the same path `set_dr_level` uses), and
        usable by hand from a script or ground station. Only the MIXTURE is affected: a
        manoeuvre pinned with `set_command` ignores the weights entirely.
        """
        self.traj_cfg.set_weight(name, weight)

    def get_maneuver_weights(self) -> dict:
        """Current mixture probabilities (normalised), for reporting and tests."""
        return self.traj_cfg.normalized_weights()

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

    def set_envelope_scale(self, scale: float) -> None:
        """
        Sets the relative tracking envelope multiplier for boundary termination.
        When scale >= 4.0 (early in training), the policy is distinctly free to explore
        almost whatever it wants (free exploration, only ground crash and loose safety sphere).
        As training progresses, it anneals down towards 1.0 - 1.5 ('within 50% extra of maneuver demand').
        """
        self.envelope_scale = float(max(0.5, scale))

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
        if ANCHOR_ACTOR_XY:
            # x,y relative to the pose this episode started at; z absolute (see
            # ACTOR_FRAME_MODE). The ERRORS below are already differences and are
            # deliberately NOT anchored - `ref.p - est` is invariant under the same
            # translation, so anchoring it would double-subtract.
            frame[O_POS:O_POS + 2] = est[:2] - self.anchor_pos[:2]
            frame[O_POS + 2] = est[2]
        else:
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

    def _compute_reference_ff(self) -> np.ndarray:
        """Reference feed-forward specific force command, world frame (REF_FF_DIM = 3).

        See the layout note at the top of the file for why this is `a + g` and why it is
        world-frame. Deliberately NOT delayed and NOT noisy: it is a commanded quantity
        computed from the reference, not a measurement, so it carries no sensor error and
        no causal lag. Adding either would only hide the signal the policy needs most.
        """
        ref = self._reference_or_default()
        return (np.asarray(ref.a, dtype=np.float64) + GRAVITY_VEC).astype(np.float32)

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
        the order given by PRIV_TARGET_GROUPS (PRIV_TARGET_DIM = 35 dims).

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

        # Estimator drift: what the actor frame's pose/velocity channels are wrong by.
        # The encoder cannot be robust to a lying estimate unless it is asked to measure
        # the lie; the same fields are what the corruption augmentation shifts when it
        # deliberately breaks those channels.
        est_drift_p = np.asarray(self.lighthouse.p_est, dtype=np.float64) - self.quad.pos
        est_drift_v = np.asarray(self.lighthouse.v_est, dtype=np.float64) - self.quad.vel

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
            est_drift_p,
            est_drift_v,
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

    def _get_stacked_obs(self, privileged_critic: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Env observation vector: [ stacked actor frames | reference feed-forward | delayed
        encoder aux | privileged ].

        The aux block is kept as a SEPARATE, single-frame block rather than being
        interleaved into the stack or appended per frame, so that the actor
        observation stays a clean PREFIX of the vector for any OBS_HISTORY_LEN
        (which keeps AsymmetricActorCriticPolicy's [:actor_obs_dim] slice valid).
        The reference feed-forward sits BETWEEN them for the same reason: the actor must
        see it, the encoder must NOT (its input contract is [o_t | aux] and is frozen), and
        "actor = prefix" only holds if everything actor-facing is at the front.
        The aux is latency-delayed to match the actor frames - an undelayed
        accelerometer would give the encoder an unrealistic peek at the current
        state that no real airframe provides.

        `privileged_critic` may be supplied by a caller that has already computed this
        instant's privileged block. `step()` needs it twice (once for the observation and
        once for the telemetry dict) and it is the most expensive block to assemble, so
        it is built once there and passed to both.
        """
        stacked_actor = np.concatenate(self.obs_history_buffer, dtype=np.float32)
        if privileged_critic is None:
            privileged_critic = self._compute_privileged_critic_obs()
        return np.concatenate(
            [stacked_actor, self._compute_reference_ff(), self._aux_delayed, privileged_critic],
            dtype=np.float32,
        )

    def get_encoder_frame(self) -> np.ndarray:
        """
        The single encoder input frame at the current step: [o_t (29) | aux_t (4)].

        o_t already carries the POST-EMA action applied at t-1 in its prev_action
        slots, so no separate action channel is needed - adding one would be a
        provable duplicate of obs[13:17] one step earlier.
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

    def _update_flip_progress(self, ref, omega_body: np.ndarray) -> None:
        """
        Integrate how far the VEHICLE has rotated through the current flip's rotation.

        Called once per `step()`, BEFORE the reward is scored, and inert for every
        reference that is not a flip (see FLIP_PROGRESS).

        THE AXIS IS A MIXTURE, NOT A CHANNEL. `Flip` applies its spin about a WORLD axis
        (`TrajectorySampler._make_flip` draws `[0,1,0]` pitch 75% of the time and
        `[1,0,0]` roll 25%), so `R = Rot(a, phi) @ R_base` and the BODY-frame axis is
        `R_base^T a`. `R_base` carries the flip's own yaw, so even a pure pitch flip
        rotates about a mixture of body-x and body-y: measured at yaw +38.4 deg it is
        `[0.621, 0.784, 0]`. Integrating only the larger component therefore UNDER-COUNTS
        the rotation by exactly `cos(yaw)`: 2*pi*0.784 = 4.925 rad instead of 6.283, which
        is what an `argmax`-based axis selection produced (measured 4.924). That is a
        silent 22% shortfall on every flip flown at a non-zero heading, and it would have
        been charged to the policy as a rotation it never failed to make.

        So the axis is latched as a UNIT VECTOR from the reference's own body rate
        (`ref.omega`, whose direction is exactly the flip axis and is measurably constant -
        spread 0.00 deg over the whole rotation), taken at the first step where that rate
        is non-zero (it is exactly zero through the climb - the spin is confined to the
        zero-thrust coast window). Latching once keeps the integral on one axis instead of
        letting gyro noise re-aim it mid-rotation.

        THE SIGN. The integral is signed and uncapped, deliberately, and unlike
        `accumulated_pitch`. A vehicle that rotates backwards, or that stalls halfway and
        falls back, must lose the credit - and a k=2 flip needs 4*pi, which
        `accumulated_pitch`'s clamp to FLIP_THRESHOLD would silently cap at half.

        PER-SEGMENT RESET. A new flip segment begins when the kind turns into "flip", or
        when the reference's spin goes BACKWARDS. The second test is what catches two
        flips back-to-back inside one chain: their `kind_at` is "flip" throughout, and
        nothing else in the env is reset between them.
        """
        kind = str(getattr(ref, "kind", "hover")) if ref is not None else "hover"
        spin_ref = float(getattr(ref, "spin", 0.0)) if ref is not None else 0.0

        if kind != "flip":
            # Not in a rotation window at all: nothing to integrate, and the next flip
            # starts from scratch.
            self.flip_spin_veh = 0.0
            self.flip_axis = None
            self.flip_progress_err = 0.0
            self._flip_last_kind = kind
            self._flip_last_spin = spin_ref
            return

        if self._flip_last_kind != "flip" or spin_ref + 1e-9 < self._flip_last_spin:
            self.flip_spin_veh = 0.0
            self.flip_axis = None

        if self.flip_axis is None:
            ref_w = np.asarray(getattr(ref, "omega", np.zeros(3)), dtype=np.float64)
            rate = float(np.linalg.norm(ref_w))
            if rate >= FLIP_SPIN_AXIS_MIN_RATE:
                self.flip_axis = ref_w / rate          # unit body-frame rotation axis

        if self.flip_axis is not None:
            self.flip_spin_veh += float(np.dot(omega_body, self.flip_axis)) * self.dt

        self.flip_progress_err = abs(spin_ref - self.flip_spin_veh)
        self._flip_last_kind = kind
        self._flip_last_spin = spin_ref

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

        WHY BOUNDED KERNELS. Each term is bounded in [0, 1], so the achievable return
        per step is comparable across a hover, a figure-8 and a 360 deg flip and PPO's
        single value head does not have to span wildly different magnitudes. They also
        saturate rather than growing without bound, so an early catastrophic error
        cannot dominate the gradient before the policy can fly at all.

        WHICH BOUNDED KERNEL is a separate decision and is NOT cosmetic - see
        REWARD_KERNEL above. A gaussian tail e^(-x^2) is effectively dead by x = 2, and a
        policy that is 2-5 tolerances away from its target is the normal case for most of
        training (and the measured case for the live point-hold at 0.2-0.7 m). The
        cauchy tail 1/(1+x^2) keeps a usable gradient there while staying flatter than
        the gaussian near the target, so it widens the basin without loosening the
        precision end of the objective.

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

        # Through a flip the rotvec error above cannot see rotation progress - it wraps,
        # reading ~0 at both ends of the turn - so the vehicle's accumulated rotation is
        # required as well. See FLIP_PROGRESS: this is inert for every other family, and
        # for a vehicle that IS rotating it matches the rotvec error rather than adding to
        # it, so only the refusal to rotate is penalised.
        if FLIP_PROGRESS and ref.kind == "flip":
            att_err = max(att_err, self.flip_progress_err)

        r_pos = _tracking_kernel(p_err, tol["pos"])
        r_vel = _tracking_kernel(v_err, tol["vel"])
        r_att = _tracking_kernel(att_err, tol["att"])
        r_rate = _tracking_kernel(w_err, tol["rate"])

        delta_action_norm = float(np.linalg.norm(action - self.prev_action))
        # The action-smoothness term keeps the gaussian SHAPE on purpose: it is a
        # regulariser, not a tracking objective, its measured value is 0.97-1.00
        # (0.97 even through a flip), and giving it heavy tails would only widen the
        # slack on the one term that is supposed to be tight.
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

        # Physical ground contact check:
        if self.quad.check_ground_contact():
            # In takeoff mode or when starting on/near the ground, allow a grace window
            # while the vehicle is upright on its landing legs during spin-up and initial liftoff.
            # Do NOT terminate as long as:
            # 1. Episode started on or near ground (spawn_pos[2] < 0.20 or kind == "takeoff")
            # 2. Quad is upright on landing legs (dcm[2, 2] > 0.50, i.e. tilt < 60 deg, not flipped over)
            # 3. Within initial liftoff grace window (t < 1.2s and has not already achieved sustained flight)
            is_takeoff = (self.spawn_pos[2] < 0.20 or getattr(self.ref, "kind", "") == "takeoff")
            is_upright = (self.quad.dcm[2, 2] > 0.50)
            in_liftoff_window = (self.t < 1.2 and not getattr(self, "_has_lifted_off", False))

            if is_takeoff and is_upright and in_liftoff_window:
                # Legitimate resting / spool-up / liftoff on ground: do not terminate
                pass
            else:
                self.termination_reason = "ground_crash"
                return True

        scale = float(getattr(self, "envelope_scale", 5.0))

        # Relative Maneuver-Specific Flight Envelopes
        # When scale >= 4.0 (early exploration), the agent is distinctly free to explore
        # and discover coarse dynamics without premature termination.
        # Below 4.0, bounds progressively engage and tighten down to ~1.5 (within 50% extra margin).
        if self.ref is not None and scale < 4.0:
            kind = str(getattr(self.ref, "kind", "hover"))
            p_rel = self.quad.pos - self.ref.p
            dist_rel = float(np.linalg.norm(p_rel))
            eff_scale = scale / 1.5  # 1.0 at target curriculum end (scale=1.5)

            if kind == "flip":
                # Flip: Bounded vertical hop and horizontal drift
                # A flip must rotate cleanly in place and NOT balloon upwards into space or drift away
                dz_hop = float(self.quad.pos[2] - self.ref.p[2])
                d_xy = float(np.linalg.norm(p_rel[:2]))
                max_ceiling_excursion = 0.60 * eff_scale
                max_xy_drift = 0.80 * eff_scale

                if dz_hop > max_ceiling_excursion:
                    self.termination_reason = "flip_ballooned_ceiling"
                    return True
                if d_xy > max_xy_drift:
                    self.termination_reason = "flip_drifted_xy"
                    return True
            else:
                # Traverse / Smooth / Acrobatic Maneuvers (slalom, figure8, orbit, waypoints, lissajous, v8, chain, hover)
                # Evaluated along its tracking tunnel: allows wide lateral traverse (e.g. slalom) as long as
                # it tracks the reference demand.
                base_tol = TRACK_TOL.get(kind, TRACK_TOL["hover"])["pos"]
                # At eff_scale=1.0, allowed deviation is 3.5x base tolerance (~50% extra margin over comfort zone)
                max_tunnel_error = max(0.50, 3.5 * base_tol) * eff_scale
                if dist_rel > max_tunnel_error:
                    self.termination_reason = f"breached_{kind}_tunnel"
                    return True

        # Outer safety guard (prevents unbounded runaway in case of complete divergence)
        # In early free exploration (scale >= 4.0), allows a generous 3.5m outer bubble
        max_outer_r = 3.5 if scale >= 4.0 else 2.5
        centre = np.array([self.anchor_pos[0], self.anchor_pos[1], VOLUME_CENTER[2]])
        d = self.quad.pos - centre
        if float(np.dot(d, d)) > max_outer_r ** 2:
            self.termination_reason = "out_of_volume"
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
        self.accumulated_roll = 0.0
        self.total_pitch_rotated = 0.0
        self.reached_90 = False
        self.has_inverted = False
        self.flip_completed = False
        self.flip_completed_time = None
        self.termination_reason = "none"
        self._has_lifted_off = False
        self.flip_spin_veh = 0.0
        self.flip_axis = None
        self.flip_progress_err = 0.0
        self._flip_last_kind = ""
        self._flip_last_spin = 0.0

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
        # derivative are zero there; the flip because its spin starts at 0; the vertical
        # figure-eight and every chain segment because they are built rest-to-rest with a
        # C2 ramp-in), so the spawn attitude is level plus a small perturbation.
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
        # Ground protection: if starting on or near ground (z < 0.06m), ensure position
        # is strictly resting on/above floor (legs resting at z ~ 0.025m) and eliminate negative vertical jitter
        if ref0.p[2] < 0.06 or spawn_pos[2] < 0.025:
            spawn_pos[2] = max(0.025, float(ref0.p[2] + abs(pos_jitter[2]) * 0.2))

        # Spawn at the REFERENCE attitude, not at level.
        #
        # The reference is level in TILT at t = 0 for some manoeuvres but not all: an
        # Orbit is already BANKED at t = 0 (a circle needs a constant centripetal
        # acceleration from the first instant) and a Lissajous with a phase offset is
        # tilted too. Its YAW is randomised on top of that. Spawning level would therefore
        # start the episode with a heading and bank error the policy can do nothing about,
        # charged against the attitude kernel and blamed on the policy.
        base_quat = self._dcm_to_quat(ref0.R)
        if self.random_initial_att:
            att_rp = 0.05 + dr * 0.12
            att_y = 0.035 + dr * 0.10
            # If resting on ground, keep attitude upright so props don't collide with floor
            if spawn_pos[2] < 0.05:
                att_rp = min(0.02, att_rp * 0.25)
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
            vel_lin = INIT_VEL_RANGE[0] + dr * (INIT_VEL_RANGE[1] - INIT_VEL_RANGE[0])
            vel_ang = INIT_RATE_RANGE[0] + dr * (INIT_RATE_RANGE[1] - INIT_RATE_RANGE[0])
            # Spawn on the reference's initial VELOCITY, then add the kick. The reference
            # has v = 0 at t = 0 for every manoeuvre, so this is the kick alone in practice.
            v_init = ref0.v + self.np_random.uniform(-vel_lin, vel_lin, size=3)
            # When spawned resting on ground, prevent kicking downwards through the floor
            if spawn_pos[2] < 0.05 and v_init[2] < 0.0:
                v_init[2] = 0.0
            self.quad.data.qvel[0:3] = v_init
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
        # The actor frame's origin for this episode. Set BEFORE the first
        # `_compute_actor_obs()` below, so the very first frame already reads (0, 0) in x,y.
        self.anchor_pos = spawn_pos.copy()
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
            "anchor_pos": self.anchor_pos.copy(),
            "accumulated_pitch": self.accumulated_pitch,
            "accumulated_roll": self.accumulated_roll,
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
            "lighthouse_fix_rejected": int(self.lighthouse.n_fix_rejected),
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
            # The gyro bias is a hardware property, not a sensor-noise option: the rate
            # loop must see the same biased measurement the actor does. Gating it on
            # obs_noise made the two disagree about the vehicle's own body rate.
            self.quad.update(
                t=self.t,
                dt=self.dt,
                wind=self.wind,
                rate_cmd=(throttle_cmd, omega_des_cmd),
                rate_pid=self.rate_pid,
                gyro_bias=self.gyro_bias,
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
        delta_roll = float(self.quad.omega[0] * self.dt)
        self.total_pitch_rotated += max(0.0, delta_pitch)
        self.accumulated_pitch = min(FLIP_THRESHOLD, max(0.0, self.accumulated_pitch + delta_pitch))
        self.accumulated_roll = min(FLIP_THRESHOLD, max(0.0, self.accumulated_roll + delta_roll))
        if not self.flip_completed and self.accumulated_pitch >= FLIP_THRESHOLD:
            self.flip_completed = True
            self.flip_completed_time = float(self.t)
        if self.quad.dcm[2, 2] < -0.2:
            self.has_inverted = True
        if self.quad.pos[2] > 0.08:
            self._has_lifted_off = True
        if self.accumulated_pitch >= 0.5 * np.pi:
            self.reached_90 = True

        # Flip rotation progress - the ONLY flight-progress quantity that feeds the reward
        # (see FLIP_PROGRESS). Runs before `_compute_reward` below so the reward sees the
        # rotation this step.
        self._update_flip_progress(
            self.ref, np.asarray(getattr(self.quad, "omega_filtered", self.quad.omega),
                                 dtype=np.float64))

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
        # Assembled once and shared with the telemetry dict below. Both describe the same
        # instant (nothing in between mutates the plant), and this is the most expensive
        # block of the observation, so recomputing it would be pure overhead.
        privileged_critic = self._compute_privileged_critic_obs()
        stacked_obs = self._get_stacked_obs(privileged_critic)

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

        if self.telemetry:
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
                "anchor_pos": self.anchor_pos.copy(),
                "accumulated_pitch": self.accumulated_pitch,
                "accumulated_roll": self.accumulated_roll,
                "total_pitch_rotated": self.total_pitch_rotated,
                "flip_spin_veh": self.flip_spin_veh,
                "flip_progress_err": self.flip_progress_err,
                "flip_axis": (None if self.flip_axis is None
                              else [float(x) for x in self.flip_axis]),
                "reference_spin": float(getattr(self.ref, "spin", 0.0)) if self.ref is not None else 0.0,
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
                "lighthouse_fix_rejected": int(self.lighthouse.n_fix_rejected),
                "actor_obs": self.get_actor_obs(),
                "privileged_obs": privileged_critic,
                "stock_obs": self._compute_stock_obs(),
                "single_obs": delayed_obs.copy(),
                "encoder_frame": self.get_encoder_frame(),
                "dr_eff": float(self.dr_eff),
                "active_disturbances": self.active_disturbances,
            }
        else:
            # Training fast path (see the `telemetry` note in __init__): the PPO path reads
            # no env info key, and an empty dict is what SB3 fills with
            # `terminal_observation` / `TimeLimit.truncated` in the worker process.
            info = {}

        # Shift action history for 2nd-order difference
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()

        return stacked_obs, reward, terminated, truncated, info


# Backwards compatibility alias
CustomQuadEnv = QuadFlipEnv


if __name__ == "__main__":
    env = QuadFlipEnv()
    obs, info = env.reset()
    print("✓ QuadFlipEnv (trajectory tracking) initialized!")
    print(f"  Observation shape : {obs.shape}")
    print(f"  Action shape      : {env.action_space.shape}")
    print(f"  Target state      : {info['target_state']}")
    print(f"  Pitch direction   : {'Front-flip (+Y)' if env.pitch_direction > 0 else 'Back-flip (-Y)'}")
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    print(f"  Sample step reward: {rew:.2f}")
    print("All checks passed.")