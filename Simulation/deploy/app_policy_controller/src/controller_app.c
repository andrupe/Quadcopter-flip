/**
 * Policy controller app - the trained encoder+PPO stack flying on a Crazyflie 2.1.
 *
 * WHAT THIS IS
 * ------------
 * An out-of-tree controller (`controllerOutOfTree`, registered by CONFIG_CONTROLLER_OOT
 * and selected by `stabilizer.controller = <index of OutOfTree>`). It runs INSIDE the 1 kHz
 * stabilizer task with direct access to the estimator state and the sensors, which is what
 * makes an onboard policy possible without any radio state transport:
 *
 *   every 1 kHz tick : feed the stock PID a rate setpoint (ours when armed, the pilot's
 *                      incoming setpoint when not)
 *   every 10th tick  : build the 45-dim actor input, run the GRU+MLP (policy_net.c),
 *                      advance the onboard reference (a hold, or a baked manoeuvre table
 *                      relocated onto the current pose at launch), map action -> rate
 *                      setpoint
 *
 * DISARMED == STOCK. When not armed this calls `controllerPid(control, setpoint, ...)`
 * with the incoming setpoint, i.e. the vehicle behaves exactly as it did before the app
 * was flashed, and the pilot's radio always has authority. Disarm mid-air is therefore
 * instantaneous and total.
 *
 * SHADOW MODE (`policy.shadow`, default 1). The full pipeline runs and logs but commands
 * nothing. This is the bring-up step from the deployment plan: fly by hand and compare the
 * logged observation against the simulator BEFORE the policy is ever allowed to command.
 * Set `policy.shadow = 0` only after that comparison has been made.
 *
 * SAFETY ENVELOPE while armed: tilt, altitude and NaN guards run every tick; any breach
 * disarms (handing control straight back to the pilot's stream) and latches a reason code
 * that is visible in the log group and the console.
 *
 * UNITS AND CONVENTIONS
 * ---------------------
 * Everything entering or leaving the policy uses the SIMULATOR's units, because the
 * network was trained on them (all constants come from the generated policy_weights.h):
 *   rates rad/s, distances m, quaternion (w, x, y, z), specific force m/s^2.
 * The Crazyflie firmware speaks deg/s, Gs and its own legacy body conventions, so the
 * conversion is done HERE, in one place (`build_frame` for inputs, `synth_setpoint` for
 * outputs), with the sign constants below doing the mapping. They are the first thing the
 * shadow log is meant to confirm - see SIGN_FROM_* below.
 *
 * COMMANDS (appchannel):
 *   0x01 ARM         sync the hold reference to the current estimate and arm
 *   0x02 DISARM      disarm immediately
 *   0x03 FLIP        launch the baked flip (shorthand for 0x05 REF_KIND_FLIP)
 *   0x04 STATUS      request an immediate status packet
 *   0x05 <kind>      launch a baked manoeuvre - flip, orbit, figure8, lissajous, slalom,
 *                    waypoints - or 0xFF to stop and hold wherever it is. Only while
 *                    armed: every manoeuvre is relocated onto the CURRENT pose and
 *                    heading at launch, and hands back to a hold anchored where it ended.
 * Status packets (0xA5 ...) are also sent every 100 ms while connected, and carry the
 * active manoeuvre byte.
 */

#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "app.h"
#include "controller.h"
#include "controller_pid.h"
#include "attitude_controller.h"
#include "app_channel.h"
#include "log.h"
#include "param.h"
#include "math3d.h"
#include "debug.h"

#include "FreeRTOS.h"
#include "task.h"

#include "policy_backend.h"            // -> policy_net.h or policy_net_stedgeai.h
#include "reference.h"
#include "analytic_flip.h"
#include "policy_guard.h"               // failsafe descent + estimator plausibility gate

#define DEBUG_MODULE "POLICY"

// ======================================================================================
// COMPILE-TIME CONTRACT WITH THE EXPORTED WEIGHTS
//
// POLICY_FRAME_ANCHORED_XY is emitted by export_policy.py straight from
// `quad_flip_env.ACTOR_FRAME_MODE`, so it describes the frame the policy was TRAINED on.
// build_frame() below must agree with it: the anchored and absolute conventions differ
// only in the CONTENT of o_t's x,y slots, so a mismatch changes nothing structurally - the
// net still runs and still commands rates, it is just conditioned on a frame it never saw.
// The `#error` turns that silent failure into a build failure. Re-export (and rebuild) if
// this ever fires:
//     .venv/bin/python Simulation/deploy/export_policy.py
// ======================================================================================
#ifndef POLICY_FRAME_ANCHORED_XY
#error "policy_weights.h predates the actor-frame anchoring change (no POLICY_FRAME_ANCHORED_XY). Re-export with Simulation/deploy/export_policy.py."
#endif

// ======================================================================================
// CONVENTIONS - the sign of each axis between the sim and the firmware.
//
// These are NOT guesses; they are recorded here so the shadow log can confirm them in one
// flight, and so a wrong sign has exactly one place to be corrected:
//
//   * gyro  : the sim's body frame is (x = nose, y = left, z = up). The BMI088 is mounted
//             the same way, but the LEGACY flight loop negates gyro.y when it feeds its
//             rate PID (controller_pid.c), which is how the "legacy CF pitch convention"
//             shows up in the control path.
//   * rates : a rate setpoint is consumed by that same (legacy) loop: the PID tracks
//             `-gyro.y` towards `setpoint.attitudeRate.pitch`, so commanding the sim's
//             pitch rate needs the sign opposite to the gyro mapping.
//   * yaw   : the state estimator's yaw is the standard z-up heading, so the yaw rate
//             channel passes through (the CRTP decoder negates its INPUT, which is a
//             pilot-side convention, not a body-frame one).
//
// SHADOW CHECK (one hand-flown minute is enough): command +X roll rate, watch gyro.x sign;
// command +Y pitch, watch gyro.y; yaw likewise; and compare `state->attitude.yaw` against
// the sim's `yaw_of(dcm)` for a slow spin on the spot.
// ======================================================================================
#define SIGN_GYRO_ROLL_TO_SIM   (+1.0f)
#define SIGN_GYRO_PITCH_TO_SIM  (+1.0f)
#define SIGN_GYRO_YAW_TO_SIM    (+1.0f)
#define SIGN_RATE_ROLL          (+1.0f)   // sim rad/s -> setpoint deg/s
#define SIGN_RATE_PITCH         (-1.0f)   // see the gyro note above
#define SIGN_RATE_YAW           (+1.0f)

// ======================================================================================
// constants
// ======================================================================================
#define POLICY_DECIMATION 10              // 1 kHz tick / 10 = the policy's 100 Hz
#define POLICY_THRUST_MAX_UNITS 60000.0f  // crtp_commander_rpyt.c MAX_THRUST (legacy units)

// Abort envelope (params override these at runtime)
#define POLICY_DEFAULT_MAX_TILT_DEG 55.0f
#define POLICY_MIN_Z 0.04f
#define POLICY_MAX_Z 2.1f   // Cage ceiling 2.0 m + 0.1 m margin
// Consecutive 1 kHz ticks below POLICY_MIN_Z before the flight is abandoned. The floor needs a
// debounce because a LEGITIMATE post-blackout innovation reaches 2.67 m and can cross it
// transiently, while a real ground contact HOLDS it. 100 ms = 10 fix intervals at the deck's
// own ~50 Hz - long enough to reject a re-acquisition excursion, short enough to disarm before
// the props have spent a second on the ground.
#define POLICY_GROUND_HOLD_TICKS 100u

// Failsafe descent defaults (see policy_guard.h). The ramp starts at the thrust that was
// last commanded, floored at hover, and runs to zero; from a 1.2 m entry that reaches the
// ground in ~0.95 s, i.e. inside the ramp, so the normal exit is the GROUND condition.
#define POLICY_FS_RAMP_S 1.2f
#define POLICY_FS_TIMEOUT_S 6.0f
#define POLICY_TICK_HZ 1000.0f            // controllerOutOfTree is called at 1 kHz

// Estimator plausibility defaults.  MEASURED in Simulation/lighthouse.py: a legitimate
// post-blackout innovation reaches 2.67 m and 1.50 m/s, so these MUST stay comfortably
// above that (an earlier draft used 1.5 m/s and rejected every fix after a 2.2 s blackout).
#define POLICY_MAX_FIX_JUMP_M 5.0f
#define POLICY_MAX_FIX_DV_MS 4.0f
#define POLICY_MAX_FIX_RANGE_M 8.0f       // 4 x FLIGHT_RADIUS, as the sim env enables it
#define POLICY_MAX_REJECT_STREAK 15       // 0.15 s at 100 Hz

// Battery: the sim's aux channel is V / V_nom with V_nom = 3.7 V (quad_mujoco.py).
#define POLICY_VBAT_NOMINAL 3.7f
#define POLICY_GRAVITY 9.81f

// The layout the ONBOARD image must reproduce, mirroring quad_flip_env. These are asserted
// rather than assumed: the generated header is the only place they come from, and a sim-side
// change that reaches the exporter but not this file would otherwise be invisible here.
_Static_assert(POLICY_REF_FF_DIM == 3, "quad_flip_env.REF_FF_DIM is 3");
_Static_assert(POLICY_ENC_IN_DIM == POLICY_O_T_DIM + POLICY_ENC_AUX_DIM,
               "the encoder frame is [o_t | aux]");

// policy_guard.h MIRRORS the firmware's setpoint mode enum so it can stay host-compilable
// (see guard_host_check.py). If the firmware ever renumbers them, this must be a build
// error rather than a silent change in what the vehicle does when it gives up.
_Static_assert(POLICY_GUARD_MODE_DISABLE == (int)modeDisable,
               "policy_guard mirrors stabilizer_types.h modeDisable");
_Static_assert(POLICY_GUARD_MODE_ABS == (int)modeAbs,
               "policy_guard mirrors stabilizer_types.h modeAbs");
_Static_assert(POLICY_GUARD_MODE_VELOCITY == (int)modeVelocity,
               "policy_guard mirrors stabilizer_types.h modeVelocity");

// appchannel opcodes
#define CMD_ARM 0x01
#define CMD_DISARM 0x02
#define CMD_FLIP 0x03                       // kept: shorthand for CMD_PLAY REF_KIND_FLIP
#define CMD_STATUS 0x04
#define CMD_PLAY 0x05                       // + one byte: REF_KIND_* to launch (0xFF = hold)
#define STATUS_MAGIC 0xA5

// abort reasons (latched for the log)
#define ABORT_NONE 0
#define ABORT_TILT 1
#define ABORT_Z 2
#define ABORT_ESTIMATE 3
#define ABORT_COUNT 4

// ======================================================================================
// state
// ======================================================================================
static policy_state_t g_policy;                       // ~3 KB, BSS (no stack use per tick)
static ref_state_t g_ref;

static uint8_t g_armed = 0;                           // 0/1 (uint8 so it can be a param/log)
static uint8_t g_arm_param = 0;                       // write-through param mirror
static uint8_t g_shadow = 1;                          // 1 = log only, never command
static uint8_t g_play_kind = REF_KIND_NONE;           // manoeuvre requested, pending next 100 Hz tick
static uint8_t g_abort_reason = ABORT_NONE;
static uint8_t g_mode = 0;                            // mirror of g_ref.mode for the log
static uint32_t g_tick = 0;                           // 1 kHz calls seen
static uint32_t g_step100 = 0;                        // 100 Hz ticks since boot
static float g_applied[POLICY_ACT_DIM];               // post-EMA applied action (obs 13:17)
static float g_last_act[POLICY_ACT_DIM];              // raw action of the last policy step
static float g_last_thrust_units = 0.0f;
static float g_thrust_scale = POLICY_THRUST_MAX_UNITS; // action +1 -> this many units
static float g_max_tilt_deg = POLICY_DEFAULT_MAX_TILT_DEG;
static float g_min_z = POLICY_MIN_Z;
static float g_max_z = POLICY_MAX_Z;
static uint32_t g_low_z_ticks = 0;             // ground-guard debounce, see policy_safety_check
static float g_last_tilt_deg = 0.0f;
static float g_last_p_err = 0.0f;

// ---- Flip Phase-Triggered State Machine ----------------------------------------------
typedef enum {
    FLIP_PHASE_IDLE = 0,
    FLIP_PHASE_CLIMB = 1,       // Phase 1: Dynamic Climb
    FLIP_PHASE_ROTATION = 2,    // Phase 2: Timed Flip Rotation
    FLIP_PHASE_RECOVERY = 3,    // Phase 3: Powered Arrest Catch & Hover
} flip_phase_t;

static flip_phase_t g_flip_phase = FLIP_PHASE_IDLE;
static uint8_t g_flip_phase_log = 0;                  // mirror for log
static bool g_flip_inverted = false;                  // true once vehicle inverts (>120 deg tilt)
static float g_flip_z0 = 0.0f;
static float g_flip_gyro_angle = 0.0f;               // cumulative pitch rotation angle (deg)
static uint32_t g_flip_climb_start_step = 0;
static uint32_t g_flip_rot_start_step = 0;

// Live tunable parametric flip configuration
static float g_tune_flip_peak_dps = ANALYTIC_FLIP_DEFAULT_PEAK_DPS;
static float g_tune_flip_pop_pct = ANALYTIC_FLIP_DEFAULT_POP_PCT;
static float g_tune_flip_rate_frac = ANALYTIC_FLIP_DEFAULT_RATE_FRAC;
// The thrust the reference is PLANNED for, in newtons. The single biggest lever on the
// flip's altitude excursion: raising it shortens `c0 = v0/u` and shrinks the climb.
// Initialised from the sim plant constant the policy was trained against (see
// ANALYTIC_FLIP_DEFAULT_MAX_THRUST in analytic_flip.h for the whole story).
// MEASURED after the 20 mm motor swap: the real vehicle hovers at 33.7% of the command
// range, i.e. `m*g/hover_fraction` = 0.96 N, so 0.60 is conservative and leaves headroom
// to raise this live via `traj_flip.max_thrust`.
static float g_tune_flip_max_thrust = POLICY_SIM_MAX_THRUST_N;
static uint8_t g_tune_flip_axis = ANALYTIC_FLIP_DEFAULT_AXIS;  // 0 = pitch, 1 = roll
static uint8_t g_tune_flip_analytic = 1;                       // 1 = analytic on-the-fly, 0 = prebaked table

static analytic_flip_state_t g_analytic_flip;
static bool g_flip_is_analytic = false;
static uint32_t g_flip_active_rot_start = REF_FLIP_ROT_START_ROW;
static uint32_t g_flip_active_catch_start = REF_FLIP_CATCH_START_ROW;

// THE FLIP REFERENCE IS NEVER RE-BASED (2026-09-25). The baked table's OWN timeline is
// authoritative: from launch it plays rows 0..REF_FLIP_ROT_START_ROW as its powered climb,
// then its ballistic coast, then its powered arrest, at 100 Hz. The phase machine below
// therefore never touches `g_ref.start_step` - it only decides WHEN EACH SAFETY ENVELOPE
// OPENS (`is_flip` / `is_flip_rec` in policy_safety_check) and logs the catch.
//
// It used to fast-forward the clock three times: into the coast, to the last coast row, and
// to the arrest row. Measured 2026-09-25 against the restored reference, the first of those
// entered the ballistic coast at row 25 (dz +0.21 m, vz +1.64 m/s) instead of the table's
// own row 60 (dz +1.18 m, vz +3.85 m/s) - 0.97 m early, with less than half the climb
// momentum the reference was built around. Every one of those jumps also puts a STEP into
// the policy's errors, which is the defect class the whole reference layer exists to avoid
// (check_trajectories section I). Trust the table: gen_references.py --check verifies it is
// feasible end to end, and it was flown clean from rest as plain playback.
// REF_FLIP_ROT_START_ROW / REF_FLIP_CATCH_START_ROW remain the phase boundaries.
_Static_assert(REF_FLIP_ROT_START_ROW < REF_FLIP_CATCH_START_ROW,
               "REF_FLIP_ROT_START_ROW/REF_FLIP_CATCH_START_ROW from the generated header "
               "are out of order - re-run gen_references.py");

// ---- T0-A / T0-B state ----------------------------------------------------------------
// The gate and the failsafe are the two things that decide whether a real flight is
// abandoned, so they live in `policy_guard.c` where `guard_host_check.py` can drive them.
static guard_t g_guard;
static failsafe_t g_failsafe;
static uint8_t g_failsafe_active = 0;         // mirrors failsafe.active for the log
static uint32_t g_failsafe_count = 0;
static float g_fs_thrust_units = 0.0f;        // what the failsafe last commanded
static uint8_t g_est_hold = 0;                // mirrors guard.hold for the log
// The feed-forward block's vertical component (a_ref_z + g). Logged because it is the one
// channel whose value is self-evident on the bench: ~9.81 in a hover, and ~0 through a
// flip's ballistic coast. It is the cheapest possible proof that the block is alive.
static float g_last_ff_z = 0.0f;
static float g_last_vbat = POLICY_VBAT_NOMINAL;
static float g_last_state_z = 0.0f;
// ORIGIN OF THE ACTOR FRAME'S x,y CHANNELS (quad_flip_env.ACTOR_FRAME_MODE). Captured at
// ARM - which is this controller's episode start, the same moment the GRU is reset - and
// subtracted from the position the frame reports. The env anchors the same way on
// `reset()`/`adopt_state()`. z is deliberately NOT anchored: the floor and ceiling are
// absolute, so an absolute z is an exact, launch-height-independent ground cue.
static float g_anchor_xy[2] = {0.0f, 0.0f};
static logVarId_t g_vbat_id = 0;
static bool g_vbat_id_valid = false;
static setpoint_t g_rate_sp;                          // our synthesized rate setpoint

// ======================================================================================
// small maths helpers
// ======================================================================================
static void dcm_from_quat(const float q[4], float R[9])   // q = (w, x, y, z)
{
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1.0f - 2.0f * (y * y + z * z);  R[1] = 2.0f * (x * y - w * z);        R[2] = 2.0f * (x * z + w * y);
    R[3] = 2.0f * (x * y + w * z);         R[4] = 1.0f - 2.0f * (x * x + z * z);  R[5] = 2.0f * (y * z - w * x);
    R[6] = 2.0f * (x * z - w * y);         R[7] = 2.0f * (y * z + w * x);         R[8] = 1.0f - 2.0f * (x * x + y * y);
}

// Rotation-vector (log map) of a rotation matrix - the sim's att_err channel.
static void rotvec_from_dcm(const float R[9], float out[3])
{
    const float tr = R[0] + R[4] + R[8];
    const float cos_angle = fmaxf(-1.0f, fminf(1.0f, 0.5f * (tr - 1.0f)));
    const float angle = acosf(cos_angle);
    if (angle < 1e-5f) {
        // small-angle: the vector is the skew-symmetric part
        out[0] = 0.5f * (R[7] - R[5]);
        out[1] = 0.5f * (R[2] - R[6]);
        out[2] = 0.5f * (R[3] - R[1]);
        return;
    }
    if (angle > 3.14159265f - 1e-4f) {
        // Near-pi singularity: R - R^T -> 0 (symmetric), so recover axis from 0.5 * (R + I) = u * u^T
        // exactly matching QuadFlipEnv._attitude_error_rotvec
        const float a00 = fmaxf(0.0f, 0.5f * (R[0] + 1.0f));
        const float a11 = fmaxf(0.0f, 0.5f * (R[4] + 1.0f));
        const float a22 = fmaxf(0.0f, 0.5f * (R[8] + 1.0f));
        float ax, ay, az;
        if (a00 >= a11 && a00 >= a22) {
            const float d = sqrtf(a00);
            ax = d;
            ay = (d > 1e-6f) ? (0.5f * R[3]) / d : 0.0f;
            az = (d > 1e-6f) ? (0.5f * R[6]) / d : 0.0f;
        } else if (a11 >= a22) {
            const float d = sqrtf(a11);
            ax = (d > 1e-6f) ? (0.5f * R[1]) / d : 0.0f;
            ay = d;
            az = (d > 1e-6f) ? (0.5f * R[7]) / d : 0.0f;
        } else {
            const float d = sqrtf(a22);
            ax = (d > 1e-6f) ? (0.5f * R[2]) / d : 0.0f;
            ay = (d > 1e-6f) ? (0.5f * R[5]) / d : 0.0f;
            az = d;
        }
        const float norm = sqrtf(ax * ax + ay * ay + az * az);
        if (norm > 1e-6f) {
            const float inv_n = 1.0f / norm;
            ax *= inv_n; ay *= inv_n; az *= inv_n;
        } else {
            ax = 0.0f; ay = 1.0f; az = 0.0f; // fallback to pitch axis
        }
        out[0] = angle * ax;
        out[1] = angle * ay;
        out[2] = angle * az;
        return;
    }
    const float s = sinf(angle);
    if (fabsf(s) < 1e-6f) {
        out[0] = out[1] = out[2] = 0.0f;
        return;
    }
    const float k = angle / (2.0f * s);
    out[0] = k * (R[7] - R[5]);
    out[1] = k * (R[2] - R[6]);
    out[2] = k * (R[3] - R[1]);
}

static float yaw_from_state(const state_t *state)
{
    // Estimator yaw (deg, z-up heading) -> radians. The sim's yaw_of() is the standard
    // atan2(R10, R00), which for a z-up body frame is this same quantity.
    return radians(state->attitude.yaw) * SIGN_GYRO_YAW_TO_SIM;
}

static float tilt_deg_from_quat(const float q[4])
{
    float R[9];
    dcm_from_quat(q, R);
    return degrees(acosf(fmaxf(-1.0f, fminf(1.0f, R[8]))));
}

// ======================================================================================
// observation assembly (the actor frame + aux, SIM units)
// ======================================================================================
static void build_frame(const sensorData_t *sensors, const state_t *state,
                        float frame[POLICY_ENC_IN_DIM], float ref_ff[POLICY_REF_FF_DIM])
{
    // -- onboard reference (hold or manoeuvre), then the four error channels ---------
    float rp[3], rv[3], rR[9], rw[3], ra[3];
    if (g_ref.mode == REF_MODE_MANOEUVRE && g_ref.kind == REF_KIND_FLIP && g_flip_is_analytic) {
        analytic_flip_sample(&g_analytic_flip, g_step100, rp, rv, rR, rw, ra);
    } else {
        ref_sample(&g_ref, g_step100, rp, rv, rR, rw, ra);
    }

    // The actor's feed-forward block: a_ref + g*e_z, world frame - exactly what the training
    // environment feeds (quad_flip_env._compute_reference_ff). Built HERE, next to the same
    // ref_sample() the error channels come from, so both describe the same instant.
    ref_feed_forward(ra, POLICY_SIM_GRAVITY, ref_ff);
    g_last_ff_z = ref_ff[2];

    // T0-B: the GATED estimate, not the raw one.  While the plausibility gate holds a
    // rejected sample, pos / vel / p_err / v_err all come from that ONE held sample, so the
    // policy input stops moving AND stays coherent - the same rule
    // Simulation/encoder/corruption.py enforces (a corruption that moved the position but
    // not the error would hand the policy the disagreement as a tell).
    const float p[3] = {g_guard.p[0], g_guard.p[1], g_guard.p[2]};
    const float v[3] = {g_guard.v[0], g_guard.v[1], g_guard.v[2]};

    // CF quaternion_t is (x, y, z, w); the sim's block is (w, x, y, z).
    const float q[4] = {state->attitudeQuaternion.w, state->attitudeQuaternion.x,
                        state->attitudeQuaternion.y, state->attitudeQuaternion.z};

    // gyro deg/s -> rad/s in the sim's body axes
    const float w[3] = {
        SIGN_GYRO_ROLL_TO_SIM * radians(sensors->gyro.x),
        SIGN_GYRO_PITCH_TO_SIM * radians(sensors->gyro.y),
        SIGN_GYRO_YAW_TO_SIM * radians(sensors->gyro.z),
    };

    float R_meas[9];
    dcm_from_quat(q, R_meas);

    // att_err = rotvec(R_ref^T R_meas)   (same form as quad_flip_env._compute_actor_obs)
    float RtR[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            RtR[3 * i + j] = rR[0 * 3 + i] * R_meas[0 * 3 + j]
                           + rR[1 * 3 + i] * R_meas[1 * 3 + j]
                           + rR[2 * 3 + i] * R_meas[2 * 3 + j];
        }
    }
    float att_err[3];
    rotvec_from_dcm(RtR, att_err);

    int o = 0;
    // O_POS(3), O_QUAT(4), O_OMEGA(3)
#if POLICY_FRAME_ANCHORED_XY
    frame[o++] = p[0] - g_anchor_xy[0];
    frame[o++] = p[1] - g_anchor_xy[1];
    frame[o++] = p[2];
#else
    frame[o++] = p[0]; frame[o++] = p[1]; frame[o++] = p[2];
#endif
    frame[o++] = q[0]; frame[o++] = q[1]; frame[o++] = q[2]; frame[o++] = q[3];
    frame[o++] = w[0]; frame[o++] = w[1]; frame[o++] = w[2];
    // O_VELXY(2), O_VELZ(1)
    frame[o++] = v[0]; frame[o++] = v[1]; frame[o++] = v[2];
    // O_PREV_ACTION(4) - the POST-EMA applied action (the sim stores exactly this)
    for (int i = 0; i < POLICY_ACT_DIM; i++) {
        frame[o++] = g_applied[i];
    }
    // O_P_ERR(3), O_V_ERR(3), O_ATT_ERR(3), O_W_ERR(3)
    for (int i = 0; i < 3; i++) { frame[o++] = rp[i] - p[i]; }
    for (int i = 0; i < 3; i++) { frame[o++] = rv[i] - v[i]; }
    for (int i = 0; i < 3; i++) { frame[o++] = att_err[i]; }
    for (int i = 0; i < 3; i++) { frame[o++] = rw[i] - w[i]; }

    // -- aux (4): specific force (m/s^2, body) + battery ---------------------------------
    if (!g_vbat_id_valid) {
        g_vbat_id = logGetVarId("pm", "vbat");
        g_vbat_id_valid = (g_vbat_id != 0);
    }
    if (g_vbat_id_valid) {
        g_last_vbat = logGetFloat(g_vbat_id);
    }
    frame[o++] = sensors->acc.x * POLICY_GRAVITY;
    frame[o++] = sensors->acc.y * POLICY_GRAVITY;
    frame[o++] = sensors->acc.z * POLICY_GRAVITY;
    frame[o++] = g_last_vbat / POLICY_VBAT_NOMINAL;

    // diagnostics
    g_last_p_err = sqrtf((rp[0] - p[0]) * (rp[0] - p[0]) + (rp[1] - p[1]) * (rp[1] - p[1])
                         + (rp[2] - p[2]) * (rp[2] - p[2]));
    g_last_tilt_deg = tilt_deg_from_quat(q);
}

// ======================================================================================
// action -> rate setpoint (SIM units -> firmware units)
// ======================================================================================
static void synth_setpoint(const float act_in[POLICY_ACT_DIM])
{
    // EMA exactly as the simulator applies it (quad_flip_env ACTION_EMA_ALPHA), INCLUDING
    // the T1-A per-step slew limit that runs before it. The two halves must stay
    // step-for-step identical: the policy was trained against this control law, so a
    // one-sided change here silently deploys a different one. POLICY_ACTION_MAX_DELTA is
    // emitted by export_policy.py straight from `quad_flip_env.ACTION_MAX_DELTA`.
    for (int i = 0; i < POLICY_ACT_DIM; i++) {
        float a = act_in[i];
        float max_delta = POLICY_ACTION_MAX_DELTA;
        if (g_flip_phase != FLIP_PHASE_IDLE) {
            max_delta = 1.0f; // Full agility during aerobatic flip maneuver
        }
        if (max_delta > 0.0f) {
            const float slew = a - g_applied[i];
            if (slew > max_delta) {
                a = g_applied[i] + max_delta;
            } else if (slew < -max_delta) {
                a = g_applied[i] - max_delta;
            }
        }
        g_applied[i] = POLICY_ACTION_EMA_ALPHA * a
                     + (1.0f - POLICY_ACTION_EMA_ALPHA) * g_applied[i];
        g_last_act[i] = act_in[i];      // the POLICY's raw output, for the log
    }

    // thrust: action [0] -> legacy thrust units (0..60000), scaled by policy.thrust_scale
    float units = (0.5f * g_applied[0] + 0.5f) * g_thrust_scale;
    // Airmode / idle floor: while the policy is armed and flying, keep motors spinning at
    // minimum idle PWM (2500 units ~ 4%) so differential torque for roll/pitch/yaw rate PID
    // has immediate authority and motors do not stall or drop to zero.
    if (units < 2500.0f) { units = 2500.0f; }
    if (units > POLICY_THRUST_MAX_UNITS) { units = POLICY_THRUST_MAX_UNITS; }
    g_last_thrust_units = units;

    // rates: action [1..3] -> deg/s in the setpoint's legacy conventions
    g_rate_sp.attitudeRate.roll  = SIGN_RATE_ROLL  * degrees(g_applied[1] * POLICY_RATE_SCALE_RP);
    g_rate_sp.attitudeRate.pitch = SIGN_RATE_PITCH * degrees(g_applied[2] * POLICY_RATE_SCALE_PITCH);
    g_rate_sp.attitudeRate.yaw   = SIGN_RATE_YAW   * degrees(g_applied[3] * POLICY_RATE_SCALE_YAW);

    // Rate-mode setpoint, exactly the shape the stock CRTP rate input produces:
    // roll/pitch/yaw follow attitudeRate, z is disabled so `thrust` is used verbatim.
    g_rate_sp.mode.roll = modeVelocity;
    g_rate_sp.mode.pitch = modeVelocity;
    g_rate_sp.mode.yaw = modeVelocity;
    g_rate_sp.mode.x = modeDisable;
    g_rate_sp.mode.y = modeDisable;
    g_rate_sp.mode.z = modeDisable;
    g_rate_sp.mode.quat = modeDisable;
    g_rate_sp.attitude.roll = 0.0f;
    g_rate_sp.attitude.pitch = 0.0f;
    g_rate_sp.thrust = units;
}

// ======================================================================================
// arming
// ======================================================================================
static void policy_arm(const state_t *state, const setpoint_t *setpoint)
{
    const float p[3] = {state->position.x, state->position.y, state->position.z};
    const float yaw = yaw_from_state(state);

    policy_reset(&g_policy);              // a handover is a new flight to the GRU
    ref_hold_sync(&g_ref, p, yaw);
    // New episode -> new frame origin. Set BEFORE the first build_frame() of the episode,
    // so the first onboard frame reads (0, 0) in x,y exactly as the first training frame
    // of an episode does.
    g_anchor_xy[0] = p[0];
    g_anchor_xy[1] = p[1];

    // Seamless handoff: inherit the pilot's manual hover thrust from setpoint->thrust
    // so there is no sudden 35% throttle drop cliff.
    // units = (0.5 * a0 + 0.5) * g_thrust_scale -> a0 = 2.0 * (units / g_thrust_scale) - 1.0.
    float pilot_thrust = (float)setpoint->thrust;
    if (pilot_thrust < 0.0f) { pilot_thrust = 0.0f; }
    if (pilot_thrust > POLICY_THRUST_MAX_UNITS) { pilot_thrust = POLICY_THRUST_MAX_UNITS; }

    float a0 = POLICY_SIM_HOVER_TRIM_A0;
    if (g_thrust_scale > 0.0f && pilot_thrust > 1000.0f) {
        a0 = 2.0f * (pilot_thrust / g_thrust_scale) - 1.0f;
        if (a0 < -0.5f) { a0 = -0.5f; }
        if (a0 > 0.85f) { a0 = 0.85f; }
    }

    memset(g_applied, 0, sizeof(g_applied));
    g_applied[0] = a0;
    g_last_act[0] = a0;
    g_last_act[1] = 0.0f;
    g_last_act[2] = 0.0f;
    g_last_act[3] = 0.0f;

    // Immediately prime g_rate_sp so any tick before the first 100 Hz decimation
    // already has a valid hover setpoint rather than stale/uninitialized data.
    g_rate_sp.mode.roll = modeVelocity;
    g_rate_sp.mode.pitch = modeVelocity;
    g_rate_sp.mode.yaw = modeVelocity;
    g_rate_sp.mode.x = modeDisable;
    g_rate_sp.mode.y = modeDisable;
    g_rate_sp.mode.z = modeDisable;
    g_rate_sp.mode.quat = modeDisable;
    g_rate_sp.attitude.roll = 0.0f;
    g_rate_sp.attitude.pitch = 0.0f;
    g_rate_sp.attitudeRate.roll = 0.0f;
    g_rate_sp.attitudeRate.pitch = 0.0f;
    g_rate_sp.attitudeRate.yaw = 0.0f;
    g_rate_sp.thrust = (0.5f * a0 + 0.5f) * g_thrust_scale;
    g_last_thrust_units = g_rate_sp.thrust;

    // Reset inner-loop PID controllers to clear any manual flight integrator trim
    attitudeControllerResetAllPID(state->attitude.roll, state->attitude.pitch, state->attitude.yaw);

    g_abort_reason = ABORT_NONE;
    g_play_kind = REF_KIND_NONE;
    g_flip_phase = FLIP_PHASE_IDLE;
    g_flip_phase_log = 0;
    g_flip_inverted = false;
    g_flip_gyro_angle = 0.0f;
    g_flip_is_analytic = false;
    // ARM is this controller's episode start. The plausibility gate adopts the pose we are
    // arming from (arming from a lie would make the gate un-usable on a fresh boot) and
    // re-seats its range anchor, and any previous failsafe is cancelled - the operator has
    // explicitly asked for control.
    const float v_arm[3] = {state->velocity.x, state->velocity.y, state->velocity.z};
    guard_start(&g_guard, p, v_arm);
    g_failsafe.active = false;
    g_failsafe_active = 0;
    g_est_hold = 0;
    g_low_z_ticks = 0;                 // a fresh episode starts with a clean ground guard
    // The failsafe's ramp floor is hover, anchored to actual hover thrust
    g_failsafe.cfg.hover_units = g_rate_sp.thrust;
    g_armed = 1;
    g_arm_param = 1;
    DEBUG_PRINT("ARMED at (%.2f %.2f %.2f) yaw %.0f deg (a0=%.3f, thrust=%.0f)%s\n",
                (double)p[0], (double)p[1], (double)p[2], (double)degrees(yaw),
                (double)a0, (double)g_rate_sp.thrust, g_shadow ? " [SHADOW]" : "");
}

static void policy_disarm(const char *why)
{
    if (g_armed) {
        DEBUG_PRINT("DISARM (%s)\n", why);
    }
    g_armed = 0;
    g_arm_param = 0;      // keep the write-through param consistent, nothing may re-arm
    g_play_kind = REF_KIND_NONE;
    g_flip_phase = FLIP_PHASE_IDLE;
    g_flip_phase_log = 0;
    g_flip_inverted = false;
    g_flip_gyro_angle = 0.0f;
    g_flip_is_analytic = false;
    g_low_z_ticks = 0;
    g_failsafe.active = false;
    g_failsafe_active = 0;
}

// ======================================================================================
// T0-A: the failsafe descent
//
// Replaces "disarm on breach".  Disarmed, this controller hands the stock PID the PILOT's
// setpoint - and there is no firmware deadman, so a mid-air abort means attitude-only
// control on a possibly stale thrust command.  That is the measured crash chain: the policy
// believed it was at 8 m, commanded zero thrust, and the vehicle dropped.  The failsafe
// keeps OWNING the setpoint instead and flies a bounded, self-levelling descent.
//
// The profile (level attitude, zero yaw rate, monotone thrust ramp, bounded) lives in
// `policy_guard.c`, where `guard_host_check.py` tests it.  This function is only the
// mapping to firmware types, and it runs at 1 kHz so the ramp is smooth.
// ======================================================================================
static void policy_failsafe_enter(const char *why)
{
    failsafe_enter(&g_failsafe, g_last_thrust_units);
    g_failsafe_active = 1;
    g_failsafe_count++;
    DEBUG_PRINT("FAILSAFE (%s): level descent from z=%.2f m, thrust %.0f -> 0 over %.1f s\n",
                why, (double)g_last_state_z, (double)g_last_thrust_units,
                (double)POLICY_FS_RAMP_S);
}

static void failsafe_step(void)
{
    // The altitude handed to the disarm test is the GATED one: a lying estimate must not be
    // able to keep the failsafe airborne, nor end it early.
    const failsafe_cmd_t c = failsafe_update(&g_failsafe, 1.0f / POLICY_TICK_HZ,
                                            g_guard.p[2]);
    failsafe_setpoint_t sp;
    failsafe_setpoint(&c, &sp);

    // The casts are safe because the _Static_asserts at the top of this file pin
    // policy_guard's mirror of `stab_mode_t` to the firmware's own enumerators.
    g_rate_sp.mode.roll = (stab_mode_t)sp.mode_roll;
    g_rate_sp.mode.pitch = (stab_mode_t)sp.mode_pitch;
    g_rate_sp.mode.yaw = (stab_mode_t)sp.mode_yaw;
    g_rate_sp.mode.x = (stab_mode_t)sp.mode_x;
    g_rate_sp.mode.y = (stab_mode_t)sp.mode_y;
    g_rate_sp.mode.z = (stab_mode_t)sp.mode_z;
    g_rate_sp.mode.quat = modeDisable;
    g_rate_sp.attitude.roll = sp.roll_deg;
    g_rate_sp.attitude.pitch = sp.pitch_deg;
    g_rate_sp.attitudeRate.yaw = sp.yaw_rate_dps;
    g_rate_sp.thrust = sp.thrust_units;
    g_fs_thrust_units = sp.thrust_units;
    g_last_thrust_units = sp.thrust_units;

    if (c.done) {
        g_failsafe_active = 0;
        policy_disarm("failsafe complete");
    }
}

static void policy_safety_check(const state_t *state, const sensorData_t *sensors)
{
    if (!g_armed || g_failsafe.active) {
        return;      // one latched reason per episode; the failsafe owns the setpoint now
    }
    if (!isfinite(state->position.x) || !isfinite(state->position.y)
        || !isfinite(state->position.z) || !isfinite(sensors->gyro.x)) {
        g_abort_reason = ABORT_ESTIMATE;
        policy_failsafe_enter("non-finite state");
        return;
    }
    // tilt straight from the quaternion: this check runs BEFORE the policy step, so it
    // must not depend on a value build_frame() produces later in the same tick.
    const float q[4] = {state->attitudeQuaternion.w, state->attitudeQuaternion.x,
                        state->attitudeQuaternion.y, state->attitudeQuaternion.z};
    g_last_tilt_deg = tilt_deg_from_quat(q);
    // THE FLIP IS EXEMPT FROM THE TILT ABORT FOR THE WHOLE MANOEUVRE (2026-09-25). A commanded
    // flip sweeps tilt 0 -> 360 deg BY DESIGN and the airframe legitimately reads ~146 deg in
    // the middle of its coast. The old exemption was phase-scoped - FLIP_PHASE_ROTATION was
    // exempt but RECOVERY got a 75 deg limit - and RECOVERY opens at the reference's own ARREST
    // row, which lands while the vehicle can still be inverted. So the safety net fired on a
    // normal flip. MEASURED, log 18:42: the reference reached row 112 at t=8.001 with tilt 146,
    // ABORT_TILT fired, the failsafe levelled the airframe and ramped thrust to 0, and the last
    // 1.5 m was a drop - the arrest rows never got to run (the vehicle had already come back to
    // 72 deg on its own by t=8.129).
    // Tilt cannot discriminate "on a commanded flip" from "lost", so during a flip it is not a
    // fault signal. This is BOUNDED: the manoeuvre is <=2.2 s and the normal limit is back the
    // moment the table ends and the reference hands back to the hold - which is also the moment
    // "did it finish level?" gets asked. The estimator gate, the z envelope below and the
    // failsafe all still apply.
    const bool is_flip = (g_ref.mode == REF_MODE_MANOEUVRE && g_ref.kind == REF_KIND_FLIP);
    const bool is_flip_rec = is_flip && (g_flip_phase == FLIP_PHASE_RECOVERY);
    if (!is_flip && g_last_tilt_deg > g_max_tilt_deg) {
        g_abort_reason = ABORT_TILT;
        policy_failsafe_enter("tilt limit");
        return;
    }
    // T0-B: this guard acts on an ESTIMATOR output.  While the plausibility gate is holding
    // a REJECTED sample it is SUSPENDED, because a lie must not be able to abandon a healthy
    // flight (measured: a run ended with all four motors at 0 and z reading 8.35 m).
    // A PLAUSIBLE z above the ceiling still trips it, and that is the point of the split:
    // 4.0 m is inside the 5.0 m plausibility band, so it reaches this test.
    //
    // *** THE FLOOR IS ALWAYS ARMED (2026-09-25). *** It used to be suspended in
    // FLIP_PHASE_RECOVERY together with the ceiling - which, combined with the flip's
    // whole-manoeuvre tilt exemption, left a flip's recovery with NO guard at all. MEASURED,
    // log 18:48: the flip fell from z 1.56 to the ground at tilt 98-109 and the log ends with
    // `armed` still 1 and the motors at 52-65k for the rest of the run - nothing ever disarmed,
    // so nothing stopped the props (the host's motors-off latch cannot help; it only fires on a
    // disarm the app did not ask for, and there was no disarm).
    // The floor is DEBOUNCED and the ceiling is not: a post-blackout jump UP is exactly the
    // plausible lie T0-B must not abort on, so the ceiling stays suspended in RECOVERY, while a
    // sustained reading on the floor is what a real ground contact looks like.
    if (state->position.z < g_min_z) {
        if (g_low_z_ticks < 0xFFFFFFFFu) { g_low_z_ticks++; }
    } else {
        g_low_z_ticks = 0;
    }
    const bool z_too_low = (g_low_z_ticks >= POLICY_GROUND_HOLD_TICKS);
    // During an active flip maneuver, allow dynamic climb headroom up to g_max_z + 0.35 m
    // so the ballistic pop does not prematurely trip the failsafe ceiling abort mid-rotation.
    const float current_max_z = is_flip ? (g_max_z + 0.35f) : g_max_z;
    const bool z_too_high = (!is_flip_rec && state->position.z > current_max_z);
    // THE FLOOR IS NOT GATED ON `g_guard.hold` (2026-09-25). It used to be - and that is a hole,
    // not a protection: `hold` suspends the check precisely when the estimate is wild, which is
    // exactly the state a crash produces, and `state->position.z` here is the RAW estimator
    // output (not the held sample the policy is fed), so the check was being skipped while the
    // raw z sat metres below the floor. The debounce above is what protects against a
    // transient lie; a sustained reading on the floor is a ground contact. The CEILING keeps the
    // `hold` gate because it has no debounce and a post-blackout jump UP is the plausible lie
    // T0-B exists to not abort on.
    if (z_too_low || (!g_guard.hold && z_too_high)) {
        g_abort_reason = ABORT_Z;
        policy_failsafe_enter("altitude limits");
        return;
    }
}

// ======================================================================================
// the out-of-tree controller (1 kHz, inside the stabilizer task)
// ======================================================================================
void controllerOutOfTreeInit(void)
{
    controllerPidInit();                  // the stock inner loop does the flying
    memset(&g_policy, 0, sizeof(g_policy));
    memset(&g_rate_sp, 0, sizeof(g_rate_sp));

    // T0-A / T0-B configuration.  Every threshold here is a measured one (see
    // policy_guard.h); they are set in one place so there is no second copy to go stale.
    guard_cfg_t gc;
    gc.max_fix_jump = POLICY_MAX_FIX_JUMP_M;
    gc.max_fix_dv = POLICY_MAX_FIX_DV_MS;
    gc.max_fix_range = POLICY_MAX_FIX_RANGE_M;
    gc.max_reject_streak = POLICY_MAX_REJECT_STREAK;
    guard_init(&g_guard, &gc);

    failsafe_cfg_t fc;
    fc.ramp_s = POLICY_FS_RAMP_S;
    fc.min_thrust_frac = 0.0f;
    fc.hover_units = (0.5f * POLICY_SIM_HOVER_TRIM_A0 + 0.5f) * g_thrust_scale;
    fc.max_thrust_units = POLICY_THRUST_MAX_UNITS;
    fc.disarm_z = POLICY_MIN_Z;
    fc.timeout_s = POLICY_FS_TIMEOUT_S;
    failsafe_init(&g_failsafe, &fc);

    DEBUG_PRINT("policy controller ready (shadow=%u thrust_scale=%.0f)\n",
                (unsigned)g_shadow, (double)g_thrust_scale);
}

bool controllerOutOfTreeTest(void)
{
    return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint,
                         const sensorData_t *sensors, const state_t *state,
                         const stabilizerStep_t stabilizerStep)
{
    g_tick++;
    g_last_state_z = state->position.z;

    // 1 kHz gyro accumulation for adaptive phase-slaved flip rotation
    if (g_armed && g_ref.mode == REF_MODE_MANOEUVRE && g_ref.kind == REF_KIND_FLIP
        && g_flip_phase == FLIP_PHASE_ROTATION) {
        const float rot_rate_dps = (g_tune_flip_axis == 1) ? fabsf(sensors->gyro.x) : fabsf(sensors->gyro.y);
        g_flip_gyro_angle += rot_rate_dps * (1.0f / POLICY_TICK_HZ);
    }

    // write-through param mirror (allows arming from the client's parameter tab)
    if (g_arm_param != g_armed) {
        if (g_arm_param) {
            policy_arm(state, setpoint);
        } else {
            policy_disarm("client parameter");
        }
    }

    const bool do_policy = (g_tick % POLICY_DECIMATION) == 0;

    if (do_policy) {
        g_step100++;
        // T0-B: gate the estimate FIRST, so the safety check below sees the same
        // plausibility verdict the frame will.
        const float p_raw[3] = {state->position.x, state->position.y, state->position.z};
        const float v_raw[3] = {state->velocity.x, state->velocity.y, state->velocity.z};
        const guard_out_t g_out = guard_update(&g_guard, p_raw, v_raw);
        g_est_hold = g_guard.hold ? 1u : 0u;
        if (g_out.escalate && !g_failsafe.active) {
            g_abort_reason = ABORT_ESTIMATE;
            policy_failsafe_enter("estimator implausible");
        }
        policy_safety_check(state, sensors);

        // The policy step is SKIPPED while the failsafe owns the setpoint: its output would
        // be discarded, and leaving it running would keep advancing the GRU on a frame
        // nobody acts on.
        if (g_armed && !g_failsafe.active) {
            // manoeuvre launch / completion (100 Hz domain)
            const float p[3] = {state->position.x, state->position.y, state->position.z};
            const float yaw = yaw_from_state(state);
            if (g_play_kind != REF_KIND_NONE) {
                const uint8_t kind = g_play_kind;
                g_play_kind = REF_KIND_NONE;
                if (kind >= REF_TABLE_COUNT) {
                    ref_play_launch(&g_ref, kind, p, yaw, g_step100);   // -> hold, at the
                    g_flip_is_analytic = false;
                    g_flip_phase = FLIP_PHASE_IDLE;
                    g_flip_phase_log = 0;
                    g_flip_inverted = false;
                    DEBUG_PRINT("stop and hold\n");                       // current pose
                } else {
                    ref_play_launch(&g_ref, kind, p, yaw, g_step100);
                    if (kind == REF_KIND_FLIP) {
                        g_flip_phase = FLIP_PHASE_CLIMB;
                        g_flip_phase_log = 1;
                        g_flip_inverted = false;
                        g_flip_gyro_angle = 0.0f;
                        g_flip_z0 = g_guard.p[2];
                        g_flip_climb_start_step = g_step100;
                        if (g_tune_flip_analytic) {
                            g_flip_is_analytic = true;
                            analytic_flip_params_t p_cfg;
                            analytic_flip_params_default(&p_cfg, POLICY_SIM_MASS_KG,
                                                         g_tune_flip_max_thrust,
                                                         POLICY_SIM_GRAVITY);
                            p_cfg.peak_dps = g_tune_flip_peak_dps;
                            p_cfg.pop_pct = g_tune_flip_pop_pct;
                            p_cfg.rate_frac = g_tune_flip_rate_frac;
                            p_cfg.axis = g_tune_flip_axis;
                            analytic_flip_init(&g_analytic_flip, &p_cfg, p, yaw, g_step100);
                            g_flip_active_rot_start = g_analytic_flip.rot_start_row;
                            g_flip_active_catch_start = g_analytic_flip.catch_start_row;
                            DEBUG_PRINT("ANALYTIC FLIP launched: peak=%.1f dps, pop=%.2f, frac=%.2f, axis=%u (rot=%u catch=%u dur=%u)\n",
                                        (double)g_tune_flip_peak_dps, (double)g_tune_flip_pop_pct,
                                        (double)g_tune_flip_rate_frac, (unsigned)g_tune_flip_axis,
                                        (unsigned)g_flip_active_rot_start, (unsigned)g_flip_active_catch_start,
                                        (unsigned)g_analytic_flip.total_rows);
                        } else {
                            g_flip_is_analytic = false;
                            g_flip_active_rot_start = REF_FLIP_ROT_START_ROW;
                            g_flip_active_catch_start = REF_FLIP_CATCH_START_ROW;
                            DEBUG_PRINT("PREBAKED FLIP launched (rot=%u catch=%u)\n",
                                        (unsigned)g_flip_active_rot_start, (unsigned)g_flip_active_catch_start);
                        }
                        DEBUG_PRINT("FLIP Phase 1 (Adaptive Climb) start at z=%.2f m\n", (double)g_flip_z0);
                    } else {
                        g_flip_is_analytic = false;
                        g_flip_phase = FLIP_PHASE_IDLE;
                        g_flip_phase_log = 0;
                        g_flip_inverted = false;
                        g_flip_gyro_angle = 0.0f;
                    }
                    DEBUG_PRINT("%s launched (%u steps, %.2f s)\n", ref_kind_name(kind),
                                 (unsigned)ref_kind_len(kind),
                                 (double)((float)ref_kind_len(kind) * REF_DT));
                }
            } else if (g_ref.mode == REF_MODE_MANOEUVRE && g_ref.kind == REF_KIND_FLIP) {
                if (g_flip_phase == FLIP_PHASE_CLIMB) {
                    const uint32_t climb_ticks = g_step100 - g_flip_climb_start_step;
                    const float dz = g_guard.p[2] - g_flip_z0;
                    const float vz = g_guard.v[2];

                    // Phase 2 Entry Trigger: the vehicle has climbed (Δz ≥ 0.35 m or
                    // vz ≥ 1.60 m/s, min 15 ticks ~ 150 ms for spool-up) OR the REFERENCE has
                    // reached its own coast row. BOTH are wanted: the measured arm fires EARLY
                    // (~row 25) and is what starts the 1 kHz gyro accounting the level-catch
                    // needs, while the row arm guarantees the phase still advances when the
                    // vehicle cannot climb (flat pack / thrust-limited) - the phase no longer
                    // moves the reference clock, so nothing else would move it.
                    const uint32_t ref_row = g_step100 - g_ref.start_step;
                    const bool trigger = ((climb_ticks >= 15u) && ((dz >= 0.35f) || (vz >= 1.60f)))
                                      || (ref_row >= g_flip_active_rot_start);
                    // Safety timeout: abandon the flip if the REFERENCE has passed its own
                    // rotation row and the phase STILL has not advanced.
                    //
                    // *** IT MUST BE SLAVED TO THE REFERENCE CLOCK, NOT THE WALL CLOCK (2026-09-29). ***
                    // The old form was `climb_ticks >= 85u`, which silently rejected every
                    // weak-pop preset: `rot_start_row` is `c0/DT` with `c0 = v0/u` and
                    // `v0 = 0.5*g*coast`, so a gentler pop makes the climb LONGER. MEASURED:
                    // 750 dps / 0.85 pop gives rot_start_row = 165 (1.65 s), while the row arm
                    // that was supposed to rescue it (`ref_row >= g_flip_active_rot_start`)
                    // arrives at row 165 - far too late to beat an 85-tick deadline. The guard
                    // killed exactly the launches it was meant to protect. With the baked flip
                    // ROT_START is 57 ticks so this never showed. The +20 rows keep a real guard:
                    // if the reference is past its rotation row and the state machine has not
                    // moved, something is wrong and holding is the safe answer.
                    const bool timeout = (ref_row >= g_flip_active_rot_start + 20u);

                    if (trigger) {
                        g_flip_phase = FLIP_PHASE_ROTATION;
                        g_flip_phase_log = 2;
                        g_flip_inverted = false;
                        g_flip_gyro_angle = 0.0f;
                        g_flip_rot_start_step = g_step100;
                        // NO CLOCK RE-BASE. The reference keeps its own timeline and reaches its
                        // own coast row at row g_flip_active_rot_start; this transition only opens
                        // the tilt exemption, so the measured dz/vz trigger is deliberately EARLY
                        // (it fires around row 25, well before the body starts inverting).
                        DEBUG_PRINT("FLIP Phase 2 (Rotation) start: dz=%.2f m, vz=%.2f m/s at %u ms "
                                    "(reference row %u of %u)\n",
                                    (double)dz, (double)vz, (unsigned)(climb_ticks * 10u),
                                    (unsigned)(g_step100 - g_ref.start_step),
                                    (unsigned)g_flip_active_rot_start);
                    } else if (timeout) {
                        DEBUG_PRINT("FLIP climb timeout (dz=%.2f m, vz=%.2f m/s): aborting to hold\n",
                                    (double)dz, (double)vz);
                        ref_play_to_hold(&g_ref, p, yaw);
                        g_flip_phase = FLIP_PHASE_IDLE;
                        g_flip_phase_log = 0;
                    }
                } else if (g_flip_phase == FLIP_PHASE_ROTATION) {
                    const uint32_t rot_ticks = g_step100 - g_flip_rot_start_step;

                    // Inverted latch: vehicle reached upside-down
                    if (g_last_tilt_deg > 120.0f || g_flip_gyro_angle > 140.0f) {
                        g_flip_inverted = true;
                    }

                    // Level Catch Trigger: drone reached inversion, completed ~360 deg,
                    // is near level, AND has actually decelerated (gyro rate < 300 deg/s).
                    // Without the rate check, the trigger fires while the body is merely
                    // PASSING THROUGH level at 1000+ deg/s, causing a second flip.
                    // 300 deg/s is well within the rate PID's arrest capability.
                    const float rot_rate_dps = (g_tune_flip_axis == 1) ? fabsf(sensors->gyro.x) : fabsf(sensors->gyro.y);
                    const bool level_exit = g_flip_inverted
                                         && g_flip_gyro_angle >= 320.0f
                                         && g_last_tilt_deg <= 35.0f
                                         && rot_rate_dps < 300.0f;
                    // Fallback: the REFERENCE has reached its own arrest rows AND vehicle is righted.
                    // Must NOT fire while inverted (prevents driving upside-down full thrust into floor).
                    const uint32_t ref_row = g_step100 - g_ref.start_step;
                    const bool rot_timeout = (ref_row >= g_flip_active_catch_start)
                                          && (g_last_tilt_deg <= 45.0f);
                    // *** ALSO REFERENCE-RELATIVE (2026-09-29). *** This was `rot_ticks >= 150u`,
                    // counted from the ROTATION ENTRY - and that entry is deliberately EARLY
                    // (the measured dz/vz arm), so the budget was already partly spent before the
                    // body started turning. MEASURED, log 16:18: entry at reference row ~74 for a
                    // flip whose rotation is rows 165..234, so a 150-tick budget expired at row
                    // 224 - 10 rows SHORT of the reference's own catch row. The flip would have
                    // been cut at ~308 deg, which also makes the 320 deg level-exit unreachable.
                    // Slaving it to the catch row keeps exactly the intended meaning: "the
                    // reference has finished rotating and the vehicle is still not righted after
                    // 200 ms - give up and hold".
                    const bool rot_abort = (ref_row >= g_flip_active_catch_start + 20u);

                    if (level_exit || rot_timeout) {
                        g_flip_phase = FLIP_PHASE_RECOVERY;
                        g_flip_phase_log = 3;

                        // Re-anchor recovery target to current level-out position (Option 1)
                        if (g_flip_is_analytic) {
                            analytic_flip_reanchor_z(&g_analytic_flip, g_guard.p[2]);
                        } else {
                            g_ref.p_shift[0] = g_guard.p[0];
                            g_ref.p_shift[1] = g_guard.p[1];
                            g_ref.p_shift[2] = g_guard.p[2] - REF_P0[REF_KIND_FLIP][2];
                        }

                        // Brake rotation immediately: clear spin rate setpoints and rate PID integrators
                        g_applied[1] = 0.0f;
                        g_applied[2] = 0.0f;
                        g_applied[3] = 0.0f;
                        g_rate_sp.attitudeRate.roll = 0.0f;
                        g_rate_sp.attitudeRate.pitch = 0.0f;
                        g_rate_sp.attitudeRate.yaw = 0.0f;
                        attitudeControllerResetAllPID(state->attitude.roll, state->attitude.pitch, state->attitude.yaw);

                        DEBUG_PRINT("FLIP Phase 3 (Catch) triggered: gyro=%.1f deg, tilt=%.1f deg at %u ms (%s), catch z=%.2f m\n",
                                    (double)g_flip_gyro_angle, (double)g_last_tilt_deg, (unsigned)(rot_ticks * 10u),
                                    level_exit ? "level trigger" : "timeout",
                                    (double)g_guard.p[2]);
                    } else if (rot_abort) {
                        DEBUG_PRINT("FLIP rotation timeout (%u ms, tilt=%.1f deg): aborting to hold\n",
                                    (unsigned)(rot_ticks * 10u), (double)g_last_tilt_deg);
                        ref_play_to_hold(&g_ref, p, yaw);
                        g_flip_phase = FLIP_PHASE_IDLE;
                        g_flip_phase_log = 0;
                    }
                } else if (g_flip_phase == FLIP_PHASE_RECOVERY) {
                    const bool finished = g_flip_is_analytic
                                        ? analytic_flip_finished(&g_analytic_flip, g_step100)
                                        : ref_play_finished(&g_ref, g_step100);
                    if (finished) {
                        ref_play_to_hold(&g_ref, p, yaw);
                        g_flip_phase = FLIP_PHASE_IDLE;
                        g_flip_phase_log = 0;
                        DEBUG_PRINT("Flip complete - holding\n");
                    }
                }
            } else if (g_ref.mode == REF_MODE_MANOEUVRE && ref_play_finished(&g_ref, g_step100)) {
                ref_play_to_hold(&g_ref, p, yaw);
                DEBUG_PRINT("%s complete - holding\n", ref_kind_name(g_ref.kind));
            }

            float frame[POLICY_ENC_IN_DIM];
            float ref_ff[POLICY_REF_FF_DIM];
            float act[POLICY_ACT_DIM];
            build_frame(sensors, state, frame, ref_ff);
            policy_step(&g_policy, frame, ref_ff, act);
            synth_setpoint(act);
        }
        g_mode = (uint8_t)g_ref.mode;
    }

    // T0-A at 1 kHz: the failsafe rebuilds the setpoint EVERY tick so the thrust ramp is
    // smooth - at the 100 Hz policy cadence it would step.  `g_armed` is still 1 here (the
    // failsafe itself disarms when it completes), so the branch below keeps using our
    // setpoint rather than handing the flight back mid-descent.
    if (g_failsafe.active && !g_shadow) {
        failsafe_step();
    }

    control->controlMode = controlModeLegacy;   // controllerPid() overwrites this; kept so
                                                // the struct is never left unlabelled
    if (g_armed && !g_shadow) {
        controllerPid(control, &g_rate_sp, sensors, state, stabilizerStep);
    } else {
        // Disarmed (or shadow): the stock behaviour, on the pilot's own setpoint.
        controllerPid(control, setpoint, sensors, state, stabilizerStep);
    }
}

// ======================================================================================
// ground side: commands in, status out
// ======================================================================================
static void send_status(void)
{
    struct {
        uint8_t magic;
        uint8_t armed;
        uint8_t mode;
        uint8_t shadow;
        uint8_t abort_reason;
        uint8_t manoeuvre;          // REF_KIND_* while playing, 0xFF = none (was padding)
        float z;
        float tilt_deg;
        float thrust_units;
        float p_err;
        float vbat;
        float act0;
    } __attribute__((packed)) msg;

    msg.magic = STATUS_MAGIC;
    msg.armed = g_armed;
    msg.mode = g_mode;
    msg.shadow = g_shadow;
    msg.abort_reason = g_abort_reason;
    msg.manoeuvre = g_ref.kind;
    msg.z = g_last_state_z;
    msg.tilt_deg = g_last_tilt_deg;
    msg.thrust_units = g_last_thrust_units;
    msg.p_err = g_last_p_err;
    msg.vbat = g_last_vbat;
    msg.act0 = g_last_act[0];
    appchannelSendDataPacketBlock(&msg, sizeof(msg));
}

void appMain(void)
{
    DEBUG_PRINT("policy app running - appchannel: 0x01 arm, 0x02 disarm, 0x03 flip, "
                "0x04 status, 0x05 <kind> manoeuvre (0x%02x = hold, %u kinds baked)\n",
                REF_KIND_NONE, (unsigned)ref_kind_count());

    uint8_t msg[2] = {0, 0};
    uint32_t status_div = 0;

    while (1) {
        const size_t n = appchannelReceiveDataPacket(msg, sizeof(msg), 100);
        if (n > 0) {
            switch (msg[0]) {
                case CMD_ARM:
                    // arming is done in the controller tick (needs the estimator state)
                    g_arm_param = 1;
                    break;
                case CMD_DISARM:
                    g_arm_param = 0;
                    break;
                case CMD_FLIP:
                    if (g_armed) {
                        g_play_kind = REF_KIND_FLIP;
                    } else {
                        DEBUG_PRINT("flip ignored: not armed\n");
                    }
                    break;
                case CMD_PLAY:
                    if (!g_armed) {
                        DEBUG_PRINT("manoeuvre ignored: not armed\n");
                    } else if (n < 2) {
                        DEBUG_PRINT("manoeuvre needs a kind byte\n");
                    } else if (msg[1] >= REF_TABLE_COUNT && msg[1] != REF_KIND_NONE) {
                        DEBUG_PRINT("no manoeuvre %u (0..%u, or 0x%02x to hold)\n",
                                    (unsigned)msg[1], (unsigned)(REF_TABLE_COUNT - 1u),
                                    REF_KIND_NONE);
                    } else {
                        g_play_kind = msg[1];
                    }
                    break;
                case CMD_STATUS:
                    send_status();
                    break;
                default:
                    DEBUG_PRINT("unknown command 0x%02x\n", msg[0]);
                    break;
            }
        }
        if (++status_div >= 2) {   // ~every 200 ms (the receive timeout paces this loop)
            status_div = 0;
            send_status();
        }
    }
}

// ======================================================================================
// params / logs
// ======================================================================================
PARAM_GROUP_START(policy)
/** >0 places the controller in shadow mode: the pipeline runs and logs but never commands. */
PARAM_ADD(PARAM_UINT8, shadow, &g_shadow)
/** arm (1) / disarm (0): same as the appchannel arm/disarm commands. */
PARAM_ADD(PARAM_UINT8, arm, &g_arm_param)
/** action +1 maps to this many legacy thrust units (60000 = firmware max setpoint). */
PARAM_ADD(PARAM_FLOAT, thrust_scale, &g_thrust_scale)
/** abort if the tilt exceeds this. */
PARAM_ADD(PARAM_FLOAT, max_tilt_deg, &g_max_tilt_deg)
/** T0-B: the altitude envelope, now runtime-tunable. Tune it by reading the real envelope
 *  off the log rather than by editing a #define and reflashing. */
PARAM_ADD(PARAM_FLOAT, min_z, &g_min_z)
PARAM_ADD(PARAM_FLOAT, max_z, &g_max_z)
PARAM_GROUP_STOP(policy)

PARAM_GROUP_START(traj_flip)
/** Target peak body rate in deg/s (e.g. 720 deg/s) */
PARAM_ADD(PARAM_FLOAT, peak_dps, &g_tune_flip_peak_dps)
/** Pop climb/arrest thrust fraction of max thrust (0.70..0.95) */
PARAM_ADD(PARAM_FLOAT, pop_pct, &g_tune_flip_pop_pct)
/** Trapezoidal rate ramp fraction of coast (0.15..0.45) */
PARAM_ADD(PARAM_FLOAT, rate_frac, &g_tune_flip_rate_frac)
/** Full-throttle thrust (N) the reference is planned for. THE altitude-excursion lever:
 *  raising it shortens the climb (c0 = v0/u) and shrinks the throw. 0.60 = the sim plant
 *  constant the policy trained against; the real vehicle measures ~0.96 N. */
PARAM_ADD(PARAM_FLOAT, max_thrust, &g_tune_flip_max_thrust)
/** Flip axis: 0 = pitch (y), 1 = roll (x) */
PARAM_ADD(PARAM_UINT8, axis, &g_tune_flip_axis)
/** Mode: 1 = parametric analytic trajectory, 0 = prebaked table */
PARAM_ADD(PARAM_UINT8, analytic, &g_tune_flip_analytic)
PARAM_GROUP_STOP(traj_flip)

LOG_GROUP_START(policy)
LOG_ADD(LOG_UINT8, armed, &g_armed)
LOG_ADD(LOG_UINT8, mode, &g_mode)
LOG_ADD(LOG_UINT8, flip_phase, &g_flip_phase_log)
LOG_ADD(LOG_UINT8, abort_reason, &g_abort_reason)
/** T0-A: 1 while the failsafe owns the setpoint (it descends level, then disarms). */
LOG_ADD(LOG_UINT8, failsafe, &g_failsafe_active)
/** T0-B: 1 while the plausibility gate is holding a rejected estimate. The altitude guard
 *  is suspended while this reads 1, so a case that trips it should be read together with
 *  `guard_z` (the HELD altitude) rather than the estimator's. */
LOG_ADD(LOG_UINT8, est_hold, &g_est_hold)
LOG_ADD(LOG_FLOAT, fs_thrust, &g_fs_thrust_units)
LOG_ADD(LOG_FLOAT, guard_z, &g_guard.p[2])
LOG_ADD(LOG_UINT32, reject_total, &g_guard.reject_total)
LOG_ADD(LOG_FLOAT, thrust_units, &g_last_thrust_units)
LOG_ADD(LOG_FLOAT, tilt_deg, &g_last_tilt_deg)
LOG_ADD(LOG_FLOAT, p_err, &g_last_p_err)
LOG_ADD(LOG_FLOAT, ff_z, &g_last_ff_z)
LOG_ADD(LOG_FLOAT, vbat, &g_last_vbat)
LOG_ADD(LOG_FLOAT, act0, &g_last_act[0])
LOG_ADD(LOG_FLOAT, act1, &g_last_act[1])
LOG_ADD(LOG_FLOAT, act2, &g_last_act[2])
LOG_ADD(LOG_FLOAT, act3, &g_last_act[3])
LOG_GROUP_STOP(policy)

LOG_GROUP_START(traj_flip)
LOG_ADD(LOG_FLOAT, gyro_deg, &g_flip_gyro_angle)
LOG_ADD(LOG_UINT32, rot_start, &g_flip_active_rot_start)
LOG_ADD(LOG_UINT32, catch_start, &g_flip_active_catch_start)
LOG_ADD(LOG_FLOAT, z_shift, &g_analytic_flip.z_shift)
LOG_GROUP_STOP(traj_flip)
