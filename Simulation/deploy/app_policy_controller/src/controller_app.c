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
#include "app_channel.h"
#include "log.h"
#include "param.h"
#include "math3d.h"
#include "debug.h"

#include "FreeRTOS.h"
#include "task.h"

#include "policy_backend.h"            // -> policy_net.h or policy_net_stedgeai.h
#include "reference.h"

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
#define POLICY_MAX_Z 3.5f

// Battery: the sim's aux channel is V / V_nom with V_nom = 3.7 V (quad_mujoco.py).
#define POLICY_VBAT_NOMINAL 3.7f
#define POLICY_GRAVITY 9.81f

// The layout the ONBOARD image must reproduce, mirroring quad_flip_env. These are asserted
// rather than assumed: the generated header is the only place they come from, and a sim-side
// change that reaches the exporter but not this file would otherwise be invisible here.
_Static_assert(POLICY_REF_FF_DIM == 3, "quad_flip_env.REF_FF_DIM is 3");
_Static_assert(POLICY_ENC_IN_DIM == POLICY_O_T_DIM + POLICY_ENC_AUX_DIM,
               "the encoder frame is [o_t | aux]");

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
static float g_last_tilt_deg = 0.0f;
static float g_last_p_err = 0.0f;
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
    float angle = acosf(fmaxf(-1.0f, fminf(1.0f, 0.5f * (tr - 1.0f))));
    if (angle < 1e-5f) {
        // small-angle: the vector is the skew-symmetric part
        out[0] = 0.5f * (R[7] - R[5]);
        out[1] = 0.5f * (R[2] - R[6]);
        out[2] = 0.5f * (R[3] - R[1]);
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
    ref_sample(&g_ref, g_step100, rp, rv, rR, rw, ra);

    // The actor's feed-forward block: a_ref + g*e_z, world frame - exactly what the training
    // environment feeds (quad_flip_env._compute_reference_ff). Built HERE, next to the same
    // ref_sample() the error channels come from, so both describe the same instant.
    ref_feed_forward(ra, POLICY_SIM_GRAVITY, ref_ff);
    g_last_ff_z = ref_ff[2];

    const float p[3] = {state->position.x, state->position.y, state->position.z};
    const float v[3] = {state->velocity.x, state->velocity.y, state->velocity.z};

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
static void synth_setpoint(const float act[POLICY_ACT_DIM])
{
    // EMA exactly as the simulator applies it (quad_flip_env ACTION_EMA_ALPHA).
    for (int i = 0; i < POLICY_ACT_DIM; i++) {
        g_applied[i] = POLICY_ACTION_EMA_ALPHA * act[i]
                     + (1.0f - POLICY_ACTION_EMA_ALPHA) * g_applied[i];
        g_last_act[i] = act[i];
    }

    // thrust: action [0] -> legacy thrust units (0..60000), scaled by policy.thrust_scale
    float units = (0.5f * g_applied[0] + 0.5f) * g_thrust_scale;
    if (units < 0.0f) { units = 0.0f; }
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
static void policy_arm(const state_t *state)
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
    // The sim's first frame of an episode carries the hover-trim action as the "prior
    // action" fiction (quad_flip_env.hover_trim_action); start the EMA there too so the
    // first onboard frames describe the same state of the world that training did.
    memset(g_applied, 0, sizeof(g_applied));
    g_applied[0] = POLICY_SIM_HOVER_TRIM_A0;
    g_abort_reason = ABORT_NONE;
    g_play_kind = REF_KIND_NONE;
    g_armed = 1;
    g_arm_param = 1;
    DEBUG_PRINT("ARMED at (%.2f %.2f %.2f) yaw %.0f deg%s\n", (double)p[0], (double)p[1],
                (double)p[2], (double)degrees(yaw), g_shadow ? " [SHADOW]" : "");
}

static void policy_disarm(const char *why)
{
    if (g_armed) {
        DEBUG_PRINT("DISARM (%s)\n", why);
    }
    g_armed = 0;
    g_arm_param = 0;      // keep the write-through param consistent, nothing may re-arm
    g_play_kind = REF_KIND_NONE;
}

static void policy_safety_check(const state_t *state, const sensorData_t *sensors)
{
    if (!g_armed) {
        return;
    }
    if (!isfinite(state->position.x) || !isfinite(state->position.y)
        || !isfinite(state->position.z) || !isfinite(sensors->gyro.x)) {
        g_abort_reason = ABORT_ESTIMATE;
        policy_disarm("non-finite state");
        return;
    }
    // tilt straight from the quaternion: this check runs BEFORE the policy step, so it
    // must not depend on a value build_frame() produces later in the same tick.
    const float q[4] = {state->attitudeQuaternion.w, state->attitudeQuaternion.x,
                        state->attitudeQuaternion.y, state->attitudeQuaternion.z};
    g_last_tilt_deg = tilt_deg_from_quat(q);
    if (g_last_tilt_deg > g_max_tilt_deg) {
        g_abort_reason = ABORT_TILT;
        policy_disarm("tilt limit");
        return;
    }
    if (state->position.z < POLICY_MIN_Z || state->position.z > POLICY_MAX_Z) {
        g_abort_reason = ABORT_Z;
        policy_disarm("altitude limits");
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

    // write-through param mirror (allows arming from the client's parameter tab)
    if (g_arm_param != g_armed) {
        if (g_arm_param) {
            policy_arm(state);
        } else {
            policy_disarm("client parameter");
        }
    }

    const bool do_policy = (g_tick % POLICY_DECIMATION) == 0;

    if (do_policy) {
        g_step100++;
        policy_safety_check(state, sensors);

        if (g_armed) {
            // manoeuvre launch / completion (100 Hz domain)
            const float p[3] = {state->position.x, state->position.y, state->position.z};
            const float yaw = yaw_from_state(state);
            if (g_play_kind != REF_KIND_NONE) {
                const uint8_t kind = g_play_kind;
                g_play_kind = REF_KIND_NONE;
                if (kind >= REF_TABLE_COUNT) {
                    ref_play_launch(&g_ref, kind, p, yaw, g_step100);   // -> hold, at the
                    DEBUG_PRINT("stop and hold\n");                       // current pose
                } else {
                    ref_play_launch(&g_ref, kind, p, yaw, g_step100);
                    DEBUG_PRINT("%s launched (%u steps, %.2f s)\n", ref_kind_name(kind),
                                (unsigned)ref_kind_len(kind),
                                (double)((float)ref_kind_len(kind) * REF_DT));
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
PARAM_GROUP_STOP(policy)

LOG_GROUP_START(policy)
LOG_ADD(LOG_UINT8, armed, &g_armed)
LOG_ADD(LOG_UINT8, mode, &g_mode)
LOG_ADD(LOG_UINT8, abort_reason, &g_abort_reason)
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
