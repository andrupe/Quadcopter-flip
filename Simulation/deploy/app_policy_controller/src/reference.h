// Onboard reference generator: HOLD (a point the policy stabilises to) and one MANOEUVRE
// at a time (playback of a baked 100 Hz table, relocated onto the live pose).
//
// WHY THIS IS SO SMALL
// --------------------
// The policy consumes the reference as ERRORS (p, v, R, omega) plus ONE feed-forward
// quantity: the specific-force command a_ref + g*e_z that quad_flip_env feeds the actor as
// its REF_FF_DIM = 3 block. It still never sees the manoeuvre's thrust directly, and it does
// not need trajectories.py ported: the tables below carry p|v|R|omega AND a, so the app can
// reconstruct the feed-forward without differentiating anything.
//
// The relocation is the SAME rigid motion as live_target.ShiftedTrajectory (verified
// sample-by-sample for every kind in that generator's --check), evaluated once at launch:
//
//     dyaw    = yaw_live - REF_YAW0[kind]
//     Rz      = rotation about world z by dyaw
//     p_shift = p_live - Rz @ REF_P0[kind]
//
// and applied per row as   p = Rz p_t + p_shift,  v = Rz v_t,  R = Rz R_t,  a = Rz a_t,
//                          w = w_t   (NOT rotated - see below).
//
// OMEGA IS NOT ROTATED. A body-frame rate is INVARIANT under a world-frame relocation of
// the whole trajectory: R' = QR gives [w']_x = R'^T R'dot = R^T Q^T Q Rdot = R^T Rdot =
// [w]_x, the same body-frame vector, because the body axes turn WITH the vehicle. Rotating
// it by Rz, by analogy with v and a, describes a DIFFERENT manoeuvre - a 90 deg yaw
// relocation of a pitch flip would claim a roll rate - and it put an error into the
// policy's w_err channel that grew with dyaw, i.e. worst exactly when the pilot happens to
// face the other way. This now mirrors trajectories.ShiftedManeuver, which has always been
// correct, and is verified by `gen_references.py --check` against live_target's
// ShiftedTrajectory.
//
// LAUNCH TRANSIENT
// ----------------
// The flip and the waypoint polynomial start at REST; the periodic families are sampled
// mid-motion, so launching one from a pilot's hover puts their initial velocity and rate
// straight into the policy's errors. The generator therefore selects each draw to MINIMISE
// that transient and checks it against what training itself presents at t = 0 (<= 0.6 m/s
// and <= 0.6 rad/s - the initial-kick range, and a quarter of the rate tolerance). Nothing
// here compensates for it: the policy was trained to converge from exactly that.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "generated/reference_tables.h"

typedef enum {
    REF_MODE_HOLD = 0,
    REF_MODE_MANOEUVRE = 1,
} ref_mode_t;

typedef struct {
    ref_mode_t mode;
    uint8_t kind;           // REF_KIND_* while playing; REF_KIND_NONE in HOLD
    float p0[3];            // HOLD anchor (world)
    float yaw0;             // HOLD heading (rad)
    // manoeuvre transform, computed at launch
    float c;                // cos(dyaw)
    float s;                // sin(dyaw)
    float p_shift[3];
    uint32_t start_step;    // 100 Hz tick at launch
} ref_state_t;

// Anchor the hold reference at the current pose. MUST be called on every arming - a hold
// anchored anywhere else hands the policy a step it did not create.
void ref_hold_sync(ref_state_t *r, const float p[3], float yaw);

// Launch manoeuvre `kind` from the current pose/heading. A kind >= REF_TABLE_COUNT (the
// app passes REF_KIND_NONE for "stop and hold") falls back to ref_hold_sync.
void ref_play_launch(ref_state_t *r, uint8_t kind, const float p[3], float yaw, uint32_t step100);

// Abort/complete a manoeuvre: hold wherever the vehicle is RIGHT NOW. Re-anchoring to
// reality (not to the table) is what keeps the handback step-free even when the manoeuvre
// was cut short.
void ref_play_to_hold(ref_state_t *r, const float p[3], float yaw);

// Sample the reference at a 100 Hz tick. For a manoeuvre, ticks past the table length clamp
// to the final row (which is a hover, so the handback to HOLD is step-free).
//
// `a` is the reference's WORLD-frame acceleration (m/s^2) and is ZERO in HOLD mode, which is
// correct: a hover at rest has no acceleration to feed forward.
void ref_sample(const ref_state_t *r, uint32_t step100,
                float p[3], float v[3], float R[9], float w[3], float a[3]);

// The policy's feed-forward block: out = a_ref + g*e_z, WORLD frame, m/s^2.
//
// This is EXACTLY the block quad_flip_env puts in the actor observation (REF_FF_DIM = 3,
// world frame, `_compute_reference_ff`): its NORM is the collective feed-forward (1 g in a
// hover, 0 in a flip's ballistic coast) and its DIRECTION is the desired thrust direction.
// Kept here, next to the reference maths, so the gravity sign cannot drift away from it.
void ref_feed_forward(const float a_ref[3], float gravity, float out[3]);

// How many manoeuvres are baked in (the host's GUI builds its buttons from this).
uint8_t ref_kind_count(void);

// Name / length of a baked manoeuvre (NULL / 0 for an out-of-range kind).
const char *ref_kind_name(uint8_t kind);
uint32_t ref_kind_len(uint8_t kind);

// True when the launched manoeuvre has played its whole table.
bool ref_play_finished(const ref_state_t *r, uint32_t step100);
