/*
 * policy_guard - the two safety primitives the app needs, and the only ones worth testing
 * WITHOUT a Crazyflie.
 *
 * WHY THIS IS A SEPARATE MODULE
 * -----------------------------
 * `controller_app.c` cannot be host-compiled: it includes app.h / controller.h / log.h /
 * param.h / FreeRTOS and links the stock controller.  The two things added here decide
 * whether a real flight is abandoned, so they must be testable on the bench, not by
 * inspection.  This module is therefore pure C over plain floats with NO firmware headers,
 * and `test/guard_check.c` (+ `Simulation/deploy/guard_host_check.py`) drives it directly.
 * `controller_app.c` keeps only the mechanical mapping to firmware types.
 *
 * 1. ESTIMATOR PLAUSIBILITY GATE (`guard_*`)
 *    The app acts on the state ESTIMATE.  The measured hardware failure was not a bad fix
 *    but a bad ESTIMATE: `stateEstimate.x` ran to +98 m at 5-9 m/s on a STATIC vehicle, and
 *    nothing downstream could tell.  Two measured consequences, both of which this gate
 *    removes:
 *      * the policy was fed the lie and commanded ZERO thrust to "come down" from a phantom
 *        8 m altitude - it could not; it cut thrust and dropped;
 *      * the altitude envelope guard fired on the lie and abandoned a healthy flight.
 *    On a rejection the gate HOLDS the last plausible estimate (so the policy input stops
 *    moving and stays COHERENT - position, velocity, p_err and v_err all come from the same
 *    held sample, which is the same coherence rule `Simulation/encoder/corruption.py`
 *    enforces) and marks it, so the caller can suspend the altitude guard.
 *
 *    THRESHOLDS ARE MEASURED, NOT ESTIMATED.  max_fix_jump 5.0 m and max_fix_dv 4.0 m/s
 *    come from the legitimate post-blackout innovation distribution in `Simulation/
 *    lighthouse.py` (p99 1.41 m / max 2.67 m, worst velocity step 1.50 m/s), and
 *    max_reject_streak 50 was derived the same way.  Do NOT re-derive them here - an
 *    earlier draft used 1.5 m/s and rejected EVERY fix after a 2.2 s blackout.
 *
 *    KNOWN GAP, deliberately not closed: a SLOW persistent ramp (the measured runaway was
 *    0.02-0.04 m/step) is invisible to any per-step test.  `max_fix_range` (distance from
 *    the anchor) is the only guard that can see it, and it is why that field exists.
 *
 * 2. FAILSAFE DESCENT (`failsafe_*`)
 *    Today a safety breach calls `policy_disarm`, which hands the stock PID the PILOT's
 *    setpoint.  There is no firmware deadman, so a mid-air abort means attitude-only
 *    control on a stale thrust command - the measured crash chain.  The failsafe instead
 *    keeps OWNING the setpoint and flies a bounded, self-levelling descent, then disarms
 *    near the ground.
 *
 *    This is the one job the angle interface is right for (see PROPOSED_CHANGES.md (2)):
 *    it is bounded, needs no reference tracking, and must not be able to invert.
 */

#ifndef POLICY_GUARD_H
#define POLICY_GUARD_H

#include <stdbool.h>
#include <stdint.h>

/* ==================================================================================
 * 1. estimator plausibility gate
 * ================================================================================== */
typedef struct {
    float max_fix_jump;         /* m, per fix                                    */
    float max_fix_dv;           /* m/s, per fix                                  */
    float max_fix_range;        /* m from the anchor; <= 0.0 disables            */
    uint16_t max_reject_streak; /* consecutive rejections that escalate          */
} guard_cfg_t;

typedef struct {
    guard_cfg_t cfg;
    float p[3];                 /* last PLAUSIBLE position (world frame)         */
    float v[3];                 /* last PLAUSIBLE velocity                       */
    float anchor[3];            /* set at ARM; used by max_fix_range             */
    bool hold;                  /* a rejection is being suppressed right now     */
    bool primed;                /* at least one plausible sample has been seen   */
    uint16_t reject_streak;     /* consecutive rejections                        */
    uint32_t reject_total;      /* latched count, for the log                    */
} guard_t;

typedef struct {
    bool rejected;              /* this sample was refused                       */
    bool escalate;              /* streak exceeded -> hand over to the failsafe  */
} guard_out_t;

void guard_init(guard_t *g, const guard_cfg_t *cfg);
/* ARM / new episode: adopt the current estimate as plausible and re-seat the anchor. */
void guard_start(guard_t *g, const float p[3], const float v[3]);
/* One 100 Hz policy tick.  Never aborts anything itself - it only reports. */
guard_out_t guard_update(guard_t *g, const float p[3], const float v[3]);

/* ==================================================================================
 * 2. failsafe descent
 *
 * The commanded ATTITUDE is constant (level, zero yaw rate); only the thrust moves.  Keeping
 * the attitude in the command struct is what makes "never inverts" a testable property
 * rather than a claim about the caller.
 * ================================================================================== */
typedef struct {
    float ramp_s;               /* thrust ramp duration                          */
    float min_thrust_frac;      /* fraction of the entry thrust the ramp ends at */
    float hover_units;          /* legacy thrust units for a hover (ramp floor)  */
    float max_thrust_units;     /* clamp (the firmware's unit ceiling)           */
    float disarm_z;             /* disarm at or below this altitude              */
    float timeout_s;            /* hard stop, so the failsafe cannot own it forever */
} failsafe_cfg_t;

typedef struct {
    failsafe_cfg_t cfg;
    bool active;
    float t;                    /* seconds since entry */
    float entry_units;
    float start_units;          /* max(entry, hover), clamped */
} failsafe_t;

typedef struct {
    float thrust_units;         /* legacy units, driven with mode.z = modeDisable */
    float roll_deg;             /* commanded attitude (modeAbs) - always 0        */
    float pitch_deg;            /* always 0                                       */
    float yaw_rate_dps;         /* always 0                                       */
    bool level;                 /* true while the command is a level descent      */
    bool done;                  /* the caller must disarm now                     */
} failsafe_cmd_t;

void failsafe_init(failsafe_t *f, const failsafe_cfg_t *cfg);
/* Enter with the thrust that was last commanded (used as the ramp start, floored at hover
 * so an entry from a zero-thrust moment - e.g. a flip's ballistic coast - still arrests the
 * fall before bleeding the thrust off). */
void failsafe_enter(failsafe_t *f, float entry_thrust_units);
/* dt_s is the tick period; z is the altitude to test for the disarm condition (pass the
 * HELD estimate, not a possibly-lying raw one). */
failsafe_cmd_t failsafe_update(failsafe_t *f, float dt_s, float z);

/* ==================================================================================
 * 3. the firmware-facing mapping, in plain fields
 *
 * Kept here so the "level, zero-angle, thrust-only" contract is asserted by the HOST TEST
 * instead of by reading controller_app.c.  The mode values are MIRRORED from the firmware's
 * `stabilizer_types.h` (`modeDisable = 0, modeAbs = 1, modeVelocity = 2`) because this
 * module must stay host-compilable; `controller_app.c` carries a `_Static_assert` against
 * the real enums, so a firmware change becomes a build error rather than a silent change in
 * what the vehicle does when it gives up.
 * ================================================================================== */
#define POLICY_GUARD_MODE_DISABLE  0
#define POLICY_GUARD_MODE_ABS      1
#define POLICY_GUARD_MODE_VELOCITY 2

typedef struct {
    int mode_roll, mode_pitch, mode_yaw, mode_x, mode_y, mode_z;
    float roll_deg, pitch_deg;
    float yaw_rate_dps;
    float thrust_units;
} failsafe_setpoint_t;

void failsafe_setpoint(const failsafe_cmd_t *c, failsafe_setpoint_t *sp);

#endif /* POLICY_GUARD_H */
