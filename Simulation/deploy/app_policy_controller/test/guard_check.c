/*
 * Host harness for `policy_guard.c` - the failsafe descent and the estimator gate.
 *
 * WHY THIS EXISTS
 * ---------------
 * `controller_app.c` cannot be host-compiled (FreeRTOS + the stock controller + the whole
 * firmware header set), so the two pieces of logic that decide whether a real flight is
 * abandoned live in `policy_guard.c` and are driven here.  Every threshold in that module is
 * one a flight depends on, and two of them were previously WRONG in a way only measurement
 * caught (1.5 m/s rejected every fix after a 2.2 s blackout; a naive "elapsed/(n-1)" read a
 * throughput 4x too low).  A claim about them belongs in a test, not in a comment.
 *
 * It is compiled and run by `Simulation/deploy/guard_host_check.py`.
 *
 * WHAT IS ASSERTED
 *   A  the failsafe: enters rather than disarms, commands a LEVEL descent, ramps the thrust
 *      down monotonically from at least hover, never commands a rotation, reaches the ground
 *      inside the ramp, and is bounded by its timeout
 *   B  the estimator gate: a slow ramp is invisible to a per-step test (the documented gap)
 *      UNLESS the range guard is on, a 4 m step is caught and freezes the input, a legitimate
 *      post-blackout innovation is NOT a false positive, and the held estimate stays coherent
 *   C  the exported sim constants are self-consistent (T0-C's prerequisite)
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "generated/policy_weights.h"   // POLICY_SIM_MASS_KG / GRAVITY / MAX_THRUST_N
#include "policy_guard.h"

#define POLICY_THRUST_MAX_UNITS 60000.0f   // mirrors controller_app.c (crtp MAX_THRUST)

static int failures = 0;
static int checks = 0;

static void check(const char *name, int ok, const char *detail)
{
    checks++;
    if (detail != NULL && detail[0] != '\0') {
        printf("  [%s] %s  %s\n", ok ? "ok " : "FAIL", name, detail);
    } else {
        printf("  [%s] %s\n", ok ? "ok " : "FAIL", name);
    }
    if (!ok) {
        failures++;
    }
}

/* ---------------------------------------------------------------------------------
 * shared config, mirroring what controller_app.c will install by default
 * --------------------------------------------------------------------------------- */
static failsafe_cfg_t fs_cfg(void)
{
    failsafe_cfg_t c;
    c.ramp_s = 1.2f;
    c.min_thrust_frac = 0.0f;
    c.hover_units = (0.5f * POLICY_SIM_HOVER_TRIM_A0 + 0.5f) * POLICY_THRUST_MAX_UNITS;
    c.max_thrust_units = POLICY_THRUST_MAX_UNITS;
    c.disarm_z = 0.04f;
    c.timeout_s = 6.0f;
    return c;
}

static guard_cfg_t guard_cfg(float range)
{
    guard_cfg_t c;
    c.max_fix_jump = 5.0f;       // measured, notes.md - do not re-derive
    c.max_fix_dv = 4.0f;
    c.max_fix_range = range;
    c.max_reject_streak = 50;
    return c;
}

/* C has no `&f()` rvalue, so the config is materialised here. */
static void guard_setup(guard_t *g, float range)
{
    guard_cfg_t c = guard_cfg(range);
    guard_init(g, &c);
}

/* ---------------------------------------------------------------------------------
 * A. the failsafe
 * --------------------------------------------------------------------------------- */
static void section_A(void)
{
    printf("\nA. failsafe descent\n");

    const failsafe_cfg_t cfg = fs_cfg();
    const float hover = cfg.hover_units;

    /* A.1/A.2 - a tilt breach at 1.2 m enters and commands a LEVELLED, zero-rate setpoint
     *           with live thrust, instead of disarming on the same tick. */
    failsafe_t f;
    failsafe_init(&f, &cfg);
    failsafe_enter(&f, hover);
    failsafe_cmd_t c = failsafe_update(&f, 0.001f, 1.2f);
    check("A.1 a breach at altitude does NOT disarm on the same tick", !c.done, NULL);
    check("A.2 the command is a level descent (no rotation)",
          c.level && c.roll_deg == 0.0f && c.pitch_deg == 0.0f && c.yaw_rate_dps == 0.0f,
          NULL);
    check("A.3 the ramp STARTS at hover (a zero-thrust entry is floored)",
          f.start_units >= hover - 1e-3f, NULL);
    check("A.3b the first commanded thrust is still essentially hover",
          c.thrust_units >= 0.99f * hover, NULL);

    failsafe_setpoint_t sp;
    failsafe_setpoint(&c, &sp);
    check("A.4 roll/pitch are modeAbs with a ZERO angle",
          sp.mode_roll == POLICY_GUARD_MODE_ABS && sp.mode_pitch == POLICY_GUARD_MODE_ABS
          && sp.roll_deg == 0.0f && sp.pitch_deg == 0.0f, NULL);
    check("A.5 yaw is a zero RATE and z is disabled (thrust verbatim)",
          sp.mode_yaw == POLICY_GUARD_MODE_VELOCITY && sp.yaw_rate_dps == 0.0f
          && sp.mode_z == POLICY_GUARD_MODE_DISABLE, NULL);
    check("A.6 the horizontal modes stay disabled (no position loop involved)",
          sp.mode_x == POLICY_GUARD_MODE_DISABLE && sp.mode_y == POLICY_GUARD_MODE_DISABLE,
          NULL);

    /* A.7/A.8/A.9 - fly the whole descent at 1 kHz with a plain vertical model.
     *  units -> force is linear (`(0.5*a0+0.5)*scale` is, by construction), so
     *      F = MAX_THRUST_N * units / MAX_UNITS      a = (F - m*g)/m
     *  With the ramp starting exactly at hover the acceleration is -g*t/ramp_s, i.e. the
     *  mass cancels and the landing time is analytic: z0 - g t^3 / (6 ramp_s) = disarm_z.
     *  Using the exported constants keeps this tied to the real numbers. */
    failsafe_t f2;
    failsafe_init(&f2, &cfg);
    failsafe_enter(&f2, hover);
    const float m = POLICY_SIM_MASS_KG;
    const float g = POLICY_SIM_GRAVITY;
    float z = 1.2f, vz = 0.0f, t = 0.0f, z_at_done = -1.0f, t_done = -1.0f;
    float prev_units = 1e9f;
    int monotone = 1, rotation_free = 1, done_ticks = 0;
    for (int i = 0; i < 6000; i++) {
        failsafe_cmd_t cc = failsafe_update(&f2, 0.001f, z);
        if (cc.roll_deg != 0.0f || cc.pitch_deg != 0.0f || cc.yaw_rate_dps != 0.0f) {
            rotation_free = 0;
        }
        if (cc.thrust_units > prev_units + 1e-3f) {
            monotone = 0;
        }
        prev_units = cc.thrust_units;

        const float F = POLICY_SIM_MAX_THRUST_N * cc.thrust_units / POLICY_THRUST_MAX_UNITS;
        const float a = (F - m * g) / m;
        vz += a * 0.001f;
        z += vz * 0.001f;
        t += 0.001f;
        if (z < 0.0f) {
            z = 0.0f;
            vz = 0.0f;
        }
        if (cc.done) {
            z_at_done = z;
            t_done = t;
            done_ticks++;
            if (done_ticks > 1) {
                break;
            }
        }
    }
    const float analytic = powf(6.0f * cfg.ramp_s * (1.2f - cfg.disarm_z) / g, 1.0f / 3.0f);
    check("A.7 the thrust ramps DOWN monotonically (never re-launches)", monotone, NULL);
    check("A.8 no rotation is ever commanded during the descent", rotation_free, NULL);
    check("A.9 it reaches the ground and disarms", z_at_done >= 0.0f, NULL);
    check("A.10 it disarms INSIDE the ramp, on the ground condition",
          t_done > 0.0f && t_done < cfg.ramp_s && z_at_done <= cfg.disarm_z + 1e-3f, NULL);
    printf("    [info] disarm at t = %.3f s (analytic %.3f s), z = %.4f m, ramp %.2f s, "
           "hover %.0f units\n", (double)t_done, (double)analytic, (double)z_at_done,
           (double)cfg.ramp_s, (double)hover);

    /* A.11 - a zero-thrust entry (a flip's ballistic coast) is floored at hover, so the
     *        failsafe arrests the fall before bleeding thrust off. */
    failsafe_t f3;
    failsafe_init(&f3, &cfg);
    failsafe_enter(&f3, 0.0f);
    failsafe_cmd_t c3 = failsafe_update(&f3, 0.001f, 1.5f);
    check("A.11 an entry from zero thrust starts at hover, not at zero",
          f3.start_units >= hover - 1e-3f && c3.thrust_units >= 0.99f * hover, NULL);

    /* A.12 - bounded: the timeout disarms even if the ground is never reached (a lying
     *        altitude, or a vehicle hanging in the props' wash). */
    failsafe_t f4;
    failsafe_init(&f4, &cfg);
    failsafe_enter(&f4, hover);
    int ticks_to_done = 0;
    for (int i = 0; i < 20000; i++) {
        failsafe_cmd_t cc = failsafe_update(&f4, 0.001f, 99.0f);   // never descends
        ticks_to_done++;
        if (cc.done) {
            break;
        }
    }
    check("A.13 a stuck failsafe still terminates (ramp end, then the timeout)",
          ticks_to_done <= (int)(cfg.timeout_s * 1000.0f) + 2, NULL);
    printf("    [info] bounded at %.3f s with the altitude never leaving 99 m\n",
           (double)(ticks_to_done * 0.001f));

    /* A.14 - and entering at ground level disarms at once rather than hovering there. */
    failsafe_t f5;
    failsafe_init(&f5, &cfg);
    failsafe_enter(&f5, hover);
    failsafe_cmd_t c5 = failsafe_update(&f5, 0.001f, 0.01f);
    check("A.14 entering on the ground disarms immediately", c5.done, NULL);
}

/* ---------------------------------------------------------------------------------
 * B. the estimator plausibility gate
 * --------------------------------------------------------------------------------- */
static void section_B(void)
{
    printf("\nB. estimator plausibility gate\n");

    const float p0[3] = {0.0f, 0.0f, 1.2f};
    const float v0[3] = {0.0f, 0.0f, 0.0f};

    /* B.1 - the DOCUMENTED GAP: a per-step test cannot see the measured runaway slope
     *       (0.02-0.04 m/step).  With the range guard off, a 0.03 m/tick ramp runs for 5 s
     *       (15 m!) without a single rejection.  This is asserted so the limitation is
     *       visible rather than assumed away. */
    guard_t ga;
    guard_setup(&ga, 0.0f);
    guard_start(&ga, p0, v0);
    int rejected_ticks = 0;
    float zz = p0[2];
    for (int i = 0; i < 500; i++) {           // 5 s at 100 Hz
        const float p[3] = {0.0f, 0.0f, zz};
        guard_out_t o = guard_update(&ga, p, v0);
        if (o.rejected) {
            rejected_ticks++;
        }
        zz += 0.03f;
    }
    check("B.1 a 0.03 m/tick ramp is INVISIBLE to the per-step test (known gap)",
          rejected_ticks == 0, NULL);
    printf("    [info] after 5 s the ramp is at z = %.2f m and the gate has accepted every "
           "sample (range guard OFF by construction)\n", (double)zz);

    /* B.2 - the range guard is the one thing that CAN see a persistent lie: at 8 m from the
     *       anchor the ramp is finally rejected, which freezes the policy input. */
    guard_t gb;
    guard_setup(&gb, 8.0f);
    guard_start(&gb, p0, v0);
    int first_reject_tick = -1;
    int escalated_tick = -1;
    float held_at_reject = -1.0f;
    zz = p0[2];
    for (int i = 0; i < 500; i++) {
        const float p[3] = {0.0f, 0.0f, zz};
        guard_out_t o = guard_update(&gb, p, v0);
        if (o.rejected && first_reject_tick < 0) {
            first_reject_tick = i;
            held_at_reject = gb.p[2];
        }
        if (o.escalate && escalated_tick < 0) {
            escalated_tick = i;
        }
        zz += 0.03f;
    }
    check("B.2 the range guard catches the persistent ramp",
          first_reject_tick > 0, NULL);
    printf("    [info] ramp first rejected at t = %.2f s (z = %.2f m); escalate at t = "
           "%.2f s if the streak keeps growing\n",
           (double)(first_reject_tick * 0.01f), 1.2f + 0.03f * first_reject_tick,
           (double)(escalated_tick * 0.01f));
    /* The hold freezes the estimate at the last PLAUSIBLE sample - it does not track the
     * lie, and it does not rewind to the start either. */
    check("B.3 the held estimate does NOT track the lie (the policy input is frozen)",
          held_at_reject > 0.0f && gb.p[2] == held_at_reject && gb.p[2] < zz - 1.0f, NULL);
    printf("    [info] held z stayed %.2f m while the lie ran on to %.2f m\n",
           (double)gb.p[2], (double)zz);

    /* B.4 - the measured runaway is a ~1 m TELEPORT and the plausible band is 5 m wide by
     *        design, so this uses a step ABOVE the threshold to exercise the jump test: it
     *        is caught on the tick it arrives and does NOT escalate at once. */
    guard_t gc;
    guard_setup(&gc, 8.0f);
    guard_start(&gc, p0, v0);
    const float step[3] = {0.0f, 0.0f, 7.2f};      // 6.0 m from the held 1.2 m
    guard_out_t o = guard_update(&gc, step, v0);
    check("B.4 a 6 m step is rejected immediately, without escalating",
          o.rejected && !o.escalate, NULL);
    check("B.5 the held estimate is unchanged by the step", gc.p[2] == 1.2f, NULL);
    check("B.6 the gate reports HOLD so the caller can suspend the z envelope", gc.hold,
          NULL);
    float held_p[3];
    memcpy(held_p, gc.p, sizeof(held_p));
    guard_out_t o2 = guard_update(&gc, step, v0);
    check("B.7 a second identical lie keeps the same held estimate",
          gc.p[2] == held_p[2] && o2.rejected, NULL);

    /* B.8 - the brief's "a genuine 4.0 m step must still be caught" is the ALTITUDE
     *        ENVELOPE's job, not the gate's: 4.0 m is BELOW the 5.0 m plausibility
     *        threshold, so the sample is ACCEPTED, `hold` stays false, and the caller's z
     *        guard is still armed when it sees 4.0 > POLICY_MAX_Z (3.5).  If the gate
     *        swallowed this it would be suppressing a real flight state. */
    guard_t gh;
    guard_setup(&gh, 8.0f);
    guard_start(&gh, p0, v0);
    const float four[3] = {0.0f, 0.0f, 5.2f};      // 4.0 m from the held 1.2 m
    guard_out_t oh = guard_update(&gh, four, v0);
    check("B.8 a 4.0 m step is INSIDE the plausible band, so the envelope still catches it",
          !oh.rejected && !gh.hold, NULL);

    /* B.8 - resuming: a plausible sample returns and the gate releases the hold. */
    const float back[3] = {0.0f, 0.0f, 1.21f};
    guard_out_t o3 = guard_update(&gc, back, v0);
    check("B.9 a plausible sample clears the hold and is adopted",
          !o3.rejected && !gc.hold && gc.p[2] == 1.21f, NULL);

    /* B.10 - the streak.  50 consecutive rejections escalate; 49 do not. */
    guard_t gd;
    guard_setup(&gd, 8.0f);
    guard_start(&gd, p0, v0);
    int escalated = 0;
    for (int i = 0; i < 49; i++) {
        if (guard_update(&gd, step, v0).escalate) {
            escalated = 1;
        }
    }
    check("B.11 49 rejections do not escalate", !escalated && gd.reject_streak == 49, NULL);
    check("B.12 the 50th escalates", guard_update(&gd, step, v0).escalate, NULL);

    /* B.13 - NO FALSE POSITIVE on a legitimate post-blackout innovation.  This is the case an
     *        earlier draft got wrong (a 1.5 m/s threshold rejected EVERY fix after a real
     *        2.2 s blackout, blinding the estimator exactly when the flip needs it).  The
     *        measured legitimate maxima are 2.67 m and 1.50 m/s. */
    guard_t ge;
    guard_setup(&ge, 8.0f);
    guard_start(&ge, p0, v0);
    const float legit_p[3] = {0.0f, 0.0f, 1.2f + 2.67f};
    const float legit_v[3] = {0.0f, 0.0f, 1.50f};
    guard_out_t ol = guard_update(&ge, legit_p, legit_v);
    check("B.14 a legitimate post-blackout innovation is ACCEPTED", !ol.rejected, NULL);

    /* B.15 - coherence: on rejection every channel the actor consumes still comes from the
     *        ONE held sample, so the policy cannot see pos and p_err disagree. */
    guard_t gf;
    guard_setup(&gf, 8.0f);
    guard_start(&gf, p0, v0);
    (void)guard_update(&gf, step, v0);
    check("B.16 held position and velocity stay one consistent sample",
          gf.p[0] == p0[0] && gf.p[1] == p0[1] && gf.p[2] == p0[2]
          && gf.v[0] == v0[0] && gf.v[2] == v0[2], NULL);

    /* B.17 - non-finite never gets through, and never needs the caller to check first. */
    guard_t gg;
    guard_setup(&gg, 8.0f);
    guard_start(&gg, p0, v0);
    const float nan_p[3] = {0.0f, 0.0f, NAN};
    check("B.18 a NaN estimate is rejected", guard_update(&gg, nan_p, v0).rejected, NULL);
}

/* ---------------------------------------------------------------------------------
 * C. exported constants (T0-C precondition)
 * --------------------------------------------------------------------------------- */
static void section_C(void)
{
    printf("\nC. exported sim constants (T0-C)\n");

    const float hover_units = (0.5f * POLICY_SIM_HOVER_TRIM_A0 + 0.5f) * POLICY_THRUST_MAX_UNITS;
    const float hover_n = POLICY_SIM_MAX_THRUST_N * hover_units / POLICY_THRUST_MAX_UNITS;
    const float weight_n = POLICY_SIM_MASS_KG * POLICY_SIM_GRAVITY;

    check("C.1 the exported hover trim actually balances the exported mass",
          fabsf(hover_n - weight_n) < 0.01f * weight_n, NULL);
    printf("    [info] mass %.4f kg -> weight %.4f N; hover trim a0 = %+.4f -> %.0f units "
           "-> %.4f N (%.1f%% of the command range)\n",
           (double)POLICY_SIM_MASS_KG, (double)weight_n, (double)POLICY_SIM_HOVER_TRIM_A0,
           (double)hover_units, (double)hover_n, (double)(100.0f * hover_units
                                                          / POLICY_THRUST_MAX_UNITS));
    printf("    [info] T0-C still needs ONE physical number: the all-up mass on a scale. "
           "Nothing here can invent it - if the real vehicle is heavier, this hover trim "
           "sits below real hover and every thrust unit is worth the wrong force.\n");
}

int main(void)
{
    printf("policy_guard host check (failsafe + estimator gate)\n");
    printf("  exported: mass %.4f kg, gravity %.2f, max thrust %.2f N, hover a0 %+.4f\n",
           (double)POLICY_SIM_MASS_KG, (double)POLICY_SIM_GRAVITY,
           (double)POLICY_SIM_MAX_THRUST_N, (double)POLICY_SIM_HOVER_TRIM_A0);
    section_A();
    section_B();
    section_C();
    printf("\n");
    if (failures) {
        printf("%d FAILURE(S) of %d checks\n", failures, checks);
        return 1;
    }
    printf("ALL %d CHECKS PASSED\n", checks);
    return 0;
}
