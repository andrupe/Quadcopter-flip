/* policy_guard.c - see policy_guard.h for WHY.  Pure C, no firmware headers. */

#include "policy_guard.h"

#include <math.h>
#include <string.h>

/* ==================================================================================
 * small helpers
 * ================================================================================== */
static float norm3(const float a[3])
{
    return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

static bool finite3(const float a[3])
{
    return isfinite(a[0]) && isfinite(a[1]) && isfinite(a[2]);
}

static void copy3(float dst[3], const float src[3])
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}

/* ==================================================================================
 * 1. estimator plausibility gate
 * ================================================================================== */
void guard_init(guard_t *g, const guard_cfg_t *cfg)
{
    memset(g, 0, sizeof(*g));
    g->cfg = *cfg;
}

void guard_start(guard_t *g, const float p[3], const float v[3])
{
    /* Trust the estimate the arm happened on: that is exactly what the bench does, and
     * refusing to start would make the gate un-armable on a fresh boot. */
    copy3(g->p, p);
    copy3(g->v, v);
    copy3(g->anchor, p);
    g->hold = false;
    g->primed = true;
    g->reject_streak = 0;
}

guard_out_t guard_update(guard_t *g, const float p[3], const float v[3])
{
    guard_out_t out;
    out.rejected = false;
    out.escalate = false;

    if (!g->primed) {
        copy3(g->p, p);
        copy3(g->v, v);
        copy3(g->anchor, p);
        g->primed = true;
        return out;
    }

    bool bad = false;
    if (!finite3(p) || !finite3(v)) {
        bad = true;
    } else {
        const float dp[3] = {p[0] - g->p[0], p[1] - g->p[1], p[2] - g->p[2]};
        const float dv[3] = {v[0] - g->v[0], v[1] - g->v[1], v[2] - g->v[2]};
        /* compared against the last PLAUSIBLE sample, so a persistent lie stays rejected
         * instead of being silently adopted one small step at a time */
        if (norm3(dp) > g->cfg.max_fix_jump || norm3(dv) > g->cfg.max_fix_dv) {
            bad = true;
        } else if (g->cfg.max_fix_range > 0.0f) {
            const float dr[3] = {p[0] - g->anchor[0], p[1] - g->anchor[1],
                                 p[2] - g->anchor[2]};
            if (norm3(dr) > g->cfg.max_fix_range) {
                bad = true;
            }
        }
    }

    if (bad) {
        /* HOLD: g->p / g->v are left untouched, which is what freezes the policy input. */
        g->hold = true;
        if (g->reject_streak < 0xFFFFu) {
            g->reject_streak++;
        }
        g->reject_total++;
        out.rejected = true;
        if (g->reject_streak >= g->cfg.max_reject_streak) {
            out.escalate = true;
        }
    } else {
        copy3(g->p, p);
        copy3(g->v, v);
        g->hold = false;
        g->reject_streak = 0;
    }
    return out;
}

/* ==================================================================================
 * 2. failsafe descent
 * ================================================================================== */
void failsafe_init(failsafe_t *f, const failsafe_cfg_t *cfg)
{
    memset(f, 0, sizeof(*f));
    f->cfg = *cfg;
}

void failsafe_enter(failsafe_t *f, float entry_thrust_units)
{
    f->active = true;
    f->t = 0.0f;
    f->entry_units = entry_thrust_units;

    float start = entry_thrust_units;
    if (!isfinite(start) || start < f->cfg.hover_units) {
        start = f->cfg.hover_units;
    }
    if (f->cfg.max_thrust_units > 0.0f && start > f->cfg.max_thrust_units) {
        start = f->cfg.max_thrust_units;
    }
    if (start < 0.0f) {
        start = 0.0f;
    }
    f->start_units = start;
}

failsafe_cmd_t failsafe_update(failsafe_t *f, float dt_s, float z)
{
    failsafe_cmd_t c;
    c.thrust_units = 0.0f;
    c.roll_deg = 0.0f;
    c.pitch_deg = 0.0f;
    c.yaw_rate_dps = 0.0f;
    c.level = true;
    c.done = false;

    if (!f->active) {
        c.done = true;
        return c;
    }

    if (!isfinite(dt_s) || dt_s < 0.0f) {
        dt_s = 0.0f;
    }
    f->t += dt_s;

    const float ramp_s = (f->cfg.ramp_s > 1e-3f) ? f->cfg.ramp_s : 1e-3f;
    float u = f->t / ramp_s;
    if (u > 1.0f) {
        u = 1.0f;
    }
    const float end_units = f->cfg.min_thrust_frac * f->start_units;
    c.thrust_units = f->start_units + (end_units - f->start_units) * u;
    if (c.thrust_units < 0.0f) {
        c.thrust_units = 0.0f;
    }

    /* Disarm on: the ground, the ramp completing, or the hard timeout.  The ground case is
     * the normal one (the ramp starts at hover and reaches it inside the ramp for any sane
     * entry altitude); the other two exist so the failsafe can never own the setpoint
     * forever, which would be worse than the disarm it replaced. */
    if (isfinite(z) && z <= f->cfg.disarm_z) {
        c.done = true;
    }
    if (f->t >= ramp_s) {
        c.done = true;
    }
    if (f->t >= f->cfg.timeout_s) {
        c.done = true;
    }
    if (c.done) {
        f->active = false;
    }
    return c;
}

/* ==================================================================================
 * 3. firmware-facing mapping
 * ================================================================================== */
void failsafe_setpoint(const failsafe_cmd_t *c, failsafe_setpoint_t *sp)
{
    /* roll/pitch are ANGLE commands (modeAbs) so the stock attitude loop levels the
     * vehicle; yaw is a zero RATE; z is disabled so `thrust_units` is used verbatim - the
     * same shape the stock CRTP rate input produces.  Nothing here can command a rotation,
     * which is the whole point of the mode. */
    sp->mode_roll = POLICY_GUARD_MODE_ABS;
    sp->mode_pitch = POLICY_GUARD_MODE_ABS;
    sp->mode_yaw = POLICY_GUARD_MODE_VELOCITY;
    sp->mode_x = POLICY_GUARD_MODE_DISABLE;
    sp->mode_y = POLICY_GUARD_MODE_DISABLE;
    sp->mode_z = POLICY_GUARD_MODE_DISABLE;
    sp->roll_deg = c->roll_deg;
    sp->pitch_deg = c->pitch_deg;
    sp->yaw_rate_dps = c->yaw_rate_dps;
    sp->thrust_units = c->thrust_units;
}
