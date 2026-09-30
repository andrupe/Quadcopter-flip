// Analytic Onboard Flip Trajectory Generator
// Evaluates closed-form Flip equations at 100 Hz based on runtime parameters.

#include "analytic_flip.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

void analytic_flip_params_default(analytic_flip_params_t *p,
                                  float mass, float max_thrust, float gravity)
{
    p->peak_dps = ANALYTIC_FLIP_DEFAULT_PEAK_DPS;
    p->pop_pct = ANALYTIC_FLIP_DEFAULT_POP_PCT;
    p->rate_frac = ANALYTIC_FLIP_DEFAULT_RATE_FRAC;
    p->axis = ANALYTIC_FLIP_DEFAULT_AXIS;
    p->mass = (mass > 0.005f) ? mass : 0.033f;
    p->max_thrust = (max_thrust > 0.1f) ? max_thrust : ANALYTIC_FLIP_DEFAULT_MAX_THRUST;
    p->gravity = (gravity > 1.0f) ? gravity : 9.81f;
}

void analytic_flip_params_sanitize(analytic_flip_params_t *p)
{
    // Clamp peak body rate between 400 and 1100 deg/s (safely below 1146 dps actor limit)
    if (p->peak_dps < 400.0f) p->peak_dps = 400.0f;
    if (p->peak_dps > 1100.0f) p->peak_dps = 1100.0f;

    // Clamp pop thrust percentage
    if (p->pop_pct < 0.70f) p->pop_pct = 0.70f;
    if (p->pop_pct > 0.95f) p->pop_pct = 0.95f;

    // Clamp rate ramp fraction
    if (p->rate_frac < 0.15f) p->rate_frac = 0.15f;
    if (p->rate_frac > 0.45f) p->rate_frac = 0.45f;

    // Axis: 0 = pitch, 1 = roll
    if (p->axis != 0u && p->axis != 1u) p->axis = 0u;

    if (p->mass < 0.01f || p->mass > 0.10f) p->mass = 0.033f;
    if (p->max_thrust < 0.2f || p->max_thrust > 2.0f) p->max_thrust = ANALYTIC_FLIP_DEFAULT_MAX_THRUST;
    if (p->gravity < 5.0f || p->gravity > 15.0f) p->gravity = 9.81f;
}

void analytic_flip_init(analytic_flip_state_t *s,
                        const analytic_flip_params_t *params,
                        const float p0[3],
                        float yaw0,
                        uint32_t step100)
{
    s->params = *params;
    analytic_flip_params_sanitize(&s->params);

    s->p0[0] = p0[0];
    s->p0[1] = p0[1];
    s->p0[2] = p0[2];
    s->yaw0 = yaw0;
    s->cy = cosf(yaw0);
    s->sy = sinf(yaw0);
    s->z_shift = 0.0f;
    s->start_step = step100;

    const float omega_target = s->params.peak_dps * ((float)M_PI / 180.0f);
    s->coast = (2.0f * (float)M_PI) / (omega_target * (1.0f - s->params.rate_frac));
    s->omega_peak = (2.0f * (float)M_PI) / (s->coast * (1.0f - s->params.rate_frac));
    s->v0 = 0.5f * s->params.gravity * s->coast;

    const float a_max_net = (s->params.pop_pct * s->params.max_thrust / s->params.mass) - s->params.gravity;
    s->u = (a_max_net > 0.5f) ? a_max_net : 0.5f;
    s->c0 = s->v0 / s->u;
    s->dur_maneuver = 2.0f * s->c0 + s->coast;
    s->dur_total = s->dur_maneuver + ANALYTIC_FLIP_TAIL_S;

    s->rot_start_row = (uint32_t)roundf(s->c0 / ANALYTIC_FLIP_DT);
    s->catch_start_row = (uint32_t)roundf((s->c0 + s->coast) / ANALYTIC_FLIP_DT);
    s->total_rows = (uint32_t)roundf(s->dur_total / ANALYTIC_FLIP_DT);
}

void analytic_flip_sample(const analytic_flip_state_t *s,
                          uint32_t step100,
                          float p[3], float v[3], float R[9], float w[3], float a[3])
{
    const uint32_t k = (step100 >= s->start_step) ? (step100 - s->start_step) : 0u;
    const float t = (float)k * ANALYTIC_FLIP_DT;

    float z, v_z, a_z;
    const float c0 = s->c0;
    const float c1 = s->c0 + s->coast;
    const float c2 = s->dur_maneuver;
    const float g = s->params.gravity;
    const float u = s->u;

    if (t <= c0) {
        z = s->p0[2] + 0.5f * u * t * t;
        v_z = u * t;
        a_z = u;
    } else if (t <= c1) {
        const float dt = t - c0;
        const float v_entry = u * c0;
        z = s->p0[2] + 0.5f * u * c0 * c0 + v_entry * dt - 0.5f * g * dt * dt;
        v_z = v_entry - g * dt;
        a_z = -g;
    } else if (t <= c2) {
        const float v_exit = u * c0 - g * s->coast;
        const float z_exit = s->p0[2] + 0.5f * u * c0 * c0 + u * c0 * s->coast - 0.5f * g * s->coast * s->coast;
        const float dt = t - c1;
        z = z_exit + v_exit * dt + 0.5f * u * dt * dt;
        v_z = v_exit + u * dt;
        a_z = u;
    } else {
        // Terminal hover tail
        z = s->p0[2];
        v_z = 0.0f;
        a_z = 0.0f;
    }

    // Apply catch altitude shift (if re-anchored)
    p[0] = s->p0[0];
    p[1] = s->p0[1];
    p[2] = z + s->z_shift;

    v[0] = 0.0f;
    v[1] = 0.0f;
    v[2] = v_z;

    a[0] = 0.0f;
    a[1] = 0.0f;
    a[2] = a_z;

    // Spin schedule
    float phi = 0.0f;
    float phi_dot = 0.0f;
    if (t > c0 && t < c1) {
        const float dt = t - c0;
        const float ramp = s->params.rate_frac * s->coast;
        const float total_phi = 2.0f * (float)M_PI;
        if (dt <= ramp) {
            phi = 0.5f * s->omega_peak * dt * dt / ramp;
            phi_dot = s->omega_peak * dt / ramp;
        } else if (dt <= s->coast - ramp) {
            phi = 0.5f * s->omega_peak * ramp + s->omega_peak * (dt - ramp);
            phi_dot = s->omega_peak;
        } else {
            const float rem = s->coast - dt;
            phi = total_phi - 0.5f * s->omega_peak * rem * rem / ramp;
            phi_dot = s->omega_peak * rem / ramp;
        }
    } else if (t >= c1) {
        phi = 2.0f * (float)M_PI;
        phi_dot = 0.0f;
    }

    const float cp = cosf(phi);
    const float sp = sinf(phi);
    const float cy = s->cy;
    const float sy = s->sy;

    if (s->params.axis == 0u) {
        // Pitch flip (world y axis / body pitch rotated by heading)
        w[0] = phi_dot * sy;
        w[1] = phi_dot * cy;
        w[2] = 0.0f;

        // R = Ry(phi) @ Rz(yaw)
        R[0] = cp * cy;  R[1] = -cp * sy; R[2] = sp;
        R[3] = sy;       R[4] = cy;       R[5] = 0.0f;
        R[6] = -sp * cy; R[7] = sp * sy;  R[8] = cp;
    } else {
        // Roll flip (world x axis / body roll rotated by heading)
        w[0] = phi_dot * cy;
        w[1] = -phi_dot * sy;
        w[2] = 0.0f;

        // R = Rx(phi) @ Rz(yaw)
        R[0] = cy;       R[1] = -sy;      R[2] = 0.0f;
        R[3] = cp * sy;  R[4] = cp * cy;  R[5] = -sp;
        R[6] = sp * sy;  R[7] = sp * cy;  R[8] = cp;
    }
}

void analytic_flip_reanchor_z(analytic_flip_state_t *s, float current_z)
{
    s->z_shift = current_z - s->p0[2];
}

bool analytic_flip_finished(const analytic_flip_state_t *s, uint32_t step100)
{
    if (step100 < s->start_step) {
        return false;
    }
    return (step100 - s->start_step) >= s->total_rows;
}
