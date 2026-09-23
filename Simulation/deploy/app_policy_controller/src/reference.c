// Onboard reference generator - see reference.h for the contract and the maths.

#include "reference.h"

#include <math.h>
#include <string.h>

static void rot_z(const ref_state_t *r, const float in[3], float out[3])
{
    out[0] = r->c * in[0] - r->s * in[1];
    out[1] = r->s * in[0] + r->c * in[1];
    out[2] = in[2];
}

void ref_hold_sync(ref_state_t *r, const float p[3], float yaw)
{
    r->mode = REF_MODE_HOLD;
    r->kind = REF_KIND_NONE;
    memcpy(r->p0, p, sizeof(r->p0));
    r->yaw0 = yaw;
    r->c = 1.0f;
    r->s = 0.0f;
    memset(r->p_shift, 0, sizeof(r->p_shift));
    r->start_step = 0;
}

void ref_play_launch(ref_state_t *r, uint8_t kind, const float p[3], float yaw,
                     uint32_t step100)
{
    if (kind >= REF_TABLE_COUNT) {
        ref_hold_sync(r, p, yaw);          // "stop and hold" - see reference.h
        return;
    }

    const float dyaw = yaw - REF_YAW0[kind];
    const float c = cosf(dyaw);
    const float s = sinf(dyaw);

    // p_shift = p_live - Rz @ REF_P0[kind]
    const float p0_rot[3] = {
        c * REF_P0[kind][0] - s * REF_P0[kind][1],
        s * REF_P0[kind][0] + c * REF_P0[kind][1],
        REF_P0[kind][2],
    };

    r->mode = REF_MODE_MANOEUVRE;
    r->kind = kind;
    r->c = c;
    r->s = s;
    r->p_shift[0] = p[0] - p0_rot[0];
    r->p_shift[1] = p[1] - p0_rot[1];
    r->p_shift[2] = p[2] - p0_rot[2];
    r->start_step = step100;
    // p0/yaw0 are not used while a manoeuvre plays, but keep them coherent so an abort can
    // fall back to a hold at the CURRENT pose without touching the caller's state.
    memcpy(r->p0, p, sizeof(r->p0));
    r->yaw0 = yaw;
}

void ref_play_to_hold(ref_state_t *r, const float p[3], float yaw)
{
    ref_hold_sync(r, p, yaw);
}

uint8_t ref_kind_count(void)
{
    return (uint8_t)REF_TABLE_COUNT;
}

const char *ref_kind_name(uint8_t kind)
{
    return (kind < REF_TABLE_COUNT) ? REF_NAME[kind] : NULL;
}

uint32_t ref_kind_len(uint8_t kind)
{
    return (kind < REF_TABLE_COUNT) ? (uint32_t)REF_LEN[kind] : 0u;
}

bool ref_play_finished(const ref_state_t *r, uint32_t step100)
{
    if (r->mode != REF_MODE_MANOEUVRE || r->kind >= REF_TABLE_COUNT) {
        return false;
    }
    return (uint32_t)(step100 - r->start_step) >= (uint32_t)REF_LEN[r->kind];
}

static void sample_hold(const ref_state_t *r, float p[3], float v[3], float R[9], float w[3],
                        float a[3])
{
    p[0] = r->p0[0];
    p[1] = r->p0[1];
    p[2] = r->p0[2];
    memset(v, 0, 3 * sizeof(float));
    memset(w, 0, 3 * sizeof(float));
    // A hold is a STATIONARY point: zero acceleration, so the feed-forward block is exactly
    // [0, 0, g]. Zeroing it is not a placeholder - it is the correct value.
    memset(a, 0, 3 * sizeof(float));
    // level attitude at the anchor heading: z_b = e_z -> R = Rz(yaw)
    const float cy = cosf(r->yaw0);
    const float sy = sinf(r->yaw0);
    R[0] = cy;   R[1] = -sy;  R[2] = 0.0f;
    R[3] = sy;   R[4] = cy;   R[5] = 0.0f;
    R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
}

void ref_feed_forward(const float a_ref[3], float gravity, float out[3])
{
    out[0] = a_ref[0];
    out[1] = a_ref[1];
    out[2] = a_ref[2] + gravity;
}

void ref_sample(const ref_state_t *r, uint32_t step100,
                float p[3], float v[3], float R[9], float w[3], float a[3])
{
    if (r->mode != REF_MODE_MANOEUVRE || r->kind >= REF_TABLE_COUNT) {
        sample_hold(r, p, v, R, w, a);
        return;
    }

    const uint32_t len = (uint32_t)REF_LEN[r->kind];
    uint32_t k = step100 - r->start_step;
    if (k >= len) {
        k = len - 1u;                          // the last row is a hover by construction
    }
    const float *row = REF_ROWS[REF_OFFSET[r->kind] + k];

    float tmp[3];
    rot_z(r, &row[0], tmp);
    p[0] = tmp[0] + r->p_shift[0];
    p[1] = tmp[1] + r->p_shift[1];
    p[2] = tmp[2] + r->p_shift[2];

    rot_z(r, &row[3], v);

    // OMEGA IS NOT ROTATED. The body frame turns WITH the vehicle, so for R' = Q R the
    // body rate is [w']_x = R'^T R'dot = R^T Q^T Q Rdot = R^T Rdot = [w]_x - the same
    // body-frame vector. Rotating it (by analogy with v and a) would describe a different
    // manoeuvre: a 90 deg yaw relocation of a pitch flip would claim a roll rate. It also
    // injected an error into the policy's w_err channel that grows with dyaw. This mirrors
    // trajectories.ShiftedManeuver, and `gen_references.py --check` verifies it against
    // live_target.ShiftedTrajectory, which does the same.
    memcpy(w, &row[15], 3 * sizeof(float));

    // `a` IS a WORLD-frame vector, so the relocation does rotate it. (Only the translation
    // leaves accelerations alone.)
    rot_z(r, &row[REF_A_OFFSET], a);

    // R = Rz @ R_table (rows of Rz times the table's row-major 3x3)
    for (int i = 0; i < 3; i++) {
        const float a = (i == 0) ? r->c : ((i == 1) ? r->s : 0.0f);
        const float b = (i == 0) ? -r->s : ((i == 1) ? r->c : 0.0f);
        const float cz = (i == 2) ? 1.0f : 0.0f;
        for (int j = 0; j < 3; j++) {
            R[3 * i + j] = a * row[6 + j] + b * row[9 + j] + cz * row[12 + j];
        }
    }
}
