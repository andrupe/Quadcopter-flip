// Deployed policy forward pass - see policy_net.h for the contract.
//
// Numerics are chosen to match the torch/numpy reference implementations that the host
// checker compares against: float32 throughout, plain division for the standardization,
// sequential accumulation for the mat-vecs, `erff` for GELU (torch's nn.GELU() default is
// the exact erf form, not the tanh approximation).

#include "policy_net.h"

#include <math.h>
#include <string.h>

static inline float sigmoidf_(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// y = tanh(W x + b), W is [out][in] row-major, x has `in` entries.
static void matvec_tanh(const float *w, const float *b, const float *x,
                        int out_dim, int in_dim, float *y)
{
    for (int i = 0; i < out_dim; i++) {
        const float *row = w + (size_t)i * (size_t)in_dim;
        float acc = b[i];
        for (int j = 0; j < in_dim; j++) {
            acc += row[j] * x[j];
        }
        y[i] = tanhf(acc);
    }
}

// y = W x + b (no activation).
static void matvec(const float *w, const float *b, const float *x,
                   int out_dim, int in_dim, float *y)
{
    for (int i = 0; i < out_dim; i++) {
        const float *row = w + (size_t)i * (size_t)in_dim;
        float acc = b[i];
        for (int j = 0; j < in_dim; j++) {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

void policy_reset(policy_state_t *st)
{
    memset(st, 0, sizeof(*st));
    st->ready = 1;
}

void policy_standardize(const float raw[POLICY_ENC_IN_DIM], float out[POLICY_ENC_IN_DIM])
{
    for (int i = 0; i < POLICY_ENC_IN_DIM; i++) {
        float v = (raw[i] - ENC_NORM_MEAN[i]) / ENC_NORM_STD[i];
        if (v > POLICY_ENC_NORM_CLIP) {
            v = POLICY_ENC_NORM_CLIP;
        } else if (v < -POLICY_ENC_NORM_CLIP) {
            v = -POLICY_ENC_NORM_CLIP;
        }
        out[i] = v;
    }
}

void policy_encoder_step(policy_state_t *st, const float frame[POLICY_ENC_IN_DIM])
{
    const int H = POLICY_GRU_HIDDEN;

#if POLICY_HAS_ENCODER
    float *x = st->x_std;
    policy_standardize(frame, x);

    // gates = W_ih x + b_ih  (input part)  and  W_hh h + b_hh  (hidden part)
    //
    // The two parts must stay SEPARATE: torch's GRU computes
    //     n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    // so b_hh passes through the reset gate. Folding b_hh into the input part instead
    // (measured: z error 0.05 at the first step) silently drops the r * b_hn term - and a
    // probe that reimplements the same wrong formula will happily agree with it.
    for (int i = 0; i < 3 * H; i++) {
        const float *wi = GRU_W_IH[i];
        const float *wh = GRU_W_HH[i];
        float a = GRU_B_IH[i];
        for (int j = 0; j < POLICY_ENC_IN_DIM; j++) {
            a += wi[j] * x[j];
        }
        st->gi[i] = a;
        float b = GRU_B_HH[i];
        for (int j = 0; j < H; j++) {
            b += wh[j] * st->h[j];
        }
        st->gh[i] = b;
    }

    for (int k = 0; k < H; k++) {
        st->gru_r[k] = sigmoidf_(st->gi[k] + st->gh[k]);
        st->gru_z[k] = sigmoidf_(st->gi[H + k] + st->gh[H + k]);
        st->gru_n[k] = tanhf(st->gi[2 * H + k] + st->gru_r[k] * st->gh[2 * H + k]);
        st->h[k] = (1.0f - st->gru_z[k]) * st->gru_n[k] + st->gru_z[k] * st->h[k];
    }

    // head: Linear -> GELU -> Linear -> Tanh. The generated weights are 2-D arrays; the
    // flat-pointer casts are exact (row-major, contiguous) and keep -Wpointer-sign quiet.
    matvec((const float *)ENC_HEAD_W1, ENC_HEAD_B1, st->h, H, H, st->head_a);
    for (int k = 0; k < H; k++) {
        const float v = st->head_a[k];
        // exact GELU: 0.5 x (1 + erf(x / sqrt(2)))
        st->head_a[k] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    }
    matvec_tanh((const float *)ENC_HEAD_W2, ENC_HEAD_B2, st->head_a, POLICY_Z_DIM, H, st->z);
#else
    (void)frame;
    memset(st->z, 0, sizeof(st->z));
#endif
}

void policy_actor(policy_state_t *st, const float o_t[POLICY_O_T_DIM],
                  const float ref_ff[POLICY_REF_FF_DIM], float act_out[POLICY_ACT_DIM])
{
    // [o_t(29) | z(16) | ref_ff(3)] - built from the state's z so the caller never assembles
    // a copy. The ORDER is the contract: it mirrors quad_flip_env's actor prefix exactly
    // (the wrapper inserts z between o_t and ref_ff), and POLICY_ACTOR_DIM_Z in policy_net.h
    // asserts the total width.
    float x[POLICY_ACTOR_DIM];
    size_t off = (size_t)POLICY_O_T_DIM;
    memcpy(x, o_t, sizeof(float) * POLICY_O_T_DIM);
#if POLICY_HAS_ENCODER
    memcpy(x + off, st->z, sizeof(float) * POLICY_Z_DIM);
    off += (size_t)POLICY_Z_DIM;
#endif
    memcpy(x + off, ref_ff, sizeof(float) * POLICY_REF_FF_DIM);

    float m[POLICY_ACT_DIM];
    // Hidden layers: `Linear -> Tanh` once per entry in POLICY_HID_LAYERS, then the action
    // head. The COUNT and the widths come from the generated header, so a 1-hidden-layer
    // policy (net_arch pi=[32]) and a 2-layer one (pi=[128,128]) need no change here -
    // only different exported weights. hid_a/hid_b ping-pong so a layer's input is never
    // overwritten before it has been consumed.
    {
        float *buf[2] = { st->hid_a, st->hid_b };
        const float *cur = x;
        for (int l = 0; l < POLICY_N_HIDDEN; l++) {
            const policy_hidden_layer_t *L = &POLICY_HID_LAYERS[l];
            float *dst = buf[l & 1];
            matvec_tanh(L->w, L->b, cur, L->out_dim, L->in_dim, dst);
            cur = dst;
        }
        matvec((const float *)POLICY_OUT_W, POLICY_OUT_B, cur,
               POLICY_ACT_DIM, POLICY_LATENT_DIM, m);
    }

    for (int i = 0; i < POLICY_ACT_DIM; i++) {
        float v = m[i];
        if (v > 1.0f) {
            v = 1.0f;
        } else if (v < -1.0f) {
            v = -1.0f;
        }
        act_out[i] = v;
    }
}

void policy_step(policy_state_t *st, const float frame[POLICY_ENC_IN_DIM],
                 const float ref_ff[POLICY_REF_FF_DIM], float act_out[POLICY_ACT_DIM])
{
    policy_encoder_step(st, frame);
    policy_actor(st, frame, ref_ff, st->act);
    memcpy(act_out, st->act, sizeof(st->act));
}
