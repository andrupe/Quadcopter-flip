// ST Edge AI Core backend for the deployed policy. See policy_net_stedgeai.h.
//
// The contract (frame layout, the meaning of every number) is policy_net.h's; this file is
// the plumbing that feeds the generated graph and carries its recurrence forward. There is
// deliberately NO arithmetic here at all - if a number looks wrong, it is a wrong export,
// not a wrong line of this file.

#include "policy_net_stedgeai.h"

#include <string.h>

// The tensor contract is checked at COMPILE TIME, not trusted: a regenerated network with
// different tensor sizes would otherwise "work" while reading garbage.
#if (STAI_NETWORK_IN_NUM != 4) || (STAI_NETWORK_OUT_NUM != 3)
#error "ST network must have exactly 4 inputs (o_t, aux, ref_ff, h_in) and 3 outputs (action, h_out, z)"
#endif
#if (STAI_NETWORK_IN_1_SIZE != POLICY_O_T_DIM) || (STAI_NETWORK_IN_2_SIZE != POLICY_ENC_AUX_DIM) \
 || (STAI_NETWORK_IN_3_SIZE != POLICY_REF_FF_DIM) || (STAI_NETWORK_IN_4_SIZE != POLICY_GRU_HIDDEN)
#error "ST network input sizes do not match [o_t(29) | aux(4) | ref_ff(3) | h_in(48)]"
#endif
#if (STAI_NETWORK_OUT_1_SIZE != POLICY_ACT_DIM) || (STAI_NETWORK_OUT_2_SIZE != POLICY_GRU_HIDDEN) \
 || (STAI_NETWORK_OUT_3_SIZE != POLICY_Z_DIM)
#error "ST network output sizes do not match the contract [action(4) | h_out(48) | z(16)]"
#endif
// The actor width the graph was built for must equal [o_t | z | ref_ff]; the graph forms
// that concatenation internally, so a mismatch here is a silent reorder, not a crash.
#define POLICY_ACTOR_DIM_Z (POLICY_O_T_DIM + POLICY_REF_FF_DIM \
                            + ((POLICY_HAS_ENCODER) ? POLICY_Z_DIM : 0))
_Static_assert(POLICY_ACTOR_DIM == POLICY_ACTOR_DIM_Z,
               "actor input must be [o_t | z | ref_ff]");

#define IN_F(st, i)  ((float *)(void *)((st)->in[i]))
#define OUT_F(st, i) ((float *)(void *)((st)->out[i]))

static void action_zero(float act_out[POLICY_ACT_DIM])
{
    for (int i = 0; i < POLICY_ACT_DIM; i++) {
        act_out[i] = 0.0f;
    }
}

// Wire the context. Nothing is ever allocated here: the activations come from the state
// struct (BSS) and the weights are the const flash array the generated code already put in
// the context (STAI_FLAG_PREALLOCATED). Returns a negative step number on failure so a
// debugger says WHERE it failed, not just that it did.
static int32_t backend_init(policy_state_t *st)
{
    stai_network *net = (stai_network *)st->ctx;
    stai_ptr acts[STAI_NETWORK_ACTIVATIONS_NUM];
    stai_size n = 0;

    if (stai_network_init(net) != STAI_SUCCESS) {
        return -1;
    }
    acts[0] = (stai_ptr)st->activations;
    if (stai_network_set_activations(net, acts, STAI_NETWORK_ACTIVATIONS_NUM) != STAI_SUCCESS) {
        return -2;
    }
    if (stai_network_get_inputs(net, st->in, &n) != STAI_SUCCESS) {
        return -3;
    }
    if (n != STAI_NETWORK_IN_NUM) {
        return -4;
    }
    if (stai_network_get_outputs(net, st->out, &n) != STAI_SUCCESS) {
        return -5;
    }
    if (n != STAI_NETWORK_OUT_NUM) {
        return -6;
    }

    st->ready = 1u;
    return 0;
}

void policy_reset(policy_state_t *st)
{
    // A reset is a NEW FLIGHT to the GRU. `ready = 0` re-runs backend_init on the next
    // step, which re-takes the tensor pointers from the context - cheap, and it keeps a
    // reset honest even if a future build moves the arena.
    memset(st->h, 0, sizeof(st->h));
    memset(st->z, 0, sizeof(st->z));
    memset(st->act, 0, sizeof(st->act));
    st->ready = 0u;
    st->runs = 0u;
    st->errors = 0u;
    st->last_error = 0;
}

void policy_step(policy_state_t *st, const float frame[POLICY_ENC_IN_DIM],
                 const float ref_ff[POLICY_REF_FF_DIM], float act_out[POLICY_ACT_DIM])
{
    if (!st->ready) {
        int32_t rc = backend_init(st);
        if (rc != 0) {
            st->errors++;
            st->last_error = rc;
            action_zero(act_out);
            return;
        }
    }

    // frame = [o_t(29) | aux(4)]. The graph takes them as two tensors - which is exactly
    // the split the `--compat` export performs to avoid a Slice - so the split happens
    // here, on the way in, instead of inside the network. ref_ff is its own tensor because
    // it must reach the ACTOR without passing through the GRU.
    memcpy(IN_F(st, 0), frame, POLICY_O_T_DIM * sizeof(float));
    memcpy(IN_F(st, 1), frame + POLICY_O_T_DIM, POLICY_ENC_AUX_DIM * sizeof(float));
    memcpy(IN_F(st, 2), ref_ff, POLICY_REF_FF_DIM * sizeof(float));
    memcpy(IN_F(st, 3), st->h, POLICY_GRU_HIDDEN * sizeof(float));

    stai_return_code rc = stai_network_run((stai_network *)st->ctx, STAI_MODE_SYNC);
    if (rc != STAI_SUCCESS) {
        st->errors++;
        st->last_error = (int32_t)rc;
        action_zero(act_out);
        return;                      // h is left as it was: the next step tries again
    }

    memcpy(act_out, OUT_F(st, 0), POLICY_ACT_DIM * sizeof(float));
    memcpy(st->h, OUT_F(st, 1), POLICY_GRU_HIDDEN * sizeof(float));   // h_out -> h_in
    memcpy(st->z, OUT_F(st, 2), POLICY_Z_DIM * sizeof(float));
    memcpy(st->act, act_out, POLICY_ACT_DIM * sizeof(float));
    st->runs++;
}
