// Deployed policy forward pass backed by an ST Edge AI Core generated network.
//
// SAME CONTRACT as policy_net.h - read that file first: it defines what the frame, the
// latent z and the action mean, and why the action is a CLIPPED MEAN and not a tanh.
// Everything it says about the DATA still holds here; only the code that computes it
// differs.
//
// WHAT THE GENERATED GRAPH DOES
// ----------------------------
//   in : o_t(29) | aux(4) | h_in(48)        raw, un-standardized
//   out: action(4) | h_out(48) | z(16)      action is already clipped to [-1, 1]
//
// Standardization (mean, reciprocal std, clip), the GRU step, the tanh-GELU and the actor
// all live INSIDE the graph: `export_onnx.py --compat` emitted the whole deployed pipeline
// as one primitive-op network precisely so that no arithmetic has to be re-derived on the
// vehicle. One `stai_network_run()` per control step; `h_out -> h_in` carries the
// recurrence.
//
// MEMORY
// ------
// Weights are const and live in flash (network_data.c); the generated code wires them into
// the context itself. The activations (1,408 B) are NOT allocated by the generated code -
// we provide them, and they sit in the state struct, i.e. in BSS, so the stabilizer task
// never pays for them on its stack. The context is opaque: touch it only through the stai
// API.
//
// FAILURE HANDLING
// ----------------
// `errors` counts steps that could not be executed (an init failure means a BUILD problem,
// not a flight one) and `last_error` holds the stai return code that caused it. A failed
// step emits a ZERO action - zero rate setpoints, mid thrust - and the controller's own
// safety envelope still applies. `runs` counts completed inferences. If the vehicle
// behaves as if the policy were dead, those three are the first thing to look at.

#pragma once

#include <stdint.h>

#include "stai.h"
#include "network.h"

// Dims and sim constants. These MUST match the ones export_policy.py wrote from the same
// checkpoint into generated/policy_weights.h (that file is the source of truth; this is a
// copy, and install_stedgeai.py diffs the two rather than trusting it).
#define POLICY_HAS_ENCODER 1
#define POLICY_ACTOR_DIM 48
#define POLICY_O_T_DIM 29
#define POLICY_REF_FF_DIM 3
#define POLICY_Z_DIM 16
#define POLICY_ENC_IN_DIM 33
#define POLICY_ENC_AUX_DIM 4
#define POLICY_GRU_HIDDEN 48
#define POLICY_N_HIDDEN 1
#define POLICY_HIDDEN_MAX 32
#define POLICY_LATENT_DIM 32
#define POLICY_ACT_DIM 4
// Which actor-frame convention the weights were trained under (quad_flip_env.ACTOR_FRAME_MODE).
// controller_app.c #errors on this being absent, and install_stedgeai.py requires EVERY
// POLICY_* constant in policy_weights.h to appear here - a constant that only exists in the
// generated header is invisible to this backend, which is exactly how this one was missed.
#define POLICY_FRAME_ANCHORED_XY 1

#define POLICY_SIM_DT 0.01f
#define POLICY_ACTION_EMA_ALPHA 0.8f
#define POLICY_RATE_SCALE_RP 20.0f
#define POLICY_RATE_SCALE_PITCH 20.0f
#define POLICY_RATE_SCALE_YAW 4.0f
#define POLICY_SIM_MAX_THRUST_N 0.6f
#define POLICY_SIM_MASS_KG 0.028f
#define POLICY_SIM_GRAVITY 9.81f
#define POLICY_SIM_HOVER_TRIM_A0 -0.0844f
#define POLICY_ENC_NORM_CLIP 10.0f

typedef struct {
    // Opaque ST context FIRST so the whole struct inherits its 8-byte alignment.
    STAI_ALIGNED(8) uint8_t ctx[STAI_NETWORK_CONTEXT_SIZE];
    // The activation arena. The generated model allocates nothing: inputs and outputs
    // point INSIDE this buffer (stai was run with allocate-inputs/allocate-outputs).
    STAI_ALIGNED(8) uint8_t activations[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];
    stai_ptr in[STAI_NETWORK_IN_NUM];       // 0 o_t | 1 aux | 2 ref_ff | 3 h_in
    stai_ptr out[STAI_NETWORK_OUT_NUM];     // 0 action | 1 h_out | 2 z

    float h[POLICY_GRU_HIDDEN];             // GRU hidden state (fed back as h_in)
    float z[POLICY_Z_DIM];                  // last latent (kept for logging/inspection)
    float act[POLICY_ACT_DIM];              // last action emitted
    uint8_t ready;                          // context initialised
    uint32_t runs;                          // inferences completed
    uint32_t errors;                        // inferences that failed
    int32_t last_error;                     // stai return code of the last failure
} policy_state_t;

// Zero the recurrent state and re-initialise the runtime context. MUST be called on every
// (re)start/handover - the GRU's hidden state describes one flight, and carrying it across
// a handover or a respawn feeds the actor a summary of a flight that no longer exists.
void policy_reset(policy_state_t *st);

// The whole deployed step: raw frame + feed-forward -> z (in st->z) -> action (in st->act
// and act_out).
//
// `ref_ff` (a_ref + g*e_z, world frame) is a SEPARATE argument, not appended to `frame`: the
// encoder's input contract is exactly [o_t | aux] and is frozen, so the block must reach the
// actor without passing through the GRU.
void policy_step(policy_state_t *st, const float frame[POLICY_ENC_IN_DIM],
                 const float ref_ff[POLICY_REF_FF_DIM], float act_out[POLICY_ACT_DIM]);
