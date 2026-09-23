// Deployed policy forward pass (encoder GRU + PPO actor), portable C.
//
// This file is deliberately free of any Crazyflie headers: the SAME source is compiled
// into the firmware app and into the host-side checker (`test/host_check.c`, driven by
// Simulation/deploy/policy_host_check.py), so the thing verified against torch on the Mac
// is byte-for-byte the thing that runs on the STM32.
//
// WHAT THIS IS
// ------------
//   frame_t (33) = [o_t(29) | aux(4)]      raw, un-standardized
//     -> standardize with the encoder's FROZEN constants (mean/std/clip)
//     -> one GRU step (hidden 48)         -> z (16)
//   actor input (48) = [o_t(29) | z(16) | ref_ff(3)]
//     -> tanh(W_l h + b_l) once per hidden layer (POLICY_N_HIDDEN of them)
//     -> POLICY_OUT_W/POLICY_OUT_B (the action head)
//     -> clip to [-1, 1]                  <- MEASURED: DiagGaussian, squash_output=False
//
// The actor's LAYER COUNT is a property of the exported weights (POLICY_N_HIDDEN and the
// POLICY_HID_LAYERS table in generated/policy_weights.h), not of this file: a single
// hidden layer from `net_arch pi=[32]` and two from `pi=[128,128]` both run through the
// same loop.
//
// `ref_ff` is the reference's specific-force command (a_ref + g*e_z, world frame) and is
// passed SEPARATELY rather than appended to `frame`, because the encoder's input contract is
// exactly [o_t | aux] and is frozen: widening `frame` would feed the GRU three channels it
// was never trained on. It is the actor's only access to the reference's acceleration.
//
// The clip (not a tanh) is what `model.predict(deterministic=True)` returns for this
// checkpoint; the export's verify step compares both and pins this down.
//
// ALL SCRATCH LIVES IN THE STATE STRUCT. The controller runs inside the stabilizer task
// at 1 kHz; nothing here allocates or uses large stack frames.

#pragma once

#include <stdint.h>

#include "generated/policy_weights.h"

// ---------------------------------------------------------------------------------------
// LAYOUT ASSERTIONS. These three widths are written by three different things (the exporter,
// this file, and the observation contract), and NOTHING else cross-checks them. Before these
// asserts existed, a mismatch in POLICY_ACTOR_DIM left the tail of the actor input as
// UNINITIALISED STACK - no error, just a policy fed garbage. Fail the build instead.
// ---------------------------------------------------------------------------------------
#define POLICY_ACTOR_DIM_Z (POLICY_O_T_DIM + POLICY_REF_FF_DIM \
                            + ((POLICY_HAS_ENCODER) ? POLICY_Z_DIM : 0))
_Static_assert(POLICY_ENC_IN_DIM == POLICY_O_T_DIM + POLICY_ENC_AUX_DIM,
               "encoder frame must be [o_t | aux]");
_Static_assert(POLICY_ACTOR_DIM == POLICY_ACTOR_DIM_Z,
               "actor input must be [o_t | z | ref_ff]");

typedef struct {
    // folded into state so every call is self-contained for one vehicle
    float h[POLICY_GRU_HIDDEN];        // GRU hidden state
    float z[POLICY_Z_DIM];             // last latent (kept for logging/inspection)
    float act[POLICY_ACT_DIM];         // last action emitted
    // scratch (kept out of the 1 kHz stack)
    float x_std[POLICY_ENC_IN_DIM];
    float gi[3 * POLICY_GRU_HIDDEN];
    float gh[3 * POLICY_GRU_HIDDEN];
    float gru_r[POLICY_GRU_HIDDEN];
    float gru_z[POLICY_GRU_HIDDEN];
    float gru_n[POLICY_GRU_HIDDEN];
    float head_a[POLICY_GRU_HIDDEN];
    // Ping-pong scratch for the actor's hidden layers. POLICY_HIDDEN_MAX is the widest
    // layer (the exporter emits it), so this covers any count; POLICY_HID_LAYERS decides
    // how many there actually are.
    float hid_a[POLICY_HIDDEN_MAX];
    float hid_b[POLICY_HIDDEN_MAX];
    uint8_t ready;
} policy_state_t;

// Zero the recurrent state and scratch. MUST be called on every (re)start/handover - the
// GRU's hidden state describes one flight and carrying it across a handover or a respawn
// feeds the actor a summary of a flight that no longer exists.
void policy_reset(policy_state_t *st);

// Freeze the input frame (standardization + clip) exactly as the encoder was trained.
void policy_standardize(const float raw[POLICY_ENC_IN_DIM], float out[POLICY_ENC_IN_DIM]);

// One GRU step: standardized frame -> z. Also exposes the hidden state for logging.
void policy_encoder_step(policy_state_t *st, const float frame[POLICY_ENC_IN_DIM]);

// Actor: [o_t(29) | z(16) | ref_ff(3)] -> 4 deterministic action channels in [-1, 1].
// Takes the state (not const): the hidden layers are computed into its scratch buffers so
// the stabilizer task never pays for them on its stack.
void policy_actor(policy_state_t *st, const float o_t[POLICY_O_T_DIM],
                  const float ref_ff[POLICY_REF_FF_DIM], float act_out[POLICY_ACT_DIM]);

// The whole deployed step: raw frame + feed-forward -> z (in st->z) -> action (in st->act
// and act_out).
void policy_step(policy_state_t *st, const float frame[POLICY_ENC_IN_DIM],
                 const float ref_ff[POLICY_REF_FF_DIM], float act_out[POLICY_ACT_DIM]);
