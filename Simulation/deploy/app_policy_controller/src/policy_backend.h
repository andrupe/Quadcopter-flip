// Which policy implementation backs the out-of-tree controller.
//
// ONE API, TWO BACKENDS - `controller_app.c` is compiled against whichever is selected
// and cannot tell the difference:
//
//   legacy   (default) : policy_net.c          hand-written float forward pass. Weights in
//                                              generated/policy_weights.c. Every step of it
//                                              (standardization, GRU semantics, the clipped
//                                              mean, the tanh-GELU) was diffed against torch
//                                              over real frames.
//   stedgeai           : policy_net_stedgeai.c One generated graph from ST Edge AI Core
//                                              (generated/stedgeai/) plus the ST runtime
//                                              archive. Same contract, different code.
//
// Selected with the APP_BACKEND make variable (`build_app.sh --stedgeai`); the C sees
// -DAPP_BACKEND_STEDGEAI=1 only for the ST backend, so a stale build directory cannot
// silently mix a backend with the other one's sources.
//
// THE INTERFACE (both backends, identical):
//
//     void policy_reset(policy_state_t *st);
//     void policy_step(policy_state_t *st,
//                      const float frame[POLICY_ENC_IN_DIM],      // [o_t(29) | aux(4)]
//                      const float ref_ff[POLICY_REF_FF_DIM],     // a_ref + g*e_z, world
//                      float act_out[POLICY_ACT_DIM]);
//
// `ref_ff` is a SEPARATE argument rather than being appended to `frame` because the encoder's
// input contract is exactly [o_t | aux] and is frozen - appending the block would feed the
// GRU three channels it was never trained on. The actor input is formed as
// [o_t | z | ref_ff] inside each backend (or inside the generated graph, for ST).

#pragma once

#if defined(APP_BACKEND_STEDGEAI) && (APP_BACKEND_STEDGEAI == 1)
#include "policy_net_stedgeai.h"
#else
#include "policy_net.h"
#endif
