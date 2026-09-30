"""
Frozen GRU history encoder, in JAX.

The encoder itself is TRAINED elsewhere (numpy/torch: ``encoder/collect_data.py`` +
``encoder/train_encoder.py`` against the frozen baseline env) and only RUN here, so this
module is the inference path only.  Its job in the environment is to turn the
``[o_t (29) | aux (4)]`` frame into the 16-dim latent ``z`` that the actor consumes.

Semantics verified against the torch module: single-layer GRU with b_ih on the INPUT part
and b_hh on the HIDDEN part (n = tanh(W_in x + b_in + r * (W_hh h + b_hh))), then
Linear(48->48) -> GELU(erf) -> Linear(48->16) -> tanh, with the frozen
(x - mean)/std standardization clipped to +-10.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .spec import ENCODER_AUX_DIM, ACTOR_TOTAL_DIM, Z_DIM

FRAME_DIM = ACTOR_TOTAL_DIM + ENCODER_AUX_DIM      # 33

DEFAULT_ENCODER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "logs", "encoder_gru.pt")


@struct.dataclass
class GRUWeights:
    w_ih: jax.Array        # (3H, F)
    w_hh: jax.Array        # (3H, H)
    b_ih: jax.Array        # (3H,)
    b_hh: jax.Array        # (3H,)
    h1_w: jax.Array        # (H, H)
    h1_b: jax.Array        # (H,)
    h3_w: jax.Array        # (Z, H)
    h3_b: jax.Array        # (Z,)
    frame_mean: jax.Array  # (F,)
    frame_std: jax.Array   # (F,)
    clip: float


def load(path: str | None = None) -> GRUWeights:
    """Load the checkpoint written by ``encoder/train_encoder.py``."""
    import torch

    if path is None:
        path = DEFAULT_ENCODER
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"no encoder checkpoint at {path}. Train one with "
                f"encoder/collect_data.py + encoder/train_encoder.py, or pass an explicit "
                f"path. The env will train a 32-dim actor without it.")
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["encoder_state"]
    norm = ckpt["norm"]
    return GRUWeights(
        w_ih=jnp.asarray(sd["gru.weight_ih_l0"].numpy(), jnp.float32),
        w_hh=jnp.asarray(sd["gru.weight_hh_l0"].numpy(), jnp.float32),
        b_ih=jnp.asarray(sd["gru.bias_ih_l0"].numpy(), jnp.float32),
        b_hh=jnp.asarray(sd["gru.bias_hh_l0"].numpy(), jnp.float32),
        h1_w=jnp.asarray(sd["head.0.weight"].numpy(), jnp.float32),
        h1_b=jnp.asarray(sd["head.0.bias"].numpy(), jnp.float32),
        h3_w=jnp.asarray(sd["head.2.weight"].numpy(), jnp.float32),
        h3_b=jnp.asarray(sd["head.2.bias"].numpy(), jnp.float32),
        frame_mean=jnp.asarray(norm["frame_mean"], jnp.float32),
        frame_std=jnp.asarray(norm["frame_std"], jnp.float32),
        clip=float(norm.get("clip", 10.0)),
    )


def head_hidden(w: GRUWeights) -> int:
    return int(w.h3_w.shape[1])


def step(w: GRUWeights, h, frame_raw):
    """One causal recurrent step: [frame 33, h 48] -> [z 16, h_next 48]."""
    frame = (frame_raw - w.frame_mean) / jnp.maximum(w.frame_std, 1e-6)
    if w.clip > 0:
        frame = jnp.clip(frame, -w.clip, w.clip)

    H = head_hidden(w)
    x_r = w.w_ih[0:H] @ frame + w.b_ih[0:H]
    x_z = w.w_ih[H:2 * H] @ frame + w.b_ih[H:2 * H]
    x_n = w.w_ih[2 * H:3 * H] @ frame + w.b_ih[2 * H:3 * H]
    h_r = w.w_hh[0:H] @ h + w.b_hh[0:H]
    h_z = w.w_hh[H:2 * H] @ h + w.b_hh[H:2 * H]
    h_n = w.w_hh[2 * H:3 * H] @ h + w.b_hh[2 * H:3 * H]

    r = jax.nn.sigmoid(x_r + h_r)
    u = jax.nn.sigmoid(x_z + h_z)
    n = jnp.tanh(x_n + r * h_n)
    h_next = (1.0 - u) * n + u * h

    a = jax.nn.gelu(w.h1_w @ h_next + w.h1_b, approximate=False)
    z = jnp.tanh(w.h3_w @ a + w.h3_b)
    return z, h_next


def zeros_state(w: GRUWeights):
    return jnp.zeros(head_hidden(w)), jnp.zeros(Z_DIM)
