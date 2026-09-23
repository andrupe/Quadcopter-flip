# -*- coding: utf-8 -*-
"""
Export the trained policy (encoder GRU + PPO actor) to C arrays for the Crazyflie firmware.

WHY THIS EXISTS
---------------
The deployed actor is fed `[o_t(29) | z(16) | ref_ff(3)]` at 100 Hz, where z comes from the
frozen GRU encoder over `[o_t | aux]` frames (see `encoder/observation_spec.py`) and ref_ff
is the reference's specific-force command (a_ref + g*e_z, world frame) that the app builds
from the baked acceleration column of `generated/reference_tables.h`. On the Crazyflie
that forward pass must run in C, so this script is the ONLY translation step from the
checkpoint to the firmware. Everything it emits is derived from the same loaders the sim
uses (`actor_input.load_checkpoint`, `history_encoder.load_encoder_checkpoint`), so the
export cannot drift from the deployment contract silently.

NOTE the actor prefix is `[o_t | z | ref_ff]` and the ENCODER's input is `[o_t | aux]`.
ref_ff sits between the two on board for exactly the reason it does in the env vector: the
encoder must not see it (its contract is frozen at 33 dims) and the actor must.

WHAT IS EXPORTED (and what is not)
----------------------------------
    actor  : policy_net (Linear/Tanh) + action_net, deterministic action = tanh(mean)
    encoder: GRU (r,z,n gate order, PyTorch layout) + head Linear/GELU/Linear/Tanh
             + the FROZEN input standardization (mean/std/clip) stored beside the weights
    meta   : dims and the sim constants the firmware needs to speak the same units

The critic (value_net) is NOT exported: it is training-only. The log_std parameter is
NOT exported either: deployment is deterministic.

GRU GATE ORDER. torch's `nn.GRU` packs the input/hidden weight rows as [reset | update |
new] (r, z, n). `policy_net.c` must read the rows in that same order - the comment in the
emitted header says so, and `--verify` would fail if it were wrong.

USAGE
-----
    .venv/bin/python Simulation/deploy/export_policy.py            # latest checkpoint
    .venv/bin/python Simulation/deploy/export_policy.py --model logs/rl_model_15000000_steps.zip
    .venv/bin/python Simulation/deploy/export_policy.py --no-verify

`--verify` (default ON) runs the numpy reconstruction of the WHOLE deployed forward pass
(standardize -> GRU step -> z -> actor MLP -> tanh) against the torch deployment path
(`encoder.step` + `model.predict`) over a real frame stream from `logs/encoder_data/`
when one is available, and reports max |delta| for z and for the action. That check
settles two things no amount of code reading can: the gate order and the tanh squash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR)) if os.path.basename(_THIS_DIR) == "deploy" else _THIS_DIR
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR, os.path.join(_SIM_DIR, "encoder")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from actor_input import Z_DIM, load_checkpoint, read_checkpoint_arch  # noqa: E402
from encoder.history_encoder import load_encoder_checkpoint  # noqa: E402
from encoder.observation_spec import ACTOR_FRAME_DIM, AUX_DIM, ENCODER_IN_DIM  # noqa: E402
from quad_flip_env import (  # noqa: E402
    ACTION_EMA_ALPHA,
    ACTOR_FRAME_MODE,
    ACTOR_TOTAL_DIM,
    ANCHOR_ACTOR_XY,
    ENCODER_AUX_DIM,
    GRAVITY,
    REF_FF_DIM,
    SIM_DT,
)

DEFAULT_OUT_DIR = os.path.join(_SIM_DIR, "deploy", "app_policy_controller", "src", "generated")
DEFAULT_MANIFEST = os.path.join(_SIM_DIR, "deploy", "manifests", "policy_export.json")
ENCODER_DATA_DIR = os.path.join(_PROJECT_ROOT, "logs", "encoder_data")

# Verification tolerances. float32 forward passes accumulate a few ulp of difference
# between torch kernels and a hand-rolled numpy path; the deployment budget is far tighter
# than either (the actor's action resolution on the wire is ~1146 dps for 2.0 of range).
Z_TOL = 2.0e-5
ACTION_TOL = 1.0e-5


# ======================================================================================
# checkpoint discovery
# ======================================================================================
def resolve_model(name: str) -> str:
    """`latest`/`auto` = newest rl_model_*.zip in logs/, else a path relative to the root."""
    if os.path.isfile(name):
        return os.path.abspath(name)
    candidate = os.path.join(_PROJECT_ROOT, name)
    if os.path.isfile(candidate):
        return candidate
    if os.path.isfile(candidate + ".zip"):
        return candidate + ".zip"
    logs = os.path.join(_PROJECT_ROOT, "logs")
    if name.lower() in ("latest", "auto") and os.path.isdir(logs):
        zips = [os.path.join(logs, f) for f in os.listdir(logs)
                if f.endswith(".zip") and f.startswith("rl_model_")]
        if zips:
            return max(zips, key=os.path.getmtime)
    raise FileNotFoundError(f"no checkpoint found for '{name}'")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ======================================================================================
# extraction
# ======================================================================================
def _find(sd: Dict[str, Any], suffix: str):
    """State-dict lookup that tolerates wrapper prefixes (e.g. `_orig_mod.`)."""
    if suffix in sd:
        return sd[suffix]
    hits = [k for k in sd if k.endswith("." + suffix)]
    if len(hits) != 1:
        raise KeyError(f"expected exactly one state-dict key ending in '{suffix}', found {hits}")
    return sd[hits[0]]


def _np(t) -> np.ndarray:
    return np.asarray(t.detach().cpu().numpy(), dtype=np.float32)


def extract_actor(model_path: str) -> Dict[str, Any]:
    """
    policy_net + action_net weights, flattened to the exact C layout.

    The actor MLP is `Linear -> Tanh` repeated once per hidden layer, then the action head.
    The NUMBER of hidden layers is READ from the checkpoint, never assumed: net_arch
    `{"pi": [32]}` gives one and `{"pi": [128, 128]}` gives two - the emitted C walks
    whatever arrives here (see `POLICY_HID_LAYERS` in the generated header).
    """
    model = load_checkpoint(model_path)
    sd = model.policy.state_dict()

    hidden: List[Tuple[np.ndarray, np.ndarray]] = []
    idx = 0
    # The extractor's policy_net is [Linear, activation, Linear, activation, ...], so the
    # weight tensors live at even indices; iterate until one is missing.
    while f"mlp_extractor.policy_net.{idx}.weight" in sd:
        hidden.append((_np(sd[f"mlp_extractor.policy_net.{idx}.weight"]),
                       _np(sd[f"mlp_extractor.policy_net.{idx}.bias"])))
        idx += 2
    if not hidden:
        raise KeyError("no mlp_extractor.policy_net.* weights in the checkpoint")

    w_out = _np(_find(sd, "action_net.weight"))
    b_out = _np(_find(sd, "action_net.bias"))
    log_std = _find(sd, "log_std")
    actor_dim = int(getattr(model.policy, "actor_obs_dim", hidden[0][0].shape[1]))
    return {
        "hidden": hidden, "w_out": w_out, "b_out": b_out,
        "log_std": _np(log_std), "actor_dim": actor_dim,
        "dist": type(model.policy.action_dist).__name__,
        "squash_output": bool(getattr(model.policy.action_dist, "squash_output", False)),
    }


def extract_encoder(encoder_path: str) -> Dict[str, Any]:
    """GRU + head weights and the frozen standardization, from the canonical loader."""
    encoder, norm, ckpt = load_encoder_checkpoint(encoder_path)
    sd = encoder.state_dict()
    return {
        "w_ih": _np(_find(sd, "gru.weight_ih_l0")),      # [3H, F] rows: reset, update, new
        "w_hh": _np(_find(sd, "gru.weight_hh_l0")),      # [3H, H]
        "b_ih": _np(_find(sd, "gru.bias_ih_l0")),
        "b_hh": _np(_find(sd, "gru.bias_hh_l0")),
        "head_w1": _np(_find(sd, "head.0.weight")),
        "head_b1": _np(_find(sd, "head.0.bias")),
        "head_w2": _np(_find(sd, "head.2.weight")),
        "head_b2": _np(_find(sd, "head.2.bias")),
        "f_in": int(encoder.f_in),
        "hidden": int(encoder.width),
        "z_dim": int(encoder.z_dim),
        "norm_mean": np.asarray(norm.frame_mean, dtype=np.float32),
        "norm_std": np.asarray(norm.frame_std, dtype=np.float32),
        "norm_clip": float(norm.clip),
        "ckpt_config": ckpt.get("config", {}),
    }


def extract_env_constants() -> Dict[str, float]:
    """The sim constants the firmware needs so its units match the trained task."""
    from quad_flip_env import QuadFlipEnv

    env = QuadFlipEnv(telemetry=False)
    q = env.quad.params
    return {
        "sim_dt": float(SIM_DT),
        "action_ema_alpha": float(ACTION_EMA_ALPHA),
        "actor_total_dim": int(ACTOR_TOTAL_DIM),
        "encoder_aux_dim": int(ENCODER_AUX_DIM),
        "max_rate_xy": float(env.max_rate_xy),
        "max_rate_pitch": float(env.max_rate_pitch),
        "max_rate_z": float(env.max_rate_z),
        "max_thrust_n": float(q["maxThr"]),
        "mass_kg": float(q["mB"]),
        "gravity": float(q["g"]),
        "hover_trim_a0": float(2.0 * q["mB"] * q["g"] / q["maxThr"] - 1.0),
    }


# ======================================================================================
# numpy reconstruction of the deployed forward pass
# ======================================================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def standardize_frame(raw: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float) -> np.ndarray:
    x = (np.asarray(raw, dtype=np.float32) - mean) / std
    if clip > 0:
        x = np.clip(x, -clip, clip)
    return x.astype(np.float32)


def gru_step(enc: Dict[str, Any], x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """One PyTorch-semantics GRU step. Rows of w_ih/w_hh are [reset | update | new]."""
    H = enc["hidden"]
    gi = enc["w_ih"] @ x + enc["b_ih"]
    gh = enc["w_hh"] @ h + enc["b_hh"]
    r = sigmoid(gi[:H] + gh[:H])
    z = sigmoid(gi[H:2 * H] + gh[H:2 * H])
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    return ((1.0 - z) * n + z * h).astype(np.float32)


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact erf-based GELU, matching `nn.GELU()` with approximate='none'."""
    from scipy.special import erf

    x64 = np.asarray(x, dtype=np.float64)
    return np.asarray(0.5 * x64 * (1.0 + erf(x64 / np.sqrt(2.0))), dtype=np.float32)


def encoder_head(enc: Dict[str, Any], h: np.ndarray) -> np.ndarray:
    a = gelu(enc["head_w1"] @ h + enc["head_b1"])
    return np.tanh(enc["head_w2"] @ a + enc["head_b2"]).astype(np.float32)


def actor_forward(act: Dict[str, Any], actor_in: np.ndarray) -> np.ndarray:
    """
    The deterministic deployed action.

    MEASURED, not assumed: this checkpoint's `action_dist` is a plain
    `DiagGaussianDistribution` with `squash_output=False`, so `model.predict(...
    deterministic=True)` returns the Gaussian MEAN, clipped to the action space
    (SB3 clips in `predict`; training samples around the mean and clips the same way).
    A tanh squash here is off by up to 0.24 of full range and is what `--verify`
    caught on the first export.

    The hidden-layer COUNT is whatever the checkpoint carries (one layer for pi=[32]), so
    this loop - not a fixed W1/W2/W3 chain - is the reference the C is diffed against.
    """
    x = actor_in
    for w, b in act["hidden"]:
        x = np.tanh(w @ x + b)
    mean = act["w_out"] @ x + act["b_out"]
    return np.clip(mean, -1.0, 1.0).astype(np.float32)


# ======================================================================================
# verification against the torch deployment path
# ======================================================================================
def _frame_stream(limit: int) -> Tuple[np.ndarray, str]:
    """
    A stream of raw 33-dim encoder frames.

    Prefers a REAL shard from the encoder corpus (the only source that exercises the
    standardization constants exactly as collected); falls back to a bounded random walk.
    """
    if os.path.isdir(ENCODER_DATA_DIR):
        shards = sorted(f for f in os.listdir(ENCODER_DATA_DIR) if f.endswith(".npz"))
        if shards:
            path = os.path.join(ENCODER_DATA_DIR, shards[0])
            with np.load(path) as d:
                for key in ("frames", "frame", "obs", "x"):
                    if key in d and d[key].ndim == 2 and d[key].shape[-1] == ENCODER_IN_DIM:
                        frames = np.asarray(d[key][:limit], dtype=np.float32)
                        return frames, f"{shards[0]}:{key}"
                for key in d.files:
                    arr = d[key]
                    if arr.ndim == 2 and arr.shape[-1] == ENCODER_IN_DIM:
                        frames = np.asarray(arr[:limit], dtype=np.float32)
                        return frames, f"{shards[0]}:{key}"
    rng = np.random.default_rng(1234)
    walk = np.cumsum(rng.normal(0.0, 0.05, size=(limit, ENCODER_IN_DIM)), axis=0)
    return walk.astype(np.float32), "synthetic random walk"


def ref_ff_stream(limit: int, seed: int = 0) -> np.ndarray:
    """
    A stream of reference feed-forward blocks: `a_ref + g*e_z`, world frame, m/s^2.

    SYNTHETIC, DELIBERATELY. The encoder corpus stores `[o_t | aux]` frames and the
    privileged targets, not the reference, so there is no real block to replay with them.
    Synthesising one is not a compromise here: the host checks are numerical equivalence
    tests, and a block that is DECORRELATED from the frames is strictly better at catching
    a channel-order or offset error than a realistic one would be (a correlated input can
    produce a plausible action even when it is wired to the wrong channel).

    The magnitudes are real: the block sits at ~1 g in a hover, peaks near 2 g during a
    flip's pop, and reaches EXACTLY zero through a ballistic coast - all three regimes are
    present here, and each channel has its own frequency and phase so the three cannot be
    confused for one another.
    """
    t = np.arange(int(limit), dtype=np.float64) * SIM_DT
    a = np.stack([
        2.5 * np.sin(2.0 * np.pi * 0.37 * t + 0.0),
        1.7 * np.sin(2.0 * np.pi * 0.61 * t + 1.1),
        3.1 * np.sin(2.0 * np.pi * 0.23 * t + 2.4),
    ], axis=1)
    ff = a.astype(np.float32)
    ff[:, 2] += np.float32(GRAVITY)
    # Ballistic-coast windows: the block is exactly zero, the single most diagnostic value.
    ff[np.sin(2.0 * np.pi * 0.13 * t + 0.7) > 0.93] = 0.0
    return ff


def verify(act: Dict[str, np.ndarray], enc: Dict[str, Any], model_path: str,
           encoder_path: str, limit: int = 3000) -> Dict[str, Any]:
    """numpy deployed path vs torch deployed path (`encoder.step` + `model.predict`)."""
    import torch

    model = load_checkpoint(model_path)
    encoder, _norm, _ckpt = load_encoder_checkpoint(encoder_path)

    frames, source = _frame_stream(limit)
    n = len(frames)
    ff = ref_ff_stream(n)

    h_np = np.zeros(enc["hidden"], dtype=np.float32)
    h_t = encoder.init_state(1)
    max_z_err = 0.0
    max_a_err = 0.0
    max_a_err_torch_z = 0.0
    bad_z = -1
    bad_a = -1

    with torch.no_grad():
        for i in range(n):
            raw = frames[i]
            ref_ff = ff[i]
            x_std = standardize_frame(raw, enc["norm_mean"], enc["norm_std"], enc["norm_clip"])

            # -- numpy deployed path ------------------------------------------------
            h_np = gru_step(enc, x_std, h_np)
            z_np = encoder_head(enc, h_np)
            actor_in_np = np.concatenate([raw[:ACTOR_FRAME_DIM], z_np,
                                          ref_ff]).astype(np.float32)
            a_np = actor_forward(act, actor_in_np)

            # -- torch deployed path ------------------------------------------------
            x_t = torch.from_numpy(x_std[None, :])
            z_t, h_t = encoder.step(x_t, h_t)
            actor_in_t = np.concatenate([raw[:ACTOR_FRAME_DIM],
                                         z_t.numpy()[0], ref_ff]).astype(np.float32)
            a_t, _ = model.predict(actor_in_t, deterministic=True)
            a_t = np.asarray(a_t, dtype=np.float32).reshape(-1)

            # also compare the action with the TORCH z, isolating the MLP from the GRU
            a_from_torch_z = actor_forward(act, actor_in_t)

            z_err = float(np.max(np.abs(z_np - z_t.numpy()[0])))
            if z_err > max_z_err:
                max_z_err, bad_z = z_err, i
            a_err = float(np.max(np.abs(a_np - a_t)))
            if a_err > max_a_err:
                max_a_err, bad_a = a_err, i
            max_a_err_torch_z = max(max_a_err_torch_z,
                                    float(np.max(np.abs(a_from_torch_z - a_t))))

    return {
        "ok": bool(max_z_err <= Z_TOL and max_a_err <= ACTION_TOL),
        "frames": int(n),
        "source": source,
        "z_tol": Z_TOL,
        "action_tol": ACTION_TOL,
        "max_z_err": max_z_err,
        "max_z_err_step": bad_z,
        "max_action_err": max_a_err,
        "max_action_err_step": bad_a,
        "max_action_err_torch_z": max_a_err_torch_z,
        "dist": act["dist"],
        "squash_output": act["squash_output"],
    }


# ======================================================================================
# C emission
# ======================================================================================
def _fmt(v: Any) -> str:
    """C float literal that round-trips a float32 exactly (9 significant digits)."""
    f = float(v)
    if not np.isfinite(f):
        raise ValueError(f"non-finite value in export: {f}")
    s = f"{f:.9g}"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def _matrix(name: str, arr: np.ndarray) -> str:
    rows = ["{" + ", ".join(_fmt(v) for v in row) + "}" for row in arr]
    return f"const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n    " + ",\n    ".join(rows) + "\n};\n"


def _vector(name: str, arr: np.ndarray) -> str:
    return f"const float {name}[{arr.size}] = {{\n    " + ",\n    ".join(_fmt(v) for v in arr) + "\n};\n"


def emit_c(out_dir: str, act: Dict[str, np.ndarray], enc: Dict[str, Any],
           consts: Dict[str, float], verify_result: Optional[Dict[str, Any]]) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    header_path = os.path.join(out_dir, "policy_weights.h")
    source_path = os.path.join(out_dir, "policy_weights.c")
    has_encoder = act["actor_dim"] in (ACTOR_TOTAL_DIM + Z_DIM, ACTOR_TOTAL_DIM + Z_DIM + REF_FF_DIM)
    n_hidden = len(act["hidden"])
    hidden_max = max(int(w.shape[0]) for w, _b in act["hidden"])
    latent_dim = int(act["hidden"][-1][0].shape[0])

    hdr = [
        "// GENERATED by Simulation/deploy/export_policy.py - DO NOT EDIT BY HAND.",
        f"// source checkpoint : {os.path.basename(consts['_model_name'])}",
        f"// source encoder    : {os.path.basename(consts['_encoder_name'])}",
        f"// generated         : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "//",
        "// Actor:   tanh(W_n ... tanh(W_1 [o_t|z|ref_ff] + b_1) ...) -> action head -> clip",
        "// Encoder: one GRU step per control step. torch's `nn.GRU` stores its rows as",
        "//          [reset | update | new] = [r | z | n]; GRU_W_IH/GRU_W_HH keep that",
        "//          order and policy_net.c must read them in it.",
        "//          h' = (1 - z) * n + z * h ;  n = tanh(W_in x + b_in + r * (W_hn h + b_hn))",
        "// Normalization is part of the encoder contract: x = clip((raw - mean) / std, +-clip).",
        "#pragma once",
        "",
        "#define POLICY_HAS_ENCODER %d" % (1 if has_encoder else 0),
        "#define POLICY_ACTOR_DIM %d" % int(act["actor_dim"]),
        "#define POLICY_O_T_DIM %d" % ACTOR_FRAME_DIM,
        "#define POLICY_REF_FF_DIM %d" % REF_FF_DIM,
        # WHICH ACTOR-FRAME CONVENTION the policy was trained against. Baked in from
        # quad_flip_env.ACTOR_FRAME_MODE rather than hand-maintained, so the firmware can
        # never disagree with the weights it is carrying: controller_app.c `#error`s on a
        # missing/other value instead of feeding the net a frame it never saw.
        "#define POLICY_FRAME_ANCHORED_XY %d  // quad_flip_env.ACTOR_FRAME_MODE = \"%s\""
        % (1 if ANCHOR_ACTOR_XY else 0, ACTOR_FRAME_MODE),
        "#define POLICY_Z_DIM %d" % int(enc["z_dim"]),
        "#define POLICY_ENC_IN_DIM %d" % int(enc["f_in"]),
        "#define POLICY_ENC_AUX_DIM %d" % AUX_DIM,
        "#define POLICY_GRU_HIDDEN %d" % int(enc["hidden"]),
        "#define POLICY_N_HIDDEN %d" % n_hidden,
        "#define POLICY_HIDDEN_MAX %d" % hidden_max,
        "#define POLICY_LATENT_DIM %d" % latent_dim,
        "#define POLICY_ACT_DIM %d" % int(act["w_out"].shape[0]),
        "",
        "// sim constants the firmware must mirror (recorded at export time)",
        "#define POLICY_SIM_DT " + _fmt(consts["sim_dt"]),
        "#define POLICY_ACTION_EMA_ALPHA " + _fmt(consts["action_ema_alpha"]),
        "#define POLICY_RATE_SCALE_RP " + _fmt(consts["max_rate_xy"]),
        "#define POLICY_RATE_SCALE_PITCH " + _fmt(consts["max_rate_pitch"]),
        "#define POLICY_RATE_SCALE_YAW " + _fmt(consts["max_rate_z"]),
        "#define POLICY_SIM_MAX_THRUST_N " + _fmt(consts["max_thrust_n"]),
        "#define POLICY_SIM_MASS_KG " + _fmt(consts["mass_kg"]),
        "#define POLICY_SIM_GRAVITY " + _fmt(consts["gravity"]),
        "#define POLICY_SIM_HOVER_TRIM_A0 " + _fmt(consts["hover_trim_a0"]),
        "#define POLICY_ENC_NORM_CLIP " + _fmt(enc["norm_clip"]),
        "",
    ]
    # Actor hidden layers: ONE DESCRIPTOR PER `Linear -> Tanh`, declared with their own
    # literal dims and walked by a single loop in policy_net.c. That is what makes the
    # layer COUNT a property of the exported weights rather than of the C: a 1-hidden-layer
    # policy (net_arch pi=[32]) and a 2-layer one (pi=[128,128]) both build unchanged.
    # POLICY_HIDDEN_MAX sizes the state struct's ping-pong scratch.
    hdr += ["typedef struct { const float *w; const float *b; int out_dim; int in_dim; } "
            "policy_hidden_layer_t;"]
    for _l, (_w, _b) in enumerate(act["hidden"]):
        hdr += [f"extern const float POLICY_HID_W_{_l}[{int(_w.shape[0])}][{int(_w.shape[1])}];",
                f"extern const float POLICY_HID_B_{_l}[{int(_w.shape[0])}];"]
    hdr += ["extern const policy_hidden_layer_t POLICY_HID_LAYERS[POLICY_N_HIDDEN];",
            "extern const float POLICY_OUT_W[POLICY_ACT_DIM][POLICY_LATENT_DIM];",
            "extern const float POLICY_OUT_B[POLICY_ACT_DIM];",
            ""]
    if has_encoder:
        hdr += [
            "extern const float GRU_W_IH[3 * POLICY_GRU_HIDDEN][POLICY_ENC_IN_DIM];",
            "extern const float GRU_W_HH[3 * POLICY_GRU_HIDDEN][POLICY_GRU_HIDDEN];",
            "extern const float GRU_B_IH[3 * POLICY_GRU_HIDDEN];",
            "extern const float GRU_B_HH[3 * POLICY_GRU_HIDDEN];",
            "extern const float ENC_HEAD_W1[POLICY_GRU_HIDDEN][POLICY_GRU_HIDDEN];",
            "extern const float ENC_HEAD_B1[POLICY_GRU_HIDDEN];",
            "extern const float ENC_HEAD_W2[POLICY_Z_DIM][POLICY_GRU_HIDDEN];",
            "extern const float ENC_HEAD_B2[POLICY_Z_DIM];",
            "extern const float ENC_NORM_MEAN[POLICY_ENC_IN_DIM];",
            "extern const float ENC_NORM_STD[POLICY_ENC_IN_DIM];",
            "",
        ]
    with open(header_path, "w") as fh:
        fh.write("\n".join(hdr))

    src = ["// GENERATED by Simulation/deploy/export_policy.py - DO NOT EDIT BY HAND.",
           '#include "policy_weights.h"', ""]
    for _l, (_w, _b) in enumerate(act["hidden"]):
        src += [_matrix(f"POLICY_HID_W_{_l}", _w), _vector(f"POLICY_HID_B_{_l}", _b)]
    _layers = ",\n    ".join(
        f'{{ (const float *)POLICY_HID_W_{_l}, POLICY_HID_B_{_l}, '
        f'{int(_w.shape[0])}, {int(_w.shape[1])} }}'
        for _l, (_w, _b) in enumerate(act["hidden"]))
    src += [f"const policy_hidden_layer_t POLICY_HID_LAYERS[POLICY_N_HIDDEN] = {{\n    "
            f"{_layers}\n}};\n"]
    src += [_matrix("POLICY_OUT_W", act["w_out"]), _vector("POLICY_OUT_B", act["b_out"])]
    if has_encoder:
        src += [_matrix("GRU_W_IH", enc["w_ih"]), _matrix("GRU_W_HH", enc["w_hh"]),
                _vector("GRU_B_IH", enc["b_ih"]), _vector("GRU_B_HH", enc["b_hh"]),
                _matrix("ENC_HEAD_W1", enc["head_w1"]), _vector("ENC_HEAD_B1", enc["head_b1"]),
                _matrix("ENC_HEAD_W2", enc["head_w2"]), _vector("ENC_HEAD_B2", enc["head_b2"]),
                _vector("ENC_NORM_MEAN", enc["norm_mean"]), _vector("ENC_NORM_STD", enc["norm_std"])]
    with open(source_path, "w") as fh:
        fh.write("\n".join(src))

    digest = hashlib.sha256()
    for _w, _b in act["hidden"]:
        digest.update(np.ascontiguousarray(_w, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(_b, dtype=np.float32).tobytes())
    for arr in (act["w_out"], act["b_out"]):
        digest.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    if has_encoder:
        for key in ("w_ih", "w_hh", "b_ih", "b_hh", "head_w1", "head_b1", "head_w2",
                    "head_b2", "norm_mean", "norm_std"):
            digest.update(np.ascontiguousarray(enc[key], dtype=np.float32).tobytes())
    out = {"header": header_path, "source": source_path,
           "weights_sha256": digest.hexdigest(),
           "header_bytes": os.path.getsize(header_path),
           "source_bytes": os.path.getsize(source_path)}
    if verify_result is not None:
        out["verify"] = verify_result
    return out


# ======================================================================================
# vecnormalize check (must stay a no-op or the export above is incomplete)
# ======================================================================================
def vecnormalize_report(model_path: str, actor_dim: int) -> Dict[str, Any]:
    """Confirm the checkpoint's VecNormalize stats are a no-op, or say loudly that they are not."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    cands = [os.path.join(os.path.dirname(model_path), f"{stem}_vecnormalize.pkl"),
             os.path.join(_PROJECT_ROOT, f"{stem}_vecnormalize.pkl")]
    for cand in cands:
        if os.path.isfile(cand):
            try:
                import gymnasium as gym
                from gymnasium import spaces
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

                class _StatsSpaceEnv(gym.Env):
                    def __init__(self, dim: int) -> None:
                        self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
                        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

                    def reset(self, **kw):
                        return np.zeros(self.observation_space.shape, np.float32), {}

                    def step(self, action):
                        return np.zeros(self.observation_space.shape, np.float32), 0.0, False, False, {}

                    def render(self, *a, **k):
                        pass

                dummy = DummyVecEnv([lambda: _StatsSpaceEnv(actor_dim)])
                vn = VecNormalize.load(cand, dummy)
                return {"file": os.path.basename(cand),
                        "norm_obs": bool(vn.norm_obs),
                        "norm_reward": bool(vn.norm_reward),
                        "noop": not bool(vn.norm_obs)}
            except Exception as exc:  # pragma: no cover - diagnostic only
                return {"file": os.path.basename(cand), "error": str(exc)}
    return {"file": None, "noop": True, "note": "no VecNormalize stats found"}


# ======================================================================================
# main
# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Export the trained policy to C for the Crazyflie")
    ap.add_argument("--model", default="latest", help="checkpoint path/name (default: latest rl_model_*)")
    ap.add_argument("--encoder", default=os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt"))
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--no-verify", action="store_true", help="skip the numpy-vs-torch check")
    ap.add_argument("--verify-frames", type=int, default=3000)
    args = ap.parse_args()

    model_path = resolve_model(args.model)
    print(f"model   : {model_path}")
    print(f"encoder : {args.encoder}")

    actor_dim, net_arch = read_checkpoint_arch(model_path)
    if actor_dim is None:
        print("ERROR: checkpoint has no policy weights")
        return 2
    print(f"actor   : {actor_dim} dims, net_arch {net_arch}")
    # The on-board image assembles [o_t | z | ref_ff] from the frame, the GRU state and the
    # baked reference acceleration, so the feed-forward widths are now deployable. Anything
    # else is not: a width this build cannot feed would emit weights the app fills only
    # partially, which builds cleanly and flies with the tail of the actor input stale.
    deployable = (
        (ACTOR_TOTAL_DIM + REF_FF_DIM, ACTOR_TOTAL_DIM + Z_DIM + REF_FF_DIM)
    )
    if actor_dim not in deployable:
        print(f"WARNING: checkpoint expects {actor_dim} actor inputs; this build deploys "
              f"{ACTOR_TOTAL_DIM + Z_DIM + REF_FF_DIM} (with encoder z{Z_DIM} + ref_ff{REF_FF_DIM}).")
        if actor_dim not in (ACTOR_TOTAL_DIM, ACTOR_TOTAL_DIM + Z_DIM):
            print("Refusing: unknown actor width.")
            return 2

    act = extract_actor(model_path)
    enc = extract_encoder(args.encoder)
    consts = extract_env_constants()
    consts["_model_name"] = os.path.basename(model_path)
    consts["_encoder_name"] = os.path.basename(args.encoder)

    print(f"GRU     : f_in {enc['f_in']}, hidden {enc['hidden']}, z {enc['z_dim']}")
    print(f"norm    : mean/std over {enc['norm_mean'].size} dims, clip {enc['norm_clip']}")
    print(f"const   : dt {consts['sim_dt']}, rates {consts['max_rate_xy']}/{consts['max_rate_z']} rad/s, "
          f"maxThr {consts['max_thrust_n']} N, mass {consts['mass_kg']} kg, hover a0 {consts['hover_trim_a0']:+.4f}")

    verify_result: Optional[Dict[str, Any]] = None
    if not args.no_verify:
        verify_result = verify(act, enc, model_path, args.encoder, limit=args.verify_frames)
        status = "PASS" if verify_result["ok"] else "FAIL"
        print(f"verify  : {status}  z err {verify_result['max_z_err']:.3g} (tol {Z_TOL:g}), "
              f"action err {verify_result['max_action_err']:.3g} (tol {ACTION_TOL:g})  "
              f"[{verify_result['frames']} frames from {verify_result['source']}]")
        print(f"          dist {verify_result['dist']}, squash_output {verify_result['squash_output']}")

    out = emit_c(args.out_dir, act, enc, consts, verify_result)
    print(f"wrote   : {out['header']}")
    print(f"wrote   : {out['source']} ({out['source_bytes']} bytes)")
    print(f"sha256  : {out['weights_sha256'][:16]}… (weights)")

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"path": os.path.relpath(model_path, _PROJECT_ROOT), "sha256": sha256_file(model_path),
                  "actor_dim": int(actor_dim), "net_arch": net_arch},
        "encoder": {"path": os.path.relpath(args.encoder, _PROJECT_ROOT),
                    "sha256": sha256_file(args.encoder), "f_in": enc["f_in"],
                    "hidden": enc["hidden"], "z_dim": enc["z_dim"],
                    "checkpoint_config": enc["ckpt_config"]},
        "vecnormalize": vecnormalize_report(model_path, act["actor_dim"]),
        "env_constants": consts,
        "weights_sha256": out["weights_sha256"],
        "verify": verify_result,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"wrote   : {args.manifest}")

    if verify_result is not None and not verify_result["ok"]:
        print("\nVERIFY FAILED - the C arrays would not reproduce the checkpoint. Not stopping "
              "the write (the arrays are still consistent with the export contract), but do "
              "NOT flash until this passes.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
