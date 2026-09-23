# -*- coding: utf-8 -*-
"""
Export the deployed policy as ONE ONNX graph: primitives only, hidden state in/out.

WHY THIS SHAPE
--------------
A micro runtime (X-CUBE-AI / ST Edge AI Core, TFLite-Micro, or any equivalent the
deployment ends up using) will not execute a raw `GRU` node. The deployed step is
therefore UNROLLED into primitive ops, with the recurrent state as an explicit tensor:

    inputs :  frame [1, 33]   raw encoder frame [o_t(29) | aux(4)]
              ref_ff[1, 3]    reference feed-forward, a_ref + g*e_z (world, m/s^2)
              h_in  [1, 48]   GRU hidden state (zeroed at arming)
    outputs:  action [1, 4]   deterministic action, clipped to [-1, 1]
              h_out [1, 48]   state for the next step (copy back into h_in)
              z     [1, 16]   latent, for logging / parity checks

The actor input the graph forms internally is `[o_t(29) | z(16) | ref_ff(3)]` = 48, the
mirror of quad_flip_env's actor prefix. ref_ff is a separate INPUT rather than three
extra columns on `frame` because the encoder consumes `frame` whole and its contract is
frozen at 33 dims - the block has to reach the actor without entering the GRU.

The ops used are: Sub, Div, Clip, MatMul, Add, Sigmoid, Tanh, Mul, Concat, Slice and
GELU (exact erf form by default; `--gelu tanh` swaps in the tanh approximation for a
runtime without Erf, and the verification then reports exactly what that costs).

`--compat` writes a second, IMPORTER-FRIENDLY variant for ST Edge AI / X-CUBE-AI-class
tools: inputs split (o_t + aux, so nothing is sliced), GRU gates pre-split into r/z/n
(no Slice), reciprocal-multiply instead of divide (no Div) and tanh GELU (no Erf).
Result: Add, Sub, Mul, MatMul, Sigmoid, Tanh, Clip, Concat - nothing an importer can
reasonably reject. The only behavioural difference is the measured GELU cost
(z 5.3e-04, action 1.6e-04).

WEIGHT ORIENTATION. Weights are pre-transposed to [in, out] and consumed with MatMul, so
the graph contains no Transpose nodes (one fewer op for the runtime to implement).

NUMERICS ARE THE CONTRACT. This graph must reproduce `model.predict` + `encoder.step`:
the deterministic action is the raw Gaussian mean CLIPPED (not a tanh), and the GRU keeps
torch's bias split (b_ih in the input part, b_hh in the hidden part, so the new gate
multiplies the reset gate through the biased hidden term). Both were measured the hard
way - see `export_policy.py` and the repo notes.

USAGE
-----
    .venv/bin/python Simulation/deploy/export_onnx.py
    .venv/bin/python Simulation/deploy/export_onnx.py --gelu tanh --frames 5000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_policy import (  # noqa: E402
    REF_FF_DIM, _frame_stream, extract_actor, extract_encoder, ref_ff_stream, resolve_model,
)
from policy_host_check import torch_reference  # noqa: E402

DEFAULT_ONNX = os.path.join(_SIM_DIR, "deploy", "models", "policy_step.onnx")
DEFAULT_MANIFEST = os.path.join(_SIM_DIR, "deploy", "manifests", "policy_onnx.json")

Z_TOL = 2.0e-5
ACTION_TOL = 2.0e-5
# The ST-compatible graph trades exact GELU for the tanh form (the only op-set concession
# that changes numerics). These tolerances are the MEASURED cost of that swap - see the
# module docstring - not a relaxation of the contract.
COMPAT_Z_TOL = 2.0e-3
COMPAT_ACTION_TOL = 5.0e-4


# ======================================================================================
# the deployed step, as plain torch ops
# ======================================================================================
def build_module(act: Dict[str, np.ndarray], enc: Dict[str, Any], gelu: str):
    import torch
    import torch.nn as nn

    H = enc["hidden"]
    Z = enc["z_dim"]

    class DeployedStep(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            t = lambda a: nn.Parameter(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)), requires_grad=False)  # noqa: E731

            # encoder normalization (frozen constants, not affine parameters)
            self.register_buffer("n_mean", torch.from_numpy(enc["norm_mean"].astype(np.float32)))
            self.register_buffer("n_std", torch.from_numpy(enc["norm_std"].astype(np.float32)))
            self.register_buffer("n_clip", torch.tensor(float(enc["norm_clip"]), dtype=torch.float32))

            # GRU, pre-transposed to [in, out]
            self.W_ih = t(enc["w_ih"].T)      # [F, 3H]
            self.b_ih = t(enc["b_ih"])        # [3H]
            self.W_hh = t(enc["w_hh"].T)      # [H, 3H]
            self.b_hh = t(enc["b_hh"])        # [3H]

            # encoder head: Linear -> GELU -> Linear -> Tanh
            self.HW1 = t(enc["head_w1"].T)    # [H, H]
            self.Hb1 = t(enc["head_b1"])
            self.HW2 = t(enc["head_w2"].T)    # [H, Z]
            self.Hb2 = t(enc["head_b2"])

            # actor: tanh per hidden layer -> linear head, then CLIP (no tanh).
            # The layer COUNT comes from the checkpoint's net_arch (one layer for pi=[32]),
            # so the graph is built from act["hidden"] rather than a hardcoded W1/W2/W3.
            self.A_hidden = nn.ParameterList()
            for _w, _b in act["hidden"]:
                self.A_hidden.append(t(_w.T))      # [in, out]
                self.A_hidden.append(t(_b))
            self.AW_out = t(act["w_out"].T)       # [latent, 4]
            self.Ab_out = t(act["b_out"])

            self.gelu_kind = gelu

        def _gelu(self, x):
            if self.gelu_kind == "erf":
                # torch's default (exact) GELU; exports as Erf-based primitives at opset 17
                return torch.nn.functional.gelu(x)
            # tanh approximation of GELU (tanh is universally available, Erf often is not)
            c = 0.7978845608028654  # sqrt(2/pi)
            return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))

        def forward(self, frame, ref_ff, h_in):
            # -- standardization + clip (the encoder contract) --------------------------
            x = (frame - self.n_mean) / self.n_std
            x = torch.clamp(x, -self.n_clip, self.n_clip)

            # -- one GRU step (bias split preserved) ------------------------------------
            gi = x @ self.W_ih + self.b_ih
            gh = h_in @ self.W_hh + self.b_hh
            r = torch.sigmoid(gi[:, :H] + gh[:, :H])
            zt = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
            n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h_new = (1.0 - zt) * n + zt * h_in

            # -- head -> z ---------------------------------------------------------------
            a = self._gelu(h_new @ self.HW1 + self.Hb1)
            z = torch.tanh(a @ self.HW2 + self.Hb2)

            # -- actor: [o_t(raw) | z | ref_ff] -> mean -> clip --------------------------
            actor_in = torch.cat([frame[:, :29], z, ref_ff], dim=1)
            a = actor_in
            for _i in range(0, len(self.A_hidden), 2):
                a = torch.tanh(a @ self.A_hidden[_i] + self.A_hidden[_i + 1])
            action = torch.clamp(a @ self.AW_out + self.Ab_out, -1.0, 1.0)

            return action, h_new, z

    return DeployedStep()


def build_module_compat(act: Dict[str, np.ndarray], enc: Dict[str, Any]):
    """
    The same deployed step, restricted to the op set model-to-C compilers accept.

    Changes vs the default graph (all numerically transparent except the GELU):
      * inputs are SPLIT (o_t[1,29] + aux[1,4]) - the actor needs the raw o_t anyway, so
        nothing has to be sliced out of a 33-vector (zero Slice ops);
      * the GRU's combined [3H] gate matrices are pre-split into r/z/n blocks at export
        time, so the gates are three separate MatMuls instead of one plus three Slices;
      * standardization multiplies by a precomputed 1/std instead of dividing (no Div);
      * GELU uses the tanh form (no Erf). MEASURED cost: z 5.3e-04, action 1.6e-04
        (~0.18 deg/s of commanded rate) - the price of the op set, not of the model.

    Resulting ops: Add, Sub, Mul, MatMul, Sigmoid, Tanh, Clip, Concat.
    """
    import torch
    import torch.nn as nn

    H = enc["hidden"]
    Z = enc["z_dim"]
    wih = enc["w_ih"].T          # [F, 3H]
    whh = enc["w_hh"].T          # [H, 3H]
    blocks = {"r": slice(0, H), "z": slice(H, 2 * H), "n": slice(2 * H, 3 * H)}

    class DeployedStepCompat(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            t = lambda a: nn.Parameter(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)), requires_grad=False)  # noqa: E731

            self.register_buffer("n_mean", torch.from_numpy(enc["norm_mean"].astype(np.float32)))
            self.register_buffer("n_inv_std", torch.from_numpy(
                (1.0 / enc["norm_std"]).astype(np.float32)))
            self.register_buffer("n_clip", torch.tensor(float(enc["norm_clip"]), dtype=torch.float32))

            for name, sl in blocks.items():
                setattr(self, f"Wih_{name}", t(wih[:, sl]))
                setattr(self, f"bih_{name}", t(enc["b_ih"][sl]))
                setattr(self, f"Whh_{name}", t(whh[:, sl]))
                setattr(self, f"bhh_{name}", t(enc["b_hh"][sl]))

            self.HW1 = t(enc["head_w1"].T)
            self.Hb1 = t(enc["head_b1"])
            self.HW2 = t(enc["head_w2"].T)
            self.Hb2 = t(enc["head_b2"])

            # same layer-list construction as the exact graph: the count is a property of
            # the checkpoint, not of this file.
            self.A_hidden = nn.ParameterList()
            for _w, _b in act["hidden"]:
                self.A_hidden.append(t(_w.T))
                self.A_hidden.append(t(_b))
            self.AW_out = t(act["w_out"].T)
            self.Ab_out = t(act["b_out"])

        def _gelu(self, x):
            c = 0.7978845608028654  # sqrt(2/pi)
            return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))

        def forward(self, o_t, aux, ref_ff, h_in):
            frame = torch.cat([o_t, aux], dim=1)
            x = torch.clamp((frame - self.n_mean) * self.n_inv_std, -self.n_clip, self.n_clip)

            # bias split preserved: b_hh stays INSIDE the reset-gate term of the new gate
            r = torch.sigmoid(x @ self.Wih_r + self.bih_r + h_in @ self.Whh_r + self.bhh_r)
            zt = torch.sigmoid(x @ self.Wih_z + self.bih_z + h_in @ self.Whh_z + self.bhh_z)
            n = torch.tanh(x @ self.Wih_n + self.bih_n
                           + r * (h_in @ self.Whh_n + self.bhh_n))
            h_new = (1.0 - zt) * n + zt * h_in

            a = self._gelu(h_new @ self.HW1 + self.Hb1)
            z = torch.tanh(a @ self.HW2 + self.Hb2)

            # o_t is already raw and unsliced, and ref_ff comes in as its own tensor, so the
            # actor input is one Concat with no Slice anywhere in the graph.
            actor_in = torch.cat([o_t, z, ref_ff], dim=1)
            a = actor_in
            for _i in range(0, len(self.A_hidden), 2):
                a = torch.tanh(a @ self.A_hidden[_i] + self.A_hidden[_i + 1])
            action = torch.clamp(a @ self.AW_out + self.Ab_out, -1.0, 1.0)
            return action, h_new, z

    return DeployedStepCompat()


# ======================================================================================
# export + verify
# ======================================================================================
def export_onnx(module, path: str, opset: int, example_inputs, input_names) -> Tuple[str, str]:
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = dict(
        input_names=input_names,
        output_names=["action", "h_out", "z"],
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(module, example_inputs, path, dynamo=False, **kwargs)
        exporter = "torchscript"
    except TypeError:
        # torch dropped the legacy exporter; the dynamo path emits the same primitive ops
        torch.onnx.export(module, example_inputs, path, dynamo=True, **kwargs)
        exporter = "dynamo"
    return exporter, path


def op_report(path: str) -> Dict[str, Any]:
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    ops: Dict[str, int] = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    params = sum(int(np.prod(init.dims)) for init in model.graph.initializer)
    param_bytes = sum(int(np.prod(init.dims)) * 4 for init in model.graph.initializer)
    return {
        "opset": [int(o.version) for o in model.opset_import],
        "ops": dict(sorted(ops.items())),
        "n_nodes": len(model.graph.node),
        "params": params,
        "flash_bytes_fp32": param_bytes,
        "file_bytes": os.path.getsize(path),
        "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
    }


def verify_onnx(path: str, frames: np.ndarray, model_path: str, encoder_path: str,
                compat: bool = False, z_tol: float = Z_TOL,
                action_tol: float = ACTION_TOL) -> Dict[str, Any]:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    expected_in = ["o_t", "aux", "ref_ff", "h_in"] if compat else ["frame", "ref_ff", "h_in"]
    if in_names != expected_in or out_names != ["action", "h_out", "z"]:
        raise RuntimeError(f"unexpected io: {in_names} -> {out_names}")

    n = len(frames)
    ff = ref_ff_stream(n)
    h = np.zeros((1, 48), dtype=np.float32)
    onnx_a = np.zeros((n, 4), dtype=np.float32)
    onnx_z = np.zeros((n, 16), dtype=np.float32)
    for i in range(n):
        frame = frames[i][None, :].astype(np.float32)
        # The SAME feed-forward block the torch reference uses, or the two cannot agree.
        ff_i = ff[i][None, :]
        if compat:
            feeds = {"o_t": frame[:, :29], "aux": frame[:, 29:], "ref_ff": ff_i, "h_in": h}
        else:
            feeds = {"frame": frame, "ref_ff": ff_i, "h_in": h}
        a, h, z = sess.run(["action", "h_out", "z"], feeds)
        onnx_a[i] = a[0]
        onnx_z[i] = z[0]

    tz, ta = torch_reference(frames, model_path, encoder_path, ref_ff=ff)

    dz = float(np.max(np.abs(onnx_z - tz)))
    da = float(np.max(np.abs(onnx_a - ta)))
    return {
        "ok": bool(dz <= z_tol and da <= action_tol),
        "frames": int(n),
        "z_tol": z_tol,
        "action_tol": action_tol,
        "max_z_err": dz,
        "max_action_err": da,
        "mean_z_err": float(np.mean(np.abs(onnx_z - tz))),
        "mean_action_err": float(np.mean(np.abs(onnx_a - ta))),
        "action_saturation_frac": float(np.mean(np.abs(onnx_a) >= 0.999)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export the deployed policy to ONNX (primitives, state in/out)")
    ap.add_argument("--model", default="latest")
    ap.add_argument("--encoder", default=os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt"))
    ap.add_argument("--out", default="", help="default: models/policy_step[_stedgeai].onnx")
    ap.add_argument("--manifest", default="", help="default: manifests/policy_onnx[_stedgeai].json")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--gelu", choices=["erf", "tanh"], default="erf")
    ap.add_argument("--compat", action="store_true",
                    help="ST Edge AI / X-CUBE-AI-friendly graph: no Slice/Div/Erf")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    model_path = resolve_model(args.model)
    act = extract_actor(model_path)
    enc = extract_encoder(args.encoder)
    if act["actor_dim"] != 29 + 16 + REF_FF_DIM:
        print(f"WARNING: actor expects {act['actor_dim']} inputs; the deployment target is "
              f"{29 + 16 + REF_FF_DIM} ([o_t|z|ref_ff]).")

    import torch

    suffix = "_stedgeai" if args.compat else ""
    out_path = args.out or os.path.join(os.path.dirname(DEFAULT_ONNX), f"policy_step{suffix}.onnx")
    manifest_path = args.manifest or os.path.join(os.path.dirname(DEFAULT_MANIFEST),
                                                 f"policy_onnx{suffix}.json")

    if args.compat:
        module = build_module_compat(act, enc)
        example_inputs = (torch.zeros(1, 29), torch.zeros(1, 4), torch.zeros(1, REF_FF_DIM),
                          torch.zeros(1, int(enc["hidden"])))
        input_names = ["o_t", "aux", "ref_ff", "h_in"]
        z_tol, action_tol = COMPAT_Z_TOL, COMPAT_ACTION_TOL
        print("variant : ST-compatible (split inputs, pre-split gates, mul-by-rcp, tanh GELU)")
    else:
        module = build_module(act, enc, args.gelu)
        example_inputs = (torch.zeros(1, 33), torch.zeros(1, REF_FF_DIM),
                          torch.zeros(1, int(enc["hidden"])))
        input_names = ["frame", "ref_ff", "h_in"]
        z_tol, action_tol = Z_TOL, ACTION_TOL

    module.eval()
    exporter, path = export_onnx(module, out_path, args.opset, example_inputs, input_names)
    rep = op_report(path)
    print(f"model   : {os.path.basename(model_path)}")
    print(f"encoder : f_in {enc['f_in']}, hidden {enc['hidden']}, z {enc['z_dim']}")
    print(f"gelu    : {'tanh (compat)' if args.compat else args.gelu}")
    print(f"export  : {exporter} exporter, opset {rep['opset']}")
    print(f"ops     : {', '.join(f'{k}x{v}' for k, v in rep['ops'].items())}")
    print(f"size    : {rep['file_bytes']} bytes on disk, {rep['params']} params "
          f"({rep['flash_bytes_fp32']} bytes as fp32 weights)")
    print(f"wrote   : {path}")

    verify: Dict[str, Any] = {}
    if not args.no_verify:
        frames, source = _frame_stream(args.frames)
        verify = verify_onnx(path, frames, model_path, args.encoder,
                             compat=args.compat, z_tol=z_tol, action_tol=action_tol)
        status = "PASS" if verify["ok"] else "FAIL"
        print(f"verify  : {status}  z err {verify['max_z_err']:.3g}  "
              f"action err {verify['max_action_err']:.3g}  "
              f"({verify['frames']} frames from {source}; tol {z_tol:g}/{action_tol:g})")
        print(f"          mean z err {verify['mean_z_err']:.2e}, mean action err "
              f"{verify['mean_action_err']:.2e}, saturated actions "
              f"{100.0 * verify['action_saturation_frac']:.1f}%")

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": os.path.relpath(model_path, _PROJECT_ROOT),
        "encoder": os.path.relpath(args.encoder, _PROJECT_ROOT),
        "onnx": os.path.relpath(path, _PROJECT_ROOT),
        "gelu": "tanh" if args.compat else args.gelu,
        "compat": bool(args.compat),
        "exporter": exporter,
        "graph": rep,
        "verify": verify or None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"wrote   : {manifest_path}")
    return 0 if (args.no_verify or verify.get("ok", False)) else 3


if __name__ == "__main__":
    raise SystemExit(main())
