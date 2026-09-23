#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install the ST Edge AI generated network into the app's source tree.

    .venv/bin/python Simulation/deploy/install_stedgeai.py

`stedgeai generate` writes into Simulation/deploy/stedgeai_out/ (gitignored); kbuild can
only descend into subdirectories of the app, so this copies the generated files into

    Simulation/deploy/app_policy_controller/src/generated/stedgeai/

which is where src/generated/Kbuild picks them up for an `APP_BACKEND=stedgeai` build.

It CHECKS, rather than assumes, the two things a silent mismatch would ruin:

  * the generated network's tensor sizes against the frame contract declared in
    policy_net_stedgeai.h - 3 in (o_t 29 / aux 4 / h_in 48), 3 out (action 4 / h_out 48 /
    z 16). A regenerated model with different shapes would otherwise "work" while reading
    the wrong bytes;
  * every POLICY_* constant in the ST header against the ones export_policy.py wrote from
    the SAME checkpoint into generated/policy_weights.h - the action mapping, the EMA and
    the hover trim. Those decide what an action MEANS, so a divergence there flies a
    different vehicle.

and it verifies the ST pack really is where the Makefile expects it (the runtime headers
and the Cortex-M4 archive), because that path is a hard dependency of the build.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.join(_HERE, "app_policy_controller")
_SRC = os.path.join(_APP, "src")
_GENERATED = os.path.join(_SRC, "generated")
_DESTS = os.path.join(_GENERATED, "stedgeai")

DEFAULT_STEDGEAI_DIR = "/Applications/ST/STEdgeAI/4.0"

# What `stedgeai generate` produces and what the build needs.
FILES = ("network.c", "network.h", "network_data.c", "network_data.h", "network_details.h")

# The frame contract, mirrored from policy_net_stedgeai.h / policy_weights.h.
EXPECTED_IN = (29, 4, 3, 48)      # o_t | aux | ref_ff | h_in
EXPECTED_OUT = (4, 48, 16)        # action | h_out | z


def _parse_defines(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.match(r"^\s*#define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", line)
            if m:
                # STRIP A TRAILING `//` COMMENT. export_policy.py annotates several
                # constants that way (e.g. `... 1  // quad_flip_env.ACTOR_FRAME_MODE = ...`),
                # and leaving it in makes the value fail the numeric comparison while
                # printing two identical-looking numbers - a genuinely confusing failure.
                out[m.group(1)] = re.sub(r"//.*$", "", m.group(2)).strip()
    return out


def _number(value: str) -> str:
    """`((4))` / `(29)` / `153568` -> `4` / `29` / `153568` (leave anything else alone)."""
    m = re.fullmatch(r"\(*\s*(\d+)\s*\)*", value)
    return m.group(1) if m else value


def _tensor_sizes(defines: dict[str, str], prefix: str, count: int) -> tuple[str, ...]:
    return tuple(_number(defines.get(f"{prefix}_{i}_SIZE", "?")) for i in range(1, count + 1))


def _tensor_count(defines: dict[str, str], name: str) -> int:
    """
    How many tensors the generated network declares.

    Read it, never assume it. This used to be a hardcoded 3, so adding the ref_ff input to
    the graph left the checker looking at IN_1..IN_3 and reporting a baffling
    "('29', '4', '3') vs ('29', '4', '3', '48')" - and a future reordering that kept three
    plausible sizes would have been accepted outright.
    """
    raw = _number(defines.get(name, "0"))
    return int(raw) if raw.isdigit() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Install the ST Edge AI network into the app")
    ap.add_argument("--from", dest="src_dir", default=os.path.join(_HERE, "stedgeai_out"),
                    help="the `stedgeai generate --output` directory")
    ap.add_argument("--stedgeai-dir", default=DEFAULT_STEDGEAI_DIR,
                    help="the ST Edge AI Core installation (default %(default)s)")
    args = ap.parse_args()

    print(f"source : {args.src_dir}")
    print(f"dest   : {_DESTS}")

    # -- 1. the generated output must be there ------------------------------------------
    missing = [f for f in FILES if not os.path.isfile(os.path.join(args.src_dir, f))]
    if missing:
        print(f"\nERROR: {args.src_dir} is missing {', '.join(missing)}", file=sys.stderr)
        print("       run: stedgeai generate --model Simulation/deploy/models/"
              "policy_step_stedgeai.onnx \\", file=sys.stderr)
        print("                --target stm32f4 --output Simulation/deploy/stedgeai_out "
              "< /dev/null", file=sys.stderr)
        return 1

    # -- 2. the ST pack must be where the Makefile thinks it is -------------------------
    inc = os.path.join(args.stedgeai_dir, "Middlewares", "ST", "AI", "Inc")
    lib = os.path.join(args.stedgeai_dir, "Middlewares", "ST", "AI", "Lib", "GCC",
                       "ARMCortexM4", "NetworkRuntime1201_CM4_GCC.a")
    for path, what in ((os.path.join(inc, "stai.h"), "runtime header stai.h"),
                       (lib, "Cortex-M4 runtime archive")):
        if not os.path.isfile(path):
            print(f"\nERROR: {what} not found at {path}", file=sys.stderr)
            print("       pass --stedgeai-dir / set STEDGEAI_DIR to the ST Edge AI Core "
                  "installation", file=sys.stderr)
            return 1
    print(f"pack   : {args.stedgeai_dir} (headers + Cortex-M4 runtime archive)  OK")

    # -- 3. tensor shapes vs the frame contract -----------------------------------------
    net = _parse_defines(os.path.join(args.src_dir, "network.h"))
    got_in = _tensor_sizes(net, "STAI_NETWORK_IN", _tensor_count(net, "STAI_NETWORK_IN_NUM"))
    got_out = _tensor_sizes(net, "STAI_NETWORK_OUT", _tensor_count(net, "STAI_NETWORK_OUT_NUM"))
    want_in = tuple(str(v) for v in EXPECTED_IN)
    want_out = tuple(str(v) for v in EXPECTED_OUT)
    if got_in != want_in or got_out != want_out:
        print("\nERROR: tensor shapes do not match the frame contract", file=sys.stderr)
        print(f"       in  generated {got_in}  expected {want_in}", file=sys.stderr)
        print(f"       out generated {got_out}  expected {want_out}", file=sys.stderr)
        print("       (re-export with `export_onnx.py --compat`; the network is stale)",
              file=sys.stderr)
        return 1
    print(f"shapes : in {got_in} out {got_out}  OK")

    # -- 4. the POLICY_* constants must agree with the exporter -------------------------
    # BIDIRECTIONAL on purpose. Comparing only the SHARED keys means a constant that the
    # exporter added but this header never learned about is invisible here - and that is not
    # hypothetical: POLICY_FRAME_ANCHORED_XY was added to the generated header, required by
    # controller_app.c, and the omission only surfaced as a confusing #error deep in the
    # firmware build. Everything the exporter emits describes the deployed contract, so the
    # ST header must mirror ALL of it.
    st_hdr = _parse_defines(os.path.join(_SRC, "policy_net_stedgeai.h"))
    ref_hdr = _parse_defines(os.path.join(_GENERATED, "policy_weights.h"))
    generated_keys = sorted(k for k in ref_hdr if k.startswith("POLICY_"))
    if not generated_keys:
        print("\nERROR: no POLICY_* constants in policy_weights.h - is it "
              "generated yet?", file=sys.stderr)
        print("       run Simulation/deploy/export_policy.py first", file=sys.stderr)
        return 1
    missing = [k for k in generated_keys if k not in st_hdr]
    if missing:
        print("\nERROR: policy_net_stedgeai.h is missing constants the exporter emitted",
              file=sys.stderr)
        for k in missing:
            print(f"       {k} = {ref_hdr[k]}  (add it to the hand-kept block)", file=sys.stderr)
        return 1
    bad = [(k, st_hdr[k], ref_hdr[k]) for k in generated_keys
           if _number(st_hdr[k]) != _number(ref_hdr[k])]
    if bad:
        print("\nERROR: the ST backend's constants disagree with the exporter's", file=sys.stderr)
        for k, a, b in bad:
            print(f"       {k}: policy_net_stedgeai.h {a}  vs  policy_weights.h {b}",
                  file=sys.stderr)
        return 1
    print(f"contract: {len(generated_keys)} POLICY_* constants present and matching  OK")

    # -- 5. install ---------------------------------------------------------------------
    os.makedirs(_DESTS, exist_ok=True)
    for name in FILES:
        shutil.copyfile(os.path.join(args.src_dir, name), os.path.join(_DESTS, name))
    print(f"copied : {len(FILES)} files -> {os.path.relpath(_DESTS, os.path.dirname(_HERE))}")

    # -- 6. what it costs, against the measured headroom ---------------------------------
    weights = int(_number(net.get("STAI_NETWORK_WEIGHTS_SIZE_BYTES", "0")))
    acts = int(_number(net.get("STAI_NETWORK_ACTIVATION_1_SIZE_BYTES", "0")))
    macc = int(_number(net.get("STAI_NETWORK_MACC_NUM", "0")))
    print("\ngenerated model")
    print(f"  weights     : {weights:,} B  (flash, const)")
    print(f"  activations : {acts:,} B  (RAM, in the state struct)")
    print(f"  macc        : {macc:,} per inference  ({macc * 100:,}/s at 100 Hz)")
    print("  st runtime  : ~15.8 KB flash / ~4.1 KB RAM (plus the kernels kept, "
          "see the build's size report)")
    print("  budget      : 549,696 B flash and 21,972 B RAM free in the app build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
