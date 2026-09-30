"""
Deterministic evaluation of an MJX/Flax policy: `Simulation/quad_mjx`.

This is the MJX counterpart of `Simulation/evaluate.py` and it reports the SAME metric --
episode return as a percentage of the per-step reward ceiling (7.3) -- per manoeuvre
family, plus overall and worst-family, so a run here can be compared directly against the
recorded SB3 baseline numbers.

It also makes the checkpoints honest, which the previous MJX evaluator did not:
  * it LOADS the trained weights (``logs/quad_mjx_policy.npz``) instead of instantiating a
    fresh network and reporting a random policy's score;
  * it reconstructs the nested Flax param tree from the flat npz via
    ``train_mjx.load_params``, and asserts the round-trip is bit-exact;
  * it validates the checkpoint's widths against the env's observation contract before
    evaluating, so a stale/mismatched npz fails loudly instead of scoring garbage.

Run:
    .venv/bin/python -u Simulation/evaluate_mjx.py
    .venv/bin/python -u Simulation/evaluate_mjx.py --episodes 5 --families flip,chain
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# _HERE must WIN over _ROOT.  The repository root contains `train_mjx.py`, a thin shim;
# if it shadows `Simulation/train_mjx.py` then `T.TOTAL_TIMESTEPS` and every other constant
# disappears.  A plain `if p not in sys.path` guard is not enough: Python already puts this
# file's own directory on the path, so the root ends up in FRONT and the shim wins.
for p in (_ROOT, _HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import jax
import jax.numpy as jnp

import train_mjx as T
from quad_mjx import spec as S
from quad_mjx.env import QuadFlipMJXEnv

# Reference points recorded from the SB3 baseline (percent of ceiling, per family).
# Chain/cauchy era, in KIND_NAMES order: hover, takeoff, waypoints, fig8, orbit, lissajous,
# slalom, v8, flip, chain.
BASELINE_REFERENCE = {
    "hover": 78, "takeoff": 87, "waypoints": 81, "fig8": 86, "orbit": 83,
    "lissajous": 92, "slalom": 90, "v8": 95, "flip": 81, "chain": 78,
}


def main():
    ap = argparse.ArgumentParser(description="Evaluate an MJX policy per manoeuvre family")
    ap.add_argument("--weights", default=os.path.join(_ROOT, "logs", "quad_mjx_policy.npz"),
                    help="flat npz written by train_mjx.run_training")
    ap.add_argument("--episodes", type=int, default=5, help="episodes per family (default: 5)")
    ap.add_argument("--families", default="", help="comma list; default = all 10")
    ap.add_argument("--dr", type=float, default=0.0, help="domain-randomisation level 0..1")
    ap.add_argument("--envelope", type=float, default=S.ENVELOPE_SCALE_END,
                    help="reference envelope scale (1.5 = the trained end value)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--random", action="store_true",
                    help="evaluate an UNTRAINED policy (explicit sanity baseline)")
    ap.add_argument("--csv", default="", help="append results to this CSV (default: none)")
    a = ap.parse_args()

    env = QuadFlipMJXEnv()
    net = T.ActorCritic()

    if a.random:
        params = T.init_params(net, env, a.seed)
        print("[warn] --random: evaluating an UNTRAINED policy; scores are a floor, "
              "not a result\n")
    else:
        if not os.path.isfile(a.weights):
            raise SystemExit(
                f"no checkpoint at {a.weights}\n"
                f"  train one first:  .venv/bin/python -u Simulation/train_mjx.py\n"
                f"  or pass --random for an untrained sanity run")
        template = T.init_params(net, env, a.seed)
        flat = T.load_params(a.weights)
        params = T.load_params(a.weights, template)

        # the round-trip has to be exact, or every downstream number is fiction
        back = T._flatten_params(params)
        if set(back) != set(flat):
            raise SystemExit(f"checkpoint keys do not match the network: "
                             f"{sorted(set(back) ^ set(flat))[:6]}")
        worst = max(float(np.max(np.abs(np.asarray(back[k]) - np.asarray(flat[k]))))
                    for k in flat)
        if worst != 0.0:
            raise SystemExit(f"param round-trip is not exact (max diff {worst:.3e})")

        # report the two INPUT widths straight off the checkpoint: this is where a missed
        # critic-dimension change (80 vs 96) shows up before any score is computed
        widths = {k: tuple(np.asarray(v).shape) for k, v in flat.items()
                  if k.endswith("fc0/kernel")}
        print(f"[load] {os.path.relpath(a.weights, _ROOT)}  "
              f"actor={env.actor_dim} critic={env.critic_dim}  round-trip exact; "
              f"first-layer kernels {widths}")

    families = [f.strip() for f in a.families.split(",") if f.strip()] or None
    if families:
        bad = [f for f in families if f not in S.KIND_INDEX]
        if bad:
            raise SystemExit(f"unknown families {bad}; expected one of {S.KIND_NAMES}")

    print(f"[eval] envelope {a.envelope:.2f}  dr {a.dr:.2f}  "
          f"{a.episodes} episode(s)/family  dtype float32")
    t0 = time.time()
    names, pct, length, track = T.evaluate_families(
        env, net, params, families=families, episodes=a.episodes,
        key=jax.random.PRNGKey(a.seed), envelope=a.envelope, dr=a.dr)
    jax.block_until_ready(jnp.asarray(pct))
    dt = time.time() - t0

    print()
    print(f"  {'family':<12} {'score':>7}  {'track':>7}  {'vs baseline':>11}  "
          f"{'mean len':>8}")
    print("  " + "-" * 56)
    for i, n in enumerate(names):
        ref = BASELINE_REFERENCE.get(n)
        delta = f"{pct[i] - ref:+7.1f}" if ref is not None else "        -"
        print(f"  {n:<12} {pct[i]:6.1f}%  {track[i]:6.1f}%  {delta:>11}  {length[i]:7.0f}")
    print("  " + "-" * 44)
    print(f"  {'overall':<12} {pct.mean():6.1f}%")
    print(f"  {'worst':<12} {pct.min():6.1f}%  ({names[int(np.argmin(pct))]})")
    print(f"\n  {len(names) * a.episodes} episodes in {dt:.1f}s")

    if a.csv:
        new = not os.path.exists(a.csv)
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, "a") as f:
            if new:
                f.write("family,score,mean_len\n")
            for i, n in enumerate(names):
                f.write("%s,%.2f,%.1f\n" % (n, pct[i], length[i]))
        print(f"  appended to {a.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
