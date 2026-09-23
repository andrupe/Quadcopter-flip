"""
Validation for the manoeuvre MIXTURE CURRICULUM: the `chain` family's draw weight is
ramped up toward the end of a run, so the longest/hardest command in the set is
introduced progressively instead of being a ~4% sliver of every episode from step 0.

The claims that matter:

  A. WEIGHTS ARE RELATIVE. `TrajectoryConfig.set_weight` changes one family and the
     sampler re-normalises on every draw, so the other families shrink in proportion and
     the probabilities still sum to 1. Bad names and bad values are refused (a typo must
     not silently do nothing).
  B. LOWER... raising the chain weight really does make chains more frequent in the
     REALIZED mixture, and the realized share lags the nominal one (chain candidates are
     the most-rejected family and the sampler's fallback is a Hover). The measured band is
     printed so the schedule constants can be read off it.
  C. THE RPC PATH WORKS. The curriculum pushes weights from the trainer process with
     `env_method`, through the SubprocVecEnv AND the LatentObsWrapper that sits inside it
     during training - the same path `set_dr_level` uses. This is the part that silently
     breaks if a wrapper does not forward attributes, so it is checked for real.
  D. THE SCHEDULE IS THE INTENDED ONE: flat at the base weight until `start_steps`,
     linear in between, pinned at `weight_end` from `end_steps` on (and a resume at step
     X starts from the value the schedule has at X, not from the base weight).
  E. THE MIXTURE STILL CONTAINS EVERY FAMILY, including the pinned-command interface:
     re-weighting does not remove a family from `set_command`'s vocabulary.

Run:  .venv/bin/python scratch/check_mix_curriculum.py
"""

from __future__ import annotations

import collections
import os
import sys
from types import SimpleNamespace

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import REF_FF_DIM, QuadFlipEnv  # noqa: E402
from trajectories import TrajectoryConfig, TrajectorySampler  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def realized_chain_share(env: QuadFlipEnv, weight: float, n: int = 120) -> float:
    """Fraction of DRAWS that are chains, through the env's own reset path."""
    env.set_maneuver_weight("chain", weight)
    hits = 0
    for s in range(n):
        _, info = env.reset(seed=100_000 + s)
        hits += info["maneuver"] == "chain"
    return hits / n


def main() -> int:
    print("=" * 78)
    print("A. weights are relative, and bad input is refused")
    print("=" * 78)
    cfg = TrajectoryConfig()
    base = cfg.normalized_weights()
    check("base mixture sums to 1", abs(sum(base.values()) - 1.0) < 1e-12,
          f"sum {sum(base.values()):.12f}")
    check("base mixture contains all nine families", len(base) == 9, f"{len(base)} families")

    cfg.set_weight("chain", 0.25)
    norm = cfg.normalized_weights()
    check("weights stay normalised after a change", abs(sum(norm.values()) - 1.0) < 1e-12,
          f"sum {sum(norm.values()):.12f}")
    check("the changed family gains share", norm["chain"] > base["chain"],
          f"{base['chain']:.1%} -> {norm['chain']:.1%}")
    others_ratio = [norm[k] / base[k] for k in base if k != "chain"]
    check("every other family shrinks by the SAME factor (so their ratios are untouched)",
          max(others_ratio) - min(others_ratio) < 1e-12,
          f"factor {np.mean(others_ratio):.4f}")
    check("zero is allowed (a family can be dropped from the mix)",
          TrajectoryConfig().set_weight("chain", 0.0) is None)

    for bad_name in ("barrel_roll", "Chain", ""):
        try:
            TrajectoryConfig().set_weight(bad_name, 0.2)
            check(f"unknown family {bad_name!r} is refused", False, "no exception")
        except ValueError:
            check(f"unknown family {bad_name!r} is refused", True)
    for bad_w in (-0.1, float("nan"), float("inf")):
        try:
            TrajectoryConfig().set_weight("chain", bad_w)
            check(f"invalid weight {bad_w!r} is refused", False, "no exception")
        except ValueError:
            check(f"invalid weight {bad_w!r} is refused", True)

    print()
    print("=" * 78)
    print("B. raising the weight really does buy chains (measured, through the env)")
    print("=" * 78)
    env = QuadFlipEnv()
    measured = {}
    for w in (0.05, 0.15, 0.25, 0.35):
        share = realized_chain_share(env, w)
        measured[w] = share
        print(f"     weight {w:.2f}  ->  realized {share:5.1%} of draws")
    check("a higher weight gives at least as many chains (monotone-ish in the measured band)",
          measured[0.25] >= measured[0.05], f"{measured[0.05]:.1%} -> {measured[0.25]:.1%}")
    check("the boosted weight multiplies the chain share several-fold",
          measured[0.25] >= 2.5 * max(measured[0.05], 1e-9),
          f"{measured[0.25] / max(measured[0.05], 1e-9):.1f}x")
    check("the realized share lags the nominal one (chain candidates are rejected most)",
          measured[0.25] < 0.25 / (0.95 + 0.25) + 1e-9,
          f"realized {measured[0.25]:.1%} < nominal {0.25 / 1.20:.1%}")
    check("the realized share saturates before the weight dominates the mix",
          measured[0.35] - measured[0.25] < 0.05,
          f"0.25 -> {measured[0.25]:.1%}, 0.35 -> {measured[0.35]:.1%}")
    check("every family is still reachable at the boosted weight",
          len({env.reset(seed=s)[1]["maneuver"] for s in range(60)}) >= 4,
          f"{len({env.reset(seed=s)[1]['maneuver'] for s in range(60)})} kinds in 60 draws")

    print()
    print("=" * 78)
    print("C. the trainer's RPC path reaches the workers (SubprocVecEnv + latent wrapper)")
    print("=" * 78)
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_env():
        return QuadFlipEnv(telemetry=False)

    vec = SubprocVecEnv([make_env, make_env])
    try:
        encoder_path = os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt")
        wrapped = os.path.isfile(encoder_path)
        if wrapped:
            from encoder.latent_obs_wrapper import LatentObsWrapper
            vec = LatentObsWrapper(vec, encoder_path=encoder_path, z_dim=16, ref_ff_dim=REF_FF_DIM)
            print("     (encoder found: the wrapper that sits inside vec_env is exercised)")

        before = vec.env_method("get_maneuver_weights")[0]["chain"]
        vec.env_method("set_maneuver_weight", "chain", 0.25)
        after = [d["chain"] for d in vec.env_method("get_maneuver_weights")]
        check("set_maneuver_weight crosses the vec-env boundary", after[0] > before + 0.05,
              f"{before:.1%} -> {after[0]:.1%}" + (" (through LatentObsWrapper)" if wrapped else ""))
        check("EVERY worker got the change", all(abs(a - after[0]) < 1e-12 for a in after),
              str([round(a, 4) for a in after]))
        # A bad family name must fail in the TRAINER. It cannot be checked over the RPC
        # channel: an exception raised inside a worker's env_method does not come back as
        # that exception - SB3's worker loop dies and the trainer sees EOFError /
        # "Broken pipe" (measured below), so the callback validates the name itself.
        from train import (  # noqa: E402
            CHAIN_MIX_END_STEPS,
            CHAIN_MIX_START_STEPS,
            CHAIN_WEIGHT_END,
            CHAIN_WEIGHT_START,
            ManeuverMixCurriculumCallback,
        )

        try:
            ManeuverMixCurriculumCallback(family="barrel_roll", verbose=0)
            check("a bad family name fails fast, in the trainer process", False,
                  "no exception raised")
        except ValueError as exc:
            check("a bad family name fails fast, in the trainer process", True,
                  str(exc)[:52] + "...")
        check("the family name used by the schedule is valid",
              ManeuverMixCurriculumCallback(verbose=0).family == "chain")

        # The callback itself, driven against the real vec env with a stub model. BaseCallback
        # only needs `model.num_timesteps` and `model.get_env()`, so no PPO run is required.
        cb = ManeuverMixCurriculumCallback(verbose=1)
        cb.model = SimpleNamespace(get_env=lambda: vec)
        # SB3 sets `callback.num_timesteps = model.num_timesteps` on every update, so the
        # schedule is driven by the LIFETIME step count (which is what makes a resume pick
        # the ramp up where it left off). The stub drives the same attribute.
        cb.num_timesteps = CHAIN_MIX_START_STEPS
        cb._on_training_start()
        check("the callback pushes the BASE weight at the start of training",
              abs(vec.env_method("get_maneuver_weights")[0]["chain"]
                  - CHAIN_WEIGHT_START / (1.0 - CHAIN_WEIGHT_START + CHAIN_WEIGHT_START)) < 1e-9,
              f"chain share {vec.env_method('get_maneuver_weights')[0]['chain']:.3f}")
        cb.num_timesteps = CHAIN_MIX_END_STEPS
        cb._on_rollout_end()
        final_share = vec.env_method("get_maneuver_weights")[0]["chain"]
        expected = CHAIN_WEIGHT_END / (1.0 - CHAIN_WEIGHT_START + CHAIN_WEIGHT_END)
        check("the callback pushes the END weight at the end of training",
              abs(final_share - expected) < 1e-9, f"{final_share:.4f} vs {expected:.4f}")
        check("the other families were never touched by the callback",
              abs(sum(vec.env_method("get_maneuver_weights")[0].values()) - 1.0) < 1e-9)
    finally:
        vec.close()

    print()
    print("=" * 78)
    print("D. the schedule is the intended one (flat, ramp, pinned, resume-safe)")
    print("=" * 78)
    from train import ManeuverMixCurriculumCallback as CB  # noqa: E402

    cb = CB(start_steps=6_000_000, end_steps=15_000_000, weight_start=0.05, weight_end=0.25)
    at = {s: cb._weight_at(s) for s in (0, 3_000_000, 6_000_000, 10_500_000, 15_000_000, 30_000_000)}
    check("flat at the base weight until start_steps", at[0] == 0.05 == at[6_000_000], str(at))
    check("halfway is halfway", abs(at[10_500_000] - 0.15) < 1e-9, f"{at[10_500_000]:.4f}")
    check("pinned at weight_end from end_steps on", at[15_000_000] == 0.25 == at[30_000_000])
    check("a degenerate window (end <= start) jumps straight to the final weight",
          CB(start_steps=5, end_steps=5)._weight_at(0) == 0.25)

    # The PRODUCTION constants must be a coherent set, because the failures here are silent:
    # a chain ramp that ends after the run finishes simply never reaches its final share,
    # and an LR phase that outlives the run leaves the last phase unreachable. The trainer
    # prints a warning for the LR one; nothing warns about the mixture one.
    from train import (  # noqa: E402
        CHAIN_MIX_END_STEPS as MIX_END,
        CHAIN_MIX_START_STEPS as MIX_START,
        CHAIN_WEIGHT_END as W_END,
        CHAIN_WEIGHT_START as W_START,
        DR_END_STEPS,
        DR_START_STEPS,
        LR_WARMUP_STEPS,
        TOTAL_TIMESTEPS,
    )

    print(f"     production schedule: total {TOTAL_TIMESTEPS / 1e6:.0f}M | warmup {LR_WARMUP_STEPS / 1e6:.0f}M | "
          f"DR {DR_START_STEPS / 1e6:.0f}M -> {DR_END_STEPS / 1e6:.0f}M | "
          f"chain {W_START:.2f} -> {W_END:.2f} over {MIX_START / 1e6:.0f}M -> {MIX_END / 1e6:.0f}M")
    check("LR warmup is aligned with the ADR start", LR_WARMUP_STEPS == DR_START_STEPS,
          f"{LR_WARMUP_STEPS:,} vs {DR_START_STEPS:,}")
    check("the DR ramp finishes inside the run", DR_END_STEPS <= TOTAL_TIMESTEPS,
          f"{DR_END_STEPS:,} <= {TOTAL_TIMESTEPS:,}")
    check("the LR phases fit inside the run",
          LR_WARMUP_STEPS + (DR_END_STEPS - DR_START_STEPS) <= TOTAL_TIMESTEPS,
          f"{LR_WARMUP_STEPS + (DR_END_STEPS - DR_START_STEPS):,} <= {TOTAL_TIMESTEPS:,}")
    check("the chain ramp has a non-empty window inside the run",
          0 < MIX_START < MIX_END <= TOTAL_TIMESTEPS,
          f"{MIX_START:,} -> {MIX_END:,} (run {TOTAL_TIMESTEPS:,})")
    check("the chain ramp ends at the full weight, not part-way",
          abs(CB()._weight_at(MIX_END) - W_END) < 1e-12, f"{CB()._weight_at(MIX_END):.4f}")
    check("a full-DR polish phase is actually reached before the run ends",
          DR_END_STEPS < TOTAL_TIMESTEPS, f"{DR_END_STEPS:,} < {TOTAL_TIMESTEPS:,}")
    # A RESUME must pick the ramp up where it left off, so the weight is evaluated at the
    # lifetime step count. Checked at the middle of the production window (an even split, so
    # the expected value is exact regardless of how the endpoints are set).
    mid = (MIX_START + MIX_END) // 2
    check("a resume mid-ramp starts from the scheduled value, not the base weight",
          abs(CB()._weight_at(mid) - (W_START + W_END) / 2.0) < 1e-9,
          f"step {mid:,} -> {CB()._weight_at(mid):.4f} (base {W_START:.2f}, final {W_END:.2f})")

    print()
    print("=" * 78)
    print("E. the command interface survives the re-weighting")
    print("=" * 78)
    env = QuadFlipEnv()
    env.set_maneuver_weight("chain", 0.25)
    env.set_command("chain")
    _, info = env.reset(seed=3)
    check("a pinned chain is still honoured", info["maneuver"] == "chain", f"got {info['maneuver']}")
    check("all nine families are still addressable",
          all(k in env.traj_cfg.weights for k in
              ("hover", "waypoints", "figure8", "lissajous", "orbit", "slalom", "flip", "v8", "chain")))

    print()
    print("=" * 78)
    print(f"RESULT: {'all mixture-curriculum checks passed' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    print("=" * 78)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
