"""Why did no flip happen in evaluate.py?

Two separate questions, measured rather than argued:
  1. Is a flip even DRAWN? evaluate.py has MANEUVER = None, so the family is sampled at
     every reset - with one episode you only see a flip ~1 time in 5.
  2. When a flip IS drawn, does the vehicle actually rotate? A flip is ~2.2 s of reference
     and at full DR the policy is at its weakest here, so aborts are expected.

evaluate.py defaults to DR_LEVEL = 1 (full randomization). This sweeps dr = 0 and dr = 1.

The flip's own tracker records whether the ROTATION HAPPENED (`flips_completed`,
`accumulated_roll`) and the env reports a termination reason, so both halves are visible.

Run:  .venv/bin/python scratch/diag_flip_execution.py [model.zip]
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Simulation"))

from quad_flip_env import QuadFlipEnv                            # noqa: E402
from actor_input import ActorInput, load_checkpoint, read_checkpoint_arch   # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def pick_model(argv):
    if argv and os.path.isfile(argv[0]):
        return argv[0]
    logs = os.path.join(_ROOT, "logs")
    import re
    zips = [os.path.join(logs, f) for f in os.listdir(logs) if f.endswith(".zip")]
    zips.sort(key=lambda p: int(re.search(r"(\d+)_steps", p).group(1))
              if re.search(r"(\d+)_steps", p) else os.path.getmtime(p))
    return zips[-1]


def main():
    model_path = pick_model(sys.argv[1:])
    dim, _ = read_checkpoint_arch(model_path)
    ai = ActorInput(dim, encoder_path=os.path.join(_ROOT, "logs", "encoder_gru.pt"))
    model = load_checkpoint(model_path)
    print(f"model: {os.path.relpath(model_path, _ROOT)}   actor {dim} dims\n")

    # --- 1. what does the DEFAULT (MANEUVER=None) actually draw? --------------------
    print("=== what evaluate.py's default actually samples (maneuver=None) ===")
    env = QuadFlipEnv(episode_seconds=15.0)
    drawn = []
    for seed in range(12):
        env.reset(seed=seed)
        drawn.append(env.traj.maneuver.kind)
    counts: dict = {}
    for k in drawn:
        counts[k] = counts.get(k, 0) + 1
    print(f"  12 draws: {counts}")
    print(f"  flips drawn: {counts.get('flip', 0)}/12"
          f"   -> with NUM_EPISODES=1 you would see one about "
          f"{100*counts.get('flip', 0)/12:.0f}% of the time\n")

    # --- 2. when a flip IS drawn, does it rotate? ----------------------------------
    # BOTH sides are tracked. The vehicle's `min dcm22` says whether it inverted; the
    # REFERENCE's says whether it was ever asked to. Without the second column a policy
    # that ignores a real flip and a sampler that quietly handed back a hover look identical.
    print("=== a PINNED flip: does the REFERENCE invert, and does the vehicle follow? ===")
    print(f"  {'dr':>4} {'seed':>4} {'kind':>6} {'dur':>5} {'steps':>6} {'t/dur':>7} "
          f"{'ended':>10} {'reason':>14} | {'REF dcm22':>10} {'ref|w|max':>10} "
          f"| {'VEH dcm22':>10} {'vehTilt':>8} {'rew/step':>9}")
    print("  " + "-" * 132)
    for dr in (0.0, 1.0):
        for seed in SEEDS:
            env = QuadFlipEnv(episode_seconds=15.0, maneuver="flip")
            env.set_dr_level(dr)
            obs, _info = env.reset(seed=seed, options={"maneuver": "flip"})
            ai.reset()
            span = float(env.traj.duration) if env.traj is not None else float("nan")
            kind = env.traj.maneuver.kind if env.traj is not None else "none"
            min_dcm22, peak_tilt, rew, n = 1.0, 0.0, 0.0, 0
            ref_min_d22, ref_peak_w = 1.0, 0.0
            while True:
                act, _ = model.predict(ai.prepare(obs), deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                rew += float(r)
                n += 1
                d22 = float(env.quad.dcm[2, 2])
                min_dcm22 = min(min_dcm22, d22)
                peak_tilt = max(peak_tilt, float(np.degrees(np.arccos(np.clip(d22, -1, 1)))))
                if env.ref is not None:
                    ref_min_d22 = min(ref_min_d22, float(env.ref.R[2, 2]))
                    ref_peak_w = max(ref_peak_w, float(np.linalg.norm(env.ref.omega)))
                if term or trunc:
                    break
            reason = info.get("termination_reason", "none") if term else "-"
            print(f"  {dr:>4.1f} {seed:>4} {kind:>6} {span:>5.2f} {n:>6} {env.t / span:>7.2f} "
                  f"{'terminated' if term else 'truncated':>10} {str(reason):>14} | "
                  f"{ref_min_d22:>+10.3f} {ref_peak_w:>10.2f} "
                  f"| {min_dcm22:>+10.3f} {peak_tilt:>7.0f}d {rew / max(1, n):>9.2f}")
        print()


if __name__ == "__main__":
    main()
