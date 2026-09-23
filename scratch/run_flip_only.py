"""PROBE: can this policy learn to flip at all?

WHY THIS EXISTS
---------------
A pinned flip is ignored outright by the trained policy: the reference inverts completely
(min body-z world component = -1.000 in 10/10 runs) and the vehicle's minimum never goes
below +0.60, i.e. it never rotates past 90 deg - while still scoring ~65-79% of the ceiling,
because hovering through a flip collects position + velocity + action (4.5 of 7.3).

That leaves two very different explanations, and the objective cannot tell them apart:

  (a) the policy CAN rotate but the reward makes NOT rotating a good deal -> fix the reward;
  (b) the policy cannot discover the rotation at all                  -> fix the curriculum.

THE EXPERIMENT: train ONE manoeuvre, from scratch, with the reward UNCHANGED. Same reward,
same seed-free PPO, same encoder - the only difference is that every episode is a flip. If a
flip appears within a few million steps, (a) holds and the mixture is what buries it. If it
still never rotates when flips are the ONLY task, the problem is deeper than the mixture and
no amount of re-weighting will fix it.

WHAT IT DOES NOT TOUCH
----------------------
* `quad_flip_model.zip`, `logs/rl_model_*_steps.zip` - PERIODIC CHECKPOINTS ARE DISABLED.
  `CheckpointCallback` writes `rl_model_<steps>_steps.zip` into `logs/` no matter what
  `model_name` is (only the CSVs and plots get the run suffix), so a 1M/2M probe would
  overwrite the real run's checkpoints. `checkpoint_freq` beyond the run length prevents it,
  and this script ASSERTS that rather than trusting it.
* The production schedules: DR starts at DR_START_STEPS (10M) and the chain ramp at
  CHAIN_MIX_START_STEPS (20M), so at a few million steps neither fires and the probe stays
  on the nominal plant with the clean-phase LR.

Run:  .venv/bin/python scratch/run_flip_only.py [steps]        (default 3,000,000)
Then: .venv/bin/python scratch/diag_flip_execution.py flip_probe.zip
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROBE_MODEL = "flip_probe"
PROBE_FAMILY = "flip"
NO_CHECKPOINTS = 10 ** 9


def main() -> int:
    import train as T

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000

    # --- the safety check that matters -------------------------------------------------
    if NO_CHECKPOINTS <= steps:
        raise SystemExit("checkpoint_freq must exceed the run length, or the probe would "
                         "overwrite the real run's logs/rl_model_*_steps.zip")
    if PROBE_MODEL == T.MODEL_NAME:
        raise SystemExit(f"probe model_name must differ from MODEL_NAME ({T.MODEL_NAME!r}) "
                         "or it would overwrite the real run's outputs")

    T.PIN_MANEUVER = PROBE_FAMILY
    if PROBE_FAMILY not in T.TrajectoryConfig().weights:
        raise SystemExit(f"{PROBE_FAMILY!r} is not a known manoeuvre family")

    print("=" * 78)
    print("FLIP-ONLY PROBE")
    print("=" * 78)
    print(f"  every episode pinned to : {PROBE_FAMILY}")
    print(f"  steps                   : {steps:,}")
    print(f"  reward                  : UNCHANGED (this is the point)")
    print(f"  periodic checkpoints    : OFF (nothing written to logs/rl_model_*)")
    print(f"  DR starts at            : {T.DR_START_STEPS:,}  -> will not fire")
    print(f"  chain ramp starts at    : {T.CHAIN_MIX_START_STEPS:,}  -> will not fire")
    print(f"  writes                  : {_PROJECT_ROOT}/{PROBE_MODEL}.zip, "
          f"logs/training_metrics_{PROBE_MODEL}.csv, logs/eval_metrics_{PROBE_MODEL}.csv")
    print("=" * 78, flush=True)

    T.train(
        total_timesteps=steps,
        num_workers=8,
        model_name=PROBE_MODEL,
        checkpoint_freq=NO_CHECKPOINTS,
    )

    print("\n" + "=" * 78)
    print("PROBE DONE - now measure whether the vehicle actually rotates:")
    print(f"  .venv/bin/python scratch/diag_flip_execution.py {PROBE_MODEL}.zip")
    print("  PASS = min dcm22 goes negative (info['has_inverted'], set below -0.2)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
