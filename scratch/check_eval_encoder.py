"""
End-to-end chain test: encoder -> PPO (with latent) -> evaluate (with latent).

This is the join that is easiest to get wrong and hardest to notice, because every
individual stage works fine on its own. Specifically it checks that:

  1. train.py attaches LatentObsWrapper and builds the actor at ACTOR_TOTAL_DIM + z.
  2. The checkpoint that train.py writes carries that actor width.
  3. Simulation/evaluate.py DETECTS that width, attaches the wrapper itself, and feeds the
     actor the same [o_t | z] slice it was trained on.

Step 3 is the one that used to be silently wrong: without the wrapper the eval path would
have sliced a raw environment observation and evaluated the policy on the wrong input.

Run:  .venv/bin/python scratch/check_eval_encoder.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

SMOKE_CKPT = "/tmp/encoder_chain.pt"
SMOKE_DATA = os.path.join(_PROJECT_ROOT, "logs", "encoder_data_smoke")
CHAIN_MODEL = "eval_chain_model"

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    # --- build a real (if tiny) encoder checkpoint -------------------------------------
    print("=" * 78)
    print("A. encoder checkpoint")
    print("=" * 78)
    # Prefer the REAL encoder when it exists: that validates the artifact PPO will actually
    # load at run time, rather than a 2-epoch stand-in built just for this test.
    real_ckpt = os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt")
    if os.path.isfile(real_ckpt):
        encoder_for_test = real_ckpt
        print(f"  using the real trained encoder: {real_ckpt}")
    else:
        if not os.path.isdir(SMOKE_DATA):
            print(f"  [skip] no real encoder and {SMOKE_DATA} missing; "
                  f"run collect_data.py --smoke first")
            return 0
        from encoder.train_encoder import train as train_enc
        train_enc(data_dir=SMOKE_DATA, out_path=SMOKE_CKPT, epochs=2, batch_size=16)
        encoder_for_test = SMOKE_CKPT
    check("encoder checkpoint available", os.path.isfile(encoder_for_test), encoder_for_test)

    # --- train PPO with the latent ------------------------------------------------------
    print()
    print("=" * 78)
    print("B. train PPO with the latent attached")
    print("=" * 78)
    import train as T
    T.ENCODER_CHECKPOINT = encoder_for_test
    T.LR_WARMUP_STEPS = 512
    T.DR_START_STEPS = 512
    T.DR_END_STEPS = 1536
    T.train(total_timesteps=2048, num_workers=2, model_name=CHAIN_MODEL, checkpoint_freq=10 ** 9)

    model_zip = os.path.join(_PROJECT_ROOT, f"{CHAIN_MODEL}.zip")
    check("PPO checkpoint written", os.path.isfile(model_zip), model_zip)

    # The saved policy's first layer width IS the actor input width.
    import zipfile, io
    with zipfile.ZipFile(model_zip) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    width = int(sd["mlp_extractor.policy_net.0.weight"].shape[1])
    from quad_flip_env import ACTOR_TOTAL_DIM
    expected = ACTOR_TOTAL_DIM + 16
    check("saved policy expects [o_t | z]", width == expected, f"{width} vs {expected}")

    # --- evaluate it --------------------------------------------------------------------
    print()
    print("=" * 78)
    print("C. evaluate detects the latent and attaches the wrapper")
    print("=" * 78)
    import evaluate as EV
    EV.ENCODER_CHECKPOINT = encoder_for_test
    EV.NUM_EPISODES = 2

    # A missing encoder must refuse rather than silently mis-slice.
    EV.ENCODER_CHECKPOINT = "/tmp/definitely_not_here.pt"
    refused = False
    try:
        EV.evaluate(model_name=f"{CHAIN_MODEL}.zip", num_episodes=1, loop=False,
                    show_viewer=False, show_plots=False, dr_level=0.0)
    except SystemExit as exc:
        refused = "no encoder was found" in str(exc)
    check("missing encoder is refused, not silently mis-sliced", refused,
          "raised SystemExit" if refused else "did NOT raise")

    # And with the encoder present it must actually run the requested episodes and stop.
    EV.ENCODER_CHECKPOINT = encoder_for_test
    ran = False
    try:
        EV.evaluate(model_name=f"{CHAIN_MODEL}.zip", num_episodes=3, loop=False,
                    show_viewer=False, show_plots=False, dr_level=0.0)
        ran = True
    except SystemExit as exc:
        print(f"    SystemExit: {exc}")
    except Exception as exc:
        print(f"    {type(exc).__name__}: {exc}")
    check("evaluation completed with the encoder attached", ran, "")

    # --- cleanup ------------------------------------------------------------------------
    for p in (model_zip, os.path.join(_PROJECT_ROOT, f"{CHAIN_MODEL}_vecnormalize.pkl")):
        if os.path.isfile(p):
            os.remove(p)
    if os.path.isdir(os.path.join(_PROJECT_ROOT, "logs")):
        for f in os.listdir(os.path.join(_PROJECT_ROOT, "logs")):
            if f.startswith(CHAIN_MODEL):
                os.remove(os.path.join(_PROJECT_ROOT, "logs", f))
    if os.path.isfile(SMOKE_CKPT):
        os.remove(SMOKE_CKPT)

    print()
    print("=" * 78)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: encoder -> PPO -> evaluate chain verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
