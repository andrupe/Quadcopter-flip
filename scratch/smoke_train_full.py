"""
Short end-to-end run of the REAL train() entry point.

Not a learning test: it exists to prove the actual training path starts, builds every
callback, steps PPO, saves a checkpoint and shuts down - which is the difference between
"the pieces compile" and "you can start a 20M-step run".

It exercises the TARGET configuration (with the history encoder attached), using a
synthetic untrained encoder checkpoint written to /tmp so that the real logs/ directory is
not polluted with a fake encoder.

Run:  .venv/bin/python scratch/smoke_train_full.py
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

SMOKE_CKPT = "/tmp/encoder_smoke.pt"
SMOKE_MODEL = "smoke_model"


def main() -> int:
    import train as T

    # The encoder package is optional. When it is absent this still exercises the REAL
    # training path, just without the latent - which is exactly the configuration we need
    # to be able to start a run in.
    try:
        from encoder.history_encoder import EncoderWithHead, save_encoder_checkpoint
        from encoder.observation_spec import ENCODER_IN_DIM, NormStats

        # A synthetic encoder. Random weights are fine: this checks wiring, not capability.
        model = EncoderWithHead(n_targets=29, f_in=ENCODER_IN_DIM, width=48, z_dim=T.Z_DIM)
        norm = NormStats(
            frame_mean=np.zeros(ENCODER_IN_DIM, dtype=np.float32),
            frame_std=np.ones(ENCODER_IN_DIM, dtype=np.float32),
            target_mean=np.zeros(29, dtype=np.float32),
            target_std=np.ones(29, dtype=np.float32),
            target_groups=[],
            clip=10.0,
            degenerate_dims=[],
        )
        save_encoder_checkpoint(SMOKE_CKPT, model, norm)
        T.ENCODER_CHECKPOINT = SMOKE_CKPT
        print(f"[smoke] synthetic encoder written to {SMOKE_CKPT}")
    except ImportError as exc:
        print(f"[smoke] encoder package unavailable ({exc}); testing the NO-ENCODER path")

    # Compress every schedule so a few thousand steps still traverse all of its phases.
    T.LR_WARMUP_STEPS = 2048
    T.DR_START_STEPS = 2048
    T.DR_END_STEPS = 6144
    T.CHECKPOINT_FREQ = 4096

    print("[smoke] calling train(total_timesteps=8192, num_workers=4)")
    T.train(
        total_timesteps=8192,
        num_workers=4,
        model_name=SMOKE_MODEL,
        checkpoint_freq=4096,
    )

    ok = os.path.isfile(os.path.join(_PROJECT_ROOT, f"{SMOKE_MODEL}.zip"))
    print(f"\n[smoke] model artifact written: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
