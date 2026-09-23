# -*- coding: utf-8 -*-
"""
Build a throwaway RANDOM-weight checkpoint with a chosen actor `net_arch`.

Why: every committed checkpoint predates `ref_ff` (and the actor-frame anchoring), so it
cannot be exported at all - `export_policy.py` refuses it. That leaves the export ->
`policy_net.c` -> `policy_host_check.py` chain untestable after an architecture change
unless a current-architecture model exists. This makes one whose only job is to be
numerically DIFFED, so the weights are irrelevant (and the head is scaled down so every
action channel is live without saturating the clip, which would hide errors).

    .venv/bin/python scratch/make_probe_model.py            # writes both probe models
    .venv/bin/python scratch/make_probe_model.py 32         # just pi=[32]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from gymnasium import spaces
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_ROOT, os.path.join(_ROOT, "Simulation"), os.path.join(_ROOT, "Simulation", "encoder")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asymmetric_policy import AsymmetricActorCriticPolicy  # noqa: E402

# The deployable with-encoder actor width: o_t(29) + z(16) + ref_ff(3).
ACTOR_DIM = 48
# Model name -> actor hidden widths. The SECOND is the pre-change architecture, kept so the
# two-hidden-layer path stays covered too.
PROBES = {
    "probe_pi32": [32],
    "probe_pi128x2": [128, 128],
}


class _NullEnv(Env):
    """A never-stepped placeholder: PPO needs an env, the weights are what we want."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1e3, 1e3, (ACTOR_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(ACTOR_DIM, np.float32), {}

    def step(self, action):
        return np.zeros(ACTOR_DIM, np.float32), 0.0, False, False, {}


def build(pi: list, path: str) -> None:
    model = PPO(
        policy=AsymmetricActorCriticPolicy,
        env=DummyVecEnv([_NullEnv]),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs=dict(
            actor_obs_dim=ACTOR_DIM,
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=pi, vf=[512, 256, 128]),
            log_std_init=-0.5,
        ),
        verbose=0,
        device="cpu",
    )
    # Scale the action head so the clip is not the only thing the diff ever sees.
    with torch.no_grad():
        model.policy.action_net.weight.mul_(0.05)
    model.save(path)
    widths = [int(m.weight.shape[0]) for m in model.policy.mlp_extractor.policy_net
              if isinstance(m, torch.nn.Linear)]
    print(f"wrote {os.path.relpath(path, _ROOT)}  pi={pi}  actual hidden widths={widths}")


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = os.path.join(_ROOT, "logs")
    os.makedirs(out_dir, exist_ok=True)
    for name, pi in PROBES.items():
        if wanted and wanted != str(pi[0]):
            continue
        build(pi, os.path.join(out_dir, name + ".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
