"""
Single-environment latent injection for scripts that do NOT run inside a VecEnv.

Training injects z with `LatentObsWrapper`, which lives between the vectorised env and
VecNormalize. Evaluation, PID tuning and checkpoint benchmarking drive a RAW `QuadFlipEnv`
step by step instead, because they need the un-reset env state for telemetry after a
`done` (wrapping in DummyVecEnv to reach LatentObsWrapper makes every `env.*` read return
the POST-reset episode - see the note in Simulation/evaluate.py). Those scripts therefore
need the same `[o_t | z]` assembly done by hand, which is what this class is for.

It is deliberately the SAME contract as the wrapper:

    in  (raw env) : [ o_t (29) | aux (4) | privileged (44) ]
    out           : [ o_t (29) | z (16) | aux (4) | privileged (44) ]

and it drives the encoder through its incremental `step()` - the deployment path a flight
controller would use - not through a batch API that only exists offline. Keeping one
implementation here means evaluate.py, tune_rate_pid.py and benchmark_checkpoints.py
cannot drift apart, which is exactly how the pre-migration `obs[:51]` slices survived in
tune_rate_pid.py after the observation layout changed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .history_encoder import load_encoder_checkpoint
from .observation_spec import ACTOR_FRAME_DIM, AUX_DIM, frame_from_env_obs


class LatentInjector:
    """
    Owns the frozen encoder's recurrent state for ONE environment.

    :param encoder_path: checkpoint written by train_encoder.py (weights + frozen norm)
    :param actor_dim: width of the actor block in the wrapped observation (o_t, 29)
    :param aux_dim: width of the encoder aux block (4)
    :param z_dim: latent width; must match the checkpoint
    :param device: torch device for the encoder
    """

    def __init__(
        self,
        encoder_path: str,
        actor_dim: int = ACTOR_FRAME_DIM,
        aux_dim: int = AUX_DIM,
        z_dim: int = 16,
        device: str = "cpu",
    ):
        encoder, norm, _ckpt = load_encoder_checkpoint(encoder_path, device=device)
        if int(encoder.z_dim) != int(z_dim):
            raise ValueError(
                f"encoder at {encoder_path} has z_dim={encoder.z_dim}, expected {z_dim}"
            )
        if int(encoder.f_in) != int(actor_dim) + int(aux_dim):
            raise ValueError(
                f"encoder at {encoder_path} expects f_in={encoder.f_in}, but this build "
                f"feeds it actor_dim({actor_dim}) + aux_dim({aux_dim})"
            )
        self.encoder = encoder
        self.norm = norm
        self.actor_dim = int(actor_dim)
        self.aux_dim = int(aux_dim)
        self.z_dim = int(z_dim)
        self.z = np.zeros((1, self.z_dim), dtype=np.float32)
        self.h = self.encoder.init_state(1)

    @torch.no_grad()
    def reset(self) -> None:
        """Zero the recurrent state. Must be called whenever the env resets."""
        self.h = self.encoder.init_state(1)
        self.z = np.zeros((1, self.z_dim), dtype=np.float32)

    @torch.no_grad()
    def inject(self, env_obs: np.ndarray) -> np.ndarray:
        """Raw env observation -> [o_t | z | aux | privileged], same as the training wrapper."""
        env_obs = np.asarray(env_obs, dtype=np.float32)
        raw = frame_from_env_obs(env_obs, self.actor_dim, self.aux_dim)
        frame = self.norm.standardize_frame(raw)[None, :]
        x = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        z, self.h = self.encoder.step(x, self.h)
        self.z = z.numpy()
        return np.concatenate(
            [env_obs[: self.actor_dim], self.z[0], env_obs[self.actor_dim :]]
        ).astype(np.float32)

    def get_history(self) -> Optional[np.ndarray]:
        """Current recurrent state (diagnostics only)."""
        return self.h.detach().cpu().numpy().copy() if self.h is not None else None
