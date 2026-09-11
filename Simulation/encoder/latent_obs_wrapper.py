"""
LatentObsWrapper - inject the frozen history encoder's output into the vec observation.

    SubprocVecEnv  ->  LatentObsWrapper  ->  VecNormalize  ->  PPO

WHY A VecEnvWrapper AND NOT THE ENV ITSELF
There is no difference in "frozenness" (the gradient is cut either way), but there is a
large difference in cost and in coupling. Running the encoder inside each SubprocVecEnv
worker would multiply the forward cost by the worker count; running it here puts all
workers' frames into ONE batched pass and keeps Simulation/quad_flip_env.py completely
torch-free - which also matters because SubprocVecEnv uses the fork/spawn start method,
so a set_num_threads() call in the trainer process does not reach the workers.

LAYOUT

    in  (env)  : [ actor stack (29*H) | aux (4)   | privileged (44) ]
    out (vec)  : [ actor stack (29*H) | z (16)    | aux (4) | privileged (44) ]

z is inserted immediately after the actor block, so:
  * the actor observation is STILL a clean prefix, obs[:actor_total_dim + z_dim]
  * the critic receives the privileged block plus z for free
  * AsymmetricActorCriticPolicy needs no change beyond the actor dimension constant

RECURRENT STATE, NOT A FRAME WINDOW
The encoder is a GRU, so this wrapper carries one hidden vector per environment and steps
it once per environment step. No observation window is buffered at all: the per-step cost
is a single GRU cell and does not grow with episode length. (The TCN this replaced had to
keep 100+ frames alive per env and re-ran its convolutions over all of them on every
update - the exact cost profile that made it undeployable on a 168 MHz Cortex-M4.)

HISTORY HANDLING
On a `done` step, SB3's SubprocVecEnv worker already reset the env and returns the RESET
observation while `info` still holds the terminal one. The recurrent state for that env is
therefore zeroed BEFORE the new frame is consumed, so the next episode begins from exactly
the same state it would at takeoff - carrying the terminal state across the boundary would
hand the encoder a history that never happened on the aircraft either.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

from .observation_spec import ACTOR_FRAME_DIM, AUX_DIM, ENCODER_IN_DIM, NormStats, frame_from_env_obs


class LatentObsWrapper(VecEnvWrapper):
    """
    :param venv: wrapped vectorized environment
    :param encoder_path: checkpoint written by train_encoder.py (weights + frozen norm)
    :param actor_total_dim: size of the actor block in the wrapped obs (29 * OBS_HISTORY_LEN)
    :param aux_dim: size of the encoder aux block (4)
    :param z_dim: latent width; must match the checkpoint
    :param device: torch device for the encoder
    :param num_threads: sets torch CPU threads. Default 1, matching train.py.
    """

    def __init__(
        self,
        venv,
        encoder_path: str,
        actor_total_dim: int = ACTOR_FRAME_DIM,
        aux_dim: int = AUX_DIM,
        z_dim: int = 16,
        device: str = "cpu",
        num_threads: Optional[int] = 1,
    ):
        base_dim = int(venv.observation_space.shape[0])
        self.actor_total_dim = int(actor_total_dim)
        self.aux_dim = int(aux_dim)
        self.priv_dim = base_dim - self.actor_total_dim - self.aux_dim
        if self.priv_dim <= 0:
            raise ValueError(
                f"env observation has {base_dim} dims, which is not larger than "
                f"actor_total_dim({self.actor_total_dim}) + aux_dim({self.aux_dim})"
            )

        out_dim = self.actor_total_dim + int(z_dim) + self.aux_dim + self.priv_dim
        super().__init__(
            venv,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(out_dim,), dtype=np.float32),
        )

        if num_threads is not None:
            torch.set_num_threads(int(num_threads))
        self.device = torch.device(device)
        self.encoder_path = encoder_path

        from .history_encoder import load_encoder_checkpoint

        encoder, norm, ckpt = load_encoder_checkpoint(encoder_path, device=str(self.device))
        if int(encoder.z_dim) != int(z_dim):
            raise ValueError(
                f"checkpoint {encoder_path} has z_dim={encoder.z_dim}, wrapper was built with z_dim={z_dim}"
            )
        if int(encoder.f_in) != ENCODER_IN_DIM:
            raise ValueError(
                f"checkpoint {encoder_path} expects {encoder.f_in} input dims, this build uses {ENCODER_IN_DIM}. "
                f"The encoder was trained against a different observation frame and must be retrained."
            )
        self.encoder = encoder
        self.norm: NormStats = norm
        self.hidden_dim = int(encoder.width)
        self.trained_meta = ckpt.get("extra", {})

        # Persistent recurrent state, one vector per env. This is the entire reason the
        # encoder is a GRU: the runtime state is a single [num_envs, hidden] tensor and a
        # step costs one GRU cell, independent of how long the episode has run.
        self.h = torch.zeros(self.num_envs, self.hidden_dim, dtype=torch.float32, device=self.device)
        self.z = np.zeros((self.num_envs, int(z_dim)), dtype=np.float32)

        self.actor_total_dim_out = self.actor_total_dim
        self.aux_offset_in = self.actor_total_dim
        self.z_offset_out = self.actor_total_dim
        self.aux_offset_out = self.actor_total_dim + int(z_dim)
        self.priv_offset_out = self.aux_offset_out + self.aux_dim

    # -- recurrent state ----------------------------------------------------------
    def _frame(self, env_obs: np.ndarray) -> np.ndarray:
        """Env obs (single or batched) -> standardized encoder frame [.., 33]."""
        raw = frame_from_env_obs(env_obs, self.actor_total_dim, self.aux_dim)
        return self.norm.standardize_frame(raw)

    def _advance(self, idx: np.ndarray, env_obs: np.ndarray) -> None:
        """One recurrent step for the selected envs, as a single batched forward."""
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        frame = self._frame(env_obs[idx])
        x = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)).to(self.device)
        with torch.inference_mode():
            z, h_next = self.encoder.step(x, self.h[idx])
        self.h[idx] = h_next
        self.z[idx] = z.detach().to("cpu").numpy()

    def _reset_state(self, idx: np.ndarray) -> None:
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        self.h[idx] = 0.0
        self.z[idx] = 0.0

    # -- output assembly ----------------------------------------------------------
    def _wrap(self, env_obs: np.ndarray) -> np.ndarray:
        actor = env_obs[:, : self.actor_total_dim]
        aux = env_obs[:, self.aux_offset_in : self.aux_offset_in + self.aux_dim]
        priv = env_obs[:, self.aux_offset_in + self.aux_dim :]
        return np.concatenate([actor, self.z, aux, priv], axis=1).astype(np.float32)

    @torch.inference_mode()
    def _wrap_one(self, raw_obs: np.ndarray, h_in: torch.Tensor) -> np.ndarray:
        """
        Wrap a SINGLE observation against a supplied hidden state, without mutating it.

        Needed for the terminal observation: it must be encoded with the state the episode
        actually ended on, and the live state is about to be zeroed for the next episode.
        """
        frame = self._frame(np.asarray(raw_obs, dtype=np.float32)[None, :])
        x = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)).to(self.device)
        z, _ = self.encoder.step(x, h_in)
        z = z.detach().to("cpu").numpy()[0]
        return np.concatenate(
            [raw_obs[: self.actor_total_dim], z, raw_obs[self.actor_total_dim:]]
        ).astype(np.float32)

    # -- VecEnv API ---------------------------------------------------------------
    def reset(self) -> np.ndarray:
        env_obs = self.venv.reset()
        all_idx = np.arange(self.num_envs)
        self._reset_state(all_idx)
        self._advance(all_idx, env_obs)
        return self._wrap(env_obs)

    def step_wait(self):
        env_obs, rewards, dones, infos = self.venv.step_wait()
        env_obs = np.asarray(env_obs)
        dones = np.asarray(dones, dtype=bool)
        done_idx = np.flatnonzero(dones)

        # SB3 bootstraps a truncated episode's value from infos[i]["terminal_observation"],
        # which the WORKER captured and which is therefore the RAW env observation. This
        # wrapper's output is wider (it inserts z), so handing the raw vector to the policy
        # raises "Unexpected observation shape (77,) ... please use (93,)" the first time
        # ANY environment finishes an episode.
        #
        # The terminal obs must be encoded with the hidden state the episode ENDED on, so it
        # has to be wrapped here - before _reset_state zeroes that state and before _advance
        # consumes the reset observation.
        for i in done_idx:
            term = infos[i].get("terminal_observation")
            if term is not None:
                infos[i]["terminal_observation"] = self._wrap_one(
                    np.asarray(term, dtype=np.float32), self.h[i : i + 1]
                )

        # Order matters: reset first, then step, so a new episode's first frame is
        # processed from a zero state exactly as it would be at takeoff.
        self._reset_state(done_idx)
        self._advance(np.arange(self.num_envs), env_obs)
        return self._wrap(env_obs), rewards, dones, infos

    # -- runtime control (used by the optional on-policy refresh loop) -------------
    def get_latent(self) -> np.ndarray:
        """Current z per env, for distribution-shift probes."""
        return self.z.copy()

    def load_encoder_weights(self, state_dict: dict) -> None:
        """
        Hot-swap encoder weights. Intended for the optional EMA refresh loop: blend
        offline, then push, and watch KL / entropy / reward for one iteration to make
        sure the actor can track the shifted input distribution.
        """
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

    def get_histories(self) -> np.ndarray:
        """Current recurrent state per env - the GRU replacement for a frame window."""
        return self.h.detach().to("cpu").numpy().copy()
