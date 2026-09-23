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

    in  (env)  : [ actor stack (29*H) | ref_ff (3) | aux (4)   | privileged (44) ]
    out (vec)  : [ actor stack (29*H) | z (16)    | ref_ff (3) | aux (4) | privileged (44) ]

z is inserted immediately after the actor block, so:
  * the actor observation is STILL a clean prefix, obs[:actor_total_dim + z_dim + ref_ff_dim]
  * the critic receives the privileged block plus z for free
  * AsymmetricActorCriticPolicy needs no change beyond the actor dimension constant

The `ref_ff` block stays OUTSIDE the encoder (the encoder's input contract is [o_t | aux]
and is frozen) but INSIDE the actor prefix, which is why it sits between the two.

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
# scipy only supplies two accurately-rounded primitives for the NumPy fast path below
# (`expit` for sigmoid, `erf` for the head's EXACT-erf GELU - torch's nn.GELU() defaults to
# approximate='none'). A missing scipy degrades to the torch path, it does not fail.
try:
    from scipy.special import erf as _np_erf, expit as _np_sigmoid

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------------
# NumPy forward pass for the incremental (per-step) encoder path
#
# WHY: this runs once per outer vec step for every env, in the TRAINER process, on the
# critical path - the workers are idle while it runs. Measured at 0.1-0.3 ms per step,
# which is 3-9% of the wall-clock of a rollout, and almost all of it is torch dispatch
# overhead for a 15k-parameter model on a batch of ~8 vectors, not arithmetic.
#
# The batch/training path (`forward_sequence`) stays on torch: it is one call per episode,
# it needs autograd, and it is what the checkpoints are trained against. Only the
# one-step deployment path is reimplemented, and it is verified against torch at
# construction (see `_build_numpy_step`) so it cannot silently diverge.
# ---------------------------------------------------------------------------------
def _to_np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32, copy=True)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact erf GELU, matching torch's nn.GELU(approximate='none')."""
    return (0.5 * x * (1.0 + _np_erf(x * np.float32(0.7071067811865476)))).astype(np.float32)


def _gru_step_numpy(frame: np.ndarray, h: np.ndarray, p: dict) -> tuple:
    """
    One GRU cell + head step.

    Matches torch's `nn.GRU` gate equations exactly, including the bias split: the input
    bias b_ih belongs to the input term and b_hh to the HIDDEN term, so b_hh passes through
    the reset gate. Folding b_hh into the input part (a natural-looking simplification) lets
    it escape the reset gate and produces a visible error from the first step - that exact
    mistake was made and caught while porting this model to C.

    Rows of weight_ih/weight_hh are laid out [reset | update | new] in blocks of `width`.
    """
    W = h.shape[1]
    w_ih, w_hh = p["w_ih"], p["w_hh"]
    b_ih, b_hh = p["b_ih"], p["b_hh"]

    x_r = frame @ w_ih[0:W].T + b_ih[0:W]
    x_z = frame @ w_ih[W:2 * W].T + b_ih[W:2 * W]
    x_n = frame @ w_ih[2 * W:3 * W].T + b_ih[2 * W:3 * W]
    h_r = h @ w_hh[0:W].T + b_hh[0:W]
    h_z = h @ w_hh[W:2 * W].T + b_hh[W:2 * W]
    h_n = h @ w_hh[2 * W:3 * W].T + b_hh[2 * W:3 * W]

    r = _np_sigmoid(x_r + h_r).astype(np.float32)
    u = _np_sigmoid(x_z + h_z).astype(np.float32)
    n = np.tanh(x_n + r * h_n).astype(np.float32)
    h_next = ((1.0 - u) * n + u * h).astype(np.float32)

    # head: Linear(width -> width) -> GELU -> Linear(width -> z_dim) -> Tanh
    a = _gelu((h_next @ p["h1_w"].T + p["h1_b"]).astype(np.float32))
    out = (a @ p["h3_w"].T + p["h3_b"]).astype(np.float32)
    return np.tanh(out).astype(np.float32), h_next


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
        ref_ff_dim: int = 0,
        z_dim: int = 16,
        device: str = "cpu",
        num_threads: Optional[int] = 1,
    ):
        base_dim = int(venv.observation_space.shape[0])
        self.actor_total_dim = int(actor_total_dim)
        self.aux_dim = int(aux_dim)
        # The reference feed-forward block rides along between the actor frame and aux: the
        # ACTOR sees it (it is inside the output prefix), the ENCODER does not (its input
        # contract is [o_t | aux] and is frozen), and the privileged block follows it.
        self.ref_ff_dim = int(ref_ff_dim)
        self.priv_dim = base_dim - self.actor_total_dim - self.ref_ff_dim - self.aux_dim
        if self.priv_dim <= 0:
            raise ValueError(
                f"env observation has {base_dim} dims, which is not larger than "
                f"actor_total_dim({self.actor_total_dim}) + ref_ff_dim({self.ref_ff_dim}) "
                f"+ aux_dim({self.aux_dim})"
            )

        out_dim = self.actor_total_dim + int(z_dim) + self.ref_ff_dim + self.aux_dim + self.priv_dim
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
        # encoder is a GRU: the runtime state is a single [num_envs, hidden] array and a
        # step costs one GRU cell, independent of how long the episode has run.
        # Kept as NUMPY (not a torch tensor) because the fast path below is numpy and the
        # state is only ever consumed by it; the torch fallback converts on use.
        self.h = np.zeros((self.num_envs, self.hidden_dim), dtype=np.float32)
        self.z = np.zeros((self.num_envs, int(z_dim)), dtype=np.float32)

        # Try the numpy per-step path; it is used only if it reproduces torch here, on real
        # weights, before training starts.
        self.np_weights = self._build_numpy_step()
        self.use_numpy_step = self.np_weights is not None

        self.actor_total_dim_out = self.actor_total_dim
        self.aux_offset_in = self.actor_total_dim + self.ref_ff_dim
        self.ref_ff_offset_in = self.actor_total_dim
        self.z_offset_out = self.actor_total_dim
        self.ref_ff_offset_out = self.actor_total_dim + int(z_dim)
        self.aux_offset_out = self.ref_ff_offset_out + self.ref_ff_dim
        self.priv_offset_out = self.aux_offset_out + self.aux_dim

    # -- fast path construction / verification ---------------------------------------
    def _build_numpy_step(self) -> Optional[dict]:
        """
        Extract the deployed weights as numpy arrays and PROVE the numpy step matches torch.

        The check runs several sequential steps from a non-zero hidden state: a single step
        from h = 0 would leave every recurrent term (w_hh, b_hh) untested, which is exactly
        where a hand-written GRU goes wrong. Returns None - disabling the fast path - if
        anything is missing or disagrees.

        MEASURED, so the win is not overstated: 37.6 -> 24.7 us per step at batch 8, i.e.
        1.5x on this component and ~0.4% of end-to-end wall-clock (a rollout is ~3.3 ms per
        outer step and ~2048 outer steps). It is kept because it is verified-equal and
        strictly faster, NOT because it moves the needle - the trainer's serial bottleneck
        is elsewhere (worker barrier + info unpickling).
        """
        if not _HAVE_SCIPY:
            return None
        if self.device.type != "cpu":
            # The extracted arrays live on the host; keep one code path per device rather
            # than silently doing the arithmetic somewhere else than the model.
            return None
        try:
            gru, head = self.encoder.gru, self.encoder.head
            p = {
                "w_ih": _to_np(gru.weight_ih_l0), "w_hh": _to_np(gru.weight_hh_l0),
                "b_ih": _to_np(gru.bias_ih_l0), "b_hh": _to_np(gru.bias_hh_l0),
                "h1_w": _to_np(head[0].weight), "h1_b": _to_np(head[0].bias),
                "h3_w": _to_np(head[2].weight), "h3_b": _to_np(head[2].bias),
            }
        except (AttributeError, IndexError, TypeError):
            return None
        if p["w_ih"].shape[0] != 3 * self.hidden_dim:
            return None

        rng = np.random.default_rng(0)
        n = max(2, min(4, self.num_envs))
        frames = [rng.normal(0.0, 1.0, size=(n, ENCODER_IN_DIM)).astype(np.float32) for _ in range(3)]

        h_np = np.zeros((n, self.hidden_dim), dtype=np.float32)
        h_t = torch.zeros(n, self.hidden_dim, dtype=torch.float32, device=self.device)
        dz = dh = 0.0
        with torch.inference_mode():
            for f in frames:
                z_np, h_np = _gru_step_numpy(f, h_np, p)
                z_t, h_t = self.encoder.step(torch.from_numpy(f).to(self.device), h_t)
                dz = max(dz, float(np.max(np.abs(z_np - _to_np(z_t)))))
                dh = max(dh, float(np.max(np.abs(h_np - _to_np(h_t)))))

        tol = 1e-4
        if dz > tol or dh > tol:
            print(f"[LatentObsWrapper] numpy encoder step DISAGREES with torch "
                  f"(dz={dz:.2e}, dh={dh:.2e} > {tol:g}); using the torch path.")
            return None
        return p

    # -- recurrent state ----------------------------------------------------------
    def _frame(self, env_obs: np.ndarray) -> np.ndarray:
        """Env obs (single or batched) -> standardized encoder frame [.., 33]."""
        raw = frame_from_env_obs(env_obs, self.actor_total_dim, self.aux_dim, self.ref_ff_dim)
        return self.norm.standardize_frame(raw)

    def _advance(self, idx: np.ndarray, env_obs: np.ndarray) -> None:
        """One recurrent step for the selected envs, as a single batched forward."""
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        frame = self._frame(env_obs[idx])
        if self.use_numpy_step:
            z, h_next = _gru_step_numpy(frame, self.h[idx], self.np_weights)
        else:
            with torch.inference_mode():
                z_t, h_next_t = self.encoder.step(
                    torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)).to(self.device),
                    torch.from_numpy(self.h[idx]).to(self.device),
                )
            z, h_next = _to_np(z_t), _to_np(h_next_t)
        self.h[idx] = h_next
        self.z[idx] = z

    def _reset_state(self, idx: np.ndarray) -> None:
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        self.h[idx] = 0.0
        self.z[idx] = 0.0
    # -- output assembly ----------------------------------------------------------
    def _wrap(self, env_obs: np.ndarray) -> np.ndarray:
        actor = env_obs[:, : self.actor_total_dim]
        ff = env_obs[:, self.ref_ff_offset_in : self.ref_ff_offset_in + self.ref_ff_dim]
        aux = env_obs[:, self.aux_offset_in : self.aux_offset_in + self.aux_dim]
        priv = env_obs[:, self.aux_offset_in + self.aux_dim :]
        return np.concatenate([actor, self.z, ff, aux, priv], axis=1).astype(np.float32)

    @torch.inference_mode()
    def _wrap_one(self, raw_obs: np.ndarray, h_in: np.ndarray) -> np.ndarray:
        """
        Wrap a SINGLE observation against a supplied hidden state, without mutating it.

        Needed for the terminal observation: it must be encoded with the state the episode
        actually ended on, and the live state is about to be zeroed for the next episode.
        """
        frame = self._frame(np.asarray(raw_obs, dtype=np.float32)[None, :])
        if self.use_numpy_step:
            z, _ = _gru_step_numpy(frame, np.asarray(h_in, dtype=np.float32), self.np_weights)
        else:
            with torch.inference_mode():
                z_t, _ = self.encoder.step(
                    torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)).to(self.device),
                    torch.from_numpy(np.asarray(h_in, dtype=np.float32)).to(self.device),
                )
            z = _to_np(z_t)
        z = z[0]
        # Rebuild with z inserted after the actor frame and the feed-forward block passed
        # through in place: the output layout must match _wrap exactly, because the policy
        # is handed this array as a terminal observation.
        a = self.actor_total_dim
        f0 = self.ref_ff_offset_in
        return np.concatenate(
            [raw_obs[:a], z, raw_obs[f0:f0 + self.ref_ff_dim], raw_obs[f0 + self.ref_ff_dim:]]
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
        # The cached numpy weights are now stale; rebuild (and re-verify) them, or fall back.
        self.np_weights = self._build_numpy_step()
        self.use_numpy_step = self.np_weights is not None

    def get_histories(self) -> np.ndarray:
        """Current recurrent state per env - the GRU replacement for a frame window."""
        return self.h.copy()
