"""
Hand a PPO checkpoint exactly the actor input it was trained on.

The actor observation is a PREFIX of the env observation vector:

    no encoder : [ o_t (29) | aux (4) | privileged (44) ]            -> actor takes [:29]
    encoder    : [ o_t (29) | z (16) | aux (4) | privileged (44) ]   -> actor takes [:45]

Slicing a raw env observation at the trained width is therefore only correct when the
history encoder is NOT part of the pipeline. With the encoder, the z block has to be
produced by running the frozen GRU over the observation stream, and the raw vector's
`aux`/`privileged` blocks must not be mistaken for it. That mistake is silent - the policy
still runs, it is just conditioned on the wrong 16 numbers - and it is exactly what the
pre-migration `obs[:51]` slice in `Simulation/tune_rate_pid.py` did after the observation
layout changed from a 3-frame 51-dim stack to 29 (or 45) dims.

Every consumer of a checkpoint (evaluate.py, tune_rate_pid.py, benchmark_checkpoints.py)
goes through this module so the three cannot drift apart again:

    arch = read_checkpoint_arch(model_path)      # what does the file expect?
    actor_input = ActorInput(arch.actor_dim)     # how do I feed it?
    model = load_checkpoint(model_path)          # rebuild + load the policy
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import ACTOR_TOTAL_DIM, ENCODER_AUX_DIM, TOTAL_OBS_DIM  # noqa: E402
from asymmetric_policy import AsymmetricActorCriticPolicy  # noqa: E402
from encoder.latent_injector import LatentInjector  # noqa: E402

Z_DIM: int = 16
ENCODER_CHECKPOINT: str = os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt")
DEFAULT_NET_ARCH: Dict[str, list] = {"pi": [128, 128], "vf": [512, 256, 128]}


def read_checkpoint_arch(model_path: str) -> Tuple[Optional[int], Optional[Dict[str, list]]]:
    """
    Read the actor input width and the pi/vf layer widths straight out of the weights.

    Reading the first policy layer is the only source of truth that cannot go stale: the
    saved `policy_kwargs` are metadata that a loader is free to ignore, while the tensor
    shapes are what `set_parameters` actually has to match.

    Returns (actor_dim, net_arch); both are None if the file could not be inspected.
    """
    actor_dim: Optional[int] = None
    net_arch: Optional[Dict[str, list]] = None
    try:
        with zipfile.ZipFile(model_path, "r") as z:
            if "policy.pth" not in z.namelist():
                return None, None
            with z.open("policy.pth") as f:
                sd = torch.load(io.BytesIO(f.read()), map_location="cpu")
        key = "mlp_extractor.policy_net.0.weight"
        if key in sd:
            actor_dim = int(sd[key].shape[1])
        pi_dims, vf_dims = [], []
        idx = 0
        while f"mlp_extractor.policy_net.{idx}.weight" in sd:
            pi_dims.append(int(sd[f"mlp_extractor.policy_net.{idx}.weight"].shape[0]))
            idx += 2
        idx = 0
        while f"mlp_extractor.value_net.{idx}.weight" in sd:
            vf_dims.append(int(sd[f"mlp_extractor.value_net.{idx}.weight"].shape[0]))
            idx += 2
        if pi_dims:
            net_arch = {"pi": pi_dims, "vf": vf_dims if vf_dims else list(DEFAULT_NET_ARCH["vf"])}
    except Exception:
        return None, None
    return actor_dim, net_arch


def load_checkpoint(model_path: str, device: str = "cpu"):
    """
    `PPO.load` with the custom objects `AsymmetricActorCriticPolicy` needs.

    The policy kwargs are overridden with the dims read from the weights, so a checkpoint
    loads even when the stored policy_kwargs are missing the `actor_obs_dim` argument
    (SB3 only substitutes `custom_objects` keys that are present in the saved JSON, so
    passing `actor_obs_dim` alone is NOT enough - it has to go in through policy_kwargs).
    """
    from stable_baselines3 import PPO

    actor_dim, net_arch = read_checkpoint_arch(model_path)
    custom: Dict[str, Any] = {"policy_class": AsymmetricActorCriticPolicy}
    if actor_dim is not None:
        policy_kwargs: Dict[str, Any] = {
            "actor_obs_dim": actor_dim,
            "activation_fn": torch.nn.Tanh,
            "net_arch": net_arch or dict(DEFAULT_NET_ARCH),
        }
        custom["actor_obs_dim"] = actor_dim
        custom["policy_kwargs"] = policy_kwargs
    return PPO.load(model_path, custom_objects=custom, device=device)


class ActorInput:
    """
    Turns raw `QuadFlipEnv` observations into the exact vector to pass to `model.predict`.

    :param actor_dim: width the checkpoint expects (29, or 45 with the encoder)
    :param encoder_path: frozen encoder checkpoint; only used when actor_dim == 45
    :param device: torch device for the encoder

    Raises ValueError for any other width: a pre-migration checkpoint's weights are
    meaningless against this environment and slicing them would produce a wrong
    evaluation with no error.
    """

    def __init__(
        self,
        actor_dim: int,
        encoder_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.actor_dim = int(actor_dim)
        self.encoder_path: Optional[str] = None
        self.injector: Optional[LatentInjector] = None

        if self.actor_dim == ACTOR_TOTAL_DIM + Z_DIM:
            self.encoder_path = encoder_path or ENCODER_CHECKPOINT
            if not os.path.isfile(self.encoder_path):
                raise ValueError(
                    f"checkpoint expects a {Z_DIM}-dim latent, but no encoder was found "
                    f"at {self.encoder_path}.\n"
                    "Rebuild it:\n"
                    "  .venv/bin/python Simulation/encoder/collect_data.py\n"
                    "  .venv/bin/python Simulation/encoder/train_encoder.py"
                )
            self.injector = LatentInjector(
                self.encoder_path, ACTOR_TOTAL_DIM, ENCODER_AUX_DIM, Z_DIM, device=device
            )
        elif self.actor_dim != ACTOR_TOTAL_DIM:
            raise ValueError(
                f"checkpoint expects a {self.actor_dim}-dim actor input, but this build "
                f"produces {ACTOR_TOTAL_DIM} (no encoder) or {ACTOR_TOTAL_DIM + Z_DIM} "
                f"(with the encoder).\n"
                "  This checkpoint predates the trajectory-tracking migration - its "
                "observation layout no longer exists, so its weights cannot be evaluated.\n"
                "  Retrain with:  .venv/bin/python Simulation/train.py"
            )

    # -- episode lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        """Reset the encoder's recurrent state. Call whenever the env resets."""
        if self.injector is not None:
            self.injector.reset()

    # -- per-step -------------------------------------------------------------------
    def prepare(self, raw_obs: np.ndarray, vec_norm: Optional[Any] = None) -> np.ndarray:
        """
        Raw env observation -> the vector `model.predict` expects for the ACTOR.

        Order matches training (LatentObsWrapper sits INSIDE VecNormalize): inject z
        first, then normalize, then slice. A VecNormalize wrapper with norm_obs=False is
        a no-op, so this is safe for the current configuration too.
        """
        return self.prepare_full(raw_obs, vec_norm)[: self.actor_dim]

    def prepare_full(self, raw_obs: np.ndarray, vec_norm: Optional[Any] = None) -> np.ndarray:
        """Raw env observation -> [o_t | z | aux | privileged], normalized but NOT sliced."""
        obs = np.asarray(raw_obs, dtype=np.float32)
        if self.injector is not None:
            if obs.shape[-1] != TOTAL_OBS_DIM:
                raise ValueError(
                    f"LatentInjector expects the raw {TOTAL_OBS_DIM}-dim env observation, "
                    f"got shape {obs.shape}"
                )
            obs = self.injector.inject(obs)
        if vec_norm is not None:
            obs = vec_norm.normalize_obs(obs)
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] < self.actor_dim:
            raise ValueError(
                f"observation has {obs.shape[-1]} dims, checkpoint needs {self.actor_dim}"
            )
        return obs

    @property
    def training_obs_dim(self) -> int:
        """Width of the observation the training-time VecNormalize statistics cover.

        LatentObsWrapper sits INSIDE VecNormalize, so with the encoder attached the
        statistics describe [o_t | z | aux | privileged], not the raw env observation.
        This is the width a stand-in venv must carry to load those statistics
        successfully (`VecNormalize.load` checks the observation space shape).
        """
        return TOTAL_OBS_DIM + (Z_DIM if self.injector is not None else 0)

    def describe(self) -> str:
        if self.injector is not None:
            return (
                f"actor input = [o_t({ACTOR_TOTAL_DIM}) | z({Z_DIM})] via "
                f"{os.path.basename(self.encoder_path or '')}"
            )
        return f"actor input = o_t({ACTOR_TOTAL_DIM}) (no encoder)"
