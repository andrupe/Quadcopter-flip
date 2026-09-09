from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation

try:
    from quad_flip_env import ACTOR_TOTAL_DIM
except ImportError:
    ACTOR_TOTAL_DIM = 51


class AsymmetricMlpExtractor(nn.Module):
    """
    Asymmetric MLP Extractor that decouples the input dimensions and layer architectures
    for the Actor (Policy) and Critic (Value) networks.

    - Actor network receives only the realistic onboard sensor features (actor_dim).
    - Critic network receives the full privileged state observation (critic_dim).
    """

    def __init__(
        self,
        actor_dim: int,
        critic_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module] = nn.Tanh,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)

        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [128, 128])
            vf_layers_dims = net_arch.get("vf", [512, 256, 128])
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        # Build Actor (Policy) network from actor_dim
        policy_net: List[nn.Module] = []
        last_layer_dim_pi = actor_dim
        for curr_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_dim
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.latent_dim_pi = last_layer_dim_pi

        # Build Critic (Value) network from critic_dim (privileged)
        value_net: List[nn.Module] = []
        last_layer_dim_vf = critic_dim
        for curr_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_dim
        self.value_net = nn.Sequential(*value_net).to(device)
        self.latent_dim_vf = last_layer_dim_vf

    def forward(self, features: Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]) -> Tuple[th.Tensor, th.Tensor]:
        if isinstance(features, tuple):
            pi_features, vf_features = features
            return self.forward_actor(pi_features), self.forward_critic(vf_features)
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """
    Asymmetric Actor-Critic (AAC) Policy for Stable-Baselines3 PPO.

    Allows the Critic to observe privileged ground-truth simulation variables and domain
    randomization parameters, while strictly constraining the Actor to realistic onboard
    sensor observations (e.g. IMU, relative altitude, previous actions).

    Enables seamless inference: model.predict() can be called with EITHER the full privileged
    observation vector OR the compact onboard actor observation slice.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        actor_obs_dim: int = ACTOR_TOTAL_DIM,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.actor_obs_dim = int(actor_obs_dim)
        if net_arch is None:
            net_arch = dict(pi=[128, 128], vf=[512, 256, 128])

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            share_features_extractor=False,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Instantiates the AsymmetricMlpExtractor with decoupled actor and critic input dims."""
        actor_dim = self.actor_obs_dim
        critic_dim = self.observation_space.shape[0]  # Full observation with privileged state
        self.mlp_extractor = AsymmetricMlpExtractor(
            actor_dim=actor_dim,
            critic_dim=critic_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def extract_features(
        self, obs: PyTorchObs, features_extractor: Optional[Any] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Extracts decoupled features:
        - Actor features: first actor_obs_dim elements of the observation vector.
        - Critic features: full observation vector (includes privileged physics & ADR features).
        """
        if isinstance(obs, dict):
            pi_obs = obs["actor"]
            vf_obs = obs["critic"]
        else:
            if obs.shape[-1] == self.actor_obs_dim:
                pi_obs = obs
                vf_obs = obs  # Fallback if evaluated without privileged features
            else:
                pi_obs = obs[..., :self.actor_obs_dim]
                vf_obs = obs

        return pi_obs, vf_obs

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """Computes action distribution using only the actor observation slice."""
        if isinstance(obs, dict):
            pi_obs = obs["actor"]
        elif obs.shape[-1] == self.actor_obs_dim:
            pi_obs = obs
        else:
            pi_obs = obs[..., :self.actor_obs_dim]

        latent_pi = self.mlp_extractor.forward_actor(pi_obs)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """Computes value estimation using the full privileged observation vector."""
        if isinstance(obs, dict):
            vf_obs = obs["critic"]
        else:
            vf_obs = obs

        latent_vf = self.mlp_extractor.forward_critic(vf_obs)
        return self.value_net(latent_vf)

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[PyTorchObs, bool]:
        """
        Gracefully handles converting observations to PyTorch tensors even when
        only the compact actor-only observation slice (shape == actor_obs_dim) is supplied.
        """
        if isinstance(observation, np.ndarray) and observation.shape[-1] == self.actor_obs_dim:
            obs_tensor = th.as_tensor(observation, dtype=th.float32, device=self.device)
            is_vectorized = observation.ndim > 1
            if not is_vectorized:
                obs_tensor = obs_tensor.unsqueeze(0)
            return obs_tensor, is_vectorized

        return super().obs_to_tensor(observation)
