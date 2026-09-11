"""
History-encoder package: a frozen causal GRU that infers unmodelled plant parameters from
the observation stream, plus the plumbing to inject its output into PPO.

    SubprocVecEnv  ->  LatentObsWrapper  ->  VecNormalize  ->  PPO

Modules:
    observation_spec     input frame contract + frozen normalization constants
    history_encoder      the GRU itself, plus losses and checkpoint I/O
    latent_obs_wrapper   VecEnvWrapper that injects z into the observation
    collect_data         pretraining data driver (mixed controllers)
    train_encoder        supervised pretraining + R^2 gating

Why the encoder exists: the policy runs on an airframe whose mass, thrust scale, motor
time constants, CoM offset and battery sag are all unknown and change between flights.
A hand-tuned controller is robust to some of that; a learned policy is not, because it is
only ever as good as the state estimate it is handed. The encoder turns the recent
sensor stream into a low-dimensional latent that PPO can condition on.
"""

from .observation_spec import (  # noqa: F401
    ACTOR_FRAME_DIM,
    AUX_DIM,
    ENCODER_IN_DIM,
    NormStats,
    frame_from_env_obs,
)
from .history_encoder import (  # noqa: F401
    EncoderWithHead,
    HistoryEncoder,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)

__all__ = [
    "ACTOR_FRAME_DIM",
    "AUX_DIM",
    "ENCODER_IN_DIM",
    "NormStats",
    "frame_from_env_obs",
    "EncoderWithHead",
    "HistoryEncoder",
    "load_encoder_checkpoint",
    "save_encoder_checkpoint",
]
