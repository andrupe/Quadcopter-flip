"""
Unit tests for the self-supervised dynamics encoder pipeline.
Verifies:
1. DynamicsPredictorHead and EncoderWithDynamicsHead shapes and forward rollouts.
2. Checkpoint save and load roundtrip.
3. Downstream compatibility with LatentObsWrapper, LatentInjector, and ActorInput.
"""

import os
import sys
import tempfile
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, _THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from encoder import (
    ACTION_DIM,
    ACTION_INDICES,
    ENCODER_IN_DIM,
    PHYS_STATE_DIM,
    PHYS_STATE_GROUPS,
    PHYS_STATE_INDICES,
    DynamicsPredictorHead,
    EncoderWithDynamicsHead,
    HistoryEncoder,
    LatentInjector,
    NormStats,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)
from actor_input import ActorInput, ACTOR_DIM_WITH_ENCODER


def test_dynamics_head_forward():
    print("Testing DynamicsPredictorHead forward pass...")
    head = DynamicsPredictorHead(
        state_dim=PHYS_STATE_DIM,
        action_dim=ACTION_DIM,
        z_dim=16,
        hidden_dim=64,
    )
    B, T = 4, 10
    s = torch.randn(B, T, PHYS_STATE_DIM)
    a = torch.randn(B, T, ACTION_DIM)
    z = torch.randn(B, T, 16)
    delta = head(s, a, z)
    assert delta.shape == (B, T, PHYS_STATE_DIM), f"Expected shape {(B, T, PHYS_STATE_DIM)}, got {delta.shape}"
    print("  -> DynamicsPredictorHead forward pass OK")


def test_encoder_with_dynamics_rollout():
    print("Testing EncoderWithDynamicsHead rollout sequence...")
    model = EncoderWithDynamicsHead(
        f_in=ENCODER_IN_DIM,
        width=48,
        z_dim=16,
        state_dim=PHYS_STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=64,
    )
    B, T = 2, 20
    x = torch.randn(B, T, ENCODER_IN_DIM)
    s = torch.randn(B, T, PHYS_STATE_DIM)
    a = torch.randn(B, T - 1, ACTION_DIM)
    horizon = 5

    z, preds, targets = model.rollout_sequence(x, s, a, horizon=horizon)
    assert z.shape == (B, T, 16), f"Expected z shape {(B, T, 16)}, got {z.shape}"
    assert len(preds) == horizon, f"Expected {horizon} predictions, got {len(preds)}"
    assert len(targets) == horizon, f"Expected {horizon} targets, got {len(targets)}"
    assert preds[0].shape == (B, T - horizon, PHYS_STATE_DIM), f"Expected {(B, T - horizon, PHYS_STATE_DIM)}, got {preds[0].shape}"
    print("  -> EncoderWithDynamicsHead rollout sequence OK")


def test_checkpoint_roundtrip_and_downstream_loading():
    print("Testing checkpoint save and load roundtrip with downstream wrappers...")
    model = EncoderWithDynamicsHead(
        f_in=ENCODER_IN_DIM,
        width=48,
        z_dim=16,
        state_dim=PHYS_STATE_DIM,
        action_dim=ACTION_DIM,
    )
    # Dummy norm stats
    dummy_frames = np.random.randn(50, ENCODER_IN_DIM).astype(np.float32)
    dummy_targets = np.random.randn(50, PHYS_STATE_DIM).astype(np.float32)
    norm = NormStats.from_frames_and_targets(dummy_frames, dummy_targets, PHYS_STATE_GROUPS)

    from quad_flip_env import ACTOR_FRAME_MODE
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_encoder_gru.pt")
        save_encoder_checkpoint(
            ckpt_path,
            model,
            norm,
            extra={"mode": "self_supervised", "horizon": 5, "frame_mode": ACTOR_FRAME_MODE},
        )
        assert os.path.exists(ckpt_path), "Checkpoint file was not created"

        # Test load_encoder_checkpoint
        encoder, loaded_norm, ckpt = load_encoder_checkpoint(ckpt_path)
        assert isinstance(encoder, HistoryEncoder), f"Expected HistoryEncoder, got {type(encoder)}"
        assert encoder.z_dim == 16, f"Expected z_dim=16, got {encoder.z_dim}"
        assert encoder.f_in == ENCODER_IN_DIM, f"Expected f_in={ENCODER_IN_DIM}, got {encoder.f_in}"
        assert ckpt["config"]["mode"] == "self_supervised"
        print("  -> load_encoder_checkpoint loaded cleanly")

        # Test LatentInjector
        injector = LatentInjector(ckpt_path)
        injector.reset()
        from quad_flip_env import TOTAL_OBS_DIM
        dummy_env_obs = np.random.randn(TOTAL_OBS_DIM).astype(np.float32)
        injected_obs = injector.inject(dummy_env_obs)
        assert len(injected_obs) == TOTAL_OBS_DIM + 16, f"Expected {TOTAL_OBS_DIM + 16}, got {len(injected_obs)}"
        print("  -> LatentInjector inject OK")

        # Test ActorInput with this checkpoint
        actor_input = ActorInput(actor_dim=ACTOR_DIM_WITH_ENCODER, encoder_path=ckpt_path)
        actor_input.reset()
        actor_obs = actor_input.prepare(dummy_env_obs)
        assert actor_obs.shape == (ACTOR_DIM_WITH_ENCODER,), f"Expected {(ACTOR_DIM_WITH_ENCODER,)}, got {actor_obs.shape}"
        print("  -> ActorInput prepare with encoder OK")

    print("ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_dynamics_head_forward()
    test_encoder_with_dynamics_rollout()
    test_checkpoint_roundtrip_and_downstream_loading()
