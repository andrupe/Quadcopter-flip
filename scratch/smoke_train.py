"""
End-to-end smoke test of the PPO wiring: SubprocVecEnv -> LatentObsWrapper -> VecNormalize
-> PPO, against the trajectory-tracking env.

This does NOT check learning. It checks the things that are silently wrong otherwise:

  A. SHAPES AGREE. The actor input the policy sees must be exactly
     ACTOR_TOTAL_DIM + z_dim, and the critic must see the full vector. If the wrapper is
     missing or misplaced, PPO still runs - it just trains on the wrong slice.
  B. THE ACTOR BLOCK IS [o_t | z], followed by aux and privileged.
  C. RECURRENT STATE RESETS PER EPISODE. A done env must start its next episode from a
     zero hidden state, not inherit the terminal one.
  D. PPO ACTUALLY STEPS. A short learn() run completes without shape or dtype errors.

Run:  .venv/bin/python scratch/smoke_train.py
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

Z_DIM = 16
CKPT = "/tmp/_smoke_encoder.pt"
N_ENVS = 3

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    from encoder.history_encoder import EncoderWithHead, save_encoder_checkpoint
    from encoder.latent_obs_wrapper import LatentObsWrapper
    from encoder.observation_spec import (
        ACTOR_FRAME_DIM,
        AUX_DIM,
        ENCODER_IN_DIM,
        NormStats,
    )
    from quad_flip_env import (
        ACTOR_TOTAL_DIM,
        PRIVILEGED_OBS_DIM,
        TOTAL_OBS_DIM,
        QuadFlipEnv,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)

    # A synthetic (untrained) encoder checkpoint, purely to exercise the plumbing.
    model = EncoderWithHead(n_targets=29, f_in=ENCODER_IN_DIM, width=48, z_dim=Z_DIM)
    norm = NormStats(
        frame_mean=np.zeros(ENCODER_IN_DIM, dtype=np.float32),
        frame_std=np.ones(ENCODER_IN_DIM, dtype=np.float32),
        target_mean=np.zeros(29, dtype=np.float32),
        target_std=np.ones(29, dtype=np.float32),
        target_groups=[],
        clip=10.0,
        degenerate_dims=[],
    )
    save_encoder_checkpoint(CKPT, model, norm)

    print("=" * 78)
    print("A. shape agreement")
    print("=" * 78)
    check("encoder frame dim matches the actor frame", ACTOR_FRAME_DIM == ACTOR_TOTAL_DIM == 29,
          f"spec {ACTOR_FRAME_DIM}, env {ACTOR_TOTAL_DIM}")
    check("encoder input is actor frame + aux", ENCODER_IN_DIM == ACTOR_TOTAL_DIM + AUX_DIM == 33,
          f"ENCODER_IN_DIM = {ENCODER_IN_DIM}")
    check("env vector is actor + aux + privileged",
          TOTAL_OBS_DIM == ACTOR_TOTAL_DIM + AUX_DIM + PRIVILEGED_OBS_DIM == 77,
          f"TOTAL_OBS_DIM = {TOTAL_OBS_DIM}")

    raw = make_vec_env(lambda: QuadFlipEnv(), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    check("raw vec env observation matches the env declaration",
          raw.observation_space.shape == (TOTAL_OBS_DIM,), f"{raw.observation_space.shape}")

    wrapped = LatentObsWrapper(raw, encoder_path=CKPT, z_dim=Z_DIM)
    expected = ACTOR_TOTAL_DIM + Z_DIM + AUX_DIM + PRIVILEGED_OBS_DIM
    check("wrapper output is actor + z + aux + privileged",
          wrapped.observation_space.shape == (expected,),
          f"{wrapped.observation_space.shape}, expected ({expected},)")

    obs = wrapped.reset()
    check("reset returns the wrapped shape", obs.shape == (N_ENVS, expected), f"{obs.shape}")

    # Step a few times BEFORE comparing the actor block against ground truth. At reset the
    # Lighthouse estimate IS truth (the filter is initialised there and no measurement has
    # been taken yet), which mirrors a real vehicle whose EKF is converged on the pad
    # before takeoff. The estimate only diverges once observe() starts adding noise.
    for _ in range(5):
        obs, _, _, _ = wrapped.step(np.zeros((N_ENVS, 4), dtype=np.float32))

    print()
    print("=" * 78)
    print("B. block layout")
    print("=" * 78)
    a0 = ACTOR_TOTAL_DIM
    aux0 = a0 + Z_DIM
    priv0 = aux0 + AUX_DIM

    o_t = obs[:, :a0]
    z_blk = obs[:, a0:aux0]
    aux_blk = obs[:, aux0:priv0]
    priv_blk = obs[:, priv0:]
    check("o_t block is 29 wide", o_t.shape[1] == 29, f"{o_t.shape}")
    check("z block is 16 wide", z_blk.shape[1] == 16, f"{z_blk.shape}")
    check("aux block is 4 wide", aux_blk.shape[1] == 4, f"{aux_blk.shape}")
    check("privileged block is 44 wide", priv_blk.shape[1] == PRIVILEGED_OBS_DIM, f"{priv_blk.shape}")
    check("z is finite", bool(np.all(np.isfinite(z_blk))), "")
    check("the actor slice contains no privileged ground truth",
          not np.allclose(o_t[:, :3], priv_blk[:, :3], atol=1e-7),
          "o_t position is the Lighthouse estimate; privileged holds ground truth")
    check("estimated velocity channels are populated", not np.allclose(o_t[:, 10:13], 0.0), "")

    print()
    print("=" * 78)
    print("C. recurrent state resets per episode")
    print("=" * 78)
    single = LatentObsWrapper(
        DummyVecEnv([lambda: QuadFlipEnv()]),
        encoder_path=CKPT, z_dim=Z_DIM,
    )
    single.reset()
    h0 = float(np.abs(single.get_histories()).max())
    check("hidden state accumulates after the first step", h0 > 0.0, f"max |h| = {h0:.4f}")

    done_seen = False
    for _ in range(1500):
        _, _, dones, _ = single.step(np.zeros((1, 4), dtype=np.float32))
        if bool(dones[0]):
            done_seen = True
            break
    check("an episode terminated within the horizon", done_seen, "needed for the reset check")
    if done_seen:
        h_done = float(np.abs(single.get_histories()[0]).max())
        # Right after a done, the worker returned the RESET observation, so the state must
        # be the result of ONE frame from zero - not the accumulated terminal state.
        check("hidden state was reset at the episode boundary", h_done < 1.0,
              f"max |h| right after done = {h_done:.4f}")

    print()
    print("=" * 78)
    print("D. PPO runs end to end")
    print("=" * 78)
    from stable_baselines3 import PPO
    from asymmetric_policy import AsymmetricActorCriticPolicy

    vec = VecNormalize(wrapped, norm_obs=False, norm_reward=False, clip_obs=10.0)
    actor_obs_dim = ACTOR_TOTAL_DIM + Z_DIM
    model = PPO(
        policy=AsymmetricActorCriticPolicy,
        env=vec,
        n_steps=256,
        batch_size=128,
        n_epochs=2,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(
            actor_obs_dim=actor_obs_dim,
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[128, 128], vf=[512, 256, 128]),
            log_std_init=-0.5,
        ),
        verbose=0,
        device="cpu",
    )
    check("policy built with the latent-augmented actor dim",
          model.policy.actor_obs_dim == actor_obs_dim, f"{model.policy.actor_obs_dim}")
    model.learn(total_timesteps=512)
    check("PPO completed a short learn() run", model.num_timesteps >= 512,
          f"{model.num_timesteps} steps")

    first_w = model.policy.mlp_extractor.policy_net[0].weight
    check("actor's first layer expects exactly [o_t | z]",
          int(first_w.shape[1]) == actor_obs_dim,
          f"{int(first_w.shape[1])} vs {actor_obs_dim}")

    print()
    print("=" * 78)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: PPO wiring smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
