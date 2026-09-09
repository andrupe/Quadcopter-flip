import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import QuadFlipEnv, ACTOR_TOTAL_DIM, TOTAL_OBS_DIM
from asymmetric_policy import AsymmetricActorCriticPolicy

def test_checkpoint(chkpt_path: str, n_episodes: int = 3, dr_level: float = 0.0):
    vec_path = chkpt_path.replace(".zip", "_vecnormalize.pkl")
    if not os.path.isfile(chkpt_path) or not os.path.isfile(vec_path):
        return None

    env = QuadFlipEnv(
        episode_seconds=8.0,
        random_initial_state=False,
        arena_radius=2.5,
        hover_gain=1.0,
    )
    env.set_dr_level(dr_level)

    dummy_vec = DummyVecEnv([lambda: env])
    vec_norm = VecNormalize.load(vec_path, dummy_vec)
    vec_norm.training = False

    model = PPO.load(
        chkpt_path,
        custom_objects=dict(
            policy_class=AsymmetricActorCriticPolicy,
            actor_obs_dim=ACTOR_TOTAL_DIM,
        ),
        device="cpu",
    )

    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        max_omega_y = 0.0
        max_tilt = 0.0
        max_accum_pitch = 0.0
        total_rew = 0.0

        for step in range(800):
            obs_norm = vec_norm.normalize_obs(obs)
            actor_obs = obs_norm[:ACTOR_TOTAL_DIM]
            act, _ = model.predict(actor_obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            total_rew += r

            omega_y = abs(float(env.quad.omega[1]))
            tilt = float(np.degrees(np.arccos(np.clip(env.quad.dcm[2, 2], -1.0, 1.0))))
            accum = float(np.degrees(env.accumulated_pitch))

            max_omega_y = max(max_omega_y, omega_y)
            max_tilt = max(max_tilt, tilt)
            max_accum_pitch = max(max_accum_pitch, accum)

            if term or trunc:
                break

        results.append({
            "steps": env.steps,
            "rew": total_rew,
            "max_omega_y": max_omega_y,
            "max_tilt": max_tilt,
            "max_accum_pitch": max_accum_pitch,
            "has_inverted": env.has_inverted,
            "flip_completed": env.flip_completed,
            "term_reason": env.termination_reason,
            "pos": env.quad.pos.copy(),
        })

    return results

if __name__ == "__main__":
    logs_dir = os.path.join(_PROJECT_ROOT, "logs")
    checkpoints = [
        "rl_model_250000_steps.zip",
        "rl_model_500000_steps.zip",
        "rl_model_750000_steps.zip",
        "rl_model_1000000_steps.zip",
        "rl_model_1500000_steps.zip",
        "rl_model_2000000_steps.zip",
        "rl_model_2250000_steps.zip",
        "rl_model_2500000_steps.zip",
    ]

    print(f"{'Checkpoint':<30} | {'Flip?':<6} | {'Invert?':<7} | {'MaxRate (rad/s)':<15} | {'PeakTilt':<9} | {'AccumPitch':<11} | {'MeanRew':<8} | {'Status'}")
    print("-" * 115)

    for cp in checkpoints:
        full_path = os.path.join(logs_dir, cp)
        if not os.path.isfile(full_path):
            continue
        res = test_checkpoint(full_path, n_episodes=3, dr_level=0.0)
        if res is None:
            continue
        flip_rate = sum(1 for r in res if r["flip_completed"]) / len(res)
        inv_rate = sum(1 for r in res if r["has_inverted"]) / len(res)
        avg_rate = np.mean([r["max_omega_y"] for r in res])
        avg_tilt = np.mean([r["max_tilt"] for r in res])
        avg_accum = np.mean([r["max_accum_pitch"] for r in res])
        avg_rew = np.mean([r["rew"] for r in res])
        reasons = [r["term_reason"] for r in res]
        common_reason = max(set(reasons), key=reasons.count)

        print(f"{cp:<30} | {f'{flip_rate*100:.0f}%':<6} | {f'{inv_rate*100:.0f}%':<7} | {avg_rate:<15.1f} | {f'{avg_tilt:.0f}°':<9} | {f'{avg_accum:.0f}°':<11} | {avg_rew:<8.1f} | {common_reason}")
