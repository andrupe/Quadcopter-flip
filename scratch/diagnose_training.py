import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_PROJECT_ROOT = "/Users/apan/University/IntelligentControl/Quadcopter-flip"
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for p in [_PROJECT_ROOT, _SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quad_flip_env import QuadFlipEnv, ACTOR_TOTAL_DIM
from asymmetric_policy import AsymmetricActorCriticPolicy
import utils

checkpoints = [
    2000000,
    3000000,
    3500000,
    4000000,
    4500000,
]

def eval_checkpoint(ckpt_steps, num_episodes=5, deterministic=True):
    model_path = os.path.join(_PROJECT_ROOT, "logs", f"rl_model_{ckpt_steps}_steps.zip")
    stats_path = os.path.join(_PROJECT_ROOT, "logs", f"rl_model_{ckpt_steps}_steps_vecnormalize.pkl")
    if not os.path.isfile(model_path):
        return f"File {model_path} does not exist"
    
    model = PPO.load(
        model_path,
        custom_objects=dict(
            policy_class=AsymmetricActorCriticPolicy,
            actor_obs_dim=ACTOR_TOTAL_DIM,
        ),
    )
    
    env = QuadFlipEnv(
        episode_seconds=8.0,
        random_initial_state=False,
        random_initial_pos=False,
        random_initial_vel=False,
        random_initial_att=False,
    )
    env.set_dr_level(0.0)  # clean sim evaluation
    
    vec_norm = None
    if os.path.isfile(stats_path):
        dummy_vec = DummyVecEnv([lambda: env])
        vec_norm = VecNormalize.load(stats_path, dummy_vec)
        vec_norm.training = False

    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        max_pitch = 0.0
        min_cos_pitch = 1.0
        flip_completed = False
        term_reason = "time_limit"
        t_final = 0.0
        tot_rew = 0.0
        max_xy = 0.0
        
        while not done:
            obs_norm = vec_norm.normalize_obs(obs) if vec_norm else obs
            actor_obs = obs_norm[:ACTOR_TOTAL_DIM]
            action, _ = model.predict(actor_obs, deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(action)
            tot_rew += rew
            t_final = info.get("t", 0.0)
            
            r, p, y = env.quad.euler
            pitch_deg = abs(np.degrees(p))
            cos_p = np.cos(p)
            if pitch_deg > max_pitch:
                max_pitch = pitch_deg
            if cos_p < min_cos_pitch:
                min_cos_pitch = cos_p
                
            pos = info.get("position", [0, 0, 0])
            xy_dist = np.linalg.norm(pos[:2])
            if xy_dist > max_xy:
                max_xy = xy_dist
                
            reached_90 = info.get("reached_90_deg", False)
            reached_180 = info.get("has_inverted", False)
            if info.get("flip_completed", False):
                flip_completed = True
                
            if term or trunc:
                done = True
                if term:
                    if pos[2] < 0.15:
                        term_reason = "ground_crash"
                    elif pos[2] > 3.0:
                        term_reason = "ceiling_breach"
                    elif xy_dist >= env.arena_radius - 0.05:
                        term_reason = "arena_xy_breach"
                    else:
                        term_reason = "other_termination"
                else:
                    term_reason = "time_limit"
                    
        results.append({
            "max_pitch": max_pitch,
            "min_cos_pitch": min_cos_pitch,
            "reached_90": reached_90,
            "reached_180": reached_180,
            "flip_completed": flip_completed,
            "term_reason": term_reason,
            "t_final": t_final,
            "tot_rew": tot_rew,
            "max_xy": max_xy,
        })
    return results

print("Evaluating Checkpoints (Deterministic = True)...")
for ckpt in checkpoints:
    res = eval_checkpoint(ckpt, num_episodes=5, deterministic=True)
    if isinstance(res, str):
        print(f"Ckpt {ckpt}: {res}")
        continue
    pitches = [r["max_pitch"] for r in res]
    terms = [r["term_reason"] for r in res]
    times = [r["t_final"] for r in res]
    flips = sum(r["flip_completed"] for r in res)
    max_xys = [r["max_xy"] for r in res]
    rews = [r["tot_rew"] for r in res]
    print(f"Step {ckpt:7d}: Flips={flips}/{len(res)} | MaxPitch: mean={np.mean(pitches):5.1f}°, max={np.max(pitches):5.1f}° | MaxXY: mean={np.mean(max_xys):.2f}m | T_end: mean={np.mean(times):.2f}s | Terms: {set(terms)} | Rew: mean={np.mean(rews):.1f}")

print("\nEvaluating Checkpoint 4.5M (Deterministic = False - stochastic actions)...")
res_stoch = eval_checkpoint(4500000, num_episodes=10, deterministic=False)
if not isinstance(res_stoch, str):
    pitches = [r["max_pitch"] for r in res_stoch]
    terms = [r["term_reason"] for r in res_stoch]
    times = [r["t_final"] for r in res_stoch]
    flips = sum(r["flip_completed"] for r in res_stoch)
    max_xys = [r["max_xy"] for r in res_stoch]
    rews = [r["tot_rew"] for r in res_stoch]
    print(f"Step 4.5M Stoch: Flips={flips}/{len(res_stoch)} | MaxPitch: mean={np.mean(pitches):5.1f}°, max={np.max(pitches):5.1f}° | MaxXY: mean={np.mean(max_xys):.2f}m | T_end: mean={np.mean(times):.2f}s | Terms: {set(terms)} | Rew: mean={np.mean(rews):.1f}")
