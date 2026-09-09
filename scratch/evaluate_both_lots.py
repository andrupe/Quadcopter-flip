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

def test_checkpoint(model_name, n_episodes=10, deterministic=True):
    model_path = os.path.join(_PROJECT_ROOT, "logs", f"{model_name}.zip")
    stats_path = os.path.join(_PROJECT_ROOT, "logs", f"{model_name}_vecnormalize.pkl")
    if not os.path.isfile(model_path):
        return None
    
    model = PPO.load(
        model_path,
        custom_objects=dict(
            policy_class=AsymmetricActorCriticPolicy,
            actor_obs_dim=ACTOR_TOTAL_DIM,
        ),
    )
    env = QuadFlipEnv(episode_seconds=8.0, random_initial_state=False)
    env.set_dr_level(0.0)
    
    vec_norm = None
    if os.path.isfile(stats_path):
        vec_norm = VecNormalize.load(stats_path, DummyVecEnv([lambda: env]))
        vec_norm.training = False
        
    res = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        max_p = 0.0
        reached_90 = False
        reached_180 = False
        flip_done = False
        tot_rew = 0.0
        t_final = 0.0
        term_reason = "time_limit"
        
        while not done:
            obs_norm = vec_norm.normalize_obs(obs) if vec_norm else obs
            actor_obs = obs_norm[:ACTOR_TOTAL_DIM]
            act, _ = model.predict(actor_obs, deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(act)
            tot_rew += rew
            t_final = info["t"]
            
            p = abs(np.degrees(env.quad.euler[1]))
            if p > max_p:
                max_p = p
            if info.get("reached_90_deg", False):
                reached_90 = True
            if info.get("has_inverted", False):
                reached_180 = True
            if info.get("flip_completed", False):
                flip_done = True
                
            if term or trunc:
                done = True
                if term:
                    pos = info["position"]
                    xy = np.linalg.norm(pos[:2])
                    if pos[2] < 0.15:
                        term_reason = "ground_crash"
                    elif pos[2] > 2.5:
                        term_reason = "ceiling_breach"
                    elif xy > env.arena_radius - 0.05:
                        term_reason = "arena_breach"
                    else:
                        term_reason = "other_termination"
                        
        res.append({
            "max_p": max_p,
            "reached_90": reached_90,
            "reached_180": reached_180,
            "flip_done": flip_done,
            "term_reason": term_reason,
            "t_final": t_final,
            "tot_rew": tot_rew,
        })
    return res

def summarize(name, res):
    if not res:
        print(f"{name:35s}: File not found")
        return
    pitches = [r["max_p"] for r in res]
    n90 = sum(r["reached_90"] for r in res)
    n180 = sum(r["reached_180"] for r in res)
    nflip = sum(r["flip_done"] for r in res)
    terms = set(r["term_reason"] for r in res)
    t_end = np.mean([r["t_final"] for r in res])
    rews = np.mean([r["tot_rew"] for r in res])
    print(f"{name:35s}: Flips={nflip:2d}/{len(res):2d} | 90°={n90:2d}/{len(res):2d} | 180°={n180:2d}/{len(res):2d} | PeakPitch={np.max(pitches):5.1f}° (mean={np.mean(pitches):5.1f}°) | T={t_end:.2f}s | Terms={terms} | Rew={rews:.1f}")

print("=== 1. PREVIOUS LOT (Prior to reward fix) ===")
for step in [3000000, 4000000, 5000000, 5500000]:
    name = f"rl_model_{step}_steps"
    res_det = test_checkpoint(name, n_episodes=5, deterministic=True)
    summarize(f"{name} (Det)", res_det)
    res_stoch = test_checkpoint(name, n_episodes=5, deterministic=False)
    summarize(f"{name} (Stoch)", res_stoch)

print("\n=== 2. CURRENT NEW RUN (With 90°/180° reward fix) ===")
for step in [500000, 1000000, 1500000, 2000000, 2500000, 3000000]:
    name = f"rl_model_{step}_steps"
    res_det = test_checkpoint(name, n_episodes=5, deterministic=True)
    if res_det is not None:
        summarize(f"{name} (Det)", res_det)
        res_stoch = test_checkpoint(name, n_episodes=5, deterministic=False)
        summarize(f"{name} (Stoch)", res_stoch)
