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

checkpoints = [
    5000000,
    7000000,
    9000000,
    10000000,
    11000000,
]

def run_evaluation(ckpt_steps, dr_level=0.0, num_episodes=5, deterministic=True):
    model_path = os.path.join(_PROJECT_ROOT, "logs", f"rl_model_{ckpt_steps}_steps.zip")
    stats_path = os.path.join(_PROJECT_ROOT, "logs", f"rl_model_{ckpt_steps}_steps_vecnormalize.pkl")
    if not os.path.isfile(model_path):
        return None
    
    model = PPO.load(
        model_path,
        custom_objects=dict(
            policy_class=AsymmetricActorCriticPolicy,
            actor_obs_dim=ACTOR_TOTAL_DIM,
        ),
    )
    
    env = QuadFlipEnv(
        episode_seconds=8.0,
        random_initial_state=True if dr_level > 0.0 else False,
        arena_radius=2.5,
    )
    env.set_dr_level(dr_level)
    
    vec_norm = VecNormalize.load(stats_path, DummyVecEnv([lambda: env]))
    vec_norm.training = False
    
    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        t_flip = None
        tot_rew = 0.0
        steps = 0
        max_xy = 0.0
        term_cause = "completed_flight"
        
        while not done and steps < 400:
            obs_norm = vec_norm.normalize_obs(obs)
            actor_obs = obs_norm[:ACTOR_TOTAL_DIM]
            act, _ = model.predict(actor_obs, deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(act)
            tot_rew += rew
            steps += 1
            
            pos = info["position"]
            xy_dist = np.linalg.norm(pos[:2])
            if xy_dist > max_xy:
                max_xy = xy_dist
                
            if info.get("flip_completed", False) and t_flip is None:
                t_flip = info["t"]
                
            if term or trunc:
                done = True
                if term:
                    if pos[2] < 0.15:
                        term_cause = "ground_crash"
                    elif pos[2] > 2.5:
                        term_cause = "ceiling_breach"
                    elif xy_dist >= env.arena_radius - 0.05:
                        term_cause = "arena_breach"
                    else:
                        term_cause = "other_termination"
                else:
                    term_cause = "time_limit"
                    
        results.append({
            "t_flip": t_flip,
            "pitch_deg": np.degrees(env.accumulated_pitch),
            "final_z": info["position"][2],
            "max_xy": max_xy,
            "final_pos": info["position"],
            "tot_rew": tot_rew,
            "flight_time": info["t"],
            "term_cause": term_cause,
        })
    return results

def print_table(title, ckpt_list, dr_level):
    print(f"\n{'='*75}")
    print(f" {title} (DR Level = {dr_level:.2f})")
    print(f"{'='*75}")
    print(f"{'Checkpoint':16s} | {'Flips':7s} | {'Flip Time':9s} | {'Pitch':7s} | {'Altitude':8s} | {'Max XY':7s} | {'Duration':8s} | {'Reward':7s} | {'Status'}")
    print(f"{'-'*16}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*15}")
    
    for ckpt in ckpt_list:
        res = run_evaluation(ckpt, dr_level=dr_level, num_episodes=5, deterministic=True)
        if res is None:
            continue
        flips = sum(1 for r in res if r["t_flip"] is not None)
        flip_times = [r["t_flip"] for r in res if r["t_flip"] is not None]
        avg_ft = f"{np.mean(flip_times):.2f}s" if flip_times else "N/A"
        pitches = np.mean([r["pitch_deg"] for r in res])
        alts = np.mean([r["final_z"] for r in res])
        max_xys = np.mean([r["max_xy"] for r in res])
        durations = np.mean([r["flight_time"] for r in res])
        rews = np.mean([r["tot_rew"] for r in res])
        causes = set(r["term_cause"] for r in res)
        status = ", ".join(causes)
        print(f"{ckpt:11,d} steps | {flips:2d}/{len(res):2d}   | {avg_ft:9s} | {pitches:5.1f}° | {alts:6.2f}m  | {max_xys:5.2f}m  | {durations:6.2f}s  | {rews:7.1f} | {status}")

print("\nRunning Multi-Checkpoint Testing...")
print_table("TEST 1: NOMINAL CLEAN SIMULATION", checkpoints, dr_level=0.0)
print_table("TEST 2: PARTIAL DOMAIN RANDOMIZATION (Wind, Motor Mismatch, Payload)", checkpoints, dr_level=0.5)
print_table("TEST 3: HEAVY SIM-TO-REAL STRESS TEST (Near-Full Disturbances)", checkpoints, dr_level=0.8)
