"""
Autonomous Experiment Orchestrator for Quadcopter Flip & Recovery.
Orchestrates sequential experiments, runs training with fast-budget curriculum (< 10M steps),
and executes standardized multi-regime evaluation benchmarks (dr=0.0, 0.5, 1.0).
"""
import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure project paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "Simulation" else _THIS_DIR
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for p in [_PROJECT_ROOT, _SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from quad_flip_env import QuadFlipEnv, ACTOR_TOTAL_DIM, TOTAL_OBS_DIM, ACTION_MODE
from asymmetric_policy import AsymmetricActorCriticPolicy
from train import PiecewiseLinearSchedule, DomainRandomizationCallback, VecNormalizeCheckpointCallback

LEDGER_PATH = os.path.join(_PROJECT_ROOT, "experiments_ledger.md")
EXPERIMENTS_DIR = os.path.join(_PROJECT_ROOT, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def evaluate_policy(
    model_path: str,
    stats_path: str,
    dr_level: float = 0.0,
    n_episodes: int = 15,
    episode_seconds: float = 8.0,
    deterministic: bool = True,
    actor_obs_dim: int = ACTOR_TOTAL_DIM,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Runs standardized evaluation across n_episodes at a specified domain randomization level."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(
        model_path,
        custom_objects=dict(
            policy_class=AsymmetricActorCriticPolicy,
            actor_obs_dim=actor_obs_dim,
        ),
    )

    base_eval_kwargs = dict(
        episode_seconds=episode_seconds,
        random_initial_state=True if dr_level > 0.0 else False,
        arena_radius=2.5,
    )
    if env_kwargs:
        eval_kw = env_kwargs.copy()
        eval_kw.pop("arena_radius", None)  # Benchmark arena remains standard 2.5m
        base_eval_kwargs.update(eval_kw)

    env = QuadFlipEnv(**base_eval_kwargs)
    env.set_dr_level(dr_level)

    vec_norm = None
    if os.path.isfile(stats_path):
        vec_norm = VecNormalize.load(stats_path, DummyVecEnv([lambda: env]))
        vec_norm.training = False

    flips_completed = 0
    flip_times = []
    max_xy_drifts = []
    final_alt_errors = []
    action_diff_norms = []
    settling_times = []
    total_rewards = []
    flight_durations = []
    termination_causes = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        t_flip = None
        tot_rew = 0.0
        steps = 0
        max_xy = 0.0
        ep_action_diffs = []
        term_cause = "completed_flight"
        t_settled = None

        while not done and steps < env.max_steps:
            obs_norm = vec_norm.normalize_obs(obs) if vec_norm else obs
            actor_obs = obs_norm[:actor_obs_dim]
            act, _ = model.predict(actor_obs, deterministic=deterministic)

            if steps > 0:
                ep_action_diffs.append(float(np.linalg.norm(act - prev_act)))
            prev_act = act.copy()

            obs, rew, term, trunc, info = env.step(act)
            tot_rew += rew
            steps += 1

            pos = info["position"]
            xy_dist = float(np.linalg.norm(pos[:2]))
            if xy_dist > max_xy:
                max_xy = xy_dist

            if info.get("flip_completed", False) and t_flip is None:
                t_flip = float(info["t"])

            # Measure settling after flip: speed < 0.25 m/s and omega < 1.5 rad/s
            if t_flip is not None and t_settled is None:
                speed = float(np.linalg.norm(info["velocity"]))
                omega_mag = float(np.linalg.norm(info["omega"]))
                if speed < 0.25 and omega_mag < 1.5:
                    t_settled = float(info["t"] - t_flip)

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
                    term_cause = "completed_flight"

        if t_flip is not None:
            flips_completed += 1
            flip_times.append(t_flip)
        if t_settled is not None:
            settling_times.append(t_settled)

        max_xy_drifts.append(max_xy)
        alt_err = abs(float(info["position"][2] - env.target_altitude))
        final_alt_errors.append(alt_err)
        total_rewards.append(tot_rew)
        flight_durations.append(float(info["t"]))
        termination_causes.append(term_cause)
        if ep_action_diffs:
            action_diff_norms.append(float(np.mean(ep_action_diffs)))

    return {
        "dr_level": dr_level,
        "n_episodes": n_episodes,
        "flip_rate": float(flips_completed / n_episodes),
        "mean_flip_time": float(np.mean(flip_times)) if flip_times else None,
        "mean_max_xy_drift": float(np.mean(max_xy_drifts)),
        "mean_alt_error": float(np.mean(final_alt_errors)),
        "mean_settling_time": float(np.mean(settling_times)) if settling_times else None,
        "mean_action_smoothness": float(np.mean(action_diff_norms)) if action_diff_norms else 0.0,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_duration": float(np.mean(flight_durations)),
        "terminations": {c: termination_causes.count(c) for c in set(termination_causes)},
    }


def run_benchmark_battery(
    model_path: str,
    stats_path: str,
    actor_obs_dim: int = ACTOR_TOTAL_DIM,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Runs standardized battery across nominal, moderate, and full domain randomization."""
    print(f"\n--- Running Benchmark Battery for {os.path.basename(model_path)} ---")
    res_clean = evaluate_policy(model_path, stats_path, dr_level=0.0, n_episodes=15, actor_obs_dim=actor_obs_dim, env_kwargs=env_kwargs)
    print(f"  [DR 0.0] Flips: {res_clean['flip_rate']*100:.0f}% | FlipTime: {res_clean['mean_flip_time']}s | XY Drift: {res_clean['mean_max_xy_drift']:.2f}m | AltErr: {res_clean['mean_alt_error']:.2f}m | Rew: {res_clean['mean_reward']:.1f}")

    res_mod = evaluate_policy(model_path, stats_path, dr_level=0.5, n_episodes=15, actor_obs_dim=actor_obs_dim, env_kwargs=env_kwargs)
    print(f"  [DR 0.5] Flips: {res_mod['flip_rate']*100:.0f}% | FlipTime: {res_mod['mean_flip_time']}s | XY Drift: {res_mod['mean_max_xy_drift']:.2f}m | AltErr: {res_mod['mean_alt_error']:.2f}m | Rew: {res_mod['mean_reward']:.1f}")

    res_stress = evaluate_policy(model_path, stats_path, dr_level=1.0, n_episodes=15, actor_obs_dim=actor_obs_dim, env_kwargs=env_kwargs)
    print(f"  [DR 1.0] Flips: {res_stress['flip_rate']*100:.0f}% | FlipTime: {res_stress['mean_flip_time']}s | XY Drift: {res_stress['mean_max_xy_drift']:.2f}m | AltErr: {res_stress['mean_alt_error']:.2f}m | Rew: {res_stress['mean_reward']:.1f}")

    # Composite robustness score (higher is better)
    # Rewards flip success, low drift, low alt error, and survival under DR 1.0
    score = (
        (res_clean["flip_rate"] * 30.0 + res_mod["flip_rate"] * 35.0 + res_stress["flip_rate"] * 35.0)
        - 15.0 * res_stress["mean_max_xy_drift"]
        - 20.0 * res_stress["mean_alt_error"]
        + 0.01 * res_stress["mean_reward"]
    )

    return {
        "clean": res_clean,
        "moderate": res_mod,
        "stress": res_stress,
        "composite_score": float(score),
    }


def train_experiment(
    exp_id: int,
    exp_name: str,
    total_timesteps: int = 4_000_000,
    dr_start: int = 1_000_000,
    dr_end: int = 3_500_000,
    lr_warmup: int = 1_000_000,
    lr_start: float = 3e-4,
    lr_mid: float = 1.5e-4,
    lr_end: float = 5e-5,
    lr_floor: float = 3e-5,
    num_workers: int = 10,
    load_model_path: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    ppo_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Runs a single training experiment with specified hyperparameters and returns paths to model and vecnorm stats."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, f"exp_{exp_id}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)
    model_name = f"model_exp_{exp_id}"
    out_model_path = os.path.join(exp_dir, f"{model_name}.zip")
    out_stats_path = os.path.join(exp_dir, f"{model_name}_vecnormalize.pkl")

    torch.set_num_threads(1)
    base_env_kwargs = dict(
        action_mode="motor",
        episode_seconds=8.0,
        target_altitude=1.2,
        spawn_altitude=1.2,
        random_initial_state=True,
        arena_radius=2.5,
    )
    if env_kwargs:
        base_env_kwargs.update(env_kwargs)

    def make_env():
        return QuadFlipEnv(**base_env_kwargs)

    vec_env = make_vec_env(make_env, n_envs=num_workers, vec_env_cls=SubprocVecEnv)

    if load_model_path and os.path.isfile(load_model_path):
        norm_source = load_model_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.isfile(norm_source):
            vec_env = VecNormalize.load(norm_source, vec_env)
            vec_env.training = True
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        print(f"Resuming/Warm-starting from: {load_model_path}")
        model = PPO.load(
            load_model_path,
            env=vec_env,
            device="cpu",
            custom_objects=dict(
                policy_class=AsymmetricActorCriticPolicy,
                actor_obs_dim=ACTOR_TOTAL_DIM,
            ),
        )
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        anneal_steps = dr_end - dr_start
        lr_schedule = PiecewiseLinearSchedule(
            total_timesteps=total_timesteps,
            warmup_steps=lr_warmup,
            anneal_steps=anneal_steps,
            lr_start=lr_start,
            lr_mid=lr_mid,
            lr_adr_end=lr_end,
            lr_floor=lr_floor,
            final_stable=False,
        )

        default_policy_kwargs = dict(
            actor_obs_dim=ACTOR_TOTAL_DIM,
            activation_fn=nn.Tanh,
            net_arch=dict(pi=[128, 128], vf=[512, 256, 128]),
            log_std_init=-0.5,
        )
        if policy_kwargs:
            default_policy_kwargs.update(policy_kwargs)

        default_ppo_kwargs = dict(
            n_steps=2048,
            batch_size=512,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            device="cpu",
        )
        if ppo_kwargs:
            default_ppo_kwargs.update(ppo_kwargs)

        model = PPO(
            policy=AsymmetricActorCriticPolicy,
            env=vec_env,
            learning_rate=lr_schedule,
            policy_kwargs=default_policy_kwargs,
            **default_ppo_kwargs,
        )

    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=exp_dir, name_prefix=model_name)
    norm_cb = VecNormalizeCheckpointCallback(save_freq=50_000, save_path=exp_dir, root_stats_path=out_stats_path, name_prefix=model_name)
    dr_cb = DomainRandomizationCallback(start_steps=dr_start, end_steps=dr_end, verbose=0)
    callbacks = [ckpt_cb, norm_cb, dr_cb]

    print(f"\nTraining Experiment {exp_id}: '{exp_name}' for {total_timesteps:,} steps...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=True)
    t_elapsed = time.time() - t0
    print(f"Training completed in {t_elapsed:.1f}s ({t_elapsed/60:.1f} min).")

    model.save(out_model_path.replace(".zip", ""))
    vec_env.save(out_stats_path)
    vec_env.close()

    return out_model_path, out_stats_path


def log_experiment_to_ledger(
    exp_id: int,
    exp_name: str,
    hypothesis: str,
    modifications: str,
    results: Dict[str, Any],
    decision: str,
):
    """Appends experiment results and verdict to the persistent scientific ledger."""
    clean = results["clean"]
    stress = results["stress"]

    entry = f"""
### Experiment {exp_id}: {exp_name}
- **Hypothesis**: {hypothesis}
- **Modifications**: {modifications}
- **Quantitative Benchmark Results**:
  - **Nominal Sim (DR 0.0)**: Flips={clean['flip_rate']*100:.0f}% | FlipTime={clean['mean_flip_time'] or 0:.2f}s | XY Drift={clean['mean_max_xy_drift']:.2f}m | AltError={clean['mean_alt_error']:.2f}m | Rew={clean['mean_reward']:.1f}
  - **Stress Test (DR 1.0)**: Flips={stress['flip_rate']*100:.0f}% | FlipTime={stress['mean_flip_time'] or 0:.2f}s | XY Drift={stress['mean_max_xy_drift']:.2f}m | AltError={stress['mean_alt_error']:.2f}m | Rew={stress['mean_reward']:.1f}
  - **Composite Score**: **{results['composite_score']:.1f}**
- **Decision & Analysis**: {decision}

---
"""
    with open(LEDGER_PATH, "a") as f:
        f.write(entry)


def init_ledger():
    if not os.path.isfile(LEDGER_PATH):
        header = """# Autonomous Quadcopter Flip: 10-Iteration Scientific Ledger

| Experiment | Title | Fast Budget | Flip Rate (DR 1.0) | Max Drift (DR 1.0) | Alt Error (DR 1.0) | Composite Score | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
"""
        with open(LEDGER_PATH, "w") as f:
            f.write(header)


if __name__ == "__main__":
    init_ledger()
    print("Experiment Orchestrator initialized successfully.")
