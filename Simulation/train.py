# Simulation/train.py
import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Ensure Simulation directory is on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from quad_velocity_env import QuadcopterVelocityEnv


def make_env():
    # Waypoint navigation task: fly to random Point B (x, y, z) and hover stably
    return QuadcopterVelocityEnv(
        task="waypoint",
        action_mode="thrust_moment", # Decoupled thrust & torques via mixerFM
        episode_seconds=15.0,        # 15-second episodes (3000 steps)
        random_wind=True,
        wind_magnitude=0.8,
    )


if __name__ == "__main__":
    num_cpu = 8

    print(f"Initializing {num_cpu} parallel simulation environments on CPU...")
    vec_env = make_vec_env(
        make_env,
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv,
    )

    # Configure PPO for waypoint navigation & hovering
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        verbose=1,
        device="cpu",
    )

    save_dir = os.path.join(_THIS_DIR, "..", "logs")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=save_dir)

    print("Starting PPO Waypoint Navigation Training for 1,000,000 steps (~3-4 minutes on M4)...")
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

    final_model_path = os.path.join(_THIS_DIR, "..", "quad_ppo_velocity")
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}.zip")
