"""
Where does PPO training time actually go? Measure it instead of guessing.

Three parts, all on the real environment and the real hyperparameters from train.py:

  A. RAW ENV        : one QuadFlipEnv in this process, no SB3, no wrapper. Isolates the
                      cost of the MuJoCo plant (10 x mj_step per control step) from
                      everything else, and profiles it.
  B. VEC ENV        : the 10-worker SubprocVecEnv, first bare then with the frozen
                      LatentObsWrapper attached. The difference between A and B is the
                      parallel speedup actually achieved (and IPC / wrapper overhead).
  C. FULL PPO       : model.learn for a few rollouts with the train.py configuration,
                      splitting wall time into collect_rollouts vs the PPO update
                      (PPO.train is monkeypatched to time itself). Then one iteration
                      under cProfile to find the trainer-process hotspots.

Run:  .venv/bin/python -u scratch/benchmark_training_pipeline.py
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import ACTOR_TOTAL_DIM, TOTAL_OBS_DIM, QuadFlipEnv  # noqa: E402

N_WORKERS = 10
N_STEPS = 2048  # SB3 PPO n_steps from train.py

ENCODER_PATH = os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt")


def _sprinkle(env: QuadFlipEnv, rng: np.random.Generator, n: int) -> float:
    """Run n steps with small random actions, resetting on done. Returns steps/s."""
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    t0 = time.perf_counter()
    done_count = 0
    for _ in range(n):
        a = (rng.uniform(-0.4, 0.4, size=4)).astype(np.float32)
        obs, _r, term, trunc, _info = env.step(a)
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done_count += 1
    dt = time.perf_counter() - t0
    return n / dt, done_count


def part_a() -> None:
    print("=" * 78)
    print("A. RAW ENV (single process, no SB3, no wrapper)")
    print("=" * 78)
    env = QuadFlipEnv()
    rng = np.random.default_rng(0)

    _sprinkle(env, rng, 100)  # warmup
    rate, dones = _sprinkle(env, rng, 1500)
    print(f"  step rate                : {rate:8.0f} steps/s  ({1e3 / rate:.3f} ms/step, {dones} episodes)")

    # Reset cost in isolation (trajectory sampling + DR draw + plant rebuild).
    t0 = time.perf_counter()
    for i in range(200):
        env.reset(seed=i)
    reset_ms = (time.perf_counter() - t0) / 200 * 1e3
    print(f"  reset cost               : {reset_ms:8.2f} ms/reset")

    pr2 = cProfile.Profile()
    pr2.enable()
    for i in range(60):
        env.reset(seed=10_000 + i)
    pr2.disable()
    buf2 = io.StringIO()
    ps2 = pstats.Stats(pr2, stream=buf2).sort_stats("cumtime")
    ps2.print_stats("Simulation/")
    print("\n  --- reset profile, project functions by cumtime (60 resets) ---")
    for line in buf2.getvalue().splitlines()[4:16]:
        print("  " + line)

    # Cost of the per-step info dict once it is pickled across the worker pipe.
    import pickle
    obs, info = env.reset(seed=0)
    _o, _r, _te, _tr, info = env.step(np.zeros(4, dtype=np.float32))
    payload = pickle.dumps((obs, 1.0, False, False, info), protocol=pickle.HIGHEST_PROTOCOL)
    t0 = time.perf_counter()
    for _ in range(2000):
        pickle.loads(payload)
    load_ms = (time.perf_counter() - t0) / 2000 * 1e3
    t0 = time.perf_counter()
    for _ in range(2000):
        pickle.dumps((obs, 1.0, False, False, info), protocol=pickle.HIGHEST_PROTOCOL)
    dump_ms = (time.perf_counter() - t0) / 2000 * 1e3
    print(f"  info payload size        : {len(payload) / 1024:8.1f} KB  "
          f"(dumps {dump_ms:.3f} ms, loads {load_ms:.3f} ms per env-step)")

    # Breakdown via cProfile over a fresh stretch.
    pr = cProfile.Profile()
    pr.enable()
    _sprinkle(env, rng, 600)
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    ps.print_stats(20)
    print("\n  --- cProfile, sorted by tottime (self time) ---")
    for line in buf.getvalue().splitlines()[4:30]:
        print("  " + line)

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumtime")
    ps.print_stats("Simulation/")
    print("\n  --- cProfile, project functions only, by cumtime ---")
    for line in buf.getvalue().splitlines()[4:30]:
        print("  " + line)


def build_vec(use_encoder: bool):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    def make_env():
        return QuadFlipEnv()

    venv = make_vec_env(make_env, n_envs=N_WORKERS, vec_env_cls=SubprocVecEnv)
    if use_encoder:
        from encoder.latent_obs_wrapper import LatentObsWrapper

        venv = LatentObsWrapper(venv, encoder_path=ENCODER_PATH, z_dim=16)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=False, clip_obs=10.0)
    return venv


def bench_vec(venv, n_outer: int, label: str) -> float:
    rng = np.random.default_rng(1)
    obs = venv.reset()
    t0 = time.perf_counter()
    for _ in range(n_outer):
        actions = rng.uniform(-0.4, 0.4, size=(N_WORKERS, 4)).astype(np.float32)
        obs, _r, _d, _i = venv.step(actions)
    dt = time.perf_counter() - t0
    rate = (n_outer * N_WORKERS) / dt
    print(f"  {label:<26}: {rate:8.0f} steps/s  ({dt / n_outer * 1e3:.2f} ms/outer-step)")
    return rate


def part_b() -> None:
    print()
    print("=" * 78)
    print(f"B. VEC ENV ({N_WORKERS} SubprocVecEnv workers)")
    print("=" * 78)
    bare = build_vec(use_encoder=False)
    bench_vec(bare, 60, "bare vec env (warmup)")
    bench_vec(bare, 400, "bare vec env")
    bare.close()

    if os.path.isfile(ENCODER_PATH):
        wrapped = build_vec(use_encoder=True)
        bench_vec(wrapped, 60, "+ LatentObsWrapper (warmup)")
        bench_vec(wrapped, 400, "+ LatentObsWrapper")
        wrapped.close()
    else:
        print(f"  (no encoder checkpoint at {ENCODER_PATH} - skipped)")


def part_d() -> None:
    """cProfile the MAIN process around the bare vec env loop: where does the serial time go?"""
    print()
    print("=" * 78)
    print("D. TRAINER-SIDE HOTSPOTS in vec_env.step (bare, no wrapper)")
    print("=" * 78)
    venv = build_vec(use_encoder=False)
    rng = np.random.default_rng(2)

    def loop(n):
        venv.reset()
        for _ in range(n):
            venv.step(rng.uniform(-0.4, 0.4, size=(N_WORKERS, 4)).astype(np.float32))

    loop(40)
    t0 = time.perf_counter()
    loop(200)
    dt = time.perf_counter() - t0
    print(f"  timed loop: {200 * N_WORKERS / dt:.0f} steps/s ({dt / 200 * 1e3:.2f} ms/outer-step)")

    pr = cProfile.Profile()
    pr.enable()
    loop(200)
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    ps.print_stats(22)
    print("\n  --- cProfile (main process only) ---")
    for line in buf.getvalue().splitlines()[4:32]:
        print("  " + line)
    venv.close()


def part_c() -> None:
    print()
    print("=" * 78)
    print("C. FULL PPO (train.py hyperparameters)")
    print("=" * 78)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    import train as train_mod
    from asymmetric_policy import AsymmetricActorCriticPolicy

    torch.set_num_threads(1)  # same as train.py

    venv = build_vec(use_encoder=os.path.isfile(ENCODER_PATH))
    actor_dim = ACTOR_TOTAL_DIM + (16 if os.path.isfile(ENCODER_PATH) else 0)

    model = PPO(
        policy=AsymmetricActorCriticPolicy,
        env=venv,
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=512,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            actor_obs_dim=actor_dim,
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[128, 128], vf=[512, 256, 128]),
            log_std_init=-0.5,
        ),
        verbose=0,
        device="cpu",
    )

    timings = {"update": 0.0}

    orig_train = PPO.train

    def timed_train(self):
        t0 = time.perf_counter()
        out = orig_train(self)
        timings["update"] += time.perf_counter() - t0
        return out

    PPO.train = timed_train
    try:
        N_ROLLOUTS = 3
        t0 = time.perf_counter()
        model.learn(total_timesteps=N_STEPS * N_WORKERS * N_ROLLOUTS)
        wall = time.perf_counter() - t0
        total_steps = N_STEPS * N_WORKERS * N_ROLLOUTS
        upd = timings["update"]
        print(f"  rollout (collect) time   : {(wall - upd) / N_ROLLOUTS:8.3f} s/rollout "
              f"({(total_steps / (wall - upd)):6.0f} steps/s)")
        print(f"  update  (PPO.train) time : {upd / N_ROLLOUTS:8.3f} s/rollout")
        print(f"  end-to-end               : {total_steps / wall:8.0f} steps/s  "
              f"({100 * upd / wall:.0f}% of wall time is the update)")
    finally:
        PPO.train = orig_train

    # One iteration under cProfile, to see the trainer-process hotspots.
    pr = cProfile.Profile()
    pr.enable()
    model.learn(total_timesteps=N_STEPS * N_WORKERS * 1, reset_num_timesteps=True)
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    ps.print_stats(30)
    print("\n  --- cProfile of one train iteration (tottime) ---")
    for line in buf.getvalue().splitlines()[4:40]:
        print("  " + line)

    # Does the PPO update want more threads? (train.py pins torch to 1)
    for nt in (1, 2, 4):
        torch.set_num_threads(nt)
        model.learn(total_timesteps=N_STEPS * N_WORKERS * 1, reset_num_timesteps=True)  # warmup
        timings["update"] = 0.0
        t0 = time.perf_counter()
        model.learn(total_timesteps=N_STEPS * N_WORKERS * 1, reset_num_timesteps=True)
        wall = time.perf_counter() - t0
        print(f"  threads={nt}: end-to-end {N_STEPS * N_WORKERS / wall:6.0f} steps/s "
              f"(update {timings['update']:.2f} s, collect {wall - timings['update']:.2f} s)")
    torch.set_num_threads(1)
    venv.close()


def _busy(n: int) -> float:
    """Module-level so spawn can pickle it. ~1 ms of pure-Python float work."""
    x = 1.2345
    for i in range(n):
        x = x * 1.0000001 + 0.0000001
    return x


def part_e() -> None:
    """
    Worker scaling sweep: throughput vs n_envs. Also a pure-Python busy-loop baseline on
    the same pool size, so an env bottleneck can be told apart from a MACHINE ceiling
    (this is a fanless 4P+6E M4, where 10 heavy processes is already more runnable work
    than the chip can sustain).
    """
    import multiprocessing as mp
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    print()
    print("=" * 78)
    print("E. WORKER SCALING SWEEP (bare env, short bursts)")
    print("=" * 78)

    # ~1 ms of pure-Python float work per call, to calibrate the chip.
    t0 = time.perf_counter()
    _busy(300_000)
    unit = time.perf_counter() - t0

    pool_sizes = [1, 2, 4, 6, 8, 10]
    base = None
    for np_ in pool_sizes:
        with mp.get_context("spawn").Pool(np_) as pool:
            pool.map(_busy, [10_000] * np_)
            t0 = time.perf_counter()
            n_tasks = 200
            pool.map(_busy, [300_000] * n_tasks)
            dt = time.perf_counter() - t0
        rate = n_tasks / dt
        if base is None:
            base = rate
        print(f"  pure python, pool={np_:>2}: {rate:8.1f} tasks/s  ({rate / base:4.1f}x)")

    print()
    base = None
    for n in [2, 4, 6, 8, 10]:
        venv = make_vec_env(lambda: QuadFlipEnv(), n_envs=n, vec_env_cls=SubprocVecEnv)
        rng = np.random.default_rng(3)
        venv.reset()
        for _ in range(30):
            venv.step(rng.uniform(-0.4, 0.4, size=(n, 4)).astype(np.float32))
        N = 300
        t0 = time.perf_counter()
        for _ in range(N):
            venv.step(rng.uniform(-0.4, 0.4, size=(n, 4)).astype(np.float32))
        dt = time.perf_counter() - t0
        rate = N * n / dt
        if base is None:
            base = rate / n  # single-worker-equivalent from the 2-worker run
        print(f"  env workers={n:>2}        : {rate:8.0f} steps/s  ({rate / (base * n):4.2f}x efficiency)")
        venv.close()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "abc"
    if "a" in which:
        part_a()
    if "b" in which:
        part_b()
    if "d" in which:
        part_d()
    if "e" in which:
        part_e()
    if "c" in which:
        part_c()
