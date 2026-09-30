"""
Environment sanity + throughput benchmark for the MJX port: `Simulation/quad_mjx`.

    .venv/bin/python -u Simulation/test_mjx_env.py                 # contract + benchmark
    .venv/bin/python -u Simulation/test_mjx_env.py --batches 256,1024
    .venv/bin/python -u Simulation/test_mjx_env.py --episode        # one full rollout

Two things this file refuses to get wrong, because both cost a lot of time to learn:

  * JAX dispatch is ASYNCHRONOUS.  Every timed call is followed by ``block_until_ready``;
    without it a "33.7 s" call is followed by a "0.00 s" call and neither is real.
  * The first call of any new shape is XLA COMPILATION.  Compile time is reported
    separately so it cannot masquerade as throughput.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# _HERE must WIN over _ROOT.  The repository root contains `train_mjx.py`, a thin shim;
# if it shadows `Simulation/train_mjx.py` then `T.TOTAL_TIMESTEPS` and every other constant
# disappears.  A plain `if p not in sys.path` guard is not enough: Python already puts this
# file's own directory on the path, so the root ends up in FRONT and the shim wins.
for p in (_ROOT, _HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import jax
import jax.numpy as jnp

import train_mjx as T
from quad_mjx import spec as S
from quad_mjx.env import QuadFlipMJXEnv


def contract(env):
    print("observation / action contract")
    st, obs = jax.jit(lambda k: env.reset(k, -1, 0.0, S.ENVELOPE_SCALE_START))(
        jax.random.PRNGKey(0))
    rows = [
        ("env obs", S.TOTAL_OBS_DIM, 80),
        ("wrapped / critic obs", obs.critic_obs.shape[0], S.CRITIC_OBS_DIM),
        ("actor obs", obs.actor_obs.shape[0], env.actor_dim),
        ("encoder frame", obs.frame.shape[0], 33),
        ("action", S.ACTION_DIM, env.action_dim),
        ("max steps", env.max_steps, int(round(S.EPISODE_SECONDS / S.SIM_DT))),
    ]
    ok = True
    for name, got, want in rows:
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'BAD'}] {name:<22} {got:<5} (expected {want})")
    print(f"  encoder loaded: {env.gru is not None}"
          f"{'  ' + os.path.relpath(env.encoder_path, _ROOT) if env.gru is not None else ''}")
    return ok


def bench(env, batches, steps, reps):
    print(f"\nthroughput (steady state, {reps} timed iterations after compile)")
    print(f"  {'batch':>7}  {'reset':>10}  {'step':>10}  {'us/env-step':>12}  "
          f"{'env-steps/s':>12}")
    out = {}
    dr, env_scale = 0.0, S.ENVELOPE_SCALE_START
    for n in batches:
        keys = jax.random.split(jax.random.PRNGKey(1), n)
        # dr / envelope are closed over so each benched call takes exactly one argument
        vr = jax.jit(jax.vmap(lambda k: env.reset(k, -1, dr, env_scale)))
        vs = jax.jit(jax.vmap(lambda s, a: env.step(s, a)))
        st, ob = vr(keys)
        jax.block_until_ready(st.data.qpos)
        act = jnp.zeros((n, S.ACTION_DIM))
        vs(st, act)[0].data.qpos.block_until_ready()

        def timeit(fn, *args):
            for _ in range(2):                      # drop two: the 2nd compile chunk
                jax.block_until_ready(fn(*args))
            t = time.time()
            for _ in range(reps):
                jax.block_until_ready(fn(*args))
            return (time.time() - t) / reps

        t_reset = timeit(vr, keys) / steps
        t_step = timeit(vs, st, act)
        per = (t_reset + t_step) / n
        print(f"  {n:>7}  {t_reset * 1e3:9.2f}ms  {t_step * 1e3:9.2f}ms  "
              f"{per * 1e6:11.1f}  {1.0 / per:12,.0f}")
        out[n] = per
    return out


def main():
    ap = argparse.ArgumentParser(description="MJX env sanity + throughput")
    ap.add_argument("--batches", default="32,256,1024")
    ap.add_argument("--steps", type=int, default=20,
                    help="control steps per rollout, to amortise the reset")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--episode", action="store_true", help="also run one full episode")
    a = ap.parse_args()
    batches = [int(x) for x in a.batches.split(",") if x.strip()]

    print(f"JAX {jax.__version__}  devices={jax.devices()}")
    env = QuadFlipMJXEnv()
    ok = contract(env)
    per = bench(env, batches, a.steps, a.reps)

    best = min(per.values())
    print(f"\nprojected full run ({T.TOTAL_TIMESTEPS:,} env steps, "
          f"NUM_ENVS={T.NUM_ENVS} x NUM_STEPS={T.NUM_STEPS}):")
    print(f"  at the best measured rate ({1 / best:,.0f} env-steps/s) -> "
          f"{30e6 * best / 3600:.1f} h")
    print(f"  NOTE this is the PLANT cost only: a rollout step also redraws a trajectory "
          f"for\nevery env that finishes, and PPO adds an update phase.")

    if a.episode:
        print("\nfull-episode rollout (termination + truncation behaviour)")
        vstep = jax.jit(lambda s, a_: env.step(s, a_))

        def body(carry, _):
            s, o = carry
            s, o, r, term, trunc = vstep(s, jnp.zeros(S.ACTION_DIM))
            return (s, o), (r, term, trunc)
        st, ob = jax.jit(lambda k: env.reset(k, -1, 0.0, S.ENVELOPE_SCALE_START))(
            jax.random.PRNGKey(3))
        (st, ob), (rew, term, trunc) = jax.lax.scan(body, (st, ob), None, length=env.max_steps)
        jax.block_until_ready(rew)
        done = np.flatnonzero(np.asarray(term | trunc))
        n = int(done[0]) + 1 if done.size else env.max_steps
        print(f"  a zero-action hover survived {n} steps ({n * S.SIM_DT:.2f}s of "
              f"{S.EPISODE_SECONDS:.1f}s), return {float(np.asarray(rew)[:n].sum()):.2f}, "
              f"termination {('code ' + str(int(st.termination))) if done.size else 'none'}")

    print("\nOK" if ok else "\nCONTRACT MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
