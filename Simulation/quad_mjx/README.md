# `quad_mjx` — the JAX / MuJoCo-MJX port

A faithful re-implementation of the frozen numpy/SB3 training stack
(`quad_flip_env.py`, `quad_mujoco.py`, `trajectories.py`, `lighthouse.py`) as a pure,
`vmap`-able MJX environment. The numpy baseline is **not modified** and is **not imported**:
`spec.py` copies its constants verbatim so the two can be diffed by eye, and
`Simulation/scratch/check_mjx_parity.py` compares them numerically.

## Why the port is shaped the way it is

JAX traces shapes at compile time, so nothing in the baseline's Python control flow
survives as-is. Every branch became a `jnp.where`, every loop a `lax.scan`/`lax.while_loop`,
and every family dispatch a `lax.switch` over a constant-shape `struct.dataclass` pytree.
Consequences worth knowing before editing:

* **All state is fixed-shape pytrees.** `EnvState` carries the model, `mjx.Data`, the plant
  state, the lighthouse, the `TrajSpec`, the GRU hidden state and the DR params.
* **Sampling is a traced rejection loop**, not a Python `while`. It uses
  `lax.while_loop` so the `max_resample` cap does not cost anything when it is not reached —
  this was worth **23×** on `env.reset` (317 ms → 14 ms at 32 envs).
* **Never drive `env.step` in a Python loop.** `env` per-timestep in eager mode is thousands
  of dispatches and a retained XLA executable each: it is both glacial and a memory hazard.
  Jit and `vmap` it. `Simulation/scratch/check_mjx_parity.py` is the worked example.

## Files

| file | contents |
|---|---|
| `spec.py` | single source of truth for constants, copied from the baseline; `PORT_NOTES`; reward kernels |
| `dynamics.py` | the plant, **including the original cascaded rate PID and mixer** |
| `trajectories.py` | all 10 reference families, relocation, terminal hover, central-difference `omega` |
| `sampler.py` | traced rejection sampler with the baseline's feasibility screens |
| `lighthouse.py` | station geometry, visibility, the fix gate, dead reckoning, streak breaker |
| `encoder.py` | inference-only frozen GRU history encoder |
| `env.py` | `QuadFlipMJXEnv`: `reset`, `step`, reward, termination, DR, failure injection |
| `wind.py` | Perlin value-noise wind |

Entry points live one level up: `Simulation/train_mjx.py`, `Simulation/evaluate_mjx.py`,
`Simulation/export_to_sb3.py`, `Simulation/test_mjx_env.py`.

## Observation contract (frozen)

```
env obs     : [ o_t 29 | ref_ff 3 | aux 4 | privileged 44 ]                = 80   TOTAL_OBS_DIM
wrapped obs : [ o_t 29 | z 16 | ref_ff 3 | aux 4 | privileged 44 ]         = 96   CRITIC_OBS_DIM
ACTOR       : [ o_t 29 | z 16 | ref_ff 3 ]                                 = 48
encoder frame: [ o_t 29 | aux 4 ]                                          = 33
```

`CRITIC_OBS_DIM` is **not** `TOTAL_OBS_DIM`: the critic consumes the *wrapped* vector, which
is what SB3's vec env exposes (the baseline's `LatentInjector` does the same splice). Both
constants carry an `assert` tying them together, because getting this wrong is invisible
until the network is built and then fails on the first `apply`.

## Deliberate deviations

Only the four in `spec.PORT_NOTES` — all of them are *fixes to the previous MJX port*, not
changes to the baseline:

1. the inner rate PID is kept exactly (gains, torque limits, ground-truth `qvel` + gyro
   bias), **not** replaced by direct moment control — that would be a domain shift;
2. the PID integrator now persists across the episode;
3. anti-windup clamps `ki * integral` in **torque units**;
4. `SUBSTEPS * PHYSICS_DT == SIM_DT` (the previous port ran physics at half speed).

## Verifying it

```bash
.venv/bin/python -u Simulation/scratch/check_mjx_parity.py   # the acceptance gate, ~4 min
.venv/bin/python -u Simulation/test_mjx_env.py               # contract + throughput
```

`check_mjx_parity.py` sections A–F diff the port against the numpy original directly
(poses to 1e-14, the closed inner loop to 1e-16, the sampler's guarantees, the reward).
Section **G** is the acceptance gate: a textbook cascaded geometric controller, with no
learning at all, must hold **≥70 % of the per-step ceiling on all 9 families over 6 seeds**.
It scores ~85 % overall, within a few points of the baseline's recorded numbers — that is
what says the sampler, plant, PID, lighthouse, reference and reward are consistent *with
each other*, closed-loop, in a way no unit comparison can.

## Performance measured here (Apple M4 CPU, `jax 0.10.2`)

| batch | `reset` | `step` | env-steps/s |
|---:|---:|---:|---:|
| 32 | 0.9 ms | 17.0 ms | 1,789 |
| 256 | 10.8 ms | 119.2 ms | 1,969 |
| 1024 | 86.3 ms | 353.2 ms | 2,330 |

≈ **3.6 h** for the 30 M-step plant at `NUM_ENVS=1024`, plus the PPO update phase.

Two measurement traps, both of which produced wrong conclusions during development: JAX
dispatch is asynchronous (**always** `block_until_ready`), and the first two calls of a new
shape are XLA compilation. Mixing those up made a 2,300 env-steps/s kernel look like
7 env-steps/s.
