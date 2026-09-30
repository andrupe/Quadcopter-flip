"""
Vectorised PPO trainer: a port of ``Simulation/train.py``.

The SCHEDULE is the baseline's, so a run here is comparable to a run there:
  * 30M steps, LR 3e-4 -> 1.5e-4 -> 5e-5 -> 3e-5 over 10M / 10M / 10M
  * entropy 0.01 -> 0.001; per-channel log_std floors (-1,-1,-1,-1) -> (-2,-3,-3,-3)
  * flight-envelope curriculum 5.0 -> 1.5, reaching 1.5 at 12M and then HELD there
  * ADR ramp dr 0 -> 1 over steps 14M..24M, i.e. AFTER the envelope has settled
  * chain-mixture ramp over 24M..30M (chain weight 0.05 -> 0.25)
  * a periodic DETERMINISTIC eval, one episode per family, reporting reward AND episode
    length, plus a dr=1 sweep every `ROBUST_EVAL_EVERY`-th evaluation

The two DIFFICULTY axes (flight envelope, ADR) are deliberately SERIALISED.  The "a clean
phase first" rule is the baseline's own - the 2026-09-11 run measured ~zero net progress
when the ADR ramp overlapped the clean phase - and the 2026-09-24 change applies the same
rule to the envelope, which had been ramping across the whole run and therefore overlapping
the ADR ramp.  See the comment above the curriculum breakpoints for the measurements.

Implementation notes that matter
  * The log_std floor is applied WHERE THE DISTRIBUTION IS BUILT
    (`max(log_std, floor_t, MIN_LOG_STD)`) rather than by mutating the parameter.  That is
    equivalent to the baseline's `log_std.data.clamp_` (which also zeroes the gradient on a
    clamped channel) and keeps the update one jitted function.  MIN_LOG_STD is applied as
    well, so the floor can never be silently undone.
  * Truncation is NOT treated as termination: the value bootstrap is masked by
    `terminated` only, so a time-limited episode still bootstraps.  Masking by
    `terminated | truncated` biases the value target on every episode that runs to its
    horizon - which, since the reference always ends in a hover, is the common case.
  * The manoeuvre-weight curriculum is applied by mutating `env.weights` around the step,
    because the sampler reads that attribute per reset.  It is the same mechanism the
    baseline uses (`env_method("set_maneuver_weight")`).
"""

from __future__ import annotations

import os
import sys
import time
from typing import NamedTuple

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.training.train_state import TrainState

from quad_mjx.env import QuadFlipMJXEnv, EnvConfig
from quad_mjx import spec as S
from quad_mjx import sampler as SAMP

# ======================================================================================
# schedule (mirrors Simulation/train.py)
# ======================================================================================
TOTAL_TIMESTEPS = 30_000_000
NUM_ENVS = 1024
NUM_STEPS = 128                     # 131072 transitions/iteration.  WAS 20, which is EXACTLY
# ONE GAE HORIZON (1/(1-GAE_LAMBDA) = 20): the rollout carried no realised return past the
# gamma-discount of a single lambda window, so the advantage was dominated by the bootstrap
# value of a critic that starts random.  train.py rolls 2560 steps/env (128x margin).  A
# 128-step rollout gives 6.4x margin.  MEASURED 2026-09-24, 10M arms, envelope 1.5:
#   num_steps 20  -> 51.6 / 52.2 / 54.1 at 2.5M / 5.0M / 7.5M
#   num_steps 128 -> 58.0 / 56.5 / 56.7  (untrained floor 56.3)
# and at the SAME 7.6M step the 128-step arm wins 8 of 10 families (takeoff 58 vs 50,
# slalom 58 vs 50, flip 67 vs 62).  It reaches at 2.5M what the 20-step arm needed 7.5M for.
NUM_EPOCHS = 5                      # train.py's n_epochs
NUM_MINIBATCHES = 256               # mb = 131072/256 = 512 = train.py's batch_size
# `TrainState.step` is owned by optax: `apply_gradients` bumps it ONCE PER MINIBATCH
# UPDATE, not once per environment step.  So one rollout iteration == this many `ts.step`,
# and every schedule lookup must divide by it before indexing a per-iteration row.
# This is the DEFAULT config's value; a run with a `PPOConfig` override has its own
# (`PPOConfig.updates_per_iter`).  Kept as a module constant because the parity checker and
# the throughput bench import it.
MINIBATCHES_PER_ITER = NUM_EPOCHS * NUM_MINIBATCHES
GAMMA = 0.997
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
ENT_COEF_START = 0.01               # train.py's ENT_COEF.  The port ran 0.001 -> 0.0001,
ENT_COEF_END = 0.001                # i.e. 10x LESS exploration pressure, with no measurement
# behind it - and all in the direction train.py's own comment warns about: "a policy that
# gives up on the flip can still collect decent average reward from hover and waypoints,
# which is exactly the kind of local optimum entropy pressure is the standard antidote to."
LR_START = 3e-4
LR_MID = 1.5e-4
LR_ADR_END = 5e-5
LR_FLOOR = 3e-5
LR_WARMUP_STEPS = 10_000_000
LR_ADR_END_STEP = 20_000_000
CHAIN_MIX_START_W = 0.05
CHAIN_MIX_END_W = 0.25
LOG_STD_FLOOR_START = (-1.0, -1.0, -1.0, -1.0)
LOG_STD_FLOOR_END = (-2.0, -3.0, -3.0, -3.0)   # train.py's measured values.  The port ran
# (-3,-4,-4,-4), a speculative tightening with no measurement behind it.
MIN_LOG_STD = -4.0
# UPPER clamp on log_std, matching `train.py`'s `max_log_std = 0.0`
# (`log_std.data.clamp_(min=floor_t, max=ceil_t)`).  The port had only a FLOOR, so sigma was
# free to grow past 1.  MEASURED 2026-09-24 on the 10M run: thrust sigma reached 1.26, so
# ~42% of thrust samples were clipped by the env.  Every sample past the clip has
# `d log pi / d mu = (a - mu)/sigma^2 > 0` while the reward it earns is UNCHANGED, so the
# update pushes the mean further into the boundary - the trained actor's mean thrust ran to
# +0.74..+1.84 against a hover trim of +0.079.  Missing this clamp is a port deviation.
#
# *** RESTORED TO `train.py`'s VALUE 2026-09-24. ***  The port's -0.7 was MY OWN choice, not
# `train.py`'s, and it was a second unmeasured tightening of exploration stacked on top of the
# 10x entropy cut.  An unclamped sigma is a real bug; -0.7 was an invention.
MAX_LOG_STD = 0.0
# `train.py` initialises its policy log_std at -0.5 (`log_std_init=-0.5`); the port used
# 0.0, i.e. sigma 1.0 out of the gate, so a third of every action sample sat at the
# |a| = 1 clip from step 0.  Restored.
LOG_STD_INIT = -0.5
ENVELOPE_START = S.ENVELOPE_SCALE_START      # 5.0
ENVELOPE_END = S.ENVELOPE_SCALE_END          # 1.5

# ---- curriculum ordering: the two difficulty axes are SERIALISED, not overlapped -------
# The envelope and the ADR ramp are independent axes, and moving both at once hands the
# policy a non-stationary TERMINATION landscape and a non-stationary DISTURBANCE landscape
# simultaneously.  The old order did exactly that, and it also put the difficult end of the
# envelope last:
#
#   envelope: 5.0 -> 1.5 spread over the WHOLE 30M (indexed by frac = step/30M), so
#             `eff = scale/1.5` - the multiplier that actually tightens the tolerances -
#             only reached 1.0 at frac = 1.0, i.e. on the LAST iteration;
#   dr:       0 -> 1 over [10M, 20M] and then held at 1.0, so the maximum disturbance
#             arrived while the envelope was still 2.67 (eff 1.78) and still moving.
#
# MEASURED 2026-09-24 on the 30M run at step 7.9M: while `envelope_scale >= 4.0` the
# reference-relative guard is OFF ENTIRELY (see `ENVELOPE_FREE_THRESHOLD` in spec.py), which
# is 29% of the budget trained with no tunnel / flip-ceiling termination at all.  The order
# below spends 12M steps reaching the FINAL constraint set and then HOLDS it for 18M, so ADR
# lands on a fixed envelope and the configuration that is actually GRADED (envelope 1.5) is
# trained against for 60% of the run instead of ~0%.
#
# The breakpoints are written in the absolute steps of the 30M production run because that is
# how they are reasoned about.  `Curriculum.for_budget` maps them onto a shorter budget, so a
# short run tests the ORDERING rather than silently skipping the curriculum (with absolute
# breakpoints a 10M run would leave dr at 0.0 for its entire life).
ENVELOPE_END_STEP = 12_000_000      # envelope reaches 1.5 here, then HOLDS at 1.5
DR_START_STEPS = 14_000_000         # ADR starts only after the envelope has settled
DR_END_STEPS = 24_000_000           # and is fully ramped 10M steps later
CHAIN_MIX_START_STEPS = 24_000_000
CHAIN_MIX_END_STEPS = TOTAL_TIMESTEPS
EVAL_EVERY_STEPS = 1_000_000
# A dr=1 sweep, mirroring Simulation/train.py's EVAL_ROBUST_DR / EVAL_ROBUST_EVERY.
# Without it the ADR curriculum is UNFALSIFIABLE: `dr` changes 0 -> 1 and nothing ever
# measures what it bought.  Nominally the eval is dr=0 at the trained-end envelope.
ROBUST_EVAL_DR = 1.0
ROBUST_EVAL_EVERY = 4
SEED = 42

# ======================================================================================
# throughput knobs
# ======================================================================================
# TIP 1: the trajectory pool.  The pool holds this many pre-drawn `TrajSpec`s per rollout
# iteration and a reset is then a slice of it, instead of a full rejection-sampled draw for
# every env on every step.  Expected terminations in a rollout are
# `batch / mean_episode_len` (~20480/260 ~ 79 at the production shape), so one pool entry
# per `TRAJ_POOL_PER_BATCH` env-steps leaves ~2.6x headroom; the index WRAPS, so an
# unusually terminal-heavy rollout degrades to trajectory reuse rather than to a reset per
# env.  Nothing about the reference distribution changes: every entry is a real `sample()`
# draw under the rollout's live weights.
TRAJ_POOL_MIN = 16          # floor, so a small smoke batch still has a usable pool
TRAJ_POOL_PER_BATCH = 100   # one pool entry per this many env-steps in a rollout

# TIP 4: train without MJX's contact pipeline.  A ground touch terminates the episode, so
# the collision/constraint/solver work is never trained through.  The ground test falls back
# to the geometric `qpos[2] <= GROUND_TERMINATE_Z`.  Set False to reproduce the old physics
# exactly (slower); the parity checker and every diagnostic build their own env, so they are
# unaffected either way.
DISABLE_CONTACTS = True

# Donate `ts`/`env_states`/`obs` so their input buffers are reused instead of copied each
# iteration.  MEASURED (already-hot machine, median of the last iterations): 0.998 s/iter
# with donation vs 1.122 s without, so it is worth keeping -- but the two arms ran in
# sequence on a fanless box, so treat the size of the win as soft.  It does NOT cause an
# extra compilation (that was a weak/strong dtype mismatch in the env state -- see the note
# in `quad_mjx/env.py::_reset_with_spec`).
DONATE_BUFFERS = True

# TIP 3: the whole cost model assumes float32.  x64 doubles the memory traffic of a
# 1024-env vmap and buys no accuracy we deploy -- `sampler` even pins int32 in places
# because x64 promotes literals.  `check_mjx_parity.py` turns x64 ON on purpose (the
# parity comparison wants float64), so warn loudly instead of silently paying 2x.
if jax.config.jax_enable_x64:
    import warnings

    warnings.warn(
        "jax_enable_x64 is ON: training would pay ~2x memory traffic for nothing. "
        "The parity checker enables it deliberately; a training run must not.",
        RuntimeWarning, stacklevel=2)


def _lerp(lo, hi, frac):
    return lo + (hi - lo) * frac


class Curriculum(NamedTuple):
    """
    The scheduled difficulty axes, as ENV-STEP breakpoints valid for one run.

    Held in absolute steps rather than fractions because that is how the breakpoints are
    reasoned about; `for_budget` is the only constructor that should be used for a run whose
    length differs from `TOTAL_TIMESTEPS`.
    """
    total: float = TOTAL_TIMESTEPS
    env_end: float = ENVELOPE_END_STEP
    dr_start: float = DR_START_STEPS
    dr_end: float = DR_END_STEPS
    chain_start: float = CHAIN_MIX_START_STEPS
    chain_end: float = CHAIN_MIX_END_STEPS
    lr_warmup: float = LR_WARMUP_STEPS
    lr_phase2_end: float = LR_ADR_END_STEP
    env_start_val: float = ENVELOPE_START
    env_end_val: float = ENVELOPE_END

    @classmethod
    def for_budget(cls, total: float) -> "Curriculum":
        """
        The production 30M shape, squeezed onto a run of `total` steps.

        Identity at `total == TOTAL_TIMESTEPS`, so the fitted 30M schedule is unchanged by
        this being a function rather than a set of constants.
        """
        f = float(total) / float(TOTAL_TIMESTEPS)
        return cls(total=float(total), env_end=ENVELOPE_END_STEP * f,
                   dr_start=DR_START_STEPS * f, dr_end=DR_END_STEPS * f,
                   chain_start=CHAIN_MIX_START_STEPS * f, chain_end=CHAIN_MIX_END_STEPS * f,
                   lr_warmup=LR_WARMUP_STEPS * f, lr_phase2_end=LR_ADR_END_STEP * f,
                   env_start_val=ENVELOPE_START, env_end_val=ENVELOPE_END)

    def summary(self) -> str:
        """One line, in millions of steps, for the run banner."""
        M = 1e6
        return (f"envelope {self.env_start_val:.1f}->{self.env_end_val:.1f} over "
                f"[0, {self.env_end / M:.2f}M] then held | dr 0->1 over "
                f"[{self.dr_start / M:.2f}M, {self.dr_end / M:.2f}M] | chain_w "
                f"{CHAIN_MIX_START_W}->{CHAIN_MIX_END_W} over "
                f"[{self.chain_start / M:.2f}M, {self.chain_end / M:.2f}M]")


DEFAULT_CURRICULUM = Curriculum.for_budget(TOTAL_TIMESTEPS)


# ======================================================================================
# PPO HYPERPARAMETERS AS A CONFIG  (port equivalence with Simulation/train.py)
# ======================================================================================
# WHY THIS IS A STRUCT.  The MJX port is not the same optimizer as SB3, and the parameters
# that were tuned on the Mac for SB3 do not transfer just because the LR endpoints were
# copied.  Both sides collect an identical 20,480 samples per iteration; what differs is HOW
# MANY optimizer steps that sample block is spent on, and the answer was 6.25x fewer:
#
#                                train.py (SB3)      train_mjx.py (original port)
#   envs x rollout                8 x 2560            1024 x 20
#   batch / iteration             20,480              20,480        (matched)
#   minibatch                     512                 2,560         (5x larger)
#   epochs                        5                   4
#   UPDATES / ITERATION           200                 32            1 / 6.25
#   updates per sample (UTD)      9.8e-3              1.6e-3        1 / 6.25
#   travel / iteration = U*LR     6.0e-2              9.6e-3        1 / 6.25
#
# So at the SAME LR endpoints the port advances the parameters 6.25x less far per unit of
# collected experience.  That is a learning-rate deficit wearing a batching costume, and
# there are two standard ways to pay it back, which pull in OPPOSITE directions:
#
#   * LINEAR SCALING (Goyal et al. 2017, 'Accurate, Large Minibatch SGD'): LR proportional to
#     batch, with warmup.  minibatch 512 -> 2560 is 5x, so LR -> 1.5e-3.
#   * CRITICAL BATCH SIZE (McCandlish et al. 2018, 'An Empirical Model of Large-Batch
#     Training'): LR should rise LINEARLY with batch only up to B_crit and as sqrt(batch)
#     beyond it, because past B_crit the extra samples are not buying gradient signal.
#     sqrt(5) = 2.24x -> LR_start ~ 6.7e-4.  For PPO, which is clipped and has no trust region
#     beyond clip_eps, sqrt-scaling is the safe branch: LR >= 1e-3 is a known instability
#     zone at clip 0.2.
#
# Holding `updates x LR` constant instead - the deficit view - would need 6.0e-2 / 32 = 1.9e-3
# at 32 updates, which is squarely in that instability zone.  So the deficit is split: some
# of it paid by MORE updates (which costs wall clock but no stability), the rest by a
# moderately larger LR.  `BASELINE_ALIGNED` is the other end of that trade - everything
# restored to the values train.py was actually tuned and measured with.
class PPOConfig(NamedTuple):
    epochs: int = NUM_EPOCHS
    minibatches: int = NUM_MINIBATCHES
    lr_scale: float = 1.0
    ent_start: float = ENT_COEF_START
    ent_end: float = ENT_COEF_END
    max_log_std: float = MAX_LOG_STD
    floor_start: tuple = LOG_STD_FLOOR_START
    floor_end: tuple = LOG_STD_FLOOR_END
    log_std_init: float = LOG_STD_INIT
    bound_coef: float = 2.0

    @property
    def updates_per_iter(self) -> int:
        return self.epochs * self.minibatches

    def entropy_at(self, frac: float) -> float:
        return _lerp(self.ent_start, self.ent_end, frac)

    def floor_at(self, frac: float) -> tuple:
        return tuple(_lerp(a, b, frac) for a, b in zip(self.floor_start, self.floor_end))

    def summary(self, batch: int) -> str:
        mb = batch // self.minibatches
        return (f"PPO: mb {mb} x {self.epochs} ep x {self.minibatches} mb = "
                f"{self.updates_per_iter} updates/iter (train.py: 512 x 5 x 40 = 200) | "
                f"LR x{self.lr_scale:.2f} | ent {self.ent_start}->{self.ent_end} | "
                f"log_std [{self.log_std_init:.2f} init, {self.max_log_std:.2f} max] | "
                f"floor {self.floor_start}->{self.floor_end}")


# Every deviation below is a value `train.py` was MEASURED with (its own comments record the
# measurements), which the port had quietly changed.  All four reductions are in the same
# direction - LESS exploration - and `train.py`'s ENT_COEF comment names the exact failure
# they cause: "a policy that gives up on the flip can still collect decent average reward
# from hover and waypoints, which is exactly the kind of local optimum entropy pressure is
# the standard antidote to."
BASELINE_ALIGNED = PPOConfig(
    ent_start=0.01,                 # train.py ENT_COEF
    ent_end=0.001,                  # train.py ENT_COEF_END
    max_log_std=0.0,                # train.py max_log_std (the port had -0.7)
    floor_start=(-1.0, -1.0, -1.0, -1.0),
    floor_end=(-2.0, -3.0, -3.0, -3.0),   # train.py LOG_STD_FLOOR_END, measured
    log_std_init=-0.5,              # train.py log_std_init
)

DEFAULT_PPO = PPOConfig()


def schedule_at(step: float, cur: Curriculum = DEFAULT_CURRICULUM,
                cfg: PPOConfig = DEFAULT_PPO) -> dict:
    """
    Every scheduled quantity at `step`, as Python floats.

    `cur` is used AS GIVEN - it must already be valid for this run's budget (build it with
    `Curriculum.for_budget`, or take the default for a 30M run).
    """
    s = float(step)
    if s < cur.lr_warmup:
        lr = _lerp(LR_START, LR_MID, s / max(1.0, cur.lr_warmup))
    elif s < cur.lr_phase2_end:
        lr = _lerp(LR_MID, LR_ADR_END,
                   (s - cur.lr_warmup) / max(1.0, cur.lr_phase2_end - cur.lr_warmup))
    else:
        lr = _lerp(LR_ADR_END, LR_FLOOR,
                   (s - cur.lr_phase2_end) / max(1.0, cur.total - cur.lr_phase2_end))
    dr = float(np.clip((s - cur.dr_start) / max(1.0, cur.dr_end - cur.dr_start), 0.0, 1.0))
    mix = float(np.clip((s - cur.chain_start)
                        / max(1.0, cur.chain_end - cur.chain_start), 0.0, 1.0))
    frac = s / max(1.0, cur.total)
    # The envelope RAMP ENDS and then holds; the clip is what makes the hold explicit rather
    # than relying on `frac` never exceeding 1.
    env_frac = float(np.clip(s / max(1.0, cur.env_end), 0.0, 1.0))
    return {
        "lr": lr * cfg.lr_scale,
        "ent": cfg.entropy_at(frac),
        "dr": dr,
        "envelope": _lerp(cur.env_start_val, cur.env_end_val, env_frac),
        "chain_w": _lerp(CHAIN_MIX_START_W, CHAIN_MIX_END_W, mix),
        "floor": cfg.floor_at(frac),
    }


def build_schedule(total_timesteps: int, batch: int,
                   cur: Curriculum | None = None,
                   cfg: PPOConfig = DEFAULT_PPO):
    cur = Curriculum.for_budget(total_timesteps) if cur is None else cur
    steps = np.arange(0, total_timesteps, batch, dtype=np.float64)
    rows = [schedule_at(s, cur, cfg) for s in steps]
    return {
        "lr": jnp.asarray([r["lr"] for r in rows], jnp.float32),
        "ent": jnp.asarray([r["ent"] for r in rows], jnp.float32),
        "floor": jnp.asarray([r["floor"] for r in rows], jnp.float32),
        "n": len(rows),
    }


# ======================================================================================
# network
# ======================================================================================
class ActorCritic(nn.Module):
    """
    pi = [32] (the 2026-09-23 architecture), vf = [512, 256, 128].

    The actor sees `[o_t | z | ref_ff]` (48 dims) and the critic the full `[o_t | z |
    ref_ff | aux | privileged]` (96), so this is asymmetric exactly as
    ``AsymmetricActorCriticPolicy`` is.  `log_std` is a learned vector initialised at 0.0,
    matching SB3's default.
    """
    action_dim: int = 4
    pi_dims: tuple = (32, 32)
    vf_dims: tuple = (512, 256, 128)
    log_std_init: float = LOG_STD_INIT

    @nn.compact
    def __call__(self, actor_obs, critic_obs):
        x = actor_obs
        for i, d in enumerate(self.pi_dims):
            x = nn.tanh(nn.Dense(d, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                                 name=f"actor_fc{i}")(x))
        action_mean = nn.Dense(self.action_dim,
                               kernel_init=nn.initializers.orthogonal(0.01),
                               name="action_net")(x)
        log_std = self.param("log_std", nn.initializers.constant(self.log_std_init),
                             (self.action_dim,))

        y = critic_obs
        for i, d in enumerate(self.vf_dims):
            y = nn.tanh(nn.Dense(d, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                                 name=f"critic_fc{i}")(y))
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), name="value_net")(y)
        return action_mean, log_std, jnp.squeeze(value, axis=-1)


def gaussian_log_prob(action, mean, log_std):
    var = jnp.exp(2.0 * log_std)
    return -0.5 * jnp.sum(((action - mean) ** 2) / var + 2.0 * log_std
                          + jnp.log(2.0 * jnp.pi), axis=-1)


def _select_env(new, old, done):
    """
    Per-env ``where(done, new, old)`` over an arbitrary pytree of env state/observation.

    ``done`` is ``(num_envs,)``, but the leaves range from ``(num_envs,)`` to
    ``(num_envs, ...)`` and include ``()`` for values vmap left unbatched (model
    constants such as ``opt.gravity``).  A bare ``jnp.where(done, r, s)`` fails to
    broadcast ``(num_envs,)`` against ``(num_envs, 3)`` -- the mask has to be reshaped to
    the leaf's rank.  An unbatched leaf is identical in every env, so ``new`` is already
    the right answer for it.
    """
    def go(n, o):
        if jnp.ndim(n) == 0:
            return n
        mask = done.reshape((-1,) + (1,) * (jnp.ndim(n) - 1))
        return jnp.where(mask, n, o)
    return jax.tree_util.tree_map(go, new, old)


def create_tx(params, lr_fn, max_grad_norm=MAX_GRAD_NORM):
    """
    Decoupled optimizer transform for Asymmetric Actor-Critic.

    Actor and Critic have completely independent parameter trees (1.7k params vs 214k params).
    Applying a single global norm clip pools their gradients together, which caused critic
    value residuals (v_loss ~ 1300) to dominate total norm and squash actor gradients by 100x.
    multi_transform clips actor and critic gradients independently so the actor always retains
    its full gradient learning signal.
    """
    def param_labels(path_tree):
        def label_fn(path, _):
            p_str = "/".join(str(getattr(p, "key", p)) for p in path)
            if any(k in p_str for k in ("actor", "action", "log_std")):
                return "actor"
            return "critic"
        return jax.tree_util.tree_map_with_path(label_fn, path_tree)

    labels = param_labels(params)
    return optax.multi_transform(
        {
            "actor": optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr_fn, eps=1e-5),
            ),
            "critic": optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr_fn, eps=1e-5),
            ),
        },
        labels,
    )


class Transition(NamedTuple):
    actor_obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array          # the RAW Gaussian sample; the env clips it to [-1, 1]
    log_prob: jax.Array        # log-prob of THAT sample, not of the clipped value
    value: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array


# ======================================================================================
# training
# ======================================================================================
def make_iteration(env: QuadFlipMJXEnv, schedule, num_envs: int, num_steps: int,
                   cfg: PPOConfig = DEFAULT_PPO):
    """Return a jitted `iterate(ts, states, obs, rng, dr, envelope, chain_w)`."""
    batch = num_envs * num_steps
    mb = batch // cfg.minibatches
    # `TrainState.step` is owned by optax and bumps once per MINIBATCH, so one rollout
    # iteration advances it by exactly this and every schedule lookup divides by it.
    updates_per_iter = cfg.updates_per_iter
    net = ActorCritic(log_std_init=cfg.log_std_init)

    # TIP 1: pool size, and the static lane index used to spread pool entries over lanes.
    pool_size = max(TRAJ_POOL_MIN, batch // TRAJ_POOL_PER_BATCH)
    lanes = jnp.arange(num_envs, dtype=jnp.int32)

    def _weights(chain_w):
        return SAMP.default_weights().at[S.KIND_INDEX["chain"]].set(chain_w)

    def step_one(state, act, weights):
        old = env.weights
        env.weights = weights
        try:
            return env.step(state, act)
        finally:
            env.weights = old

    # donate_argnums: `ts`, `env_states` and `obs` are rebound from the return value every
    # iteration, so their input buffers can be reused instead of copied.  Set
    # `DONATE_BUFFERS = False` to compare (measured ~11% slower, but see the caveat there).
    @jax.jit(donate_argnums=(0, 1, 2) if DONATE_BUFFERS else ())
    def iterate(ts: TrainState, env_states, obs, rng, dr, envelope, chain_w):
        step_idx = jnp.minimum(ts.step // updates_per_iter, schedule["n"] - 1)
        floor = schedule["floor"][step_idx]
        ent = schedule["ent"][step_idx]
        weights = _weights(chain_w)

        # --- TIP 1: the trajectory POOL ---------------------------------------------
        # `env.reset` used to run for EVERY env on EVERY step, and its result was then
        # discarded for all but the few lanes that actually finished.  With 1024 envs and
        # ~260-step episodes that is ~250x the necessary work, spent on the most expensive
        # call in the codebase (the rejection sampler -- the reference notes measure the
        # reference draw at 94% of a reset).  XLA cannot elide it: a data-dependent
        # `while_loop` under vmap runs every lane, and the reset result is materialised for
        # all of them.  So draw the pool ONCE per rollout and make reset a slice.
        rng, k_pool = jax.random.split(rng)
        pool = SAMP.sample_batch(jax.random.split(k_pool, pool_size), env.traj_cfg,
                                 weights, mass=S.MASS)

        def rollout(carry, step_i):
            env_states, obs, rng = carry
            rng, k_act, k_reset = jax.random.split(rng, 3)
            mean, log_std, value = net.apply(ts.params, obs.actor_obs, obs.critic_obs)
            log_std = jnp.clip(log_std, jnp.maximum(floor, MIN_LOG_STD), cfg.max_log_std)
            # THE SAMPLE IS MEASURED BEFORE IT IS CLIPPED, and the RAW sample is what gets
            # stored.  SB3 does exactly this - it keeps <unclipped action, log_prob of that
            # action> in the buffer and clips only what reaches the env
            # (`clipped_actions = np.clip(actions, low, high)` in `collect_rollouts`).
            #
            # Clipping FIRST and taking the log-prob of the BOUNDARY is not a cosmetic
            # difference, it is a runaway.  The surrogate's gradient in mu is
            # `(mu - clip(a))/sigma^2`, which points AWAY from the boundary with magnitude
            # proportional to |mu|, so the update pushes the mean out without bound.
            # MEASURED 2026-09-24 on the 30M run:
            #   * actor mean std across states 7.1-8.1 (untrained: 0.003), 92.4% of steps
            #     had a channel clipped, 56-88% of the mean itself sat outside [-1, 1];
            #   * |log_prob(clipped) - log_prob(raw)| averaged 286 nats (untrained: 0.97),
            #     which also makes the importance ratio meaningless;
            #   * reward/step fell 5.16 (iter 0) -> 2.0 and stayed there, and the final
            #     policy scored 38.1% of ceiling vs 56.3% for an UNTRAINED net.
            raw = mean + jax.random.normal(k_act, mean.shape) * jnp.exp(log_std)
            action = jnp.clip(raw, -1.0, 1.0)
            log_prob = gaussian_log_prob(raw, mean, log_std)

            nxt, nobs, reward, term, trunc = jax.vmap(
                lambda s, a: step_one(s, a, weights))(env_states, action)

            done = term | trunc
            # Every lane gets a distinct pool entry, and the offset advances with the step
            # so a lane never resets onto the trajectory it used at the previous step.
            idx = (step_i + lanes) % pool_size
            specs = jax.tree_util.tree_map(lambda a: a[idx], pool)
            rstates, robs = jax.vmap(env.reset_from_spec, in_axes=(0, 0, None, None))(
                jax.random.split(k_reset, num_envs), specs, dr, envelope)
            nxt = _select_env(rstates, nxt, done)
            nobs = _select_env(robs, nobs, done)
            return (nxt, nobs, rng), Transition(
                obs.actor_obs, obs.critic_obs, raw, log_prob, value, reward, term, trunc)

        (env_states, obs, rng), traj = jax.lax.scan(
            rollout, (env_states, obs, rng), jnp.arange(num_steps), length=num_steps)

        _, _, last_val = net.apply(ts.params, obs.actor_obs, obs.critic_obs)

        def gae_step(carry, tr):
            gae, next_val = carry
            delta = tr.reward + GAMMA * next_val * (1.0 - tr.terminated) - tr.value
            gae = delta + GAMMA * GAE_LAMBDA * (1.0 - tr.terminated) * gae
            return (gae, tr.value), (gae, gae + tr.value)

        _, (adv, ret) = jax.lax.scan(gae_step, (jnp.zeros(num_envs), last_val),
                                     traj, reverse=True)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        a_flat = traj.actor_obs.reshape((-1, env.actor_dim))
        c_flat = traj.critic_obs.reshape((-1, S.CRITIC_OBS_DIM))
        act_flat = traj.action.reshape((-1, S.ACTION_DIM))
        lp_flat = traj.log_prob.reshape((-1,))
        adv_flat = adv.reshape((-1,))
        ret_flat = ret.reshape((-1,))
        val_flat = traj.value.reshape((-1,))

        def update_epoch(carry, rng):
            ts, = carry
            rng, k_perm = jax.random.split(rng)
            perm = jax.random.permutation(k_perm, batch)
            A, C = a_flat[perm], c_flat[perm]
            AC, LP = act_flat[perm], lp_flat[perm]
            AD, RT = adv_flat[perm], ret_flat[perm]
            VL = val_flat[perm]
            mbs = (A.reshape((cfg.minibatches, mb, -1)), C.reshape((cfg.minibatches, mb, -1)),
                   AC.reshape((cfg.minibatches, mb, -1)), LP.reshape((cfg.minibatches, mb)),
                   AD.reshape((cfg.minibatches, mb)), RT.reshape((cfg.minibatches, mb)),
                   VL.reshape((cfg.minibatches, mb)))

            def mb_step(ts, m):
                ao, co, ac, lp_, ad, rt, old_v = m

                def loss_fn(params):
                    mu, log_sd, val = net.apply(params, ao, co)
                    log_sd = jnp.clip(log_sd, jnp.maximum(floor, MIN_LOG_STD), cfg.max_log_std)
                    ratio = jnp.exp(gaussian_log_prob(ac, mu, log_sd) - lp_)
                    s1 = ratio * ad
                    s2 = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * ad
                    pi_loss = -jnp.mean(jnp.minimum(s1, s2))

                    # PPO clipped value loss
                    v_clipped = old_v + jnp.clip(val - old_v, -10.0, 10.0)
                    v_loss_unclipped = (val - rt) ** 2
                    v_loss_clipped = (v_clipped - rt) ** 2
                    v_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))

                    entropy = jnp.mean(jnp.sum(log_sd + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi))))
                    bound_loss = jnp.mean(jnp.square(jnp.maximum(0.0, jnp.abs(mu) - 1.0)))
                    total = pi_loss + VF_COEF * v_loss - ent * entropy + cfg.bound_coef * bound_loss
                    return total, (pi_loss, v_loss, entropy)

                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                return ts.apply_gradients(grads=grads), aux

            ts, aux = jax.lax.scan(mb_step, ts, mbs)
            return (ts,), (jnp.mean(aux[0]), jnp.mean(aux[1]), jnp.mean(aux[2]))

        (ts,), ep = jax.lax.scan(update_epoch, (ts,), jax.random.split(rng, cfg.epochs))
        return ts, env_states, obs, rng, {"rew": traj.reward.mean(),
                                          "pi": ep[0].mean(), "vf": ep[1].mean(),
                                          "ent": ep[2].mean()}

    return iterate, net


# ======================================================================================
# host loop
# ======================================================================================
def run_training(total_timesteps: int = TOTAL_TIMESTEPS, num_envs: int = NUM_ENVS,
                 num_steps: int = NUM_STEPS, seed: int = SEED,
                 eval_every: int = EVAL_EVERY_STEPS, verbose: bool = True,
                 logs_dir: str | None = None, tag: str = "",
                 encoder_path: str | None = None,
                 curriculum: Curriculum | None = None,
                 robust_eval_dr: float = ROBUST_EVAL_DR,
                 robust_eval_every: int = ROBUST_EVAL_EVERY,
                 cfg: PPOConfig = DEFAULT_PPO,
                 init_weights: str | None = None):
    batch = num_envs * num_steps
    updates_per_iter = cfg.updates_per_iter
    # A silent `reshape((minibatches, mb, -1))` failure would surface as an opaque XLA error
    # several compilations in, so reject it here where the numbers are still visible.
    if batch % cfg.minibatches:
        raise SystemExit(f"num_envs*num_steps = {batch} is not divisible by "
                         f"minibatches = {cfg.minibatches}")
    n_iters = max(1, total_timesteps // batch)
    cur = Curriculum.for_budget(total_timesteps) if curriculum is None else curriculum
    # `encoder_path=None` keeps the env's own default (`logs/encoder_gru.pt`).  The pipeline
    # passes the checkpoint it just trained, so a run can never silently pick up an older
    # encoder than the one Stage 2 produced.
    env = QuadFlipMJXEnv(config=EnvConfig(disable_contacts=DISABLE_CONTACTS),
                         encoder_path=encoder_path)
    schedule = build_schedule(total_timesteps, batch, cur, cfg)
    iterate, net = make_iteration(env, schedule, num_envs, num_steps, cfg)
    if verbose:
        pool_size = max(TRAJ_POOL_MIN, batch // TRAJ_POOL_PER_BATCH)
        print(f"[mjx] backend={jax.default_backend()} x64={jax.config.jax_enable_x64} "
              f"float32-default | contacts={'OFF' if DISABLE_CONTACTS else 'ON'} | "
              f"traj pool={pool_size}/rollout (was 1 reset per env per step)")
        # The PPO config is printed next to the batch because the two together are what
        # determine how far the parameters travel per unit of collected experience, and that
        # is the quantity that did NOT transfer from train.py.
        print(f"[mjx] batch {num_envs} x {num_steps} = {batch:,} "
              f"({n_iters:,} iterations)")
        print(f"[mjx] {cfg.summary(batch)}")
        # The curriculum is printed because it is the one thing that determines WHEN the run
        # stops being an exploration phase.  `envelope` is also on the per-iteration line and
        # in the metrics CSV - an invisible curriculum is how a 29%-of-budget free phase went
        # unnoticed.
        print(f"[mjx] curriculum: {cur.summary()}")
        print(f"[mjx] eval: every {eval_every:,} steps at envelope "
              f"{S.ENVELOPE_SCALE_END:.2f}, dr=0"
              + (f" + dr={robust_eval_dr:.2f} every {max(1, robust_eval_every)}-th eval"
                 if robust_eval_dr > 0.0 else ""))

    logs = logs_dir or os.path.join(_ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    metrics_path = os.path.join(logs, f"training_metrics_mjx{tag}.csv")
    # ONE FILE PER RUN.  This file used to be opened in append mode and to write its header
    # only when it was new, but `step` restarts at 0 every run - so a second run silently
    # interleaved its rows with the first one's and every plot of the file drew several runs
    # on the same axis.  MEASURED 2026-09-24: `logs/training_metrics_mjx.csv` held 2,579 rows
    # after a run that only reached iteration 723.  The stale file is archived instead.
    for _stale in (metrics_path,
                   os.path.join(logs, f"eval_metrics_mjx{tag}.csv"),
                   os.path.join(logs, f"eval_robust_mjx{tag}.csv")):
        _archive_stale_metrics(_stale)
    csv = open(metrics_path, "w")
    csv.write("step,reward,policy_loss,value_loss,entropy,lr,dr,envelope,chain_w\n")

    rng = jax.random.PRNGKey(seed)
    rng, k_net, k_env = jax.random.split(rng, 3)
    params = init_params(net, env, seed)
    if init_weights is not None:
        if not os.path.isfile(init_weights):
            raise FileNotFoundError(f"--init-weights checkpoint not found: {init_weights}")
        if verbose:
            print(f"[mjx] warm-starting parameters from {init_weights}")
        params = load_params(init_weights, params)
    tx = create_tx(
        params,
        lambda c: schedule["lr"][jnp.minimum(c // updates_per_iter, schedule["n"] - 1)],
        max_grad_norm=MAX_GRAD_NORM,
    )
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    # jitted: an eager batched reset re-dispatches every primitive of a 40-attempt sampler
    v_reset = jax.jit(jax.vmap(lambda k, d, e: env.reset(k, -1, d, e),
                               in_axes=(0, None, None)))
    env_states, obs = v_reset(jax.random.split(k_env, num_envs), 0.0, cur.env_start_val)

    t0 = time.time()
    history = []
    n_eval = 0
    for it in range(n_iters):
        row = schedule_at(it * batch, cur, cfg)
        ts, env_states, obs, rng, stats = iterate(
            ts, env_states, obs, rng,
            jnp.asarray(row["dr"], jnp.float32),
            jnp.asarray(row["envelope"], jnp.float32),
            jnp.asarray(row["chain_w"], jnp.float32))
        step = int(ts.step) // updates_per_iter * batch   # ENV steps, for the schedule
        csv.write("%d,%.4f,%.6f,%.6f,%.4f,%.3e,%.3f,%.3f,%.3f\n" % (
            step, float(stats["rew"]), float(stats["pi"]), float(stats["vf"]),
            float(stats["ent"]), row["lr"], row["dr"], row["envelope"], row["chain_w"]))
        csv.flush()
        history.append((step, float(stats["rew"])))
        if verbose and (it % 10 == 0 or it == n_iters - 1):
            print("  iter %5d/%d  step %9d  R/step %7.3f  pi %8.4f  vf %9.2f  "
                  "ent %.4f  lr %.2e  dr %.2f  env %.2f  chain_w %.2f  (%.0fs)"
                  % (it, n_iters, step, float(stats["rew"]), float(stats["pi"]),
                     float(stats["vf"]), float(stats["ent"]), row["lr"], row["dr"],
                     row["envelope"], row["chain_w"], time.time() - t0))
        if eval_every and (step // eval_every) != ((step - batch) // eval_every):
            n_eval += 1
            # dr=1 only every `robust_eval_every`-th eval (mirrors train.py): the robust
            # sweep is a second full vmap over every family, so running it at every eval
            # would make a diagnostic a measurable fraction of the wall clock.
            r_dr = (robust_eval_dr if robust_eval_dr > 0.0
                    and n_eval % max(1, robust_eval_every) == 0 else 0.0)
            run_eval(env, net, ts.params, step, logs, robust_dr=r_dr, tag=tag)
    csv.close()

    ckpt = os.path.join(logs, f"quad_mjx_policy{tag}.npz")
    np.savez(ckpt, **_flatten_params(ts.params))
    if verbose:
        done = int(ts.step) // updates_per_iter * batch
        print(f"\n[done] {done:,} env steps in {time.time() - t0:.0f}s -> {ckpt}")
    return ts, net, env, history


def _flatten_params(params):
    """Flatten a Flax param tree to name -> array, so a reload is unambiguous."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = "/".join(str(getattr(p, "key", p)) for p in path)
        out[name] = np.array(leaf)
    return out


def load_params(path: str, template=None):
    """
    Inverse of ``_flatten_params``.

    With no ``template`` it returns the FLAT ``{name: array}`` dict (what
    ``export_to_sb3.py`` wants, since it maps those names onto SB3's own keys).  Pass the
    tree from ``net.init`` and it returns a real nested Flax param tree, because a flat
    dict is NOT a valid pytree for ``net.apply`` -- the slash-joined names have to be
    split back into the nested structure.
    """
    flat = dict(np.load(path))
    if template is None:
        return {k: jnp.asarray(v) for k, v in flat.items()}
    leaves = []
    for p, _leaf in jax.tree_util.tree_flatten_with_path(template)[0]:
        name = "/".join(str(getattr(q, "key", q)) for q in p)
        if name not in flat:
            raise KeyError(f"{os.path.basename(path)} is missing {name!r}; "
                           f"it has {sorted(flat)[:6]}...")
        leaves.append(jnp.asarray(flat[name]))
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template), leaves)


def init_params(net, env, seed: int = 0):
    """A correctly-shaped param tree to use as the template for ``load_params``."""
    return net.init(jax.random.PRNGKey(seed),
                    jnp.zeros((1, env.actor_dim)), jnp.zeros((1, S.CRITIC_OBS_DIM)))


def _build_eval_fn(env: QuadFlipMJXEnv, net):
    TRACK_CEILING = (S.TRACK_W_POS + S.TRACK_W_VEL + S.TRACK_W_ATT + S.TRACK_W_RATE)

    @jax.jit
    def run(params, idx, keys, envelope, dr):
        def one(key, i):
            st, obs = env.reset(key, i, dr, envelope)

            def body(carry, _):
                s, o = carry
                mean, _, _ = net.apply(params, o.actor_obs[None], o.critic_obs[None])
                s, o, r, term, trunc = env.step(s, mean[0])
                # carry the per-term breakdown: it EXCLUDES the termination penalty, so the
                # tracking score below is comparable across penalty values and is not moved
                # by how long the episode happened to survive.
                return (s, o), (r, term | trunc, o.reward_terms)

            (s, o), (rew, done, terms) = jax.lax.scan(body, (st, obs), None,
                                                     length=env.max_steps)
            # step i still counts if the episode ended ON step i, so the mask is
            # "no termination strictly before i"
            ended_before = jnp.concatenate(
                [jnp.zeros(1, bool), jnp.cumsum(done.astype(jnp.int32))[:-1] > 0])
            alive = ~ended_before
            # FIXED-HORIZON return over env.max_steps (post-termination counted as 0 reward).
            # The alive-steps divisor rewarded early termination; fixed-horizon eliminates this.
            score = jnp.sum(jnp.where(alive, rew, 0.0)) / env.max_steps
            trk = (S.TRACK_W_POS * terms[:, 0] + S.TRACK_W_VEL * terms[:, 1]
                   + S.TRACK_W_ATT * terms[:, 2] + S.TRACK_W_RATE * terms[:, 3])
            track = 100.0 * jnp.sum(jnp.where(alive, trk, 0.0)) / (env.max_steps * TRACK_CEILING)
            return (100.0 * score / S.REWARD_CEILING_PER_STEP, jnp.sum(alive), track)

        return jax.vmap(one)(keys, idx)

    return run


def make_eval_fn(env: QuadFlipMJXEnv, net):
    """
    Batched, jitted deterministic evaluation.

    The naive version ran `env.step` in a python loop -- 1500 eager dispatches per episode
    x 10 families, every eval.  This vmapps all (family, episode) pairs into ONE compiled
    scan, and masks the reward so a terminated episode contributes exactly the steps it
    actually lived.

    CACHED on `env`, and that is not premature: `jax.jit` attaches its compilation cache to
    the FUNCTION IT RETURNS, and this used to be constructed fresh for every eval, so every
    eval recompiled the entire scan.  MEASURED 2026-09-24 at the 64-env test shape, a
    nominal + dr=1 pair cost 2 x ~46 s of pure compilation, and a 30M run performs 30 such
    evals.  Everything that actually varies between evals (`params`, `dr`, `envelope`) is a
    TRACED argument, so a single compilation serves all of them.

    Validated by IDENTITY rather than by `id()`, and the cached tuple keeps `net` alive, so
    a recycled `id()` cannot hand back a function closed over a dead env.
    """
    cached = getattr(env, "_eval_fn_cache", None)
    if cached is not None and cached[0] is net:
        return cached[1]
    run = _build_eval_fn(env, net)
    env._eval_fn_cache = (net, run)
    return run


def evaluate_families(env: QuadFlipMJXEnv, net, params, families=None, episodes: int = 1,
                      key=None, envelope=None, dr: float = 0.0):
    """
    Return (names, mean_percent_per_family, mean_episode_length_per_family,
    mean_TRACKING_percent_per_family).

    The tracking score uses only the four tracking terms - no termination penalty and no
    action-smoothness term - so it is comparable across `TERMINATION_PENALTY` settings and is
    not inflated by an early termination truncating the degrading tail of an episode.
    """
    names = list(families) if families else list(S.KIND_NAMES)
    idx = jnp.asarray([S.KIND_INDEX[n] for n in names for _ in range(episodes)], jnp.int32)
    key = jax.random.PRNGKey(1000) if key is None else key
    keys = jax.random.split(key, idx.shape[0])
    dr = jnp.asarray(dr, jnp.float32)
    env_scale = jnp.asarray(S.ENVELOPE_SCALE_END if envelope is None else envelope, jnp.float32)
    pct, length, track = make_eval_fn(env, net)(params, idx, keys, env_scale, dr)
    pct = np.asarray(pct).reshape((len(names), episodes))
    length = np.asarray(length).reshape((len(names), episodes))
    track = np.asarray(track).reshape((len(names), episodes))
    return names, pct.mean(axis=1), length.mean(axis=1), track.mean(axis=1)


def _archive_stale_metrics(path: str) -> bool:
    """
    Move a metrics file left by a PREVIOUS run aside, so a new run starts from a clean file.

    These CSVs are written in append mode and their `step` column restarts at 0 on every run,
    so without this a second run interleaves its rows with the first run's and any plot of
    the file overlays two runs on one axis.  Archived under the `*.legacy.csv` naming this
    repo already uses for superseded metrics.

    Assumes one run per file.  This trainer always starts from a fresh parameter init (there
    is no resume path), so a legitimate append into an existing file does not exist.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    base, ext = os.path.splitext(path)
    os.replace(path, f"{base}.{time.strftime('%Y%m%d_%H%M%S')}.legacy{ext}")
    return True


def _append_eval_row(path: str, header: str, row: str) -> None:
    """
    Append `row` to `path`, writing `header` first if the file is new.

    The header is re-checked on every call because these files are opened in APPEND mode: a
    changed column set would otherwise leave a file whose header describes the earlier rows
    and not the later ones.  A mismatch ARCHIVES the stale file under the `*.legacy.csv`
    convention this repo already uses for superseded metrics.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            first = f.readline().strip()
        if first != header.strip():
            base, ext = os.path.splitext(path)
            os.replace(path, f"{base}.{time.strftime('%Y%m%d_%H%M%S')}.legacy{ext}")
    new_file = not os.path.exists(path)
    with open(path, "a") as f:
        if new_file:
            f.write(header)
        f.write(row)


def run_eval(env: QuadFlipMJXEnv, net, params, step: int, logs: str,
             episodes_per_family: int = 5, verbose: bool = True, robust_dr: float = 0.0,
             tag: str = ""):
    """
    Deterministic eval: `episodes_per_family` episodes per family, scored as a percentage
    of the per-step ceiling (7.3), at the trained-end envelope.

    Reports the reward percentage AND the mean episode LENGTH.  The length is the quantity
    that responds to the flight-envelope curriculum - the percentage is a mean over ALIVE
    steps, so a termination dilutes its own penalty to `-TERMINATION_PENALTY / n` and the
    number is nearly blind to the thing the curriculum is moving.  `evaluate_families`
    already computed it; it used to be discarded into `_len`.

    `robust_dr > 0` adds a second sweep at that randomization level and logs the nominal
    cost, which is what makes the ADR curriculum falsifiable.
    """
    def sweep(dr: float):
        names, pct, length, track = evaluate_families(env, net, params,
                                                      episodes=episodes_per_family,
                                                      envelope=S.ENVELOPE_SCALE_END, dr=dr)
        return names, pct, length, track

    names, pct, elen, etrack = sweep(0.0)
    # `tag` is threaded through so a smoke / test run cannot append its evals to the
    # production CSV.  The TRAINING metrics CSV was always tagged; the eval one was not.
    _append_eval_row(os.path.join(logs, f"eval_metrics_mjx{tag}.csv"),
                     "step," + ",".join(names) + ",overall,worst,ep_len,track\n",
                     "%d,%s,%.2f,%.2f,%.1f,%.2f\n"
                     % (step, ",".join("%.2f" % v for v in pct), float(pct.mean()),
                        float(pct.min()), float(elen.mean()), float(etrack.mean())))
    if verbose:
        print("    [eval %9d] overall %.1f%%  track %.1f%%  worst %s %.1f%%  ep_len "
              "%.0f/%d (worst %s %.0f)   %s"
              % (step, float(pct.mean()), float(etrack.mean()),
                 names[int(np.argmin(pct))], float(pct.min()),
                 float(elen.mean()), env.max_steps, names[int(np.argmin(elen))],
                 float(elen.min()),
                 " ".join("%s=%.0f" % (names[i][:4], pct[i]) for i in range(len(names)))))

    if robust_dr > 0.0:
        r_names, r_pct, r_len, _r_trk = sweep(robust_dr)
        _append_eval_row(os.path.join(logs, f"eval_robust_mjx{tag}.csv"),
                         "step,dr," + ",".join(r_names) + ",overall,worst,ep_len\n",
                         "%d,%.2f,%s,%.2f,%.2f,%.1f\n"
                         % (step, robust_dr, ",".join("%.2f" % v for v in r_pct),
                            float(r_pct.mean()), float(r_pct.min()), float(r_len.mean())))
        if verbose:
            print("    [eval %9d] ROBUST dr=%.2f  overall %.1f%%  worst %s %.1f%%  "
                  "ep_len %.0f   (nominal cost %+.1f pts, %+.0f steps)"
                  % (step, robust_dr, float(r_pct.mean()),
                     r_names[int(np.argmin(r_pct))], float(r_pct.min()),
                     float(r_len.mean()), float(pct.mean()) - float(r_pct.mean()),
                     float(elen.mean()) - float(r_len.mean())), flush=True)
    return pct


# ======================================================================================
# CLI
# ======================================================================================
# The smoke shape is the one the deployment chain is validated at: a rollout batch that is
# still a multiple of NUM_MINIBATCHES (64*20 = 1280 = 8*160), so every reshape in the PPO
# update is exercised, and eval disabled because it costs a full vmap per family and proves
# nothing about the plumbing.  It writes `quad_mjx_policy_smoke.npz` and
# `training_metrics_mjx_smoke.csv`, so a smoke run can never be mistaken for a real one.
SMOKE_TIMESTEPS = 40_960
SMOKE_NUM_ENVS = 64
SMOKE_TAG = "_smoke"


def build_parser():
    import argparse

    ap = argparse.ArgumentParser(
        description="Vectorised MJX PPO trainer (port of Simulation/train.py)")
    ap.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
                    help=f"env steps to train (default {TOTAL_TIMESTEPS:,})")
    ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
    ap.add_argument("--num-steps", type=int, default=NUM_STEPS,
                    help="rollout length per iteration (batch = num_envs * num_steps)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY_STEPS,
                    help="deterministic eval period in env steps; 0 disables")
    # Curriculum ordering knobs, as FRACTIONS of the run.  None means "use the constant".
    # They exist so an ordering can be A/B'd without editing this file; the defaults are the
    # serialised order described in the module docstring.
    ap.add_argument("--envelope-end-frac", type=float, default=None,
                    help="fraction of the run at which the envelope reaches 1.5 and holds; "
                         "a value > 1 holds it loose for the whole run "
                         f"(default {ENVELOPE_END_STEP / TOTAL_TIMESTEPS:.3f})")
    ap.add_argument("--dr-start-frac", type=float, default=None,
                    help="fraction of the run at which the ADR ramp starts "
                         f"(default {DR_START_STEPS / TOTAL_TIMESTEPS:.3f})")
    ap.add_argument("--dr-end-frac", type=float, default=None,
                    help="fraction of the run at which dr reaches 1.0 "
                         f"(default {DR_END_STEPS / TOTAL_TIMESTEPS:.3f})")
    ap.add_argument("--chain-start-frac", type=float, default=None,
                    help="fraction of the run at which the chain-mixture ramp starts "
                         f"(default {CHAIN_MIX_START_STEPS / TOTAL_TIMESTEPS:.3f})")
    ap.add_argument("--robust-eval-dr", type=float, default=ROBUST_EVAL_DR,
                    help="randomization level of the robustness eval; 0 disables it")
    ap.add_argument("--robust-eval-every", type=int, default=ROBUST_EVAL_EVERY,
                    help="run the robustness eval every N-th evaluation")
    ap.add_argument("--logs-dir", default=None, help="output directory (default <root>/logs)")
    ap.add_argument("--encoder", default=None,
                    help="encoder checkpoint to freeze (default logs/encoder_gru.pt)")
    ap.add_argument("--init-weights", default=None,
                    help="path to flat npz checkpoint to warm-start parameters from")
    ap.add_argument("--envelope-start", type=float, default=None,
                    help=f"initial envelope scale (default {ENVELOPE_START})")
    ap.add_argument("--tag", default="", help="suffix for the npz / metrics filenames")
    # ---- PPO hyperparameters: the port-equivalence knobs ---------------------------------
    # Defaults are None so that `--baseline-aligned` is not clobbered field by field; the
    # resolution order is BASELINE_ALIGNED (if asked) -> any individual flag -> constant.
    ap.add_argument("--epochs", type=int, default=None,
                    help=f"PPO epochs per rollout (default {NUM_EPOCHS})")
    ap.add_argument("--minibatches", type=int, default=None,
                    help=f"PPO minibatches per epoch (default {NUM_MINIBATCHES}); "
                         f"updates/iter = epochs * minibatches, and train.py runs 200")
    ap.add_argument("--lr-scale", type=float, default=None,
                    help="multiplier applied to ALL FOUR LR endpoints. The port does 6.25x "
                         "fewer updates than train.py on the same rollout, so this is "
                         "where the sqrt(batch) rule is paid back")
    ap.add_argument("--ent-start", type=float, default=None,
                    help=f"entropy coefficient at step 0 (default {ENT_COEF_START}; "
                         f"train.py uses 0.01)")
    ap.add_argument("--ent-end", type=float, default=None,
                    help=f"entropy coefficient at the end (default {ENT_COEF_END}; "
                         f"train.py uses 0.001)")
    ap.add_argument("--max-log-std", type=float, default=None,
                    help=f"UPPER clamp on log_std (default {MAX_LOG_STD}; "
                         f"train.py uses 0.0)")
    ap.add_argument("--log-std-init", type=float, default=None,
                    help=f"initial log_std (default {LOG_STD_INIT}; train.py uses -0.5)")
    ap.add_argument("--log-std-floor-end", type=float, nargs=4, default=None,
                    metavar=("THRUST", "ROLL", "PITCH", "YAW"),
                    help="per-channel log_std floor at the END of the run "
                         f"(default {LOG_STD_FLOOR_END}; train.py uses -2 -3 -3 -3)")
    ap.add_argument("--bound-coef", type=float, default=None,
                    help="penalty weight on action means outside [-1, 1] (default 2.0)")
    ap.add_argument("--baseline-aligned", action="store_true",
                    help="restore every hyperparameter to the value Simulation/train.py "
                         "was tuned and MEASURED with: ent 0.01->0.001, max_log_std 0.0, "
                         "floors (-2,-3,-3,-3), log_std_init -0.5")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run that validates the plumbing")
    return ap


def build_ppo_config(args) -> PPOConfig:
    """
    The PPO hyperparameters for this run, from `--baseline-aligned` plus any overrides.

    Validated here rather than inside the jitted update, so a bad combination is a startup
    message and not a recompilation.
    """
    cfg = BASELINE_ALIGNED if args.baseline_aligned else DEFAULT_PPO
    over = {}
    for flag, field, cast in (("epochs", "epochs", int),
                              ("minibatches", "minibatches", int),
                              ("lr_scale", "lr_scale", float),
                              ("ent_start", "ent_start", float),
                              ("ent_end", "ent_end", float),
                              ("max_log_std", "max_log_std", float),
                              ("log_std_init", "log_std_init", float),
                              ("bound_coef", "bound_coef", float)):
        val = getattr(args, flag)
        if val is not None:
            over[field] = cast(val)
    if args.log_std_floor_end is not None:
        over["floor_end"] = tuple(float(x) for x in args.log_std_floor_end)
    if over:
        cfg = cfg._replace(**over)
    if cfg.epochs < 1 or cfg.minibatches < 1:
        raise SystemExit(f"--epochs and --minibatches must both be >= 1 "
                         f"(got {cfg.epochs}, {cfg.minibatches})")
    if cfg.lr_scale <= 0.0:
        raise SystemExit(f"--lr-scale must be > 0, got {cfg.lr_scale}")
    return cfg


def build_curriculum(args, total: int) -> Curriculum:
    """
    The production curriculum shape for a run of `total` steps, with any `--*-frac`
    overrides applied.

    Rejects an out-of-order ramp rather than silently producing a schedule that never
    reaches its end value, which is the failure mode the ordering change exists to fix.
    """
    cur = Curriculum.for_budget(total)
    over = {}
    for frac, field in ((args.envelope_end_frac, "env_end"),
                        (args.dr_start_frac, "dr_start"),
                        (args.dr_end_frac, "dr_end"),
                        (args.chain_start_frac, "chain_start")):
        if frac is not None:
            over[field] = float(frac) * total
    if over:
        cur = cur._replace(**over)
    if getattr(args, "envelope_start", None) is not None:
        cur = cur._replace(env_start_val=float(args.envelope_start))
    if not 0.0 <= cur.dr_start < cur.dr_end <= cur.total:
        raise SystemExit(f"--dr-start-frac / --dr-end-frac out of order: dr "
                         f"{cur.dr_start:.0f}..{cur.dr_end:.0f} of {cur.total:.0f} steps")
    # `env_end` is deliberately NOT clamped to `total`: a fraction > 1 means the ramp never
    # completes, i.e. the envelope stays loose for the whole run.  That is what a short
    # A/B needs - with the ramp confined to the run, a 30-iteration check collapses the
    # envelope to 1.5 and switches the reference-relative guard ON, which changes the very
    # thing under test.
    if cur.env_end <= 0.0:
        raise SystemExit(f"--envelope-end-frac must be > 0, got "
                         f"{cur.env_end / max(1.0, total):.3f}")
    if not 0.0 <= cur.chain_start < cur.chain_end:
        raise SystemExit(f"--chain-start-frac out of order: "
                         f"{cur.chain_start:.0f}..{cur.chain_end:.0f}")
    return cur


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    total, num_envs, num_steps = args.total_timesteps, args.num_envs, args.num_steps
    eval_every, tag = args.eval_every, args.tag
    if args.smoke:
        total, num_envs, eval_every = SMOKE_TIMESTEPS, SMOKE_NUM_ENVS, 0
        tag = tag or SMOKE_TAG

    batch = num_envs * num_steps
    cfg = build_ppo_config(args)
    if batch % cfg.minibatches:
        # A silent `reshape((minibatches, mb, -1))` failure here would surface as an
        # opaque XLA error three compilations in, so reject it up front.
        raise SystemExit(f"num_envs*num_steps = {batch} is not divisible by "
                         f"minibatches = {cfg.minibatches}")

    if args.smoke:
        print(f"[mjx] SMOKE: {total:,} steps, {num_envs} envs x {num_steps} steps")

    run_training(total_timesteps=total, num_envs=num_envs, num_steps=num_steps,
                 seed=args.seed, eval_every=eval_every, logs_dir=args.logs_dir, tag=tag,
                 encoder_path=args.encoder,
                 curriculum=build_curriculum(args, total),
                 robust_eval_dr=args.robust_eval_dr,
                 robust_eval_every=args.robust_eval_every,
                 cfg=cfg,
                 init_weights=args.init_weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
