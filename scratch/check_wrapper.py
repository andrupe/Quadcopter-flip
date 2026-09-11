"""
Smoke test for LatentObsWrapper: layout, auto-reset history handling, batched z, and
measured in-situ overhead.

Uses DummyVecEnv rather than SubprocVecEnv because SB3's DummyVecEnv reproduces the
SAME auto-reset contract (reset observation returned, terminal observation stashed in
info), so this exercises exactly the logic that matters while staying debuggable.

Run:  .venv/bin/python scratch/check_wrapper.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from quad_flip_env import (  # noqa: E402
    ACTOR_TOTAL_DIM,
    ENCODER_AUX_DIM,
    PRIV_TARGET_DIM,
    PRIV_TARGET_GROUPS,
    TOTAL_OBS_DIM,
    QuadFlipEnv,
)
from encoder.history_encoder import EncoderWithHead, HistoryEncoder, save_encoder_checkpoint  # noqa: E402
from encoder.latent_obs_wrapper import LatentObsWrapper  # noqa: E402
from encoder.observation_spec import ENCODER_IN_DIM, NormStats, frame_from_env_obs  # noqa: E402

Z_DIM = 16
CKPT = os.path.join(_PROJECT_ROOT, "logs", "_smoke_encoder.pt")

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def make_env():
    env = QuadFlipEnv(
        episode_seconds=3.0,
        obs_noise=True,
        random_wind=True,
        random_initial_state=True,
        random_battery=True,
        curriculum_arena=False,
        arena_radius_start=4.0,
        arena_radius_end=4.0,
    )
    env.set_dr_level(1.0)
    env.set_dr_headroom(1.5)
    return env


print("=" * 78)
print("A. harvest a small dataset to build realistic frozen constants")
print("=" * 78)
venv_raw = DummyVecEnv([make_env for _ in range(3)])
venv_raw.seed(0)
obs = venv_raw.reset()

frames, targets = [], []
t0 = time.perf_counter()
for _ in range(500):
    obs, rew, dones, infos = venv_raw.step(np.random.uniform(-0.6, 0.6, size=(3, 4)).astype(np.float32))
    for i in range(3):
        frames.append(frame_from_env_obs(obs[i], ACTOR_TOTAL_DIM, ENCODER_AUX_DIM))
        targets.append(venv_raw.env_method("get_priv_targets", indices=[i])[0])
frames = np.asarray(frames, dtype=np.float64)
targets = np.asarray(targets, dtype=np.float64)
dt_harvest = time.perf_counter() - t0
check("frame dims", frames.shape[1] == ENCODER_IN_DIM, f"= {frames.shape}")
check("target dims", targets.shape[1] == PRIV_TARGET_DIM, f"= {targets.shape}")
check("frames finite", np.all(np.isfinite(frames)))
check("targets finite", np.all(np.isfinite(targets)))
print(f"        harvested {len(frames)} frames in {dt_harvest:.2f}s")

print()
print("=" * 78)
print("B. build and save a checkpoint (weights + frozen constants)")
print("=" * 78)
norm = NormStats.from_arrays(frames, targets, PRIV_TARGET_GROUPS)
check("frame means finite", np.all(np.isfinite(norm.frame_mean)))
check("frame stds positive", np.all(norm.frame_std > 0))
check("target stds positive", np.all(norm.target_std > 0))
print(f"        frame std range      : {norm.frame_std.min():.4f} .. {norm.frame_std.max():.4f}")
print(f"        target std range     : {norm.target_std.min():.6f} .. {norm.target_std.max():.4f}")
print(f"        v_batt_norm std      : {norm.frame_std[20]:.4f}")
print(f"        aero_wrench std (6d) : {np.round(norm.target_std[-6:], 6)}")

model = EncoderWithHead(n_targets=PRIV_TARGET_DIM, f_in=ENCODER_IN_DIM, width=48, z_dim=Z_DIM)
save_encoder_checkpoint(CKPT, model, norm, extra={"smoke_test": True})
check("checkpoint written", os.path.isfile(CKPT), CKPT)
check("receptive field covers the 100-step window", model.encoder.receptive_field >= 100,
      f"RF = {model.encoder.receptive_field}")
print(f"        encoder params = {sum(p.numel() for p in model.parameters()):,}")

print()
print("=" * 78)
print("C. LatentObsWrapper layout and behaviour")
print("=" * 78)
venv = DummyVecEnv([make_env for _ in range(3)])
wrapped = LatentObsWrapper(
    venv, encoder_path=CKPT, actor_total_dim=ACTOR_TOTAL_DIM, aux_dim=ENCODER_AUX_DIM,
    z_dim=Z_DIM, history_len=100, z_update_every=1,
)
expected_dim = ACTOR_TOTAL_DIM + Z_DIM + ENCODER_AUX_DIM + (TOTAL_OBS_DIM - ACTOR_TOTAL_DIM - ENCODER_AUX_DIM)
check("wrapped observation_space == 81", wrapped.observation_space.shape == (81,),
      f"= {wrapped.observation_space.shape}")
check("base env obs == 65", wrapped.priv_dim == 44, f"priv_dim = {wrapped.priv_dim}")

wobs = wrapped.reset()
check("reset obs dims", wobs.shape == (3, 81), f"= {wobs.shape}")
check("reset obs finite", np.all(np.isfinite(wobs)))

# At reset the window is the first frame repeated T times. The encoder has biases, so
# the expected latents are NOT zero - the correct assertion is CONSISTENCY: z must equal
# what the encoder produces for that exact repeated-frame prefill.
inner_envs = wrapped.venv.envs
for i in range(3):
    raw_i = inner_envs[i]._get_stacked_obs()
    frame_i = norm.standardize_frame(frame_from_env_obs(raw_i, ACTOR_TOTAL_DIM, ENCODER_AUX_DIM))
    x_i = torch.from_numpy(np.tile(frame_i[None, None, :], (1, 100, 1)).astype(np.float32))
    with torch.inference_mode():
        z_expected = wrapped.encoder(x_i).numpy()[0]
    check(
        f"env {i}: reset z equals the encoder's response to the repeated-frame prefill",
        np.allclose(wobs[i, ACTOR_TOTAL_DIM:ACTOR_TOTAL_DIM + Z_DIM], z_expected, atol=1e-5),
        f"|z|max={np.abs(z_expected).max():.4f}",
    )

# z must actually vary once the drone is moving
seen_z = []
for _ in range(60):
    wobs, rew, dones, infos = wrapped.step(np.random.uniform(-1, 1, size=(3, 4)).astype(np.float32))
    seen_z.append(wobs[:, ACTOR_TOTAL_DIM:ACTOR_TOTAL_DIM + Z_DIM].copy())
seen_z = np.asarray(seen_z)
check("obs stays finite over 60 steps", np.all(np.isfinite(wobs)))
check("z becomes non-trivial", np.abs(seen_z[-1]).max() > 1e-4, f"max|z| = {np.abs(seen_z[-1]).max():.5f}")
check("z is tanh-bounded to [-1,1]", np.abs(seen_z).max() <= 1.0 + 1e-6, f"max|z| = {np.abs(seen_z).max():.6f}")

# layout checks against the raw env
for i in range(3):
    inner = inner_envs[i]
    check(f"env {i}: inner env observation_space is 65", inner.observation_space.shape == (65,),
          f"= {inner.observation_space.shape}")

# The most important one: z must land at index 17..33 and the priv block at 37..81
priv_from_wrapper = wobs[0, 37:]
priv_from_env = np.concatenate([inner_envs[0]._compute_privileged_critic_obs()])
check("priv block passed through unchanged", np.allclose(priv_from_wrapper, priv_from_env, atol=1e-6))
aux_from_wrapper = wobs[0, 33:37]
check("aux block passed through unchanged",
      np.allclose(aux_from_wrapper, inner_envs[0]._aux_delayed, atol=1e-6),
      f"{np.round(aux_from_wrapper, 4)}")

print()
print("=" * 78)
print("D. auto-reset reseeds history instead of carrying the crashed trajectory")
print("=" * 78)


def expected_prefill_z(idx: int) -> np.ndarray:
    """What z must be if the window is the current env state's first frame repeated T times."""
    raw_i = inner_envs[idx]._get_stacked_obs()
    frame_i = norm.standardize_frame(frame_from_env_obs(raw_i, ACTOR_TOTAL_DIM, ENCODER_AUX_DIM))
    x_i = torch.from_numpy(np.tile(frame_i[None, None, :], (1, 100, 1)).astype(np.float32))
    with torch.inference_mode():
        return wrapped.encoder(x_i).numpy()[0]


# Zero action drives the drone into the ground quickly -> many resets.
resets = 0
reset_matched = None
reset_dev = None
for k in range(1200):
    wobs, rew, dones, infos = wrapped.step(np.zeros((3, 4), dtype=np.float32))
    if dones.any():
        resets += int(dones.sum())
        if reset_matched is None:
            idx = int(np.flatnonzero(dones)[0])
            z_got = wobs[idx, ACTOR_TOTAL_DIM:ACTOR_TOTAL_DIM + Z_DIM]
            z_exp = expected_prefill_z(idx)
            reset_dev = float(np.abs(z_got - z_exp).max())
            reset_matched = bool(np.allclose(z_got, z_exp, atol=1e-5))
    if resets >= 3:
        break
check("episodes terminated and auto-reset", resets >= 3, f"{resets} resets in {k + 1} steps")
check(
    "z at an auto-reset is a fresh prefill of the NEW episode (not carried over)",
    bool(reset_matched),
    f"max deviation from the recomputed prefill = {reset_dev}",
)
check("obs finite after many resets", np.all(np.isfinite(wobs)))

print()
print("=" * 78)
print("E. z_update_every holds z between refreshes")
print("=" * 78)
K = 5
venv2 = DummyVecEnv([make_env for _ in range(2)])
w2 = LatentObsWrapper(
    venv2, encoder_path=CKPT, actor_total_dim=ACTOR_TOTAL_DIM, aux_dim=ENCODER_AUX_DIM,
    z_dim=Z_DIM, history_len=100, z_update_every=K,
)
w2.reset()
zs = []
for k in range(3 * K + 2):
    obs2, _, _, _ = w2.step(np.random.uniform(-1, 1, size=(2, 4)).astype(np.float32))
    zs.append(obs2[:, ACTOR_TOTAL_DIM:ACTOR_TOTAL_DIM + Z_DIM].copy())
zs = np.asarray(zs)
changed = [k for k in range(len(zs) - 1) if not np.allclose(zs[k], zs[k + 1], atol=0)]
gaps = np.diff(changed)
check(
    f"z refreshes exactly every {K} steps",
    len(changed) >= 2 and np.all(gaps == K),
    f"refresh indices = {changed}, gaps = {list(gaps)}",
)
check("z is constant between refreshes", len(changed) <= len(zs) // K + 1)

print()
print("=" * 78)
print("F. measured in-situ overhead")
print("=" * 78)
N = 400
venv3 = DummyVecEnv([make_env for _ in range(10)])
venv3.seed(1)
venv3.reset()
a = np.random.uniform(-0.5, 0.5, size=(10, 4)).astype(np.float32)
t0 = time.perf_counter()
for _ in range(N):
    venv3.step(a)
base = (time.perf_counter() - t0) / N * 1e6

w3 = LatentObsWrapper(
    venv3, encoder_path=CKPT, actor_total_dim=ACTOR_TOTAL_DIM, aux_dim=ENCODER_AUX_DIM,
    z_dim=Z_DIM, history_len=100, z_update_every=1,
)
w3.reset()
t0 = time.perf_counter()
for _ in range(N):
    w3.step(a)
eager = (time.perf_counter() - t0) / N * 1e6

venv4 = DummyVecEnv([make_env for _ in range(10)])
w4 = LatentObsWrapper(
    venv4, encoder_path=CKPT, actor_total_dim=ACTOR_TOTAL_DIM, aux_dim=ENCODER_AUX_DIM,
    z_dim=Z_DIM, history_len=100, z_update_every=5,
)
w4.reset()
t0 = time.perf_counter()
for _ in range(N):
    w4.step(a)
lazy5 = (time.perf_counter() - t0) / N * 1e6

print(f"        raw 10-env step          : {base:8.1f} us")
print(f"        + encoder every step     : {eager:8.1f} us  ({100 * (eager - base) / base:+.1f}%)")
print(f"        + encoder every 5 steps  : {lazy5:8.1f} us  ({100 * (lazy5 - base) / base:+.1f}%)")
print("        (in-process DummyVecEnv, so this is an upper bound on the wall-clock")
print("         share; real training runs the vec step in 10 parallel processes)")

print()
print("=" * 78)
print("G. encoder cost sweep (batched over 10 envs, 1 CPU thread)")
print("=" * 78)
torch.set_num_threads(1)
xs = torch.randn(10, 100, ENCODER_IN_DIM)
SWEEP = [
    ("width=48  6 blocks  eager", 48, (1, 2, 4, 8, 16, 32), False),
    ("width=48  5 blocks  eager", 48, (1, 2, 4, 8, 16), False),
    ("width=32  6 blocks  eager", 32, (1, 2, 4, 8, 16, 32), False),
    ("width=32  5 blocks  eager", 32, (1, 2, 4, 8, 16), False),
    ("width=48  6 blocks  jit  ", 48, (1, 2, 4, 8, 16, 32), True),
    ("width=32  5 blocks  jit  ", 32, (1, 2, 4, 8, 16), True),
    ("width=24  5 blocks  jit  ", 24, (1, 2, 4, 8, 16), True),
]
print(f"        {'config':<26}{'params':>9}{'RF':>6}{'us/batch':>10}{'us/env-step':>13}")
for name, width, dil, use_jit in SWEEP:
    enc = HistoryEncoder(f_in=ENCODER_IN_DIM, width=width, z_dim=Z_DIM, dilations=dil).eval()
    nparams = sum(p.numel() for p in enc.parameters())
    rf = enc.receptive_field
    if use_jit:
        try:
            enc = torch.jit.script(enc).eval()
        except Exception as exc:  # noqa: BLE001
            print(f"        {name:<26}{nparams:>9,}{rf:>6}   jit failed: {type(exc).__name__}")
            continue
    with torch.inference_mode():
        for _ in range(5):
            enc(xs)
        t0 = time.perf_counter()
        reps = 40
        for _ in range(reps):
            enc(xs)
        dt = (time.perf_counter() - t0) / reps
    print(f"        {name:<26}{nparams:>9,}{rf:>6}{dt * 1e6:>10.0f}{dt * 1e6 / 10:>13.0f}")

print()
print("        reference: 10 SubprocVecEnv workers advance one vec-step in ~1880 us wall")
print("        (measured 5330 total steps/s during the 20M-step run, incl. optimizer time)")

if os.path.isfile(CKPT):
    os.remove(CKPT)

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all wrapper checks passed")
print("=" * 78)
