"""
Phase 0 observability probes for the frozen-history-encoder design.

Answers three questions before any implementation is committed to:

  1. Does the simulated wind actually perturb the plant, or is it decorative?
     (scene.xml sets density/viscosity but no geom sets fluidshape, so whether
     MuJoCo's fluid model is doing any work is unverified.)

  2. Is the MJCF `accelerometer` sensor readable, and does it return specific
     force (gravity + motion)? It is never read anywhere in the Python code today.

  3. What does the proposed causal TCN cost per batched vec-step? The measured
     PPO baseline is ~5330 steps/s over 10 workers = 1.88 ms per worker-step, so
     the batched encoder must stay well under ~0.4 ms per vec-step.

Run:  .venv/bin/python scratch/probe_phase0.py
"""

from __future__ import annotations

import os
import random
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from quad_flip_env import QuadFlipEnv  # noqa: E402

SEED = 123
STEPS = 600
ARENA = 6.0


def make_env() -> QuadFlipEnv:
    """Fully nominal env: no sensor noise, no DR, fixed spawn, full wind plumbing."""
    return QuadFlipEnv(
        episode_seconds=10.0,
        obs_noise=False,
        random_wind=True,
        random_initial_state=False,
        random_battery=False,
        curriculum_arena=False,
        arena_radius_start=ARENA,
        arena_radius_end=ARENA,
    )


def action_at(k: int) -> np.ndarray:
    """Deterministic excitation: oscillating body-rate commands + mild throttle."""
    return np.array(
        [
            0.30 + 0.10 * np.sin(0.040 * k),
            0.55 * np.sin(0.060 * k),   # roll rate  +/- 3.3 rad/s
            0.25 * np.sin(0.090 * k),   # pitch rate +/- 5.0 rad/s
            0.10 * np.sin(0.030 * k),   # yaw rate   +/- 0.4 rad/s
        ],
        dtype=np.float32,
    )


def rollout(vel_w: float, label: str) -> dict:
    """Run one deterministic episode with the wind magnitude pinned to `vel_w`."""
    # Seed the *global* random module too: Wind.reseed() draws from it and would
    # otherwise hand the two runs different Perlin seeds.
    random.seed(0)
    env = make_env()
    env.set_dr_level(1.0)
    obs, info = env.reset(seed=SEED)

    # reset() sets velW_max from dr_level; pin magnitude explicitly and keep the
    # baseline breeze consistent so the ONLY difference between runs is vel_w.
    env.wind.velW_max = float(vel_w)
    env.wind.velW_med = float(vel_w) * 0.3

    vel, pos, qfrc_p, qfrc_a, wind_v, accel = [], [], [], [], [], []
    for k in range(STEPS):
        velW, qW1, qW2 = env.wind.randomWind(env.t)
        wind_v.append(
            [
                velW * np.cos(qW1) * np.cos(qW2),
                velW * np.sin(qW1) * np.cos(qW2),
                velW * np.sin(qW2),
            ]
        )
        obs, rew, term, trunc, info = env.step(action_at(k))
        vel.append(env.quad.vel.copy())
        pos.append(env.quad.pos.copy())
        qfrc_p.append(env.quad.data.qfrc_passive[:6].copy())
        qfrc_a.append(env.quad.data.qfrc_actuator[:6].copy())
        accel.append(np.asarray(env.quad.data.qfrc_applied[:6], dtype=float).copy())
        if term or trunc:
            break

    out = {
        "label": label,
        "n": len(vel),
        "vel": np.asarray(vel),
        "pos": np.asarray(pos),
        "qfrc_passive": np.asarray(qfrc_p),
        "qfrc_actuator": np.asarray(qfrc_a),
        "wind": np.asarray(wind_v),
    }
    env.close()
    return out


# ---------------------------------------------------------------------------
# Probe 1 - accelerometer availability
# ---------------------------------------------------------------------------
def probe_accelerometer() -> None:
    print("=" * 78)
    print("PROBE 1 - MJCF sensors / accelerometer")
    print("=" * 78)

    env = make_env()
    m = env.quad.model
    print(f"sensor count: {m.nsensor}")
    for i in range(m.nsensor):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
        adr = int(m.sensor_adr[i])
        dim = int(m.sensor_dim[i])
        print(f"  [{i}] {name:<18} adr={adr:<3} dim={dim}")

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")
    print(f"\naccelerometer sensor id = {sid}")

    if sid >= 0:
        adr = int(m.sensor_adr[sid])
        obs, _ = env.reset(seed=SEED)
        sf = np.asarray(env.quad.data.sensordata[adr : adr + 3], dtype=float)
        print(f"sensordata @ reset (near-static) : {np.round(sf, 4)}")
        env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        sf = np.asarray(env.quad.data.sensordata[adr : adr + 3], dtype=float)
        print(f"norm                             : {np.linalg.norm(sf):.4f} m/s^2")
        print("  -> |a| ~ 9.81 and pointing along body +z means specific force (OK)")

    print()
    print("fluid model settings:")
    print(f"  opt.density  = {m.opt.density}")
    print(f"  opt.viscosity= {m.opt.viscosity}")
    print(f"  opt.wind     = {np.round(np.asarray(m.opt.wind), 4)}")
    if hasattr(m, "geom_fluid"):
        gf = np.asarray(m.geom_fluid)
        print(f"  geom_fluid   = shape {gf.shape}, nonzero rows {int((np.abs(gf).sum(axis=1) > 0).sum())}")
    else:
        print("  geom_fluid   = not exposed")

    env.close()


# ---------------------------------------------------------------------------
# Probe 2b - isolate the wind force at a known state
# ---------------------------------------------------------------------------
HOVER_TRIM_A0 = -0.0844  # 2*m*g/maxThr - 1 for m=0.028, g=9.81, maxThr=0.60


def wind_force_at_rest(vel_w: float, n_steps: int = 50):
    """
    Start from an identical, exactly-known state (rest, level, 1.2 m) under a
    hover-throttle command, and record the passive (fluid) wrench. Because both
    runs share a bit-identical initial state, the difference at step k is purely
    the wind contribution. This gives the wind force directly, without the
    chaotic amplification of the open-loop rollout.
    """
    random.seed(0)
    env = make_env()
    env.set_dr_level(0.0)
    env.reset(seed=SEED)
    env.wind.velW_max = float(vel_w)
    env.wind.velW_med = float(vel_w) * 0.3

    q = env.quad
    q.data.qpos[0:3] = [0.0, 0.0, 1.2]
    q.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    q.data.qvel[:] = 0.0
    mujoco.mj_forward(q.model, q.data)
    q._update_state_properties()

    act = np.array([HOVER_TRIM_A0, 0.0, 0.0, 0.0], dtype=np.float32)
    passive, vel = [], []
    for _ in range(n_steps):
        env.step(act)
        passive.append(q.data.qfrc_passive[:6].copy())
        vel.append(q.vel.copy())
    env.close()
    return np.asarray(passive), np.asarray(vel)


def probe_wind_force() -> None:
    print("=" * 78)
    print("PROBE 2b - wind force at a known state (rest, level, hover throttle)")
    print("=" * 78)

    runs = {w: wind_force_at_rest(w) for w in (0.0, 0.5, 1.0)}
    base_p, base_v = runs[0.0]

    print(f"{'wind':>6}{'max|F_wind|':>14}{'equiv accel':>14}{'max|dv| over 0.5s':>20}")
    for w, (p, v) in runs.items():
        df = np.abs(p[:, :3] - base_p[:, :3]).max()
        dv = np.abs(v - base_v).max()
        print(f"{w:>6.1f}{df:>14.6f}{df / 0.028:>14.4f}{dv:>20.4f}")

    print()
    print("  hover thrust = 0.2747 N; body mass = 0.028 kg")
    df1 = np.abs(runs[1.0][0][:, :3] - base_p[:, :3]).max()
    print(f"  wind = 1.0 m/s produces {df1:.5f} N => {df1 / 0.028:.4f} m/s^2")
    print(f"  that is {100 * df1 / 0.2747:.2f}% of hover thrust, over 0.5 s")
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Probe 2 - does wind actually perturb the plant?
# ---------------------------------------------------------------------------
def probe_wind() -> None:
    print("=" * 78)
    print("PROBE 2 - wind efficacy (identical seed, only wind magnitude differs)")
    print("=" * 78)

    a = rollout(0.0, "wind=0.0 (control)")
    a2 = rollout(0.0, "wind=0.0 (repeat)")
    b = rollout(1.0, "wind=1.0 (full ADR)")

    print(f"{'run':<22}{'steps':>6}{'max|v|':>10}{'max|qp_pass|':>14}{'max|qp_act|':>13}")
    for r in (a, a2, b):
        print(
            f"{r['label']:<22}{r['n']:>6}"
            f"{np.abs(r['vel']).max():>10.4f}"
            f"{np.abs(r['qfrc_passive']).max():>14.6f}"
            f"{np.abs(r['qfrc_actuator']).max():>13.4f}"
        )

    n = min(a["n"], a2["n"], b["n"])
    d_control = np.abs(a["vel"][:n] - a2["vel"][:n]).max()
    d_wind = np.abs(a["vel"][:n] - b["vel"][:n]).max()

    # Relative air velocity actually felt by the airframe - the quantity an
    # encoder would have to infer. Compare its size to the ground velocity.
    rel_air = np.abs(b["wind"][:n] - b["vel"][:n]).max() if n else 0.0

    print()
    print(f"determinism check  max|dv| wind=0 vs wind=0 : {d_control:.6f} m/s")
    print(f"wind effect        max|dv| wind=0 vs wind=1 : {d_wind:.6f} m/s")
    print(f"max relative air speed seen                 : {rel_air:.4f} m/s")
    print(f"max |qfrc_passive[0:3]| at wind=1.0         : "
          f"{np.abs(b['qfrc_passive'][:n, :3]).max():.6f} N "
          f"(hover thrust is {0.028 * 9.81:.4f} N)")
    print(f"max |qfrc_passive[3:6]| at wind=1.0         : "
          f"{np.abs(b['qfrc_passive'][:n, 3:6]).max():.6f} N.m")

    verdict = (
        "WIND IS ACTIVE"
        if (d_control < 1e-9 and d_wind > 10 * max(d_control, 1e-9) and d_wind > 1e-3)
        else "WIND LOOKS INERT / UNOBSERVABLE"
    )
    print(f"\nverdict: {verdict}")
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Probe 3 - encoder cost
# ---------------------------------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, c: int, d: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, dilation=d)
        self.conv2 = nn.Conv1d(c, c, 3, dilation=d)
        self.n1, self.n2 = nn.GroupNorm(1, c), nn.GroupNorm(1, c)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.n1(self.conv1(F.pad(x, (2 * self.conv1.dilation[0], 0)))))
        h = self.act(self.n2(self.conv2(F.pad(h, (2 * self.conv2.dilation[0], 0)))))
        return x + h


class HistoryEncoder(nn.Module):
    def __init__(self, f_in: int, width: int = 48, z_dim: int = 16):
        super().__init__()
        self.stem = nn.Conv1d(f_in, width, 1)
        self.blocks = nn.ModuleList(TCNBlock(width, d) for d in [1, 2, 4, 8, 16, 32])
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, z_dim), nn.Tanh())

    def forward(self, x):
        h = self.stem(x.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, :, -1])


def probe_encoder_cost() -> None:
    print("=" * 78)
    print("PROBE 3 - causal TCN cost (batched over 10 vec envs)")
    print("=" * 78)

    torch.set_num_threads(1)
    for f_in, T, n_envs in [(21, 100, 10), (17, 100, 10), (21, 100, 1)]:
        enc = HistoryEncoder(f_in, 48, 16).eval()
        n_params = sum(p.numel() for p in enc.parameters())
        x = torch.randn(n_envs, T, f_in)
        with torch.inference_mode():
            for _ in range(5):
                enc(x)
            t0 = time.perf_counter()
            reps = 50
            for _ in range(reps):
                enc(x)
            dt = (time.perf_counter() - t0) / reps
        per_env_us = dt / n_envs * 1e6
        print(
            f"  f_in={f_in:<3} T={T:<4} n_envs={n_envs:<3} params={n_params:>7,}  "
            f"{dt * 1e3:>7.3f} ms/batch  =  {per_env_us:>7.1f} us/env-step"
        )

    print()
    print("  baseline PPO throughput: 5330 steps/s over 10 workers = 1876 us/worker-step")
    print("  budget: keep overhead under ~20% -> under ~375 us per worker-step")
    print("=" * 78)


if __name__ == "__main__":
    probe_accelerometer()
    probe_wind_force()
    probe_wind()
    probe_encoder_cost()
