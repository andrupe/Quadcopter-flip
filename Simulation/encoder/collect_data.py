"""
Pretraining data driver for the frozen history encoder.

Writes (encoder frame, privileged physics target) pairs to .npz shards, which
train_encoder.py consumes.

WHY THE CONTROLLER MIX IS BUILT AROUND TRACKING
The encoder exists to identify unmodelled plant parameters - mass, thrust scale, motor
time constants, CoM offset, battery sag - from the recent sensor stream. That only works
if the data contains the excitation that makes those parameters visible, which means the
closed loop has to be flown in a way that varies the forces and moments the plant sees.

The obvious controller is the one that will actually be deployed: a cascaded geometric
tracker following the reference. Everything else here is that tracker with extra
excitation layered on it, so the corpus stays inside the operating envelope the policy
will occupy. Purely random actions are deliberately NOT used as the main source: a
previous version of this driver did that, and open-loop excitation tumbled the vehicle in
under a second, producing data that was nearly all crashing.

Exciter variants, all applied on top of the tracker:
    none          nominal closed loop
    bandlimited   low-pass filtered uniform noise on the rate commands
    chirp         slow frequency sweep on roll/pitch, to sweep the plant's response
    steps         random step offsets, to provoke aggressive transients

On top of that, controller gains are randomised per episode (soft through aggressive) and
the domain-randomisation level is drawn across the full widened envelope, so the encoder
sees the whole range of plants PPO will later encounter.

Run:
    # quick check that it works end to end
    .venv/bin/python Simulation/encoder/collect_data.py --smoke

    # a real corpus (~10 min)
    .venv/bin/python Simulation/encoder/collect_data.py --frames 1200000 --out logs/encoder_data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_SIM_DIR)
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quad_flip_env import QuadFlipEnv  # noqa: E402
from trajectories import GRAVITY, dcm_from_thrust_dir_and_yaw  # noqa: E402

EXCITER_WEIGHTS: Dict[str, float] = {
    "none": 0.35,
    "bandlimited": 0.30,
    "chirp": 0.15,
    "steps": 0.20,
}


def vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


class TrackerExciter:
    """
    Cascaded geometric tracking controller plus optional excitation.

    The tracking part is deliberately the textbook controller - position error to desired
    acceleration to desired attitude and collective thrust to a body-rate command - rather
    than anything tuned for this reward, so the corpus reflects a plausible flight
    controller instead of one overfitted to the encoder.
    """

    def __init__(self, rng: np.random.Generator, exciter: str = "none"):
        self.rng = rng
        self.kind = exciter
        # Gain ranges chosen so the closed loop spans sluggish to aggressive; the encoder
        # needs to see both, since the disturbance the plant puts on the loop depends on
        # how hard the loop is driving.
        self.kp = float(rng.uniform(4.0, 10.0))
        self.kd = float(rng.uniform(2.5, 6.5))
        self.kr = float(rng.uniform(3.5, 9.0))
        self.kw = float(rng.uniform(0.8, 2.5))
        self._t = 0.0
        self._noise = np.zeros(3)
        self._step = np.zeros(3)
        self._step_timer = 0.0
        self.f0 = float(rng.uniform(0.1, 0.4))
        self.f1 = float(rng.uniform(1.5, 4.0))
        self.amp = float(rng.uniform(0.08, 0.30))

    def reset(self) -> None:
        self._t = 0.0
        self._noise = np.zeros(3)
        self._step = np.zeros(3)
        self._step_timer = 0.0

    def _excitation(self, dt: float) -> np.ndarray:
        if self.kind == "bandlimited":
            # First-order low-pass on white noise: keeps the command inside the rate
            # loop's bandwidth instead of demanding something the actuators cannot follow.
            alpha = 0.12
            self._noise = (1.0 - alpha) * self._noise + alpha * self.rng.normal(0.0, 1.0, size=3)
            return 0.35 * self._noise
        if self.kind == "chirp":
            self._t += dt
            # Linear sweep across the episode length.
            dur = max(1.0, self.period)
            frac = float(np.clip(self._t / dur, 0.0, 1.0))
            f = self.f0 + (self.f1 - self.f0) * frac
            phase = 2.0 * np.pi * (self.f0 * self._t + 0.5 * (self.f1 - self.f0) * self._t * frac)
            return self.amp * np.array([np.sin(phase), np.sin(phase + 1.0), 0.0])
        if self.kind == "steps":
            self._step_timer -= dt
            if self._step_timer <= 0.0:
                self._step_timer = float(self.rng.uniform(0.10, 0.35))
                self._step = self.rng.uniform(-0.35, 0.35, size=3)
            return self._step
        return np.zeros(3)

    def __call__(self, env: QuadFlipEnv, dt: float) -> np.ndarray:
        ref = env.ref
        q = env.quad
        self.period = getattr(env.traj, "duration", 2.0)

        a_des = ref.a + self.kp * (ref.p - q.pos) + self.kd * (ref.v - q.vel)
        up = a_des + np.array([0.0, 0.0, GRAVITY])
        z_b = dcm_from_thrust_dir_and_yaw(up, 0.0)[:, 2]
        thrust = float(q.base_mass * float(np.dot(up, z_b)))

        R_err = ref.R.T @ q.dcm
        e_R = 0.5 * vee(R_err - R_err.T)
        omega_cmd = -self.kr * e_R - self.kw * (q.omega - ref.omega) + ref.omega

        exc = self._excitation(dt)
        max_thr = float(q.params["maxThr"])
        a0 = float(np.clip(2.0 * thrust / max_thr - 1.0 + 0.10 * exc[0], -1.0, 1.0))
        return np.array([
            a0,
            np.clip((omega_cmd[0] + exc[0]) / env.max_rate_xy, -1.0, 1.0),
            np.clip((omega_cmd[1] + exc[1]) / env.max_rate_pitch, -1.0, 1.0),
            np.clip((omega_cmd[2] + exc[2]) / env.max_rate_z, -1.0, 1.0),
        ], dtype=np.float32)


def collect(
    total_frames: int,
    out_dir: str,
    shard_frames: int = 80_000,
    dr_headroom: float = 1.3,
    dr_level: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    env = QuadFlipEnv(episode_seconds=8.0)
    # Widen the randomisation envelope beyond what PPO will visit, so the frozen encoder
    # has already seen the extremes before the ADR curriculum gets there.
    env.set_dr_headroom(dr_headroom)
    env.set_dr_level(dr_level)

    kinds = list(EXCITER_WEIGHTS)
    probs = np.array([EXCITER_WEIGHTS[k] for k in kinds], dtype=np.float64)
    probs /= probs.sum()

    buf_frames: List[np.ndarray] = []
    buf_targets: List[np.ndarray] = []
    ep_lens: List[int] = []
    n_frames = 0
    n_eps = 0
    n_shard = 0
    counts: Dict[str, int] = {}
    t0 = time.time()

    while n_frames < total_frames:
        kind = str(rng.choice(kinds, p=probs))
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ctrl = TrackerExciter(rng, kind)
        ctrl.reset()
        counts[kind] = counts.get(kind, 0) + 1

        ep_frames: List[np.ndarray] = []
        ep_targets: List[np.ndarray] = []
        while True:
            action = ctrl(env, env.dt)
            _, _, term, trunc, info = env.step(action)
            ep_frames.append(info["encoder_frame"])
            ep_targets.append(env.get_priv_targets())
            if term or trunc:
                break

        ep_frames_a = np.asarray(ep_frames, dtype=np.float32)
        ep_targets_a = np.asarray(ep_targets, dtype=np.float32)
        buf_frames.append(ep_frames_a)
        buf_targets.append(ep_targets_a)
        ep_lens.append(int(ep_frames_a.shape[0]))
        n_frames += int(ep_frames_a.shape[0])
        n_eps += 1

        if sum(a.shape[0] for a in buf_frames) >= shard_frames:
            _write_shard(out_dir, n_shard, buf_frames, buf_targets, ep_lens)
            n_shard += 1
            buf_frames, buf_targets, ep_lens = [], [], []

        if verbose and n_eps % 25 == 0:
            el = time.time() - t0
            print(f"  {n_frames:>9,} frames | {n_eps:>5} episodes | "
                  f"{n_frames / max(1e-6, el):>7.0f} fps | {dict(sorted(counts.items()))}",
                  flush=True)

    if buf_frames:
        _write_shard(out_dir, n_shard, buf_frames, buf_targets, ep_lens)
        n_shard += 1

    print(f"\nWrote {n_shard} shards to {out_dir}")
    print(f"  frames   : {n_frames:,}")
    print(f"  episodes : {n_eps:,}")
    print(f"  mean len : {n_frames / max(1, n_eps):.0f} steps")
    print(f"  mix      : {dict(sorted(counts.items()))}")
    print(f"  elapsed  : {time.time() - t0:.0f}s")
    print(f"\nNext:  .venv/bin/python Simulation/encoder/train_encoder.py --data {out_dir}")


def _write_shard(out_dir: str, idx: int, frames: List[np.ndarray],
                 targets: List[np.ndarray], ep_lens: List[int]) -> None:
    """
    Concatenate episodes into one shard and record the boundaries.

    Episode boundaries are preserved because the encoder is a GRU: its hidden state is an
    unbounded summary of everything it has seen, so supervision MUST be episode-sequential.
    A randomly sampled window would hand the model a state it would never actually be in.
    """
    path = os.path.join(out_dir, f"shard_{idx:04d}.npz")
    np.savez_compressed(
        path,
        frames=np.concatenate(frames, axis=0),
        targets=np.concatenate(targets, axis=0),
        ep_lens=np.asarray(ep_lens, dtype=np.int32),
    )
    print(f"  -> {os.path.basename(path)}  "
          f"({sum(ep_lens):,} frames, {len(ep_lens)} episodes)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect encoder pretraining data.")
    ap.add_argument("--frames", type=int, default=1_200_000)
    ap.add_argument("--out", type=str, default=os.path.join(_PROJECT_ROOT, "logs", "encoder_data"))
    ap.add_argument("--shard-frames", type=int, default=80_000)
    ap.add_argument("--dr-headroom", type=float, default=1.3)
    ap.add_argument("--dr-level", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run, for validating the plumbing")
    args = ap.parse_args()

    frames = 20_000 if args.smoke else args.frames
    out = args.out + ("_smoke" if args.smoke else "")
    print(f"Collecting {frames:,} frames -> {out}")
    collect(
        total_frames=frames,
        out_dir=out,
        shard_frames=min(args.shard_frames, frames),
        dr_headroom=args.dr_headroom,
        dr_level=args.dr_level,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
