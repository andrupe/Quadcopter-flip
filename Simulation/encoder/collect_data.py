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

    # a real corpus (default 5M frames, ~45 min at the measured ~2k fps on ONE core;
    # 8 workers cut that to ~15 min on this machine)
    .venv/bin/python Simulation/encoder/collect_data.py --frames 5000000 --out logs/encoder_data

    # how much does the worker count actually buy? worth measuring once per machine
    .venv/bin/python Simulation/encoder/collect_data.py --frames 200000 --out /tmp/w1 --workers 1
    .venv/bin/python Simulation/encoder/collect_data.py --frames 200000 --out /tmp/w8 --workers 8

CORPUS SIZE (5M frames, up from 1.2M)
The 1.2M-frame corpus held 3,246 episodes at a mean length of 370 steps. Two things made
that too small to train against any more:

  * it was collected at the OLD 8 s horizon (max episode length 800 steps), so it contains
    no example of the multi-command CHAINS or the vertical figure-eight added on
    2026-09-15 - 4% of the draws and ~7% of the episode TIME of the current mixture are
    simply absent;
  * the encoder is now also asked to predict the estimator's own drift, which is a
    genuinely heteroscedastic target that needs many blackout/re-acquisition events to fit.

5M frames is ~13k episodes of the current mixture, i.e. ~4x the episode count and a
regime where every manoeuvre family and every exciter gets thousands of episodes. Raising
it further costs only collection wall-time and RAM in the trainer (~2.3 GB for frames +
targets), not model size - the deployed network is unchanged.
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

from quad_flip_env import ACTOR_FRAME_MODE, QuadFlipEnv  # noqa: E402
from trajectories import GRAVITY, dcm_from_thrust_dir_and_yaw  # noqa: E402

FRAME_MODE: str = ACTOR_FRAME_MODE

EXCITER_WEIGHTS: Dict[str, float] = {
    "none": 0.25,
    "bandlimited": 0.25,
    "chirp": 0.15,
    "steps": 0.15,
    "acrobatic": 0.20,
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
        if self.kind == "acrobatic":
            self._step_timer -= dt
            if self._step_timer <= 0.0:
                self._step_timer = float(self.rng.uniform(0.12, 0.35))
                # Command sharp rate burst in roll or pitch to excite high-rate rotational dynamics
                axis = int(self.rng.integers(0, 2))
                pulse = np.zeros(3)
                pulse[axis] = float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.40, 0.85))
                self._step = pulse
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
        # Suppress aggressive roll/pitch disturbance while on the ground before liftoff
        if env.spawn_pos[2] < 0.08 and not getattr(env, "_has_lifted_off", False):
            exc = np.zeros(3)

        max_thr = float(q.params["maxThr"])
        a0 = float(np.clip(2.0 * thrust / max_thr - 1.0 + 0.10 * exc[0], -1.0, 1.0))
        return np.array([
            a0,
            np.clip((omega_cmd[0] + exc[0]) / env.max_rate_xy, -1.0, 1.0),
            np.clip((omega_cmd[1] + exc[1]) / env.max_rate_pitch, -1.0, 1.0),
            np.clip((omega_cmd[2] + exc[2]) / env.max_rate_z, -1.0, 1.0),
        ], dtype=np.float32)


def _collect_worker(job: Dict) -> Dict:
    """
    One collection process: build a PRIVATE env, fly `frames_budget` frames, write shards.
    """
    import mujoco
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

    out_dir = job["out_dir"]
    worker = int(job["worker"])
    rng = np.random.default_rng(int(job["seed"]))

    env = QuadFlipEnv(episode_seconds=15.0)   # matches the training horizon
    env.set_dr_headroom(float(job["dr_headroom"]))
    env.set_dr_level(float(job["dr_level"]))
    env.set_envelope_scale(5.0)  # unconstrained exploration envelope during pretraining data collection

    kinds = list(EXCITER_WEIGHTS)
    probs = np.array([EXCITER_WEIGHTS[k] for k in kinds], dtype=np.float64)
    probs /= probs.sum()

    budget = int(job["frames_budget"])
    shard_frames = int(job["shard_frames"])
    tag = f"w{worker:02d}"

    buf_frames: List[np.ndarray] = []
    buf_targets: List[np.ndarray] = []
    ep_lens: List[int] = []
    n_frames = 0
    n_eps = 0
    n_shard = 0
    counts: Dict[str, int] = {}
    t0 = time.time()

    while n_frames < budget:
        kind = str(rng.choice(kinds, p=probs))
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        # In 20% of episodes, inject asymmetric motor efficiency degradation
        if rng.uniform() < 0.20:
            bad_rotor = int(rng.integers(0, 4))
            efficiencies = np.ones(4, dtype=np.float64)
            efficiencies[bad_rotor] = float(rng.uniform(0.75, 0.90))
            env.quad.apply_hardware_distortions(motor_efficiencies=efficiencies)

        # In 25% of episodes, inject random recovery kicks so encoder sees toss/recovery regimes
        if rng.uniform() < 0.25:
            v_kick = rng.uniform(-1.0, 1.0, size=3)
            w_kick = rng.uniform(-2.5, 2.5, size=3)
            # If on ground, prevent kicking down into the floor
            if env.spawn_pos[2] < 0.08 and v_kick[2] < 0.0:
                v_kick[2] = float(rng.uniform(0.1, 1.0))
            env.quad.data.qvel[0:3] = v_kick
            env.quad.data.qvel[3:6] = w_kick
            mujoco.mj_forward(env.quad.model, env.quad.data)
            env.quad._update_state_properties()

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
            _write_shard(out_dir, n_shard, buf_frames, buf_targets, ep_lens, tag=tag)
            n_shard += 1
            buf_frames, buf_targets, ep_lens = [], [], []

        if bool(job["verbose"]) and n_eps % 25 == 0:
            el = time.time() - t0
            print(f"  [{tag}] {n_frames:>9,} frames | {n_eps:>5} episodes | "
                  f"{n_frames / max(1e-6, el):>7.0f} fps | {dict(sorted(counts.items()))}",
                  flush=True)

    if buf_frames:
        _write_shard(out_dir, n_shard, buf_frames, buf_targets, ep_lens, tag=tag)
        n_shard += 1

    return {
        "worker": worker,
        "frames": n_frames,
        "episodes": n_eps,
        "shards": n_shard,
        "counts": counts,
        "elapsed": time.time() - t0,
    }


def collect(
    total_frames: int,
    out_dir: str,
    shard_frames: int = 80_000,
    dr_headroom: float = 1.3,
    dr_level: float = 1.0,
    seed: int = 0,
    workers: int = 1,
    verbose: bool = True,
) -> None:
    """
    Collect `total_frames` frames into `out_dir`, over `workers` processes.

    WHY PARALLEL. Collection is embarrassingly parallel by construction - every episode
    draws its own plant, own lighthouse geometry and own manoeuvre, and shards are
    independent files - but this driver used to fly every episode in ONE process, so the
    default 5M-frame corpus ran on a single core for ~45 min while 9 others idled. The
    measured scaling ceiling on this machine is ~3.7x for pure-Python workers, so `workers`
    maps to roughly a 3x wall-clock cut rather than an 8x one.

    WARNING: the reader (train_encoder.load_episodes) globs EVERY `shard_*.npz` in
    `out_dir`, so a re-collection into a directory that already holds shards mixes the two
    corpora. Collect into a fresh directory (the default is per-run and the smoke run gets
    its own suffix).
    """
    os.makedirs(out_dir, exist_ok=True)
    workers = max(1, int(workers))

    # Split the budget evenly, with the remainder on the last worker. Every worker writes
    # its own tag-prefixed shards, so no index coordination is needed between them.
    base = int(total_frames) // workers
    remainder = int(total_frames) - base * workers
    jobs = []
    for w in range(workers):
        n = base + (remainder if w == workers - 1 else 0)
        if n <= 0:
            continue
        jobs.append({
            "worker": w,
            "out_dir": out_dir,
            "frames_budget": n,
            "shard_frames": int(shard_frames),
            "dr_headroom": float(dr_headroom),
            "dr_level": float(dr_level),
            # Distinct, well-separated seed streams: one shared rng would give several
            # workers the same plants.
            "seed": int(seed) + 1_000_003 * w,
            "verbose": bool(verbose) and workers == 1,
        })

    print(f"Collecting {total_frames:,} frames over {len(jobs)} process(es) "
          f"({jobs[0]['frames_budget']:,} frames each)", flush=True)
    t0 = time.time()

    if len(jobs) == 1:
        results = [_collect_worker(jobs[0])]
    else:
        import multiprocessing as mp

        # spawn, not fork: each child must build its OWN MuJoCo model, and fork inherits
        # the parent's already-initialised MuJoCo/BLAS state.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(jobs)) as pool:
            results = pool.map(_collect_worker, jobs)

    n_frames = sum(int(r["frames"]) for r in results)
    n_eps = sum(int(r["episodes"]) for r in results)
    n_shard = sum(int(r["shards"]) for r in results)
    counts: Dict[str, int] = {}
    for r in results:
        for k, v in r["counts"].items():
            counts[k] = counts.get(k, 0) + int(v)

    print(f"\nWrote {n_shard} shards to {out_dir}")
    print(f"  frames   : {n_frames:,}")
    print(f"  episodes : {n_eps:,}")
    print(f"  mean len : {n_frames / max(1, n_eps):.0f} steps")
    print(f"  mix      : {dict(sorted(counts.items()))}")
    print(f"  elapsed  : {time.time() - t0:.0f}s "
          f"({n_frames / max(1e-6, time.time() - t0):,.0f} fps aggregate)")
    print(f"\nNext:  .venv/bin/python Simulation/encoder/train_encoder.py --data {out_dir}")


def _write_shard(out_dir: str, idx: int, frames: List[np.ndarray],
                 targets: List[np.ndarray], ep_lens: List[int], tag: str = "") -> None:
    """
    Concatenate episodes into one shard and record the boundaries.

    Episode boundaries are preserved because the encoder is a GRU: its hidden state is an
    unbounded summary of everything it has seen, so supervision MUST be episode-sequential.
    A randomly sampled window would hand the model a state it would never actually be in.

    `tag` namespaces the file per worker (shard_w03_0017.npz) so parallel workers cannot
    collide and no index needs to be shared. train_encoder.load_episodes globs
    `shard_*.npz`, so the tagged names are picked up unchanged.
    """
    name = f"shard_{tag}_{idx:04d}.npz" if tag else f"shard_{idx:04d}.npz"
    path = os.path.join(out_dir, name)
    np.savez_compressed(
        path,
        frames=np.concatenate(frames, axis=0),
        targets=np.concatenate(targets, axis=0),
        ep_lens=np.asarray(ep_lens, dtype=np.int32),
        # WHICH ACTOR-FRAME CONVENTION produced these frames (quad_flip_env.ACTOR_FRAME_MODE).
        # Consumed by train_encoder.load_episodes, which REFUSES a corpus collected under a
        # different one: the x,y content of o_t changed, so the numbers are not
        # interchangeable, and silently mixing them would train the encoder on a signal the
        # deployed pipeline never produces.
        frame_mode=np.array(FRAME_MODE),
    )
    print(f"  -> {name}  "
          f"({sum(ep_lens):,} frames, {len(ep_lens)} episodes)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect encoder pretraining data.")
    ap.add_argument("--frames", type=int, default=5_000_000)
    ap.add_argument("--out", type=str, default=os.path.join(_PROJECT_ROOT, "logs", "encoder_data"))
    ap.add_argument("--shard-frames", type=int, default=80_000)
    ap.add_argument("--dr-headroom", type=float, default=1.3)
    ap.add_argument("--dr-level", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help="collection processes (default: min(8, ncpu-2); the measured "
                         "pure-Python scaling ceiling on this machine is ~3.7x, so this "
                         "maps to ~3x wall-clock, not Nx). 1 = the old serial behaviour.")
    ap.add_argument("--smoke", action="store_true", help="tiny run, for validating the plumbing")
    args = ap.parse_args()

    frames = 20_000 if args.smoke else args.frames
    out = args.out + ("_smoke" if args.smoke else "")
    if args.workers is None:
        workers = 1 if args.smoke else min(8, max(1, (os.cpu_count() or 4) - 2))
    else:
        workers = max(1, int(args.workers))
    print(f"Collecting {frames:,} frames -> {out}")
    collect(
        total_frames=frames,
        out_dir=out,
        shard_frames=min(args.shard_frames, frames),
        dr_headroom=args.dr_headroom,
        dr_level=args.dr_level,
        seed=args.seed,
        workers=workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
