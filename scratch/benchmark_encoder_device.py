"""
Is the encoder training step faster on the Apple-Silicon GPU (MPS) than on the CPU?

Motivation: the model is TINY (GRU f_in=33, hidden=48, z=16, ~15.1k params) and the
recurrent pass is inherently sequential over T~366 batch steps. That is the regime where a
discrete GPU usually LOSES to a good CPU, because each timestep is a small GEMM whose
launch overhead dominates. This script measures it instead of guessing.

It times a full pass over a fixed subset of real episodes - identical batches on both
devices - using the SAME model, loss and batching code as train_encoder.py (imported, not
reimplemented), so the numbers correspond to what a real run does.

Batch size is swept as well: a GPU that loses at batch 64 can still win at 256, and that
would be actionable.

Run:  .venv/bin/python scratch/benchmark_encoder_device.py
"""

from __future__ import annotations

import os
import sys
import time
from glob import glob
from typing import List, Tuple

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the real training script so the model/loss/batching are byte-for-byte the ones
# a full run uses. Importing it only sets sys.path; nothing executes at import time.
import encoder.train_encoder as te  # noqa: E402
from encoder.observation_spec import ENCODER_IN_DIM, NormStats  # noqa: E402

DATA_DIR = os.path.join(_PROJECT_ROOT, "logs", "encoder_data")
N_EPISODES = 500            # subset, enough for stable timing
THREADS = 4                 # repo-measured optimum for the CPU path
FULL_CORPUS_EPISODES = 3300  # what an actual 1.2M-frame training run has to chew through


def load_some(data_dir: str, n_eps: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Same split-by-ep_lens logic as train_encoder.load_episodes, stopped early."""
    episodes: List[Tuple[np.ndarray, np.ndarray]] = []
    for sp in sorted(glob(os.path.join(data_dir, "shard_*.npz"))):
        d = np.load(sp, allow_pickle=False)
        frames, targets, lens = d["frames"], d["targets"], d["ep_lens"]
        if frames.shape[-1] != ENCODER_IN_DIM:
            raise ValueError(f"{sp}: {frames.shape[-1]} dims, expected {ENCODER_IN_DIM}")
        off = 0
        for L in lens:
            L = int(L)
            episodes.append((frames[off:off + L], targets[off:off + L]))
            off += L
            if len(episodes) >= n_eps:
                return episodes
    return episodes


def prep_episodes(episodes, norm):
    return [(norm.standardize_frame(f).astype(np.float32),
             norm.standardize_targets(t).astype(np.float32)) for f, t in episodes]


def make_batches(eps, batch_size: int, seed: int = 0):
    """One epoch of shuffled batches - identical across devices."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eps))
    out = []
    for i in range(0, len(order), batch_size):
        batch = [eps[j] for j in order[i:i + batch_size]]
        lens = np.asarray([len(f) for f, _ in batch], dtype=np.int64)
        T = int(lens.max())
        B = len(batch)
        x = np.zeros((B, T, ENCODER_IN_DIM), dtype=np.float32)
        y = np.zeros((B, T, target_dim), dtype=np.float32)
        m = np.zeros((B, T), dtype=np.float32)
        for b, (f, t) in enumerate(batch):
            L = len(f)
            x[b, :L] = f
            y[b, :L] = t
            m[b, :L] = 1.0
        out.append((x, y, m, lens))
    return out


print("=" * 78)
print("ENCODER TRAINING STEP: CPU vs MPS")
print("=" * 78)
print(f"  torch {torch.__version__}   mps_available={torch.backends.mps.is_available()}")
print(f"  os.cpu_count()={os.cpu_count()}")

print(f"\nLoading {N_EPISODES} episodes from {DATA_DIR} ...", flush=True)
episodes = load_some(DATA_DIR, N_EPISODES)
target_dim = episodes[0][1].shape[1]
n_frames = sum(len(f) for f, _ in episodes)
t_avg = n_frames / len(episodes)
print(f"  {len(episodes)} episodes, {n_frames:,} frames, mean T={t_avg:.0f}, "
      f"targets={target_dim} (episodes padded to the batch max T)")

# Normalization is a fixed preprocessing step; both devices get byte-identical inputs.
f_all = np.concatenate([f for f, _ in episodes], axis=0)
t_all = np.concatenate([t for _, t in episodes], axis=0)
groups = [(f"dim{i}", 1) for i in range(target_dim)]
norm = NormStats.from_frames_and_targets(f_all, t_all, target_groups=groups)
eps_n = prep_episodes(episodes, norm)
del episodes, f_all, t_all


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def run_epoch(device: str, batch_size: int, timed: bool = True):
    """Full pass over the subset on `device`. Returns (seconds, mean_loss_mu) or None."""
    if device == "cpu":
        torch.set_num_threads(THREADS)
    model = te.EncoderWithHead(n_targets=target_dim, f_in=ENCODER_IN_DIM, width=48, z_dim=16)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batched = make_batches(eps_n, batch_size)

    def step(x, y, m):
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        mt = torch.from_numpy(m).to(device)
        _, mu, logvar = model.forward_sequence(xt)
        se = ((mu - yt) ** 2).sum(dim=-1)
        loss_mu = (se * mt).sum() / mt.sum().clamp(min=1.0) / model.n_targets
        loss_var = te.gaussian_nll(mu.detach(), model.clamp_logvar(logvar), yt)
        loss = loss_mu + 0.1 * loss_var
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss_mu.item())

    step(*batched[0][:3])                     # warmup (kernel autotune, lazy allocs)
    sync(device)
    if not timed:
        return None

    t0 = time.perf_counter()
    losses = [step(*b[:3]) for b in batched]
    sync(device)
    return time.perf_counter() - t0, float(np.mean(losses)), len(batched)


print()
print("A. Is MPS even usable for this model?")
print("-" * 78)
mps_ok = torch.backends.mps.is_available()
mps_note = ""
if mps_ok:
    try:
        run_epoch("mps", 64, timed=False)
        print("  MPS runs the GRU forward/backward: OK")
    except Exception as e:                                    # noqa: BLE001
        mps_ok = False
        mps_note = f"{type(e).__name__}: {e}"
        print(f"  MPS FAILED: {mps_note}")
else:
    print("  MPS not available in this torch build")

print()
print("B. One full pass over 500 episodes (identical batches, same seed)")
print("-" * 78)
print(f"  {'device':>6} {'batch':>6} {'steps':>6} {'best s':>8} {'ms/step':>9} "
      f"{'loss_mu':>9}  extrapolated to 3,300 episodes")
results = {}
REPS = 3
for device in (["cpu", "mps"] if mps_ok else ["cpu"]):
    for bs in (64, 128, 256):
        times = []
        for _ in range(REPS):
            sec, loss, n_steps = run_epoch(device, bs)
            times.append(sec)
        best = min(times)          # best-of-N: least affected by other processes
        results[(device, bs)] = best
        full = best * FULL_CORPUS_EPISODES / len(eps_n)
        print(f"  {device.upper():>6} {bs:>6} {n_steps:>6} "
              f"{best:>8.2f} {(best / n_steps) * 1e3:>9.1f} "
              f"{loss:>9.4f}  {full / 60:>6.1f} min/epoch")

print()
print("C. Verdict")
print("-" * 78)
cpu_sec = results.get(("cpu", 64))
mps_sec = results.get(("mps", 64))
if mps_sec is None:
    print("  MPS unusable for this model; stay on CPU.")
else:
    print(f"  batch 64: CPU {cpu_sec:.2f}s vs MPS {mps_sec:.2f}s "
          f"-> MPS runs at {cpu_sec / mps_sec:.2f}x the CPU speed")
    best_dev = min(results, key=results.get)
    print(f"  fastest overall: {best_dev[0].upper()} at batch {best_dev[1]} "
          f"({results[best_dev]:.2f}s/pass)")
print()
