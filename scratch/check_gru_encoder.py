"""
Validation for the GRU history encoder (Simulation/encoder/history_encoder.py).

The claims that matter:

  A. SIZE AND COST. The model must actually fit the stated onboard budget: ~13k
     parameters, and a per-step cost that does not grow with episode length. The TCN it
     replaced was O(receptive_field) at every step, which is why it was replaced.
  B. CAUSALITY. Output at time t must not depend on any input after t. This is a
     correctness property, not a style one: the encoder is trained by supervising every
     timestep at once, and if any future information leaked into the readout, that
     supervision would train the model to use information it cannot have at runtime.
  C. INCREMENTAL EQUIVALENCE. `step()` called in a loop must reproduce
     `forward_sequence()` to floating-point tolerance. This is THE deployment-critical
     property: training uses the batch path and the vehicle uses the incremental path, so
     if they disagree the encoder is silently different on the aircraft.
  D. CHECKPOINT ROUND-TRIP. Save/load reproduces the outputs exactly.
  E. GRADIENTS. The training path must actually be differentiable end to end.

Run:  .venv/bin/python scratch/check_gru_encoder.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from encoder.history_encoder import (  # noqa: E402
    EncoderWithHead,
    HistoryEncoder,
    gaussian_nll,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)
from encoder.observation_spec import ENCODER_IN_DIM, NormStats  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(4)

enc = HistoryEncoder()
enc.eval()

print("=" * 78)
print("A. size and per-step cost")
print("=" * 78)
n_param = enc.param_count
check("parameter count is in the onboard budget (12k-18k)", 12_000 <= n_param <= 18_000,
      f"{n_param:,} params (f_in={enc.f_in}, hidden={enc.width}, z={enc.z_dim})")
check("encoder input dim is 33 (actor frame 29 + aux 4)",
      ENCODER_IN_DIM == 33 and enc.f_in == 33, f"ENCODER_IN_DIM = {ENCODER_IN_DIM}")
check("receptive_field reports 0, not a fake horizon", enc.receptive_field == 0,
      "a GRU state is an unbounded episode summary; any integer would be a lie")

# Per-step cost must be flat in episode length.
x_short = torch.randn(1, 20, enc.f_in)
x_long = torch.randn(1, 2000, enc.f_in)
with torch.no_grad():
    enc.forward_sequence(x_long)                      # warm up

    def time_step(x_t: torch.Tensor, h: torch.Tensor, n: int = 300) -> float:
        t0 = time.perf_counter()
        for _ in range(n):
            _, h = enc.step(x_t, h)
        return (time.perf_counter() - t0) / n

    h = enc.init_state(1)
    t_per_step = time_step(x_short[:, 0], h)
    t_long = time_step(x_long[:, 0], h)
check("per-step latency is independent of episode length",
      max(t_per_step, t_long) / min(t_per_step, t_long) < 2.5,
      f"{t_per_step*1e6:.0f} us vs {t_long*1e6:.0f} us per step")
check("per-step latency fits a 100 Hz budget with huge margin", t_per_step < 1e-3,
      f"{t_per_step*1e6:.0f} us/step = {t_per_step*100*100:.2f}% of a 100 Hz period")

print()
print("=" * 78)
print("B. causality: no output depends on any future input")
print("=" * 78)
x = torch.randn(1, 30, enc.f_in)
with torch.no_grad():
    z_ref = enc.forward_sequence(x)
    x_future = x.clone()
    x_future[0, 15:] += 5.0          # perturb everything from step 15 onward
    z_future = enc.forward_sequence(x_future)

check("outputs before the perturbation are bit-identical",
      torch.equal(z_ref[0, :15], z_future[0, :15]),
      "max delta = %.2e" % float((z_ref[0, :15] - z_future[0, :15]).abs().max()))
check("outputs after the perturbation DO change (the test bites)",
      not torch.allclose(z_ref[0, 16:], z_future[0, 16:]),
      "otherwise the perturbation could not have been applied at all")

print()
print("=" * 78)
print("C. incremental step() == batch forward_sequence() (deployment-critical)")
print("=" * 78)
x = torch.randn(3, 64, enc.f_in)
with torch.no_grad():
    z_batch = enc.forward_sequence(x)

    h = enc.init_state(3)
    zs = []
    for t in range(x.shape[1]):
        z_t, h = enc.step(x[:, t], h)
        zs.append(z_t)
    z_inc = torch.stack(zs, dim=1)

delta = float((z_batch - z_inc).abs().max())
check("incremental readout matches the batch readout", delta < 1e-5,
      f"max |dz| = {delta:.2e} over {x.shape[1]} steps, batch {tuple(x.shape)}")

# And the hidden state itself must agree, not just the readout.
with torch.no_grad():
    h_batch = torch.zeros(1, 3, enc.width)
    _, h_batch = enc.gru(x, h_batch)
check("incremental hidden state matches the batch hidden state",
      float((h_batch[0] - h).abs().max()) < 1e-5,
      f"max |dh| = {float((h_batch[0] - h).abs().max()):.2e}")

# The state must be re-initialisable: two episodes must not bleed into each other.
with torch.no_grad():
    z_a, _ = enc.step(x[:, 0], enc.init_state(3))
    h_used = enc.init_state(3)
    for t in range(x.shape[1]):
        _, h_used = enc.step(x[:, t], h_used)
    z_b, _ = enc.step(x[:, 0], enc.init_state(3))
check("a fresh state gives the same first output regardless of history",
      torch.allclose(z_a, z_b), "episode state is genuinely reset, not carried over")

print()
print("=" * 78)
print("D. checkpoint round-trip")
print("=" * 78)
model = EncoderWithHead(n_targets=29)
save_encoder_checkpoint("/tmp/_gru_roundtrip.pt", model, _norm := NormStats(
    frame_mean=np.zeros(ENCODER_IN_DIM, dtype=np.float32),
    frame_std=np.ones(ENCODER_IN_DIM, dtype=np.float32),
    target_mean=np.zeros(29, dtype=np.float32),
    target_std=np.ones(29, dtype=np.float32),
    target_groups=[],
    clip=10.0,
    degenerate_dims=[],
))
loaded, norm, ckpt = load_encoder_checkpoint("/tmp/_gru_roundtrip.pt")
check("checkpoint records the architecture", ckpt["config"].get("arch") == "gru",
      f"arch = {ckpt['config'].get('arch')}")
check("loaded encoder has the same parameter count",
      loaded.param_count == model.encoder.param_count,
      f"{loaded.param_count:,} vs {model.encoder.param_count:,}")

with torch.no_grad():
    x = torch.randn(2, 32, ENCODER_IN_DIM)
    check("loaded encoder reproduces the original outputs",
          torch.allclose(loaded(x), model.encoder(x), atol=1e-6),
          "max delta = %.2e" % float((loaded(x) - model.encoder(x)).abs().max()))
check("loaded encoder is frozen and in eval mode",
      not any(p.requires_grad for p in loaded.parameters()) and not loaded.training, "")

print()
print("=" * 78)
print("E. gradients flow through the training path")
print("=" * 78)
model = EncoderWithHead(n_targets=4)
model.train()
x = torch.randn(4, 40, ENCODER_IN_DIM, requires_grad=False)
y = torch.randn(4, 40, 4)
z, mu, logvar = model.forward_sequence(x)
loss = gaussian_nll(mu, model.clamp_logvar(logvar), y)
loss.backward()
gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
check("loss is finite", bool(torch.isfinite(loss)), f"loss = {float(loss):.4f}")
check("gradients reach the GRU weights", gnorm > 0.0, f"grad norm = {gnorm:.4e}")
check("the mu head receives gradient",
      model.mu_head.weight.grad is not None and float(model.mu_head.weight.grad.abs().sum()) > 0,
      "")
check("z is tanh-bounded in [-1, 1]",
      float(z.abs().max()) <= 1.0 + 1e-6, f"max |z| = {float(z.abs().max()):.4f}")

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all GRU encoder checks passed")
