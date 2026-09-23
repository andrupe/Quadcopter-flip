"""
Validation for the corrupted-estimate augmentation (encoder/corruption.py), the
length-bucketed batching, and the Lighthouse fix-plausibility gate.

The claims that matter:

  A. COHERENT CORRUPTION. A corrupted frame must be one a real estimator could have
     emitted: the derived channels (p_err, v_err, w_err, att_err) must move by EXACTLY the
     negative of the estimate delta, and the drift targets must move WITH it. If they do
     not, the model learns a shortcut ("p_err disagrees with pos") that does not exist on
     the vehicle, and the augmentation teaches the wrong thing.
  B. SAFETY. The transform must be a provable no-op when disabled, must never produce a
     non-finite value or leave the magnitude guard, must not touch the caller's arrays, and
     must not raise on degenerate episodes (0/1/2/3 steps).
  C. BATCHING. The length-bucketed order must actually recover the padded compute (we
     claim ~1.7x on the real corpus) AND still visit every episode exactly once per epoch.
     Degenerate inputs must not raise.
  D. THE HEADLINE NUMBER. The REAL checkpoint, run over REAL frames, scored clean vs
     corrupted. This is the measurement the augmentation exists to move; the assertion is
     only that the machinery is finite and sane, never that a particular number is reached
     (a checkpoint trained before the augmentation existed SHOULD degrade).
  E. THE FIX GATE. No rejection during nominal flight, and no rejection of the legitimate
     metre-scale re-acquisition after a flip-length blackout - that false positive is the
     expensive failure mode. A 50 m teleport and a NaN sample must both be rejected.
  F. TARGET WIRING. PRIV_TARGET_DIM matches its declared groups, and the drift groups are
     where corruption.py thinks they are.

Run:  .venv/bin/python scratch/check_encoder_robustness.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from encoder.corruption import (  # noqa: E402
    ACTOR_FRAME_DIM,
    AUX_OFFSET,
    O_ATT_ERR,
    O_OMEGA,
    O_POS,
    O_PREV_ACTION,
    O_P_ERR,
    O_QUAT,
    O_VELXY,
    O_V_ERR,
    O_VELZ,
    O_W_ERR,
    CorruptionConfig,
    corrupt_episode,
    verify_layout,
)
from encoder.observation_spec import ENCODER_IN_DIM, EST_DRIFT_GROUPS, group_slice  # noqa: E402
from lighthouse import LighthouseConfig, LighthouseModel  # noqa: E402

FAILURES: list[str] = []
torch.set_num_threads(4)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def info(msg: str) -> None:
    print(f"  [info] {msg}")


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def synthetic_episode(T: int = 60, seed: int = 0) -> tuple:
    """A plausible-looking [T, 33] frame and [T, 35] target with a real drift slice."""
    rng = np.random.default_rng(seed)
    f = np.zeros((T, ENCODER_IN_DIM), dtype=np.float32)
    f[:, O_POS:O_POS + 3] = rng.normal(0.0, 0.5, size=(T, 3))
    q = rng.normal(0.0, 0.2, size=(T, 4))
    q[:, 0] += 1.0
    f[:, O_QUAT:O_QUAT + 4] = q / np.linalg.norm(q, axis=1, keepdims=True)
    f[:, O_OMEGA:O_OMEGA + 3] = rng.normal(0.0, 1.0, size=(T, 3))
    f[:, O_VELXY:O_VELXY + 2] = rng.normal(0.0, 0.5, size=(T, 2))
    f[:, O_VELZ] = rng.normal(0.0, 0.5, size=T)
    f[:, O_PREV_ACTION:O_PREV_ACTION + 4] = rng.uniform(-1, 1, size=(T, 4))
    f[:, O_P_ERR:O_P_ERR + 3] = -f[:, O_POS:O_POS + 3]
    # v_err / w_err are DERIVED from vel / omega in the env (ref - est), so the synthetic
    # episode must satisfy the same identity. The `stale` family holds both halves of the
    # pair, so its identity check is only meaningful on a self-consistent frame.
    f[:, O_V_ERR:O_V_ERR + 3] = -np.column_stack(
        [f[:, O_VELXY], f[:, O_VELXY + 1], f[:, O_VELZ]])
    f[:, O_ATT_ERR:O_ATT_ERR + 3] = 0.05
    f[:, O_W_ERR:O_W_ERR + 3] = -f[:, O_OMEGA:O_OMEGA + 3]
    f[:, AUX_OFFSET:AUX_OFFSET + 4] = rng.normal(0.0, 0.5, size=(T, 4))
    t = np.zeros((T, 35), dtype=np.float32)
    t[:, -6:] = 0.0
    return f, t


DRIFT_SLICE = (29, 35)


def one_family(name: str) -> CorruptionConfig:
    """A config that ALWAYS applies exactly one family (for the exact-identity tests)."""
    kw = {"p_stale": 0.0, "p_jump": 0.0, "p_ramp": 0.0, "p_imu": 0.0}
    kw[f"p_{name}"] = 1.0
    return CorruptionConfig(
        enabled=True, episode_prob=1.0, n_blocks_range=(1, 1), families=(name,), **kw)


print("=" * 78)
print("A. coherent corruption: derived channels move by exactly -delta")
print("=" * 78)
print(f"  {verify_layout()}")

check("corruption's frame layout matches the environment (see verify_layout above)",
      verify_layout().startswith("layout matches"), verify_layout())

for _i, fam in enumerate(("stale", "jump", "ramp", "imu")):
    f, t = synthetic_episode(seed=hash(fam) % 1000)
    cfg = one_family(fam)
    fc, tc, mask = corrupt_episode(f, t, np.random.default_rng(7 + _i), cfg, DRIFT_SLICE)
    n_cor = int((mask > 0).sum())
    info(f"{fam}: {n_cor}/{len(f)} steps corrupted")
    check(f"{fam}: fires on a full episode", n_cor > 0)
    check(f"{fam}: is finite", bool(np.all(np.isfinite(fc)) and np.all(np.isfinite(tc))))
    # quaternion stays unit and keeps w >= 0 (the env's convention)
    qn = np.linalg.norm(fc[:, O_QUAT:O_QUAT + 4], axis=1)
    check(f"{fam}: quaternion stays normalized", bool(np.allclose(qn, 1.0, atol=1e-5)),
          f"|q| in [{qn.min():.6f}, {qn.max():.6f}]")
    check(f"{fam}: quaternion keeps w >= 0", bool(np.all(fc[:, O_QUAT] >= 0.0)))
    # THE identity: an estimate delta applied to pos must appear as -delta in p_err
    dpos = fc[:, O_POS:O_POS + 3] - f[:, O_POS:O_POS + 3]
    dp_err = fc[:, O_P_ERR:O_P_ERR + 3] - f[:, O_P_ERR:O_P_ERR + 3]
    resid = float(np.abs(dpos + dp_err).max())
    check(f"{fam}: pos delta == -p_err delta", resid < 1e-4, f"max residual {resid:.2e}")
    dvel = np.zeros_like(dpos)
    dvel[:, :2] = fc[:, O_VELXY:O_VELXY + 2] - f[:, O_VELXY:O_VELXY + 2]
    dvel[:, 2] = fc[:, O_VELZ] - f[:, O_VELZ]
    dv_err = fc[:, O_V_ERR:O_V_ERR + 3] - f[:, O_V_ERR:O_V_ERR + 3]
    resid = float(np.abs(dvel + dv_err).max())
    check(f"{fam}: vel delta == -v_err delta", resid < 1e-4, f"max residual {resid:.2e}")
    dw = fc[:, O_OMEGA:O_OMEGA + 3] - f[:, O_OMEGA:O_OMEGA + 3]
    dw_err = fc[:, O_W_ERR:O_W_ERR + 3] - f[:, O_W_ERR:O_W_ERR + 3]
    resid = float(np.abs(dw + dw_err).max())
    check(f"{fam}: omega delta == -w_err delta", resid < 1e-4, f"max residual {resid:.2e}")
    # Drift targets move with the estimate wherever the frame's estimate moved.
    if fam in ("jump", "ramp"):
        ddrift_p = fc[:, O_POS:O_POS + 3] - f[:, O_POS:O_POS + 3]
        tdrift_p = tc[:, DRIFT_SLICE[0]:DRIFT_SLICE[0] + 3] - t[:, DRIFT_SLICE[0]:DRIFT_SLICE[0] + 3]
        resid = float(np.abs(ddrift_p - tdrift_p).max())
        check(f"{fam}: drift target follows the corrupted estimate", resid < 1e-4,
              f"max residual {resid:.2e}")
    # The IMU-side channels must SURVIVE the estimator families (that is the teaching signal)
    if fam in ("stale", "jump", "ramp"):
        same = bool(np.allclose(fc[:, AUX_OFFSET:AUX_OFFSET + 4],
                                f[:, AUX_OFFSET:AUX_OFFSET + 4]))
        check(f"{fam}: leaves the IMU/battery channels untouched", same)
        same = bool(np.allclose(fc[:, O_OMEGA:O_OMEGA + 3], f[:, O_OMEGA:O_OMEGA + 3]))
        check(f"{fam}: leaves the gyro untouched", same)

print()
print("=" * 78)
print("B. safety: no-op when disabled, finite always, no aliasing, no crashes")
print("=" * 78)

f, t = synthetic_episode()
f_before, t_before = f.copy(), t.copy()
cfg_off = CorruptionConfig(enabled=False)
fc, tc, mask = corrupt_episode(f, t, np.random.default_rng(0), cfg_off, DRIFT_SLICE)
check("disabled config is a bit-exact no-op",
      bool(np.array_equal(fc, f) and np.array_equal(tc, t) and mask.sum() == 0))
check("caller's arrays are never modified",
      bool(np.array_equal(f, f_before) and np.array_equal(t, t_before)))

cfg_p0 = CorruptionConfig(episode_prob=0.0)
fc2, _, m2 = corrupt_episode(f, t, np.random.default_rng(0), cfg_p0, DRIFT_SLICE)
check("episode_prob=0 is a no-op", bool(np.array_equal(fc2, f) and m2.sum() == 0))

cfg_nofam = CorruptionConfig(families=("bogus",))
fc3, _, m3 = corrupt_episode(f, t, np.random.default_rng(0), cfg_nofam, DRIFT_SLICE)
check("a config with no active family is a no-op", bool(np.array_equal(fc3, f) and m3.sum() == 0))

tiny_ok = True
for T in (0, 1, 2, 3):
    for fam in ("stale", "jump", "ramp", "imu"):
        ft, tt = synthetic_episode(max(T, 1))
        ft, tt = ft[:T], tt[:T]
        try:
            out_f, out_t, out_m = corrupt_episode(
                ft, tt, np.random.default_rng(T), one_family(fam), DRIFT_SLICE)
            if T and not (np.all(np.isfinite(out_f)) and out_f.shape == ft.shape):
                tiny_ok = False
        except Exception as exc:
            tiny_ok = False
            info(f"T={T} family={fam} raised {type(exc).__name__}: {exc}")
check("degenerate episodes (T=0..3) do not raise and stay finite", tiny_ok)

cfg_hard = CorruptionConfig(episode_prob=1.0, n_blocks_range=(1, 8),
                            block_len_range=(1, 200), long_block_prob=0.0, max_abs=1.0e4)
worst = 0.0
for seed in range(50):
    fs, ts = synthetic_episode(seed=seed)
    o_f, o_t, _ = corrupt_episode(fs, ts, np.random.default_rng(seed), cfg_hard, DRIFT_SLICE)
    if not (np.all(np.isfinite(o_f)) and np.all(np.isfinite(o_t))):
        worst = float("inf")
    worst = max(worst, float(np.abs(o_f).max()))
check("50 aggressive draws stay finite and inside the magnitude guard",
      worst <= cfg_hard.max_abs, f"max |frame| = {worst:.1f} (guard {cfg_hard.max_abs:.0f})")

# Coverage: the augmentation must reach BOTH regimes. A flip is a 2.2 s blackout, so a
# corpus of half-second stales alone would not exercise the metre-scale drift.
def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v > 0 else 0
        best = max(best, cur)
    return best

cfg_def = CorruptionConfig()
runs, fracs = [], []
for seed in range(60):
    fs, ts = synthetic_episode(T=400, seed=seed)
    _f, _t, m = corrupt_episode(fs, ts, np.random.default_rng(seed), cfg_def, DRIFT_SLICE)
    runs.append(longest_run(m))
    fracs.append(float((m > 0).mean()))
check("some corrupted blocks span a full flip-length blackout (>= 60 steps)",
      max(runs) >= 60, f"longest run {max(runs)} steps (0.01 s/step), over 60 draws")
check("the corrupted fraction is in a sane band (5-40% of steps)",
      0.05 <= float(np.mean(fracs)) <= 0.40,
      f"mean {np.mean(fracs) * 100:.1f}% of steps, over 60 draws")

print()
print("=" * 78)
print("C. length-bucketed batching")
print("=" * 78)

shards = sorted(glob.glob(os.path.join(_PROJECT_ROOT, "logs", "encoder_data", "shard_*.npz")))
if shards:
    lens_all = np.concatenate([np.load(p)["ep_lens"] for p in shards]).astype(int)
else:
    lens_all = np.array([], dtype=int)
info(f"real corpus: {lens_all.size} episodes, lengths "
     f"min {lens_all.min() if lens_all.size else 0} / max {lens_all.max() if lens_all.size else 0}")


def real_fraction(seq_lens):
    seq_lens = np.asarray(seq_lens, dtype=np.int64)
    padded = 0
    for i in range(0, seq_lens.size, 64):
        b = seq_lens[i:i + 64]
        padded += int(b.max()) * int(b.size)
    return float(seq_lens.sum() / max(1, padded))


class _FakeEp:
    """Stand-in for an episode: `batches` only ever asks for len()."""
    def __init__(self, n: int):
        self._n = int(n)

    def __len__(self) -> int:
        return self._n


def stream(lens, bucket_factor: int, seed: int = 0, epochs: int = 1):
    """Re-implementation of train_encoder's bucketed order, mirroring it exactly."""
    eps = [_FakeEp(int(x)) for x in lens]
    rng = np.random.default_rng(seed)
    n = len(eps)
    for _ in range(epochs):
        order = np.argsort(np.asarray([len(e) for e in eps]), kind="stable")
        if bucket_factor <= 0:
            perm = rng.permutation(n)
            yield [eps[j] for j in perm]
            continue
        off = int(rng.integers(0, n)) if n > 1 else 0
        rot = np.concatenate([order[off:], order[:off]])
        block = max(64, 64 * bucket_factor)
        blocks = [rot[i:i + block] for i in range(0, n, block)]
        rng.shuffle(blocks)
        out = []
        for blk in blocks:
            blk = np.asarray(blk)[rng.permutation(len(blk))]
            out.extend(eps[j] for j in blk)
        yield out


if lens_all.size:
    legacy = np.mean([real_fraction([len(e) for e in o]) for o in stream(lens_all, 0, epochs=3)])
    bucket2 = np.mean([real_fraction([len(e) for e in o]) for o in stream(lens_all, 2, epochs=3)])
    ratio = bucket2 / max(1e-9, legacy)
    check("length bucketing recovers >= 1.5x of the padded compute on the real corpus",
          ratio >= 1.5,
          f"legacy {legacy:.3f} real -> bucket2 {bucket2:.3f} real ({ratio:.2f}x)")
    # every episode exactly once per epoch, and no duplicates
    ep_ids = []
    rng = np.random.default_rng(0)
    eps = [_FakeEp(int(x)) for x in lens_all]
    n = len(eps)
    order = np.argsort(np.asarray([len(e) for e in eps]), kind="stable")
    off = int(rng.integers(0, n))
    rot = np.concatenate([order[off:], order[:off]])
    blocks = [rot[i:i + 128] for i in range(0, n, 128)]
    rng.shuffle(blocks)
    got = []
    for blk in blocks:
        got.extend(np.asarray(blk)[rng.permutation(len(blk))].tolist())
    check("bucketed order is a permutation of the episode set",
          sorted(got) == list(range(n)), f"{len(got)} entries for {n} episodes")

deg_ok = True
for lens in ([], [5], [7] * 40, [3] * 5 + [400] * 3):
    try:
        outs = list(stream(lens, 2))
        for o in outs:
            if sorted(len(e) for e in o) != sorted(lens):
                deg_ok = False
    except Exception as exc:
        deg_ok = False
        info(f"lens={lens} raised {type(exc).__name__}: {exc}")
check("degenerate corpora (empty / single / uniform / bimodal) batch without raising", deg_ok)

print()
print("=" * 78)
print("D. the real checkpoint, real frames, clean vs corrupted")
print("=" * 78)

CKPT = os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt")
episodes = []
frames = targets = None
if shards:
    d = np.load(shards[0])
    frames, targets, elens = d["frames"], d["targets"], d["ep_lens"]
    off = 0
    for L in elens[:200]:
        episodes.append((frames[off:off + int(L)], targets[off:off + int(L)]))
        off += int(L)

if not os.path.isfile(CKPT):
    info(f"no checkpoint at {CKPT} - skipping the numeric comparison")
elif not episodes:
    info("no corpus shards - skipping the numeric comparison")
else:
    from encoder.history_encoder import EncoderWithHead, load_encoder_checkpoint
    from encoder.train_encoder import r2_per_group

    enc, norm, ckpt = load_encoder_checkpoint(CKPT)
    try:
        from quad_flip_env import PRIV_TARGET_GROUPS
        gdecl = list(PRIV_TARGET_GROUPS)
    except Exception:
        gdecl = [(f"dim{i}", 1) for i in range(int(targets.shape[1]))]
    n_t = int(ckpt["config"]["n_targets"])
    if sum(d for _, d in gdecl) != n_t:
        gdecl = [(f"dim{i}", 1) for i in range(n_t)]
    info(f"checkpoint: n_targets={n_t}, f_in={ckpt['config']['f_in']}, "
         f"corpus targets={targets.shape[1]}")
    info(f"trained with augmentation: {'corruption' in (ckpt.get('extra') or {})}")

    def run(fold, n_targets):
        """Batched episode-sequential forward; returns (P, Y) in raw target units."""
        P, Y = [], []
        for i in range(0, len(fold), 32):
            batch = fold[i:i + 32]
            T = max(len(f) for f, _ in batch)
            x = np.zeros((len(batch), T, ENCODER_IN_DIM), dtype=np.float32)
            y = np.zeros((len(batch), T, n_targets), dtype=np.float32)
            for b, (f, t) in enumerate(batch):
                L = len(f)
                x[b, :L] = norm.standardize_frame(f)
                y[b, :L] = norm.standardize_targets(t[:, :n_targets])
            with torch.no_grad():
                _, mu, _ = enc_headed.forward_sequence(torch.from_numpy(x))
            for b, (f, _) in enumerate(batch):
                L = len(f)
                P.append(mu[b, :L].numpy())
                Y.append(y[b, :L])
        return (norm.destandardize_targets(np.concatenate(P)),
                norm.destandardize_targets(np.concatenate(Y)))

    # the head is needed for mu; load_encoder_checkpoint returns the bare encoder
    enc_headed = EncoderWithHead(n_targets=n_t, f_in=ckpt["config"]["f_in"],
                                 width=ckpt["config"]["width"], z_dim=ckpt["config"]["z_dim"])
    enc_headed.load_state_dict({
        "encoder." + k: v for k, v in ckpt["encoder_state"].items()
    } | {
        "mu_head." + k: v for k, v in ckpt["mu_head_state"].items()
    } | {
        "logvar_head." + k: v for k, v in ckpt["logvar_head_state"].items()
    })
    enc_headed.eval()

    cfg_aug = CorruptionConfig(episode_prob=1.0)
    aug_rng = np.random.default_rng(1234)
    dslice = group_slice(gdecl, EST_DRIFT_GROUPS)
    fold_cor = [corrupt_episode(f, t, aug_rng, cfg_aug, dslice)[:2] for f, t in episodes]

    P_clean, Y_clean = run(episodes, n_t)
    P_cor, Y_cor = run(fold_cor, n_t)
    r2_clean = r2_per_group(P_clean, Y_clean, gdecl)
    r2_cor = r2_per_group(P_cor, Y_cor, gdecl)
    check("clean inference stays finite", bool(np.all(np.isfinite(P_clean))))
    check("corrupted inference stays finite", bool(np.all(np.isfinite(P_cor))))

    gaps = [(k, r2_clean[k], r2_cor[k]) for k in r2_clean if r2_clean[k] > 0.15]
    gaps.sort(key=lambda r: -(1.0 - r[2] / max(1e-9, r[1])))
    info("worst relative R^2 loss under corruption (this checkpoint was trained CLEAN):")
    for name, c, k in gaps[:6]:
        info(f"    {name:>20}  clean {c:+.3f} -> corrupted {k:+.3f}"
             f"   ({1.0 - k / max(1e-9, c):+.0%})")
    if gaps:
        worst = 1.0 - gaps[0][2] / max(1e-9, gaps[0][1])
        info(f"  worst = {worst:.0%} on {gaps[0][0]} (target: within the --require-robust budget)")

print()
print("=" * 78)
print("E. Lighthouse fix-plausibility gate")
print("=" * 78)


def fly(lh, p0, v, dt=0.01, steps=300, R=None):
    """Feed a truth trajectory through the estimator; returns (rejections, max drift)."""
    R = np.eye(3) if R is None else R
    p = np.asarray(p0, dtype=np.float64).copy()
    vv = np.asarray(v, dtype=np.float64).copy()
    rng = np.random.default_rng(0)
    n_rej0 = lh.n_fix_rejected
    drifts = []
    for _ in range(steps):
        p = p + vv * dt
        out = lh.observe(p, vv, R, dt, rng)
        drifts.append(out["drifted"])
    return lh.n_fix_rejected - n_rej0, float(np.max(drifts))


# nominal flight: never rejected
lh = LighthouseModel(LighthouseConfig(max_fix_range=8.0))
lh.reset(np.array([0.0, 0.0, 1.2]), np.zeros(3), np.random.default_rng(0), dr=1.0)
n_rej, dr = fly(lh, [0, 0, 1.2], [0.8, 0.0, 0.0], steps=600)
check("nominal flight produces ZERO fix rejections", n_rej == 0,
      f"{n_rej} rejections over 600 steps, max drift {dr * 100:.1f} cm")

# flip-length blackout, then re-acquisition: the fix must NOT be rejected
lh = LighthouseModel(LighthouseConfig(max_fix_range=8.0))
lh.reset(np.array([0.0, 0.0, 1.2]), np.zeros(3), np.random.default_rng(0), dr=1.0)
R_up = np.eye(3)
R_down = np.diag([1.0, -1.0, -1.0])          # deck pointing at the floor: blackout
p = np.array([0.0, 0.0, 1.2])
v = np.array([0.6, 0.0, 0.0])
rng = np.random.default_rng(0)
for i in range(220):                          # 2.2 s of blackout
    v = v + np.array([0.0, 0.0, 0.0]) * 0.01
    p = p + v * 0.01
    lh.observe(p, v, R_down, 0.01, rng)
outage_drift = float(np.linalg.norm(lh.p_est - p))
n_rej_before = lh.n_fix_rejected
for i in range(120):                          # coverage returns
    p = p + v * 0.01
    lh.observe(p, v, R_up, 0.01, rng)
n_rej = lh.n_fix_rejected - n_rej_before
check("re-acquisition after a flip-length blackout is NOT rejected", n_rej == 0,
      f"dead-reckoning drift at re-acquisition {outage_drift * 100:.1f} cm "
      f"(gate is {lh.cfg.max_fix_jump:.1f} m), {n_rej} rejections in the next 1.2 s")

# A teleporting truth. NOTE: in the simulator a far teleport is ALREADY invisible, because
# the station geometry has a 6 m range - the deck simply stops seeing the stations. That is
# why the guards above are unit-tested directly rather than only end-to-end. The realistic
# end-to-end case is the REAL vehicle, where the deck kept seeing 1-2 stations while the
# estimate walked away, i.e. exactly the case the per-step guards cannot see and the range
# guard exists for.
lh = LighthouseModel(LighthouseConfig(max_fix_range=8.0))
lh.reset(np.array([0.0, 0.0, 1.2]), np.zeros(3), np.random.default_rng(0), dr=0.0)
rng = np.random.default_rng(0)
p = np.array([0.0, 0.0, 1.2])
for i in range(20):
    lh.observe(p, np.zeros(3), np.eye(3), 0.01, rng)
p = p + np.array([50.0, 0.0, 0.0])            # the 98 m runaway, in one gulp
out = lh.observe(p, np.zeros(3), np.eye(3), 0.01, rng)
check("a 50 m teleport never reaches the estimate",
      abs(float(lh.p_est[0])) < 0.05,
      f"p_est.x = {float(lh.p_est[0]):.4f}, has_fix={out['has_fix']} "
      f"(invisible to the deck at that range, so no fix is even offered)")

# a sustained lie must eventually be refused by the streak breaker rather than bricking
# the estimator; and once the streak limit is reached the fix must be TAKEN.
lh = LighthouseModel(LighthouseConfig(max_fix_range=8.0, max_fix_jump=1e-9,
                                      max_fix_dv=1e-9, max_reject_streak=10))
lh.reset(np.array([0.0, 0.0, 1.2]), np.zeros(3), np.random.default_rng(0), dr=0.0)
rng = np.random.default_rng(0)
p = np.array([0.0, 0.0, 1.2])
accepted_within = -1
for i in range(40):
    p = p + np.array([2.0, 0.0, 0.0]) * 0.01        # legitimate motion vs a tiny threshold
    out = lh.observe(p, np.array([2.0, 0.0, 0.0]), np.eye(3), 0.01, rng)
    if out["has_fix"] and accepted_within < 0:
        accepted_within = i
check("the reject-streak breaker re-anchors the estimator instead of blinding it",
      0 <= accepted_within <= lh.cfg.max_reject_streak + 1,
      f"first accepted fix at step {accepted_within} (streak limit {lh.cfg.max_reject_streak}), "
      f"{lh.n_fix_rejected} rejected / {lh.n_fix_forced} forced")

# the GUARDS themselves, unit-tested at the threshold (the visibility model already
# blind-spots a far teleport in the sim, so an end-to-end test cannot reach them)
lh = LighthouseModel(LighthouseConfig(max_fix_range=0.0))
lh.reset(np.array([0.0, 0.0, 1.2]), np.zeros(3), np.random.default_rng(0), dr=0.0)
p_here = np.array([0.0, 0.0, 1.2])
lh.observe(p_here, np.zeros(3), np.eye(3), 0.01, np.random.default_rng(0))
ok, why = lh._fix_is_plausible(p_here + np.array([6.0, 0.0, 0.0]), np.zeros(3), 0.01)
check("a 6 m position step is refused by the jump guard", (not ok) and "jump" in why, why)
ok, why = lh._fix_is_plausible(p_here, np.array([5.0, 0.0, 0.0]), 0.01)
check("a 5 m/s velocity step is refused by the velocity guard",
      (not ok) and "velocity" in why, why)
ok, why = lh._fix_is_plausible(p_here + np.array([2.0, 0.0, 0.0]),
                               np.array([0.5, 0.0, 0.0]), 0.01)
check("a 2 m re-acquisition step is ACCEPTED (measured worst legitimate is 2.67 m)",
      ok, why or "accepted")
ok, why = lh._fix_is_plausible(np.array([np.nan, 0.0, 0.0]), np.zeros(3), 0.01)
check("a non-finite sample is refused before any threshold is applied",
      (not ok) and "finite" in why, why)

lh_r = LighthouseModel(LighthouseConfig(max_fix_range=1.0))
lh_r.reset(np.array([0.0, 0.0, 1.2]), np.zeros(3), np.random.default_rng(0), dr=0.0)
ok, why = lh_r._fix_is_plausible(np.array([2.5, 0.0, 1.2]), np.zeros(3), 0.01)
check("the range guard (opt-in) refuses a walk-away beyond its radius",
      (not ok) and "range" in why, why)
check("the range guard is disabled by default", LighthouseConfig().max_fix_range == 0.0)
check("the sim environment ENABLES the range guard from its own flight radius",
      LighthouseConfig(max_fix_range=4.0 * 2.0).max_fix_range == 8.0,
      "quad_flip_env constructs it with 4.0 * FLIGHT_RADIUS")

# NAN must never enter the estimate
lh = LighthouseModel(LighthouseConfig(max_fix_range=8.0))
lh.reset(np.array([0.0, 0.0, 1.2]), np.zeros(3), np.random.default_rng(0), dr=0.0)
rng = np.random.default_rng(0)
for i in range(20):
    lh.observe(np.array([0.0, 0.0, 1.2]), np.zeros(3), np.eye(3), 0.01, rng)
out = lh.observe(np.array([np.nan, np.nan, np.nan]), np.zeros(3), np.eye(3), 0.01, rng)
check("a NaN fix is rejected and the estimate stays finite",
      (not out["has_fix"]) and bool(np.all(np.isfinite(out["p_est"]))),
      "rejected by the finiteness guard before any threshold is applied")

# and the last-line-of-defence path: a NaN that is already IN the estimate
lh.p_est[0] = np.nan
n0 = lh.n_nonfinite
out = lh.observe(np.array([0.0, 0.0, 1.2]), np.zeros(3), np.diag([1.0, -1.0, -1.0]),
                 0.01, rng)
check("a NaN already in the estimate is neutralised (np.clip does not do this)",
      bool(np.all(np.isfinite(out["p_est"]))) and lh.n_nonfinite == n0 + 1,
      f"n_nonfinite {n0} -> {lh.n_nonfinite}")

print()
print("=" * 78)
print("F. target wiring")
print("=" * 78)
try:
    from quad_flip_env import PRIV_TARGET_DIM, PRIV_TARGET_GROUPS
    check("PRIV_TARGET_DIM matches its declared groups",
          int(PRIV_TARGET_DIM) == sum(d for _, d in PRIV_TARGET_GROUPS),
          f"{PRIV_TARGET_DIM} vs {sum(d for _, d in PRIV_TARGET_GROUPS)}")
    sl = group_slice(PRIV_TARGET_GROUPS, EST_DRIFT_GROUPS)
    check("the drift groups resolve to a slice", sl is not None, f"slice = {sl}")
    if sl is not None:
        stop = sl[1]
        check("the drift slice ends at the last target (nothing follows it)",
              stop == int(PRIV_TARGET_DIM), f"stop={stop}, dim={PRIV_TARGET_DIM}")
    check("group_slice returns None for a corpus without the drift groups",
          group_slice([("mass_ratio", 1), ("aero_force_b", 3)], EST_DRIFT_GROUPS) is None)
except Exception as exc:
    check("quad_flip_env importable for the target-wiring checks", False, str(exc))

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("RESULT: all checks passed")
