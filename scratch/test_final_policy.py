# -*- coding: utf-8 -*-
"""
EXTENSIVE TEST CAMPAIGN ON A TRAINED CHECKPOINT.

This is the "how is it actually doing" tool. It is deliberately NOT evaluate.py: that
program is one interactive episode with a viewer and a plot, which is the right thing to
fly but the wrong thing to MEASURE with (one episode on one reference draw, chosen by a
different RNG stream every time you run it).

Every section here is headless, deterministic and independent, and the scoring protocol is
the TRAINING CALLBACK'S OWN (train.py::_sweep): the same families, the same seeds k=0..n-1,
the same encoder injector, the same 7.3 ceiling - so a number printed here is directly
comparable to the "[Eval @ N] ... % of 7.30" line the run reported for itself. Anything
that did not match the callback would be measuring a different task.

SECTIONS
  [1] PER-FAMILY, nominal plant (dr=0)      - the headline. Which families work.
  [2] PER-FAMILY, full DR (dr=1)            - what the ADR ramp buys and what it costs.
  [3] WHERE THE REWARD GOES                 - pos/vel/att/rate decomposition, both.
  [4] HOW TERMINATED EPISODES END           - the crash census, per family and per dr.
  [5] DR SWEEP                              - degradation curve over dr = 0..1.
  [6] FLIP EXECUTION                        - does the vehicle actually INVERT? (behaviour,
                                              not reward: the flip reward column is known
                                              to be uninformative - see diag_flip_execution)
  [7] ENCODER ABLATION                      - zero the latent. Does the policy USE z?
  [8] LIGHTHOUSE FAILURE                    - loss/outage/runaway/teleport. Does it survive
                                              a lying estimator?
  [9] REPEATABILITY                         - same seed twice, byte-identical score?

Run:  .venv/bin/python scratch/test_final_policy.py [model.zip] [episodes_per_family]
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")     # never try to open a window

import numpy as np                              # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "Simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quad_flip_env as qfe                                          # noqa: E402
from quad_flip_env import (                                          # noqa: E402
    QuadFlipEnv, ACTOR_TOTAL_DIM, REF_FF_DIM, REWARD_CEILING_PER_STEP,
    TRACK_W_POS, TRACK_W_VEL, TRACK_W_ATT, TRACK_W_RATE, EPISODE_SECONDS,
)
from trajectories import TrajectoryConfig                            # noqa: E402
from actor_input import load_checkpoint, read_checkpoint_arch        # noqa: E402

try:
    from encoder.latent_injector import LatentInjector
except ImportError:                                                  # pragma: no cover
    LatentInjector = None

Z_DIM = 16
Z_OFF = ACTOR_TOTAL_DIM                       # 29: the latent sits right after o_t
TERM_NAMES = ("pos", "vel", "att", "rate")
TERM_W = {"pos": TRACK_W_POS, "vel": TRACK_W_VEL, "att": TRACK_W_ATT, "rate": TRACK_W_RATE}


# ======================================================================================
# model / injector
# ======================================================================================
def pick_model(argv):
    if argv:
        return argv
    import re
    logs = os.path.join(_ROOT, "logs")
    cand = [os.path.join(logs, f) for f in os.listdir(logs) if f.endswith(".zip")]
    if not cand:
        raise SystemExit("no checkpoints in logs/")
    stepped = [p for p in cand if re.search(r"(\d+)_steps", os.path.basename(p))]
    if stepped:
        return max(stepped, key=lambda p: int(re.search(r"(\d+)_steps", p).group(1)))
    return max(cand, key=os.path.getmtime)


class ZeroLatent:
    """Same injector, latent forced to zero: the encoder-ablation arm of [7].

    The GRU still runs (same cost, same call pattern) - only the 16 numbers the policy
    receives from it are zeroed. That isolates "does the policy USE z" from "is the policy
    fast enough", which zeroing the whole encoder would not.
    """

    def __init__(self, inner, z_off=Z_OFF, z_dim=Z_DIM):
        self.inner, self.z_off, self.z_dim = inner, z_off, z_dim

    def reset(self):
        self.inner.reset()

    def inject(self, obs):
        v = np.array(self.inner.inject(obs), dtype=np.float32, copy=True)
        v[self.z_off:self.z_off + self.z_dim] = 0.0
        return v


# ======================================================================================
# the sweep - a faithful copy of train.py::_sweep
# ======================================================================================
def sweep(env, model, injector, dr, families, episodes, want_terms=False, want_behavior=False):
    """One deterministic pass. Returns (per_family, steps, ret, terms, behaviour)."""
    env.set_dr_level(dr)
    rec: list = []
    orig = None
    if want_terms:
        orig = qfe._tracking_kernel

        def _spy(err, tol):
            v = orig(err, tol)
            rec.append((float(err), float(tol), float(v)))
            return v

        qfe._tracking_kernel = _spy

    per_family, reasons, behavior = {}, {}, {}
    total_ret, total_steps = 0.0, 0
    err_sum, kern_sum, n_terms = [0.0] * 4, [0.0] * 4, 0
    try:
        for kind in families:
            fam_ret, fam_steps = 0.0, 0
            fam_reasons: dict = {}
            fam_perr, fam_inv, fam_tilt, fam_n = [], [], [], 0
            for k in range(episodes):
                obs, _ = env.reset(seed=k, options={"maneuver": kind})
                if injector is not None:
                    injector.reset()
                fam_n += 1
                perr_ep, inv_ep, tilt_ep = [], False, 0.0
                done = False
                while not done:
                    o = injector.inject(obs) if injector is not None else obs
                    action, _ = model.predict(o, deterministic=True)
                    rec.clear()
                    obs, reward, term, trunc, info = env.step(action)
                    if want_terms and len(rec) == 4:
                        for i in range(4):
                            err_sum[i] += rec[i][0]
                            kern_sum[i] += rec[i][2]
                        n_terms += 1
                    fam_ret += float(reward)
                    fam_steps += 1
                    if env.ref is not None:
                        perr_ep.append(float(np.linalg.norm(env.quad.pos - env.ref.p)))
                    d22 = float(env.quad.dcm[2, 2])
                    tilt_ep = max(tilt_ep, float(np.degrees(np.arccos(np.clip(d22, -1, 1)))))
                    if d22 < -0.5:
                        inv_ep = True
                    done = bool(term or trunc)
                # Read the ATTRIBUTE, not info["termination_reason"]: the env only fills
                # `info` when telemetry=True (quad_flip_env.py: `if telemetry: ... else:
                # info = {}`), and this sweep deliberately runs the training callback's
                # telemetry=False fast path. The attribute is maintained either way, so
                # reading it here is the only way to get a truthful crash census without
                # paying for the telemetry dict on every one of ~70k steps.
                reason = (getattr(env, "termination_reason", "?") if term else "completed")
                fam_reasons[reason] = fam_reasons.get(reason, 0) + 1
                fam_perr.append(float(np.mean(perr_ep)) if perr_ep else 0.0)
                fam_inv.append(inv_ep)
                fam_tilt.append(tilt_ep)
            per_family[kind] = fam_ret / max(1, fam_steps)
            reasons[kind] = fam_reasons
            behavior[kind] = {
                "mean_perr": float(np.mean(fam_perr)),
                "inverted": int(np.sum(fam_inv)),
                "n": fam_n,
                "peak_tilt": float(np.max(fam_tilt)),
            }
            total_ret += fam_ret
            total_steps += fam_steps
    finally:
        if orig is not None:
            qfe._tracking_kernel = orig

    terms = None
    if want_terms and n_terms:
        terms = {"n": n_terms,
                 "err": [e / n_terms for e in err_sum],
                 "kern": [k / n_terms for k in kern_sum]}
    return per_family, total_steps, total_ret, terms, (reasons, behavior)


def pct(v):
    return 100.0 * v / REWARD_CEILING_PER_STEP


def bar(v):
    """A crude 0..100 visual, so a table is readable at a glance."""
    n = int(round(max(0.0, min(100.0, v)) / 5.0))
    return "#" * n + "." * (20 - n)


# ======================================================================================
def main() -> int:
    model_path = pick_model(sys.argv[1] if len(sys.argv) > 1 else None)
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    families = list(TrajectoryConfig().weights)
    actor_dim, _ = read_checkpoint_arch(model_path)
    model = load_checkpoint(model_path)
    enc = os.path.join(_ROOT, "logs", "encoder_gru.pt")
    injector = (LatentInjector(enc, z_dim=Z_DIM, ref_ff_dim=REF_FF_DIM)
                if (LatentInjector and os.path.isfile(enc)) else None)

    print("=" * 100)
    print("FINAL POLICY TEST CAMPAIGN")
    print(f"  model    : {os.path.relpath(model_path, _ROOT)}   actor {actor_dim} dims")
    print(f"  encoder  : {'ON' if injector is not None else 'OFF (no encoder.pt!)'}"
          f"   latent z={Z_DIM} at obs[{Z_OFF}:{Z_OFF + Z_DIM}]")
    print(f"  protocol : train.py::_sweep - {episodes} episodes/family, seeds 0..{episodes-1}, "
          f"deterministic, {EPISODE_SECONDS:.0f} s, ceiling {REWARD_CEILING_PER_STEP:.1f}")
    print(f"  families : {', '.join(families)}")
    print("=" * 100)

    env = QuadFlipEnv(episode_seconds=EPISODE_SECONDS, telemetry=False)
    results = {}
    t_all = time.time()

    # -- [1] nominal ------------------------------------------------------------------
    print("\n[1] PER-FAMILY, nominal plant (dr=0)  - the headline number")
    t0 = time.time()
    fam0, st0, ret0, _, (rz0, bh0) = sweep(env, model, injector, 0.0, families, episodes,
                                           want_behavior=True)
    ov0 = ret0 / max(1, st0)
    print(f"    {'family':>10} {'reward/step':>12} {'% of 7.3':>9}  {'':20} "
          f"{'mean|p_err|':>11} {'inverted':>9} {'peak tilt':>10}")
    for k in sorted(fam0, key=lambda x: fam0[x]):
        b = bh0[k]
        print(f"    {k:>10} {fam0[k]:>12.3f} {pct(fam0[k]):>8.1f}%  {bar(pct(fam0[k]))} "
              f"{b['mean_perr']:>11.3f} {b['inverted']:>4}/{b['n']:<4} "
              f"{b['peak_tilt']:>9.0f}d")
    print(f"    {'OVERALL':>10} {ov0:>12.3f} {pct(ov0):>8.1f}%  {bar(pct(ov0))} "
          f"   ({st0} steps, {time.time() - t0:.0f}s)")
    results["nominal"] = ov0

    # -- [2] robust -------------------------------------------------------------------
    print("\n[2] PER-FAMILY, full domain randomisation (dr=1)  - the sim-to-real number")
    t0 = time.time()
    fam1, st1, ret1, _, (rz1, bh1) = sweep(env, model, injector, 1.0, families, episodes,
                                           want_behavior=True)
    ov1 = ret1 / max(1, st1)
    print(f"    {'family':>10} {'dr=0':>9} {'dr=1':>9} {'cost':>8}  {'':20}")
    for k in sorted(fam0, key=lambda x: fam0[x] - fam1[x]):
        d = pct(fam1[k]) - pct(fam0[k])
        print(f"    {k:>10} {pct(fam0[k]):>8.1f}% {pct(fam1[k]):>8.1f}% {d:>+7.1f}  "
              f"{bar(pct(fam1[k]))}")
    print(f"    {'OVERALL':>10} {pct(ov0):>8.1f}% {pct(ov1):>8.1f}% {pct(ov1) - pct(ov0):>+7.1f}  "
          f"{bar(pct(ov1))}   ({st1} steps, {time.time() - t0:.0f}s)")
    results["robust"] = ov1

    # -- [3] reward decomposition ------------------------------------------------------
    print("\n[3] WHERE THE REWARD GOES  (each term as a fraction of its own weight)")
    for label, dr in (("dr=0", 0.0), ("dr=1", 1.0)):
        _, _, _, terms, _ = sweep(env, model, injector, dr, families, episodes,
                                  want_terms=True)
        if terms:
            parts = "  ".join(f"{n}={TERM_W[n] * terms['kern'][i]:.2f}/{TERM_W[n]:.1f}"
                              for i, n in enumerate(TERM_NAMES))
            errs = "  ".join(f"{n}_err={terms['err'][i]:.3f}"
                             for i, n in enumerate(TERM_NAMES))
            print(f"    {label}: {parts}")
            print(f"           {errs}")

    # -- [4] termination census --------------------------------------------------------
    print("\n[4] HOW EPISODES END  (per family, dr=0 | dr=1)")
    print(f"    {'family':>10}  {'dr=0':<40} {'dr=1':<40}")
    for k in families:
        a = "  ".join(f"{r}:{c}" for r, c in sorted(rz0[k].items()))
        b = "  ".join(f"{r}:{c}" for r, c in sorted(rz1[k].items()))
        print(f"    {k:>10}  {a:<40} {b:<40}")
    ok0 = sum(c for k in rz0 for rr, c in rz0[k].items() if rr == "completed")
    ok1 = sum(c for k in rz1 for rr, c in rz1[k].items() if rr == "completed")
    tot = len(families) * episodes
    print(f"    {'SURVIVED':>10}  {ok0}/{tot} episodes ran to the horizon"
          f"   |  {ok1}/{tot} at dr=1")

    # -- [5] DR sweep ------------------------------------------------------------------
    print("\n[5] DR SWEEP  - what each level of randomization costs (3 episodes/family)")
    print(f"    {'dr':>5} {'reward/step':>12} {'% of 7.3':>9}  {'':20}")
    for dr in (0.0, 0.25, 0.5, 0.75, 1.0):
        f, s, r, _, _ = sweep(env, model, injector, dr, families, 3)
        o = r / max(1, s)
        print(f"    {dr:>5.2f} {o:>12.3f} {pct(o):>8.1f}%  {bar(pct(o))}")

    # -- [6] flip execution ------------------------------------------------------------
    print("\n[6] FLIP EXECUTION  - behaviour, not reward (the flip reward column is "
          "known to be uninformative)")
    print(f"    {'dr':>5} {'seed':>5} {'ended':>10} {'reason':>14} {'t/dur':>7} "
          f"{'REF dcm22':>10} | {'VEH dcm22':>10} {'inverted':>9} {'peak tilt':>10} "
          f"{'alt drop':>9}")
    n_inv = n_tot = 0
    for dr in (0.0, 1.0):
        for seed in (0, 1, 2, 3, 4):
            e = QuadFlipEnv(episode_seconds=EPISODE_SECONDS, maneuver="flip")
            e.set_dr_level(dr)
            obs, _ = e.reset(seed=seed, options={"maneuver": "flip"})
            if injector is not None:
                injector.reset()
            span = float(e.traj.duration) if e.traj is not None else float("nan")
            z0 = float(e.quad.pos[2])
            min_d22, ref_min, peak_tilt, zmin = 1.0, 1.0, 0.0, z0
            while True:
                o = injector.inject(obs) if injector is not None else obs
                act, _ = model.predict(o, deterministic=True)
                obs, _r, term, trunc, info = e.step(act)
                d22 = float(e.quad.dcm[2, 2])
                min_d22 = min(min_d22, d22)
                peak_tilt = max(peak_tilt, float(np.degrees(np.arccos(np.clip(d22, -1, 1)))))
                zmin = min(zmin, float(e.quad.pos[2]))
                if e.ref is not None:
                    ref_min = min(ref_min, float(e.ref.R[2, 2]))
                if term or trunc:
                    break
            inv = min_d22 < -0.5
            n_inv += int(inv)
            n_tot += 1
            reason = info.get("termination_reason", "?") if term else "-"
            print(f"    {dr:>5.1f} {seed:>5} {'term' if term else 'trunc':>10} "
                  f"{str(reason):>14} {e.t / span:>7.2f} {ref_min:>+10.3f} | "
                  f"{min_d22:>+10.3f} {'YES' if inv else 'no':>9} {peak_tilt:>9.0f}d "
                  f"{zmin - z0:>+9.2f}")
            e.close()
    print(f"    INVERTED: {n_inv}/{n_tot} episodes  ({100.0 * n_inv / max(1, n_tot):.0f}%)")

    # -- [7] encoder ablation ----------------------------------------------------------
    if injector is not None:
        print("\n[7] ENCODER ABLATION  - same rollout with the latent forced to zero")
        _, st, ret, _, _ = sweep(env, model, injector, 0.0, families, 3)
        real = pct(ret / max(1, st))
        _, st, ret, _, _ = sweep(env, model, ZeroLatent(injector), 0.0, families, 3)
        zero = pct(ret / max(1, st))
        print(f"    latent real : {real:>6.1f}%")
        print(f"    latent ZERO : {zero:>6.1f}%   ({zero - real:+.1f} points)")
        print("    (a large drop means the policy genuinely depends on the encoder's history;"
              "\n     ~0 means the latent is decorative and the encoder can be dropped)")

    # -- [8] lighthouse failure --------------------------------------------------------
    print("\n[8] LIGHTHOUSE FAILURE  - estimator lies / goes blind mid-episode (3 seeds each)")
    try:
        from evaluate import LighthouseFailure
        print(f"    {'mode':>9} {'reward/step':>12} {'% of 7.3':>9} {'vs none':>9} "
              f"{'survived':>10} {'peak lie':>9}")
        base = None
        for mode in ("none", "loss", "outage", "runaway", "teleport"):
            f, s, r, _, (rz, _bh) = sweep_failure(env, model, injector, mode, LighthouseFailure)
            o = r / max(1, s)
            if base is None:
                base = pct(o)
            ok = sum(c for k in rz for rr, c in rz[k].items() if rr == "completed")
            tot = len(families) * 3
            print(f"    {mode:>9} {o:>12.3f} {pct(o):>8.1f}% {pct(o) - base:>+8.1f} "
                  f"{ok:>5}/{tot:<4} {LighthouseFailure.LAST_MAX_LIE:>9.2f}")
    except Exception as exc:  # noqa: BLE001
        print(f"    skipped: {type(exc).__name__}: {exc}")

    # -- [9] repeatability -------------------------------------------------------------
    print("\n[9] REPEATABILITY  - the same sweep twice; a deterministic eval must be identical")
    _, _, r_a, _, _ = sweep(env, model, injector, 0.0, families, 3)
    _, _, r_b, _, _ = sweep(env, model, injector, 0.0, families, 3)
    print(f"    run A {r_a:.6f}   run B {r_b:.6f}   delta {abs(r_a - r_b):.3e}"
          f"   {'IDENTICAL' if abs(r_a - r_b) < 1e-9 else 'NOT DETERMINISTIC'}")

    env.close()
    print("\n" + "=" * 100)
    print(f"DONE in {time.time() - t_all:.0f}s   |   nominal {pct(results['nominal']):.1f}%  "
          f"robust {pct(results['robust']):.1f}%")
    print("=" * 100)
    return 0


def sweep_failure(env, model, injector, mode, FailureCls):
    """A mixture sweep with a lighthouse failure injected mid-episode."""
    families = list(TrajectoryConfig().weights)
    per_family, reasons = {}, {}
    total_ret, total_steps, max_lie = 0.0, 0, 0.0
    for kind in families:
        fam_ret, fam_steps, fam_reasons = 0.0, 0, {}
        for k in range(3):
            # Save the TRUE observe before patching. install() closes over it privately and
            # exposes no way back, so a second install() on the same env would raise the
            # double-install guard - this is the restore path, and restoring is mandatory
            # on EVERY iteration or the next episode's injection is silently a no-op.
            pristine = env.lighthouse.observe
            inj = FailureCls(env.lighthouse, mode=mode, at_s=1.0, severity=1.0,
                             outage_s=0.5, period_s=1.0, teleport_m=6.0, z_too=True)
            inj.install()
            obs, _ = env.reset(seed=k, options={"maneuver": kind})
            inj.reset()
            if injector is not None:
                injector.reset()
            while True:
                o = injector.inject(obs) if injector is not None else obs
                action, _ = model.predict(o, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                total_ret += float(reward)
                fam_ret += float(reward)
                fam_steps += 1
                max_lie = max(max_lie, float(inj.max_lie))
                if term or trunc:
                    break
            reason = info.get("termination_reason", "?") if term else "completed"
            fam_reasons[reason] = fam_reasons.get(reason, 0) + 1
            env.lighthouse.observe = pristine
            env.lighthouse.cfg.min_stations_for_fix = inj._orig_min
        per_family[kind] = fam_ret / max(1, fam_steps)
        reasons[kind] = fam_reasons
        total_steps += fam_steps
    FailureCls.LAST_MAX_LIE = max_lie
    return per_family, total_steps, total_ret, None, (reasons, {})


if __name__ == "__main__":
    raise SystemExit(main())
