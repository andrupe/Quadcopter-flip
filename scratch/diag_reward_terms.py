"""Per-term reward decomposition for a trained policy, evaluated at dr=0 and dr=1.

WHY THIS EXISTS
---------------
The deterministic eval callback reports ONE number (reward/step = x% of 7.30).
That single number cannot distinguish:
  * a POSITION-tracking deficit (weight 3.0, the largest term), from
  * an ATTITUDE (2.0) or RATE (0.8) one, or
  * the nominal cost of domain randomization (does dr=1 actually cost much?).
This script answers all three by instrumenting `_tracking_kernel`, the single
function every tracking term funnels through, and reporting the raw error next
to its tolerance so the number can be read against what a controller can hold.

The eval callback also only ever measures dr=0, so the robustness that the ADR
ramp BUYS is currently invisible; only its cost is visible.

Run:
    .venv/bin/python scratch/diag_reward_terms.py [model.zip] [dr,dr,...]
Example:
    .venv/bin/python scratch/diag_reward_terms.py logs/rl_model_14000000_steps.zip 0 1
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Simulation"))

import quad_flip_env as qfe                                    # noqa: E402
from quad_flip_env import QuadFlipEnv, TRACK_TOL               # noqa: E402
from actor_input import ActorInput, load_checkpoint, read_checkpoint_arch   # noqa: E402

FAMILIES = ["hover", "waypoints", "figure8", "lissajous", "orbit",
            "slalom", "flip", "v8", "chain"]
TERMS = ("pos", "vel", "att", "rate")
WEIGHTS = {
    "pos": qfe.TRACK_W_POS,
    "vel": qfe.TRACK_W_VEL,
    "att": qfe.TRACK_W_ATT,
    "rate": qfe.TRACK_W_RATE,
}
W_ACT = qfe.W_ACTION_SMOOTH

# --- instrumentation -------------------------------------------------------------
# `_compute_reward` calls _tracking_kernel exactly ONCE per term, in the order
# pos, vel, att, rate. Nothing else in the module calls it (verified by grep), so
# chunks of four align with one control step. We clear the record immediately
# before env.step() and discard any step whose chunk is not exactly four long,
# so a stray call could never silently mislabel a term.
_rec = []
_orig_kernel = qfe._tracking_kernel


def _spy(err, tol):
    v = _orig_kernel(err, tol)
    _rec.append((float(err), float(tol), float(v)))
    return v


qfe._tracking_kernel = _spy


def run_episode(model, actor_input, dr, family, seed: int = 0):
    """
    One deterministic episode of `family`; returns the per-step records.

    SEEDED on purpose. An unseeded reset draws a different reference for every `dr` level,
    and the reference-to-reference spread of this metric is several points - which is larger
    than the whole DR effect being measured, so an unseeded sweep reports draw luck as a
    trend. With a fixed seed every level flies the SAME references and only the plant changes.
    """
    env = QuadFlipEnv(episode_seconds=15.0, maneuver=family)
    env.set_dr_level(dr)
    obs, _info = env.reset(seed=seed, options={"maneuver": family})
    actor_input.reset()

    step_rows, n_bad = [], 0
    for _ in range(env.max_steps):
        x = actor_input.prepare(obs)
        action, _ = model.predict(x, deterministic=True)
        _rec.clear()
        obs, rew, terminated, truncated, _info = env.step(action)

        if len(_rec) != 4:
            n_bad += 1
        else:
            step_rows.append(([_rec[i][0] for i in range(4)],
                              [_rec[i][1] for i in range(4)],
                              [_rec[i][2] for i in range(4)],
                              float(rew)))
        if terminated or truncated:
            break

    if n_bad:
        print(f"    [warn] {family}: {n_bad} step(s) had != 4 kernel calls, skipped")
    return step_rows


def summarise(rows):
    errs = np.array([r[0] for r in rows])          # (T, 4)
    tols = np.array([r[1] for r in rows])          # (T, 4)
    kern = np.array([r[2] for r in rows])          # (T, 4)
    rew = np.array([r[3] for r in rows])           # (T,)
    out = {"steps": len(rows), "reward": float(rew.mean()) if len(rew) else np.nan}
    out["reward_kernel_only"] = float(
        sum(WEIGHTS[t] * kern[:, i].mean() for i, t in enumerate(TERMS)))
    # The reward is the sum of exactly five terms, so the action-smoothness term is
    # obtained EXACTLY as the residual - no need to re-derive its kernel formula here.
    out["action"] = out["reward"] - out["reward_kernel_only"]
    for i, t in enumerate(TERMS):
        out[f"err_{t}"] = float(errs[:, i].mean())
        out[f"tol_{t}"] = float(tols[:, i].mean())
        out[f"r_{t}"] = float(kern[:, i].mean())
        out[f"c_{t}"] = float(WEIGHTS[t] * kern[:, i].mean())
    return out


def print_table(title, table):
    print(f"\n{title}")
    print(f"  ceiling = {qfe.REWARD_CEILING_PER_STEP:.2f}/step "
          f"(pos {WEIGHTS['pos']} | vel {WEIGHTS['vel']} | att {WEIGHTS['att']} "
          f"| rate {WEIGHTS['rate']} | action {W_ACT})")
    head = (f"  {'family':>10} {'steps':>6} {'p_err':>7} {'v_err':>7} {'a_err':>7} "
            f"{'w_err':>7} | {'pos':>5} {'vel':>5} {'att':>5} {'rate':>5} {'rest':>5} "
            f"| {'total':>6} {'%':>5}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    agg = []

    def line(name, s):
        pct = 100.0 * s["reward"] / qfe.REWARD_CEILING_PER_STEP
        print(f"  {name:>10} {s['steps']:>6} {s['err_pos']:>7.3f} {s['err_vel']:>7.3f} "
              f"{s['err_att']:>7.3f} {s['err_rate']:>7.3f} | "
              f"{s['c_pos']:>5.2f} {s['c_vel']:>5.2f} {s['c_att']:>5.2f} "
              f"{s['c_rate']:>5.2f} {s['action']:>5.2f} | {s['reward']:>6.2f} {pct:>4.1f}%")
        agg.append(s)

    for fam, s in table.items():
        line(fam, s)
    if agg:
        mean = {k: float(np.mean([s[k] for s in agg]))
                for k in agg[0] if k != "steps"}
        mean["steps"] = int(np.mean([s["steps"] for s in agg]))
        print("  " + "-" * (len(head) - 2))
        line("MEAN", mean)
    print(f"\n  tolerances: "
          + " | ".join(f"{f} pos={TRACK_TOL[f]['pos']:.2f} vel={TRACK_TOL[f]['vel']:.2f} "
                       f"att={TRACK_TOL[f]['att']:.2f} rate={TRACK_TOL[f]['rate']:.2f}"
                       for f in ("hover", "orbit", "flip")))


def main():
    argv = sys.argv[1:]
    model_name = argv[0] if argv else "latest"
    dr_levels = [float(x) for x in argv[1:]] or [0.0, 1.0]

    model_path = model_name
    if model_name.lower() in ("latest", "auto") or not os.path.isfile(model_name):
        logs = os.path.join(_ROOT, "logs")
        zips = [os.path.join(logs, f) for f in os.listdir(logs) if f.endswith(".zip")]
        if not zips:
            raise SystemExit("no checkpoint found in logs/")
        import re

        def key(p):
            m = re.search(r"(\d+)_steps", os.path.basename(p))
            return int(m.group(1)) if m else os.path.getmtime(p)

        model_path = sorted(zips, key=key)[-1]

    print(f"model  : {os.path.relpath(model_path, _ROOT)}")
    dim, _ = read_checkpoint_arch(model_path)
    if dim is None:
        raise SystemExit("could not read the actor width from the checkpoint")
    actor_input = ActorInput(dim, encoder_path=os.path.join(_ROOT, "logs", "encoder_gru.pt"))
    model = load_checkpoint(model_path)
    print(f"actor  : {dim} dims | {actor_input.describe()}")

    for dr in dr_levels:
        table = {}
        for fam in FAMILIES:
            table[fam] = summarise(run_episode(model, actor_input, dr, fam))
        overall = float(np.mean([s["reward"] for s in table.values()]))
        print_table(f"=== dr = {dr:.2f}  (overall {overall:.3f}/step = "
                    f"{100*overall/qfe.REWARD_CEILING_PER_STEP:.2f}% of ceiling) ===",
                    table)


if __name__ == "__main__":
    main()
