"""Per-term reward decomposition: SCRIPTED geometric controller vs the trained POLICY.

WHY: the policy sits at ~50% of the 7.30 ceiling while a hand-written controller reaches
78-95%. The aggregate does not say WHETHER that gap is position tracking (weight 3.0),
attitude (2.0), or something the reference makes unreachable. This runs both controllers
on the SAME seeds - so the same reference draws - and reports the same decomposition.

The scripted controller is read OUT OF check_env_tracking.py by AST rather than re-typed:
importing that module would execute the whole validation suite, and copying `track_action`
would let the two drift apart.

Run:  .venv/bin/python scratch/diag_scripted_vs_policy.py [model.zip] [dr]
"""
import ast
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Simulation"))
sys.path.insert(0, os.path.join(_ROOT, "scratch"))

import quad_flip_env as qfe                                      # noqa: E402
from quad_flip_env import QuadFlipEnv                            # noqa: E402
from actor_input import ActorInput, load_checkpoint, read_checkpoint_arch   # noqa: E402

FAMILIES = ["hover", "waypoints", "figure8", "lissajous", "orbit",
            "slalom", "flip", "v8", "chain"]
TERM_NAMES = ("pos", "vel", "att", "rate")
WEIGHTS = {"pos": qfe.TRACK_W_POS, "vel": qfe.TRACK_W_VEL,
           "att": qfe.TRACK_W_ATT, "rate": qfe.TRACK_W_RATE}
# Recorded 6-seed means from check_env_tracking section B, for a sanity check on this
# harness: if the scripted column does not land near these, the harness is wrong.
RECORDED = {"hover": 78, "waypoints": 87, "figure8": 81, "orbit": 86, "lissajous": 83,
            "slalom": 92, "flip": 90, "v8": 95, "chain": 81}


def load_scripted_controller():
    """Exec only the imports + function defs of check_env_tracking.py (no suite side effects)."""
    src = open(os.path.join(_ROOT, "scratch", "check_env_tracking.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns: dict = {"__name__": "check_env_tracking_extract"}
    exec(compile(ast.Module(keep, type_ignores=[]), "check_env_tracking.py", "exec"), ns)
    # Anything the defs reference that the filtered AST did not carry in.
    for name in ("np", "GRAVITY", "MASS_NOMINAL", "dcm_from_thrust_dir_and_yaw", "QuadFlipEnv"):
        if name not in ns:
            for mod in (qfe,):
                if hasattr(mod, name):
                    ns[name] = getattr(mod, name)
    missing = [n for n in ("track_action", "np", "GRAVITY", "MASS_NOMINAL",
                           "dcm_from_thrust_dir_and_yaw") if n not in ns]
    if missing:
        raise SystemExit(f"could not extract from check_env_tracking.py: {missing}")
    return ns


_rec = []
_orig_kernel = qfe._tracking_kernel


def _spy(err, tol):
    v = _orig_kernel(err, tol)
    _rec.append((float(err), float(tol), float(v)))
    return v


qfe._tracking_kernel = _spy


def _attitude_split(env):
    """
    Split the attitude error into TILT and HEADING.

    `_attitude_error_rotvec` returns the rotation vector of R_ref^T R in the CURRENT BODY
    frame, so its z component is NOT the heading error and must not be read as one. The
    physically meaningful split is on the axes:
      * tilt   = angle between the two thrust axes (body z). This one is what position
                 tracking has to fight, so a large tilt error cannot coexist with good
                 position tracking.
      * heading = difference of the two yaw angles, i.e. how far off the commanded
                 heading is. Invisible to the position term.
    """
    R_ref = np.asarray(env.ref.R, dtype=float)
    R = np.asarray(env.quad.dcm, dtype=float)
    z_ref, z = R_ref[:, 2], R[:, 2]
    tilt = float(np.arccos(np.clip(float(np.dot(z_ref, z)), -1.0, 1.0)))
    yaw_ref = float(np.arctan2(R_ref[1, 0], R_ref[0, 0]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    heading = float((yaw_ref - yaw + np.pi) % (2.0 * np.pi) - np.pi)
    # SIGNED heading error is what the lag/offset test needs: sign(ref - actual) > 0 while
    # the reference is turning means the vehicle is BEHIND it.
    return tilt, abs(heading), heading


def run(controller, family, seed, dr, model=None, actor_input=None):
    """One episode; `controller(env) -> action`."""
    env = QuadFlipEnv(episode_seconds=15.0, maneuver=family)
    env.set_dr_level(dr)
    obs, _info = env.reset(seed=seed, options={"maneuver": family})
    if actor_input is not None:
        actor_input.reset()

    errs, kerns, rets, n = [], [], [], 0
    tilts, heads, yaw_rows, att_tols = [], [], [], []
    # The reference kind per step, so the tilt-only projection below can exclude flips -
    # their reward no longer keys on the rotvec error at all. See the projection's note.
    kinds: list = []
    for _ in range(env.max_steps):
        if model is not None:
            action = model.predict(actor_input.prepare(obs), deterministic=True)[0]
        else:
            action = controller(env)
        _rec.clear()
        obs, rew, term, trunc, _info = env.step(action)
        if len(_rec) == 4:
            errs.append([_rec[i][0] for i in range(4)])
            kerns.append([_rec[i][2] for i in range(4)])
            rets.append(float(rew))
            tl, hd, hd_signed = _attitude_split(env)
            tilts.append(tl)
            heads.append(hd)
            # The tolerance the reward ACTUALLY used for the attitude term this step. The
            # spy captured it, so a chain (whose tolerance changes per segment via
            # `kind_at`) is handled exactly rather than approximated by the family entry.
            att_tols.append(_rec[2][1])
            kinds.append(str(getattr(env.ref, "kind", "?")))
            yaw_rows.append((hd_signed,
                             float(env.ref.omega[2]),
                             float(action[3]) * float(env.max_rate_z)))
            n += 1
        if term or trunc:
            break

    if n == 0:
        return None
    kern = np.array(kerns).mean(axis=0)
    return {
        "n": n,
        "reward": float(np.mean(rets)),
        "kern": kern,
        "err": np.array(errs).mean(axis=0),
        "contrib": np.array([WEIGHTS[t] * kern[i] for i, t in enumerate(TERM_NAMES)]),
        "tilt": float(np.mean(tilts)),
        "head": float(np.mean(heads)),
        "tilts": np.asarray(tilts, dtype=float),
        "att_tol": np.asarray(att_tols, dtype=float),
        "kern_att": np.asarray([k[2] for k in kerns], dtype=float),
        "kinds": np.asarray(kinds, dtype=object),
        "yaw": np.array(yaw_rows) if yaw_rows else np.zeros((0, 3)),
    }


def main():
    argv = sys.argv[1:]
    model_name = argv[0] if argv else "latest"
    dr = float(argv[1]) if len(argv) > 1 else 0.0

    if model_name.lower() in ("latest", "auto") or not os.path.isfile(model_name):
        import re
        logs = os.path.join(_ROOT, "logs")
        zips = [os.path.join(logs, f) for f in os.listdir(logs) if f.endswith(".zip")]
        zips.sort(key=lambda p: int(re.search(r"(\d+)_steps", p).group(1))
                  if re.search(r"(\d+)_steps", p) else os.path.getmtime(p))
        model_name = zips[-1]

    print(f"model    : {os.path.relpath(model_name, _ROOT)}   dr={dr:.2f}")
    dim, _ = read_checkpoint_arch(model_name)
    actor_input = ActorInput(dim, encoder_path=os.path.join(_ROOT, "logs", "encoder_gru.pt"))
    model = load_checkpoint(model_name)

    sc = load_scripted_controller()
    track_action = sc["track_action"]

    pol_rows, scr_rows = {}, {}
    for fam in FAMILIES:
        for seed in (0,):
            pol_rows[fam] = run(None, fam, seed, dr, model=model, actor_input=actor_input)
            scr_rows[fam] = run(track_action, fam, seed, dr)

    def pct(s):
        return 100.0 * s["reward"] / qfe.REWARD_CEILING_PER_STEP

    head = (f"  {'family':>10} | {'policy%':>7} {'script%':>7} {'recorded':>8} {'gap':>6} "
            f"| {'pol: pos vel att rate':>23} | {'scr: pos vel att rate':>23}")
    print("\n=== SCRIPTED vs POLICY, per term (dr=%.2f, 1 seed/family) ===" % dr)
    print(head)
    print("  " + "-" * (len(head) - 2))
    gaps, pol_sum, scr_sum = [], [], []
    for fam in FAMILIES:
        p, s = pol_rows[fam], scr_rows[fam]
        if p is None or s is None:
            continue
        gap = pct(p) - pct(s)
        gaps.append(gap)
        pol_sum.append(p["contrib"])
        scr_sum.append(s["contrib"])
        pc = " ".join(f"{v:4.2f}" for v in p["contrib"])
        scc = " ".join(f"{v:4.2f}" for v in s["contrib"])
        print(f"  {fam:>10} | {pct(p):>6.1f}% {pct(s):>6.1f}% {RECORDED.get(fam, 0):>7}% "
              f"{gap:>+5.1f} | {pc:>23} | {scc:>23}")

    pm = np.mean(pol_sum, axis=0)
    sm = np.mean(scr_sum, axis=0)
    mean_pol_pct = float(np.mean([pct(p) for p in pol_rows.values() if p]))
    mean_scr_pct = float(np.mean([pct(s) for s in scr_rows.values() if s]))
    print("  " + "-" * (len(head) - 2))
    print(f"  {'MEAN':>10} | {mean_pol_pct:>6.1f}% {mean_scr_pct:>6.1f}% "
          f"{'':>8} {np.mean(gaps):>+5.1f} | "
          f"{' '.join(f'{v:4.2f}' for v in pm):>23} | {' '.join(f'{v:4.2f}' for v in sm):>23}")

    print("\n=== WHERE the gap comes from (mean weighted contribution, reward/step) ===")
    print(f"  {'term':>6} {'weight':>6} {'policy':>7} {'script':>7} {'gap':>7} "
          f"{'% of gap':>9}")
    total_gap = mean_pol_pct - mean_scr_pct
    total_gap_rs = total_gap / 100.0 * qfe.REWARD_CEILING_PER_STEP
    for i, t in enumerate(TERM_NAMES):
        d = sm[i] - pm[i]
        share = 100.0 * d / total_gap_rs if abs(total_gap_rs) > 1e-9 else float("nan")
        print(f"  {t:>6} {WEIGHTS[t]:>6.1f} {pm[i]:>7.2f} {sm[i]:>7.2f} {d:>+7.2f} "
              f"{share:>8.0f}%")
    act_p = np.mean([p["reward"] - p["contrib"].sum() for p in pol_rows.values()])
    act_s = np.mean([s["reward"] - s["contrib"].sum() for s in scr_rows.values()])
    d = act_s - act_p
    print(f"  {'action':>6} {0.5:>6.1f} {act_p:>7.2f} {act_s:>7.2f} {d:>+7.2f} "
          f"{100.0*d/total_gap_rs if abs(total_gap_rs)>1e-9 else float('nan'):>8.0f}%")

    # --- is the attitude error TILT or HEADING? ------------------------------------
    print("\n=== ATTITUDE ERROR: tilt (body-z angle) vs heading (yaw) ===")
    print("  The reward penalises the whole rotation vector, so a heading error costs as")
    print("  much as a tilt error - but only tilt can disturb position tracking.")
    print(f"  {'family':>10} {'att_tol':>8} | {'pol tilt':>8} {'pol head':>8} | "
          f"{'scr tilt':>8} {'scr head':>8}")
    print("  " + "-" * 62)
    pt, ph, st, sh = [], [], [], []
    for fam in FAMILIES:
        p, s = pol_rows[fam], scr_rows[fam]
        if p is None or s is None:
            continue
        tol = qfe.TRACK_TOL[fam]["att"]
        pt.append(p["tilt"])
        ph.append(p["head"])
        st.append(s["tilt"])
        sh.append(s["head"])
        print(f"  {fam:>10} {tol:>8.2f} | {p['tilt']:>8.3f} {p['head']:>8.3f} | "
              f"{s['tilt']:>8.3f} {s['head']:>8.3f}")
    print("  " + "-" * 62)
    print(f"  {'MEAN':>10} {'':>8} | {np.mean(pt):>8.3f} {np.mean(ph):>8.3f} | "
          f"{np.mean(st):>8.3f} {np.mean(sh):>8.3f}")
    tot_p, tot_s = np.mean(pt) + np.mean(ph), np.mean(st) + np.mean(sh)
    print(f"\n  policy: tilt {100*np.mean(pt)/tot_p:.0f}% of its attitude error, "
          f"heading {100*np.mean(ph)/tot_p:.0f}%")
    print(f"  scripted: tilt {100*np.mean(st)/tot_s:.0f}%, heading {100*np.mean(sh)/tot_s:.0f}%")

    # --- is the heading error a FEEDFORWARD LAG or a constant OFFSET? ---------------
    print("\n=== YAW: feedforward lag, or does the policy simply ignore heading? ===")
    print("  A proportional-only loop with no rate feedforward chasing a rotating reference")
    print("  settles at a LAG proportional to the rate; ignoring yaw gives a CONSTANT offset.")
    print("  `track_action` is exact because it adds `+ ref.omega`.\n")

    def yaw_stats(label, rows):
        rows = np.asarray(rows, dtype=float)
        if len(rows) < 8:
            print(f"  {label}: too few steps")
            return
        hs, rr, cmd = rows[:, 0], rows[:, 1], rows[:, 2]
        corr = (float(np.corrcoef(hs, rr)[0, 1])
                if np.std(rr) > 1e-9 and np.std(hs) > 1e-9 else float("nan"))
        b, a = np.polyfit(rr, hs, 1)
        pred = a + b * rr
        r2 = 1.0 - float(np.sum((hs - pred) ** 2)) / max(1e-12, float(np.sum((hs - hs.mean()) ** 2)))
        quiet, driven = np.abs(rr) < 0.10, np.abs(rr) > 0.50
        print(f"  {label}   ({len(rows)} steps)")
        print(f"    corr(heading_err, ref_yaw_rate) = {corr:+.3f}")
        print(f"    heading_err = {a:+.3f} + {b:+.3f} * ref_yaw_rate      (R^2 = {r2:.2f})")
        if quiet.sum() > 4:
            print(f"    |heading| with |ref_rate|<0.10  ({int(quiet.sum()):>5} steps): "
                  f"{np.abs(hs[quiet]).mean():.3f} rad  <- the OFFSET floor")
        if driven.sum() > 4:
            agree = float(np.mean(np.sign(cmd[driven]) == np.sign(rr[driven])))
            print(f"    |heading| with |ref_rate|>0.50  ({int(driven.sum()):>5} steps): "
                  f"{np.abs(hs[driven]).mean():.3f} rad  <- the DRIVEN part")
            print(f"    commanded yaw rate {cmd[driven].mean():+.3f} vs reference "
                  f"{rr[driven].mean():+.3f} rad/s   sign agrees on {100*agree:.0f}%")
            print(f"    feedforward gain needed to null the lag: ~{b:.2f} s")
        print()

    yaw_stats("POLICY",
              np.vstack([pol_rows[f]["yaw"] for f in FAMILIES if pol_rows[f] is not None]))
    yaw_stats("SCRIPTED",
              np.vstack([scr_rows[f]["yaw"] for f in FAMILIES if scr_rows[f] is not None]))

    # --- IS THE YAW CHANNEL ALIVE? -------------------------------------------------
    # SIGN CONVENTION, derived from track_action: `_attitude_error_rotvec(R_ref, R)` is the
    # rotvec of R_ref^T R, so its z component is e_R[2] = (yaw_actual - yaw_ref) = -hd.
    # `track_action` commands omega[2] = -kr*e_R[2] + ref.omega[2] = +kr*hd + ref_rate,
    # so a WORKING yaw regulator must show a clearly POSITIVE corr(hd, commanded_rate).
    # (An earlier version of this print said NEGATIVE - that was backwards.)
    print("=== IS THE YAW CHANNEL ALIVE?  corr(heading_err, commanded_yaw_rate) ===")
    print("  A working regulator has a clearly POSITIVE correlation (turn toward the error).")
    print("  ~0 means the yaw output does not depend on the heading error at all.\n")
    print(f"  {'family':>10} {'hd[0]':>7} {'hd[10]':>7} {'hd[-1]':>7} {'mean|hd|':>9} "
          f"| {'pol corr':>8} {'scr corr':>8}")
    print("  " + "-" * 66)
    for fam in FAMILIES:
        p, s = pol_rows[fam], scr_rows[fam]
        if p is None or s is None or len(p["yaw"]) < 12:
            continue
        hp, cp = p["yaw"][:, 0], p["yaw"][:, 2]
        hs_, cs = s["yaw"][:, 0], s["yaw"][:, 2]

        def cc(a, b):
            return (float(np.corrcoef(a, b)[0, 1])
                    if np.std(a) > 1e-9 and np.std(b) > 1e-9 else float("nan"))

        print(f"  {fam:>10} {hp[0]:>+7.3f} {hp[10]:>+7.3f} {hp[-1]:>+7.3f} "
              f"{np.abs(hp).mean():>9.3f} | {cc(hp, cp):>+8.2f} {cc(hs_, cs):>+8.2f}")

    # How much of the heading error is present at t=0 and still there at the end?
    # Episodes differ in length, so concatenate the 1-D traces (never vstack 2-D).
    allp = np.concatenate([pol_rows[f]["yaw"][:, 0] for f in FAMILIES
                           if pol_rows[f] is not None])
    alls = np.concatenate([scr_rows[f]["yaw"][:, 0] for f in FAMILIES
                           if scr_rows[f] is not None])
    print(f"\n  policy   mean|hd| {np.abs(allp).mean():.3f} rad   "
          f"(per-family hd[0] and hd[-1] are in the table above)")
    print(f"  scripted mean|hd| {np.abs(alls).mean():.3f} rad")

    # --- EXACT PROJECTION: what the score would be if heading tracked ---------------
    # The reward's attitude kernel is 1/(1+(rotvec_norm/tol)^2). My tilt is the angle
    # between the two body-z axes, i.e. the rotvec norm the SAME state would have if the
    # heading error were zero. Recomputing the kernel from tilt alone therefore gives the
    # score this very policy would earn with a working yaw loop - same weights, same
    # tolerances (the spy recorded the tolerance actually used each step), same episodes.
    #
    # FLIP STEPS ARE EXCLUDED, because that premise no longer holds for them: under
    # FLIP_PROGRESS the flip's attitude error is `max(rotvec_err, rotation_progress_err)`,
    # so the kernel the spy recorded is not a function of the rotvec at all and "tilt
    # only" would predict a gain that is not available. Excluded steps are charged ZERO
    # gain (the sum is divided by the full step count) rather than dropped from the
    # denominator, so a family that is part flip cannot be credited with a gain on the
    # steps the projection cannot speak to.
    print("\n=== PROJECTION: score with tilt-only attitude error (i.e. heading tracked) ===")
    print("    (flip steps excluded - see the note in the source; coverage shown per family)")
    print(f"  {'family':>10} {'now':>6} {'then':>6} {'gain':>6} | {'att now':>7} "
          f"{'att then':>8} {'head':>6} {'cov':>5}")
    print("  " + "-" * 69)
    nows, thens = [], []
    for fam in FAMILIES:
        p = pol_rows[fam]
        if p is None or len(p["tilts"]) == 0:
            continue
        kinds = p["kinds"]
        keep = kinds != "flip"
        cov = float(np.mean(keep))
        if not np.any(keep):
            print(f"  {fam:>10} {100*p['reward']/7.30:>5.1f}% {'n/a':>6} {'n/a':>6} | "
                  f"{WEIGHTS['att']*p['kern_att'].mean():>7.2f} {'n/a':>8} {p['head']:>6.3f} "
                  f"{cov:>5.2f}   (all steps are flips - not projectable)")
            continue
        hyp_all = 1.0 / (1.0 + (p["tilts"] / p["att_tol"]) ** 2)
        hyp = hyp_all[keep]
        kern = p["kern_att"][keep]
        # Zero gain on the excluded steps, so the denominator stays the full episode.
        gain = float(np.sum(WEIGHTS["att"] * (hyp - kern)) / len(kinds))
        now, then = p["reward"], p["reward"] + gain
        nows.append(now)
        thens.append(then)
        print(f"  {fam:>10} {100*now/7.30:>5.1f}% {100*then/7.30:>5.1f}% "
              f"{100*gain/7.30:>+5.1f} | {WEIGHTS['att']*kern.mean():>7.2f} "
              f"{WEIGHTS['att']*hyp.mean():>8.2f} {p['head']:>6.3f} {cov:>5.2f}")
    print("  " + "-" * 62)
    a = 100 * float(np.mean(nows)) / 7.30
    b = 100 * float(np.mean(thens)) / 7.30
    print(f"  {'MEAN':>10} {a:>5.1f}% {b:>5.1f}% {b-a:>+5.1f}")
    print(f"\n  => a working yaw loop alone is worth {b-a:+.1f} points "
          f"(gate is 75%, scripted controller is 88.5%)")


if __name__ == "__main__":
    main()
