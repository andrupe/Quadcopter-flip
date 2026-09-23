"""
Επιλογή του καλύτερου PPO checkpoint με βάση τον ΑΝΕΞΑΡΤΗΤΟ ΓΕΩΜΕΤΡΙΚΟ
ανιχνευτή επιτυχίας - όχι με βάση το reward.

Το έχουμε επιβεβαιώσει πειραματικά: το checkpoint με το υψηλότερο reward
μπορεί να είναι ποιοτικά χειρότερο (π.χ. περιστρέφεται ασταμάτητα και
μαζεύει bonus επιβίωσης). Το τελευταίο checkpoint επίσης δεν είναι
απαραίτητα το καλύτερο.

Χρήση:
    py -3.11 select_best_checkpoint.py --seeds 5
    py -3.11 select_best_checkpoint.py --seeds 5 --every 2 --save ppo_best
"""

import argparse
import contextlib
import io
import os
import pickle
import re
import shutil
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(_ROOT, "Simulation")
# Ο PID ελεγκτής ζει στο controllers/Pid - χρειάζεται στο path για το import.
_PID = os.path.join(_ROOT, "controllers", "Pid")
for _p in (_ROOT, _SIM, _PID):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.quad_flip_env import QuadFlipEnv
from algor import evaluate_flip_success


def list_checkpoints(logs_dir):
    out = []
    if not os.path.isdir(logs_dir):
        return out
    for f in os.listdir(logs_dir):
        m = re.match(r"rl_model_(\d+)_steps\.zip$", f)
        if m:
            steps = int(m.group(1))
            stats = os.path.join(logs_dir, f"rl_model_{steps}_steps_vecnormalize.pkl")
            out.append((steps, os.path.join(logs_dir, f),
                        stats if os.path.isfile(stats) else None))
    return sorted(out)


def evaluate(model, vn, env, seeds, deterministic):
    rows = []
    for s in seeds:
        obs, info = env.reset(seed=s)
        z0 = float(info["position"][2])
        pitch, zs, ws = [], [z0], []
        R = 0.0
        for t in range(env.max_steps):
            o = vn.normalize_obs(obs) if vn is not None else obs
            a, _ = model.predict(o, deterministic=deterministic)
            q = env.quad.quat
            pitch.append(float(np.arctan2(2 * (q[0] * q[2] + q[3] * q[1]),
                                          1 - 2 * (q[1] ** 2 + q[2] ** 2))))
            ws.append(env.quad.omega.copy())
            obs, r, term, trunc, info = env.step(a)
            R += r
            zs.append(float(info["position"][2]))
            if term or trunc:
                break
        with contextlib.redirect_stdout(io.StringIO()):
            geo = evaluate_flip_success(np.unwrap(pitch).tolist(), zs, ws)
        rows.append(dict(
            success=bool(geo and env.flip_completed and not term),
            survived=float(info["t"]),
            reward=R,
            att_err=float(np.degrees(np.arccos(np.clip(env.quad.dcm[2, 2], -1, 1)))),
            xy=float(np.linalg.norm(info["position"][:2])),
        ))
    n = len(rows)
    return dict(
        rate=sum(r["success"] for r in rows) / n,
        survived=float(np.mean([r["survived"] for r in rows])),
        reward=float(np.mean([r["reward"] for r in rows])),
        att_err=float(np.mean([r["att_err"] for r in rows])),
        xy=float(np.mean([r["xy"] for r in rows])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=str, default=os.path.join(_ROOT, "logs"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--every", type=int, default=1,
                    help="αξιολόγηση κάθε N-οστού checkpoint (για ταχύτητα)")
    ap.add_argument("--min-steps", type=int, default=0,
                    help="παράλειψη checkpoints κάτω από αυτό το βήμα")
    ap.add_argument("--dr", type=float, default=None,
                    help="επιλογή με βάση την απόδοση σε ΑΥΤΟ το dr_level αντί για "
                         "καθαρές συνθήκες. ΑΠΑΡΑΙΤΗΤΟ για μοντέλα ADR: τα πρώιμα "
                         "checkpoints είναι εκπαιδευμένα σχεδόν καθαρά, οπότε "
                         "κερδίζουν άδικα στο καθαρό τεστ")
    ap.add_argument("--vary", action="store_true",
                    help="τυχαία αρχική κατάσταση ώστε τα seeds να διαφέρουν πραγματικά. "
                         "Χωρίς αυτό κάθε seed δίνει ΤΟ ΙΔΙΟ επεισόδιο και το ποσοστό "
                         "είναι πάντα 0%% ή 100%%.")
    ap.add_argument("--stochastic", action="store_true",
                    help="αξιολόγηση με θόρυβο δράσης αντί για τη μέση δράση")
    ap.add_argument("--save", type=str, default=None,
                    help="όνομα στο οποίο θα αντιγραφεί το καλύτερο checkpoint")
    args = ap.parse_args()

    from stable_baselines3 import PPO

    cps = list_checkpoints(args.logs)
    cps = [c for c in cps if c[0] >= args.min_steps][::args.every]
    if not cps:
        print(f"Δεν βρέθηκαν checkpoints στο {args.logs}")
        return

    if args.dr is None:
        env = QuadFlipEnv(obs_noise=False, random_wind=False,
                          random_initial_state=args.vary, random_battery=False)
        cond = "καθαρό + τυχαία αρχική κατάσταση" if args.vary else "καθαρό"
    else:
        env = QuadFlipEnv()          # προεπιλογές: θόρυβος + άνεμος ενεργά
        env.set_dr_level(args.dr)
        cond = f"dr_level = {args.dr:.2f}"

    seeds = list(range(args.seeds))

    print(f"Αξιολόγηση {len(cps)} checkpoints x {args.seeds} seeds "
          f"({'στοχαστικά' if args.stochastic else 'ντετερμινιστικά'}), {cond}")
    if args.dr is None and not args.vary:
        print("ΠΡΟΣΟΧΗ: χωρίς --vary όλα τα seeds δίνουν το ίδιο επεισόδιο "
              "-> το ποσοστό θα είναι πάντα 0% ή 100%.")
    print()
    print(f"{'βήματα':>11} {'επιτυχία':>9} {'επιβίωση':>9} {'στάση':>8} {'xy':>8} {'reward':>9}")
    print("-" * 60)

    results = []
    for steps, mp, sp in cps:
        model = PPO.load(mp, device="cpu")
        vn = None
        if sp:
            with open(sp, "rb") as f:
                vn = pickle.load(f)
        if args.dr is not None:
            env.set_dr_level(args.dr)
        r = evaluate(model, vn, env, seeds, deterministic=not args.stochastic)
        r.update(steps=steps, model=mp, stats=sp)
        results.append(r)
        print(f"{steps:>11,} {r['rate']*100:>8.0f}% {r['survived']:>8.2f}s "
              f"{r['att_err']:>7.1f}° {r['xy']:>7.2f}m {r['reward']:>9.0f}")

    env.close()

    # Κατάταξη: πρώτα ποσοστό επιτυχίας, μετά χρόνος επιβίωσης, μετά σφάλμα στάσης
    best = max(results, key=lambda r: (r["rate"], r["survived"], -r["att_err"]))
    print("\n" + "=" * 60)
    print(f"ΚΑΛΥΤΕΡΟ: {best['steps']:,} βήματα -> επιτυχία {best['rate']*100:.0f}%, "
          f"επιβίωση {best['survived']:.2f}s, στάση {best['att_err']:.1f}°")

    by_reward = max(results, key=lambda r: r["reward"])
    if by_reward["steps"] != best["steps"]:
        print(f"ΠΡΟΣΟΧΗ: με κριτήριο το reward θα επιλεγόταν το {by_reward['steps']:,} "
              f"(reward {by_reward['reward']:.0f}, επιτυχία {by_reward['rate']*100:.0f}%) "
              f"- διαφορετικό, και γι' αυτό δεν κρίνουμε με reward.")
    print("=" * 60)

    if args.save:
        dst = os.path.join(_ROOT, args.save + ".zip")
        shutil.copyfile(best["model"], dst)
        print(f"\nΑντιγράφηκε -> {dst}")
        if best["stats"]:
            dst_s = os.path.join(_ROOT, args.save + "_vecnormalize.pkl")
            shutil.copyfile(best["stats"], dst_s)
            print(f"Αντιγράφηκε -> {dst_s}")
        print(f"\nΣύγκρινε με:  py -3.11 compare_pid_ppo.py --model {args.save} --episodes 10")


if __name__ == "__main__":
    main()
