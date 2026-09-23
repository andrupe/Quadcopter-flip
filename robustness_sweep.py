"""
Καμπύλη αντοχής: ποσοστό επιτυχίας συναρτήσει του επιπέδου διαταραχής.

Αντί για δύο μεμονωμένα νούμερα (καθαρό και dr=1.0), σαρώνει όλη την κλίμακα
ώστε να φανεί ΠΟΥ σπάει ο καθένας, όχι απλώς ότι σπάει.

Όλοι οι ελεγκτές αξιολογούνται στα ΙΔΙΑ seeds και στα ΙΔΙΑ επίπεδα, με τον
ανεξάρτητο γεωμετρικό ανιχνευτή επιτυχίας.

Χρήση:
    py -3.11 robustness_sweep.py
    py -3.11 robustness_sweep.py --episodes 30
    py -3.11 robustness_sweep.py --models ppo_shaped_best ppo_adr_best
    py -3.11 robustness_sweep.py --csv robustness.csv
"""

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(_ROOT, "Simulation")
for _p in (_ROOT, _SIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.quad_flip_env import QuadFlipEnv
from compare_pid_ppo import PIDPolicy, PPOPolicy, run_episode


DEFAULT_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="ονόματα μοντέλων PPO (χωρίς .zip). Προεπιλογή: ό,τι υπάρχει")
    ap.add_argument("--levels", nargs="*", type=float, default=DEFAULT_LEVELS)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--no-pid", action="store_true", help="παράλειψη του PID")
    ap.add_argument("--csv", type=str, default=None,
                    help="αρχείο CSV για διάγραμμα")
    args = ap.parse_args()

    # Ποια μοντέλα να συμπεριληφθούν
    if args.models is None:
        candidates = ["ppo_shaped_best", "ppo_adr_best", "ppo_control_best"]
        models = [m for m in candidates
                  if os.path.isfile(os.path.join(_ROOT, m + ".zip"))]
    else:
        models = list(args.models)

    missing = [m for m in models if not os.path.isfile(os.path.join(_ROOT, m + ".zip"))]
    for m in missing:
        print(f"[!] Δεν βρέθηκε: {m}.zip - παραλείπεται")
    models = [m for m in models if m not in missing]

    names = ([] if args.no_pid else ["PID"]) + models
    if not names:
        print("Κανένας ελεγκτής προς αξιολόγηση.")
        return

    print("=" * 78)
    print(f"ΚΑΜΠΥΛΗ ΑΝΤΟΧΗΣ | {args.episodes} επεισόδια ανά σημείο | "
          f"ίδια seeds παντού")
    print("=" * 78)
    print("Το dr_level κλιμακώνει ΤΑΥΤΟΧΡΟΝΑ: άνεμο, θόρυβο αισθητήρων, μάζα,")
    print("κέντρο μάζας, ανισορροπία κινητήρων, μπαταρία, καθυστέρηση, αρχική κλίση.")
    print()

    results = {n: {} for n in names}

    for lvl in args.levels:
        env = QuadFlipEnv()          # προεπιλογές: θόρυβος + άνεμος ενεργά
        env.set_dr_level(lvl)
        for name in names:
            if name == "PID":
                pol = PIDPolicy(dt=env.dt)
            else:
                pol = PPOPolicy(os.path.join(_ROOT, name + ".zip"),
                                os.path.join(_ROOT, name + "_vecnormalize.pkl"))
            rows = []
            for s in range(args.episodes):
                env.set_dr_level(lvl)   # το reset δεν το αλλάζει, αλλά για ασφάλεια
                rows.append(run_episode(env, pol, seed=s))
            n = len(rows)
            results[name][lvl] = dict(
                rate=100.0 * sum(r["success"] for r in rows) / n,
                crash=100.0 * sum(r["crashed"] for r in rows) / n,
                att=float(np.mean([r["att_err"] for r in rows])),
                omega=float(np.mean([r["final_omega"] for r in rows])),
                effort=float(np.mean([r["effort"] for r in rows])),
                reward=float(np.mean([r["reward"] for r in rows])),
            )
            r = results[name][lvl]
            print(f"  dr={lvl:4.2f}  {name:<20s} επιτυχία {r['rate']:5.1f}%  "
                  f"συντριβές {r['crash']:5.1f}%  στάση {r['att']:6.2f}°  "
                  f"reward {r['reward']:6.0f}")
        env.close()
        print()

    # --- Συγκεντρωτικός πίνακας ---------------------------------------------------
    print("=" * 78)
    print("ΠΟΣΟΣΤΟ ΕΠΙΤΥΧΙΑΣ (%)")
    print("=" * 78)
    hdr = "dr_level".ljust(12) + "".join(f"{n:>20s}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for lvl in args.levels:
        row = f"{lvl:<12.2f}"
        for n in names:
            row += f"{results[n][lvl]['rate']:>19.1f}%"
        print(row)

    print()
    print("=" * 78)
    print("ΠΟΣΟΣΤΟ ΣΥΝΤΡΙΒΩΝ (%)")
    print("=" * 78)
    print(hdr)
    print("-" * len(hdr))
    for lvl in args.levels:
        row = f"{lvl:<12.2f}"
        for n in names:
            row += f"{results[n][lvl]['crash']:>19.1f}%"
        print(row)

    if args.csv:
        path = os.path.join(_ROOT, args.csv)
        with open(path, "w", encoding="utf-8") as f:
            f.write("controller,dr_level,success_pct,crash_pct,att_deg,"
                    "final_omega,effort,reward\n")
            for n in names:
                for lvl in args.levels:
                    r = results[n][lvl]
                    f.write(f"{n},{lvl},{r['rate']:.2f},{r['crash']:.2f},"
                            f"{r['att']:.3f},{r['omega']:.4f},{r['effort']:.3f},"
                            f"{r['reward']:.1f}\n")
        print(f"\nCSV για διάγραμμα: {path}")


if __name__ == "__main__":
    main()
