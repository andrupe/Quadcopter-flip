"""
Δίκαιη σύγκριση cascaded-PID vs PPO στις ΙΔΙΕΣ συνθήκες και ΙΔΙΑ seeds.

Και οι δύο controllers αξιολογούνται με τον ίδιο ανεξάρτητο γεωμετρικό
ανιχνευτή επιτυχίας - όχι με το reward, όπως ζητά ρητά το εκφώνημα.

Χρήση:
    py -3.11 compare_pid_ppo.py --model ppo_5m
    py -3.11 compare_pid_ppo.py --model ppo_5m --episodes 20
    py -3.11 compare_pid_ppo.py --model ppo_5m --dr 1.0     # stress test
"""

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(_ROOT, "Simulation")
# Οι ελεγκτές ζουν στο controllers/ - πρέπει να είναι στο path για το import.
_PID = os.path.join(_ROOT, "controllers", "Pid")
for _p in (_ROOT, _SIM, _PID):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.quad_flip_env import QuadFlipEnv
from algor import FlipController, evaluate_flip_success


# ----------------------------------------------------------------------------------
# Wrappers ώστε PID και PPO να έχουν την ίδια διεπαφή (reset / act)
# ----------------------------------------------------------------------------------
class PIDPolicy:
    name = "Cascaded PID"

    def __init__(self, dt):
        self.c = FlipController(dt=dt)

    def reset(self, seed=None):
        self.c.reset(seed=seed)

    def act(self, obs):
        action, info = self.c.act(obs)
        return action


class PPOPolicy:
    def __init__(self, model_path, stats_path=None):
        from stable_baselines3 import PPO
        self.name = f"PPO ({os.path.basename(model_path)})"
        self.model = PPO.load(model_path, device="cpu")
        self.vec_norm = None
        if stats_path and os.path.isfile(stats_path):
            # Φορτώνουμε ΜΟΝΟ τις στατιστικές κανονικοποίησης παρατηρήσεων.
            import pickle
            with open(stats_path, "rb") as f:
                vn = pickle.load(f)
            self.vec_norm = vn
            print(f"  Φορτώθηκαν στατιστικές VecNormalize: {stats_path}")
        else:
            print("  ΠΡΟΣΟΧΗ: δεν βρέθηκαν στατιστικές VecNormalize -> "
                  "οι παρατηρήσεις δεν κανονικοποιούνται, η πολιτική θα συμπεριφερθεί λάθος!")

    def reset(self, seed=None):
        pass

    def act(self, obs):
        o = obs
        if self.vec_norm is not None:
            o = self.vec_norm.normalize_obs(obs)
        action, _ = self.model.predict(o, deterministic=True)
        return action


# ----------------------------------------------------------------------------------
def run_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    policy.reset(seed=seed)

    z0 = float(info["position"][2])
    zmin = z0
    R = 0.0
    effort = 0.0
    t_complete = None
    pitch_hist, z_hist, omega_hist = [], [z0], []
    omega_f = np.zeros(3)
    dcm_f = np.eye(3)
    terminated = False

    for t in range(env.max_steps):
        action = policy.act(obs)
        q = env.quad.quat
        pitch_hist.append(float(np.arctan2(2.0 * (q[0] * q[2] + q[3] * q[1]),
                                           1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))))
        omega_hist.append(env.quad.omega.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        R += reward
        effort += float(np.sum(np.square(np.asarray(action)))) * env.dt
        zmin = min(zmin, float(info["position"][2]))
        z_hist.append(float(info["position"][2]))
        omega_f = info["omega"]
        dcm_f = env.quad.dcm.copy()

        if env.flip_completed and t_complete is None:
            t_complete = float(info["t"])

        if terminated or truncated:
            break

    att_err = float(np.degrees(np.arccos(np.clip(dcm_f[2, 2], -1.0, 1.0))))
    # Ανεξάρτητος γεωμετρικός έλεγχος (σιωπηλός)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        geo_ok = evaluate_flip_success(np.unwrap(pitch_hist).tolist(), z_hist, omega_hist)

    return dict(
        success=bool(geo_ok and env.flip_completed and not terminated),
        geo_ok=bool(geo_ok),
        env_flip=bool(env.flip_completed),
        crashed=bool(terminated),
        t_complete=t_complete,
        alt_loss=max(0.0, z0 - zmin),
        zmin=zmin,
        att_err=att_err,
        final_omega=float(np.linalg.norm(omega_f)),
        xy_final=float(np.linalg.norm(info["position"][:2])),
        effort=effort,
        reward=R,
        steps=t + 1,
    )


def summarize(name, rows):
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    done = [r for r in rows if r["t_complete"] is not None]

    def m(key, src=None):
        src = src if src is not None else rows
        vals = [r[key] for r in src if r[key] is not None]
        return np.mean(vals) if vals else float("nan")

    print(f"\n  {name}")
    print(f"    Επιτυχία (γεωμετρική)   : {len(ok)}/{n}  ({100*len(ok)/n:.0f}%)")
    print(f"    Συντριβές               : {sum(r['crashed'] for r in rows)}/{n}")
    print(f"    Χρόνος ολοκλ. flip      : {m('t_complete', done):.3f} s "
          f"({len(done)}/{n} ολοκλήρωσαν)")
    print(f"    Μέγ. απώλεια ύψους      : {max(r['alt_loss'] for r in rows)*100:.1f} cm")
    print(f"    Τελικό σφάλμα στάσης    : {m('att_err'):.2f}°")
    print(f"    Τελική γωνιακή ταχύτητα : {m('final_omega'):.4f} rad/s")
    print(f"    Τελικό σφάλμα θέσης XY  : {m('xy_final')*1000:.1f} mm")
    print(f"    Control effort (Σa²·dt) : {m('effort'):.2f}")
    print(f"    Reward                  : {m('reward'):.0f} "
          f"[{min(r['reward'] for r in rows):.0f} - {max(r['reward'] for r in rows):.0f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="ppo_5m",
                    help="όνομα μοντέλου χωρίς .zip (π.χ. ppo_5m)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--dr", type=float, default=None,
                    help="επίπεδο domain randomization 0.0-1.0 (παράλειψη = καθαρό benchmark)")
    ap.add_argument("--vary", action="store_true",
                    help="τυχαία αρχική κατάσταση (θέση/στάση/ταχύτητα) ΧΩΡΙΣ θόρυβο ή "
                         "άνεμο. Απαραίτητο για ουσιαστικό ποσοστό επιτυχίας: χωρίς "
                         "αυτό κάθε seed δίνει ΤΟ ΙΔΙΟ επεισόδιο και το ποσοστό είναι "
                         "πάντα 0% ή 100%.")
    args = ap.parse_args()

    model_path = os.path.join(_ROOT, args.model + ".zip")
    stats_path = os.path.join(_ROOT, args.model + "_vecnormalize.pkl")

    clean = args.dr is None
    if clean:
        env_kw = dict(obs_noise=False, random_wind=False,
                      random_initial_state=args.vary, random_battery=False)
        label = ("ΚΑΘΑΡΟ + ΤΥΧΑΙΑ ΑΡΧΙΚΗ ΚΑΤΑΣΤΑΣΗ" if args.vary
                 else "ΚΑΘΑΡΟ BENCHMARK (ΠΡΟΣΟΧΗ: όλα τα seeds δίνουν το ίδιο επεισόδιο)")
    else:
        env_kw = {}
        label = f"ΜΕ ΔΙΑΤΑΡΑΧΕΣ (dr_level = {args.dr:.2f})"

    print("=" * 70)
    print(f"ΣΥΓΚΡΙΣΗ PID vs PPO | {label} | {args.episodes} επεισόδια")
    print("=" * 70)

    policies = []
    env = QuadFlipEnv(**env_kw)
    policies.append(PIDPolicy(dt=env.dt))

    if os.path.isfile(model_path):
        print(f"\nΦόρτωση PPO: {model_path}")
        policies.append(PPOPolicy(model_path, stats_path))
    else:
        print(f"\n[!] Δεν βρέθηκε μοντέλο: {model_path}")
        print("    Τρέξε πρώτα:  py -3.11 train_ppo_baseline.py --steps 5000000 "
              "--workers 4 --name " + args.model)
        print("    Συνεχίζω μόνο με τον PID.\n")

    for pol in policies:
        if not clean:
            env.set_dr_level(args.dr)
        rows = [run_episode(env, pol, seed=s) for s in range(args.episodes)]
        summarize(pol.name, rows)

    env.close()
    print("\n" + "=" * 70)
    print("Σημείωση: η επιτυχία κρίνεται από τον ανεξάρτητο γεωμετρικό ανιχνευτή,")
    print("όχι από το reward - όπως ζητά ρητά το εκφώνημα του project.")
    print("=" * 70)


if __name__ == "__main__":
    main()
