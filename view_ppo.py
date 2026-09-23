"""
Οπτικοποίηση εκπαιδευμένης πολιτικής PPO στον 3D viewer του MuJoCo.

Ίδια λογική και ίδια flags με το view_pid.py, αλλά αντί για τον cascaded PID
τρέχει ένα αποθηκευμένο μοντέλο PPO.

Χρήση:
    py -3.11 view_ppo.py                          # ppo_shaped_best, αργή κίνηση 0.25x
    py -3.11 view_ppo.py --speed 1.0              # πραγματικός χρόνος
    py -3.11 view_ppo.py --loop                   # συνεχής επανάληψη επεισοδίων
    py -3.11 view_ppo.py --model ppo_5m           # η ΑΠΟΤΥΧΗΜΕΝΗ πολιτική (πριν τη διόρθωση)
    py -3.11 view_ppo.py --model ppo_best          # το baseline (χωρίς shaped reward)
    py -3.11 view_ppo.py --disturbed              # με άνεμο, θόρυβο, τυχαία αρχική κατάσταση
    py -3.11 view_ppo.py --fixed-start            # σταθερή αρχική κατάσταση (ίδιο επεισόδιο κάθε φορά)
    py -3.11 view_ppo.py --no-viewer              # headless (μόνο τα στατιστικά)

Χειρισμός στο παράθυρο: ποντίκι = περιστροφή/zoom, ESC ή κλείσιμο = έξοδος.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(_ROOT, "Simulation")
# Ο PID ελεγκτής ζει στο controllers/Pid - χρειάζεται στο path για το import.
_PID = os.path.join(_ROOT, "controllers", "Pid")
for _p in (_ROOT, _SIM, _PID):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.quad_flip_env import QuadFlipEnv
from algor import evaluate_flip_success


def parse_args():
    p = argparse.ArgumentParser(description="MuJoCo viewer για πολιτική PPO")
    p.add_argument("--model", type=str, default="ppo_shaped_best",
                   help="όνομα μοντέλου χωρίς .zip")
    p.add_argument("--speed", type=float, default=0.25,
                   help="ταχύτητα αναπαραγωγής (1.0 = πραγματικός χρόνος, 0.25 = slow motion)")
    p.add_argument("--seed", type=int, default=0, help="seed του επεισοδίου")
    p.add_argument("--loop", action="store_true", help="επανάληψη επεισοδίων συνέχεια")
    p.add_argument("--disturbed", action="store_true",
                   help="ενεργοποίηση άνεμου / θορύβου / πλήρους τυχαιοποίησης")
    p.add_argument("--fixed-start", action="store_true",
                   help="σταθερή αρχική κατάσταση (χωρίς αυτό κάθε επεισόδιο διαφέρει)")
    p.add_argument("--no-viewer", action="store_true", help="τρέξιμο χωρίς παράθυρο")
    p.add_argument("--no-track", action="store_true",
                   help="σταθερή κάμερα (χωρίς παρακολούθηση του drone)")
    return p.parse_args()


class PPOPolicy:
    def __init__(self, name):
        from stable_baselines3 import PPO
        model_path = os.path.join(_ROOT, name + ".zip")
        stats_path = os.path.join(_ROOT, name + "_vecnormalize.pkl")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Δεν βρέθηκε το μοντέλο: {model_path}")
        self.name = name
        self.model = PPO.load(model_path, device="cpu")
        self.vn = None
        if os.path.isfile(stats_path):
            with open(stats_path, "rb") as f:
                self.vn = pickle.load(f)
        else:
            print("ΠΡΟΣΟΧΗ: λείπουν οι στατιστικές VecNormalize -> "
                  "η πολιτική θα συμπεριφερθεί λάθος!")

    def reset(self, seed=None):
        pass

    def act(self, obs):
        o = self.vn.normalize_obs(obs) if self.vn is not None else obs
        action, _ = self.model.predict(o, deterministic=True)
        return action


def make_env(args):
    if args.disturbed:
        return QuadFlipEnv()  # defaults του env: πλήρεις διαταραχές
    return QuadFlipEnv(
        obs_noise=False,
        random_wind=False,
        # Χωρίς τυχαία αρχική κατάσταση ΚΑΘΕ επεισόδιο είναι πανομοιότυπο,
        # οπότε το --loop δεν θα έδειχνε τίποτα καινούργιο.
        random_initial_state=not args.fixed_start,
        random_battery=False,
    )


def termination_reason(env):
    if env.quad.check_ground_contact():
        return "Πρόσκρουση στο έδαφος"
    if env.quad.pos[2] > 2.5:
        return "Έξοδος από το ταβάνι (> 2.5 m)"
    if float(np.linalg.norm(env.quad.pos[:2])) > env.arena_radius:
        return f"Έξοδος από το arena (|xy| > {env.arena_radius} m)"
    return "Αποκλίνουσα κατάσταση"


def run(env, policy, viewer, args):
    """Τρέχει ένα επεισόδιο. Επιστρέφει False αν ο χρήστης έκλεισε το παράθυρο."""
    obs, info = env.reset(seed=args.seed)
    policy.reset(seed=args.seed)

    pitch_hist, z_hist, omega_hist = [], [float(info["position"][2])], []
    total_reward = 0.0
    alive = True
    max_perturb = 0.0
    teleports = 0
    prev_pos = env.quad.pos.copy()
    announced_inverted = False
    announced_flip = False

    if viewer is not None:
        viewer.sync()

    for step in range(env.max_steps):
        if viewer is not None and not viewer.is_running():
            alive = False
            break

        step_start = time.time()

        q = env.quad.quat
        pitch_hist.append(float(np.arctan2(2.0 * (q[0] * q[2] + q[3] * q[1]),
                                           1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))))
        omega_hist.append(env.quad.omega.copy())

        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        z_hist.append(float(info["position"][2]))
        max_perturb = max(max_perturb, float(np.abs(env.quad.data.xfrc_applied).max()))

        # Ο viewer έχει ΔΥΟ τρόπους να επέμβει:
        #   1. mjv_applyPerturbForce -> γράφει στο xfrc_applied (πιάνεται πάνω)
        #   2. mjv_applyPerturbPose  -> γράφει ΑΠΕΥΘΕΙΑΣ στο qpos, δηλαδή
        #      τηλεμεταφέρει το σώμα χωρίς κανένα ίχνος σε δυνάμεις.
        # Τον δεύτερο τον πιάνουμε μόνο ως φυσικά αδύνατο άλμα θέσης:
        # 0.1 m σε ένα βήμα των 10 ms αντιστοιχεί σε 10 m/s.
        cur_pos = env.quad.pos.copy()
        if float(np.linalg.norm(cur_pos - prev_pos)) > 0.1:
            teleports += 1
        prev_pos = cur_pos

        # Ορόσημα του ελιγμού
        if not announced_inverted and env.has_inverted:
            print(f"  t={info['t']:5.2f}s  -> ανάποδη θέση (z = {info['position'][2]:.2f} m)")
            announced_inverted = True
        if not announced_flip and env.flip_completed:
            print(f"  t={info['t']:5.2f}s  -> FLIP ΟΛΟΚΛΗΡΩΘΗΚΕ "
                  f"(z = {info['position'][2]:.2f} m, "
                  f"|ω| = {np.linalg.norm(info['omega']):.2f} rad/s)")
            announced_flip = True

        if viewer is not None:
            if not args.no_track:
                viewer.cam.lookat[:] = env.quad.pos
            viewer.sync()
            target_dt = env.dt / max(0.01, args.speed)
            sleep = target_dt - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

        if terminated or truncated:
            if terminated:
                print(f"  ΤΕΡΜΑΤΙΣΜΟΣ στο t={info['t']:.2f}s: {termination_reason(env)}")
            break

    print(f"\n  Reward: {total_reward:.1f} | env.flip_completed = {env.flip_completed} "
          f"({np.rad2deg(env.accumulated_pitch):.0f}°) | ελάχ. ύψος = {min(z_hist):.2f} m")

    if max_perturb > 0.0 or teleports > 0:
        print("  ΠΡΟΣΟΧΗ: το επεισόδιο επηρεάστηκε από τον viewer - ΔΕΝ είναι "
              "έγκυρο ως benchmark.")
        if max_perturb > 0.0:
            print(f"    - εξωτερική δύναμη (max |xfrc| = {max_perturb:.3f})")
        if teleports > 0:
            print(f"    - {teleports} άλματα θέσης (σύρσιμο του drone με Ctrl)")

    evaluate_flip_success(np.unwrap(pitch_hist).tolist(), z_hist, omega_hist)
    return alive


def main():
    args = parse_args()
    env = make_env(args)
    try:
        policy = PPOPolicy(args.model)
    except FileNotFoundError as e:
        print(e)
        print("\nΔιαθέσιμα μοντέλα:")
        for f in sorted(os.listdir(_ROOT)):
            if f.endswith(".zip"):
                print("  ", f[:-4])
        return

    mode = "ΜΕ ΔΙΑΤΑΡΑΧΕΣ" if args.disturbed else "ΚΑΘΑΡΟ"
    start = "σταθερή αρχική κατάσταση" if args.fixed_start else "τυχαία αρχική κατάσταση"
    print(f"PPO: {args.model} | {mode} | {start} | speed={args.speed}x")

    viewer = None
    if not args.no_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(env.quad.model, env.quad.data)
            viewer.cam.lookat[:] = [0.0, 0.0, 1.2]
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 135.0
            print("Άνοιξε ο 3D viewer του MuJoCo. ESC ή κλείσιμο παραθύρου για έξοδο.\n")
        except Exception as e:
            print(f"Δεν άνοιξε ο viewer ({e}). Συνεχίζω headless.\n")

    try:
        while True:
            if not run(env, policy, viewer, args):
                break
            if not args.loop or viewer is None:
                break
            print("\n" + "-" * 60)
            args.seed += 1  # επόμενο επεισόδιο, διαφορετική αρχική κατάσταση
            time.sleep(0.5)
    finally:
        if viewer is not None and viewer.is_running():
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
