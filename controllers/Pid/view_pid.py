"""
Οπτικοποίηση του cascaded-PID controller στον 3D viewer του MuJoCo.

Χρήση:
    py -3.11 view_pid.py                  # αργή κίνηση 0.25x (το flip κρατάει μόλις 0.4s)
    py -3.11 view_pid.py --speed 1.0      # πραγματικός χρόνος
    py -3.11 view_pid.py --loop           # συνεχής επανάληψη επεισοδίων
    py -3.11 view_pid.py --disturbed      # με άνεμο, θόρυβο και τυχαία αρχική κατάσταση
    py -3.11 view_pid.py --no-viewer      # headless (μόνο τα στατιστικά)

Χειρισμός στο παράθυρο: ποντίκι = περιστροφή/zoom, ESC ή κλείσιμο = έξοδος.
"""

import argparse
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

# Το αρχείο ζει στο <repo>/controllers/Pid, το περιβάλλον δύο επίπεδα πάνω.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO, os.path.join(_REPO, "Simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.quad_flip_env import QuadFlipEnv
from algor import FlipController, evaluate_flip_success


def parse_args():
    p = argparse.ArgumentParser(description="MuJoCo viewer για τον PID flip controller")
    p.add_argument("--speed", type=float, default=0.25,
                   help="ταχύτητα αναπαραγωγής (1.0 = πραγματικός χρόνος, 0.25 = slow motion)")
    p.add_argument("--seed", type=int, default=0, help="seed του επεισοδίου")
    p.add_argument("--loop", action="store_true", help="επανάληψη επεισοδίων συνέχεια")
    p.add_argument("--disturbed", action="store_true",
                   help="ενεργοποίηση άνεμου / θορύβου / τυχαίας αρχικής κατάστασης")
    p.add_argument("--no-viewer", action="store_true", help="τρέξιμο χωρίς παράθυρο")
    p.add_argument("--no-track", action="store_true",
                   help="σταθερή κάμερα (χωρίς παρακολούθηση του drone)")
    return p.parse_args()


def make_env(disturbed):
    if disturbed:
        return QuadFlipEnv()  # defaults του env: πλήρεις διαταραχές
    return QuadFlipEnv(
        obs_noise=False,
        random_wind=False,
        random_initial_state=False,
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


def run(env, controller, viewer, args):
    """Τρέχει ένα επεισόδιο. Επιστρέφει False αν ο χρήστης έκλεισε το παράθυρο."""
    obs, info = env.reset(seed=args.seed)
    controller.reset(seed=args.seed)

    pitch_hist, z_hist, omega_hist = [], [float(info["position"][2])], []
    total_reward = 0.0
    prev_phase = controller.phase
    alive = True
    # Ο passive viewer εφαρμόζει ΠΡΑΓΜΑΤΙΚΗ εξωτερική δύναμη στο σώμα όταν κάνεις
    # διπλό κλικ + Ctrl-σύρσιμο (γράφεται στο data.xfrc_applied). Χρήσιμο για να
    # δοκιμάσεις απόρριψη διαταραχών, αλλά αλλοιώνει το καθαρό benchmark - οπότε
    # το ανιχνεύουμε και το δηλώνουμε αντί να περάσει σιωπηλά.
    max_perturb = 0.0
    teleports = 0
    prev_pos = env.quad.pos.copy()

    if viewer is not None:
        viewer.sync()

    for step in range(env.max_steps):
        if viewer is not None and not viewer.is_running():
            alive = False
            break

        step_start = time.time()

        action, dbg = controller.act(obs)
        pitch_hist.append(dbg["pitch_raw"])
        omega_hist.append(dbg["omega"])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        z_hist.append(float(info["position"][2]))
        max_perturb = max(max_perturb, float(np.abs(env.quad.data.xfrc_applied).max()))

        # Το mjv_applyPerturbPose γράφει απευθείας στο qpos (τηλεμεταφορά) και
        # δεν αφήνει ίχνος σε δυνάμεις. Το πιάνουμε ως φυσικά αδύνατο άλμα.
        cur_pos = env.quad.pos.copy()
        if float(np.linalg.norm(cur_pos - prev_pos)) > 0.1:
            teleports += 1
        prev_pos = cur_pos

        # Εκτύπωση στις αλλαγές φάσης
        if dbg["phase"] != prev_phase:
            print(f"  t={info['t']:5.2f}s  -> φάση {dbg['phase']:8s} "
                  f"(γωνία flip = {np.rad2deg(dbg['flip_angle']):6.1f}°, "
                  f"z = {info['position'][2]:.2f} m)")
            prev_phase = dbg["phase"]

        if viewer is not None:
            if not args.no_track:
                viewer.cam.lookat[:] = env.quad.pos
            viewer.sync()
            # Ρύθμιση ρυθμού αναπαραγωγής σε πραγματικό χρόνο
            target_dt = env.dt / max(0.01, args.speed)
            sleep = target_dt - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

        if terminated or truncated:
            if terminated:
                print(f"  ΤΕΡΜΑΤΙΣΜΟΣ στο t={info['t']:.2f}s: {termination_reason(env)}")
            break

    flip_deg = np.rad2deg(env.accumulated_pitch)
    print(f"\n  Reward: {total_reward:.1f} | env.flip_completed = {env.flip_completed} "
          f"({flip_deg:.0f}° καταγεγραμμένα) | ελάχ. ύψος = {min(z_hist):.2f} m")

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
    env = make_env(args.disturbed)
    controller = FlipController(dt=env.dt)

    mode = "ΜΕ ΔΙΑΤΑΡΑΧΕΣ" if args.disturbed else "ΚΑΘΑΡΟ BENCHMARK"
    print(f"Cascaded-PID flip controller | {mode} | seed={args.seed} | speed={args.speed}x")

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
            if not run(env, controller, viewer, args):
                break
            if not args.loop or viewer is None:
                break
            print("\n" + "-" * 60)
            time.sleep(0.5)
    finally:
        if viewer is not None and viewer.is_running():
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
