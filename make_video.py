"""
Παραγωγή demo video για την παράδοση του project.

Το εκφώνημα ζητά: "a short demo video showing representative successes and
failures". Το script παράγει ένα MP4 με σκηνές που δείχνουν και τα δύο.

Χρήση:
    py -3.11 make_video.py                       # προεπιλεγμένες σκηνές
    py -3.11 make_video.py --out demo.mp4 --fps 30
    py -3.11 make_video.py --width 1280 --height 720
"""

import argparse
import os
import pickle
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(_ROOT, "Simulation")
# Ο PID ελεγκτής ζει στο controllers/Pid - χρειάζεται στο path για το import.
_PID = os.path.join(_ROOT, "controllers", "Pid")
for _p in (_ROOT, _SIM, _PID):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.quad_flip_env import QuadFlipEnv


# ----------------------------------------------------------------------------------
class PPOPolicy:
    def __init__(self, name):
        from stable_baselines3 import PPO
        self.model = PPO.load(os.path.join(_ROOT, name + ".zip"), device="cpu")
        stats = os.path.join(_ROOT, name + "_vecnormalize.pkl")
        with open(stats, "rb") as f:
            self.vn = pickle.load(f)

    def reset(self, seed=None):
        pass

    def act(self, obs):
        a, _ = self.model.predict(self.vn.normalize_obs(obs), deterministic=True)
        return a


class PIDPolicy:
    def __init__(self, dt):
        from algor import FlipController
        self.c = FlipController(dt=dt)

    def reset(self, seed=None):
        self.c.reset(seed=seed)

    def act(self, obs):
        a, _ = self.c.act(obs)
        return a


# ----------------------------------------------------------------------------------
def draw_overlay(frame, lines, color=(255, 255, 255)):
    """Γράφει κείμενο πάνω στο καρέ (χωρίς εξωτερικές γραμματοσειρές)."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    pad, lh = 12, 16
    box_h = pad + lh * len(lines)
    d.rectangle([0, 0, img.width, box_h], fill=(0, 0, 0))
    for i, (text, col) in enumerate(lines):
        d.text((pad, 6 + i * lh), text, fill=col)
    return np.asarray(img)


def render_scene(writer, title, policy_name, policy, env, seed,
                 max_seconds, width, height, every, dr_level=None):
    """Τρέχει ένα επεισόδιο και γράφει τα καρέ στο video."""
    renderer = mujoco.Renderer(env.quad.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.distance = 1.6
    cam.elevation = -12.0
    cam.azimuth = 130.0

    obs, info = env.reset(seed=seed)
    policy.reset(seed=seed)

    z0 = float(info["position"][2])
    zmin = z0
    n_steps = min(env.max_steps, int(max_seconds / env.dt))
    outcome = "ΟΛΟΚΛΗΡΩΘΗΚΕ"
    outcome_col = (120, 255, 120)

    for t in range(n_steps):
        action = policy.act(obs)
        obs, r, term, trunc, info = env.step(action)
        pos = info["position"]
        zmin = min(zmin, float(pos[2]))

        if t % every == 0:
            cam.lookat[:] = pos
            renderer.update_scene(env.quad.data, camera=cam)
            frame = renderer.render()
            flip_deg = np.rad2deg(env.accumulated_pitch)
            lines = [
                (title, (255, 255, 255)),
                (f"{policy_name}", (180, 210, 255)),
                (f"t={info['t']:4.2f}s   ypsos={pos[2]:4.2f}m   "
                 f"peristrofi={flip_deg:3.0f}deg", (200, 200, 200)),
                (f"flip_completed={env.flip_completed}   "
                 f"|w|={np.linalg.norm(info['omega']):4.1f} rad/s", (200, 200, 200)),
            ]
            writer.append_data(draw_overlay(frame, lines))

        if term or trunc:
            if term:
                if env.quad.check_ground_contact():
                    outcome, outcome_col = "ΣΥΝΤΡΙΒΗ ΣΤΟ ΕΔΑΦΟΣ", (255, 120, 120)
                elif float(np.linalg.norm(env.quad.pos[:2])) > env.arena_radius:
                    outcome, outcome_col = "ΕΞΟΔΟΣ ΑΠΟ ΤΟ ARENA", (255, 120, 120)
                elif env.quad.pos[2] > 2.5:
                    outcome, outcome_col = "ΥΠΕΡΒΑΣΗ ΤΑΒΑΝΙΟΥ", (255, 120, 120)
                else:
                    outcome, outcome_col = "ΤΕΡΜΑΤΙΣΜΟΣ", (255, 120, 120)
            break

    # Πάγωμα τελευταίου καρέ με το αποτέλεσμα
    cam.lookat[:] = env.quad.pos
    renderer.update_scene(env.quad.data, camera=cam)
    frame = renderer.render()
    tail = [
        (title, (255, 255, 255)),
        (f"{policy_name}", (180, 210, 255)),
        (outcome, outcome_col),
        (f"elach. ypsos={zmin:4.2f}m   peristrofi="
         f"{np.rad2deg(env.accumulated_pitch):3.0f}deg", (200, 200, 200)),
    ]
    for _ in range(20):
        writer.append_data(draw_overlay(frame, tail))

    renderer.close()
    print(f"  {title}: {outcome} (elach. ypsos {zmin:.2f}m)")


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="demo_ppo.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--every", type=int, default=2,
                    help="καρέ ανά N βήματα προσομοίωσης (2 -> 50Hz δείγματα)")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="μέγιστη διάρκεια προσομοίωσης ανά σκηνή")
    args = ap.parse_args()

    out = os.path.join(_ROOT, args.out)
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=None)

    # Σκηνές: επιτυχία με την τελική πολιτική, αποτυχία με την αρχική.
    scenes = []
    if os.path.isfile(os.path.join(_ROOT, "ppo_shaped_best.zip")):
        scenes.append(("EPITYXIA - PPO me shaped reward",
                       "ppo_shaped_best (20M vimata)", "ppo_shaped_best", 0))
        scenes.append(("EPITYXIA - allo epeisodio",
                       "ppo_shaped_best (20M vimata)", "ppo_shaped_best", 7))
    if os.path.isfile(os.path.join(_ROOT, "ppo_5m.zip")):
        scenes.append(("APOTYXIA - PPO prin ti diorthosi",
                       "ppo_5m (arxiko reward, ent_coef=0.01)", "ppo_5m", 0))

    print(f"Παραγωγή {out} ({args.width}x{args.height}, {args.fps}fps)\n")
    for title, label, model, seed in scenes:
        env = QuadFlipEnv(obs_noise=False, random_wind=False,
                          random_initial_state=True, random_battery=False)
        env.set_dr_level(0.0)
        pol = PPOPolicy(model)
        render_scene(writer, title, label, pol, env, seed,
                     args.seconds, args.width, args.height, args.every)
        env.close()

    writer.close()
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"\nΈτοιμο: {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
