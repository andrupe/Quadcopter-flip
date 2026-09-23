"""
Αρχείο παράδοσης για τον ελεγκτή PPO.

Υλοποιεί ΑΚΡΙΒΩΣ τη διεπαφή που ορίζει το εκφώνημα:

    class Controller:
        def reset(self, seed=None): ...
        def act(self, observation): return action, info

Χρήση από script αξιολόγησης:

    from ppo_controller import Controller

    controller = Controller()
    obs, info = env.reset(seed=0)
    controller.reset(seed=0)
    for t in range(MAX_STEPS):
        action, dbg = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

Αυτοέλεγχος:
    py -3.11 ppo_controller.py
    py -3.11 ppo_controller.py --model ppo_adr_best
"""

import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Το μοντέλο που παραδίδεται. Άλλαξέ το εδώ αν επιλέξεις άλλο checkpoint.
#
# ppo_final: PPO με σχηματισμένο reward + domain randomization + 8M βήματα
# εδραίωσης στο dr_level=1.0. Μετρημένη επιτυχία ανά επίπεδο διαταραχής
# (20 επεισόδια/σημείο, ανεξάρτητος γεωμετρικός ανιχνευτής):
#
#   dr_level   0.00   0.25   0.50   0.75   1.00
#   ppo_final  100%   100%    95%    40%    40%
#   PID        100%   100%    95%    55%    25%
#   ppo_shaped 100%   100%    25%    10%     5%
#
# Επιβεβαίωση του κύριου ευρήματος με μεγαλύτερο δείγμα (60 επεισόδια,
# dr_level = 1.00): ppo_final 28/60 = 47%, PID 16/60 = 27%.
# Έλεγχος δύο αναλογιών: z = 2.27, p = 0.023 -> σημαντικό στο 5%.
#
# ΠΡΟΣΟΧΗ: στον φάκελο παράδοσης πρέπει να μπουν ΚΑΙ ΤΑ ΔΥΟ αρχεία:
#   ppo_final.zip  και  ppo_final_vecnormalize.pkl
DEFAULT_MODEL = "ppo_final"


class Controller:
    """
    Ελεγκτής PPO για το flip του quadcopter.

    ΣΗΜΑΝΤΙΚΟ: η πολιτική εκπαιδεύτηκε με VecNormalize, δηλαδή με
    κανονικοποιημένες παρατηρήσεις. Οι στατιστικές κανονικοποίησης φορτώνονται
    μαζί με τα βάρη και εφαρμόζονται σε κάθε βήμα. Χωρίς αυτές η πολιτική
    δέχεται παρατηρήσεις σε λάθος κλίμακα και η συμπεριφορά της καταρρέει.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        from stable_baselines3 import PPO

        model_path = os.path.join(_HERE, model_name + ".zip")
        stats_path = os.path.join(_HERE, model_name + "_vecnormalize.pkl")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Δεν βρέθηκαν τα βάρη της πολιτικής: {model_path}")

        self.model_name = model_name
        self.model = PPO.load(model_path, device=device)

        if not os.path.isfile(stats_path):
            raise FileNotFoundError(
                f"Δεν βρέθηκαν οι στατιστικές VecNormalize: {stats_path}\n"
                "Η πολιτική εκπαιδεύτηκε με κανονικοποιημένες παρατηρήσεις και "
                "χωρίς αυτές δεν λειτουργεί σωστά."
            )
        import pickle
        with open(stats_path, "rb") as f:
            self._vecnorm = pickle.load(f)
        # Αξιολόγηση: οι στατιστικές δεν ενημερώνονται.
        self._vecnorm.training = False

        self._steps = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        """
        Η πολιτική είναι MLP χωρίς επαναλαμβανόμενη κατάσταση, οπότε δεν
        υπάρχει εσωτερική μνήμη να μηδενιστεί. Η υπογραφή διατηρείται όπως
        ορίζει το εκφώνημα και το seed γίνεται δεκτό χωρίς να αλλάζει τίποτα
        στο περιβάλλον.
        """
        self._steps = 0

    # ------------------------------------------------------------------
    def act(self, observation):
        """
        Επιστρέφει (action, info).

        action : np.ndarray shape (4,), τιμές στο [-1, 1], μία ανά κινητήρα.
        info   : dict με διαγνωστικά μόνο για logging/visualization. Δεν
                 τροποποιεί περιβάλλον, reward, τερματισμό ή seeds.
        """
        obs = np.asarray(observation, dtype=np.float32)
        obs_norm = self._vecnorm.normalize_obs(obs)

        action, _ = self.model.predict(obs_norm, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        self._steps += 1
        info = {
            "policy": self.model_name,
            "step": self._steps,
            "action_norm": float(np.linalg.norm(action)),
        }
        return action, info


# Εναλλακτικό όνομα, αν το script αξιολόγησης το ψάχνει πιο περιγραφικά.
PPOFlipController = Controller


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Αυτοέλεγχος του ελεγκτή PPO")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    # Το αρχείο ζει στο <repo>/controllers/PPO, το περιβάλλον δύο επίπεδα πάνω.
    _REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
    for _p in (_HERE, _REPO, os.path.join(_REPO, "Simulation")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from Simulation.quad_flip_env import QuadFlipEnv

    print(f"Έλεγχος διεπαφής για: {args.model}")
    controller = Controller(args.model)

    # Έλεγχος ότι το act() επιστρέφει ΔΥΟ τιμές, όπως ορίζει το εκφώνημα.
    env = QuadFlipEnv(obs_noise=False, random_wind=False,
                      random_initial_state=True, random_battery=False)
    obs, _ = env.reset(seed=0)
    controller.reset(seed=0)
    out = controller.act(obs)
    assert isinstance(out, tuple) and len(out) == 2, \
        "Το act() ΠΡΕΠΕΙ να επιστρέφει (action, info)"
    a, dbg = out
    assert a.shape == (4,), f"Λάθος σχήμα δράσης: {a.shape}"
    assert float(np.abs(a).max()) <= 1.0, "Η δράση ξεφεύγει από το [-1, 1]"
    print(f"  act() -> (action{tuple(a.shape)}, info με κλειδιά {list(dbg)})  OK")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        controller.reset(seed=ep)
        total, zmin = 0.0, float(info["position"][2])
        for t in range(env.max_steps):
            action, _dbg = controller.act(obs)
            obs, r, term, trunc, info = env.step(action)
            total += r
            zmin = min(zmin, float(info["position"][2]))
            if term or trunc:
                break
        print(f"  επεισόδιο {ep}: reward={total:7.1f}  flip={env.flip_completed}  "
              f"ελάχ.ύψος={zmin:.2f}m  βήματα={t+1}")
    env.close()
    print("\nΗ διεπαφή είναι συμβατή με το εκφώνημα.")
