"""
Πειραματικός σχηματισμός reward (reward shaping) για το PPO.

ΔΕΝ τροποποιεί το quad_flip_env.py. Είναι gymnasium.Wrapper που ΠΡΟΣΘΕΤΕΙ
όρους πάνω στο υπάρχον reward, ώστε το baseline να παραμένει αναπαραγώγιμο.

Κίνητρο (μετρημένο στη σύγκριση PID vs PPO στα 8M βήματα):

  Μετρική              PID      PPO     λόγος
  control effort       0.91    25.81     28x
  απώλεια ύψους       2.8cm   63.7cm     23x

ΟΡΟΣ 1 - Μέγεθος δράσης, ΜΟΝΟ στη φάση αιώρησης
  Ο υπάρχων r_action του env τιμωρεί ΜΟΝΟ τη μεταβολή ||a_t - a_{t-1}||
  (TOL_ACTION_SMOOTH). Το μέγεθος ||a_t|| δεν τιμωρείται καθόλου, οπότε η
  πολιτική μπορεί να κρατά τους κινητήρες μόνιμα κοντά στον κορεσμό χωρίς
  κόστος.

  Μετρημένη κατανομή του effort (seed 0, καθαρό benchmark):

                σύνολο   φάση flip   αιώρηση
      PID         0.89    0.67          0.22  (25%)
      PPO        15.12    0.93         14.19  (94%)

  Στην ΙΔΙΑ ΤΗΝ ΤΟΥΜΠΑ τα δύο είναι ισοδύναμα (0.93 vs 0.67). Όλη η σπατάλη
  του PPO είναι στην αιώρηση. Γι' αυτό ο όρος εφαρμόζεται ΜΟΝΟ όταν
  flip_completed=True: αλλιώς θα αποθαρρύναμε μια τούμπα που είναι ήδη καλή.

  Μέση ||a|| στην αιώρηση: PID ~0.17, PPO ~1.36. Με ανοχή 1.0 ο PID παίρνει
  ~0.97 ενώ το PPO ~0.16 - ισχυρή κλίση ακριβώς στην περιοχή λειτουργίας του.

ΟΡΟΣ 2 - Απώλεια ύψους στη φάση flip
  Ο υπάρχων r_alt_down = exp(-(alt_drop/0.30)^2) είναι ΚΟΡΕΣΜΕΝΟΣ για το PPO:
  alt_drop=0.64 -> x=2.13 -> exp(-4.55) ~ 0.01, με σχεδόν μηδενική παράγωγο.
  Δηλαδή δεν υπάρχει κλίση για να μάθει. Η σωστή κίνηση είναι ΕΥΡΥΤΕΡΗ ανοχή
  (0.6), που τοποθετεί το x στο ~1.07 - εκεί όπου η exp(-x^2) έχει τη μέγιστη
  κλίση της. Στένεμα της ανοχής θα χειροτέρευε τον κορεσμό.

Χρήση:
    py -3.11 train_ppo_baseline.py --steps ... --shaped
"""

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

# --- Παράμετροι σχηματισμού (συντονίστε εδώ) ---------------------------------------
W_EFFORT = 0.6        # βάρος του όρου μεγέθους δράσης (μόνο στην αιώρηση)
TOL_EFFORT = 1.0      # ανοχή ||a|| στην αιώρηση (PID ~0.17, PPO ~1.36)

W_ALT_FLIP = 1.0      # βάρος του επιπλέον όρου ύψους (μόνο στη φάση flip)
TOL_ALT_FLIP = 0.60   # m - ευρύτερη από την TOL_ALT_DOWN=0.30 του env, επίτηδες


class ShapedFlipReward(gym.Wrapper):
    """
    Προσθέτει όρους reward χωρίς να αλλάζει παρατήρηση, χώρο δράσης,
    συνθήκες τερματισμού ή seeds.
    """

    def __init__(self, env, w_effort=W_EFFORT, tol_effort=TOL_EFFORT,
                 w_alt_flip=W_ALT_FLIP, tol_alt_flip=TOL_ALT_FLIP):
        super().__init__(env)
        self.w_effort = float(w_effort)
        self.tol_effort = float(tol_effort)
        self.w_alt_flip = float(w_alt_flip)
        self.tol_alt_flip = float(tol_alt_flip)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped

        # Το env εφαρμόζει EMA στη δράση· το prev_action κρατά αυτή που όντως
        # χρησιμοποιήθηκε σε αυτό το βήμα.
        a = np.asarray(base.prev_action, dtype=np.float64)
        a_norm = float(np.linalg.norm(a))
        shaped = 0.0
        r_effort = float("nan")

        if base.flip_completed:
            # Αιώρηση: εδώ είναι το 94% της σπατάλης του PPO.
            r_effort = float(np.exp(-((a_norm / self.tol_effort) ** 2)))
            shaped += self.w_effort * r_effort
        else:
            # Φάση flip: ΚΑΜΙΑ ποινή μεγέθους - η τούμπα χρειάζεται μεγάλες
            # εντολές και είναι ήδη εξίσου οικονομική με του PID.
            # Επιπλέον σχηματισμός ύψους, με ανοχή που αφήνει πραγματική κλίση
            # (ο όρος του env με TOL=0.30 είναι κορεσμένος στο ~0.01).
            alt_drop = max(0.0, float(base.target_state[2] - base.quad.pos[2]))
            r_alt = float(np.exp(-((alt_drop / self.tol_alt_flip) ** 2)))
            shaped += self.w_alt_flip * r_alt

        # Διαγνωστικά για logging - δεν επηρεάζουν το περιβάλλον.
        info["shaping"] = {
            "a_norm": a_norm,
            "r_effort": r_effort,
            "shaped_total": shaped,
        }
        return obs, float(reward + shaped), terminated, truncated, info


def wrap(env):
    """Βοηθητικό για χρήση μέσα από το train.py."""
    return ShapedFlipReward(env)
