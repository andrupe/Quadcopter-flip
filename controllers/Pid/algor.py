"""
Baseline PID controller για το Project 1 (Quadcopter Flip).

Δομή: CASCADED PID (δύο βρόχοι ανά άξονα)
  - Στάση    : PID γωνίας -> επιθυμητός ρυθμός -> PID ρυθμού -> διαφορική ροπή
  - Ύψος     : PID ύψους  -> επιθυμητό vz      -> PID vz     -> collective thrust
  - Θέση XY  : PID θέσης  -> επιθυμητό vxy     -> PID vxy    -> setpoint κλίσης
               (τροφοδοτεί τον εξωτερικό βρόχο στάσης -> cascade 3 επιπέδων)

Μηχανή καταστάσεων 3 φάσεων:
  1. CLIMB   : σταθεροποίηση + ανοδική ώθηση για απόθεμα ύψους
  2. FLIP    : εντολή σταθερού ρυθμού pitch (μόνο ο εσωτερικός βρόχος ρυθμού)
  3. RECOVER : επαναφορά στάσης, κράτημα ύψους και φρενάρισμα της οριζόντιας
               παράσυρσης που δημιουργεί το ίδιο το flip
"""

import numpy as np

# Ο ελεγκτής ΔΕΝ εξαρτάται από το περιβάλλον: χρειάζεται μόνο numpy.
# Το QuadFlipEnv εισάγεται μόνο όταν το αρχείο τρέχει αυτόνομα (βλ. __main__),
# ώστε το `from algor import FlipController` να δουλεύει από οπουδήποτε.


# ----------------------------------------------------------------------------------
# Διάταξη παρατήρησης (ένα frame = 18 τιμές, το env επιστρέφει 3 stacked frames = 54)
#   [0:3]   rel_pos_body  (target - pos, εκφρασμένο στο body frame)
#   [3:7]   quat [w, x, y, z]   (κανονικοποιημένο ώστε w >= 0)
#   [7:10]  vel_body
#   [10:13] omega [p, q, r]  (body frame, rad/s)
#   [13:17] prev_action
#   [17]    flip_progress
# ----------------------------------------------------------------------------------
SINGLE_OBS_DIM = 18
SLICE_RELPOS = slice(0, 3)
SLICE_QUAT = slice(3, 7)
SLICE_VEL = slice(7, 10)
SLICE_OMEGA = slice(10, 13)

# Διάταξη κινητήρων στο quadcopter.xml: [FL, FR, RR, RL]
#   FL(+x,+y)  FR(+x,-y)  RR(-x,-y)  RL(-x,+y)
# Ροπή από δύναμη F στη θέση r:  tau = r x F,  με F = (0, 0, F)
#   tau_x = +r_y * F     -> roll  : [+1, -1, -1, +1]
#   tau_y = -r_x * F     -> pitch : [-1, -1, +1, +1]   (θετικό q => μύτη κάτω, front-flip)
#   tau_z από το gear    -> yaw   : [-1, +1, -1, +1]
MIX_ROLL = np.array([+1.0, -1.0, -1.0, +1.0])
MIX_PITCH = np.array([-1.0, -1.0, +1.0, +1.0])
MIX_YAW = np.array([-1.0, +1.0, -1.0, +1.0])


def quat_to_dcm(q):
    """Quaternion [w, x, y, z] -> πίνακας στροφής body->world."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def pitch_from_quat(q):
    """Γωνία pitch στο (-pi, pi]. Αναλλοίωτη στην κανονικοποίηση q -> -q."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * y + z * x), 1.0 - 2.0 * (x * x + y * y))


def roll_from_dcm(R):
    """Roll από τον DCM. Έγκυρο μόνο κοντά σε όρθια στάση (φάσεις CLIMB/RECOVER)."""
    return np.arctan2(R[2, 1], R[2, 2])


class PIDController:
    """PID με anti-windup (clamp ολοκληρωτή) και παράγωγο επί της μέτρησης."""

    def __init__(self, kp, ki, kd, limits=(-1.0, 1.0), integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limits = limits
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_measurement = None

    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement

        # Παράγωγος επί της μέτρησης: αποφεύγει το "derivative kick" σε άλματα setpoint.
        if self.prev_measurement is None:
            derivative = 0.0
        else:
            derivative = -(measurement - self.prev_measurement) / dt
        self.prev_measurement = measurement

        raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = float(np.clip(raw, self.limits[0], self.limits[1]))

        # Anti-windup: ολοκληρώνουμε μόνο όταν δεν είμαστε σε κορεσμό προς την ίδια φορά.
        if abs(raw - output) < 1e-9 or (raw - output) * error < 0.0:
            self.integral = float(np.clip(
                self.integral + error * dt, -self.integral_limit, self.integral_limit
            ))
        return output


class FlipController:
    """
    Cascaded-PID controller για ένα πλήρες 360° flip γύρω από τον άξονα pitch.
    """

    # --- Παράμετροι φάσεων ---------------------------------------------------------
    T_CLIMB = 0.30          # s, διάρκεια ανοδικής ώθησης πριν το flip
    CLIMB_COLLECTIVE = 0.55  # normalized collective κατά την ανάβαση
    FLIP_RATE = 22.0        # rad/s, επιθυμητός ρυθμός pitch κατά το flip
    FLIP_TARGET = 2.0 * np.pi
    ALPHA_BRAKE = 250.0     # rad/s^2, συντηρητική εκτίμηση γωνιακής επιτάχυνσης πέδησης
    FLIP_TIMEOUT = 1.5      # s, ασφαλιστική έξοδος από τη φάση FLIP
    MAX_ATT_RATE = 12.0     # rad/s, όριο εντολής ρυθμού από τον βρόχο γωνίας
    MAX_VZ = 2.0            # m/s, όριο εντολής ταχύτητας από τον βρόχο ύψους
    MAX_VXY = 1.5           # m/s, όριο εντολής οριζόντιας ταχύτητας
    MAX_TILT = 0.65         # rad (~37°), όριο εντολής κλίσης από τον βρόχο θέσης XY

    def __init__(self, dt=0.01):
        self.dt = float(dt)

        # --- Βρόχος στάσης: εξωτερικός (γωνία) -> εσωτερικός (ρυθμός) ---------------
        # Εξωτερικοί: καθαρά αναλογικοί, παράγουν εντολή γωνιακής ταχύτητας.
        self.pid_pitch_angle = PIDController(kp=8.0, ki=0.0, kd=0.0,
                                             limits=(-self.MAX_ATT_RATE, self.MAX_ATT_RATE))
        self.pid_roll_angle = PIDController(kp=8.0, ki=0.0, kd=0.0,
                                            limits=(-self.MAX_ATT_RATE, self.MAX_ATT_RATE))
        # Εσωτερικοί: παράγουν διαφορική εντολή κινητήρων (normalized).
        # kp = 0.05 -> σφάλμα ~20 rad/s κορεσμοί στο 1.0 (πλήρης εξουσία στο flip).
        self.pid_pitch_rate = PIDController(kp=0.050, ki=0.02, kd=0.0018,
                                            limits=(-1.0, 1.0), integral_limit=5.0)
        self.pid_roll_rate = PIDController(kp=0.050, ki=0.02, kd=0.0018,
                                           limits=(-1.0, 1.0), integral_limit=5.0)
        self.pid_yaw_rate = PIDController(kp=0.050, ki=0.01, kd=0.0,
                                          limits=(-0.5, 0.5), integral_limit=5.0)

        # --- Βρόχος ύψους: εξωτερικός (z) -> εσωτερικός (vz) -----------------------
        self.pid_z = PIDController(kp=2.0, ki=0.0, kd=0.0, limits=(-self.MAX_VZ, self.MAX_VZ))
        self.pid_vz = PIDController(kp=0.90, ki=0.40, kd=0.02,
                                    limits=(-1.0, 1.0), integral_limit=2.0)

        # --- Βρόχος θέσης XY: θέση -> ταχύτητα -> γωνία κλίσης ---------------------
        # Το flip δίνει οριζόντια ταχύτητα στο drone. Χωρίς αυτόν τον βρόχο
        # παρασύρεται και βγαίνει εκτός arena (|xy| > 1.5 m -> terminated).
        self.pid_x = PIDController(kp=3.5, ki=0.0, kd=0.0, limits=(-self.MAX_VXY, self.MAX_VXY))
        self.pid_y = PIDController(kp=3.5, ki=0.0, kd=0.0, limits=(-self.MAX_VXY, self.MAX_VXY))
        self.pid_vx = PIDController(kp=0.80, ki=0.30, kd=0.06,
                                    limits=(-self.MAX_TILT, self.MAX_TILT), integral_limit=2.0)
        self.pid_vy = PIDController(kp=0.80, ki=0.30, kd=0.06,
                                    limits=(-self.MAX_TILT, self.MAX_TILT), integral_limit=2.0)

        self.reset()

    # ------------------------------------------------------------------------------
    def reset(self, seed=None):
        self.t = 0.0
        self.phase = "CLIMB"
        self.phase_t = 0.0
        self.flip_angle = 0.0   # δικό μας ολοκλήρωμα του q (ανεξάρτητο από το env)
        for pid in (self.pid_pitch_angle, self.pid_roll_angle, self.pid_pitch_rate,
                    self.pid_roll_rate, self.pid_yaw_rate, self.pid_z, self.pid_vz,
                    self.pid_x, self.pid_y, self.pid_vx, self.pid_vy):
            pid.reset()

    # ------------------------------------------------------------------------------
    @staticmethod
    def _unpack(observation):
        """Κρατάει μόνο το ΤΕΛΕΥΤΑΙΟ (τρέχον) frame από το stacked observation."""
        obs = np.asarray(observation, dtype=np.float64).ravel()
        cur = obs[-SINGLE_OBS_DIM:]
        quat = cur[SLICE_QUAT]
        R = quat_to_dcm(quat)
        return {
            "R": R,
            "quat": quat,
            # rel_pos_body είναι (target - pos) στο body frame -> γύρνα το στο world.
            "pos_err_world": R @ cur[SLICE_RELPOS],
            "vel_world": R @ cur[SLICE_VEL],
            "omega": cur[SLICE_OMEGA],
        }

    def _altitude_collective(self, s):
        """Cascade ύψους: σφάλμα z -> επιθυμητό vz -> collective εντολή."""
        z_err = float(s["pos_err_world"][2])          # target_z - z
        vz = float(s["vel_world"][2])
        # Ο εξωτερικός βρόχος δουλεύει με σφάλμα, οπότε setpoint=z_err, measurement=0.
        vz_des = self.pid_z.compute(z_err, 0.0, self.dt)
        return self.pid_vz.compute(vz_des, vz, self.dt)

    def _position_tilt(self, s):
        """
        Cascade θέσης XY: σφάλμα θέσης -> επιθυμητή ταχύτητα -> επιθυμητή κλίση.

        Για μικρές γωνίες, η επιτάχυνση από την ώση είναι:
            a_x ≈ +g * sin(pitch)   (R[0,2] = +sin(theta))
            a_y ≈ -g * sin(roll)    (R[1,2] = -sin(phi))
        Άρα το roll setpoint παίρνει αντίθετο πρόσημο από το y.
        """
        err = s["pos_err_world"]     # (target - pos) στο world frame
        vel = s["vel_world"]

        vx_des = self.pid_x.compute(float(err[0]), 0.0, self.dt)
        vy_des = self.pid_y.compute(float(err[1]), 0.0, self.dt)

        pitch_des = self.pid_vx.compute(vx_des, float(vel[0]), self.dt)
        roll_des = -self.pid_vy.compute(vy_des, float(vel[1]), self.dt)
        return roll_des, pitch_des

    def _attitude_hold(self, s, roll_des=0.0, pitch_des=0.0):
        """Cascade στάσης: γωνίες -> ρυθμοί -> διαφορικές εντολές (roll, pitch, yaw)."""
        R, omega = s["R"], s["omega"]
        pitch = pitch_from_quat(s["quat"])
        roll = roll_from_dcm(R)

        q_des = self.pid_pitch_angle.compute(pitch_des, pitch, self.dt)
        p_des = self.pid_roll_angle.compute(roll_des, roll, self.dt)

        u_pitch = self.pid_pitch_rate.compute(q_des, omega[1], self.dt)
        u_roll = self.pid_roll_rate.compute(p_des, omega[0], self.dt)
        u_yaw = self.pid_yaw_rate.compute(0.0, omega[2], self.dt)
        return u_roll, u_pitch, u_yaw

    # ------------------------------------------------------------------------------
    def act(self, observation):
        s = self._unpack(observation)
        omega = s["omega"]
        q_meas = float(omega[1])

        # Ολοκλήρωση του ρυθμού pitch: συνεχής γωνία χωρίς wrap στο +-pi.
        self.flip_angle += q_meas * self.dt

        # ---------------- Μηχανή καταστάσεων ---------------------------------------
        if self.phase == "CLIMB" and self.phase_t >= self.T_CLIMB:
            self._switch("FLIP")
        elif self.phase == "FLIP":
            remaining = self.FLIP_TARGET - self.flip_angle
            # Απόσταση πέδησης: όση γωνία χρειάζεται για να μηδενίσει ο ρυθμός.
            brake_lead = max(0.35, q_meas ** 2 / (2.0 * self.ALPHA_BRAKE))
            if remaining <= brake_lead or self.phase_t >= self.FLIP_TIMEOUT:
                self._switch("RECOVER")

        # ---------------- Έξοδοι ανά φάση ------------------------------------------
        if self.phase == "CLIMB":
            # Σταθεροποίηση στάσης + σταθερή ανοδική ώθηση (απόθεμα ύψους για το flip).
            u_roll, u_pitch, u_yaw = self._attitude_hold(s)
            collective = self.CLIMB_COLLECTIVE

        elif self.phase == "FLIP":
            # Μόνο ο ΕΣΩΤΕΡΙΚΟΣ βρόχος: εντολή σταθερού ρυθμού pitch.
            # Ο βρόχος γωνίας είναι άχρηστος εδώ (θα πάλευε να κρατήσει pitch=0).
            u_pitch = self.pid_pitch_rate.compute(self.FLIP_RATE, q_meas, self.dt)
            # Roll/yaw: μόνο απόσβεση ρυθμού -> κρατάει το flip καθαρά στον άξονα y.
            u_roll = self.pid_roll_rate.compute(0.0, omega[0], self.dt)
            u_yaw = self.pid_yaw_rate.compute(0.0, omega[2], self.dt)
            # Το διάνυσμα ώσης περιστρέφεται μαζί με το σώμα: δίνουμε γκάζι μόνο
            # όσο δείχνει προς τα πάνω (R[2,2] = cos της κλίσης), αλλιώς κρατάμε
            # ουδέτερο collective ώστε να υπάρχει περιθώριο ροπής και προς τις δύο φορές.
            up = float(s["R"][2, 2])
            collective = 0.70 * up if up > 0.25 else 0.0

        else:  # RECOVER
            # Πλήρης cascade τριών επιπέδων: θέση XY -> κλίση -> ρυθμός -> κινητήρες.
            roll_des, pitch_des = self._position_tilt(s)
            u_roll, u_pitch, u_yaw = self._attitude_hold(s, roll_des, pitch_des)
            collective = self._altitude_collective(s)

        # ---------------- Mixer -----------------------------------------------------
        action = (collective
                  + u_roll * MIX_ROLL
                  + u_pitch * MIX_PITCH
                  + u_yaw * MIX_YAW)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        self.t += self.dt
        self.phase_t += self.dt

        info = {
            "phase": self.phase,
            "pitch_raw": pitch_from_quat(s["quat"]),
            "flip_angle": self.flip_angle,
            "omega": omega.copy(),
            "collective": collective,
            "u_pitch": u_pitch,
        }
        return action, info

    def _switch(self, phase):
        self.phase = phase
        self.phase_t = 0.0
        # Καθαρισμός ολοκληρωτών στην αλλαγή φάσης (αλλιώς μεταφέρεται windup).
        self.pid_pitch_rate.reset()
        self.pid_roll_rate.reset()
        self.pid_pitch_angle.reset()
        self.pid_roll_angle.reset()
        if phase == "RECOVER":
            self.pid_z.reset()
            self.pid_vz.reset()
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_vx.reset()
            self.pid_vy.reset()


# ======================================================================================
# Ανεξάρτητος γεωμετρικός ανιχνευτής επιτυχίας (δεν χρησιμοποιεί το reward)
# ======================================================================================
def evaluate_flip_success(pitch_hist, z_hist, ang_vel_hist, min_altitude_allowed=0.15):
    started_upright = abs(pitch_hist[0]) < 0.2
    passed_inverted = any(abs(abs(p) - np.pi) < 0.5 for p in pitch_hist)
    completed_rotation = max(pitch_hist) > (2.0 * np.pi - 0.8)

    min_altitude = min(z_hist)
    no_ground_collision = min_altitude > min_altitude_allowed

    final_ang_vel = float(np.linalg.norm(ang_vel_hist[-1]))
    small_final_vel = final_ang_vel < 1.0

    final_upright = abs(pitch_hist[-1] - 2.0 * np.pi) < 0.5 or abs(pitch_hist[-1]) < 0.5

    is_successful = (
        started_upright
        and passed_inverted
        and completed_rotation
        and final_upright
        and no_ground_collision
        and small_final_vel
    )

    print("\n--- ΑΞΙΟΛΟΓΗΣΗ ΕΠΙΤΥΧΙΑΣ (SUCCESS DETECTOR) ---")
    print(f"1. Ξεκίνησε όρθιο            : {started_upright}")
    print(f"2. Πέρασε από ανάποδη θέση   : {passed_inverted}")
    print(f"3. Ολοκλήρωσε 360°           : {completed_rotation} (max pitch = {max(pitch_hist):.2f} rad)")
    print(f"4. Επέστρεψε όρθιο           : {final_upright} (τελικό pitch = {pitch_hist[-1]:.2f} rad)")
    print(f"5. Αποφυγή εδάφους           : {no_ground_collision} (ελάχ. ύψος = {min_altitude:.2f} m)")
    print(f"6. Μικρή τελική γων. ταχύτητα: {small_final_vel} ({final_ang_vel:.2f} rad/s)")
    print(f">> ΑΠΟΤΕΛΕΣΜΑ: {'ΕΠΙΤΥΧΙΑ' if is_successful else 'ΑΠΟΤΥΧΙΑ'}")
    return is_successful


def run_episode(env, controller, max_steps=None, verbose=False, seed=0):
    obs, info = env.reset(seed=seed)
    controller.reset(seed=seed)

    if max_steps is None:
        max_steps = env.max_steps

    pitch_hist_raw = []
    z_hist = [float(info["position"][2])]
    ang_vel_hist = []
    total_reward = 0.0

    for t in range(max_steps):
        action, dbg = controller.act(obs)
        pitch_hist_raw.append(dbg["pitch_raw"])
        ang_vel_hist.append(dbg["omega"])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        z_hist.append(float(info["position"][2]))

        if verbose and t % 10 == 0:
            print(f"t={t * env.dt:5.2f}s [{dbg['phase']:7s}] "
                  f"z={info['position'][2]:5.2f} "
                  f"pitch={dbg['pitch_raw']:+6.2f} "
                  f"flip={dbg['flip_angle']:+6.2f} "
                  f"q={dbg['omega'][1]:+7.2f} "
                  f"col={dbg['collective']:+5.2f}")

        if terminated or truncated:
            print(f"\nΤέλος επεισοδίου στο t={t * env.dt:.2f}s "
                  f"(terminated={terminated}, truncated={truncated})")
            break

    pitch_hist = np.unwrap(pitch_hist_raw).tolist()  # unwrap ΜΙΑ φορά, στο τέλος
    return pitch_hist, z_hist, ang_vel_hist, total_reward


if __name__ == "__main__":
    import os
    import sys

    # Το περιβάλλον βρίσκεται δύο επίπεδα πάνω: <repo>/Simulation
    _REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for _p in (_REPO, os.path.join(_REPO, "Simulation")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from Simulation.quad_flip_env import QuadFlipEnv

    # Αρχικό benchmark του project: χωρίς άνεμο, θόρυβο ή τυχαία αρχική κατάσταση.
    env = QuadFlipEnv(
        obs_noise=False,
        random_wind=False,
        random_initial_state=False,
        random_battery=False,
    )
    controller = FlipController(dt=env.dt)

    pitch_hist, z_hist, ang_vel_hist, total_reward = run_episode(env, controller, verbose=True)
    env.close()

    print(f"\nΣυνολικό reward: {total_reward:.1f}")
    evaluate_flip_success(pitch_hist, z_hist, ang_vel_hist)
