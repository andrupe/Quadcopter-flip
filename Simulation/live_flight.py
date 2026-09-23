# -*- coding: utf-8 -*-
"""
Live flight: the real plant, the real 1 kHz rate loop, and a PILOT-CHOSEN reference for
the trained encoder+PPO stack - switchable mid-flight, from a GUI or the controller.

WHAT THIS IS
------------
`evaluate.py` flies a checkpoint down a trajectory the SAMPLER generated.
`manual_flight.py` gives the sticks to the pilot and never touches the policy.
This program combines them into one live flight, with one rule:

        the REFERENCE (the "trajectory") is decided by the human, live.

Three ways to fly, switchable at any instant without resetting anything:

    MANUAL     (ACRO or ASSISTED)   the pilot flies the plant directly, exactly as in
                                    manual_flight.py - same outer loop, same rate PID.
    POLICY     (your sticks)        the trained encoder+PPO flies, tracking a reference
                                    generated from YOUR sticks (velocity + climb +
                                    heading). Centre the sticks and the reference settles
                                    to a point - a hover - and the policy holds it. The
                                    sampler that generates training references is
                                    replaced by you; nothing else about the policy path
                                    changes.
    TRAJECTORY (one manoeuvre)      the trained policy executes a training-distribution
                                    manoeuvre - a flip, an orbit, ... - relocated to start
                                    from wherever you are hovering (the same
                                    flatness-consistent reference the policy was trained
                                    on, just shifted and re-aimed). When it finishes it
                                    hands back to POLICY hover; at any moment you can
                                    take over with the controller.

The intended live-test workflow is exactly the one that was asked for:

    pick up the controller -> hover it yourself (MANUAL)
      -> GUI: POLICY (the policy takes the hover, reference = your sticks)
      -> GUI: FLIP (the policy executes the flip from your hover)
      -> it settles -> POLICY hover -> GUI: MANUAL (you take back control)

WHAT IS *NOT* TOUCHED (deliberately)
------------------------------------
`quad_flip_env.py`, `trajectories.py`, `encoder/*`, `actor_input.py`, `train.py` and
`evaluate.py` are unmodified - the training task, the acrobatics that run WITHOUT a human
and the evaluation path are bit-identical to what they were. This program only supplies a
different REFERENCE SOURCE (a duck-typed `traj.sample(t)` - see `live_target.py`), relaxes
the fixed training flight sphere for live flying (`live_policy_env.py`), and adds the
input/handover/GUI layers around them. The policy is driven through `ActorInput` and the
env's own `step()`, i.e. the identical deployment path evaluate.py uses.

RADIO PATH
----------
Every command this program produces - manual or policy - is converted once into a
`flight_link.RadioSetpoint` (rate + thrust in vehicle units) and handed to a
`CommandLink`. Today that link is `NullLink` (sim only). `--radio` selects the
`CrazyradioLink` placeholder, which documents the CRTP packet and refuses to pretend.
Nothing above the link changes when it is implemented.

HOW TO RUN
----------
    .venv/bin/python Simulation/live_flight.py            (VS Code: click Run)

The panel opens in its own process (tkinter cannot own a window under mjpython). Keys in
the 3D window: 1 MANUAL-acro, 2 MANUAL-assisted, 3 POLICY(human); 4 hover-hold, 5 flip,
6 orbit, 7 figure-8, 8 lissajous, 9 slalom, 0 waypoints; ENTER respawn; ESC quit. Pad:
`-` hand over / take back, `+` hover now, X match velocity (manual) / hover (policy),
bottom face button level/hover, B respawn.

WHERE THE FLIGHT STARTS
-----------------------
On the FLOOR, resting on its landing legs, the way a real quad waits for its pilot - the
floor is a RUNWAY until the vehicle has climbed above `LiveFlightEnv.GROUND_RELEASE_Z`
(0.20 m) for the first time, and a crash surface after that. So the flight begins at the
bottom: spool up and take off (GAME mode: hold ZR; ACRO: SPACE/R raises the collective),
then hand over to the policy whenever you are airborne.

The pose is tweakable - `START_POS` / `START_YAW_DEG` in the config block, or:

    --spawn X Y Z        metres, e.g. `--spawn 0.5 -0.5 0.015` (a ground start elsewhere)
    --spawn-yaw DEG      heading, e.g. `--spawn-yaw 90`

A start z ABOVE the release height spawns straight into a hover instead (the previous
behaviour, and what the checker's hover-to-hover stack tests use: `--spawn 0 0 1.2`).
Respawn (ENTER, pad B, or the panel) always returns to this same pose.

THE FLIP BUTTON (top-right shoulder, R/RB)
------------------------------------------
Works in BOTH flight programs and in both manual and policy flying: hold it and the left
stick rolls/pitches the vehicle directly (right = roll right, left = roll left) while the
COLLECTIVE IS FLOWN FOR THE PILOT on the trained flip's own vertical profile - a pop with
the body upright, ZERO thrust through the rotation (thrust acts along body z, so firing it
inverted slams the vehicle into the floor), then an arrest and a banked altitude hold. The
rotation only starts after the 0.25 s pop, so the pop goes UP and not sideways. In POLICY
mode the policy steps aside for the duration (the encoder keeps being fed, so it does not
cold-start when you let go); release the button and it takes the vehicle back, anchored to
wherever the roll left it.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -- GUI-thread guard (identical to evaluate.py / manual_flight.py) ---------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "Simulation" else _THIS_DIR
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _is_gui_thread() -> bool:
    if sys.platform != "darwin":
        return True
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        return bool(libc.pthread_main_np())
    except Exception:
        return True


_WANTS_WINDOW = _is_gui_thread()
if sys.platform == "darwin" and not _WANTS_WINDOW:
    import matplotlib
    matplotlib.use("Agg")          # the telemetry PNG is still written
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

import manual_flight as mf  # noqa: E402  (manual loop, HUD style, plotting)
from quad_flip_env import QuadFlipEnv  # noqa: E402
from actor_input import ActorInput, load_checkpoint, read_checkpoint_arch  # noqa: E402
from live_policy_env import LiveFlightEnv  # noqa: E402
from live_target import HumanTarget, ShiftedTrajectory, yaw_of  # noqa: E402
from pad_mapper import PadMapper  # noqa: E402
from flight_link import NullLink, RadioSetpoint, make_link  # noqa: E402
from flight_gui import DEFAULT_PORT, GuiServer  # noqa: E402
import gamepad as gp  # noqa: E402

# ======================================================================================
# CONFIGURATION (edit here, or drive everything live from the panel)
# ======================================================================================
MODEL_NAME: str = "latest"            # checkpoint in logs/ ("latest" = newest .zip)
ENCODER_CHECKPOINT: str = os.path.join(_PROJECT_ROOT, "logs", "encoder_gru.pt")
DR_LEVEL: float = 0.0                 # 0 = nominal plant (manual sandbox behaviour)
TELEMETRY_PATH: str = os.path.join(_PROJECT_ROOT, "live_flight_telemetry.png")
PAD_CALIBRATION_PATH: str = os.path.join(_PROJECT_ROOT, "logs", "pad_calibration.json")
PAD_ENABLE: bool = True
START_MODE: str = "manual_acro"       # manual_acro | manual_assisted | policy_human
GUI_ENABLE: bool = True
GUI_PORT: int = DEFAULT_PORT

# --- START STATE (tweak these, or use --spawn / --spawn-yaw) ---------------------------
# The flight begins ON THE FLOOR, resting on its landing legs, the way a real quad waits
# for its pilot: the floor is a RUNWAY until the vehicle has climbed above
# `LiveFlightEnv.GROUND_RELEASE_Z` (0.20 m) once, and a crash after that. Spool up and
# take off (GAME mode: hold ZR; ACRO: SPACE/R raises the latched collective).
#
# Any pose can be given: START_POS is (x, y, z) in metres and START_YAW_DEG the heading.
# A z ABOVE the release height spawns straight into a hover instead (the old behaviour -
# e.g. `--spawn 0 0 1.2`). Respawn (ENTER, pad B, or the panel) returns to this pose.
GROUND_REST_Z: float = 0.015          # m: legs are spheres at -0.01 r=0.003 under the body
START_POS: Tuple[float, float, float] = (0.0, 0.0, GROUND_REST_Z)
START_YAW_DEG: float = 0.0

# Policy-mode stick scaling: full stick = this reference velocity / yaw rate.
#
# Deliberately modest. The policy was trained on references inside a 1.5 m x 1.5 m box
# (see TrajectoryConfig.bounds_xy) at 1.2-2.0 m altitude, and a reference that TRANSLATES
# for seconds at a time leaves that box however gently it is flown - the trained task is
# oscillatory tracking, not cruise. Inside the box the policy tracks a live reference to
# ~0.2-0.4 m; far outside it, it still flies (and holds a hover) but its corrections get
# lazy, which is a property of the checkpoint's training envelope, not of this layer.
# Manual mode has no such limit - the pilot is the pilot.
POLICY_STICK_V: float = 1.5           # m/s
POLICY_STICK_UP: float = 1.0          # m/s
POLICY_STICK_YAW: float = 1.0         # rad/s (the reference cap is live_target.HUMAN_YAW_RATE_MAX)

MODE_MANUAL_ACRO = "MANUAL (acro)"
MODE_MANUAL_ASSISTED = "MANUAL (assisted)"
MODE_MANUAL_GAME = "MANUAL (game)"
MODE_POLICY_HUMAN = "POLICY (your sticks)"
MODE_POLICY_TRAJ = "POLICY (trajectory)"

# Telemetry codes. Indexing a literal dict with `self.mode` is how a flight that had just
# gone perfectly well died with `KeyError: 'MANUAL (game)'` the moment a new mode was
# added: the log must never be able to end a flight, so unknown modes are logged as -1.
MODE_CODES = {MODE_MANUAL_GAME: 0, MODE_MANUAL_ACRO: 1, MODE_MANUAL_ASSISTED: 2,
              MODE_POLICY_HUMAN: 3, MODE_POLICY_TRAJ: 4}
MODE_LABELS = {code: name for name, code in MODE_CODES.items()}
MODE_LABELS[-1] = "other"

# Pad buttons are defined by the manual sandbox (gamepad.py only publishes raw HID
# button numbers). Importing them by name keeps one source of truth.
PAD_BTN_MODE = mf.PAD_BTN_MODE            # left face button: manual control set / hover now
PAD_BTN_MODE_ALT = mf.PAD_BTN_MODE_ALT    # '-' : hand over to the policy / take it back
PAD_BTN_MATCH = mf.PAD_BTN_MATCH_ALT      # '+' : hover now / release the trajectory
PAD_BTN_RESPAWN = mf.PAD_BTN_RESPAWN
PAD_BTN_CLIMB = mf.PAD_BTN_CLIMB          # ZR: climb
PAD_BTN_DESCEND = mf.PAD_BTN_DESCEND      # ZL: descend
PAD_BTN_FLIP = mf.PAD_BTN_FLIP            # top-right shoulder: flip (GAME mode)

# Policy-mode keyboard, for flying without a pad. The viewer reports presses only (no
# key-up), so these are LATCHED like every other keyboard channel in this project:
# W/S = fwd +-0.5 m/s, A/D = left/right +-0.5, R/F = up/down +-0.5, Q/E = yaw +-0.25 rad/s,
# SPACE = clear everything (hover). The commanded speed is shown in the HUD.
POLICY_KEY_V_STEP: float = 0.5        # m/s per press
POLICY_KEY_YAW_STEP: float = 0.25     # rad/s per press

_KEY_TRAJ = {ord("4"): "hover", ord("5"): "flip", ord("6"): "orbit", ord("7"): "figure8",
             ord("8"): "lissajous", ord("9"): "slalom", ord("0"): "waypoints"}
_KEY_MODE = {ord("1"): MODE_MANUAL_GAME, ord("2"): MODE_MANUAL_ACRO,
             ord("3"): MODE_POLICY_HUMAN}


def _resolve_model(name: str) -> str:
    """Same lookup order as evaluate.py, plus 'latest' = newest .zip in logs/."""
    if os.path.isfile(name):
        return os.path.abspath(name)
    candidate = os.path.join(_PROJECT_ROOT, name)
    if os.path.isfile(candidate):
        return candidate
    if os.path.isfile(candidate + ".zip"):
        return candidate + ".zip"
    logs = os.path.join(_PROJECT_ROOT, "logs")
    if name.lower() in ("latest", "auto") and os.path.isdir(logs):
        zips = [os.path.join(logs, f) for f in os.listdir(logs) if f.endswith(".zip")]
        if zips:
            # Prefer the periodic rl_model_<steps>_steps.zip checkpoints when present.
            preferred = [z for z in zips if os.path.basename(z).startswith("rl_model_")]
            pool = preferred or zips
            return max(pool, key=os.path.getmtime)
    return candidate


def _python_for_gui() -> str:
    """A python that can open a tkinter window (mjpython cannot own one)."""
    import shutil
    for name in ("python3", "python"):
        sibling = os.path.join(os.path.dirname(sys.executable), name)
        if os.path.isfile(sibling):
            return sibling
    return shutil.which("python3") or sys.executable


class LiveFlight:
    """One live flight: plant + policy + pad + panel + link, with live handovers."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.running = True
        self.messages: List[str] = []
        self.crashed = False
        self.end_reason = "quit by the pilot"
        self.last_thrust = 0.0                      # N, last command applied to the plant
        self.last_omega = np.zeros(3)
        self.last_source = "manual"
        self.traj_kind: Optional[str] = None
        self.traj_t0 = 0.0
        self.traj_duration = 0.0
        self._key_cmd = np.zeros(4)                 # fwd, left, up, yaw (latched keyboard)
        self._last_cmd = (0.0, 0.0, 0.0, 0.0)       # fwd, left, up, yaw (whatever was set last)
        self._policy_flip_active = False            # the flip button, in POLICY mode
        self._policy_flip_t = 0.0                   # time since it engaged (the pop clock)
        self._steps_done = 0

        # -- environment: the training task, live-flight termination ---------------------
        # A CLEAN spawn (no position/velocity/attitude kicks, no wind, no battery sag) is
        # right for a live flight: the pilot picks the vehicle up at a known hover and the
        # only disturbance is what they or the policy do. Domain randomisation stays
        # available through --dr for the same reason it exists in training.
        self.env = LiveFlightEnv(
            telemetry=True, maneuver="hover",
            random_initial_state=False, random_wind=True, random_battery=True,
        )
        self.env.set_dr_level(float(args.dr))
        self.env.reset(options={"maneuver": "hover"})
        self.dt = float(self.env.dt)
        assert abs(self.dt - mf.DT) < 1e-12, "manual loop and env must share a control period"

        # -- where the flight starts (config block above, or --spawn / --spawn-yaw) ------
        spawn = getattr(args, "spawn", None)
        self.start_pos = np.asarray(START_POS if spawn is None else spawn, dtype=np.float64)
        yaw_arg = getattr(args, "spawn_yaw", None)
        self.start_yaw = math.radians(float(START_YAW_DEG if yaw_arg is None else yaw_arg))

        # -- the pilot's reference source ------------------------------------------------
        yaw0 = yaw_of(self.env.quad.dcm)
        self.human = HumanTarget(self.env.quad.pos, yaw0=yaw0)
        self.env.set_external_reference(self.human)

        # -- manual control (shared with the manual sandbox, same plant object) ----------
        self.cmd = mf.ManualCommand(spawn_z=float(self.start_pos[2]))
        self.ctrl = mf.ManualFlightController(self.env.quad, self.cmd)
        self.cmd.throttle = self.ctrl.hover_throttle
        mf.PAD_INVERT_LEFT_X = False       # the mapper hands the manual loop LOGICAL sticks
        mf.PAD_INVERT_LEFT_Y = False
        mf.PAD_INVERT_RIGHT_X = False
        mf.PAD_INVERT_RIGHT_Y = False

        # -- pad ------------------------------------------------------------------------
        self.mapper = PadMapper.load(PAD_CALIBRATION_PATH, layout="ow")
        self.pad = None
        if PAD_ENABLE and not args.no_pad:
            self.pad = gp.HIDGamepad()
            # NOTE: `start()` returns None (it spawns the reader thread), so it must NOT
            # be used as a truth value - testing it that way silently dropped every
            # controller, which is exactly how "the policy isn't tracking my sticks"
            # presented (the pad was never read at all). Availability is `available`,
            # reported by `describe()`/`error` afterwards.
            if self.pad.available:
                self.pad.start()
                # `HIDGamepad` keeps the device name privately (`PadState.name` is the
                # snapshot's copy), so read it defensively: a missing attribute here must
                # not take the flight program down before it starts.
                device = (getattr(self.pad, "name", None)
                          or getattr(self.pad, "_name", "")
                          or "device")
                self.messages.append(f"pad: {device} reader started")
            else:
                self.messages.append(f"pad: not available ({self.pad.error or 'no device'}); keyboard only")
                self.pad = None
        self.pad_state = None

        # -- policy stack (encoder + PPO), loaded exactly like evaluate.py ---------------
        self.policy_ok = False
        self.encoder_on = False
        self.vec_norm = None
        self.actor_input = None
        self.model = None
        self.obs = np.zeros(self.env.total_obs_dim, dtype=np.float32)
        if not args.no_policy:
            self._load_policy(args.model)
        if self.policy_ok:
            self.obs = self.env._get_stacked_obs()   # a fresh frame for the first predict

        # -- link (sim-only today; crazyradio arrives later) -----------------------------
        self.link = make_link(radio=bool(args.radio))
        try:
            self.link.open()
        except NotImplementedError as exc:
            # --radio today means "the wiring is in place, the driver is not": say so and
            # fly sim-only rather than dying in the middle of a command.
            self.link = NullLink()
            self.messages.append(f"radio link unavailable ({exc}) - flying sim-only")

        # -- put the vehicle on the START pose (the whole stack is anchored on it) -------
        self._place_at_start()
        self.messages.append(
            f"start: x {self.start_pos[0]:+.2f} y {self.start_pos[1]:+.2f} "
            f"z {self.start_pos[2]:.2f} m, heading {math.degrees(self.start_yaw):+.0f} deg"
            + ("  (on the floor - spool up and take off)"
               if self.env.ground_start else "  (airborne start)"))

        # -- panel ----------------------------------------------------------------------
        self.gui: Optional[GuiServer] = None
        self.gui_proc: Optional[subprocess.Popen] = None
        self.status_t = 0.0
        self.mode = ""
        start = {"manual_acro": MODE_MANUAL_ACRO,
                 "manual_game": MODE_MANUAL_GAME,
                 "manual_assisted": MODE_MANUAL_ASSISTED,
                 "policy_human": MODE_POLICY_HUMAN}.get(START_MODE, MODE_MANUAL_ACRO)
        self.enter_mode(start)
        if start != MODE_MANUAL_ACRO:
            self.messages.append(f"start mode from config: {start}")

        # -- telemetry -------------------------------------------------------------------
        self.log = mf._new_log()
        self.viewer = None

    # -- policy -------------------------------------------------------------------------
    def _load_policy(self, model_name: str) -> None:
        model_path = _resolve_model(model_name)
        if not os.path.isfile(model_path):
            self.messages.append(f"policy: no checkpoint at {model_path} - policy modes disabled")
            return
        try:
            actor_dim, _arch = read_checkpoint_arch(model_path)
            if actor_dim is None:
                self.messages.append(f"policy: {os.path.basename(model_path)} has no policy weights")
                return
            self.actor_input = ActorInput(actor_dim, encoder_path=ENCODER_CHECKPOINT)
            self.encoder_on = self.actor_input.injector is not None
            self.model = load_checkpoint(model_path)
            self.policy_ok = True
            self.messages.append(
                f"policy: {os.path.basename(model_path)} loaded "
                f"(actor {actor_dim}, encoder {'on' if self.encoder_on else 'off'})"
            )
        except Exception as exc:                     # missing encoder, old checkpoint, ...
            self.messages.append(f"policy: could not load {os.path.basename(model_path)} ({exc})")

        # VecNormalize statistics are a no-op with norm_obs=False but load them if present,
        # exactly as evaluate.py does.
        if self.policy_ok:
            stem = os.path.splitext(os.path.basename(model_path))[0]
            for cand in (os.path.join(os.path.dirname(model_path), f"{stem}_vecnormalize.pkl"),
                         os.path.join(_PROJECT_ROOT, f"{stem}_vecnormalize.pkl")):
                if os.path.isfile(cand):
                    try:
                        from evaluate import _StatsSpaceEnv
                        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                        dummy = DummyVecEnv([lambda: _StatsSpaceEnv(self.actor_input.training_obs_dim)])
                        self.vec_norm = VecNormalize.load(cand, dummy)
                        self.vec_norm.training = False
                        self.messages.append(f"policy: VecNormalize stats from {os.path.basename(cand)}")
                    except Exception as exc:
                        self.messages.append(f"policy: VecNormalize stats ignored ({exc})")
                    break

    # -- modes --------------------------------------------------------------------------
    def enter_mode(self, mode: str, announce: bool = True) -> None:
        """Hand over between manual flight and the policy, seamlessly."""
        if mode in (MODE_POLICY_HUMAN, MODE_POLICY_TRAJ) and not self.policy_ok:
            self.messages.append("policy modes unavailable (no checkpoint)")
            return
        was_policy = self.mode.startswith("POLICY")
        self._policy_flip_active = False          # never carry a flip across a handover
        if mode == MODE_MANUAL_ACRO:
            self.cmd.set_mode(mf.MODE_ACRO)
            self._manual_takeover(was_policy)
        elif mode == MODE_MANUAL_GAME:
            self.cmd.set_mode(mf.MODE_GAME)
            self._manual_takeover(was_policy)
        elif mode == MODE_MANUAL_ASSISTED:
            self.cmd.set_mode(mf.MODE_VELOCITY)
            self._manual_takeover(was_policy)
        else:
            if mode == MODE_POLICY_HUMAN:
                self.human.sync_to(self.env.quad.pos, v=None, yaw=yaw_of(self.env.quad.dcm))
                self.env.set_external_reference(self.human)
                self.messages.append(
                    "the policy tracks a live reference best inside its training volume "
                    "(~1 m around spawn); manual mode has no such limit"
                )
            self.env.adopt_state(self.env.quad.pos, self.env.quad.quat,
                                 self.env.quad.vel, self.env.quad.omega)
            if self.actor_input is not None:
                self.actor_input.reset()             # a handover is a new flight to the GRU
            self.obs = self.env._get_stacked_obs()
        self.mode = mode
        self.traj_kind = None
        if announce:
            self.messages.append(f"mode -> {mode}")

    def _manual_takeover(self, from_policy: bool) -> None:
        """
        Give the pilot the sticks from wherever the policy left the vehicle.

        The collective is inherited from the policy's own last command so the thrust does
        not step - BUT clamped to +-20% of hover trim. Unclamped, a take-back in the
        middle of a manoeuvre hands the pilot whatever the policy was doing (measured: 100%
        mid-flip, i.e. a rocket when upright and a slam when inverted). Inside the band the
        handover is continuity; outside it, the pilot gets a flyable aircraft and a message
        saying what changed.
        """
        if from_policy:
            hover = float(self.ctrl.hover_throttle)
            inherited = float(np.clip(self.last_thrust / self.env.quad.params["maxThr"], 0.0, 1.0))
            self.cmd.throttle = float(np.clip(inherited, 0.8 * hover, 1.2 * hover))
            self.messages.append(
                f"collective {100.0 * self.cmd.throttle:.0f}% "
                f"(policy was at {100.0 * inherited:.0f}%, hover trim {100.0 * hover:.0f}%; "
                f"SPACE/R adds, CTRL/F removes)"
            )
        self.cmd.rot_pad[:] = 0.0
        self.cmd._rot_budget[:] = 0.0
        self.cmd.match_velocity = False
        self.cmd.level_now = False
        self.env.set_external_reference(self.human)

    def launch_trajectory(self, kind: str) -> None:
        """Hand the policy a training manoeuvre relocated onto the current hover."""
        if not self.policy_ok:
            self.messages.append("trajectory launch needs a loaded policy")
            return
        if self.mode != MODE_POLICY_TRAJ and not self.mode.startswith("POLICY"):
            self.enter_mode(MODE_POLICY_HUMAN, announce=False)
        traj = self.env.sampler.sample(self.env.np_random, mass=float(self.env.quad.base_mass),
                                       kind=kind)
        got = str(getattr(traj.maneuver, "kind", "?"))
        if kind != "hover" and got != kind:
            self.messages.append(f"sampler could not build a {kind}; flying {got} instead")
        shifted = ShiftedTrajectory(traj, p0=self.env.quad.pos, yaw0=yaw_of(self.env.quad.dcm),
                                    t0=self.env.t)
        self.env.set_external_reference(shifted)
        if self.actor_input is not None:
            self.actor_input.reset()                 # a new reference: new latent state
        self.mode = MODE_POLICY_TRAJ
        self.traj_kind = got
        self.traj_t0 = self.env.t
        self.traj_duration = float(traj.duration)
        self.messages.append(f"policy is flying: {got} ({traj.duration:.1f} s from your hover)")

    # -- per-step: manual ---------------------------------------------------------------
    def _manual_step(self) -> None:
        pad = self.pad_state
        if pad is not None:
            self._pad_buttons_manual(pad)
            actions = self.cmd.apply_gamepad(self._pad_shim(pad), self.dt)
            for a in actions:
                if a.strip():
                    self.messages.append(a)
        thrust, omega_des = self.ctrl.update()
        self.env.quad.update(t=self.env.t, dt=self.dt, wind=self.env.wind,
                             rate_cmd=(thrust, omega_des), rate_pid=self.env.rate_pid,
                             gyro_bias=self.env.gyro_bias)
        self.env.t += self.dt
        self.env.steps += 1
        # Keep the sensor model and the (hover) reference alive while the pilot flies, so
        # a handover to the policy starts from a converged estimate and zero reference error.
        self.env.lighthouse.observe(self.env.quad.pos, self.env.quad.vel, self.env.quad.dcm,
                                    self.dt, self.env.np_random)
        self.human.sync_to(self.env.quad.pos, v=None, yaw=yaw_of(self.env.quad.dcm))
        self.env.ref = self.human.sample(self.env.t)
        # Ground latch: manual steps do not go through `env.step()`, so this is where a
        # ground-start flight becomes "airborne" - and the floor terminal again.
        self.env.ground_touch_is_fatal()
        self.last_thrust, self.last_omega, self.last_source = float(thrust), omega_des, "manual"

    def _policy_flip_step(self, pad) -> None:
        """The flip button, in POLICY mode: the pilot gets the sticks, the policy steps aside.

        The vehicle is still flown through the ENV'S OWN action interface - the same
        (throttle, roll, pitch, yaw-rate) channel the policy writes - so the plant, the 1 kHz
        rate loop, the mixer and the observation history all keep running exactly as they do
        under the policy. What the pilot asks for is the manual GAME flip: the left stick
        straight to body rates and the collective flown on the flip profile (pop up, ZERO
        thrust through the rotation, arrest) instead of having to be flown by hand.

        The encoder is still fed (its latent state has to be current when the policy takes
        back over) but `model.predict` is deliberately NOT called: this is the pilot's roll,
        the policy only gets it back afterwards - which it is trained to do (it recovers from
        inverted states).
        """
        roles = self.mapper.roles(pad)
        rate_max = math.radians(mf.GAME_FLIP_RATE_DEG)
        omega = np.array([
            roles["lx"] * rate_max,                                   # + = roll right
            roles["ly"] * rate_max,                                   # + = nose down
            -roles["heading"] * math.radians(mf.GAME_HEADING_RATE_DEG),
        ], dtype=np.float64)
        if not self._policy_flip_active:
            self._policy_flip_t = 0.0                                 # the pop starts now
        self._policy_flip_t += self.dt
        hover_thrust = float(self.env.hover_trim_action[0] + 1.0) * 0.5 \
            * float(self.env.quad.params["maxThr"])
        rate_frac = float(np.max(np.abs(omega[:2])) / max(1e-9, self.env.max_rate_xy))
        thrust = mf.game_flip_thrust(self._policy_flip_t, float(self.env.quad.dcm[2, 2]),
                                     float(self.env.quad.vel[2]), hover_thrust)
        del rate_frac                         # kept out of the profile (attitude decides)
        if self._policy_flip_t < mf.GAME_FLIP_POP_S:
            # The pop needs the body upright: gate the rotation until it is done, exactly as
            # the manual flip button does (the stick is still read every step).
            omega[0] = 0.0
            omega[1] = 0.0
        if self.actor_input is not None:
            self.actor_input.prepare(self.obs, self.vec_norm)          # keep the GRU warm
        action = np.zeros(4, dtype=np.float32)
        action[0] = 2.0 * thrust / float(self.env.quad.params["maxThr"]) - 1.0
        action[1] = omega[0] / max(1e-9, self.env.max_rate_xy)
        action[2] = omega[1] / max(1e-9, self.env.max_rate_pitch)
        action[3] = omega[2] / max(1e-9, self.env.max_rate_z)
        self.obs, _reward, terminated, _truncated, info = self.env.step(
            np.clip(action, -1.0, 1.0))
        self.last_thrust = float(info.get("throttle", 0.0))
        self.last_omega = np.asarray(info.get("omega_des", omega), dtype=np.float64)
        self.last_source = "pilot-flip"
        if not self._policy_flip_active:
            self._policy_flip_active = True
            self.messages.append("FLIP (policy mode): stick = roll/pitch, collective flown "
                                 "for you - release to hand it back to the policy")
        if terminated:
            reason = info.get("termination_reason", self.env.termination_reason)
            self.crashed = True
            self.end_reason = f"{reason} (pilot flip)"
            self.running = False

    def _policy_flip_release(self) -> None:
        """Hand the vehicle back to the policy, hovering where the roll left it."""
        self._policy_flip_active = False
        self.human.sync_to(self.env.quad.pos, v=None, yaw=yaw_of(self.env.quad.dcm))
        self.env.set_external_reference(self.human)
        self.messages.append("flip released - the policy has it again (holding this hover)")

    def _policy_step(self) -> None:
        pad = self.pad_state
        if pad is not None:
            self._pad_buttons_policy(pad)
        if self.mode == MODE_POLICY_HUMAN:
            self._policy_human_command(pad)
            # The aim point trails the ESTIMATED position (what the policy sees), never truth.
            self.human.update(self.dt, vehicle_p=self.env.lighthouse.p_est)

        # The flip button works here too: hold the top-right shoulder and the left stick
        # rolls/pitches the vehicle directly while the collective is flown on the flip
        # profile, and the policy steps aside. Releasing it hands the vehicle back, hovering where the roll left it.
        flip_held = (self.mode == MODE_POLICY_HUMAN and pad is not None and pad.connected
                     and PAD_BTN_FLIP in pad.buttons)
        if flip_held:
            self._policy_flip_step(pad)
            return
        if self._policy_flip_active:
            self._policy_flip_release()

        if self.actor_input is None or self.model is None:
            # The policy failed to load (missing encoder, a checkpoint from another
            # observation layout, a mismatched actor-frame mode, ...). That is recorded in
            # `messages` at load time; without this guard the next line would raise an
            # opaque AttributeError on None and bury the actual reason.
            why = next((m for m in reversed(self.messages) if m.startswith("policy:")),
                       "the policy was never loaded")
            raise RuntimeError(
                f"POLICY mode requested but {why}.\n"
                f"  Switch to a manual mode, or fix the policy/encoder and restart."
            )

        obs_input = self.actor_input.prepare(self.obs, self.vec_norm)
        action, _ = self.model.predict(obs_input, deterministic=True)
        self.obs, _reward, terminated, _truncated, info = self.env.step(action)
        self.last_thrust = float(info.get("throttle", 0.0))
        self.last_omega = np.asarray(info.get("omega_des", np.zeros(3)), dtype=np.float64)
        self.last_source = "policy"
        if terminated:
            reason = info.get("termination_reason", self.env.termination_reason)
            self.crashed = True
            self.end_reason = f"{reason} (policy flight)"
            self.running = False
            return
        # A finished manoeuvre hands back to the hover that is tracking the sticks.
        if self.mode == MODE_POLICY_TRAJ and (self.env.t - self.traj_t0) >= self.traj_duration:
            self.human.sync_to_reference(self.env.ref)
            self.env.set_external_reference(self.human)
            self.mode = MODE_POLICY_HUMAN
            self.messages.append(f"{self.traj_kind or 'trajectory'} complete - policy is holding a hover")

    def _policy_human_command(self, pad) -> None:
        """Sticks -> reference commands, the VIDEO-GAME way (same sticks as GAME mode).

        left stick = movement (forward/back + strafe) in the heading frame, right stick X =
        heading, ZL/ZR = climb/descend. Centre = hover. Priority: a deflected stick owns the
        command; otherwise the latched keyboard; otherwise, with a pad connected, zero;
        otherwise (no pad at all) the last command is kept so an external source can drive it.
        """
        fwd = left = up = yaw = 0.0
        pad_deflected = False
        if pad is not None and pad.connected:
            roles = self.mapper.roles(pad)
            fwd = roles["move_fwd"] * POLICY_STICK_V
            left = roles["move_left"] * POLICY_STICK_V
            climb = float(pad.right_trigger) - float(pad.left_trigger)
            if PAD_BTN_CLIMB in pad.buttons:
                climb += 1.0
            if PAD_BTN_DESCEND in pad.buttons:
                climb -= 1.0
            up = float(np.clip(climb, -1.0, 1.0)) * POLICY_STICK_UP
            yaw = -roles["heading"] * POLICY_STICK_YAW     # +yaw command = turn right
            pad_deflected = any(abs(v) > 1e-9 for v in (fwd, left, up, yaw))
        if pad_deflected:
            self.human.set_command(fwd=fwd, left=left, up=up, yaw_rate=yaw)
            self._last_cmd = (fwd, left, up, yaw)
        elif bool(np.any(np.abs(self._key_cmd) > 0.0)):
            self.human.set_command(fwd=float(self._key_cmd[0]), left=float(self._key_cmd[1]),
                                   up=float(self._key_cmd[2]), yaw_rate=float(self._key_cmd[3]))
            self._last_cmd = tuple(float(v) for v in self._key_cmd)
        elif pad is not None and pad.connected:
            self.human.set_command(0.0, 0.0, 0.0, 0.0)
            self._last_cmd = (0.0, 0.0, 0.0, 0.0)
        # else: no input device - leave the command as it is (external source)

    def _policy_key(self, key: int) -> bool:
        """Latched keyboard commands for policy mode. Returns True if consumed."""
        if key == mf.K_SPACE or key == mf.K_X:
            self._key_cmd[:] = 0.0
            self.messages.append("hover: command cleared")
            return True
        step_v = POLICY_KEY_V_STEP
        step_y = POLICY_KEY_YAW_STEP
        delta = {mf.K_W: (step_v, 0), mf.K_S: (-step_v, 0),
                 mf.K_A: (0, step_v), mf.K_D: (0, -step_v)}.get(key)
        if delta is not None:
            self._key_cmd[0] += delta[0]
            self._key_cmd[1] += delta[1]
        elif key in (mf.K_R, mf.K_F):
            self._key_cmd[2] += step_v if key == mf.K_R else -step_v
        elif key in (mf.K_Q, mf.K_E):
            # The command is a rotation rate about z, where POSITIVE is a left turn
            # (the project's convention), so "turn right" is negative.
            turn_right = key == mf.K_E
            self._key_cmd[3] += -step_y if turn_right else +step_y
        else:
            return False
        self._key_cmd[0] = float(np.clip(self._key_cmd[0], -POLICY_STICK_V, POLICY_STICK_V))
        self._key_cmd[1] = float(np.clip(self._key_cmd[1], -POLICY_STICK_V, POLICY_STICK_V))
        self._key_cmd[2] = float(np.clip(self._key_cmd[2], -POLICY_STICK_UP, POLICY_STICK_UP))
        self._key_cmd[3] = float(np.clip(self._key_cmd[3], -POLICY_STICK_YAW, POLICY_STICK_YAW))
        self.messages.append(f"command fwd {self._key_cmd[0]:+.1f} left {self._key_cmd[1]:+.1f} "
                             f"up {self._key_cmd[2]:+.1f} yaw {self._key_cmd[3]:+.2f}")
        return True

    # -- pad plumbing --------------------------------------------------------------------
    def _pad_shim(self, pad):
        """Logical sticks presented to the manual loop (layout applied, signs corrected).

        Two conventions, because the manual sandbox has two stick worlds: ACRO/ASSISTED
        consume role-mapped values (roll/throttle/pitch/yaw), while GAME consumes the raw
        logical sticks with its own video-game equations (left stick = movement, right
        stick X = heading). live_flight sets the sandbox's PAD_INVERT_* flags to False, so
        "logical" means '+1 = right / up' in both cases.

        `PAD_BTN_MODE_ALT` (-) is held back from the manual mapping: in this program it
        means "hand over to the policy / take back control", not "switch the manual
        control set".
        """
        roles = self.mapper.roles(pad)
        if self.cmd.mode == mf.MODE_GAME:
            left_x, left_y, right_x, right_y = roles["lx"], roles["ly"], roles["rx"], roles["ry"]
        else:
            left_x, left_y, right_x, right_y = (roles["roll"], roles["throttle"],
                                                roles["yaw"], roles["pitch"])
        return gp.PadState(
            connected=pad.connected, name=pad.name,
            left_x=left_x, left_y=left_y, right_x=right_x, right_y=right_y,
            left_trigger=pad.left_trigger, right_trigger=pad.right_trigger,
            buttons=set(pad.buttons),
            pressed=tuple(b for b in pad.pressed if b != PAD_BTN_MODE_ALT),
            raw_axes=dict(pad.raw_axes), analog_triggers=pad.analog_triggers,
        )

    def _pad_buttons_manual(self, pad) -> None:
        if PAD_BTN_MODE_ALT in pad.pressed:             # '-' : hand over to the policy
            if self.policy_ok:
                self.enter_mode(MODE_POLICY_HUMAN)
            else:
                self.messages.append("no policy loaded - staying manual")

    def _pad_buttons_policy(self, pad) -> None:
        pressed = set(pad.pressed)
        if PAD_BTN_MODE_ALT in pressed:                 # '-' : take back control
            self.enter_mode(MODE_MANUAL_ACRO)
        if PAD_BTN_MODE in pressed or PAD_BTN_MATCH in pressed:
            if self.mode == MODE_POLICY_TRAJ:
                self.human.sync_to_reference(self.env.ref)
                self.env.set_external_reference(self.human)
                self.mode = MODE_POLICY_HUMAN
                self.messages.append("trajectory released - holding a hover")
            else:
                self.human.clear_command()
                self._key_cmd[:] = 0.0
                self.messages.append("hover: commands cleared")
        if PAD_BTN_RESPAWN in pressed:
            self.respawn()

    # -- keyboard ------------------------------------------------------------------------
    def handle_key(self, key: int) -> None:
        if key == mf.K_ESCAPE:
            self.running = False
            self.end_reason = "quit by the pilot (ESC)"
            return
        if key == mf.K_ENTER:
            self.respawn()
            return
        if key in _KEY_MODE:
            self.enter_mode(_KEY_MODE[key])
            return
        if key in _KEY_TRAJ:
            self.launch_trajectory(_KEY_TRAJ[key])
            return
        if self.mode.startswith("MANUAL"):
            line = self.cmd.handle_key(key)
            if line:
                self.messages.append(line)
        else:
            self._policy_key(key)

    # -- panel commands ------------------------------------------------------------------
    def handle_gui(self, command: str) -> None:
        try:
            if command.startswith("mode:"):
                name = command.split(":", 1)[1]
                self.enter_mode({"manual_acro": MODE_MANUAL_ACRO,                                 "manual_game": MODE_MANUAL_GAME,                                 "manual_assisted": MODE_MANUAL_ASSISTED,
                                 "policy_human": MODE_POLICY_HUMAN}.get(name, name))
            elif command.startswith("traj:"):
                self.launch_trajectory(command.split(":", 1)[1])
            elif command == "pad:calibrate":
                self.mapper.calibrate_start(self.pad_state)
                self.messages.append("calibration started - follow the panel instructions")
            elif command == "pad:save":
                self.mapper.save(PAD_CALIBRATION_PATH)
                self.messages.append(f"calibration saved to {os.path.basename(PAD_CALIBRATION_PATH)}")
            elif command.startswith("pad:layout:"):
                layout = self.mapper.set_layout(command.split(":", 2)[2])
                self.messages.append(f"pad layout -> {'Outer Wilds roles' if layout == 'ow' else 'Mode 2'}")
            elif command == "pad:swap_sticks":
                self.mapper.swap_sticks()
                self.messages.append("pad: left/right sticks swapped (check the stick readout)")
            elif command.startswith("pad:invert:"):
                _, _, channel, value = command.split(":")
                self.mapper.set_invert(channel, bool(int(value)))
                self.messages.append(f"pad {channel} invert -> {bool(int(value))}")
            elif command == "flight:respawn":
                self.respawn()
            elif command == "flight:quit":
                self.running = False
                self.end_reason = "quit from the panel"
        except Exception as exc:
            self.messages.append(f"panel command '{command}' failed: {exc}")

    def _place_at_start(self) -> None:
        """
        Teleport the vehicle - and everything anchored to it - to the START pose.

        The env's own `reset()` spawns at the training volume's centre (0, 0, 1.2) because
        that is where its reference lives; a LIVE flight may start anywhere (the default is
        on the floor), so once the env is set up the whole stack is re-seated where the
        pilot asked: the plant, the Lighthouse (anchored on the spawn point), the reference
        the actor frame compares against, and the observation history.

        `adopt_state` is exactly that primitive - it is the handover path - which is why
        this is also what a RESPAWN does. `arm_ground_start` then decides whether the floor
        under this pose is a runway or a crash surface.
        """
        pos = np.asarray(self.start_pos, dtype=np.float64)
        quat = QuadFlipEnv._euler_to_quat(0.0, 0.0, float(self.start_yaw))
        self.human.sync_to(pos, v=None, yaw=float(self.start_yaw))
        self.env.set_external_reference(self.human)
        self.env.adopt_state(pos, quat, vel=np.zeros(3), omega=np.zeros(3))
        self.env.arm_ground_start(float(pos[2]) <= self.env.GROUND_RELEASE_Z)

        # Manual channels: the altitude setpoint starts AT the start height (so nothing
        # commands a climb until the pilot asks for one) and every command is cleared.
        self.cmd.z_des = float(pos[2])
        self.cmd.yaw_des = float(self.start_yaw)
        self.cmd.v_cmd[:] = 0.0
        self.cmd.rot_pad[:] = 0.0
        self.cmd._rot_budget[:] = 0.0
        self.cmd.match_velocity = False
        self.cmd.level_now = False
        self.cmd.game_flip = False
        self.cmd.throttle = self.ctrl.hover_throttle
        self._key_cmd[:] = 0.0
        self.crashed = False

        if self.actor_input is not None:
            self.actor_input.reset()             # a (re)start is a new flight to the GRU
        if self.policy_ok:
            self.obs = self.env._get_stacked_obs()

    def respawn(self) -> None:
        self.env.reset(options={"maneuver": "hover"})
        if self.mode == MODE_POLICY_TRAJ:
            self.mode = MODE_POLICY_HUMAN
            self.traj_kind = None
        self._place_at_start()
        self.messages.append("respawned at the start pose"
                             + (" (on the floor)" if self.env.ground_start else ""))

    # -- HUD / status ---------------------------------------------------------------------
    def hud_texts(self) -> List[Tuple[Any, Any, str, str]]:
        q = self.env.quad
        tilt = math.degrees(math.acos(float(np.clip(q.dcm[2, 2], -1.0, 1.0))))
        ref = self.env.ref
        ref_err = float(np.linalg.norm(q.pos - ref.p)) if ref is not None else 0.0
        rate = np.degrees(self.last_omega)
        if self.mode == MODE_POLICY_TRAJ:
            left = max(0.0, self.traj_duration - (self.env.t - self.traj_t0))
            head = f"POLICY - executing {self.traj_kind or '?'}  ({left:4.1f} s left)"
            sub = f"ref z {ref.p[2]:.2f} m  err {ref_err:.2f} m  thrust {100 * self.last_thrust / q.params['maxThr']:3.0f}%"
        elif self.mode == MODE_POLICY_HUMAN:
            head = "POLICY - tracking your sticks (left = movement, right X = heading, ZL/ZR = altitude)"
            c = self._last_cmd
            sub = (f"cmd fwd {c[0]:+4.1f} left {c[1]:+4.1f} up {c[2]:+4.1f} yaw {c[3]:+4.2f} | "
                   f"ref err {ref_err:.2f} m  thrust {100 * self.last_thrust / q.params['maxThr']:3.0f}%")
        else:
            head = f"{self.mode} - you have the sticks"
            sub = f"throttle {100 * self.cmd.throttle:3.0f}%  rate [p{rate[0]:+4.0f} q{rate[1]:+4.0f} r{rate[2]:+4.0f}] deg/s"
        live = (f"t {self.env.t:6.1f}s  z {q.pos[2]:5.2f} m  speed {float(np.linalg.norm(q.vel)):5.2f} m/s  "
                f"tilt {tilt:4.0f} deg  [{self.env.quad.pos[0]:+.2f} {self.env.quad.pos[1]:+.2f}]")
        health = (f"policy {'ON' if self.policy_ok else 'off'} (encoder {'on' if self.encoder_on else 'off'})  "
                  f"link {self.link.name}  1/2/3 modes  4-0 trajectories  ENTER respawn  ESC quit")
        help_left = ("PAD: - hand over/take back   + hover now   "
                     "GAME mode: top-right shoulder = FLIP while held")
        if self.pad_state is not None and self.pad_state.connected:
            pad_line = (f"PAD {self.pad_state.name[:30]}  {self.mapper.calibration_report()}")
        else:
            pad_line = "PAD none - keyboard only"
        texts = [
            (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, head, sub),
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT, live, health),
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
             pad_line, self.messages[-1] if self.messages else ""),
        ]
        return texts

    def status_dict(self) -> Dict[str, Any]:
        q = self.env.quad
        ref = self.env.ref
        axes: Dict[str, float] = {}
        if self.pad_state is not None:
            sticks = self.mapper.sticks(self.pad_state, shaped=False)
            axes = {k: float(v) for k, v in sticks.items()}
        cal = self.mapper.cal
        prompt = cal.instruction if cal.active else ""
        return {
            "mode": self.mode,
            "trajectory": self.traj_kind or "-",
            "trajectory_left": max(0.0, self.traj_duration - (self.env.t - self.traj_t0))
            if self.mode == MODE_POLICY_TRAJ else 0.0,
            "policy": self.policy_ok,
            "encoder": self.encoder_on,
            "pad": self.pad_state.name if (self.pad_state is not None and self.pad_state.connected) else "none",
            "z": float(q.pos[2]),
            "speed": float(np.linalg.norm(q.vel)),
            "ref_err": float(np.linalg.norm(q.pos - ref.p)) if ref is not None else 0.0,
            "tilt": math.degrees(math.acos(float(np.clip(q.dcm[2, 2], -1.0, 1.0)))),
            "thrust": 100.0 * self.last_thrust / float(q.params["maxThr"]),
            "rate": float(np.degrees(np.linalg.norm(self.last_omega))),
            "link": self.link.describe(),
            "axes": axes,
            "inverts": self.mapper.invert_flags(),
            "layout": self.mapper.layout,
            "calibration": self.mapper.calibration_report(),
            "calibration_done": bool(cal.done),
            "calibration_failed": cal.failed,
            "calibration_prompt": prompt,
            "message": self.messages[-1] if self.messages else "",
        }

    # -- main loop -------------------------------------------------------------------------
    def run(self, show_viewer: bool = True) -> None:
        if self.gui is None and GUI_ENABLE and not self.args.no_gui:
            self._start_gui()
        if show_viewer:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.env.quad.model, self.env.quad.data)
                look = np.array([self.start_pos[0], self.start_pos[1],
                                 max(0.8, float(self.start_pos[2]))])
                self.viewer.cam.lookat = look
                self.viewer.cam.distance = mf.CAMERA_DISTANCE
                self.viewer.cam.elevation = mf.CAMERA_ELEVATION_DEG
                self.viewer.key_callback = self._key_callback
            except Exception as exc:
                print(f"Note: could not launch the viewer ({exc}); running headless.")
                self.viewer = None

        print("\nLive flight running. 1/2/3 = modes, 4-0 = trajectories, ENTER = respawn, ESC = quit.")
        for m in self.messages:
            print(f"  {m}")
        self.messages.clear()
        budget = int(getattr(self.args, "steps", 0) or 0)

        try:
            while self.running:
                step_start = time.time()
                if self.viewer is not None and not self.viewer.is_running():
                    self.end_reason = "viewer closed"
                    break

                self._poll_gui()
                self.pad_state = self.pad.state() if self.pad is not None else None
                self._poll_calibration()

                if self.mode.startswith("POLICY"):
                    self._policy_step()
                else:
                    self._manual_step()

                # link: every command, one conversion, one sink.
                self.link.send(RadioSetpoint.from_sim(
                    self.env.t, self.last_thrust, self.last_omega,
                    float(self.env.quad.params["maxThr"]), self.last_source))

                if not np.all(np.isfinite(self.env.quad.state)):
                    self.crashed = True
                    self.end_reason = "diverged"
                    break
                # Ground: fatal once the flight can be said to have taken off. A ground
                # start may idle, drift and hop on the floor until it climbs away; after
                # that the floor is the floor again.
                ground_is_fatal = self.env.ground_touch_is_fatal()
                if self.env.quad.check_ground_contact() and ground_is_fatal:
                    self.crashed = True
                    self.end_reason = "crashed into the ground"
                    break

                self._push_log()
                self._update_viewer()
                self._publish_status()

                if budget and self._steps_done >= budget:
                    self.end_reason = f"step budget reached ({budget})"
                    break

                remaining = self.dt - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self._finish()

    def _poll_gui(self) -> None:
        if self.gui is None:
            return
        for msg in self.gui.poll():
            if msg.get("type") == "command":
                self.handle_gui(str(msg.get("command", "")))

    def _poll_calibration(self) -> None:
        if self.mapper.cal.active and self.pad_state is not None:
            before = self.mapper.cal.step
            self.mapper.calibrate_update(self.pad_state)
            if not self.mapper.cal.active:
                self.messages.append(self.mapper.calibration_report()
                                     if self.mapper.cal.done else
                                     f"calibration: {self.mapper.cal.failed}")
            elif self.mapper.cal.step != before:
                self.messages.append(f"calibration: {self.mapper.cal.instruction}")

    def _key_callback(self, keycode: int) -> None:
        # The viewer reports presses only (see manual_flight.py); ESC is the shared quit.
        self.handle_key(int(keycode))

    def _update_viewer(self) -> None:
        if self.viewer is None:
            return
        # Chase camera locked to the vehicle's HEADING (same helper as the manual sandbox;
        # `azimuth` is the direction the camera looks, measured against mjv_updateScene).
        mf._update_camera(self.viewer, self.env.quad)
        self.viewer.set_texts(self.hud_texts())
        self.viewer.sync()

    def _publish_status(self) -> None:
        if self.gui is None or (self.env.t - self.status_t) < 0.1:
            return
        self.status_t = self.env.t
        self.gui.publish(self.status_dict())

    # -- telemetry ----------------------------------------------------------------------
    def _push_log(self) -> None:
        ref = self.env.ref
        q = self.env.quad
        # Fill the manual sandbox's log keys from the REFERENCE where it has one, so the
        # existing plotter shows reference vs actual for policy modes too.
        self.cmd.v_cmd[0] = float(ref.v[0]) if ref is not None else 0.0
        self.cmd.v_cmd[1] = float(ref.v[1]) if ref is not None else 0.0
        self.cmd.z_des = float(ref.p[2]) if ref is not None else float(q.pos[2])
        self.cmd.yaw_des = yaw_of(ref.R) if ref is not None else yaw_of(q.dcm)
        mf._push(self.log, q, self.cmd, self.env.t, float(self.last_thrust), self.last_omega)
        self.log["mode"][-1] = MODE_CODES.get(self.mode, -1)
        self.log["throttle"][-1] = float(self.last_thrust / self.env.quad.params["maxThr"])
        self.log.setdefault("ref_p", []).append(ref.p.copy() if ref is not None else q.pos.copy())
        self._steps_done += 1

    def summary(self) -> None:
        log = self.log
        if not log["t"]:
            print("No telemetry recorded.")
            return
        data = {k: np.asarray(v, dtype=np.float64) for k, v in log.items()}
        t = data["t"]
        modes = data["mode"]
        shares = "  ".join(f"{MODE_LABELS[int(m)].lower()} {100.0 * float(np.mean(modes == m)):.0f}%"
                           for m in np.unique(modes)
                           if float(np.mean(modes == m)) > 0.005)
        policy_frames = modes >= 2
        has_ref = "ref_p" in data and len(data["ref_p"]) == len(data["pos"])
        print("\n" + "=" * 78)
        print("LIVE FLIGHT SUMMARY")
        print(f"  Duration   : {t[-1]:.2f} s | mode share: {shares}")
        print(f"  Max speed  : {float(np.linalg.norm(data['vel'], axis=1).max()):.2f} m/s")
        print(f"  Attitude   : roll {np.degrees(data['euler'][:, 0]).min():+.0f}.."
              f"{np.degrees(data['euler'][:, 0]).max():+.0f}, "
              f"pitch {np.degrees(data['euler'][:, 1]).min():+.0f}.."
              f"{np.degrees(data['euler'][:, 1]).max():+.0f}, "
              f"yaw {np.degrees(data['euler'][:, 2]).min():+.0f}.."
              f"{np.degrees(data['euler'][:, 2]).max():+.0f} deg")
        print(f"  Altitude   : {data['pos'][:, 2].min():.2f} .. {data['pos'][:, 2].max():.2f} m")
        if has_ref and bool(np.any(policy_frames)):
            err = np.linalg.norm(data["pos"] - data["ref_p"], axis=1)[policy_frames]
            print(f"  Policy err : mean {float(np.mean(err)):.3f} m, max {float(np.max(err)):.3f} m "
                  f"while the policy was flying")
        print(f"  End        : {self.end_reason}")
        print(f"  Link       : {self.link.describe()}")
        try:
            mf._plot_telemetry(data, TELEMETRY_PATH)
            print(f"  Telemetry  : {TELEMETRY_PATH}")
        except Exception as exc:
            print(f"  Telemetry  : plotting failed ({exc})")
        print("=" * 78 + "\n")

    # -- panel process --------------------------------------------------------------------
    def _start_gui(self) -> None:
        self.gui = GuiServer(port=int(self.args.port))
        if not self.gui.start():
            self.messages.append(f"panel: could not listen on port {self.args.port} ({self.gui.error})")
            self.gui = None
            return
        try:
            self.gui_proc = subprocess.Popen(
                [_python_for_gui(), os.path.join(_SIM_DIR, "flight_gui.py"),
                 "--port", str(self.args.port)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.messages.append(f"panel: control window on port {self.args.port}")
        except Exception as exc:
            self.messages.append(f"panel: could not start the GUI process ({exc})")

    def _finish(self) -> None:
        self.summary()
        if self.gui is not None:
            self.gui.publish(self.status_dict())
            self.gui.stop()
        if self.gui_proc is not None and self.gui_proc.poll() is None:
            self.gui_proc.terminate()
        if self.pad is not None:
            self.pad.stop()
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass


# ======================================================================================
# entry point
# ======================================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Live flight with a human-chosen reference")
    parser.add_argument("--model", default=MODEL_NAME, help="checkpoint name or path")
    parser.add_argument("--dr", type=float, default=DR_LEVEL, help="domain randomisation 0..1")
    parser.add_argument("--port", type=int, default=GUI_PORT, help="control-panel port")
    parser.add_argument("--no-gui", action="store_true", help="do not start the panel")
    parser.add_argument("--no-pad", action="store_true", help="ignore controllers")
    parser.add_argument("--no-policy", action="store_true", help="manual flight only")
    parser.add_argument("--radio", action="store_true",
                        help="use the (not yet implemented) crazyradio link placeholder")
    parser.add_argument("--spawn", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                        help="start position in metres (default: on the floor, "
                             "(0.0, 0.0, %.3f))" % GROUND_REST_Z)
    parser.add_argument("--spawn-yaw", type=float, default=None,
                        help="start heading in degrees (default %.1f)" % START_YAW_DEG)
    parser.add_argument("--headless", action="store_true", help="no 3D window (testing)")
    parser.add_argument("--steps", type=int, default=0, help="stop after N steps (testing)")
    args = parser.parse_args()

    live = LiveFlight(args)
    live.run(show_viewer=not args.headless)
    return 0


if __name__ == "__main__":
    # macOS GUI trampoline: launch_passive requires mjpython (identical to evaluate.py).
    if sys.platform == "darwin":
        is_mjpython = hasattr(mujoco.viewer, "_MJPYTHON") and mujoco.viewer._MJPYTHON is not None
        if not is_mjpython and os.environ.get("_MJP_TRAMPOLINED") != "1":
            import shutil
            mjpython_path = os.path.join(os.path.dirname(sys.executable), "mjpython")
            if not os.path.isfile(mjpython_path):
                mjpython_path = shutil.which("mjpython")
            if mjpython_path and os.path.isfile(mjpython_path):
                os.environ["_MJP_TRAMPOLINED"] = "1"
                os.execv(mjpython_path, [mjpython_path] + sys.argv)
    raise SystemExit(main())
