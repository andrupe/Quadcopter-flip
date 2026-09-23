# -*- coding: utf-8 -*-
"""
Manual flight sandbox: fly the Crazyflie yourself in the MuJoCo viewer.

WHY THIS FILE EXISTS
--------------------
evaluate.py flies a trained checkpoint. This script hands the sticks to the pilot: the
SAME plant, the SAME 1 kHz inner-loop rate PID and the SAME viewer, but the collective
and attitude setpoints come from the keyboard instead of a policy.

    .venv/bin/python Simulation/manual_flight.py     (VS Code: click Run)

On macOS the interactive viewer requires `mjpython`; the script re-execs itself through
it automatically (the same trampoline as evaluate.py). Nothing is loaded from logs/, so
the script works with or without a trained model; and unlike training/evaluation there
are NO boundaries at all - the flight volume, the reference trajectory, the reward and
the domain randomisation are all absent, so a key press always does the same thing.
The run ends when you press ESC, close the window, or the vehicle touches the ground.

WHAT ACTUALLY RUNS (and what therefore does NOT)
------------------------------------------------
The full real-time chain is: keyboard/pad -> ManualFlightController (100 Hz, the pilot)
-> rate setpoints + collective -> utils.rate_pid.RatePIDController (1 kHz - the same
controller training and evaluation use) -> utils.mixer.mixerFM -> quad_mujoco motor and
body dynamics -> MuJoCo. What is missing is the perception/decision half of the trained
stack: no lighthouse/observation model, no actor input frame, no GRU history encoder and
no PPO policy (torch is not even imported). So this flies the plant and its inner loop
exactly as the learned controller experiences them, with a human standing in for the
policy.

HOW THE KEYBOARD INPUT WORKS - PLEASE READ THIS ONE PARAGRAPH
-------------------------------------------------------------
MuJoCo's Python viewer reports key PRESSES only: there is no key-up event, and macOS
auto-repeat arrives as GLFW_REPEAT, which the viewer's callback filters out (see
`IsKeyDownEvent` in mujoco/simulate/glfw_adapter.cc, act == GLFW_PRESS). A "hold the
key" control law therefore cannot be built on top of it, so this script uses TRIMMED
commands: every press nudges a latched setpoint, and an overlay in the window always
shows what the setpoints currently are. To undo, press the opposite key the same number
of times, or SPACE (hover) / ENTER (respawn).

CONNECT A GAMEPAD AND THE STICKS ARE CONTINUOUS - NO TRIMMING NEEDED
--------------------------------------------------------------------
A HID controller (Xbox, DualSense/DualShock, Switch Pro, 8BitDo, ...) connected to the
Mac is read directly through IOKit by `gamepad.py` (its own thread, so it works under
mjpython where the main thread belongs to the Cocoa GUI). The sticks spring back to the
centre, so they command continuously exactly like a real transmitter. The pad drives
whichever control set is active:

    ACRO (default)                      ASSISTED (after M)
    left stick  X = roll rate           left stick = fly forward/back/left/right
                Y = throttle (latched)  right stick = tilt (enters ANGLE mode)
    right stick X = yaw rate            LT / RT (or buttons 7/8) = altitude down/up
                Y = pitch rate          LB / RB = yaw left/right
    LB / RB     = roll                  D-pad = altitude / yaw steps
    LT / RT     = throttle down / up    face button = hover (panic)
    D-pad       = throttle steps        button 2 = respawn
    X / Square  = match velocity
    bottom btn  = level (keeps yaw)
    left btn    = switch control set

In both sets a deflected stick owns its channel and the keyboard drives every channel
the pad is not currently using. If a pad's axes read backwards (they are reported in raw
HID sign), flip the PAD_INVERT_* flags in the configuration block - the window overlay
prints the live stick values so it is obvious which one is wrong.

KEYBOARD, FOR REFERENCE (ACRO set - see the section below for what it does)
    W / S (I / K)  pitch                                A / D   roll
    Q / E (J / L)  yaw                                  X       match velocity
    SPACE / R      collective up   CTRL / F  collective down
    C              level, heading kept   M  switch control set
    ENTER          respawn                               ESC    quit

KEYBOARD, ASSISTED SET (what M switches to)
    W / S        velocity command  forward / back     0.5 m/s per press (+-2.5 max)
    A / D        velocity command  left / right       0.5 m/s per press (+-2.5 max)
    I / K        pitch trim  nose down / up           5 deg per press (+-80 max)
    J / L        roll trim   left / right             5 deg per press (+-80 max)
    Q / E        yaw setpoint left / right            15 deg per press (wraps)
    R / F        altitude setpoint up / down          0.25 m per press (0.3 - 6 m)
    SPACE        hover: stop, level, zero every trim, hold the current altitude
    ENTER        respawn at the start state and clear all commands
    ESC          quit (window closes, flight summary + telemetry plot are printed)

THREE CONTROL SETS - M SWITCHES BETWEEN THEM (default: GAME)
------------------------------------------------------------
GAME - a video-game stick layout, plus a flip button
The vehicle is still the real one - the plant, the 1 kHz rate loop and the mixer are
untouched - but the sticks are mapped the way a game maps them:

    left stick ....... movement: forward/back and left/right, as a VELOCITY command in
                       the heading frame (the quad banks into it and brakes on release)
    right stick X .... heading: turn while the stick is held, then hold the new heading
    ZR / ZL .......... climb / descend (the trigger buttons)
    top-right shoulder  FLIP: hold it and the collective is FLOWN for you on a flip
                       profile (pop up, thrust cut while inverted, catch on the way
                       down - the shape the trained flip uses) while the left stick
                       commands body rates directly: push right to roll right, left to
                       roll left, forward/back to pitch. The rotation only starts after
                       the 0.25 s pop, so the pop goes UP and not sideways. Release and
                       the movement mapping returns (the heading it left you on becomes
                       the new heading setpoint, so nothing snaps back).
    keyboard ........ W/A/S/D movement, Q/E heading, R/F altitude, V = flip toggle,
                       SPACE = hover (stop, level, hold the altitude)

The camera follows the VEHICLE'S HEADING (CAMERA_LOCK_TO_HEADING): the free camera's
`azimuth` is the direction it looks (measured with mjv_updateScene, not assumed), so the
view stays behind the tail and the world turns with you - the same feel as the game's
chase camera. The mouse can still zoom and raise it; only the heading is locked.

ACRO - a REAL acro quadcopter, flown with Outer Wilds' control roles
The vehicle is unchanged: ordinary quadrotor physics, the same 1 kHz rate loop, the same
mixer and motors. What the pilot commands is exactly what a real acro quad's flight
controller commands - a LATCHED COLLECTIVE on the throttle, and BODY RATES on the
three steering channels:

    W / S ........ pitch nose down / up    -> travel forward / back, as in the game
    A / D ........ roll left / right       -> travel left / right, as in the game
    Q / E (J/L) .. yaw left / right
    I / K ........ pitch (the game's aiming input; same as W/S)
    SPACE / R .... more collective         CTRL / F ... less collective
    X ............ MATCH VELOCITY: brake to zero ground velocity and hold it
    C ............ level the attitude, keeping the heading
    M ............ switch to the assisted pair below

The point of this set is that NOTHING is stabilised for the pilot: no auto-levelling, no
auto-braking, no altitude hold. A drone left tilted keeps accelerating, and it coasts at
constant velocity once levelled - which is exactly how an acro quad behaves, and also
what the game's ship feels like. Counter-tilt (or Match Velocity) to stop. Keyboard
rotation keys are pulses (key release cannot be detected through the viewer's API) and
the keyboard throttle is stepped; a gamepad gives continuous sticks.

ASSISTED - the earlier pair, kept for relaxed flying
VELOCITY (W/A/S/D) holds the ground velocity you command and brakes to a stop by itself;
ANGLE (I/J/K/L) holds the tilt you command. Both keep an altitude hold running, so the
collective is handled for you.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No reference trajectory, no reward, no reward-shaped termination, no flight sphere, no
lighthouse/observation model and no domain randomisation. The plant is nominal so that
one press produces one predictable response: this script is a joystick, not a benchmark.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

# macOS GUI trampoline support: matplotlib may only open a window on the main thread,
# and under mjpython the script runs on a secondary thread (see utils/display.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "Simulation" else _THIS_DIR
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _is_gui_thread() -> bool:
    """True when the current thread may own a GUI window (always true off macOS)."""
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
    matplotlib.use("Agg")          # plotting still works, it just writes a file
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

from quadFiles.quad_mujoco import QuadcopterMuJoCo
from quad_flip_env import QuadFlipEnv                      # rotation helpers only
from utils.rate_pid import RatePIDController
from utils.windModel import Wind
from trajectories import dcm_from_thrust_dir_and_yaw

try:
    from gamepad import HAT_DOWN, HAT_LEFT, HAT_RIGHT, HAT_UP, HIDGamepad
except Exception as _pad_import_error:      # pragma: no cover - non-macOS or broken IOKit
    HIDGamepad = None                       # type: ignore[assignment]
    HAT_UP = HAT_RIGHT = HAT_DOWN = HAT_LEFT = -1
    _PAD_IMPORT_ERROR = str(_pad_import_error)

if TYPE_CHECKING:                           # type-only import (annotations stay strings)
    from gamepad import PadState

# ======================================================================================
# FLIGHT CONFIGURATION (edit here, then Run)
# ======================================================================================
DT: float = 0.01                  # Outer (command) loop period, s. 100 Hz, same as the
                                  # training env. The inner rate PID is untouched and
                                  # still runs at the physics rate (1 kHz, 10 substeps).
SPAWN_POS: Tuple[float, float, float] = (0.0, 0.0, 1.2)   # Start state: level hover
PLAYBACK_SPEED: float = 1.0       # 1.0 = real time, 0.5 = half speed (slow motion)
CAMERA_FOLLOW: bool = True        # Keep the camera on the vehicle - there are no bounds!
SHOW_PLOTS: bool = True           # Save (and, on a GUI thread, show) telemetry at the end
VERBOSE_KEYS: bool = True         # Echo every key press and the setpoint it produced
WIND_SPEED: float = 0.0           # m/s of Perlin gusts; 0 = still air
TELEMETRY_PATH: str = os.path.join(_PROJECT_ROOT, "manual_flight_telemetry.png")

# --- VELOCITY mode --------------------------------------------------------------------
V_STEP: float = 0.5               # m/s added per W/A/S/D press
V_MAX: float = 2.5                # m/s cap per axis
KP_V: float = 2.5                 # 1/s: velocity error -> horizontal acceleration

# --- ANGLE mode -----------------------------------------------------------------------
TILT_STEP_DEG: float = 5.0        # deg added per I/K/J/L press
TILT_MAX_DEG: float = 80.0        # deg cap per axis (past that the rotors cannot hold)
KR_RP: float = 6.0                # 1/s: attitude error -> body rate (roll/pitch)
KR_YAW: float = 3.0               # 1/s: attitude error -> body rate (yaw)
MAX_RATE_RP: float = 20.0         # rad/s rate-command clip (the policy's own authority)
MAX_RATE_YAW: float = 4.0

# --- shared ---------------------------------------------------------------------------
MAX_TILT_DEG: float = 80.0        # Hard cap on the commanded tilt (velocity mode)
# Altitude hold is a CASCADE, the way a commercial flight controller does it: the position
# error becomes a BOUNDED climb/descent rate, and the rate error becomes vertical
# acceleration with integral action.
#   * the rate bound is what stops a large setpoint step from becoming a dive (a single PD
#     on the position error lets a 1 m step build ~1.2 m/s of sink and overshoot by ~0.4 m);
#   * the integrator is not decoration: above ~10 m/s the fluid drag on a PITCHED airframe
#     is a real vertical force (measured 0.29 g in a 20 deg attitude hold at 20 m/s) and
#     only integral action removes the droop it causes. It is anti-windup guarded below.
KP_VZ: float = 1.5                # 1/s: altitude error -> commanded vertical rate
VZ_MAX: float = 1.5               # m/s cap on the commanded climb / descent rate
KV_Z: float = 3.0                 # 1/s: vertical rate error -> vertical acceleration
KI_VZ: float = 1.5                # 1/s^2: integral on the rate error (drag droop)
A_Z_INT_MAX: float = 5.0          # m/s^2 cap on the integral term (anti-windup)
A_Z_MIN: float = -4.0             # m/s^2 descent-acceleration limit
A_Z_MAX: float = 6.0              # m/s^2 climb-acceleration limit
Z_STEP: float = 0.25              # m per R/F press
Z_MIN: float = 0.30               # m; below this the ground stop triggers first
Z_MAX: float = 6.00               # m
YAW_STEP_RAD: float = math.radians(15.0)

# --- ACRO mode: a REAL quad (collective + body rates) flown with Outer Wilds' key roles --
# This is not a spacecraft model. The vehicle is still an ordinary rate-controlled
# quadcopter: the throttle channel sets COLLECTIVE thrust along body z, the steering keys
# set BODY RATES, and all translation comes from tilting, exactly as a real drone does.
# What is borrowed from Outer Wilds is the CONTROL LAYOUT and its feel:
#   * thrust/steering are momentary - release and nothing is compensated for you;
#   * there is NO auto-levelling and NO auto-braking, so letting the drone stay tilted keeps
#     accelerating it (coast instead of hover) - which is also what an acro quad does;
#   * W/S and A/D do what they do in the game (forward/back, left/right travel), which for a
#     quad means pitching and rolling - that is how a real drone accelerates sideways;
#   * SPACE / CTRL are the game's up/down thrusters = more / less collective;
#   * X is Match Velocity, the game's brake, implemented as the position-hold a real
#     flight controller has (it is the only assist, and it disengages the moment you touch
#     a rotation key or the throttle).
START_MODE: str = "GAME"           # GAME, ACRO or the assisted pair (M cycles)
# VIDOEOGAME set (GAME) ------------------------------------------------
# left stick = movement (forward/back/strafe) in the heading frame, right stick X =
# heading, ZL/ZR = altitude. The top-right shoulder is the FLIP button: while it is held
# the LEFT stick goes straight to body rates (roll on X, pitch on Y) and the COLLECTIVE is
# flown for the pilot on the flip profile below - which is how a video game lets you do a
# roll without also having to fly the throttle at the same time.
GAME_FLIP_RATE_DEG: float = 720.0   # deg/s of body rate at full left stick, flip held
# The flip button's vertical profile, copied from the TRAINED flip (trajectories.Flip):
# the rotation happens in a ZERO-THRUST ballistic window, because thrust acts along body z
# and firing it while rotated 90 deg would push the vehicle sideways. Thrust acts only
# while upright: a pop before the roll and an arrest after it. Measured before this
# existed: a 360 deg roll at hover thrust lost ~1 m of altitude and hit the floor.
GAME_FLIP_POP_S: float = 0.25      # s of pop before the rotation is released (the climb phase)
GAME_FLIP_POP_X: float = 1.9       # pop / arrest thrust as a multiple of hover trim (~2x,
                                   # the same ratio the trained flip's climb phase uses)
GAME_FLIP_CATCH_DCM: float = 0.3   # z_b.z above which the vehicle counts as upright again
GAME_FLIP_CATCH_VZ: float = -0.15  # m/s of sink that triggers the arrest
GAME_FLIP_HOLD_DCM: float = 0.5    # z_b.z above which the altitude hold takes over
GAME_HEADING_RATE_DEG: float = 90.0  # deg/s of heading change at full right stick
PAD_BTN_FLIP: int = 6               # top-right shoulder (R / RB): hold = flip mode
K_V = ord("V")                      # keyboard equivalent: V toggles the flip mode
# CAMERA -----------------------------------------------------------------
# Measured against MuJoCo's own camera maths (mjv_updateScene): the free camera's
# `azimuth` IS the direction it looks, in the world xy-plane, in degrees. So locking the
# view to the vehicle's heading is exactly `azimuth = yaw`, with the camera ending up
# behind the tail. CAMERA_YAW_OFFSET_DEG exists only to look from another angle.
CAMERA_LOCK_TO_HEADING: bool = True
CAMERA_YAW_OFFSET_DEG: float = 0.0
CAMERA_DISTANCE: float = 2.2
CAMERA_ELEVATION_DEG: float = -18.0
ACRO_RATE_MAX_DEG: float = 360.0   # deg/s of body rate at full stick (flips need 250+)
ACRO_KEY_RATE_DEG: float = 216.0   # deg/s a keyboard rotation pulse commands (60% of stick)
ACRO_KEY_ROT_DEG: float = 75.0     # rotation ONE key press delivers (as an angle budget)
ACRO_KEY_ROT_MAX_DEG: float = 720.0  # cap on the budget so rapid tapping stays sane
ACRO_THROTTLE_STEP: float = 0.06   # keyboard press = +-6% collective (a latched throttle)
ACRO_THROTTLE_RATE: float = 0.5    # 1/s of collective change while a trigger is held
ACRO_MATCH_KP: float = 2.2         # 1/s: match-velocity brake gain
ACRO_MATCH_MAX_ACCEL: float = 8.0  # m/s^2 cap on the match-velocity brake
ACRO_LEVEL_S: float = 2.0          # s the level assist keeps working after it is asked for

# --- gamepad (macOS HID game controller; reader lives in gamepad.py) -------------------
# The sticks are CONTINUOUS, so with a pad connected the latched-trim workaround is not
# needed: left stick = velocity command (springs back to hover), right stick = attitude
# (springs back to level), triggers = altitude setpoint, shoulders = yaw. The keyboard
# keeps working for every channel a pad is not currently driving.
PAD_ENABLE: bool = True
PAD_DEADZONE: float = 0.10        # ignore stick noise around the centre
PAD_EXPO: float = 0.35            # 0 = linear, 1 = cubic: finer control near the centre
V_STICK_MAX: float = 3.0          # m/s at full left-stick deflection
TILT_STICK_MAX_DEG: float = 35.0  # deg at full right-stick deflection
YAW_BUTTON_RATE_DEG: float = 90.0 # deg/s of yaw while LB/RB (L1/R1) is held
Z_TRIGGER_RATE: float = 1.5       # m/s of altitude-setpoint travel at full trigger
# Sign fixes. Most HID pads report "stick pushed forward" as a NEGATIVE value; if yours
# does not, the overlay shows the live stick values, so flip the flag that looks wrong.
PAD_INVERT_LEFT_X: bool = False
PAD_INVERT_LEFT_Y: bool = True
PAD_INVERT_RIGHT_X: bool = False
PAD_INVERT_RIGHT_Y: bool = True
# Button usages. HID buttons are numbered by the pad (1 = the first button in its
# descriptor, which is a face button on every controller). The terminal prints every
# button press with its usage number, and `scratch/gamepad_probe.py` prints them with the
# axis table, so an exotic pad is easy to re-map here. Defaults:
#   any face button except #2  -> hover (panic)
#   face button #2             -> respawn        (Xbox B, PlayStation Circle, Switch X/B)
#   5 / 6                      -> yaw left / right while held (Xbox LB/RB, PS L1/R1)
#   7 / 8                      -> descend / climb, ONLY on pads without analog triggers
#                                 (Nintendo Switch Pro: ZL/ZR are digital buttons)
PAD_BTN_HOVER: Tuple[int, ...] = (1, 3, 4)   # red button = hover
PAD_BTN_RESPAWN: int = 2
PAD_BTN_YAW_LEFT: int = 5
PAD_BTN_YAW_RIGHT: int = 6
PAD_BTN_DESCEND: int = 7
PAD_BTN_CLIMB: int = 8
# ACRO mode buttons. Match velocity sits on X / Square, the button the game uses for it;
# the two small system buttons work in both modes so a pad can always switch between the
# acro quad and the assisted pair.
PAD_BTN_MATCH_VELOCITY: int = 4    # X (Xbox) / Square (PS) / X (Switch top)
PAD_BTN_LEVEL: int = 1             # bottom face button: levelling assist
PAD_BTN_MODE: int = 3              # left face button: back to the assisted modes
PAD_BTN_MODE_ALT: int = 9          # minus / select
PAD_BTN_MATCH_ALT: int = 10        # plus / start
# ======================================================================================

# GLFW key codes as delivered to the viewer's key_callback (letters are ASCII, and they
# are the same codes for lower/upper case because GLFW reports the physical key).
K_W, K_A, K_S, K_D = ord("W"), ord("A"), ord("S"), ord("D")
K_I, K_J, K_K, K_L = ord("I"), ord("J"), ord("K"), ord("L")
K_Q, K_E = ord("Q"), ord("E")
K_R, K_F = ord("R"), ord("F")
K_SPACE, K_ESCAPE, K_ENTER = 32, 256, 257
K_LEFT_CTRL, K_RIGHT_CTRL = 341, 345
K_X, K_C, K_M = ord("X"), ord("C"), ord("M")

MODE_VELOCITY = "VELOCITY"
MODE_ANGLE = "ANGLE"
MODE_ACRO = "ACRO"
MODE_GAME = "GAME"

TILT_STEP_RAD = math.radians(TILT_STEP_DEG)
TILT_MAX_RAD = math.radians(TILT_MAX_DEG)


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]; the attitude error uses the shortest rotation anyway."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _shape_stick(value: float, deadzone: float = PAD_DEADZONE,
                 expo: float = PAD_EXPO) -> float:
    """Stick shaping: deadzone, then an expo curve. Returns 0 inside the deadzone."""
    v = float(value)
    if abs(v) <= deadzone:
        return 0.0
    v = math.copysign((abs(v) - deadzone) / max(1e-6, 1.0 - deadzone), v)
    return (1.0 - expo) * v + expo * v * v * v


def game_flip_thrust(t_elapsed: float, dcm22: float, vz: float, hover_thrust: float) -> float:
    """Collective thrust for the flip button: pop, ballistic window, arrest, banked hold.

    The shape is the one the trained flip uses (`trajectories.Flip`): the rotation happens
    with NO thrust, because thrust acts along body z and firing it while rolled ~90 deg
    pushes the vehicle sideways rather than up. Sequencing matters as much as the shape -
    measured: if the roll starts during the pop, the pop is spent sideways and the flip then
    free-falls over a metre.

        pop     t < GAME_FLIP_POP_S      ~2x hover, body upright, rotation gated off
        roll    steeply rolled (inverted) ZERO thrust: pure ballistic rotation
        arrest  upright again and sinking ~2x hover until the sink stops
        bank    mildly tilted             altitude hold, 1/z_b.z compensated for the tilt

    Measured with this profile: a full turn from 1.20 m stays inside 1.09..1.79 m and ends
    level and stationary, where a roll at frozen hover trim hit the floor.
    """
    if t_elapsed < GAME_FLIP_POP_S:
        return GAME_FLIP_POP_X * hover_thrust
    if dcm22 < GAME_FLIP_CATCH_DCM:
        return 0.0                       # inverted: ballistic, thrust would push sideways
    if vz < GAME_FLIP_CATCH_VZ:
        return GAME_FLIP_POP_X * hover_thrust      # upright and falling: catch it
    # Banked but not inverted: hold altitude. A tilted quad needs 1/cos(tilt) more thrust,
    # and beyond ~60 deg that exceeds the vehicle's authority - then the honest answer is to
    # ask for everything it has (the caller clips) rather than to pretend.
    return hover_thrust / max(GAME_FLIP_HOLD_DCM, dcm22)


def _update_camera(viewer, quad: QuadcopterMuJoCo) -> None:
    """Chase camera: follow the vehicle, and (optionally) orbit with its HEADING.

    `azimuth` is the direction the camera looks, in the world xy-plane, in degrees -
    measured with `mjv_updateScene` rather than assumed - so `azimuth = yaw` puts the
    camera behind the tail looking along the nose, which is the video-game view. Only the
    azimuth is forced: the mouse can still zoom and raise or lower the camera, it just
    cannot un-lock the heading.
    """
    viewer.cam.lookat[:] = quad.pos
    if CAMERA_LOCK_TO_HEADING:
        viewer.cam.azimuth = math.degrees(float(quad.psi)) + CAMERA_YAW_OFFSET_DEG


class ManualCommand:
    """Latched pilot command: one key press nudges one setpoint.

    The latch is a property of the viewer's input API (press events only, see the module
    docstring), not a flight-control decision - the setpoints it produces are the same
    quantities a held stick would command continuously, and the window overlay always
    shows their current values.
    """

    def __init__(self, spawn_z: float):
        self.reset(spawn_z)

    def reset(self, spawn_z: float) -> None:
        self.mode: str = START_MODE
        self.v_cmd = np.zeros(2, dtype=np.float64)     # body frame [forward, left], m/s
        self.tilt = np.zeros(2, dtype=np.float64)      # body frame [pitch, roll], rad
        self.yaw_des: float = 0.0                      # rad, wrapped
        self.z_des: float = float(spawn_z)             # m
        # ACRO channels: a latched collective (like a real throttle stick) plus momentary
        # body-rate commands from the pad and from keyboard rotation pulses.
        self.throttle: float = 0.458                   # 0..1 of max collective thrust
        self.rot_pad = np.zeros(3, dtype=np.float64)   # body rates from sticks, rad/s
        # Keyboard rotation is an ANGLE BUDGET, not a wall-clock pulse: a key press adds
        # ~75 deg of rotation, the controller spends it at ACRO_KEY_RATE_DEG and stops when
        # it runs out. Simulation time is the only clock that makes sense here - the flight
        # loop is paced in sim time, and a wall-clock pulse would behave differently
        # depending on how fast the machine renders.
        self._rot_budget = np.zeros(3, dtype=np.float64)
        self._rot_key_rate = math.radians(ACRO_KEY_RATE_DEG)
        self.match_velocity: bool = False
        self.level_now: bool = False                   # one-shot flag, cleared by the loop
        # GAME mode's flip button (held on the pad, toggled with V on the keyboard): while
        # it is on, the left stick commands body rates and the collective is flown on the
        # flip profile - see _game_flip_law.
        self.game_flip: bool = False
        # Gamepad channel ownership: a deflected stick takes its channel over, and
        # releasing it hands the channel back to the keyboard in a neutral state.
        self._pad_velocity_active: bool = False
        self._pad_attitude_active: bool = False

    # -- mode plumbing ----------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Public mode switch (the keyboard/pad handlers call _enter the same way)."""
        self._enter(mode)

    def _enter(self, mode: str) -> None:
        """Switch control mode. The channels of the mode being left stop being commanded."""
        if mode == self.mode:
            return
        self.mode = mode
        self.match_velocity = False
        self.game_flip = False
        # Whichever set we are leaving stops commanding anything, so a stale trim or a
        # half-spent rotation budget cannot leak across a mode switch.
        self.v_cmd[:] = 0.0
        self.tilt[:] = 0.0
        self.rot_pad[:] = 0.0
        self._rot_budget[:] = 0.0

    # -- ACRO helpers -----------------------------------------------------------------
    def push_rotation(self, axis: int, direction: float,
                      angle_deg: float = ACRO_KEY_ROT_DEG) -> None:
        """Add rotation to the keyboard budget on one body axis [deg, signed]."""
        limit = math.radians(ACRO_KEY_ROT_MAX_DEG)
        self._rot_budget[axis] = float(np.clip(
            self._rot_budget[axis] + math.copysign(math.radians(angle_deg), direction),
            -limit, limit))
        if abs(direction) > 0.0:
            self.match_velocity = False          # steering takes over from the assist

    def rot_rate(self, dt: Optional[float] = None) -> np.ndarray:
        """Current body-rate command: pad sticks when they are deflected, otherwise the
        remaining keyboard rotation budget. Passing `dt` spends that budget (the flight
        loop does; callers that just want to look pass nothing)."""
        rate = self.rot_pad.copy()
        for axis in range(3):
            budget = self._rot_budget[axis]
            if abs(rate[axis]) < 1e-9 and abs(budget) > 1e-9:
                pulse = math.copysign(self._rot_key_rate, budget)
                rate[axis] = pulse
                if dt is not None:
                    spent = min(abs(pulse) * dt, abs(budget))
                    self._rot_budget[axis] -= math.copysign(spent, budget)
        return rate

    def bump_throttle(self, delta: float) -> float:
        self.throttle = float(np.clip(self.throttle + delta, 0.0, 1.0))
        if abs(delta) > 0.0:
            self.match_velocity = False        # thrusting takes over, like the game
        return self.throttle

    def toggle_match_velocity(self) -> bool:
        self.match_velocity = not self.match_velocity
        return self.match_velocity

    # -- one key press each -----------------------------------------------------------
    def handle_key(self, key: int) -> Optional[str]:
        """Apply ONE key press. Returns a terminal status line, or None if unbound."""
        if key == K_M:
            return self.toggle_assist()
        if self.mode == MODE_ACRO:
            return self._handle_key_acro(key)
        if self.mode == MODE_GAME:
            return self._handle_key_game(key)
        return self._handle_key_assisted(key)

    def toggle_assist(self, label: str = "M  mode") -> str:
        """Cycle the control sets: GAME -> ACRO -> ASSISTED (VELOCITY/ANGLE) -> GAME."""
        if self.mode == MODE_GAME:
            self._enter(MODE_ACRO)
            return (f"{label:20s} -> ACRO: real quad, rate control, no auto-level, "
                    f"no auto-brake, X = match velocity")
        if self.mode == MODE_ACRO:
            self._enter(MODE_VELOCITY)
            return (f"{label:20s} -> ASSISTED (VELOCITY): holds the speed you command, "
                    f"auto-levels, brakes by itself")
        self._enter(MODE_GAME)
        return (f"{label:20s} -> GAME: left stick = movement, right stick X = heading, "
                f"ZL/ZR = altitude, top-right shoulder = flip")

    def _handle_key_acro(self, key: int) -> Optional[str]:
        """ACRO: momentary rotation (as an angle budget per press, since key release cannot
        be detected) and a latched collective. Signs are body-frame: +pitch = nose down,
        +roll = right, +yaw = left - so W/S and A/D travel forward/back and left/right, as
        they do in the game."""
        pulse = f"({ACRO_KEY_RATE_DEG:.0f} deg/s, {ACRO_KEY_ROT_DEG:.0f} deg per press)"
        if key == K_W or key == K_I:
            self.push_rotation(1, +1.0)
            return f"{'W  pitch fwd':20s} -> nose down {pulse}"
        if key == K_S or key == K_K:
            self.push_rotation(1, -1.0)
            return f"{'S  pitch back':20s} -> nose up {pulse}"
        if key == K_A:
            self.push_rotation(0, -1.0)
            return f"{'A  roll left':20s} -> tilt left {pulse}"
        if key == K_D:
            self.push_rotation(0, +1.0)
            return f"{'D  roll right':20s} -> tilt right {pulse}"
        if key == K_Q or key == K_J:
            self.push_rotation(2, +1.0)
            return f"{'Q/J yaw left':20s} -> nose left {pulse}"
        if key == K_E or key == K_L:
            self.push_rotation(2, -1.0)
            return f"{'E/L yaw right':20s} -> nose right {pulse}"
        if key == K_SPACE or key == K_R:
            self.bump_throttle(+ACRO_THROTTLE_STEP)
            return f"{'SPACE/R throttle+':20s} -> {100.0 * self.throttle:3.0f}% collective"
        if key in (K_LEFT_CTRL, K_RIGHT_CTRL, K_F):
            self.bump_throttle(-ACRO_THROTTLE_STEP)
            return f"{'CTRL/F throttle-':20s} -> {100.0 * self.throttle:3.0f}% collective"
        if key == K_X:
            on = self.toggle_match_velocity()
            return (f"{'X  match velocity':20s} -> "
                    f"{'ENGAGED (braking to a hover)' if on else 'released'}")
        if key == K_C:
            self.level_now = True
            return f"{'C  level':20s} -> levelling the attitude, yaw kept"
        return None

    def _handle_key_game(self, key: int) -> Optional[str]:
        """GAME keyboard: WASD = movement, Q/E = heading, R/F = altitude, SPACE = hover,
        V = the flip button (a toggle here - the viewer reports presses only).

        While the flip mode is ON the collective is FLOWN on the flip profile, so the
        thrust keys are inert and
        WASD/QE become roll/pitch/yaw pulses through the same angle-budget mechanism the
        ACRO keys use.
        """
        if key == K_V:
            self.game_flip = not self.game_flip
            self.rot_pad[:] = 0.0
            self._rot_budget[:] = 0.0
            if self.game_flip:
                return (f"{'V  flip ON':20s} -> collective flown (pop/cut/arrest), WASD = roll/pitch/yaw "
                        f"({ACRO_KEY_ROT_DEG:.0f} deg per press)")
            return f"{'V  flip off':20s} -> back to movement + altitude hold"
        if self.game_flip:
            # Rotation only: thrust, level and match-velocity have nothing to act on while
            # the collective is flown on the flip profile.
            if key in (K_SPACE, K_R, K_F, K_LEFT_CTRL, K_RIGHT_CTRL, K_X, K_C):
                return None
            return self._handle_key_acro(key)
        if key == K_W:
            return self._trim_velocity(0, +V_STEP, "W  forward", switch=False)
        if key == K_S:
            return self._trim_velocity(0, -V_STEP, "S  back", switch=False)
        if key == K_A:
            return self._trim_velocity(1, +V_STEP, "A  left", switch=False)
        if key == K_D:
            return self._trim_velocity(1, -V_STEP, "D  right", switch=False)
        if key == K_Q or key == K_E:
            # +yaw is a LEFT turn in this project's convention, so E turns right.
            return self._trim_yaw(+YAW_STEP_RAD if key == K_Q else -YAW_STEP_RAD,
                                  "Q  heading left" if key == K_Q else "E  heading right")
        if key == K_R:
            return self._trim_altitude(+Z_STEP, "R  climb")
        if key == K_F:
            return self._trim_altitude(-Z_STEP, "F  descend")
        if key == K_SPACE:
            return self.hover()
        return None

    def _handle_key_assisted(self, key: int) -> Optional[str]:
        if key == K_W:
            return self._trim_velocity(0, +V_STEP, "W  forward")
        if key == K_S:
            return self._trim_velocity(0, -V_STEP, "S  back")
        if key == K_A:
            return self._trim_velocity(1, +V_STEP, "A  left")
        if key == K_D:
            return self._trim_velocity(1, -V_STEP, "D  right")
        if key == K_I:
            return self._trim_angle(0, +TILT_STEP_RAD, "I  pitch nose down")
        if key == K_K:
            return self._trim_angle(0, -TILT_STEP_RAD, "K  pitch nose up")
        if key == K_J:
            return self._trim_angle(1, -TILT_STEP_RAD, "J  roll left")
        if key == K_L:
            return self._trim_angle(1, +TILT_STEP_RAD, "L  roll right")
        if key == K_Q:
            return self._trim_yaw(+YAW_STEP_RAD, "Q  yaw left")
        if key == K_E:
            return self._trim_yaw(-YAW_STEP_RAD, "E  yaw right")
        if key == K_R:
            return self._trim_altitude(+Z_STEP, "R  climb")
        if key == K_F:
            return self._trim_altitude(-Z_STEP, "F  descend")
        if key == K_SPACE:
            return self.hover()
        return None

    def _trim_velocity(self, axis: int, delta: float, label: str, switch: bool = True) -> str:
        # `switch=False` is what GAME mode uses: its movement keys are ordinary trims there,
        # and forcing VELOCITY would silently move the pilot out of the GAME set (which is
        # exactly what it used to do before GAME existed).
        if switch:
            self._enter(MODE_VELOCITY)
        self.v_cmd[axis] = float(np.clip(self.v_cmd[axis] + delta, -V_MAX, V_MAX))
        return (f"{label:20s} -> v_cmd {self.v_cmd[0]:+.2f} fwd / {self.v_cmd[1]:+.2f} left m/s")

    def _trim_angle(self, axis: int, delta: float, label: str) -> str:
        self._enter(MODE_ANGLE)
        self.tilt[axis] = float(np.clip(self.tilt[axis] + delta, -TILT_MAX_RAD, TILT_MAX_RAD))
        return (f"{label:20s} -> pitch {math.degrees(self.tilt[0]):+5.0f} deg / "
                f"roll {math.degrees(self.tilt[1]):+5.0f} deg")

    def _trim_yaw(self, delta: float, label: str) -> str:
        self.yaw_des = _wrap_pi(self.yaw_des + delta)
        return f"{label:20s} -> yaw {math.degrees(self.yaw_des):+5.0f} deg"

    def _trim_altitude(self, delta: float, label: str) -> str:
        self.z_des = float(np.clip(self.z_des + delta, Z_MIN, Z_MAX))
        return f"{label:20s} -> z_des {self.z_des:.2f} m"

    def hover(self, label: str = "SPACE  hover") -> str:
        """Panic button: stop, level, hold the current altitude.

        In the assisted set this also returns the command set to VELOCITY (leaving ANGLE),
        which is what the pad's hover button has always done. GAME mode STAYS in GAME -
        its velocity law already holds altitude, and switching sets under the pilot's
        fingers would silently change what the sticks mean.
        """
        if self.mode != MODE_GAME:
            self.mode = MODE_VELOCITY
        self.v_cmd[:] = 0.0
        self.tilt[:] = 0.0
        self.game_flip = False
        self.rot_pad[:] = 0.0
        self._rot_budget[:] = 0.0
        self.match_velocity = False
        return f"{label:20s} -> stop, level, hold z={self.z_des:.2f} m"

    # -- gamepad ----------------------------------------------------------------------
    def apply_gamepad(self, pad: "PadState", dt: float) -> List[str]:
        """Fold a gamepad snapshot into the same command state the keyboard writes.

        Sticks are absolute and own their channel while deflected: left stick = velocity
        command (body frame), right stick = attitude. A deflected right stick switches to
        ANGLE mode, and RELEASING it returns to VELOCITY mode with a level attitude, which
        is what makes the self-centering stick behave like a commercial drone stick (let
        go = level off, and the velocity loop brakes to the left stick's command).
        Shoulders/triggers/D-pad share the yaw and altitude setpoints with the keyboard.
        In ACRO mode the same sticks drive the real quad channels instead: left stick =
        throttle (Y) and roll (X), right stick = pitch (Y) and yaw (X), triggers = throttle,
        X/Square = match velocity, the bottom face button = level, Y = mode switch.
        GAME mode reads the sticks the video-game way instead: left = movement, right X =
        heading, ZL/ZR = altitude, the top-right shoulder = the flip button.

        Returns one-shot action tokens for the caller to announce/execute: "hover",
        "respawn" (the command state itself is already updated for both).
        """
        actions: List[str] = []
        if pad is None or not pad.connected:
            return actions
        if self.mode == MODE_ACRO:
            return self._apply_gamepad_acro(pad, dt)
        if self.mode == MODE_GAME:
            return self._apply_gamepad_game(pad, dt)
        return self._apply_gamepad_assisted(pad, dt)

    def _apply_gamepad_game(self, pad: "PadState", dt: float) -> List[str]:
        """GAME: the video-game mapping.

        left stick ... movement: forward/back + strafe, as a VELOCITY command in the
                       heading frame (the law banks the quad to fly it and brakes when the
                       stick is released)
        right stick X  heading: the heading setpoint is integrated from the stick, so the
                       quad turns while the stick is held and holds the new heading after
        ZR (right trigger button)  climb     ZL (left trigger button)  descend
        top-right shoulder (R/RB)  FLIP: while held the collective is flown on a flip
                       and the LEFT stick commands body rates - push right, roll right;
                       push left, roll left; push forward/back for pitch. Release and the
                       movement mapping comes straight back.
        """
        actions: List[str] = []
        lx = _shape_stick(-pad.left_x if PAD_INVERT_LEFT_X else pad.left_x)
        ly = _shape_stick(-pad.left_y if PAD_INVERT_LEFT_Y else pad.left_y)
        rx = _shape_stick(-pad.right_x if PAD_INVERT_RIGHT_X else pad.right_x)

        flip_held = PAD_BTN_FLIP in pad.buttons
        if flip_held != self.game_flip:
            self.game_flip = flip_held
            self.rot_pad[:] = 0.0
            self._rot_budget[:] = 0.0
            actions.append("flip ON (collective flown, left stick = roll/pitch)"
                           if flip_held else "flip off")

        if self.game_flip:
            # Thrust is flown by the flip profile, so the movement and altitude commands
            # stand still while the
            # pilot rolls; the stick feeds rates directly (+roll = right, +pitch = nose down).
            rate_max = math.radians(GAME_FLIP_RATE_DEG)
            self.v_cmd[:] = 0.0
            self.tilt[:] = 0.0
            self.rot_pad[:] = [lx * rate_max, ly * rate_max, -rx * rate_max * 0.15]
            return actions

        # --- left stick: movement in the heading frame --------------------------------
        if lx or ly:
            self._pad_velocity_active = True
            self.v_cmd[0] = ly * V_STICK_MAX          # + = forward
            self.v_cmd[1] = -lx * V_STICK_MAX         # v_cmd is [forward, LEFT]
        elif self._pad_velocity_active:
            self._pad_velocity_active = False
            self.v_cmd[:] = 0.0

        # --- right stick X: heading (right = turn right, which is -yaw here) -----------
        if abs(rx) > 1e-9:
            self.yaw_des = _wrap_pi(self.yaw_des - math.radians(GAME_HEADING_RATE_DEG) * rx * dt)

        # --- ZL / ZR: altitude. Buttons on pads without analog triggers (Switch Pro). ----
        throttle_axis = pad.right_trigger - pad.left_trigger
        if pad.analog_triggers == 0:
            if PAD_BTN_CLIMB in pad.buttons:
                throttle_axis += 1.0
            if PAD_BTN_DESCEND in pad.buttons:
                throttle_axis -= 1.0
        if abs(throttle_axis) > 0.05:
            self.z_des = float(np.clip(self.z_des + throttle_axis * Z_TRIGGER_RATE * dt,
                                       Z_MIN, Z_MAX))

        # --- button edges --------------------------------------------------------------
        for button in pad.pressed:
            if button in PAD_BTN_HOVER:
                self.hover("PAD  hover")
                actions.append("hover")
            elif button == PAD_BTN_RESPAWN:
                actions.append("respawn")
            elif button in (PAD_BTN_MODE_ALT, PAD_BTN_MODE):
                actions.append(self.toggle_assist("PAD  mode"))
            elif button == HAT_LEFT:
                self.yaw_des = _wrap_pi(self.yaw_des + YAW_STEP_RAD)
            elif button == HAT_RIGHT:
                self.yaw_des = _wrap_pi(self.yaw_des - YAW_STEP_RAD)
            elif button == HAT_UP:
                self.z_des = float(np.clip(self.z_des + Z_STEP, Z_MIN, Z_MAX))
            elif button == HAT_DOWN:
                self.z_des = float(np.clip(self.z_des - Z_STEP, Z_MIN, Z_MAX))
        return actions

    def _apply_gamepad_acro(self, pad: "PadState", dt: float) -> List[str]:
        """ACRO: the sticks drive the real quad channels (rates + collective).

        right stick ... pitch (Y) and yaw (X): the game's aiming input
        left stick .... X = roll rate, Y = throttle (hold to change, release to keep)
        triggers ...... throttle down / up
        LB / RB ....... roll (the game's roll shoulders)
        D-pad up/down . throttle steps (fine trim)
        X / Square .... Match Velocity toggle (the game's brake)
        bottom button . level attitude, yaw kept
        left button ... switch back to the assisted modes
        """
        actions: List[str] = []
        rate_max = math.radians(ACRO_RATE_MAX_DEG)
        # Rotation: signs are body-frame (+pitch = nose down, +roll = right, +yaw = left).
        pitch_stick = _shape_stick(-pad.right_y if PAD_INVERT_RIGHT_Y else pad.right_y)
        yaw_stick = _shape_stick(-pad.right_x if PAD_INVERT_RIGHT_X else pad.right_x)
        roll_stick = _shape_stick(-pad.left_x if PAD_INVERT_LEFT_X else pad.left_x)
        if PAD_BTN_YAW_LEFT in pad.buttons:      # shoulders are roll in this mode (Q/E)
            roll_stick -= 1.0
        if PAD_BTN_YAW_RIGHT in pad.buttons:
            roll_stick += 1.0
        roll_stick = float(np.clip(roll_stick, -1.0, 1.0))
        self.rot_pad[:] = [roll_stick * rate_max, pitch_stick * rate_max, -yaw_stick * rate_max]
        if any(abs(rate) > 1e-6 for rate in self.rot_pad):
            self.match_velocity = False          # steering takes over from the assist

        # Throttle: a latched collective, changed while a stick/trigger is held.
        throttle_axis = (pad.right_trigger - pad.left_trigger
                         + _shape_stick(-pad.left_y if PAD_INVERT_LEFT_Y else pad.left_y))
        if abs(throttle_axis) > 1e-6:
            self.bump_throttle(float(np.clip(throttle_axis, -1.0, 1.0)) * ACRO_THROTTLE_RATE * dt)
        if HAT_UP in pad.pressed:
            self.bump_throttle(+ACRO_THROTTLE_STEP)
        if HAT_DOWN in pad.pressed:
            self.bump_throttle(-ACRO_THROTTLE_STEP)

        for button in pad.pressed:
            if button == PAD_BTN_RESPAWN:
                actions.append("respawn")
            elif button in (PAD_BTN_MATCH_VELOCITY, PAD_BTN_MATCH_ALT):
                on = self.toggle_match_velocity()
                actions.append("match-velocity ENGAGED (braking to a hover)" if on
                               else "match velocity released")
            elif button == PAD_BTN_LEVEL:
                self.level_now = True
                actions.append("level")
            elif button in (PAD_BTN_MODE, PAD_BTN_MODE_ALT):
                actions.append(self.toggle_assist("PAD  mode"))
        return actions

    def _apply_gamepad_assisted(self, pad: "PadState", dt: float) -> List[str]:
        """Assisted modes: sticks command velocity/tilt setpoints (see apply_gamepad)."""
        actions: List[str] = []

        # --- right stick: attitude (do this first: it can switch modes) ---------------
        rx = _shape_stick(-pad.right_x if PAD_INVERT_RIGHT_X else pad.right_x)
        ry = _shape_stick(-pad.right_y if PAD_INVERT_RIGHT_Y else pad.right_y)
        if rx or ry:
            if not self._pad_attitude_active:
                self._pad_attitude_active = True
                self._enter(MODE_ANGLE)
            self.tilt[0] = ry * TILT_STICK_MAX_DEG * math.pi / 180.0    # + = nose down
            self.tilt[1] = rx * TILT_STICK_MAX_DEG * math.pi / 180.0    # + = roll right
        elif self._pad_attitude_active:
            self._pad_attitude_active = False
            self.tilt[:] = 0.0
            self._enter(MODE_VELOCITY)      # released: hand back to the braking assist loop

        # --- left stick: velocity command in the body frame ---------------------------
        lx = _shape_stick(-pad.left_x if PAD_INVERT_LEFT_X else pad.left_x)
        ly = _shape_stick(-pad.left_y if PAD_INVERT_LEFT_Y else pad.left_y)
        if lx or ly:
            self._pad_velocity_active = True
            self.v_cmd[0] = ly * V_STICK_MAX          # + = forward
            self.v_cmd[1] = -lx * V_STICK_MAX         # v_cmd is [forward, LEFT]
        elif self._pad_velocity_active:
            self._pad_velocity_active = False
            self.v_cmd[:] = 0.0

        # --- shoulders: yaw rate while held -------------------------------------------
        yaw_rate = 0.0
        if PAD_BTN_YAW_LEFT in pad.buttons:
            yaw_rate += YAW_BUTTON_RATE_DEG
        if PAD_BTN_YAW_RIGHT in pad.buttons:
            yaw_rate -= YAW_BUTTON_RATE_DEG
        if yaw_rate:
            self.yaw_des = _wrap_pi(self.yaw_des + math.radians(yaw_rate) * dt)

        # --- triggers: move the altitude setpoint (release holds the new altitude) -----
        # Analog triggers are ideal (Xbox/PlayStation). Pads without them (Switch Pro
        # reports ZL/ZR as buttons) fall back to buttons 7/8 as descend/climb while held.
        throttle_axis = pad.right_trigger - pad.left_trigger
        if pad.analog_triggers == 0:
            if PAD_BTN_CLIMB in pad.buttons:
                throttle_axis += 1.0
            if PAD_BTN_DESCEND in pad.buttons:
                throttle_axis -= 1.0
        if abs(throttle_axis) > 0.05:
            self.z_des = float(np.clip(self.z_des + throttle_axis * Z_TRIGGER_RATE * dt,
                                       Z_MIN, Z_MAX))

        # --- button edges --------------------------------------------------------------
        for button in pad.pressed:
            if button in PAD_BTN_HOVER:
                self.hover("PAD  hover")
                actions.append("hover")
            elif button == PAD_BTN_RESPAWN:
                actions.append("respawn")
            elif button in (PAD_BTN_MODE_ALT, PAD_BTN_MODE):
                actions.append(self.toggle_assist("PAD  mode"))
            elif button == HAT_LEFT:
                self.yaw_des = _wrap_pi(self.yaw_des + YAW_STEP_RAD)
            elif button == HAT_RIGHT:
                self.yaw_des = _wrap_pi(self.yaw_des - YAW_STEP_RAD)
            elif button == HAT_UP:
                self.z_des = float(np.clip(self.z_des + Z_STEP, Z_MIN, Z_MAX))
            elif button == HAT_DOWN:
                self.z_des = float(np.clip(self.z_des - Z_STEP, Z_MIN, Z_MAX))
        return actions


class ManualFlightController:
    """Keyboard setpoints -> (collective thrust [N], body-rate setpoint [rad/s]).

    The plant closes the rate loop itself at 1 kHz (`QuadcopterMuJoCo.update` with
    `rate_cmd=`), so this outer loop only has to be right at 100 Hz: velocity -> tilt ->
    attitude error -> body rate, plus the altitude -> collective channel. The rotation
    helpers are the environment's own (`QuadFlipEnv._attitude_error_rotvec`), so the
    manual pilot and the trained policy cannot disagree about what an attitude error is.
    """

    def __init__(self, quad: QuadcopterMuJoCo, command: ManualCommand):
        self.quad = quad
        self.cmd = command
        self.g = float(quad.params["g"])
        # Payload-aware mass: `body_mass` already includes anything the model carries.
        self.mass = float(quad.model.body_mass[quad.body_id])
        self.max_thrust = float(quad.params["maxThr"])
        self.a_tilt_max = self.g * math.tan(math.radians(MAX_TILT_DEG))
        self.kr = np.array([KR_RP, KR_RP, KR_YAW], dtype=np.float64)
        self.rate_max = np.array([MAX_RATE_RP, MAX_RATE_RP, MAX_RATE_YAW], dtype=np.float64)
        # Collective that exactly cancels weight: what a pilot trims to before take-off, and
        # the value the ACRO throttle starts at after a respawn.
        self.hover_throttle = float(np.clip(self.mass * self.g / max(1e-9, self.max_thrust),
                                            0.0, 1.0))
        self.z_int = 0.0
        # Which control law ran last, plus the snapshots the assists need. Switching laws
        # clears the shared altitude integrator and re-captures the hover/heading hold, so
        # an assist always engages from the state the pilot actually left the drone in.
        self._law = ""
        self._match_z = 0.0
        self._match_yaw = 0.0
        self._level_yaw = 0.0
        self._level_until = 0.0
        self._flip_t = 0.0            # time since the flip button engaged

    def reset(self) -> None:
        """Clear the integrator and the assist snapshots (called on respawn)."""
        self.z_int = 0.0
        self._law = ""
        self._level_until = 0.0

    # -- law selection ------------------------------------------------------------------
    def _select_law(self) -> str:
        """Which control law this step should run.

        A real acro quad flies on (collective, body rates) and nothing else - that is the
        ACRO law below. The two assists are conveniences a real flight controller also has:
        Match Velocity (X) is a position hold, and the level button is an attitude hold;
        both are cancelled the moment the pilot touches a rotation input or the throttle.
        """
        cmd = self.cmd
        if cmd.mode == MODE_GAME:
            # GAME's normal state IS the velocity law below (movement in the heading frame
            # with the altitude hold); the flip button swaps in a rate law with the
            # collective flown.
            return "game_flip" if cmd.game_flip else MODE_GAME
        if cmd.mode != MODE_ACRO:
            return cmd.mode
        if cmd.level_now:
            cmd.level_now = False
            self._level_yaw = float(self.quad.psi)
            self._level_until = time.time() + ACRO_LEVEL_S
            return "level"
        pilot_rate = cmd.rot_rate()
        if time.time() < self._level_until and not np.any(np.abs(pilot_rate) > 1e-9):
            return "level"
        if cmd.match_velocity:
            return "match"
        return "acro"

    def update(self) -> Tuple[float, np.ndarray]:
        law = self._select_law()
        if law != self._law:
            # Re-capture the snapshots on every law change: the match-velocity hold targets
            # the altitude and heading the pilot was actually flying at.
            self.z_int = 0.0
            if law == "game_flip":
                self._flip_t = 0.0            # the pop starts now
            if self._law == "game_flip" and law != "game_flip":
                # Releasing the flip button must not snap the heading back to where it was
                # before the roll: the movement law adopts whatever heading the roll left.
                self.cmd.yaw_des = float(self.quad.psi)
            self._law = law
            self._match_z = float(self.quad.pos[2])
            self._match_yaw = float(self.quad.psi)
        if law == "acro":
            return self._acro_law()
        if law == "match":
            # Match Velocity = the position hold a real flight controller has: brake to zero
            # ground velocity (which, in gravity, is a hover) and stay there.
            return self._velocity_law(np.zeros(2), self._match_z, self._match_yaw,
                                      kp_v=ACRO_MATCH_KP, a_xy_max=ACRO_MATCH_MAX_ACCEL)
        if law == "level":
            return self._level_law()
        if law == "game_flip":
            return self._game_flip_law()
        if law == MODE_ANGLE:
            return self._angle_law()
        return self._velocity_law(self.cmd.v_cmd, self.cmd.z_des, self.cmd.yaw_des)

    # -- the real acro-quad law ----------------------------------------------------------
    def _game_flip_law(self) -> Tuple[float, np.ndarray]:
        """The flip button: sequenced pop -> ballistic roll -> arrest, left stick = rates.

        The stick is pure rate control (full stick = GAME_FLIP_RATE_DEG) once the pop is
        over; before that the rotation is gated off so the pop actually goes up. The
        collective is flown by `game_flip_thrust`, not frozen at trim: a roll at hover
        thrust loses about a metre and hits the floor (measured), while this profile costs
        roughly 0.2 m of altitude for a full turn.
        """
        self._flip_t += DT
        rate = self.cmd.rot_rate(DT)
        thrust = game_flip_thrust(self._flip_t, float(self.quad.dcm[2, 2]),
                                  float(self.quad.vel[2]),
                                  self.hover_throttle * self.max_thrust)
        omega_des = np.clip(rate, -self.rate_max, self.rate_max)
        if self._flip_t < GAME_FLIP_POP_S:
            # The pop only works with the body upright, so the rotation is gated until it is
            # over (the stick is still read every step: the roll simply starts afterwards).
            omega_des[0] = 0.0
            omega_des[1] = 0.0
        return float(thrust), omega_des

    def _acro_law(self) -> Tuple[float, np.ndarray]:
        """Collective straight from the throttle, body rates straight from the pilot.

        This is exactly how a real acro quadcopter is flown: the throttle channel is
        COLLECTIVE thrust along body z, the other three channels are body-rate commands
        that the (real, 1 kHz) rate loop tracks. Nothing is stabilised for the pilot: no
        auto-levelling, no braking, and a drone left tilted keeps accelerating. The rates
        are clipped only to the vehicle's own authority (the same 20 rad/s roll/pitch and
        4 rad/s yaw the trained policy has), so the sticks cannot ask for more than the
        rotors can deliver.
        """
        thrust = float(np.clip(self.cmd.throttle, 0.0, 1.0)) * self.max_thrust
        omega_des = np.clip(self.cmd.rot_rate(DT), -self.rate_max, self.rate_max)
        return thrust, omega_des

    def _level_law(self) -> Tuple[float, np.ndarray]:
        """Attitude-hold assist (the pad's level button / C): roll and pitch to zero - the
        yaw captured when it engaged - while the pilot keeps the collective."""
        R_des = QuadFlipEnv._quat_to_dcm(
            QuadFlipEnv._euler_to_quat(0.0, 0.0, self._level_yaw))
        att_err = QuadFlipEnv._attitude_error_rotvec(R_des, self.quad.dcm)
        omega_des = np.clip(-self.kr * att_err, -self.rate_max, self.rate_max)
        thrust = float(np.clip(self.cmd.throttle, 0.0, 1.0)) * self.max_thrust
        return thrust, omega_des

    # -- assisted laws -------------------------------------------------------------------
    def _vertical_accel(self, z_des: float) -> float:
        """Cascaded altitude hold: position error -> bounded climb/descent rate -> accel."""
        q = self.quad
        e_z = z_des - q.pos[2]
        vz_des = float(np.clip(KP_VZ * e_z, -VZ_MAX, VZ_MAX))
        e_vz = vz_des - q.vel[2]
        raw_a_z = KV_Z * e_vz + self.z_int
        # Conditional integration (standard anti-windup): integrate unless the output is
        # already saturated in the direction the error pushes - a saturated descent must
        # not park the integrator at its limit and hold the vehicle below the setpoint.
        if (A_Z_MIN < raw_a_z < A_Z_MAX
                or (raw_a_z >= A_Z_MAX and e_vz < 0.0)
                or (raw_a_z <= A_Z_MIN and e_vz > 0.0)):
            self.z_int = float(np.clip(self.z_int + KI_VZ * DT * e_vz,
                                       -A_Z_INT_MAX, A_Z_INT_MAX))
        return float(np.clip(raw_a_z, A_Z_MIN, A_Z_MAX))

    def _velocity_law(self, v_cmd: np.ndarray, z_des: float, yaw_des: float,
                      kp_v: float = KP_V,
                      a_xy_max: Optional[float] = None) -> Tuple[float, np.ndarray]:
        q, g = self.quad, self.g
        a_z = self._vertical_accel(z_des)
        # --- horizontal: velocity error -> desired tilt -------------------------------
        # The command is given in the BODY frame (forward = where the nose points), so it
        # is rotated into the world frame with the current heading first; the pilot's
        # "forward" therefore always means "where the nose is pointing".
        cy, sy = math.cos(q.psi), math.sin(q.psi)
        v_des = np.array([
            cy * v_cmd[0] - sy * v_cmd[1],
            sy * v_cmd[0] + cy * v_cmd[1],
        ])
        a_xy = kp_v * (v_des - q.vel[:2])

        # Never ask for a tilt the vehicle cannot reach: clip the horizontal acceleration
        # so the commanded attitude stays inside the caller's limit (MAX_TILT_DEG for the
        # velocity loop, a tighter brake cap for match velocity).
        limit = self.a_tilt_max if a_xy_max is None else float(a_xy_max)
        a_h = float(np.linalg.norm(a_xy))
        if a_h > limit:
            a_xy = a_xy * (limit / a_h)

        # Flatness mapping: body z must point along (a + g e_z), which is exactly the
        # attitude that produces the requested acceleration at thrust |a + g e_z|.
        s_vec = np.array([a_xy[0], a_xy[1], a_z + g])
        R_des = dcm_from_thrust_dir_and_yaw(s_vec, yaw_des)
        # Thrust = component of the desired force along the CURRENT thrust axis: a
        # self-correcting law - if the vehicle lags the commanded tilt, it pushes harder,
        # and once aligned it produces exactly |a + g e_z|.
        thrust = self.mass * float(np.dot(s_vec, q.dcm[:, 2]))
        return self._attitude_to_rate(thrust, R_des)

    def _angle_law(self) -> Tuple[float, np.ndarray]:
        q, g, cmd = self.quad, self.g, self.cmd
        a_z = self._vertical_accel(cmd.z_des)
        # --- ANGLE: the commanded Euler angles ARE the attitude setpoint -----------------
        # Built with the environment's own ZYX helpers, so roll/pitch/yaw here mean the
        # same thing as `quad.euler` and as the trajectory references.
        R_des = QuadFlipEnv._quat_to_dcm(
            QuadFlipEnv._euler_to_quat(cmd.tilt[1], cmd.tilt[0], cmd.yaw_des))
        # Altitude hold must be compensated for the tilt: the vertical component of the
        # thrust is T * z_b[2], so holding a_z at a roll/pitch of theta needs
        # T = m (g + a_z) / cos(theta). Clipped at 0.25 (75 deg) so the law stays bounded
        # when the vehicle is on its side or inverted - past that the rotors saturate
        # anyway, and an inverted vehicle accelerates towards the ground, which is exactly
        # what a pilot at full collective would feel.
        support = float(max(q.dcm[2, 2], 0.25))
        thrust = self.mass * (g + a_z) / support
        return self._attitude_to_rate(thrust, R_des)

    def _attitude_to_rate(self, thrust: float,
                          R_des: np.ndarray) -> Tuple[float, np.ndarray]:
        """Finish an assisted law: clip the collective, turn the attitude error into a
        body-rate setpoint (rotation vector of R_des^T R: no double cover, no singularity
        at a half flip - the sign convention is checked in scratch/check_manual_flight.py)."""
        thrust = float(np.clip(thrust, 0.0, self.max_thrust))
        att_err = QuadFlipEnv._attitude_error_rotvec(R_des, self.quad.dcm)
        omega_des = np.clip(-self.kr * att_err, -self.rate_max, self.rate_max)
        return thrust, omega_des


# ======================================================================================
# FLIGHT LOOP
# ======================================================================================
def _new_log() -> Dict[str, list]:
    return {k: [] for k in (
        "t", "pos", "vel", "euler", "omega", "omega_des", "thrust", "w_motor",
        "v_cmd", "tilt", "z_des", "yaw_des", "mode", "throttle", "rate_cmd",
    )}


def _push(log: Dict[str, list], quad: QuadcopterMuJoCo, cmd: ManualCommand, t: float,
          thrust: float, omega_des: np.ndarray) -> None:
    log["t"].append(t)
    log["pos"].append(quad.pos.copy())
    log["vel"].append(quad.vel.copy())
    log["euler"].append(quad.euler.copy())
    log["omega"].append(quad.omega.copy())
    log["omega_des"].append(np.asarray(omega_des, dtype=np.float64).copy())
    log["thrust"].append(thrust)
    log["w_motor"].append(quad.wMotor.copy())
    log["v_cmd"].append(cmd.v_cmd.copy())
    log["tilt"].append(cmd.tilt.copy())
    log["z_des"].append(cmd.z_des)
    log["yaw_des"].append(cmd.yaw_des)
    log["mode"].append({MODE_GAME: 3, MODE_VELOCITY: 0, MODE_ANGLE: 1, MODE_ACRO: 2}[cmd.mode])
    log["throttle"].append(float(cmd.throttle))
    log["rate_cmd"].append(cmd.rot_rate().copy())


def _status_texts(cmd: ManualCommand, quad: QuadcopterMuJoCo, t: float,
                  crashed: bool, status: str, pad: Optional["PadState"] = None) -> List[tuple]:
    """Overlay text: setpoints (top left), live state (top right), help and pad below."""
    if crashed:
        head = f"STOPPED - {status}"
        sub = "ENTER: respawn    ESC: quit and plot"
    elif cmd.mode == MODE_ACRO:
        rate = np.degrees(cmd.rot_rate())
        head = "ACRO - real quad, rate control (no auto-level, no auto-brake)"
        sub = (f"throttle {100.0 * cmd.throttle:3.0f}%   "
               f"rate [p{rate[0]:+4.0f} q{rate[1]:+4.0f} r{rate[2]:+4.0f}] deg/s   "
               f"{'MATCH VELOCITY' if cmd.match_velocity else 'manual'}")
    elif cmd.mode == MODE_GAME:
        if cmd.game_flip:
            rate = np.degrees(cmd.rot_rate())
            head = "FLIP - collective flown (pop/cut/arrest), left stick = roll/pitch rates"
            sub = (f"rate [p{rate[0]:+4.0f} q{rate[1]:+4.0f} r{rate[2]:+4.0f}] deg/s   "
                   f"release the button / V to fly again")
        else:
            head = "GAME - left stick = movement, right stick X = heading, ZL/ZR = altitude"
            sub = (f"v_cmd {cmd.v_cmd[0]:+.2f} fwd / {cmd.v_cmd[1]:+.2f} left m/s   "
                   f"z_des {cmd.z_des:.2f} m   heading {math.degrees(cmd.yaw_des):+.0f} deg")
    elif cmd.mode == MODE_VELOCITY:
        head = "VELOCITY MODE (W/A/S/D  or left stick)"
        sub = (f"v_cmd {cmd.v_cmd[0]:+.2f} fwd / {cmd.v_cmd[1]:+.2f} left m/s   "
               f"z_des {cmd.z_des:.2f} m   yaw {math.degrees(cmd.yaw_des):+.0f} deg")
    else:
        head = "ANGLE MODE (I/J/K/L or right stick) - no self-braking"
        sub = (f"pitch {math.degrees(cmd.tilt[0]):+.0f} / roll {math.degrees(cmd.tilt[1]):+.0f} deg   "
               f"z_des {cmd.z_des:.2f} m   yaw {math.degrees(cmd.yaw_des):+.0f} deg")

    speed = float(np.linalg.norm(quad.vel))
    tilt = math.degrees(math.acos(float(np.clip(quad.dcm[2, 2], -1.0, 1.0))))
    rate_degs = np.degrees(quad.omega)
    live_left = (f"t {t:6.2f} s    speed {speed:5.2f} m/s    vz {quad.vel[2]:+5.2f} m/s    "
                 f"alt {quad.pos[2]:4.2f} m    tilt {tilt:4.0f} deg")
    live_right = (f"pos [{quad.pos[0]:+.2f} {quad.pos[1]:+.2f} {quad.pos[2]:+.2f}] m    "
                  f"rate [{rate_degs[0]:+5.0f} {rate_degs[1]:+5.0f} {rate_degs[2]:+5.0f}] deg/s")
    if cmd.mode == MODE_GAME:
        help_left = ("W/S fwd/back   A/D left/right   Q/E heading   R/F altitude   "
                     "SPACE hover   V flip mode")
        help_right = ("pad: left stick = movement, right stick X = heading, ZL/ZR = "
                      "altitude, top-right shoulder = FLIP (hold)   M next set   ENTER respawn")
    elif cmd.mode == MODE_ACRO:
        help_left = ("W/S pitch fwd/back   A/D roll   Q/E yaw   SPACE/CTRL or R/F throttle")
        help_right = ("X match velocity   C level   M assisted modes   ENTER respawn   "
                      "ESC quit   (keys = short pulses/bumps, pad sticks are continuous)")
    else:
        help_left = ("W/S fwd/back  A/D left/right   I/K pitch  J/L roll   Q/E yaw   R/F altitude")
        help_right = ("SPACE hover   M acro mode   ENTER respawn   ESC quit   "
                      "(keys trim, they do not hold)")
    texts = [
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, head, sub),
        (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT, live_left, live_right),
        (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, help_left, help_right),
    ]
    if pad is not None:
        if pad.connected:
            pad_left = f"PAD {pad.name[:34]}"
            pad_right = (f"L({pad.left_x:+.2f},{pad.left_y:+.2f}) R({pad.right_x:+.2f},{pad.right_y:+.2f}) "
                         f"T({pad.left_trigger:.2f},{pad.right_trigger:.2f}) b{sorted(pad.buttons)}")
        else:
            pad_left, pad_right = "PAD (none - keyboard only)", "plug a controller in any time"
        texts.append((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                      pad_left, pad_right))
    return texts


def _print_help() -> None:
    print("=" * 78)
    print("TWO CONTROL SETS - M (or the pad's minus/left-face button) switches between them.")
    print("-" * 78)
    print("=" * 78)
    print("GAME (default): video-game stick layout, with a flip button.")
    print("  left stick ....... movement: forward/back and left/right (velocity command)")
    print("  right stick X .... heading (turn while held, then hold the new heading)")
    print("  ZR / ZL .......... climb / descend (right / left trigger button)")
    print("  TOP-RIGHT shoulder  FLIP: hold it - the collective locks at hover trim and")
    print("                      the LEFT stick rolls/pitches directly (right = roll right)")
    print("    keyboard: W/A/S/D movement, Q/E heading, R/F altitude, V = flip toggle,")
    print("              SPACE = hover (stop, level, hold altitude)")
    print("-" * 78)
    print("ACRO: a REAL quad - collective + body rates, exactly like an acro flight")
    print("  drone's flight controller. No auto-levelling, no auto-braking: a drone left")
    print("  tilted keeps accelerating, so counter-tilt (or Match Velocity) to stop.")
    print("    W / S ........ pitch nose down / up  -> travel forward / back")
    print("    A / D ........ roll left / right     -> travel left / right")
    print("    Q / E (J/L) .. yaw left / right")
    print("    I / K ........ pitch (same as W/S, the game's aiming input)")
    print("    SPACE / R .... more collective       CTRL / F .... less collective")
    print("    X ............ MATCH VELOCITY: brake to a hover and hold it")
    print("    C ............ level the attitude, keeping the heading")
    print("  Keyboard rotation = short pulses (~75 deg each); the pad sticks are continuous.")
    print("  The throttle is LATCHED (like a real throttle stick): press to change it.")
    print("-" * 78)
    print("ASSISTED (the earlier set): VELOCITY mode holds the speed you command and brakes")
    print("  for you; ANGLE mode holds the tilt you command.")
    print("    W / S ........ fly forward / back            (0.5 m/s per press, +-2.5 max)")
    print("    A / D ........ fly left / right              (0.5 m/s per press, +-2.5 max)")
    print("    I / K ........ pitch nose down / up          (5 deg per press, +-80 max)")
    print("    J / L ........ roll left / right             (5 deg per press, +-80 max)")
    print("    Q / E ........ yaw setpoint left / right     (15 deg per press)")
    print("    R / F ........ altitude up / down            (0.25 m per press)")
    print("    SPACE ........ hover: stop, level, clear trims")
    print("-" * 78)
    print("ALWAYS: ENTER = respawn    ESC = quit (prints the summary and writes the plot)")
    print("-" * 78)
    print("GAMEPAD (any HID controller: Xbox / DualSense / DualShock / Switch Pro / 8BitDo):")
    print("  (Switch-style pads that only stream their own report format are decoded directly)")
    print("  GAME mode: left stick = movement, right stick X = heading, ZL/ZR = altitude,")
    print("             TOP-RIGHT shoulder = FLIP (hold; collective locks, stick rolls)")
    print("  ACRO mode: right stick = pitch/yaw, left stick = roll + throttle,")
    print("             LB/RB = roll, triggers = throttle, X/Square = MATCH VELOCITY,")
    print("             bottom face button = level, left face button = assisted modes")
    print("  ASSISTED: left stick = fly, right stick = tilt, LB/RB = yaw,")
    print("             triggers (or buttons 7/8) = altitude, face buttons = hover,")
    print("             button 2 = respawn, D-pad = altitude/yaw steps")
    print("  In both: minus/select (9) switches between the acro quad and the assists.")
    print("  Sticks own their channel while deflected; the keyboard drives the rest.")
    print("=" * 78)


def run_manual(
    playback_speed: float = PLAYBACK_SPEED,
    show_plots: bool = SHOW_PLOTS,
    camera_follow: bool = CAMERA_FOLLOW,
    verbose_keys: bool = VERBOSE_KEYS,
) -> Dict[str, np.ndarray]:
    """Fly by hand until ESC, the window closes, or the vehicle hits the ground."""
    quad = QuadcopterMuJoCo()
    rate_pid = RatePIDController(max_torque_xy=0.01, max_torque_z=0.003)
    wind = Wind("PERLIN", WIND_SPEED) if WIND_SPEED > 0.0 else Wind("NONE")
    command = ManualCommand(spawn_z=SPAWN_POS[2])
    controller = ManualFlightController(quad, command)

    def respawn() -> float:
        """Back to the start state: level, stationary, trims cleared, clock restarted."""
        quad.reset(pos=np.array(SPAWN_POS, dtype=np.float64),
                   quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                   thrust_scale=1.0)
        rate_pid.reset()
        controller.reset()
        command.reset(SPAWN_POS[2])
        # Trim the collective to exactly the hover value, like a pilot would before take-off.
        command.throttle = controller.hover_throttle
        return 0.0

    _print_help()
    print(f"Start state: pos={np.round(SPAWN_POS, 2)} m, level, at rest. "
          f"Wind: {WIND_SPEED:.2f} m/s.")

    # ---- gamepad (optional) -------------------------------------------------------------- 
    # The reader lives in gamepad.py (IOKit HID, own thread - mjpython owns the main one).
    # A missing or unplugged controller is never fatal: the sim falls back to the keyboard,
    # and a controller plugged in mid-flight is picked up as soon as it enumerates.
    pad = None
    if PAD_ENABLE:
        if HIDGamepad is None:
            print(f"Gamepad: unavailable on this platform ({_PAD_IMPORT_ERROR}); keyboard only.")
        else:
            pad = HIDGamepad()
            if pad.available:
                pad.start()
                print("Gamepad: HID reader running - plug a controller in before or during the run.")
            else:
                print(f"Gamepad: IOKit unavailable ({pad.error}); keyboard only.")
                pad = None

    # Key presses arrive on the viewer's UI thread; a list append is atomic under the GIL,
    # and the flight loop drains the queue at the top of every step.
    pending_keys: List[int] = []

    def key_callback(keycode: int) -> None:
        pending_keys.append(int(keycode))

    try:
        viewer = mujoco.viewer.launch_passive(quad.model, quad.data, key_callback=key_callback)
    except Exception as exc:
        print(f"\n[Error] Could not open the interactive viewer: {exc}")
        print("        On macOS this script must run through mjpython (it re-execs itself "
              "automatically when started with .venv/bin/python).")
        return {}

    viewer.cam.lookat[:] = SPAWN_POS
    viewer.cam.distance = CAMERA_DISTANCE
    viewer.cam.elevation = CAMERA_ELEVATION_DEG
    if CAMERA_LOCK_TO_HEADING:
        print("Camera locked to the vehicle's heading (mouse can still zoom/raise it).")
    print("Viewer open. Fly!\n")

    log = _new_log()
    t = respawn()
    crashed = False
    status = "flying"
    running = True
    last_overlay: Optional[str] = None
    pad_was_connected = False

    try:
        while running and viewer.is_running():
            step_start = time.time()

            # ---- gamepad -----------------------------------------------------------
            pad_state = pad.state() if pad is not None else None
            if pad_state is not None:
                if pad_state.connected != pad_was_connected:
                    pad_was_connected = pad_state.connected
                    if pad_state.connected:
                        print(f"[PAD] connected: {pad_state.name}")
                        if command.mode == MODE_GAME:
                            print("[PAD] GAME: left stick = movement, right stick X = heading, "
                                  "ZL/ZR = altitude, top-right shoulder = FLIP (hold)")
                        elif command.mode == MODE_ACRO:
                            print("[PAD] ACRO: left stick = roll + throttle, right stick = pitch "
                                  "+ yaw, triggers = throttle, X = match velocity")
                        else:
                            print("[PAD] ASSISTED: left stick = velocity, right stick = tilt, "
                                  "triggers = altitude, LB/RB = yaw, A = hover, B = respawn")
                    else:
                        print("[PAD] disconnected - keyboard only")
                for action in command.apply_gamepad(pad_state, DT):
                    if action == "respawn":
                        t = respawn()
                        crashed = False
                        status = "flying"
                        print(f"[t=0.00s] {'PAD  respawn':20s} -> back to the start state")
                    elif action == "hover":
                        print(f"[t={t:7.2f}s] {'PAD  hover':20s} -> stop, level, "
                              f"hold z={command.z_des:.2f} m")
                    else:
                        print(f"[t={t:7.2f}s] PAD  {action}")

            # ---- keyboard ----------------------------------------------------------
            keys, pending_keys[:] = list(pending_keys), []
            for key in keys:
                if key == K_ESCAPE:
                    print("\n[ESC] closing the viewer ...")
                    running = False
                    break
                if key == K_ENTER:
                    t = respawn()
                    crashed = False
                    status = "flying"
                    print(f"[ENTER] respawned at {np.round(SPAWN_POS, 2)} m")
                    continue
                message = command.handle_key(key)
                if message and verbose_keys:
                    print(f"[t={t:7.2f}s] {message}")
            if not running:
                break

            # ---- control + physics -------------------------------------------------
            if not crashed:
                thrust, omega_des = controller.update()
                quad.update(t, DT, wind=wind, rate_cmd=(thrust, omega_des), rate_pid=rate_pid)
                t += DT
                _push(log, quad, command, t, thrust, omega_des)

                # The only two stops besides ESC. No flight volume, no time limit: the
                # vehicle is free to go anywhere as long as it stays in one piece.
                if not np.all(np.isfinite(quad.state)):
                    crashed, status = True, "divergent state (NaN)"
                elif quad.check_ground_contact():
                    crashed, status = True, "ground contact"
                if crashed:
                    vz = float(quad.vel[2])
                    print(f"\n[STOP] {status} at t={t:.2f} s, "
                          f"pos=[{quad.pos[0]:+.2f} {quad.pos[1]:+.2f} {quad.pos[2]:+.2f}] m, "
                          f"vel=[{quad.vel[0]:+.2f} {quad.vel[1]:+.2f} {quad.vel[2]:+.2f}] m/s "
                          f"(vz {vz:+.2f})")
                    print("       ENTER: respawn    ESC: quit and plot\n")

            # ---- viewer ------------------------------------------------------------
            if camera_follow:
                _update_camera(viewer, quad)
            overlay = _status_texts(command, quad, t, crashed, status, pad_state)
            key_text = repr(overlay)
            if key_text != last_overlay:
                try:
                    viewer.set_texts(overlay)
                    last_overlay = key_text
                except Exception:
                    pass
            viewer.sync()

            # Real-time pacing: the physics is stepped exactly DT per iteration, the sleep
            # only keeps the loop from running faster than wall-clock (playback_speed < 1
            # gives slow motion, e.g. 0.25 for 4x).
            slack = DT / max(0.05, playback_speed) - (time.time() - step_start)
            if slack > 0.0:
                time.sleep(slack)
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        if pad is not None:
            pad.stop()

    return _summarise(log, crashed, status, plot=show_plots)


# ======================================================================================
# SUMMARY + TELEMETRY
# ======================================================================================
def _summarise(log: Dict[str, list], crashed: bool, status: str,
               plot: bool = True) -> Dict[str, np.ndarray]:
    if not log["t"]:
        print("No telemetry recorded - the viewer closed before the first step.")
        return {}

    data = {k: np.asarray(v, dtype=np.float64) for k, v in log.items()}
    t = data["t"]
    speed = np.linalg.norm(data["vel"], axis=1)
    # Attitude is reported per Euler axis (what a pilot reads off a display); the total
    # tilt from vertical is shown live in the window overlay instead.
    horiz = data["pos"][:, :2]
    travel = float(np.sum(np.linalg.norm(np.diff(horiz, axis=0), axis=1))) if len(t) > 1 else 0.0

    print("\n" + "=" * 78)
    print("FLIGHT SUMMARY")
    modes = data["mode"]
    shares = [(name, float(np.mean(modes == code)))
              for name, code in ((MODE_GAME, 3), (MODE_ACRO, 2), (MODE_VELOCITY, 0),
                                 (MODE_ANGLE, 1))]
    share_text = "  ".join(f"{name.lower()} {100.0 * frac:.0f}%"
                           for name, frac in shares if frac > 0.005) or "n/a"
    print(f"  Duration    : {t[-1]:.2f} s ({len(t)} steps at {1.0 / DT:.0f} Hz) | "
          f"mode share: {share_text}")
    print(f"  Max speed   : {speed.max():.2f} m/s")
    print(f"  Attitude    : roll {np.degrees(data['euler'][:, 0]).min():+.0f}.."
          f"{np.degrees(data['euler'][:, 0]).max():+.0f} deg, "
          f"pitch {np.degrees(data['euler'][:, 1]).min():+.0f}.."
          f"{np.degrees(data['euler'][:, 1]).max():+.0f} deg, "
          f"yaw {np.degrees(data['euler'][:, 2]).min():+.0f}.."
          f"{np.degrees(data['euler'][:, 2]).max():+.0f} deg")
    # In ACRO the pilot owns the collective, so there is no altitude setpoint to compare
    # against; only mention one if the assisted loops were actually flown.
    target = (f" (last assisted target {data['z_des'][-1]:.2f} m)"
              if bool(np.any(modes != 2)) else "")
    print(f"  Altitude    : {data['pos'][:, 2].min():.2f} .. {data['pos'][:, 2].max():.2f} m"
          f"{target}")
    print(f"  Travel      : {travel:.1f} m of horizontal path, "
          f"final position [{data['pos'][-1, 0]:+.2f} {data['pos'][-1, 1]:+.2f} "
          f"{data['pos'][-1, 2]:+.2f}] m")
    print(f"  End state   : {'STOPPED - ' + status if crashed else 'quit by the pilot'}")
    if plot:
        save_path = TELEMETRY_PATH
        try:
            _plot_telemetry(data, save_path)
            print(f"  Telemetry   : {save_path}")
        except Exception as exc:
            print(f"  Telemetry   : plotting failed ({exc})")
    print("=" * 78 + "\n")
    return data


def _plot_telemetry(data: Dict[str, np.ndarray], save_path: str) -> None:
    """Six panels: position, speed, attitude, body rates, collective, motor speeds."""
    t = data["t"]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 9.0), sharex=True)
    font = 9

    ax = axes[0, 0]
    ax.plot(t, data["pos"][:, 0], label="x")
    ax.plot(t, data["pos"][:, 1], label="y")
    ax.plot(t, data["pos"][:, 2], label="z", color="tab:green")
    ax.plot(t, data["z_des"], "--", color="tab:green", alpha=0.6, label="z target")
    ax.set_ylabel("position [m]", fontsize=font)
    ax.legend(fontsize=font, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, np.linalg.norm(data["vel"], axis=1), label="|v|", color="tab:red")
    ax.plot(t, data["vel"][:, 2], label="vz", alpha=0.7)
    ax.plot(t, np.linalg.norm(data["v_cmd"], axis=1), "--", color="tab:gray",
            label="|v| command")
    ax.set_ylabel("speed [m/s]", fontsize=font)
    ax.legend(fontsize=font, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for i, name in enumerate(("roll", "pitch", "yaw")):
        ax.plot(t, np.degrees(data["euler"][:, i]), label=name)
    ax.plot(t, np.degrees(data["tilt"][:, 0]), "--", color="tab:orange", alpha=0.7,
            label="pitch trim")
    ax.plot(t, np.degrees(data["tilt"][:, 1]), "--", color="tab:red", alpha=0.7,
            label="roll trim")
    ax.set_ylabel("attitude [deg]", fontsize=font)
    ax.legend(fontsize=font, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for i, name in enumerate(("p (roll)", "q (pitch)", "r (yaw)")):
        ax.plot(t, data["omega"][:, i], label=name)
        ax.plot(t, data["omega_des"][:, i], "--", alpha=0.6)
    ax.set_ylabel("body rate [rad/s]", fontsize=font)
    ax.legend(fontsize=font, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.plot(t, data["thrust"], color="tab:purple", label="collective [N]")
    hover = data["thrust"][0] if len(data["thrust"]) else 0.0
    ax.axhline(hover, color="tab:gray", ls=":", label=f"initial {hover:.3f} N")
    ax.set_ylabel("thrust [N]", fontsize=font)
    ax.set_xlabel("time [s]", fontsize=font)
    ax.legend(fontsize=font, loc="upper left")
    ax.grid(alpha=0.3)
    # The pilot's latched throttle (ACRO has no altitude loop, so this is the input).
    ax_t = ax.twinx()
    ax_t.plot(t, 100.0 * data["throttle"], color="tab:olive", lw=1.0, alpha=0.7,
              label="throttle [%]")
    ax_t.set_ylabel("throttle [%]", fontsize=font, color="tab:olive")
    ax_t.tick_params(axis="y", labelcolor="tab:olive")
    ax_t.set_ylim(0.0, 100.0)

    ax = axes[2, 1]
    for i in range(4):
        ax.plot(t, data["w_motor"][:, i], label=f"motor {i + 1}")
    ax.set_ylabel("motor speed [rad/s]", fontsize=font)
    ax.set_xlabel("time [s]", fontsize=font)
    ax.legend(fontsize=font, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle("Manual flight telemetry")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    # Only the main thread can own a window; under mjpython (script on a background
    # thread) the file above is the deliverable and show() is skipped.
    if _WANTS_WINDOW:
        try:
            if plt.get_backend().lower() != "agg":
                plt.show(block=False)
        except Exception as exc:
            print(f"Note: interactive plot display failed ({exc}); use the saved PNG.")
    plt.close(fig)


if __name__ == "__main__":
    # macOS GUI trampoline: launch_passive requires mjpython for interactive windowing
    # (identical to evaluate.py).
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

    run_manual()
