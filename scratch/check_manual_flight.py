# -*- coding: utf-8 -*-
"""
Headless validation for Simulation/manual_flight.py.

The keyboard itself cannot be tested from here (the viewer only reports key presses to a
GUI callback), so this script drives the SAME ManualCommand / ManualFlightController the
window drives, by calling `handle_key` with the same GLFW codes the viewer delivers, and
asserts the closed-loop response of the real plant:

  A. the key map itself (every bound key trims exactly what it advertises)
  B. VELOCITY mode: hover is stationary, a velocity command is reached and held, SPACE
     brakes the vehicle, and forward/left really are forward/left
  C. ANGLE mode: the commanded tilt is reached AND held (no self-braking), roll/pitch/yaw
     signs match the vehicle's own Euler readout
  D. altitude hold: R/F move the setpoint, the loop follows it in both modes
  E. the two stop conditions the flight loop can trigger (ground contact, divergence)

Run:  .venv/bin/python scratch/check_manual_flight.py
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")          # never open a window from a test

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from manual_flight import (  # noqa: E402
    DT,
    K_A,
    K_C,
    K_D,
    K_E,
    K_F,
    K_I,
    K_J,
    K_K,
    K_L,
    K_Q,
    K_R,
    K_S,
    K_SPACE,
    K_W,
    K_X,
    MODE_ACRO,
    MODE_ANGLE,
    MODE_GAME,
    MODE_VELOCITY,
    ManualCommand,
    ManualFlightController,
    SPAWN_POS,
    START_MODE,
)
from quadFiles.quad_mujoco import QuadcopterMuJoCo  # noqa: E402
from utils.rate_pid import RatePIDController  # noqa: E402

import manual_flight as mf  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


class Flight:
    """One sortie: the exact objects and call sequence manual_flight.run_manual uses."""

    def __init__(self, mode: str = MODE_VELOCITY):
        self.quad = QuadcopterMuJoCo()
        self.pid = RatePIDController(max_torque_xy=0.01, max_torque_z=0.003)
        self.cmd = ManualCommand(spawn_z=SPAWN_POS[2])
        self.ctrl = ManualFlightController(self.quad, self.cmd)
        self.cmd.set_mode(mode)
        self.cmd.throttle = self.ctrl.hover_throttle      # pilot trims to hover
        self.quad.reset(pos=np.array(SPAWN_POS, dtype=np.float64),
                        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                        thrust_scale=1.0)
        self.t = 0.0
        self.log: dict[str, list] = {k: [] for k in
                                     ("t", "pos", "vel", "euler", "omega", "omega_des", "tilt")}

    def press(self, *keys: int) -> None:
        for key in keys:
            self.cmd.handle_key(key)

    def hover_at(self, z: float) -> "Flight":
        """Teleport into a settled hover at altitude z.

        ACRO has no altitude loop: the pilot owns the collective, so a physics check only
        means something when the drone starts level, throttled to hover and not moving.
        """
        self.quad.reset(pos=np.array([0.0, 0.0, z], dtype=np.float64),
                        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                        thrust_scale=1.0)
        self.cmd.throttle = self.ctrl.hover_throttle
        self.cmd.rot_pad[:] = 0.0
        self.cmd._rot_budget[:] = 0.0
        return self.run(0.6)

    def run(self, seconds: float) -> "Flight":
        for _ in range(int(round(seconds / DT))):
            thrust, omega_des = self.ctrl.update()
            self.quad.update(self.t, DT, rate_cmd=(thrust, omega_des), rate_pid=self.pid)
            self.t += DT
            self.log["t"].append(self.t)
            self.log["pos"].append(self.quad.pos.copy())
            self.log["vel"].append(self.quad.vel.copy())
            self.log["euler"].append(self.quad.euler.copy())
            self.log["omega"].append(self.quad.omega.copy())
            self.log["omega_des"].append(np.asarray(omega_des).copy())
            self.log["tilt"].append(self.cmd.tilt.copy())
        return self

    def arr(self, key: str) -> np.ndarray:
        return np.asarray(self.log[key], dtype=np.float64)

    # -- convenience ---------------------------------------------------------------
    def pos(self) -> np.ndarray:
        return self.quad.pos.copy()

    def vel(self) -> np.ndarray:
        return self.quad.vel.copy()

    def euler_deg(self) -> np.ndarray:
        return np.degrees(self.quad.euler.copy())


print("=" * 78)
print("A. key map (each press trims the setpoint it advertises)")
print("=" * 78)
print(f"   (default start mode is {START_MODE}; the assisted sections pin VELOCITY first)")
cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
for key in (K_W, K_S, K_A, K_D, K_I, K_K, K_J, K_L, K_Q, K_E, K_R, K_F, K_SPACE):
    check(f"key {key} is bound", cmd.handle_key(key) is not None)
check("unbound key is ignored", cmd.handle_key(ord("Z")) is None)

cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
cmd.handle_key(K_W)
check("W: velocity command forward", abs(cmd.v_cmd[0] - 0.5) < 1e-12 and cmd.mode == MODE_VELOCITY,
      f"v_cmd={cmd.v_cmd}, mode={cmd.mode}")
cmd.handle_key(K_I)
cmd.handle_key(K_I)
check("I x2: +10 deg pitch, mode switches to ANGLE",
      abs(math.degrees(cmd.tilt[0]) - 10.0) < 1e-9 and cmd.mode == MODE_ANGLE,
      f"pitch={math.degrees(cmd.tilt[0]):.2f} deg, mode={cmd.mode}")
check("switching modes clears the channel being left",
      np.allclose(cmd.v_cmd, 0.0), f"v_cmd={cmd.v_cmd}")
cmd.handle_key(K_W)
check("back to VELOCITY: tilt trim is dropped",
      np.allclose(cmd.tilt, 0.0) and abs(cmd.v_cmd[0] - 0.5) < 1e-12)
cmd.handle_key(K_W)
cmd.handle_key(K_W)
cmd.handle_key(K_W)
cmd.handle_key(K_W)
cmd.handle_key(K_W)
cmd.handle_key(K_W)
cmd.handle_key(K_W)
cmd.handle_key(K_W)
check("velocity command saturates at V_MAX", cmd.v_cmd[0] <= 2.5 + 1e-12,
      f"v_cmd_fwd={cmd.v_cmd[0]:.2f}")
cmd.handle_key(K_SPACE)
check("SPACE: stop + level + velocity mode",
      np.allclose(cmd.v_cmd, 0.0) and np.allclose(cmd.tilt, 0.0) and cmd.mode == MODE_VELOCITY)
cmd.handle_key(K_R)
check("R: altitude setpoint steps by 0.25 m", abs(cmd.z_des - 1.45) < 1e-12,
      f"z_des={cmd.z_des:.2f}")
cmd.handle_key(K_Q)
check("Q: yaw setpoint steps by 15 deg", abs(math.degrees(cmd.yaw_des) - 15.0) < 1e-9)

print()
print("=" * 78)
print("B. VELOCITY mode: hover, command tracking, braking, axis signs")
print("=" * 78)
f = Flight().run(4.0)
drift = float(np.linalg.norm(f.pos()[:2]))
check("idle hover: holds altitude", abs(f.pos()[2] - SPAWN_POS[2]) < 0.05,
      f"z={f.pos()[2]:.3f} m")
check("idle hover: does not drift", drift < 0.05, f"|xy|={drift * 1000:.0f} mm")
check("idle hover: stays level", float(np.max(np.abs(f.euler_deg()[:2]))) < 2.0,
      f"max |roll,pitch|={float(np.max(np.abs(f.euler_deg()[:2]))):.2f} deg")

f = Flight()
f.press(K_W, K_W, K_W)                     # +1.5 m/s forward
f.run(4.0)
v = f.vel()
check("W: forward speed reaches the command", abs(v[0] - 1.5) < 0.15,
      f"vx={v[0]:+.2f} m/s (commanded +1.50)")
check("W: no sideways motion", abs(v[1]) < 0.05, f"vy={v[1]:+.3f} m/s")
check("W: altitude held while flying", abs(f.pos()[2] - SPAWN_POS[2]) < 0.10,
      f"z={f.pos()[2]:.3f} m")
check("W: it flew in +x", f.pos()[0] > 2.0, f"x={f.pos()[0]:+.2f} m")
check("W: nose pitched DOWN in the transient (x_body points into the motion)",
      float(np.max(np.degrees(f.arr('euler')[:200, 1]))) > 3.0,
      f"peak pitch={float(np.max(np.degrees(f.arr('euler')[:200, 1]))):+.1f} deg")

f.run(1.5)                                 # keep flying, then brake
f.press(K_SPACE)
f.run(2.5)
check("SPACE: brakes to a stop", float(np.linalg.norm(f.vel())) < 0.10,
      f"|v|={float(np.linalg.norm(f.vel())):.3f} m/s")

f = Flight()
f.press(K_A, K_A)                          # +1.0 m/s LEFT (body y)
f.run(4.0)
check("A: flies left (+y) at the commanded speed",
      abs(f.vel()[1] - 1.0) < 0.15 and f.pos()[1] > 1.0,
      f"vy={f.vel()[1]:+.2f} m/s, y={f.pos()[1]:+.2f} m")
check("A: no forward motion", abs(f.vel()[0]) < 0.05, f"vx={f.vel()[0]:+.3f} m/s")

f = Flight()
f.press(K_D)                               # -0.5 m/s right
f.run(3.0)
check("D: flies right (-y)", f.vel()[1] < -0.35 and f.pos()[1] < -0.4,
      f"vy={f.vel()[1]:+.2f} m/s, y={f.pos()[1]:+.2f} m")

f = Flight()
f.press(K_W, K_W)                          # +1.0 m/s forward
f.run(2.0)
f.press(K_S, K_S, K_S)                     # -1.0 + 0.5*... -> -0.5 m/s: trims ADD,
                                           # so returning to zero takes as many presses
f.run(3.0)
check("S: trims subtract, driving the command negative", f.vel()[0] < -0.4,
      f"vx={f.vel()[0]:+.2f} m/s (commanded -0.50)")
f.press(K_SPACE)
f.run(2.0)
check("commands are additive from either side (SPACE still stops it)",
      float(np.linalg.norm(f.vel())) < 0.10, f"|v|={float(np.linalg.norm(f.vel())):.3f} m/s")

print()
print("=" * 78)
print("C. ANGLE mode: the commanded tilt is held (and it does not self-brake)")
print("=" * 78)
f = Flight()
f.press(K_I, K_I, K_I, K_I)                # +20 deg nose down
f.run(1.5)
check("I x4: pitch settles at +20 deg",
      abs(f.euler_deg()[1] - 20.0) < 4.0, f"pitch={f.euler_deg()[1]:+.2f} deg")
f.run(1.5)
check("angle is HELD, not trimmed back towards level",
      abs(f.euler_deg()[1] - 20.0) < 5.0, f"pitch={f.euler_deg()[1]:+.2f} deg after 3 s")
check("tilted flight accelerates without braking (speed > 3 m/s)",
      f.vel()[0] > 3.0, f"vx={f.vel()[0]:+.2f} m/s")
# The attitude hold has no speed limit, so by 3 s this vehicle is doing ~9 m/s and the
# fluid drag on the pitched airframe is a real (measured) vertical force; the integral
# term in the altitude loop is what keeps the vehicle up. A few tens of cm of transient
# droop is the honest expectation, and it must NOT be a runaway sink.
check("altitude hold compensates the tilt (no runaway sink)",
      abs(f.pos()[2] - SPAWN_POS[2]) < 0.30 and f.vel()[2] > -0.6,
      f"z={f.pos()[2]:.3f} m, vz={f.vel()[2]:+.2f} m/s")

f = Flight()
f.press(K_J, K_J, K_J, K_J)                # roll left = -20 deg
f.run(3.0)
check("J x4: rolls to -20 deg",
      abs(f.euler_deg()[0] + 20.0) < 4.0, f"roll={f.euler_deg()[0]:+.2f} deg")
check("roll left flies left (+y)", f.vel()[1] > 2.0, f"vy={f.vel()[1]:+.2f} m/s")

f = Flight()
f.press(K_L, K_L, K_L, K_L)                # roll right = +20 deg
f.run(3.0)
check("L x4: rolls to +20 deg and flies right",
      abs(f.euler_deg()[0] - 20.0) < 4.0 and f.vel()[1] < -2.0,
      f"roll={f.euler_deg()[0]:+.2f} deg, vy={f.vel()[1]:+.2f} m/s")

f = Flight()
f.press(K_Q, K_Q)                          # yaw +30 deg
f.run(2.5)
check("Q x2: yaw tracks +30 deg", abs(f.euler_deg()[2] - 30.0) < 4.0,
      f"yaw={f.euler_deg()[2]:+.2f} deg")
check("yaw hold leaves the vehicle stationary", float(np.linalg.norm(f.vel())) < 0.15,
      f"|v|={float(np.linalg.norm(f.vel())):.3f} m/s")

f = Flight()
f.press(K_W, K_W)                          # 1.0 m/s forward
f.press(K_K, K_K)                          # then -10 deg pitch (angle mode)
f.run(2.0)
check("K: pitch trim goes nose-up (-10 deg)", abs(f.euler_deg()[1] + 10.0) < 4.0,
      f"pitch={f.euler_deg()[1]:+.2f} deg")

f = Flight()
f.press(K_I, K_I, K_I, K_I)
f.run(1.0)
f.press(K_SPACE)                           # recovery: velocity loop + level
f.run(2.5)
check("SPACE recovers from ANGLE mode (level, then held)",
      abs(f.euler_deg()[1]) < 4.0 and abs(f.vel()[0]) < 0.15,
      f"pitch={f.euler_deg()[1]:+.2f} deg, vx={f.vel()[0]:+.3f} m/s")

print()
print("=" * 78)
print("D. altitude channel (in both modes)")
print("=" * 78)
f = Flight()
f.press(K_R, K_R)                          # +0.5 m -> 1.7 m
f.run(4.0)
check("R x2: climbs to 1.70 m", abs(f.pos()[2] - 1.70) < 0.08, f"z={f.pos()[2]:.3f} m")
f.press(K_F, K_F, K_F, K_F)                # -1.0 m -> 0.7 m
f.run(4.0)
check("F x4: descends to 0.70 m", abs(f.pos()[2] - 0.70) < 0.10, f"z={f.pos()[2]:.3f} m")

f = Flight()
f.press(K_R, K_R)
f.press(K_I, K_I, K_I, K_I)                # climb AND tilt (angle mode: thrust compensation)
f.run(3.0)
check("altitude hold still works while pitched 20 deg in ANGLE mode",
      abs(f.pos()[2] - 1.70) < 0.25, f"z={f.pos()[2]:.3f} m (target 1.70)")

print()
print("=" * 78)
print("E. the two stop conditions the flight loop watches")
print("=" * 78)
f = Flight()
f.quad.data.qpos[2] = 0.02                 # simulate having sunk to the floor
import mujoco  # noqa: E402
mujoco.mj_forward(f.quad.model, f.quad.data)
f.quad._update_state_properties()
check("ground contact is detected (stop #1)", bool(f.quad.check_ground_contact()))
f = Flight()
f.quad.data.qpos[0] = float("nan")
mujoco.mj_forward(f.quad.model, f.quad.data)
f.quad._update_state_properties()
check("non-finite state is detectable (stop #2)", not bool(np.all(np.isfinite(f.quad.state))))
f = Flight().run(2.0)
check("a healthy flight reports finite state and no ground contact",
      bool(np.all(np.isfinite(f.quad.state))) and not bool(f.quad.check_ground_contact()))

print()
print("=" * 78)
print("F. gamepad mapping (synthetic pad states - the mapping is what runs in flight)")
print("=" * 78)
from gamepad import HAT_DOWN, HAT_LEFT, HAT_RIGHT, HAT_UP, PadState  # noqa: E402


def pad_state(**kwargs) -> PadState:
    """A connected pad at rest, with only the fields a test cares about overridden."""
    defaults = dict(connected=True, name="test pad", analog_triggers=0)
    defaults.update(kwargs)
    return PadState(**defaults)


cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
# Left stick pushed forward is NEGATIVE in raw HID sign, which must fly the drone forward.
cmd.apply_gamepad(pad_state(left_y=-0.5), DT)
expected = mf._shape_stick(0.5) * mf.V_STICK_MAX      # deadzone + expo applied
check("left stick forward -> velocity command forward",
      abs(cmd.v_cmd[0] - expected) < 1e-9 and cmd.mode == MODE_VELOCITY,
      f"v_cmd={np.round(cmd.v_cmd, 2)} (expect {expected:.2f} fwd)")
check("stick shaping is applied (expo makes half-stick less than half command)",
      cmd.v_cmd[0] < 0.5 * mf.V_STICK_MAX, f"v_fwd={cmd.v_cmd[0]:.2f} of "
      f"{0.5 * mf.V_STICK_MAX:.2f} for a linear stick")
cmd.apply_gamepad(pad_state(left_x=+1.0), DT)
check("left stick right -> velocity command to the RIGHT (negative 'left' component)",
      cmd.v_cmd[1] < -0.5, f"v_cmd={np.round(cmd.v_cmd, 2)}")
cmd.apply_gamepad(pad_state(), DT)
check("releasing the left stick clears the velocity command",
      np.allclose(cmd.v_cmd, 0.0), f"v_cmd={np.round(cmd.v_cmd, 2)}")

cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
cmd.apply_gamepad(pad_state(right_x=+1.0), DT)
check("right stick right -> ANGLE mode with positive roll",
      cmd.mode == MODE_ANGLE and abs(math.degrees(cmd.tilt[1]) - mf.TILT_STICK_MAX_DEG) < 1.0,
      f"mode={cmd.mode} roll={math.degrees(cmd.tilt[1]):+.1f} deg")
cmd.apply_gamepad(pad_state(right_y=-1.0), DT)
check("right stick forward -> nose-down pitch (same ANGLE mode)",
      cmd.mode == MODE_ANGLE and abs(math.degrees(cmd.tilt[0]) - mf.TILT_STICK_MAX_DEG) < 1.0,
      f"pitch={math.degrees(cmd.tilt[0]):+.1f} deg")
cmd.apply_gamepad(pad_state(right_y=-1.0, left_y=-1.0), DT)
check("both sticks at once: attitude holds AND the velocity command is kept",
      cmd.mode == MODE_ANGLE and cmd.v_cmd[0] > 0.5,
      f"mode={cmd.mode} v_fwd={cmd.v_cmd[0]:.2f}")
cmd.apply_gamepad(pad_state(), DT)
check("releasing the attitude stick returns to VELOCITY mode, level",
      cmd.mode == MODE_VELOCITY and np.allclose(cmd.tilt, 0.0), f"mode={cmd.mode}")

cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
for _ in range(20):
    cmd.apply_gamepad(pad_state(buttons={mf.PAD_BTN_CLIMB}, analog_triggers=0), DT)
check("trigger-less pad: button 8 climbs (altitude setpoint rises)",
      cmd.z_des > 1.2 + 0.1, f"z_des={cmd.z_des:.2f} m")
for _ in range(40):
    cmd.apply_gamepad(pad_state(buttons={mf.PAD_BTN_DESCEND}, analog_triggers=0), DT)
check("trigger-less pad: button 7 descends", cmd.z_des < 1.2, f"z_des={cmd.z_des:.2f} m")

cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
for _ in range(20):
    cmd.apply_gamepad(pad_state(analog_triggers=2, right_trigger=1.0), DT)
check("analog trigger climbs on a pad that has them", cmd.z_des > 1.2 + 0.1,
      f"z_des={cmd.z_des:.2f} m")
z_before = cmd.z_des
cmd.apply_gamepad(pad_state(analog_triggers=2, buttons={mf.PAD_BTN_CLIMB}), DT)
check("buttons 7/8 are ignored when the pad has real analog triggers",
      abs(cmd.z_des - z_before) < 1e-12, f"z_des={cmd.z_des:.4f}")

cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
cmd.apply_gamepad(pad_state(buttons={mf.PAD_BTN_YAW_LEFT}), DT)
check("shoulder button yaws left", cmd.yaw_des > 0.0, f"yaw={math.degrees(cmd.yaw_des):+.2f} deg")
before = cmd.yaw_des
cmd.apply_gamepad(pad_state(buttons={mf.PAD_BTN_YAW_RIGHT}), DT)
check("the other shoulder yaws right (through the same setpoint)",
      cmd.yaw_des < before, f"yaw={math.degrees(cmd.yaw_des):+.2f} deg")

cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
cmd.apply_gamepad(pad_state(pressed=(HAT_UP,)), DT)
check("D-pad up steps the altitude setpoint", abs(cmd.z_des - 1.45) < 1e-9,
      f"z_des={cmd.z_des:.2f} m")
cmd.apply_gamepad(pad_state(), DT)
check("a D-pad held across polls steps ONCE (the pad layer reports edges; that the edge"
      " is not repeated while held is covered by check_gamepad.py)",
      abs(cmd.z_des - 1.45) < 1e-9, f"z_des={cmd.z_des:.2f} m")
cmd.apply_gamepad(pad_state(pressed=(HAT_DOWN, HAT_LEFT)), DT)
check("D-pad down + left steps altitude back down and yaws left",
      abs(cmd.z_des - 1.20) < 1e-9 and cmd.yaw_des > 0.0,
      f"z_des={cmd.z_des:.2f} m yaw={math.degrees(cmd.yaw_des):+.0f} deg")

cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
cmd.apply_gamepad(pad_state(right_x=1.0, left_y=-1.0), DT)
actions = cmd.apply_gamepad(pad_state(pressed=(mf.PAD_BTN_RESPAWN,)), DT)
check("the respawn button is reported to the caller as a one-shot action",
      actions == ["respawn"], f"actions={actions}")
cmd.apply_gamepad(pad_state(right_x=1.0, left_y=-1.0), DT)
actions = cmd.apply_gamepad(pad_state(pressed=(mf.PAD_BTN_HOVER[0],)), DT)
check("a face button is reported as hover", actions == ["hover"], f"actions={actions}")
check("hover clears the sticks' effect (level, stop, altitude kept)",
      cmd.mode == MODE_VELOCITY and np.allclose(cmd.v_cmd, 0.0) and np.allclose(cmd.tilt, 0.0),
      f"mode={cmd.mode} v_cmd={np.round(cmd.v_cmd, 2)} tilt={np.round(cmd.tilt, 3)}")

cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
cmd.apply_gamepad(pad_state(left_y=-0.05), DT)      # inside the deadzone
check("stick noise inside the deadzone is ignored", np.allclose(cmd.v_cmd, 0.0))
check("...so the keyboard still owns the channel",
      cmd.handle_key(K_W) is not None and cmd.v_cmd[0] > 0.0, f"v_cmd={np.round(cmd.v_cmd, 2)}")
cmd.apply_gamepad(pad_state(), DT)
check("a centred pad does not wipe a keyboard command",
      abs(cmd.v_cmd[0] - 0.5) < 1e-12, f"v_cmd={np.round(cmd.v_cmd, 2)}")

cmd = ManualCommand(spawn_z=1.2)
cmd.set_mode(MODE_VELOCITY)
cmd.apply_gamepad(PadState(connected=False), DT)
check("a disconnected pad changes nothing", np.allclose(cmd.v_cmd, 0.0) and cmd.mode == MODE_VELOCITY)

print()
print("=" * 78)
print("G. ACRO mode: a real quad (collective + rates) with the Outer-Wilds mapping")
print("=" * 78)
f = Flight(mode=MODE_ACRO)
check("a flight pinned to ACRO flies in ACRO (not the GAME default)",
      f.cmd.mode == MODE_ACRO and START_MODE == MODE_GAME,
      f"START_MODE={START_MODE}, pinned mode={f.cmd.mode}")

f = Flight(mode=MODE_ACRO).run(4.0)
check("throttle trimmed to hover holds altitude with no input",
      abs(f.pos()[2] - SPAWN_POS[2]) < 0.06, f"z={f.pos()[2]:.3f} m")
check("and it does not rotate on its own (rate loop holds zero rate)",
      float(np.max(np.abs(f.euler_deg()))) < 1.0, f"max angle={float(np.max(np.abs(f.euler_deg()))):.2f} deg")

# One key press spends a rotation budget: the pitch must change by a real amount, and the
# budget must run out by itself (a press is a pulse, not a latching trim).
f = Flight(mode=MODE_ACRO)
f.cmd.handle_key(K_W)
check("a rotation key press arms a budget", abs(f.cmd._rot_budget[1]) > 0.1,
      f"budget={math.degrees(f.cmd._rot_budget[1]):.0f} deg")
f.run(1.0)
check("the budget is spent and the rotation stops",
      abs(f.cmd._rot_budget[1]) < 1e-6 and f.euler_deg()[1] > 25.0,
      f"pitch={f.euler_deg()[1]:+.1f} deg, budget left={math.degrees(f.cmd._rot_budget[1]):.1f} deg")

# Flying: settle in a hover with room underneath, pitch the nose down, then let go. A
# real quad keeps the attitude (no auto-levelling) and keeps accelerating (no auto-brake)
# - and, because the thrust is along body z, it also starts to sink.
f = Flight(mode=MODE_ACRO).hover_at(8.0)
check("hover holds altitude up high too (no altitude loop, just collective)",
      abs(f.pos()[2] - 8.0) < 0.15, f"z={f.pos()[2]:.2f} m")
z_high = f.pos()[2]
f.cmd.push_rotation(1, +1.0, 25.0)           # a 25 deg nose-down pulse
f.run(0.5)
pitch_set = f.euler_deg()[1]
v_after_pulse = f.vel()[0]
f.run(0.6)                                   # hands off
check("the drone keeps the attitude the pulse left it at (no auto-levelling)",
      abs(f.euler_deg()[1] - pitch_set) < 3.0,
      f"pitch {pitch_set:+.1f} -> {f.euler_deg()[1]:+.1f} deg")
check("and keeps accelerating (no auto-brake, unlike the assisted modes)",
      f.vel()[0] > v_after_pulse + 0.5,
      f"vx {v_after_pulse:+.2f} -> {f.vel()[0]:+.2f} m/s at {f.euler_deg()[1]:+.0f} deg")
check("it also sinks while tilted, as a real quad does (thrust is along body z)",
      f.pos()[2] < z_high - 0.2, f"z {z_high:.2f} -> {f.pos()[2]:.2f} m")

# Counter-pitch reverses the acceleration.
speed_in = float(np.linalg.norm(f.vel()))
f.cmd.push_rotation(1, -1.0, 50.0)
f.run(1.2)
check("counter-pitch bleeds the speed off (the game's counter-thrust)",
      float(np.linalg.norm(f.vel())) < speed_in,
      f"|v| {speed_in:.2f} -> {float(np.linalg.norm(f.vel())):.2f} m/s")

# Match Velocity: the one assist, and the game's brake.
f = Flight(mode=MODE_ACRO).hover_at(8.0)
f.cmd.push_rotation(1, +1.0, 20.0)
f.run(0.8)                                   # fly forward a little
z_engage = f.pos()[2]
f.cmd.toggle_match_velocity()
f.run(3.0)
check("MATCH VELOCITY brakes to a hover",
      float(np.linalg.norm(f.vel())) < 0.15, f"|v|={float(np.linalg.norm(f.vel())):.3f} m/s")
check("...and holds the altitude and heading it was engaged at",
      abs(f.pos()[2] - z_engage) < 0.5 and abs(f.euler_deg()[0]) < 10.0 and abs(f.euler_deg()[1]) < 12.0,
      f"z {z_engage:.2f} -> {f.pos()[2]:.2f} m, roll={f.euler_deg()[0]:+.1f} pitch={f.euler_deg()[1]:+.1f} deg")
f.cmd.handle_key(K_W)
check("touching a rotation key hands control back to the pilot",
      not f.cmd.match_velocity, f"match_velocity={f.cmd.match_velocity}")
f.cmd._rot_budget[:] = 0.0

# Throttle: a latched collective, stepped by the keyboard like a real throttle stick.
f = Flight(mode=MODE_ACRO)
throttle0 = f.cmd.throttle
f.cmd.handle_key(K_SPACE)
check("SPACE adds collective", abs(f.cmd.throttle - throttle0 - mf.ACRO_THROTTLE_STEP) < 1e-9,
      f"{100 * throttle0:.0f}% -> {100 * f.cmd.throttle:.0f}%")
f.run(2.0)
check("more collective climbs", f.pos()[2] > SPAWN_POS[2] + 0.3, f"z={f.pos()[2]:.2f} m")
f.cmd.handle_key(K_F)
f.cmd.handle_key(K_F)
f.run(2.0)
check("less collective descends", f.vel()[2] < 0.0, f"vz={f.vel()[2]:+.2f} m/s")

# Aggressive manoeuvre: a fast roll inverts the drone while it is airborne. It STAYS
# inverted (no auto-level) and its thrust now points at the ground - real quad physics.
f = Flight(mode=MODE_ACRO).hover_at(12.0)
f.cmd.rot_pad[0] = math.pi / 0.5             # 360 deg/s of roll for 0.5 s
f.run(0.5)
f.cmd.rot_pad[0] = 0.0
f.run(0.05)
check("a fast roll inverts the drone (the aggressive part is fully available)",
      f.quad.dcm[2, 2] < -0.8, f"z_b.z={f.quad.dcm[2, 2]:+.2f} (inverted = -1)")
vz_in = f.vel()[2]
f.run(0.15)
vz_out = f.vel()[2]
a_inverted = (vz_out - vz_in) / 0.15
check("inverted thrust pushes it towards the ground (real quad physics, no cheats)",
      a_inverted < -11.0,
      f"a={a_inverted:+.1f} m/s^2 (free fall plus airframe drag alone is > -9.8)")
f.cmd.handle_key(K_C)
f.cmd.throttle = 0.95                        # the pilot's part of the recovery
f.run(1.4)
check("C (level) recovers the attitude before it lands",
      f.quad.dcm[2, 2] > 0.7 and f.pos()[2] > 0.5,
      f"z_b.z={f.quad.dcm[2, 2]:+.2f}, z={f.pos()[2]:.2f} m")
check("...and the collective is what stops the fall (level does not fly it for you)",
      f.vel()[2] > 0.5, f"vz={f.vel()[2]:+.2f} m/s, z={f.pos()[2]:.2f} m")

# Mode switching must clear the set being left. M CYCLES: GAME -> ACRO -> ASSISTED -> GAME.
cmd = ManualCommand(spawn_z=1.2)
check("the default command starts in GAME", cmd.mode == MODE_GAME)
cmd.handle_key(K_W)                          # a GAME movement step
cmd.toggle_match_velocity()
msg_game_to_acro = cmd.toggle_assist()
check("M switches GAME -> ACRO and clears the movement channels",
      cmd.mode == MODE_ACRO and np.allclose(cmd.v_cmd, 0.0) and not cmd.match_velocity,
      f"mode={cmd.mode} v_cmd={cmd.v_cmd}")
cmd.rot_pad[1] = 1.0
cmd._rot_budget[0] = 0.5
msg_acro_to_assisted = cmd.toggle_assist()
check("M switches ACRO -> ASSISTED and clears the acro channels",
      cmd.mode == MODE_VELOCITY and np.allclose(cmd.rot_pad, 0.0)
      and np.allclose(cmd._rot_budget, 0.0), f"mode={cmd.mode} rot_pad={cmd.rot_pad}")
cmd.handle_key(K_W)                          # assisted trim
msg_assisted_to_game = cmd.toggle_assist()
check("M switches ASSISTED -> GAME and clears the assisted channels",
      cmd.mode == MODE_GAME and np.allclose(cmd.v_cmd, 0.0) and np.allclose(cmd.tilt, 0.0),
      f"mode={cmd.mode} v_cmd={cmd.v_cmd}")
check("every mode switch says what it changed",
      all(word in msg for msg, word in ((msg_game_to_acro, "ACRO"),
                                        (msg_acro_to_assisted, "ASSISTED"),
                                        (msg_assisted_to_game, "GAME"))),
      msg_assisted_to_game.strip())

print()
print("=" * 78)
print("H. GAME mode: video-game sticks, the flip button, the heading-locked camera")
print("=" * 78)
check("GAME is the start mode", mf.START_MODE == mf.MODE_GAME, mf.START_MODE)

f = Flight(mode=mf.MODE_GAME)
check("the start mode is GAME on a fresh command object", f.cmd.mode == mf.MODE_GAME, f.cmd.mode)

# left stick = movement (raw sign: forward/left/up are NEGATIVE, right is POSITIVE)
f.cmd.apply_gamepad(pad_state(left_y=-0.6), DT)
check("left stick forward commands forward velocity",
      f.cmd.v_cmd[0] > 1.0 and abs(f.cmd.v_cmd[1]) < 1e-9 and f.cmd.mode == mf.MODE_GAME,
      f"v_cmd {np.round(f.cmd.v_cmd, 2)}")
f.cmd.apply_gamepad(pad_state(), DT)
f.cmd.apply_gamepad(pad_state(left_x=+0.6), DT)
check("left stick right strafes right (v_cmd[1] is LEFT, so it goes negative)",
      f.cmd.v_cmd[1] < -1.0 and abs(f.cmd.v_cmd[0]) < 1e-9, f"v_cmd {np.round(f.cmd.v_cmd, 2)}")
f.cmd.apply_gamepad(pad_state(), DT)
check("releasing the stick stops the movement command",
      float(np.linalg.norm(f.cmd.v_cmd)) < 1e-9, f"v_cmd {np.round(f.cmd.v_cmd, 2)}")

# right stick X = heading
f.cmd.apply_gamepad(pad_state(right_x=+1.0), DT)
step = math.degrees(f.cmd.yaw_des)
f.cmd.apply_gamepad(pad_state(right_x=+1.0), DT)
check("right stick right turns the heading right, while held",
      step < 0.0 and abs(step + mf.GAME_HEADING_RATE_DEG * DT) < 0.01,
      f"{step:+.2f} deg in one {DT * 1000:.0f} ms step (rate {mf.GAME_HEADING_RATE_DEG:.0f} deg/s)")
f.cmd.apply_gamepad(pad_state(), DT)
held = f.cmd.yaw_des
f.cmd.apply_gamepad(pad_state(), DT)
check("...and holds that heading when the stick is released",
      abs(f.cmd.yaw_des - held) < 1e-12, f"heading {math.degrees(f.cmd.yaw_des):+.1f} deg")

# ZL / ZR = altitude
z0 = f.cmd.z_des
for _ in range(10):
    f.cmd.apply_gamepad(pad_state(buttons=(mf.PAD_BTN_CLIMB,)), DT)
z_up = f.cmd.z_des
for _ in range(10):
    f.cmd.apply_gamepad(pad_state(buttons=(mf.PAD_BTN_DESCEND,)), DT)
check("ZR (right trigger button) climbs and ZL descends",
      z_up > z0 + 0.05 and f.cmd.z_des < z_up - 0.05,
      f"z_des {z0:.2f} -> {z_up:.2f} -> {f.cmd.z_des:.2f} m")

# the flip button: thrust is FLOWN (pop -> ballistic -> arrest), left stick rolls/pitches
f.cmd.apply_gamepad(pad_state(buttons=(mf.PAD_BTN_FLIP,), left_x=+1.0), DT)
thrust, omega = f.ctrl.update()
hover_n = f.ctrl.hover_throttle * f.ctrl.max_thrust
check("FLIP pressed: the collective pops to ~2x hover and the rotation is gated so the "
      "pop goes UP, not sideways",
      f.cmd.game_flip and abs(thrust - mf.GAME_FLIP_POP_X * hover_n) < 1e-9
      and float(np.linalg.norm(omega)) < 1e-9 and float(np.linalg.norm(f.cmd.v_cmd)) < 1e-9,
      f"thrust {thrust:.4f} N ({thrust / hover_n:.1f}x hover), omega {np.round(omega, 3)} "
      f"during the {mf.GAME_FLIP_POP_S * 1000:.0f} ms pop, v_cmd zeroed")
f.ctrl._flip_t = mf.GAME_FLIP_POP_S + 0.05      # past the pop: the roll is live now
_, omega_right = f.ctrl.update()
check("...then FLIP + left stick RIGHT rolls right at the full rate",
      omega_right[0] > 0.9 * math.radians(mf.GAME_FLIP_RATE_DEG),
      f"roll {omega_right[0]:+.1f} rad/s")
f.cmd.apply_gamepad(pad_state(buttons=(mf.PAD_BTN_FLIP,), left_x=-1.0), DT)
f.ctrl._flip_t = mf.GAME_FLIP_POP_S + 0.05      # past the pop: the roll is live now
_, omega_left = f.ctrl.update()
check("FLIP + left stick LEFT rolls left (after the pop)",
      omega_left[0] < -0.9 * math.radians(mf.GAME_FLIP_RATE_DEG), f"roll {omega_left[0]:+.1f} rad/s")
f.cmd.apply_gamepad(pad_state(buttons=(mf.PAD_BTN_FLIP,), left_y=-1.0), DT)
f.ctrl._flip_t = mf.GAME_FLIP_POP_S + 0.05
_, omega_pitch = f.ctrl.update()
check("FLIP + left stick forward pitches the nose down (after the pop)",
      omega_pitch[1] > 0.9 * math.radians(mf.GAME_FLIP_RATE_DEG), f"pitch {omega_pitch[1]:+.1f} rad/s")
check("while inverted the flip profile cuts the thrust (thrust along a rolled body z "
      "would push sideways)",
      mf.game_flip_thrust(0.6, -0.9, -1.0, hover_n) == 0.0
      and mf.game_flip_thrust(0.6, 0.9, -1.0, hover_n) > 1.5 * hover_n,
      "inverted -> 0 N, upright and sinking -> 1.9x hover")
f.cmd.apply_gamepad(pad_state(), DT)
# one more step so the law switch is applied (this is where the heading is adopted)
f.ctrl.update()
check("releasing the button returns to the movement law",
      (not f.cmd.game_flip) and f.ctrl._law == mf.MODE_GAME, f"law={f.ctrl._law}")

# keyboard: V toggles the flip, WASD/QE/RF drive movement/heading/altitude
f.cmd.handle_key(mf.K_W)
check("GAME keyboard: W = forward velocity step", f.cmd.v_cmd[0] > 0, f"v_cmd {np.round(f.cmd.v_cmd, 2)}")
msg = f.cmd.handle_key(mf.K_V)
f.cmd.handle_key(mf.K_D)                      # a rotation key while the flip is on
check("GAME keyboard: V locks the flip mode and then D becomes a roll-right pulse",
      f.cmd.game_flip and f.cmd._rot_budget[0] > 0 and "flip ON" in (msg or ""),
      f"budget {np.degrees(f.cmd._rot_budget[0]):.0f} deg roll")
z_locked = f.cmd.z_des
f.cmd.handle_key(mf.K_SPACE)
check("...and the thrust keys are inert while it is locked",
      f.cmd.z_des == z_locked, f"z_des stayed at {f.cmd.z_des:.2f} m")
f.cmd.handle_key(mf.K_V)
check("V again releases the flip mode", not f.cmd.game_flip)

# flying it: a forward command should move the vehicle and hold the altitude
f = Flight(mode=mf.MODE_GAME)
z_start = f.pos()[2]
for _ in range(200):
    f.cmd.apply_gamepad(pad_state(left_y=-0.8), DT)
    f.run(DT)
moved = float(np.linalg.norm(f.pos()[:2]))
check("GAME flight: a held forward stick flies it forward at a held altitude",
      moved > 0.4 and abs(f.pos()[2] - z_start) < 0.35,
      f"moved {moved:.2f} m horizontally, altitude {z_start:.2f} -> {f.pos()[2]:.2f} m")

# and the flip button actually does a flip in the air, without hitting the floor
f = Flight(mode=mf.MODE_GAME)
f.cmd.game_flip = True
spin = 0.0
zmin = 9.9
for _ in range(80):                            # pop (0.25 s) + one full turn (0.5 s)
    f.cmd.rot_pad[0] = math.radians(mf.GAME_FLIP_RATE_DEG)
    f.run(DT)
    spin += abs(float(f.quad.omega[0])) * DT
    zmin = min(zmin, float(f.pos()[2]))
f.cmd.game_flip = False
f.cmd.rot_pad[0] = 0.0
f.run(2.5)
check("GAME flip button: a full turn from 1.2 m never reaches the floor",
      math.degrees(spin) > 300.0 and zmin > 0.8 and f.quad.dcm[2, 2] > 0.9,
      f"rotated {math.degrees(spin):.0f} deg, min altitude {zmin:.2f} m, "
      f"ended level (z_b.z={f.quad.dcm[2, 2]:+.2f}) at z={f.pos()[2]:.2f} m")

# camera: azimuth IS the look direction (measured against mjv_updateScene), so the lock
# is provable rather than a guess.
import mujoco  # noqa: E402

class _FakeViewer:
    def __init__(self):
        self.cam = mujoco.MjvCamera()

for yaw in (0.0, 0.5, -1.2):
    viewer = _FakeViewer()
    f.quad.reset(pos=np.zeros(3), quat=mf.QuadFlipEnv._euler_to_quat(0.0, 0.0, yaw),
                 thrust_scale=1.0)
    mf._update_camera(viewer, f.quad)
    azimuth_ok = abs(viewer.cam.azimuth - math.degrees(yaw)) < 1e-9
    scene = mujoco.MjvScene(f.quad.model, maxgeom=64)
    mujoco.mjv_updateScene(f.quad.model, f.quad.data, mujoco.MjvOption(), None,
                           viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    fwd = np.asarray(scene.camera[0].forward)
    nose = np.asarray(f.quad.dcm[:, 0])          # body x axis = the nose, in world
    align = float(np.degrees(math.acos(np.clip(
        float(np.dot(fwd[:2], nose[:2]) / max(1e-9, np.linalg.norm(fwd[:2]) * np.linalg.norm(nose[:2]))),
        -1.0, 1.0))))
    check(f"camera locked to the heading (yaw {math.degrees(yaw):+.0f} deg): looks along the nose",
          azimuth_ok and align < 1.0,
          f"azimuth {viewer.cam.azimuth:+.1f} deg, camera forward is {align:.2f} deg off the nose")

print()
print("=" * 78)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
