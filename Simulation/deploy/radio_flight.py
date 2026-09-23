#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fly the real Crazyflie over the Crazyradio PA, with the trained policy on a switch.

    .venv/bin/python Simulation/deploy/radio_flight.py --uri radio://0/80/2M/E7E7E7E7E7
    # really flying it: --arm-ok --hover 50 --set policy.shadow=0

This is the real-vehicle twin of `Simulation/live_flight.py`: the same workflow, the same
kind of GUI, but the plant is the drone and the handover goes through the app's appchannel
instead of a Python mode switch.

    STOCK  (do nothing)   the app is DISARMED: your sticks are streamed over CRTP and the
                          stock rate PID flies it, exactly as if the app were not there
    HOVER  (policy)       appchannel 0x01 ARM: the policy takes the hover at the moment you
                          press it and holds it. Press again (or KILL) to get it back
    FLIP   (policy)       appchannel 0x03: the policy flies the baked flip from the hover
                          it is holding; it returns to hover on its own
    MANOEUVRE (policy)    appchannel 0x05 <kind>: the same handover for ANY baked table -
                          flip / orbit / figure8 / lissajous / slalom / waypoints. 0xFF
                          stops and holds wherever the vehicle is. Each table is relocated
                          onto the pose and heading the drone has AT LAUNCH, and hands back
                          to a hold anchored where it ended (see gen_references.py). The
                          kind list is read from the generated header, so the GUI cannot
                          offer a manoeuvre the firmware does not have.
    KILL                  appchannel 0x02 DISARM + a motors-off setpoint. Always available,
                          on the GUI button, the pad, and the keyboard

TAKE-OFF IS YOURS, AND THAT IS DELIBERATE. Arming syncs the hover reference to wherever
the vehicle is, so arming on the ground means holding the ground (and the app's own z
envelope will refuse it). Fly it up in STOCK, settle, then hand over.

WHAT THE HOST MUST SEND (verified against the firmware, not assumed)
-------------------------------------------------------------------
`cflib`'s `send_setpoint_manual(roll, pitch, yawrate, thrust_pct, rate)` is the modern
generic-commander setpoint; the firmware's `manualDecoder` maps it to the stabilizer as
    roll/pitch -> setpoint->attitude.*      (modeAbs,   degrees)   when rate=False
               -> setpoint->attitudeRate.*  (modeVelocity, deg/s)   when rate=True
    yawrate    -> setpoint->attitudeRate.yaw  (deg/s, always rate)
    thrust     -> setpoint->thrust            (raw units, see below)
Two details that matter and are easy to get wrong:
  * cflib PACKS `-pitch`. Combined with the app's own `SIGN_RATE_PITCH = -1` derivation
    (the legacy loop tracks `-gyro.y`), that makes `pitch=+P` a NOSE-DOWN command. So
    "stick forward = go forward" is `pitch = +P`, which is what this script sends.
  * `send_setpoint_manual(..., thrust_pct=0)` is NOT off: it packs raw thrust 10001, above
    the firmware's MIN_THRUST of 1000, so the motors idle. The only real "off" is
    `send_stop_setpoint()`, which this script uses for KILL, for idle and on exit.

THRUST IS A HOVER-ANCHORED STICK, because a gamepad's stick self-centres. The centre
position IS the hover command (`--hover`, percent), so letting go holds altitude instead of
falling out of the sky. The sim's hover is 45.8 % of maximum FORCE, which is not the same
number as this command - calibrate `--hover` on the bench (and read it off the policy
itself: while the policy hovers, `policy.thrust_units / 60000 * 100` is the answer).

KEYBOARD (no pad needed)
------------------------
This tool has no 3D window, so unlike `manual_flight.py` / `live_flight.py` there is no
MuJoCo `key_callback` to hang bindings on - the keys come straight from the terminal
(`TerminalKeys`, cbreak mode) and work with the pad connected, absent, or ignored.

    w / up-arrow      climb      +--key-thrust-step (default 2 %) on the thrust CENTRE
    s / down-arrow    descend    -step
    space             centre     back to --hover
    x                 KILL       motors off + disarm (same as the pad/GUI button)
    q / ESC           quit
    Ctrl-C            kill + quit (unchanged - cbreak leaves ISIG alone, so the signal
                                 still arrives; raw mode would have swallowed it)

Climb/descend TRIM THE CENTRE rather than acting as momentary switches, for the same reason
the viewer bindings are latched: a terminal cannot report key RELEASE either, so "hold to
climb" would latch on or depend on the OS auto-repeat rate. One press is one step, holding
the key auto-repeats into a ramp, and the current centre is printed on every change. With no
pad connected the stick contribution is zero, so the centre IS the commanded thrust - which
is exactly the take-off control you want: raise it until the vehicle lifts, and that number
is your real `--hover`.

SAFETY, HONESTLY
----------------
There is NO firmware-side deadman: if this script dies mid-air the vehicle keeps the last
setpoint. The script therefore (a) slew-limits thrust, (b) sends stop+disarm on every exit
path including Ctrl-C, (c) has a KILL on the pad and the GUI, and (d) refuses to arm unless
`--arm-ok` is given. Props off for the first run of every new mapping.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flight_gui import DEFAULT_PORT, GuiServer  # noqa: E402  (the same loopback transport)
import gamepad as gp  # noqa: E402
from pad_mapper import PadMapper  # noqa: E402
import drone_link  # noqa: E402

CMD_ARM = 0x01
CMD_DISARM = 0x02
CMD_FLIP = 0x03
CMD_STATUS = 0x04
CMD_PLAY = 0x05
KIND_NONE = 0xFF          # the app's "not playing" / stop-and-hold sentinel
STATUS_MAGIC = 0xA5

MODE_STOCK = "STOCK (your sticks)"
MODE_VELHOLD = "STOCK +VEL-HOLD (position hold)"
MODE_HOVER = "POLICY (hover)"

PAD_CALIBRATION_PATH = os.path.join(_PROJECT_ROOT, "logs", "pad_calibration.json")

# The firmware's OWN telemetry, which the app's status packet cannot show us.
#
# WHY THIS EXISTS: the status packet's `vbat` is NOT a battery reading.
# controller_app.c has `static float g_last_vbat = POLICY_VBAT_NOMINAL;` (= 3.7f) and only
# overwrites it `if (g_vbat_id_valid)`, i.e. when `logGetVarId("pm","vbat")` resolves. So a
# vbat that sits at exactly 3.70 forever means the lookup FAILED and the column is a
# placeholder - it must never be read as "the battery did not sag".
#
# `motor.m1..m4` are the MIXER'S OWN OUTPUT (`motor_ratios`, 0..65535, written by
# powerDistribution), i.e. the firmware's answer to "did the command reach the motors".
# `motor.m1_rpm..m4_rpm` would be better still, but they exist ONLY under
# `#ifdef CONFIG_MOTORS_ESC_PROTOCOL_DSHOT_BIDIRECTIONAL` and this image runs OneShot125
# (`CONFIG_MOTORS_ESC_PROTOCOL_ONESHOT125=y`), so they are absent and the startup line says so.
# ALSO: the status packet's `tilt` and `p_err` are DEAD while disarmed.
# `policy_safety_check()` opens with `if (!g_armed) { return; }` and `g_last_tilt_deg` is only
# assigned after that, and `g_last_p_err` is only set inside `build_frame()` - which needs the
# policy armed. Both therefore read exactly 0.0 for an entire STOCK flight and must not be
# read as "the vehicle is perfectly level". (`z` IS live: `g_last_state_z` is assigned at the top
# of `controllerOutOfTree`, before any armed check.) So log the firmware's own values instead.
#
# `motor.*_rpm` would be nice but they only exist under
# `#ifdef CONFIG_MOTORS_ESC_PROTOCOL_DSHOT_BIDIRECTIONAL`, and this image is OneShot125.
# Sizes: 8 x float + 4 x uint16 = 48 B, so this spans MULTIPLE 26 B blocks - start_fw_log packs
# greedily and asserts the budget rather than discovering it from an AttributeError.
FW_LOG_VARS = [
    "stateEstimate.x", "stateEstimate.y",             # horizontal position - the drift
    "stateEstimate.vx", "stateEstimate.vy",           # and its rate
    "stabilizer.roll", "stabilizer.pitch",            # the REAL attitude
    "pm.vbat",                                        # the firmware's own battery reading
    "motor.m1", "motor.m2", "motor.m3", "motor.m4",   # mixer output into the ESCs
    # The estimator's own health. A position that JUMPS (measured: y +0.82 m and z -1.10 m in
    # single samples, i.e. 8-13 m/s equivalent on a vehicle that was barely moving) is almost
    # always the deck losing base stations, and it corrupts p_err for the policy.
    "lighthouse.status", "lighthouse.bsReceive",
    # *** THE DECISIVE ONE: `comSync` is `uartSynchronized` in lighthouse_core.c. ***
    # The whole lighthouse task is a nested loop: `waitForUartSynchFrame()` (which requires
    # UART_FRAME_LENGTH = 12 consecutive 0xff bytes and has NO TIMEOUT) then an inner loop
    # `while (getUartFrameRaw(&frame))`. The inner loop exits on the FIRST invalid frame and
    # sets uartSynchronized = false. `deckHealthCheck()` and `updateSystemStatus()` are BOTH
    # called only from INSIDE that inner loop, so once it exits, `systemStatus` FREEZES at
    # statusNotReceiving (0) and the deck watchdog stops running. That is a latch: it explains
    # a reception loss that never recovers even with the motors off, and "restarting the drone
    # fixes it".
    # comSync therefore splits the two remaining hypotheses apart:
    #   comSync = 0, bsReceive = 0  -> the UART byte stream is broken/dead: firmware-side latch
    #                                 (the task is spinning in waitForUartSynchFrame)
    #   comSync = 1, bsReceive = 0  -> the deck IS talking and sending sync frames, but detects
    #                                 NO base-station pulses: optics / EMI / dark
    # `bsActive` is received AND calibrated (narrower than bsReceive); `bsCalUd` tells the
    # statusMissingData (orange slow blink) case from statusNotReceiving (orange solid).
    "lighthouse.comSync", "lighthouse.bsActive", "lighthouse.bsCalUd",
    # The PIPELINE RATES - the honest way to answer "are we overloading the CPU?".
    # `cycleRt`/`frameRt` are the deck SPI cycle and frame rates, and `preThRt`/`postThRt`
    # bracket `throttleLh2Samples()` in `lighthouse_throttle.c`. A starved Lighthouse task
    # shows up here as a SAGGING rate; a starved CPU would also show it as a LOW `bsReceive`
    # (that bit is set the instant a sweep arrives, before the throttle, so it is a pure
    # signal measure). `disProb` is the throttle's discard probability.
    # These matter because "loses the Lighthouse under power but not by hand" differs ONLY by
    # the motors running - i.e. vibration / ESC electrical noise / a sagging 1S rail.
    "lighthouse.cycleRt", "lighthouse.frameRt",
    "lighthouse.preThRt", "lighthouse.postThRt", "lighthouse.disProb",
]

# The baked manoeuvres, read from the SAME generated header the firmware was compiled with,
# so the GUI can never offer a kind the drone does not have.
REF_HEADER = os.path.join(_HERE, "app_policy_controller", "src", "generated",
                          "reference_tables.h")


def load_kinds() -> List[Tuple[int, str]]:
    """[(index, name)] from generated/reference_tables.h's REF_KIND_* enum."""
    try:
        with open(REF_HEADER, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return []
    found = re.findall(r"REF_KIND_([A-Z0-9]+)\s*=\s*(\d+)", text)
    return sorted(((int(i), n.lower()) for n, i in found), key=lambda kv: kv[0])

# Pad buttons (same names manual_flight publishes; see Simulation/gamepad.py)
BTN_KILL = 3          # X / square: kill at all times
BTN_HOVER = 4         # Y / triangle: hand over to the policy
BTN_STOCK = 2         # B / circle: take it back
BTN_FLIP = 6          # top-right shoulder: flip (only while the policy has it)


def commands_from_pad(roles: Dict[str, float], hover_pct: float, span_pct: float,
                      max_angle_deg: float, max_yaw_dps: float,
                      invert_roll: bool = False, invert_pitch: bool = False
                      ) -> Tuple[float, float, float, float]:
    """
    Sticks -> (roll_deg, pitch_deg, yawrate_dps, thrust_pct). Pure, so it is unit-testable
    without a radio or a pad (see the self-test at the bottom of this file).

    Sign derivation (see the module docstring): roll=+deg is roll RIGHT, pitch=+deg is
    NOSE-DOWN. `yawrate` is the ONE channel the firmware takes in the opposite sense - the
    estimator's yaw rate is positive for a LEFT turn (the same convention the app encodes
    as SIGN_RATE_YAW = +1) - so the pilot's "turn right" intent is negated here.
    """
    roll = (-1.0 if invert_roll else 1.0) * roles.get("roll", 0.0) * max_angle_deg
    pitch = (-1.0 if invert_pitch else 1.0) * roles.get("pitch", 0.0) * max_angle_deg
    yaw = -float(roles.get("yaw", 0.0)) * max_yaw_dps          # + = right turn
    thrust = hover_pct + float(roles.get("throttle", 0.0)) * span_pct
    return roll, pitch, yaw, thrust


class TerminalKeys:
    """
    Read single keypresses from the terminal, no Enter required.

    WHY NOT THE VIEWER: `manual_flight.py` / `live_flight.py` take their keys from MuJoCo's
    `key_callback`. This tool drives the REAL drone and has no 3D window, so there is no
    viewer to hook - the keyboard has to come from the terminal.

    cbreak, NOT raw. cbreak turns off line buffering and echo but leaves ISIG alone, so
    Ctrl-C still raises KeyboardInterrupt and the documented "Ctrl-C = kill + quit" panic
    path keeps working. Raw mode would swallow it as the byte 0x03 and silently remove the
    panic button. Terminal attributes are saved and restored on close for the same reason
    (a crashed script must not leave the shell without echo).
    """

    # macOS and Linux both send these for the arrow keys
    ARROWS = {"\x1b[A": "up", "\x1b[B": "down", "\x1b[C": "right", "\x1b[D": "left"}

    def __init__(self) -> None:
        self.enabled = False
        self._saved = None
        if not sys.stdin.isatty():
            return                      # piped / headless: nothing to read
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self.enabled = True
        except Exception:
            self.enabled = False

    def read(self) -> List[str]:
        """Non-blocking: every key pressed since the last call, as names."""
        if not self.enabled:
            return []
        import select
        out: List[str] = []
        while True:
            if not select.select([sys.stdin], [], [], 0)[0]:
                return out
            ch = sys.stdin.read(1)
            if not ch:
                return out
            if ch == "\x1b":
                # an escape sequence: gather whatever follows immediately. A bare ESC (no
                # continuation) becomes "esc"; the short wait only happens on ESC itself.
                seq = ch
                while select.select([sys.stdin], [], [], 0.01)[0]:
                    nxt = sys.stdin.read(1)
                    if not nxt:
                        break
                    seq += nxt
                out.append(self.ARROWS.get(seq, "esc"))
            else:
                out.append(ch)
        return out

    def close(self) -> None:
        if self._saved is None:
            return
        try:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
        except Exception:
            pass
        self._saved = None
        self.enabled = False


class RadioFlight:
    """One radio flight: link + pilot setpoints + appchannel handovers + panel."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.running = True
        self.messages: List[str] = []
        self._con_line: List[str] = []           # char-wise firmware console -> whole lines
        self.mode = MODE_STOCK
        self.armed = False
        self.hover_pct = float(args.hover)
        self.span_pct = float(args.thrust_span)
        self.max_pct = float(args.max_thrust)
        self.last_status: Dict[str, Any] = {}
        self.kinds = load_kinds()                   # [(index, name)] baked into the firmware
        self.flip_index = next((i for i, n in self.kinds if n == "flip"), 0)
        self.manoeuvre = KIND_NONE
        self._play_at = 0.0
        self.applied = (0.0, 0.0, 0.0, 0.0)      # what we last sent (for the HUD/log)
        self._thrust_sent = 0.0
        self._killed = False
        # The FIRMWARE's arming state - a different gate from our app's `armed`. Nothing
        # moves until this is true; see firmware_arm().
        self._motors_ok = False
        self.fw: Dict[str, float] = {}           # latest firmware log values (motor RPM etc.)
        self.fw_sup: Dict[str, Any] = {}         # latest supervisor flags
        self._fw_logs: List[Any] = []
        self._sup_t = 0.0
        # Use the stock firmware's velocity loop instead of attitude-only manual setpoints.
        self._vel_hold = bool(getattr(args, "vel_hold", False))
        self.mode = self._stock_label()
        self._lock = threading.Lock()
        self._gui: Optional[GuiServer] = None
        self._gui_proc: Optional[subprocess.Popen] = None

        # -- pad ------------------------------------------------------------------------
        self.mapper = PadMapper.load(PAD_CALIBRATION_PATH, layout="ow")
        self.pad = gp.HIDGamepad()
        if self.pad.available:
            self.pad.start()
            self.messages.append(f"pad: {getattr(self.pad, '_name', 'device')} reader started")
        else:
            self.messages.append(f"pad: not available ({self.pad.error}) - GUI only")
            self.pad = None
        self.pad_state = None
        self._prev_buttons: set = set()

        # -- link ------------------------------------------------------------------------
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

        # `crtp.CLASSES` is empty until this runs and `get_link_driver()` iterates it, so
        # without it every URI resolves to None ("No driver found or malformed URI").
        cflib.crtp.init_drivers()

        if not drone_link.dongle_preflight(args.uri):
            raise RuntimeError("Crazyradio is wedged at the USB level - replug the dongle "
                               "(NOT a URI problem: see the line above)")

        self.cf = Crazyflie(rw_cache=os.path.join(_PROJECT_ROOT, ".cf_cache"))
        self.cf.appchannel.packet_received.add_callback(self._on_status)
        # `receivedChar` in cflib 0.1.33, `received_char` in some other builds: resolve it
        # rather than guessing (this is the kind of thing only hardware exposes).
        console_cb = (getattr(self.cf.console, "received_char", None)
                      or getattr(self.cf.console, "receivedChar", None))
        if console_cb is not None:
            console_cb.add_callback(self._on_console)
        self._sf = SyncCrazyflie(args.uri, cf=self.cf)
        # This retries the link indefinitely with no output of its own, so say what it is
        # doing: otherwise a taken dongle or a powered-off drone looks like a plain hang.
        print(f"  link        : connecting to {args.uri} (retries until the drone answers)",
              flush=True)
        self._sf.__enter__()
        print("  link        : connected", flush=True)
        self.messages.append(f"link: {args.uri}")
        time.sleep(1.0)

        # motors-off until the pilot asks otherwise, and clear the firmware's thrust lock
        # (a COMMANDER_PRIORITY_DISABLE latches `thrustLocked` until a zero-thrust packet
        # arrives - without this the first spool-up does nothing).
        for _ in range(3):
            self._send_stop()
            time.sleep(0.05)

        # The firmware's supervisor decides whether motors may run at all, and it is NOT the
        # same gate as our app's `armed` param. Read it out loud: if it is not "Can be armed"
        # then arming will be refused and no amount of thrust will do anything.
        print(f"  supervisor  : {self.supervisor_states()}", flush=True)
        if "Is locked" in self.supervisor_state_names():
            print("  supervisor  : *** LOCKED - POWER-CYCLE THE DRONE BEFORE GOING FURTHER. ***",
                  flush=True)
            print("  supervisor  : Locked has NO software exit (the transitionsLocked table in "
                  "supervisor_state_machine.c only points back at Locked), and the emergency-"
                  "stop latch behind it is never cleared. Nothing below will work.", flush=True)
        elif self.args.arm_ok and not self.args.no_motors:
            print("  supervisor  : motors are NOT live yet - press 'a' to arm "
                  "(or just press 'w')", flush=True)

        ctrl = self._read_param("stabilizer.controller")
        self.messages.append(f"stabilizer.controller = {ctrl}"
                             + ("" if ctrl == "6" else "  (not 6: the app is not the controller!)"))
        shadow = self._read_param("policy.shadow")
        if shadow is None:
            self.messages.append("policy.shadow unreadable - is the policy image flashed?")
        elif str(shadow) in ("1", "1.0"):
            self.messages.append("*** policy.shadow = 1: arming will compute but COMMAND NOTHING. "
                                 "Set policy.shadow = 0 before expecting the policy to fly. ***")

        # `--set name=value` (repeatable). The one that matters is `policy.shadow=0`: it is
        # what turns the app from "compute and log" into "compute and fly".
        for spec in (self.args.set or []):
            name, _, value = spec.partition("=")
            name, value = name.strip(), value.strip()
            if not name or not value:
                print(f"  param       : --set {spec!r} IGNORED - needs NAME=VALUE", flush=True)
                continue
            # A param name must be the COMPLETE `group.name` form: cflib resolves it with
            # `toc.get_element_by_complete_name()` and the wire format is `group\0name\0`, so
            # a short name like `lh2maxRate` raises. This MUST be visible on a terminal-only
            # session - the old code queued it in `self.messages`, which only the GUI renders,
            # so a misnamed param silently did nothing and looked like "the setting worked".
            try:
                self.cf.param.set_value(name, value)
                got = self._read_param(name)
                print(f"  param       : {name} = {value}"
                      + (f"   (reads back {got})" if got is not None else ""), flush=True)
            except Exception as exc:
                print(f"  param       : {name} NOT SET ({exc}) - a param name needs its "
                      f"group, e.g. lighthouse.lh2maxRate or policy.shadow", flush=True)
            time.sleep(0.2)

    # -- link helpers --------------------------------------------------------------------
    def _on_console(self, ch: str) -> None:
        """
        Print the drone's console straight to our terminal.

        This used to be a deliberate `pass` and it threw away exactly the evidence needed for
        two separate failures:
          * the BOOT BANNER - which is the only way to tell that the MCU RESET (a reset drops
            the radio link AND re-runs the deck init, which is exactly "loses the base
            stations and the link LED goes orange at the same moment");
          * the firmware's ASSERT / hard-fault messages, which name the actual cause;
          * the Lighthouse `LHFL:` lines (bitstream CRC, "FPGA not booted", deck flasher),
            which nothing else reports.
        cflib delivers ONE CHARACTER per callback, so this is a char-wise flush.
        """
        if ch == "\0":
            return
        self._con_line.append(ch)
        # Treat BOTH \n and \r as end-of-line: CRLF consoles then yield a real line plus an
        # empty one (skipped below), and a console that only ever emits a bare \r - which
        # would otherwise buffer forever and print nothing - still gets flushed.
        if ch not in ("\n", "\r"):
            if len(self._con_line) > 400:        # runaway guard: no terminator at all
                self._con_line.append("\n")
            else:
                return
        line = "".join(self._con_line).replace("\n", "").replace("\r", "")
        self._con_line.clear()
        if not line.strip():
            return
        # Flag the lines that carry a verdict, so they are not lost in the wall of
        # "thrust centre -> NN%" chatter. `Watchdog: no frames received from deck` is the
        # firmware's OWN verdict on the UART link and is the read-out for the comSync
        # hypothesis: if it appears, the byte stream died (firmware-side latch); if it
        # never appears while lh_recv == 0, the deck is still talking and the problem is
        # optical. "FPGA not booted. Lighthouse disabled!" means the task is parked in a
        # `while(1) vTaskDelay(portMAX_DELAY)` and the Lighthouse is OFF for the whole boot.
        key = ("Watchdog:", "ASSERT", "HardFault", "FPGA not booted", "LHFL:",
               "rate is off", "out of bounds", "not getting data")
        if any(k in line for k in key):
            print(f"\n  !! FW: {line.strip()}\n", flush=True)
        else:
            print(f"{line}\n", end="", flush=True)

    def _on_status(self, data: bytes) -> None:
        import struct
        if len(data) < 30 or data[0] != STATUS_MAGIC:
            return
        (magic, armed, mode, shadow, abort, manoeuvre,
         z, tilt, thrust, p_err, vbat, act0) = struct.unpack("<6B6f", data[:30])
        with self._lock:
            self.last_status = dict(armed=armed, mode=mode, shadow=shadow, abort=abort,
                                    manoeuvre=manoeuvre, z=z, tilt=tilt, thrust=thrust,
                                    p_err=p_err, vbat=vbat, act0=act0)
        self.armed = armed == 1
        self.manoeuvre = manoeuvre
        # `mode` is the app's own answer: 1 = MANOEUVRE (a relocated table), 0 = HOLD. Use
        # it, so the panel stays honest when a table ends on its own - the broadcast is
        # 10 Hz, so give a fresh launch a few ticks before believing a "not playing" byte.
        if mode != 1 and self.mode != MODE_HOVER and self.mode.startswith("POLICY ("):
            if time.time() - self._play_at > 0.30:
                self.mode = MODE_HOVER if self.armed else self._stock_label()

    def _read_param(self, name: str, timeout: float = 2.0) -> Optional[str]:
        getter = getattr(self.cf.param, "get_value", None)
        if callable(getter):
            try:
                return str(getter(name, timeout=timeout))
            except Exception:
                pass
        done = threading.Event()
        box: Dict[str, str] = {}

        def cb(_n, v):
            box["v"] = v
            done.set()

        try:
            self.cf.param.add_update_callback(group=name.split(".")[0],
                                              name=name.split(".", 1)[1], cb=cb)
            self.cf.param.request_param_update(name)
        except Exception:
            return None
        done.wait(timeout)
        return box.get("v")

    def send_cmd(self, cmd: int) -> None:
        self.cf.appchannel.send_packet(bytes([cmd]))

    def _raw_stop(self) -> None:
        """Send the two stop packets. Deliberately does NOT touch our own log state."""
        try:
            self.cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)   # legacy RPYT, raw thrust 0
        except Exception:
            pass
        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass

    def _send_stop(self) -> None:
        """Motors off, and reset what we believe we have sent."""
        self._raw_stop()
        self._thrust_sent = 0.0
        self.applied = (0.0, 0.0, 0.0, 0.0)

    def _stock_label(self) -> str:
        """The mode label for manual flight - VEL-HOLD changes what the vehicle is doing."""
        return MODE_VELHOLD if self._vel_hold else MODE_STOCK

    def send_velocity_setpoint(self, vx: float, vy: float, vz: float,
                               yawrate_dps: float) -> None:
        """
        Send a WORLD-FRAME VELOCITY setpoint: the stock firmware's own position hold.

        cflib has no sender for this type, so the packet is built by hand. Layout verified
        against the firmware: `crtp_commander_generic.c` has
            enum packet_type { ... velocityWorldType = 8, ... };
            struct velocityPacket_s { float vx, vy, vz, yawrate; } __attribute__((packed));
        and `velocityDecoder` sets
            setpoint->mode.x = setpoint->mode.y = setpoint->mode.z = modeVelocity;
            setpoint->attitudeRate.yaw = values->yawrate;

        WHY THIS FIXES THE DRIFT: in `position_controller_pid.c::positionController`,
        `mode.x == modeAbs` runs the position PID, otherwise `setpoint->velocity.x` is used
        DIRECTLY and `velocityController` closes a velocity loop. `manualDecoder` - what
        `send_setpoint_manual` uses - sets mode.x/y = modeDisable, i.e. ATTITUDE ONLY with no
        corrective loop, which is exactly why the vehicle drifts on its own.

        Wire format is `[type][vx][vy][vz][yawrate]` = 17 bytes; the firmware asserts the 16
        bytes AFTER the type byte equal sizeof(velocityPacket_s).
        """
        import struct as _struct
        from cflib.crazyflie.commander import SET_SETPOINT_CHANNEL, TYPE_VELOCITY_WORLD
        from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

        pk = CRTPPacket()
        pk.port = CRTPPort.COMMANDER_GENERIC
        pk.channel = SET_SETPOINT_CHANNEL
        pk.data = _struct.pack("<Bffff", TYPE_VELOCITY_WORLD,
                               float(vx), float(vy), float(vz), float(yawrate_dps))
        self.cf.send_packet(pk)

    # -- firmware supervisor: the gate that actually decides whether motors may run ------
    def supervisor_state_names(self) -> list:
        """The firmware supervisor's active states, as words. Empty list if unreadable."""
        try:
            bits = self.cf.supervisor.read_bitfield()
            return list(self.cf.supervisor.decode_bitfield(bits))
        except Exception:
            return []

    def supervisor_states(self) -> str:
        """The firmware's own view of itself, in words. Never raises."""
        try:
            bits = self.cf.supervisor.read_bitfield()
            names = list(self.cf.supervisor.decode_bitfield(bits))
        except Exception as exc:
            return f"(unreadable: {exc})"
        return f"0x{bits:03x} " + (f"[{', '.join(names)}]" if names else "[nothing set]")

    def explain_arming_refusal(self) -> None:
        """Say WHICH condition is refusing the arm, and what clears it."""
        names = self.supervisor_state_names()
        print(f"  supervisor  : {self.supervisor_states()}", flush=True)
        if "Is locked" in names:
            # Verified in firmware/src/modules/src/supervisor_state_machine.c: the
            # transitionsLocked table has a single entry and it points at Locked itself
            # (triggerCombiner=never, blockerCombiner=always), so nothing can leave it.
            print("  supervisor  : *** LOCKED - POWER-CYCLE THE DRONE. There is NO software "
                  "way out of this state. ***", flush=True)
            print("  supervisor  : it is caused by an EMERGENCY STOP, and that latch is "
                  "permanent: crtp_supervisor.c sets isEmergencyStopRequested=true and "
                  "never clears it (only a reboot resets it).", flush=True)
            stop = self._read_param("supervisor.stop")
            if stop is not None:
                print(f"  supervisor  : supervisor.stop = {stop}  (1 = the emergency-stop "
                      f"PARAM is the cause; 0 = it came from a CRTP request or the "
                      f"emergency-stop watchdog expiring)", flush=True)
        elif "Is tumbled" in names:
            print("  supervisor  : the drone reports TUMBLED - set it level and keep it still.",
                  flush=True)
        elif "Is crashed" in names:
            print("  supervisor  : the drone reports CRASHED - send a crash-recovery request "
                  "or power-cycle it.", flush=True)
        elif "Can be armed" in names:
            print("  supervisor  : the drone SAYS it can be armed, so the request or the "
                  "read-back is not getting through - retry, and check the link.", flush=True)
        else:
            print("  supervisor  : the 'Can be armed' bit is not set and none of the known "
                  "blockers is reported - a pre-flight check is failing.", flush=True)

    def poll_supervisor(self) -> None:
        """Cache the firmware supervisor flags for the CSV. Non-fatal, ~1 Hz."""
        try:
            self.fw_sup = {"bits": int(self.cf.supervisor.read_bitfield()),
                           "armed": bool(self.cf.supervisor.is_armed),
                           "can_fly": bool(self.cf.supervisor.can_fly),
                           "locked": bool(self.cf.supervisor.is_locked)}
        except Exception:
            pass

    def start_fw_log(self, period_ms: int = 20) -> None:
        if getattr(self.args, "no_fw_log", False):
            print("  fw log      : DISABLED (--no-fw-log) - the fw_* CSV columns will read -1",
                  flush=True)
            return
        print(f"  fw log      : {period_ms} ms -> ~{1000.0 / period_ms:.0f} packets/s per block",
              flush=True)
        """
        Log the firmware's motor ratios/RPMs and its own vbat into `self.fw`.

        Reports and survives every failure: a flight tool must not refuse to fly because a
        log variable moved. Variable types are taken from the TOC rather than assumed, and
        the byte budget is CHECKED against `LogConfig.MAX_LEN` (26 B/block) instead of
        discovered by getting an AttributeError at capture time.

        PERIOD MATTERS FOR ANALYSIS: the CSV is written by the 50 Hz control loop, so a log
        period of 100 ms makes each logged change span ~4 CSV rows and dividing one by the
        24 ms row spacing inflates every implied speed by ~4x. At 20 ms the two are in step.
        """
        try:
            from cflib.crazyflie.log import LogConfig, LogTocElement

            # `cf.log` is Optional in cflib's typing and its TOC is only populated once the
            # link has walked the log TOC, so check rather than assume.
            toc = getattr(self.cf.log, "toc", None)
            if toc is None:
                print("  firmware log: no log TOC available - motor/battery columns will "
                      "read -1", flush=True)
                return

            wanted: List[Tuple[str, str, int]] = []
            missing: List[str] = []
            for var in FW_LOG_VARS:
                elem = toc.get_element_by_complete_name(var)
                if elem is None:
                    missing.append(var)
                    continue
                ctype = elem.ctype
                wanted.append((var, ctype,
                               LogTocElement.get_size_from_id(
                                   LogTocElement.get_id_from_cstring(ctype))))
            if missing:
                print(f"  firmware log: not in this firmware, skipped: {', '.join(missing)}",
                      flush=True)
            if not wanted:
                print("  firmware log: none of the telemetry variables is in this firmware - "
                      "those CSV columns will read -1", flush=True)
                return

            # Greedy pack into <= MAX_LEN blocks. One block is NOT enough for position +
            # velocity + attitude + battery + four motor ratios, and cflib only enforces the
            # limit in add_config() - i.e. after every variable has already been queued.
            chunks: List[List[Tuple[str, str, int]]] = []
            cur: List[Tuple[str, str, int]] = []
            used = 0
            for item in wanted:
                if cur and used + item[2] > LogConfig.MAX_LEN:
                    chunks.append(cur)
                    cur, used = [], 0
                cur.append(item)
                used += item[2]
            if cur:
                chunks.append(cur)

            def cb(_ts, data, _lg):
                with self._lock:
                    self.fw.update({k: float(v) for k, v in data.items()})

            blocks: List[Any] = []
            for idx, chunk in enumerate(chunks):
                lg = LogConfig(name=f"flight{idx}", period_in_ms=period_ms)
                for name, ctype, _size in chunk:
                    lg.add_variable(name, ctype)
                self.cf.log.add_config(lg)
                lg.data_received_cb.add_callback(cb)
                blocks.append(lg)
            for lg in blocks:
                lg.start()
            self._fw_logs = blocks
            print(f"  firmware log: {len(wanted)} variables in {len(chunks)} block(s), "
                  f"max payload "
                  f"{max(sum(s for _n, _c, s in c) for c in chunks)} B "
                  f"(limit {LogConfig.MAX_LEN} B/block)", flush=True)
            print(f"  link load   : {len(chunks)} block(s) x {1000.0 / period_ms:.0f}/s = "
                  f"~{len(chunks) * 1000.0 / period_ms:.0f} log packets/s of the CRTP TX "
                  f"queue (depth 200). Only the log uses the BLOCKING send, and it does so "
                  f"from the worker task (priority 1, the lowest) - so this cannot stall the "
                  f"stabilizer, Kalman or Lighthouse tasks. The console uses a 0-timeout "
                  f"send and DROPS instead, marking it with '<F>'", flush=True)
        except Exception as exc:
            print(f"  firmware log: not started ({exc})", flush=True)

    def firmware_arm(self, do_arm: bool, settle: float = 1.5) -> bool:
        """
        ARM/DISARM the FIRMWARE's supervisor, then CONFIRM by reading it back.

        This is a DIFFERENT gate from our app's `armed` param, and it is the one that decides
        whether the motors may turn at all. From `stabilizer.c`:

            const bool canFly = supervisorCanFly();
            crtpCommanderBlock(!canFly);
            ...
            if (!canFly) { setpoint = (setpoint_t){0}; }      // your thrust, discarded

        `canFly` is true only in ReadyToFly / Flying / WarningLevelOut / Landed, and the only
        routes there are an arming request over CRTP (this call) or an external transmitter
        (`extrx.c`). The commander does NOT arm it, and neither does our appchannel command -
        `armSystem` over CRTP is the only thing that works.

        MEASURED 2026-09-17: without this, a commanded 60% thrust is zeroed before the
        controller ever sees it. Motors never spin, the battery never sags and `z` never
        moves - which reads exactly like "not enough thrust", and is why raising
        `--max-thrust` changed nothing at all.
        """
        try:
            self.cf.supervisor.send_arming_request(bool(do_arm))
        except Exception as exc:
            self.messages.append(f"arming request failed: {exc}")
            return False
        deadline = time.time() + settle
        last = ""
        while time.time() < deadline:
            time.sleep(0.1)
            try:
                is_armed = self.cf.supervisor.is_armed
                can_fly = self.cf.supervisor.can_fly
            except Exception:
                break
            last = f"armed={is_armed} can_fly={can_fly}"
            if is_armed == bool(do_arm):
                self._motors_ok = bool(do_arm)
                print(f"  supervisor  : {'ARMED - motors are live' if do_arm else 'disarmed'}"
                      f"  ({last})", flush=True)
                return True
        self._motors_ok = False
        print(f"  supervisor  : NOT {'armed' if do_arm else 'disarmed'} after {settle:.1f}s "
              f"({last or 'no reply from the supervisor'})", flush=True)
        if do_arm:
            self.explain_arming_refusal()
        return False

    # -- modes ---------------------------------------------------------------------------
    def arm(self) -> None:
        if not self.args.arm_ok:
            self.messages.append("refusing to arm: pass --arm-ok (props off for the first run)")
            return
        # Firmware FIRST: with the supervisor unarmed the app's policy output is zeroed by
        # stabilizer.c, so arming the app alone would look like it worked and do nothing.
        if not self.firmware_arm(True):
            self.messages.append("*** the FIRMWARE refused to arm - the policy cannot move the "
                                 "motors until it does ***")
            return
        self.send_cmd(CMD_ARM)
        self.mode = MODE_HOVER
        self.messages.append("HOVER: armed - the policy holds the hover where it was when "
                             "you pressed it (appchannel 0x01)")

    def disarm(self) -> None:
        self.send_cmd(CMD_DISARM)
        self.mode = self._stock_label()
        self.messages.append("STOCK: app disarmed - your sticks have it back (appchannel 0x02); "
                             "firmware still armed, 'x' to stop the motors")

    def flip(self) -> None:
        """Shorthand the pad button uses: launch the baked flip."""
        self.play(self.flip_index, "flip")

    def play(self, kind: int, name: str) -> None:
        """
        Launch a baked manoeuvre.

        The app relocates it onto the CURRENT pose and heading at the next 100 Hz tick and
        hands back to a hold anchored where it ended, so this is exactly the live_flight
        "FLIP / 6 orbit / 7 figure-8 ..." button, on the real vehicle.

        Printed to the TERMINAL as well as queued: `self.messages` is only rendered by the
        panel, so with --no-gui a refusal would otherwise be completely silent.
        """
        if not self.armed:
            print(f"  {name}: the POLICY must be flying first - press 'h' "
                  f"(and start the session with --set policy.shadow=0)", flush=True)
            return
        self.cf.appchannel.send_packet(bytes([CMD_PLAY, kind & 0xFF]))
        self._play_at = time.time()
        self.mode = f"POLICY ({name})"
        print(f"  {name} launched (appchannel 0x05 {kind}) - relocated onto the current pose, "
              f"hands back to hover at the end", flush=True)

    def hold(self) -> None:
        """Stop whatever is playing and hold wherever the vehicle is."""
        if not self.armed:
            print("  nothing to hold: the POLICY is not flying (press 'h')", flush=True)
            return
        self.cf.appchannel.send_packet(bytes([CMD_PLAY, KIND_NONE]))
        self._play_at = time.time()
        self.mode = MODE_HOVER
        print("  manoeuvre stopped - holding here", flush=True)

    def kill(self) -> None:
        self._killed = True
        self._send_stop()
        try:
            self.firmware_arm(False, settle=0.5)
        except Exception:
            pass
        try:
            self.send_cmd(CMD_DISARM)
        except Exception:
            pass
        self.mode = self._stock_label()
        self.messages.append("*** KILL: motors off + disarmed (firmware too) ***")

    # -- keyboard ------------------------------------------------------------------------
    def nudge_thrust(self, delta: float) -> None:
        """
        Move the thrust CENTRE by `delta` percent (that is what climb/descend do).

        Trimming the centre rather than adding a momentary offset is deliberate. A terminal
        cannot report key RELEASE - exactly the same limitation the MuJoCo viewer has - so a
        "hold to climb" binding would either latch on or depend on the OS auto-repeat rate.
        Stepping the centre is the same model as the panel's +/-5% buttons: each press is one
        step, holding the key auto-repeats into a ramp, and the value is readable back.

        Printed straight to the terminal, not just queued in `self.messages`: those are only
        rendered by the PANEL, so with --no-gui a queued message would be invisible.
        """
        ceiling = float(self.args.max_thrust)
        new = float(np.clip(self.hover_pct + delta, 0.0, ceiling))
        if abs(new - self.hover_pct) < 1e-9:
            if delta > 0:
                # This is a REAL trap: the sim's hover maps to ~46% of the command range for
                # a 28 g model, and the physical vehicle is heavier, so a default --max-thrust
                # of 70 can sit BELOW the thrust actually needed to leave the ground. Nothing
                # responds, and the natural conclusion is "it is broken" rather than "I am
                # clamped". Say what to change.
                print(f"  thrust centre already at the ceiling ({new:.0f}%, --max-thrust "
                      f"{ceiling:.0f}%)", flush=True)
                if ceiling < 100.0:
                    print(f"    if it still will not lift, raise the ceiling: "
                          f"--max-thrust 100", flush=True)
                else:
                    print("    this is already full authority - if it will not lift at 100%, "
                          "it is not a command problem: suspect the battery under load "
                          "(a 1S pack near 3.7 V sags hard) or the props/motors", flush=True)
            else:
                print(f"  thrust centre already at zero (limit {ceiling:.0f}% is the ceiling)",
                      flush=True)
            return
        self.hover_pct = new
        note = ""
        if self.armed:
            note = "   (ARMED: the policy owns thrust - disarm to fly it by hand)"
        print(f"  thrust centre -> {new:.0f}%{note}", flush=True)
        self.messages.append(f"thrust centre -> {new:.0f}%")

    def handle_key(self, key: str) -> None:
        """Keyboard flight control (see the docstring in `run()` for the binding table)."""
        k = key.lower()
        if k in ("w", "up"):
            if not self._motors_ok and self.args.arm_ok and not self.args.no_motors:
                # "Raise the thrust" while nothing is live is unambiguous intent, and the only
                # reason it did nothing before is that the FIRMWARE was never armed.
                print("  arming the firmware (that is what makes the motors live)", flush=True)
                self.firmware_arm(True)
            self.nudge_thrust(+self.args.key_thrust_step)
        elif k in ("s", "down"):
            self.nudge_thrust(-self.args.key_thrust_step)
        elif k == "a":
            if self.args.no_motors:
                print("  --no-motors: not arming", flush=True)
            else:
                self.firmware_arm(True)
        elif k == "h":
            # Hand control to the POLICY. This is the mode that HOLDS POSITION: the stock inner
            # loop is attitude-only, so anything before this drifts and nothing can stop it.
            self.arm()
        elif k == "d":
            self.disarm()
        elif k == "v":
            if self.armed:
                print("  velocity hold is a STOCK-mode feature - press 'd' first", flush=True)
            else:
                self._vel_hold = not self._vel_hold
                self.mode = self._stock_label()
                self._thrust_sent = 0.0
                print("  velocity hold "
                      + ("ON - the stock position loop holds station (x/y velocity -> 0)"
                         if self._vel_hold else
                         "OFF - attitude only, so it WILL drift"), flush=True)
        elif k.isdigit() and self.kinds:
            # Manoeuvres by the SAME index the startup line prints and appchannel 0x05 uses, so
            # the number you type IS the number the GUI shows - no 0-vs-1-based confusion.
            idx = int(k)
            match = [t for t in self.kinds if t[0] == idx]
            if not match:
                print(f"  no baked manoeuvre {idx}; have "
                      + ", ".join(f"{i}={n}" for i, n in self.kinds), flush=True)
            else:
                self.play(match[0][0], match[0][1])
        elif k == ".":
            self.hold()
        elif k == " ":
            print(f"  thrust centre -> {self.args.hover:.0f}% (--hover)", flush=True)
            self.hover_pct = float(self.args.hover)
        elif k == "x":
            self.kill()
        elif k in ("q", "esc", "\x03"):
            self.running = False
            print("  quit", flush=True)

    # -- per-tick setpoint ---------------------------------------------------------------
    def pilot_tick(self) -> None:
        """Stream the pilot's setpoint. Harmless while armed (the app ignores it) and
        necessary while disarmed, because that IS the stock control path."""
        pad = self.pad_state
        roles = self.mapper.roles(pad) if (pad is not None and pad.connected) else {
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "throttle": 0.0}
        roll, pitch, yaw, thrust = commands_from_pad(
            roles, self.hover_pct, self.span_pct, self.args.max_angle, self.args.max_yaw,
            invert_roll=self.args.invert_roll, invert_pitch=self.args.invert_pitch)
        thrust = float(np.clip(thrust, 0.0, self.max_pct))
        # slew limit: a stick slam must not become a thrust step (the rate PID can only do
        # so much, and the props cannot do more)
        step = self.args.thrust_slew * self.dt
        self._thrust_sent = float(np.clip(thrust, self._thrust_sent - step, self._thrust_sent + step))
        self.applied = (roll, pitch, yaw, self._thrust_sent)
        if self.args.no_motors:
            return
        if not self._motors_ok:
            # The firmware supervisor has not armed us, so stabilizer.c would zero this
            # setpoint anyway. Send a real stop so the props genuinely stay still, and reset
            # the slewed value so arming ramps up from zero instead of jumping straight to the
            # centre. `applied` deliberately keeps the COMMANDED value, so the log still shows
            # what you asked for while nothing is turning.
            self._raw_stop()
            self._thrust_sent = 0.0
            return
        if self._vel_hold:
            # ALTITUDE IS A CLIMB RATE HERE: the thrust centre's offset from `--hover` maps to
            # +-`--climb-rate`, so the centre is a LEVEL HOLD and w/s climb and descend. x/y
            # are held at zero velocity, which is the point of the mode.
            vz = ((self.hover_pct - float(self.args.hover)) / 50.0) * self.args.climb_rate
            try:
                self.send_velocity_setpoint(0.0, 0.0, vz, yaw)
            except Exception as exc:
                self.messages.append(f"velocity setpoint send failed: {exc}")
            return
        try:
            self.cf.commander.send_setpoint_manual(roll, pitch, yaw, self._thrust_sent,
                                                   rate=self.args.rate_mode)
        except Exception as exc:
            self.messages.append(f"setpoint send failed: {exc}")

    # -- pad buttons ---------------------------------------------------------------------
    def pad_buttons(self) -> None:
        pad = self.pad_state
        if pad is None:
            return
        now = set(pad.pressed)
        for b, action in ((BTN_KILL, self.kill), (BTN_HOVER, self.arm),
                          (BTN_STOCK, self.disarm), (BTN_FLIP, self.flip)):
            if b in now and b not in self._prev_buttons:
                action()
        self._prev_buttons = now

    # -- panel ---------------------------------------------------------------------------
    def handle_gui(self, command: str) -> None:
        try:
            if command == "mode:stock":
                self.disarm()
            elif command == "mode:hover":
                self.arm()
            elif command == "mode:flip":
                self.flip()
            elif command == "hold":
                self.hold()
            elif command.startswith("traj:"):
                name = command.split(":", 1)[1]
                match = [k for k in self.kinds if k[1] == name]
                if match:
                    self.play(match[0][0], match[0][1])
                else:
                    self.messages.append(f"no baked manoeuvre called '{name}' "
                                         f"(have: {', '.join(k[1] for k in self.kinds)})")
            elif command == "kill":
                self.kill()
            elif command.startswith("hover:"):
                self.hover_pct = float(command.split(":", 1)[1])
                self.messages.append(f"hover thrust -> {self.hover_pct:.0f}%")
            elif command == "quit":
                self.running = False
        except Exception as exc:
            self.messages.append(f"panel command '{command}' failed: {exc}")

    def status_dict(self) -> Dict[str, Any]:
        with self._lock:
            st = dict(self.last_status)
        return {
            "mode": self.mode,
            "armed": self.armed,
            "app_abort": st.get("abort", 0),
            "manoeuvre": st.get("manoeuvre", KIND_NONE),
            "manoeuvre_name": next((n for i, n in self.kinds
                                    if i == st.get("manoeuvre")), ""),
            "kinds": [{"index": i, "name": n} for i, n in self.kinds],
            "shadow": st.get("shadow", -1),
            "z": st.get("z", 0.0),
            "tilt": st.get("tilt", 0.0),
            "policy_thrust_pct": st.get("thrust", 0.0) / 600.0,      # raw units -> %
            "vbat": st.get("vbat", 0.0),
            "p_err": st.get("p_err", 0.0),
            "act0": st.get("act0", 0.0),
            "cmd": self.applied,
            "hover": self.hover_pct,
            "link": self.args.uri,
            "pad": self.pad_state.name if (self.pad_state is not None and self.pad_state.connected) else "none",
            "message": self.messages[-1] if self.messages else "",
        }

    # -- main loop -----------------------------------------------------------------------
    dt = 0.02                                     # 50 Hz setpoint stream
    LOG_EVERY = 0.1

    def run(self) -> None:
        if not self.args.no_gui:
            self._start_gui()
        print("\nradio flight. GUI: STOCK / HOVER / manoeuvres / HOLD / KILL.  Ctrl-C = kill + quit.")
        self._keys = None if self.args.no_keys else TerminalKeys()
        if self._keys is not None and self._keys.enabled:
            print(f"  KEYBOARD: w / up-arrow = climb   s / down-arrow = descend "
                  f"({self.args.key_thrust_step:.0f}% per press, hold to repeat)")
            print("            a = ARM the firmware (makes the motors live; 'w' does it too)")
            print("            v = VELOCITY HOLD: the stock position loop holds station"
                  " (kills the drift)")
            print("            h = HOVER: hand control to the POLICY (it holds position)")
            print("            d = back to STOCK (attitude only - it WILL drift)")
            print("            space = centre thrust   x = KILL (motors off)   q / ESC = quit")
        elif not self.args.no_keys:
            print("  KEYBOARD: not available (stdin is not a terminal) - use the panel")
        if self.kinds:
            print("  baked manoeuvres (also in the GUI) - press the NUMBER to launch one: "
                  + ", ".join(f"{i}={n}" for i, n in self.kinds))
            print("    they need the POLICY flying first ('h'), then relocate onto the current "
                  "pose; '.' stops and holds")
        else:
            print(f"  no generated header at {REF_HEADER} - run gen_references.py to bake tables")
        for m in self.messages:
            print(f"  {m}")
        self.messages.clear()
        self._log = open(os.path.join(_PROJECT_ROOT, "logs", "radio_flight_log.csv"), "w")
        # `app_vbat` is the app's status-packet value, which is a PLACEHOLDER unless the app
        # resolved pm.vbat (see FW_LOG_VARS). `pm_vbat` is the firmware's own reading and is
        # the one to trust. The m*_rpm/m* columns are what prove whether motors turned.
        self._log.write("t,mode,armed,roll,pitch,yaw,thrust_pct,z,tilt,policy_thrust_pct,"
                        "app_vbat,fw_armed,can_fly,locked,sup_bits,"
                        "x,y,vx,vy,fw_roll,fw_pitch,m1,m2,m3,m4,pm_vbat,lh_status,lh_recv,"
                        "lh_sync,lh_active,lh_calud,"
                        "cycle_rt,frame_rt,pre_th_rt,post_th_rt,dis_prob\n")
        self.start_fw_log(period_ms=int(getattr(self.args, "log_period", 20)))
        self.poll_supervisor()
        t0 = time.time()
        status_t = 0.0
        try:
            while self.running:
                tick = time.time()
                if self._keys is not None:
                    for key in self._keys.read():
                        self.handle_key(key)
                self.pad_state = self.pad.state() if self.pad is not None else None
                self.pad_buttons()
                if not self._killed:
                    self.pilot_tick()
                # the app pushes status on its own (~5 Hz); ask explicitly now and then so
                # the panel stays live even before the first push
                if time.time() - status_t > 0.5:
                    status_t = time.time()
                    self.send_cmd(CMD_STATUS)
                # The firmware supervisor, at 1 Hz: the flags are what explain a run that
                # commanded thrust and nothing happened.
                if time.time() - self._sup_t > 1.0:
                    self._sup_t = time.time()
                    self.poll_supervisor()
                if self._gui is not None:
                    for msg in self._gui.poll():
                        if msg.get("type") == "command":
                            self.handle_gui(str(msg.get("command", "")))
                    if time.time() - getattr(self, "_pub_t", 0.0) > 0.1:
                        self._pub_t = time.time()
                        self._gui.publish(self.status_dict())
                t = time.time() - t0
                st = self.last_status
                fw = dict(self.fw)
                sup = self.fw_sup
                self._log.write(
                    f"{t:.3f},{self.mode},{int(self.armed)},"
                    f"{self.applied[0]:.1f},{self.applied[1]:.1f},{self.applied[2]:.1f},"
                    f"{self.applied[3]:.1f},{st.get('z', 0):.3f},{st.get('tilt', 0):.1f},"
                    f"{st.get('thrust', 0) / 600.0:.1f},{st.get('vbat', 0):.2f},"
                    f"{int(sup.get('armed', 0))},{int(sup.get('can_fly', 0))},"
                    f"{int(sup.get('locked', 0))},{int(sup.get('bits', -1))},"
                    f"{fw.get('stateEstimate.x', -1):.3f},{fw.get('stateEstimate.y', -1):.3f},"
                    f"{fw.get('stateEstimate.vx', -1):.3f},{fw.get('stateEstimate.vy', -1):.3f},"
                    f"{fw.get('stabilizer.roll', -1):.1f},"
                    f"{fw.get('stabilizer.pitch', -1):.1f},"
                    f"{fw.get('motor.m1', -1):.0f},{fw.get('motor.m2', -1):.0f},"
                    f"{fw.get('motor.m3', -1):.0f},{fw.get('motor.m4', -1):.0f},"
                    f"{fw.get('pm.vbat', -1):.2f},"
                    f"{fw.get('lighthouse.status', -1):.0f},"
                    f"{fw.get('lighthouse.bsReceive', -1):.0f},"
                    f"{fw.get('lighthouse.comSync', -1):.0f},"
                    f"{fw.get('lighthouse.bsActive', -1):.0f},"
                    f"{fw.get('lighthouse.bsCalUd', -1):.0f},"
                    f"{fw.get('lighthouse.cycleRt', -1):.1f},"
                    f"{fw.get('lighthouse.frameRt', -1):.1f},"
                    f"{fw.get('lighthouse.preThRt', -1):.1f},"
                    f"{fw.get('lighthouse.postThRt', -1):.1f},"
                    f"{fw.get('lighthouse.disProb', -1):.2f}\n")
                self._log.flush()
                rest = self.dt - (time.time() - tick)
                if rest > 0:
                    time.sleep(rest)
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self._finish()

    def _start_gui(self) -> None:
        self._gui = GuiServer(port=int(self.args.port))
        if not self._gui.start():
            self.messages.append(f"panel: could not listen on {self.args.port}")
            self._gui = None
            return
        try:
            self._gui_proc = subprocess.Popen(
                [sys.executable, os.path.join(_HERE, "radio_gui.py"),
                 "--port", str(self.args.port)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.messages.append(f"panel: window on port {self.args.port}")
        except Exception as exc:
            self.messages.append(f"panel: could not start the GUI ({exc})")

    def _finish(self) -> None:
        # Restore the terminal FIRST: everything below prints, and a half-configured tty
        # would garble the shutdown messages (and leave the shell without echo if we died).
        keys = getattr(self, "_keys", None)
        if keys is not None:
            keys.close()
        # motors off, repeatedly, and disarm: the last setpoint is what a dead script would
        # otherwise leave behind (there is no firmware-side deadman).
        for _ in range(5):
            self._send_stop()
            try:
                self.send_cmd(CMD_DISARM)
            except Exception:
                pass
            time.sleep(0.05)
        # Disarm the FIRMWARE too. Stop packets alone do kill the motors, but leaving the
        # supervisor armed means the next thing to send a setpoint flies the vehicle.
        try:
            self.firmware_arm(False, settle=0.5)
        except Exception:
            pass
        try:
            for lg in self._fw_logs:
                lg.stop()
        except Exception:
            pass
        try:
            self._log.close()
        except Exception:
            pass
        if self._gui is not None:
            self._gui.publish(self.status_dict())
            self._gui.stop()
        if self._gui_proc is not None and self._gui_proc.poll() is None:
            self._gui_proc.terminate()
        if self.pad is not None:
            self.pad.stop()
        try:
            self._sf.__exit__(None, None, None)
        except Exception:
            pass
        print("stopped: motors off, disarmed.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fly the real Crazyflie with the policy on a switch")
    ap.add_argument("--uri", default=None,
                    help="cflib URI (default: whatever radio_config.py remembered in "
                         "logs/drone_uri.txt, else the factory "
                         f"{drone_link.DEFAULT_URI})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT + 1)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--arm-ok", action="store_true",
                    help="allow HOVER/manoeuvres (without it, arming is refused)")
    ap.add_argument("--set", action="append", metavar="NAME=VALUE",
                    help="write a param after connecting (repeatable). The usual one is "
                         "--set policy.shadow=0, without which arming computes but "
                         "commands nothing")
    ap.add_argument("--hover", type=float, default=50.0,
                    help="centre-stick thrust (%%) = the hover command. CALIBRATE IT. "
                         "Read the answer off the policy: policy.thrust_units/60000*100 "
                         "while it hovers.")
    ap.add_argument("--thrust-span", type=float, default=25.0, help="full stick = +-this %%")
    ap.add_argument("--max-thrust", type=float, default=70.0, help="hard ceiling (%%)")
    ap.add_argument("--max-angle", type=float, default=25.0, help="deg at full stick")
    ap.add_argument("--max-yaw", type=float, default=90.0, help="deg/s at full stick")
    ap.add_argument("--thrust-slew", type=float, default=60.0, help="%%/s thrust slew limit")
    ap.add_argument("--rate-mode", action="store_true",
                    help="send RATES instead of angles (acro; no self-levelling)")
    ap.add_argument("--invert-roll", action="store_true")
    ap.add_argument("--invert-pitch", action="store_true")
    ap.add_argument("--no-motors", action="store_true",
                    help="BENCH ONLY: compute and print setpoints, send only stop+disarm")
    ap.add_argument("--vel-hold", action="store_true",
                    help="start in VELOCITY HOLD (stock position loop, no drift); toggle with 'v'")
    ap.add_argument("--climb-rate", type=float, default=0.5,
                    help="VELOCITY HOLD: m/s commanded at the thrust extreme (default 0.5)")
    ap.add_argument("--no-keys", action="store_true",
                    help="do not grab the terminal for keyboard flight control")
    ap.add_argument("--key-thrust-step", type=float, default=2.0,
                    help="%% of thrust added/removed per climb/descend keypress (default 2; "
                         "finer is better for finding the lift-off point)")
    # The firmware log is the ONLY part of the link that can block, and it can only do so from
    # the worker task (WORKER_TASK_PRI = 1, the lowest in the system), so it cannot stall the
    # stabilizer, the Kalman or the lighthouse task. These two flags exist so the claim can be
    # TESTED rather than argued: fly with --no-fw-log and if the Lighthouse still dies, the
    # downlink is definitively not the cause.
    ap.add_argument("--log-period", type=int, default=20, metavar="MS",
                    help="firmware log period (default 20 ms = 50 Hz per block). 20 ms is "
                         "deliberate - at 100 ms the implied speeds were 4.2x too high - but "
                         "raising it trades speed accuracy for link margin")
    ap.add_argument("--no-fw-log", action="store_true",
                    help="do NOT start any firmware log. Use this to test whether OUR telemetry "
                         "is implicated: the CRTP TX queue is 200 deep and logRunBlock uses "
                         "the BLOCKING crtpSendPacketBlock, so this removes that load "
                         "entirely (the CSV loses the fw_* columns, which will read -1)")
    args = ap.parse_args()
    args.uri = drone_link.preferred_uri(args.uri)

    flight = RadioFlight(args)
    flight.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
