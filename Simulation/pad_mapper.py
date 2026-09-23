# -*- coding: utf-8 -*-
"""
Gamepad -> logical stick channels, with a live calibration that fixes mis-mapped axes.

THE PROBLEM THIS SOLVES
-----------------------
"Left is acting like right" is not one failure mode, it is at least four, and they are
indistinguishable from inside the flight code:

  1. a sign inversion on an axis (push left, aircraft rolls right);
  2. the two horizontal axes swapped between the sticks (left stick's X is read where the
     right stick's X belongs - which is exactly what "left acts like right" sounds like);
  3. an X/Y swap inside one stick (a nibble-order error in a vendor report decoding looks
     identical to this);
  4. a different convention than assumed (Mode 2 puts yaw on the left stick's X and roll
     on the right's, Outer Wilds does the opposite).

Guessing between these from a chat message is how you end up flip-flopping defaults. So
the flight program reads sticks through this mapper, which supports all four corrections
AT RUNTIME:

  * `bindings`: which raw axis feeds each logical channel, and with which sign;
  * `layout`: Outer-Wilds roles (roll on the left stick) or Mode 2 (roll on the right);
  * `calibrate_*`: a four-step guided routine ("push the LEFT stick LEFT", ...) that
    MEASURES the binding instead of assuming it, and can be run any time from the GUI.

Logical channels are always:

    lx, ly, rx, ry      +1 = right / up, in the PAD's own frame

so the flight code above this module never has to know what the hardware did.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

LOGICAL_CHANNELS: Tuple[str, ...] = ("lx", "ly", "rx", "ry")
RAW_AXES: Tuple[str, ...] = ("left_x", "left_y", "right_x", "right_y")

PAD_DEADZONE: float = 0.10
PAD_EXPO: float = 0.35
CALIBRATE_MOVE_THRESHOLD: float = 0.55   # raw deflection that counts as "the pilot moved it"
CALIBRATE_RETURN_THRESHOLD: float = 0.20  # back to centre before the next instruction
CALIBRATE_STEP_TIMEOUT: float = 8.0       # s per instruction before giving up


def shape_stick(value: float, deadzone: float = PAD_DEADZONE, expo: float = PAD_EXPO) -> float:
    """Deadzone + cubic-blend expo. Identical shaping to the manual sandbox."""
    v = float(value)
    if abs(v) <= deadzone:
        return 0.0
    v = math.copysign((abs(v) - deadzone) / (1.0 - deadzone), v)
    return (1.0 - expo) * v + expo * v ** 3


@dataclass
class AxisBinding:
    """logical = sign * raw attribute (the sign carries the pad's own convention)."""

    attr: str
    sign: float = 1.0

    def evaluate(self, pad) -> float:
        return float(self.sign) * float(getattr(pad, self.attr, 0.0))


@dataclass
class CalibrationState:
    """Progress of the guided routine (owned by the flight loop, mirrored to the GUI)."""

    active: bool = False
    step: int = 0
    instruction: str = ""
    waiting_for_centre: bool = True
    started: float = 0.0
    step_started: float = 0.0
    message: str = ""
    bindings: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    baseline: Dict[str, float] = field(default_factory=dict)
    done: bool = False
    failed: str = ""


_INSTRUCTIONS = (
    ("lx", "push the LEFT stick LEFT  (and hold it)"),
    ("ly", "push the LEFT stick UP  (and hold it)"),
    ("rx", "push the RIGHT stick LEFT  (and hold it)"),
    ("ry", "push the RIGHT stick UP  (and hold it)"),
)
# The instruction names the direction the LOGICAL channel must read negative/positive:
# "left"/"down" => the logical value is negative when the pilot does it, so sign = -1 x raw.
_INSTRUCTION_DIRECTION = {"lx": -1.0, "ly": +1.0, "rx": -1.0, "ry": +1.0}


class PadMapper:
    """
    Raw pad -> logical sticks. `layout` decides what the logical sticks MEAN; it is applied
    by the caller through `roles()` so one mapper serves every control set.
    """

    def __init__(
        self,
        layout: str = "ow",
        inverts: Optional[Dict[str, bool]] = None,
        bindings: Optional[Dict[str, Tuple[str, float]]] = None,
    ) -> None:
        self.layout = layout if layout in ("ow", "mode2") else "ow"
        self.deadzone = PAD_DEADZONE
        self.expo = PAD_EXPO
        inv = dict(inverts or {})
        # Defaults reproduce the manual sandbox's conventions: X axes standard, Y axes
        # reported "up = negative" by the pads this was developed against.
        default_sign = {
            "lx": -1.0 if inv.get("lx", False) else 1.0,
            "ly": -1.0 if inv.get("ly", True) else 1.0,
            "rx": -1.0 if inv.get("rx", False) else 1.0,
            "ry": -1.0 if inv.get("ry", True) else 1.0,
        }
        self.axes: Dict[str, AxisBinding] = {}
        default_attr = {"lx": "left_x", "ly": "left_y", "rx": "right_x", "ry": "right_y"}
        for ch in LOGICAL_CHANNELS:
            if bindings is not None and ch in bindings:
                attr, sign = bindings[ch]
                self.axes[ch] = AxisBinding(str(attr), float(sign))
            else:
                self.axes[ch] = AxisBinding(default_attr[ch], default_sign[ch])
        self.cal = CalibrationState()

    # -- configuration ------------------------------------------------------------------
    def set_layout(self, layout: str) -> str:
        self.layout = layout if layout in ("ow", "mode2") else "ow"
        return self.layout

    def set_invert(self, channel: str, invert: bool) -> None:
        """Flip ONE logical channel's sign (keeps the axis binding)."""
        if channel in self.axes:
            binding = self.axes[channel]
            binding.sign = -abs(binding.sign) if invert else abs(binding.sign)

    def invert_flags(self) -> Dict[str, bool]:
        return {ch: self.axes[ch].sign < 0.0 for ch in LOGICAL_CHANNELS}

    def swap_sticks(self) -> None:
        """Swap which physical stick feeds the left/right logical channels.

        The one-click fix for a pad whose two sticks arrive in the opposite order
        ("left acts like right"): lx <-> rx and ly <-> ry, signs kept.
        """
        self.axes["lx"], self.axes["rx"] = self.axes["rx"], self.axes["lx"]
        self.axes["ly"], self.axes["ry"] = self.axes["ry"], self.axes["ly"]

    def export(self) -> Dict[str, object]:
        return {
            "layout": self.layout,
            "axes": {ch: [self.axes[ch].attr, self.axes[ch].sign] for ch in LOGICAL_CHANNELS},
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.export(), f, indent=2)

    @classmethod
    def load(cls, path: str, layout: str = "ow") -> "PadMapper":
        mapper = cls(layout=layout)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            axes = {ch: (str(v[0]), float(v[1])) for ch, v in dict(data.get("axes", {})).items()
                    if ch in LOGICAL_CHANNELS and len(v) == 2}
            if axes:
                mapper = cls(layout=str(data.get("layout", layout)), bindings=axes)
        except Exception:
            pass
        return mapper

    # -- per-frame ----------------------------------------------------------------------
    def sticks(self, pad, shaped: bool = True) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for ch in LOGICAL_CHANNELS:
            value = self.axes[ch].evaluate(pad)
            out[ch] = shape_stick(value, self.deadzone, self.expo) if shaped else value
        return out

    def roles(self, pad) -> Dict[str, float]:
        """
        The sticks in the terms each control set uses, in PILOT INTENT units:

            roll, pitch     +1 = roll right / pitch nose down     (the ACRO layout)
            yaw             +1 = turn the nose RIGHT (the consumer negates it into a
                            body-rate command, because +yaw about z is a LEFT turn in
                            this project's convention)
            throttle        +1 = more collective / climb
            move_fwd        +1 = push the LEFT stick forward     (video-game movement)
            move_left       +1 = push the LEFT stick left
            heading         +1 = push the RIGHT stick right      (turn right)

        The video-game channels are deliberately INDEPENDENT of the layout selector:
        movement is always the left stick and heading is always the right stick's X, which
        is what makes the same sticks work in GAME mode and in the policy's stick-tracking
        mode. The layout only decides where ROLL and YAW sit for ACRO flying.
        """
        s = self.sticks(pad)
        if self.layout == "mode2":
            # Mode 2: left stick = throttle + yaw (rudder), right stick = pitch + roll.
            roll, yaw = s["rx"], s["lx"]
        else:
            # Outer Wilds roles: left stick = translate (roll / main thrust), right = aim.
            roll, yaw = s["lx"], s["rx"]
        return {
            "roll": roll, "pitch": s["ry"], "yaw": yaw, "throttle": s["ly"],
            "move_fwd": s["ly"], "move_left": -s["lx"], "heading": s["rx"],
            "lx": s["lx"], "ly": s["ly"], "rx": s["rx"], "ry": s["ry"],
        }

    # -- guided calibration -------------------------------------------------------------
    def calibrate_start(self, pad=None) -> str:
        self.cal = CalibrationState(active=True, step=0, started=time.time(),
                                    step_started=time.time(), waiting_for_centre=True)
        self.cal.instruction = _INSTRUCTIONS[0][1]
        self.cal.message = "centre all sticks, then follow the instructions"
        return self.cal.message

    def calibrate_update(self, pad, now: Optional[float] = None) -> CalibrationState:
        """
        Advance the guided routine. Call ~every frame while `cal.active`.

        Mechanics: wait for the sticks to be at rest, show the instruction, then watch the
        four raw axes and take the one that moves furthest. Its SIGN relative to the
        instructed direction gives the binding sign, so the routine fixes sign errors, axis
        swaps and stick swaps in one pass.
        """
        now = time.time() if now is None else now
        cal = self.cal
        if not cal.active:
            return cal
        raw = {name: float(getattr(pad, name, 0.0)) for name in RAW_AXES}

        if cal.done:
            return cal

        if cal.waiting_for_centre:
            if max(abs(v) for v in raw.values()) < CALIBRATE_RETURN_THRESHOLD:
                cal.waiting_for_centre = False
                cal.step_started = now
                cal.baseline = dict(raw)
                channel, text = _INSTRUCTIONS[cal.step]
                cal.instruction = text
                cal.message = f"step {cal.step + 1}/{len(_INSTRUCTIONS)}"
            elif now - cal.started > 4.0 * CALIBRATE_STEP_TIMEOUT:
                cal.active = False
                cal.failed = "sticks never returned to centre"
            return cal

        if now - cal.step_started > CALIBRATE_STEP_TIMEOUT:
            cal.active = False
            cal.failed = f"no movement seen for: {cal.instruction}"
            return cal

        channel, _text = _INSTRUCTIONS[cal.step]
        delta = {name: raw[name] - cal.baseline.get(name, 0.0) for name in RAW_AXES}
        attr = max(delta, key=lambda name: abs(delta[name]))
        move = delta[attr]
        if abs(move) < CALIBRATE_MOVE_THRESHOLD:
            return cal

        # The instructed direction tells us the intended LOGICAL sign; the measured raw
        # sign gives the pad's convention. logical = sign * raw with sign chosen so the
        # instructed motion produces the intended logical value.
        intended = _INSTRUCTION_DIRECTION[channel]
        sign = 1.0 if (move * intended) > 0.0 else -1.0
        cal.bindings[channel] = (attr, sign)

        cal.step += 1
        cal.waiting_for_centre = True
        cal.step_started = now
        if cal.step >= len(_INSTRUCTIONS):
            bound = list(cal.bindings.values())
            if len({b[0] for b in bound}) < len(LOGICAL_CHANNELS):
                # Two channels landed on the same raw axis: the pilot answered twice with
                # the same stick, which cannot fly. Keep the measurement but say so.
                cal.message = "warning: two channels share an axis - repeat and move both sticks"
            for ch, (attr_name, sign_value) in cal.bindings.items():
                self.axes[ch] = AxisBinding(attr_name, sign_value)
            cal.active = False
            cal.done = True
            cal.instruction = "calibration complete"
            cal.message = "calibration applied (use 'Save calibration' to keep it)"
            return cal

        cal.instruction = _INSTRUCTIONS[cal.step][1]
        cal.message = f"step {cal.step + 1}/{len(_INSTRUCTIONS)}"
        return cal

    def calibration_report(self) -> str:
        if self.cal.failed:
            return f"calibration FAILED: {self.cal.failed}"
        if not self.cal.done:
            return "not calibrated"
        parts = [f"{ch}={self.axes[ch].attr}x{self.axes[ch].sign:+.0f}" for ch in LOGICAL_CHANNELS]
        return "calibrated: " + "  ".join(parts)
