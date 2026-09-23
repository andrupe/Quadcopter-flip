# -*- coding: utf-8 -*-
"""
Headless checks for Simulation/gamepad.py - the macOS IOKit HID reader.

A real controller cannot be attached from a test script, so this file does two things:

  * It drives the HID *parsing* code with a stub IOKit (the same call pattern the real
    callbacks use, fed synthetic element/value tuples). That covers the parts that are
    easy to get wrong and impossible to see in the sim: rest-position classification
    (stick vs trigger), normalization, stick pairing for the Xbox AND Sony layouts, hat
    switch folding and button edges.
  * It exercises the real IOKit plumbing as far as this machine allows: the manager must
    start, enumerate (possibly zero controllers), report a usable state and stop cleanly.
    That is the part that proves the ctypes layer is wired correctly.

Run:  .venv/bin/python scratch/check_gamepad.py
"""
from __future__ import annotations

import os
import sys

import numpy as np  # noqa: F401  (kept for symmetry with the other check scripts)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gamepad import (  # noqa: E402
    HAT_DOWN,
    HAT_LEFT,
    HAT_RIGHT,
    HAT_UP,
    HIDGamepad,
    _AXIS_USAGES,
    _HID_PAGE_BUTTON,
    _HID_PAGE_GENERIC_DESKTOP,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


class _StubIOKit:
    """Mimics the handful of IOKit calls gamepad.py makes, fed (page, usage, lo, hi, raw).

    `value` and the element handle are the same tuple here, which is exactly how the real
    code uses them: the element supplies page/usage/range, the value supplies the reading.
    """

    def IOHIDValueGetElement(self, value):
        return value

    def IOHIDElementGetUsagePage(self, elem):
        return elem[0]

    def IOHIDElementGetUsage(self, elem):
        return elem[1]

    def IOHIDElementGetLogicalMin(self, elem):
        return elem[2]

    def IOHIDElementGetLogicalMax(self, elem):
        return elem[3]

    def IOHIDValueGetIntegerValue(self, value):
        return value[4]


def make_pad() -> HIDGamepad:
    """A HIDGamepad whose IOKit has been replaced by the stub (no hardware involved)."""
    pad = HIDGamepad()
    pad._iokit = _StubIOKit()
    pad._connected = True
    pad._name = "stub pad"
    return pad


def axis(pad: HIDGamepad, usage: int, raw: int, lo: int, hi: int) -> None:
    pad._on_value(None, 0, None, (_HID_PAGE_GENERIC_DESKTOP, usage, lo, hi, raw))


def button(pad: HIDGamepad, usage: int, down: bool) -> None:
    pad._on_value(None, 0, None, (_HID_PAGE_BUTTON, usage, 0, 1, 1 if down else 0))


print("=" * 78)
print("A. rest-position classification: sticks self-centre, triggers do not")
print("=" * 78)
pad = make_pad()
# Xbox-style 16-bit signed sticks resting at 0 = middle of the range.
for usage in _AXIS_USAGES:
    axis(pad, usage, 0, -32768, 32767)
with pad._lock:
    kinds = {u: v[1] for (p, u), v in pad._axes.items() if p == _HID_PAGE_GENERIC_DESKTOP}
check("signed axes resting at 0 classify as sticks",
      all(k.startswith("stick") for k in kinds.values()), f"{kinds}")

pad = make_pad()
axis(pad, 0x30, 0, 0, 1023)            # rests at the minimum -> trigger / slider
with pad._lock:
    kind = pad._axes[(_HID_PAGE_GENERIC_DESKTOP, 0x30)][1]
check("an axis resting at its minimum classifies as a trigger", kind == "trigger", kind)

pad = make_pad()
axis(pad, 0x31, -15000, -32768, 32767)      # seen mid-motion: neither centre nor an end
with pad._lock:
    kind = pad._axes[(_HID_PAGE_GENERIC_DESKTOP, 0x31)][1]
check("an axis first seen mid-motion is held as 'stick_offcenter', not misread as a trigger",
      kind == "stick_offcenter", kind)
axis(pad, 0x31, 0, -32768, 32767)           # later it centres
with pad._lock:
    kind = pad._axes[(_HID_PAGE_GENERIC_DESKTOP, 0x31)][1]
check("...and is upgraded to a confident stick once a sample lands near the middle",
      kind == "stick", kind)

print()
print("=" * 78)
print("B. normalization")
print("=" * 78)
pad = make_pad()
# A pad reports continuously, so by the time the pilot moves a stick the axis has already
# been seen at rest - seed the sticks at centre exactly like the hardware does.
for usage in (0x30, 0x31, 0x32, 0x35):
    axis(pad, usage, 128, 0, 255)
axis(pad, 0x30, -32768, -32768, 32767)
check("stick full left normalizes to -1", abs(pad.state().left_x + 1.0) < 1e-3,
      f"{pad.state().left_x:+.6f}")
axis(pad, 0x30, 32767, -32768, 32767)
check("stick full right normalizes to +1", abs(pad.state().left_x - 1.0) < 1e-3,
      f"{pad.state().left_x:+.6f}")
axis(pad, 0x30, 0, -32768, 32767)
check("stick centred normalizes to 0 (16-bit signed has no exact centre: ~1.5e-5)",
      abs(pad.state().left_x) < 1e-3, f"{pad.state().left_x:+.2e}")

pad = make_pad()
axis(pad, 0x32, 0, 0, 1023)
axis(pad, 0x35, 0, 0, 1023)
axis(pad, 0x30, 0, -32768, 32767)
axis(pad, 0x31, 0, -32768, 32767)
axis(pad, 0x32, 512, 0, 1023)
check("trigger halfway normalizes to 0.5", abs(pad.state().left_trigger - 0.5) < 0.01,
      f"{pad.state().left_trigger:.4f}")
axis(pad, 0x32, 0, 0, 1023)
check("released trigger normalizes to 0", abs(pad.state().left_trigger) < 1e-6)

print()
print("=" * 78)
print("C. stick pairing - the same code must read both common layouts")
print("=" * 78)
# Xbox Wireless / One S: X,Y = left stick, Rx,Ry = right stick, Z,Rz = triggers.
pad = make_pad()
for usage in (0x30, 0x31, 0x33, 0x34):
    axis(pad, usage, 0, -32768, 32767)
for usage in (0x32, 0x35):
    axis(pad, usage, 0, 0, 1023)
axis(pad, 0x33, 32767, -32768, 32767)      # right stick full right
axis(pad, 0x31, -32768, -32768, 32767)     # left stick full forward
axis(pad, 0x35, 1023, 0, 1023)             # right trigger squeezed
st = pad.state()
check("Xbox layout: left stick = X,Y", abs(st.left_y + 1.0) < 1e-3 and abs(st.left_x) < 1e-3,
      f"L=({st.left_x:+.2f},{st.left_y:+.2f})")
check("Xbox layout: right stick = Rx,Ry", abs(st.right_x - 1.0) < 1e-3,
      f"R=({st.right_x:+.2f},{st.right_y:+.2f})")
check("Xbox layout: triggers = Z,Rz", abs(st.right_trigger - 1.0) < 1e-3
      and abs(st.left_trigger) < 1e-3, f"T=({st.left_trigger:.2f},{st.right_trigger:.2f})")

# DualShock/DualSense: X,Y = left stick, Z,Rz = right stick, Rx,Ry = triggers.
pad = make_pad()
for usage in (0x30, 0x31, 0x32, 0x35):
    axis(pad, usage, 128, 0, 255)
for usage in (0x33, 0x34):
    axis(pad, usage, 0, 0, 255)
axis(pad, 0x32, 255, 0, 255)               # right stick full right
axis(pad, 0x31, 0, 0, 255)                 # left stick full forward (0 in unsigned range)
axis(pad, 0x34, 255, 0, 255)               # right trigger squeezed
st = pad.state()
check("Sony layout: left stick = X,Y", abs(st.left_y + 1.0) < 1e-3,
      f"L=({st.left_x:+.2f},{st.left_y:+.2f})")
check("Sony layout: right stick = Z,Rz", abs(st.right_x - 1.0) < 1e-3,
      f"R=({st.right_x:+.2f},{st.right_y:+.2f})")
check("Sony layout: triggers = Rx,Ry", abs(st.right_trigger - 1.0) < 1e-3
      and abs(st.left_trigger) < 1e-3, f"T=({st.left_trigger:.2f},{st.right_trigger:.2f})")

# Nintendo Switch Pro Controller, values taken from its real descriptor as read on this
# machine (gamepad.describe_devices): 4 stick axes 0..65535 resting near 32768, hat 0..7,
# 16 digital buttons, and NO analog triggers - ZL/ZR are buttons.
pad = make_pad()
for usage in (0x30, 0x31, 0x33, 0x34):
    axis(pad, usage, 32768, 0, 65535)
axis(pad, 0x30, 0, 0, 65535)               # left stick hard left
axis(pad, 0x31, 0, 0, 65535)               # left stick hard forward (Y = 0 at the top)
axis(pad, 0x33, 65535, 0, 65535)           # right stick hard right
st = pad.state()
check("Switch Pro: four stick axes pair as (X,Y) + (Rx,Ry)",
      st.left_y < -0.9 and st.right_x > 0.9, f"L=({st.left_x:+.2f},{st.left_y:+.2f}) "
      f"R=({st.right_x:+.2f},{st.right_y:+.2f})")
check("Switch Pro: no analog triggers are invented",
      st.analog_triggers == 0 and st.left_trigger == 0.0 and st.right_trigger == 0.0,
      f"analog_triggers={st.analog_triggers}")
pad._on_value(None, 0, None, (_HID_PAGE_GENERIC_DESKTOP, 0x39, 0, 7, 8))     # centred
st = pad.state()
check("Switch Pro: a neutral hat (8) presses no direction",
      not (st.buttons & {HAT_UP, HAT_RIGHT, HAT_DOWN, HAT_LEFT}), f"buttons={st.buttons}")

print()
print("=" * 78)
print("D. buttons and the hat switch")
print("=" * 78)
pad = make_pad()
axis(pad, 0x30, 0, -32768, 32767)
axis(pad, 0x31, 0, -32768, 32767)
button(pad, 1, True)
st = pad.state()
check("button press appears in buttons and as an edge", st.buttons == {1} and st.pressed == (1,),
      f"buttons={st.buttons} pressed={st.pressed}")
st = pad.state()
check("the edge is consumed by the next poll", st.pressed == () and st.buttons == {1},
      f"pressed={st.pressed}")
check("holding a button does not re-trigger the edge", pad.state().pressed == ())
button(pad, 1, False)
check("release clears the button", pad.state().buttons == set())

pad._on_value(None, 0, None, (_HID_PAGE_GENERIC_DESKTOP, 0x39, 0, 8, 0))     # hat up
st = pad.state()
check("hat 0 folds into HAT_UP", st.buttons == {HAT_UP} and st.pressed == (HAT_UP,),
      f"buttons={st.buttons}")
pad._on_value(None, 0, None, (_HID_PAGE_GENERIC_DESKTOP, 0x39, 0, 8, 4))     # hat down
st = pad.state()
check("hat 4 replaces the previous direction with HAT_DOWN",
      st.buttons == {HAT_DOWN} and st.pressed == (HAT_DOWN,), f"buttons={st.buttons}")
pad._on_value(None, 0, None, (_HID_PAGE_GENERIC_DESKTOP, 0x39, 0, 8, 5))     # hat down-left
st = pad.state()
check("diagonal hats report both directions", st.buttons == {HAT_DOWN, HAT_LEFT},
      f"buttons={sorted(st.buttons)}")
pad._on_value(None, 0, None, (_HID_PAGE_GENERIC_DESKTOP, 0x39, 0, 8, 8))     # centred
st = pad.state()
check("hat centred clears all four directions",
      not (st.buttons & {HAT_UP, HAT_RIGHT, HAT_DOWN, HAT_LEFT}), f"buttons={st.buttons}")

# Real bytes captured from the user's pad: it identifies as a Nintendo Switch Pro and
# streams the vendor "full" report forever (it never accepts the standard-report
# subcommand), so the reader decodes the format itself. With the sticks at rest every
# axis must land on 2048 = centre and no button may read as pressed.
_SWITCH_SAMPLES = (
    "25800000000008800008800c4cfcc800ae10dbffa800feff4bfcce00b610dbffa600fcff4bfccb00b110dcffa7000100",
    "30800000000008800008800c4dfcc900be10dcffa50001004cfcc800ae10dbffa800feff4bfcce00b610dbffa600fcff",
    "3c800000000008800008800c4dfcc700af10daffa800fbff4dfcc900be10dcffa50001004cfcc800ae10dbffa800feff",
)
pad = make_pad()
for hex_blob in _SWITCH_SAMPLES:
    check("switch decoder accepts a real 48-byte report",
          pad._decode_switch_report(bytes.fromhex(hex_blob)))
st = pad.state()
check("switch decoder: sticks at rest decode to centre on both sticks",
      abs(st.left_x) < 0.01 and abs(st.left_y) < 0.01 and abs(st.right_x) < 0.01
      and abs(st.right_y) < 0.01,
      f"L=({st.left_x:+.3f},{st.left_y:+.3f}) R=({st.right_x:+.3f},{st.right_y:+.3f})")
check("switch decoder: no phantom buttons from an idle report", not st.buttons,
      f"buttons={sorted(st.buttons)}")
check("switch decoder: pad reports no analog triggers (ZL/ZR are buttons)",
      st.analog_triggers == 0, f"analog_triggers={st.analog_triggers}")

# Synthetic report: A+B pressed, D-pad up, left stick hard right + hard forward,
# right stick hard back. Only the encoding rule is exercised here, not the pad.
blob = bytearray(48)
blob[1] = 0x80
blob[2] = 0x08 | 0x04 | 0x40 | 0x80          # A, B, R, ZR
blob[3] = 0x02                                # plus
blob[4] = 0x02 | 0x40 | 0x80                  # D-pad up, L, ZL
blob[5] = 0xFF; blob[6] = 0x0F                # left X = 4095 (right), left Y = 0 (forward)
blob[9] = 0xF0; blob[10] = 0xFF               # right Y = 4095 (back)
pad = make_pad()
pad._decode_switch_report(bytes(blob))
st = pad.state()
check("switch decoder: full-deflection sticks decode with the raw HID sign",
      abs(st.left_x - 1.0) < 0.01 and abs(st.left_y + 1.0) < 0.01 and abs(st.right_y - 1.0) < 0.01,
      f"L=({st.left_x:+.2f},{st.left_y:+.2f}) R=({st.right_x:+.2f},{st.right_y:+.2f})")
check("switch decoder: A/B land on usages 2/1 and shoulders on 5/6, triggers on 7/8",
      {1, 2, 5, 6, 7, 8, 10, HAT_UP} <= st.buttons, f"buttons={sorted(st.buttons)}")
check("switch decoder: button edges are reported once",
      {1, 2, HAT_UP} <= set(st.pressed), f"pressed={st.pressed}")
check("switch decoder: a short blob is rejected instead of misread",
      not pad._decode_switch_report(b"\x30\x80\x00"))

print()
print("=" * 78)
print("E. the real IOKit plumbing (no controller required)")
print("=" * 78)
pad = HIDGamepad()
check("IOKit/CoreFoundation loaded", pad.available, pad.error or "")
if pad.available:
    pad.start()
    import time
    time.sleep(0.6)
    st = pad.state()
    check("manager starts and reports a state without hardware", isinstance(st.connected, bool),
          f"connected={st.connected} name={st.name!r} error={pad.error or 'none'}")
    check("matching is game-controller only (no keyboard/trackpad prompt on macOS)",
          "keyboard" not in st.name.lower() and "trackpad" not in st.name.lower(),
          f"matched name={st.name!r}")
    check("an idle pad reads centred, not deflected", not st.any_stick_deflected(),
          f"L=({st.left_x:+.2f},{st.left_y:+.2f}) R=({st.right_x:+.2f},{st.right_y:+.2f})")
    pad.stop()
    check("reader thread stops cleanly", pad._thread is None or not pad._thread.is_alive())

print()
print("=" * 78)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
