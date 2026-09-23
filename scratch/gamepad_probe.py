# -*- coding: utf-8 -*-
"""
Live gamepad probe: show EXACTLY what the Mac reports for your controller.

Run this with the controller already connected (Bluetooth or USB), then move ONE stick
or press ONE button at a time:

    .venv/bin/python scratch/gamepad_probe.py            # 30 s live readout
    .venv/bin/python scratch/gamepad_probe.py --seconds 10

It prints, every half second:

  * the device name the HID layer matched,
  * every analog element with its HID usage, the kind the reader assigned (stick or
    trigger - decided by where the element RESTED) and its live normalized value,
  * the current sticks in the same left/right pairing the flight code uses,
  * every button usage currently held.

Use it to answer three questions:
  1. Is the controller seen at all?            -> "devices matched"
  2. Which way do the sticks read?             -> left_y is usually NEGATIVE when the
     left stick is pushed FORWARD (that is why PAD_INVERT_*_Y defaults to True in
     manual_flight.py). If your pad reads the other way, flip those flags.
  3. Are the shoulders/triggers where we think?-> buttons 5/6 should be LB/RB (L1/R1) and
     the two "trigger" elements should be LT/RT (L2/R2). Anything else: tell me the
     numbers and the mapping constants get updated.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gamepad import HIDGamepad, HAT_DOWN, HAT_LEFT, HAT_RIGHT, HAT_UP  # noqa: E402

_HAT_NAMES = {HAT_UP: "HAT_UP", HAT_RIGHT: "HAT_RIGHT", HAT_DOWN: "HAT_DOWN", HAT_LEFT: "HAT_LEFT"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to watch")
    parser.add_argument("--raw", action="store_true",
                        help="also hex-dump the raw vendor reports (use when nothing moves)")
    args = parser.parse_args()

    pad = HIDGamepad()
    if not pad.available:
        print(f"HID layer unavailable: {pad.error}")
        return 1
    pad.start()
    print(f"Listening for game controllers for {args.seconds:.0f} s - start wiggling sticks...")
    print("(nothing appears below? the pad is not matched as Game Pad / Joystick /")
    print(" Multi-Axis: check it is connected, and try a USB cable instead of Bluetooth)")

    last_buttons: set = set()
    t0 = time.time()
    seen_device = False
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(0.5)
            state = pad.state()
            with pad._lock:
                matched = bool(pad._device_names)
                axes = dict(pad._axes)
            if matched and not seen_device:
                seen_device = True
                print(f"\nDEVICE MATCHED: {state.name or next(iter(pad._device_names.values()))}")
                print("  element table (straight from the HID descriptor):")
                for line in pad.describe():
                    print("  " + line)
                if axes:
                    print("  axes the reader classified:")
                    for (page, usage), (_norm, kind, rest, lo, hi) in sorted(axes.items()):
                        print(f"    usage 0x{usage:02X}: {kind:14s} range=[{lo}, {hi}] rest={rest}")
                else:
                    print("  no standard HID axis values yet (report mode 0x3F not active).")
                    print("  Two things can be going on:")
                    print("    * a Nintendo-style pad only streams its vendor report until the host")
                    print("      asks for the standard one - this script now sends that request")
                    print("      automatically (watch for the 'handshake' line below), and")
                    print("    * a sleeping pad sends nothing at all: pick it up and press buttons.")
                print("  live values follow (sticks raw HID sign, triggers 0..1):\n")
            if not matched:
                continue
            pressed = {u for u in state.buttons if u not in last_buttons}
            released = {u for u in last_buttons if u not in state.buttons}
            last_buttons = set(state.buttons)
            line = (f"L({state.left_x:+.2f},{state.left_y:+.2f}) "
                    f"R({state.right_x:+.2f},{state.right_y:+.2f}) "
                    f"T({state.left_trigger:.2f},{state.right_trigger:.2f})"
                    f"{'  [no standard reports]' if not axes else ''}")
            if pad.note:
                line += f"   [{pad.note}]"
                pad.note = ""
            line += f"   (handshake attempts: {pad._handshake_sent}, vendor reports: {pad._vendor_hits})"
            if pressed:
                names = [f"b{u}" for u in sorted(pressed)]
                names += [_HAT_NAMES[u] for u in sorted(pressed) if u in _HAT_NAMES]
                line += "   PRESSED " + " ".join(names)
            if released:
                line += "   released " + " ".join(f"b{u}" for u in sorted(released))
            print(line)
            if args.raw:
                for raw_line in pad.raw_report_lines(limit=3):
                    print(f"      raw: {raw_line}")
    finally:
        pad.stop()
    print("\nprobe done" + ("" if seen_device else " - no controller was matched"))
    return 0 if seen_device else 1


if __name__ == "__main__":
    sys.exit(main())
