# -*- coding: utf-8 -*-
"""
Diagnose a matched-but-silent game controller on macOS (IOKit HID).

`gamepad_probe.py` shows "no reports" when a controller is matched by name but no input
values ever arrive. That has several very different causes, and this script tells them
apart:

  1. the device is never opened       -> IOHIDDeviceOpen returns a non-zero IOReturn
  2. the device is opened by someone   -> kIOReturnExclusiveAccess (0xE00002C5); try
     else exclusively                     seizing it (the script does that automatically)
  3. the device is open but ASLEEP     -> it matches, GetValue returns 0 for every axis
     (Bluetooth link idle)                and nothing changes no matter what you press
  4. the device is sending, but the    -> the per-device value-callback counter stays 0
     manager-level callback never fires  while GetValue values change
  5. everything works                  -> values move and the counter climbs

HOW TO RUN IT - this needs your hands:

    .venv/bin/python scratch/gamepad_diagnose.py --seconds 30

While it runs, do this in order and watch the numbers:
  * press a face button a few times,
  * move the LEFT stick in circles, then the RIGHT stick,
  * press every shoulder/trigger button once.

Read the verdict at the end. "device is SILENT" means the Mac is not receiving input
reports at all: unplug/replug or re-pair the controller (Bluetooth settings), make sure
it is not still connected to a console, then run this again.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gamepad import (  # noqa: E402
    HIDGamepad,
    _AXIS_USAGES,
    _HID_PAGE_BUTTON,
    _HID_PAGE_GENERIC_DESKTOP,
    cf_value_to_str,
)

_kIOHIDOptionsTypeNone = 0
_kIOHIDOptionsTypeSeizeDevice = 1

_DEVICE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_void_p)
_VALUE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_void_p)

_PROPERTIES = ("Product", "Manufacturer", "Transport", "VendorID", "ProductID",
               "VersionNumber", "LocationID", "PrimaryUsagePage", "PrimaryUsage")


class Diagnostic:
    def __init__(self) -> None:
        self.pad = HIDGamepad()
        self.error = ""
        if not self.pad.available:
            return
        self.iokit = self.pad._iokit
        self.cf = self.pad._cf
        self.iokit.IOHIDDeviceOpen.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.iokit.IOHIDDeviceOpen.restype = ctypes.c_int
        self.iokit.IOHIDDeviceClose.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.iokit.IOHIDDeviceRegisterInputValueCallback.argtypes = [ctypes.c_void_p,
                                                                     ctypes.c_void_p,
                                                                     ctypes.c_void_p]
        self.iokit.IOHIDDeviceScheduleWithRunLoop.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                              ctypes.c_void_p]

        self.lock = threading.Lock()
        self.devices: Dict[int, str] = {}
        self.opened: Dict[int, int] = {}          # device ptr -> IOReturn
        self.axis_elements: Dict[Tuple[int, int], int] = {}   # (dev, usage) -> element ref
        self.callback_hits = 0
        self.thread: Optional[threading.Thread] = None
        self.runloop = None

        self.cb_match = _DEVICE_CALLBACK(self._on_match)
        self.cb_remove = _DEVICE_CALLBACK(lambda *a: None)
        self.cb_value = _VALUE_CALLBACK(self._on_value)

    # -- helpers ------------------------------------------------------------------------
    def cfstr(self, text: str):
        return self.pad._make_cfstring(text)

    def prop(self, device, name: str) -> str:
        """Read a device property as text. Type-checked: CFStringGetCString on a CFNumber
        raises an Objective-C exception that kills the process outright."""
        try:
            value = self.iokit.IOHIDDeviceGetProperty(device, self.cfstr(name))
            return cf_value_to_str(self.cf, value)
        except Exception as exc:
            return f"(err {exc})"

    def read(self, device, element) -> Optional[int]:
        try:
            ref = ctypes.c_void_p(0)
            rc = self.iokit.IOHIDDeviceGetValue(ctypes.c_void_p(device), element,
                                                ctypes.byref(ref))
            if rc == 0 and ref:
                return int(self.iokit.IOHIDValueGetIntegerValue(ref))
        except Exception:
            pass
        return None

    # -- HID callbacks (reader thread) ---------------------------------------------------
    def _on_value(self, context, result, sender, value) -> None:
        with self.lock:
            self.callback_hits += 1
        if self.callback_hits <= 5:
            try:
                elem = self.iokit.IOHIDValueGetElement(value)
                page = int(self.iokit.IOHIDElementGetUsagePage(elem))
                usage = int(self.iokit.IOHIDElementGetUsage(elem))
                print(f"    [device callback #{self.callback_hits}] page=0x{page:04X} "
                      f"usage=0x{usage:02X} raw={int(self.iokit.IOHIDValueGetIntegerValue(value))}")
            except Exception:
                pass

    def _on_match(self, context, result, sender, device) -> None:
        try:
            name = self.prop(device, "Product")
            with self.lock:
                self.devices[int(device)] = name
            print(f"\n[matched] {name}")
            for key in _PROPERTIES:
                print(f"    {key:18s} = {self.prop(device, key)}")

            rc = self.iokit.IOHIDDeviceOpen(device, _kIOHIDOptionsTypeNone)
            how = "shared"
            if rc != 0:
                rc2 = self.iokit.IOHIDDeviceOpen(device, _kIOHIDOptionsTypeSeizeDevice)
                how = f"seized (shared open failed {rc:#x})"
                rc = rc2
            with self.lock:
                self.opened[int(device)] = rc
            print(f"    IOHIDDeviceOpen  = {rc:#x} ({'success' if rc == 0 else 'FAILED'}) "
                  f"[{how}]")
            if rc != 0:
                if rc in (-0x1ffffd3b, 0xE00002C5 - 0x100000000):
                    print("    -> kIOReturnExclusiveAccess: another process has the device")
                elif rc in (-0x1ffffd1e, 0xE00002E2 - 0x100000000):
                    print("    -> kIOReturnNotPermitted: macOS denied access (permissions)")

            # Per-device callbacks + scheduling: the manager-level callback should already
            # cover this, but a per-device registration is the belt-and-braces path and
            # the diagnostic is exactly where we want to know which one works.
            try:
                self.iokit.IOHIDDeviceRegisterInputValueCallback(
                    device, ctypes.cast(self.cb_value, ctypes.c_void_p), None)
                self.iokit.IOHIDDeviceScheduleWithRunLoop(device, self.runloop,
                                                          self.pad._mode)
                print("    per-device value callback registered")
            except Exception as exc:
                print(f"    per-device registration failed: {exc}")

            # Cache the axis elements so the main thread can poll them.
            array = self.iokit.IOHIDDeviceCopyMatchingElements(device, None, 0)
            if array:
                count = int(self.cf.CFArrayGetCount(ctypes.c_void_p(array)))
                found = []
                for i in range(count):
                    elem = self.cf.CFArrayGetValueAtIndex(ctypes.c_void_p(array), i)
                    page = int(self.iokit.IOHIDElementGetUsagePage(elem))
                    usage = int(self.iokit.IOHIDElementGetUsage(elem))
                    if page == _HID_PAGE_GENERIC_DESKTOP and usage in _AXIS_USAGES:
                        with self.lock:
                            self.axis_elements[(int(device), usage)] = elem
                        found.append(f"0x{usage:02X} (now {self.read(device, elem)})")
                print(f"    axis elements cached: {', '.join(found) if found else 'none'}")
        except Exception as exc:
            print(f"[matched] handler failed: {exc}")

    # -- lifecycle ----------------------------------------------------------------------
    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="hid-diagnose", daemon=True)
        self.thread.start()
        time.sleep(0.8)

    def _run(self) -> None:
        cf, iokit = self.cf, self.iokit
        self.runloop = cf.CFRunLoopGetCurrent()
        manager = iokit.IOHIDManagerCreate(None, 0)
        iokit.IOHIDManagerSetDeviceMatchingMultiple(manager, self.pad._build_matching())
        iokit.IOHIDManagerRegisterDeviceMatchingCallback(
            manager, ctypes.cast(self.cb_match, ctypes.c_void_p), None)
        iokit.IOHIDManagerRegisterInputValueCallback(
            manager, ctypes.cast(self.cb_value, ctypes.c_void_p), None)
        iokit.IOHIDManagerScheduleWithRunLoop(manager, self.runloop, self.pad._mode)
        rc = iokit.IOHIDManagerOpen(manager, 0)
        print(f"IOHIDManagerOpen -> {rc:#x}")
        cf.CFRunLoopRun()

    def stop(self) -> None:
        if self.runloop:
            try:
                self.cf.CFRunLoopStop(self.runloop)
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to watch")
    args = parser.parse_args()

    diag = Diagnostic()
    if not diag.pad.available:
        print(f"HID layer unavailable: {diag.pad.error}")
        return 1

    print(__doc__.split("HOW TO RUN IT")[0].strip().splitlines()[-1])
    diag.start()
    print(f"\nWatching for {args.seconds:.0f} s. Press buttons, then move each stick.")

    t0 = time.time()
    moved: Dict[int, Tuple[int, int]] = {}
    samples = 0
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(0.5)
            samples += 1
            with diag.lock:
                devices = dict(diag.devices)
                elements = dict(diag.axis_elements)
                hits = diag.callback_hits
                opened = dict(diag.opened)
            if not devices:
                continue
            bits = []
            for (dev, usage), elem in sorted(elements.items()):
                value = diag.read(dev, elem)
                if value is None:
                    continue
                lo, hi = moved.get(usage, (value, value))
                moved[usage] = (min(lo, value), max(hi, value))
                bits.append(f"0x{usage:02X}={value}")
            changed = " ".join(f"0x{u:02X}[{lo}..{hi}]"
                               for u, (lo, hi) in sorted(moved.items()) if hi != lo)
            print(f"  t={time.time() - t0:5.1f}s  callbacks={hits:5d}  "
                  f"values({' '.join(bits) if bits else 'none'})"
                  f"{'  moved: ' + changed if changed else ''}")
    finally:
        diag.stop()

    print("\n" + "=" * 78)
    with diag.lock:
        hits = diag.callback_hits
        opened = dict(diag.opened)
    if not opened:
        print("VERDICT: no controller was matched at all -> it is not connected to this Mac.")
        return 1
    if any(rc != 0 for rc in opened.values()):
        print("VERDICT: the controller could not be OPENED (see IOReturn above). Another")
        print("         process is holding it, or macOS denied access.")
        return 1
    if hits == 0:
        print("VERDICT: device is SILENT - it is open, but macOS is receiving no input")
        print("         reports from it. Re-pair / re-plug the controller, make sure it is")
        print("         awake (press a button) and not still talking to a console.")
        return 1
    if not any(hi != lo for lo, hi in moved.values()):
        print(f"VERDICT: callbacks ARE arriving ({hits}), but no axis ever moved.")
        print("         You probably did not touch the sticks - run it again and move them.")
        return 1
    print(f"VERDICT: WORKING - {hits} value callbacks and the axes moved.")
    print("         If gamepad_probe.py still shows nothing, the fault is in its callback")
    print("         path, not in the controller.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
