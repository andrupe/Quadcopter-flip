# -*- coding: utf-8 -*-
"""
Minimal HID gamepad reader for macOS, built on ctypes + IOKit.

WHY THIS EXISTS (and why it is not a pip package)
-------------------------------------------------
`manual_flight.py` runs under mjpython, which means the *main* thread belongs to the
Cocoa/GUI runtime and the flight loop runs on a secondary thread. Any input library that
wants to own the main thread (pygame/SDL event pumps, Tk, most GUI toolkits) is therefore
not usable, and the MuJoCo viewer only reports key PRESSES (see manual_flight.py). IOKit's
IOHIDManager has no such constraint: it is a plain C API that can be scheduled on a
run loop owned by whatever thread you like, so this module runs a dedicated reader thread
and the flight loop just reads a snapshot.

WHAT IT DOES
------------
* Matches HID devices whose top-level collection is Game Pad / Joystick / Multi-Axis
  Controller (the standard shapes for Xbox, DualShock/DualSense, Switch Pro, 8BitDo, ...).
* Normalizes every analog element to a hardware-neutral form:
      sticks   -> [-1, +1]   (raw HID sign; most pads report stick-forward as NEGATIVE,
                              the flight code applies its own invert flags)
      triggers -> [ 0, +1]
  The element's "kind" is learned from where it RESTED when first seen: an axis parked at
  the middle of its logical range is a self-centering stick, one parked at the minimum is
  a trigger. That is what makes the same code work for the Xbox layout (X,Y | Rx,Ry with
  Z,Rz as triggers) and the Sony layout (X,Y | Z,Rz with Rx,Ry as triggers) without a
  per-device table.
* Exposes buttons by HID usage, plus a hat switch folded into four synthetic buttons.
* Tracks device connect / disconnect so the sim can say what it found.

Public API:
    pad = HIDGamepad(); pad.start()
    state = pad.state()          # PadState snapshot, thread-safe
    pad.stop()

    devices = probe_devices(seconds=2.0)   # diagnostics: what HID sees, see gamepad_probe
"""
from __future__ import annotations

import ctypes
import ctypes.util
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

__all__ = ["PadState", "HIDGamepad", "HAT_UP", "HAT_RIGHT", "HAT_DOWN", "HAT_LEFT"]

# --- HID constants --------------------------------------------------------------------
_HID_PAGE_GENERIC_DESKTOP = 0x01
_HID_PAGE_BUTTON = 0x09
_HID_USAGE_JOYSTICK = 0x04
_HID_USAGE_GAMEPAD = 0x05
_HID_USAGE_MULTIAXIS = 0x08
_HID_USAGE_HAT = 0x39

# Analog usages, in the order that pairs them (X,Y) then (Z,Rz) then (Rx,Ry).
_AXIS_USAGES: Tuple[int, ...] = (0x30, 0x31, 0x32, 0x33, 0x34, 0x35)

# Synthetic "button" usages for the hat switch (0x39 reports one integer 0..8, not bits).
HAT_UP = 0x90
HAT_RIGHT = 0x91
HAT_DOWN = 0x92
HAT_LEFT = 0x93

# IOHIDElementType values, for the diagnostics table only.
_ELEMENT_TYPES: Dict[int, str] = {
    1: "Input_Misc",
    2: "Input_Button",
    3: "Input_Axis",
    4: "Input_ScanCodes",
    129: "Output",
    257: "Feature",
    513: "Collection",
}
_HAT_BITS: Dict[int, Tuple[int, ...]] = {
    0: (HAT_UP,),
    1: (HAT_UP, HAT_RIGHT),
    2: (HAT_RIGHT,),
    3: (HAT_DOWN, HAT_RIGHT),
    4: (HAT_DOWN,),
    5: (HAT_DOWN, HAT_LEFT),
    6: (HAT_LEFT,),
    7: (HAT_UP, HAT_LEFT),
}

# Fraction of the logical span that decides "parked at the middle" vs "parked at the end".
_REST_TOLERANCE = 0.20

# --- Nintendo Switch Pro / Joy-Con ----------------------------------------------------
# These pads advertise a standard gamepad collection (16-bit axes, buttons, hat) but do
# NOT populate it until the host asks for the "simple HID" input report: out of the box
# they stream their vendor report (page 0xFF01) only, which is why macOS's own HID layer
# shows their sticks as a constant 0. macOS itself does not send the request unless the
# GameController framework is driving the pad (and on this machine it does not see it).
# The handshake below is the documented Nintendo subcommand:
#   output report 1, packet counter, 8 bytes rumble, 0x03 (SET_INPUT_REPORT_MODE), 0x3F
# Two padding layouts circulate in the wild (with and without the report ID inside the
# 48-byte buffer), so the reader alternates until real axis values arrive.
_SWITCH_BT_REPORT_SIZE = 48
_SWITCH_CMD_SET_INPUT_REPORT_MODE = 0x03
_SWITCH_MODE_SIMPLE_HID = 0x3F
_VENDOR_PAGE = 0xFF01
# Some pads never switch report mode (Nintendo clones in particular) and simply stream the
# vendor "full" report forever, so the reader decodes that format as well. The layout is
# the one from Linux hid-nintendo.c with the report ID stripped by macOS:
#   0 timer | 1 battery | 2,3,4 buttons | 5..7 left stick | 8..10 right stick | 11 vibrator
#   12..47 three IMU frames
_SWITCH_FULL_REPORT_USAGE = 0x30
_SWITCH_STICK_CENTRE = 2048          # 12-bit sticks, 0..4095

# CFNumberType / CFStringEncoding
_kCFNumberSInt32Type = 3
_kCFStringEncodingUTF8 = 0x08000100

# IOHID device-matching dictionary keys. Modern macOS does NOT export the kIOHID*Key
# symbols as data from IOKit (dlsym fails), but the headers define them as plain string
# literals, and CFString/CFDictionary compare by CONTENT, so building our own CFStrings
# from the same literals is equivalent. Values taken from IOHIDDeviceKeys.h.
_KEY_DEVICE_USAGE_PAGE = "DeviceUsagePage"
_KEY_DEVICE_USAGE = "DeviceUsage"
_KEY_PRODUCT = "Product"
_KEY_RUN_LOOP_DEFAULT_MODE = "kCFRunLoopDefaultMode"


@dataclass
class PadState:
    """Snapshot of a HID gamepad, hardware-neutral and thread-safe to read."""

    connected: bool = False
    name: str = ""
    # Sticks in raw HID sign: -1 .. +1 per axis (forward is usually NEGATIVE on Y).
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    # Analog triggers: 0 (released) .. 1 (fully pressed).
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    # How many real analog trigger axes the pad exposes. Pads with digital triggers only
    # (Nintendo Switch Pro, most 8BitDo pads in Switch mode) report 0 here, which is the
    # signal the flight code uses to fall back to trigger BUTTONS for the altitude channel.
    analog_triggers: int = 0
    # Buttons currently held (HID usages on page 0x09, plus the HAT_* codes).
    buttons: Set[int] = field(default_factory=set)
    # Buttons that went down since the previous state() call (edge, for one-shot actions).
    pressed: Tuple[int, ...] = ()
    # Raw normalized axes keyed by HID usage - diagnostics only (see gamepad_probe).
    raw_axes: Dict[int, float] = field(default_factory=dict)

    def any_stick_deflected(self, deadzone: float = 0.0) -> bool:
        return (abs(self.left_x) > deadzone or abs(self.left_y) > deadzone
                or abs(self.right_x) > deadzone or abs(self.right_y) > deadzone)


# --- ctypes plumbing ------------------------------------------------------------------
def _cfunctions() -> Tuple[ctypes.CDLL, ctypes.CDLL]:
    """Load CoreFoundation and IOKit and declare the signatures used below."""
    cf_path = ctypes.util.find_library("CoreFoundation") or \
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    iokit_path = ctypes.util.find_library("IOKit") or \
        "/System/Library/Frameworks/IOKit.framework/IOKit"
    cf = ctypes.CDLL(cf_path)
    iokit = ctypes.CDLL(iokit_path)

    cf.CFRunLoopGetCurrent.restype = ctypes.c_void_p
    cf.CFRunLoopRun.restype = None
    cf.CFRunLoopStop.argtypes = [ctypes.c_void_p]
    cf.CFRunLoopStop.restype = None
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFGetTypeID.restype = ctypes.c_ulong
    cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
    cf.CFStringGetTypeID.restype = ctypes.c_ulong
    cf.CFStringGetTypeID.argtypes = []
    cf.CFNumberGetTypeID.restype = ctypes.c_ulong
    cf.CFNumberGetTypeID.argtypes = []
    cf.CFNumberGetValue.restype = ctypes.c_bool
    cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    cf.CFNumberCreate.restype = ctypes.c_void_p
    cf.CFNumberCreate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    cf.CFDictionaryCreateMutable.restype = ctypes.c_void_p
    cf.CFDictionaryCreateMutable.argtypes = [ctypes.c_void_p, ctypes.c_long,
                                             ctypes.c_void_p, ctypes.c_void_p]
    cf.CFDictionarySetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    cf.CFArrayCreate.restype = ctypes.c_void_p
    cf.CFArrayCreate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                 ctypes.c_long, ctypes.c_void_p]

    iokit.IOHIDManagerCreate.restype = ctypes.c_void_p
    iokit.IOHIDManagerCreate.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    iokit.IOHIDManagerSetDeviceMatchingMultiple.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    iokit.IOHIDManagerRegisterDeviceMatchingCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                                 ctypes.c_void_p]
    iokit.IOHIDManagerRegisterDeviceRemovalCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                                ctypes.c_void_p]
    iokit.IOHIDManagerRegisterInputValueCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                             ctypes.c_void_p]
    iokit.IOHIDManagerScheduleWithRunLoop.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                      ctypes.c_void_p]
    iokit.IOHIDManagerUnscheduleFromRunLoop.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_void_p]
    iokit.IOHIDManagerOpen.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    iokit.IOHIDManagerOpen.restype = ctypes.c_int
    iokit.IOHIDManagerClose.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    iokit.IOHIDDeviceGetProperty.restype = ctypes.c_void_p
    iokit.IOHIDDeviceGetProperty.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    iokit.IOHIDDeviceOpen.restype = ctypes.c_int
    iokit.IOHIDDeviceOpen.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    iokit.IOHIDDeviceRegisterInputValueCallback.argtypes = [ctypes.c_void_p,
                                                            ctypes.c_void_p,
                                                            ctypes.c_void_p]
    iokit.IOHIDDeviceScheduleWithRunLoop.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p]
    iokit.IOHIDDeviceSetReport.restype = ctypes.c_int
    iokit.IOHIDDeviceSetReport.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long,
                                           ctypes.POINTER(ctypes.c_ubyte), ctypes.c_long]
    iokit.IOHIDDeviceCopyMatchingElements.restype = ctypes.c_void_p
    iokit.IOHIDDeviceCopyMatchingElements.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                      ctypes.c_uint32]
    iokit.IOHIDDeviceGetValue.restype = ctypes.c_int
    iokit.IOHIDDeviceGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_void_p)]
    iokit.IOHIDElementGetType.restype = ctypes.c_uint32
    iokit.IOHIDElementGetType.argtypes = [ctypes.c_void_p]

    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    cf.CFRelease.argtypes = [ctypes.c_void_p]

    iokit.IOHIDValueGetElement.restype = ctypes.c_void_p
    iokit.IOHIDValueGetElement.argtypes = [ctypes.c_void_p]
    iokit.IOHIDValueGetIntegerValue.restype = ctypes.c_long
    iokit.IOHIDValueGetIntegerValue.argtypes = [ctypes.c_void_p]
    iokit.IOHIDValueGetLength.restype = ctypes.c_long
    iokit.IOHIDValueGetLength.argtypes = [ctypes.c_void_p]
    iokit.IOHIDValueGetBytePtr.restype = ctypes.POINTER(ctypes.c_ubyte)
    iokit.IOHIDValueGetBytePtr.argtypes = [ctypes.c_void_p]
    iokit.IOHIDElementGetUsage.restype = ctypes.c_uint32
    iokit.IOHIDElementGetUsage.argtypes = [ctypes.c_void_p]
    iokit.IOHIDElementGetUsagePage.restype = ctypes.c_uint32
    iokit.IOHIDElementGetUsagePage.argtypes = [ctypes.c_void_p]
    iokit.IOHIDElementGetLogicalMin.restype = ctypes.c_long
    iokit.IOHIDElementGetLogicalMin.argtypes = [ctypes.c_void_p]
    iokit.IOHIDElementGetLogicalMax.restype = ctypes.c_long
    iokit.IOHIDElementGetLogicalMax.argtypes = [ctypes.c_void_p]

    return cf, iokit


_DEVICE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_void_p)
_VALUE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_void_p)


def cf_value_to_str(cf: ctypes.CDLL, value) -> str:
    """Render a CF property as text WITHOUT asking for the wrong type.

    CFStringGetCString on a CFNumber (VendorID, ProductID, VersionNumber, ...) raises an
    Objective-C exception that terminates the process - there is no Python-level catch for
    it - so the type must be checked first.
    """
    if not value:
        return "-"
    try:
        type_id = int(cf.CFGetTypeID(ctypes.c_void_p(value)))
        if type_id == int(cf.CFStringGetTypeID()):
            buf = ctypes.create_string_buffer(512)
            if cf.CFStringGetCString(ctypes.c_void_p(value), buf, 512, _kCFStringEncodingUTF8):
                return buf.value.decode("utf-8", "replace")
            return "(unreadable string)"
        if type_id == int(cf.CFNumberGetTypeID()):
            num = ctypes.c_int64(0)
            if cf.CFNumberGetValue(ctypes.c_void_p(value), 4, ctypes.byref(num)):  # SInt64
                return str(int(num.value))
            return "(unreadable number)"
        return f"(CF type {type_id})"
    except Exception as exc:
        return f"(err {exc})"


class HIDGamepad:
    """A background-thread IOHIDManager reader for Game Pad / Joystick HID devices.

    Build it, call start(), then poll state(). Construction never raises on a supported
    platform; if IOKit cannot be reached the object simply reports `connected = False`
    (the flight loop then falls back to the keyboard).
    """

    def __init__(self, match_all_generic_desktop: bool = False):
        self.error: str = ""
        self._cf: Optional[ctypes.CDLL] = None
        self._iokit: Optional[ctypes.CDLL] = None
        self._manager = None
        self._runloop = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._match_all = bool(match_all_generic_desktop)

        self._lock = threading.Lock()
        self._name = ""
        self._device_names: Dict[int, str] = {}          # device ptr -> product name
        self._device_refs: Dict[int, int] = {}           # device ptr -> device ref
        self._connected = False
        # (page, usage) -> normalized value, kind, rest value, logical range
        self._axes: Dict[Tuple[int, int], Tuple[float, str, int, int, int]] = {}
        self._buttons: Dict[int, bool] = {}
        self._edges: List[int] = []
        self._cf_keepalive: List[object] = []            # CF objects the manager must keep
        # Nintendo style wake-up / report-mode handshake bookkeeping (see _maybe_handshake)
        self._vendor_hits: int = 0
        self._handshake_sent: int = 0
        self._last_handshake: float = 0.0
        self.note: str = ""
        # Raw byte samples of vendor reports (diagnostics: lets a pad whose subcommands
        # are ignored be decoded from its own report format instead).
        self.raw_samples: List[Tuple[int, bytes]] = []

        try:
            self._cf, self._iokit = _cfunctions()
            self._usage_page_key = self._make_cfstring(_KEY_DEVICE_USAGE_PAGE)
            self._usage_key = self._make_cfstring(_KEY_DEVICE_USAGE)
            self._product_key = self._make_cfstring(_KEY_PRODUCT)
            self._mode = self._make_cfstring(_KEY_RUN_LOOP_DEFAULT_MODE)
        except Exception as exc:      # pragma: no cover - only on a broken/other platform
            self.error = f"IOKit/CoreFoundation unavailable: {exc}"

        # ctypes callbacks must outlive the calls that register them.
        self._cb_match = _DEVICE_CALLBACK(self._on_device_matched)
        self._cb_remove = _DEVICE_CALLBACK(self._on_device_removed)
        self._cb_value = _VALUE_CALLBACK(self._on_value)

    def _make_cfstring(self, text: str):
        """Create a CFStringRef and keep it alive for the lifetime of the manager."""
        assert self._cf is not None, "CoreFoundation not loaded"
        ref = self._cf.CFStringCreateWithCString(None, text.encode("utf-8"),
                                                 _kCFStringEncodingUTF8)
        if not ref:
            raise RuntimeError(f"CFStringCreateWithCString({text!r}) failed")
        self._cf_keepalive.append(ref)
        return ref

    # -- lifecycle ---------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._iokit is not None

    def start(self) -> None:
        if not self.available or (self._thread is not None and self._thread.is_alive()):
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, name="hid-gamepad", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stopping.set()
        if self._runloop:
            try:
                self._cf.CFRunLoopStop(self._runloop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:                       # the reader thread
        try:
            self._run_reader()
        except Exception as exc:                  # never die silently: state() reports it
            self.error = f"gamepad reader thread failed: {exc}"
            with self._lock:
                self._connected = False

    def _run_reader(self) -> None:
        cf, iokit = self._cf, self._iokit
        assert cf is not None and iokit is not None, "frameworks not loaded"
        self._runloop = cf.CFRunLoopGetCurrent()
        manager = iokit.IOHIDManagerCreate(None, 0)
        if not manager:
            raise RuntimeError("IOHIDManagerCreate returned NULL")
        self._manager = manager
        iokit.IOHIDManagerSetDeviceMatchingMultiple(manager, self._build_matching())
        iokit.IOHIDManagerRegisterDeviceMatchingCallback(
            manager, ctypes.cast(self._cb_match, ctypes.c_void_p), None)
        iokit.IOHIDManagerRegisterDeviceRemovalCallback(
            manager, ctypes.cast(self._cb_remove, ctypes.c_void_p), None)
        iokit.IOHIDManagerRegisterInputValueCallback(
            manager, ctypes.cast(self._cb_value, ctypes.c_void_p), None)
        iokit.IOHIDManagerScheduleWithRunLoop(manager, self._runloop, self._mode)
        rc = iokit.IOHIDManagerOpen(manager, 0)
        if rc != 0:
            self.error = f"IOHIDManagerOpen failed (IOReturn {rc:#x})"
        cf.CFRunLoopRun()                          # blocks until stop()
        try:
            iokit.IOHIDManagerUnscheduleFromRunLoop(manager, self._runloop, self._mode)
            iokit.IOHIDManagerClose(manager, 0)
        except Exception:
            pass

    def _build_matching(self):
        """CFArray of CFDictionary device matchings: one per game-controller usage.

        Only Game Pad / Joystick / Multi-Axis are matched by default. Matching the whole
        Generic Desktop page would also pull in the built-in keyboard and trackpad, which
        on current macOS raises an Input Monitoring permission prompt - not something a
        flight sim should do just to look for a pad.
        """
        cf = self._cf
        assert cf is not None, "CoreFoundation not loaded"
        usages = [_HID_USAGE_GAMEPAD, _HID_USAGE_JOYSTICK, _HID_USAGE_MULTIAXIS]
        if self._match_all:
            usages.insert(0, 0)          # page-only entry: every Generic Desktop device
        dicts = []
        for usage in usages:
            d = cf.CFDictionaryCreateMutable(None, 2, None, None)
            page_num = ctypes.c_int32(_HID_PAGE_GENERIC_DESKTOP)
            page_ref = cf.CFNumberCreate(None, _kCFNumberSInt32Type, ctypes.byref(page_num))
            cf.CFDictionarySetValue(d, self._usage_page_key, page_ref)
            if usage:
                usage_num = ctypes.c_int32(usage)
                usage_ref = cf.CFNumberCreate(None, _kCFNumberSInt32Type, ctypes.byref(usage_num))
                cf.CFDictionarySetValue(d, self._usage_key, usage_ref)
                self._cf_keepalive.extend((usage_ref,))
            self._cf_keepalive.extend((page_ref, d))
            dicts.append(d)
        ptrs = (ctypes.c_void_p * len(dicts))(*[ctypes.c_void_p(d) for d in dicts])
        array = cf.CFArrayCreate(None, ptrs, len(dicts), None)
        self._cf_keepalive.append(array)
        return array

    # -- HID callbacks (reader thread) --------------------------------------------------
    def _on_device_matched(self, context, result, sender, device) -> None:
        name = "(unknown)"
        try:
            if device:
                # Type-checked read: VendorID/ProductID are CFNumbers, and asking a
                # number for its C string aborts the process.
                prop = self._iokit.IOHIDDeviceGetProperty(device, self._product_key)
                name = cf_value_to_str(self._cf, prop)
                with self._lock:
                    self._device_names[int(device)] = name
                    self._device_refs[int(device)] = int(device)
                    self._name = name
                    self._connected = True
                self._iokit.IOHIDDeviceOpen(device, 0)
                # Register and schedule PER DEVICE. The manager-level input callback is
                # silently unreliable for some Bluetooth pads - measured with a Switch Pro,
                # which delivered nothing at manager level and ~25 reports/s at device
                # level, which is exactly why an earlier version looked like a dead pad.
                self._iokit.IOHIDDeviceRegisterInputValueCallback(
                    device, ctypes.cast(self._cb_value, ctypes.c_void_p), None)
                self._iokit.IOHIDDeviceScheduleWithRunLoop(device, self._runloop, self._mode)
        except Exception:
            pass

    def _on_device_removed(self, context, result, sender, device) -> None:
        with self._lock:
            self._device_names.pop(int(device), None)
            self._device_refs.pop(int(device), None)
            self._connected = bool(self._device_names)
            self._name = next(iter(self._device_names.values()), "")
            if not self._connected:
                self._axes.clear()
                self._buttons.clear()
                self._vendor_hits = 0
                self._handshake_sent = 0
                self.raw_samples.clear()

    def _on_value(self, context, result, sender, value) -> None:
        try:
            iokit = self._iokit
            elem = iokit.IOHIDValueGetElement(value)
            if not elem:
                return
            page = int(iokit.IOHIDElementGetUsagePage(elem))
            usage = int(iokit.IOHIDElementGetUsage(elem))
            raw = int(iokit.IOHIDValueGetIntegerValue(value))

            if page == _HID_PAGE_BUTTON:
                pressed = raw != 0
                with self._lock:
                    if pressed and not self._buttons.get(usage, False):
                        self._edges.append(usage)
                    self._buttons[usage] = pressed
                return

            if page != _HID_PAGE_GENERIC_DESKTOP:
                if page == _VENDOR_PAGE:
                    # A vendor report means the pad is awake but still in its own mode; the
                    # handshake in _maybe_handshake() asks it to use the standard one, and
                    # the decoder below covers pads that never accept that request. Keep a
                    # byte sample too, for diagnostics.
                    length = int(iokit.IOHIDValueGetLength(value))
                    blob = b""
                    if 0 < length <= 128:
                        ptr = iokit.IOHIDValueGetBytePtr(value)
                        blob = bytes(ptr[:length])
                    with self._lock:
                        self._vendor_hits += 1
                        if len(self.raw_samples) < 24 and blob:
                            self.raw_samples.append((usage, blob))
                    if blob and usage == _SWITCH_FULL_REPORT_USAGE:
                        self._decode_switch_report(blob)
                return
            if usage in _AXIS_USAGES:
                lo = int(iokit.IOHIDElementGetLogicalMin(elem))
                hi = int(iokit.IOHIDElementGetLogicalMax(elem))
                norm, kind, rest = self._normalize(usage, raw, lo, hi)
                with self._lock:
                    self._axes[(page, usage)] = (norm, kind, rest, lo, hi)
            elif usage == _HID_USAGE_HAT:
                with self._lock:
                    self._buttons = {u: v for u, v in self._buttons.items()
                                     if u not in (HAT_UP, HAT_RIGHT, HAT_DOWN, HAT_LEFT)}
                    for bit in _HAT_BITS.get(raw, ()):
                        if not self._buttons.get(bit, False):
                            self._edges.append(bit)
                        self._buttons[bit] = True
        except Exception:
            pass

    def _normalize(self, usage: int, raw: int, lo: int, hi: int) -> Tuple[float, str, int]:
        """Map a raw element value onto [-1, 1] (stick) or [0, 1] (trigger).

        The kind is decided from where the axis RESTS: the middle of its logical range
        means a self-centering stick, the minimum means a trigger or slider. This is what
        lets one code path read both the Xbox layout (Z/Rz = triggers) and the Sony layout
        (Rx/Ry = triggers).
        """
        span = float(max(1, hi - lo))
        mid = lo + 0.5 * span
        known = self._axes.get((_HID_PAGE_GENERIC_DESKTOP, usage))
        if known is None:
            if abs(raw - mid) <= _REST_TOLERANCE * span:
                kind, rest = "stick", raw
            elif abs(raw - lo) <= _REST_TOLERANCE * span:
                kind, rest = "trigger", raw
            else:
                # Seen mid-motion (the pilot was already holding the stick): assume a
                # self-centering axis around whatever the logical middle is, and upgrade
                # to a confident "stick" as soon as a sample lands near the middle.
                kind, rest = "stick_offcenter", int(round(mid))
        else:
            kind, rest = known[1], known[2]
            if kind == "stick_offcenter" and abs(raw - mid) <= _REST_TOLERANCE * span:
                kind, rest = "stick", raw
        if kind == "trigger":
            return float(min(1.0, max(0.0, (raw - lo) / span))), kind, rest
        if kind == "stick_offcenter":
            half = max(float(rest - lo), float(hi - rest), 1.0)
            return float(min(1.0, max(-1.0, (raw - rest) / half))), kind, rest
        return float(min(1.0, max(-1.0, 2.0 * (raw - lo) / span - 1.0))), kind, rest

    # -- public snapshot ----------------------------------------------------------------
    def state(self) -> PadState:
        self._maybe_handshake()
        with self._lock:
            sticks = sorted((usage for (page, usage), v in self._axes.items()
                             if page == _HID_PAGE_GENERIC_DESKTOP and v[1].startswith("stick")))
            triggers = sorted(usage for (page, usage), v in self._axes.items()
                              if page == _HID_PAGE_GENERIC_DESKTOP and v[1] == "trigger")
            axes = {usage: self._axes[(_HID_PAGE_GENERIC_DESKTOP, usage)][0]
                    for usage in sticks + triggers}
            state = PadState(
                connected=self._connected and bool(self._axes),
                name=self._name,
                buttons={u for u, down in self._buttons.items() if down},
                pressed=tuple(self._edges),
                raw_axes=axes,
                analog_triggers=len(triggers),
            )
            self._edges = []

        def axis(index: int) -> float:
            return axes[sticks[index]] if len(sticks) > index else 0.0

        # Pairs are taken in usage order: (X,Y) then (Z,Rz) or (Rx,Ry) - whichever pair
        # exists, lower usage is the horizontal axis.
        state.left_x, state.left_y = axis(0), axis(1)
        state.right_x, state.right_y = axis(2), axis(3)
        if len(triggers) > 0:
            state.left_trigger = axes[triggers[0]]
        if len(triggers) > 1:
            state.right_trigger = axes[triggers[1]]
        # A trigger-only pad (or a racing wheel) may expose no stick axes at all: keep the
        # device usable by letting the two triggers act as the vertical axes.
        if not sticks and len(triggers) >= 2:
            state.left_y = axes[triggers[0]] - axes[triggers[1]]
        return state

    # -- Nintendo raw-report fallback -----------------------------------------------------
    def _decode_switch_report(self, blob: bytes) -> bool:
        """Publish a Switch "full" input report (report 0x30) as if it were HID elements.

        Verified against real bytes from a Switch-Pro-identifying pad: with the sticks at
        rest every axis decodes to 2048 = centre and the IMU frame reads ~1 g, which is how
        this layout was confirmed. Sticks keep the raw HID sign convention (forward/up is
        NEGATIVE) so the flight code's invert flags mean the same thing for every pad, and
        the buttons are published under the usages the pad's own descriptor declares
        (1..16) so the PAD_BTN_* mapping applies unchanged. The D-pad becomes the synthetic
        HAT_* codes the rest of the code already understands.
        """
        if len(blob) < 12:
            return False
        buttons_1, buttons_2, buttons_3 = blob[2], blob[3], blob[4]
        left_x = blob[5] | ((blob[6] & 0x0F) << 8)
        left_y = (blob[6] >> 4) | (blob[7] << 4)
        right_x = blob[8] | ((blob[9] & 0x0F) << 8)
        right_y = (blob[9] >> 4) | (blob[10] << 4)
        sticks = {0x30: left_x, 0x31: left_y, 0x33: right_x, 0x34: right_y}
        buttons = {
            1: bool(buttons_1 & 0x04), 2: bool(buttons_1 & 0x08),      # B (bottom), A (right)
            3: bool(buttons_1 & 0x01), 4: bool(buttons_1 & 0x02),      # Y (left), X (top)
            5: bool(buttons_3 & 0x40), 6: bool(buttons_1 & 0x40),      # L, R
            7: bool(buttons_3 & 0x80), 8: bool(buttons_1 & 0x80),      # ZL, ZR
            9: bool(buttons_2 & 0x01), 10: bool(buttons_2 & 0x02),     # minus, plus
            11: bool(buttons_2 & 0x08), 12: bool(buttons_2 & 0x04),    # stick presses
            13: bool(buttons_2 & 0x10), 14: bool(buttons_2 & 0x20),    # home, capture
            HAT_DOWN: bool(buttons_3 & 0x01), HAT_UP: bool(buttons_3 & 0x02),
            HAT_RIGHT: bool(buttons_3 & 0x04), HAT_LEFT: bool(buttons_3 & 0x08),
        }
        centre = float(_SWITCH_STICK_CENTRE)
        with self._lock:
            for usage, raw in sticks.items():
                norm = float(min(1.0, max(-1.0, (raw - centre) / centre)))
                self._axes[(_HID_PAGE_GENERIC_DESKTOP, usage)] = (
                    norm, "stick", _SWITCH_STICK_CENTRE, 0, 4095)
            for usage, down in buttons.items():
                if down and not self._buttons.get(usage, False):
                    self._edges.append(usage)
                self._buttons[usage] = down
            if not self.note:
                self.note = "decoding the pad's own (Switch) report format"
        return True

    # -- Nintendo report-mode handshake --------------------------------------------------
    def _send_input_mode(self, device: int, attempt: int) -> int:
        """Send SET_INPUT_REPORT_MODE(0x3F) to a Nintendo-style pad.

        Byte layout taken from the Linux driver (`struct joycon_subcmd_request`): output id
        0x01, packet counter, EIGHT RUMBLE BYTES, then the subcommand and its argument.
        The rumble bytes are left at zero on purpose - a packet whose payload is misplaced
        lands there instead and the pad buzzes, which is exactly what happened while this
        was being figured out.
        """
        payload = bytearray(_SWITCH_BT_REPORT_SIZE)
        payload[0] = 0x01                          # 0x01 = rumble + subcommand report
        payload[1] = attempt & 0x0F                # packet counter, wraps at 0x10
        # payload[2:10] = rumble data: zero = silence
        payload[10] = _SWITCH_CMD_SET_INPUT_REPORT_MODE
        payload[11] = _SWITCH_MODE_SIMPLE_HID
        buf = (ctypes.c_ubyte * _SWITCH_BT_REPORT_SIZE)(*payload)
        return int(self._iokit.IOHIDDeviceSetReport(
            ctypes.c_void_p(device), 1, 1, buf, _SWITCH_BT_REPORT_SIZE))

    def _maybe_handshake(self) -> None:
        """While a pad streams ONLY vendor reports, ask it to switch to the standard one.

        Throttled to 4 attempts/s and capped, so a pad that simply does not support the
        request is asked a few times and then left alone (the sim keeps working from the
        keyboard). Called from state(), i.e. from the flight loop, which keeps the whole
        handshake out of the HID callback thread.
        """
        now = time.time()
        with self._lock:
            if (not self._device_refs or self._axes or self._vendor_hits == 0
                    or self._handshake_sent >= 60 or now - self._last_handshake < 0.33):
                return
            self._last_handshake = now
            self._handshake_sent += 1
            attempt = self._handshake_sent
            device = next(iter(self._device_refs))
        rc = self._send_input_mode(device, attempt)
        with self._lock:
            if attempt == 1:
                self.note = ("pad streams vendor reports only - asking for the standard "
                             f"HID report (SET_INPUT_REPORT_MODE 0x{_SWITCH_MODE_SIMPLE_HID:02X}, "
                             f"first attempt rc={rc:#x})")
            if attempt >= 60 and not self._axes:
                self.note = ("pad ignored the standard-report request - running in raw mode "
                             "if its report format is supported")

    def raw_report_lines(self, limit: int = 8) -> List[str]:
        """Hex dump of the vendor reports seen so far (diagnostics / decoder work)."""
        with self._lock:
            samples = list(self.raw_samples)[:limit]
        return [f"vendor usage=0x{usage:04X} len={len(blob):3d}  {blob.hex()}"
                for usage, blob in samples]

    # -- diagnostics --------------------------------------------------------------------
    def describe(self) -> List[str]:
        """Element table of every matched device, even when no values are flowing yet.

        This is the first thing to look at when a pad is matched (its name shows up) but
        its sticks stay at zero: it prints each element's usage page / usage / logical
        range / current raw value straight from the HID descriptor, so a pad that reports
        on vendor pages, or that exposes only Feature elements until a handshake, is
        obvious instead of a mystery.
        """
        if not self.available:
            return [f"IOKit unavailable: {self.error}"]
        cf, iokit = self._cf, self._iokit
        assert cf is not None and iokit is not None, "frameworks not loaded"
        with self._lock:
            devices = dict(self._device_names)
        if not devices:
            return ["no game controller matched"]
        lines: List[str] = []
        for ptr, name in devices.items():
            lines.append(f"device: {name}")
            array = iokit.IOHIDDeviceCopyMatchingElements(ctypes.c_void_p(ptr), None, 0)
            if not array:
                lines.append("  (descriptor not available - device may not be open)")
                continue
            count = int(cf.CFArrayGetCount(ctypes.c_void_p(array)))
            lines.append(f"  {count} elements:")
            for i in range(count):
                elem = cf.CFArrayGetValueAtIndex(ctypes.c_void_p(array), i)
                page = int(iokit.IOHIDElementGetUsagePage(elem))
                usage = int(iokit.IOHIDElementGetUsage(elem))
                etype = int(iokit.IOHIDElementGetType(elem))
                lo = int(iokit.IOHIDElementGetLogicalMin(elem))
                hi = int(iokit.IOHIDElementGetLogicalMax(elem))
                value_ref = ctypes.c_void_p(0)
                rc = iokit.IOHIDDeviceGetValue(ctypes.c_void_p(ptr), elem,
                                               ctypes.byref(value_ref))
                if rc == 0 and value_ref:
                    raw = int(iokit.IOHIDValueGetIntegerValue(value_ref))
                    shown = f"{raw}"
                else:
                    shown = "n/a"
                lines.append(
                    f"    page=0x{page:04X} usage=0x{usage:02X} "
                    f"type={_ELEMENT_TYPES.get(etype, etype)} range=[{lo}, {hi}] value={shown}"
                )
            cf.CFRelease(ctypes.c_void_p(array))
        return lines


def probe_devices(seconds: float = 3.0, match_all_generic_desktop: bool = False) -> List[str]:
    """Diagnostics: list HID devices and every element they report for a few seconds.

    Returns human-readable lines; `gamepad_probe.py` prints them. Used to check a pad's
    axis layout (which usage is which stick/trigger) before trusting the flight mapping.

    The default matches game controllers only. Passing match_all_generic_desktop=True
    also matches keyboards and trackpads, which on current macOS can raise an Input
    Monitoring permission prompt - only do that deliberately.
    """
    pad = HIDGamepad(match_all_generic_desktop=match_all_generic_desktop)
    if not pad.available:
        return [f"IOKit unavailable: {pad.error}"]
    pad.start()
    deadline = time.time() + seconds
    while time.time() < deadline:
        time.sleep(0.1)
    state = pad.state()
    with pad._lock:                                   # diagnostics: read the raw tables
        devices = dict(pad._device_names)
        axes = dict(pad._axes)
        buttons = dict(pad._buttons)
    pad.stop()
    lines = [f"devices matched: {len(devices)}"]
    for ptr, name in devices.items():
        lines.append(f"  - {name}")
    lines.append(f"analog elements seen: {len(axes)}")
    for (page, usage), (norm, kind, rest, lo, hi) in sorted(axes.items()):
        lines.append(f"  - usage 0x{usage:02X} page 0x{page:02X}: kind={kind:15s} "
                     f"range=[{lo}, {hi}] rest={rest} value={norm:+.3f}")
    lines.append(f"buttons seen: {sorted(buttons)}  currently held: "
                 f"{sorted(u for u, d in buttons.items() if d)}")
    return lines


def describe_devices(seconds: float = 3.0) -> List[str]:
    """Match game controllers, wait, and dump their HID element tables.

    The tool to use when a pad is matched (its name appears) but nothing moves: it shows
    what the device actually declares, element by element, with the current raw value.
    """
    pad = HIDGamepad()
    if not pad.available:
        return [f"IOKit unavailable: {pad.error}"]
    pad.start()
    time.sleep(seconds)
    lines = pad.describe()
    with pad._lock:
        live_axes = len(pad._axes)
        buttons = sum(1 for down in pad._buttons.values() if down)
    lines.append(f"analog elements parsed by the reader: {live_axes}"
                 f" | buttons currently held: {buttons}")
    if pad.error:
        lines.append(f"reader note: {pad.error}")
    pad.stop()
    return lines
