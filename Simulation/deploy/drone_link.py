#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Where is the drone? One place that knows.

The factory URI is `radio://0/80/2M/E7E7E7E7E7`, but the channel / datarate / address a
drone actually uses are **stored in its config block, not compiled into the firmware**
(`firmware/src/hal/src/radiolink.c` boots from `configblockGetRadioChannel/Speed/Address()`,
and `configblockeeprom.c` only rewrites that EEPROM when its magic/version/checksum is
invalid). Flashing - warm boot or cold boot - does not change it. A second-hand board, or one
from a lab where every drone gets its own address, therefore ignores the factory URI:

  * the Crazyradio opens, then `Too many packets lost`;
  * `crtp.scan_interfaces()` finds nothing, because it scans every channel but only at the
    DEFAULT address, so a changed address is invisible to it;
  * USB and cold-boot flashing still work, because the nRF51 bootloader uses its own
    hard-coded settings (channel 0 / 110) and ignores the config block.

`radio_config.py` reads those stored settings over the cable and records the matching URI
here; every other tool resolves `--uri` through this module, so it is typed once, ever.
"""
from __future__ import annotations

import os
from typing import Optional

DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"      # the factory Crazyflie address

_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_URI_PATH = os.path.join(os.path.dirname(os.path.dirname(_DIR)),
                              "logs", "drone_uri.txt")


def saved_uri() -> Optional[str]:
    """The URI a previous `radio_config.py` run learned, if it is still there."""
    try:
        with open(SAVED_URI_PATH, "r", encoding="utf-8") as fh:
            uri = fh.read().strip()
    except OSError:
        return None
    return uri if uri.startswith("radio://") else None


def save_uri(uri: str) -> None:
    os.makedirs(os.path.dirname(SAVED_URI_PATH), exist_ok=True)
    with open(SAVED_URI_PATH, "w", encoding="utf-8") as fh:
        fh.write(uri.strip() + "\n")


def preferred_uri(uri: Optional[str] = None) -> str:
    """`--uri` if given, else the remembered one, else the factory default."""
    return uri or saved_uri() or DEFAULT_URI


def source(uri: Optional[str] = None) -> str:
    """A one-line explanation of where the URI came from, for the console."""
    if uri:
        return "--uri"
    if saved_uri():
        return f"remembered in {os.path.relpath(SAVED_URI_PATH)} (radio_config.py)"
    return "factory default"


def radio_owners() -> list:
    """Our own tools that are running and therefore OWN the Crazyradio.

    MEASURED 2026-09-15: the dongle is a single-owner USB device - cflib keeps one
    handle open for the whole session - so while `radio_flight.py` is alive, EVERY other
    tool that tries to open it fails. macOS/libusb reports that as

        usb.core.USBError: [Errno 19] No such device (it may have been disconnected)

    which reads like a dead or unplugged dongle and is why this looked random: "it worked
    once, then it didn't" == "the previous session was still running". A physical replug
    appeared to fix it only because it invalidated the other process's handle.
    """
    names = ("radio_flight", "radio_gui", "flight_gui", "bench_bringup",
             "lighthouse_check", "radio_config", "cfclient")
    try:
        import subprocess
        out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return []
    me = os.getpid()
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid == me or "python" not in cmd.lower():
            continue
        if any(n in cmd for n in names):
            hits.append(f"pid {pid}: {cmd.strip()[:78]}")
    return hits


def dongle_preflight(uri: Optional[str] = None) -> bool:
    """Say - in ONE line - when the Crazyradio cannot be opened, and WHY.

    Two distinct causes, and they need different fixes:

    1. Another one of our tools is still running and holds the dongle (the common case).
       Fix: stop that process (Ctrl-C in its terminal) - replugging is not the cure.
    2. The dongle is genuinely stuck at the USB level. Fix: unplug/replug it, directly
       into the Mac, not a hub. Software reset cannot clear it (`dev.reset()` fails with
       the same ENODEV).

    Both surface as `Couldn't load link driver: [Errno 19] No such device` plus a long
    traceback, so the tool opens the device first and only blames ownership or hardware
    once that has actually failed - that way the check can never invent a failure.
    Returns False only when the caller should stop.
    """
    target = preferred_uri(uri)
    if not target.startswith("radio://"):
        return True
    try:
        import usb.core
    except Exception:                   # pragma: no cover - cflib always pulls pyusb
        return True

    print(f"  crazyradio  : checking {target}")
    device = usb.core.find(idVendor=0x1915, idProduct=0x7777)
    failure: Optional[str] = None
    if device is None:
        failure = "not on the USB bus"
    else:
        try:
            device.set_configuration(1)
        except Exception as exc:
            failure = str(exc)

    if failure is None:
        # set_configuration() succeeding is NOT proof of exclusive access: libusb on macOS
        # opens the device non-exclusively, so a live `cfclient` still reads as OK here and
        # the real connect then blocks with no output at all. Measured 2026-09-17. So scan
        # for owners EVEN ON SUCCESS and warn - this is exactly the case that looked like a
        # hang with a green checkmark above it.
        print("  crazyradio  : OK")
        owners = radio_owners()
        if owners:
            print("  crazyradio  : BUT another session is running and can hold this dongle:")
            for who in owners:
                print(f"  crazyradio  :    {who}")
            print("  crazyradio  : the USB probe cannot rule this out - if the connect below")
            print("  crazyradio  : hangs, kill that process and try again. Use SIGKILL:")
            print("  crazyradio  : cfclient is a Qt GUI and ignores plain SIGTERM.")
        return True

    owners = radio_owners()
    if owners:
        print(f"  crazyradio  : IN USE - another tool owns the dongle ({failure})")
        for who in owners:
            print(f"  crazyradio  :    {who}")
        print("  crazyradio  : STOP that process first (Ctrl-C in its terminal).")
        print("  crazyradio  : The dongle is a single-owner device: one session at a time.")
        print("  crazyradio  : No replug needed - and a replug only masks this.")
        return False

    if "No such device" in failure or "ENODEV" in failure:
        print(f"  crazyradio  : WEDGED at the USB level ({failure})")
        print("  crazyradio  : no other tool is running, so UNPLUG the dongle, wait ~5 s,")
        print("  crazyradio  : plug it back in - directly into the Mac, not a hub.")
        return False

    print(f"  crazyradio  : could not probe ({failure}) - continuing")
    return True
