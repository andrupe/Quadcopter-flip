#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read (or rewrite) the Crazyflie 2.x configuration block - the one that holds the radio
settings - over USB, i.e. over the link that still works when the radio settings are wrong.

    .venv/bin/python Simulation/deploy/radio_config.py                # read + report
    .venv/bin/python Simulation/deploy/radio_config.py --defaults     # 80 / 2M / E7E7E7E7E7
    .venv/bin/python Simulation/deploy/radio_config.py --channel 100 --speed 1

WHY THIS EXISTS. The channel, datarate and address your drone actually uses are NOT what the
firmware was compiled with. `firmware/src/hal/src/radiolink.c` does

    radiolinkSetChannel(configblockGetRadioChannel());
    radiolinkSetDatarate(configblockGetRadioSpeed());
    radiolinkSetAddress(configblockGetRadioAddress());

and `configblockeeprom.c` only rewrites that EEPROM when its magic/version/checksum is
invalid. So a board that was previously used on another channel or address KEEPS it, and
flashing firmware - warm boot or cold boot - does not change it. The symptoms are a specific
and confusing combination:

  * `radio://0/80/2M/E7E7E7E7E7` opens the dongle and then dies with "Too many packets lost";
  * `cflib.crtp.scan_interfaces()` finds nothing, because it scans every CHANNEL at the
    DEFAULT ADDRESS only - a changed address is invisible to it;
  * USB works perfectly, and cold-boot flashing works, because the nRF51 bootloader uses its
    own hard-coded radio settings (`radio://0/0/2M/...` and `radio://0/110/2M/...`) and
    ignores the config block entirely.

This is the scriptable form of cfclient's "Configure 2.x" dialog: the config block is a
`MemoryElement.TYPE_I2C` element with the fields `radio_channel`, `radio_speed` (0 = 250K,
1 = 1M, 2 = 2M), `radio_address` (version 1 blocks only) plus the roll/pitch trims, which are
preserved.

A reboot is required for a write to take effect (`radiolinkInit` runs at boot).
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Any, Dict, Optional

import drone_link

DEFAULT_URI = "usb://0"                     # the cable: the radio is what we are fixing
SPEEDS = {0: "250K", 1: "1M", 2: "2M"}
FACTORY = {"radio_channel": 80, "radio_speed": 2, "radio_address": 0xE7E7E7E7E7}

SAVED_URI_PATH = drone_link.SAVED_URI_PATH      # where the other tools look for the URI


def save_uri(uri: str) -> None:
    drone_link.save_uri(uri)
    _say(f"saved   : {uri} -> {os.path.relpath(SAVED_URI_PATH)}")
    _say("          bench_bringup.py / lighthouse_check.py / radio_flight.py will use it")


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _fmt_address(addr: int) -> str:
    return f"{addr:010X}"


def uri_for(values: Dict[str, Any]) -> str:
    """The cflib URI that matches this config block."""
    speed = SPEEDS.get(int(values.get("radio_speed", 2)), "?")
    addr = values.get("radio_address")
    if addr is None:                        # version 0 block: no address field
        return f"radio://0/{values.get('radio_channel')}/{speed}/<address from firmware default>"
    return f"radio://0/{values.get('radio_channel')}/{speed}/{_fmt_address(int(addr))}"


def _i2c_mem(cf, timeout: float = 10.0):
    from cflib.crazyflie.mem import MemoryElement

    t0 = time.time()
    mem = None
    while time.time() - t0 < timeout:
        mems = cf.mem.get_mems(MemoryElement.TYPE_I2C)
        if mems:
            mem = mems[0]
            break
        time.sleep(0.2)
    if mem is None:
        raise SystemExit("no I2C memory element found - this does not look like a CF2.x config "
                         "block (or the memory subsystem did not answer)")
    # cflib reads the 1-wire / deck memories right after connecting; let that pass finish
    # before asking for the config block.
    time.sleep(1.5)
    return mem


def _read_block(mem, timeout: float = 10.0, attempts: int = 3) -> None:
    """
    Read the block, retrying.

    `I2CElement.update()` starts with `if not self._update_finished_cb:` - so if an earlier
    update ever left that callback set (a read that was dropped, e.g. while the connection
    setup was still walking the memory TOC), every later call is a SILENT NO-OP and the
    read never happens. Hence: clear it, ask, wait, repeat.
    """
    for attempt in range(1, attempts + 1):
        done = threading.Event()
        mem._update_finished_cb = None                      # cflib quirk, see docstring
        mem.update(lambda *_a: done.set())                  # called back as cb(mem)
        if done.wait(timeout):
            return
        _say(f"  (no answer yet, retry {attempt}/{attempts})")
    raise SystemExit("the config block did not answer (timeout) - is the drone powered and "
                     "is this really a CF2.x?")


def read_config(cf, timeout: float = 10.0) -> Dict[str, Any]:
    mem = _i2c_mem(cf, timeout)
    _read_block(mem, timeout)
    if not getattr(mem, "valid", False):
        raise SystemExit("the config block read back INVALID (bad magic or checksum) - the "
                         "firmware rewrites the factory defaults on the next boot")
    return dict(mem.elements)


def write_config(cf, values: Dict[str, Any], timeout: float = 10.0) -> None:
    """Write `values` on top of whatever is already there (trims are kept)."""
    mem = _i2c_mem(cf, timeout)
    _read_block(mem, timeout)

    mem.elements.update(values)
    if "radio_address" in values and int(mem.elements.get("version", 0)) == 0:
        # a version 0 block has no address field at all: writing one means upgrading it
        # (this is also how the address gets out of the compiled-in default)
        mem.elements["version"] = 1

    done = threading.Event()
    # NOTE: the write callback is called as cb(mem, addr) - TWO arguments (i2c_element.py
    # write_done), unlike update(), so accept whatever cflib passes.
    mem.write_data(lambda *_a: done.set())
    if not done.wait(timeout):
        raise SystemExit("the config block write did not complete (timeout)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Read/write the CF2.x config block (radio "
                                             "settings) - over USB by default")
    ap.add_argument("--uri", default=DEFAULT_URI,
                    help=f"cflib URI (default: {DEFAULT_URI}; use a radio URI only if the "
                         f"radio already works)")
    ap.add_argument("--defaults", action="store_true",
                    help=f"write the factory defaults (channel 80, 2M, "
                         f"{_fmt_address(FACTORY['radio_address'])})")
    ap.add_argument("--channel", type=int, help="radio channel 0-125")
    ap.add_argument("--speed", type=int, choices=sorted(SPEEDS), help="0=250K 1=1M 2=2M")
    ap.add_argument("--address", type=lambda s: int(s, 0),
                    help="40-bit radio address, e.g. 0xE7E7E7E7E7")
    args = ap.parse_args()

    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

    cflib.crtp.init_drivers()               # CLASSES is empty until this runs

    requested: Dict[str, Any] = {}
    if args.defaults:
        requested.update(FACTORY)
    if args.channel is not None:
        requested["radio_channel"] = args.channel
    if args.speed is not None:
        requested["radio_speed"] = args.speed
    if args.address is not None:
        requested["radio_address"] = args.address

    _say(f"link : {args.uri}")
    cf = Crazyflie(rw_cache=".cf_cache")
    with SyncCrazyflie(args.uri, cf=cf) as scf:
        cf = scf.cf
        time.sleep(0.5)
        before = read_config(cf)
        _say()
        _say("current config block")
        _say(f"  version       : {before.get('version')}")
        _say(f"  radio channel : {before.get('radio_channel')}")
        _say(f"  radio speed   : {before.get('radio_speed')} "
             f"({SPEEDS.get(int(before.get('radio_speed', -1)), '?')})")
        addr = before.get("radio_address")
        _say(f"  radio address : " + (f"0x{_fmt_address(int(addr))}" if addr is not None
                                      else "n/a (version 0 block -> the firmware default, "
                                           f"0x{_fmt_address(FACTORY['radio_address'])})"))
        _say(f"  roll / pitch  : {before.get('roll_trim')} / {before.get('pitch_trim')}")
        _say()
        _say(f"  -> the drone answers at : {uri_for(before)}")
        if "<" not in uri_for(before):
            save_uri(uri_for(before))

        if not requested:
            _say()
            _say("read-only run. To put it back on the factory settings:")
            _say("  .venv/bin/python Simulation/deploy/radio_config.py --defaults")
            _say("(then power-cycle the drone: the radio settings are applied at boot)")
            return 0

        _say()
        _say(f"writing : {requested}")
        write_config(cf, requested)
        time.sleep(0.5)
        after = read_config(cf)
        bad = {k: (v, after.get(k)) for k, v in requested.items()
               if after.get(k) is None or int(after.get(k)) != int(v)}
        _say()
        _say("after write  : channel %s, speed %s (%s), address %s" % (
            after.get("radio_channel"), after.get("radio_speed"),
            SPEEDS.get(int(after.get("radio_speed", -1)), "?"), uri_for(after).split("/")[-1]))
        for k, (want, got) in bad.items():
            _say(f"  MISMATCH   : {k} asked {want}, read back {got}")
        if bad:
            _say("verdict      : *** the write did not stick ***")
            _say("               the settings already in the EEPROM keep working, so the")
            _say("               drone is not bricked - retry, or change one field at a")
            _say("               time to see which one is refused")
            return 1
        _say("verdict      : OK")
        _say()
        _say(f"NOW POWER-CYCLE THE DRONE, then use : {uri_for(after)}")
        save_uri(uri_for(after))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
