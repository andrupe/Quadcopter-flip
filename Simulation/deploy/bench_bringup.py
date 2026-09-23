#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bench bring-up for the policy app: link, controller select, props-off arm/disarm, and the
SHADOW capture that confirms the SIGN_* assumptions.

    .venv/bin/python Simulation/deploy/bench_bringup.py            # radio, factory address
    .venv/bin/python Simulation/deploy/bench_bringup.py --uri radio://0/80/2M/<ADDRESS>
    .venv/bin/python Simulation/deploy/bench_bringup.py --scan     # find an unknown address

RADIO IS THE PATH. Everything runs over the Crazyradio PA on
`radio://0/80/2M/E7E7E7E7E7` - the factory address, which is the default, so no `--uri` is
needed unless the address has been changed. `--scan` enumerates every channel and datarate
when the vehicle does not answer at the default (10-30 s).

WHAT IT DOES, IN ORDER
  1. connects over the radio (default address; `--scan` to find another)
  2. prints the console banner and reads the app's params -> proves the policy app is
     really in the flashed image
  3. sets `stabilizer.controller = 6` (OutOfTree) if it is not already there
  4. REFUSES to go further unless `policy.shadow == 1`. This script never commands the
     policy: it is the props-off / in-hand gate that has to pass BEFORE shadow is cleared
  5. arm then disarm through the appchannel (0x01 / 0x02) and confirms the state from the
     `policy` log group
  6. SHADOW CAPTURE: asks the pilot to carry the vehicle through four motions (roll right,
     nose down, yaw left, lift 0.5 m) while logging `policy.*`, `stateEstimate.*`,
     `gyro.*`, `acc.*` and `pm.vbat` at 10 Hz; writes logs/bench_shadow.csv; then CHECKS
     THE SIGN_* CONSTANTS in controller_app.c against what the gyro actually reported.

WHY THE SIGN CHECK IS THE POINT
  `controller_app.c` maps the firmware's body frame onto the simulator's with six constants
  (SIGN_*). A wrong one makes the policy fly a MIRRORED vehicle - plausibly, which is the
  dangerous kind of failure. The gyro is the cheapest place to catch it, and a vehicle in
  the hand is enough: a right roll, a nose-down pitch and a left yaw each have exactly one
  expected sign in the sim's frame (x = nose, y = left, z = up, right-handed).

  The sim's convention, and therefore the expectation:
      roll right   (right wing down)        -> gyro.x POSITIVE
      nose down                             -> gyro.y POSITIVE
      yaw left     (counter-clockwise seen from above) -> gyro.z POSITIVE
  The app currently assumes +1 for all three gyro signs. If a measured sign disagrees, the
  report says which constant to flip - change it in controller_app.c, rebuild, re-run.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import drone_link  # noqa: E402  (one place that knows where the drone is)
_OUT_CSV = os.path.join(_PROJECT_ROOT, "logs", "bench_shadow.csv")

CMD_ARM = 0x01
CMD_DISARM = 0x02
CMD_STATUS = 0x04
STATUS_MAGIC = 0xA5

# `policy.abort_reason` codes from controller_app.c (ABORT_NONE..ABORT_COUNT).
ABORT_NAMES = {0: "none", 1: "tilt limit", 2: "z outside the flight envelope",
               3: "non-finite estimate", 4: "count"}

# The four motions, what to do, and which gyro axis/telltale is checked.
# (label, instruction, axis index, expected sign, what it proves)
MOTIONS: Tuple[Tuple[str, str, int, float, str], ...] = (
    ("roll right", "ROLL RIGHT: bank the right side down ~45 deg, hold, then level",
     0, +1.0, "SIGN_GYRO_ROLL_TO_SIM"),
    ("nose down", "PITCH NOSE DOWN: dip the nose ~45 deg, hold, then level",
     1, +1.0, "SIGN_GYRO_PITCH_TO_SIM"),
    ("yaw left", "YAW LEFT: rotate the whole vehicle counter-clockwise seen from ABOVE, "
                 "then back", 2, +1.0, "SIGN_GYRO_YAW_TO_SIM"),
    ("lift", "LIFT: raise it ~0.5 m and lower it again (checks stateEstimate.z)", -1, +1.0,
     "POLICY_MIN_Z / z scale"),
)

LOG_VARS = (
    "policy.armed", "policy.mode", "policy.abort_reason", "policy.thrust_units",
    "policy.tilt_deg", "policy.p_err", "policy.vbat", "policy.act0",
    # policy.ff_z is the feed-forward block's vertical component (a_ref_z + g) and is the
    # cheapest proof that the block is alive on the vehicle: it reads ~9.81 whenever the
    # reference is a HOVER, and 0 while a flip is in its ballistic coast. If the app ever
    # lost the block (or fed the actor a shifted vector) this is the one channel that says
    # so on the bench, without flying.
    "policy.ff_z",
    "stateEstimate.x", "stateEstimate.y", "stateEstimate.z", "stateEstimate.yaw",
    "gyro.x", "gyro.y", "gyro.z",
    "acc.x", "acc.y", "acc.z",
)


def _say(msg: str = "") -> None:
    print(msg, flush=True)


DEFAULT_URI = drone_link.DEFAULT_URI            # the factory Crazyflie address


def _saved_uri() -> Optional[str]:
    """Kept for callers that still ask bench_bringup; the real one is drone_link."""
    return drone_link.saved_uri()


def _find_link(uri: Optional[str] = None, scan: bool = False) -> str:
    """
    The Crazyradio link.

    The factory address is E7E7E7E7E7, so that is the default and connecting is immediate.
    `scan=True` enumerates every radio channel and datarate instead (10-30 s) - use it when
    the address has been changed, or when the vehicle simply does not answer.

    A URI learned by `radio_config.py` (the drone's STORED radio settings, which are not the
    factory ones on a second-hand board) is remembered in `logs/drone_uri.txt` and used
    automatically - otherwise every command would need `--uri radio://0/<ch>/<rate>/<addr>`.
    """
    import cflib.crtp

    # MANDATORY, and before the early returns below: `crtp.CLASSES` is EMPTY until this
    # runs, and `get_link_driver()` iterates it - so without it every URI resolves to None
    # and SyncCrazyflie dies with "No driver found or malformed URI: radio://...".
    # (Cheap and idempotent.)
    cflib.crtp.init_drivers()

    if uri:
        return uri
    saved = drone_link.saved_uri()
    if saved:
        _say(f"uri     : {saved}  ({drone_link.source()}; --uri overrides)")
        return saved
    if not scan:
        return DEFAULT_URI
    from cflib.crtp import scan_interfaces

    _say("scanning radio interfaces (10-30 s)...")
    found = scan_interfaces()
    radio = [u for u, _ in found if u.startswith("radio://")]
    if not radio:
        raise SystemExit(
            "no Crazyflie found on the radio.\n"
            "  * is the Crazyradio PA plugged in, and the Crazyflie powered?\n"
            "  * is the address still E7E7E7E7E7? pass --uri radio://0/80/2M/<ADDRESS>\n"
            "  * a slower datarate may be needed on a busy channel (250K / 1M)")
    return radio[0]


class _MultiLog:
    """`stop()` for a group of log blocks.

    One `LogConfig` can only carry `LogConfig.MAX_LEN` (26) BYTES of payload, and the
    shadow capture wants 18 mixed uint8/float variables, so it has to be spread over
    several blocks. This is the bit of the interface the capture loop uses.
    """

    def __init__(self, blocks: List[Any]) -> None:
        self.blocks = blocks

    def stop(self) -> None:
        for block in self.blocks:
            try:
                block.stop()
            except Exception:
                pass


class Bench:
    def __init__(self, cf) -> None:
        self.cf = cf
        self.console: List[str] = []
        self.status: Dict[int, object] = {}
        self.rows: List[Tuple[float, Dict[str, float]]] = []
        self._lock = threading.Lock()

    # -- plumbing ----------------------------------------------------------------------
    def attach(self) -> None:
        # cflib spells this `receivedChar` in 0.1.33 (both venvs here) and `received_char`
        # in some other builds - resolve it at runtime instead of guessing.
        console_cb = (getattr(self.cf.console, "received_char", None)
                      or getattr(self.cf.console, "receivedChar", None))
        if console_cb is not None:
            console_cb.add_callback(self.console.append)
        self.cf.appchannel.packet_received.add_callback(self._on_status)

    def _on_status(self, data: bytes) -> None:
        if len(data) < 30 or data[0] != STATUS_MAGIC:
            return
        import struct
        (magic, armed, mode, shadow, abort_reason, pad,
         z, tilt, thrust, p_err, vbat, act0) = struct.unpack("<6B6f", data[:30])
        with self._lock:
            self.status.update(dict(armed=armed, mode=mode, shadow=shadow, abort=abort_reason,
                                    z=z, tilt=tilt, thrust=thrust, p_err=p_err, vbat=vbat,
                                    act0=act0))

    def status_now(self) -> Dict[str, float]:
        self.cf.appchannel.send_packet(bytes([CMD_STATUS]))
        time.sleep(0.25)
        with self._lock:
            return dict(self.status)

    def read_param(self, name: str, timeout: float = 3.0) -> Optional[str]:
        # cflib has a blocking getter on newer versions; the callback dance below is the
        # fallback and works without the full param table having been downloaded.
        getter = getattr(self.cf.param, "get_value", None)
        if callable(getter):
            try:
                return str(getter(name, timeout=timeout))
            except Exception:
                pass
        done = threading.Event()
        box: Dict[str, str] = {}

        def cb(_name, value):
            box["v"] = value
            done.set()

        try:
            self.cf.param.add_update_callback(group=name.split(".")[0],
                                              name=name.split(".", 1)[1], cb=cb)
        except Exception:
            return None
        self.cf.param.request_param_update(name)
        done.wait(timeout)
        return box.get("v")

    def set_param(self, name: str, value: str) -> None:
        self.cf.param.set_value(name, value)
        time.sleep(0.2)

    def send(self, cmd: int) -> None:
        self.cf.appchannel.send_packet(bytes([cmd]))

    # -- logging -----------------------------------------------------------------------
    def start_log(self, period_ms: int = 100):
        """Start logging `LOG_VARS`, split across as many blocks as the payload allows.

        cflib caps a block at `LogConfig.MAX_LEN` = 26 BYTES, and it enforces that in
        `add_config()`, NOT in `add_variable()` - the latter just queues a name in
        `default_fetch_as` and cannot fail. Adding all 18 variables to one block therefore
        got all the way to the end and then raised

            AttributeError: The log configuration is too large or has an invalid parameter

        at the START of the shadow capture, with nothing captured. So: resolve each
        variable's real byte size from the log TOC and pack greedily into blocks of
        <= 26 B. Types are taken from the TOC explicitly so the accounting cannot drift.
        """
        from cflib.crazyflie.log import LogConfig, LogTocElement

        wanted: List[Tuple[str, str, int]] = []
        missing: List[str] = []
        for var in LOG_VARS:
            elem = self.cf.log.toc.get_element_by_complete_name(var)
            if elem is None:
                missing.append(var)
                continue
            # `elem` exposes the C type NAME ('float', 'uint8_t', ...) while the size
            # table is keyed by the type ID, so go through get_id_from_cstring: passing
            # the name straight to get_size_from_id raises a KeyError whose message then
            # dies on a "%d" format ("Type [%d] not found ... a real number is required").
            ctype = elem.ctype
            wanted.append((var, ctype,
                           LogTocElement.get_size_from_id(
                               LogTocElement.get_id_from_cstring(ctype))))
        if missing:
            _say(f"note: not in this firmware, skipped: {', '.join(missing)}")
        if not wanted:
            raise RuntimeError("none of LOG_VARS is in this firmware's log TOC")

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

        t0 = time.time()

        def cb(_ts, data, _lg):
            with self._lock:
                self.rows.append((time.time() - t0, dict(data)))

        blocks: List[Any] = []
        for idx, chunk in enumerate(chunks):
            lg = LogConfig(name=f"bench{idx}", period_in_ms=period_ms)
            for name, ctype, _size in chunk:
                lg.add_variable(name, ctype)
            self.cf.log.add_config(lg)
            lg.data_received_cb.add_callback(cb)
            blocks.append(lg)

        for lg in blocks:
            lg.start()
        _say(f"logging : {len(blocks)} block(s), {len(wanted)}/{len(LOG_VARS)} variables, "
             f"max payload {max(sum(s for _n, _c, s in c) for c in chunks)} B "
             f"(limit {LogConfig.MAX_LEN} B/block)")
        return _MultiLog(blocks), t0


def main() -> int:
    ap = argparse.ArgumentParser(description="Bench bring-up + shadow capture (props off)")
    ap.add_argument("--uri", default=None,
                    help=f"cflib URI (default: {DEFAULT_URI})")
    ap.add_argument("--scan", action="store_true",
                    help="scan radio channels/datarates for the address (10-30 s)")
    ap.add_argument("--seconds", type=float, default=6.0, help="capture per motion")
    ap.add_argument("--yes", action="store_true", help="skip the props-off confirmation")
    ap.add_argument("--no-capture", action="store_true", help="link/params/arm test only")
    args = ap.parse_args()

    _say("=" * 78)
    _say("POLICY APP BENCH BRING-UP  (props off if this is the first run)")
    _say("=" * 78)
    if not args.yes:
        ans = input("Are the PROPELLERS REMOVED and is the vehicle on the bench? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            _say("aborted - remove the propellers first (this tool is not a flight tool).")
            return 2

    uri = _find_link(args.uri, scan=args.scan)
    _say(f"link    : {uri}")
    if not drone_link.dongle_preflight(uri):
        return 3

    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

    cf = Crazyflie(rw_cache=os.path.join(_PROJECT_ROOT, ".cf_cache"))
    bench = Bench(cf)
    windows: List[Tuple[str, float, float, int, float, str]] = []

    with SyncCrazyflie(uri, cf=cf) as scf:
        cf = scf.cf
        bench.attach()
        time.sleep(1.5)

        ban = "".join(bench.console).strip()
        if ban:
            _say("-" * 78)
            _say("console:")
            _say(ban[:1200])
            _say("-" * 78)

        # -- 1. is our app in this image? ---------------------------------------------
        shadow = bench.read_param("policy.shadow")
        if shadow is None:
            _say("*** `policy.shadow` is not readable: this is NOT the policy image.")
            _say("    flash Simulation/deploy/app_policy_controller/build/cf2.bin and retry.")
            return 1
        _say(f"app     : present (policy.shadow = {shadow})")

        for name in ("policy.thrust_scale", "policy.max_tilt_deg", "stabilizer.controller"):
            _say(f"  {name} = {bench.read_param(name)}")

        # -- 2. controller selection ---------------------------------------------------
        ctrl = bench.read_param("stabilizer.controller")
        if ctrl != "6":
            _say(f"setting stabilizer.controller: {ctrl} -> 6 (OutOfTree)")
            bench.set_param("stabilizer.controller", "6")
            _say(f"  now {bench.read_param('stabilizer.controller')}")
        else:
            _say("controller: already 6 (OutOfTree)")

        if str(shadow) not in ("1", "1.0"):
            _say("*** shadow mode is OFF. This tool refuses to continue: clear it only when")
            _say("    you are actually flying, and only after this capture has passed.")
            return 1

        # -- lighthouse: do the stations actually reach the deck? ------------------------
        # One second of checks that decides whether the next hour is a calibration job or
        # a flying job (see lighthouse_check.py for the standalone version).
        _say()
        try:
            from lighthouse_check import report as lh_report
            lh_report(cf)
        except Exception as exc:
            _say(f"  lighthouse: report failed ({exc})")

        # -- 3. arm / disarm, props off ------------------------------------------------
        _say()
        _say("arm/disarm test (shadow mode: the policy computes but commands nothing)")
        bench.send(CMD_ARM)
        # Poll instead of one sample: the app can arm and then disarm itself within a tick,
        # and we want to know WHICH happened and WHY.
        ever_armed = False
        st = {}
        for _ in range(12):
            time.sleep(0.1)
            st = bench.status_now() or st
            ever_armed = ever_armed or int(st.get("armed", 0)) == 1
        abort = int(st.get("abort", 0))
        _say(f"  after 0x01 ARM    : armed={st.get('armed')} (was armed: {ever_armed}) "
             f"mode={st.get('mode')} abort={abort} ({ABORT_NAMES.get(abort, '?')}) "
             f"z={float(st.get('z', 0.0)):.2f} vbat={float(st.get('vbat', 0.0)):.2f}")

        # The app REFUSES to fly from the ground on purpose: `policy_safety_check` disarms
        # with ABORT_Z whenever z leaves [POLICY_MIN_Z, POLICY_MAX_Z] = [0.04, 3.5] m, and on
        # a bench the estimate is ~0 - or meaningless until the Lighthouse geometry exists.
        # `armed -> auto-disarmed for z` is the envelope doing its job, so it counts as a
        # PASS here (and it is the one thing that CANNOT be verified on the ground).
        ground_refusal = (not ever_armed) and abort == 2 and bool(st)
        armed_ok = ever_armed or ground_refusal

        bench.send(CMD_DISARM)
        time.sleep(0.5)
        st2 = bench.status_now()
        _say(f"  after 0x02 DISARM : armed={st2.get('armed')} mode={st2.get('mode')}")
        disarmed_ok = int(st2.get("armed", 0)) == 0
        if ground_refusal:
            _say("  note: the app refused to arm because z is outside the flight envelope")
            _say("        - EXPECTED on the bench (it is designed not to fly from the")
            _say("        ground). The command path itself worked; a sustained arm needs the")
            _say("        Lighthouse fix or the vehicle in the air.")
        ok = armed_ok and disarmed_ok
        _say(f"  -> arm/disarm {'OK' if ok else '*** FAILED ***'} "
             f"(status packets: {'OK' if st else 'none'})")

        if args.no_capture:
            _say("\n--no-capture: stopping here.")
            return 0 if ok else 1

        # -- 4. shadow capture ---------------------------------------------------------
        _say()
        _say("=" * 78)
        _say("SHADOW CAPTURE - hold the vehicle in your hand and follow the prompts")
        _say("(the props must be off; nothing here commands a motor - shadow mode)")
        _say("=" * 78)
        lg, t0 = bench.start_log()
        try:
            for label, instruction, _axis, _sign, _why in MOTIONS:
                _say()
                _say(f">>> {instruction}")
                if label == "lift":
                    _say(f"    (starting in {2.0:.0f} s, then {args.seconds:.0f} s to do it)")
                else:
                    _say(f"    (starting in 2.0 s, then {args.seconds:.0f} s of that motion)")
                time.sleep(2.0)
                start = time.time() - t0
                time.sleep(args.seconds)
                windows.append((label, start, time.time() - t0, _axis, _sign, _why))
        except KeyboardInterrupt:
            _say("\ncapture interrupted - analysing what was recorded")
        finally:
            lg.stop()
            time.sleep(0.3)

    # -- 5. analysis -------------------------------------------------------------------
    return analyse(bench.rows, windows)


def analyse(rows: List[Tuple[float, Dict[str, float]]],
            windows: List[Tuple[str, float, float, int, float, str]]) -> int:
    if not rows:
        _say("no log data was received - is the log group `policy` in this image?")
        return 1

    keys = sorted({k for _t, d in rows for k in d})
    with open(_OUT_CSV, "w", encoding="utf-8") as fh:
        fh.write("t," + ",".join(keys) + "\n")
        for t, d in rows:
            fh.write(f"{t:.3f}," + ",".join(str(d.get(k, "")) for k in keys) + "\n")
    _say(f"\ncapture : {len(rows)} samples -> {os.path.relpath(_OUT_CSV, _PROJECT_ROOT)}")

    def mean_in(start: float, end: float, key: str) -> Optional[float]:
        vals = [d[key] for t, d in rows if start <= t <= end and key in d]
        return sum(vals) / len(vals) if vals else None

    def range_in(start: float, end: float, key: str) -> Optional[Tuple[float, float]]:
        vals = [d[key] for t, d in rows if start <= t <= end and key in d]
        return (min(vals), max(vals)) if vals else None

    _say()
    _say("--- sign check (sim frame: x = nose, y = left, z = up) ---")
    ok = True
    for label, start, end, axis, expect, why in windows:
        if axis < 0:
            rng = range_in(start, end, "stateEstimate.z")
            if rng is None:
                _say(f"  {label:<11}: stateEstimate.z not logged - cannot check z scale")
                ok = False
                continue
            span = rng[1] - rng[0]
            moved = span > 0.05
            ok = ok and moved
            _say(f"  {label:<11}: z ranged {rng[0]:.3f} .. {rng[1]:.3f} m (span {span:.3f})"
                 f"  ({'moves with the vehicle' if moved else '*** NO vertical change: z is stuck ***'})")
            continue
        name = ("gyro.x", "gyro.y", "gyro.z")[axis]
        m = mean_in(start, end, name)
        if m is None:
            _say(f"  {label:<11}: {name} not logged - cannot check {why}")
            ok = False
            continue
        good = (m > 0) == (expect > 0)
        ok = ok and good
        _say(f"  {label:<11}: {name} mean {m:+8.1f} deg/s  expected {'+' if expect > 0 else '-'}"
             f"  -> {'consistent with ' + why if good else '*** INVERTED: flip ' + why + ' ***'}")

    tilt = mean_in(rows[0][0], rows[-1][0], "policy.tilt_deg")
    vbat = mean_in(rows[0][0], rows[-1][0], "policy.vbat")
    _say()
    _say(f"policy  : tilt {tilt:.1f} deg, vbat {vbat:.2f} V"
         if tilt is not None and vbat is not None else "policy  : (no policy rows)")
    _say()
    if ok:
        _say("VERDICT: all sign assumptions confirmed - the SIGN_* block in controller_app.c "
             "is consistent with the hardware")
    else:
        _say("VERDICT: *** at least one check disagrees - fix the SIGN_* constant in "
             "controller_app.c, rebuild with build_app.sh, then re-run ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
