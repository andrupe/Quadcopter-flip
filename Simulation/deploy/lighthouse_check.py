#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lighthouse: is it working, does it need calibrating, and is the calibration any good?

    .venv/bin/python Simulation/deploy/lighthouse_check.py                  # report + verdict
    .venv/bin/python Simulation/deploy/lighthouse_check.py --monitor 30     # live, 30 s
    .venv/bin/python Simulation/deploy/lighthouse_check.py --point 0 0 1.0  # verify at a mark
    .venv/bin/python Simulation/deploy/lighthouse_check.py --set systemType=2 --set method=1
    .venv/bin/python Simulation/deploy/lighthouse_check.py --reset-calib    # wipe geometry

WHY THIS EXISTS
---------------
The base stations need no calibration - they are factory devices that broadcast. What needs
estimating once per room is the INSTALLATION GEOMETRY (where each station sits and points),
and that estimate lives in YOUR drone. A drone that came from somewhere else carries the
previous room's geometry, which does not fail loudly: it produces a plausible, wrong
position. So the checks below come first, and the verification at the end is what actually
closes the job.

Run the estimation itself in cfclient (its Lighthouse tab drives the whole sequence: put the
deck in estimation mode, carry the drone around the volume, write the result). This tool is
the GATE before it and the VERDICT after it. Everything runs over the Crazyradio PA
(`radio://0/80/2M/E7E7E7E7E7`, the factory address and the default here).

WHAT IT READS (names taken from the firmware, all in the `lighthouse` group)
  PARAMS  systemType (1=V1, 2=V2) | method (0=CrossingBeam, 1=Sweep in EKF) |
          bsAvailable (RO bitmap of stations the system can use) | bsCalibReset
  LOGS    status        0 = nothing received | 1 = stations seen but geometry/calibration
                        missing | 2 = base-station data is going to the estimator
          bsReceive/bsActive     which stations are being received / used
          bsGeoVal/bsCalVal      which stations have VALID geometry / calibration
          bsCalCon/bsCalUd       convergence and "updated" bookkeeping
  All the bitmaps are per-station-channel bit fields: bit 0 is channel 1.

Everything is discovered from the TOC at runtime, so a firmware that renames or drops a
variable degrades into a clear message instead of a traceback.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CSV_PATH = os.path.join(_PROJECT_ROOT, "logs", "lighthouse_log.csv")

# Read in this order so the report reads top-down.
# `deck.bcLighthouse4` is the Lighthouse deck driver's own `static bool isInit`
# (`firmware/src/deck/drivers/src/lighthouse.c`), set at the END of its `init` - and that
# `init` is only called for decks the one-wire scan actually enumerated (`deck.c::deckInit`
# loops over `deckCount()`). So it is the ONE field that separates "the deck is not there"
# from every station- and calibration-level problem. Read it first; see `_advice`.
PARAM_NAMES = ("deck.bcLighthouse4", "lighthouse.systemType", "lighthouse.method",
               "lighthouse.bsAvailable",
               # the config THRESHOLDS: if these are too strict the deck hears pulses and
               # throws the angles away, which looks exactly like "no base stations".
               "lighthouse.sweepStd", "lighthouse.sweepStd2",
               "lighthouse.fwdToEstimator", "lighthouse.lh2maxRate")
LOG_NAMES = ("lighthouse.status", "lighthouse.bsReceive", "lighthouse.bsActive",
             "lighthouse.bsGeoVal", "lighthouse.bsCalVal", "lighthouse.bsCalCon",
             "lighthouse.bsCalUd", "lighthouse.comSync")
# The CAUSE signals, as opposed to the outcome flags above. `lighthouse.width*` are pulse
# widths and `bs0Rt`/`bs1Rt` are per-station frame rates: non-zero PROVES light is reaching
# the deck's photodiodes. They are LOG_ADD_DEBUG in the firmware, so they may be absent -
# `_start_blocks` reports that instead of pretending they read 0.
CAUSE_LOG_NAMES = ("lighthouse.validAngles",
                   "lighthouse.width0", "lighthouse.width1", "lighthouse.width2",
                   "lighthouse.bs0Rt", "lighthouse.bs1Rt",
                   # The PIPELINE RATES. `cycleRt`/`frameRt` are the deck SPI cycle and
                   # frame rates and `preThRt`/`postThRt` bracket throttleLh2Samples(), so
                   # if the main CPU were starving the Lighthouse task (the "are we
                   # overloading the CPU" question) these are what would sag. `disProb` is
                   # the throttle's own discard probability, so a high value means samples
                   # are being thrown away by lighthouse.lh2maxRate (default 50/s).
                   "lighthouse.cycleRt", "lighthouse.frameRt",
                   "lighthouse.preThRt", "lighthouse.postThRt",
                   "lighthouse.disProb")
EST_NAMES = ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z")

STATUS_TEXT = {
    0: "no base stations received",
    1: "stations received but geometry/calibration MISSING",
    2: "base-station data is reaching the estimator",
}


# ======================================================================================
# plumbing (self-contained: this file is also imported by bench_bringup.py)
# ======================================================================================
def read_param(cf, name: str, timeout: float = 2.0) -> Optional[str]:
    getter = getattr(cf.param, "get_value", None)
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
        cf.param.add_update_callback(group=name.split(".")[0], name=name.split(".", 1)[1], cb=cb)
        cf.param.request_param_update(name)
    except Exception:
        return None
    done.wait(timeout)
    return box.get("v")


def _toc_names(cf, kind: str) -> Dict[str, str]:
    """{complete_name: type-string} for a group prefix, straight from the TOC."""
    toc = getattr(getattr(cf, kind), "toc", None)
    table = getattr(toc, "toc", {}) if toc is not None else {}
    out: Dict[str, str] = {}
    for group, elems in table.items():
        for name, elem in elems.items():
            out[f"{group}.{name}"] = str(getattr(elem, "type", "?"))
    return out


def available(cf) -> Tuple[List[str], List[str]]:
    logs = _toc_names(cf, "log")
    params = _toc_names(cf, "param")
    return ([n for n in LOG_NAMES if n in logs], [n for n in PARAM_NAMES if n in params])


def self_fmt(d: Dict[str, float], name: str, spec: str, width: int) -> str:
    """Format d[name], or `width` spaces when the firmware does not publish it."""
    v = d.get(name)
    if v is None:
        return " " * width
    try:
        return spec.format(float(v))
    except Exception:
        return " " * width


def cause(cf, seconds: float) -> int:
    """
    Answer "is it the SETTINGS or is it the OPTICS?" instead of guessing.

    `report()` says WHAT is missing. These variables say WHY:
      * `lighthouse.width0/1/2` are PULSE WIDTHS and `lighthouse.bs0Rt`/`bs1Rt` are
        per-station frame rates - non-zero PROVES light is reaching the photodiodes;
      * `lighthouse.validAngles` says whether the pulse processor made angles of it.

    So the verdict is mechanical:
      light arriving but bsReceive == 0 -> pulses reach the sensors and are REJECTED. That
        is a calibration/threshold question - genuinely SETTINGS. `lighthouse.sweepStd` and
        `sweepStd2` are the angle-acceptance thresholds and are printed by report().
      nothing arriving at all -> aim, height, occlusion, or that station is off. NO
        PARAMETER CAN CREATE PHOTONS, so this is not a settings problem.

    RUN IT TWICE - once with the drone on the bench, once with someone HOLDING it at the
    intended flight altitude (~1.2 m) - and compare. The altitude dependence is the point.
    """
    names = list(LOG_NAMES) + list(CAUSE_LOG_NAMES) + ["stateEstimate.z"]
    blocks, _keep, missing = _start_blocks(cf, names, 100, "lhcaus")
    if not blocks:
        print("cause   : nothing to log")
        return 1
    if missing:
        print("cause   : not published here (LOG_ADD_DEBUG, can be compiled out): "
              + ", ".join(missing))
    rows: List[Dict[str, float]] = []

    def cb(_ts, d, _lg):
        rows.append({k: float(v) for k, v in d.items()})

    for lg in blocks:
        lg.data_received_cb.add_callback(cb)
    for lg in blocks:
        lg.start()
    print(f"cause   : sampling {seconds:.0f} s - hold the drone at the test altitude NOW "
          f"if that is what you are testing")
    print("  t       z   status recv vAng  width0 width1 width2  bs0Rt  bs1Rt"
          "   | cycles/frames pre/post-throttle disProb")
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            time.sleep(0.5)
            if rows:
                d = rows[-1]
                print(f"  {time.time() - t0:5.1f} "
                      + self_fmt(d, "stateEstimate.z", "{:>6.2f}", 6) + " "
                      + self_fmt(d, "lighthouse.status", "{:>6.0f}", 6) + " "
                      + f"{len(bits(d.get('lighthouse.bsReceive', 0))):>4} "
                      + self_fmt(d, "lighthouse.validAngles", "{:>4.0f}", 4) + " "
                      + self_fmt(d, "lighthouse.width0", "{:>6.0f}", 6) + " "
                      + self_fmt(d, "lighthouse.width1", "{:>6.0f}", 6) + " "
                      + self_fmt(d, "lighthouse.width2", "{:>6.0f}", 6) + " "
                      + self_fmt(d, "lighthouse.bs0Rt", "{:>6.1f}", 6) + " "
                      + self_fmt(d, "lighthouse.bs1Rt", "{:>6.1f}", 6)
                      + "   | "
                      + self_fmt(d, "lighthouse.cycleRt", "{:.0f}", 4) + "/"
                      + self_fmt(d, "lighthouse.frameRt", "{:.0f}", 4) + " "
                      + self_fmt(d, "lighthouse.preThRt", "{:.0f}", 4) + "/"
                      + self_fmt(d, "lighthouse.postThRt", "{:.0f}", 4) + " "
                      + self_fmt(d, "lighthouse.disProb", "{:.2f}", 4))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        for lg in blocks:
            lg.stop()

    def peak(name: str) -> Optional[float]:
        vals = [r[name] for r in rows if name in r]
        return max(vals) if vals else None

    print()
    if not rows:
        print("cause   : NO SAMPLES - the log never started")
        return 1
    widths = [peak(f"lighthouse.width{i}") for i in range(3)]
    rates = [peak("lighthouse.bs0Rt"), peak("lighthouse.bs1Rt")]
    light = [v for v in widths + rates if v is not None]
    recv = peak("lighthouse.bsReceive")
    vang = peak("lighthouse.validAngles")
    n_recv = len(bits(recv)) if recv is not None else 0
    # The CPU-starvation question, answered by measurement rather than arithmetic.
    cyc = peak("lighthouse.cycleRt")
    frm = peak("lighthouse.frameRt")
    pre = peak("lighthouse.preThRt")
    post = peak("lighthouse.postThRt")
    dp = peak("lighthouse.disProb")
    if cyc is not None or frm is not None or pre is not None:
        print(f"cause   : pipeline rates -> cycle {cyc} / frame {frm} / "
              f"pre-throttle {pre} / post-throttle {post}, discardProb {dp}")
        if cyc is not None and cyc > 0:
            print("          the deck SPI cycle is RUNNING, so the Lighthouse task is "
                  "being serviced. A starved CPU would show this at 0.")
        elif cyc == 0:
            print("          *** cycle rate is ZERO - the Lighthouse task is NOT being "
                  "serviced at all. That is a CPU/starvation symptom. ***")
        if pre and post is not None and post < 0.8 * pre:
            print(f"          NOTE: {100 * (1 - post / pre):.0f}% of samples are being "
                  "thrown away by throttleLh2Samples - raise `lighthouse.lh2maxRate` "
                  "(default 50/s).")
    else:
        print("cause   : pipeline rates not published here (LOG_ADD_DEBUG), so CPU "
              "starvation cannot be measured directly - judge from `bsReceive`.")
    if not light and vang is None:
        print("cause   : INCONCLUSIVE - this firmware publishes neither the pulse widths nor "
              "the per-station rates, so light arrival cannot be measured directly. Judge "
              "from `bsReceive` and the geometry instead.")
        return 2
    shown = " ".join(f"{v:.0f}" for v in light) if light else "none"
    print(f"cause   : peak width/rate evidence = [{shown}]")
    print(f"          peak bsReceive = {n_recv} station(s), peak validAngles = "
          f"{'-' if vang is None else f'{vang:.0f}'}")
    if n_recv > 0:
        print("cause   : PULSES ARE RECEIVED at this pose, so it is NOT an optics problem "
              "here. If it fails higher up, re-run holding the drone there.")
    elif light and any(v > 0 for v in light):
        print("cause   : *** LIGHT ARRIVES BUT NO STATION IS DECODED. Pulses reach the "
              "photodiodes and are being rejected - that IS a settings/calibration "
              "question. Check lighthouse.sweepStd / sweepStd2 (angle-acceptance "
              "thresholds, printed by report) and the stored geometry. ***")
    else:
        print("cause   : *** NO LIGHT AT ALL - no pulse width, no station frame rate. No "
              "parameter can create photons. This is aim / height / occlusion, or that "
              "base station is switched off. ***")
    return 0


def _log_var_types(cf) -> Dict[str, Tuple[str, int]]:
    """{complete_name: (c_type, payload_bytes)} straight from the log TOC.

    Types come from the TOC rather than being assumed, because the byte budget is what
    decides the block split and cflib enforces 26 B/block.
    """
    from cflib.crazyflie.log import LogTocElement

    toc = getattr(getattr(cf, "log", None), "toc", None)
    table = getattr(toc, "toc", {}) if toc is not None else {}
    out: Dict[str, Tuple[str, int]] = {}
    for group, elems in table.items():
        for name, elem in elems.items():
            ctype = getattr(elem, "ctype", None) or getattr(elem, "type", None)
            try:
                size = int(LogTocElement.get_size_from_id(
                    LogTocElement.get_id_from_cstring(str(ctype))))
            except Exception:
                continue
            if size:
                out[f"{group}.{name}"] = (str(ctype), size)
    return out


def _start_blocks(cf, names: List[str], period_ms: int = 100, label: str = "lh"):
    """Build (but do not start) LogConfig blocks, packed greedily into <= 26 B each.

    A SINGLE block with a per-variable `try/except add_variable` does NOT work: the call
    only queues a name in `default_fetch_as` and cannot fail, while `add_config()` is where
    the byte limit is enforced and it raises `AttributeError`. So ONE variable too many
    silently takes the whole capture with it. Names the firmware does not publish (the
    LOG_ADD_DEBUG ones can be compiled out) are returned as `missing` instead of pretending
    they read 0.
    """
    from cflib.crazyflie.log import LogConfig

    types = _log_var_types(cf)
    keep = [n for n in names if n in types]
    missing = [n for n in names if n not in types]
    chunks: List[List[str]] = []
    cur: List[str] = []
    used = 0
    for n in keep:
        if cur and used + types[n][1] > LogConfig.MAX_LEN:
            chunks.append(cur)
            cur, used = [], 0
        cur.append(n)
        used += types[n][1]
    if cur:
        chunks.append(cur)
    blocks = []
    for i, chunk in enumerate(chunks):
        lg = LogConfig(name=f"{label}{i}", period_in_ms=period_ms)
        for n in chunk:
            lg.add_variable(n, types[n][0])
        cf.log.add_config(lg)
        blocks.append(lg)
    return blocks, keep, missing


def sample_log(cf, names: List[str], seconds: float) -> Dict[str, float]:
    """One short multi-block capture; returns the last sample of each variable."""
    if not names:
        return {}
    want = list(names) + [v for v in EST_NAMES if v in _log_var_types(cf)]
    blocks, _keep, _missing = _start_blocks(cf, want, 100, "lhcheck")
    data: Dict[str, float] = {}
    done = threading.Event()

    def cb(_ts, d, _lg):
        data.update({k: float(v) for k, v in d.items()})
        done.set()

    for lg in blocks:
        lg.data_received_cb.add_callback(cb)
    for lg in blocks:
        lg.start()
    done.wait(seconds)
    for lg in blocks:
        lg.stop()
    time.sleep(0.1)
    return data


def bits(value: float, width: int = 16) -> List[int]:
    v = int(value)
    return [i + 1 for i in range(width) if (v >> i) & 1]


def _as_int(value: Any) -> Optional[int]:
    """
    Coerce a param/log value to int, or None if it is not a number.

    `read_param` returns `str(...)`, so a FAILED read yields the string "None" - which an
    `is not None` test happily accepts and `int()` then explodes on. A diagnostic must never
    die from the very condition it is meant to report, so coerce defensively here.
    """
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _stations_line(label: str, value: Optional[float]) -> str:
    if value is None:
        return f"    {label:<12}: (not in this firmware)"
    b = bits(value)
    return f"    {label:<12}: {value:>5.0f}   channels {b if b else '-'}"


# ======================================================================================
# the report
# ======================================================================================
def report(cf, verbose: bool = True) -> Dict[str, object]:
    """Print the Lighthouse state and return it as a dict (bench_bringup.py reuses this)."""
    log_names, param_names = available(cf)
    if not log_names and not param_names:
        print("  lighthouse: NOT PRESENT in this firmware (no lighthouse.* in the TOC)")
        return {}

    params = {n: read_param(cf, n) for n in param_names}
    data = sample_log(cf, log_names, 0.7)

    status_raw = data.get("lighthouse.status")
    status = int(status_raw) if status_raw is not None else None
    out: Dict[str, object] = {"params": params, "log": data, "status": status}

    if verbose:
        print("  LIGHTHOUSE")
        deck_flag = _as_int(params.get("deck.bcLighthouse4"))
        if deck_flag is not None:
            state = "detected and initialised" if deck_flag else \
                    "NOT DETECTED - the deck driver never initialised"
            print(f"    {'deck':<12}: {deck_flag}  ({state})")
        for n in PARAM_NAMES:
            if n in params and not n.startswith("deck."):
                label = n.split(".", 1)[1]
                note = ("   <- CONFIGURED from systemType, NOT a visibility reading"
                        if label == "bsAvailable" else "")
                print(f"    {label:<12}: {params[n]}{note}")
        if status is not None:
            print(f"    status      : {status}  ({STATUS_TEXT.get(status, 'unknown')})")
        for key, label in (("lighthouse.bsReceive", "receiving"),
                           ("lighthouse.bsActive", "active"),
                           ("lighthouse.bsGeoVal", "geo valid"),
                           ("lighthouse.bsCalVal", "cal valid"),
                           ("lighthouse.bsCalUd", "cal updated")):
            _v = data.get(key)
            bits_ = bits(_v) if _v is not None else None
            if _v is None:
                print(f"    {label:<12}: (not in this firmware)")
            else:
                print(f"    {label:<12}: {_v:>5.0f}   channels {bits_ if bits_ else '-'}")
        if status is not None:
            print(f"    next step   : {_advice(status, data, params)}")

    out["advice"] = _advice(status, data, params) if status is not None else ""
    return out


def _advice(status: Optional[int], data: Dict[str, float],
            params: Optional[Mapping[str, Any]] = None) -> str:
    """
    What to do next.

    `lighthouse.status` is the ESTIMATOR's view of the world. The field that says what the
    deck can actually HEAR is `lighthouse.bsReceive` - `lighthouse.bsAvailable` is a trap:
    it is CONFIGURED from `systemType` alone (`lighthouseUpdateSystemType()` sets all bits
    for V2), so it reads 15 even with every base station switched off. They disagree in a
    common and confusing way - a station list that looks full but status 0 - and reading
    the wrong one turns a "no base stations" problem into a bogus calibration job.
    """
    if status is None:
        return "status not available - cannot judge"
    # Outranks every station- and calibration-level diagnosis: if the deck driver never
    # initialised there are no receivers at all, so advice about powering base stations or
    # recalibrating geometry is wrong and misleading.
    deck_flag = _as_int((params or {}).get("deck.bcLighthouse4"))
    if deck_flag is not None and not deck_flag:
        return ("THE LIGHTHOUSE DECK IS NOT DETECTED (`deck.bcLighthouse4` == 0): its driver's "
                "init never ran, so the BASE STATIONS, the geometry and the bitstream are all "
                "irrelevant until this is fixed. The deck was not enumerated on the one-wire "
                "bus - reseat it on the expansion header and check the contact. This is NOT "
                "the bitstream fault: that one reads `deck.bcLighthouse4` == 1 with "
                "`LHFL: Bitstream CRC32 ... [FAIL]` in the boot console")
    if status == 0:
        # bsReceive, NOT bsAvailable: the latter is merely the configured V2 bitmask.
        seen_raw = _as_int(data.get("lighthouse.bsReceive"))
        if seen_raw is None:
            seen_raw = _as_int((params or {}).get("lighthouse.bsReceive"))
        seen = bits(seen_raw) if seen_raw is not None else []
        if seen:
            return (f"stations {seen} ARE heard by the deck, but the estimator has no "
                    "geometry -> THIS DRONE NEEDS CALIBRATING: run the estimation in "
                    "cfclient's Lighthouse tab (motion, not flight - props off, carry it "
                    "slowly around the volume)")
        return ("the deck is hearing NOTHING (`bsReceive` = 0 - no base-station pulses at "
                "all). Check the stations are POWERED, are actually v2 (matching "
                "`lighthouse.systemType`), and have line of sight to the deck. Do NOT use "
                "`bsAvailable` as evidence here - it is CONFIGURED from the system type and "
                "reads 15 for v2 even with every station switched off")
    if status == 1:
        return ("stations seen but geometry/calibration missing -> THIS DRONE NEEDS "
                "CALIBRATING: run the estimation in cfclient's Lighthouse tab "
                "(it needs motion, not flight - carry it around, props off)")
    geo = bits(data.get("lighthouse.bsGeoVal", 0.0))
    recv = bits(data.get("lighthouse.bsReceive", 0.0))
    if geo and set(recv) - set(geo):
        return (f"stations {sorted(set(recv) - set(geo))} are received but have no valid "
                "geometry -> recalibrate (consider `--reset-calib` first)")
    return ("geometry is in use -> VERIFY it: put the drone at a measured point and run "
            "`--point X Y Z` (a -same- systematic offset means a bad geometry)")


# ======================================================================================
# the steps
# ======================================================================================
def verify_point(cf, point: Tuple[float, float, float], hold: float,
                 tolerance: float) -> int:
    """Sample the estimate while the vehicle sits at a TAPE-MEASURED position."""
    log_names, _ = available(cf)
    need = [n for n in EST_NAMES if n in _toc_names(cf, "log")]
    if len(need) != 3:
        print("stateEstimate.x/y/z are not in this firmware's TOC - cannot verify")
        return 1
    print(f"hold the vehicle at ({point[0]:+.3f}, {point[1]:+.3f}, {point[2]:+.3f}) m "
          f"for {hold:.0f} s ...")
    data = sample_log(cf, log_names, hold)
    est = (data.get("stateEstimate.x"), data.get("stateEstimate.y"),
           data.get("stateEstimate.z"))
    if any(v is None for v in est):
        print("no estimate samples arrived - is an estimator running?")
        return 1
    # The `any(... is None)` guard above already handled the missing case; filtering here is
    # what actually narrows the type (an `any()` over a tuple does not).
    est_xyz = tuple(float(v) for v in est if v is not None)
    d = tuple(e - p for e, p in zip(est_xyz, point))
    err = math.sqrt(sum(v * v for v in d))
    print(f"  estimate : {est_xyz[0]:+.3f} {est_xyz[1]:+.3f} {est_xyz[2]:+.3f} m")
    print(f"  error    : dx {d[0]:+.3f}  dy {d[1]:+.3f}  dz {d[2]:+.3f}  |e| {err:.3f} m")
    ok = err <= tolerance
    print(f"  verdict  : {'OK' if ok else 'FAIL'} (tolerance {tolerance:.2f} m)")
    if not ok:
        print("  a uniform offset in one axis (or a rotation) is a GEOMETRY problem, not a "
              "tuning one: re-estimate the base stations")
    return 0 if ok else 1


def monitor(cf, seconds: float) -> int:
    """Live status + estimate, and a CSV next to the other logs."""
    log_names, _ = available(cf)
    need = log_names + [n for n in EST_NAMES if n in _toc_names(cf, "log")]
    if not need:
        print("nothing to monitor")
        return 1
    blocks, keep, _missing = _start_blocks(cf, need, 100, "lhmon")
    fh = open(CSV_PATH, "w", encoding="utf-8")
    fh.write("t," + ",".join(keep) + "\n")
    t0 = time.time()
    print(f"monitoring {seconds:.0f} s -> {os.path.relpath(CSV_PATH, _PROJECT_ROOT)}")
    print("  t      status  recv  geo   x       y       z")
    latest: Dict[str, float] = {}

    def cb(_ts, d, _lg):
        latest.update({k: float(v) for k, v in d.items()})
        fh.write(f"{time.time() - t0:.3f}," + ",".join(str(d.get(k, "")) for k in keep) + "\n")

    for lg in blocks:
        lg.data_received_cb.add_callback(cb)
    for lg in blocks:
        lg.start()
    try:
        while time.time() - t0 < seconds:
            time.sleep(0.5)
            x, y, z = (latest.get("stateEstimate.x"), latest.get("stateEstimate.y"),
                       latest.get("stateEstimate.z"))
            print(f"  {time.time() - t0:5.1f}  {int(latest.get('lighthouse.status', -1)):>6}  "
                  f"{len(bits(latest.get('lighthouse.bsReceive', 0))):>4}  "
                  f"{len(bits(latest.get('lighthouse.bsGeoVal', 0))):>4}  "
                  f"{f'{x:+.3f}' if x is not None else '  -   '} "
                  f"{f'{y:+.3f}' if y is not None else '  -   '} "
                  f"{f'{z:+.3f}' if z is not None else '  -   '}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        for lg in blocks:
            lg.stop()
        fh.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Lighthouse status / calibration gate / verification")
    ap.add_argument("--uri", default=None,
                    help="cflib URI (default: radio://0/80/2M/E7E7E7E7E7)")
    ap.add_argument("--scan", action="store_true",
                    help="scan radio channels/datarates for the address (10-30 s)")
    ap.add_argument("--monitor", type=float, default=0.0, help="live monitor for N seconds")
    ap.add_argument("--point", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                    help="verify the estimate against a tape-measured position")
    ap.add_argument("--hold", type=float, default=5.0, help="seconds to average at --point")
    ap.add_argument("--tolerance", type=float, default=0.15, help="--point pass threshold (m)")
    ap.add_argument("--cause", type=float, default=0.0,
                    help="diagnose WHY stations are missing: sample pulse widths, "
                         "per-station frame rates and the valid-angle counter for N seconds")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="write a param, e.g. --set systemType=2 (repeatable; group "
                         "`lighthouse` is assumed unless a dot is given)")
    ap.add_argument("--reset-calib", action="store_true",
                    help="write lighthouse.bsCalibReset=1 (the geometry must then be "
                         "re-estimated - use when the drone came from another room)")
    args = ap.parse_args()

    from bench_bringup import _find_link
    from drone_link import dongle_preflight
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

    uri = _find_link(args.uri, scan=args.scan)
    print(f"link    : {uri}")
    if not dongle_preflight(uri):
        return 3
    cf = Crazyflie(rw_cache=os.path.join(_PROJECT_ROOT, ".cf_cache"))
    rc = 0
    with SyncCrazyflie(uri, cf=cf) as scf:
        cf = scf.cf
        time.sleep(1.0)

        if args.set:
            for item in args.set:
                name, _, value = item.partition("=")
                if "." not in name:
                    name = f"lighthouse.{name}"
                print(f"set     : {name} = {value}")
                cf.param.set_value(name, value)
                time.sleep(0.2)
                print(f"          now {read_param(cf, name)}")

        if args.reset_calib:
            print("reset   : lighthouse.bsCalibReset = 1  (the stored geometry is now "
                  "marked for re-estimation)")
            cf.param.set_value("lighthouse.bsCalibReset", "1")
            time.sleep(0.3)

        state = report(cf)

        if args.monitor > 0:
            print()
            rc = monitor(cf, args.monitor) or rc
        if args.cause > 0:
            print()
            rc = cause(cf, args.cause) or rc
        if args.point is not None:
            print()
            rc = verify_point(cf, tuple(args.point), args.hold, args.tolerance) or rc

        if (not args.set and not args.reset_calib and not args.monitor
                and args.point is None and args.cause <= 0):
            st = state.get("status")
            if st == 1:
                rc = 1                     # stations seen but no usable geometry: not ready
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
