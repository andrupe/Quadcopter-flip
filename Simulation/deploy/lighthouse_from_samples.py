# -*- coding: utf-8 -*-
"""
Turn a Lighthouse SAMPLE file into base-station geometry and (optionally) write it to the drone.

WHY THIS EXISTS
---------------
cfclient's *Lighthouse positioning* tab is the documented way to estimate the installation
geometry, but it is a GUI flow - you cannot script it, diff it, or run it over SSH. The
computation behind that button is plain cflib, so this does exactly the same thing headlessly:

    samples.yaml  ->  LhGeoEstimationManager.estimate_geometry  ->  base-station poses
                  ->  LighthouseConfigWriter.write_and_store_config(geos=...)  ->  the drone

The two YAML formats are DIFFERENT things and are easy to confuse:

  * the SAMPLE file (`samples.yaml`, tag `!LhGeoInputContainerData`) is INPUT. It holds the
    poses the vehicle was carried to, each with the calibrated sweep angles the deck measured
    from every base station that was visible. 47 samples = 1 origin + 1 x-axis + 5 xy-plane
    + 16 xyz-space, plus `verification` samples that are deliberately NOT used for the fit
    and are therefore an honest held-out test.
  * the CONFIG file (`type: lighthouse_system_configuration`) is OUTPUT: geos/calibs/
    systemType, the thing that gets written to the drone.

`--point` in `lighthouse_check.py` is the independent end-to-end check of the result. Do not
trust this script's own error stats alone: those measure the fit, not the aeroplane.

USAGE
-----
    # solve and report; writes NOTHING (this is the default, on purpose)
    .venv/bin/python Simulation/deploy/lighthouse_from_samples.py samples.yaml

    # ...and write the geometry to the drone (persists across reboot)
    .venv/bin/python Simulation/deploy/lighthouse_from_samples.py samples.yaml --write

    # also save the solved config so it can be re-applied later without re-solving
    .venv/bin/python Simulation/deploy/lighthouse_from_samples.py samples.yaml \
        --save-config logs/lighthouse_config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Dict, Optional

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_PROJECT_ROOT, _HERE, os.path.join(_PROJECT_ROOT, "Simulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cflib.crtp  # noqa: E402
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # noqa: E402
from cflib.crazyflie.mem import LighthouseBsGeometry  # noqa: E402
from cflib.crazyflie.mem import MemoryElement  # noqa: E402
from cflib.localization.lighthouse_config_manager import (  # noqa: E402
    LighthouseConfigFileManager,
    LighthouseConfigWriter,
)
# Importing the manager is also what REGISTERS the `!LhGeoInputContainerData` YAML
# constructor (`yaml.add_constructor` runs at module import), so this import is load-bearing
# for loading the sample file - without it PyYAML cannot construct the tag.
from cflib.localization.lighthouse_geo_estimation_manager import (  # noqa: E402
    LhGeoInputContainerData,
    LhGeoEstimationManager,
)

import drone_link  # noqa: E402  (one place that knows where the drone is)


# A BACKSTOP, not a measured gate. We have exactly one sample set, and picking tight bounds
# from a single observation is how thresholds end up rejecting good data (see the notes on
# the Lighthouse fix gate). cflib's own `progress_is_ok` is the hard failure signal; these
# numbers only catch a solution that is obviously broken, and `--force` overrides them.
MAX_MEAN_ERR_M = 0.050
MAX_WORST_ERR_M = 0.100


def load_samples(path: str) -> LhGeoInputContainerData:
    """Load a cfclient sample file. Accepts both `{'data': ..., 'file_type_version': n}` and
    a bare container document (older dumps)."""
    with open(path, "r") as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)

    if isinstance(doc, dict) and "data" in doc:
        version = doc.get("file_type_version")
        if version is not None and int(version) != 1:
            raise SystemExit(f"ERROR: unsupported sample file version {version} (expected 1)")
        container = doc["data"]
    else:
        container = doc

    if not isinstance(container, LhGeoInputContainerData):
        raise SystemExit(f"ERROR: {path} is not a Lighthouse sample file "
                         f"(got {type(container).__name__})")
    return container


def describe_container(container: LhGeoInputContainerData) -> None:
    empty = container.EMPTY_POSE_SAMPLE
    print("samples")
    print(f"  sensor_positions : {np.asarray(container.sensor_positions).shape[0]} deck sensors")
    print(f"  origin           : {'1' if container.origin != empty else 'MISSING (required)'}")
    print(f"  x_axis           : {len(container.x_axis)}")
    print(f"  xy_plane         : {len(container.xy_plane)}")
    print(f"  xyz_space        : {len(container.xyz_space)}")
    print(f"  verification     : {len(container.verification)}  (held out of the fit)")


def solve(container: LhGeoInputContainerData):
    sol = LhGeoEstimationManager.estimate_geometry(container)
    # NOTE: `has_converged` stays False here and that is NOT a failure - `estimate_geometry`
    # never assigns it (the parent's progress_info/progress_is_ok are the real signals).
    print("solution")
    print(f"  progress         : {sol.progress_info}  (ok={sol.progress_is_ok})")
    if not sol.progress_is_ok:
        print(f"  FAILURE          : {sol.general_failure_info}")
    return sol


def report_positions(sol) -> None:
    poses = sol.bs_poses
    print(f"base stations ({len(poses)} solved)")
    for bs_id in sorted(poses):
        t = np.asarray(poses[bs_id].translation, dtype=np.float64)
        print(f"  BS{bs_id + 1}: x {t[0]:+.3f}  y {t[1]:+.3f}  z {t[2]:+.3f}   |p| {np.linalg.norm(t):.3f} m")
    # All stations at the same height is the strongest cheap sanity signal: a real mounting
    # is level, and a solve that tilts the whole constellation shows up here immediately.
    zs = [float(np.asarray(poses[b].translation)[2]) for b in poses]
    if zs:
        print(f"  height spread    : {max(zs) - min(zs):.3f} m (a level installation is ~0)")


def report_quality(sol) -> bool:
    """Print the fit and held-out errors. Returns False if the backstop is exceeded."""
    es, vs = sol.error_stats, sol.verification_stats
    print("quality")
    print(f"  fit        (n={len(sol.bs_poses)} bs): mean {es.mean * 1000:6.1f} mm  "
          f"max {es.max * 1000:6.1f} mm  std {es.std * 1000:5.1f} mm")
    print(f"  held out   (verification): mean {vs.mean * 1000:6.1f} mm  "
          f"max {vs.max * 1000:6.1f} mm  std {vs.std * 1000:5.1f} mm")
    ok = True
    if not sol.progress_is_ok:
        print("  VERDICT    : the solver did not complete - do NOT write this")
        ok = False
    elif es.mean > MAX_MEAN_ERR_M or es.max > MAX_WORST_ERR_M:
        print(f"  VERDICT    : fit error exceeds the backstop "
              f"(mean<{MAX_MEAN_ERR_M * 1000:.0f} mm, max<{MAX_WORST_ERR_M * 1000:.0f} mm)")
        ok = False
    else:
        print("  VERDICT    : sane (held-out error is the honest number to trust)")
    return ok


def read_stored_geometry(cf, ids=(0, 1, 2, 3)) -> Dict[int, str]:
    """Read back what the drone currently holds, so the change is visible."""
    mems = cf.mem.get_mems(MemoryElement.TYPE_LH)
    if not mems:
        return {}
    out: Dict[int, str] = {}
    for bs_id in ids:
        box: Dict[int, object] = {}
        mems[0].read_geo_data(bs_id, lambda _m, d, i=bs_id, b=box: b.update({i: d}))
        time.sleep(0.9)
        g = box.get(bs_id)
        if g is None:
            out[bs_id] = "no reply"
        elif not getattr(g, "valid", False):
            out[bs_id] = "invalid"
        else:
            o = np.asarray(g.origin, dtype=np.float64)
            out[bs_id] = f"({o[0]:+.3f}, {o[1]:+.3f}, {o[2]:+.3f})"
    return out


def write_geometry(uri: str, sol, timeout: float = 60.0) -> bool:
    """
    Write only the GEOMETRY (cfclient does the same: `write_and_store_config(cb, geos=...)`).

    Calibration is deliberately NOT passed: the per-station sweep curves are a property of
    the base stations and are already on the drone (and `bsCalVal` shows which are valid).
    Passing calibs=None means "leave that data alone" - writing None-valued calibration would
    invalidate it.
    """
    cflib.crtp.init_drivers()
    done = threading.Event()
    result = {"ok": False}

    def stored(success):
        result["ok"] = bool(success)
        done.set()

    with SyncCrazyflie(uri) as scf:
        print("\ncurrent stored geometry")
        before = read_stored_geometry(scf.cf)
        for bs_id in sorted(before):
            print(f"  BS{bs_id + 1}: {before[bs_id]}")

        geo_dict: Dict[int, LighthouseBsGeometry] = {}
        for bs_id, pose in sol.bs_poses.items():
            geo = LighthouseBsGeometry()          # exactly the cfclient conversion
            geo.origin = np.asarray(pose.translation).tolist()
            geo.rotation_matrix = np.asarray(pose.rot_matrix).tolist()
            geo.valid = True
            geo_dict[bs_id] = geo

        print(f"\nwriting geometry for {len(geo_dict)} base station(s) ...")
        writer = LighthouseConfigWriter(scf.cf)
        writer.write_and_store_config(stored, geos=geo_dict)
        if not done.wait(timeout):
            print(f"  TIMED OUT after {timeout:.0f}s (the write may still be running)")
            return False
        print(f"  write_and_store_config callback -> {'success' if result['ok'] else 'FAILED'}")

        # Read back so the success is EVIDENCE rather than a callback's opinion. The callback
        # reports the PERSIST step and has been seen to say FAILED while the data was in fact
        # correct in every base station - so the drone's own copy is what decides this, and
        # the two are reported separately rather than conflated.
        print("\nnewly stored geometry (read back from the drone)")
        after = read_stored_geometry(scf.cf)
        matches = 0
        for bs_id in sorted(after):
            want = geo_dict.get(bs_id)
            want_s = (f"({np.asarray(want.origin)[0]:+.3f}, {np.asarray(want.origin)[1]:+.3f}, "
                      f"{np.asarray(want.origin)[2]:+.3f})") if want is not None else "-"
            got = after[bs_id]
            ok = "OK" if got == want_s else "MISMATCH"
            if got == want_s:
                matches += 1
            print(f"  BS{bs_id + 1}: {got:28s} wrote {want_s:28s} {ok}")
        verified = matches == len(geo_dict) and bool(geo_dict)
        print(f"  read-back verification: {matches}/{len(geo_dict)} "
              f"{'MATCH - the drone holds this geometry' if verified else 'DOES NOT MATCH'}")

    return verified


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate Lighthouse geometry from a sample file")
    ap.add_argument("samples", help="cfclient sample file (e.g. samples.yaml)")
    ap.add_argument("--write", action="store_true",
                    help="write the solved geometry to the drone (default: report only)")
    ap.add_argument("--uri", default=None, help="override the remembered drone URI")
    ap.add_argument("--force", action="store_true", help="write even if the backstop trips")
    ap.add_argument("--save-config", default=None, metavar="PATH",
                    help="also save the solved config as a lighthouse_system_configuration file")
    args = ap.parse_args()

    container = load_samples(args.samples)
    describe_container(container)
    print()
    sol = solve(container)
    report_positions(sol)
    print()
    sane = report_quality(sol)

    if args.save_config and sol.bs_poses:
        geos = {}
        for bs_id, pose in sol.bs_poses.items():
            geo = LighthouseBsGeometry()
            geo.origin = np.asarray(pose.translation).tolist()
            geo.rotation_matrix = np.asarray(pose.rot_matrix).tolist()
            geo.valid = True
            geos[bs_id] = geo
        # calibs={} on purpose: this file carries geometry only, and LighthouseConfigFileManager
        # skips invalid entries, so an empty calibs dict writes no calibration at all.
        LighthouseConfigFileManager.write(args.save_config, geos=geos, calibs={},
                                          system_type=LighthouseConfigFileManager.SYSTEM_TYPE_V2)
        print(f"\nsaved config -> {args.save_config}")

    if not args.write:
        print("\n(report only - re-run with --write to apply this to the drone)")
        return 0 if sane else 1

    if not sane and not args.force:
        print("\nREFUSING to write: the backstop tripped. Fix the samples, or pass --force.")
        return 1

    uri = drone_link.preferred_uri(args.uri)
    print(f"\nuri: {uri}  ({drone_link.source()})")
    if not drone_link.dongle_preflight(uri):
        return 3

    ok = write_geometry(uri, sol)
    if ok:
        print("\nNext: verify against a TAPE-MEASURED mark - a fit this good can still be "
              "systematically offset if the origin sample was placed badly:")
        print("  .venv/bin/python Simulation/deploy/lighthouse_check.py --point X Y Z")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
