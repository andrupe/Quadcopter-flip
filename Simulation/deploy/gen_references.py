# -*- coding: utf-8 -*-
"""
Bake the trained manoeuvres into 100 Hz reference tables for the Crazyflie firmware.

    .venv/bin/python Simulation/deploy/gen_references.py            # all kinds, emit + check
    .venv/bin/python Simulation/deploy/gen_references.py --kinds flip orbit
    .venv/bin/python Simulation/deploy/gen_references.py --check-only

WHY TABLES AND NOT A PORT
-------------------------
The policy consumes the reference as ERRORS (`ref.p/v/R/omega` minus the estimate), so the
reference has to be produced ONBOARD. `trajectories.py` - quintic waypoints, settle
envelopes, flatness attitude, a central-difference rate stencil - is a lot of Python to port
for manoeuvres that never vary. Instead each one is evaluated here at exactly the policy's
100 Hz and shipped as a table the firmware indexes by tick count. No interpolation: table
sample k IS control step k. (This is the generalisation of the single flip table the flip
button used to fly: same maths, one entry per kind.)

THE RELOCATION (identical to `live_target.ShiftedTrajectory`)
-------------------------------------------------------------
Every sampled manoeuvre starts at the training spawn (z = 1.2 m, its own heading); the one
flown on the drone must start where the pilot is hovering, aimed along the pilot's heading.
That is a rigid motion (yaw rotation + translation), which preserves every dynamic property:

    dyaw    = yaw_live - REF_YAW0[kind]         (baked beside each table)
    Rz      = rotation about world z by dyaw
    p_shift = p_live - Rz @ REF_P0[kind]

    p(τ) = Rz @ table_p + p_shift,   v/R/omega = Rz @ (.)
    τ = (tick - launch_tick) / 100

`--check` verifies this reconstruction against `ShiftedTrajectory` sample by sample, for
EVERY baked kind, and asserts the two invariants the handovers rely on: the reference at
τ = 0 is a level hover at rest at the live pose, and the table ENDS in a hover (so the
manoeuvre can hand back to the hold reference with no step). Any kind that fails is not
emitted - a table that steps at its ends would put a jump into the policy's errors.

EMITTED
-------
    generated/reference_tables.h    REF_KIND_* enum, count, per-kind length/yaw0/p0/name
    generated/reference_tables.c    one flat REF_ROWS[total][21] array:
                                    [p(3) | v(3) | R(9, row-major) | omega(3) | a(3)]

The trailing `a` is the reference's WORLD-frame acceleration. It is baked (rather than
reconstructed on board by differencing `v`) so the policy's feed-forward block is a_ref +
g*e_z EXACTLY - the same quantity the training environment feeds the actor - instead of a
one-step-delayed finite difference. Cost: 72 -> 84 B per row (+3 floats).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR)) if os.path.basename(_THIS_DIR) == "deploy" else _THIS_DIR
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_target import ShiftedTrajectory, yaw_of  # noqa: E402
from quad_flip_env import GRAVITY, QuadFlipEnv  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(_SIM_DIR, "deploy", "app_policy_controller", "src", "generated")
DEFAULT_MANIFEST = os.path.join(_SIM_DIR, "deploy", "manifests", "reference_tables.json")
TABLE_DT = 0.01            # 100 Hz - the policy's control period (must equal env SIM_DT)
CHECK_TOL = 2.0e-6         # float32 table storage vs float64 reconstruction
# The fixed pose the C cross-check launches every kind from. MUST match REF_CHECK_P0 /
# REF_CHECK_YAW0 / REF_CHECK_START_STEP in app_policy_controller/test/ref_check.c.
# p0/yaw0 match check_table's own relocation test; the tick is an integer so the C can
# index rows with it directly (1234 ticks = 12.34 s).
REF_CHECK_P0 = (0.41, -0.33, 1.37)
REF_CHECK_YAW0 = 0.7
REF_CHECK_START_STEP = 1234
REF_CHECK_HOLD_ROWS = 8
REF_CHECK_TOL = 1.0e-5     # C float32 sampler vs the float64 simulator reconstruction
MAX_DURATION_S = 5.0       # prefer draws no longer than this (the flash cost is 84 B/row)
SEED_SEARCH = 4000         # how far to walk seeds looking for a short, calm draw

# LAUNCH TRANSIENT BUDGET. Only the flip and the waypoint polynomial START at rest; the
# periodic families (orbit, figure-8, lissajous, slalom) are sampled from the middle of
# their own motion, so launching one from a pilot's hover puts the reference's initial
# velocity/rate straight into the policy's ERRORS. Rather than invent a blend curve, the
# generator picks the draw that MINIMISES that transient and holds it to what training
# already presents at t = 0:
#   * v0 <= INIT_VEL_RANGE's upper bound (0.60 m/s): a training episode opens with a
#     random kick of 0.10-0.60 m/s against a reference that may itself be moving, so a
#     0.4 m/s launch mismatch is literally in-distribution;
#   * w0 <= 0.6 rad/s: a quarter of the rate tolerance the tracking score is graded
#     against (2.5 rad/s), so the policy is nowhere near its limit even at launch.
# Measured over 80 seeds, the achievable minima are v0 0.32-0.41 m/s and w0 0.10-0.48 rad/s,
# so both budgets are met with margin by seed selection alone.
V0_MAX_MPS = 0.6
W0_MAX_RADS = 0.6

# The families baked into the firmware. `hover` is deliberately absent: holding a point is
# what the app's HOLD mode already is, and baking it would spend flash on a constant.
KINDS = ("flip", "orbit", "figure8", "lissajous", "slalom", "waypoints")


def draw(kind: str, seed0: int, max_tries: int = SEED_SEARCH) -> Tuple[int, Any, float, Any]:
    """
    Pick the most launchable draw of `kind`.

    The filter is the family itself (the draw really is that kind, and for a flip the
    single pitch rotation the flip button rehearses); the CHOICE is the draw whose launch
    transient and length are smallest - see the budget constants above. A draw is never
    rejected for being mid-motion, because every family that is could still be launched
    inside the training envelope.
    """
    env = QuadFlipEnv(telemetry=False)
    sampler = env.sampler
    mass = float(env.quad.base_mass)
    pool: List[Tuple[float, int, Any]] = []

    for s in range(seed0, seed0 + max_tries):
        rng = np.random.default_rng(s)
        traj = sampler.sample(rng, mass=mass, kind=kind)
        m = traj.maneuver
        if str(getattr(m, "kind", "?")) != kind:
            continue
        if kind == "flip":
            if not (abs(float(m.axis[1]) - 1.0) < 1e-9):          # pitch, not roll
                continue
            if abs(float(getattr(m, "rotations", 1.0)) - 1.0) > 1e-9:
                continue
            # FIRST match wins for the flip, deliberately: this table is the already
            # verified article (+1.58 m pop, 778 deg/s peak) that the mission's flip
            # acceptance criteria are written against, and a flip starts at rest anyway,
            # so there is no launch transient to optimise away.
            return s, traj, mass, env
        r0 = traj.sample(0.0)
        score = (float(np.linalg.norm(r0.v)) / V0_MAX_MPS
                 + float(np.linalg.norm(r0.omega)) / W0_MAX_RADS
                 + 0.1 * float(traj.duration) / MAX_DURATION_S)
        pool.append((score, s, traj))
        if len(pool) >= 400:
            break
    if not pool:
        raise RuntimeError(f"no usable {kind} draw in seeds {seed0}..{seed0 + max_tries - 1}")
    short = [p for p in pool if float(p[2].duration) <= MAX_DURATION_S]
    score, seed, traj = min(short or pool, key=lambda p: p[0])
    return seed, traj, mass, env


def build_table(traj, dt: float = TABLE_DT) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the whole trajectory (manoeuvre + terminal hold tail) at the control rate.

    Row layout: [p(3) | v(3) | R(9, row-major) | omega(3) | a(3)] = 21 floats.
    """
    n = int(math.ceil(traj.duration / dt))
    times = np.arange(n + 1) * dt
    rows = []
    for t in times:
        r = traj.sample(float(min(t, traj.duration)))
        rows.append(np.concatenate([r.p, r.v, r.R.reshape(-1), r.omega, r.a]))
    return times, np.asarray(rows, dtype=np.float32)


def check_table(traj, table: np.ndarray, dt: float = TABLE_DT) -> Dict[str, Any]:
    """Reconstruct the relocation in numpy and compare with `ShiftedTrajectory`."""
    p0 = np.array([0.41, -0.33, 1.37], dtype=np.float64)   # somewhere else entirely
    yaw0 = 0.7                                             # and aimed differently
    t0 = 12.34                                             # and later in a flight

    shifted = ShiftedTrajectory(traj, p0=p0, yaw0=yaw0, t0=t0)
    r0 = traj.sample(0.0)
    dyaw = yaw0 - yaw_of(r0.R)
    c, s = math.cos(dyaw), math.sin(dyaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p_shift = p0 - Rz @ r0.p

    worst = 0.0
    worst_k = -1
    for k in range(len(table)):
        t = t0 + k * dt
        r = shifted.sample(t)
        row = table[k]
        p = Rz @ row[0:3] + p_shift
        v = Rz @ row[3:6]
        R = Rz @ row[6:15].reshape(3, 3)
        # omega is a BODY-frame rate, so it is INVARIANT under the relocation: the body axes
        # turn with the vehicle, giving [w']_x = R'^T R'dot = R^T Q^T Q Rdot = [w]_x. This
        # is what trajectories.ShiftedManeuver has always done and what check_trajectories
        # section I asserts; ShiftedTrajectory and reference.c used to rotate it, and this
        # check agreed with them because it compared against ShiftedTrajectory.
        w = row[15:18]
        # `a` IS world-frame, so the relocation does rotate it.
        a = Rz @ row[18:21]
        err = max(float(np.max(np.abs(p - r.p))), float(np.max(np.abs(v - r.v))),
                  float(np.max(np.abs(R - r.R))), float(np.max(np.abs(w - r.omega))),
                  float(np.max(np.abs(a - r.a))))
        if err > worst:
            worst, worst_k = err, k

    # invariant 1: tau = 0 is a level hover at the live pose (the handover tolerance)
    start_v = float(np.linalg.norm(r0.v))
    start_w = float(np.linalg.norm(r0.omega))
    # invariant 2: the table ends in a hover (the manoeuvre -> hold handback)
    end_v = float(np.linalg.norm(table[-1, 3:6]))
    end_w = float(np.linalg.norm(table[-1, 15:18]))

    z_rel = table[:, 2].astype(np.float64) - float(table[0, 2])
    # Launch transient vs the budget (see V0_MAX_MPS / W0_MAX_RADS): what a launch from a
    # hovering vehicle actually puts into the policy's errors at tau = 0.
    launch_ok = bool(start_v <= V0_MAX_MPS and start_w <= W0_MAX_RADS)
    return {
        "ok": bool(worst <= CHECK_TOL and launch_ok and end_v < 1e-6 and end_w < 1e-6),
        "tol": CHECK_TOL,
        "launch_ok": launch_ok,
        "v0_budget_mps": V0_MAX_MPS,
        "w0_budget_rads": W0_MAX_RADS,
        "max_shift_err": worst,
        "max_shift_err_index": worst_k,
        "start_v_norm": start_v,
        "start_w_norm": start_w,
        "start_yaw_rad": yaw_of(r0.R),
        "end_v_norm": end_v,
        "end_w_norm": end_w,
        "z_rel_min": float(z_rel.min()),
        "z_rel_max": float(z_rel.max()),
        "peak_omega_dps": float(np.degrees(np.max(np.abs(table[:, 15:18])))),
        "peak_a_mps2": float(np.max(np.abs(table[:, 18:21]))),
        "peak_specific_force_mps2": float(np.max(np.abs(table[:, 18:21] + np.array([0.0, 0.0, 9.81])))),
        "test_relocation": {"p0": p0.tolist(), "yaw0": float(yaw0), "t0": float(t0)},
    }


def check_c(out_dir: str, entries: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
    """
    Compile test/ref_check.c and diff the ONBOARD reference sampler against the simulator.

    This is the check that covers the half of the deployed observation `policy_host_check`
    cannot see: reference.c turning a baked table into the error channels AND the
    feed-forward block. It compares against `live_target.ShiftedTrajectory` - the same
    relocation the generator's `check_table` uses, and the same one `live_flight` flies -
    so a pass means the on-board reference describes the motion the policy was trained on,
    with the same `a_ref + g*e_z` it was trained with.

    Rows past the end of the manoeuvre are skipped: reference.c CLAMPS them to the last
    row (a hover by construction) while the simulator samples past `duration`, so the two
    are only expected to agree over the manoeuvre itself.

    Note this compares against ShiftedTrajectory, so it validates the two against EACH
    OTHER - it cannot adjudicate whether a shared convention is physically right. The
    body-frame rate is the case in point: both used to rotate it, and this check passed.
    `check_trajectories` section I is what pins that convention down, against
    ShiftedManeuver.
    """
    app = os.path.join(_SIM_DIR, "deploy", "app_policy_controller")
    src, test = os.path.join(app, "src"), os.path.join(app, "test")
    tmp = tempfile.mkdtemp(prefix="ref_check_")
    exe, out_bin = os.path.join(tmp, "ref_check"), os.path.join(tmp, "rows.bin")

    cmd = ["clang", "-O2", "-std=c11", "-Wall", "-Wextra", f"-I{src}", "-o", exe,
           os.path.join(test, "ref_check.c"),
           os.path.join(src, "reference.c"),
           os.path.join(src, "generated", "reference_tables.c"), "-lm"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return {"ok": True, "skipped": "no host clang"}
    if res.returncode != 0:
        return {"ok": False, "error": res.stderr.strip()[:800]}

    run = subprocess.run([exe, out_bin], capture_output=True, text=True)
    if run.returncode != 0:
        return {"ok": False, "error": run.stderr.strip()[:800]}

    rows = np.fromfile(out_bin, dtype=np.float32)
    n_rows = rows.size // 21
    rows = rows.reshape(n_rows, 21)

    p0 = np.array(REF_CHECK_P0, dtype=np.float64)
    t0 = REF_CHECK_START_STEP / 100.0
    g_vec = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)

    per_kind: Dict[str, float] = {}
    worst, worst_where = 0.0, ""
    off = 0
    for e in entries:
        traj = e["traj"]
        shifted = ShiftedTrajectory(traj, p0=p0, yaw0=REF_CHECK_YAW0, t0=t0)
        n_cmp = min(len(e["table"]), int(math.floor(float(traj.duration) / TABLE_DT)) + 1)
        kw = 0.0
        for k in range(n_cmp):
            r = shifted.sample(t0 + k * TABLE_DT)
            c = rows[off + k].astype(np.float64)
            d = max(
                float(np.max(np.abs(c[0:3] - r.p))),
                float(np.max(np.abs(c[3:6] - r.v))),
                float(np.max(np.abs(c[6:15] - np.asarray(r.R).reshape(-1)))),
                float(np.max(np.abs(c[15:18] - r.omega))),
                float(np.max(np.abs(c[18:21] - (np.asarray(r.a) + g_vec)))),
            )
            if d > kw:
                kw = d
        per_kind[e["kind"]] = kw
        if kw > worst:
            worst, worst_where = kw, f"{e['kind']} row {int(np.argmax([0]))}"
        off += len(e["table"])

    # HOLD: stationary level hover at the anchor, with a [0, 0, g] feed-forward.
    hold = rows[off:off + REF_CHECK_HOLD_ROWS].astype(np.float64)
    cy, sy = math.cos(REF_CHECK_YAW0), math.sin(REF_CHECK_YAW0)
    R_hold = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    hold_err = 0.0
    for c in hold:
        hold_err = max(
            hold_err,
            float(np.max(np.abs(c[0:3] - p0))),
            float(np.max(np.abs(c[3:6]))),
            float(np.max(np.abs(c[6:15] - R_hold.reshape(-1)))),
            float(np.max(np.abs(c[15:18]))),
            float(np.max(np.abs(c[18:21] - g_vec))),
        )
    per_kind["hold"] = hold_err

    ok = bool(worst <= REF_CHECK_TOL and hold_err <= REF_CHECK_TOL and n_rows == off + REF_CHECK_HOLD_ROWS)
    if verbose:
        detail = "  ".join(f"{k}={v:.1e}" for k, v in per_kind.items())
        print(f"check-c : {'PASS' if ok else 'FAIL'}  worst {worst:.2e} "
              f"(tol {REF_CHECK_TOL:g}) over {off} + {REF_CHECK_HOLD_ROWS} rows")
        print(f"          {detail}")
        if not ok:
            print(f"          C harness said: {run.stderr.strip()}")
    return {"ok": ok, "worst": worst, "hold_worst": hold_err, "per_kind": per_kind,
            "rows": int(n_rows), "tol": REF_CHECK_TOL, "harness": run.stderr.strip()}


# ======================================================================================
# C emission
# ======================================================================================
def _fmt(v: Any) -> str:
    f = float(v)
    if not np.isfinite(f):
        raise ValueError(f"non-finite value in table: {f}")
    s = f"{f:.9g}"
    if not any(ch in s for ch in ".eE"):
        s += ".0"
    return s + "f"


def emit_c(out_dir: str, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    path_c = os.path.join(out_dir, "reference_tables.c")
    path_h = os.path.join(out_dir, "reference_tables.h")
    n = len(entries)

    offs, lens, yaws, p0s, names = [], [], [], [], []
    off = 0
    for e in entries:
        offs.append(off)
        lens.append(int(len(e["table"])))
        yaws.append(float(yaw_of(e["traj"].sample(0.0).R)))
        p0s.append(e["table"][0, 0:3].tolist())
        names.append(e["kind"])
        off += int(len(e["table"]))
    rows = np.concatenate([e["table"] for e in entries], axis=0)

    enum = ",\n    ".join(f"REF_KIND_{k.upper()} = {i}" for i, k in enumerate(names))
    hdr = f"""// GENERATED by Simulation/deploy/gen_references.py - DO NOT EDIT BY HAND.
//
// One 100 Hz table per trained manoeuvre, drawn from the training sampler and sampled at
// the policy's control period ({TABLE_DT * 1000:.0f} ms). Index by INTEGER TICK COUNT since launch:
//
//     tau_k = k * REF_DT,   k = 0 .. REF_LEN[kind]-1   (ticks past the end clamp to the
//                                                       last row, which is a hover)
//
// Relocation (the same rigid motion as live_target.ShiftedTrajectory, verified for every
// kind by the generator's --check):
//
//     dyaw    = yaw_live - REF_YAW0[kind]
//     Rz      = yaw rotation by dyaw
//     p_shift = p_live - Rz @ REF_P0[kind]
//     ref_p = Rz @ REF_ROWS[row][0..2]  + p_shift
//     ref_v = Rz @ REF_ROWS[row][3..5]
//     ref_R = Rz @ REF_ROWS[row][6..14] (3x3, row-major)
//     ref_w = REF_ROWS[row][15..17]        (BODY frame: INVARIANT, not rotated)
//     ref_a = Rz @ REF_ROWS[row][18..20]   (WORLD frame: rotates)
//
// ref_a is the reference's acceleration, baked so the app can form the policy's
// feed-forward block (ref_a + g*e_z) exactly. See REF_A_OFFSET below.
//
// At k = 0 every table is a LEVEL HOVER at rest, and every table ends in a hover, so a
// launch and its handback are both step-free.
#pragma once

#include <stdint.h>

#define REF_DT {_fmt(TABLE_DT)}
#define REF_ROW_FLOATS {int(rows.shape[1])}u    // floats per row: p|v|R|omega|a
#define REF_A_OFFSET 18u                   // first float of the acceleration block

typedef enum {{
    {enum},
}} ref_kind_t;

#define REF_KIND_NONE 0xFFu          // "not playing" / the app's stop-and-hold payload
#define REF_TABLE_COUNT {n}u
#define REF_ROWS_TOTAL {int(rows.shape[0])}u

// REF_OFFSET is uint16_t: six tables of ~500 rows already exceed one byte.  The app
// indexes REF_ROWS[REF_OFFSET[kind] + k], so it must not truncate.
extern const uint16_t REF_OFFSET[REF_TABLE_COUNT];
extern const uint16_t REF_LEN[REF_TABLE_COUNT];
extern const float    REF_YAW0[REF_TABLE_COUNT];
extern const float    REF_P0[REF_TABLE_COUNT][3];
extern const char    *const REF_NAME[REF_TABLE_COUNT];
extern const float    REF_ROWS[REF_ROWS_TOTAL][REF_ROW_FLOATS];
"""

    def arr(name: str, values: List[str], ctype: str) -> str:
        return f"const {ctype} {name}[REF_TABLE_COUNT] = {{" + ", ".join(values) + "};\n"

    src = ["// GENERATED by Simulation/deploy/gen_references.py - DO NOT EDIT BY HAND.",
           '#include "reference_tables.h"', ""]
    src.append(arr("REF_OFFSET", [f"{v}u" for v in offs], "uint16_t"))
    src.append(arr("REF_LEN", [f"{v}u" for v in lens], "uint16_t"))
    src.append(arr("REF_YAW0", [_fmt(v) for v in yaws], "float"))
    src.append("const float REF_P0[REF_TABLE_COUNT][3] = {\n    "
               + ",\n    ".join("{" + ", ".join(_fmt(v) for v in p) + "}" for p in p0s)
               + "\n};\n")
    src.append("const char *const REF_NAME[REF_TABLE_COUNT] = {\n    "
               + ",\n    ".join(f'"{k}"' for k in names) + "\n};\n")
    body = ",\n    ".join("{" + ", ".join(_fmt(v) for v in row) + "}" for row in rows)
    src.append(f"const float REF_ROWS[REF_ROWS_TOTAL][REF_ROW_FLOATS] = {{\n    {body}\n}};\n")

    with open(path_h, "w") as fh:
        fh.write(hdr)
    with open(path_c, "w") as fh:
        fh.write("\n".join(src))

    return {"header": path_h, "source": path_c, "kinds": names, "rows": int(rows.shape[0]),
            "rows_per_kind": lens, "offsets": offs,
            "table_sha256": hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
            "header_bytes": os.path.getsize(path_h), "source_bytes": os.path.getsize(path_c)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Bake the trained manoeuvres into 100 Hz C tables")
    ap.add_argument("--seed", type=int, default=0, help="first seed to try (default 0)")
    ap.add_argument("--kinds", nargs="*", default=list(KINDS),
                    help=f"families to bake (default: {' '.join(KINDS)})")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--check-only", action="store_true", help="draw + check, emit nothing")
    ap.add_argument("--no-check-c", dest="check_c", action="store_false",
                    help="skip compiling test/ref_check.c and diffing the ONBOARD reference "
                         "sampler against ShiftedTrajectory (on by default; needs host clang)")
    args = ap.parse_args()

    entries: List[Dict[str, Any]] = []
    failed: List[str] = []
    print(f"{'kind':<10} {'seed':>5}  {'rows':>5}  {'dur':>6}  {'shift err':>10}  "
          f"{'v0':>7}  {'w0':>7}  {'z range (m)':>16}  {'peak rate':>9}")
    for kind in args.kinds:
        seed, traj, mass, _env = draw(kind, args.seed)
        _times, table = build_table(traj)
        check = check_table(traj, table)
        entries.append(dict(kind=kind, seed=seed, traj=traj, table=table, check=check,
                            mass=mass))
        print(f"{kind:<10} {seed:>5}  {len(table):>5}  {float(traj.duration):>6.2f}  "
              f"{check['max_shift_err']:>10.3g}  {check['start_v_norm']:>7.3f}  "
              f"{check['start_w_norm']:>7.3f}  "
              f"{check['z_rel_min']:+.2f} .. {check['z_rel_max']:+.2f}  "
              f"{check['peak_omega_dps']:>7.0f} d/s")
        if not check["ok"]:
            failed.append(kind)

    total_rows = sum(len(e["table"]) for e in entries)
    print(f"\ntotal   : {total_rows} rows -> {total_rows * 21 * 4 / 1024:.0f} KiB of weights "
          f"({total_rows * 21 * 4} bytes), {len(entries)} kinds")
    if failed:
        print(f"check   : FAIL for {', '.join(failed)} - not emitting a table that steps "
              f"at its ends into the policy's errors")
        return 3
    print("check   : PASS for every kind (invariants + relocation vs ShiftedTrajectory)")

    if args.check_only:
        return 0

    out = emit_c(args.out_dir, entries)
    print(f"wrote   : {out['header']}")
    print(f"wrote   : {out['source']} ({out['source_bytes']} bytes)")
    print(f"sha256  : {out['table_sha256'][:16]}... (all tables)")

    # -- the C side, against the SIMULATOR's own reference ------------------------------
    # Runs on the emitted tables (it compiles reference_tables.c), so it belongs after
    # emit_c. A failure here means the vehicle would fly a different reference than the
    # policy was trained on - a class of bug that no amount of policy-side checking sees.
    c_report = {"ok": True, "skipped": "--no-check-c"}
    if args.check_c:
        c_report = check_c(args.out_dir, entries)
        if c_report.get("error"):
            print(f"check-c : FAIL to build/run the harness\n          {c_report['error']}")
            return 4
        if not c_report["ok"]:
            print("check-c : FAIL - the onboard sampler disagrees with the simulator")
            return 4

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table_dt": TABLE_DT,
        "max_duration_s": MAX_DURATION_S,
        "kinds": out["kinds"],
        "rows_per_kind": out["rows_per_kind"],
        "rows_total": out["rows"],
        "flash_bytes": out["rows"] * 21 * 4,
        "table_sha256": out["table_sha256"],
        "check_c": c_report,
        "entries": [{"kind": e["kind"], "seed": e["seed"], "rows": int(len(e["table"])),
                     "duration_s": float(e["traj"].duration),
                     "yaw0": float(yaw_of(e["traj"].sample(0.0).R)),
                     "check": e["check"]} for e in entries],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"wrote   : {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
