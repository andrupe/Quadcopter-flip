#!/usr/bin/env python3
"""Host verification for analytic_flip.c vs trajectories.py Flip."""

import math
import os
import subprocess
import sys
import tempfile
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trajectories import Flip, Trajectory

def test_analytic_flip():
    print("Testing analytic_flip.c vs trajectories.py...")
    src_c = os.path.join(_HERE, "app_policy_controller", "src", "analytic_flip.c")
    src_h = os.path.join(_HERE, "app_policy_controller", "src", "analytic_flip.h")

    # Write a test harness in C that outputs row data for a requested configuration
    harness_c = f"""
#include <stdio.h>
#include <stdlib.h>
#include "{src_h}"

int main(int argc, char **argv) {{
    if (argc < 7) return 1;
    float peak_dps = atof(argv[1]);
    float pop_pct = atof(argv[2]);
    float rate_frac = atof(argv[3]);
    uint8_t axis = atoi(argv[4]);
    float yaw0 = atof(argv[5]);
    float z0 = atof(argv[6]);

    analytic_flip_params_t p;
    analytic_flip_params_default(&p, 0.033f, 0.60f, 9.81f);
    p.peak_dps = peak_dps;
    p.pop_pct = pop_pct;
    p.rate_frac = rate_frac;
    p.axis = axis;

    analytic_flip_state_t s;
    float p0[3] = {{0.1f, -0.2f, z0}};
    analytic_flip_init(&s, &p, p0, yaw0, 100u);

    printf("%u %u %u\\n", s.rot_start_row, s.catch_start_row, s.total_rows);

    for (uint32_t k = 0; k < s.total_rows; k++) {{
        float pos[3], vel[3], R[9], w[3], a[3];
        analytic_flip_sample(&s, 100u + k, pos, vel, R, w, a);
        printf("%u %.6f %.6f %.6f  %.6f %.6f %.6f  %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f  %.6f %.6f %.6f  %.6f %.6f %.6f\\n",
               k,
               pos[0], pos[1], pos[2],
               vel[0], vel[1], vel[2],
               R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
               w[0], w[1], w[2],
               a[0], a[1], a[2]);
    }}
    return 0;
}}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_c_file = os.path.join(tmpdir, "test_harness.c")
        test_bin = os.path.join(tmpdir, "test_harness")
        with open(test_c_file, "w") as f:
            f.write(harness_c)

        cmd = ["clang", "-O2", "-Wall", "-Wextra", "-I", os.path.dirname(src_h),
               src_c, test_c_file, "-lm", "-o", test_bin]
        subprocess.check_call(cmd)

        test_cases = [
            # peak_dps, pop_pct, rate_frac, axis, yaw0, z0
            (720.0, 0.90, 0.30, 0, 0.7, 1.2),   # pitch flip, positive yaw
            (680.0, 0.88, 0.28, 0, -1.2, 1.5),  # pitch flip, negative yaw
            (800.0, 0.92, 0.33, 0, 0.0, 1.1),   # pitch flip, zero yaw
            (740.0, 0.90, 0.30, 1, 0.5, 1.3),   # roll flip, positive yaw
            (700.0, 0.85, 0.32, 1, -2.1, 1.4),  # roll flip, negative yaw
        ]

        for peak_dps, pop_pct, rate_frac, axis, yaw0, z0 in test_cases:
            axis_name = "pitch" if axis == 0 else "roll"
            res = subprocess.check_output([
                test_bin, str(peak_dps), str(pop_pct), str(rate_frac),
                str(axis), str(yaw0), str(z0)
            ]).decode().strip().split("\n")

            meta = res[0].split()
            rot_start, catch_start, total_rows = int(meta[0]), int(meta[1]), int(meta[2])

            # Ground truth in Python
            omega_target = math.radians(peak_dps)
            coast = 2.0 * math.pi / (omega_target * (1.0 - rate_frac))
            ax_vec = [0.0, 1.0, 0.0] if axis == 0 else [1.0, 0.0, 0.0]
            fl = Flip(p0=[0.1, -0.2, z0], axis=ax_vec, rotations=1.0, coast=coast,
                      yaw=yaw0, mass=0.033, rate_frac=rate_frac, accel_frac=pop_pct)
            traj = Trajectory(fl, tail=1.0)

            max_dp = 0.0
            max_dv = 0.0
            max_da = 0.0
            max_dR = 0.0
            max_dw = 0.0

            for line in res[1:]:
                parts = [float(x) for x in line.split()]
                k = int(parts[0])
                t = k * 0.01
                p_c = np.array(parts[1:4])
                v_c = np.array(parts[4:7])
                R_c = np.array(parts[7:16]).reshape(3, 3)
                w_c = np.array(parts[16:19])
                a_c = np.array(parts[19:22])

                r_py = traj.sample(t)
                max_dp = max(max_dp, float(np.linalg.norm(r_py.p - p_c)))
                max_dv = max(max_dv, float(np.linalg.norm(r_py.v - v_c)))
                max_da = max(max_da, float(np.linalg.norm(r_py.a - a_c)))
                max_dR = max(max_dR, float(np.linalg.norm(r_py.R - R_c)))
                max_dw = max(max_dw, float(np.linalg.norm(r_py.omega - w_c)))

            print(f"  [PASS] {axis_name} {peak_dps:.0f} dps (yaw={yaw0:+.1f}): "
                  f"rows={total_rows} rot={rot_start} catch={catch_start} | "
                  f"dp={max_dp:.1e}m dv={max_dv:.1e} da={max_da:.1e} dR={max_dR:.1e} dw={max_dw:.1e}")
            assert max_dp < 1e-5, f"dp too high: {max_dp}"
            assert max_dv < 1e-5, f"dv too high: {max_dv}"
            assert max_da < 1e-4, f"da too high: {max_da}"
            assert max_dR < 1e-5, f"dR too high: {max_dR}"
            assert max_dw < 0.02, f"dw too high: {max_dw}"

    print("ALL ANALYTIC FLIP TESTS PASSED!")
    return 0

if __name__ == "__main__":
    sys.exit(test_analytic_flip())
