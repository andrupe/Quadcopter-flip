"""
Equivalence check for the scalar ("fast math") hot paths.

The reference generator, the rate PID and the mixer were rewritten in scalar form
because numpy's dispatch on 3- and 4-element vectors costs ~10x the arithmetic and
these calls run at 1 kHz inside every worker process. A rewrite like that is only
acceptable if it is numerically the SAME FUNCTION, so this script compares each fast
implementation against the exact vector formulation it replaced:

    dcm_from_thrust_dir_and_yaw   vs np.cross + np.linalg.norm + np.column_stack
    _normalize                    vs v / np.linalg.norm(v)
    RatePID.update                vs the vector expression (sub, mul, clip, copy)
    mixerFM                       vs np.dot + np.clip + np.sqrt

Tolerance is tight because the operations and their order are preserved - the reference
helpers and the rate PID reproduce the vector code BIT-EXACTLY. The mixer is the one
exception: np.dot on a 4x4 @ 4-vector dispatches to a BLAS gemv that accumulates with
FMA in a different order than an explicit sum, so that check is relative (measured
~2e-15, a couple of ULP) rather than exact. That is unavoidable without keeping BLAS in
the hot loop, and it is a rounding difference, not a different function.

Run:  .venv/bin/python scratch/check_fast_math.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "Simulation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trajectories import _normalize, dcm_from_thrust_dir_and_yaw  # noqa: E402
from utils.rate_pid import RatePIDController  # noqa: E402
from utils.mixer import mixerFM  # noqa: E402

FAILURES: list[str] = []


def check(name: str, worst: float, tol: float, detail: str = "") -> None:
    ok = worst <= tol
    if not ok:
        FAILURES.append(name)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name:<44} worst |diff| = {worst:.3e}  "
          f"(tol {tol:.0e}){('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------------
# reference implementations - the code exactly as it was before the rewrite
# ---------------------------------------------------------------------------------
def ref_normalize(v, fallback=None):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0]) if fallback is None else np.asarray(fallback, dtype=np.float64)
    return v / n


def ref_dcm(z_b, yaw):
    z_b = ref_normalize(z_b)
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_b = np.cross(z_b, x_c)
    if np.linalg.norm(y_b) < 1e-6:
        x_c = np.array([np.cos(yaw + np.pi / 2.0), np.sin(yaw + np.pi / 2.0), 0.0])
        y_b = np.cross(z_b, x_c)
    y_b = ref_normalize(y_b)
    x_b = np.cross(y_b, z_b)
    return np.column_stack([x_b, y_b, z_b])


def ref_pid_update(pid: RatePIDController, omega_des, omega_meas, dt) -> np.ndarray:
    """The vector formulation the fast loop replaced."""
    omega_des = np.asarray(omega_des, dtype=np.float64)
    omega_meas = np.asarray(omega_meas, dtype=np.float64)
    dt = max(1e-6, float(dt))

    error = omega_des - omega_meas
    p_term = pid.kp * error
    pid.integral += pid.ki * error * dt
    pid.integral = np.clip(pid.integral, -pid.max_integral, pid.max_integral)
    i_term = pid.integral.copy()
    if pid.prev_omega is not None:
        d_omega = (omega_meas - pid.prev_omega) / dt
        d_term = -pid.kd * d_omega
    else:
        d_term = np.zeros(3, dtype=np.float64)
    pid.prev_omega = omega_meas.copy()
    raw_torque = p_term + i_term + d_term
    clamped = np.clip(raw_torque, -pid.max_torque, pid.max_torque)
    pid.last_torque = clamped.copy()
    return clamped


def ref_mixer(quad, thr, moment) -> np.ndarray:
    t = np.array([thr, moment[0], moment[1], moment[2]])
    return np.sqrt(np.clip(np.dot(quad.params["mixerFMinv"], t),
                           quad.params["minWmotor"] ** 2, quad.params["maxWmotor"] ** 2))


# ---------------------------------------------------------------------------------
# A. normalize + dcm
# ---------------------------------------------------------------------------------
def part_a() -> None:
    print("A. reference-generator helpers")
    rng = np.random.default_rng(0)

    cases = [np.array([0.3, -0.4, 9.81]), np.array([1e-3, 0.0, 1.0]),
             np.array([0.0, 0.0, 0.0])]  # zero vector exercises the fallback
    for _ in range(2000):
        cases.append(rng.normal(0.0, 1.0, size=3))
    for _ in range(200):
        cases.append(rng.normal(0.0, 1e-6, size=3))  # near-degenerate magnitudes

    worst_norm = 0.0
    worst_dcm = 0.0
    worst_degenerate = 0.0
    for v in cases:
        worst_norm = max(worst_norm, float(np.abs(_normalize(v) - ref_normalize(v)).max()))
        for yaw in (0.0, 0.7, -2.5, np.pi / 2.0, 3.14159):
            got = dcm_from_thrust_dir_and_yaw(v, yaw)
            ref = ref_dcm(v, yaw)
            worst_dcm = max(worst_dcm, float(np.abs(got - ref).max()))

    # The heading-degenerate branch: z_b parallel to [cos yaw, sin yaw, 0].
    for yaw in (0.0, 1.1, -2.0):
        z_b = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        worst_degenerate = max(worst_degenerate, float(np.abs(dcm_from_thrust_dir_and_yaw(z_b, yaw) - ref_dcm(z_b, yaw)).max()))

    check("_normalize (2200 vectors incl. zero)", worst_norm, 1e-16)
    check("dcm_from_thrust_dir_and_yaw (11000 cases)", worst_dcm, 1e-14)
    check("dcm heading-degenerate branch", worst_degenerate, 1e-14)

    # Colinearity sanity: the result must be a proper rotation matrix.
    for _ in range(200):
        yaw = float(rng.uniform(-np.pi, np.pi))
        R = dcm_from_thrust_dir_and_yaw(rng.normal(0.0, 1.0, size=3), yaw)
        worst_det = float(abs(np.linalg.det(R) - 1.0))
        worst_orth = float(np.abs(R.T @ R - np.eye(3)).max())
    check("dcm is orthonormal (det=1)", max(worst_det, worst_orth), 1e-12)


# ---------------------------------------------------------------------------------
# B. rate PID
# ---------------------------------------------------------------------------------
def part_b() -> None:
    print("\nB. RatePID.update")
    rng = np.random.default_rng(1)
    fast = RatePIDController()
    ref = RatePIDController()

    worst_tau = 0.0
    worst_state = 0.0
    t = 0.0
    # Random walk through the torque and integral limits so both saturations engage.
    for i in range(20000):
        t += 0.001
        omega_meas = rng.normal(0.0, 8.0, size=3)
        omega_des = rng.normal(0.0, 12.0, size=3) * (1.0 if i % 7 else 30.0)
        tau_f = fast.update(omega_des, omega_meas, 0.001)
        tau_r = ref_pid_update(ref, omega_des, omega_meas, 0.001)
        worst_tau = max(worst_tau, float(np.abs(tau_f - tau_r).max()))
        worst_state = max(
            worst_state,
            float(np.abs(fast.integral - ref.integral).max()),
            float(np.abs(fast.last_torque - ref.last_torque).max()),
            float(np.abs(fast.prev_omega - ref.prev_omega).max()),
        )
        if i % 5000 == 0:
            fast.reset()
            ref.reset()

    check("torque, 20k steps (saturating)", worst_tau, 1e-15)
    check("integral / prev / last state", worst_state, 1e-15)

    # List inputs and a non-array dt path (the API accepts both).
    f2, r2 = RatePIDController(), RatePIDController()
    tau_f = f2.update([0.5, -0.25, 0.1], [0.4, -0.2, 0.05], np.float64(0.001))
    tau_r = ref_pid_update(r2, np.array([0.5, -0.25, 0.1]), np.array([0.4, -0.2, 0.05]), np.float64(0.001))
    check("list inputs", float(np.abs(tau_f - tau_r).max()), 1e-15)


# ---------------------------------------------------------------------------------
# C. mixer
# ---------------------------------------------------------------------------------
def part_c() -> None:
    print("\nC. mixerFM")
    from quadFiles.quad_mujoco import QuadcopterMuJoCo

    quad = QuadcopterMuJoCo()  # real params: mixerFMinv, minWmotor, maxWmotor
    rng = np.random.default_rng(2)

    worst = 0.0
    for _ in range(5000):
        thr = float(rng.uniform(0.0, 0.6))
        moment = rng.uniform(-0.01, 0.01, size=3)
        got = mixerFM(quad, thr, moment)
        ref_w = ref_mixer(quad, thr, moment)
        scale = max(1.0, float(np.abs(ref_w).max()))
        worst = max(worst, float(np.abs(got - ref_w).max()) / scale)
    # Drive both limits: huge moments saturate every rotor.
    for _ in range(500):
        got = mixerFM(quad, 0.0, np.array([1.0, -1.0, 1.0]))
        ref_w = ref_mixer(quad, 0.0, np.array([1.0, -1.0, 1.0]))
        scale = max(1.0, float(np.abs(ref_w).max()))
        worst = max(worst, float(np.abs(got - ref_w).max()) / scale)
    check("w_cmd, 5.5k draws (incl. saturation)", worst, 1e-13, detail="relative (BLAS FMA order)")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("RESULT: scalar hot paths are numerically equivalent to the vector versions")
