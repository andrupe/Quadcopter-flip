"""
Validation for the trajectory-tracking migration of Simulation/quad_flip_env.py.

What matters here is that the task is well-posed and SOLVABLE, not just that it runs:

  A. OBSERVATION CONTRACT. 77 dims, actor frame is a clean prefix, and every declared
     field offset actually contains what the layout comment claims.
  B. THE TASK IS SOLVABLE. A plain cascaded geometric controller, with no learning at all,
     must score well on every manoeuvre. If a hand-written controller cannot track the
     reference, the reference is wrong or the reward is mis-scaled, and no amount of PPO
     will fix it. This is the single most important check in the file.
  C. LIGHTHOUSE IS IN THE LOOP. A flip must produce a real blackout, and the actor frame
     must show the dead-reckoned estimate drifting away from truth while it lasts.
  D. EPISODES END IN THE TERMINAL HOVER. The episode horizon follows the trajectory, and
     the final reference is a genuine hold.
  E. COMMAND INTERFACE. A pinned manoeuvre is honoured, and an unknown one is rejected
     rather than silently ignored.
  F. REWARD SANITY. Finite, non-negative, and bounded by the sum of the kernel weights.

Run:  .venv/bin/python scratch/check_env_tracking.py
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

from quad_flip_env import (  # noqa: E402
    ACTOR_SINGLE_OBS_DIM,
    ENCODER_AUX_DIM,
    O_ATT_ERR,
    O_OMEGA,
    O_POS,
    O_PREV_ACTION,
    O_P_ERR,
    O_QUAT,
    O_VELXY,
    O_VELZ,
    O_V_ERR,
    O_W_ERR,
    TOTAL_OBS_DIM,
    QuadFlipEnv,
)
from trajectories import GRAVITY, MASS_NOMINAL, dcm_from_thrust_dir_and_yaw  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def track_action(env: QuadFlipEnv, kp: float = 7.0, kd: float = 4.5,
                 kr: float = 6.0, kw: float = 1.5) -> np.ndarray:
    """
    Plain cascaded geometric controller: position error -> desired acceleration ->
    desired attitude + collective thrust -> body-rate command. No learning involved.

    This exists to prove the reference is followable. It is deliberately the textbook
    controller rather than anything tuned for this reward.
    """
    ref = env.ref
    q = env.quad

    a_des = ref.a + kp * (ref.p - q.pos) + kd * (ref.v - q.vel)
    z_b = dcm_from_thrust_dir_and_yaw(a_des + np.array([0.0, 0.0, GRAVITY]), 0.0)[:, 2]
    thrust = float(MASS_NOMINAL * float(np.dot(a_des + np.array([0.0, 0.0, GRAVITY]), z_b)))

    # Attitude error between the reference attitude and the actual one, body frame.
    R_err = ref.R.T @ q.dcm
    e_R = 0.5 * vee(R_err - R_err.T)
    omega_cmd = -kr * e_R + ref.omega

    max_thr = float(q.params["maxThr"])
    a0 = float(np.clip(2.0 * thrust / max_thr - 1.0, -1.0, 1.0))
    return np.array([
        a0,
        np.clip(omega_cmd[0] / env.max_rate_xy, -1.0, 1.0),
        np.clip(omega_cmd[1] / env.max_rate_pitch, -1.0, 1.0),
        np.clip(omega_cmd[2] / env.max_rate_z, -1.0, 1.0),
    ], dtype=np.float32)


def run_episode(maneuver=None, seed: int = 0, max_reward_weight: float = 6.8):
    env = QuadFlipEnv()
    opts = {"maneuver": maneuver} if maneuver else None
    obs, info = env.reset(seed=seed, options=opts)

    total = 0.0
    n = 0
    min_lh_visible = 99
    max_drift = 0.0
    saw_outage = False
    rewards = []
    while True:
        obs, r, term, trunc, info = env.step(track_action(env))
        total += r
        n += 1
        rewards.append(r)
        if info["lighthouse_fix"]:
            min_lh_visible = min(min_lh_visible, info["lighthouse_visible"])
        else:
            saw_outage = True
            min_lh_visible = 0
        max_drift = max(max_drift, info["lighthouse_drift_m"])
        if term or trunc:
            break

    return {
        "env": env, "info": info, "total": total, "steps": n,
        "mean": total / max(1, n), "min_lh_visible": min_lh_visible,
        "max_drift": max_drift, "saw_outage": saw_outage,
        "final_ref": env.ref, "terminated": term, "rewards": rewards,
    }


print("=" * 78)
print("A. observation contract")
print("=" * 78)
env = QuadFlipEnv()
obs, info = env.reset(seed=0)
check("total observation dim is 77", obs.shape == (TOTAL_OBS_DIM,) and TOTAL_OBS_DIM == 77,
      f"shape {obs.shape}, declared {TOTAL_OBS_DIM}")
check("actor frame is a clean prefix of the env vector",
      ACTOR_SINGLE_OBS_DIM == 29 and ENCODER_AUX_DIM == 4,
      f"actor {ACTOR_SINGLE_OBS_DIM}, aux {ENCODER_AUX_DIM}")

actor = obs[:ACTOR_SINGLE_OBS_DIM]
ref, lh = env.ref, env.lighthouse
check("position channel carries the Lighthouse estimate",
      np.allclose(actor[O_POS:O_POS + 3], lh.p_est, atol=1e-5),
      f"|obs - est| = {np.linalg.norm(actor[O_POS:O_POS+3] - lh.p_est):.2e}")
check("horizontal velocity channel is the estimate",
      np.allclose(actor[O_VELXY:O_VELXY + 2], lh.v_est[:2], atol=1e-5), "")
check("vertical velocity channel is the estimate",
      abs(actor[O_VELZ] - lh.v_est[2]) < 1e-5, "")
check("position error channel is ref - estimate",
      np.allclose(actor[O_P_ERR:O_P_ERR + 3], ref.p - lh.p_est, atol=1e-5), "")
check("velocity error channel is ref - estimate",
      np.allclose(actor[O_V_ERR:O_V_ERR + 3], ref.v - lh.v_est, atol=1e-5), "")
check("quaternion channel is unit norm",
      abs(np.linalg.norm(actor[O_QUAT:O_QUAT + 4]) - 1.0) < 1e-5,
      f"|q| = {np.linalg.norm(actor[O_QUAT:O_QUAT+4]):.6f}")
check("attitude error is ~0 at reset (spawn is on the reference)",
      np.linalg.norm(actor[O_ATT_ERR:O_ATT_ERR + 3]) < 0.15,
      f"|e_R| = {np.linalg.norm(actor[O_ATT_ERR:O_ATT_ERR+3]):.4f} rad (spawn jitter is intentional)")
check("body-rate error is finite",
      np.all(np.isfinite(actor[O_W_ERR:O_W_ERR + 3])), "")
check("prev-action channel is the hover-trim prefill at reset",
      np.allclose(actor[O_PREV_ACTION:O_PREV_ACTION + 4], env.hover_trim_action, atol=1e-6), "")
check("encoder frame is 33 dims (actor + aux)",
      info["encoder_frame"].shape == (ACTOR_SINGLE_OBS_DIM + ENCODER_AUX_DIM,),
      f"{info['encoder_frame'].shape}")

print()
print("=" * 78)
print("B. the task is solvable by a textbook controller (the important one)")
print("=" * 78)
print("   A cascaded geometric controller with no learning at all. Target: mean reward")
print("   above 75% of the maximum achievable per step, on every manoeuvre.")
MAX_STEP_REWARD = 7.3     # 3.0 pos + 1.0 vel + 2.0 att + 0.8 rate + 0.5 action
results = {}
for name in ("hover", "waypoints", "figure8", "flip"):
    res = run_episode(maneuver=name, seed=3)
    results[name] = res
    frac = res["mean"] / MAX_STEP_REWARD
    check(f"{name}: scripted controller scores >= 75% of max",
          frac >= 0.75, f"mean {res['mean']:.2f} / {MAX_STEP_REWARD:.1f} = {frac*100:.0f}%")
    check(f"{name}: no termination", not res["terminated"],
          f"reason = {res['info']['termination_reason']}")

print()
print("=" * 78)
print("C. Lighthouse is genuinely in the loop")
print("=" * 78)
flip = results["flip"]
check("a flip produces a real blackout", flip["saw_outage"] and flip["min_lh_visible"] == 0,
      f"min visible stations = {flip['min_lh_visible']}")
check("the estimate drifts from truth during the blackout",
      flip["max_drift"] > 0.005, f"max drift {flip['max_drift']*1000:.1f} mm")
hover = results["hover"]
check("a hover keeps coverage (no blackout)",
      not hover["saw_outage"], f"outage seen = {hover['saw_outage']}")

print()
print("=" * 78)
print("D. episodes end in the terminal hover")
print("=" * 78)
for name, res in results.items():
    e = res["env"]
    fref = res["final_ref"]
    check(f"{name}: episode length follows the trajectory",
          abs(e.t - e.traj.duration) < 2.5 * e.dt,
          f"t = {e.t:.2f} s, traj = {e.traj.duration:.2f} s, steps {res['steps']}")
    check(f"{name}: final reference is a hover",
          np.linalg.norm(fref.v) < 1e-6 and np.linalg.norm(fref.omega) < 1e-3 and fref.R[2, 2] > 0.999,
          f"|v| {np.linalg.norm(fref.v):.1e}, |omega| {np.linalg.norm(fref.omega):.1e}, "
          f"tilt {np.degrees(np.arccos(np.clip(fref.R[2, 2], -1, 1))):.3f} deg")

print()
print("=" * 78)
print("E. command interface")
print("=" * 78)
env = QuadFlipEnv()
env.set_command("figure8")
_, info = env.reset(seed=1)
check("a pinned command is honoured", info["maneuver"] == "figure8", f"got {info['maneuver']}")
try:
    env.set_command("barrel_roll")
    check("an unknown command is rejected", False, "no exception raised")
except ValueError as exc:
    check("an unknown command is rejected", True, str(exc)[:48] + "...")

env.set_command(None)
seen = set()
for s in range(12):
    _, i = env.reset(seed=100 + s)
    seen.add(i["maneuver"])
check("mixture sampling returns multiple manoeuvres", len(seen) >= 2, f"seen {sorted(seen)}")

print()
print("=" * 78)
print("F. reward sanity")
print("=" * 78)
for name, res in results.items():
    rw = np.array(res["rewards"])
    check(f"{name}: reward finite, non-negative and bounded",
          bool(np.all(np.isfinite(rw))) and rw.min() >= -1e-6 and rw.max() <= MAX_STEP_REWARD + 1e-6,
          f"[{rw.min():.2f}, {rw.max():.2f}]")

print()
print("=" * 78)
print("G. determinism")
print("=" * 78)
a = run_episode(maneuver="waypoints", seed=17)["total"]
b = run_episode(maneuver="waypoints", seed=17)["total"]
check("same seed reproduces the episode return", abs(a - b) < 1e-6, f"{a:.6f} vs {b:.6f}")

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all tracking env checks passed")
