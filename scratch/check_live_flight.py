# -*- coding: utf-8 -*-
"""
Validation for the live-flight layer: human reference generation, manoeuvre relocation,
the live env, the full encoder+PPO path, the pad mapper/calibration and the panel
protocol.

Everything here is additive to the training pipeline; the last section re-states the
contract that the crucial files (quad_flip_env.py, trajectories.py, encoder/, train.py,
evaluate.py) are NOT touched by this feature.

Run:  .venv/bin/python scratch/check_live_flight.py
"""

from __future__ import annotations

import math
import os
import socket
import sys
import tempfile
import time
import types

import numpy as np
import matplotlib
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_SIM = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in (_PROJECT_ROOT, _SIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trajectories import Maneuver, omega_from_dcm  # noqa: E402
from live_target import (HumanTarget, ShiftedTrajectory, yaw_of,  # noqa: E402
                         HUMAN_ACC_MAX, HUMAN_V_MAX, HUMAN_FOREVER)
from live_policy_env import LiveFlightEnv  # noqa: E402
import live_flight as lf  # noqa: E402
import flight_gui as fg  # noqa: E402
import gamepad as gp  # noqa: E402
from pad_mapper import PadMapper, RAW_AXES  # noqa: E402
from flight_link import NullLink, RadioSetpoint  # noqa: E402

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# Raw device convention, as documented in gamepad.py and assumed by the pad mapper's
# defaults: pushing RIGHT reports +1, pushing UP reports -1, on every axis.
RAW_RIGHT = +1.0
RAW_UP = -1.0


# ======================================================================================
section("A. HumanTarget: the pilot as a trajectory generator")
# ======================================================================================
g = 9.81
m = 0.028
target = HumanTarget(np.array([0.0, 0.0, 1.2]), yaw0=0.0)

# hover: no command, the reference must not move at all
p0 = target.p_ref.copy()
for _ in range(200):
    target.update(0.01)
check("no command = a stationary hover reference",
      float(np.linalg.norm(target.p_ref - p0)) < 1e-12 and float(np.linalg.norm(target.v_ref)) < 1e-12,
      f"moved {1e3 * float(np.linalg.norm(target.p_ref - p0)):.3e} mm")
ref = target.sample(0.0)
check("the hover reference is level and carries hover thrust",
      ref.R[2, 2] > 0.9999 and abs(ref.thrust_ff - m * g) < 1e-6,
      f"z_b.z={ref.R[2, 2]:.4f} thrust={ref.thrust_ff:.4f} N (m g={m * g:.4f})")

# forward command: converge to the commanded speed, acceleration capped
target.clear_command()
target.set_command(fwd=2.0, left=0.0, up=0.0, yaw_rate=0.0)
accels, speeds = [], []
for _ in range(300):
    target.update(0.01)
    accels.append(float(np.linalg.norm(target.a_ref)))
    speeds.append(float(np.linalg.norm(target.v_ref)))
check("a forward command converges to the commanded speed",
      abs(target.v_ref[0] - 2.0) < 0.05 and abs(target.v_ref[1]) < 1e-9,
      f"v_ref=[{target.v_ref[0]:+.3f} {target.v_ref[1]:+.3f} {target.v_ref[2]:+.3f}] m/s")
check("the reference acceleration is capped (a policy envelope, not a step)",
      max(accels) <= HUMAN_ACC_MAX * 1.001, f"peak {max(accels):.2f} <= {HUMAN_ACC_MAX} m/s^2")
check("the reference speed is capped",
      max(speeds) <= HUMAN_V_MAX * 1.001, f"peak {max(speeds):.2f} <= {HUMAN_V_MAX} m/s")
check("attitude stays flatness-consistent with the reference acceleration",
      float(np.linalg.norm(target.sample().R - Maneuver.flat_attitude(target.a_ref, target.yaw_ref))) < 1e-12,
      "R == flat_attitude(a, yaw)")
check("the reference travels in the heading frame (world x = nose here)",
      target.p_ref[0] > 0.3 and abs(target.p_ref[1]) < 1e-9, f"x={target.p_ref[0]:.2f} m")

# yaw command: capped, and the body rate tracks it
target2 = HumanTarget(np.array([0.0, 0.0, 1.2]))
target2.set_command(0.0, 0.0, 0.0, yaw_rate=1.0)
yaws = []
for _ in range(200):
    target2.update(0.01)
    yaws.append(target2.yaw_ref)
check("a yaw command turns the reference heading",
      yaws[-1] > yaws[0] + 1.0 and abs(target2.sample().omega[2] - 1.0) < 0.05,
      f"yaw {yaws[0]:+.2f} -> {yaws[-1]:+.2f} rad, omega_z={target2.sample().omega[2]:+.3f} "
      f"(same sign as the command)")

# handover: sync_to must adopt the state with no step, then decay a non-zero velocity
t3 = HumanTarget(np.array([0.0, 0.0, 2.0]))
t3.sync_to(np.array([1.0, -1.0, 3.0]), v=np.array([1.5, 0.0, 0.0]), yaw=0.4)
check("sync_to adopts position, velocity and heading exactly",
      np.allclose(t3.p_ref, [1.0, -1.0, 3.0]) and np.allclose(t3.v_ref, [1.5, 0.0, 0.0])
      and abs(t3.yaw_ref - 0.4) < 1e-12,
      f"p={np.round(t3.p_ref, 2)} v={np.round(t3.v_ref, 2)} yaw={t3.yaw_ref:.2f}")
for _ in range(400):
    t3.update(0.01)
check("after a handover the reference brakes itself to a hover (no command given)",
      float(np.linalg.norm(t3.v_ref)) < 0.02, f"|v_ref|={float(np.linalg.norm(t3.v_ref)):.4f} m/s")
check("and the hover it settles on is where it stopped, not the origin",
      abs(t3.p_ref[2] - 3.0) < 0.12 and t3.p_ref[0] > 1.0, f"p={np.round(t3.p_ref, 2)}")
check("the reference source duck-types Trajectory (duration + kind)",
      t3.duration >= HUMAN_FOREVER * 0.999 and t3.maneuver.kind == "human",
      f"duration={t3.duration:.0e} kind={t3.maneuver.kind}")

# ======================================================================================
section("B. ShiftedTrajectory: a training manoeuvre relocated onto the live hover")
# ======================================================================================
env_probe = LiveFlightEnv(telemetry=False, maneuver="hover")
traj = env_probe.sampler.sample(env_probe.np_random, mass=float(env_probe.quad.base_mass), kind="flip")
check("the sampler can produce a flip for live execution",
      getattr(traj.maneuver, "kind", "") == "flip", f"kind={traj.maneuver.kind}")

p_live = np.array([1.5, -0.7, 1.35])
yaw_live = 0.9
t0 = 37.0
shifted = ShiftedTrajectory(traj, p0=p_live, yaw0=yaw_live, t0=t0)
r0 = shifted.sample(t0)
r0_ref = traj.sample(0.0)
check("at launch the shifted manoeuvre IS the current hover position and heading",
      np.allclose(r0.p, p_live, atol=1e-9) and float(np.linalg.norm(r0.v)) < 1e-9
      and abs(yaw_of(r0.R) - yaw_live) < 1e-9,
      f"p={np.round(r0.p, 2)} |v|={float(np.linalg.norm(r0.v)):.1e} yaw={math.degrees(yaw_of(r0.R)):.1f} deg")
check("the launch state carries the manoeuvre's OWN acceleration/thrust (a flip starts with an up-push)",
      np.allclose(r0.a, r0_ref.a) and abs(r0.thrust_ff - r0_ref.thrust_ff) < 1e-9,
      f"|a|={float(np.linalg.norm(r0.a)):.2f} m/s^2 thrust={r0.thrust_ff:.3f} N "
      f"(level attitude: z_b.z={r0.R[2, 2]:.3f})")
check("and its heading is the live heading", abs(yaw_of(r0.R) - yaw_live) < 1e-9,
      f"yaw={math.degrees(yaw_of(r0.R)):.1f} deg")
check("duration accounts for the live clock offset",
      abs(shifted.duration - (t0 + traj.duration)) < 1e-9, f"{shifted.duration:.2f} s")

worst_da = worst_dw = worst_dthr = 0.0
worst_rot_v = worst_rot_R = worst_rot_a = worst_w_component = 0.0
peak_rate = 0.0
inverted_ref = False
# The relocation's own Rz, rebuilt here rather than read off the object: this section is
# supposed to be an independent statement of the contract, not a restatement of the code.
_dyaw = yaw_live - yaw_of(r0_ref.R)
_c, _s = math.cos(_dyaw), math.sin(_dyaw)
Rz = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])
for tau in np.linspace(0.0, traj.duration, 60):
    a = traj.sample(tau)
    b = shifted.sample(t0 + tau)
    worst_da = max(worst_da, abs(float(np.linalg.norm(a.a)) - float(np.linalg.norm(b.a))))
    worst_dw = max(worst_dw, float(np.linalg.norm(a.omega)) - float(np.linalg.norm(b.omega)))
    worst_dthr = max(worst_dthr, abs(a.thrust_ff - b.thrust_ff))
    # COMPONENT-WISE, which is the part norms cannot see:
    worst_rot_v = max(worst_rot_v, float(np.max(np.abs(np.asarray(b.v) - Rz @ np.asarray(a.v)))))
    worst_rot_R = max(worst_rot_R, float(np.max(np.abs(np.asarray(b.R) - Rz @ np.asarray(a.R)))))
    worst_rot_a = max(worst_rot_a, float(np.max(np.abs(np.asarray(b.a) - Rz @ np.asarray(a.a)))))
    worst_w_component = max(worst_w_component,
                            float(np.max(np.abs(np.asarray(b.omega) - np.asarray(a.omega)))))
    peak_rate = max(peak_rate, float(np.linalg.norm(a.omega)))
    inverted_ref = inverted_ref or bool(b.R[2, 2] < -0.5)
check("the relocation is a rigid motion: |a|, |omega| and thrust are preserved",
      worst_da < 1e-9 and worst_dw < 1e-9 and worst_dthr < 1e-9,
      f"max d|a|={worst_da:.1e}, d|omega|={worst_dw:.1e}, dthrust={worst_dthr:.1e}")
check("v, R and a are ROTATED by the relocation",
      worst_rot_v < 1e-9 and worst_rot_R < 1e-9 and worst_rot_a < 1e-9,
      f"max |dv|={worst_rot_v:.1e}, |dR|={worst_rot_R:.1e}, |da|={worst_rot_a:.1e}")
# THE ONE THAT WOULD HAVE CAUGHT THE BUG. A norm test cannot see a rate that has been
# rotated, because rotation preserves it. The body frame turns WITH the vehicle, so the
# body-frame rate must come through the relocation UNCHANGED; rotating it (by analogy with
# v/R/a) describes a different manoeuvre and charges the policy a w_err it cannot explain.
# The |dyaw| term below is why this matters in practice: the error grows with the heading
# difference, i.e. it is worst when the pilot is facing away from the table's own heading.
check("the relocation leaves the BODY-frame rate UNCHANGED (not rotated with v/R/a)",
      worst_w_component < 1e-9,
      f"max |domega| = {worst_w_component:.2e} rad/s at dyaw={math.degrees(_dyaw):+.0f} deg "
      f"(rotating it would displace a {peak_rate:.1f} rad/s rate by up to "
      f"{2.0 * peak_rate * abs(math.sin(_dyaw / 2.0)):.1f} rad/s)")
check("the rate channel is actually exercised (a zero-rate reference proves nothing)",
      peak_rate > 1.0, f"peak |omega| over the relocation = {peak_rate:.2f} rad/s")
check("the flip reference itself goes inverted (the training manoeuvre survived)",
      inverted_ref, "some sample has z_b.z < -0.5")

# ======================================================================================
section("C. LiveFlightEnv: live termination, state adoption, external reference")
# ======================================================================================
env = LiveFlightEnv(telemetry=False, maneuver="hover")
env.reset(options={"maneuver": "hover"})
env.quad.data.qpos[0] = 5.0                 # far outside the training sphere
env.quad.data.qpos[2] = 2.0
import mujoco  # noqa: E402
mujoco.mj_forward(env.quad.model, env.quad.data)
env.quad._update_state_properties()
check("the training flight sphere no longer terminates a live flight",
      env._check_termination() is False, f"termination_reason={env.termination_reason}")

env.quad.data.qpos[2] = 0.02
mujoco.mj_forward(env.quad.model, env.quad.data)
env.quad._update_state_properties()
check("ground contact is still terminal", env._check_termination() is True
      and env.termination_reason == "ground_crash", env.termination_reason)

# ...but a flight that STARTS on the floor may sit there: `arm_ground_start` turns floor
# contact into a runway until the vehicle has climbed above GROUND_RELEASE_Z once, after
# which the floor is terminal again (the latch is what makes a descent past it a crash).
env.arm_ground_start()
check("an armed ground start turns floor contact into a runway, not a crash",
      env._check_termination() is False and not env.airborne,
      f"armed={env.ground_start}, airborne={env.airborne}, z={float(env.quad.pos[2]):.3f} m")
env.quad.data.qpos[2] = env.GROUND_RELEASE_Z + 0.30          # take off
mujoco.mj_forward(env.quad.model, env.quad.data)
env.quad._update_state_properties()
env._check_termination()
env.quad.data.qpos[2] = 0.02                                 # ...and touch down again
mujoco.mj_forward(env.quad.model, env.quad.data)
env.quad._update_state_properties()
check("...and once it has taken off, touching down is terminal again",
      env._check_termination() is True and env.termination_reason == "ground_crash",
      f"airborne={env.airborne}")
env.arm_ground_start(False)
check("...and disarming restores the plain ground-is-terminal rule",
      env._check_termination() is True, f"ground_start={env.ground_start}")

human = HumanTarget(np.array([0.0, 0.0, 1.4]), yaw0=0.3)
env2 = LiveFlightEnv(telemetry=False, maneuver="hover")
env2.reset(options={"maneuver": "hover"})
env2.set_external_reference(human)
env2.adopt_state(np.array([0.4, -0.2, 1.4]), env2.quad.quat.copy(),
                 np.zeros(3), np.zeros(3))
obs = env2._get_stacked_obs()
check("adopt_state rebuilds the observation stack (shape + finite)",
      obs.shape == (env2.total_obs_dim,) and bool(np.all(np.isfinite(obs))),
      f"shape={obs.shape}")
check("the installer's reference is the one the actor frame compares against",
      float(np.linalg.norm(env2.ref.p - human.p_ref)) < 1e-12,
      f"ref p={np.round(env2.ref.p, 2)}")
drift = float(np.linalg.norm(env2.lighthouse.p_est - env2.quad.pos))
check("the lighthouse is re-anchored on the adopted state", drift < 0.05, f"drift={drift:.3f} m")
check("the env reports the human source as its manoeuvre kind",
      env2.traj.maneuver.kind == "human", env2.traj.maneuver.kind)

# ======================================================================================
section("D. The full stack: manual -> policy (your sticks) -> flip -> take back")
# ======================================================================================
args = types.SimpleNamespace(model="latest", dr=0.0, port=51998, no_gui=True, no_pad=True,
                             no_policy=False, radio=False, headless=True, steps=0,
                             # These checks fly hover-to-hover, so they start AIRBORNE at
                             # the training volume's centre (the old spawn). The floor
                             # start - the program's default - has its own section (G).
                             spawn=[0.0, 0.0, 1.2], spawn_yaw=0.0)
live = lf.LiveFlight(args)
check("checkpoint + encoder loaded through the shared ActorInput path",
      live.policy_ok and live.encoder_on, f"model={'loaded' if live.policy_ok else 'MISSING'}")

if live.policy_ok:
    for _ in range(100):
        live._manual_step()
    z_manual = float(live.env.quad.pos[2])
    check("manual flight hovers before the handover",
          abs(z_manual - 1.2) < 0.15 and float(np.linalg.norm(live.env.quad.vel)) < 0.5,
          f"z={z_manual:.2f} m |v|={float(np.linalg.norm(live.env.quad.vel)):.2f} m/s")

    live.enter_mode(lf.MODE_POLICY_HUMAN)
    errs_est, errs_truth = [], []
    vmax = 0.0
    for k in range(600):
        live._policy_step()
        errs_est.append(float(np.linalg.norm(live.env.lighthouse.p_est - live.env.ref.p)))
        errs_truth.append(float(np.linalg.norm(live.env.quad.pos - live.env.ref.p)))
        if k >= 400:
            vmax = max(vmax, float(np.linalg.norm(live.env.quad.vel)))
    tail_est = float(np.mean(errs_est[400:]))
    tail_truth = float(np.mean(errs_truth[400:]))
    # These bounds assert that the policy FLIES the human reference and holds station on
    # it - not a quality score. Its own hover skill is the limit here: the reward kernel
    # tolerates 0.30 m, a deterministic eval of this checkpoint measures 0.3-0.5 m, and a
    # STATIC point held for 6 s is mildly out of distribution (training references move).
    # Measured across draws: 0.27-0.67 m. What matters is that it is bounded and slow.
    check("the policy takes the hover and holds the human's reference",
          tail_est < 0.8 and vmax < 0.6,
          f"settled tracking error (estimate) {tail_est:.3f} m, truth {tail_truth:.3f} m, "
          f"peak |v| after settling {vmax:.2f} m/s")
    check("the reference really is the human target",
          live.env.traj.maneuver.kind == "human", live.env.traj.maneuver.kind)

    # the pilot commands a MODEST move (inside the training volume, where the checkpoint
    # is expected to be tight - a translating reference leaves that box and the policy's
    # corrections get lazy, which is the checkpoint's envelope, not this layer's job)
    p_before = live.env.quad.pos.copy()
    live.human.set_command(fwd=0.8, left=0.0, up=0.0)
    for _ in range(100):
        live._policy_step()
    moved = float(np.linalg.norm(live.env.quad.pos - p_before))
    speed = float(np.linalg.norm(live.env.quad.vel))
    lag = float(np.linalg.norm(live.env.quad.pos - live.env.ref.p))
    check("a stick command moves the reference and the policy follows it",
          moved > 0.15 and speed > 0.25 and lag < 0.6,
          f"moved {moved:.2f} m from the start of the move at {speed:.2f} m/s, lag {lag:.2f} m")
    live.human.clear_command()
    for _ in range(350):
        live._policy_step()
    err_hover = float(np.linalg.norm(live.env.quad.pos - live.env.ref.p))
    ref_speed = float(np.linalg.norm(live.env.ref.v))
    check("centring the sticks anchors the aim point and the policy returns to it",
          ref_speed < 0.02 and err_hover < 1.0 and float(np.linalg.norm(live.env.quad.vel)) < 0.6,
          f"reference speed {ref_speed:.3f} m/s (anchored), lag {err_hover:.2f} m, "
          f"|v|={float(np.linalg.norm(live.env.quad.vel)):.2f} m/s")

    # a flip, executed from the live hover by the trained policy. The sampler draws flips of
    # very different difficulty (measured durations 2-12 s, singles and doubles), so this
    # samples up to three draws and reports the score: the claim under test is that the
    # LAYER can hand the policy a relocated manoeuvre, not that every draw is easy.
    z_before = float(live.env.quad.pos[2])
    attempts, successes = 0, 0
    best = ""
    for _attempt in range(3):
        attempts += 1
        live.launch_trajectory("flip")
        inverted = False
        upright_again = False
        spin = 0.0
        zmin = 9.9
        steps = 0
        while live.mode == lf.MODE_POLICY_TRAJ and steps < 2500:
            live._policy_step()
            steps += 1
            q = live.env.quad
            spin += float(np.linalg.norm(q.omega)) * live.dt     # sign-agnostic rotation
            inverted = inverted or bool(q.dcm[2, 2] < -0.5)
            if inverted and q.dcm[2, 2] > 0.8:
                upright_again = True
            zmin = min(zmin, float(q.pos[2]))
        ok_draw = inverted and upright_again and spin > math.radians(300.0)
        successes += int(ok_draw)
        best = (f"draw {attempts}: inverted={inverted}, recovered={upright_again}, "
                f"rotated {math.degrees(spin):.0f} deg in {steps} steps, min z={zmin:.2f} m")
        if ok_draw:
            break
        for _ in range(200):                     # settle back onto the hover point
            live._policy_step()
    check("the layer can hand the policy a relocated flip from a live hover",
          successes > 0, f"{successes}/{attempts} draws flipped - {best}")
    check("it keeps altitude through the flip (no crash)",
          zmin > 0.35, f"min z={zmin:.2f} m (started {z_before:.2f} m)")
    check("the manoeuvre hands back to POLICY hover when it finishes",
          live.mode == lf.MODE_POLICY_HUMAN, live.mode)

    # take back control with the controller, from a settled hover (as a pilot would)
    for _ in range(100):
        live._policy_step()
    policy_thrust = live.last_thrust
    live.enter_mode(lf.MODE_MANUAL_ACRO)
    hover = float(live.ctrl.hover_throttle)
    max_thr = float(live.env.quad.params["maxThr"])
    expected = float(np.clip(np.clip(policy_thrust / max_thr, 0.0, 1.0), 0.8 * hover, 1.2 * hover))
    check("taking back control inherits the collective, clamped to a flyable band",
          abs(live.cmd.throttle - expected) < 1e-9
          and any("collective" in msg for msg in live.messages),
          f"collective {100 * live.cmd.throttle:.0f}% "
          f"(policy {100 * policy_thrust / max_thr:.0f}%, hover {100 * hover:.0f}%)")
    for _ in range(100):
        live._manual_step()
    check("the manual loop flies on after the handover",
          bool(np.all(np.isfinite(live.env.quad.state))) and not live.env.quad.check_ground_contact(),
          f"z={float(live.env.quad.pos[2]):.2f} m")

    # ---- PAD-DRIVEN paths (the coverage gap that let two pad bugs through) ---------
    # Both control loops must run with a CONNECTED pad, and the policy must translate the
    # stick intent into the matching reference command. NOTE: this runs BEFORE the panel
    # command block below, because those commands deliberately re-map the sticks - an
    # earlier version of this test ran them the other way round and "proved" the mapping
    # wrong while it was reading a swapped one.
    def syn(Lx=0.0, Ly=0.0, Rx=0.0, Ry=0.0, pressed=(), buttons=()):
        return gp.PadState(connected=True, name="synthetic", left_x=Lx, left_y=Ly,
                           right_x=Rx, right_y=Ry, pressed=tuple(pressed),
                           buttons=set(buttons), raw_axes={}, analog_triggers=0)

    live.pad_state = syn()
    live._manual_step()                       # pad path in manual mode (shim + buttons)
    check("manual mode runs with a pad connected (no crash, sticks reach the loop)",
          bool(np.all(np.isfinite(live.env.quad.state))) and live.mode == lf.MODE_MANUAL_ACRO,
          f"mode={live.mode}")

    live.enter_mode(lf.MODE_POLICY_HUMAN)
    live.pad_state = syn(Ly=RAW_UP)           # left stick forward = movement
    live._policy_step()
    fwd, left_cmd, up_cmd, yaw_cmd = live._last_cmd
    check("policy mode: left stick forward commands forward",
          fwd > 0.5 and abs(up_cmd) < 1e-9 and abs(left_cmd) < 1e-9,
          f"cmd fwd {fwd:+.2f} left {left_cmd:+.2f} up {up_cmd:+.2f}")
    live.pad_state = syn(Lx=RAW_RIGHT)        # left stick right = strafe RIGHT
    live._policy_step()
    fwd, left_cmd, up_cmd, yaw_cmd = live._last_cmd
    check("policy mode: left stick right commands a rightward strafe",
          left_cmd < -0.5 and abs(fwd) < 1e-9 and abs(up_cmd) < 1e-9,
          f"cmd fwd {fwd:+.2f} left {left_cmd:+.2f} up {up_cmd:+.2f} (left is negative = right)")
    live.pad_state = syn(Rx=RAW_RIGHT)        # right stick X = heading
    live._policy_step()
    fwd, left_cmd, up_cmd, yaw_cmd = live._last_cmd
    check("policy mode: right stick right commands a right turn",
          yaw_cmd < -0.1, f"yaw_rate {yaw_cmd:+.2f} rad/s (negative = nose right)")
    live.pad_state = syn(buttons=(lf.PAD_BTN_CLIMB,))     # ZR = climb
    live._policy_step()
    fwd, left_cmd, up_cmd, yaw_cmd = live._last_cmd
    check("policy mode: ZR (right trigger button) commands a climb",
          up_cmd > 0.5 and abs(fwd) < 1e-9, f"up {up_cmd:+.2f} m/s")
    live.pad_state = syn(buttons=(lf.PAD_BTN_DESCEND,))   # ZL = descend
    live._policy_step()
    fwd, left_cmd, up_cmd, yaw_cmd = live._last_cmd
    check("policy mode: ZL commands a descent", up_cmd < -0.5, f"up {up_cmd:+.2f} m/s")
    live.pad_state = syn(Ly=RAW_UP)
    live._policy_step()
    target = live.env.ref.p.copy()
    for _ in range(50):
        live._policy_step()
    moved = float(np.linalg.norm(live.env.ref.p - target))
    check("holding a stick keeps moving the aim point (and the policy drives at it)",
          moved > 0.2, f"reference advanced {moved:.2f} m in 0.5 s of held stick")

    # the pad buttons that switch modes must work through the pad path too
    live.pad_state = syn(pressed=(lf.PAD_BTN_MODE_ALT,))
    live._policy_step()
    check("the pad's '-' button hands control back from the pad path",
          live.mode == lf.MODE_MANUAL_ACRO, live.mode)

    # GAME mode runs through the same shim (its sticks are the video-game ones)
    live.enter_mode(lf.MODE_MANUAL_GAME)
    live.pad_state = syn(Ly=RAW_UP)
    live._manual_step()
    check("manual GAME runs with a pad and flies forward on the left stick",
          live.cmd.mode == lf.mf.MODE_GAME and live.cmd.v_cmd[0] > 1.0
          and bool(np.all(np.isfinite(live.env.quad.state))),
          f"mode={live.cmd.mode} v_cmd={np.round(live.cmd.v_cmd, 2)}")
    live.pad_state = syn(buttons=(lf.PAD_BTN_FLIP,), Lx=RAW_RIGHT)
    live._manual_step()                          # first step: the pop (rotation gated)
    live.ctrl._flip_t = lf.mf.GAME_FLIP_POP_S + 0.05
    live._manual_step()                          # past the pop: the roll is live
    thrust, omega = live.ctrl.update()
    check("manual GAME: the top-right shoulder button rolls right (thrust is flown for "
          "the pilot by the flip profile)",
          live.cmd.game_flip and omega[0] > 5.0,
          f"roll {omega[0]:+.1f} rad/s")
    live.pad_state = syn()
    live._manual_step()

    # panel command surface (the exact strings the panel emits; these DO remap the pad)
    with tempfile.TemporaryDirectory() as tmp:
        lf.PAD_CALIBRATION_PATH = os.path.join(tmp, "cal.json")
        for cmd in ("mode:manual_assisted", "mode:policy_human", "traj:hover",
                    "pad:calibrate", "pad:save", "pad:layout:mode2", "pad:layout:ow",
                    "pad:swap_sticks", "pad:invert:lx:1", "pad:invert:lx:0",
                    "flight:respawn", "mode:manual_acro", "mode:manual_game",
                    "mode:manual_acro"):
            live.handle_gui(cmd)
    check("every panel command is accepted", live.mode == lf.MODE_MANUAL_ACRO,
          f"mode={live.mode}, {len(live.messages)} messages")
    # ...one of those commands was the one-click stick SWAP, which is a TOGGLE: restore
    # the default mapping before the stick-driven policy tests below.
    live.mapper = PadMapper()

    # The telemetry log must never be able to end a flight: a new mode once crashed the
    # whole program with `KeyError: 'MANUAL (game)'` AFTER a successful flight.
    logged_modes = []
    saved_mode = live.mode
    for mode in (lf.MODE_MANUAL_GAME, lf.MODE_MANUAL_ACRO, lf.MODE_MANUAL_ASSISTED,
                 lf.MODE_POLICY_HUMAN):
        live.enter_mode(mode)
        (live._policy_step if mode.startswith("POLICY") else live._manual_step)()
        try:
            live._push_log()
            logged_modes.append(int(live.log["mode"][-1]))
        except Exception as exc:
            logged_modes.append(f"EXC {exc}")
    live.enter_mode(saved_mode)
    check("telemetry logging works in every mode (an unknown mode logs as -1, never raises)",
          all(isinstance(m, int) for m in logged_modes)
          and logged_modes == [lf.MODE_CODES[m] for m in (lf.MODE_MANUAL_GAME, lf.MODE_MANUAL_ACRO,
                                                          lf.MODE_MANUAL_ASSISTED,
                                                          lf.MODE_POLICY_HUMAN)],
          f"codes {logged_modes}")

    # The flip button in POLICY mode: the pilot rolls it, the policy takes it back.
    live.enter_mode(lf.MODE_POLICY_HUMAN)
    live.pad_state = syn()
    for _ in range(200):
        live._policy_step()
    z_before = float(live.env.quad.pos[2])
    inverted = False
    zmin = 9.9
    for k in range(80):                       # pop + one full turn, held
        live.pad_state = syn(Lx=RAW_RIGHT, buttons=(lf.PAD_BTN_FLIP,))
        live._policy_step()
        inverted = inverted or bool(live.env.quad.dcm[2, 2] < -0.5)
        zmin = min(zmin, float(live.env.quad.pos[2]))
        if k == 30:
            check("policy flip: the pilot's roll is applied through the env's action interface",
                  live.last_source == "pilot-flip"
                  and live.last_omega[0] > 5.0,
                  f"source={live.last_source} roll={math.degrees(live.last_omega[0]):+.0f} dps "
                  f"(a body-rate request through the env action, so the trained plant and "
                  f"rate loop fly it)")
    live.pad_state = syn()
    for _ in range(250):
        live._policy_step()
    check("policy flip: a full turn stays airborne and the policy re-catches it",
          inverted and zmin > 0.5 and live.last_source == "policy"
          and float(live.env.quad.dcm[2, 2]) > 0.7 and float(live.env.quad.pos[2]) > 0.6,
          f"inverted={inverted}, min z={zmin:.2f} m (started {z_before:.2f}), "
          f"recovered to z={float(live.env.quad.pos[2]):.2f} m by the policy")
else:
    print("  [info] no checkpoint - policy sections skipped (manual flight still runs)")

# ======================================================================================
section("E. Pad mapper + guided calibration (the 'left acts like right' fix)")
# ======================================================================================
def pad(Lx=0.0, Ly=0.0, Rx=0.0, Ry=0.0):
    """Raw device values. The DEFAULT convention (which the mapper's own defaults and
    gamepad.py both document) is right = +1 and up = -1 on every axis."""
    return gp.PadState(connected=True, name="synthetic", left_x=Lx, left_y=Ly,
                       right_x=Rx, right_y=Ry)


def push_right(raw_left_axis: bool) -> float:
    """Raw value produced by pushing the given stick RIGHT (default convention)."""
    return +1.0


mapper = PadMapper()
roles = mapper.roles(pad(Lx=RAW_RIGHT))
check("Outer-Wilds layout: left stick right = roll right (travels right)",
      roles["roll"] > 0.5 and roles["yaw"] == 0.0, f"roll={roles['roll']:+.2f}")
roles = mapper.roles(pad(Rx=RAW_RIGHT))
check("Outer-Wilds layout: right stick right = turn right", roles["yaw"] > 0.5, f"yaw={roles['yaw']:+.2f}")
roles = mapper.roles(pad(Ly=RAW_UP))
check("left stick up = more collective, not a sign fight", roles["throttle"] > 0.5,
      f"throttle={roles['throttle']:+.2f}")
roles = mapper.roles(pad(Ry=RAW_UP))
check("right stick up = nose down (aircraft sense)", roles["pitch"] > 0.5, f"pitch={roles['pitch']:+.2f}")
roles = mapper.roles(pad(Lx=RAW_RIGHT))
check("a right push on the left stick commands a RIGHTWARD strafe (left is negative)",
      roles["move_left"] < -0.5, f"move_left={roles['move_left']:+.2f}")
roles = mapper.roles(pad(Ly=RAW_UP))
check("movement forward comes from the LEFT stick, video-game style",
      roles["move_fwd"] > 0.5, f"move_fwd={roles['move_fwd']:+.2f}")
roles = mapper.roles(pad(Rx=RAW_RIGHT))
check("heading comes from the RIGHT stick's X (right = turn right)",
      roles["heading"] > 0.5, f"heading={roles['heading']:+.2f}")
mapper.set_layout("mode2")
roles = mapper.roles(pad(Lx=RAW_RIGHT, Rx=RAW_RIGHT))
check("Mode 2 layout: left stick right = yaw right, right stick right = roll right",
      roles["yaw"] > 0.5 and roles["roll"] > 0.5, f"yaw={roles['yaw']:+.2f} roll={roles['roll']:+.2f}")
mapper.set_layout("ow")

# The measured fix: a pad whose axes arrive swapped + inverted (the reported symptom).
def wired_pad(Lx=0.0, Ly=0.0, Rx=0.0, Ry=0.0):
    """Raw attributes as a badly-wired pad reports them, for physical Lx/Ly/Rx/Ry.

    Physical RIGHT is +1 and physical UP is -1 (the default convention), but this pad
    reports the left stick's Y where the X belongs, the right stick's Y where the X
    belongs, and with the opposite sign.
    """
    return gp.PadState(connected=True, name="wired",
                       left_x=-Ly, left_y=-Lx, right_x=-Ry, right_y=-Rx)


messy = PadMapper()
before = messy.roles(wired_pad(Lx=-1.0))        # the pilot pushes the LEFT stick LEFT
check("before calibration the messy pad really is wrong (that is the bug)",
      abs(before["lx"]) < 1e-9 and abs(before["ly"]) > 0.5,
      f"a LEFT push reads lx={before['lx']:+.2f} ly={before['ly']:+.2f} (leaks into the other stick)")

messy.calibrate_start()
now = time.time()
messy.calibrate_update(pad(), now=now)          # sticks at rest: the routine starts step 1
now += 0.05
for move in (wired_pad(Lx=-1.0), wired_pad(Ly=-1.0), wired_pad(Rx=-1.0), wired_pad(Ry=-1.0)):
    messy.calibrate_update(move, now=now)       # the pilot performs the instruction
    now += 0.05
    if messy.cal.done:
        break
    messy.calibrate_update(pad(), now=now)      # back to centre, ready for the next step
    now += 0.05
check("the 4-step routine completes on a swapped + inverted pad",
      messy.cal.done, messy.calibration_report())
after_left = messy.roles(wired_pad(Lx=-1.0))
after_up = messy.roles(wired_pad(Ly=-1.0))
after_right = messy.roles(wired_pad(Rx=-1.0))
after_ry = messy.roles(wired_pad(Ry=-1.0))
check("after calibration a physical LEFT push reads as LEFT",
      after_left["lx"] < -0.5, f"lx={after_left['lx']:+.2f}")
check("...and the other three channels land correctly too",
      after_up["ly"] > 0.5 and after_right["rx"] < -0.5 and after_ry["ry"] > 0.5,
      f"ly={after_up['ly']:+.2f} rx={after_right['rx']:+.2f} ry={after_ry['ry']:+.2f}")

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "cal.json")
    messy.save(path)
    loaded = PadMapper.load(path)
    check("the calibration saves and reloads",
          loaded.roles(wired_pad(Lx=-1.0))["lx"] < -0.5,
          str(loaded.export()["axes"]))

swapped = PadMapper()
swapped.swap_sticks()
left_push = swapped.sticks(pad(Lx=push_right(True)))
check("one-click stick swap fixes a pad whose sticks arrive in the wrong order",
      left_push["rx"] > 0.5 and abs(left_push["lx"]) < 1e-9,
      f"a left-stick push now feeds rx={left_push['rx']:+.2f}")

# ======================================================================================
section("F. Panel protocol and the radio-ready link")
# ======================================================================================
port = fg._free_port()
server = fg.GuiServer(port=port)
started = server.start()
check("the sim-side panel server starts", started, f"port {port}")
client = socket.create_connection(("127.0.0.1", port), timeout=2.0)
client.sendall(fg.encode({"type": "command", "command": "traj:flip"}))
deadline = time.time() + 2.0
got = []
while time.time() < deadline and not got:
    got = server.poll()
    time.sleep(0.02)
check("a panel command reaches the sim", got and got[0].get("command") == "traj:flip", str(got))
server.publish({"mode": "POLICY (trajectory)", "z": 1.2})
decoder = fg.LineDecoder()
msgs = []
deadline = time.time() + 2.0
while time.time() < deadline and not msgs:
    try:
        msgs += decoder.feed(client.recv(4096))
    except socket.timeout:
        pass
check("status flows back to the panel",
      any(m.get("type") == "status" for m in msgs), f"{len(msgs)} messages")
client.close()
server.stop()

check("the panel's command list is exactly what handle_gui understands",
      all(c["cmd"].split(":")[0] in ("mode", "traj") for c in fg.MODES + fg.TRAJECTORIES),
      f"{len(fg.MODES)} modes, {len(fg.TRAJECTORIES)} trajectories")

link = NullLink()
sp = RadioSetpoint.from_sim(t=1.5, thrust_newtons=0.3, omega_rads=np.array([1.0, -2.0, 0.5]),
                            max_thrust=0.6, source="policy")
link.send(sp)
clamped = RadioSetpoint.from_sim(t=0.0, thrust_newtons=9.0, omega_rads=np.array([50.0, 0.0, 0.0]),
                                 max_thrust=0.6).clamped()
check("commands convert to the vehicle's own units (deg/s + thrust %)",
      abs(sp.roll_rate_dps - math.degrees(1.0)) < 1e-9
      and abs(sp.thrust_pct - 50.0) < 1e-9 and link.sent == 1,
      sp.summary())
check("clamping respects the gyro range and the thrust range",
      abs(clamped.roll_rate_dps - math.degrees(20.0)) < 1e-9 and clamped.thrust_pct == 100.0,
      clamped.summary())

# ======================================================================================
section("G. Starting on the floor (tweakable start pose)")
# ======================================================================================
gargs = types.SimpleNamespace(model="latest", dr=0.0, port=51999, no_gui=True, no_pad=True,
                              no_policy=False, radio=False, headless=True, steps=0)


def floor_pad(**kw):
    """A connected synthetic pad (section D's helper is nested inside its own block)."""
    buttons = tuple(kw.get("buttons", ()))
    return gp.PadState(connected=True, name="synthetic", left_x=kw.get("lx", 0.0),
                       left_y=kw.get("ly", 0.0), right_x=kw.get("rx", 0.0),
                       right_y=kw.get("ry", 0.0), pressed=buttons, buttons=set(buttons),
                       raw_axes={}, analog_triggers=0)


floor = lf.LiveFlight(gargs)                # the default --spawn: on the floor
check("a live flight starts ON THE FLOOR by default, runway armed",
      float(floor.env.quad.pos[2]) < 0.05 and bool(floor.env.ground_start)
      and not floor.env.airborne,
      f"z={float(floor.env.quad.pos[2]):.3f} m (leg rest 0.013), "
      f"z_des={floor.cmd.z_des:.3f} m, runway={floor.env.ground_start}")
for _ in range(60):
    floor._manual_step()
check("...and idling on the floor does not end the flight",
      (not floor.crashed) and floor.env._check_termination() is False,
      f"crashed={floor.crashed}, z={float(floor.env.quad.pos[2]):.3f} m, "
      f"reason={floor.env.termination_reason}")

# The policy may be handed the vehicle while it is still on the floor: the reference sits
# at the ground with it, and the flight must survive holding there (the runway covers the
# untrained low-altitude regime until the pilot climbs away).
#
# GUARDED on policy_ok, like section D. Without a usable checkpoint `_policy_step` RAISES
# ("policy: no checkpoint at ... - policy modes disabled"), which ABORTED this script
# instead of skipping: the whole suite became unrunnable from a state as ordinary as "the
# committed checkpoints predate an observation-layout change". A missing policy is a
# configuration, not a test failure - the manual sections are still worth running.
if floor.policy_ok:
    floor.enter_mode(lf.MODE_POLICY_HUMAN)
    floor.pad_state = None
    for _ in range(20):
        floor._policy_step()
    check("the policy can be handed a vehicle that is still on the floor",
          (not floor.crashed) and floor.running and float(floor.env.ref.p[2]) < 0.10,
          f"source={floor.last_source}, ref z={float(floor.env.ref.p[2]):.3f} m, "
          f"thrust {100 * floor.last_thrust / float(floor.env.quad.params['maxThr']):.0f}%")
else:
    print("  [info] no checkpoint - the floor handover check is skipped")
floor.enter_mode(lf.MODE_MANUAL_GAME)

# Take off the way the pilot does: GAME mode with ZR held (which steps the altitude
# setpoint up past manual_flight's Z_MIN floor and lets the altitude law do the flying).
climbed_early = 0.0
for k in range(300):
    floor.pad_state = floor_pad(buttons=(lf.PAD_BTN_CLIMB,))
    floor._manual_step()
    if k == 40:
        climbed_early = float(floor.env.quad.pos[2])
floor.pad_state = floor_pad()
climbed = float(floor.env.quad.pos[2])
check("GAME mode takes off from the floor on ZR (the altitude law does the flying)",
      climbed > 0.25 and not floor.crashed and floor.env.airborne,
      f"z {climbed_early:.2f} m after 0.4 s -> {climbed:.2f} m after 3 s, "
      f"airborne={floor.env.airborne}")
check("...and after the takeoff the floor is a crash surface again",
      floor.env.ground_touch_is_fatal() is True, "runway spent")

floor.respawn()
check("respawn returns to the START pose (back on the floor, runway re-armed)",
      float(floor.env.quad.pos[2]) < 0.05 and bool(floor.env.ground_start)
      and not floor.env.airborne and not floor.crashed,
      f"z={float(floor.env.quad.pos[2]):.3f} m, z_des={floor.cmd.z_des:.3f} m")

# --spawn / --spawn-yaw move the start pose; a start ABOVE the release height spawns in a
# hover instead and keeps the floor terminal from the first step (the old behaviour).
pargs = types.SimpleNamespace(model="latest", dr=0.0, port=52000, no_gui=True, no_pad=True,
                              no_policy=True, radio=False, headless=True, steps=0,
                              spawn=[0.6, -0.4, 0.9], spawn_yaw=90.0)
placed = lf.LiveFlight(pargs)
check("--spawn / --spawn-yaw are honoured (x, y, z and heading)",
      bool(np.allclose(np.array([0.6, -0.4, 0.9]), placed.env.quad.pos, atol=0.02))
      and abs(math.degrees(float(placed.env.quad.psi)) - 90.0) < 2.0,
      f"pos {np.round(placed.env.quad.pos, 2)}, heading "
      f"{math.degrees(float(placed.env.quad.psi)):+.1f} deg")
check("...and an airborne start keeps the floor terminal (no runway)",
      (not placed.env.ground_start) and placed.env.ground_touch_is_fatal() is True,
      f"ground_start={placed.env.ground_start}")

# ======================================================================================
section("H. The training pipeline is untouched")
# ======================================================================================
for name in ("quad_flip_env.py", "trajectories.py", "train.py", "evaluate.py",
             "actor_input.py", "encoder/latent_injector.py", "encoder/history_encoder.py"):
    path = os.path.join(_SIM, name)
    with open(path, "r") as f:
        text = f.read()
    check(f"{name} has no live-flight edits",
          "live_target" not in text and "LiveFlight" not in text and "pad_mapper" not in text,
          "no imports or hooks added")

print()
print("=" * 78)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
