"""
Validation for actor-frame anchoring (quad_flip_env.ACTOR_FRAME_MODE = "anchored_xy").

The claim is worth stating exactly:

    Translating an entire episode horizontally by delta - vehicle, reference and guard
    together - leaves the ACTOR INPUT bit-identical, and therefore any fixed policy's
    action bit-identical.

If that holds, the policy is genuinely position-agnostic: it does not matter where in the
room you start, and the arbitrary origin of a Lighthouse geometry calibration stops being
a hidden input to the controller. Before this change the same experiment moved the 15M
checkpoint's rate command by 54% of its own magnitude at delta = 0.5 m, because o_t[0:3]
carried the raw room-frame estimate.

WHAT IS CHECKED
  A. THE FRAME. x,y read exactly 0 at reset (wherever the spawn is); z stays absolute; the
     error channels are NOT double-subtracted; estimate+anchor moved together leaves the
     frame bit-identical.
  B. THE ENCODER INHERITS IT. The encoder frame is sliced out of o_t, so it must be
     invariant too - that is what stops absolute x,y leaking back in through z.
  C. THE GUARD MOVED WITH THE ANCHOR. The volume test is measured from the episode anchor,
     so an episode translated by delta terminates at the same RELATIVE excursion.
  D. WHOLE EPISODE. Two identically-seeded envs, one translated (state + relocated
     trajectory via the production `ShiftedManeuver`), stepped with the same actions for
     150 steps: the actor frame sequence must match.
  E. NULLS. A vertical translation must NOT be invariant (the ground is real), and the
     ablation switch must restore the pre-change absolute layout.

Run:  .venv/bin/python scratch/check_frame_anchor.py
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

import mujoco  # noqa: E402
import quad_flip_env as E  # noqa: E402
from quad_flip_env import (  # noqa: E402
    ACTOR_FRAME_MODE,
    ANCHOR_ACTOR_XY,
    O_POS,
    O_P_ERR,
    O_V_ERR,
    QuadFlipEnv,
)
from trajectories import ShiftedManeuver, Trajectory  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def info(msg: str) -> None:
    print(f"  [info] {msg}")


def make_env() -> QuadFlipEnv:
    env = QuadFlipEnv(episode_seconds=15.0, telemetry=True)
    env.set_dr_level(1.0)          # full randomisation: the spawn must vary
    return env


print("=" * 78)
print(f"A. the actor frame  (ACTOR_FRAME_MODE = {ACTOR_FRAME_MODE!r})")
print("=" * 78)

check("the build is in the anchored configuration",
      bool(ANCHOR_ACTOR_XY) and ACTOR_FRAME_MODE == "anchored_xy", f"{ACTOR_FRAME_MODE}")

env = make_env()
env.reset(seed=3, options={"maneuver": "orbit"})
frame = env._compute_actor_obs()
anchor = env.anchor_pos.copy()
info(f"spawn/anchor = ({anchor[0]:+.3f}, {anchor[1]:+.3f}, {anchor[2]:+.3f}) m "
     f"(drawn per episode, so this is a real offset)")
check("x,y of the first frame read exactly 0",
      abs(float(frame[O_POS])) < 1e-6 and abs(float(frame[O_POS + 1])) < 1e-6,
      f"o_t[0:2] = ({float(frame[O_POS]):+.2e}, {float(frame[O_POS + 1]):+.2e})")
est_z = float(env.lighthouse.p_est[2])
check("z is still ABSOLUTE (the ground cue survives)",
      abs(float(frame[O_POS + 2]) - est_z) < 1e-6,
      f"o_t[2] = {float(frame[O_POS + 2]):.4f} vs the estimate z {est_z:.4f}")

_dp = env.ref.p - env.lighthouse.p_est
check("p_err is not anchored (it is already a difference)",
      bool(np.allclose(frame[O_P_ERR:O_P_ERR + 3], _dp, atol=1e-5)),
      f"max |p_err - (ref.p - est)| = {float(np.abs(frame[O_P_ERR:O_P_ERR + 3] - _dp).max()):.2e}")
_dv = env.ref.v - env.lighthouse.v_est
check("v_err is not anchored either",
      bool(np.allclose(frame[O_V_ERR:O_V_ERR + 3], _dv, atol=1e-5)),
      f"max |v_err - (ref.v - est_v)| = {float(np.abs(frame[O_V_ERR:O_V_ERR + 3] - _dv).max()):.2e}")

f_before = frame.copy()
# A translation moves the ESTIMATE, the ANCHOR and the REFERENCE together. Moving only the
# first two would be a different experiment (the vehicle moved relative to its reference),
# and p_err - which is ref.p - est and is already a difference - would rightly change.
#
# `_compute_actor_obs()` DRAWS FRESH SENSOR NOISE on every call, so the RNG is reseeded
# identically before each read: otherwise this comparison would measure noise, not
# geometry. (Noise is drawn from a fixed distribution that does not depend on position, so
# holding it equal is exactly the right control.)
def frame_reseeded(env_: QuadFlipEnv, seed: int = 4242) -> np.ndarray:
    env_.np_random = np.random.default_rng(seed)
    return env_._compute_actor_obs()


f_before = frame_reseeded(env)
env.anchor_pos[:2] += 2.0
env.lighthouse.p_est[:2] += 2.0
env.ref.p[:2] += 2.0
f_after = frame_reseeded(env)
check("estimate + anchor + reference translated together -> frame BIT-IDENTICAL",
      bool(np.array_equal(f_before, f_after)),
      f"max |dframe| = {float(np.abs(f_after - f_before).max()):.2e}")
env.anchor_pos[:2] -= 2.0
env.lighthouse.p_est[:2] -= 2.0
env.ref.p[:2] -= 2.0

print()
print("=" * 78)
print("B. the encoder's input frame inherits the anchoring (z cannot leak absolute x,y)")
print("=" * 78)

enc_before = env.get_encoder_frame().copy()
# NOTE: with observation latency the encoder consumes the DELAYED frame, so the comparison
# has to be against the same buffer the encoder is actually fed, not against the instant's
# frame. (At dr=1.0 the latency is drawn up to OBS_LATENCY_MAX_STEPS.)
delayed_before = np.asarray(env.obs_history_buffer[-1], dtype=np.float32).copy()
check("the encoder frame's leading 29 dims ARE the (delayed) actor frame it consumes",
      bool(np.array_equal(enc_before[:29], delayed_before)),
      f"max |diff| = {float(np.abs(enc_before[:29] - delayed_before).max()):.2e}")
env.anchor_pos[:2] += 3.5
env.lighthouse.p_est[:2] += 3.5
env.ref.p[:2] += 3.5
enc_after = env.get_encoder_frame().copy()
check("the encoder frame is identical under the same translation",
      bool(np.array_equal(enc_before, enc_after)),
      f"max |dframe| = {float(np.abs(enc_after - enc_before).max()):.2e}")
env.anchor_pos[:2] -= 3.5
env.lighthouse.p_est[:2] -= 3.5
env.ref.p[:2] -= 3.5

print()
print("=" * 78)
print("C. the flight-volume guard moved with the anchor")
print("=" * 78)

env2 = make_env()
env2.reset(seed=11, options={"maneuver": "hover"})
R = float(env2.flight_radius)


def place(env_: QuadFlipEnv, x: float, y: float, z: float) -> None:
    env_.quad.data.qpos[0] = x
    env_.quad.data.qpos[1] = y
    env_.quad.data.qpos[2] = z
    mujoco.mj_forward(env_.quad.model, env_.quad.data)
    env_.quad._update_state_properties()


for dx in (0.0, 1.7, -2.4):
    env2.anchor_pos = np.array([dx, -0.6, 1.2], dtype=np.float64)
    place(env2, dx + R * 0.98, -0.6, 1.2)
    inside = not env2._check_termination()
    place(env2, dx + R * 1.02, -0.6, 1.2)
    outside = env2._check_termination()
    check(f"anchor x={dx:+.1f}: inside the sphere is safe, 2% outside is not",
          inside and outside and env2.termination_reason == "out_of_volume",
          f"inside={inside}, outside={outside} ({env2.termination_reason})")

print()
print("=" * 78)
print("D. whole episode: same actions, translated episode, identical frame sequence")
print("=" * 78)

DELTA = np.array([0.9, -0.7])           # small enough to leave the station geometry alone
DELTA3 = np.array([DELTA[0], DELTA[1], 0.0])

env_a = make_env()
env_a.reset(seed=5, options={"maneuver": "figure8"})
env_b = make_env()
env_b.reset(seed=5, options={"maneuver": "figure8"})

# Episode B is episode A translated by DELTA. The plant, the estimate, the lighthouse
# (its stations and its own range anchor) and the REFERENCE all move - the reference via
# the production relocation helper, so this exercises the same code the live layer uses.
p_s = env_b.traj.sample(0.0).p.copy()
env_b.traj = Trajectory(ShiftedManeuver(env_b.traj.maneuver, p_s + DELTA3, 0.0))
env_b.ref = env_b.traj.sample(env_b.t)
env_b.anchor_pos[:2] += DELTA
env_b.quad.data.qpos[0:2] += DELTA
env_b.quad.data.qpos[2] = env_a.quad.data.qpos[2]
mujoco.mj_forward(env_b.quad.model, env_b.quad.data)
env_b.quad._update_state_properties()
env_b.lighthouse.p_est[:2] += DELTA
env_b.lighthouse.stations[:, :2] += DELTA
env_b.lighthouse._anchor[:2] += DELTA

rng = np.random.default_rng(0)
actions = [rng.uniform(-0.35, 0.35, size=4).astype(np.float32) for _ in range(150)]
fa, fb, ra, rb, ta, tb = [], [], [], [], [], []
for a in actions:
    _, rewa, term_a, trunc_a, _ = env_a.step(a)
    _, rewb, term_b, trunc_b, _ = env_b.step(a)
    fa.append(env_a._compute_actor_obs().copy())
    fb.append(env_b._compute_actor_obs().copy())
    ra.append(rewa)
    rb.append(rewb)
    ta.append(bool(term_a))
    tb.append(bool(term_b))
    if term_a or trunc_a or term_b or trunc_b:
        break

Fa, Fb = np.asarray(fa), np.asarray(fb)
check("the two episodes ran the same number of steps (same termination)",
      len(fa) == len(fb) and ta == tb,
      f"{len(fa)} steps, terminations {sum(ta)} vs {sum(tb)}")
check("translated episode produces the same actor FRAME sequence",
      bool(np.allclose(Fa, Fb, atol=1e-5, rtol=0.0)),
      f"max |dframe| = {float(np.abs(Fa - Fb).max()):.2e} over {len(fa)} steps")
check("...and the same REWARD sequence",
      bool(np.allclose(np.asarray(ra), np.asarray(rb), atol=1e-5, rtol=0.0)),
      f"max |dreward| = {float(np.abs(np.asarray(ra) - np.asarray(rb)).max()):.2e}")

print()
print("=" * 78)
print("E. nulls: what must NOT be invariant")
print("=" * 78)

env_z = make_env()
env_z.reset(seed=7, options={"maneuver": "hover"})
fz_before = env_z._compute_actor_obs().copy()
env_z.anchor_pos[2] += 1.0          # a VERTICAL translation moves the ground with it
env_z.lighthouse.p_est[2] += 1.0
env_z.ref.p[2] += 1.0
fz_after = env_z._compute_actor_obs().copy()
check("a VERTICAL translation DOES change the frame (z stays absolute)",
      not bool(np.array_equal(fz_before, fz_after)),
      f"o_t[2] {float(fz_before[O_POS + 2]):.3f} -> {float(fz_after[O_POS + 2]):.3f}")

saved = E.ANCHOR_ACTOR_XY
try:
    E.ANCHOR_ACTOR_XY = False
    env_abs = make_env()
    env_abs.reset(seed=3, options={"maneuver": "orbit"})
    frame_abs = env_abs._compute_actor_obs()
    check("the ablation (ANCHOR_ACTOR_XY=False) restores room coordinates",
          bool(np.allclose(frame_abs[O_POS:O_POS + 3], env_abs.lighthouse.p_est, atol=1e-6)),
          f"o_t[0:3] = ({float(frame_abs[O_POS]):+.3f}, {float(frame_abs[O_POS + 1]):+.3f}, "
          f"{float(frame_abs[O_POS + 2]):+.3f})")
finally:
    E.ANCHOR_ACTOR_XY = saved

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("RESULT: frame anchoring verified")
