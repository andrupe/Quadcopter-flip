"""
Phase 1 sanity checks for the frozen-history-encoder data path.

Verifies the new observation layout, the encoder aux channels, the DR headroom
knob, and - most importantly - cross-validates get_priv_targets() against the
privileged vector that the critic already consumed before this change.

Run:  .venv/bin/python scratch/check_phase1.py
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
    ACTOR_TOTAL_DIM,
    ENCODER_AUX_DIM,
    PRIV_TARGET_DIM,
    PRIV_TARGET_GROUPS,
    PRIVILEGED_OBS_DIM,
    TOTAL_OBS_DIM,
    QuadFlipEnv,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def make_env() -> QuadFlipEnv:
    return QuadFlipEnv(
        episode_seconds=4.0,
        obs_noise=True,
        random_wind=True,
        random_initial_state=True,
        random_battery=True,
        curriculum_arena=False,
        arena_radius_start=4.0,
        arena_radius_end=4.0,
    )


print("=" * 78)
print("A. observation layout")
print("=" * 78)
env = make_env()
check("ACTOR_SINGLE_OBS_DIM == 29", ACTOR_SINGLE_OBS_DIM == 29, f"= {ACTOR_SINGLE_OBS_DIM}")
check("ENCODER_AUX_DIM == 4", ENCODER_AUX_DIM == 4, f"= {ENCODER_AUX_DIM}")
check("ACTOR_TOTAL_DIM == 29", ACTOR_TOTAL_DIM == 29, f"= {ACTOR_TOTAL_DIM}")
check("PRIVILEGED_OBS_DIM == 44", PRIVILEGED_OBS_DIM == 44, f"= {PRIVILEGED_OBS_DIM}")
check(
    "TOTAL_OBS_DIM == 77",
    TOTAL_OBS_DIM == 29 + 4 + 44,
    f"= {TOTAL_OBS_DIM}",
)
check("PRIV_TARGET_DIM == 32", PRIV_TARGET_DIM == sum(d for _, d in PRIV_TARGET_GROUPS), f"= {PRIV_TARGET_DIM}")
check("observation_space == TOTAL_OBS_DIM", env.observation_space.shape == (TOTAL_OBS_DIM,), f"= {env.observation_space.shape}")

print()
print("=" * 78)
print("B. reset frame: hover-trim prefill and aux channels")
print("=" * 78)
env.set_dr_level(0.0)
obs, info = env.reset(seed=7)
check("obs dims", obs.shape == (TOTAL_OBS_DIM,), f"= {obs.shape}")
check("env obs == info actor_obs + aux + priv", obs.shape[0] == 77)

prev = obs[13:17]
check(
    "prior action at reset == hover trim",
    np.allclose(prev, env.hover_trim_action, atol=1e-6),
    f"= {np.round(prev, 4)}",
)
print(f"        hover trim action          = {np.round(env.hover_trim_action, 4)}")

aux = obs[29:33]
check("aux accel ~ 9.81 on body +z at rest", abs(aux[2] - 9.81) < 0.5, f"= {np.round(aux[:3], 3)}")
check("aux accel xy ~ 0", np.abs(aux[:2]).max() < 0.2, f"= {np.round(aux[:2], 3)}")
check("aux v_batt_norm ~ 1.0 at no DR", abs(aux[3] - 1.0) < 0.05, f"= {aux[3]:.4f}")

check(
    "obs[:29] matches info['single_obs'] (no stacking)",
    np.allclose(obs[:29], info["single_obs"], atol=1e-6),
)
check(
    "obs[33:] matches info['privileged_obs']",
    np.allclose(obs[33:], info["privileged_obs"], atol=1e-6),
)

print()
print("=" * 78)
print("C. get_priv_targets() cross-validation vs the privileged vector")
print("=" * 78)
env.set_dr_level(1.0)
for seed in (11, 12, 13):
    obs, info = env.reset(seed=seed)
    for _ in range(25):  # let the plant move so vel / wrench / motor speeds are non-trivial
        obs, _, term, trunc, info = env.step(np.array([0.1, 0.3, -0.2, 0.05], dtype=np.float32))
        if term or trunc:
            env.reset(seed=seed + 100)
    tgt = env.get_priv_targets()
    priv = info["privileged_obs"]

    ok = tgt.shape == (PRIV_TARGET_DIM,)
    check(f"seed {seed}: target dims == 32", ok, f"= {tgt.shape}")

    # group slices
    mass_ratio = tgt[0]
    com = tgt[1:4]
    radial = tgt[4:8]
    eff = tgt[8:12]
    tau = tgt[12:14]
    thrust_scale = tgt[14]
    sag = tgt[15]
    gyro_bias = tgt[16:19]
    vel = tgt[19:22]
    omega = tgt[22:26]
    wrench = tgt[26:32]

    check(f"seed {seed}: mass_ratio == priv total_mass/base_mass",
          abs(mass_ratio - priv[30] / env.quad.base_mass) < 1e-5,
          f"{mass_ratio:.5f} vs {priv[30] / env.quad.base_mass:.5f}")
    check(f"seed {seed}: com_offset_b == priv[26:29]",
          np.allclose(com, priv[26:29], atol=1e-6),
          f"{np.round(com, 5)} vs {np.round(priv[26:29], 5)}")
    check(f"seed {seed}: motor_efficiency == priv[31:35]",
          np.allclose(eff, priv[31:35], atol=1e-5),
          f"{np.round(eff, 4)}")
    check(f"seed {seed}: motor_tau == priv[35:37]",
          np.allclose(tau, priv[35:37], atol=1e-6),
          f"{np.round(tau, 5)}")
    check(f"seed {seed}: thrust_scale == priv[38]",
          abs(thrust_scale - priv[38]) < 1e-5, f"{thrust_scale:.4f} vs {priv[38]:.4f}")
    check(f"seed {seed}: dynamic_sag_coef == priv[37]",
          abs(sag - priv[37]) < 1e-5, f"{sag:.5f} vs {priv[37]:.5f}")
    check(f"seed {seed}: gyro_bias_b == priv[39:42]",
          np.allclose(gyro_bias, priv[39:42], atol=1e-6), f"{np.round(gyro_bias, 4)}")
    check(f"seed {seed}: true_vel_w == quad.vel",
          np.allclose(vel, env.quad.vel, atol=1e-6), f"{np.round(vel, 4)}")
    check(f"seed {seed}: motor_speed_norm == wMotor/maxW",
          np.allclose(omega, env.quad.wMotor / env.max_w, atol=1e-6), f"{np.round(omega, 4)}")
    check(f"seed {seed}: rotor_radial_err is small and finite",
          np.all(np.isfinite(radial)) and np.abs(radial).max() < 5e-3,
          f"max |err| = {np.abs(radial).max() * 1000:.3f} mm")
    check(f"seed {seed}: aero_wrench_b finite", np.all(np.isfinite(wrench)),
          f"|F| = {np.linalg.norm(wrench[:3]):.5f} N")

print()
print("=" * 78)
print("D. DR headroom")
print("=" * 78)
env2 = make_env()
env2.set_dr_level(1.0)
env2.set_dr_headroom(1.5)
obs, info = env2.reset(seed=99)
dist = env2.active_disturbances
check("dr_level stays 1.0", env2.dr_level == 1.0, f"= {env2.dr_level}")
check("dr_eff == 1.5", abs(env2.dr_eff - 1.5) < 1e-9, f"= {env2.dr_eff}")
check("dr_eff recorded in active_disturbances", abs(dist["dr_eff"] - 1.5) < 1e-9)

# Statistical check that the headroom really widens the envelope past the nominal caps.
N = 400
pay, com, ts, eff_min = [], [], [], []
for i in range(N):
    _, info_i = env2.reset(seed=1000 + i)
    d = info_i["active_disturbances"]
    pay.append(d["payload_mass_g"])
    com.append(max(abs(d["com_offset"][0]), abs(d["com_offset"][1])))
    ts.append(d["thrust_scale"])
    eff_min.append(min(d["motor_efficiencies"]))
pay, com, ts, eff_min = map(np.asarray, (pay, com, ts, eff_min))
print(
    f"        over {N} resets: payload max {pay.max():.3f} g | "
    f"|com|xy max {com.max() * 1000:.3f} mm | thrust_scale [{ts.min():.3f}, {ts.max():.3f}] | "
    f"min motor eff {eff_min.min():.3f}"
)
check(
    "payload envelope extends past the nominal 4.5 g cap",
    pay.max() > 0.0045 * 1000,
    f"max = {pay.max():.3f} g (nominal cap 4.500 g)",
)
check(
    "|com|xy envelope extends past the nominal 2.5 mm cap",
    com.max() > 0.0025,
    f"max = {com.max() * 1000:.3f} mm (nominal cap 2.500 mm)",
)
check(
    "thrust_scale envelope extends past the nominal [0.85, 1.10] band",
    ts.min() < 0.85 or ts.max() > 1.10,
    f"[{ts.min():.3f}, {ts.max():.3f}]",
)
check(
    "motor efficiency goes below the nominal 0.92 floor",
    eff_min.min() < 0.92,
    f"min = {eff_min.min():.3f} (nominal floor 0.920)",
)
check(
    "headroom stays inside the 1.5x analytic bounds",
    pay.max() <= 6.75 + 1e-6            # grams  (4.5 g nominal cap * 1.5)
    and com.max() <= 0.00375 + 1e-9     # metres (2.5 mm nominal cap * 1.5)
    and ts.min() >= 0.775 - 1e-6        # 1 - 1.5 * 0.15
    and ts.max() <= 1.15 + 1e-6,        # 1 + 1.5 * 0.10
    f"payload {pay.max():.4f} g, |com| {com.max() * 1000:.4f} mm, "
    f"thrust_scale [{ts.min():.4f}, {ts.max():.4f}]",
)
check("targets finite under headroom", np.all(np.isfinite(tgt)))
check(
    "headroom widens the target envelope",
    abs(tgt[0] - 1.0) > 0.0 or np.abs(tgt[8:12] - 1.0).max() > 0.0,
    f"mass_ratio={tgt[0]:.4f}, min eff={tgt[8:12].min():.4f}",
)

print()
print("=" * 78)
print("F. observation latency is applied to the aux block too")
print("=" * 78)
env4 = QuadFlipEnv(
    episode_seconds=3.0,
    obs_noise=False,
    random_wind=False,
    random_initial_state=True,
    random_battery=False,
    curriculum_arena=False,
    arena_radius_start=6.0,
    arena_radius_end=6.0,
)
env4.set_dr_level(1.0)

# Find a reset that landed on the maximum injected latency (2 steps at full DR).
found = False
for s in range(60):
    obs, info = env4.reset(seed=500 + s)
    if env4.obs_latency == 2:
        found = True
        break
check("found an episode with obs_latency == 2", found, f"after {s + 1} resets")

if found:
    check("aux_buffer length == latency + 1", len(env4.aux_buffer) == 3, f"= {len(env4.aux_buffer)}")
    check(
        "obs aux block == buffered (delayed) aux",
        np.allclose(obs[29:33], env4.aux_buffer[0], atol=1e-6),
        f"{np.round(obs[29:33], 4)}",
    )
    check(
        "info['encoder_frame'] == obs[:33]",
        np.allclose(info["encoder_frame"], obs[:33], atol=1e-6),
    )

    # Excite the plant hard so the delayed and current specific force differ.
    for _ in range(6):
        obs, _, term, trunc, info = env4.step(np.array([0.6, 0.0, 0.9, 0.0], dtype=np.float32))
        if term or trunc:
            break
    current = env4._compute_encoder_aux()
    delayed = env4._aux_delayed
    check(
        "aux is genuinely delayed (current != delayed under excitation)",
        not np.allclose(current, delayed, atol=1e-4),
        f"current z={current[2]:.4f} delayed z={delayed[2]:.4f}",
    )
    check(
        "obs aux block still tracks the delayed value, not the current one",
        not np.allclose(obs[29:33], current, atol=1e-4),
    )

print()
print("=" * 78)
print("E. full episode integrity")
print("=" * 78)
env3 = make_env()
env3.set_dr_level(1.0)
env3.set_dr_headroom(1.5)
obs, info = env3.reset(seed=5)
steps, bad = 0, 0
while True:
    obs, rew, term, trunc, info = env3.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    steps += 1
    if not np.all(np.isfinite(obs)) or not np.isfinite(rew):
        bad += 1
    if not np.all(np.isfinite(env3.get_priv_targets())):
        bad += 1
    if term or trunc:
        break
check("episode ran", steps > 0, f"{steps} steps, reason={info['termination_reason']}")
check("no non-finite obs/reward/targets", bad == 0, f"{bad} bad steps")

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all Phase 1 checks passed")
print("=" * 78)
