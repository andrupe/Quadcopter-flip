#!/usr/bin/env python3
import os
import sys
import time
import csv
import json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_SIM_DIR = os.path.join(_ROOT, "Simulation")
for _p in [_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import quad_flip_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from actor_input import ActorInput, load_checkpoint, read_checkpoint_arch

# Nominal checkpoint labels. If none of these exist the benchmark falls back to whatever
# rl_model_*_steps.zip files are actually in logs/ (see _discover_checkpoints).
CHECKPOINTS = [
    ("15.5M", os.path.join(_ROOT, "logs", "rl_model_15500000_steps")),
    ("16.0M", os.path.join(_ROOT, "logs", "rl_model_16000000_steps")),
    ("16.5M", os.path.join(_ROOT, "logs", "rl_model_16500000_steps")),
    ("17.0M", os.path.join(_ROOT, "logs", "rl_model_17000000_steps")),
    ("17.5M", os.path.join(_ROOT, "logs", "rl_model_17500000_steps")),
    ("18.0M", os.path.join(_ROOT, "logs", "rl_model_18000000_steps")),
    ("18.5M", os.path.join(_ROOT, "logs", "rl_model_18500000_steps")),
    ("19.0M", os.path.join(_ROOT, "logs", "rl_model_19000000_steps")),
    ("19.5M", os.path.join(_ROOT, "logs", "rl_model_19500000_steps")),
    ("20.0M", os.path.join(_ROOT, "logs", "rl_model_20000000_steps")),
]


def _discover_checkpoints():
    """Configured list when present, otherwise the newest checkpoints in logs/ (max 10)."""
    configured = [(label, base) for label, base in CHECKPOINTS if os.path.isfile(f"{base}.zip")]
    if configured:
        return configured
    from glob import glob
    import re

    def _step(path: str) -> int:
        m = re.search(r"rl_model_(\d+)_steps", os.path.basename(path))
        return int(m.group(1)) if m else 0

    zips = sorted(glob(os.path.join(_ROOT, "logs", "rl_model_*_steps.zip")), key=_step)
    return [(f"{_step(p) / 1e6:.1f}M", p[:-4]) for p in zips[-10:]]

DR_LEVELS = [0.65, 0.80, 1.00]
EPISODES_PER_DR = 25
EPISODE_SECONDS = 8.0
MAX_STEPS = int(EPISODE_SECONDS / 0.01)  # 800 steps

def run_benchmark():
    checkpoints = _discover_checkpoints()
    if not checkpoints:
        print("No checkpoints found (nothing matches the configured list and no "
              "logs/rl_model_*_steps.zip exists). Nothing to benchmark.")
        return
    print("=" * 80)
    print(f"LARGE SCALE CHECKPOINT BENCHMARK ({len(checkpoints)} checkpoints x {len(DR_LEVELS)} DR levels x {EPISODES_PER_DR} episodes = {len(checkpoints)*len(DR_LEVELS)*EPISODES_PER_DR} total flights)")
    print(f"DR Levels        : {DR_LEVELS}")
    print(f"Episodes / Level : {EPISODES_PER_DR} (identical paired seeds per checkpoint)")
    print(f"Flight Duration  : {EPISODE_SECONDS}s ({MAX_STEPS} steps)")
    print("=" * 80)

    env = quad_flip_env.QuadFlipEnv(
        episode_seconds=EPISODE_SECONDS,
        random_initial_state=True,
    )
    dummy = DummyVecEnv([lambda: env])

    all_episode_records = []
    summary_rows = []

    start_bench_time = time.time()

    for ckpt_label, ckpt_base in checkpoints:
        zip_path = f"{ckpt_base}.zip"
        norm_path = f"{ckpt_base}_vecnormalize.pkl"

        if not os.path.isfile(zip_path):
            print(f"Skipping {ckpt_label}: file not found ({zip_path})")
            continue

        try:
            actor_dim, _ = read_checkpoint_arch(zip_path)
            actor_input = ActorInput(actor_dim) if actor_dim is not None else ActorInput(29)
            model = load_checkpoint(zip_path, device="cpu")
        except (ValueError, RuntimeError) as exc:
            print(f"Skipping {ckpt_label}: {exc}")
            continue

        if os.path.isfile(norm_path):
            vec_norm = VecNormalize.load(norm_path, dummy)
            vec_norm.training = False
            if getattr(vec_norm, "norm_obs", False):
                print(f"Note: {os.path.basename(norm_path)} has norm_obs=True; feeding "
                      f"normalised observations (shapes match only if it came from this build).")
        else:
            vec_norm = None

        print(f"\nEvaluating Checkpoint: {ckpt_label} ({os.path.basename(zip_path)})")
        print(f"  Actor input: {actor_input.describe()}")

        for dr in DR_LEVELS:
            env.set_dr_level(dr)
            
            flips = 0
            survivals = 0
            returns = []
            flight_times = []
            min_altitudes = []
            post_flip_xy_drifts = []
            post_flip_z_errors = []
            final_xy_drifts = []
            final_z_errors = []
            term_reasons = {}

            t_dr_start = time.time()

            for ep_idx in range(EPISODES_PER_DR):
                seed = 1000 + ep_idx
                obs, info = env.reset(seed=seed)
                actor_input.reset()

                ep_reward = 0.0
                min_z = float(env.quad.pos[2])
                xy_errs_post_flip = []
                z_errs_post_flip = []

                for step in range(MAX_STEPS):
                    action, _ = model.predict(actor_input.prepare(obs, vec_norm), deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_reward += float(reward)

                    current_z = float(env.quad.pos[2])
                    min_z = min(min_z, current_z)

                    if env.flip_completed:
                        xy_err = float(np.linalg.norm(env.quad.pos[:2] - env.target_state[:2]))
                        z_err = float(abs(env.quad.pos[2] - env.target_state[2]))
                        xy_errs_post_flip.append(xy_err)
                        z_errs_post_flip.append(z_err)

                    if terminated or truncated:
                        break

                flip_done = bool(env.flip_completed)
                survived = not terminated
                term_reason = info.get("termination_reason", "none") if terminated else "completed"
                term_reasons[term_reason] = term_reasons.get(term_reason, 0) + 1

                flips += int(flip_done)
                survivals += int(survived)
                returns.append(ep_reward)
                flight_times.append(float(info.get("t", (step + 1) * env.dt)))
                min_altitudes.append(min_z)

                mean_post_xy = float(np.mean(xy_errs_post_flip)) if xy_errs_post_flip else float(np.linalg.norm(env.quad.pos[:2]))
                mean_post_z = float(np.mean(z_errs_post_flip)) if z_errs_post_flip else float(abs(env.quad.pos[2] - 1.2))
                post_flip_xy_drifts.append(mean_post_xy)
                post_flip_z_errors.append(mean_post_z)

                final_xy = float(np.linalg.norm(env.quad.pos[:2] - env.target_state[:2]))
                final_z = float(abs(env.quad.pos[2] - env.target_state[2]))
                final_xy_drifts.append(final_xy)
                final_z_errors.append(final_z)

                all_episode_records.append({
                    "checkpoint": ckpt_label,
                    "dr": dr,
                    "episode": ep_idx,
                    "seed": seed,
                    "flip": flip_done,
                    "survived": survived,
                    "reward": ep_reward,
                    "flight_time_s": flight_times[-1],
                    "steps": step + 1,
                    "min_z_m": min_z,
                    "post_xy_drift_m": mean_post_xy,
                    "post_z_err_m": mean_post_z,
                    "final_xy_drift_m": final_xy,
                    "final_z_err_m": final_z,
                    "termination_reason": term_reason,
                })

            dr_duration = time.time() - t_dr_start
            flip_rate = (flips / EPISODES_PER_DR) * 100.0
            surv_rate = (survivals / EPISODES_PER_DR) * 100.0
            mean_ret = float(np.mean(returns))
            mean_time = float(np.mean(flight_times))
            mean_xy = float(np.mean(post_flip_xy_drifts))
            mean_z_err = float(np.mean(post_flip_z_errors))
            mean_min_z = float(np.mean(min_altitudes))

            print(f"  DR={dr:0.2f} [{dr_duration:4.1f}s]: Flip={flip_rate:5.1f}% | Surv={surv_rate:5.1f}% | Time={mean_time:4.2f}s | Ret={mean_ret:7.1f} | MinAlt={mean_min_z:0.2f}m | XY_drift={mean_xy:0.2f}m | Z_err={mean_z_err:0.2f}m | Terms={term_reasons}")

            summary_rows.append({
                "checkpoint": ckpt_label,
                "dr": dr,
                "flip_rate_pct": flip_rate,
                "survival_rate_pct": surv_rate,
                "mean_return": mean_ret,
                "mean_flight_time_s": mean_time,
                "mean_min_alt_m": mean_min_z,
                "mean_post_xy_drift_m": mean_xy,
                "mean_post_z_err_m": mean_z_err,
                "ground_crash_count": term_reasons.get("ground_crash", 0),
                "out_of_volume_count": term_reasons.get("out_of_volume", 0),
            })

    total_time = time.time() - start_bench_time
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE in {total_time:.1f}s ({total_time/60.0:.2f} min)")
    print("=" * 80)

    if not all_episode_records or not summary_rows:
        print("No episodes were evaluated; nothing to write.")
        return

    # Save detailed CSVs
    ep_csv_path = os.path.join(_ROOT, "logs", "benchmark_episodes.csv")
    with open(ep_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_episode_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_episode_records)

    sum_csv_path = os.path.join(_ROOT, "logs", "benchmark_summary.csv")
    with open(sum_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print Summary Tables
    ckpt_names = [c[0] for c in checkpoints]

    def print_table(title, metric_key, fmt):
        print("\n" + "=" * 80)
        print(title)
        print(f"{'Checkpoint':<12} | {'DR=0.65':<10} | {'DR=0.80':<10} | {'DR=1.00':<10} | {'Mean':<10}")
        print("-" * 62)
        for ckpt in ckpt_names:
            vals = [r[metric_key] for r in summary_rows if r["checkpoint"] == ckpt]
            if len(vals) == 3:
                m_val = np.mean(vals)
                print(f"{ckpt:<12} | {fmt(vals[0]):<10} | {fmt(vals[1]):<10} | {fmt(vals[2]):<10} | {fmt(m_val):<10}")

    print_table("SURVIVAL RATE (%) BY CHECKPOINT & DR LEVEL:", "survival_rate_pct", lambda v: f"{v:5.1f}%")
    print_table("MEAN EPISODE RETURN BY CHECKPOINT & DR LEVEL:", "mean_return", lambda v: f"{v:7.1f}")
    print_table("MEAN FLIGHT TIME (SECONDS) BY CHECKPOINT & DR LEVEL (Max 8.0s):", "mean_flight_time_s", lambda v: f"{v:5.2f}s")
    print_table("MEAN POST-FLIP XY DRIFT (METERS) BY CHECKPOINT & DR LEVEL:", "mean_post_xy_drift_m", lambda v: f"{v:5.2f}m")
    print_table("MEAN POST-FLIP ALTITUDE ERROR (METERS) BY CHECKPOINT & DR LEVEL:", "mean_post_z_err_m", lambda v: f"{v:5.2f}m")

    # Overall Ranking
    print("\n" + "=" * 80)
    print("OVERALL CHECKPOINT RANKING (Sorted by Mean Survival Rate, then DR=1.00 Return):")
    print(f"{'Rank':<5} | {'Checkpoint':<12} | {'Surv@0.65':<10} | {'Surv@0.80':<10} | {'Surv@1.00':<10} | {'Mean Surv':<10} | {'Ret@1.00':<10} | {'Mean Ret':<10}")
    print("-" * 90)

    overall_scores = []
    for ckpt in ckpt_names:
        row_065 = next((r for r in summary_rows if r["checkpoint"] == ckpt and r["dr"] == 0.65), None)
        row_080 = next((r for r in summary_rows if r["checkpoint"] == ckpt and r["dr"] == 0.80), None)
        row_100 = next((r for r in summary_rows if r["checkpoint"] == ckpt and r["dr"] == 1.00), None)

        if row_065 and row_080 and row_100:
            surv_mean = (row_065["survival_rate_pct"] + row_080["survival_rate_pct"] + row_100["survival_rate_pct"]) / 3.0
            ret_mean = (row_065["mean_return"] + row_080["mean_return"] + row_100["mean_return"]) / 3.0
            overall_scores.append({
                "ckpt": ckpt,
                "s065": row_065["survival_rate_pct"],
                "s080": row_080["survival_rate_pct"],
                "s100": row_100["survival_rate_pct"],
                "s_mean": surv_mean,
                "r100": row_100["mean_return"],
                "r_mean": ret_mean,
            })

    overall_scores.sort(key=lambda x: (x["s_mean"], x["s100"], x["r100"]), reverse=True)
    for idx, sc in enumerate(overall_scores, 1):
        print(f"{idx:<5} | {sc['ckpt']:<12} | {sc['s065']:5.1f}%    | {sc['s080']:5.1f}%    | {sc['s100']:5.1f}%    | {sc['s_mean']:5.1f}%    | {sc['r100']:7.1f}   | {sc['r_mean']:7.1f}")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_benchmark()
