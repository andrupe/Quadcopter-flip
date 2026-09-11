#!/usr/bin/env python3
"""Quadcopter Flip & Stabilization Evaluation Runner (Option C + Hierarchical Rate PID).

ONE-CLICK EVALUATION WORKFLOW:
  * In VS Code: Open this file and click the "Run Python File" (Play button ▶) in the top-right.
  * In Terminal: Run `.venv/bin/python evaluate.py`
  * Flags:
      --viewer / --no-viewer   : Toggle real-time 3D MuJoCo interactive window
      --plot / --no-plot       : Toggle flight telemetry plots & PDF export
      --dr 0.0..1.0            : Domain Randomization intensity (0.0=nominal, 1.0=full stress)
      --loop                   : Keep looping flights indefinitely in 3D viewer
      --speed 1.0              : Playback speed (0.5 = 2x slow-mo, 1.0 = real-time)
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure repository root and Simulation directories are on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_ROOT, "Simulation")
for _p in [_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.evaluate import evaluate, MODEL_NAME, EPISODE_SECONDS, DR_LEVEL, HOVER_GAIN

# ======================================================================================
# EVALUATION CONFIGURATION (Edit options below, then click Run ▶ in VS Code)
# ======================================================================================
DEFAULT_MODEL: str = "latest"                         # "latest" auto-selects newest checkpoint in logs/
SHOW_VIEWER: bool = True              # Launch real-time interactive 3D MuJoCo viewer window
SHOW_PLOTS: bool = True               # Display 2D telemetry matplotlib plots and export PDF
EVAL_DR_LEVEL: float = 0.0            # 0.0 = nominal clean sim, 1.0 = full sim-to-real stress
FLIGHT_SECONDS: float = 8.0           # Episode flight time (seconds)
NUM_EPISODES: int = 1                 # Episodes to run before plotting
LOOP_VIEWER: bool = False             # True: continuous looping in 3D viewer
PLAYBACK_SPEED: float = 1.0           # 1.0 = real-time, 0.5 = 2x slow-motion
# ======================================================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quadcopter Flip Policy in MuJoCo")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model checkpoint name/path")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=SHOW_VIEWER, help="Interactive 3D viewer")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=SHOW_PLOTS, help="Show telemetry plots")
    parser.add_argument("--dr", type=float, default=EVAL_DR_LEVEL, help="Domain Randomization level [0.0, 1.0]")
    parser.add_argument("--duration", type=float, default=FLIGHT_SECONDS, help="Episode duration in seconds")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of evaluation episodes")
    parser.add_argument("--loop", action="store_true", default=LOOP_VIEWER, help="Loop continuously in viewer")
    parser.add_argument("--speed", type=float, default=PLAYBACK_SPEED, help="Playback speed scale")
    args = parser.parse_args()

    # macOS GUI trampoline: interactive viewer requires mjpython on macOS
    if sys.platform == "darwin" and args.viewer:
        try:
            import mujoco.viewer
            is_mjpython = hasattr(mujoco.viewer, "_MJPYTHON") and mujoco.viewer._MJPYTHON is not None
        except Exception:
            is_mjpython = False

        if not is_mjpython and os.environ.get("_MJP_TRAMPOLINED") != "1":
            import shutil
            mjpython_path = os.path.join(os.path.dirname(sys.executable), "mjpython")
            if not os.path.isfile(mjpython_path):
                mjpython_path = shutil.which("mjpython")
            if mjpython_path and os.path.isfile(mjpython_path):
                os.environ["_MJP_TRAMPOLINED"] = "1"
                os.execv(mjpython_path, [mjpython_path] + sys.argv)

    # Fallback to latest checkpoint in logs/ if model does not exist yet
    model_to_use = args.model
    if not os.path.isfile(model_to_use) and not os.path.isfile(f"{model_to_use}.zip") and not os.path.isfile(os.path.join(_ROOT, f"{model_to_use}.zip")):
        # Find latest zip in logs/
        logs_dir = os.path.join(_ROOT, "logs")
        if os.path.isdir(logs_dir):
            zips = [f for f in os.listdir(logs_dir) if f.endswith(".zip")]
            if zips:
                # pick one with largest step count or most recently modified
                zips.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
                model_to_use = os.path.join(logs_dir, zips[0])

    evaluate(
        model_name=model_to_use,
        episode_seconds=args.duration,
        num_episodes=args.episodes,
        dr_level=args.dr,
        hover_gain=HOVER_GAIN,
        show_viewer=args.viewer,
        show_plots=args.plot,
        loop=args.loop,
        eval_actor_only=True,
        random_initial_pos=False,
        random_initial_vel=True,
        random_initial_att=True,
        playback_speed=args.speed,
    )


if __name__ == "__main__":
    main()
