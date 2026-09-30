#!/usr/bin/env python3
"""
Packaging script for 5th AI-Hub Artificial Intelligence Competition (2026) Submission.
Builds the complete submission archive: AI_Hub_2026_Submission_Quadcopter_Flip.zip
"""

import os
import zipfile
from pathlib import Path

def create_submission_zip():
    repo_root = Path(__file__).resolve().parent.parent
    sub_dir = repo_root / "submission_ai_hub_2026"
    zip_path = repo_root / "AI_Hub_2026_Submission_Quadcopter_Flip.zip"

    files_to_pack = [
        # Documents
        (sub_dir / "paper.pdf", "paper.pdf"),
        (sub_dir / "paper.tex", "paper.tex"),
        (sub_dir / "presentation.pdf", "presentation.pdf"),
        (sub_dir / "SCRIPT_8MIN.md", "SCRIPT_8MIN.md"),
        (sub_dir / "01_AI_Hub_Submission_Form.md", "01_AI_Hub_Submission_Form.md"),
        (sub_dir / "02_Technical_Project_Description.md", "02_Technical_Project_Description.md"),
        (sub_dir / "03_Evaluation_Criteria_Alignment_and_Media_Guide.md", "03_Evaluation_Criteria_Alignment_and_Media_Guide.md"),
        
        # Participant Docs
        (sub_dir / "participant_documents" / "README.md", "participant_documents/README.md"),

        # Figures
        (sub_dir / "figures" / "crazyflie_hardware.jpg", "figures/crazyflie_hardware.jpg"),
        (sub_dir / "figures" / "lighthouse_deck.jpg", "figures/lighthouse_deck.jpg"),
        (sub_dir / "figures" / "motors_16mm_vs_20mm.jpg", "figures/motors_16mm_vs_20mm.jpg"),
        (sub_dir / "figures" / "lipo_battery.jpg", "figures/lipo_battery.jpg"),
        (sub_dir / "figures" / "crazyradio_dongle.jpg", "figures/crazyradio_dongle.jpg"),
        (sub_dir / "figures" / "live_flight_telemetry.png", "figures/live_flight_telemetry.png"),
        (sub_dir / "figures" / "flight_trajectories_mjx.png", "figures/flight_trajectories_mjx.png"),
        (sub_dir / "figures" / "training_curves_mjx.png", "figures/training_curves_mjx.png"),
        (sub_dir / "figures" / "rate_tracking_plot.png", "figures/rate_tracking_plot.png"),
        (sub_dir / "figures" / "manual_flight_telemetry.png", "figures/manual_flight_telemetry.png"),
        (sub_dir / "figures" / "telemetry_plots.pdf", "figures/telemetry_plots.pdf"),

        # Key Code & Architecture Artifacts
        (repo_root / "Simulation" / "deploy" / "manifests" / "policy_export.json", "code_artifacts/policy_export.json"),
        (repo_root / "Simulation" / "deploy" / "app_policy_controller" / "src" / "controller_app.c", "code_artifacts/controller_app.c"),
        (repo_root / "Simulation" / "deploy" / "app_policy_controller" / "src" / "policy_guard.c", "code_artifacts/policy_guard.c"),
        (repo_root / "Simulation" / "deploy" / "app_policy_controller" / "src" / "policy_guard.h", "code_artifacts/policy_guard.h"),
        (repo_root / "Simulation" / "deploy" / "app_policy_controller" / "src" / "analytic_flip.c", "code_artifacts/analytic_flip.c"),
        (repo_root / "Simulation" / "deploy" / "app_policy_controller" / "src" / "analytic_flip.h", "code_artifacts/analytic_flip.h"),
        (repo_root / "Simulation" / "quad_flip_env.py", "code_artifacts/quad_flip_env.py"),
        (repo_root / "logs" / "eval_metrics_mjx.csv", "evaluation_metrics/eval_metrics_mjx.csv"),
        (repo_root / "logs" / "eval_robust_mjx.csv", "evaluation_metrics/eval_robust_mjx.csv"),
    ]

    # Also check if any PDF or documents are in participant_documents
    part_dir = sub_dir / "participant_documents"
    for item in part_dir.iterdir():
        if item.is_file() and item.name != "README.md":
            files_to_pack.append((item, f"participant_documents/{item.name}"))

    print(f"Creating submission zip archive at: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arc_name in files_to_pack:
            if file_path.exists():
                print(f"  Adding: {arc_name}")
                zf.write(file_path, arc_name)
            else:
                print(f"  Warning: File not found: {file_path}")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nSUCCESS: Archive created ({size_mb:.2f} MB). Ready for AI-Hub eClass upload!")

if __name__ == "__main__":
    create_submission_zip()
