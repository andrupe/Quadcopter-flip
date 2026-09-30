#!/usr/bin/env bash
# ==============================================================================
# Parametric Aerobatics Flip Experiments Runner
#
# Usage:
#   ./experiments.sh <num>     # Run experiment 1 to 12
#   ./experiments.sh check     # Analyze the latest flight log
#   ./experiments.sh list      # List all experiment commands
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON="${ROOT_DIR}/.venv/bin/python"
SCRIPT="${ROOT_DIR}/Simulation/deploy/radio_flight.py"
BASE_CMD="${PYTHON} ${SCRIPT} --arm-ok --hover 50 --set policy.shadow=0"

case "${1:-list}" in
    1)
        echo ">>> Running Exp 1: Pure Hover & Control Authority Baseline"
        ${BASE_CMD}
        ;;
    2)
        echo ">>> Running Exp 2: Conservative Pitch Flip (750 deg/s, 85% pop)"
        ${BASE_CMD} --flip-dps 750 --flip-pop 0.85
        ;;
    3)
        echo ">>> Running Exp 3: Calibrated Flip (850 deg/s, 90% pop)"
        ${BASE_CMD} --flip-dps 850 --flip-pop 0.90
        ;;
    4)
        echo ">>> Running Exp 4: High-Agility Snappy Flip (950 deg/s, 92% pop) [RECOMMENDED]"
        ${BASE_CMD} --flip-dps 950 --flip-pop 0.92
        ;;
    5)
        echo ">>> Running Exp 5: Extreme Fast Rotation (1050 deg/s, 92% pop)"
        ${BASE_CMD} --flip-dps 1050 --flip-pop 0.92
        ;;
    6)
        echo ">>> Running Exp 6: Soft Pop for Low Ceilings (950 deg/s, 82% pop)"
        ${BASE_CMD} --flip-dps 950 --flip-pop 0.82
        ;;
    7)
        echo ">>> Running Exp 7: Strong Pop / Maximum Punch (950 deg/s, 95% pop)"
        ${BASE_CMD} --flip-dps 950 --flip-pop 0.95
        ;;
    8)
        echo ">>> Running Exp 8: Soft Motor Ramp (950 deg/s, 92% pop, 35% ramp fraction)"
        ${BASE_CMD} --flip-dps 950 --flip-pop 0.92 --flip-rate-frac 0.35
        ;;
    9)
        echo ">>> Running Exp 9: Sharp Box Ramp (950 deg/s, 92% pop, 22% ramp fraction)"
        ${BASE_CMD} --flip-dps 950 --flip-pop 0.92 --flip-rate-frac 0.22
        ;;
    10)
        echo ">>> Running Exp 10: Standard Roll Flip (900 deg/s, 90% pop)"
        ${BASE_CMD} --flip-axis roll --flip-dps 900 --flip-pop 0.90
        ;;
    11)
        echo ">>> Running Exp 11: High-Speed Roll Flip (1000 deg/s, 92% pop)"
        ${BASE_CMD} --flip-axis roll --flip-dps 1000 --flip-pop 0.92
        ;;
    12)
        echo ">>> Running Exp 12: Prebaked Table Playback"
        ${BASE_CMD} --flip-table
        ;;
    check|eval|log)
        echo ">>> Analyzing latest flight log in logs/radio_flight_log.csv..."
        ${PYTHON} -c "
import csv
import os

path = 'logs/radio_flight_log.csv'
if not os.path.exists(path):
    print('No flight log found at', path)
    exit(0)

with open(path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    print('Log file is empty.')
    exit(0)

phase_col = 'fw_policy_flip_phase' if 'fw_policy_flip_phase' in rows[0] else None
flip_rows = [r for r in rows if phase_col and float(r.get(phase_col, 0) or 0) > 0]

if flip_rows:
    gyro_col = 'fw_traj_flip_gyro_deg' if 'fw_traj_flip_gyro_deg' in rows[0] else None
    z_col = 'fw_stateEstimate.z' if 'fw_stateEstimate.z' in rows[0] else ('stateEstimate.z' if 'stateEstimate.z' in rows[0] else None)
    shift_col = 'fw_traj_flip_z_shift' if 'fw_traj_flip_z_shift' in rows[0] else None

    print('================ FLIP TELEMETRY SUMMARY ================')
    if gyro_col:
        max_gyro = max(float(r[gyro_col]) for r in flip_rows if r[gyro_col])
        print(f'Max Gyro Angle : {max_gyro:.1f} deg  (Target: ~360 deg)')
    if z_col:
        z_vals = [float(r[z_col]) for r in flip_rows if r[z_col]]
        if z_vals:
            print(f'Peak Altitude  : {max(z_vals):.2f} m')
            print(f'Min Altitude   : {min(z_vals):.2f} m')
            print(f'Entry Altitude : {z_vals[0]:.2f} m')
            print(f'Exit Altitude  : {z_vals[-1]:.2f} m')
    if shift_col:
        shift_vals = [float(r[shift_col]) for r in flip_rows if r[shift_col]]
        if shift_vals:
            print(f'Catch Z Shift  : {shift_vals[-1]:.2f} m')
    print('========================================================')
else:
    print('Flight log found, but no active flip phase was detected.')
"
        ;;
    list|*)
        echo "======================================================================="
        echo "  PARAMETRIC AEROBATICS FLIP EXPERIMENTS"
        echo "======================================================================="
        echo "Usage: ./experiments.sh <number 1-12>   or   ./experiments.sh check"
        echo ""
        echo "  1) Pure Hover & Authority Baseline"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0"
        echo ""
        echo "  2) Conservative Pitch Flip (750 deg/s, 85% pop)"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-dps 750 --flip-pop 0.85"
        echo ""
        echo "  3) Calibrated Pitch Flip (850 deg/s, 90% pop)"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-dps 850 --flip-pop 0.90"
        echo ""
        echo "  4) High-Agility Snappy Flip (950 deg/s, 92% pop) [RECOMMENDED]"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-dps 950 --flip-pop 0.92"
        echo ""
        echo "  5) Extreme Fast Rotation (1050 deg/s, 92% pop)"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-dps 1050 --flip-pop 0.92"
        echo ""
        echo "  6) Soft Pop for Low Ceilings (950 deg/s, 82% pop)"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-dps 950 --flip-pop 0.82"
        echo ""
        echo "  7) Strong Pop / Maximum Punch (950 deg/s, 95% pop)"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-dps 950 --flip-pop 0.95"
        echo ""
        echo "  8) Soft Motor Ramp (950 deg/s, 92% pop, 35% ramp fraction)"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-dps 950 --flip-pop 0.92 --flip-rate-frac 0.35"
        echo ""
        echo "  9) Sharp Box Ramp (950 deg/s, 92% pop, 22% ramp fraction)"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-dps 950 --flip-pop 0.92 --flip-rate-frac 0.22"
        echo ""
        echo " 10) Standard Roll Flip (900 deg/s, 90% pop)"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-axis roll --flip-dps 900 --flip-pop 0.90"
        echo ""
        echo " 11) High-Speed Roll Flip (1000 deg/s, 92% pop)"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-axis roll --flip-dps 1000 --flip-pop 0.92"
        echo ""
        echo " 12) Prebaked Table Playback"
        echo "     .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0 --flip-table"
        echo "======================================================================="
        ;;
esac
