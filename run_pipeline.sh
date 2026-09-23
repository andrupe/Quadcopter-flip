#!/usr/bin/env bash
# ==============================================================================
# Master Pipeline Runner: Full End-to-End Quadcopter Training
#
# Steps executed in order:
#   1. Collect 5M-frame pretraining corpus (Simulation/encoder/collect_data.py)
#   2. Train 5-step self-supervised history encoder (Simulation/encoder/train_encoder.py)
#   3. Train PPO tracking policy with adaptive envelope (Simulation/train.py)
#
# Usage:
#   ./run_pipeline.sh                     # Full production run
#   ./run_pipeline.sh --smoke             # Quick smoke validation run
#   ./run_pipeline.sh --skip-collect      # Skip step 1 if corpus already collected
#   ./run_pipeline.sh --skip-encoder      # Skip step 2 if encoder already trained
#   ./run_pipeline.sh --skip-ppo          # Run steps 1 & 2 only
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON="${ROOT_DIR}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    PYTHON="$(command -v python3)"
fi

# Pipeline Defaults
FRAMES=${FRAMES:-5000000}
WORKERS=${WORKERS:-8}
ENCODER_EPOCHS=${ENCODER_EPOCHS:-100}
ENCODER_HORIZON=${ENCODER_HORIZON:-5}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-30000000}

SKIP_COLLECT=false
SKIP_ENCODER=false
SKIP_PPO=false
SKIP_FIRMWARE=false
RESUME=false
FIRMWARE_BACKEND="legacy"
FLASH_AFTER=false
FLASH_COLD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-collect)
            SKIP_COLLECT=true
            shift
            ;;
        --skip-encoder)
            SKIP_ENCODER=true
            shift
            ;;
        --skip-ppo)
            SKIP_PPO=true
            shift
            ;;
        --skip-firmware)
            SKIP_FIRMWARE=true
            shift
            ;;
        --firmware-only)
            SKIP_COLLECT=true
            SKIP_ENCODER=true
            SKIP_PPO=true
            SKIP_FIRMWARE=false
            shift
            ;;
        --flash)
            FLASH_AFTER=true
            shift
            ;;
        --flash-cold)
            FLASH_AFTER=true
            FLASH_COLD=true
            shift
            ;;
        --stedgeai)
            FIRMWARE_BACKEND="stedgeai"
            shift
            ;;
        --legacy)
            FIRMWARE_BACKEND="legacy"
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --smoke)
            FRAMES=20000
            WORKERS=2
            ENCODER_EPOCHS=2
            TOTAL_TIMESTEPS=40960
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoke           Run a minimal end-to-end smoke test"
            echo "  --skip-collect    Skip data collection (Stage 1)"
            echo "  --skip-encoder    Skip encoder training (Stage 2)"
            echo "  --skip-ppo        Skip PPO policy training (Stage 3)"
            echo "  --skip-firmware   Skip firmware generation & compilation (Stage 4)"
            echo "  --firmware-only   Only export policy and compile firmware (Stage 4)"
            echo "  --flash           Flash firmware to drone via radio at pipeline end"
            echo "  --flash-cold      Cold-boot flash firmware to drone at pipeline end"
            echo "  --stedgeai        Use ST Edge AI backend instead of legacy C"
            echo "  --legacy          Use legacy C net backend (default)"
            echo "  --resume          Resume PPO training from previous checkpoint"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

PIPELINE_START=$(date +%s)

echo "======================================================================="
echo "  QUADCOPTER FLIGHT PIPELINE RUNNER"
echo "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python    : ${PYTHON}"
echo "  Directory : ${ROOT_DIR}"
echo "  Config    : frames=${FRAMES} | workers=${WORKERS} | encoder_epochs=${ENCODER_EPOCHS} | ppo_steps=${TOTAL_TIMESTEPS}"
echo "======================================================================="

mkdir -p logs/encoder_data

# ------------------------------------------------------------------------------
# STAGE 1: Data Collection
# ------------------------------------------------------------------------------
if [ "${SKIP_COLLECT}" = false ]; then
    echo ""
    echo "======================================================================="
    echo "  [STAGE 1/3] Collecting Encoder Pretraining Corpus (${FRAMES} frames)"
    echo "  Includes: 33g–38g mass, ground-to-hover takeoffs, acrobatic pulses,"
    echo "            asymmetric motor faults, and dynamic recovery kicks."
    echo "======================================================================="
    T_START=$(date +%s)

    # If old shards exist, move them to legacy backup so new collection is clean
    if compgen -G "logs/encoder_data/*.npz" > /dev/null; then
        echo "  Archiving existing shards to logs/encoder_data_legacy/ ..."
        mkdir -p logs/encoder_data_legacy
        mv logs/encoder_data/*.npz logs/encoder_data_legacy/ 2>/dev/null || true
    fi

    "${PYTHON}" Simulation/encoder/collect_data.py \
        --frames "${FRAMES}" \
        --out logs/encoder_data \
        --workers "${WORKERS}"

    T_END=$(date +%s)
    echo ">>> Stage 1 completed in $((T_END - T_START))s."
else
    echo ">>> Skipping Stage 1 (Data Collection)."
fi

# ------------------------------------------------------------------------------
# STAGE 2: Train Self-Supervised History Encoder
# ------------------------------------------------------------------------------
if [ "${SKIP_ENCODER}" = false ]; then
    echo ""
    echo "======================================================================="
    echo "  [STAGE 2/3] Training 5-Step Self-Supervised Forward Dynamics Encoder"
    echo "  Target    : logs/encoder_gru.pt (horizon=${ENCODER_HORIZON}, epochs=${ENCODER_EPOCHS})"
    echo "======================================================================="
    T_START=$(date +%s)

    "${PYTHON}" Simulation/encoder/train_encoder.py \
        --mode self_supervised \
        --horizon "${ENCODER_HORIZON}" \
        --epochs "${ENCODER_EPOCHS}" \
        --out logs/encoder_gru.pt

    T_END=$(date +%s)
    echo ">>> Stage 2 completed in $((T_END - T_START))s."
else
    echo ">>> Skipping Stage 2 (Encoder Training)."
fi

# ------------------------------------------------------------------------------
# STAGE 3: Train PPO Tracking Policy
# ------------------------------------------------------------------------------
if [ "${SKIP_PPO}" = false ]; then
    echo ""
    echo "======================================================================="
    echo "  [STAGE 3/4] Training PPO Policy (${TOTAL_TIMESTEPS} steps)"
    echo "  Features  : LatentObsWrapper (48 actor dims), takeoff capability,"
    echo "              unconstrained early exploration, and relative envelope curriculum."
    echo "======================================================================="
    T_START=$(date +%s)
    if [ "${RESUME}" = true ]; then
        "${PYTHON}" Simulation/train.py \
            --total-timesteps "${TOTAL_TIMESTEPS}" \
            --num-workers "${WORKERS}" \
            --resume
    else
        "${PYTHON}" Simulation/train.py \
            --total-timesteps "${TOTAL_TIMESTEPS}" \
            --num-workers "${WORKERS}"
    fi

    T_END=$(date +%s)
    echo ">>> Stage 3 completed in $((T_END - T_START))s."
else
    echo ">>> Skipping Stage 3 (PPO Training)."
fi

# ------------------------------------------------------------------------------
# STAGE 4: Export Policy & Build Crazyflie Firmware
# ------------------------------------------------------------------------------
if [ "${SKIP_FIRMWARE}" = false ]; then
    echo ""
    echo "======================================================================="
    echo "  [STAGE 4/4] Exporting Policy & Compiling Crazyflie Firmware"
    echo "  Backend   : ${FIRMWARE_BACKEND}"
    echo "======================================================================="
    T_START=$(date +%s)

    # 4a. Check / Bake reference trajectory tables
    echo "  [4a] Verifying reference trajectory tables..."
    "${PYTHON}" Simulation/deploy/gen_references.py

    # 4b. Export policy and encoder weights to C arrays
    echo "  [4b] Exporting policy (PPO actor + GRU history encoder) to C..."
    MODEL_PATH="quad_flip_model.zip"
    if [ ! -f "${MODEL_PATH}" ]; then
        # Check if checkpoint exists in logs/
        LATEST_LOG_MODEL=$(ls -t logs/rl_model_*_steps.zip 2>/dev/null | head -n1 || true)
        if [ -n "${LATEST_LOG_MODEL}" ]; then
            MODEL_PATH="${LATEST_LOG_MODEL}"
        fi
    fi

    if [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: No trained policy model found (${MODEL_PATH}). Cannot export firmware." >&2
        exit 1
    fi

    "${PYTHON}" Simulation/deploy/export_policy.py --model "${MODEL_PATH}"

    # 4c. Compile out-of-tree policy controller firmware
    echo "  [4c] Compiling Crazyflie firmware binary (${FIRMWARE_BACKEND})..."
    if [ "${FIRMWARE_BACKEND}" = "stedgeai" ]; then
        ./Simulation/deploy/build_app.sh --stedgeai
    else
        ./Simulation/deploy/build_app.sh --legacy
    fi

    T_END=$(date +%s)
    echo ">>> Stage 4 completed in $((T_END - T_START))s."
else
    echo ">>> Skipping Stage 4 (Firmware Build)."
fi

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))

echo ""
echo "======================================================================="
echo "  PIPELINE COMPLETED SUCCESSFULLY"
echo "  Total elapsed time: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo "  Saved Encoder     : logs/encoder_gru.pt"
echo "  Saved Policy      : quad_flip_model.zip"
echo "  Firmware Binary   : Simulation/deploy/app_policy_controller/build/cf2.bin"
echo "======================================================================="
echo ""
echo "-----------------------------------------------------------------------"
echo "  DEPLOYMENT & FLASHING INSTRUCTIONS"
echo "-----------------------------------------------------------------------"
echo "1. Flash the firmware onto the Crazyflie via radio:"
echo "   ./flash.sh"
echo "   (or cold boot recovery if unresponsive: ./flash.sh --cold)"
echo "   (or directly: .venv-client/bin/cfloader flash Simulation/deploy/app_policy_controller/build/cf2.bin stm32-fw -w radio://0/80/2M/E7E7E7E7E7)"
echo ""
echo "2. Verify Lighthouse Positioning & Geometry (CRITICAL):"
echo "   .venv/bin/python Simulation/deploy/lighthouse_check.py"
echo "   * status 2 = geometry valid & base-station pulses feeding Kalman estimator"
echo "   * status 1 = base stations visible but geometry missing (calibrate in cfclient)"
echo "   * status 0 = no base stations detected"
echo ""
echo "3. Props-off Bench Test (Verifies sign convention & shadow mode):"
echo "   .venv/bin/python Simulation/deploy/bench_bringup.py"
echo ""
echo "4. Arm and Fly:"
echo "   .venv/bin/python Simulation/deploy/radio_flight.py \\"
echo "       --arm-ok --hover 50 --set policy.shadow=0"
echo "-----------------------------------------------------------------------"

if [ "${FLASH_AFTER}" = true ]; then
    echo ""
    echo "======================================================================="
    echo "  AUTO-FLASH TRIGGERED (--flash)"
    echo "======================================================================="
    if [ "${FLASH_COLD}" = true ]; then
        ./flash.sh --cold
    else
        ./flash.sh
    fi
fi

