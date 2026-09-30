#!/usr/bin/env bash
# ==============================================================================
# MJX Pipeline Runner: encoder + PPO retrain on the JAX / MuJoCo-MJX stack
#
# This is the MJX counterpart to `run_pipeline.sh`.  `run_pipeline.sh` drives the
# numpy/SB3 path (`Simulation/train.py`); this one drives `Simulation/train_mjx.py`.
# Both share Stages 1 and 2 verbatim, because the HISTORY ENCODER IS NOT A JAX
# COMPONENT -- `quad_mjx/encoder.py` is inference-only and loads the checkpoint that
# `encoder/train_encoder.py` (numpy/torch, against the frozen baseline env) writes.
#
# Steps executed in order:
#   1. Collect 5M-frame pretraining corpus   (Simulation/encoder/collect_data.py)
#   2. Train the 5-step history encoder      (Simulation/encoder/train_encoder.py)
#   3. Train PPO with the vectorised MJX env (Simulation/train_mjx.py)
#   4. Convert the JAX/Flax weights to an SB3 .zip the deploy chain can read
#                                            (Simulation/export_to_sb3.py)
#   5. Export to C and compile firmware      (deploy/export_policy.py, build_app.sh)
#
# Usage:
#   ./run_pipeline_mjx.sh                     # Full production run (30M steps)
#   ./run_pipeline_mjx.sh --smoke             # Fast end-to-end plumbing validation
#   ./run_pipeline_mjx.sh --skip-collect      # Reuse an existing corpus
#   ./run_pipeline_mjx.sh --skip-encoder      # Reuse logs/encoder_gru.pt
#   ./run_pipeline_mjx.sh --skip-ppo          # Stages 1-2 only
#   ./run_pipeline_mjx.sh --skip-firmware     # Everything except the firmware build
#   ./run_pipeline_mjx.sh --firmware-only     # Re-export + rebuild from existing weights
#
# Overridable via environment:
#   FRAMES WORKERS ENCODER_EPOCHS ENCODER_HORIZON ENCODER_BUCKET TOTAL_TIMESTEPS SEED
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON="${ROOT_DIR}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: ${PYTHON} not found. This pipeline needs the project venv." >&2
    exit 1
fi

# Pipeline Defaults
FRAMES=${FRAMES:-5000000}
WORKERS=${WORKERS:-8}
# 200 epochs at bucket-factor 2 is the configuration the robustness gate was measured at.
# Raising BUCKET_FACTOR looks tempting (0 -> 0.52 real-frame fraction, 2 -> 0.89, 8 -> 0.80)
# but 8 is measurably WORSE than 2, so leave it alone.
ENCODER_EPOCHS=${ENCODER_EPOCHS:-200}
ENCODER_HORIZON=${ENCODER_HORIZON:-5}
ENCODER_BUCKET=${ENCODER_BUCKET:-2}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-30000000}
SEED=${SEED:-42}
# The production robustness gate. Left ON by default; read the override note at the Stage 2
# call site before setting it false.
ENCODER_REQUIRE_ROBUST=${ENCODER_REQUIRE_ROBUST:-true}

# Artifacts the deploy chain consumes.  Named here once, and passed EXPLICITLY at every
# call site: `export_policy.py` defaults to `--model latest`, which globs the newest
# `logs/rl_model_*.zip` -- a leftover SB3 checkpoint -- so relying on the default after an
# MJX run would silently export the PREVIOUS policy into the firmware.
ENCODER_OUT="logs/encoder_gru.pt"
MJX_NPZ="logs/quad_mjx_policy.npz"
SB3_ZIP="quad_flip_model.zip"
EXPORT_OUT_DIR="Simulation/deploy/app_policy_controller/src/generated"
EXPORT_MANIFEST="Simulation/deploy/manifests/policy_export.json"
REF_OUT_DIR="Simulation/deploy/app_policy_controller/src/generated"
REF_MANIFEST="Simulation/deploy/manifests/reference_tables.json"

SMOKE=false
SKIP_COLLECT=false
SKIP_ENCODER=false
SKIP_PPO=false
SKIP_FIRMWARE=false
SMOKE_FIRMWARE=false
RESUME=false
FIRMWARE_BACKEND="legacy"
FLASH_AFTER=false
FLASH_COLD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-collect)  SKIP_COLLECT=true; shift ;;
        --skip-encoder)  SKIP_ENCODER=true; shift ;;
        --skip-ppo)      SKIP_PPO=true; shift ;;
        --skip-firmware) SKIP_FIRMWARE=true; shift ;;
        --firmware-only)
            SKIP_COLLECT=true; SKIP_ENCODER=true; SKIP_PPO=true; SKIP_FIRMWARE=false; shift ;;
        --smoke)         SMOKE=true; shift ;;
        --smoke-firmware)
            # Opt in to compiling a firmware image from SMOKE weights.  Off by default
            # because it overwrites cf2.bin -- the binary the drone actually flashes.
            SMOKE=true; SMOKE_FIRMWARE=true; shift ;;
        --flash)         FLASH_AFTER=true; shift ;;
        --flash-cold)    FLASH_AFTER=true; FLASH_COLD=true; shift ;;
        --stedgeai)      FIRMWARE_BACKEND="stedgeai"; shift ;;
        --legacy)        FIRMWARE_BACKEND="legacy"; shift ;;
        --resume)        RESUME=true; shift ;;
        -h|--help)
            sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

if [ "${SMOKE}" = true ]; then
    FRAMES=20000
    WORKERS=1
    ENCODER_EPOCHS=2
    SKIP_COLLECT=false
    SKIP_ENCODER=false
    SKIP_PPO=false
    # Divert every artifact to a *_smoke name so a smoke run can never be mistaken for, or
    # overwrite, a real one.  The trainer independently tags its npz/metrics `_smoke`.
    ENCODER_OUT="logs/encoder_gru_smoke.pt"
    MJX_NPZ="logs/quad_mjx_policy_smoke.npz"
    SB3_ZIP="logs/quad_flip_model_smoke.zip"
    EXPORT_OUT_DIR="logs/smoke_generated"
    # The manifest has to move too: diverting only --out-dir still let a smoke run rewrite
    # `manifests/policy_export.json` to describe a 2-epoch policy, which is the provenance
    # record for what is actually in the firmware image.
    EXPORT_MANIFEST="logs/smoke_generated/policy_export.json"
    # Same reasoning for the baked reference tables: identical bytes, but regenerating them
    # churns the committed manifest's timestamp for no reason.
    REF_OUT_DIR="logs/smoke_generated"
    REF_MANIFEST="logs/smoke_generated/reference_tables.json"
fi

# Both heavy stages write into this directory; create it up front so gen_references (which
# runs before export_policy) does not have to create it itself.
if [ "${SMOKE}" = true ]; then mkdir -p logs/smoke_generated; fi

PIPELINE_START=$(date +%s)

echo "======================================================================="
echo "  QUADCOPTER FLIGHT PIPELINE (MJX)"
if [ "${SMOKE}" = true ]; then
    echo "  MODE      : SMOKE (plumbing validation - NOT a deployable policy)"
fi
echo "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python    : ${PYTHON}"
echo "  Directory : ${ROOT_DIR}"
echo "  Config    : frames=${FRAMES} | workers=${WORKERS} | encoder_epochs=${ENCODER_EPOCHS} | ppo_steps=${TOTAL_TIMESTEPS}"
echo "======================================================================="

mkdir -p logs

# ------------------------------------------------------------------------------
# STAGE 1: Data Collection
# ------------------------------------------------------------------------------
if [ "${SKIP_COLLECT}" = false ]; then
    echo ""
    echo "======================================================================="
    echo "  [STAGE 1/5] Collecting Encoder Pretraining Corpus (${FRAMES} frames)"
    echo "  Includes: mass variation, ground-to-hover takeoffs, acrobatic pulses,"
    echo "            asymmetric motor faults, and dynamic recovery kicks."
    echo "======================================================================="
    T_START=$(date +%s)

    # Collect into a STAGING directory and swap only AFTER a successful collection.
    #
    # The obvious order -- move the old shards aside, then collect -- leaves
    # logs/encoder_data EMPTY for the entire duration of the collection, so ANY
    # interruption (Ctrl-C, a crash, a full disk, closing the laptop) strands the project:
    # every downstream command then fails with a confusing "no shard_*.npz found", and the
    # corpus is only recoverable if you think to look in encoder_data_legacy/.  Staging
    # costs the disk of one corpus and removes that failure mode completely -- if anything
    # goes wrong the canonical directory still holds the previous corpus, untouched.
    if [ "${SMOKE}" = true ]; then
        # collect_data appends "_smoke" to --out itself; the smoke corpus lives in its own
        # directory and the canonical one is never touched.
        COLLECT_OUT="logs/encoder_data"
        ENCODER_DATA="logs/encoder_data_smoke"
    else
        COLLECT_OUT="logs/encoder_data_staging"
        ENCODER_DATA="logs/encoder_data"
        rm -rf "${COLLECT_OUT}"
        mkdir -p "${COLLECT_OUT}"
    fi

    COLLECT_ARGS=(--frames "${FRAMES}" --out "${COLLECT_OUT}"
                  --workers "${WORKERS}" --seed "${SEED}")
    if [ "${SMOKE}" = true ]; then COLLECT_ARGS+=(--smoke); fi

    "${PYTHON}" -u Simulation/encoder/collect_data.py "${COLLECT_ARGS[@]}"

    if [ "${SMOKE}" = false ]; then
        if ! compgen -G "${COLLECT_OUT}/*.npz" > /dev/null; then
            echo "ERROR: collection produced no shards in ${COLLECT_OUT}; the existing" >&2
            echo "       corpus in ${ENCODER_DATA} has been left untouched." >&2
            exit 1
        fi
        if compgen -G "${ENCODER_DATA}/*.npz" > /dev/null; then
            # Timestamped subdirectory -- NOT the flat logs/encoder_data_legacy/.  Shard
            # names are deterministic (shard_w00_0000.npz, ...), so a flat move would
            # silently OVERWRITE the previous archive and destroy the older corpus.  A
            # nested directory is invisible to train_encoder.load_episodes, which globs
            # `shard_*.npz` in exactly one directory.
            ARCHIVE_DIR="logs/encoder_data_legacy/$(date '+%Y%m%d_%H%M%S')"
            echo "  Archiving previous shards to ${ARCHIVE_DIR}/ ..."
            mkdir -p "${ARCHIVE_DIR}"
            mv "${ENCODER_DATA}"/*.npz "${ARCHIVE_DIR}/"
        fi
        mv "${COLLECT_OUT}"/*.npz "${ENCODER_DATA}/"
        rmdir "${COLLECT_OUT}" 2>/dev/null || true
        echo "  Promoted $(ls "${ENCODER_DATA}"/*.npz | wc -l | tr -d ' ') shards into ${ENCODER_DATA}/"
    fi

    T_END=$(date +%s)
    echo ">>> Stage 1 completed in $((T_END - T_START))s."
else
    echo ">>> Skipping Stage 1 (Data Collection)."
fi

# Stage 1 sets ENCODER_DATA when it actually runs.  When it is SKIPPED, derive the same
# value here, so there is exactly one definition of where the corpus lives.
if [ -z "${ENCODER_DATA:-}" ]; then
    ENCODER_DATA="logs/encoder_data"
    if [ "${SMOKE}" = true ]; then ENCODER_DATA="logs/encoder_data_smoke"; fi
fi

if [ "${SKIP_ENCODER}" = false ] && { [ ! -d "${ENCODER_DATA}" ] || ! compgen -G "${ENCODER_DATA}/*.npz" > /dev/null; }; then
    echo "ERROR: no encoder shards in ${ENCODER_DATA}." >&2
    echo "       Run Stage 1, or pass --skip-encoder to reuse ${ENCODER_OUT}." >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# STAGE 2: Train Self-Supervised History Encoder
# ------------------------------------------------------------------------------
if [ "${SKIP_ENCODER}" = false ]; then
    echo ""
    echo "======================================================================="
    echo "  [STAGE 2/5] Training 5-Step Self-Supervised Forward Dynamics Encoder"
    echo "  Data      : ${ENCODER_DATA}"
    echo "  Target    : ${ENCODER_OUT} (horizon=${ENCODER_HORIZON}, epochs=${ENCODER_EPOCHS}, bucket=${ENCODER_BUCKET})"
    if [ "${SMOKE}" = true ]; then
        echo "  Gate      : not enforced under --smoke"
    elif [ "${ENCODER_REQUIRE_ROBUST}" = true ]; then
        echo "  Gate      : --require-robust (fails the run rather than shipping a fragile encoder)"
    else
        echo "  Gate      : ENFORCEMENT OFF (ENCODER_REQUIRE_ROBUST=false) - ratio reported only"
    fi
    echo "======================================================================="
    T_START=$(date +%s)

    ENCODER_ARGS=(--mode self_supervised --data "${ENCODER_DATA}"
                  --horizon "${ENCODER_HORIZON}" --epochs "${ENCODER_EPOCHS}"
                  --bucket-factor "${ENCODER_BUCKET}" --seed "${SEED}"
                  --out "${ENCODER_OUT}")
    # --require-robust enforces the corrupted-fold budget. A 2-epoch smoke run cannot meet it
    # (nothing has been learned yet), so it is a PRODUCTION-only assertion -- leaving it on
    # under --smoke would always abort Stage 2 and mask the plumbing the smoke run validates.
    #
    # ENCODER_REQUIRE_ROBUST=false turns the ENFORCEMENT off but keeps the ratio REPORTED.
    # THE EVIDENCE FOR THAT OVERRIDE (2026-09-24, all measured, see
    # /memories/repo/encoder_robustness_metric.md): five arms at 60 epochs, 3000 episodes,
    # seed 42, all scored on ONE FIXED corrupted fold with the instrument decoupled from the
    # training knobs -- corrupted-step weighting 5x (NULL), input un-saturation --clip 60
    # (NULL), doubled corruption exposure (NULL), more GRU width (helps clean and delta_q,
    # does NOT touch delta_w), and the clean-training CONTROL --no-corrupt (indistinguishable
    # from the augmented model on the corrupted fold). Nothing moved the statistic, and the
    # production ratio is driven by clean skill growing rather than by any of these knobs.
    # So the gate as constructed is not passable by this model class, and a gate nothing can
    # pass is a blocker, not a gate. Turning it off is a deliberate trade: acceptance moves to
    # the POLICY-level Lighthouse robustness (Simulation/scratch/test_final_policy.py --
    # outage / loss / runaway / teleport), which is the thing that actually flies. Shipping a
    # policy and measuring that beats blocking forever on a proxy for an undeployed head.
    if [ "${SMOKE}" = false ] && [ "${ENCODER_REQUIRE_ROBUST}" = true ]; then
        ENCODER_ARGS+=(--require-robust)
    fi

    "${PYTHON}" -u Simulation/encoder/train_encoder.py "${ENCODER_ARGS[@]}"

    T_END=$(date +%s)
    echo ">>> Stage 2 completed in $((T_END - T_START))s."
else
    echo ">>> Skipping Stage 2 (Encoder Training)."
    if [ ! -f "${ENCODER_OUT}" ]; then
        echo "ERROR: --skip-encoder but ${ENCODER_OUT} does not exist." >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# STAGE 3: Train PPO Policy (vectorised MJX)
# ------------------------------------------------------------------------------
if [ "${SKIP_PPO}" = false ]; then
    echo ""
    echo "======================================================================="
    echo "  [STAGE 3/5] Training MJX PPO Policy"
    if [ "${SMOKE}" = true ]; then
        echo "  Steps     : SMOKE (40960 steps / 64 envs, eval disabled)"
    else
        echo "  Steps     : ${TOTAL_TIMESTEPS}"
    fi
    echo "  Encoder   : ${ENCODER_OUT} (frozen; passed explicitly)"
    echo "======================================================================="
    T_START=$(date +%s)

    if [ "${SMOKE}" = false ] && [ -f "${MJX_NPZ}" ]; then
        ARCHIVE="${MJX_NPZ%.npz}.$(date '+%Y%m%d_%H%M%S').npz"
        echo "  Archiving previous weights to ${ARCHIVE} ..."
        mv "${MJX_NPZ}" "${ARCHIVE}"
    fi

    MJX_ARGS=(--total-timesteps "${TOTAL_TIMESTEPS}" --seed "${SEED}"
              --encoder "${ENCODER_OUT}")
    if [ "${SMOKE}" = true ]; then
        # --smoke pins the step count and the env count and disables eval; overriding them
        # here would defeat the point of the flag.
        MJX_ARGS=(--smoke --encoder "${ENCODER_OUT}")
    fi

    "${PYTHON}" -u train_mjx.py "${MJX_ARGS[@]}"

    T_END=$(date +%s)
    echo ">>> Stage 3 completed in $((T_END - T_START))s."

    if [ ! -f "${MJX_NPZ}" ]; then
        echo "ERROR: trainer finished but ${MJX_NPZ} was not written." >&2
        exit 1
    fi
else
    echo ">>> Skipping Stage 3 (PPO Training)."
    if [ ! -f "${MJX_NPZ}" ]; then
        echo "ERROR: --skip-ppo but ${MJX_NPZ} does not exist." >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# STAGE 4: Convert JAX weights -> SB3 .zip
# ------------------------------------------------------------------------------
# Everything downstream (evaluate.py, live_flight.py, deploy/export_policy.py) reads SB3
# state-dict keys, so the Flax tree has to be transcoded.  `--verify` round-trips the zip
# back through SB3 and diffs it against the Flax net, which is what catches a silently
# wrong key mapping -- without it a bad export looks like a bad policy.
echo ""
echo "======================================================================="
echo "  [STAGE 4/5] Converting MJX weights to SB3 checkpoint"
echo "  In        : ${MJX_NPZ}"
echo "  Out       : ${SB3_ZIP}"
echo "======================================================================="
T_START=$(date +%s)

"${PYTHON}" -u Simulation/export_to_sb3.py \
    --weights "${MJX_NPZ}" \
    --output "${SB3_ZIP}" \
    --verify

T_END=$(date +%s)
echo ">>> Stage 4 completed in $((T_END - T_START))s."

# ------------------------------------------------------------------------------
# STAGE 5: Export Policy & Build Crazyflie Firmware
# ------------------------------------------------------------------------------
if [ "${SKIP_FIRMWARE}" = false ]; then
    echo ""
    echo "======================================================================="
    echo "  [STAGE 5/5] Exporting Policy & Compiling Crazyflie Firmware"
    echo "  Backend   : ${FIRMWARE_BACKEND}"
    echo "======================================================================="
    T_START=$(date +%s)

    echo "  [5a] Verifying reference trajectory tables..."
    "${PYTHON}" -u Simulation/deploy/gen_references.py \
        --out-dir "${REF_OUT_DIR}" \
        --manifest "${REF_MANIFEST}"

    echo "  [5b] Exporting policy (PPO actor + GRU history encoder) to C..."
    "${PYTHON}" -u Simulation/deploy/export_policy.py \
        --model "${SB3_ZIP}" \
        --encoder "${ENCODER_OUT}" \
        --out-dir "${EXPORT_OUT_DIR}" \
        --manifest "${EXPORT_MANIFEST}"

    if [ "${SMOKE}" = true ] && [ "${SMOKE_FIRMWARE}" = false ]; then
        echo ""
        echo "  [5c] SKIPPED firmware compile: the weights are SMOKE weights."
        echo "       cf2.bin is left untouched.  Re-run without --smoke to build a real image."
    else
        echo "  [5c] Compiling Crazyflie firmware binary (${FIRMWARE_BACKEND})..."
        if [ "${FIRMWARE_BACKEND}" = "stedgeai" ]; then
            ./Simulation/deploy/build_app.sh --stedgeai
        else
            ./Simulation/deploy/build_app.sh --legacy
        fi
    fi

    T_END=$(date +%s)
    echo ">>> Stage 5 completed in $((T_END - T_START))s."
else
    echo ">>> Skipping Stage 5 (Export & Firmware Build)."
fi

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))

echo ""
echo "======================================================================="
echo "  PIPELINE COMPLETED SUCCESSFULLY"
echo "  Total elapsed time: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo "  Saved Encoder     : ${ENCODER_OUT}"
echo "  Saved MJX weights : ${MJX_NPZ}"
echo "  Saved SB3 ckpt    : ${SB3_ZIP}"
echo "  Firmware Binary   : Simulation/deploy/app_policy_controller/build/cf2.bin"
if [ "${SMOKE}" = true ]; then
    echo ""
    echo "  *** SMOKE RUN: none of the above is flight-worthy. ***"
fi
echo "======================================================================="
echo ""
echo "-----------------------------------------------------------------------"
echo "  DEPLOYMENT & FLASHING INSTRUCTIONS"
echo "-----------------------------------------------------------------------"
echo "1. Evaluate the policy over the manoeuvre family suite:"
echo "   .venv/bin/python Simulation/evaluate_mjx.py --weights ${MJX_NPZ}"
echo ""
echo "   NOTE: Simulation/evaluate.py takes NO command-line arguments - it is driven by"
echo "   the module-level MODEL_NAME constant, which defaults to \"latest\" and globs the"
echo "   newest logs/rl_model_*.zip.  After an MJX run that is a LEFTOVER SB3 checkpoint,"
echo "   so it would score the previous policy.  To use it on the converted zip, set"
echo "   MODEL_NAME = \"${SB3_ZIP%.zip}\" in Simulation/evaluate.py (or drop the zip into"
echo "   logs/ under an rl_model_*_steps.zip name)."
echo ""
echo "2. Flash the firmware onto the Crazyflie via radio:"
echo "   ./flash.sh"
echo "   (or cold boot recovery if unresponsive: ./flash.sh --cold)"
echo ""
echo "3. Verify Lighthouse Positioning & Geometry (CRITICAL):"
echo "   .venv/bin/python Simulation/deploy/lighthouse_check.py"
echo "   * status 2 = geometry valid & base-station pulses feeding Kalman estimator"
echo "   * status 1 = base stations visible but geometry missing (calibrate in cfclient)"
echo "   * status 0 = no base stations detected"
echo ""
echo "4. Props-off Bench Test (Verifies sign convention & shadow mode):"
echo "   .venv/bin/python Simulation/deploy/bench_bringup.py"
echo ""
echo "5. Arm and Fly:"
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
