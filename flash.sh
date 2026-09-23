#!/usr/bin/env bash
# ==============================================================================
# Flash Crazyflie Firmware using .venv-client cfloader
#
# Usage:
#   ./flash.sh                # Normal warm-boot radio flash to default/saved URI
#   ./flash.sh --cold         # Cold-boot flash (power-cycle drone in 10s window)
#   ./flash.sh --uri <URI>    # Flash to specific radio URI
#   ./flash.sh --client       # Launch cfclient GUI from .venv-client
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

VENV_CLIENT="${ROOT_DIR}/.venv-client"
CFLOADER="${VENV_CLIENT}/bin/cfloader"
CFCLIENT="${VENV_CLIENT}/bin/cfclient"
BIN="Simulation/deploy/app_policy_controller/build/cf2.bin"

if [ ! -d "${VENV_CLIENT}" ]; then
    echo "ERROR: .venv-client not found at ${VENV_CLIENT}." >&2
    echo "Create it with:" >&2
    echo "  python3 -m venv .venv-client" >&2
    echo "  .venv-client/bin/pip install -U pip cfclient" >&2
    exit 1
fi

# Fallback to python -m if direct wrapper isn't present
if [ ! -x "${CFLOADER}" ]; then
    CFLOADER="${VENV_CLIENT}/bin/python -m cfloader"
fi

# Handle --client shortcut
if [[ "${1:-}" == "--client" || "${1:-}" == "-g" || "${1:-}" == "client" ]]; then
    echo "Launching cfclient from .venv-client..."
    exec "${CFCLIENT}"
fi

MODE="warm"
CUSTOM_URI=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cold|--cold-boot)
            MODE="cold"
            shift
            ;;
        -w|--warm|--warm-boot)
            MODE="warm"
            shift
            ;;
        --uri)
            CUSTOM_URI="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -w, --warm        Warm-boot radio flash (default)"
            echo "  -c, --cold        Cold-boot flash (restart drone within 10s)"
            echo "  --uri <URI>       Specify radio URI (overrides saved URI)"
            echo "  --client          Launch cfclient GUI"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

if [ ! -f "${BIN}" ]; then
    echo "Firmware binary not found at ${BIN}."
    echo "Building firmware now via Simulation/deploy/build_app.sh ..."
    ./Simulation/deploy/build_app.sh
fi

echo "======================================================================="
echo "  CRAZYFLIE FIRMWARE FLASHER (.venv-client)"
echo "  Binary : ${BIN} ($(ls -lh "${BIN}" | awk '{print $5}'))"
echo "  Loader : ${CFLOADER}"
echo "======================================================================="

if [ "${MODE}" = "cold" ]; then
    echo ""
    echo ">>> Starting COLD-BOOT flashing."
    echo ">>> POWER-CYCLE the Crazyflie (battery disconnect/reconnect) in the next 10 seconds!"
    echo ""
    ${CFLOADER} flash "${BIN}" stm32-fw -c
else
    TARGET_URI="radio://0/80/2M/E7E7E7E7E7"
    if [ -n "${CUSTOM_URI}" ]; then
        TARGET_URI="${CUSTOM_URI}"
    elif [ -f "logs/drone_uri.txt" ]; then
        SAVED="$(cat logs/drone_uri.txt | tr -d ' \n\r')"
        if [[ "${SAVED}" == radio://* ]]; then
            TARGET_URI="${SAVED}"
            echo "Using remembered radio URI from logs/drone_uri.txt: ${TARGET_URI}"
        fi
    fi

    echo ""
    echo ">>> Flashing over radio to: ${TARGET_URI}"
    echo ">>> (If connection fails, use './flash.sh --cold' to force cold-boot)"
    echo ""
    ${CFLOADER} flash "${BIN}" stm32-fw -w "${TARGET_URI}"
fi

echo ""
echo "======================================================================="
echo "  FLASH COMPLETED"
echo "  Next steps:"
echo "  1. Check Lighthouse: .venv/bin/python Simulation/deploy/lighthouse_check.py"
echo "  2. Bench Bringup   : .venv/bin/python Simulation/deploy/bench_bringup.py"
echo "  3. Live Flight     : .venv/bin/python Simulation/deploy/radio_flight.py --arm-ok --hover 50 --set policy.shadow=0"
echo "======================================================================="
