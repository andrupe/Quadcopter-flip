#!/bin/zsh
# Build the policy-controller app for the Crazyflie 2.x (out-of-tree).
#
# One command, deterministic on macOS:
#   Simulation/deploy/build_app.sh                # legacy backend (default)
#   Simulation/deploy/build_app.sh --stedgeai     # ST Edge AI Core generated network
#
# It (1) finds the ARM toolchain, (2) generates the default cf2 config, (3) merges
# app-config with the tracked stdlib-only merger (the firmware's merge_config.sh is
# GNU-only and fails SILENTLY on macOS - see apply_oot_config.py), (4) builds, (5)
# reports flash/RAM and confirms what is really in the image.
set -e

BACKEND=legacy
case "${1:-}" in
  ""|--legacy)  BACKEND=legacy ;;
  --stedgeai)   BACKEND=stedgeai ;;
  *) echo "usage: build_app.sh [--legacy | --stedgeai]" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HERE/app_policy_controller"
TOOLS_DIR="${TOOLS_DIR:-/Users/apan/University/IntelligentControl/tools}"
TOOLCHAIN="$TOOLS_DIR/arm-gnu-toolchain-14.3.rel1-darwin-arm64-arm-none-eabi/bin"

if [ -x "$TOOLCHAIN/arm-none-eabi-gcc" ]; then
  export PATH="$TOOLCHAIN:$PATH"
elif ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "arm-none-eabi-gcc not found: set TOOLS_DIR or install the ARM toolchain" >&2
  exit 1
fi

cd "$APP"

PY="$(cd "$HERE/../.." && pwd)/.venv/bin/python"
[ -x "$PY" ] || PY=python3

if [ "$BACKEND" = "stedgeai" ]; then
  # Copies the generated network into the source tree and cross-checks its shapes and
  # constants against the exporter before a single file is compiled.
  "$PY" "$HERE/install_stedgeai.py" ${STEDGEAI_DIR:+--stedgeai-dir "$STEDGEAI_DIR"}
  echo
fi

# 0. A BACKEND SWITCH MUST REBUILD FROM SCRATCH. kbuild tracks APP_BACKEND for COMPILING
#    (a different .o is produced) but not for the final link, so a legacy tree rebuilt as
#    --stedgeai keeps the already-linked legacy objects: the app's own objects are correct
#    and the IMAGE is not. Measured exactly that way - stai_network_run missing AND
#    POLICY_HID_W_0 still present at the same time, which is a state the asserts below can
#    detect but cannot explain. Cleaning on the transition costs a full rebuild, once.
STAMP=build/.app_backend
PREV="$( [ -f "$STAMP" ] && cat "$STAMP" || echo "" )"
if [ "$PREV" != "$BACKEND" ]; then
  if [ -n "$PREV" ]; then
    echo "backend changed ($PREV -> $BACKEND): make clean"
  fi
  make clean >/dev/null 2>&1 || true
  mkdir -p build
  printf '%s' "$BACKEND" > "$STAMP"
fi

# 1. config skeleton (exists -> cheap no-op)
make cf2_defconfig >/dev/null

# 2. merge OUR config (mandatory: the build's own merge is a silent no-op on macOS)
"$PY" "$HERE/apply_oot_config.py" build/.config app-config

# 3. build
echo "backend: $BACKEND"
if [ "$BACKEND" = "stedgeai" ]; then
  make APP_BACKEND=stedgeai -j"$(sysctl -n hw.ncpu)"
else
  make -j"$(sysctl -n hw.ncpu)"
fi

# 4. prove what is in the image
echo
arm-none-eabi-size build/cf2.elf
if ! arm-none-eabi-nm build/cf2.elf | grep -q "controllerOutOfTreeInit"; then
  echo "policy controller: *** MISSING - the app was not linked (check app-config merge) ***" >&2
  exit 1
fi
echo "policy controller: PRESENT in the image"
if [ "$BACKEND" = "stedgeai" ]; then
  # The ST path must carry the generated network and the runtime, and must NOT carry the
  # legacy float weight arrays that nothing references any more.
  arm-none-eabi-nm build/cf2.elf | grep -q "stai_network_run" \
    && echo "ST backend:        stai_network_run PRESENT" \
    || { echo "ST backend: *** stai_network_run missing - the runtime was not linked ***" >&2; exit 1; }
  if arm-none-eabi-nm build/cf2.elf | grep -q "POLICY_HID_W_0"; then
    echo "ST backend: *** POLICY_HID_W_0 is still in the image (legacy weights linked) ***" >&2
    exit 1
  fi
  echo "ST backend:        legacy float arrays absent"
else
  arm-none-eabi-nm build/cf2.elf | grep -q "POLICY_HID_W_0" \
    && echo "legacy backend:    POLICY_HID_W_0 PRESENT" \
    || { echo "legacy backend: *** POLICY_HID_W_0 missing - the weights were not linked ***" >&2; exit 1; }
fi
echo "artifact: $APP/build/cf2.bin"
