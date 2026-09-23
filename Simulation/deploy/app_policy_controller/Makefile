# Out-of-tree build of the policy controller app against the local firmware clone.
#
# The app lives HERE (tracked in the project repo); the firmware clone is disposable and
# gitignored, which is why CRAZYFLIE_BASE points at it explicitly.
#
#   make cf2_defconfig   (once, in this directory)
#   make -j$(sysctl -n hw.ncpu)
#
# macOS needs GNU readlink/sed/cp for the app-layer build:
#   brew install coreutils gnu-sed

CRAZYFLIE_BASE := $(abspath ../../../firmware)

#
# App-layer configuration (must enable CONFIG_APP_ENABLE + CONFIG_CONTROLLER_OOT).
#
OOT_CONFIG := $(PWD)/app-config

#
# Which policy backend to compile: `legacy` (default, the hand-written float net) or
# `stedgeai` (the ST Edge AI Core generated network + the ST runtime archive).
#   make APP_BACKEND=stedgeai      (or Simulation/deploy/build_app.sh --stedgeai)
# Both are exported so the kbuild files below can branch on APP_BACKEND, and so the
# ST pack location stays one variable.
#
APP_BACKEND ?= legacy
STEDGEAI_DIR ?= /Applications/ST/STEdgeAI/4.0
export APP_BACKEND STEDGEAI_DIR

#
# Our sources include each other by name (policy_net.h, generated/policy_weights.h,
# generated/stedgeai/network.h).
#
EXTRA_CFLAGS += -I$(PWD)/src

ifeq ($(APP_BACKEND),stedgeai)
EXTRA_CFLAGS += -DAPP_BACKEND_STEDGEAI=1
EXTRA_CFLAGS += -I$(PWD)/src/generated/stedgeai
EXTRA_CFLAGS += -I$(STEDGEAI_DIR)/Middlewares/ST/AI/Inc
endif

include $(CRAZYFLIE_BASE)/tools/make/oot.mk
