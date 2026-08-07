
# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

# get current directory name
CURRENT_DIR := $(notdir $(CURDIR))

BUILD_DIR := ../../build/test/$(CURRENT_DIR)

# ————————————————————————————————————————————————————————————————
# NPU runtime backend selection (mirrors the CMake FLM_USE_HRX flag).
#   FLM_USE_HRX=0 -> XRT (default)   FLM_USE_HRX=1 -> HRX
# Prebuilt engine libs are consumed from the matching per-backend subdir.
# ————————————————————————————————————————————————————————————————
FLM_USE_HRX ?= 0

ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -DUSEAVX2=1
CXX_FLAGS += -mavx2 -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -MMD -MP
CXX_FLAGS += -DDEV_BUILD
CXX_FLAGS += -fopenmp
CXX_FLAGS += -DCMAKE_INSTALL_PREFIX="\"/opt/fastflowlm\""
CXX_FLAGS += -DCMAKE_XCLBIN_PREFIX="\"/opt/fastflowlm/share/flm/xclbins\""
#NOTE: TODO: FIXME: Either deprecate makefile, or keep the parameter sync with ../CMAKELists.txt, otherwise it is error-prone
CXX_FLAGS += -D__FLM_VERSION__="\"0.9.34\""
CXX_FLAGS += -D__NPU_VERSION__="\"32.0.203.304\""
CXX_FLAGS += -DDISABLE_ABI_CHECK=1

ifeq ($(FLM_USE_HRX),1)
# --- HRX backend ---
# Consumed from the pinned release artifact fetched by
# hrx-integration/fetch-hrx-release.sh (pin: hrx-integration/hrx-release.env).
# The artifact's env.sh exports HRX_DIR / HRX_BUILD; source it before building
# or override HRX_DIR / HRX_BUILD on the make command line.
LIB_DIR := ../../lib/hrx
HRX_MK_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT  := $(abspath $(HRX_MK_DIR)/../..)
HRX_RELEASE_DIR   ?= $(REPO_ROOT)/hrx-integration/.hrx-release
HRX_ARTIFACT_ROOT ?= $(firstword $(wildcard $(HRX_RELEASE_DIR)/hrx-amdxdna-*/))
HRX_DIR   ?= $(HRX_ARTIFACT_ROOT)HRX_DIR
HRX_BUILD ?= $(HRX_ARTIFACT_ROOT)HRX_BUILD
ifeq ($(wildcard $(HRX_BUILD)/libhrx/src/libhrx/libhrx.so),)
$(error HRX artifact not found at "$(HRX_BUILD)". Run hrx-integration/fetch-hrx-release.sh (or `source <artifact>/env.sh`) before building, or override HRX_DIR/HRX_BUILD on the make command line.)
endif
HRX_INC := -I$(HRX_DIR)/libhrx/include -I$(HRX_DIR)/runtime/src \
           -I$(HRX_BUILD)/runtime/src -I$(HRX_BUILD)/_deps/flatcc-src/include
HRX_LIBS := $(HRX_BUILD)/libhrx/src/libhrx/libhrx.so $(HRX_BUILD)/libflatcc_runtime.a
HRX_RPATH := -Wl,-rpath,$(HRX_BUILD)/libhrx/src/libhrx
CXX_FLAGS += -DFLM_USE_HRX=1 $(HRX_INC)
RUNTIME_LDFLAGS := $(HRX_RPATH) $(HRX_LIBS)
else
# --- XRT backend (default) ---
LIB_DIR := ../../lib/xrt
XRT_PREFIX ?= /opt/xilinx/xrt
CXX_FLAGS += -I$(XRT_PREFIX)/include
RUNTIME_LDFLAGS := -L$(XRT_PREFIX)/lib -Wl,-rpath,$(XRT_PREFIX)/lib -lxrt_coreutil -laiebu
endif

LDFLAGS += -lboost_program_options -lboost_filesystem
LDFLAGS += -L$(LIB_DIR)
# Resolve the prebuilt engine .so (and bundled aiebu) at run time.
LDFLAGS += -Wl,-rpath,$(abspath $(LIB_DIR))
LDFLAGS += -L../../build/tokenizers-cpp
LDFLAGS += -L../../build/tokenizers-cpp/sentencepiece/src
LDFLAGS += -ltokenizers_cpp -ltokenizers_c -lsentencepiece
LDFLAGS += $(RUNTIME_LDFLAGS)
DEPENDENCY_LDFLAGS += -lmha -ldequant -lgemm -llm_head -lq4_npu_eXpress


SOURCES += ../../common/utils.cpp

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

endif 
