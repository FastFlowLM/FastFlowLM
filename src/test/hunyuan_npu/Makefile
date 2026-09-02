# =============================================================================
# Hy-MT2-1.8B (hunyuan-dense) NPU Test Makefile
# =============================================================================
#
# Test harness for the text-only hunyuan-dense engine. Links against the
# prebuilt libhunyuan_npu in ../../lib/$(FLM_RUNTIME).
#
# Usage:
#   make        - Build the test executable
#   make test   - Build and run the danmaku benchmark against the hunyuan:1.8b tag
#   make clean  - Remove all built files
#
# Benchmark knobs (passed through to the executable):
#   WARMUP=3        - leading turns discarded before the statistics are collected
#   COUNT=0         - turns to run, 0 = the whole corpus in bench_samples.hpp
#   LENGTH=512      - max generated tokens per turn
#   CSV=bench.csv   - per-turn CSV written next to the executable, empty to skip
#   PIN=1           - 1 pins the constant instruction in the KV cache at init,
#                     0 re-prefills it every turn (the A/B reference)
#
# =============================================================================
-include ../common.mk

WARMUP ?= 3
COUNT  ?= 0
LENGTH ?= 512
CSV    ?= bench.csv
PIN    ?= 1

BENCH_ARGS := --warmup $(WARMUP) --count $(COUNT) --Length $(LENGTH) --pin $(PIN)
ifneq ($(CSV),)
BENCH_ARGS += --csv $(CSV)
endif

SOURCES += test.cpp
SOURCES += ../../common/AutoModel/automodel.cpp
SOURCES += ../../common/AutoModel/modeling_hunyuan.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/models/hunyuan/hunyuan_npu.hpp
HEADERS += ../../include/AutoModel/modeling_hunyuan.hpp
HEADERS += bench_samples.hpp

ifeq ($(WSL), 0)

LDFLAGS += -lhunyuan_npu

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_hunyuan

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_hunyuan: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_hunyuan
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ./test_hunyuan --model hunyuan:1.8b $(BENCH_ARGS)

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_hunyuan.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/hunyuan_npu"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_hunyuan"

clean:
	rm -rf $(BUILD_DIR)

test: directories $(BUILD_DIR)/test_hunyuan.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_hunyuan.exe -m hunyuan:1.8b -p 0 $(BENCH_ARGS)"

.PHONY: all clean test directories

endif
