# =============================================================================
# Qwen NPU Library Makefile
# =============================================================================
#
# Author: Zhenyu Xu (zxu3@clemson.edu)
# Created: 2024
#
# This Makefile is used to build the Qwen NPU library components.
#
# Usage:
#   make        - Build all targets
#   make clean  - Remove all built files
#   make help   - Show this help message
#
# =============================================================================
-include ../common.mk

SOURCES += $(wildcard ../../common/AutoModel/automodel.cpp)
SOURCES += $(wildcard ../../common/AutoModel/modeling_gemma3_text.cpp)
SOURCES += $(wildcard ../../common/tokenizer/tokenizer.cpp)
SOURCES += $(wildcard ../../common/modules/sampler.cpp)
SOURCES += $(wildcard test.cpp)

HEADERS += ../../include/gemma_text/gemma_text_npu.hpp
HEADERS += ../../include/gemma_text/gemma_text_npu_sequence.hpp
HEADERS += ../../include/gemma_text/gemma_text_lm_head.hpp

ifeq ($(WSL), 0)

LDFLAGS += -lgemma_text_npu

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_gemma_text_npu test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_gemma_text_npu: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_gemma_text_npu
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && export LD_LIBRARY_PATH=../../../lib && ./test_gemma_text_npu --model gemma3:1b -s 1

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_gemma_text_npu.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/gemma_text_npu"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_gemma_text_npu_target"

clean:
	rm -rf $(BUILD_DIR)


test: directories $(BUILD_DIR)/test_gemma_text_npu.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_gemma_text_npu.exe -m gemma3:270m -s 1 -p 0"

.PHONY: all clean test directories

endif 
