# =============================================================================
# Qwen3.6 MoE NPU Library Makefile
# =============================================================================
#
# This Makefile is used to build the Qwen3.6 MoE NPU library components.
#
# Usage:
#   make        - Build all targets
#   make clean  - Remove all built files
#   make help   - Show this help message
#
# =============================================================================
-include ../common.mk


SOURCES += $(wildcard ../../common/AutoModel/automodel.cpp)
SOURCES += $(wildcard ../../common/AutoModel/modeling_qwen3_6_moe.cpp)
SOURCES += $(wildcard ../../common/AutoModel/modeling_qwen3_6_moe_image.cpp)
SOURCES += $(wildcard ../../common/image_process_utils/imageproc.cpp)
SOURCES += $(wildcard ../../common/image_process_utils/imageprocAVX512.cpp)
SOURCES += $(wildcard ../../common/image/image_reader.cpp)
SOURCES += $(wildcard ../../common/tokenizer/tokenizer.cpp)
SOURCES += $(wildcard ../../common/modules/sampler.cpp)
SOURCES += $(wildcard test.cpp)

HEADERS += ../../include/qwen3_6_moe/qwen3_6_moe_npu.hpp
HEADERS += ../../include/qwen3_6_moe/qwen3_6_moe_npu_sequence.hpp
HEADERS += ../../include/modules/embedding.hpp
HEADERS += ../../include/modules/lm_head.hpp
HEADERS += ../../include/modules/gemm.hpp
HEADERS += ../../include/modules/dequant.hpp


ifeq ($(WSL), 0)

LDFLAGS += -lqwen3_6_moe_npu -lmha -ldequant -lgemm
LDFLAGS += -lavformat -lavcodec -lavutil -lswscale -lswresample

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_qwen3_6_moe_npu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_qwen3_6_moe_npu: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_qwen3_6_moe_npu
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && export LD_LIBRARY_PATH=../../../lib && ./test_qwen3_6_moe_npu --model qwen3.6-moe -s 1

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test


host: $(BUILD_DIR)/test_qwen3_6_moe_npu.exe


directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_qwen3_6_moe_npu.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/qwen3_6_moe_npu"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_qwen3_6_moe_npu_target"

clean:
	rm -rf $(BUILD_DIR)

test: directories $(BUILD_DIR)/test_qwen3_6_moe_npu.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_qwen3_6_moe_npu.exe -m qwen3.6-moe -s 1 -p 0"

.PHONY: all clean test directories

endif
