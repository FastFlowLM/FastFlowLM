# =============================================================================
# Qwen3.5 Omni NPU Test Makefile
# =============================================================================
#
# Author: Zhenyu Xu (zxu3@clemson.edu)
# Created: 2024
#
# This Makefile builds the qwen3_5_omni test. The Qwen3_5_Omni wrapper inherits
# AutoModel (while owning the qwen3_5_omni engine directly, since that engine does
# not inherit from causal_lm), so this build includes AutoModel/automodel.cpp
# alongside the tokenizer and sampler helpers.
#
# Usage:
#   make        - Build all targets
#   make clean  - Remove all built files
#
# =============================================================================
-include ../common.mk


SOURCES += test.cpp
SOURCES += ../../common/AutoModel/automodel.cpp
SOURCES += ../../common/AutoModel/modeling_qwen3_5_omni.cpp
SOURCES += ../../common/AutoModel/modeling_qwen3_5_omni_image.cpp
SOURCES += ../../common/AutoModel/modeling_qwen3_5_omni_audio.cpp
SOURCES += ../../common/image_process_utils/imageproc.cpp
SOURCES += ../../common/image_process_utils/imageprocAVX512.cpp
SOURCES += ../../common/audio_process_utils/audioproc.cpp
SOURCES += ../../common/audio_process_utils/audioprocAVX512.cpp
SOURCES += ../../common/image/image_reader.cpp
SOURCES += ../../common/audio/audio_reader.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/models/qwen3_5_omni/qwen3_5_omni.hpp
HEADERS += ../../include/AutoModel/modeling_qwen3_5_omni.hpp

# omni mode is default;
# 0: text
# 1: images
# 2: audios
# 3: omni
TYPE ?= 3


ifeq ($(WSL), 0)

LDFLAGS += -lqwen3_5_omni_npu
LDFLAGS += -lavformat -lavcodec -lavutil -lswscale -lswresample
LDFLAGS += -lfftw3f

# Map each source to an object file in BUILD_DIR (basenames are unique across
# the source set). VPATH lets the pattern rule locate sources in their dirs.
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(SOURCES)))

all: $(BUILD_DIR)/test_qwen3_5_omni_npu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile each source only when it (or one of its headers) changes.
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Link only when object files change.
$(BUILD_DIR)/test_qwen3_5_omni_npu: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_qwen3_5_omni_npu
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && export LD_LIBRARY_PATH=../../../lib && ./test_qwen3_5_omni_npu --model qwen3.5-omni --type $(TYPE)

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_qwen3_5_omni_npu.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/qwen3_5_omni_npu"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_qwen3_5_omni_npu"

clean:
	rm -rf $(BUILD_DIR)


test: directories $(BUILD_DIR)/test_qwen3_5_omni_npu.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_qwen3_5_omni_npu.exe -m qwen3.5:9b -p 0"

.PHONY: all clean test directories

endif
