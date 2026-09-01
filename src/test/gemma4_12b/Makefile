# =============================================================================
# Gemma4-12B NPU Test Makefile
# =============================================================================
#
# Test harness for the Gemma4-12B omni model. Links against the
# prebuilt libgemma4_12b_npu engine in ../../lib/$(FLM_RUNTIME).
#
# Usage:
#   make        - Build the test executable
#   make test   - Build and run it against the gemma4-12b tag
#
# Test scenario is selected with -t: 0 text, 1 image, 2 audio, 3 omni.
#   make clean  - Remove all built files
#
# =============================================================================
-include ../common.mk

# Test scenario: 0 = text, 1 = image, 2 = audio, 3 = omni (image + audio)
TYPE ?= 3

SOURCES += test.cpp
SOURCES += ../../common/AutoModel/automodel.cpp
SOURCES += ../../common/AutoModel/modeling_gemma4_12b.cpp
SOURCES += ../../common/AutoModel/modeling_gemma4_12b_image.cpp
SOURCES += ../../common/AutoModel/modeling_gemma4_12b_audio.cpp
SOURCES += ../../common/image_process_utils/imageproc.cpp
SOURCES += ../../common/image_process_utils/imageprocAVX512.cpp
SOURCES += ../../common/image/image_reader.cpp
SOURCES += ../../common/audio/audio_reader.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/models/gemma4_12b/gemma4_12b_npu.hpp
HEADERS += ../../include/AutoModel/modeling_gemma4_12b.hpp

ifeq ($(WSL), 0)

LDFLAGS += -lgemma4_12b_npu
LDFLAGS += -lavformat -lavcodec -lavutil -lswscale -lswresample

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_gemma4_12b

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_gemma4_12b: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_gemma4_12b
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ./test_gemma4_12b --model gemma4-it:12b -s 1 -t $(TYPE)

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_gemma4_12b.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/gemma4_12b"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_gemma4_12b"

clean:
	rm -rf $(BUILD_DIR)

test: directories $(BUILD_DIR)/test_gemma4_12b.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_gemma4_12b.exe -m gemma4-it:12b -s 1 -p 0 -t $(TYPE)"

.PHONY: all clean test directories

endif
