cmake_minimum_required(VERSION 3.22)
project(qwen3_asr_preprocessor_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
find_library(FFTW3F_LIBRARY NAMES fftw3f libfftw3f.so.3 REQUIRED)

add_executable(test_qwen3_asr_preprocessor
    test.cpp
    ../../common/audio/qwen3_asr_preprocessor.cpp
    ../../common/audio_process_utils/audioproc.cpp
    ../../common/audio_process_utils/audioprocAVX512.cpp
)

target_include_directories(test_qwen3_asr_preprocessor PRIVATE ../../include)
target_compile_definitions(test_qwen3_asr_preprocessor PRIVATE USEAVX2=1 USEAVX512=1)
target_compile_options(test_qwen3_asr_preprocessor PRIVATE
    -O3 -ffast-math -mavx2 -mfma -mavx512f -mavx512dq -mavx512bw -mavx512vl
)
target_link_libraries(test_qwen3_asr_preprocessor PRIVATE ${FFTW3F_LIBRARY})

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_qwen3_asr_preprocessor PRIVATE OpenMP::OpenMP_CXX)
endif()

enable_testing()
add_test(NAME qwen3_asr_preprocessor COMMAND test_qwen3_asr_preprocessor)
