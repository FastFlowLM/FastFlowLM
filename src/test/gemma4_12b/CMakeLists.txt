cmake_minimum_required(VERSION 3.22)
project(gemma4_12b VERSION 1.0.0 LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../CMakeLists.txt)
npu_test_setup()

add_npu_test(
    test_gemma4_12b
    test/gemma4_12b
    USE_AUTOMODEL
    USE_TOKENIZER
    USE_SAMPLER
    SOURCES
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_gemma4_12b.cpp"
)

target_compile_definitions(test_gemma4_12b PUBLIC
    __FLM_VERSION__="0.9.45"
)

target_link_libraries(test_gemma4_12b PUBLIC
    gemma4_12b_npu
    xrt_coreutil
)

# Add test target
add_custom_target(test_gemma4_12b_target
    DEPENDS test_gemma4_12b
    COMMENT "Building test_gemma4_12b executable"
)
