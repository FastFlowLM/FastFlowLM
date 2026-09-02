cmake_minimum_required(VERSION 3.22)
project(hunyuan_npu VERSION 1.0.0 LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../CMakeLists.txt)
npu_test_setup()

add_npu_test(
    test_hunyuan
    test/hunyuan_npu
    USE_AUTOMODEL
    USE_TOKENIZER
    USE_SAMPLER
    SOURCES
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_hunyuan.cpp"
)

target_compile_definitions(test_hunyuan PUBLIC
    __FLM_VERSION__="0.9.45"
)

# Add OpenMP compiler flags if available
if(OpenMP_CXX_FOUND)
    target_compile_options(test_hunyuan PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:/openmp>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fopenmp>
    )
endif()

target_link_libraries(test_hunyuan PUBLIC
    hunyuan_npu
    xrt_coreutil
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_hunyuan PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add test target
add_custom_target(test_hunyuan_target
    DEPENDS test_hunyuan
    COMMENT "Building test_hunyuan executable"
)
