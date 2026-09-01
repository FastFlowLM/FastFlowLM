cmake_minimum_required(VERSION 3.22)
project(phi4_corelib_aie4_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(RYZENAI_CORELIB_INCLUDE_DIR
          NAMES ryzenai/corelib.h
          REQUIRED)

get_filename_component(
    FASTFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    ABSOLUTE)
set(XRT_INCLUDE_DIR
    "C:/dev/XRT/src/runtime_src/core/include"
    CACHE PATH "Where XRT headers live")
set(XRT_LIB_DIR
    "C:/dev/xrtNPUfromDLL"
    CACHE PATH "Where XRT libraries live")

include("${FASTFLOW_SOURCE_DIR}/common/corelib/corelib_sources.cmake")
add_library(flm_corelib_aie4_testlib STATIC
    ${FLM_CORELIB_AIE4_SOURCES})
target_include_directories(flm_corelib_aie4_testlib PUBLIC
    ${FASTFLOW_SOURCE_DIR}/include
    ${RYZENAI_CORELIB_INCLUDE_DIR}
    ${XRT_INCLUDE_DIR})
target_link_directories(flm_corelib_aie4_testlib PUBLIC
    ${XRT_LIB_DIR})
target_compile_definitions(flm_corelib_aie4_testlib PUBLIC
    FLM_ENABLE_CORELIB_AIE4=1
    FLM_CORELIB_TESTING=1
    DEV_BUILD=1
    __WINDOWS__
    USEAVX2=1
    DISABLE_ABI_CHECK=1
    _ENABLE_EXTENDED_ALIGNED_STORAGE
    CMAKE_INSTALL_PREFIX=\"${FASTFLOW_SOURCE_DIR}/build/phi4_corelib_aie4-tests\"
    CMAKE_XCLBIN_PREFIX=\"${FASTFLOW_SOURCE_DIR}/xclbins\"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
    WIN32_LEAN_AND_MEAN
    NOMINMAX)
target_compile_options(flm_corelib_aie4_testlib PRIVATE /fp:precise)
target_link_libraries(flm_corelib_aie4_testlib PUBLIC
    xrt_coreutil
    shell32
    ole32)

add_library(fake_ryzenai_corelib SHARED
    fake_corelib.cpp)
target_include_directories(fake_ryzenai_corelib PRIVATE
    ${RYZENAI_CORELIB_INCLUDE_DIR})
target_compile_definitions(fake_ryzenai_corelib PRIVATE
    RYZENAI_CORELIB_EXPORT=1
    FLM_FAKE_CORELIB_DLL=1)

function(add_corelib_host_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME}
        ${TEST_SOURCE}
        fake_corelib.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE
        flm_corelib_aie4_testlib)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${XRT_LIB_DIR}")
endfunction()

enable_testing()
add_corelib_host_test(test_corelib_api test_corelib_api.cpp)
add_corelib_host_test(test_phi4_manifest test_phi4_manifest.cpp)
add_corelib_host_test(test_phi4_shape_plan test_phi4_shape_plan.cpp)
add_corelib_host_test(test_phi4_weights test_phi4_weights.cpp)
add_corelib_host_test(
    test_corelib_fatal_record
    test_corelib_fatal_record.cpp)
add_dependencies(
    test_corelib_fatal_record
    fake_ryzenai_corelib)
target_link_libraries(test_corelib_fatal_record PRIVATE
    advapi32)
