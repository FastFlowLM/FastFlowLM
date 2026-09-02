cmake_minimum_required(VERSION 3.22)
project(phi4_corelib_aie4_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(RYZENAI_CORELIB_INCLUDE_DIR
          NAMES ryzenai/corelib.h
          REQUIRED)
find_path(BOOST_INCLUDE_DIR
          NAMES boost/any.hpp
          HINTS
              "$ENV{BOOST_INCLUDEDIR}"
              "$ENV{BOOST_ROOT}"
              "$ENV{CONDA_PREFIX}/Library/include"
              "$ENV{USERPROFILE}/anaconda3/Library/include"
              "C:/dev/boost_1_88_0"
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
    ${BOOST_INCLUDE_DIR}
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
if(MSVC)
    target_compile_options(flm_corelib_aie4_testlib PRIVATE
        /arch:AVX2
        /fp:precise
        $<$<CONFIG:Release>:/O2>)
endif()
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
add_executable(test_model_catalog
    test_model_catalog.cpp
    ${FASTFLOW_SOURCE_DIR}/pull/model_overlay.cpp)
target_include_directories(test_model_catalog PRIVATE
    ${FASTFLOW_SOURCE_DIR}/include
    ${FASTFLOW_SOURCE_DIR}/pull)
target_compile_definitions(test_model_catalog PRIVATE
    FLM_TEST_SOURCE_DIR="${FASTFLOW_SOURCE_DIR}")
add_test(NAME test_model_catalog COMMAND test_model_catalog)

add_library(test_model_downloader_compile OBJECT
    ${FASTFLOW_SOURCE_DIR}/pull/model_downloader.cpp)
target_include_directories(test_model_downloader_compile PRIVATE
    ${FASTFLOW_SOURCE_DIR}/include
    ${FASTFLOW_SOURCE_DIR}/pull
    ${FASTFLOW_SOURCE_DIR}/../third_party/tokenizers-cpp/include
    ${BOOST_INCLUDE_DIR}
    ${XRT_INCLUDE_DIR})
target_compile_definitions(test_model_downloader_compile PRIVATE
    __FLM_VERSION__="${FLM_VERSION}")

add_executable(test_generation_limit
    test_generation_limit.cpp
    ${FASTFLOW_SOURCE_DIR}/server/generation_limit.cpp
    ${FASTFLOW_SOURCE_DIR}/server/npu_access_manager.cpp)
target_include_directories(test_generation_limit PRIVATE
    ${FASTFLOW_SOURCE_DIR}/include)
add_test(NAME test_generation_limit COMMAND test_generation_limit)

add_corelib_host_test(test_corelib_api test_corelib_api.cpp)
add_corelib_host_test(test_phi4_manifest test_phi4_manifest.cpp)
add_corelib_host_test(test_phi4_shape_plan test_phi4_shape_plan.cpp)
add_corelib_host_test(test_phi4_host test_phi4_host.cpp)
add_corelib_host_test(test_phi4_weights test_phi4_weights.cpp)
add_corelib_host_test(test_phi4_engine test_phi4_engine.cpp)
target_link_libraries(test_phi4_engine PRIVATE
    advapi32)
add_corelib_host_test(
    test_corelib_fatal_record
    test_corelib_fatal_record.cpp)
add_dependencies(
    test_corelib_fatal_record
    fake_ryzenai_corelib)
target_link_libraries(test_corelib_fatal_record PRIVATE
    advapi32)

set(PHI4_FRONTEND_SOURCES
    ${FASTFLOW_SOURCE_DIR}/common/AutoModel/automodel.cpp
    ${FASTFLOW_SOURCE_DIR}/common/AutoModel/modeling_phi4.cpp
    ${FASTFLOW_SOURCE_DIR}/server/generation_limit.cpp)

function(add_phi4_frontend_test TEST_NAME ENABLE_AIE4)
    add_executable(${TEST_NAME}
        test_phi4_frontend.cpp
        ${PHI4_FRONTEND_SOURCES})
    target_include_directories(${TEST_NAME} PRIVATE
        ${FASTFLOW_SOURCE_DIR}/include
        ${FASTFLOW_SOURCE_DIR}/../third_party/tokenizers-cpp/include
        ${BOOST_INCLUDE_DIR}
        ${XRT_INCLUDE_DIR})
    target_compile_definitions(${TEST_NAME} PRIVATE
        FLM_CORELIB_TESTING=1
        DEV_BUILD=1
        __WINDOWS__
        USEAVX2=1
        DISABLE_ABI_CHECK=1
        _ENABLE_EXTENDED_ALIGNED_STORAGE
        CMAKE_INSTALL_PREFIX="${FASTFLOW_SOURCE_DIR}/build/phi4_corelib_aie4-tests"
        CMAKE_XCLBIN_PREFIX="${FASTFLOW_SOURCE_DIR}/xclbins"
        WIN32_LEAN_AND_MEAN
        NOMINMAX)
    target_compile_options(${TEST_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2 /fp:precise>)
    target_link_directories(${TEST_NAME} PRIVATE
        ${XRT_LIB_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE
        xrt_coreutil)

    if(ENABLE_AIE4)
        target_compile_definitions(${TEST_NAME} PRIVATE
            FLM_ENABLE_CORELIB_AIE4=1)
        target_include_directories(${TEST_NAME} PRIVATE
            ${RYZENAI_CORELIB_INCLUDE_DIR})
        target_link_libraries(${TEST_NAME} PRIVATE
            flm_corelib_aie4_testlib)
        add_dependencies(${TEST_NAME}
            fake_ryzenai_corelib)
    endif()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${XRT_LIB_DIR}")
endfunction()

add_phi4_frontend_test(test_phi4_frontend_off FALSE)
add_phi4_frontend_test(test_phi4_frontend_on TRUE)

function(add_phi4_frontend_compile_check TARGET_NAME ENABLE_AIE4)
    add_library(${TARGET_NAME} OBJECT
        ${PHI4_FRONTEND_SOURCES})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${FASTFLOW_SOURCE_DIR}/include
        ${FASTFLOW_SOURCE_DIR}/../third_party/tokenizers-cpp/include
        ${BOOST_INCLUDE_DIR}
        ${XRT_INCLUDE_DIR})
    target_compile_definitions(${TARGET_NAME} PRIVATE
        DEV_BUILD=1
        __WINDOWS__
        USEAVX2=1
        DISABLE_ABI_CHECK=1
        _ENABLE_EXTENDED_ALIGNED_STORAGE
        CMAKE_INSTALL_PREFIX="${FASTFLOW_SOURCE_DIR}/build/phi4_corelib_aie4-tests"
        CMAKE_XCLBIN_PREFIX="${FASTFLOW_SOURCE_DIR}/xclbins"
        WIN32_LEAN_AND_MEAN
        NOMINMAX)
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2 /fp:precise>)
    if(ENABLE_AIE4)
        target_compile_definitions(${TARGET_NAME} PRIVATE
            FLM_ENABLE_CORELIB_AIE4=1)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${RYZENAI_CORELIB_INCLUDE_DIR})
    endif()
endfunction()

add_phi4_frontend_compile_check(
    phi4_frontend_compile_off
    FALSE)
add_phi4_frontend_compile_check(
    phi4_frontend_compile_on
    TRUE)

set(TASK10_PRODUCTION_SOURCES
    ${FASTFLOW_SOURCE_DIR}/server/generation_limit.cpp
    ${FASTFLOW_SOURCE_DIR}/server/npu_access_manager.cpp
    ${FASTFLOW_SOURCE_DIR}/server/rest_handler.cpp
    ${FASTFLOW_SOURCE_DIR}/server/server.cpp
    ${FASTFLOW_SOURCE_DIR}/runner/runner.cpp
    ${FASTFLOW_SOURCE_DIR}/src/main.cpp)

set(TASK10_STUB_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/task10-compile-stubs")
# These production translation units include declaration-only FFmpeg headers
# transitively. The standalone host suite does not link or execute FFmpeg.
foreach(TASK10_STUB_HEADER IN ITEMS
        libavcodec/avcodec.h
        libswscale/swscale.h
        libavutil/imgutils.h
        libavutil/frame.h
        libavutil/pixfmt.h)
    get_filename_component(
        TASK10_STUB_PARENT
        "${TASK10_STUB_INCLUDE_DIR}/${TASK10_STUB_HEADER}"
        DIRECTORY)
    file(MAKE_DIRECTORY "${TASK10_STUB_PARENT}")
    file(WRITE
        "${TASK10_STUB_INCLUDE_DIR}/${TASK10_STUB_HEADER}"
        "#pragma once\n")
endforeach()

function(add_task10_production_compile_check TARGET_NAME ENABLE_AIE4)
    add_library(${TARGET_NAME} OBJECT
        ${TASK10_PRODUCTION_SOURCES})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${FASTFLOW_SOURCE_DIR}/include
        ${FASTFLOW_SOURCE_DIR}/runner
        ${FASTFLOW_SOURCE_DIR}/server
        ${FASTFLOW_SOURCE_DIR}/pull
        ${FASTFLOW_SOURCE_DIR}/../third_party/tokenizers-cpp/include
        ${TASK10_STUB_INCLUDE_DIR}
        ${BOOST_INCLUDE_DIR}
        ${XRT_INCLUDE_DIR})
    target_compile_definitions(${TARGET_NAME} PRIVATE
        DEV_BUILD=1
        __WINDOWS__
        USEAVX2=1
        DISABLE_ABI_CHECK=1
        _ENABLE_EXTENDED_ALIGNED_STORAGE
        CMAKE_INSTALL_PREFIX="${FASTFLOW_SOURCE_DIR}/build/phi4_corelib_aie4-tests"
        CMAKE_XCLBIN_PREFIX="${FASTFLOW_SOURCE_DIR}/xclbins"
        __FLM_VERSION__="task10-test"
        __NPU_VERSION__="0.0.0.0"
        WIN32_LEAN_AND_MEAN
        NOMINMAX)
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2 /fp:precise>)
    if(ENABLE_AIE4)
        target_compile_definitions(${TARGET_NAME} PRIVATE
            FLM_ENABLE_CORELIB_AIE4=1)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${RYZENAI_CORELIB_INCLUDE_DIR})
    endif()
endfunction()

add_task10_production_compile_check(
    task10_production_compile_off
    FALSE)
add_task10_production_compile_check(
    task10_production_compile_on
    TRUE)
