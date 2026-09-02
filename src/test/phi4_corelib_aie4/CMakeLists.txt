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

# FLM_VERSION and NPU_VERSION are compiled into several tests, and this file
# never declared either of them.
#
# That is not a harmless omission. test_model_catalog asserts
# `__FLM_VERSION__ >= the version packaged in the AIE4 overlay`, so an empty
# string makes its sscanf return EOF and the test fail. It went unnoticed
# because the one machine the suite had ever been configured on carried
# `FLM_VERSION:UNINITIALIZED=1.0.4` in its cache from a hand-typed -D, so the
# suite passed there and failed on the first fresh configure anywhere else --
# which is exactly what happened on the AIE4 target.
#
# The values are read from the product's own CMakePresets.json rather than
# retyped here, so the standalone suite cannot drift away from the version the
# product actually ships.
if(NOT DEFINED FLM_VERSION OR NOT DEFINED NPU_VERSION)
    set(_flm_presets "${FASTFLOW_SOURCE_DIR}/CMakePresets.json")
    if(NOT EXISTS "${_flm_presets}")
        message(FATAL_ERROR
            "Cannot default FLM_VERSION/NPU_VERSION: ${_flm_presets} is "
            "missing. Pass -DFLM_VERSION= and -DNPU_VERSION= explicitly.")
    endif()
    file(READ "${_flm_presets}" _flm_presets_text)
    string(JSON _flm_common GET "${_flm_presets_text}"
           "configurePresets" 0 "cacheVariables")
    if(NOT DEFINED FLM_VERSION)
        string(JSON FLM_VERSION GET "${_flm_common}" "FLM_VERSION")
    endif()
    if(NOT DEFINED NPU_VERSION)
        string(JSON NPU_VERSION GET "${_flm_common}" "NPU_VERSION")
    endif()
    unset(_flm_presets_text)
    unset(_flm_common)
endif()
# Validated rather than trusted: an empty or malformed version reaches the
# tests as a compile definition and fails them at run time with a message that
# says nothing about where the value came from.
if(NOT FLM_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR
        "FLM_VERSION must be MAJOR.MINOR.PATCH, got '${FLM_VERSION}'")
endif()
if(NOT NPU_VERSION MATCHES "^[0-9]+(\\.[0-9]+)+$")
    message(FATAL_ERROR
        "NPU_VERSION must be dotted numeric, got '${NPU_VERSION}'")
endif()
message(STATUS
    "Phi-4 AIE4 test suite: FLM_VERSION=${FLM_VERSION} "
    "NPU_VERSION=${NPU_VERSION}")

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
    FLM_TEST_SOURCE_DIR="${FASTFLOW_SOURCE_DIR}"
    __FLM_VERSION__="${FLM_VERSION}")
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
target_compile_definitions(test_generation_limit PRIVATE
    FLM_TEST_SOURCE_DIR="${FASTFLOW_SOURCE_DIR}")
add_test(NAME test_generation_limit COMMAND test_generation_limit)

# Validates FastFlow against the REAL corelib rather than the fake.
#
# With no runtime directory it reports SKIPPED to CTest via
# SKIP_RETURN_CODE, NOT Passed: this is the suite's highest-value check and
# a green-but-inert result would read as coverage it does not have. A
# configured-but-missing DLL is still a hard failure, so a stale or absent
# runtime cannot hide behind a green suite either.
set(RYZENAI_CORELIB_RUNTIME_DIR
    ""
    CACHE PATH
    "Directory holding the real ryzenai_corelib.dll and its closure")
set(RYZENAI_CORELIB_EXTRA_DLL_DIRS
    ""
    CACHE STRING
    "Semicolon-separated directories completing corelib's DLL closure")
string(REPLACE ";" "\\;" RYZENAI_CORELIB_EXTRA_DLL_DIRS_ESCAPED
    "${RYZENAI_CORELIB_EXTRA_DLL_DIRS}")
add_corelib_host_test(test_real_corelib test_real_corelib.cpp)
target_compile_definitions(test_real_corelib PRIVATE
    FLM_REAL_CORELIB_RUNTIME_DIR="${RYZENAI_CORELIB_RUNTIME_DIR}"
    FLM_REAL_CORELIB_EXTRA_DLL_DIRS="${RYZENAI_CORELIB_EXTRA_DLL_DIRS_ESCAPED}")
set_tests_properties(test_real_corelib PROPERTIES SKIP_RETURN_CODE 77)
if(RYZENAI_CORELIB_RUNTIME_DIR)
    set_tests_properties(test_real_corelib PROPERTIES
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${XRT_LIB_DIR};PATH=path_list_prepend:${RYZENAI_CORELIB_RUNTIME_DIR}")
endif()

# The AIE4 hardware suite. These need a real NPU as well as a real DLL, so
# unlike test_real_corelib they cannot run anywhere but the target -- and like
# it, they report SKIPPED rather than Passed when the runtime directory is
# unset, because a green-and-inert hardware check reads as coverage it does not
# have.
function(add_corelib_hardware_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE
        flm_corelib_aie4_testlib
        shell32
        ole32
        advapi32)
    target_compile_definitions(${TEST_NAME} PRIVATE
        FLM_REAL_CORELIB_RUNTIME_DIR="${RYZENAI_CORELIB_RUNTIME_DIR}"
        FLM_REAL_CORELIB_EXTRA_DLL_DIRS="${RYZENAI_CORELIB_EXTRA_DLL_DIRS_ESCAPED}")
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES
        SKIP_RETURN_CODE 77
        # A hardware run loads a 3 GB model and packs its weights, and two
        # AIE4 contexts at once fail in ways that look like defects. RUN_
        # SERIAL keeps ctest from overlapping them.
        RUN_SERIAL ON
        TIMEOUT 3600)
    if(RYZENAI_CORELIB_RUNTIME_DIR)
        set_tests_properties(${TEST_NAME} PROPERTIES
            ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:${XRT_LIB_DIR};PATH=path_list_prepend:${RYZENAI_CORELIB_RUNTIME_DIR}")
    endif()
endfunction()

add_corelib_hardware_test(test_phi4_hardware test_phi4_hardware.cpp)
add_corelib_hardware_test(test_fatal_child test_fatal_child.cpp)

# The end-to-end harness is NOT registered with add_test. It requires a model
# directory and explicit token IDs, and the reference comparator must run to
# completion before it starts; run_hardware_suite.ps1 owns that sequencing.
# Registering it here with no arguments would produce a test that fails for
# want of a flag on every machine, which is worse than one that is absent.
add_executable(test_phi4_e2e test_phi4_e2e.cpp)
target_link_libraries(test_phi4_e2e PRIVATE
    flm_corelib_aie4_testlib
    shell32
    ole32
    advapi32)
# picosha2.h, for hashing the corelib the loader actually handed us. The same
# header backs flm::pull::CalculateFileSha256, so the artifact's hash is
# comparable with the ones the model catalog records.
target_include_directories(test_phi4_e2e PRIVATE
    ${FASTFLOW_SOURCE_DIR}/pull)
target_compile_definitions(test_phi4_e2e PRIVATE
    FLM_REAL_CORELIB_RUNTIME_DIR="${RYZENAI_CORELIB_RUNTIME_DIR}"
    FLM_REAL_CORELIB_EXTRA_DLL_DIRS="${RYZENAI_CORELIB_EXTRA_DLL_DIRS_ESCAPED}")

# The packaging and clean-environment checks run by default rather than only
# when someone remembers the command. Every regression guard in that script --
# the derived closure, the CLOSURE-2 negative control, the no-corelib-import
# rule, and the requirement that the ordinary NPU2 installer still builds
# without an AIE4 closure -- is invisible if it is never invoked.
#
# It reports SKIPPED via SKIP_RETURN_CODE when its optional inputs are absent,
# so a partial run never reads as full coverage.
find_program(FLM_POWERSHELL_EXECUTABLE
             NAMES pwsh powershell
             DOC "PowerShell used to run the packaging tests")

# Builds the argument vector for test_packaged_runtime.ps1.
#
# `-DependencyDir` takes one argument holding a semicolon-separated list, so
# the separator must be escaped before it is appended. `list(JOIN ... ";" ...)`
# followed by `list(APPEND)` does NOT achieve that: the join produces a string
# that append immediately re-splits, yielding one argv entry per directory.
#
# The consequence in PowerShell is worse than a lost directory. `-DependencyDir`
# binds only the first entry, and each orphan then binds POSITIONALLY -- the
# next positional parameter is `$FlmExe` -- so a second dependency directory
# silently becomes the path to the executable under test, turning an honest
# SKIPPED into a spurious flm.exe run against a directory.
function(flm_packaged_runtime_args output corelib_dir dependency_dirs)
    set(_args "")
    if(corelib_dir)
        list(APPEND _args -CorelibRuntimeDir "${corelib_dir}")
    endif()
    if(dependency_dirs)
        string(REPLACE ";" "\\;" _escaped "${dependency_dirs}")
        list(APPEND _args -DependencyDir "${_escaped}")
    endif()
    set(${output} "${_args}" PARENT_SCOPE)
endfunction()

# Self-check on the escaping, run at configure time against a synthetic
# multi-entry value so it does not depend on how this machine's cache happens
# to be populated. A single-entry cache masks the defect completely, which is
# how it survived the previous round.
flm_packaged_runtime_args(_flm_argv_probe "C:/core" "C:/one;C:/two")
list(LENGTH _flm_argv_probe _flm_argv_probe_length)
if(NOT _flm_argv_probe_length EQUAL 4)
    message(FATAL_ERROR
        "test_packaged_runtime argument construction flattened a multi-entry "
        "dependency list into ${_flm_argv_probe_length} arguments instead of "
        "4. The extra entries bind positionally in PowerShell, so -FlmExe "
        "would receive a directory: [${_flm_argv_probe}]")
endif()
list(GET _flm_argv_probe 3 _flm_argv_probe_deps)
if(NOT _flm_argv_probe_deps STREQUAL "C:/one;C:/two")
    message(FATAL_ERROR
        "test_packaged_runtime dropped a dependency directory: "
        "[${_flm_argv_probe_deps}]")
endif()

if(FLM_POWERSHELL_EXECUTABLE)
    flm_packaged_runtime_args(_packaged_runtime_args
        "${RYZENAI_CORELIB_RUNTIME_DIR}"
        "${RYZENAI_CORELIB_EXTRA_DLL_DIRS}")
    add_test(NAME test_packaged_runtime
             COMMAND "${FLM_POWERSHELL_EXECUTABLE}"
                     -NoProfile
                     -ExecutionPolicy Bypass
                     -File
                     "${CMAKE_CURRENT_LIST_DIR}/test_packaged_runtime.ps1"
                     # The CMake that configured this suite, not whatever
                     # `cmake` PATH happens to find. On the AIE4 target PATH
                     # finds Cygwin's 3.14.5, which cannot run
                     # StageAie4Runtime.cmake's required 3.24 -- so the
                     # closure guard failed for a reason unrelated to the
                     # closure.
                     -CmakeExe "${CMAKE_COMMAND}"
                     ${_packaged_runtime_args})
    set_tests_properties(test_packaged_runtime PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1800)
else()
    message(WARNING
        "PowerShell was not found, so test_packaged_runtime is not "
        "registered. The AIE4 packaging and clean-environment guards will "
        "not run in this build.")
endif()

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
