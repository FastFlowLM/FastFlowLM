cmake_minimum_required(VERSION 3.22)
project(phi4_corelib_aie4_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ryzenai-corelib is OPTIONAL here, and that is not a relaxation.
#
# This suite owns the guards that answer "does FastFlow still build with the
# AIE4 feature OFF" -- phi4_frontend_compile_off and task10_production_compile_off
# compile the real production TUs (main.cpp, rest_handler.cpp, runner.cpp,
# modeling_phi4.cpp) with FLM_ENABLE_CORELIB_AIE4 undefined. With REQUIRED here
# those targets could only be CONFIGURED on a machine that has corelib, i.e. the
# "builds without corelib" proof could never be run anywhere corelib was absent,
# which is the only place it means anything.
#
# So the search is allowed to fail, and the guards-only path below configures
# just those two targets and returns. Every target that genuinely needs corelib
# is either past that return or hard-fails by name inside the functions that add
# it -- the ON path is unchanged and still cannot be configured without corelib.
find_path(RYZENAI_CORELIB_INCLUDE_DIR
          NAMES ryzenai/corelib.h
          DOC "Directory containing ryzenai/corelib.h")
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

# ---------------------------------------------------------------------------
# The feature-OFF compile guards, and the guards-only early return
# ---------------------------------------------------------------------------
#
# These two functions and their feature-OFF invocations live HERE, ahead of
# everything that needs ryzenai-corelib, for one reason: a machine with no
# corelib must still be able to configure and build the proof that FastFlow
# compiles without it. Their feature-ON twins are invoked further down, past the
# return below, where corelib is known to exist.
#
# ENABLE_AIE4 is what makes a target need corelib, so that is where the hard
# failure lives -- naming the target rather than a header.

set(PHI4_FRONTEND_SOURCES
    ${FASTFLOW_SOURCE_DIR}/common/AutoModel/automodel.cpp
    ${FASTFLOW_SOURCE_DIR}/common/AutoModel/modeling_phi4.cpp
    ${FASTFLOW_SOURCE_DIR}/server/generation_limit.cpp)

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
        if(NOT RYZENAI_CORELIB_INCLUDE_DIR)
            message(FATAL_ERROR
                "${TARGET_NAME} compiles with FLM_ENABLE_CORELIB_AIE4=1 and "
                "therefore needs ryzenai/corelib.h, which was not found. Pass "
                "-DRYZENAI_CORELIB_INCLUDE_DIR=<dir> or add the corelib "
                "install prefix to CMAKE_PREFIX_PATH. Only the feature-OFF "
                "compile guards can be configured without it.")
        endif()
        target_compile_definitions(${TARGET_NAME} PRIVATE
            FLM_ENABLE_CORELIB_AIE4=1)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${RYZENAI_CORELIB_INCLUDE_DIR})
    endif()
endfunction()

add_phi4_frontend_compile_check(
    phi4_frontend_compile_off
    FALSE)

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
        if(NOT RYZENAI_CORELIB_INCLUDE_DIR)
            message(FATAL_ERROR
                "${TARGET_NAME} compiles with FLM_ENABLE_CORELIB_AIE4=1 and "
                "therefore needs ryzenai/corelib.h, which was not found. Pass "
                "-DRYZENAI_CORELIB_INCLUDE_DIR=<dir> or add the corelib "
                "install prefix to CMAKE_PREFIX_PATH. Only the feature-OFF "
                "compile guards can be configured without it.")
        endif()
        target_compile_definitions(${TARGET_NAME} PRIVATE
            FLM_ENABLE_CORELIB_AIE4=1)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${RYZENAI_CORELIB_INCLUDE_DIR})
    endif()
endfunction()

add_task10_production_compile_check(
    task10_production_compile_off
    FALSE)

# The offline acceptance guards, run by `ctest` rather than by remembering.
#
# verify_acceptance_guards.ps1 carries 72 assertions -- the harness's guard
# bodies driven against the committed record, the committed acceptance document
# re-rendered from that record by the harness's own render block, and an AST
# lint of the harness's call sites. Until now NOTHING invoked it: not a
# CMakeLists, not a .cmake, not a workflow. An entire verification tier that
# runs only when a human remembers is a verification tier that reports on runs
# it did not have.
#
# It needs no hardware, no corelib, no model and no flm.exe -- only PowerShell
# and the committed record -- so it is registered BEFORE the corelib-less
# early return below, and it is the one test in this suite that a machine
# without an AIE4 device can run in full. `enable_testing()` is called here
# rather than 60 lines down for the same reason.
enable_testing()
find_program(FLM_POWERSHELL_EXECUTABLE
             NAMES pwsh powershell
             DOC "PowerShell used to run the packaging and guard tests")
if(FLM_POWERSHELL_EXECUTABLE)
    add_test(NAME test_acceptance_guards
             COMMAND "${FLM_POWERSHELL_EXECUTABLE}"
                     -NoProfile
                     -ExecutionPolicy Bypass
                     -File
                     "${CMAKE_CURRENT_LIST_DIR}/verify_acceptance_guards.ps1")
    # No SKIP_RETURN_CODE. There is no input it can be missing: if the
    # committed record is not there, that is a failure and not a skip.
    set_tests_properties(test_acceptance_guards PROPERTIES TIMEOUT 900)
else()
    message(WARNING
        "PowerShell was not found, so test_acceptance_guards is not "
        "registered. The offline acceptance guards will not run in this "
        "build.")
endif()

if(NOT RYZENAI_CORELIB_INCLUDE_DIR)
    # Guards-only configure. Everything below this point either includes
    # ryzenai/corelib.h or links something that does, so stopping here is the
    # difference between "the without-corelib guard cannot be configured" and
    # "the without-corelib guard is the only thing configured". Say which
    # targets exist, so a green configure is not mistaken for the full suite.
    message(STATUS
        "ryzenai/corelib.h not found: configuring the feature-OFF compile "
        "guards only (phi4_frontend_compile_off, task10_production_compile_off). "
        "The corelib tests, the hardware suite and the feature-ON guards are "
        "NOT configured. Pass -DRYZENAI_CORELIB_INCLUDE_DIR=<dir> for the full "
        "suite.")
    return()
endif()

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

# Task 13's performance and memory baseline. Like test_phi4_e2e it is NOT
# registered with add_test: it needs a model directory, a token plan and two
# revision identifiers that only the caller can supply, and design section 4
# makes its output a record rather than a gate, so a ctest entry would either
# fail everywhere for want of a flag or pass without measuring anything.
# run_hardware_suite.ps1 owns the invocation.
add_executable(benchmark_phi4_aie4 benchmark_phi4_aie4.cpp)
target_link_libraries(benchmark_phi4_aie4 PRIVATE
    flm_corelib_aie4_testlib
    shell32
    ole32
    advapi32
    # Step 5: GetProcessMemoryInfo with PROCESS_MEMORY_COUNTERS_EX.
    psapi
    # The NPU SKU and driver version, asked of the OS device tree rather than
    # transcribed from a session transcript.
    setupapi
    # GetFileVersionInfo, for the DynamicDispatch / RyzenMM / XRT identities.
    version)
target_include_directories(benchmark_phi4_aie4 PRIVATE
    ${FASTFLOW_SOURCE_DIR}/pull)
target_compile_definitions(benchmark_phi4_aie4 PRIVATE
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
#
# FLM_POWERSHELL_EXECUTABLE is found near the top of this file, because
# test_acceptance_guards needs it on the corelib-less path too.

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

# The feature-ON halves of the two compile guards. Their feature-OFF twins, and
# the functions themselves, are defined near the top of this file so that they
# survive the guards-only return -- see the corelib find_path comment there.
add_phi4_frontend_compile_check(
    phi4_frontend_compile_on
    TRUE)

add_task10_production_compile_check(
    task10_production_compile_on
    TRUE)

# ---------------------------------------------------------------------------
# Task 14: the continuation-threshold calibrator, and the artifact-drift guard
# ---------------------------------------------------------------------------
#
# `kContinuationAppendThreshold` in phi4_corelib_aie4_tuning.hpp is GENERATED
# from docs/docs/benchmarks/phi4_aie4_baseline.json. Nothing in the C++ build
# can notice if the header, the published benchmark document and the
# measurement drift apart -- the header would still compile, the tests would
# still pass, and the shipped constant would no longer correspond to any
# measurement. `--check` is the only guard against that, so it has to be
# reachable from a runner rather than only from a human who remembers.
#
# The unittest module registered here runs the calibrator's own suite AND, in
# CheckedInArtefactsTest, invokes the calibrator with `--check` against the
# committed header, baseline, crossover history and benchmark document.
#
# FLM_PYTHON_EXECUTABLE may be set explicitly, and run_hardware_suite.ps1 does
# set it, for the same reason it passes -CmakeExe: on the AIE4 target PATH
# resolves `python3` to C:/cygwin/bin/python3, a Cygwin symlink that Windows
# cannot execute at all. A found-but-unusable interpreter is worse than none,
# because it registers a test that can never pass for a reason having nothing
# to do with what the test checks -- which is exactly what happened on the
# first run of this block.
if(NOT FLM_PYTHON_EXECUTABLE)
    find_program(FLM_PYTHON_EXECUTABLE
                 NAMES python3 python
                 DOC "Python used to run the offline tooling tests")
endif()

# Whatever was found or supplied, prove it RUNS before trusting it. `-E` and a
# trivial expression rather than `--version`, so a wrapper that prints a banner
# and exits zero without an interpreter behind it still fails here.
set(FLM_PYTHON_USABLE FALSE)
if(FLM_PYTHON_EXECUTABLE)
    execute_process(
        COMMAND "${FLM_PYTHON_EXECUTABLE}" -c "import sys; sys.exit(0)"
        RESULT_VARIABLE _flm_python_probe
        OUTPUT_QUIET
        ERROR_QUIET)
    if(_flm_python_probe EQUAL 0)
        set(FLM_PYTHON_USABLE TRUE)
    else()
        message(WARNING
            "FLM_PYTHON_EXECUTABLE=${FLM_PYTHON_EXECUTABLE} does not run "
            "(exit ${_flm_python_probe}); ignoring it. Pass a working "
            "interpreter with -DFLM_PYTHON_EXECUTABLE=<path>.")
    endif()
endif()

if(FLM_PYTHON_USABLE)
    # WORKING_DIRECTORY is the repository root because the tests import
    # `tools.calibrate_phi4_corelib_continuation` as a package and resolve the
    # committed artifacts relative to it.
    add_test(NAME test_phi4_continuation_calibration
             COMMAND "${FLM_PYTHON_EXECUTABLE}"
                     -m unittest
                     tools.tests.test_calibrate_phi4_corelib_continuation
                     -v)
    set_tests_properties(test_phi4_continuation_calibration PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../../.."
        TIMEOUT 300)
else()
    # Not silently omitted. Registering a test that FAILS is the honest
    # outcome: without Python this build cannot check that the shipped
    # threshold still matches the measurement it claims to come from, and a
    # green suite would say it did. Install Python or accept a red test.
    add_test(NAME test_phi4_continuation_calibration
             COMMAND "${CMAKE_COMMAND}" -E false)
    set_tests_properties(test_phi4_continuation_calibration PROPERTIES
        TIMEOUT 60)
    message(WARNING
        "No usable Python. test_phi4_continuation_calibration is registered "
        "as a FAILING test: the generated kContinuationAppendThreshold cannot "
        "be checked against the measurement it was derived from. Pass "
        "-DFLM_PYTHON_EXECUTABLE=<path> to fix it.")
endif()
