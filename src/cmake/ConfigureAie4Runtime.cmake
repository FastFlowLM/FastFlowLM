# Packaging support for the optional Phi-4 AIE4 corelib runtime.
#
# The corelib DLL is resolved at run time by absolute path and `flm.exe` never
# links `ryzenai_corelib.lib`, so building the product with
# `FLM_ENABLE_CORELIB_AIE4=ON` needs the corelib *include* directory and
# nothing else. The runtime directory is therefore a packaging input, required
# by the install/package step and never by a feature-ON configure. Making it a
# configure-time requirement would contradict calling it packaging-only and
# would break the ordinary developer build.
#
# The file list itself is derived, never transcribed: see StageAie4Runtime.cmake
# and design `CLOSURE-1`.

set(
    RYZENAI_CORELIB_RUNTIME_DIR
    ""
    CACHE PATH
    "Packaging-only directory containing the ryzenai-corelib runtime DLLs")
set(
    XRT_RUNTIME_DIR
    ""
    CACHE PATH
    "Packaging-only directory containing stageable XRT runtime DLLs")
set(
    FLM_AIE4_DEPENDENCY_DIRS
    ""
    CACHE STRING
    "Additional directories searched for the optional AIE4 DLL closure")

set(FLM_AIE4_STAGE_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/StageAie4Runtime.cmake")

function(flm_aie4_runtime_configured output)
    if(FLM_ENABLE_CORELIB_AIE4
       AND RYZENAI_CORELIB_RUNTIME_DIR
       AND EXISTS "${RYZENAI_CORELIB_RUNTIME_DIR}/ryzenai_corelib.dll")
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Warns at configure time, at most. A developer who only wants to compile the
# AIE4 code paths should not be stopped here; the install step is where the
# missing input actually matters, and that is where it fails.
function(flm_aie4_warn_if_unstageable)
    if(NOT FLM_ENABLE_CORELIB_AIE4)
        return()
    endif()
    flm_aie4_runtime_configured(_configured)
    if(NOT _configured)
        message(WARNING
            "FLM_ENABLE_CORELIB_AIE4=ON without a usable "
            "RYZENAI_CORELIB_RUNTIME_DIR. flm.exe will still build, because "
            "it resolves ryzenai_corelib.dll at run time by absolute path. "
            "Installing or packaging the AIE4 feature will fail until "
            "RYZENAI_CORELIB_RUNTIME_DIR points at the directory holding the "
            "ryzenai_corelib.dll you intend to ship.")
    endif()
endfunction()

function(flm_aie4_stage_command output stage_dir report audit)
    # FLM_AIE4_DEPENDENCY_DIRS is a CMake list. Interpolating it into a
    # command argument unescaped turns each `;` into an argument separator, so
    # `-DFLM_AIE4_EXTRA_DIRS=a;b` reaches cmake as two argv entries: the
    # variable silently loses everything after the first directory, and the
    # staging then fails telling the developer to add a directory they already
    # added. Escaping keeps the whole list in one argument.
    string(REPLACE ";" "\\;" _extra_dirs "${FLM_AIE4_DEPENDENCY_DIRS}")
    set(${output}
        "${CMAKE_COMMAND}"
        "-DFLM_AIE4_CORELIB_DIR=${RYZENAI_CORELIB_RUNTIME_DIR}"
        "-DFLM_AIE4_XRT_DIR=${XRT_RUNTIME_DIR}"
        "-DFLM_AIE4_EXTRA_DIRS=${_extra_dirs}"
        "-DFLM_AIE4_DESTINATION=${stage_dir}"
        "-DFLM_AIE4_REPORT=${report}"
        "-DFLM_AIE4_AUDIT=${audit}"
        -P "${FLM_AIE4_STAGE_SCRIPT}"
        PARENT_SCOPE)
endfunction()

# Stages the derived closure beside a just-built binary so a developer can run
# the AIE4 path without an install. Skipped, with no error, when the runtime
# directory is not configured.
#
# The staged directory is what the installer scripts copy, so it gets the
# shippable report. The audit record, which names build-machine absolute
# paths, stays in the build tree.
function(flm_aie4_stage_for_target target)
    if(NOT FLM_ENABLE_CORELIB_AIE4 OR NOT WIN32)
        return()
    endif()
    flm_aie4_runtime_configured(_configured)
    if(NOT _configured)
        return()
    endif()
    foreach(_stage_dir IN LISTS ARGN)
        string(MD5 _stage_id "${_stage_dir}")
        flm_aie4_stage_command(_command
            "${_stage_dir}"
            "${_stage_dir}/aie4-closure.txt"
            "${CMAKE_BINARY_DIR}/aie4-closure-audit-${_stage_id}.txt")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${_command}
            COMMENT "Deriving the Phi-4 AIE4 runtime closure")
    endforeach()
endfunction()

# Emits the install rule. The derivation runs at install time, against the
# ryzenai_corelib.dll actually being shipped, so the packaged closure can never
# be a stale list captured when the project was configured.
function(flm_aie4_install_runtime)
    cmake_parse_arguments(_arg "" "DESTINATION;COMPONENT" "" ${ARGN})
    if(NOT FLM_ENABLE_CORELIB_AIE4)
        return()
    endif()
    if(NOT WIN32)
        message(FATAL_ERROR
            "FLM_ENABLE_CORELIB_AIE4 runtime packaging is Windows-only")
    endif()
    set(_component_args "")
    if(_arg_COMPONENT)
        set(_component_args COMPONENT ${_arg_COMPONENT})
    endif()
    install(CODE "
set(FLM_AIE4_CORELIB_DIR [[${RYZENAI_CORELIB_RUNTIME_DIR}]])
set(FLM_AIE4_XRT_DIR [[${XRT_RUNTIME_DIR}]])
set(FLM_AIE4_EXTRA_DIRS [[${FLM_AIE4_DEPENDENCY_DIRS}]])
set(FLM_AIE4_DESTINATION [[${_arg_DESTINATION}]])
set(FLM_AIE4_REPORT
    \"\${CMAKE_INSTALL_PREFIX}/${_arg_DESTINATION}/aie4-closure.txt\")
set(FLM_AIE4_AUDIT [[${CMAKE_BINARY_DIR}/aie4-closure-audit-install.txt]])
"
        ${_component_args})
    install(SCRIPT "${FLM_AIE4_STAGE_SCRIPT}" ${_component_args})
endfunction()
