# Derives and stages the optional Phi-4 AIE4 corelib runtime closure.
#
# Design `CLOSURE-1`: the closure is defined by what the shipped
# `ryzenai_corelib.dll` actually imports, enumerated with a dependency walker
# against that exact binary. It is never transcribed, because different
# DynamicDispatch linkages and different dependency builds import different
# sets: the 223 MB dev-box binary statically links DynamicDispatch while the
# 0.8 MB target binary loads it as separate DLLs, and neither closure
# validates the other.
#
# This file is a standalone script. It runs both under `cmake -P` (developer
# staging beside `flm.exe`) and under `install(SCRIPT)` (packaging), so the
# packaged closure and the closure a developer runs against are produced by
# the same derivation rather than two lists that can drift apart.
#
# Inputs:
#   FLM_AIE4_CORELIB_DIR   directory holding ryzenai_corelib.dll (required)
#   FLM_AIE4_XRT_DIR       directory holding stageable XRT DLLs (optional)
#   FLM_AIE4_EXTRA_DIRS    additional dependency search directories
#   FLM_AIE4_DESTINATION   directory to stage into; relative paths resolve
#                          against CMAKE_INSTALL_PREFIX
#   FLM_AIE4_REPORT        optional path for the SHIPPABLE closure report:
#                          file names and hashes only, no build-machine paths
#   FLM_AIE4_AUDIT         optional path for the build-side audit record, which
#                          does name absolute source paths and must stay in the
#                          build tree

cmake_minimum_required(VERSION 3.24)

if(NOT WIN32 AND NOT CMAKE_HOST_WIN32)
    message(FATAL_ERROR
        "The Phi-4 AIE4 runtime closure is Windows-only.")
endif()

if(NOT FLM_AIE4_CORELIB_DIR)
    message(FATAL_ERROR
        "RYZENAI_CORELIB_RUNTIME_DIR is required to install or package the "
        "Phi-4 AIE4 feature (FLM_ENABLE_CORELIB_AIE4=ON). Point it at the "
        "directory holding the ryzenai_corelib.dll you intend to ship, then "
        "re-run the install step. Building flm.exe does not need it: the "
        "corelib DLL is resolved at runtime by absolute path and flm.exe "
        "never links ryzenai_corelib.lib.")
endif()

if(NOT IS_DIRECTORY "${FLM_AIE4_CORELIB_DIR}")
    message(FATAL_ERROR
        "RYZENAI_CORELIB_RUNTIME_DIR does not name a directory: "
        "${FLM_AIE4_CORELIB_DIR}")
endif()

set(_flm_aie4_root "${FLM_AIE4_CORELIB_DIR}/ryzenai_corelib.dll")
if(NOT EXISTS "${_flm_aie4_root}")
    message(FATAL_ERROR
        "RYZENAI_CORELIB_RUNTIME_DIR contains no ryzenai_corelib.dll: "
        "${FLM_AIE4_CORELIB_DIR}")
endif()

# Search order matters. Directories supplied for this package win over
# anything the machine happens to provide, so a build box with an ambient
# conda or toolchain prefix stages the DLLs we chose rather than the ones it
# stumbled across.
set(_flm_aie4_search_dirs "${FLM_AIE4_CORELIB_DIR}")
if(FLM_AIE4_XRT_DIR AND IS_DIRECTORY "${FLM_AIE4_XRT_DIR}")
    list(APPEND _flm_aie4_search_dirs "${FLM_AIE4_XRT_DIR}")
elseif(FLM_AIE4_XRT_DIR)
    message(FATAL_ERROR
        "XRT_RUNTIME_DIR does not name a directory: ${FLM_AIE4_XRT_DIR}")
endif()
foreach(_flm_aie4_dir IN LISTS FLM_AIE4_EXTRA_DIRS)
    if(_flm_aie4_dir)
        if(NOT IS_DIRECTORY "${_flm_aie4_dir}")
            message(FATAL_ERROR
                "FLM_AIE4_DEPENDENCY_DIRS entry is not a directory: "
                "${_flm_aie4_dir}")
        endif()
        list(APPEND _flm_aie4_search_dirs "${_flm_aie4_dir}")
    endif()
endforeach()
list(REMOVE_DUPLICATES _flm_aie4_search_dirs)

# `dyn_bins.dll` holds DynamicDispatch's precompiled binaries and is opened by
# NAME at runtime, so it is never an import and a dependency walker cannot see
# it. It has to be found by presence, and WHERE it lives depends on the
# DynamicDispatch linkage:
#
#   * statically linked DD -- the dev box -- puts it beside
#     ryzenai_corelib.dll, because Transaction resolves it against the
#     directory of the module that linked DD in;
#   * shared DD -- the AIE4 target -- puts it beside dyn_dispatch_core.dll.
#
# Searching only the corelib directory is therefore correct on one box and
# silently wrong on the other. It is silent because nothing fails to load: the
# process starts, and every shape query then comes back "Shape list size: 0",
# which surfaces as `matmul_bf16_weights_create_onnx failed ... not supported
# in this supported shape list` at weight-packing time. That is a long way from
# the missing file, which is why this searches every directory the closure is
# allowed to draw from rather than assuming a linkage.
set(_flm_aie4_runtime_loaded "")
foreach(_flm_aie4_dir IN LISTS _flm_aie4_search_dirs)
    if(EXISTS "${_flm_aie4_dir}/dyn_bins.dll")
        list(APPEND _flm_aie4_runtime_loaded
            "${_flm_aie4_dir}/dyn_bins.dll")
        break()
    endif()
endforeach()
if(NOT _flm_aie4_runtime_loaded)
    message(STATUS
        "No dyn_bins.dll in any search directory. That is expected only for "
        "a DynamicDispatch build that embeds its binaries; if the staged "
        "runtime later reports \"Shape list size: 0\", this is why.")
endif()

# The Visual C++ runtime is deliberately not staged. `flm.exe` itself imports
# MSVCP140/VCRUNTIME140, so the redistributable is already a product-wide
# prerequisite and the AIE4 feature adds no new one. Copying a build machine's
# conda or toolchain copy beside ryzenai_corelib.dll would ship a second,
# possibly older, runtime next to the one the rest of the process already
# loaded.
set(_flm_aie4_pre_exclude
    "^api-ms-win-.*"
    "^ext-ms-.*"
    "^[Mm][Ss][Vv][Cc][Pp]1[0-9]+.*\\.dll$"
    "^[Vv][Cc][Rr][Uu][Nn][Tt][Ii][Mm][Ee]1[0-9]+.*\\.dll$"
    "^[Cc][Oo][Nn][Cc][Rr][Tt]1[0-9]+.*\\.dll$")
set(_flm_aie4_post_exclude
    "^[A-Za-z]:[\\\\/][Ww][Ii][Nn][Dd][Oo][Ww][Ss][\\\\/].*"
    "^.*[\\\\/][Ss][Yy][Ss][Tt][Ee][Mm]32[\\\\/].*"
    "^.*[\\\\/][Ss][Yy][Ss][Ww][Oo][Ww]64[\\\\/].*")

file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES
        "${_flm_aie4_root}"
        ${_flm_aie4_runtime_loaded}
    RESOLVED_DEPENDENCIES_VAR _flm_aie4_resolved
    UNRESOLVED_DEPENDENCIES_VAR _flm_aie4_unresolved
    CONFLICTING_DEPENDENCIES_PREFIX _flm_aie4_conflicting
    DIRECTORIES ${_flm_aie4_search_dirs}
    PRE_EXCLUDE_REGEXES ${_flm_aie4_pre_exclude}
    POST_EXCLUDE_REGEXES ${_flm_aie4_post_exclude})

if(_flm_aie4_unresolved)
    list(JOIN _flm_aie4_unresolved "\n  " _flm_aie4_unresolved_text)
    message(FATAL_ERROR
        "The Phi-4 AIE4 runtime closure is incomplete. "
        "${_flm_aie4_root} imports DLLs that were not found in "
        "RYZENAI_CORELIB_RUNTIME_DIR, XRT_RUNTIME_DIR, "
        "FLM_AIE4_DEPENDENCY_DIRS, or the approved system directories:\n"
        "  ${_flm_aie4_unresolved_text}\n"
        "Add the directory that provides them to FLM_AIE4_DEPENDENCY_DIRS. "
        "Do not rely on them being on PATH: a closure that only loads "
        "because a build machine had a conda or toolchain prefix on PATH "
        "fails on the target with Win32 error 126.")
endif()

if(_flm_aie4_conflicting_FILENAMES)
    list(JOIN _flm_aie4_conflicting_FILENAMES ", " _flm_aie4_conflict_text)
    message(FATAL_ERROR
        "The Phi-4 AIE4 runtime closure resolved conflicting copies of: "
        "${_flm_aie4_conflict_text}. Narrow the search directories so each "
        "DLL has one unambiguous source.")
endif()

set(_flm_aie4_files ${_flm_aie4_root} ${_flm_aie4_runtime_loaded})
list(APPEND _flm_aie4_files ${_flm_aie4_resolved})
list(REMOVE_DUPLICATES _flm_aie4_files)
list(SORT _flm_aie4_files)

set(_flm_aie4_destination "${FLM_AIE4_DESTINATION}")
if(NOT _flm_aie4_destination)
    message(FATAL_ERROR "FLM_AIE4_DESTINATION was not set")
endif()
if(NOT IS_ABSOLUTE "${_flm_aie4_destination}")
    set(_flm_aie4_destination
        "${CMAKE_INSTALL_PREFIX}/${_flm_aie4_destination}")
endif()

file(MAKE_DIRECTORY "${_flm_aie4_destination}")
foreach(_flm_aie4_file IN LISTS _flm_aie4_files)
    message(STATUS "Staging AIE4 runtime: ${_flm_aie4_file}")
    file(COPY "${_flm_aie4_file}"
         DESTINATION "${_flm_aie4_destination}"
         FOLLOW_SYMLINK_CHAIN)
endforeach()

# Two records, deliberately different.
#
# The shippable report lists the derived closure by name and SHA-256 only. It
# is written into the staged directory, which the installer scripts copy
# verbatim, so it must not carry absolute paths from the machine that built it:
# a customer artifact naming a developer's conda prefix leaks build-machine
# layout for no benefit to the reader. The hashes are what a recipient can
# actually act on, since they verify the staged bits.
#
# It also carries the identity of the corelib itself, ahead of the file list.
# The first question asked about a wrong number in the field is "which runtime
# produced it", and until this project put the resolved DLL's SHA-256 into its
# own test artifacts, nothing could answer it -- an investigation ran for days
# on the assumption that two runs had loaded the same library, with no record
# either way. A shipped artifact needs the same answer available to a reader
# who has only the installed directory.
#
# Deliberately bounded: corelib version and corelib hash, nothing else. The
# FastFlow commit, the model revision and the driver identity live in the
# baseline identity block and the acceptance record. Copying them here would
# create a second source that drifts.
if(FLM_AIE4_REPORT)
    # Probed against the STAGED copy, not the source copy. The staged file is
    # the one that ships, and it is the one whose closure sits beside it -- so
    # a load failure here is a real statement about the artifact rather than
    # about the build machine's PATH.
    set(_flm_aie4_staged_root "${_flm_aie4_destination}/ryzenai_corelib.dll")
    if(NOT EXISTS "${_flm_aie4_staged_root}")
        message(FATAL_ERROR
            "Staging did not produce ${_flm_aie4_staged_root}")
    endif()
    get_filename_component(_flm_aie4_probe
        "${CMAKE_CURRENT_LIST_DIR}/ReadCorelibVersion.ps1" ABSOLUTE)
    if(NOT EXISTS "${_flm_aie4_probe}")
        message(FATAL_ERROR
            "The corelib version probe is missing: ${_flm_aie4_probe}")
    endif()
    execute_process(
        COMMAND powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass
                -File "${_flm_aie4_probe}"
                -Dll "${_flm_aie4_staged_root}"
        OUTPUT_VARIABLE _flm_aie4_version
        ERROR_VARIABLE _flm_aie4_version_error
        RESULT_VARIABLE _flm_aie4_version_status
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Hard failure, not a placeholder.
    #
    # The only way this probe fails is that the staged closure cannot be
    # loaded or does not export the entry point -- which means the feature
    # being packaged does not work. Writing "unknown" into the report and
    # carrying on would ship a broken runtime under a record that reads like a
    # successful one, which is the exact failure this project keeps finding.
    if(NOT _flm_aie4_version_status EQUAL 0
       OR NOT _flm_aie4_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
        message(FATAL_ERROR
            "Could not read ryzenai_corelib_get_version from the staged "
            "${_flm_aie4_staged_root}.\n"
            "  status: ${_flm_aie4_version_status}\n"
            "  stdout: ${_flm_aie4_version}\n"
            "  stderr: ${_flm_aie4_version_error}\n"
            "The staged closure must load before it can be shipped.")
    endif()
    file(SHA256 "${_flm_aie4_staged_root}" _flm_aie4_root_hash)
    set(_flm_aie4_report_text
        "corelib_version\t${_flm_aie4_version}\ncorelib_sha256\t${_flm_aie4_root_hash}\n")
    foreach(_flm_aie4_file IN LISTS _flm_aie4_files)
        get_filename_component(_flm_aie4_leaf "${_flm_aie4_file}" NAME)
        file(SHA256 "${_flm_aie4_file}" _flm_aie4_hash)
        string(APPEND _flm_aie4_report_text
            "staged\t${_flm_aie4_leaf}\t${_flm_aie4_hash}\n")
    endforeach()
    file(WRITE "${FLM_AIE4_REPORT}" "${_flm_aie4_report_text}")
endif()

# The audit record stays in the build tree and does name the source path of
# every staged DLL. That path is what shows, after the fact, that a shipped
# dependency came from the intended package rather than from whatever the
# build machine happened to have.
if(FLM_AIE4_AUDIT)
    set(_flm_aie4_audit_text "root\t${_flm_aie4_root}\n")
    foreach(_flm_aie4_dir IN LISTS _flm_aie4_search_dirs)
        string(APPEND _flm_aie4_audit_text "search\t${_flm_aie4_dir}\n")
    endforeach()
    foreach(_flm_aie4_file IN LISTS _flm_aie4_files)
        get_filename_component(_flm_aie4_leaf "${_flm_aie4_file}" NAME)
        string(APPEND _flm_aie4_audit_text
            "staged\t${_flm_aie4_leaf}\t${_flm_aie4_file}\n")
    endforeach()
    file(WRITE "${FLM_AIE4_AUDIT}" "${_flm_aie4_audit_text}")
endif()
