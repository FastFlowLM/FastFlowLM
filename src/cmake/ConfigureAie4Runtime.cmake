set(
    RYZENAI_CORELIB_RUNTIME_DIR
    ""
    CACHE PATH
    "Packaging-only directory containing the ryzenai-corelib runtime DLLs")
set(
    XRT_RUNTIME_DIR
    ""
    CACHE PATH
    "Packaging-only directory containing the XRT runtime DLLs")
set(
    FLM_AIE4_DEPENDENCY_DIRS
    ""
    CACHE STRING
    "Additional directories searched for the optional AIE4 DLL closure")

function(_flm_aie4_require_directory variable description)
    if(NOT ${variable})
        message(FATAL_ERROR
            "${variable} is required when FLM_ENABLE_CORELIB_AIE4=ON "
            "(${description})")
    endif()
    if(NOT IS_DIRECTORY "${${variable}}")
        message(FATAL_ERROR
            "${variable} does not name a directory: ${${variable}}")
    endif()
endfunction()

function(_flm_aie4_find_dependency output filename)
    set(_search_dirs
        "${RYZENAI_CORELIB_RUNTIME_DIR}"
        ${FLM_AIE4_DEPENDENCY_DIRS})
    if(CMAKE_SOURCE_DIR)
        list(APPEND _search_dirs "${CMAKE_SOURCE_DIR}/lib")
    endif()
    set(_found "")
    foreach(_directory IN LISTS _search_dirs)
        if(_directory AND EXISTS "${_directory}/${filename}")
            get_filename_component(
                _found
                "${_directory}/${filename}"
                ABSOLUTE)
            break()
        endif()
    endforeach()
    set(${output} "${_found}" PARENT_SCOPE)
endfunction()

function(flm_collect_aie4_runtime_files output)
    if(NOT FLM_ENABLE_CORELIB_AIE4)
        set(${output} "" PARENT_SCOPE)
        return()
    endif()
    if(NOT WIN32)
        message(FATAL_ERROR
            "FLM_ENABLE_CORELIB_AIE4 runtime packaging is Windows-only")
    endif()

    _flm_aie4_require_directory(
        RYZENAI_CORELIB_RUNTIME_DIR
        "ryzenai_corelib.dll, ryzen_mm.dll, and dyn_bins.dll")
    _flm_aie4_require_directory(
        XRT_RUNTIME_DIR
        "xrt_coreutil.dll and the XRT device runtime")

    set(_runtime_files "")
    foreach(_filename IN ITEMS
        ryzenai_corelib.dll
        ryzen_mm.dll
        dyn_bins.dll
        spdlog.dll
        fmt.dll
        libprotobuf.dll
        zlib.dll
        libutf8_validity.dll
        abseil_dll.dll)
        _flm_aie4_find_dependency(_dependency "${_filename}")
        if(NOT _dependency)
            message(FATAL_ERROR
                "The AIE4 runtime closure is incomplete: ${_filename} "
                "was not found in RYZENAI_CORELIB_RUNTIME_DIR or "
                "FLM_AIE4_DEPENDENCY_DIRS")
        endif()
        list(APPEND _runtime_files "${_dependency}")
    endforeach()

    foreach(_filename IN ITEMS zlib1.dll)
        _flm_aie4_find_dependency(_dependency "${_filename}")
        if(_dependency)
            list(APPEND _runtime_files "${_dependency}")
        endif()
    endforeach()

    foreach(_filename IN ITEMS
        xrt_coreutil.dll
        xrt_core.dll
        xrt_umddml.dll)
        if(NOT EXISTS "${XRT_RUNTIME_DIR}/${_filename}")
            message(FATAL_ERROR
                "The AIE4 XRT closure is incomplete: ${_filename} "
                "was not found in XRT_RUNTIME_DIR")
        endif()
    endforeach()
    file(GLOB _xrt_runtime_dlls "${XRT_RUNTIME_DIR}/*.dll")
    if(NOT _xrt_runtime_dlls)
        message(FATAL_ERROR
            "XRT_RUNTIME_DIR contains no runtime DLLs: ${XRT_RUNTIME_DIR}")
    endif()
    list(APPEND _runtime_files ${_xrt_runtime_dlls})
    list(REMOVE_DUPLICATES _runtime_files)
    list(SORT _runtime_files)
    set(${output} "${_runtime_files}" PARENT_SCOPE)
endfunction()
