cmake_minimum_required(VERSION 3.22)
project(flm LANGUAGES CXX VERSION 0.1.0)

if(NOT DEFINED FLM_VERSION)
    message(FATAL_ERROR "FLM_VERSION must be specified externally. Use -DFLM_VERSION=<version> when running cmake.")
endif()

if(NOT DEFINED NPU_VERSION)
    message(FATAL_ERROR "NPU_VERSION must be specified externally. Use -DNPU_VERSION=<version> when running cmake.")
endif()

# Set build type to Release
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CMake to stop at first error
set(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)
set(CMAKE_ERROR_DEPRECATED ON)


# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/)


# Force output directories to be absolute
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# ———————————————————————————————————————————————
# User-tweakable static build settings
# ———————————————————————————————————————————————
option(FLM_PORTABLE_BUILD "Build portable distribution with bundled libraries (XRT dynamic, FFmpeg static)" OFF)

if(WIN32)
    set(XRT_INCLUDE_DIR C:/dev/XRT/src/runtime_src/core/include CACHE PATH   "Where XRT headers live")
    set(XRT_LIB_DIR     C:/dev/xrtNPUfromDLL     CACHE PATH   "Where XRT libs live")
else()
    # On Linux, find pkg-config first, then use it to find XRT
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(XRT xrt)
        if(XRT_FOUND)
            message(STATUS "Found XRT via pkg-config")
            message(STATUS "  XRT include dirs: ${XRT_INCLUDE_DIRS}")
            message(STATUS "  XRT library dirs: ${XRT_LIBRARY_DIRS}")
            message(STATUS "  XRT libraries: ${XRT_LIBRARIES}")
        endif()
    endif()

    # Check if we need to fetch dependencies from source for portable build
    # For portable build: always build FFmpeg statically, only fetch XRT if not found
    set(NEED_FETCH_XRT FALSE)
    set(NEED_BUILD_FFMPEG_STATIC FALSE)
    set(NEED_BUILD_ZLIB_STATIC FALSE)

    if(FLM_PORTABLE_BUILD)
        # Always build FFmpeg statically for portable builds
        set(NEED_BUILD_FFMPEG_STATIC TRUE)
        set(NEED_BUILD_ZLIB_STATIC TRUE)
        message(STATUS "Portable build: FFmpeg and zlib will be built statically")

        if(NOT XRT_FOUND)
            set(NEED_FETCH_XRT TRUE)
            message(STATUS "XRT not found on system, will fetch from source for portable build")
        else()
            message(STATUS "XRT found on system, will use it for portable build")
        endif()
    endif()

    # Include FetchContent if needed for either XRT, FFmpeg, or zlib
    if(NEED_FETCH_XRT OR NEED_BUILD_FFMPEG_STATIC OR NEED_BUILD_ZLIB_STATIC)
        include(FetchContent)
    endif()

    if(NEED_FETCH_XRT)
        message(STATUS "Fetching XRT from source...")

        # ———————————————————————————————————————————————
        # 1. Fetch XRT first (XDNA driver depends on it)
        # ———————————————————————————————————————————————
        set(XRT_GIT_REPO "https://github.com/Xilinx/XRT.git" CACHE STRING "XRT Git repository URL")
        set(XRT_GIT_TAG "2.21.75" CACHE STRING "XRT Git tag/branch to fetch")

        FetchContent_Declare(
            xrt_source
            GIT_REPOSITORY ${XRT_GIT_REPO}
            GIT_TAG ${XRT_GIT_TAG}
            GIT_SHALLOW TRUE
        )

        # Configure XRT build options
        set(XRT_NATIVE_BUILD ON CACHE BOOL "Native XRT build" FORCE)
        set(BUILD_DOCS OFF CACHE BOOL "Build documentation" FORCE)

        # Don't compile Alveo support - only base XRT needed
        set(XRT_BASE 1 CACHE STRING "Build base XRT only" FORCE)
        set(XRT_ALVEO 0 CACHE STRING "Disable Alveo support" FORCE)

        # Disable building tools/utilities that have errors or aren't needed
        set(XRT_BUILD_XBMGMT OFF CACHE BOOL "Build xbmgmt" FORCE)
        set(XRT_BUILD_XBUTIL OFF CACHE BOOL "Build xbutil" FORCE)
        set(XRT_BUILD_XBT OFF CACHE BOOL "Build xbt" FORCE)

        message(STATUS "Fetching XRT sources...")
        FetchContent_MakeAvailable(xrt_source)

        # Set XRT paths for XDNA driver
        set(XRT_REPO_DIR ${xrt_source_SOURCE_DIR})  # Root of XRT repo
        set(XRT_BUILD_DIR ${xrt_source_BINARY_DIR})
        set(XRT_INCLUDE_DIRS ${XRT_REPO_DIR}/src/runtime_src/core/include)

        message(STATUS "XRT fetched successfully")
        message(STATUS "  XRT repo dir: ${XRT_REPO_DIR}")
        message(STATUS "  XRT build dir: ${XRT_BUILD_DIR}")

        # Set variables for later use
        set(XRT_BUILT_FROM_SOURCE TRUE)
        set(XRT_LIBRARY_DIRS ${XRT_BUILD_DIR}/src/runtime_src)

        # Note: XDNA userspace plugin is NOT built here as it requires exact version
        # matching with XRT and has compatibility issues. The plugin should be:
        # 1. Installed from system packages (libxrt-npu2)
        # 2. Or placed in /usr/lib/ or lib/ directory manually
        message(STATUS "Note: XDNA plugin (libxrt_driver_xdna.so.2) must be provided separately")
    endif()

    # ———————————————————————————————————————————————
    # Build FFmpeg as static libraries for portable builds
    # ———————————————————————————————————————————————
    if(NEED_BUILD_FFMPEG_STATIC)
        set(FFMPEG_GIT_REPO "https://github.com/FFmpeg/FFmpeg.git" CACHE STRING "FFmpeg Git repository URL")
        set(FFMPEG_GIT_TAG "n7.1" CACHE STRING "FFmpeg Git tag/branch to fetch")

        FetchContent_Declare(
            ffmpeg_source
            GIT_REPOSITORY ${FFMPEG_GIT_REPO}
            GIT_TAG ${FFMPEG_GIT_TAG}
            GIT_SHALLOW TRUE
        )

        message(STATUS "Fetching FFmpeg sources...")
        FetchContent_Populate(ffmpeg_source)

        set(FFMPEG_SOURCE_DIR ${ffmpeg_source_SOURCE_DIR})
        set(FFMPEG_BUILD_DIR ${CMAKE_BINARY_DIR}/ffmpeg-build)
        set(FFMPEG_INSTALL_DIR ${CMAKE_BINARY_DIR}/ffmpeg-install)

        # Configure and build FFmpeg as static libraries
        message(STATUS "Configuring FFmpeg for static build...")
        file(MAKE_DIRECTORY ${FFMPEG_BUILD_DIR})

        # Run FFmpeg configure
        execute_process(
            COMMAND ${FFMPEG_SOURCE_DIR}/configure
                --prefix=${FFMPEG_INSTALL_DIR}
                --enable-static
                --disable-shared
                --disable-programs
                --disable-doc
                --disable-htmlpages
                --disable-manpages
                --disable-podpages
                --disable-txtpages
                --enable-pic
                --enable-zlib
                --enable-avcodec
                --enable-avformat
                --enable-avutil
                --enable-swscale
                --enable-swresample
            WORKING_DIRECTORY ${FFMPEG_BUILD_DIR}
            RESULT_VARIABLE FFMPEG_CONFIGURE_RESULT
            OUTPUT_FILE ${FFMPEG_BUILD_DIR}/configure.log
            ERROR_FILE ${FFMPEG_BUILD_DIR}/configure.log
        )

        if(NOT FFMPEG_CONFIGURE_RESULT EQUAL 0)
            message(FATAL_ERROR "FFmpeg configure failed. Check ${FFMPEG_BUILD_DIR}/configure.log")
        endif()

        message(STATUS "Building FFmpeg static libraries...")
        # Limit parallelism to avoid resource exhaustion during FFmpeg build
        # FFmpeg compilation is memory-intensive, so use at most 4 jobs
        if(DEFINED CMAKE_BUILD_PARALLEL_LEVEL AND CMAKE_BUILD_PARALLEL_LEVEL GREATER 4)
            set(FFMPEG_BUILD_JOBS 4)
        elseif(DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
            set(FFMPEG_BUILD_JOBS ${CMAKE_BUILD_PARALLEL_LEVEL})
        else()
            set(FFMPEG_BUILD_JOBS 2)
        endif()
        execute_process(
            COMMAND make -j${FFMPEG_BUILD_JOBS}
            WORKING_DIRECTORY ${FFMPEG_BUILD_DIR}
            RESULT_VARIABLE FFMPEG_BUILD_RESULT
            OUTPUT_FILE ${FFMPEG_BUILD_DIR}/build.log
            ERROR_FILE ${FFMPEG_BUILD_DIR}/build.log
        )

        if(NOT FFMPEG_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "FFmpeg build failed. Check ${FFMPEG_BUILD_DIR}/build.log")
        endif()

        message(STATUS "Installing FFmpeg static libraries...")
        execute_process(
            COMMAND make install
            WORKING_DIRECTORY ${FFMPEG_BUILD_DIR}
            RESULT_VARIABLE FFMPEG_INSTALL_RESULT
        )

        if(NOT FFMPEG_INSTALL_RESULT EQUAL 0)
            message(FATAL_ERROR "FFmpeg install failed")
        endif()

        set(FFMPEG_BUILT_FROM_SOURCE TRUE)
        set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INSTALL_DIR}/include)
        set(FFMPEG_LIBRARY_DIRS ${FFMPEG_INSTALL_DIR}/lib)

        message(STATUS "FFmpeg built successfully")
        message(STATUS "  FFmpeg include dir: ${FFMPEG_INCLUDE_DIRS}")
        message(STATUS "  FFmpeg library dir: ${FFMPEG_LIBRARY_DIRS}")
    endif()

    # ———————————————————————————————————————————————
    # Build zlib as static library for portable builds
    # ———————————————————————————————————————————————
    if(NEED_BUILD_ZLIB_STATIC)
        message(STATUS "Fetching zlib source...")

        FetchContent_Declare(
            zlib_source
            GIT_REPOSITORY "https://github.com/madler/zlib.git"
            GIT_TAG "v1.3.1"
            GIT_SHALLOW TRUE
        )

        FetchContent_MakeAvailable(zlib_source)

        set(ZLIB_BUILT_FROM_SOURCE TRUE)
        set(ZLIB_INCLUDE_DIRS ${zlib_source_SOURCE_DIR})
        set(ZLIB_LIBRARY_DIRS ${zlib_source_BINARY_DIR})
        set(ZLIB_STATIC_LIB ${zlib_source_BINARY_DIR}/libz.a)

        message(STATUS "zlib built successfully")
        message(STATUS "  zlib static lib: ${ZLIB_STATIC_LIB}")
    endif()

    # Fallback to manual paths if pkg-config didn't find XRT and FLM_FETCH_XDNA is not enabled
    if(NOT XRT_FOUND AND NOT XRT_BUILT_FROM_SOURCE)
        set(XRT_INCLUDE_DIR /opt/xilinx/xrt/include CACHE PATH   "Where XRT headers live")
        set(XRT_LIB_DIR     /opt/xilinx/xrt/lib     CACHE PATH   "Where XRT libs live")
    endif()
endif()

# ———————————————————————————————————————————————
# Add tokenizers-cpp subproject
# ———————————————————————————————————————————————
# Ensure C++17 is available for tokenizers-cpp
if(CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/../third_party/tokenizers-cpp
                 ${CMAKE_BINARY_DIR}/tokenizers-cpp
                 EXCLUDE_FROM_ALL)

# ———————————————————————————————————————————————
# Gather your sources
# ———————————————————————————————————————————————
file(GLOB SOURCES "src/*.cpp" "runner/*.cpp" "common/*.cpp" "common/*/*.cpp" "server/*.cpp" "pull/*.cpp" )
file(GLOB HEADERS "include/*.hpp" "runner/*.hpp" "common/*.hpp" "common/*/*.hpp" "server/*.hpp" "pull/*.hpp")

# Exclude files that depend on missing libraries for Linux
if(NOT WIN32)
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3_image\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3_text\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gpt_oss\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_lfm2\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_phi4\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen2\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3vl\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3vl_image\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_whisper\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_whisper_audio\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma_embedding\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*auto_embedding_model\\.cpp$")

    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/common/image_process_utils/imageprocAVX512.cpp
        ${CMAKE_SOURCE_DIR}/common/audio_process_utils/audioprocAVX512.cpp
        PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512vl;-mavx512bw;-mfma"
    )

    # Define a macro to indicate limited model support on Linux
    # add_compile_definitions(FASTFLOWLM_LINUX_LIMITED_MODELS=1)
endif()


add_executable(flm ${SOURCES} ${HEADERS})

if(NOT WIN32)
    find_package(Threads REQUIRED)
    find_package(Boost REQUIRED COMPONENTS program_options)
    find_package(CURL REQUIRED)
    # PkgConfig is already found earlier for XRT detection
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3_PKG fftw3)
        pkg_check_modules(FFTW3F_PKG fftw3f)
        pkg_check_modules(FFTW3L_PKG fftw3l)
        pkg_check_modules(AVFORMAT_PKG libavformat)
        pkg_check_modules(AVCODEC_PKG libavcodec)
        pkg_check_modules(AVUTIL_PKG libavutil)
        pkg_check_modules(SWSCALE_PKG libswscale)
        pkg_check_modules(SWRESAMPLE_PKG libswresample)
    endif()
    find_path(READLINE_INCLUDE_DIR readline/readline.h)
    find_library(READLINE_LIBRARY readline)
    find_library(READLINE_TINFO_LIBRARY tinfo)
    find_library(READLINE_NCURSES_LIBRARY ncurses)
endif()

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(flm PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
)

# Add XRT include directories
if(NOT WIN32 AND XRT_FOUND)
    target_include_directories(flm PUBLIC ${XRT_INCLUDE_DIRS})
else()
    target_include_directories(flm PUBLIC ${XRT_INCLUDE_DIR})
endif()

# Add FFmpeg include directories if built from source
if(FFMPEG_BUILT_FROM_SOURCE)
    target_include_directories(flm PUBLIC ${FFMPEG_INCLUDE_DIRS})
endif()

if(WIN32)
    target_include_directories(flm PUBLIC
        C:/dev/boost_1_88_0
        C:/dev/vcpkg/installed/x64-windows/include/
    )
endif()

target_compile_definitions(flm PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
)

if(WIN32)
    target_compile_definitions(flm PUBLIC
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        # Additional definitions for static linking
        CURL_STATICLIB
        BOOST_ALL_NO_LIB
        BOOST_ALL_STATIC_LINK
        # Handle legacy stdio functions
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        __WINDOWS__
    )
endif()

target_link_directories(flm PUBLIC
    ${CMAKE_SOURCE_DIR}/lib
)

# Add XRT library directories
if(NOT WIN32 AND XRT_FOUND)
    target_link_directories(flm PUBLIC ${XRT_LIBRARY_DIRS})
else()
    target_link_directories(flm PUBLIC ${XRT_LIB_DIR})
endif()

if(WIN32)
    target_link_directories(flm PUBLIC
        C:/dev/boost_1_88_0/stage/lib
        C:/dev/vcpkg/installed/x64-windows/lib
    )
endif()

# Link static libraries first
if(MSVC)
    target_link_libraries(flm PUBLIC ${STATIC_LIBS})
endif()

# ———————————————————————————————————————————————
# Link XRT libraries (always dynamic)
# ———————————————————————————————————————————————
if(XRT_BUILT_FROM_SOURCE)
    # If built from source, link to the shared library targets
    if(TARGET xrt_coreutil)
        target_link_libraries(flm PUBLIC xrt_coreutil)
        message(STATUS "  Linked: xrt_coreutil (shared)")
    endif()
elseif(NOT WIN32 AND XRT_FOUND)
    # Use pkg-config provided libraries
    target_link_libraries(flm PUBLIC ${XRT_LIBRARIES})
else()
    # Fallback to manual linking
    target_link_libraries(flm PUBLIC xrt_coreutil)
endif()

# Link your custom libraries (these may still be DLLs if no static versions available)
target_link_libraries(flm PUBLIC
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
    qwen3_5vl_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
    gpt_oss_npu
    whisper_npu
    gemma_embedding
    lfm2_npu
    phi4_npu
    nanbeige_npu
    dequant
    gemm
    lm_head
    mha
)

if(WIN32)
    target_link_libraries(flm PUBLIC
        aiebu_static
    )
else()
    target_link_libraries(flm PUBLIC
        aiebu
    )
endif()

# Link FFmpeg libraries (static if built from source, dynamic otherwise)
if(FFMPEG_BUILT_FROM_SOURCE)
    # Link static FFmpeg libraries
    target_link_directories(flm PUBLIC ${FFMPEG_LIBRARY_DIRS})
    if(ZLIB_BUILT_FROM_SOURCE)
        set(ZLIB_LIB ${ZLIB_STATIC_LIB})
    else()
        set(ZLIB_LIB z)
    endif()
    target_link_libraries(flm PUBLIC
        ${FFMPEG_LIBRARY_DIRS}/libavformat.a
        ${FFMPEG_LIBRARY_DIRS}/libavcodec.a
        ${FFMPEG_LIBRARY_DIRS}/libavutil.a
        ${FFMPEG_LIBRARY_DIRS}/libswscale.a
        ${FFMPEG_LIBRARY_DIRS}/libswresample.a
        ${ZLIB_LIB}
        drm
    )
    message(STATUS "Linking static FFmpeg libraries")
else()
    # Link dynamic FFmpeg libraries (default)
    target_link_libraries(flm PUBLIC
        avformat
        avcodec
        avutil
        swscale
        swresample
    )
endif()

if(WIN32)
    target_link_libraries(flm PUBLIC
        libcurl
        libboost_program_options-vc143-mt-x64-1_88
        libfftw3-3
        libfftw3f-3
        libfftw3l-3
    )
else()
    target_link_libraries(flm PUBLIC
        CURL::libcurl
        Boost::program_options
    )

    target_compile_options(flm PUBLIC -mavx -mavx2)

    if(FFTW3_PKG_FOUND)
        target_include_directories(flm PUBLIC ${FFTW3_PKG_INCLUDE_DIRS})
        target_link_libraries(flm PUBLIC ${FFTW3_PKG_LIBRARIES})
    else()
        target_link_libraries(flm PUBLIC fftw3)
    endif()
    if(FFTW3F_PKG_FOUND)
        target_include_directories(flm PUBLIC ${FFTW3F_PKG_INCLUDE_DIRS})
        target_link_libraries(flm PUBLIC ${FFTW3F_PKG_LIBRARIES})
    else()
        target_link_libraries(flm PUBLIC fftw3f)
    endif()
    if(FFTW3L_PKG_FOUND)
        target_include_directories(flm PUBLIC ${FFTW3L_PKG_INCLUDE_DIRS})
        target_link_libraries(flm PUBLIC ${FFTW3L_PKG_LIBRARIES})
    else()
        target_link_libraries(flm PUBLIC fftw3l)
    endif()

    # Only use pkg-config FFmpeg if not building static FFmpeg from source
    if(NOT FFMPEG_BUILT_FROM_SOURCE)
        if(AVFORMAT_PKG_FOUND)
            target_include_directories(flm PUBLIC ${AVFORMAT_PKG_INCLUDE_DIRS})
            target_link_libraries(flm PUBLIC ${AVFORMAT_PKG_LIBRARIES})
        endif()
        if(AVCODEC_PKG_FOUND)
            target_include_directories(flm PUBLIC ${AVCODEC_PKG_INCLUDE_DIRS})
            target_link_libraries(flm PUBLIC ${AVCODEC_PKG_LIBRARIES})
        endif()
        if(AVUTIL_PKG_FOUND)
            target_include_directories(flm PUBLIC ${AVUTIL_PKG_INCLUDE_DIRS})
            target_link_libraries(flm PUBLIC ${AVUTIL_PKG_LIBRARIES})
        endif()
        if(SWSCALE_PKG_FOUND)
            target_include_directories(flm PUBLIC ${SWSCALE_PKG_INCLUDE_DIRS})
            target_link_libraries(flm PUBLIC ${SWSCALE_PKG_LIBRARIES})
        endif()
        if(SWRESAMPLE_PKG_FOUND)
            target_include_directories(flm PUBLIC ${SWRESAMPLE_PKG_INCLUDE_DIRS})
            target_link_libraries(flm PUBLIC ${SWRESAMPLE_PKG_LIBRARIES})
        endif()
    endif()

    if(READLINE_LIBRARY)
        target_compile_definitions(flm PUBLIC FASTFLOWLM_USE_READLINE=1)
        if(READLINE_INCLUDE_DIR)
            target_include_directories(flm PUBLIC ${READLINE_INCLUDE_DIR})
        endif()
        target_link_libraries(flm PUBLIC ${READLINE_LIBRARY})
        if(READLINE_TINFO_LIBRARY)
            target_link_libraries(flm PUBLIC ${READLINE_TINFO_LIBRARY})
        elseif(READLINE_NCURSES_LIBRARY)
            target_link_libraries(flm PUBLIC ${READLINE_NCURSES_LIBRARY})
        endif()
    endif()

    # Link with dl library for dlopen() support (used for preloading bundled libraries)
    target_link_libraries(flm PUBLIC dl)
endif()

# ———————————————————————————————————————————————
# Link tokenizers-cpp libraries
# ———————————————————————————————————————————————
# The tokenizers-cpp subproject provides these targets:
# - tokenizers_cpp: main tokenizers C++ library
# - tokenizers_c: tokenizers C FFI bindings  
# - sentencepiece: sentencepiece tokenizer library
target_link_libraries(flm PRIVATE
    tokenizers_cpp
)

if(WIN32)
    target_link_libraries(flm PRIVATE
        ntdll wsock32 ws2_32 Bcrypt
        iphlpapi userenv psapi
    )
else()
    target_link_libraries(flm PRIVATE Threads::Threads)
endif()


# ———————————————————————————————————————————————
# Copy the build flm.exe into the out directory (for local dev)
# ———————————————————————————————————————————————
if(WIN32)
    add_custom_command(TARGET flm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:flm>
            ${CMAKE_SOURCE_DIR}/out/flm.exe
    )
endif()

# Add a custom target to check for remaining DLL dependencies
if(WIN32)
    add_custom_target(check_dependencies ALL
        COMMAND ${CMAKE_COMMAND} -E echo "Checking for DLL dependencies..."
        COMMAND dumpbin /dependents $<TARGET_FILE:flm> | findstr /i ".dll"
        COMMENT "Checking for remaining DLL dependencies"
        DEPENDS flm
    )
else()
    if(NOT CMAKE_INSTALL_LIBDIR)
        set(CMAKE_INSTALL_LIBDIR "lib")
    endif()

    file(GLOB so_libs "${CMAKE_SOURCE_DIR}/lib/*.so")
    install(FILES ${so_libs} DESTINATION lib)

    # Bundle system libraries for portable distribution
    if(FLM_PORTABLE_BUILD)
        # Copy XRT and XDNA libraries while preserving symlinks
        install(CODE "
            # Find source directory for XRT libraries
            set(XRT_SOURCE_DIR \"\")
            if(EXISTS \"/usr/lib/libxrt_coreutil.so.2\")
                set(XRT_SOURCE_DIR \"/usr/lib\")
            elseif(EXISTS \"/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2\")
                set(XRT_SOURCE_DIR \"/usr/lib/x86_64-linux-gnu\")
            endif()

            if(XRT_SOURCE_DIR)
                # Find all XRT and Boost library files (including symlinks)
                file(GLOB XRT_FILES \"\${XRT_SOURCE_DIR}/libxrt*.so*\")
                file(GLOB BOOST_FILES \"\${XRT_SOURCE_DIR}/libboost_program_options.so*\")

                foreach(LIB_FILE \${XRT_FILES} \${BOOST_FILES})
                    get_filename_component(LIB_NAME \"\${LIB_FILE}\" NAME)
                    set(DEST_FILE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/\${LIB_NAME}\")

                    # Check if it's a symlink
                    if(IS_SYMLINK \"\${LIB_FILE}\")
                        # Get symlink target (relative path)
                        file(READ_SYMLINK \"\${LIB_FILE}\" LINK_TARGET)
                        message(STATUS \"Creating symlink: \${LIB_NAME} -> \${LINK_TARGET}\")
                        execute_process(
                            COMMAND \${CMAKE_COMMAND} -E create_symlink \"\${LINK_TARGET}\" \"\${DEST_FILE}\"
                        )
                    else()
                        # Copy regular file
                        message(STATUS \"Copying: \${LIB_NAME}\")
                        file(COPY \"\${LIB_FILE}\" DESTINATION \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/\"
                             FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                             GROUP_READ GROUP_EXECUTE
                                             WORLD_READ WORLD_EXECUTE)
                    endif()
                endforeach()

                # Patch XRT libraries to use \$ORIGIN as RPATH so they find
                # dependencies in the same lib/ directory instead of system paths
                find_program(PATCHELF_EXECUTABLE patchelf)
                if(PATCHELF_EXECUTABLE)
                    file(GLOB XRT_LIBS \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/libxrt_*.so.*\")
                    foreach(XRT_LIB \${XRT_LIBS})
                        get_filename_component(XRT_LIB_NAME \"\${XRT_LIB}\" NAME)
                        message(STATUS \"Setting RPATH on \${XRT_LIB_NAME}\")
                        execute_process(
                            COMMAND \${PATCHELF_EXECUTABLE} --set-rpath \"\$ORIGIN\" \"\${XRT_LIB}\"
                            RESULT_VARIABLE RET
                        )
                        if(NOT RET EQUAL 0)
                            message(WARNING \"Failed to set RPATH on \${XRT_LIB_NAME}\")
                        endif()
                    endforeach()
                else()
                    message(STATUS \"patchelf not found, skipping RPATH patching for XRT libraries\")
                endif()

                # XRT internally constructs library paths using \$XILINX_XRT prefix
                # combined with lib/x86_64-linux-gnu. Create the expected structure
                # with symlinks so XRT can find libxrt_core.so.2 from its bundled location.
                set(XRT_MULTIARCH_DIR \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu\")
                file(MAKE_DIRECTORY \${XRT_MULTIARCH_DIR})

                # Create symlinks for XRT libraries that XRT coreutil looks up
                foreach(LIB_NAME libxrt_core.so libxrt_core.so.2 libxrt_core.so.2.21.75
                                     libxrt_coreutil.so libxrt_coreutil.so.2 libxrt_coreutil.so.2.21.75
                                     libxrt_driver_xdna.so.2 libxrt_driver_xdna.so.2.21.75
                                     libxrt_hip.so libxrt_hip.so.2 libxrt_hip.so.2.21.75
                                     libxrt_hwemu.so libxrt_hwemu.so.2 libxrt_hwemu.so.2.21.75
                                     libxrt_noop.so libxrt_noop.so.2 libxrt_noop.so.2.21.75
                                     libxrt_swemu.so libxrt_swemu.so.2 libxrt_swemu.so.2.21.75
                                     libxrt++.so libxrt++.so.2 libxrt++.so.2.21.75)
                    if(EXISTS \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/\${LIB_NAME}\")
                        set(LINK_TARGET \"../\${LIB_NAME}\")
                        set(LINK_DEST \"\${XRT_MULTIARCH_DIR}/\${LIB_NAME}\")
                        if(IS_SYMLINK \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/\${LIB_NAME}\")
                            # Read the original symlink target and make it relative to multiarch dir
                            file(READ_SYMLINK \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/\${LIB_NAME}\" ORIG_TARGET)
                            set(LINK_TARGET \"../\${ORIG_TARGET}\")
                        endif()
                        message(STATUS \"Creating symlink in multiarch dir: \${LIB_NAME} -> \${LINK_TARGET}\")
                        execute_process(
                            COMMAND \${CMAKE_COMMAND} -E create_symlink \"\${LINK_TARGET}\" \"\${LINK_DEST}\"
                        )
                    endif()
                endforeach()
            else()
                message(WARNING \"XRT libraries not found in /usr/lib or /usr/lib/x86_64-linux-gnu\")
            endif()
        ")
        # Copy FFTW libraries while preserving symlinks so the portable
        # distribution does not depend on libfftw3 being installed on the host.
        install(CODE "
            # Collect FFTW shared objects from the standard library locations
            set(FFTW_FILES \"\")
            foreach(SEARCH_DIR \"/usr/lib\" \"/usr/lib/x86_64-linux-gnu\")
                file(GLOB DIR_FFTW \"\${SEARCH_DIR}/libfftw3*.so*\")
                list(APPEND FFTW_FILES \${DIR_FFTW})
            endforeach()

            if(FFTW_FILES)
                foreach(LIB_FILE \${FFTW_FILES})
                    get_filename_component(LIB_NAME \"\${LIB_FILE}\" NAME)
                    set(DEST_FILE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/\${LIB_NAME}\")

                    if(IS_SYMLINK \"\${LIB_FILE}\")
                        file(READ_SYMLINK \"\${LIB_FILE}\" LINK_TARGET)
                        message(STATUS \"Creating symlink: \${LIB_NAME} -> \${LINK_TARGET}\")
                        execute_process(
                            COMMAND \${CMAKE_COMMAND} -E create_symlink \"\${LINK_TARGET}\" \"\${DEST_FILE}\"
                        )
                    else()
                        message(STATUS \"Copying: \${LIB_NAME}\")
                        file(COPY \"\${LIB_FILE}\" DESTINATION \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/\"
                             FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                             GROUP_READ GROUP_EXECUTE
                                             WORLD_READ WORLD_EXECUTE)
                    endif()
                endforeach()

                # Patch FFTW libraries to use \$ORIGIN as RPATH so any transitive
                # dependencies resolve from the bundled lib/ directory.
                find_program(PATCHELF_EXECUTABLE patchelf)
                if(PATCHELF_EXECUTABLE)
                    file(GLOB FFTW_LIBS \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/libfftw3*.so.*\")
                    foreach(FFTW_LIB \${FFTW_LIBS})
                        if(NOT IS_SYMLINK \"\${FFTW_LIB}\")
                            get_filename_component(FFTW_LIB_NAME \"\${FFTW_LIB}\" NAME)
                            message(STATUS \"Setting RPATH on \${FFTW_LIB_NAME}\")
                            execute_process(
                                COMMAND \${PATCHELF_EXECUTABLE} --set-rpath \"\$ORIGIN\" \"\${FFTW_LIB}\"
                                RESULT_VARIABLE RET
                            )
                            if(NOT RET EQUAL 0)
                                message(WARNING \"Failed to set RPATH on \${FFTW_LIB_NAME}\")
                            endif()
                        endif()
                    endforeach()
                else()
                    message(STATUS \"patchelf not found, skipping RPATH patching for FFTW libraries\")
                endif()
            else()
                message(WARNING \"FFTW libraries not found in /usr/lib or /usr/lib/x86_64-linux-gnu\")
            endif()
        ")
    endif()

    if(FLM_PORTABLE_BUILD)
        # Portable build: flat structure with ./flm, ./lib/, ./xclbins/
        set_target_properties(flm PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
    else()
        # Standard FHS layout
        set_target_properties(flm PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    endif()
endif()

if(FLM_PORTABLE_BUILD)
    # Portable build: flat layout
    set_target_properties(flm PROPERTIES INSTALL_RPATH "$ORIGIN/lib")

    # Create wrapper script
    configure_file(
        "${CMAKE_SOURCE_DIR}/flm-wrapper.sh.in"
        "${CMAKE_BINARY_DIR}/flm-wrapper.sh"
        @ONLY
    )

    # Install everything flatly - renaming happens during packaging
    install(TARGETS flm RUNTIME DESTINATION .)
    install(FILES model_list.json DESTINATION .)
    install(FILES model_info.json DESTINATION .)
    install(DIRECTORY xclbins DESTINATION .)
    install(FILES "${CMAKE_BINARY_DIR}/flm-wrapper.sh" DESTINATION . RENAME flm-wrapper.sh)
else()
    # Standard FHS layout
    install(TARGETS flm
        RUNTIME DESTINATION bin
        BUNDLE DESTINATION .
    )
    install(FILES model_list.json DESTINATION share/flm)
    install(FILES model_info.json DESTINATION share/flm)
    install(DIRECTORY xclbins DESTINATION share/flm)
endif()

if(NOT WIN32 AND NOT FLM_PORTABLE_BUILD AND NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr" AND NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    install(CODE "
        message(STATUS \"Creating symlink for flm\")
        file(MAKE_DIRECTORY \"$ENV{DESTDIR}/usr/local/bin\")
        execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
            \"${CMAKE_INSTALL_PREFIX}/bin/flm\"
            \"$ENV{DESTDIR}/usr/local/bin/flm\")"
    )
endif()
