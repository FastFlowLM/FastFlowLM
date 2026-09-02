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
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_gemma4_12b_image.cpp"   
        "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_gemma4_12b_audio.cpp"             
        "${CMAKE_SOURCE_DIR}/../../common/image_process_utils/imageproc.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/image_process_utils/imageprocAVX512.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/image/image_reader.cpp"    
        "${CMAKE_SOURCE_DIR}/../../common/audio/audio_reader.cpp"       
        "${CMAKE_SOURCE_DIR}/../../common/audio_process_utils/audioproc.cpp"
        "${CMAKE_SOURCE_DIR}/../../common/audio_process_utils/audioprocAVX512.cpp"            
)

target_compile_definitions(test_gemma4_12b PUBLIC
    __FLM_VERSION__="0.9.45"
    USEAVX512=1    
)


# Add OpenMP compiler flags if available
if(OpenMP_CXX_FOUND)
    target_compile_options(test_gemma4_12b PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:/openmp>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fopenmp>
    )
endif()

target_link_libraries(test_gemma4_12b PUBLIC
    gemma4_12b_npu
    xrt_coreutil
    avformat
    avcodec
    avutil
    swscale
    zlib    
)


# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_gemma4_12b PUBLIC OpenMP::OpenMP_CXX)
endif()



# Add test target
add_custom_target(test_gemma4_12b_target
    DEPENDS test_gemma4_12b
    COMMENT "Building test_gemma4_12b executable"
)
