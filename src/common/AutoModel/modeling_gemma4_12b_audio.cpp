/// \file modeling_Qwen3VL_image.cpp
/// \brief Gemma4e image processing implementation
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the Gemma4e image processing functionality

#include "AutoModel/modeling_gemma4_12b.hpp"
#include "audio_process_utils/audioproc.hpp"
#include "base64.hpp"
#include <utility>
#include <cmath>
#include <algorithm>
#include <numeric>
audio_data_t Gemma4_12B::load_audio_base64(const std::string &base64_str, int resample_rate, MonoDownmixMode downmix) {
    audio_data_t empty_result;
    audio_data_t result;
    // Decode base64 to raw bytes
    std::string audio_bytes = base64::from_base64(base64_str);
    if (!audio_reader_.load_audio_from_memory(reinterpret_cast<const uint8_t*>(audio_bytes.data()), audio_bytes.size(), result, resample_rate, downmix)) {
        std::cerr << "Failed to load audio from base64 string" << std::endl;
        exit(-1);
        //return empty_result;
    }
    return result;
}




audio_data_t Gemma4_12B::load_audio(const std::string &filename, int resample_rate, MonoDownmixMode downmix) {
    audio_data_t empty_result;
    audio_data_t result;
    
    if (!audio_reader_.load_audio(filename, result, resample_rate, downmix)) {
        std::cerr << "Failed to load audio: " << filename << std::endl;
        exit(-1);
        //return empty_result;
    }
    return result;
}

void Gemma4_12B::extract_waveform_features(audio_data_t& audio, 
    int audio_samples_per_token,
    std::vector<bf16> & result_vector,
    int & num_frames
){

    assert(audio.channels == 1);
    num_frames = (audio.num_samples +audio_samples_per_token-1)/audio_samples_per_token;
    

    result_vector.resize(num_frames * audio_samples_per_token); // new elems are zero-init

    for(size_t i = 0; i < audio.samples.size(); i++){
        result_vector[i] = audio.samples[i]; // float -> bf16
    }
    // padding already zeroed by resize()

}




// std::vector<audio_data_t> Gemma4_12B::clip_audio_length(audio_data_t& audio, double max_duration_second) {
//     std::vector<audio_data_t> audio_chunks;
//     size_t max_frames = static_cast<size_t>(max_duration_second * audio.sample_rate);

//     size_t total_frames = audio.num_frames;
//     size_t total_samples = audio.num_samples;
//     size_t chunk_start_frame = 0;

//     while (chunk_start_frame < total_frames) {
//         size_t chunk_end_frame = std::min(chunk_start_frame + max_frames, total_frames);
//         size_t chunk_start_sample = chunk_start_frame * audio.channels;
//         size_t chunk_end_sample = chunk_end_frame * audio.channels;

//         audio_data_t chunk;
//         chunk.sample_rate = audio.sample_rate;
//         chunk.channels = audio.channels;
//         chunk.num_frames = chunk_end_frame - chunk_start_frame;
//         chunk.num_samples = chunk.num_frames * audio.channels;
//         chunk.duration_seconds = static_cast<double>(chunk.num_frames) / audio.sample_rate;
//         chunk.samples.assign(audio.samples.begin() + chunk_start_sample, audio.samples.begin() + chunk_end_sample);

//         audio_chunks.push_back(std::move(chunk));

//         chunk_start_frame = chunk_end_frame;
//     }

//     return audio_chunks;
// }


