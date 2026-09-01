/// \file modeling_Qwen3VL_image.cpp
/// \brief Gemma4e image processing implementation
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the Gemma4e image processing functionality

#include "AutoModel/modeling_gemma4_12b.hpp"
#include "audio_process_utils/audioproc.hpp"
#include "base64.hpp"
#include <cassert>
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
    int audio_max_soft_tokens,
    std::vector<bf16> & result_vector,
    int & num_frames,
    size_t sample_offset
){

    assert(audio.channels == 1);
    const size_t total_samples = audio.samples.size();
    const size_t remaining = (sample_offset < total_samples) ? (total_samples - sample_offset) : 0;
    num_frames = static_cast<int>((remaining + audio_samples_per_token - 1) / audio_samples_per_token);

    // One token is 640 samples = 40 ms; the model accepts at most 750 of them
    // (30 s). Anything past that is dropped, as in the reference extractor.
    // Both bounds come from processor_config.json.
    if (audio_max_soft_tokens > 0 && num_frames > audio_max_soft_tokens) {
        num_frames = audio_max_soft_tokens;
    }

    result_vector.assign(static_cast<size_t>(num_frames) * audio_samples_per_token, bf16(0.0f));

    const size_t num_to_copy = std::min(remaining, result_vector.size());
    for(size_t i = 0; i < num_to_copy; i++){
        result_vector[i] = audio.samples[sample_offset + i]; // float -> bf16
    }
    // tail padding already zeroed by assign()

}

void Gemma4_12B::extract_waveform_features_chunked(audio_data_t& audio,
    int audio_samples_per_token,
    int audio_max_soft_tokens,
    std::vector<std::vector<bf16>> & chunk_features,
    std::vector<int> & chunk_num_frames
){
    chunk_features.clear();
    chunk_num_frames.clear();

    const size_t total_samples = audio.samples.size();
    if (total_samples == 0) {
        return;
    }

    // Without a positive cap a clip is never split: one chunk holds everything.
    const size_t samples_per_chunk = (audio_max_soft_tokens > 0)
        ? static_cast<size_t>(audio_max_soft_tokens) * audio_samples_per_token
        : total_samples;

    for (size_t offset = 0; offset < total_samples; offset += samples_per_chunk) {
        std::vector<bf16> features;
        int num_frames = 0;
        extract_waveform_features(audio, audio_samples_per_token, audio_max_soft_tokens,
                                  features, num_frames, offset);
        if (num_frames <= 0) {
            break;
        }
        chunk_features.push_back(std::move(features));
        chunk_num_frames.push_back(num_frames);
    }
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


