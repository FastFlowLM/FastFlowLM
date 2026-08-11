/// \file modeling_qwen3_5_omni_audio.cpp
/// \brief Qwen3_5_Omni audio preprocessing (log-mel spectrogram).
/// \author FastFlowLM Team
/// \note 128-bin log-mel pipeline ported from the gemma4e audio path; the omni
///       audio payload is field-for-field identical to gemma4e's.

#include "AutoModel/modeling_qwen3_5_omni.hpp"
#include "audio_process_utils/audioproc.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>

audio_data_t Qwen3_5_Omni::load_audio(const std::string& filename, int resample_rate, MonoDownmixMode downmix) {
    audio_data_t result;
    if (!audio_reader_.load_audio(filename, result, resample_rate, downmix)) {
        header_print("ERROR", "Qwen3_5_Omni failed to load audio: " << filename);
        exit(-1);
    }
    return result;
}

audio_data_t Qwen3_5_Omni::load_audio_base64(const std::string& base64_str, int resample_rate, MonoDownmixMode downmix) {
    audio_data_t result;
    std::string audio_bytes = base64::from_base64(base64_str);
    if (!audio_reader_.load_audio_from_memory(
            reinterpret_cast<const uint8_t*>(audio_bytes.data()), audio_bytes.size(),
            result, resample_rate, downmix)) {
        header_print("ERROR", "Qwen3_5_Omni failed to load audio from base64 string");
        exit(-1);
    }
    return result;
}

std::vector<audio_data_t> Qwen3_5_Omni::clip_audio_length(audio_data_t& audio, double max_duration_second) {
    std::vector<audio_data_t> audio_chunks;
    size_t max_frames = static_cast<size_t>(max_duration_second * audio.sample_rate);

    size_t total_frames = audio.num_frames;
    size_t chunk_start_frame = 0;

    while (chunk_start_frame < total_frames) {
        size_t chunk_end_frame = std::min(chunk_start_frame + max_frames, total_frames);
        size_t chunk_start_sample = chunk_start_frame * audio.channels;
        size_t chunk_end_sample = chunk_end_frame * audio.channels;

        audio_data_t chunk;
        chunk.sample_rate = audio.sample_rate;
        chunk.channels = audio.channels;
        chunk.num_frames = chunk_end_frame - chunk_start_frame;
        chunk.num_samples = chunk.num_frames * audio.channels;
        chunk.duration_seconds = static_cast<double>(chunk.num_frames) / audio.sample_rate;
        chunk.samples.assign(audio.samples.begin() + chunk_start_sample,
                             audio.samples.begin() + chunk_end_sample);

        audio_chunks.push_back(std::move(chunk));
        chunk_start_frame = chunk_end_frame;
    }
    return audio_chunks;
}

/// \brief Number of audio soft tokens produced from a mel-frame count.
/// \note The engine's audio encoder is not implemented yet; use the gemma4e-style
///       two conv2d downsampling simulation (kernel=3, stride=2, padding=1) as a
///       reasonable placeholder so token expansion stays self-consistent.
std::string Qwen3_5_Omni::get_audio_tokens(int audio_length ) {
    auto audio_token_per_second = std::ceil(

        audio_sampling_rate / audio_hop_length/ (std::pow(2,audio_downsample_time ))
    );
    // std::cout << "audio_token_per_second "<< audio_token_per_second<<std::endl;
    // std::cout << "audio_length is "<< audio_length<<std::endl;
    int tokens_interval = audio_token_per_second * audio_timestamp_interval;


    int num_full_chunks = std::floor(audio_length/ tokens_interval);
    int num_residual_tokens = audio_length% tokens_interval;
    

    std::string audio_paceholders = "";
    for(int i = 0; i < num_full_chunks; i++){
        char buf[128];
        std::snprintf(buf, sizeof(buf), "<%.1f seconds>", (double)(i * audio_timestamp_interval));
       
        audio_paceholders += buf;
        for(int j = 0; j < tokens_interval; j++){
            audio_paceholders += "<|audio_pad|>";
       
        }
            
    }
    if(num_residual_tokens > 0){
        char buf[128];
        std::snprintf(buf, sizeof(buf), "<%.1f seconds>", (double)(num_full_chunks * audio_timestamp_interval));
        audio_paceholders += buf;
    
        for(int j = 0; j < num_residual_tokens; j++){
         
            audio_paceholders += "<|audio_pad|>";
        }
            
    }

   return audio_paceholders;
}

void Qwen3_5_Omni::extract_spectrogram(std::vector<audio_data_t>& audio_inputs,
                                       qwen3_5_omni_audio_payload_t& audio_payload) {
    // Matches _torch_extract_fbank_features() from HF Qwen3.5-Omni feature extractor.
    // Fixed audio encoder config (identical to Whisper-large-v2):
    constexpr int   n_fft            = 400;
    constexpr int   num_mel_filters  = 128;
    constexpr float min_frequency    = 0.0f;
    constexpr float max_frequency    = 8000.0f;
    constexpr float clamp_min        = 1e-10f;
    constexpr float dynamic_range    = 8.0f;
    constexpr float log_offset       = 4.0f;
    constexpr float log_scale        = 4.0f;

    const int hop_length         = audio_hop_length;    // 160
    const int sampling_rate      = audio_sampling_rate; // 16000
    const int num_frequency_bins = n_fft / 2 + 1;       // 201

    // Pre-compute window and mel filters once — shared across all audio chunks.
    // torch.hann_window(n_fft) is periodic=True in PyTorch convention.
    std::vector<float> window = audioproc::window_function_optimized(
        n_fft, "hann", /*periodic=*/true);

    // mel_filters shape: [num_frequency_bins x num_mel_filters] (all 201 freq bins, Slaney norm+scale).
    // Whisper feature extractor uses mel_scale="slaney", norm="slaney".
    std::vector<float> mel_filters = audioproc::mel_filter_bank_optimized(
        num_frequency_bins, num_mel_filters,
        min_frequency, max_frequency,
        sampling_rate, /*apply_slaney_norm=*/true, /*slaney_mel_scale=*/true);

    audio_payload.num_audios = static_cast<unsigned int>(audio_inputs.size());
    audio_payload.mel_spectrograms.resize(audio_payload.num_audios);
    audio_payload.mel_spectrogram_frames_per_audio.resize(audio_payload.num_audios);
    audio_payload.mel_spectrogram_bins_per_audio.resize(audio_payload.num_audios);

    for (unsigned int audio_idx = 0; audio_idx < audio_payload.num_audios; audio_idx++) {
        const audio_data_t& audio_input = audio_inputs[audio_idx];
        const int n_samples = static_cast<int>(audio_input.num_frames);

        // Allocate power spectrum buffer for worst-case frame count.
        const int num_frames_alloc = audioproc::stft_num_frames(n_samples, n_fft, hop_length, /*center=*/true);
        if (num_frames_alloc <= 0) continue;

        // STFT power spectrum: torch.stft(..., center=True) with reflect padding.
        // stft[..., :-1] drops the last TIME frame, so allocate for all frames.
        std::vector<float> power_spec(static_cast<size_t>(num_frames_alloc) * num_frequency_bins);
        const int num_frames = audioproc::stft_power_optimized(
            audio_input.samples.data(), n_samples,
            window.data(), n_fft, hop_length,
            /*center=*/true, audioproc::StftPadMode::reflect,
            power_spec.data());

        // stft[..., :-1]: drop the last time frame.
        const int num_frames_used = num_frames - 1;
        if (num_frames_used <= 0) continue;

        // mel_filters.T @ magnitudes  (mel_filters: [num_frequency_bins x num_mel_filters])
        std::vector<float> mel_spec(static_cast<size_t>(num_frames_used) * num_mel_filters);
        audioproc::mel_spectrogram_optimized(
            power_spec.data(), mel_filters.data(), mel_spec.data(),
            num_frames_used, num_frequency_bins, num_mel_filters);

        // log10(clamp(mel_spec, min=1e-10))
        const int total = num_frames_used * num_mel_filters;
        std::vector<float> log_mel(total);
        audioproc::log_mel_floor_optimized</*UseClamp=*/true, /*Base=*/10>(
            mel_spec.data(), log_mel.data(), total, clamp_min);

        // log_spec = max(log_spec, log_spec.max() - 8.0)
        float max_val = audioproc::reduce_max(log_mel.data(), total);
        audioproc::clamp_below_max(log_mel.data(), total, max_val, dynamic_range);

        // log_spec = (log_spec + 4.0) / 4.0
        audioproc::affine_scale(log_mel.data(), total, log_offset, log_scale);

        // Convert to bf16 and store
        std::vector<bf16> log_mel_bf16(total);
        for (int i = 0; i < total; i++)
            log_mel_bf16[i] = static_cast<bf16>(log_mel[i]);

        audio_payload.mel_spectrograms[audio_idx]              = std::move(log_mel_bf16);
        audio_payload.mel_spectrogram_frames_per_audio[audio_idx] = num_frames_used;
        audio_payload.mel_spectrogram_bins_per_audio[audio_idx]   = num_mel_filters;
    }
}




void Qwen3_5_Omni::pad_audio(std::vector<audio_data_t> & audio_inputs){
    // This does padding to audio_data for two reason
    // 1. Preprocessing, the audio_datat  per_channel samples need to be multiple of audio_sampling_rate
    // 2. Padding to ensure no partial window_attention 



    for(int audio_idx = 0; audio_idx< audio_inputs.size(); audio_idx++){

        


        audio_data_t& cur_audio = audio_inputs[audio_idx];
        int pad_mult_of_sample_rate =  this->audio_sampling_rate - (cur_audio.num_samples %this->audio_sampling_rate);


        // predict encoder_length, aka a slight reverse of _get_feat_extract_output_length()
        // NOTE: execpt the input in _get_feat_extract_output_length() assume samples/hop_length

        int processed_x = (pad_mult_of_sample_rate+ cur_audio.num_samples) / this->audio_hop_length;


        int ceil_downChunk_div_twoTodownTime = std::ceil(audio_downsample_chunk_size / std::pow(2,audio_downsample_time));

        int cur_x_window_atten_len =  processed_x/audio_downsample_chunk_size * ceil_downChunk_div_twoTodownTime + \
            std::ceil( (processed_x%audio_downsample_chunk_size) / std::pow(2,audio_downsample_time));
        

        int pad_window_attn_len = static_cast<int>(std::ceil(static_cast<float>(cur_x_window_atten_len) / audio_window_attention_length)) * audio_window_attention_length;
        // std::cout << "audio_window_attention_length" << audio_window_attention_length << std::endl;
        // std::cout << "cur_x_window_atten_len " << cur_x_window_atten_len<< std::endl;
        // std::cout << "pad_window_attn_len " <<pad_window_attn_len <<std::endl;
        int required_window_attn_pad_len = pad_window_attn_len- cur_x_window_atten_len;
        //std::cout << "required_window_attn_pad_len" << required_window_attn_pad_len<< std::endl;



        int required_addition_x_pad = audio_hop_length*(

           audio_downsample_chunk_size*std::floor(required_window_attn_pad_len/ ceil_downChunk_div_twoTodownTime ) \
           
           + std::min( (audio_downsample_chunk_size-1),  int(std::pow(2,audio_downsample_time)*(required_window_attn_pad_len%ceil_downChunk_div_twoTodownTime )))
        );

        pad_audio_data( audio_inputs[audio_idx], required_addition_x_pad + pad_mult_of_sample_rate );
        // std::cout << "DEBUG: sasmples length after padded is " << audio_inputs[audio_idx].num_frames<< std::endl;

        // std::cout << "DEBUG: predict seq_length after padded is " << _get_feat_extract_output_length(  
        //     audio_inputs[audio_idx].num_frames/ audio_hop_length
        // ) << std::endl;
    }



}