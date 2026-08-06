/// \file qwen3_asr_preprocessor.cpp
/// \brief Qwen3-ASR waveform-to-log-mel preprocessing.

#include "audio/qwen3_asr_preprocessor.hpp"
#include "audio_process_utils/audioproc.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Qwen3ASRPreprocessor::Qwen3ASRPreprocessor() : Qwen3ASRPreprocessor(Config{}) {}

Qwen3ASRPreprocessor::Qwen3ASRPreprocessor(Config config) : config_(config) {
    if (config_.sampling_rate <= 0 || config_.n_fft <= 0 || config_.hop_length <= 0 ||
        config_.num_mel_bins <= 0 || config_.n_window < 0 || config_.min_length < 0) {
        throw std::invalid_argument("invalid Qwen3-ASR preprocessor configuration");
    }
}

int Qwen3ASRPreprocessor::feature_output_length(int mel_frames) {
    if (mel_frames <= 0) return 0;

    // Exact integer equivalent of Transformers' three stride-2 CNN length
    // transforms. A complete 100-frame chunk produces 13 output tokens.
    const int remainder = mel_frames % 100;
    const int remainder_output = (remainder + 7) / 8;
    return remainder_output + (mel_frames / 100) * 13;
}

qwen3_asr_features_t Qwen3ASRPreprocessor::extract(const float* samples, std::size_t num_samples) const {
    if (num_samples > 0 && samples == nullptr) {
        throw std::invalid_argument("samples must not be null when num_samples is non-zero");
    }

    const std::size_t padded_samples = std::max<std::size_t>(num_samples, config_.min_length);
    std::vector<float> waveform(padded_samples, 0.0f);
    if (num_samples > 0) {
        std::copy(samples, samples + num_samples, waveform.begin());
    }

    const int num_frequency_bins = config_.n_fft / 2 + 1;
    const int allocated_frames = audioproc::stft_num_frames(
        static_cast<int>(waveform.size()), config_.n_fft, config_.hop_length, /*center=*/true);
    if (allocated_frames <= 1) {
        return {};
    }

    const std::vector<float> window = audioproc::window_function_optimized(
        config_.n_fft, "hann", /*periodic=*/true);
    const std::vector<float> mel_filters = audioproc::mel_filter_bank_optimized(
        num_frequency_bins,
        config_.num_mel_bins,
        0.0f,
        static_cast<float>(config_.sampling_rate) / 2.0f,
        config_.sampling_rate,
        /*apply_slaney_norm=*/true,
        /*slaney_mel_scale=*/true);

    std::vector<float> power_spec(
        static_cast<std::size_t>(allocated_frames) * num_frequency_bins);
    const int stft_frames = audioproc::stft_power_optimized(
        waveform.data(),
        static_cast<int>(waveform.size()),
        window.data(),
        config_.n_fft,
        config_.hop_length,
        /*center=*/true,
        audioproc::StftPadMode::reflect,
        power_spec.data());

    // Qwen3ASRFeatureExtractor uses stft[..., :-1].
    const int valid_frames = stft_frames - 1;
    if (valid_frames <= 0) {
        return {};
    }

    // audioproc emits [frames, mel_bins].
    std::vector<float> frame_major(
        static_cast<std::size_t>(valid_frames) * config_.num_mel_bins);
    audioproc::mel_spectrogram_optimized(
        power_spec.data(),
        mel_filters.data(),
        frame_major.data(),
        valid_frames,
        num_frequency_bins,
        config_.num_mel_bins);

    const int feature_count = valid_frames * config_.num_mel_bins;
    std::vector<float> log_mel(feature_count);
    // Use the scalar log10 path here. The AVX512 helper intentionally uses a
    // low-order logarithm approximation which is fast but introduces errors up
    // to ~2e-2 after Qwen's normalization, large enough to affect ASR parity.
    audioproc::log_mel_floor</*UseClamp=*/true, /*Base=*/10>(
        frame_major.data(), log_mel.data(), feature_count, 1e-10f);

    const float max_value = audioproc::reduce_max(log_mel.data(), feature_count);
    audioproc::clamp_below_max(log_mel.data(), feature_count, max_value, 8.0f);
    audioproc::affine_scale(log_mel.data(), feature_count, 4.0f, 4.0f);

    int padded_frames = valid_frames;
    const int frame_multiple = config_.n_window * 2;
    if (frame_multiple > 1) {
        const int remainder = padded_frames % frame_multiple;
        if (remainder != 0) padded_frames += frame_multiple - remainder;
    }

    qwen3_asr_features_t result;
    result.num_mel_bins = config_.num_mel_bins;
    result.num_frames = padded_frames;
    result.valid_frames = valid_frames;
    result.input_features.assign(
        static_cast<std::size_t>(config_.num_mel_bins) * padded_frames, 0.0f);
    result.attention_mask.assign(padded_frames, 0);
    std::fill(result.attention_mask.begin(), result.attention_mask.begin() + valid_frames, 1);

    // Transformers layout is [mel_bins, frames].
    for (int frame = 0; frame < valid_frames; ++frame) {
        for (int mel = 0; mel < config_.num_mel_bins; ++mel) {
            result.input_features[static_cast<std::size_t>(mel) * padded_frames + frame] =
                log_mel[static_cast<std::size_t>(frame) * config_.num_mel_bins + mel];
        }
    }

    return result;
}
