/// \file qwen3_asr_preprocessor.hpp
/// \brief Qwen3-ASR waveform-to-log-mel preprocessing.
#pragma once

#include <cstddef>
#include <vector>

struct qwen3_asr_features_t {
    // Feature-major layout matching Transformers input_features:
    // [num_mel_bins, num_frames], contiguous in row-major order.
    std::vector<float> input_features;
    std::vector<int> attention_mask;
    int num_mel_bins = 0;
    int num_frames = 0;
    int valid_frames = 0;
};

class Qwen3ASRPreprocessor {
public:
    struct Config {
        int sampling_rate = 16000;
        int n_fft = 400;
        int hop_length = 160;
        int num_mel_bins = 128;
        int n_window = 50;
        int min_length = 8000;
    };

    Qwen3ASRPreprocessor();
    explicit Qwen3ASRPreprocessor(Config config);

    /// Convert mono float32 PCM into Qwen3-ASR log-mel features.
    ///
    /// This follows Hugging Face Qwen3ASRFeatureExtractor:
    /// - zero-pad waveforms shorter than min_length;
    /// - periodic Hann window and centered reflect-padded STFT;
    /// - Slaney mel scale and normalization;
    /// - log10 clamp, dynamic-range clamp, and (x + 4) / 4 scaling;
    /// - right-pad the mel time axis to a multiple of 2 * n_window.
    qwen3_asr_features_t extract(const float* samples, std::size_t num_samples) const;
    qwen3_asr_features_t extract(const std::vector<float>& samples) const {
        return extract(samples.data(), samples.size());
    }

    /// Number of soft audio tokens produced by the three stride-2 CNN layers.
    static int feature_output_length(int mel_frames);

    const Config& config() const { return config_; }

private:
    Config config_;
};
