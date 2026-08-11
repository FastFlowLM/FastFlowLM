



#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <iostream>

struct audio_data_t {
    /// Interleaved PCM float32 samples normalized to [-1.0, 1.0].
    /// Layout: [L0, R0, L1, R1, ...] for stereo, [S0, S1, ...] for mono.
    /// Total element count = num_frames * channels.
    std::vector<float> samples;

    int sample_rate = 0;            // sample rate after resampling (= target_sample_rate)
    int original_sample_rate = 0;   // original sample rate of the source file
    int channels = 0;               // number of channels in output (same as original)
    int original_channels = 0;      // original number of channels in the source file
    double duration_seconds = 0.0;  // duration in seconds (= num_frames / sample_rate)
    size_t num_samples = 0;         // total float count in samples vector (= num_frames * channels)
    size_t num_frames = 0;          // number of per-channel frames (= num_samples / channels)
};

/// Append `num_frames` of zero-valued frames to `audio` and update all metadata.
/// Padding is zero-filled (silence) and appended at the end.
/// @param audio      audio to pad in-place
/// @param num_frames number of per-channel frames of silence to append
inline void pad_audio_data(audio_data_t& audio, size_t num_frames) {
    if (num_frames == 0) return;
    if (audio.channels <= 0) {
        std::cerr << "[pad_audio_data] WARNING: audio.channels=" << audio.channels
                  << " is invalid; skipping pad of " << num_frames << " frames.\n";
        return;
    }
    const size_t pad_samples = num_frames * static_cast<size_t>(audio.channels);
    audio.samples.insert(audio.samples.end(), pad_samples, 0.0f);
    audio.num_frames   += num_frames;
    audio.num_samples  += pad_samples;
    audio.duration_seconds = static_cast<double>(audio.num_frames) / audio.sample_rate;
}

/// Controls how multi-channel audio is downmixed to mono.
enum class MonoDownmixMode {
    NONE,   ///< Keep original channels, no downmix
    MEAN,   ///< Simple average: 1/N per channel (matches librosa's to_mono)
    RMS     ///< Energy-preserving: 1/sqrt(N) per channel (FFmpeg default)
};

class AudioReader {
public:
    AudioReader();
    ~AudioReader();

    /// Load audio from a file, decode, resample to target_sample_rate, and convert to mono float32.
    /// Supports any format FFmpeg can demux (mp3, wav, flac, ogg, etc.).
    /// @param filename          path to the audio file
    /// @param out_audio         output struct filled with samples and metadata
    /// @param target_sample_rate desired output sample rate (default 16000)
    /// @return true on success
    bool load_audio(const std::string& filename, audio_data_t& out_audio, int target_sample_rate = 16000, MonoDownmixMode downmix = MonoDownmixMode::NONE);

    /// Load audio from an in-memory buffer (e.g. received over network).
    /// @param data              raw file bytes (mp3, wav, etc.)
    /// @param size              byte count
    /// @param out_audio         output struct filled with samples and metadata
    /// @param target_sample_rate desired output sample rate (default 16000)
    /// @param downmix           how to downmix multi-channel audio to mono
    /// @return true on success
    bool load_audio_from_memory(const uint8_t* data, size_t size, audio_data_t& out_audio, int target_sample_rate = 16000, MonoDownmixMode downmix = MonoDownmixMode::NONE);

    // does a simple clipping and simply discarded the audio beyond the max duration.
    bool clip_audio_length(audio_data_t& audio, double max_duration_second);

private:
    bool decode_audio(struct AVFormatContext* format_ctx, audio_data_t& out_audio, int target_sample_rate, MonoDownmixMode downmix);
};




