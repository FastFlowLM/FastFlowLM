#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <stdexcept>
#include "fftw3.h"

namespace audioproc {

    // ---------------------------------------------------------------
    //  Mel-scale conversion helpers
    //  mel_scale: "htk" (2595*log10(1+f/700)) or "slaney" (piecewise linear/log)
    // ---------------------------------------------------------------
    inline float hertz_to_mel(float freq, bool slaney = false) {
        if (!slaney)
            return 2595.0f * std::log10(1.0f + freq / 700.0f);
        // Slaney piecewise: linear below 1000 Hz, log above
        constexpr float min_log_hertz = 1000.0f;
        constexpr float min_log_mel   = 15.0f;
        constexpr float logstep       = 27.0f / 1.8562979903656263f; // 27/log(6.4)
        if (freq >= min_log_hertz)
            return min_log_mel + std::log(freq / min_log_hertz) * logstep;
        return 3.0f * freq / 200.0f;
    }

    inline float mel_to_hertz(float mel, bool slaney = false) {
        if (!slaney)
            return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
        // Slaney inverse
        constexpr float min_log_hertz = 1000.0f;
        constexpr float min_log_mel   = 15.0f;
        constexpr float logstep       = 27.0f / 1.8562979903656263f;
        if (mel >= min_log_mel)
            return min_log_hertz * std::exp((mel - min_log_mel) / logstep);
        return 200.0f * mel / 3.0f;
    }

    // Compute real-to-complex FFT (rfft) for a batch of frames.
    // Equivalent to np.fft.rfft(frames, n=fft_length, axis=-1)
    //
    // frames:        input float array, row-major [num_frames * frame_length]
    // out_complex:   output interleaved (re,im) [num_frames * (fft_length/2+1) * 2]
    // num_frames:    number of frames in the batch
    // frame_length:  number of samples per frame (will be zero-padded to fft_length)
    // fft_length:    FFT size (must be >= frame_length)
    void rfft_batch(
        const float* frames,
        float* out_complex,
        int num_frames,
        int frame_length,
        int fft_length);

    // Compute rfft then magnitude |stft| for a batch of frames.
    // Equivalent to np.abs(np.fft.rfft(frames, n=fft_length, axis=-1))
    //
    // out_magnitude: output float array [num_frames * (fft_length/2+1)]
    void rfft_magnitude_batch(
        const float* frames,
        float* out_magnitude,
        int num_frames,
        int frame_length,
        int fft_length);

    // Auto-dispatching optimized versions (AVX512 when available, scalar fallback)
    void rfft_batch_optimized(
        const float* frames,
        float* out_complex,
        int num_frames,
        int frame_length,
        int fft_length);

    void rfft_magnitude_batch_optimized(
        const float* frames,
        float* out_magnitude,
        int num_frames,
        int frame_length,
        int fft_length);

    // Compute rfft then squared power re²+im² for a batch of frames.
    // Equivalent to torch.stft(...).abs()**2  (no sqrt).
    // out_power: output float array [num_frames * (fft_length/2+1)]
    void stft_power_batch(
        const float* frames,
        float* out_power,
        int num_frames,
        int frame_length,
        int fft_length);

    void stft_power_batch_optimized(
        const float* frames,
        float* out_power,
        int num_frames,
        int frame_length,
        int fft_length);

    // ---------------------------------------------------------------
    //  Mel filter bank generation
    //  Equivalent to HuggingFace audio_utils.mel_filter_bank() with
    //  mel_scale="slaney", norm="slaney", triangularize_in_mel_space=False.
    // ---------------------------------------------------------------

    // Generate mel filter bank matrix [num_frequency_bins x num_mel_filters],
    // stored in row-major order.
    //
    // num_frequency_bins:   fft_length / 2 + 1
    // num_mel_filters:      e.g. 128
    // min_frequency:        lowest frequency of interest in Hz (e.g. 0)
    // max_frequency:        highest frequency of interest in Hz (e.g. 8000)
    // sampling_rate:        sample rate of the audio waveform (e.g. 16000)
    // apply_slaney_norm:    if true, apply Slaney area-normalization
    // slaney_mel_scale:     if true, use Slaney piecewise mel scale; else HTK
    std::vector<float> mel_filter_bank(
        int num_frequency_bins,
        int num_mel_filters,
        float min_frequency,
        float max_frequency,
        int sampling_rate,
        bool apply_slaney_norm = false,
        bool slaney_mel_scale  = false);

    // Compute mel spectrogram:  out = magnitude_spec @ mel_filters
    // magnitude_spec: [num_frames x num_frequency_bins]  (row-major)
    // mel_filters:    [num_frequency_bins x num_mel_filters]  (row-major)
    // out:            [num_frames x num_mel_filters]  (row-major)
    void mel_spectrogram(
        const float* magnitude_spec,
        const float* mel_filters,
        float* out,
        int num_frames,
        int num_frequency_bins,
        int num_mel_filters);

    // Auto-dispatching optimized versions
    std::vector<float> mel_filter_bank_optimized(
        int num_frequency_bins,
        int num_mel_filters,
        float min_frequency,
        float max_frequency,
        int sampling_rate,
        bool apply_slaney_norm = false,
        bool slaney_mel_scale  = false);

    void mel_spectrogram_optimized(
        const float* magnitude_spec,
        const float* mel_filters,
        float* out,
        int num_frames,
        int num_frequency_bins,
        int num_mel_filters);

    // ---------------------------------------------------------------
    //  Window function generation
    //  Equivalent to HuggingFace audio_utils.window_function().
    //  Supported window names: "boxcar", "hamming", "hann", "povey"
    // ---------------------------------------------------------------

    // Generate a window of length window_length.
    //
    // window_length:  number of samples in the window
    // name:           "boxcar", "hamming", "hann", or "povey"
    // periodic:       if true, generate periodic window (length+1 then drop last)
    // frame_length:   if > 0, zero-pad or embed window into this size
    // center:         if true and frame_length > 0, center window in the frame
    std::vector<float> window_function(
        int window_length,
        const std::string& name = "hann",
        bool periodic = true,
        int frame_length = 0,
        bool center = true);

    // Auto-dispatching optimized version
    std::vector<float> window_function_optimized(
        int window_length,
        const std::string& name = "hann",
        bool periodic = true,
        int frame_length = 0,
        bool center = true);

    // ---------------------------------------------------------------
    //  Fused unfold + window multiply
    //  Extracts overlapping frames from a pre-padded waveform and
    //  multiplies element-wise by the window.
    //
    //  waveform:          pre-padded waveform [padded_length]
    //  window:            window coefficients [frame_length]
    //  out_windowed:      output [num_frames * frame_length], row-major
    //  num_frames:        number of output frames
    //  frame_length:      samples per output frame
    //  hop_length:        stride between successive frames
    // ---------------------------------------------------------------
    void apply_window_frames(
        const float* waveform,
        const float* window,
        float* out_windowed,
        int num_frames,
        int frame_length,
        int hop_length);

    void apply_window_frames_optimized(
        const float* waveform,
        const float* window,
        float* out_windowed,
        int num_frames,
        int frame_length,
        int hop_length);

    // ---------------------------------------------------------------
    //  High-level STFT power spectrum  (re²+im² per bin)
    //  Equivalent to torch.stft(waveform, n_fft, hop_length,
    //      window=window, center=center, return_complex=True).abs()**2
    //
    //  Padding modes when center=true:
    //    "reflect" — mirrors samples at each edge (torch default)
    //    "constant" — zero-pads each edge
    //
    //  Returns the number of output frames written to out_power.
    //  out_power must be pre-allocated to at least
    //    stft_num_frames(n_samples, n_fft, hop_length, center) * (n_fft/2+1) floats.
    // ---------------------------------------------------------------
    enum class StftPadMode { reflect, constant };

    int stft_num_frames(int n_samples, int n_fft, int hop_length, bool center);

    // Compute STFT power spectrum from a raw (un-padded) waveform.
    // window: [n_fft] coefficients.  out_power: [num_frames x (n_fft/2+1)].
    int stft_power(
        const float*  waveform,
        int           n_samples,
        const float*  window,
        int           n_fft,
        int           hop_length,
        bool          center,
        StftPadMode   pad_mode,
        float*        out_power);

    int stft_power_optimized(
        const float*  waveform,
        int           n_samples,
        const float*  window,
        int           n_fft,
        int           hop_length,
        bool          center,
        StftPadMode   pad_mode,
        float*        out_power);

    // ---------------------------------------------------------------
    //  Reduce max over [in, in+count) and clamp-from-below in-place.
    //  Mirrors:  log_spec = max(log_spec, log_spec.max() - dynamic_range)
    // ---------------------------------------------------------------
    float reduce_max(const float* in, int count);

    void clamp_below_max(float* inout, int count, float max_val, float dynamic_range);

    // ---------------------------------------------------------------
    //  Affine scale in-place: out[i] = (in[i] + offset) / scale
    // ---------------------------------------------------------------
    void affine_scale(float* inout, int count, float offset, float scale);

    // ---------------------------------------------------------------
    //  Vectorized log with optional floor-add or clamp, selectable base.
    //
    //  Two modes controlled by template parameters:
    //    UseClamp=false (default): out[i] = log_base(in[i] + floor)
    //    UseClamp=true:            out[i] = log_base(max(in[i], floor))
    //
    //  Base=0 (default): natural log (ln)
    //  Base=10:          log10  (torch fbank path: clamp_min=1e-10, log10)
    // ---------------------------------------------------------------
    template<bool UseClamp = false, int Base = 0>
    void log_mel_floor(
        const float* in,
        float* out,
        int count,
        float floor);

    template<bool UseClamp = false, int Base = 0>
    void log_mel_floor_optimized(
        const float* in,
        float* out,
        int count,
        float floor);

} // namespace audioproc
