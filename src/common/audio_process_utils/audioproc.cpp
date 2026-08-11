#include "audio_process_utils/audioproc.hpp"
#include "audio_process_utils/audioprocAVX512.hpp"
#include <algorithm>
#include <cassert>
// Ensure M_PI is available on MSVC and other platforms that omit it
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace audioproc {

    void rfft_batch(
        const float* frames,
        float* out_complex,
        int num_frames,
        int frame_length,
        int fft_length)
    {
        const int n_bins = fft_length / 2 + 1;

        // Allocate FFTW buffers
        float* fft_in = (float*)fftwf_malloc(fft_length * sizeof(float));
        fftwf_complex* fft_out = (fftwf_complex*)fftwf_malloc(n_bins * sizeof(fftwf_complex));

        // Create plan (FFTW_ESTIMATE for fast planning)
        fftwf_plan plan = fftwf_plan_dft_r2c_1d(fft_length, fft_in, fft_out, FFTW_ESTIMATE);

        for (int f = 0; f < num_frames; ++f) {
            // Copy frame data
            std::memcpy(fft_in, frames + f * frame_length, frame_length * sizeof(float));
            // Zero-pad if fft_length > frame_length
            if (fft_length > frame_length) {
                std::memset(fft_in + frame_length, 0, (fft_length - frame_length) * sizeof(float));
            }

            fftwf_execute(plan);

            // Copy interleaved (re, im) output
            float* dst = out_complex + f * n_bins * 2;
            for (int k = 0; k < n_bins; ++k) {
                dst[2 * k]     = fft_out[k][0]; // real
                dst[2 * k + 1] = fft_out[k][1]; // imag
            }
        }

        fftwf_destroy_plan(plan);
        fftwf_free(fft_in);
        fftwf_free(fft_out);
    }

    // Internal helper: runs FFTW batch and applies a per-bin scalar reduction.
    // Reduce: (float re, float im) -> float
    template<typename Reduce>
    static void rfft_reduce_batch_impl(
        const float* frames,
        float* out,
        int num_frames,
        int frame_length,
        int fft_length,
        Reduce reduce)
    {
        const int n_bins = fft_length / 2 + 1;

        float* fft_in          = (float*)fftwf_malloc(fft_length * sizeof(float));
        fftwf_complex* fft_out = (fftwf_complex*)fftwf_malloc(n_bins * sizeof(fftwf_complex));
        fftwf_plan plan = fftwf_plan_dft_r2c_1d(fft_length, fft_in, fft_out, FFTW_ESTIMATE);

        for (int f = 0; f < num_frames; ++f) {
            std::memcpy(fft_in, frames + f * frame_length, frame_length * sizeof(float));
            if (fft_length > frame_length)
                std::memset(fft_in + frame_length, 0, (fft_length - frame_length) * sizeof(float));

            fftwf_execute(plan);

            float* dst = out + f * n_bins;
            for (int k = 0; k < n_bins; ++k)
                dst[k] = reduce(fft_out[k][0], fft_out[k][1]);
        }

        fftwf_destroy_plan(plan);
        fftwf_free(fft_in);
        fftwf_free(fft_out);
    }

    void rfft_magnitude_batch(
        const float* frames,
        float* out_magnitude,
        int num_frames,
        int frame_length,
        int fft_length)
    {
        rfft_reduce_batch_impl(frames, out_magnitude, num_frames, frame_length, fft_length,
            [](float re, float im) { return std::sqrt(re * re + im * im); });
    }

    void stft_power_batch(
        const float* frames,
        float* out_power,
        int num_frames,
        int frame_length,
        int fft_length)
    {
        rfft_reduce_batch_impl(frames, out_power, num_frames, frame_length, fft_length,
            [](float re, float im) { return re * re + im * im; });
    }

    // Auto-dispatching versions
    void rfft_batch_optimized(
        const float* frames,
        float* out_complex,
        int num_frames,
        int frame_length,
        int fft_length)
    {
        if (avx512::has_avx512f()) {
            avx512::rfft_batch_avx512(frames, out_complex, num_frames, frame_length, fft_length);
        } else {
            rfft_batch(frames, out_complex, num_frames, frame_length, fft_length);
        }
    }

    void rfft_magnitude_batch_optimized(
        const float* frames,
        float* out_magnitude,
        int num_frames,
        int frame_length,
        int fft_length)
    {
        if (avx512::has_avx512f()) {
            avx512::rfft_magnitude_batch_avx512(frames, out_magnitude, num_frames, frame_length, fft_length);
        } else {
            rfft_magnitude_batch(frames, out_magnitude, num_frames, frame_length, fft_length);
        }
    }

    void stft_power_batch_optimized(
        const float* frames,
        float* out_power,
        int num_frames,
        int frame_length,
        int fft_length)
    {
        if (avx512::has_avx512f()) {
            avx512::stft_power_batch_avx512(frames, out_power, num_frames, frame_length, fft_length);
        } else {
            stft_power_batch(frames, out_power, num_frames, frame_length, fft_length);
        }
    }

    // ---------------------------------------------------------------
    //  Mel filter bank — scalar implementation
    // ---------------------------------------------------------------

    std::vector<float> mel_filter_bank(
        int num_frequency_bins,
        int num_mel_filters,
        float min_frequency,
        float max_frequency,
        int sampling_rate,
        bool apply_slaney_norm,
        bool slaney_mel_scale)
    {
        // mel_freqs: num_mel_filters + 2 linearly spaced points in mel domain
        float mel_min = hertz_to_mel(min_frequency, slaney_mel_scale);
        float mel_max = hertz_to_mel(max_frequency, slaney_mel_scale);

        const int n_points = num_mel_filters + 2;
        std::vector<float> filter_freqs(n_points);
        for (int i = 0; i < n_points; ++i) {
            float mel = mel_min + (mel_max - mel_min) * i / (n_points - 1);
            filter_freqs[i] = mel_to_hertz(mel, slaney_mel_scale);
        }

        // fft_freqs: linearly spaced 0 .. sampling_rate/2
        std::vector<float> fft_freqs(num_frequency_bins);
        float half_sr = static_cast<float>(sampling_rate) / 2.0f;
        for (int i = 0; i < num_frequency_bins; ++i) {
            fft_freqs[i] = half_sr * i / (num_frequency_bins - 1);
        }

        // _create_triangular_filter_bank:
        //   filter_diff = np.diff(filter_freqs)                  [n_points-1]
        //   slopes = fft_freqs[:,None] - filter_freqs[None,:]    sign flipped below
        //   down_slopes = -slopes[:, :-2] / filter_diff[:-1]
        //   up_slopes   =  slopes[:, 2:]  / filter_diff[1:]
        //   result = max(0, min(down_slopes, up_slopes))

        std::vector<float> filter_diff(n_points - 1);
        for (int i = 0; i < n_points - 1; ++i) {
            filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
        }

        // Output: [num_frequency_bins x num_mel_filters], row-major
        std::vector<float> mel_filters(
            static_cast<size_t>(num_frequency_bins) * num_mel_filters, 0.0f);

        for (int b = 0; b < num_frequency_bins; ++b) {
            float f = fft_freqs[b];
            for (int m = 0; m < num_mel_filters; ++m) {
                // slope values
                float slope_left  = filter_freqs[m]     - f;   // slopes[:, m]
                float slope_right = filter_freqs[m + 2]  - f;  // slopes[:, m+2]

                float down = -slope_left  / filter_diff[m];     // down_slopes
                float up   =  slope_right / filter_diff[m + 1]; // up_slopes

                float val = std::max(0.0f, std::min(down, up));
                mel_filters[b * num_mel_filters + m] = val;
            }
        }

        // Optional Slaney area-normalization
        if (apply_slaney_norm) {
            for (int m = 0; m < num_mel_filters; ++m) {
                float enorm = 2.0f / (filter_freqs[m + 2] - filter_freqs[m]);
                for (int b = 0; b < num_frequency_bins; ++b) {
                    mel_filters[b * num_mel_filters + m] *= enorm;
                }
            }
        }

        return mel_filters;
    }

    // ---------------------------------------------------------------
    //  Mel spectrogram matmul — scalar implementation
    // ---------------------------------------------------------------

    void mel_spectrogram(
        const float* magnitude_spec,
        const float* mel_filters,
        float* out,
        int num_frames,
        int num_frequency_bins,
        int num_mel_filters)
    {
        // out[f, m] = sum_b( magnitude_spec[f, b] * mel_filters[b, m] )
        for (int f = 0; f < num_frames; ++f) {
            const float* spec_row = magnitude_spec + f * num_frequency_bins;
            float* out_row = out + f * num_mel_filters;

            for (int m = 0; m < num_mel_filters; ++m) {
                float sum = 0.0f;
                for (int b = 0; b < num_frequency_bins; ++b) {
                    sum += spec_row[b] * mel_filters[b * num_mel_filters + m];
                }
                out_row[m] = sum;
            }
        }
    }

    // ---------------------------------------------------------------
    //  Auto-dispatching optimized versions
    // ---------------------------------------------------------------

    std::vector<float> mel_filter_bank_optimized(
        int num_frequency_bins,
        int num_mel_filters,
        float min_frequency,
        float max_frequency,
        int sampling_rate,
        bool apply_slaney_norm,
        bool slaney_mel_scale)
    {
        // AVX512 path does not support Slaney mel scale yet — fall through to scalar.
        if (avx512::has_avx512f() && !slaney_mel_scale) {
            return avx512::mel_filter_bank_avx512(num_frequency_bins, num_mel_filters,
                                                   min_frequency, max_frequency,
                                                   sampling_rate, apply_slaney_norm);
        }
        return mel_filter_bank(num_frequency_bins, num_mel_filters,
                               min_frequency, max_frequency,
                               sampling_rate, apply_slaney_norm, slaney_mel_scale);
    }

    void mel_spectrogram_optimized(
        const float* magnitude_spec,
        const float* mel_filters,
        float* out,
        int num_frames,
        int num_frequency_bins,
        int num_mel_filters)
    {
        if (avx512::has_avx512f()) {
            avx512::mel_spectrogram_avx512(magnitude_spec, mel_filters, out,
                                            num_frames, num_frequency_bins, num_mel_filters);
        } else {
            mel_spectrogram(magnitude_spec, mel_filters, out,
                            num_frames, num_frequency_bins, num_mel_filters);
        }
    }

    // ---------------------------------------------------------------
    //  Window function — scalar implementation
    // ---------------------------------------------------------------

    std::vector<float> window_function(
        int window_length,
        const std::string& name,
        bool periodic,
        int frame_length,
        bool center)
    {
        const int length = periodic ? window_length + 1 : window_length;
        std::vector<float> window(length);

        if (name == "boxcar") {
            for (int i = 0; i < length; ++i) {
                window[i] = 1.0f;
            }
        } else if (name == "hamming" || name == "hamming_window") {
            // Hamming: 0.54 - 0.46 * cos(2*pi*n / (N-1))
            const float inv = (length > 1) ? 1.0f / (length - 1) : 0.0f;
            for (int i = 0; i < length; ++i) {
                window[i] = 0.54f - 0.46f * std::cos(2.0f * static_cast<float>(M_PI) * i * inv);
            }
        } else if (name == "hann" || name == "hann_window") {
            // Hann: 0.5 - 0.5 * cos(2*pi*n / (N-1))
            const float inv = (length > 1) ? 1.0f / (length - 1) : 0.0f;
            for (int i = 0; i < length; ++i) {
                window[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i * inv);
            }
        } else if (name == "povey") {
            // Povey: hann^0.85
            const float inv = (length > 1) ? 1.0f / (length - 1) : 0.0f;
            for (int i = 0; i < length; ++i) {
                float hann = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i * inv);
                window[i] = std::pow(hann, 0.85f);
            }
        } else {
            throw std::runtime_error("Unknown window function '" + name + "'");
        }

        // Drop last sample for periodic window
        if (periodic) {
            window.resize(window_length);
        }

        // If no frame_length requested, return the window as-is
        if (frame_length <= 0) {
            return window;
        }

        if (window_length > frame_length) {
            throw std::runtime_error("window_length (" + std::to_string(window_length)
                + ") may not be larger than frame_length (" + std::to_string(frame_length) + ")");
        }

        // Zero-pad / embed into frame_length buffer
        std::vector<float> padded(frame_length, 0.0f);
        int offset = center ? (frame_length - window_length) / 2 : 0;
        std::memcpy(padded.data() + offset, window.data(), window_length * sizeof(float));
        return padded;
    }

    // ---------------------------------------------------------------
    //  Auto-dispatching optimized window_function
    // ---------------------------------------------------------------

    std::vector<float> window_function_optimized(
        int window_length,
        const std::string& name,
        bool periodic,
        int frame_length,
        bool center)
    {
        if (avx512::has_avx512f()) {
            return avx512::window_function_avx512(window_length, name, periodic, frame_length, center);
        } else {
            return window_function(window_length, name, periodic, frame_length, center);
        }
    }

    // ---------------------------------------------------------------
    //  Scalar apply_window_frames
    // ---------------------------------------------------------------

    void apply_window_frames(
        const float* waveform,
        const float* window,
        float* out_windowed,
        int num_frames,
        int frame_length,
        int hop_length)
    {
        for (int f = 0; f < num_frames; f++) {
            const int offset = f * hop_length;
            for (int n = 0; n < frame_length; n++) {
                out_windowed[f * frame_length + n] = waveform[offset + n] * window[n];
            }
        }
    }

    void apply_window_frames_optimized(
        const float* waveform,
        const float* window,
        float* out_windowed,
        int num_frames,
        int frame_length,
        int hop_length)
    {
        if (avx512::has_avx512f()) {
            avx512::apply_window_frames_avx512(waveform, window, out_windowed, num_frames, frame_length, hop_length);
        } else {
            apply_window_frames(waveform, window, out_windowed, num_frames, frame_length, hop_length);
        }
    }

    // ---------------------------------------------------------------
    //  High-level STFT power spectrum
    // ---------------------------------------------------------------

    int stft_num_frames(int n_samples, int n_fft, int hop_length, bool center)
    {
        const int effective = center ? n_samples + n_fft : n_samples;
        return (effective - n_fft) / hop_length + 1;
    }

    static void reflect_pad(const float* src, int n, int pad, float* dst)
    {
        // dst has length n + 2*pad.  Left pad: src[1..pad] reversed.
        // Right pad: src[n-2..n-1-pad] reversed.
        for (int i = 0; i < pad; i++)
            dst[pad - 1 - i] = (i + 1 < n) ? src[i + 1] : 0.0f;
        std::memcpy(dst + pad, src, static_cast<size_t>(n) * sizeof(float));
        for (int i = 0; i < pad; i++)
            dst[pad + n + i] = (n - 2 - i >= 0) ? src[n - 2 - i] : 0.0f;
    }

    int stft_power(
        const float* waveform,
        int          n_samples,
        const float* window,
        int          n_fft,
        int          hop_length,
        bool         center,
        StftPadMode  pad_mode,
        float*       out_power)
    {
        const int num_frames = stft_num_frames(n_samples, n_fft, hop_length, center);
        if (num_frames <= 0) return 0;

        if (!center) {
            std::vector<float> windowed(static_cast<size_t>(num_frames) * n_fft);
            apply_window_frames(waveform, window, windowed.data(), num_frames, n_fft, hop_length);
            stft_power_batch(windowed.data(), out_power, num_frames, n_fft, n_fft);
            return num_frames;
        }

        // center=true: pad n_fft/2 on each side then run uncentered
        const int pad = n_fft / 2;
        const int padded_len = n_samples + 2 * pad;
        std::vector<float> padded(padded_len, 0.0f);

        if (pad_mode == StftPadMode::reflect) {
            reflect_pad(waveform, n_samples, pad, padded.data());
        } else {
            std::memcpy(padded.data() + pad, waveform, static_cast<size_t>(n_samples) * sizeof(float));
        }

        std::vector<float> windowed(static_cast<size_t>(num_frames) * n_fft);
        apply_window_frames(padded.data(), window, windowed.data(), num_frames, n_fft, hop_length);
        stft_power_batch(windowed.data(), out_power, num_frames, n_fft, n_fft);
        return num_frames;
    }

    int stft_power_optimized(
        const float* waveform,
        int          n_samples,
        const float* window,
        int          n_fft,
        int          hop_length,
        bool         center,
        StftPadMode  pad_mode,
        float*       out_power)
    {
        const int num_frames = stft_num_frames(n_samples, n_fft, hop_length, center);
        if (num_frames <= 0) return 0;

        if (!center) {
            std::vector<float> windowed(static_cast<size_t>(num_frames) * n_fft);
            apply_window_frames_optimized(waveform, window, windowed.data(), num_frames, n_fft, hop_length);
            stft_power_batch_optimized(windowed.data(), out_power, num_frames, n_fft, n_fft);
            return num_frames;
        }

        const int pad = n_fft / 2;
        const int padded_len = n_samples + 2 * pad;
        std::vector<float> padded(padded_len, 0.0f);

        if (pad_mode == StftPadMode::reflect) {
            reflect_pad(waveform, n_samples, pad, padded.data());
        } else {
            std::memcpy(padded.data() + pad, waveform, static_cast<size_t>(n_samples) * sizeof(float));
        }

        std::vector<float> windowed(static_cast<size_t>(num_frames) * n_fft);
        apply_window_frames_optimized(padded.data(), window, windowed.data(), num_frames, n_fft, hop_length);
        stft_power_batch_optimized(windowed.data(), out_power, num_frames, n_fft, n_fft);
        return num_frames;
    }

    // ---------------------------------------------------------------
    //  reduce_max / clamp_below_max / affine_scale
    // ---------------------------------------------------------------

    float reduce_max(const float* in, int count)
    {
        assert(count > 0);
        float m = in[0];
        for (int i = 1; i < count; i++)
            m = std::max(m, in[i]);
        return m;
    }

    void clamp_below_max(float* inout, int count, float max_val, float dynamic_range)
    {
        float threshold = max_val - dynamic_range;
        for (int i = 0; i < count; i++)
            inout[i] = std::max(inout[i], threshold);
    }

    void affine_scale(float* inout, int count, float offset, float scale)
    {
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < count; i++)
            inout[i] = (inout[i] + offset) * inv_scale;
    }

    // ---------------------------------------------------------------
    //  Scalar log_mel_floor
    //  UseClamp=false: out[i] = log_base(in[i] + floor)
    //  UseClamp=true:  out[i] = log_base(max(in[i], floor))
    //  Base=0: natural log, Base=10: log10
    // ---------------------------------------------------------------

    template<bool UseClamp, int Base>
    void log_mel_floor(
        const float* in,
        float* out,
        int count,
        float floor)
    {
        const float log_base_inv = (Base > 1) ? (1.0f / std::log(static_cast<float>(Base))) : 1.0f;
        for (int i = 0; i < count; i++) {
            float x = UseClamp ? std::max(in[i], floor) : (in[i] + floor);
            out[i] = (Base > 1) ? std::log(x) * log_base_inv : std::log(x);
        }
    }

    // Explicit instantiations needed for linker
    template void log_mel_floor<false, 0>(const float*, float*, int, float);
    template void log_mel_floor<true,  0>(const float*, float*, int, float);
    template void log_mel_floor<false, 10>(const float*, float*, int, float);
    template void log_mel_floor<true,  10>(const float*, float*, int, float);

    template<bool UseClamp, int Base>
    void log_mel_floor_optimized(
        const float* in,
        float* out,
        int count,
        float floor)
    {
        if (avx512::has_avx512f()) {
            avx512::log_mel_floor_avx512<UseClamp, Base>(in, out, count, floor);
        } else {
            log_mel_floor<UseClamp, Base>(in, out, count, floor);
        }
    }

    template void log_mel_floor_optimized<false, 0>(const float*, float*, int, float);
    template void log_mel_floor_optimized<true,  0>(const float*, float*, int, float);
    template void log_mel_floor_optimized<false, 10>(const float*, float*, int, float);
    template void log_mel_floor_optimized<true,  10>(const float*, float*, int, float);

} // namespace audioproc
