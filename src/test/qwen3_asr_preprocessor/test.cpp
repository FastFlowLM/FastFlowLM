#include "audio/qwen3_asr_preprocessor.hpp"
#include "audio/qwen3_asr_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

std::vector<float> make_waveform(int num_samples) {
    // Deterministic broadband signal. Using integer-generated noise avoids
    // platform-specific libm differences in the C++/Python parity fixture.
    std::vector<float> samples(num_samples);
    std::uint32_t state = 0x12345678U;
    for (int i = 0; i < num_samples; ++i) {
        state = state * 1664525U + 1013904223U;
        const float unit = static_cast<float>(state >> 8) / 16777216.0f;
        samples[i] = unit * 0.4f - 0.2f;
    }
    return samples;
}

bool write_features(const std::string& path, const qwen3_asr_features_t& features) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;

    const std::int32_t header[] = {
        features.num_mel_bins,
        features.num_frames,
        features.valid_frames,
    };
    output.write(reinterpret_cast<const char*>(header), sizeof(header));
    output.write(
        reinterpret_cast<const char*>(features.input_features.data()),
        static_cast<std::streamsize>(features.input_features.size() * sizeof(float)));
    return output.good();
}

bool expect(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

} // namespace

int main(int argc, char** argv) {
    bool ok = true;

    ok &= expect(Qwen3ASRPreprocessor::feature_output_length(0) == 0, "0 mel frames");
    ok &= expect(Qwen3ASRPreprocessor::feature_output_length(1) == 1, "1 mel frame");
    ok &= expect(Qwen3ASRPreprocessor::feature_output_length(8) == 1, "8 mel frames");
    ok &= expect(Qwen3ASRPreprocessor::feature_output_length(9) == 2, "9 mel frames");
    ok &= expect(Qwen3ASRPreprocessor::feature_output_length(99) == 13, "99 mel frames");
    ok &= expect(Qwen3ASRPreprocessor::feature_output_length(100) == 13, "100 mel frames");
    ok &= expect(Qwen3ASRPreprocessor::feature_output_length(101) == 14, "101 mel frames");
    ok &= expect(Qwen3ASRPreprocessor::feature_output_length(3000) == 390, "3000 mel frames");

    ok &= expect(qwen3_asr_normalize_language("  cHINese ") == "Chinese", "normalize language");
    ok &= expect(qwen3_asr_is_supported_language("ENGLISH"), "supported language");
    ok &= expect(!qwen3_asr_is_supported_language("Klingon"), "unsupported language");

    const std::string prompt = qwen3_asr_build_prompt("hot words", 2, "english");
    ok &= expect(prompt ==
        "<|im_start|>system\nhot words<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage English<asr_text>",
        "Qwen3-ASR chat prompt");

    const auto parsed = qwen3_asr_parse_output("language Chinese<asr_text>你好，世界。 ");
    ok &= expect(parsed.language == "Chinese" && parsed.text == "你好，世界。", "parse tagged output");
    const auto silent = qwen3_asr_parse_output("language None<asr_text>");
    ok &= expect(silent.language.empty() && silent.text.empty(), "parse silent audio");
    const auto forced = qwen3_asr_parse_output("plain transcription", "French");
    ok &= expect(forced.language == "French" && forced.text == "plain transcription", "parse forced language output");

    Qwen3ASRPreprocessor preprocessor;

    const auto short_features = preprocessor.extract(make_waveform(4000));
    ok &= expect(short_features.valid_frames == 50, "min_length creates 50 valid mel frames");
    ok &= expect(short_features.num_frames == 100, "mel frames pad to 2*n_window");
    ok &= expect(short_features.num_mel_bins == 128, "128 mel bins");
    ok &= expect(short_features.input_features.size() == 128U * 100U, "feature tensor shape");
    ok &= expect(std::accumulate(short_features.attention_mask.begin(), short_features.attention_mask.end(), 0) == 50,
                 "attention mask preserves valid frame count");

    const auto features = preprocessor.extract(make_waveform(16000));
    ok &= expect(features.valid_frames == 100, "one second creates 100 mel frames");
    ok &= expect(features.num_frames == 100, "one second needs no mel padding");
    ok &= expect(features.input_features.size() == 128U * 100U, "one-second feature tensor shape");
    ok &= expect(std::all_of(features.input_features.begin(), features.input_features.end(),
                            [](float value) { return std::isfinite(value); }),
                 "all features are finite");

    if (argc == 3 && std::string(argv[1]) == "--dump") {
        ok &= expect(write_features(argv[2], features), "write parity fixture");
    }

    if (!ok) return 1;
    std::cout << "Qwen3-ASR preprocessing checks passed\n";
    return 0;
}
