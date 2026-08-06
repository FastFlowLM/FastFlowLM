/// \file qwen3_asr_utils.cpp
/// \brief Prompt and output helpers for Qwen3-ASR.

#include "audio/qwen3_asr_utils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>

namespace {

constexpr std::array<std::string_view, 30> supported_languages = {
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch",
    "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian",
    "Greek", "Romanian", "Hungarian", "Macedonian",
};

std::string trim(std::string_view value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string ascii_lower(std::string_view value) {
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

} // namespace

std::string qwen3_asr_normalize_language(std::string_view language) {
    std::string result = trim(language);
    if (result.empty()) return result;

    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    result.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(result.front())));
    return result;
}

bool qwen3_asr_is_supported_language(std::string_view language) {
    const std::string normalized = qwen3_asr_normalize_language(language);
    return std::find(supported_languages.begin(), supported_languages.end(), normalized) != supported_languages.end();
}

std::string qwen3_asr_build_prompt(
    std::string_view context,
    int audio_token_count,
    std::optional<std::string_view> forced_language) {
    if (audio_token_count <= 0) {
        throw std::invalid_argument("audio_token_count must be positive");
    }

    std::string language;
    if (forced_language.has_value()) {
        language = qwen3_asr_normalize_language(*forced_language);
        if (!qwen3_asr_is_supported_language(language)) {
            throw std::invalid_argument("unsupported Qwen3-ASR language: " + language);
        }
    }

    constexpr std::string_view audio_token = "<|audio_pad|>";
    std::string prompt;
    prompt.reserve(context.size() + static_cast<std::size_t>(audio_token_count) * audio_token.size() + 160);
    prompt += "<|im_start|>system\n";
    prompt += context;
    prompt += "<|im_end|>\n<|im_start|>user\n<|audio_start|>";
    for (int i = 0; i < audio_token_count; ++i) prompt += audio_token;
    prompt += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";

    if (!language.empty()) {
        prompt += "language ";
        prompt += language;
        prompt += "<asr_text>";
    }
    return prompt;
}

qwen3_asr_result_t qwen3_asr_parse_output(
    std::string_view raw_output,
    std::optional<std::string_view> forced_language) {
    qwen3_asr_result_t result;
    const std::string raw = trim(raw_output);
    if (raw.empty()) return result;

    if (forced_language.has_value()) {
        result.language = qwen3_asr_normalize_language(*forced_language);
        result.text = raw;
        return result;
    }

    constexpr std::string_view asr_tag = "<asr_text>";
    const std::size_t tag_position = raw.find(asr_tag);
    if (tag_position == std::string::npos) {
        result.text = raw;
        return result;
    }

    const std::string metadata = trim(std::string_view(raw).substr(0, tag_position));
    result.text = trim(std::string_view(raw).substr(tag_position + asr_tag.size()));

    const std::string metadata_lower = ascii_lower(metadata);
    if (metadata_lower.find("language none") != std::string::npos) {
        // Silent/empty audio. Preserve unexpected text but do not claim a language.
        return result;
    }

    std::size_t line_start = 0;
    while (line_start <= metadata.size()) {
        const std::size_t line_end = metadata.find('\n', line_start);
        const std::string line = trim(std::string_view(metadata).substr(
            line_start,
            line_end == std::string::npos ? std::string::npos : line_end - line_start));
        const std::string line_lower = ascii_lower(line);
        constexpr std::string_view prefix = "language ";
        if (line_lower.starts_with(prefix)) {
            result.language = qwen3_asr_normalize_language(std::string_view(line).substr(prefix.size()));
            break;
        }
        if (line_end == std::string::npos) break;
        line_start = line_end + 1;
    }

    return result;
}
