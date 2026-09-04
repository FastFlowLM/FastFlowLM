/// \file qwen3_asr_utils.hpp
/// \brief Prompt and output helpers for Qwen3-ASR.
#pragma once

#include <optional>
#include <string>
#include <string_view>

struct qwen3_asr_result_t {
    std::string language;
    std::string text;
};

/// Normalize a language name to the canonical Qwen3-ASR spelling.
/// Returns an empty string for an empty input.
std::string qwen3_asr_normalize_language(std::string_view language);

/// Whether a canonical or case-insensitive language name is supported.
bool qwen3_asr_is_supported_language(std::string_view language);

/// Build the exact single-audio prompt used by Qwen3-ASR's chat template.
/// When forced_language is set, generation starts after
/// "language <Language><asr_text>" and decoded output is text-only.
std::string qwen3_asr_build_prompt(
    std::string_view context,
    int audio_token_count,
    std::optional<std::string_view> forced_language = std::nullopt);

/// Parse generated text into language and transcription.
/// If forced_language is provided, raw_output is treated as transcription-only.
qwen3_asr_result_t qwen3_asr_parse_output(
    std::string_view raw_output,
    std::optional<std::string_view> forced_language = std::nullopt);
