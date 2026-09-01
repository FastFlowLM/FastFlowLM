#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>

enum class GenerationEndpoint {
    Generate,
    OpenAiChatCompletion,
    OpenAiCompletion
};

struct ParsedGenerationLimit {
    bool explicit_limit;
    int value;
};

ParsedGenerationLimit ParseGenerationLimit(
    const nlohmann::ordered_json& request,
    GenerationEndpoint endpoint);

int GenerationLoopLimit(
    const ParsedGenerationLimit& parsed,
    bool uses_corelib_aie4) noexcept;

std::optional<int> RequestedMaxNewTokens(
    const ParsedGenerationLimit& parsed) noexcept;

int OllamaChatGenerationLoopLimit(
    const nlohmann::ordered_json& request);

nlohmann::ordered_json ModelErrorResponse(
    std::string_view message,
    int http_code,
    bool session_cleared);

int HttpStatusForResponse(
    const nlohmann::ordered_json& response) noexcept;

bool UseFinalStreamingErrorChunk(
    bool stream_started) noexcept;

std::optional<int> CliRequestedMaxNewTokens(
    int generate_limit) noexcept;

std::string CliModelErrorNotice(
    std::string_view message,
    bool session_cleared);

bool IsCorelibAie4ModelInfo(
    const nlohmann::ordered_json& model_info) noexcept;
