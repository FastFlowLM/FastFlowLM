#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>

enum class GenerationEndpoint {
    Generate,
    OllamaChat,
    OpenAiChatCompletion,
    OpenAiCompletion
};

// Every route whose handler reaches the causal engine's generate() or
// generate_with_prompt(), and the field each one reads its limit from.
//
// This table exists because /api/chat was missed. Three of the four
// endpoints were wired to ParseGenerationLimit by inspection, and
// inspection is what let the fourth keep its own private limit path --
// bypassing explicit-limit detection, the admission rule, the HTTP 400
// path and session_cleared reporting on the AIE4 tag.
struct GenerationRoute {
    std::string_view method;
    std::string_view path;
    GenerationEndpoint endpoint;
};

std::span<const GenerationRoute> GenerationRoutes() noexcept;

std::optional<GenerationEndpoint> GenerationEndpointForRoute(
    std::string_view method,
    std::string_view path) noexcept;

// Throws when the route is absent from the table, so a generation route
// cannot reach the engine without declaring how its limit is parsed.
GenerationEndpoint RequireGenerationEndpoint(
    std::string_view method,
    std::string_view path);

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

std::array<std::string, 2> OpenAiStreamingErrorFrames(
    const nlohmann::ordered_json& error_response);

void SendOpenAiStreamingError(
    const nlohmann::ordered_json& error_response,
    const std::function<void(
        const nlohmann::ordered_json&,
        bool)>& send_streaming_response);

bool UseFinalStreamingErrorChunk(
    bool stream_started) noexcept;

std::optional<int> CliRequestedMaxNewTokens(
    int generate_limit) noexcept;

std::string CliModelErrorNotice(
    std::string_view message,
    bool session_cleared);

bool IsCorelibAie4ModelInfo(
    const nlohmann::ordered_json& model_info) noexcept;
