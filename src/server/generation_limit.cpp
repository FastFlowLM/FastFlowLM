#include <server/generation_limit.hpp>

#include <stdexcept>

namespace {

constexpr int kLegacyDefaultGenerationLimit = 4096;
constexpr int kNoExplicitGenerationLimit = -1;

constexpr std::array<GenerationRoute, 4> kGenerationRoutes{{
    {"POST", "/api/generate", GenerationEndpoint::Generate},
    {"POST", "/api/chat", GenerationEndpoint::OllamaChat},
    {"POST",
     "/v1/chat/completions",
     GenerationEndpoint::OpenAiChatCompletion},
    {"POST", "/v1/completions", GenerationEndpoint::OpenAiCompletion},
}};

ParsedGenerationLimit ParseField(
    const nlohmann::ordered_json& request,
    std::string_view field) {
    if (!request.contains(field)) {
        return {false, kNoExplicitGenerationLimit};
    }
    return {true, request.at(field).get<int>()};
}

}  // namespace

std::span<const GenerationRoute> GenerationRoutes() noexcept {
    return kGenerationRoutes;
}

std::optional<GenerationEndpoint> GenerationEndpointForRoute(
    std::string_view method,
    std::string_view path) noexcept {
    for (const GenerationRoute& route : kGenerationRoutes) {
        if (route.method == method && route.path == path) {
            return route.endpoint;
        }
    }
    return std::nullopt;
}

GenerationEndpoint RequireGenerationEndpoint(
    std::string_view method,
    std::string_view path) {
    const auto endpoint = GenerationEndpointForRoute(method, path);
    if (!endpoint.has_value()) {
        throw std::logic_error(
            "generation route " + std::string(method) + " " +
            std::string(path) +
            " is missing from GenerationRoutes(); add it there so its "
            "limit field and admission rule are declared");
    }
    return *endpoint;
}

ParsedGenerationLimit ParseGenerationLimit(
    const nlohmann::ordered_json& request,
    GenerationEndpoint endpoint) {
    switch (endpoint) {
        case GenerationEndpoint::Generate:
        case GenerationEndpoint::OpenAiCompletion:
            return ParseField(request, "max_tokens");
        case GenerationEndpoint::OllamaChat: {
            // Ollama nests the limit, and reads it with the same
            // presence/absence rule as the flat endpoints.
            const nlohmann::ordered_json options =
                request.value(
                    "options",
                    nlohmann::ordered_json::object());
            return ParseField(options, "num_predict");
        }
        case GenerationEndpoint::OpenAiChatCompletion: {
            const ParsedGenerationLimit max_tokens =
                ParseField(request, "max_tokens");
            if (max_tokens.explicit_limit) {
                return max_tokens;
            }
            return ParseField(request, "max_completion_tokens");
        }
    }
    return {false, kNoExplicitGenerationLimit};
}

int GenerationLoopLimit(
    const ParsedGenerationLimit& parsed,
    bool uses_corelib_aie4) noexcept {
    if (parsed.explicit_limit) {
        return parsed.value;
    }
    return uses_corelib_aie4
               ? kNoExplicitGenerationLimit
               : kLegacyDefaultGenerationLimit;
}

std::optional<int> RequestedMaxNewTokens(
    const ParsedGenerationLimit& parsed) noexcept {
    if (!parsed.explicit_limit) {
        return std::nullopt;
    }
    return parsed.value;
}

int OllamaChatGenerationLoopLimit(
    const nlohmann::ordered_json& request) {
    // Retained as the legacy (non-AIE4) spelling, and defined in terms of
    // the shared rule so the two cannot drift apart.
    return GenerationLoopLimit(
        ParseGenerationLimit(request, GenerationEndpoint::OllamaChat),
        false);
}

nlohmann::ordered_json ModelErrorResponse(
    std::string_view message,
    int http_code,
    bool session_cleared) {
    return {
        {"error", {
            {"message", std::string(message)},
            {"type", http_code == 400
                         ? "invalid_request_error"
                         : "server_error"},
            {"code", http_code},
            {"session_cleared", session_cleared},
        }},
    };
}

int HttpStatusForResponse(
    const nlohmann::ordered_json& response) noexcept {
    try {
        if (!response.is_object() || !response.contains("error")) {
            return 200;
        }
        const auto& error = response.at("error");
        if (!error.is_object() || !error.contains("code")) {
            return 200;
        }
        const int code = error.at("code").get<int>();
        return code == 400 || code == 500 ? code : 200;
    } catch (...) {
        return 200;
    }
}

std::array<std::string, 2> OpenAiStreamingErrorFrames(
    const nlohmann::ordered_json& error_response) {
    return {
        "data: " + error_response.dump() + "\n\n",
        "data: [DONE]\n\n",
    };
}

void SendOpenAiStreamingError(
    const nlohmann::ordered_json& error_response,
    const std::function<void(
        const nlohmann::ordered_json&,
        bool)>& send_streaming_response) {
    const auto frames =
        OpenAiStreamingErrorFrames(error_response);
    send_streaming_response(
        nlohmann::ordered_json(frames[0]),
        false);
    send_streaming_response(
        nlohmann::ordered_json(frames[1]),
        true);
}

bool UseFinalStreamingErrorChunk(
    bool stream_started) noexcept {
    return stream_started;
}

std::optional<int> CliRequestedMaxNewTokens(
    int generate_limit) noexcept {
    if (generate_limit <= 0) {
        return std::nullopt;
    }
    return generate_limit;
}

std::string CliModelErrorNotice(
    std::string_view message,
    bool session_cleared) {
    if (session_cleared) {
        return
            "ERROR: AIE4 inference failed before submission; the current "
            "conversation was cleared.";
    }
    return "ERROR: " + std::string(message);
}

bool IsCorelibAie4ModelInfo(
    const nlohmann::ordered_json& model_info) noexcept {
    try {
        if (!model_info.is_object() ||
            !model_info.contains("details")) {
            return false;
        }
        const auto& details = model_info.at("details");
        return details.is_object() &&
               details.value("execution_backend", std::string{}) ==
                   "corelib_aie4";
    } catch (...) {
        return false;
    }
}
