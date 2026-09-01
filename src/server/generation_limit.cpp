#include <server/generation_limit.hpp>

namespace {

constexpr int kLegacyDefaultGenerationLimit = 4096;
constexpr int kNoExplicitGenerationLimit = -1;

ParsedGenerationLimit ParseField(
    const nlohmann::ordered_json& request,
    std::string_view field) {
    if (!request.contains(field)) {
        return {false, kNoExplicitGenerationLimit};
    }
    return {true, request.at(field).get<int>()};
}

}  // namespace

ParsedGenerationLimit ParseGenerationLimit(
    const nlohmann::ordered_json& request,
    GenerationEndpoint endpoint) {
    switch (endpoint) {
        case GenerationEndpoint::Generate:
        case GenerationEndpoint::OpenAiCompletion:
            return ParseField(request, "max_tokens");
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
    const nlohmann::ordered_json options =
        request.value(
            "options",
            nlohmann::ordered_json::object());
    return options.value(
        "num_predict",
        kLegacyDefaultGenerationLimit);
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
