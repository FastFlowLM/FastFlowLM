#include <server/generation_limit.hpp>
#include <server/npu_access_manager.hpp>
#include <server/serve_lifecycle.hpp>

#include "test_support.hpp"

#include <array>
#include <csignal>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using nlohmann::ordered_json;

void CheckParsed(
    const ParsedGenerationLimit& parsed,
    bool explicit_limit,
    int value) {
    CHECK(parsed.explicit_limit == explicit_limit);
    CHECK(parsed.value == value);
}

void TestHandlerSpecificPresence() {
    CheckParsed(
        ParseGenerationLimit(
            ordered_json{
                {"max_tokens", 11},
                {"max_completion_tokens", 22}},
            GenerationEndpoint::Generate),
        true,
        11);
    CheckParsed(
        ParseGenerationLimit(
            ordered_json{{"max_completion_tokens", 22}},
            GenerationEndpoint::Generate),
        false,
        -1);

    CheckParsed(
        ParseGenerationLimit(
            ordered_json{
                {"max_tokens", 31},
                {"max_completion_tokens", 32}},
            GenerationEndpoint::OpenAiChatCompletion),
        true,
        31);
    CheckParsed(
        ParseGenerationLimit(
            ordered_json{{"max_completion_tokens", 32}},
            GenerationEndpoint::OpenAiChatCompletion),
        true,
        32);

    CheckParsed(
        ParseGenerationLimit(
            ordered_json{
                {"max_tokens", 41},
                {"max_completion_tokens", 42}},
            GenerationEndpoint::OpenAiCompletion),
        true,
        41);
    CheckParsed(
        ParseGenerationLimit(
            ordered_json{{"max_completion_tokens", 42}},
            GenerationEndpoint::OpenAiCompletion),
        false,
        -1);

    // /api/chat nests its limit, and it is now parsed by the same rule
    // rather than by a private one.
    CheckParsed(
        ParseGenerationLimit(
            ordered_json{{"options", {{"num_predict", 51}}}},
            GenerationEndpoint::OllamaChat),
        true,
        51);
    CheckParsed(
        ParseGenerationLimit(
            ordered_json{{"options", ordered_json::object()}},
            GenerationEndpoint::OllamaChat),
        false,
        -1);
    // A flat max_tokens is not the Ollama field and must not be read.
    CheckParsed(
        ParseGenerationLimit(
            ordered_json{{"max_tokens", 52}},
            GenerationEndpoint::OllamaChat),
        false,
        -1);

    for (const GenerationEndpoint endpoint : {
             GenerationEndpoint::Generate,
             GenerationEndpoint::OllamaChat,
             GenerationEndpoint::OpenAiChatCompletion,
             GenerationEndpoint::OpenAiCompletion}) {
        CheckParsed(
            ParseGenerationLimit(ordered_json::object(), endpoint),
            false,
            -1);
    }
}

void TestEndpointDefaultsAndPropagation() {
    const ParsedGenerationLimit omitted{false, -1};
    const ParsedGenerationLimit explicit_limit{true, 73};

    CHECK(GenerationLoopLimit(omitted, true) == -1);
    CHECK(GenerationLoopLimit(omitted, false) == 4096);
    CHECK(GenerationLoopLimit(explicit_limit, true) == 73);
    CHECK(GenerationLoopLimit(explicit_limit, false) == 73);

    CHECK(!RequestedMaxNewTokens(omitted).has_value());
    CHECK(RequestedMaxNewTokens(explicit_limit) == std::optional<int>(73));
}

// Legacy behaviour on /api/chat must not change: an omitted num_predict
// is still the 4096 soft bound, and an explicit one is still honoured.
void TestOllamaChatLegacyLimitIsUnchanged() {
    const ordered_json default_request = {
        {"options", ordered_json::object()},
    };
    CHECK(OllamaChatGenerationLoopLimit(default_request) == 4096);
    CHECK(OllamaChatGenerationLoopLimit(ordered_json::object()) == 4096);

    const ordered_json explicit_request = {
        {"options", {{"num_predict", 91}}},
    };
    CHECK(OllamaChatGenerationLoopLimit(explicit_request) == 91);

    // The legacy spelling and the shared rule are the same function.
    for (const ordered_json& request :
         {default_request, explicit_request, ordered_json::object()}) {
        const ParsedGenerationLimit parsed =
            ParseGenerationLimit(
                request,
                GenerationEndpoint::OllamaChat);
        CHECK(
            GenerationLoopLimit(parsed, false) ==
            OllamaChatGenerationLoopLimit(request));
    }
}

// On AIE4 an omitted /api/chat limit must mean "until the context cap",
// not the legacy 4096, and an explicit one must reach the admission rule
// through requested_max_new_tokens. Both were bypassed before.
void TestOllamaChatAie4LimitAndAdmission() {
    const ordered_json omitted = {{"options", ordered_json::object()}};
    const ParsedGenerationLimit omitted_parsed =
        ParseGenerationLimit(omitted, GenerationEndpoint::OllamaChat);
    CHECK(GenerationLoopLimit(omitted_parsed, true) == -1);
    CHECK(!RequestedMaxNewTokens(omitted_parsed).has_value());

    const ordered_json over_limit = {
        {"options", {{"num_predict", 8192}}},
    };
    const ParsedGenerationLimit over_parsed =
        ParseGenerationLimit(over_limit, GenerationEndpoint::OllamaChat);
    CHECK(GenerationLoopLimit(over_parsed, true) == 8192);
    CHECK(
        RequestedMaxNewTokens(over_parsed) == std::optional<int>(8192));
}

// The rule the /api/chat gap was missing. Three of four endpoints were
// covered by inspection; this derives the set instead of restating it.
//
// It reads the two production sources, finds every RestHandler::handle_*
// whose body reaches the causal engine's generate() or
// generate_with_prompt(), and maps those handlers to the routes server.cpp
// registers for them. Each such handler must then satisfy two conditions,
// because the defect that motivated this rule had BOTH shapes available
// and took the second:
//
//   1. its route is declared in GenerationRoutes() -- catches a fifth
//      endpoint appearing with no declaration at all; and
//   2. it obtains its limit through the shared functions --
//      ParseGenerationLimit, GenerationLoopLimit and RequestedMaxNewTokens
//      -- and calls no other *GenerationLoopLimit.
//
// handle_chat was declared and routed all along. What it did wrong was
// compute its limit privately, through OllamaChatGenerationLoopLimit, and
// a rule that only checked the route set would have passed it. Condition 2
// is what makes this guard able to catch the bug it exists for.
std::string ReadSource(const char* relative) {
    const std::filesystem::path path =
        std::filesystem::path(FLM_TEST_SOURCE_DIR) / relative;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error(
            "failed to read " + path.string());
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

struct GeneratingHandler {
    std::string name;
    std::string body;
};

std::vector<GeneratingHandler> GeneratingHandlers() {
    const std::string source = ReadSource("server/rest_handler.cpp");
    std::vector<GeneratingHandler> generating;
    constexpr std::string_view kDefinition = "void RestHandler::handle_";
    for (std::size_t at = source.find(kDefinition);
         at != std::string::npos;
         at = source.find(kDefinition, at + 1)) {
        const std::size_t name_start =
            at + std::string("void RestHandler::").size();
        const std::size_t name_end = source.find('(', name_start);
        if (name_end == std::string::npos) {
            continue;
        }
        const std::string name =
            source.substr(name_start, name_end - name_start);
        const std::size_t body_end =
            source.find(kDefinition, at + 1);
        const std::string body = source.substr(
            at,
            body_end == std::string::npos
                ? std::string::npos
                : body_end - at);
        if (
            body.find("auto_chat_engine->generate(") !=
                std::string::npos ||
            body.find("auto_chat_engine->generate_with_prompt(") !=
                std::string::npos) {
            generating.push_back({name, body});
        }
    }
    return generating;
}

bool IsIdentifierCharacter(char value) noexcept {
    return value == '_' ||
           (value >= '0' && value <= '9') ||
           (value >= 'a' && value <= 'z') ||
           (value >= 'A' && value <= 'Z');
}

// True when `body` calls something whose name ENDS in the given suffix but
// is not exactly it -- OllamaChatGenerationLoopLimit for
// GenerationLoopLimit, say. That is the private-path shape.
std::string FindQualifiedVariantCall(
    const std::string& body,
    const std::string& call) {
    for (std::size_t at = body.find(call);
         at != std::string::npos;
         at = body.find(call, at + 1)) {
        if (at == 0 || !IsIdentifierCharacter(body[at - 1])) {
            continue;
        }
        std::size_t start = at;
        while (start > 0 && IsIdentifierCharacter(body[start - 1])) {
            --start;
        }
        return body.substr(start, at + call.size() - start - 1);
    }
    return {};
}

// The (method, path) the handler declares to RequireGenerationEndpoint.
std::pair<std::string, std::string> DeclaredRoute(
    const std::string& body) {
    constexpr std::string_view kCall = "RequireGenerationEndpoint(";
    const std::size_t at = body.find(kCall);
    if (at == std::string::npos) {
        return {};
    }
    const std::size_t method_start = body.find('"', at + kCall.size());
    if (method_start == std::string::npos) {
        return {};
    }
    const std::size_t method_end = body.find('"', method_start + 1);
    const std::size_t path_start = body.find('"', method_end + 1);
    const std::size_t path_end = body.find('"', path_start + 1);
    if (path_end == std::string::npos) {
        return {};
    }
    return {
        body.substr(method_start + 1, method_end - method_start - 1),
        body.substr(path_start + 1, path_end - path_start - 1)};
}

std::map<std::string, std::pair<std::string, std::string>>
RegisteredRoutesByHandler() {
    const std::string source = ReadSource("server/server.cpp");
    std::map<std::string, std::pair<std::string, std::string>> routes;
    constexpr std::string_view kRegister = "register_handler(\"";
    for (std::size_t at = source.find(kRegister);
         at != std::string::npos;
         at = source.find(kRegister, at + 1)) {
        std::size_t cursor = at + kRegister.size();
        const std::size_t method_end = source.find('"', cursor);
        if (method_end == std::string::npos) {
            continue;
        }
        const std::string method =
            source.substr(cursor, method_end - cursor);
        const std::size_t path_start = source.find('"', method_end + 1);
        if (path_start == std::string::npos) {
            continue;
        }
        const std::size_t path_end = source.find('"', path_start + 1);
        if (path_end == std::string::npos) {
            continue;
        }
        const std::string path =
            source.substr(path_start + 1, path_end - path_start - 1);

        // The handler this route dispatches to, before the next
        // registration begins.
        const std::size_t next = source.find(kRegister, at + 1);
        const std::string block = source.substr(
            path_end,
            next == std::string::npos
                ? std::string::npos
                : next - path_end);
        constexpr std::string_view kCall = "rest_handler->";
        const std::size_t call_at = block.find(kCall);
        if (call_at == std::string::npos) {
            continue;
        }
        const std::size_t call_start = call_at + kCall.size();
        const std::size_t call_end = block.find('(', call_start);
        if (call_end == std::string::npos) {
            continue;
        }
        routes.emplace(
            block.substr(call_start, call_end - call_start),
            std::pair<std::string, std::string>(method, path));
    }
    return routes;
}

void TestEveryGenerationRouteIsDeclared() {
    const auto generating = GeneratingHandlers();
    const auto registered = RegisteredRoutesByHandler();
    CHECK(!generating.empty());
    CHECK(!registered.empty());

    std::set<std::string> declared;
    for (const GenerationRoute& route : GenerationRoutes()) {
        declared.insert(
            std::string(route.method) + " " + std::string(route.path));
        CHECK(
            GenerationEndpointForRoute(route.method, route.path) ==
            std::optional<GenerationEndpoint>(route.endpoint));
        CHECK(
            requires_npu_access(
                std::string(route.method),
                std::string(route.path)));
    }

    std::set<std::string> discovered;
    for (const GeneratingHandler& handler : generating) {
        const auto found = registered.find(handler.name);
        if (found == registered.end()) {
            throw std::runtime_error(
                "generation handler " + handler.name +
                " is not registered on any route in server.cpp");
        }
        const std::string route =
            found->second.first + " " + found->second.second;
        discovered.insert(route);

        // Condition 1: the route is declared.
        if (!declared.contains(route)) {
            throw std::runtime_error(
                "route " + route + " reaches the causal engine but is "
                "missing from GenerationRoutes(); it would bypass the "
                "AIE4 admission rule");
        }

        // Condition 2: the handler goes through the shared rule. This is
        // the one that catches the /api/chat defect, which was declared
        // and routed but parsed its own limit.
        for (const std::string& required : {
                 std::string("ParseGenerationLimit("),
                 std::string("GenerationLoopLimit("),
                 std::string("RequestedMaxNewTokens("),
                 std::string("RequireGenerationEndpoint(")}) {
            if (handler.body.find(required) == std::string::npos) {
                throw std::runtime_error(
                    handler.name + " serves " + route +
                    " but never calls " + required +
                    "; a generation handler must obtain its limit "
                    "through the shared rule, not privately");
            }
        }
        const std::string variant =
            FindQualifiedVariantCall(
                handler.body,
                "GenerationLoopLimit(");
        if (!variant.empty()) {
            throw std::runtime_error(
                handler.name + " serves " + route + " and calls " +
                variant +
                "; that is the private limit path /api/chat used, and it "
                "bypasses the admission rule and the HTTP 400 response");
        }

        // The route it declares must be the route it is registered on, so
        // a copy-pasted path cannot silently select another endpoint's
        // limit field.
        const auto declared_route = DeclaredRoute(handler.body);
        if (declared_route != found->second) {
            throw std::runtime_error(
                handler.name + " is registered on " + route +
                " but declares RequireGenerationEndpoint(\"" +
                declared_route.first + "\", \"" +
                declared_route.second + "\")");
        }
        CHECK(
            GenerationEndpointForRoute(
                declared_route.first,
                declared_route.second)
                .has_value());
    }
    // And no declared route is stale.
    CHECK(discovered == declared);
    CHECK(declared.size() == 4);
    CHECK(declared.contains("POST /api/chat"));

    // A route that does not generate has no endpoint, and an unknown one
    // fails loudly rather than silently defaulting.
    CHECK(
        !GenerationEndpointForRoute("POST", "/v1/embeddings")
             .has_value());
    CHECK(
        !GenerationEndpointForRoute("GET", "/api/chat").has_value());
    bool threw = false;
    try {
        (void)RequireGenerationEndpoint("POST", "/api/not-a-route");
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(
        RequireGenerationEndpoint("POST", "/api/chat") ==
        GenerationEndpoint::OllamaChat);
}

void TestNestedModelErrorAndHttpStatus() {
    const ordered_json bad_request =
        ModelErrorResponse("too many tokens", 400, false);
    CHECK(bad_request["error"]["message"] == "too many tokens");
    CHECK(bad_request["error"]["type"] == "invalid_request_error");
    CHECK(bad_request["error"]["code"] == 400);
    CHECK(bad_request["error"]["session_cleared"] == false);
    CHECK(HttpStatusForResponse(bad_request) == 400);

    const ordered_json server_error =
        ModelErrorResponse("submission failed", 500, true);
    CHECK(server_error["error"]["message"] == "submission failed");
    CHECK(server_error["error"]["type"] == "server_error");
    CHECK(server_error["error"]["code"] == 500);
    CHECK(server_error["error"]["session_cleared"] == true);
    CHECK(HttpStatusForResponse(server_error) == 500);

    CHECK(HttpStatusForResponse(ordered_json{{"ok", true}}) == 200);
    CHECK(
        HttpStatusForResponse(
            ordered_json{{"error", "legacy string error"}}) == 200);
    CHECK(!UseFinalStreamingErrorChunk(false));
    CHECK(UseFinalStreamingErrorChunk(true));
}

void TestOpenAiStreamingErrorFramingAndParsing() {
    const ordered_json error =
        ModelErrorResponse("submission failed", 500, true);
    std::vector<std::pair<std::string, bool>> transmitted;
    SendOpenAiStreamingError(
        error,
        [&](const ordered_json& data, bool is_final) {
            CHECK(data.is_string());
            transmitted.emplace_back(
                data.get<std::string>(),
                is_final);
        });
    const std::string expected_error =
        "data: " + error.dump() + "\n\n";

    CHECK(transmitted.size() == 2);
    CHECK(transmitted[0].first == expected_error);
    CHECK(!transmitted[0].second);
    CHECK(transmitted[1].first == "data: [DONE]\n\n");
    CHECK(transmitted[1].second);

    const auto parse_sse_data = [](const std::string& frame) {
        CHECK(frame.starts_with("data: "));
        CHECK(frame.ends_with("\n\n"));
        return frame.substr(6, frame.size() - 8);
    };
    const ordered_json parsed =
        ordered_json::parse(
            parse_sse_data(transmitted[0].first));
    CHECK(parsed["error"]["session_cleared"] == true);
    CHECK(parse_sse_data(transmitted[1].first) == "[DONE]");
}

void TestCliLimitAndRecoverableNotice() {
    CHECK(!CliRequestedMaxNewTokens(-1).has_value());
    CHECK(!CliRequestedMaxNewTokens(0).has_value());
    CHECK(CliRequestedMaxNewTokens(17) == std::optional<int>(17));

    CHECK(
        CliModelErrorNotice("capacity exceeded", false) ==
        "ERROR: capacity exceeded");
    CHECK(
        CliModelErrorNotice("ignored", true) ==
        "ERROR: AIE4 inference failed before submission; the current "
        "conversation was cleared.");
}

void TestAie4ModelInfoDetection() {
    CHECK(
        IsCorelibAie4ModelInfo(
            ordered_json{
                {"details", {{"execution_backend", "corelib_aie4"}}}}));
    CHECK(
        !IsCorelibAie4ModelInfo(
            ordered_json{{"details", {{"family", "phi4"}}}}));
    CHECK(!IsCorelibAie4ModelInfo(ordered_json::object()));
}

enum class HandlerExit {
    NonStreaming,
    Streaming,
    Exception,
    CancellationFinal
};

void RunProductionShapedGateInterleaving(HandlerExit exit) {
    CHECK(NPUAccessManager::is_npu_available());
    CHECK(NPUAccessManager::get_active_npu_requests() == 0);
    CHECK(NPUAccessManager::try_acquire_npu_access());

    std::mutex mutex;
    std::condition_variable ready;
    bool final_response_sent = false;
    bool second_attempted_while_handler_active = false;
    bool second_attempt_rejected = false;
    bool response_was_streaming = false;
    bool cancellation_finalized = false;
    bool post_response_cleanup_completed = false;
    bool handler_returned = false;
    bool completion_point_reached = false;
    bool second_insert_entered = false;
    bool second_entered_before_cleanup = false;
    std::exception_ptr second_error;

    std::thread second([&] {
        try {
            {
                std::unique_lock lock(mutex);
                ready.wait(lock, [&] { return final_response_sent; });
            }

            const bool acquired_early =
                NPUAccessManager::try_acquire_npu_access();
            {
                std::lock_guard lock(mutex);
                second_attempted_while_handler_active = true;
                second_attempt_rejected = !acquired_early;
            }
            ready.notify_all();
            if (acquired_early) {
                NPUAccessManager::release_npu_access();
                return;
            }

            {
                std::unique_lock lock(mutex);
                ready.wait(lock, [&] { return completion_point_reached; });
            }
            if (!NPUAccessManager::try_acquire_npu_access()) {
                throw std::runtime_error(
                    "queued request did not acquire released NPU gate");
            }
            {
                std::lock_guard lock(mutex);
                second_insert_entered = true;
                second_entered_before_cleanup =
                    !post_response_cleanup_completed ||
                    !handler_returned;
            }
            NPUAccessManager::release_npu_access();
        } catch (...) {
            second_error = std::current_exception();
            ready.notify_all();
        }
    });

    {
        NPURequestCompletionGuard completion([&] {
            NPUAccessManager::release_npu_access();
            {
                std::lock_guard lock(mutex);
                completion_point_reached = true;
            }
            ready.notify_all();
        });

        try {
            {
                std::lock_guard lock(mutex);
                response_was_streaming =
                    exit == HandlerExit::Streaming ||
                    exit == HandlerExit::CancellationFinal;
                cancellation_finalized =
                    exit == HandlerExit::CancellationFinal;
                final_response_sent = true;
            }
            ready.notify_all();
            {
                std::unique_lock lock(mutex);
                ready.wait(
                    lock,
                    [&] {
                        return second_attempted_while_handler_active;
                    });
            }

            if (exit == HandlerExit::Exception) {
                throw std::runtime_error("forced handler failure");
            }
        } catch (const std::exception&) {
            // Production process_task catches before its completion guard exits.
        }

        {
            std::lock_guard lock(mutex);
            post_response_cleanup_completed = true;
            handler_returned = true;
        }
    }

    second.join();
    if (second_error) {
        std::rethrow_exception(second_error);
    }

    CHECK(second_attempt_rejected);
    CHECK(
        response_was_streaming ==
        (exit == HandlerExit::Streaming ||
         exit == HandlerExit::CancellationFinal));
    CHECK(
        cancellation_finalized ==
        (exit == HandlerExit::CancellationFinal));
    CHECK(post_response_cleanup_completed);
    CHECK(handler_returned);
    CHECK(second_insert_entered);
    CHECK(!second_entered_before_cleanup);
    CHECK(NPUAccessManager::is_npu_available());
    CHECK(NPUAccessManager::get_active_npu_requests() == 0);
}

void TestCompleteRequestGate() {
    CHECK(requires_npu_access("POST", "/api/generate"));
    CHECK(requires_npu_access("POST", "/api/chat"));
    CHECK(requires_npu_access("POST", "/v1/chat/completions"));
    CHECK(requires_npu_access("POST", "/v1/completions"));
    CHECK(!requires_npu_access("GET", "/v1/completions"));

    for (const HandlerExit exit : {
             HandlerExit::NonStreaming,
             HandlerExit::Streaming,
             HandlerExit::Exception,
             HandlerExit::CancellationFinal}) {
        RunProductionShapedGateInterleaving(exit);
    }

    CHECK(NPUAccessManager::try_acquire_npu_access());
    {
        NPURequestCompletionGuard completion([] {
            throw std::runtime_error("forced queue handoff failure");
        });
    }
    CHECK(NPUAccessManager::is_npu_available());
    CHECK(NPUAccessManager::get_active_npu_requests() == 0);
}

volatile std::sig_atomic_t g_signal_observation = 0;

void PriorSignalHandler(int) {
    g_signal_observation = 1;
}

void ServeSignalHandler(int) {
    g_signal_observation = 2;
}

void TestServeSignalScopeAndShutdownOrder() {
    const auto original = std::signal(SIGINT, PriorSignalHandler);
    CHECK(original != SIG_ERR);
    {
        ScopedSignalHandler scope(SIGINT, ServeSignalHandler);
        g_signal_observation = 0;
        CHECK(std::raise(SIGINT) == 0);
        CHECK(g_signal_observation == 2);
    }

    g_signal_observation = 0;
    CHECK(std::raise(SIGINT) == 0);
    CHECK(g_signal_observation == 1);
    CHECK(std::signal(SIGINT, original) != SIG_ERR);

    std::vector<std::string> order;
    const bool healthy = CompleteServeShutdown(
        [&] { order.push_back("stop-admission-and-wait-inflight"); },
        [&] { order.push_back("destroy-server-handler-engine"); },
        [&] {
            order.push_back("shutdown-corelib");
            return true;
        });
    CHECK(healthy);
    CHECK(
        order == std::vector<std::string>({
                     "stop-admission-and-wait-inflight",
                     "destroy-server-handler-engine",
                     "shutdown-corelib"}));
}

}  // namespace

int main() {
    try {
        TestHandlerSpecificPresence();
        TestEndpointDefaultsAndPropagation();
        TestOllamaChatLegacyLimitIsUnchanged();
        TestOllamaChatAie4LimitAndAdmission();
        TestEveryGenerationRouteIsDeclared();
        TestNestedModelErrorAndHttpStatus();
        TestOpenAiStreamingErrorFramingAndParsing();
        TestCliLimitAndRecoverableNotice();
        TestAie4ModelInfoDetection();
        TestCompleteRequestGate();
        TestServeSignalScopeAndShutdownOrder();
        std::cout << "test_generation_limit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_generation_limit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
