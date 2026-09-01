#include <server/generation_limit.hpp>
#include <server/npu_access_manager.hpp>
#include <server/serve_lifecycle.hpp>

#include "test_support.hpp"

#include <array>
#include <csignal>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
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

    for (const GenerationEndpoint endpoint : {
             GenerationEndpoint::Generate,
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

void TestOllamaChatLimitStaysSoftOnly() {
    const ordered_json default_request = {
        {"options", ordered_json::object()},
    };
    CHECK(OllamaChatGenerationLoopLimit(default_request) == 4096);

    const ordered_json explicit_request = {
        {"options", {{"num_predict", 91}}},
    };
    CHECK(OllamaChatGenerationLoopLimit(explicit_request) == 91);
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
        TestOllamaChatLimitStaysSoftOnly();
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
