#include <server/generation_limit.hpp>
#include <server/npu_access_manager.hpp>

#include "test_support.hpp"

#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

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

void RunForcedGateInterleaving(bool queue_during_insert) {
    CHECK(NPUAccessManager::is_npu_available());
    CHECK(NPUAccessManager::get_active_npu_requests() == 0);

    std::mutex mutex;
    std::condition_variable ready;
    bool insert_entered = false;
    bool generate_entered = false;
    bool second_attempted = false;
    bool second_attempt_rejected = false;
    bool generate_completed = false;
    bool gate_released = false;
    bool second_insert_entered = false;
    bool second_entered_early = false;
    std::exception_ptr first_error;
    std::exception_ptr second_error;

    std::thread first([&] {
        try {
            if (!NPUAccessManager::try_acquire_npu_access()) {
                throw std::runtime_error(
                    "first request did not acquire NPU access");
            }
            {
                std::unique_lock lock(mutex);
                insert_entered = true;
                ready.notify_all();
                if (queue_during_insert) {
                    ready.wait(
                        lock,
                        [&] { return second_attempted; });
                }
                generate_entered = true;
                ready.notify_all();
                if (!queue_during_insert) {
                    ready.wait(
                        lock,
                        [&] { return second_attempted; });
                }
                generate_completed = true;
            }
            NPUAccessManager::release_npu_access();
            {
                std::lock_guard lock(mutex);
                gate_released = true;
            }
            ready.notify_all();
        } catch (...) {
            first_error = std::current_exception();
            ready.notify_all();
        }
    });

    std::thread second([&] {
        try {
            {
                std::unique_lock lock(mutex);
                ready.wait(
                    lock,
                    [&] {
                        return queue_during_insert
                                   ? insert_entered
                                   : generate_entered;
                    });
            }

            const bool acquired_early =
                NPUAccessManager::try_acquire_npu_access();
            {
                std::lock_guard lock(mutex);
                second_attempt_rejected = !acquired_early;
                if (acquired_early) {
                    second_insert_entered = true;
                    second_entered_early = !generate_completed;
                }
                second_attempted = true;
            }
            ready.notify_all();
            if (acquired_early) {
                NPUAccessManager::release_npu_access();
                return;
            }

            {
                std::unique_lock lock(mutex);
                ready.wait(lock, [&] { return gate_released; });
            }
            if (!NPUAccessManager::try_acquire_npu_access()) {
                throw std::runtime_error(
                    "queued request did not acquire released NPU gate");
            }
            {
                std::lock_guard lock(mutex);
                second_insert_entered = true;
                second_entered_early = !generate_completed;
            }
            NPUAccessManager::release_npu_access();
        } catch (...) {
            second_error = std::current_exception();
            ready.notify_all();
        }
    });

    first.join();
    second.join();
    if (first_error) {
        std::rethrow_exception(first_error);
    }
    if (second_error) {
        std::rethrow_exception(second_error);
    }

    CHECK(second_attempt_rejected);
    CHECK(generate_completed);
    CHECK(second_insert_entered);
    CHECK(!second_entered_early);
    CHECK(NPUAccessManager::is_npu_available());
    CHECK(NPUAccessManager::get_active_npu_requests() == 0);
}

void TestCompleteRequestGate() {
    CHECK(requires_npu_access("POST", "/api/generate"));
    CHECK(requires_npu_access("POST", "/api/chat"));
    CHECK(requires_npu_access("POST", "/v1/chat/completions"));
    CHECK(requires_npu_access("POST", "/v1/completions"));
    CHECK(!requires_npu_access("GET", "/v1/completions"));

    RunForcedGateInterleaving(true);
    RunForcedGateInterleaving(false);
}

}  // namespace

int main() {
    try {
        TestHandlerSpecificPresence();
        TestEndpointDefaultsAndPropagation();
        TestOllamaChatLimitStaysSoftOnly();
        TestNestedModelErrorAndHttpStatus();
        TestCliLimitAndRecoverableNotice();
        TestAie4ModelInfoDetection();
        TestCompleteRequestGate();
        std::cout << "test_generation_limit: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_generation_limit: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
