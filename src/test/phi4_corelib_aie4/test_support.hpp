#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

inline void Check(
    bool condition,
    std::string_view expression,
    const char* file,
    int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            " CHECK failed: " + std::string(expression));
    }
}

#define CHECK(expression) \
    Check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

template <class Function>
void CheckThrowsContains(Function&& function, std::string_view expected) {
    try {
        function();
    } catch (const std::exception& error) {
        CHECK(std::string_view(error.what()).find(expected) !=
              std::string_view::npos);
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

#if defined(FLM_PHI4_FRONTEND_TEST_SUPPORT)

#include <cstdint>
#include <new>
#include <optional>
#include <utility>
#include <vector>

class FakeEngine final : public causal_lm {
public:
    enum class FailureKind {
        None,
        Corelib,
        Standard,
        Unknown,
    };

    explicit FakeEngine(std::uint32_t max_length)
        : max_length(max_length) {}

    buffer<bf16> forward(int id) override {
#if defined(FLM_ENABLE_CORELIB_AIE4)
        if (fail_next_forward) {
            fail_next_forward = false;
            throw flm::corelib::CorelibError(
                ryzenai_corelib_status_failure,
                "fake_forward",
                "injected pre-submit forward failure",
                "failure");
        }
        MaybeFail(
            forward_failure,
            successful_forwards_before_failure,
            "fake_forward");
#endif
        forward_tokens.push_back(id);
        ++position;
        return MakeLogits();
    }

    buffer<bf16> prefill(
        std::vector<int>& ids,
        void* payload) override {
#if defined(FLM_ENABLE_CORELIB_AIE4)
        if (fail_next_prefill) {
            fail_next_prefill = false;
            throw flm::corelib::CorelibError(
                ryzenai_corelib_status_failure,
                "fake_prefill",
                "injected pre-submit prefill failure",
                "failure");
        }
        MaybeFail(
            prefill_failure,
            successful_prefills_before_failure,
            "fake_prefill");
#endif
        prefill_calls.push_back(ids);
        prefill_payloads.push_back(payload);
        position += static_cast<int>(ids.size());
        return MakeLogits();
    }

    void set_context_length(int length) override {
        position = length;
    }

    void load_weights(Q4NX&) override {}

    void update_max_length(std::uint32_t requested) override {
        if (requested == 0 || requested > 4096) {
            throw std::out_of_range(
                "fake AIE4 maximum length must be in 1..4096");
        }
        if (requested < static_cast<std::uint32_t>(position)) {
            throw std::out_of_range(
                "fake AIE4 maximum length cannot be below current");
        }
        max_length = requested;
    }

    void clear_context() override {
        ++clear_count;
        position = 0;
        checkpoint_position.reset();
    }

    buffer<bf16> get_k_cache(int, int) override {
        return MakeLogits();
    }

    buffer<bf16> get_v_cache(int, int) override {
        return MakeLogits();
    }

    int get_current_context_length() override {
        return position;
    }

    int checkpoint() override {
        checkpoint_position = position;
        return position;
    }

    int restore() override {
        if (!checkpoint_position.has_value()) {
            throw std::logic_error("fake engine has no checkpoint");
        }
        position = *checkpoint_position;
        return position;
    }

    static buffer<bf16> MakeLogits() {
        return buffer<bf16>(1);
    }

#if defined(FLM_ENABLE_CORELIB_AIE4)
    static void MaybeFail(
        FailureKind& failure,
        int& successful_calls_before_failure,
        const char* operation) {
        if (failure == FailureKind::None) {
            return;
        }
        if (successful_calls_before_failure > 0) {
            --successful_calls_before_failure;
            return;
        }

        const FailureKind injected = std::exchange(
            failure,
            FailureKind::None);
        if (injected == FailureKind::Corelib) {
            throw flm::corelib::CorelibError(
                ryzenai_corelib_status_failure,
                operation,
                "injected pre-submit corelib failure",
                "failure");
        }
        if (injected == FailureKind::Standard) {
            throw std::bad_alloc();
        }
        throw 17;
    }
#endif

    std::uint32_t max_length;
    int position = 0;
    int clear_count = 0;
    std::optional<int> checkpoint_position;
    std::vector<std::vector<int>> prefill_calls;
    std::vector<void*> prefill_payloads;
    std::vector<int> forward_tokens;
#if defined(FLM_ENABLE_CORELIB_AIE4)
    bool fail_next_prefill = false;
    bool fail_next_forward = false;
    FailureKind prefill_failure = FailureKind::None;
    FailureKind forward_failure = FailureKind::None;
    int successful_prefills_before_failure = 0;
    int successful_forwards_before_failure = 0;
#endif
};

#endif
