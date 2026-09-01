#include "test_support.hpp"

#include <AutoModel/modeling_phi4.hpp>
#include <models/phi4/phi4_corelib_aie4_tuning.hpp>

#if defined(FLM_ENABLE_CORELIB_AIE4)
#include <corelib/corelib_runtime.hpp>
#include <models/phi4/phi4_corelib_aie4.hpp>
#endif

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::vector<int> g_encoded_tokens;
int g_sample_token = 7;
int g_sampler_reset_count = 0;
int g_sampler_sample_count = 0;

std::filesystem::path CurrentExecutablePath() {
    std::wstring buffer(32768, L'\0');
    const DWORD written = GetModuleFileNameW(
        nullptr,
        buffer.data(),
        static_cast<DWORD>(buffer.size()));
    if (written == 0 || written >= buffer.size()) {
        throw std::runtime_error("GetModuleFileNameW failed");
    }
    buffer.resize(written);
    return std::filesystem::path(buffer);
}

class TempModelPackage final {
public:
    explicit TempModelPackage(
        std::vector<int> eos_ids = {200020, 199999},
        std::optional<int> hidden_size = 3072) {
        const auto base = std::filesystem::temp_directory_path();
        for (int attempt = 0; attempt < 100; ++attempt) {
            path_ = base /
                    ("flm-phi4-frontend-" +
                     std::to_string(GetCurrentProcessId()) + "-" +
                     std::to_string(GetTickCount64()) + "-" +
                     std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(path_, error)) {
                break;
            }
            if (attempt == 99) {
                throw std::runtime_error(
                    "failed to create temporary Phi-4 package");
            }
        }

        nlohmann::json config = {
            {"model_type", "phi4"},
            {"num_hidden_layers", 32},
            {"hidden_size", hidden_size.value_or(3072)},
            {"intermediate_size", 8192},
            {"num_attention_heads", 24},
            {"num_key_value_heads", 8},
            {"head_dim", 128},
            {"vocab_size", 200064},
            {"rms_norm_eps", 1.0e-5},
        };
        WriteJson(path_ / "config.json", config);

        nlohmann::json tokenizer_config = {
            {"chat_template",
             "{% for message in messages %}"
             "{{ message['content'] }}"
             "{% endfor %}"},
            {"eos_token_id", std::move(eos_ids)},
        };
        WriteJson(path_ / "tokenizer_config.json", tokenizer_config);
        WriteText(path_ / "tokenizer.json", "{}");
    }

    ~TempModelPackage() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    TempModelPackage(const TempModelPackage&) = delete;
    TempModelPackage& operator=(const TempModelPackage&) = delete;

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    static void WriteJson(
        const std::filesystem::path& path,
        const nlohmann::json& value) {
        WriteText(path, value.dump());
    }

    static void WriteText(
        const std::filesystem::path& path,
        std::string_view value) {
        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw std::runtime_error(
                "failed to create frontend test package file");
        }
        output.write(
            value.data(),
            static_cast<std::streamsize>(value.size()));
    }

    std::filesystem::path path_;
};

class FakeEngine final : public causal_lm {
public:
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
#endif
        forward_tokens.push_back(id);
        ++position;
        return MakeLogits();
    }

    buffer<bf16> prefill(
        std::vector<int>& ids,
        void*) override {
#if defined(FLM_ENABLE_CORELIB_AIE4)
        if (fail_next_prefill) {
            fail_next_prefill = false;
            throw flm::corelib::CorelibError(
                ryzenai_corelib_status_failure,
                "fake_prefill",
                "injected pre-submit prefill failure",
                "failure");
        }
#endif
        prefill_calls.push_back(ids);
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

    std::uint32_t max_length;
    int position = 0;
    int clear_count = 0;
    std::optional<int> checkpoint_position;
    std::vector<std::vector<int>> prefill_calls;
    std::vector<int> forward_tokens;
#if defined(FLM_ENABLE_CORELIB_AIE4)
    bool fail_next_prefill = false;
    bool fail_next_forward = false;
#endif
};

struct FactoryState {
    int calls = 0;
    bool last_was_corelib = false;
    FakeEngine* engine = nullptr;
};

FactoryState g_factory;

nlohmann::ordered_json ModelInfo(
    int default_context_length,
    std::optional<std::string> execution_backend = std::nullopt) {
    nlohmann::ordered_json details = {
        {"family", "phi4"},
    };
    if (execution_backend.has_value()) {
        details["execution_backend"] = *execution_backend;
    }
    return {
        {"default_context_length", default_context_length},
        {"details", std::move(details)},
    };
}

chat_meta_info_t Meta() {
    chat_meta_info_t meta;
    meta.max_prefill_len = 64;
    return meta;
}

lm_uniform_input_t Prompt(
    std::optional<int> requested = std::nullopt) {
    lm_uniform_input_t input;
    input.prompt = "frontend-test";
    input.requested_max_new_tokens = requested;
    return input;
}

template <class Function>
void CheckRequestError(
    Function&& function,
    int expected_code,
    bool expected_session_cleared,
    std::string_view expected_message) {
    try {
        function();
    } catch (const ModelRequestError& error) {
        CHECK(error.http_code() == expected_code);
        CHECK(error.session_cleared() == expected_session_cleared);
        CHECK(
            std::string_view(error.what()).find(expected_message) !=
            std::string_view::npos);
        return;
    }
    throw std::runtime_error("expected ModelRequestError was not thrown");
}

}  // namespace

Tokenizer::Tokenizer(const std::string&) {
    is_doubled_encoded = false;
}

Tokenizer::~Tokenizer() = default;

std::vector<int> Tokenizer::encode(const std::string&) {
    return g_encoded_tokens;
}

std::string Tokenizer::decode(const std::vector<int>&) {
    return "decoded";
}

std::string Tokenizer::run_time_decoder(int token) {
    return "token-" + std::to_string(token);
}

SafeTensors::~SafeTensors() = default;

Sampler::Sampler(int features, sampler_config& config)
    : in_features(features),
      rep_penalty(config.rep_penalty),
      freq_penalty(config.freq_penalty),
      pre_penalty(config.pre_penalty),
      top_k(config.top_k),
      top_p(config.top_p),
      min_p(config.min_p),
      temperature(config.temperature),
      total_tokens(0),
      freq_penalty_window(config.freq_penalty_window),
      rep_penalty_window(config.rep_penalty_window),
      repeat_last_n(config.repeat_last_n),
      use_optimized_sampling(config.use_optimized_sampling) {
    logits.resize(1);
    counters.resize(1);
    token_positions.resize(1, -1);
}

void Sampler::reset_penalties() {
    ++g_sampler_reset_count;
    std::fill(counters.begin(), counters.end(), 0);
    std::fill(token_positions.begin(), token_positions.end(), -1);
    token_counts_sparse.clear();
    token_history.clear();
    total_tokens = 0;
}

int Sampler::sample(buffer<bf16>&) {
    ++g_sampler_sample_count;
    ++total_tokens;
    token_history.push_back(g_sample_token);
    return g_sample_token;
}

namespace utils {

std::string get_executable_directory() {
    return CurrentExecutablePath().parent_path().string();
}

}  // namespace utils

namespace flm::phi4::testing {

class Phi4FrontendTestAccess final {
public:
    using Factory = std::function<std::unique_ptr<causal_lm>(
        bool,
        const LM_Config&,
        npu_xclbin_manager*,
        const std::filesystem::path&,
        std::uint32_t)>;

    static void InstallFactory() {
        g_factory = {};
        Phi4::engine_factory_for_testing_ =
            [](bool corelib,
               const LM_Config&,
               npu_xclbin_manager*,
               const std::filesystem::path&,
               std::uint32_t max_length) {
                ++g_factory.calls;
                g_factory.last_was_corelib = corelib;
                auto engine =
                    std::make_unique<FakeEngine>(max_length);
                g_factory.engine = engine.get();
                return engine;
            };
    }

    static void RemoveFactory() {
        Phi4::engine_factory_for_testing_ = {};
        g_factory = {};
    }

    static bool HasLegacyNpu(const Phi4& model) {
        return model.npu != nullptr;
    }

#if defined(FLM_ENABLE_CORELIB_AIE4)
    static bool HasRuntime(const Phi4& model) {
        return model.corelib_runtime_ != nullptr;
    }

    static void SetMetrics(
        Phi4& model,
        const Phi4Aie4Metrics& metrics) {
        model.metrics_for_testing_ = metrics;
    }

    static void ForceRoute(
        Phi4& model,
        ForcedContinuationRoute route) {
        model.forced_continuation_route_ = route;
    }
#endif

    static const std::vector<int>& History(const Phi4& model) {
        return model.token_history;
    }
};

}  // namespace flm::phi4::testing

namespace {

using flm::phi4::ContinuationRoute;
using flm::phi4::ForcedContinuationRoute;
using flm::phi4::SelectContinuationRoute;
using flm::phi4::testing::Phi4FrontendTestAccess;

struct FactoryScope final {
    FactoryScope() {
        Phi4FrontendTestAccess::InstallFactory();
    }

    ~FactoryScope() {
        Phi4FrontendTestAccess::RemoveFactory();
    }
};

std::unique_ptr<Phi4> Load(
    const TempModelPackage& package,
    nlohmann::ordered_json model_info,
    int requested_context = -1,
    bool preemption = false,
    flm_rt::device* device =
        reinterpret_cast<flm_rt::device*>(std::uintptr_t{1})) {
    auto model = std::make_unique<Phi4>(device);
    model->load_model(
        package.path().string(),
        std::move(model_info),
        requested_context,
        preemption);
    return model;
}

void TestContinuationSelector() {
    CHECK(
        SelectContinuationRoute(
            0,
            ForcedContinuationRoute::Automatic) ==
        ContinuationRoute::Append);
    CHECK(
        SelectContinuationRoute(
            1,
            ForcedContinuationRoute::Automatic) ==
        ContinuationRoute::Reprefill);
    CHECK(
        SelectContinuationRoute(
            999,
            ForcedContinuationRoute::Append) ==
        ContinuationRoute::Append);
    CHECK(
        SelectContinuationRoute(
            0,
            ForcedContinuationRoute::Reprefill) ==
        ContinuationRoute::Reprefill);
}

void TestLegacyRoutingAndUnknownBackend() {
    TempModelPackage package({200020});
    FactoryScope factory;

    auto legacy = Load(package, ModelInfo(1024));
    CHECK(g_factory.calls == 1);
    CHECK(!g_factory.last_was_corelib);
    CHECK(Phi4FrontendTestAccess::HasLegacyNpu(*legacy));

    FakeEngine* legacy_engine = g_factory.engine;
    g_encoded_tokens = {1, 2};
    auto meta = Meta();
    auto input = Prompt();
    CHECK(legacy->insert(meta, input));
    legacy_engine->prefill_calls.clear();
    g_encoded_tokens = {1, 2, 3, 4};
    CHECK(legacy->insert(meta, input));
    CHECK(legacy_engine->prefill_calls.size() == 1);
    CHECK(
        legacy_engine->prefill_calls[0] ==
        std::vector<int>({3, 4}));

    legacy->set_max_length(512);
    CHECK(legacy->get_max_length() == 1024);
    CHECK(legacy_engine->max_length == 512);
    CHECK(
        legacy->show_profile() ==
        legacy->AutoModel::show_profile());

    CheckThrowsContains(
        [&] {
            auto model = Load(
                package,
                ModelInfo(64, "invented_backend"));
            (void)model;
        },
        "invented_backend");
    CHECK(g_factory.calls == 1);
}

#if !defined(FLM_ENABLE_CORELIB_AIE4)

void TestFeatureOffRejectsCorelibTag() {
    TempModelPackage package;
    FactoryScope factory;

    CheckThrowsContains(
        [&] {
            auto model = Load(
                package,
                ModelInfo(64, "corelib_aie4"),
                -1,
                false,
                nullptr);
            (void)model;
        },
        "without Phi-4 AIE4 corelib support");
    CHECK(g_factory.calls == 0);
}

#else

void ConfigureFakeCorelibDll() {
    const auto fake_dll =
        CurrentExecutablePath().parent_path() /
        "fake_ryzenai_corelib.dll";
    CHECK(std::filesystem::exists(fake_dll));
    if (_wputenv_s(
            L"RYZENAI_CORELIB_PATH",
            fake_dll.c_str()) != 0) {
        throw std::runtime_error(
            "failed to configure RYZENAI_CORELIB_PATH");
    }
}

void TestCorelibRoutingAndPreemption() {
    TempModelPackage package;
    FactoryScope factory;

    CheckThrowsContains(
        [&] {
            auto model = Load(
                package,
                ModelInfo(64, "corelib_aie4"),
                -1,
                true,
                nullptr);
            (void)model;
        },
        "preemption");
    CHECK(g_factory.calls == 0);

    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    CHECK(g_factory.calls == 1);
    CHECK(g_factory.last_was_corelib);
    CHECK(!Phi4FrontendTestAccess::HasLegacyNpu(*model));
    CHECK(Phi4FrontendTestAccess::HasRuntime(*model));
}

void TestInitialAndAtomicCaps() {
    TempModelPackage package;
    FactoryScope factory;

    CheckThrowsContains(
        [&] {
            auto model = Load(
                package,
                ModelInfo(4097, "corelib_aie4"),
                -1,
                false,
                nullptr);
            (void)model;
        },
        "1..4096");
    CHECK(g_factory.calls == 0);

    auto model = Load(
        package,
        ModelInfo(1024, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;
    CHECK(model->get_max_length() == 1024);
    CHECK(engine->max_length == 1024);

    CheckThrowsContains(
        [&] { model->set_max_length(4097); },
        "1..4096");
    CHECK(model->get_max_length() == 1024);
    CHECK(engine->max_length == 1024);

    model->clear_context();
    CHECK(model->get_current_context_length() == 0);
    model->set_max_length(512);
    CHECK(model->get_max_length() == 512);
    CHECK(engine->max_length == 512);

    g_encoded_tokens.assign(10, 11);
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    CHECK(model->get_current_context_length() == 10);
    CHECK(engine->position == 10);

    CheckThrowsContains(
        [&] { model->set_max_length(9); },
        "current");
    CHECK(model->get_max_length() == 512);
    CHECK(engine->max_length == 512);
    CHECK(engine->position == 10);
}

void TestRenderedCapacityIsAtomic() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(512, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;
    engine->prefill_calls.clear();
    const int clear_count = engine->clear_count;

    g_encoded_tokens.assign(500, 12);
    auto meta = Meta();
    auto input = Prompt(13);
    CheckRequestError(
        [&] { (void)model->insert(meta, input); },
        400,
        false,
        "500");
    CHECK(engine->position == 0);
    CHECK(engine->clear_count == clear_count);
    CHECK(engine->prefill_calls.empty());
    CHECK(Phi4FrontendTestAccess::History(*model).empty());

    input.requested_max_new_tokens = std::nullopt;
    CHECK(model->insert(meta, input));
    CHECK(engine->position == 500);

    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Append);
    g_encoded_tokens.push_back(13);
    input.requested_max_new_tokens = 12;
    const auto calls_before_append_rejection =
        engine->prefill_calls.size();
    CheckRequestError(
        [&] { (void)model->insert(meta, input); },
        400,
        false,
        "501");
    CHECK(engine->position == 500);
    CHECK(
        engine->prefill_calls.size() ==
        calls_before_append_rejection);
    CHECK(Phi4FrontendTestAccess::History(*model).size() == 500);
}

void TestForcedAppendAndCancellationAlignment() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens = {1, 2};
    auto first_meta = Meta();
    auto first = Prompt();
    CHECK(model->insert(first_meta, first));
    engine->prefill_calls.clear();

    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Append);
    g_encoded_tokens = {1, 2, 3, 4, 5};
    auto second_meta = Meta();
    auto second = Prompt();
    int cancellation_checks = 0;
    const bool inserted = model->insert(
        second_meta,
        second,
        [&] { return cancellation_checks++ == 1; });
    CHECK(!inserted);
    CHECK(second_meta.stop_reason == CANCEL_DETECTED);
    CHECK(engine->prefill_calls.size() == 1);
    CHECK(engine->prefill_calls[0] == std::vector<int>({3}));
    CHECK(engine->position == 3);
    CHECK(model->get_current_context_length() == 3);
    CHECK(
        Phi4FrontendTestAccess::History(*model) ==
        std::vector<int>({1, 2, 3}));
}

void TestForcedAndAutomaticReprefill() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens = {1, 2};
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));

    engine->prefill_calls.clear();
    const int clear_before_forced = engine->clear_count;
    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Reprefill);
    g_encoded_tokens = {1, 2, 3, 4};
    CHECK(model->insert(meta, input));
    CHECK(engine->clear_count == clear_before_forced + 1);
    CHECK(engine->prefill_calls.size() == 1);
    CHECK(
        engine->prefill_calls[0] ==
        std::vector<int>({1, 2, 3, 4}));
    CHECK(engine->position == 4);

    engine->prefill_calls.clear();
    const int clear_before_automatic = engine->clear_count;
    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Automatic);
    g_encoded_tokens = {1, 2, 3, 4, 5};
    CHECK(model->insert(meta, input));
    CHECK(engine->clear_count == clear_before_automatic + 1);
    CHECK(engine->prefill_calls.size() == 1);
    CHECK(
        engine->prefill_calls[0] ==
        std::vector<int>({1, 2, 3, 4, 5}));
}

void TestEosValidationAndFrontendStop() {
    FactoryScope factory;
    TempModelPackage invalid_package({200020});
    CheckThrowsContains(
        [&] {
            auto model = Load(
                invalid_package,
                ModelInfo(64, "corelib_aie4"),
                -1,
                false,
                nullptr);
            (void)model;
        },
        "199999");

    TempModelPackage package;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;
    g_encoded_tokens = {10, 11};
    g_sample_token = 200020;
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    std::ostringstream output;
    CHECK(model->generate(meta, 8, output).empty());
    CHECK(engine->forward_tokens.empty());
    g_sample_token = 7;
}

void TestRecoverableFailuresClearSession() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens = {20};
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Append);
    engine->checkpoint();
    engine->fail_next_prefill = true;
    const int resets_before_insert = g_sampler_reset_count;
    g_encoded_tokens = {20, 21};
    CheckRequestError(
        [&] { (void)model->insert(meta, input); },
        500,
        true,
        "current conversation was cleared");
    CHECK(engine->position == 0);
    CHECK(!engine->checkpoint_position.has_value());
    CHECK(Phi4FrontendTestAccess::History(*model).empty());
    CHECK(g_sampler_reset_count > resets_before_insert);

    g_encoded_tokens = {31, 32};
    CHECK(model->insert(meta, input));
    engine->checkpoint();
    engine->fail_next_forward = true;
    const int resets_before_generate = g_sampler_reset_count;
    std::ostringstream output;
    CheckRequestError(
        [&] { (void)model->generate(meta, 8, output); },
        500,
        true,
        "current conversation was cleared");
    CHECK(engine->position == 0);
    CHECK(!engine->checkpoint_position.has_value());
    CHECK(Phi4FrontendTestAccess::History(*model).empty());
    CHECK(g_sampler_reset_count > resets_before_generate);
}

void TestProfileSplit() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);

    flm::phi4::Phi4Aie4Metrics metrics;
    metrics.model_load_ns = 101;
    metrics.weight_pack_ns = 202;
    metrics.dispatch_count = 303;
    metrics.synchronize_count = 404;
    metrics.helper_transition_counts = {1, 2, 3, 4, 5, 6};
    Phi4FrontendTestAccess::SetMetrics(*model, metrics);

    g_encoded_tokens = {1};
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Append);
    g_encoded_tokens = {1, 2};
    CHECK(model->insert(meta, input));

    const std::string profile = model->show_profile();
    CHECK(profile.find("corelib_aie4") != std::string::npos);
    CHECK(profile.find("Continuation route: append") != std::string::npos);
    CHECK(profile.find("Append threshold: 0") != std::string::npos);
    CHECK(
        profile.find("fake_ryzenai_corelib.dll") !=
        std::string::npos);
    CHECK(profile.find("Dispatches: 303") != std::string::npos);
    CHECK(profile.find("Synchronizations: 404") != std::string::npos);
    CHECK(profile.find("Helper transitions: 1/2/3/4/5/6") !=
          std::string::npos);
    CHECK(profile.find("Cold model load: 101 ns") !=
          std::string::npos);
    CHECK(profile.find("Cold weight pack: 202 ns") !=
          std::string::npos);
    CHECK(profile.find("Continuation time:") != std::string::npos);
}

#endif

}  // namespace

int main() {
    try {
        TestContinuationSelector();
        TestLegacyRoutingAndUnknownBackend();
#if defined(FLM_ENABLE_CORELIB_AIE4)
        ConfigureFakeCorelibDll();
        TestCorelibRoutingAndPreemption();
        TestInitialAndAtomicCaps();
        TestRenderedCapacityIsAtomic();
        TestForcedAppendAndCancellationAlignment();
        TestForcedAndAutomaticReprefill();
        TestEosValidationAndFrontendStop();
        TestRecoverableFailuresClearSession();
        TestProfileSplit();
        Phi4FrontendTestAccess::RemoveFactory();
        flm::corelib::CorelibRuntime::ShutdownProcess();
        std::cout << "test_phi4_frontend_on: PASS\n";
#else
        TestFeatureOffRejectsCorelibTag();
        std::cout << "test_phi4_frontend_off: PASS\n";
#endif
        return 0;
    } catch (const std::exception& error) {
#if defined(FLM_ENABLE_CORELIB_AIE4)
        Phi4FrontendTestAccess::RemoveFactory();
        try {
            flm::corelib::CorelibRuntime::ShutdownProcess();
        } catch (...) {
        }
#endif
        std::cerr << "test_phi4_frontend: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
