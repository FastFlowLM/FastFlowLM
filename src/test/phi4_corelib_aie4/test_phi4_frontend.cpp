#include <AutoModel/modeling_phi4.hpp>
#include <models/phi4/phi4_corelib_aie4_tuning.hpp>
#include <server/generation_limit.hpp>

#if defined(FLM_ENABLE_CORELIB_AIE4)
#include <corelib/corelib_runtime.hpp>
#include <models/phi4/phi4_corelib_aie4.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>
#endif

#define FLM_PHI4_FRONTEND_TEST_SUPPORT
#include "test_support.hpp"

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <deque>
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
std::deque<int> g_sample_tokens;
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
        std::optional<int> hidden_size = 3072,
        nlohmann::json config_overrides = nlohmann::json::object()) {
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
        for (const auto& [key, value] : config_overrides.items()) {
            config[key] = value;
        }
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

    void OverwriteFile(
        std::string_view name,
        std::string_view contents) const {
        WriteText(path_ / name, contents);
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

void SetSamples(std::initializer_list<int> samples) {
    g_sample_tokens.assign(samples.begin(), samples.end());
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
    const int sampled_token = g_sample_tokens.empty()
                                  ? g_sample_token
                                  : g_sample_tokens.front();
    if (!g_sample_tokens.empty()) {
        g_sample_tokens.pop_front();
    }
    token_history.push_back(sampled_token);
    return sampled_token;
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

    static int LastToken(const Phi4& model) {
        return model.last_token;
    }

    static bool SharedInsert(
        Phi4& model,
        chat_meta_info_t& meta,
        std::vector<int>& tokens,
        void* payload) {
        return model._shared_insert(
            meta,
            tokens,
            [] { return false; },
            payload,
            0);
    }
};

}  // namespace flm::phi4::testing

namespace {

class NonPhiModel final : public AutoModel {
public:
    NonPhiModel() : AutoModel(nullptr, "non-phi-test") {}

    std::string generate(
        chat_meta_info_t&,
        int,
        std::ostream&,
        std::function<bool()>) override {
        return {};
    }

    bool insert(
        chat_meta_info_t&,
        lm_uniform_input_t&,
        std::function<bool()>) override {
        return true;
    }

    std::string generate_with_prompt(
        chat_meta_info_t&,
        lm_uniform_input_t&,
        int,
        std::ostream&) override {
        return {};
    }

    std::string apply_chat_template(
        nlohmann::ordered_json&,
        nlohmann::ordered_json) override {
        return {};
    }
};

using flm::phi4::ContinuationRoute;
using flm::phi4::ForcedContinuationRoute;
using flm::phi4::kContinuationAppendThreshold;
using flm::phi4::SelectContinuationRoute;

// The release-fixed threshold is generated by
// `tools/calibrate_phi4_corelib_continuation.py`; the tests below are written
// against the symbol, not its current value. This bound is the one thing about
// the value they do assert: a threshold at or above the physical 4096-row
// cache could never be reached by a suffix that also fits, so it would make
// the Automatic append branch dead in a way no route test would notice.
static_assert(
    kContinuationAppendThreshold < 4096,
    "continuation append threshold must be below the physical KV row count");
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

// Every branch of `SelectContinuationRoute`, expressed against the generated
// `kContinuationAppendThreshold` rather than against the number it happens to
// hold today.
//
// The earlier version of this test asserted that suffix 1 re-prefills under
// Automatic. That was true only because the threshold was still the
// placeholder zero, so the test and the header agreed on an assumption neither
// of them stated -- and the test would have gone on passing while silently
// covering nothing, because with a zero threshold the Automatic append branch
// is unreachable. Calibration set the threshold to a positive value and the
// assertion became wrong. What follows enumerates the cases the FUNCTION
// branches on: the zero-suffix short circuit under all three forced values,
// both forced overrides above and below the threshold, and both sides of the
// Automatic comparison including its exact boundary.
void TestContinuationSelector() {
    constexpr std::size_t kThreshold = kContinuationAppendThreshold;

    // A zero-length suffix re-prefills whatever the caller forced: there is no
    // token to append, and this precedes the forced switch.
    for (const ForcedContinuationRoute forced : {
             ForcedContinuationRoute::Automatic,
             ForcedContinuationRoute::Append,
             ForcedContinuationRoute::Reprefill}) {
        CHECK(
            SelectContinuationRoute(0, forced) ==
            ContinuationRoute::Reprefill);
    }

    // A forced route ignores the threshold in both directions.
    for (const std::size_t suffix : {
             std::size_t{1},
             kThreshold,
             kThreshold + 1,
             std::size_t{999}}) {
        if (suffix == 0) {
            continue;  // covered above; `kThreshold` may legitimately be 0
        }
        CHECK(
            SelectContinuationRoute(
                suffix,
                ForcedContinuationRoute::Append) ==
            ContinuationRoute::Append);
        CHECK(
            SelectContinuationRoute(
                suffix,
                ForcedContinuationRoute::Reprefill) ==
            ContinuationRoute::Reprefill);
    }

    // Automatic: append at and below the threshold, re-prefill above it.
    //
    // A zero threshold is a legitimate calibration outcome -- Section 10.7
    // selects it when no sampled suffix length wins or the winners are not
    // prefix-contiguous -- and it makes the append side unreachable. Asserting
    // it separately keeps this test from silently becoming vacuous if a later
    // recalibration lands there.
    if (kThreshold == 0) {
        CHECK(
            SelectContinuationRoute(
                1,
                ForcedContinuationRoute::Automatic) ==
            ContinuationRoute::Reprefill);
    } else {
        for (std::size_t suffix = 1; suffix <= kThreshold; ++suffix) {
            CHECK(
                SelectContinuationRoute(
                    suffix,
                    ForcedContinuationRoute::Automatic) ==
                ContinuationRoute::Append);
        }
    }
    for (const std::size_t suffix : {
             kThreshold + 1,
             kThreshold + 2,
             std::size_t{4096}}) {
        CHECK(
            SelectContinuationRoute(
                suffix,
                ForcedContinuationRoute::Automatic) ==
            ContinuationRoute::Reprefill);
    }
}

void TestLegacyRoutingAndUnknownBackend() {
    TempModelPackage package({200020});
    FactoryScope factory;

    auto legacy = Load(package, ModelInfo(1024));
    CHECK(g_factory.calls == 1);
    CHECK(!g_factory.last_was_corelib);
    CHECK(!legacy->uses_corelib_aie4());
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

void TestLoadedBackendControlsOmittedLimit() {
    const ParsedGenerationLimit omitted{false, -1};
    const nlohmann::ordered_json misleading_catalog_info = {
        {"details", {{"execution_backend", "corelib_aie4"}}},
    };
    CHECK(IsCorelibAie4ModelInfo(misleading_catalog_info));

    NonPhiModel non_phi;
    CHECK(!non_phi.uses_corelib_aie4());
    CHECK(
        GenerationLoopLimit(
            omitted,
            non_phi.uses_corelib_aie4()) == 4096);

    TempModelPackage package({200020});
    FactoryScope factory;
    auto legacy_phi = Load(package, ModelInfo(1024));
    CHECK(!legacy_phi->uses_corelib_aie4());
    CHECK(
        GenerationLoopLimit(
            omitted,
            legacy_phi->uses_corelib_aie4()) == 4096);
}

void TestLegacyExactRepeatPreservesEmptyPrefillPayload() {
    TempModelPackage package({200020});
    FactoryScope factory;
    auto model = Load(package, ModelInfo(64));
    FakeEngine* engine = g_factory.engine;

    int first_payload = 1;
    int repeated_payload = 2;
    auto meta = Meta();
    std::vector<int> first{1, 2};
    CHECK(Phi4FrontendTestAccess::SharedInsert(
        *model,
        meta,
        first,
        &first_payload));
    const int samples_before_repeat = g_sampler_sample_count;

    std::vector<int> repeated{1, 2};
    CHECK(Phi4FrontendTestAccess::SharedInsert(
        *model,
        meta,
        repeated,
        &repeated_payload));
    CHECK(engine->prefill_calls.size() == 2);
    CHECK(engine->prefill_calls.back().empty());
    CHECK(engine->prefill_payloads.back() == &repeated_payload);
    CHECK(g_sampler_sample_count == samples_before_repeat + 1);
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

// Design `MODEL-2`. The overlay config.json is a restatement of the model
// contract, so any disagreement with the validated constants must fail the
// load rather than quietly reconfigure the model. Each corruption below is a
// value the AIE4 frontend and weight loader would otherwise trust.
void TestCorruptOverlayConfigFailsLoad() {
    const nlohmann::json corruptions[] = {
        {{"num_hidden_layers", 28}},
        {{"hidden_size", 4096}},
        {{"intermediate_size", 8960}},
        {{"num_attention_heads", 32}},
        {{"num_key_value_heads", 4}},
        {{"head_dim", 64}},
        {{"vocab_size", 200065}},
        {{"rms_norm_eps", 1.0e-6}},
        {{"model_type", "phi3"}},
    };
    for (const auto& corruption : corruptions) {
        TempModelPackage package({200020, 199999}, 3072, corruption);
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
            "config.json");
        // The engine must never be constructed from a package whose declared
        // contract does not match the validated one.
        CHECK(g_factory.calls == 0);
    }

    // A config.json that is not even parseable must fail the same way rather
    // than falling through to a partially initialized model.
    {
        TempModelPackage package;
        package.OverwriteFile("config.json", "{not json");
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
            "config.json");
        CHECK(g_factory.calls == 0);
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
    CHECK(model->uses_corelib_aie4());
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

void TestEnginePositionIsAuthoritativeForCapUpdate() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens.assign(10, 11);
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    engine->position = 11;

    model->set_max_length(11);
    CHECK(model->get_max_length() == 11);
    CHECK(engine->max_length == 11);

    CheckThrowsContains(
        [&] { model->set_max_length(10); },
        "engine position");
    CHECK(model->get_max_length() == 11);
    CHECK(engine->max_length == 11);
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

void CheckAligned(
    Phi4& model,
    const FakeEngine& engine,
    const std::vector<int>& expected_history) {
    CHECK(Phi4FrontendTestAccess::History(model) == expected_history);
    CHECK(
        model.get_current_context_length() ==
        static_cast<int>(expected_history.size()));
    CHECK(
        engine.position ==
        static_cast<int>(expected_history.size()));
}

void TestDefaultChatLimitWithoutExplicitRequestIsAdmitted() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(4096, "corelib_aie4"),
        -1,
        false,
        nullptr);

    g_encoded_tokens = {41};
    SetSamples({200020});
    auto meta = Meta();
    auto input = Prompt();
    std::ostringstream output;
    CHECK(
        model->generate_with_prompt(
            meta,
            input,
            4096,
            output)
            .empty());
    CHECK(!input.requested_max_new_tokens.has_value());
    CheckAligned(*model, *g_factory.engine, {41, 200020});
}

void TestEndpointLimitsAtLoweredCap() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(512, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    for (const GenerationEndpoint endpoint : {
             GenerationEndpoint::Generate,
             GenerationEndpoint::OpenAiChatCompletion,
             GenerationEndpoint::OpenAiCompletion}) {
        const ParsedGenerationLimit omitted =
            ParseGenerationLimit(
                nlohmann::ordered_json::object(),
                endpoint);
        CHECK(!omitted.explicit_limit);
        CHECK(GenerationLoopLimit(omitted, true) == -1);

        g_encoded_tokens = {41};
        g_sample_tokens.clear();
        g_sample_token = 7;
        auto meta = Meta();
        auto input = Prompt(RequestedMaxNewTokens(omitted));
        CHECK(model->insert(meta, input));
        std::ostringstream output;
        (void)model->generate(
            meta,
            GenerationLoopLimit(omitted, true),
            output);
        CHECK(meta.generated_tokens == 511);
        CHECK(model->get_current_context_length() == 512);
        CHECK(engine->position == 512);

        model->clear_context();
        nlohmann::ordered_json explicit_request;
        if (endpoint ==
            GenerationEndpoint::OpenAiChatCompletion) {
            explicit_request["max_completion_tokens"] = 512;
        } else {
            explicit_request["max_tokens"] = 512;
        }
        const ParsedGenerationLimit explicit_limit =
            ParseGenerationLimit(explicit_request, endpoint);
        CHECK(explicit_limit.explicit_limit);
        CHECK(explicit_limit.value == 512);

        meta = Meta();
        input = Prompt(RequestedMaxNewTokens(explicit_limit));
        const int clear_count = engine->clear_count;
        const auto prefill_count = engine->prefill_calls.size();
        CheckRequestError(
            [&] { (void)model->insert(meta, input); },
            400,
            false,
            "512");
        CHECK(engine->position == 0);
        CHECK(engine->clear_count == clear_count);
        CHECK(engine->prefill_calls.size() == prefill_count);
        CHECK(Phi4FrontendTestAccess::History(*model).empty());
    }
}

void TestLengthStopCommitsTokensBeforeForcedAppend() {
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
    SetSamples({101, 102});
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    std::ostringstream output;
    CHECK(
        model->generate(meta, 2, output) ==
        "token-101token-102");
    CHECK(meta.generated_tokens == 2);
    CHECK(meta.stop_reason == MAX_LENGTH_REACHED);
    CheckAligned(*model, *engine, {1, 2, 101, 102});

    model->set_max_length(16);
    CHECK(model->get_max_length() == 16);
    CHECK(engine->max_length == 16);

    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Append);
    engine->prefill_calls.clear();
    g_encoded_tokens = {1, 2, 101, 102, 3};
    SetSamples({103});
    auto append_meta = Meta();
    CHECK(model->insert(append_meta, input));
    CHECK(engine->prefill_calls == std::vector<std::vector<int>>({{3}}));
    CheckAligned(*model, *engine, {1, 2, 101, 102, 3});
}

void TestEosStopCommitsTokenBeforeCapUpdateAndAppend() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens = {10};
    SetSamples({101, 200020});
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    std::ostringstream output;
    CHECK(model->generate(meta, 8, output) == "token-101");
    CHECK(meta.generated_tokens == 2);
    CHECK(meta.stop_reason == EOT_DETECTED);
    CheckAligned(*model, *engine, {10, 101, 200020});

    model->set_max_length(8);
    CHECK(model->get_max_length() == 8);
    CHECK(engine->max_length == 8);

    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Append);
    engine->prefill_calls.clear();
    g_encoded_tokens = {10, 101, 200020, 11};
    SetSamples({102});
    auto append_meta = Meta();
    CHECK(model->insert(append_meta, input));
    CHECK(engine->prefill_calls == std::vector<std::vector<int>>({{11}}));
    CheckAligned(*model, *engine, {10, 101, 200020, 11});
}

// The decode loop must stop one step BEFORE the declared context length.
//
// Measured on the AIE4 target 2026-09-02: a default /api/chat request with no
// limit field ran the decode loop to position 4095, where the token attention
// kernel has no 4096-wide window. The refusal arrives from flat_mha after q, k
// and v have been submitted in the same step, so it is past the irrevocable
// boundary and the process was TERMINATED, taking the server with it. The
// fatal record read:
//
//   {"status":3,"call":"flat_mha","phase":"flat_mha","layer":0,"rows":1,
//    "position":4095,
//    "detail":"no token attention kernel ships for a 4096-token window"}
//
// Design 11.2 and SEQ-4 require an unbounded request to stop before the
// user-visible total would exceed the cap. That bound existed for the
// explicit-limit path and not for the AIE4 no-limit path, where
// GenerationLoopLimit returns kNoExplicitGenerationLimit and the loop was
// bounded only by MAX_L -- one step too many -- and by EOS.
//
// This pins the boundary WITHOUT relying on EOS arriving: the sampler here
// never emits an end token, which is exactly the condition under which the
// defect fired.
void TestDecodeStopsBelowTheTokenAttentionWindow() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(4096, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;
    CHECK(engine->max_length == 4096);

    // One prompt token, then an unending stream of ordinary tokens. Nothing
    // here will ever stop the loop except the cap.
    g_encoded_tokens = {1};
    std::vector<int> samples;
    samples.reserve(5000);
    for (int index = 0; index < 5000; ++index) {
        samples.push_back(500 + (index % 100));
    }
    g_sample_tokens.assign(samples.begin(), samples.end());

    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));

    std::ostringstream output;
    // A limit far beyond the cap, standing in for the no-limit path: on AIE4
    // GenerationLoopLimit returns kNoExplicitGenerationLimit when no field is
    // present, so the cap is the only thing that can stop this.
    (void)model->generate(meta, 100000, output);

    CHECK(meta.stop_reason == MAX_LENGTH_REACHED);
    // 4095, not 4096. The last position the engine is asked to run at is
    // 4094, whose attention window is 4095 -- the largest the token kernel
    // serves. One more would be the step that killed the server.
    CHECK(
        model->get_current_context_length() ==
        static_cast<int>(flm::phi4::constants::kMaxDecodeWindow));
    CHECK(engine->position == flm::phi4::constants::kMaxDecodeWindow);
    CHECK(engine->position < 4096);
}

// Admission must refuse a request it cannot finish.
//
// A rendered prompt plus an explicit output that totals exactly the declared
// 4096 was previously admitted, because the cap compared against MAX_L. The
// final decode step of such a request is the unservable one, so admitting it
// meant accepting a request whose only possible outcome was a dead process.
void TestAdmissionRefusesTheUnservableFinalStep() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(4096, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens.assign(10, 7);

    // 10 + 4086 == 4096 == MAX_L. Refused: the last step cannot be served.
    auto meta = Meta();
    auto rejected = Prompt(4086);
    CheckRequestError(
        [&] { (void)model->insert(meta, rejected); },
        400,
        false,
        "exceeds the active context cap");
    CHECK(engine->position == 0);

    // 10 + 4085 == 4095. Admitted, because every step of it can run.
    SetSamples({101});
    auto accepted = Prompt(4085);
    CHECK(model->insert(meta, accepted));
    CHECK(engine->position == 10);
}

void TestActiveCapNeverEmitsUncommittedToken() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(3, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens = {1, 2};
    SetSamples({101, 102});
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    std::ostringstream output;
    CHECK(model->generate(meta, 8, output) == "token-101");
    CHECK(meta.generated_tokens == 1);
    CHECK(meta.stop_reason == MAX_LENGTH_REACHED);
    CHECK(engine->forward_tokens == std::vector<int>({101}));
    CHECK(g_sample_tokens.size() == 1);
    CheckAligned(*model, *engine, {1, 2, 101});
}

void TestCancellationLeavesOnlyCommittedTokensVisible() {
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
    SetSamples({101, 102});
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    int cancellation_checks = 0;
    std::ostringstream output;
    CHECK(
        model->generate(
            meta,
            8,
            output,
            [&] { return cancellation_checks++ == 1; }) ==
        "token-101");
    CHECK(meta.generated_tokens == 1);
    CHECK(meta.stop_reason == CANCEL_DETECTED);
    CHECK(engine->forward_tokens == std::vector<int>({101}));
    CheckAligned(*model, *engine, {1, 2, 101});
}

void TestExactRepeatReprefillsAndSamplesFreshToken() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens = {1};
    SetSamples({101});
    auto first_meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(first_meta, input));
    std::ostringstream first_output;
    CHECK(model->generate(first_meta, 1, first_output) == "token-101");
    CheckAligned(*model, *engine, {1, 101});

    const int clear_before = engine->clear_count;
    engine->prefill_calls.clear();
    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Append);
    g_encoded_tokens = {1, 101};
    SetSamples({202});
    auto repeated_meta = Meta();
    CHECK(model->insert(repeated_meta, input));
    CHECK(engine->clear_count == clear_before + 1);
    CHECK(
        engine->prefill_calls ==
        std::vector<std::vector<int>>({{1, 101}}));
    std::ostringstream repeated_output;
    CHECK(
        model->generate(repeated_meta, 1, repeated_output) ==
        "token-202");
    CHECK(
        engine->forward_tokens ==
        std::vector<int>({101, 202}));
    CheckAligned(*model, *engine, {1, 101, 202});
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

// The frontend's Automatic route on BOTH sides of the generated threshold,
// plus the forced re-prefill it has to keep honouring.
//
// The earlier version of this test drove Automatic with a one-token suffix and
// asserted a re-prefill. That only held because the threshold was the
// placeholder zero: the test never exercised the Automatic append path at all,
// and its expectation inverted the moment calibration produced a positive
// constant. The suffix lengths below are derived from
// `kContinuationAppendThreshold` so that one case lands at the boundary the
// selector compares against and the other lands one token past it, whatever
// that constant is.
void TestForcedAndAutomaticReprefill() {
    constexpr int kThreshold =
        static_cast<int>(kContinuationAppendThreshold);

    TempModelPackage package;
    FactoryScope factory;
    // Enough context for the history plus the widest suffix this test builds.
    auto model = Load(
        package,
        ModelInfo(4 * (kThreshold + 4) + 64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    auto meta = Meta();
    auto input = Prompt();
    std::vector<int> history = {1, 2};
    g_encoded_tokens = history;
    CHECK(model->insert(meta, input));

    // Forced re-prefill: the whole rendered history goes through as one call
    // after a clear, regardless of how short the suffix is.
    engine->prefill_calls.clear();
    const int clear_before_forced = engine->clear_count;
    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Reprefill);
    history = {1, 2, 3, 4};
    g_encoded_tokens = history;
    CHECK(model->insert(meta, input));
    CHECK(engine->clear_count == clear_before_forced + 1);
    CHECK(engine->prefill_calls.size() == 1);
    CHECK(engine->prefill_calls[0] == history);
    CHECK(engine->position == static_cast<int>(history.size()));

    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Automatic);

    // Automatic at the boundary: a suffix of exactly `kThreshold` tokens is
    // "at most the threshold", so it appends one row at a time and never
    // clears. A zero threshold makes that unreachable, and Section 10.7 allows
    // a zero threshold, so that case asserts the other behaviour rather than
    // skipping.
    if (kThreshold > 0) {
        engine->prefill_calls.clear();
        const int clear_before_append = engine->clear_count;
        const int appended_from = static_cast<int>(history.size());
        std::vector<int> appended;
        for (int index = 0; index < kThreshold; ++index) {
            history.push_back(1000 + index);
            appended.push_back(1000 + index);
        }
        g_encoded_tokens = history;
        CHECK(model->insert(meta, input));
        CHECK(engine->clear_count == clear_before_append);
        CHECK(
            engine->prefill_calls.size() ==
            static_cast<std::size_t>(kThreshold));
        for (int index = 0; index < kThreshold; ++index) {
            CHECK(
                engine->prefill_calls[static_cast<std::size_t>(index)] ==
                std::vector<int>({appended[static_cast<std::size_t>(index)]}));
        }
        CHECK(engine->position == appended_from + kThreshold);
        CHECK(Phi4FrontendTestAccess::History(*model) == history);
    } else {
        engine->prefill_calls.clear();
        const int clear_before_zero = engine->clear_count;
        history.push_back(1000);
        g_encoded_tokens = history;
        CHECK(model->insert(meta, input));
        CHECK(engine->clear_count == clear_before_zero + 1);
        CHECK(engine->prefill_calls.size() == 1);
        CHECK(engine->prefill_calls[0] == history);
    }

    // Automatic one token past the boundary: a suffix of `kThreshold + 1`
    // clears and re-prefills the complete rendered history as one call.
    engine->prefill_calls.clear();
    const int clear_before_automatic = engine->clear_count;
    for (int index = 0; index <= kThreshold; ++index) {
        history.push_back(2000 + index);
    }
    g_encoded_tokens = history;
    CHECK(model->insert(meta, input));
    CHECK(engine->clear_count == clear_before_automatic + 1);
    CHECK(engine->prefill_calls.size() == 1);
    CHECK(engine->prefill_calls[0] == history);
    CHECK(engine->position == static_cast<int>(history.size()));
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
    CHECK(engine->forward_tokens == std::vector<int>({200020}));
    CheckAligned(*model, *engine, {10, 11, 200020});
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

void CheckPartialAppendFailureClearsSession(
    FakeEngine::FailureKind failure) {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens = {1};
    SetSamples({101});
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    engine->checkpoint();
    Phi4FrontendTestAccess::ForceRoute(
        *model,
        ForcedContinuationRoute::Append);
    engine->prefill_failure = failure;
    engine->successful_prefills_before_failure = 1;
    const int resets_before = g_sampler_reset_count;
    g_encoded_tokens = {1, 2, 3, 4};

    CheckRequestError(
        [&] { (void)model->insert(meta, input); },
        500,
        true,
        "current conversation was cleared");
    CHECK(engine->prefill_calls.back() == std::vector<int>({2}));
    CHECK(engine->position == 0);
    CHECK(!engine->checkpoint_position.has_value());
    CHECK(model->get_current_context_length() == 0);
    CHECK(Phi4FrontendTestAccess::History(*model).empty());
    CHECK(Phi4FrontendTestAccess::LastToken(*model) == -1);
    CHECK(g_sampler_reset_count > resets_before);
}

void TestPartialAppendStandardFailureClearsCommittedPrefix() {
    CheckPartialAppendFailureClearsSession(
        FakeEngine::FailureKind::Standard);
}

void TestPartialAppendUnknownFailureClearsCommittedPrefix() {
    CheckPartialAppendFailureClearsSession(
        FakeEngine::FailureKind::Unknown);
}

void TestStandardGenerateFailureClearsSession() {
    TempModelPackage package;
    FactoryScope factory;
    auto model = Load(
        package,
        ModelInfo(64, "corelib_aie4"),
        -1,
        false,
        nullptr);
    FakeEngine* engine = g_factory.engine;

    g_encoded_tokens = {31, 32};
    SetSamples({101});
    auto meta = Meta();
    auto input = Prompt();
    CHECK(model->insert(meta, input));
    engine->checkpoint();
    engine->forward_failure = FakeEngine::FailureKind::Standard;
    const int resets_before = g_sampler_reset_count;
    std::ostringstream output;

    CheckRequestError(
        [&] { (void)model->generate(meta, 8, output); },
        500,
        true,
        "current conversation was cleared");
    CHECK(engine->position == 0);
    CHECK(!engine->checkpoint_position.has_value());
    CHECK(model->get_current_context_length() == 0);
    CHECK(Phi4FrontendTestAccess::History(*model).empty());
    CHECK(Phi4FrontendTestAccess::LastToken(*model) == -1);
    CHECK(g_sampler_reset_count > resets_before);
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
    // Telemetry must report the generated constant, not a number this test
    // pinned when the header still held the placeholder zero.
    CHECK(
        profile.find(
            "Append threshold: " +
            std::to_string(kContinuationAppendThreshold)) !=
        std::string::npos);
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
        TestLoadedBackendControlsOmittedLimit();
        TestLegacyExactRepeatPreservesEmptyPrefillPayload();
#if defined(FLM_ENABLE_CORELIB_AIE4)
        ConfigureFakeCorelibDll();
        TestCorruptOverlayConfigFailsLoad();
        TestCorelibRoutingAndPreemption();
        TestInitialAndAtomicCaps();
        TestEnginePositionIsAuthoritativeForCapUpdate();
        TestRenderedCapacityIsAtomic();
        TestDefaultChatLimitWithoutExplicitRequestIsAdmitted();
        TestEndpointLimitsAtLoweredCap();
        TestLengthStopCommitsTokensBeforeForcedAppend();
        TestEosStopCommitsTokenBeforeCapUpdateAndAppend();
        TestDecodeStopsBelowTheTokenAttentionWindow();
        TestAdmissionRefusesTheUnservableFinalStep();
        TestActiveCapNeverEmitsUncommittedToken();
        TestCancellationLeavesOnlyCommittedTokensVisible();
        TestExactRepeatReprefillsAndSamplesFreshToken();
        TestForcedAppendAndCancellationAlignment();
        TestForcedAndAutomaticReprefill();
        TestEosValidationAndFrontendStop();
        TestRecoverableFailuresClearSession();
        TestPartialAppendStandardFailureClearsCommittedPrefix();
        TestPartialAppendUnknownFailureClearsCommittedPrefix();
        TestStandardGenerateFailureClearsSession();
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
