/// \file phi4.cpp
/// \brief phi4 class
/// \author FastFlowLM Team
/// \date 2025-09-04
/// \version 0.9.25
/// \note This is a source file for the phi4 class

#include "AutoModel/modeling_phi4.hpp"

#if defined(FLM_ENABLE_CORELIB_AIE4)
#include <models/phi4/phi4_corelib_constants.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

constexpr std::string_view kCorelibBackend = "corelib_aie4";
constexpr int kPhi4Eos = 200020;
constexpr int kPhi4End = 199999;

enum class Phi4Backend {
    Legacy,
    CorelibAie4
};

Phi4Backend ResolveBackend(const json& model_info) {
    const auto details = model_info.find("details");
    if (
        details == model_info.end() ||
        !details->is_object() ||
        !details->contains("execution_backend")) {
        return Phi4Backend::Legacy;
    }

    const auto& value = details->at("execution_backend");
    if (!value.is_string()) {
        throw std::invalid_argument(
            "Phi-4 details.execution_backend must be a string");
    }
    const std::string backend = value.get<std::string>();
    if (backend == kCorelibBackend) {
        return Phi4Backend::CorelibAie4;
    }
    throw std::invalid_argument(
        "Unknown Phi-4 execution backend '" + backend + "'");
}

std::uint32_t ResolveAie4ContextLength(
    const json& model_info,
    int requested_context_length) {
    std::int64_t value = requested_context_length;
    if (requested_context_length == -1) {
        if (!model_info.contains("default_context_length")) {
            throw std::invalid_argument(
                "Phi-4 AIE4 model metadata has no default_context_length");
        }
        value = model_info.at("default_context_length").get<std::int64_t>();
    }
    if (value <= 0 || value > 4096) {
        throw std::out_of_range(
            "Phi-4 AIE4 maximum length must be in 1..4096");
    }
    return static_cast<std::uint32_t>(value);
}

void RequireAie4FrontendFile(
    const std::filesystem::path& model_path,
    std::string_view filename) {
    const auto path = model_path / filename;
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Phi-4 AIE4 package is missing required frontend file: " +
            path.string());
    }
}

#if defined(FLM_ENABLE_CORELIB_AIE4)

// Design `MODEL-2`. The overlay `config.json` exists only because the upstream
// repository ships none, and it restates the Section 5.1 constants so
// FastFlow's existing readers keep working. It is never an independent source
// of truth. `flm::phi4::constants` holds those same Section 5.1 values and is
// what the manifest loader validates the ONNX initializers against, so
// requiring the overlay to equal them is what makes a disagreement between the
// overlay and the real weights a hard load failure instead of a silent
// reconfiguration of the model.
void RequireAie4OverlayMatchesModelConstants(
    const std::filesystem::path& model_path) {
    namespace constants = flm::phi4::constants;
    const auto path = model_path / "config.json";
    json config;
    try {
        std::ifstream input(path, std::ios::binary);
        input.exceptions(std::ios::badbit);
        config = json::parse(input);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "Phi-4 AIE4 config.json could not be parsed: " + path.string() +
            ": " + error.what());
    }
    if (!config.is_object()) {
        throw std::runtime_error(
            "Phi-4 AIE4 config.json must be a JSON object: " + path.string());
    }

    const auto require_integer =
        [&](std::string_view field, std::int64_t expected) {
            const auto found = config.find(field);
            if (found == config.end() || !found->is_number_integer()) {
                throw std::runtime_error(
                    "Phi-4 AIE4 config.json is missing integer field '" +
                    std::string(field) + "': " + path.string());
            }
            const std::int64_t actual = found->get<std::int64_t>();
            if (actual != expected) {
                throw std::runtime_error(
                    "Phi-4 AIE4 config.json disagrees with the validated "
                    "model constants: " +
                    std::string(field) + " is " + std::to_string(actual) +
                    ", expected " + std::to_string(expected) +
                    ". The overlay restates the model contract and cannot "
                    "redefine it; regenerate it from the packaged model "
                    "rather than editing it.");
            }
        };

    require_integer("num_hidden_layers", constants::kLayerCount);
    require_integer("hidden_size", constants::kHiddenSize);
    require_integer("intermediate_size", constants::kIntermediateSize);
    require_integer("num_attention_heads", constants::kQueryHeadCount);
    require_integer("num_key_value_heads", constants::kKvHeadCount);
    require_integer("head_dim", constants::kHeadSize);
    require_integer("vocab_size", constants::kVocabularySize);

    const auto model_type = config.find("model_type");
    if (
        model_type == config.end() ||
        !model_type->is_string() ||
        model_type->get<std::string>() != "phi4") {
        throw std::runtime_error(
            "Phi-4 AIE4 config.json must declare model_type \"phi4\": " +
            path.string());
    }

    const auto epsilon = config.find("rms_norm_eps");
    if (epsilon == config.end() || !epsilon->is_number()) {
        throw std::runtime_error(
            "Phi-4 AIE4 config.json is missing numeric field "
            "'rms_norm_eps': " +
            path.string());
    }
    if (epsilon->get<double>() != constants::kRmsEpsilon) {
        throw std::runtime_error(
            "Phi-4 AIE4 config.json disagrees with the validated model "
            "constants: rms_norm_eps does not equal the packed epsilon. "
            "The overlay restates the model contract and cannot redefine it.");
    }
}

#endif  // FLM_ENABLE_CORELIB_AIE4

void ConfigureDefaultSampler(Phi4& model) {
    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.9;
    config.min_p = 0.1;
    config.temperature = 0.8;
    model.set_sampler(config);
}

}  // namespace

#if defined(FLM_CORELIB_TESTING)
Phi4::EngineFactoryForTesting Phi4::engine_factory_for_testing_;
#endif

/************              Phi4 family            **************/
Phi4::Phi4(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Phi4") {}

void Phi4::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    const Phi4Backend backend = ResolveBackend(model_info);
    if (backend == Phi4Backend::CorelibAie4) {
#if !defined(FLM_ENABLE_CORELIB_AIE4)
        throw std::runtime_error(
            "This binary was built without Phi-4 AIE4 corelib support");
#else
        if (enable_preemption) {
            throw std::invalid_argument(
                "Phi-4 AIE4 corelib execution does not support preemption");
        }

        const std::uint32_t context_length =
            ResolveAie4ContextLength(
                model_info,
                default_context_length);
        const std::filesystem::path package_path(model_path);
        RequireAie4FrontendFile(package_path, "config.json");
        RequireAie4FrontendFile(package_path, "tokenizer.json");
        RequireAie4FrontendFile(
            package_path,
            "tokenizer_config.json");
        RequireAie4OverlayMatchesModelConstants(package_path);

        this->_shared_initialize_model_state(
            model_path,
            model_info,
            static_cast<int>(context_length));
        this->npu.reset();
        this->enable_preemption = false;
        this->setup_tokenizer(model_path, true);

        auto runtime =
            flm::corelib::CorelibRuntime::GetOrCreate(
                utils::get_executable_directory());
        std::unique_ptr<causal_lm> engine;
#if defined(FLM_CORELIB_TESTING)
        if (!engine_factory_for_testing_) {
            throw std::logic_error(
                "Phi-4 frontend test engine factory is not installed");
        }
        engine = engine_factory_for_testing_(
            true,
            *this->lm_config,
            nullptr,
            package_path,
            context_length);
#else
        engine = std::make_unique<flm::phi4::phi4_corelib_aie4>(
            *this->lm_config,
            package_path,
            runtime,
            context_length);
#endif
        engine->clear_context();

        this->lm_engine = std::move(engine);
        this->corelib_runtime_ = std::move(runtime);
        this->uses_corelib_aie4_ = true;
        this->last_continuation_route_.reset();
        this->last_continuation_ns_ = 0;
        this->append_continuation_ns_ = 0;
        this->reprefill_continuation_ns_ = 0;
        this->sampler.reset();
        ConfigureDefaultSampler(*this);
#endif
    } else {
#if defined(FLM_ENABLE_CORELIB_AIE4)
        this->uses_corelib_aie4_ = false;
        this->corelib_runtime_.reset();
        this->last_continuation_route_.reset();
#if defined(FLM_CORELIB_TESTING)
        this->metrics_for_testing_.reset();
#endif
#endif
        this->_shared_load_model(
            model_path,
            model_info,
            default_context_length,
            enable_preemption);

        std::unique_ptr<causal_lm> engine;
#if defined(FLM_CORELIB_TESTING)
        if (!engine_factory_for_testing_) {
            throw std::logic_error(
                "Phi-4 frontend test engine factory is not installed");
        }
        engine = engine_factory_for_testing_(
            false,
            *this->lm_config,
            this->npu.get(),
            std::filesystem::path(model_path),
            this->MAX_L);
#else
        this->q4nx = std::make_unique<Q4NX>(this->model_path);
        engine = std::make_unique<phi4_npu>(
            *this->lm_config,
            this->npu.get(),
            this->MAX_L);
        engine->load_weights(*this->q4nx);
        this->q4nx.reset();
#endif

        engine->clear_context();
        this->setup_tokenizer(model_path, false);
        this->lm_engine = std::move(engine);
        this->sampler.reset();
        ConfigureDefaultSampler(*this);
    }

    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Phi4::setup_tokenizer(
    const std::string& model_path,
    bool require_aie4_eos) {
#ifdef _WIN32
    std::string tokenizer_config_path = model_path + "\\tokenizer_config.json";
#else
    std::string tokenizer_config_path = model_path + "/tokenizer_config.json";
#endif
    std::ifstream fs_config(tokenizer_config_path, std::ios::in | std::ios::binary);
    if (fs_config.fail()) {
        throw std::runtime_error(
            "Cannot open " + tokenizer_config_path);
    }
    std::string data_config;
    fs_config.seekg(0, std::ios::end);
    const auto end = fs_config.tellg();
    if (end < 0) {
        throw std::runtime_error(
            "Cannot read " + tokenizer_config_path);
    }
    size_t size_config = static_cast<size_t>(end);
    fs_config.seekg(0, std::ios::beg);
    data_config.resize(size_config);
    fs_config.read(
        data_config.data(),
        static_cast<std::streamsize>(size_config));
    if (!fs_config && size_config != 0) {
        throw std::runtime_error(
            "Cannot read " + tokenizer_config_path);
    }
    fs_config.close();
    auto tokenizer_config = nlohmann::json::parse(data_config);

    if (
        !tokenizer_config.contains("chat_template") ||
        !tokenizer_config.at("chat_template").is_string()) {
        throw std::invalid_argument(
            "Phi-4 tokenizer_config.json requires a string chat_template");
    }
    if (!tokenizer_config.contains("eos_token_id")) {
        throw std::invalid_argument(
            "Phi-4 tokenizer_config.json requires eos_token_id");
    }

    auto chat_template = std::make_unique<minja::chat_template>(
        tokenizer_config.at("chat_template").get<std::string>(),
        "",
        "");
    std::vector<int> parsed_eos_ids;

    const auto& eos_ids = tokenizer_config.at("eos_token_id");
    if (eos_ids.is_number_integer()) {
        parsed_eos_ids.push_back(eos_ids.get<int>());
    } else if (eos_ids.is_array()) {
        for (const auto& token : eos_ids) {
            parsed_eos_ids.push_back(token.get<int>());
        }
    } else {
        throw std::invalid_argument(
            "Phi-4 tokenizer_config.json eos_token_id must be "
            "an integer or array");
    }

    if (require_aie4_eos) {
        const auto has_id = [&](int id) {
            return std::find(
                       parsed_eos_ids.begin(),
                       parsed_eos_ids.end(),
                       id) != parsed_eos_ids.end();
        };
        if (!has_id(kPhi4Eos) || !has_id(kPhi4End)) {
            throw std::invalid_argument(
                "Phi-4 AIE4 tokenizer_config.json must contain "
                "EOS token IDs 200020 and 199999");
        }
    }

    this->has_bos_token = false;
    this->chat_tmpl = std::move(chat_template);
    this->bos_token_id = -1;
    this->eos_token = "";
    this->eos_token_ids = std::move(parsed_eos_ids);
    this->user_system_prompt = "";
    this->extra_context["user_system_prompt"] = this->user_system_prompt;
}

std::string Phi4::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->chat_tmpl->apply(inputs);
}

#if defined(FLM_ENABLE_CORELIB_AIE4)
void Phi4::validate_aie4_capacity(
    size_t rendered_tokens,
    std::optional<int> requested_max_new_tokens) const {
    if (
        requested_max_new_tokens.has_value() &&
        *requested_max_new_tokens < 0) {
        throw ModelRequestError(
            400,
            false,
            "Phi-4 AIE4 requested_max_new_tokens cannot be negative");
    }

    const size_t active_cap = this->MAX_L;
    const size_t requested =
        requested_max_new_tokens.has_value()
            ? static_cast<size_t>(*requested_max_new_tokens)
            : 0;
    const bool prompt_has_no_generation_room =
        rendered_tokens >= active_cap;
    const bool explicit_request_exceeds_cap =
        requested_max_new_tokens.has_value() &&
        (rendered_tokens > active_cap ||
         requested > active_cap - rendered_tokens);
    if (
        prompt_has_no_generation_room ||
        explicit_request_exceeds_cap) {
        std::stringstream message;
        message
            << "Phi-4 AIE4 request exceeds the active context cap "
            << active_cap << ": rendered prompt has "
            << rendered_tokens << " tokens";
        if (requested_max_new_tokens.has_value()) {
            message << " and requested output has "
                    << requested << " tokens";
        }
        throw ModelRequestError(
            400,
            false,
            message.str());
    }
}

void Phi4::clear_after_corelib_error() {
    AutoModel::clear_context();
}

std::string Phi4::generate_aie4(
    chat_meta_info_t& meta_info,
    int length_limit,
    std::ostream& os,
    std::function<bool()> is_cancelled) {
    std::string result;
    stop_reason_t reason = EOT_DETECTED;
    int generated_this_call = 0;

    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();

    if (this->last_token == -1) {
        throw std::logic_error(
            "Phi-4 AIE4 generation has no sampled token");
    }

    while (
        this->last_token != -1 &&
        this->total_tokens < this->MAX_L) {
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            buffer_.clear();
            current_mode_ = StreamEventType::CONTENT;
            tool_name_.clear();
            is_in_tool_block_ = false;
            break;
        }

        const int committed_token = this->last_token;
        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> logits =
            this->lm_engine->forward(committed_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->token_history.push_back(committed_token);
        ++this->total_tokens;
        ++meta_info.generated_tokens;
        ++generated_this_call;

        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(committed_token)) {
            const std::string token_str =
                this->tokenizer->run_time_decoder(committed_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);

        if (this->is_eos(committed_token)) {
            this->last_token = -1;
            break;
        }
        if (
            length_limit > 0 &&
            generated_this_call >= length_limit) {
            this->last_token = -1;
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (this->total_tokens >= this->MAX_L) {
            this->last_token = -1;
            reason = MAX_LENGTH_REACHED;
            break;
        }

        this->profiler_list[SAMPLING_TIME].start();
        this->last_token = this->sampler->sample(logits);
        this->profiler_list[SAMPLING_TIME].stop(1);
    }

    if (this->total_tokens >= this->MAX_L) {
        this->last_token = -1;
        reason = MAX_LENGTH_REACHED;
        header_print(
            "WARNING",
            "Max length reached, stopping generation...");
    }
    meta_info.decoding_duration =
        static_cast<uint64_t>(
            time_utils::cast_to_us(
                this->profiler_list[DECODING_TIME]
                    .get_total_time())
                .first) *
        1e3;
    meta_info.stop_reason = reason;
    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);
    return result;
}
#endif

bool Phi4::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

#if defined(FLM_ENABLE_CORELIB_AIE4)
    if (this->uses_corelib_aie4_) {
        this->validate_aie4_capacity(
            tokens.size(),
            input.requested_max_new_tokens);
        meta_info.stop_reason = EOT_DETECTED;

        const size_t history_size = this->token_history.size();
        const size_t matched = this->_matching_prefix_length(tokens);
        const bool prefix_hit =
            history_size != 0 && matched == history_size;
        const size_t suffix_tokens =
            prefix_hit ? tokens.size() - matched : tokens.size();
        const flm::phi4::ContinuationRoute route =
            prefix_hit
                ? flm::phi4::SelectContinuationRoute(
                      suffix_tokens,
                      this->forced_continuation_route_)
                : flm::phi4::ContinuationRoute::Reprefill;
        const PrefixHitAction action =
            route == flm::phi4::ContinuationRoute::Append
                ? PrefixHitAction::AppendSuffixOneByOne
                : PrefixHitAction::RecomputeFull;
        const auto started = std::chrono::steady_clock::now();

        bool inserted = false;
        try {
            inserted = this->_shared_insert(
                meta_info,
                tokens,
                std::move(is_cancelled),
                nullptr,
                0,
                action);
        } catch (const ModelRequestError&) {
            throw;
        } catch (const flm::corelib::CorelibError&) {
            this->clear_after_corelib_error();
            throw ModelRequestError(
                500,
                true,
                "AIE4 inference failed before submission; the "
                "current conversation was cleared.");
        } catch (const std::exception&) {
            this->clear_after_corelib_error();
            throw ModelRequestError(
                500,
                true,
                "AIE4 inference failed before submission; the "
                "current conversation was cleared.");
        } catch (...) {
            this->clear_after_corelib_error();
            throw ModelRequestError(
                500,
                true,
                "AIE4 inference failed before submission; the "
                "current conversation was cleared.");
        }

        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started)
                .count();
        this->last_continuation_ns_ =
            static_cast<std::uint64_t>(
                std::max<std::int64_t>(elapsed, 0));
        this->last_continuation_route_ = route;
        if (route == flm::phi4::ContinuationRoute::Append) {
            this->append_continuation_ns_ +=
                this->last_continuation_ns_;
        } else {
            this->reprefill_continuation_ns_ +=
                this->last_continuation_ns_;
        }
        return inserted;
    }
#endif

    return this->_shared_insert(
        meta_info,
        tokens,
        std::move(is_cancelled));
}


std::string Phi4::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os,  std::function<bool()> is_cancelled) {
#if defined(FLM_ENABLE_CORELIB_AIE4)
    if (this->uses_corelib_aie4_) {
        try {
            return this->generate_aie4(
                meta_info,
                length_limit,
                os,
                std::move(is_cancelled));
        } catch (const ModelRequestError&) {
            throw;
        } catch (const flm::corelib::CorelibError&) {
            this->clear_after_corelib_error();
            throw ModelRequestError(
                500,
                true,
                "AIE4 inference failed before submission; the "
                "current conversation was cleared.");
        } catch (const std::exception&) {
            this->clear_after_corelib_error();
            throw ModelRequestError(
                500,
                true,
                "AIE4 inference failed before submission; the "
                "current conversation was cleared.");
        } catch (...) {
            this->clear_after_corelib_error();
            throw ModelRequestError(
                500,
                true,
                "AIE4 inference failed before submission; the "
                "current conversation was cleared.");
        }
    }
#endif
    return this->_shared_generate(
        meta_info,
        length_limit,
        os,
        std::move(is_cancelled));
}

std::string Phi4::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}

void Phi4::set_max_length(unsigned int requested_max_length) {
#if defined(FLM_ENABLE_CORELIB_AIE4)
    if (this->uses_corelib_aie4_) {
        if (
            requested_max_length == 0 ||
            requested_max_length > 4096) {
            throw std::out_of_range(
                "Phi-4 AIE4 maximum length must be in 1..4096");
        }
        const int engine_position =
            this->lm_engine->get_current_context_length();
        if (
            requested_max_length <
            static_cast<unsigned int>(engine_position)) {
            throw std::out_of_range(
                "Phi-4 AIE4 maximum length cannot be below the "
                "current engine position");
        }

        this->lm_engine->update_max_length(requested_max_length);
        this->MAX_L = requested_max_length;
        return;
    }
#endif
    AutoModel::set_max_length(requested_max_length);
}

#if defined(FLM_ENABLE_CORELIB_AIE4)
const flm::phi4::Phi4Aie4Metrics&
Phi4::aie4_metrics() const {
#if defined(FLM_CORELIB_TESTING)
    if (this->metrics_for_testing_.has_value()) {
        return *this->metrics_for_testing_;
    }
#endif
    const auto* engine =
        dynamic_cast<const flm::phi4::phi4_corelib_aie4*>(
            this->lm_engine.get());
    if (engine == nullptr) {
        throw std::logic_error(
            "Phi-4 AIE4 profile requested for a non-corelib engine");
    }
    return engine->metrics();
}
#endif

std::string Phi4::show_profile() {
    const std::string base = AutoModel::show_profile();
#if defined(FLM_ENABLE_CORELIB_AIE4)
    if (this->uses_corelib_aie4_) {
        const auto& metrics = this->aie4_metrics();
        std::stringstream profile;
        profile << base;
        profile << "  Phi-4 AIE4:" << std::endl;
        profile << "    Engine: corelib_aie4" << std::endl;
        profile << "    Continuation route: "
                << (this->last_continuation_route_.has_value()
                        ? flm::phi4::ContinuationRouteName(
                              *this->last_continuation_route_)
                        : "none")
                << std::endl;
        profile << "    Append threshold: "
                << flm::phi4::kContinuationAppendThreshold
                << std::endl;
        profile << "    Corelib DLL: "
                << this->corelib_runtime_->api()->library_path().string()
                << std::endl;
        profile << "    Helper transitions: "
                << metrics.helper_transition_counts[0] << "/"
                << metrics.helper_transition_counts[1] << "/"
                << metrics.helper_transition_counts[2] << "/"
                << metrics.helper_transition_counts[3] << "/"
                << metrics.helper_transition_counts[4] << "/"
                << metrics.helper_transition_counts[5]
                << std::endl;
        profile << "    Cold model load: "
                << metrics.model_load_ns << " ns" << std::endl;
        profile << "    Cold weight pack: "
                << metrics.weight_pack_ns << " ns" << std::endl;
        profile << "    Continuation time: "
                << this->last_continuation_ns_ << " ns" << std::endl;
        profile << "    Warm append total: "
                << this->append_continuation_ns_ << " ns" << std::endl;
        profile << "    Warm reprefill total: "
                << this->reprefill_continuation_ns_ << " ns"
                << std::endl;
        profile << "    Dispatches: "
                << metrics.dispatch_count << std::endl;
        profile << "    Synchronizations: "
                << metrics.synchronize_count << std::endl;
        profile << "    Packed weights: "
                << metrics.packed_weight_bytes << " bytes"
                << std::endl;
        profile << "    Mapped source: "
                << metrics.mapped_source_bytes << " bytes"
                << std::endl;
        profile << "    KV storage: "
                << metrics.kv_bytes << " bytes" << std::endl;
        profile << "    Scratch storage: "
                << metrics.scratch_bytes << " bytes" << std::endl;
        return profile.str();
    }
#endif
    return base;
}