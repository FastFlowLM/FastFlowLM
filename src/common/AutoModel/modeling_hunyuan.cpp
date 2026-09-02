/// \file modeling_hunyuan.cpp
/// \brief Hunyuan class
/// \author FastFlowLM Team
/// \date 2026-09-01
/// \version 0.9.45
/// \note AutoModel wrapper for the `hunyuan-dense` engine (Hy-MT2-1.8B).

#include "AutoModel/modeling_hunyuan.hpp"
#include "models/hunyuan/hunyuan_npu.hpp"

/************              hunyuan-dense family            **************/
Hunyuan::Hunyuan(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Hunyuan") {
    // the translator emits one short line per turn and the caller already has it
    // from the stream / return value, so the raw dump would only double the log
    this->log_raw_output = false;
    // every turn either rewinds to the pinned prefix or clears, so the kv state
    // the trailing eos forward exists to preserve is discarded either way
    this->forward_on_eos = false;
}

void Hunyuan::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    this->lm_engine = std::make_unique<hunyuan_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    // free the mmap'd weights immediately
    this->q4nx.reset();

    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // Defaults published with the checkpoint (general.sampling.* in the GGUF).
    sampler_config config;
    config.top_k = 1;
    config.top_p = 0.8;
    config.temperature = 0.0;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Hunyuan::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

/// \note The template renders `system` only when it is messages[0]; every other
///       role but user / assistant is dropped. It never branches on `tools`, so
///       the argument is accepted for the interface and ignored.
std::string Hunyuan::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->chat_tmpl->apply(inputs);
}

chat_template_type_t Hunyuan::get_chat_template_type() {
    // One marker per role turn plus a trailing generation prompt: the same shape
    // the prompt cache assumes for chat_ml.
    return chat_template_type_t::chat_ml;
}

bool Hunyuan::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    // Language only: drop any multimodal payload rather than failing the request.
    if (!input.images.empty()) {
        header_print("WARNING", "Hy-MT2-1.8B is a text-only model, ignoring " << input.images.size() << " image(s)");
        input.images.clear();
        input.image_payload_types.clear();
    }
    if (!input.audios.empty()) {
        header_print("WARNING", "Hy-MT2-1.8B is a text-only model, ignoring " << input.audios.size() << " audio input(s)");
        input.audios.clear();
        input.audio_payload_types.clear();
    }

    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        for (auto& message : input.messages) {
            message.erase("images");
            message.erase("audios");
        }
        messages = input.messages;
    }
    else { // a pure text, usually from the cli
        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
    }

    // The template renders a system turn only at messages[0], and that is where
    // set_system_prompt() primed the pinned prefix, so it goes in front. A caller
    // that brought its own system turn keeps it: overriding it here would leave
    // the prompt no longer matching what is pinned.
    if (!this->system_prompt.empty() && messages[0].value("role", "") != "system") {
        messages.insert(messages.begin(), nlohmann::ordered_json{
            {"role", "system"}, {"content", this->system_prompt} });
    }
    templated_text = this->apply_chat_template(messages, input.tools);

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // hardware
    if (meta_info.restore_allowed && this->system_tokens > 0) {
        // rewind to the pinned system turn instead of clearing: _shared_insert
        // then prefix-matches against checkpoint_his and prefills only the tail.
        // The token history has to move back with the cache or that match fails.
        hunyuan_npu* engine = dynamic_cast<hunyuan_npu*>(this->lm_engine.get());
        this->total_tokens = engine->restore();
        this->token_history = this->checkpoint_his;
        // the turns are independent translations, so nothing carries over
        this->sampler->reset_penalties();
        this->last_token = -1;
    }
    return this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);
}

int Hunyuan::set_system_prompt(const std::string& system_text) {
    this->system_prompt = system_text;
    this->clear_context();
    this->system_tokens = 0;
    if (system_text.empty()) {
        return 0;  // no system turn to pin, every prompt prefills in full
    }

    // The system turn is not the whole of what the prompts share: the user-turn
    // marker that opens after it is constant too. Template two prompts that
    // differ only in the first character of the user text and take their longest
    // common token prefix, so the reusable span is exact, merges across the
    // role boundary included.
    auto templated = [&](const std::string& user_text) {
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        messages.push_back({ {"role", "system"}, {"content", system_text} });
        messages.push_back({ {"role", "user"}, {"content", user_text} });
        return this->apply_chat_template(messages);
    };
    std::vector<int> probe_a = this->tokenizer->encode(templated("A"));
    std::vector<int> probe_b = this->tokenizer->encode(templated("\xe4\xbd\xa0"));  // U+4F60, a cjk probe

    size_t shared = 0;
    while (shared < probe_a.size() && shared < probe_b.size() && probe_a[shared] == probe_b[shared]) {
        shared++;
    }
    if (shared == 0) {
        header_print("WARNING", "Could not isolate a reusable system prefix, falling back to full prefill");
        return 0;
    }

    std::vector<int> tokens(probe_a.begin(), probe_a.begin() + shared);
    chat_meta_info_t meta_info;
    meta_info.restore_allowed = false;
    if (!this->_shared_insert(meta_info, tokens, [] { return false; }, nullptr)) {
        this->clear_context();
        return 0;
    }

    // pin it: restore() comes back here, and checkpoint_his is what
    // _shared_insert prefix-matches the next prompt against
    this->checkpoint_his = this->token_history;
    hunyuan_npu* engine = dynamic_cast<hunyuan_npu*>(this->lm_engine.get());
    engine->checkpoint();
    this->system_tokens = static_cast<int>(shared);

    // the system turn costs nothing per turn from here on, so it should not sit
    // in the per-turn prefill statistics either
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    return this->system_tokens;
}

std::string Hunyuan::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Hunyuan::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
