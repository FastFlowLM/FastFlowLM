/// \file modeling_granite.hpp
/// \brief IBM Granite (dense) family.
/// \note Pairs the open host engine in models/granite/granite_npu.hpp with
///       FLM's tokenizer, chat template and sampler. granite-4.2 is a thinking
///       and tool-calling model, so the sampler defaults follow the reasoning
///       families rather than plain Llama3's.

#pragma once
#include "AutoModel/automodel.hpp"

/************              Granite family            **************/
class Granite : public AutoModel {
private:
    void setup_tokenizer(std::string model_path);

public:
    Granite(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
};
