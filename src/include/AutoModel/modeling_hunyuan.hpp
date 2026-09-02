/// \file modeling_hunyuan.hpp
/// \brief Hunyuan class
/// \author FastFlowLM Team
/// \date 2026-09-01
/// \version 0.9.45
/// \note AutoModel wrapper for the `hunyuan-dense` engine (Hy-MT2-1.8B).
///       Language only: the family ships no vision / audio tower, so images and
///       audios on the uniform input are dropped with a warning.
///       The chat template has neither reasoning markers nor tool calls, so the
///       base-class pass-through stream / non-stream parsers are kept as is.

#pragma once
#include "AutoModel/automodel.hpp"

/************              hunyuan-dense family            **************/
class Hunyuan : public AutoModel {
private:
    void setup_tokenizer(std::string model_path);

public:
    Hunyuan(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    chat_template_type_t get_chat_template_type() override;
};
