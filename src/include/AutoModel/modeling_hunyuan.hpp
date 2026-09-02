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
///       The turns are independent translations driven by one constant system
///       prompt, so set_system_prompt() prefills that turn once and pins it with
///       a KV-cache checkpoint; every later turn rewinds to it instead of
///       clearing, and only the source text is prefilled.

#pragma once
#include "AutoModel/automodel.hpp"

/************              hunyuan-dense family            **************/
class Hunyuan : public AutoModel {
private:
    void setup_tokenizer(std::string model_path);

    /// \brief the system turn prepended to every prompt, empty when unset
    std::string system_prompt;

    /// \brief tokens pinned by set_system_prompt(), 0 when it never ran
    int system_tokens = 0;

public:
    Hunyuan(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    chat_template_type_t get_chat_template_type() override;

    /// \brief install a constant system prompt and pin its prefill in the KV cache
    /// \param system_text the system turn to prepend to every prompt
    /// \return the number of tokens pinned, 0 if the prefix could not be isolated
    /// \note call once after load_model(). insert() then prepends the system turn
    ///       to any prompt that does not carry one of its own, and turns that pass
    ///       restore_allowed rewind to this point instead of re-prefilling it.
    ///       An empty string clears the system prompt and unpins.
    int set_system_prompt(const std::string& system_text);

    /// \brief the system prompt currently prepended to every prompt
    const std::string& get_system_prompt() const { return this->system_prompt; }

    /// \brief tokens currently pinned ahead of every prompt
    int get_system_tokens() const { return this->system_tokens; }
};
