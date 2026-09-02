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
///       prompt, so that turn is prefilled once and pinned with a KV-cache
///       checkpoint; every later turn rewinds to it instead of clearing, and only
///       the source text is prefilled. The pin is taken from whichever system
///       turn insert() sees, so a REST caller gets it without any extra call;
///       set_system_prompt() is the in-process shortcut that also installs a
///       default system turn for prompts that carry none.

#pragma once
#include "AutoModel/automodel.hpp"

/************              hunyuan-dense family            **************/
class Hunyuan : public AutoModel {
private:
    void setup_tokenizer(std::string model_path);

    /// \brief prefill the reusable prefix of `system_text` and pin it
    /// \return the number of tokens pinned, 0 if none could be isolated
    /// \note clears the context. An empty string only unpins.
    int _pin_system_prefix(const std::string& system_text);

    /// \brief the system turn prepended to every prompt, empty when unset
    /// \note only set through set_system_prompt(); a system turn that arrives on
    ///       the request is used as it comes and never adopted as the default.
    std::string system_prompt;

    /// \brief the system text the current pin was built from, empty when unpinned
    std::string pinned_system_text;

    /// \brief the pinned token prefix, the history restore() rewinds to
    std::vector<int> system_his;

    /// \brief tokens currently pinned, 0 when nothing is
    int system_tokens = 0;

    /// \brief whether insert() may pin the system turn it sees, and rewind to it
    bool pin_enabled = true;

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
    /// \note optional. insert() prepends this turn to any prompt that does not
    ///       carry one of its own, and pins it. A caller that brings its own
    ///       system message -- every REST request does -- gets the same pin
    ///       without calling this at all: insert() pins the system turn it sees
    ///       and re-pins only when that text changes.
    ///       An empty string clears the system prompt and unpins.
    int set_system_prompt(const std::string& system_text);

    /// \brief allow or forbid pinning the system turn in the KV cache
    /// \note on by default. Turning it off drops any existing pin, so every
    ///       prompt prefills in full -- the control arm of an A/B, essentially.
    void set_prefix_pinning(bool enabled);

    /// \brief whether insert() pins the system turn it sees
    bool get_prefix_pinning() const { return this->pin_enabled; }

    /// \brief the system prompt currently prepended to every prompt
    const std::string& get_system_prompt() const { return this->system_prompt; }

    /// \brief tokens currently pinned ahead of every prompt
    int get_system_tokens() const { return this->system_tokens; }
};
