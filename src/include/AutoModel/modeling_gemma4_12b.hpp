/// \file modeling_gemma4_12b.hpp
/// \brief Gemma4_12B class
/// \author FastFlowLM Team
/// \date 2026-08-25
/// \version 0.9.45
/// \note This is a header file for the Gemma4_12B class.
///       Gemma4-12B is an omni model (text + image + audio); only the text
///       interface is wired up here for now. The image/audio front-ends are
///       intentionally left out -- see modeling_gemma4e.hpp for the
///       multi-modal reference implementation.

#pragma once
#include "AutoModel/automodel.hpp"
#include "metrices.hpp"
#include "typedef.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

/************              Gemma4_12B (text only)            **************/
class Gemma4_12B : public AutoModel {
private:
    bool enable_think = false;

    void setup_tokenizer(std::string model_path);

    StreamResult parse_stream_content_impl(const std::string content, bool is_final);

public:
    Gemma4_12B(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    NonStreamResult parse_nstream_content(const std::string response_text) override;
    StreamResult parse_stream_content(const std::string content) override;
    StreamResult parse_stream_content_final(const std::string content) override;
    chat_template_type_t get_chat_template_type() override {
        return chat_template_type_t::gemma4;
    }

    /// \brief Configure a parameter with type-erased value
    /// \param parameter_name the name of the parameter
    /// \param value the value to set (can be any type)
    /// \return true if the parameter was configured successfully, false otherwise
    bool configure_parameter(std::string parameter_name, const std::any& value) override {
        if (parameter_name == "enable_think") {
            try {
                this->enable_think = std::any_cast<bool>(value);
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "reasoning_effort") {
            std::string reasoning_effort;
            try {
                reasoning_effort = std::any_cast<std::string>(value);
                if (reasoning_effort == "high" || reasoning_effort == "medium" || reasoning_effort == "low")
                    this->enable_think = true;
                else if (reasoning_effort == "none")
                    this->enable_think = false;
                else
                    header_print("WARNING", "Reasoning effort must be 'none', 'low', 'medium' or 'high'!");
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "toggle_think") {
            this->enable_think = !this->enable_think;
            return true;
        }
        else if (parameter_name == "system_prompt") {
            try {
                this->user_system_prompt = std::any_cast<std::string>(value);
                this->extra_context["user_system_prompt"] = this->user_system_prompt;
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }

        return false;
    }
};
