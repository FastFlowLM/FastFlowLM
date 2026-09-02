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
#include "image/image_reader.hpp"
#include "audio/audio_reader.hpp"
#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
/************              Gemma4_12B (text only)            **************/
class Gemma4_12B : public AutoModel {
private:
    bool enable_think = false;

    void setup_tokenizer(std::string model_path);

    StreamResult parse_stream_content_impl(const std::string content, bool is_final);

    static constexpr int boc_token_id = 100; // begin of channel token id
    static constexpr int eoc_token_id = 101; // end of channel token id
    static constexpr int thought_token_id = 45518; // 'thought'
    static constexpr int new_line_token_id = 107; // '\n'


    static constexpr int boi_token_id = 255999; // begin of image token id
    static constexpr int image_token_id = 258880; // image token id
    static constexpr int eoi_token_id = 258882; // end of image token id

    static constexpr int boa_token_id = 256000; // begin of audio token id
    static constexpr int audio_token_id = 258881; // audio token id
    static constexpr int eoa_token_id = 258883; // end of audio token id

    ImageReader image_reader_;
    gemma4_12b_image_t load_image(const std::string& filename);
    gemma4_12b_image_t load_image_base64(const std::string& base64_string);


    AudioReader audio_reader_;
    audio_data_t load_audio(const std::string &filename, int resample_rate, MonoDownmixMode downmix = MonoDownmixMode::NONE);
    audio_data_t load_audio_base64(const std::string &base64_str, int resample_rate, MonoDownmixMode downmix);

    /// \note audio_samples_per_token (640 = 40 ms at 16 kHz) and
    ///       audio_max_soft_tokens (750 = 30 s) are preprocessing constants read
    ///       from processor_config.json; see gemma4_12b_npu::load_audio_preprocess_parameters.
    void extract_waveform_features(audio_data_t& audio,
        int audio_samples_per_token,
        int audio_max_soft_tokens,
        std::vector<bf16> & result_vector,
        int & num_frames,
        size_t sample_offset = 0
    );

    /// \brief Split an audio clip into consecutive chunks of at most
    ///        audio_max_soft_tokens frames (750 = 30 s) and extract the raw
    ///        waveform features of every chunk.
    /// \note The reference feature extractor caps a single clip at 30 s. A longer
    ///       clip must therefore be fed as several clips, each with its own
    ///       <|audio> ... <audio|> block, or everything past the first 30 s is
    ///       silently dropped.
    void extract_waveform_features_chunked(audio_data_t& audio,
        int audio_samples_per_token,
        int audio_max_soft_tokens,
        std::vector<std::vector<bf16>> & chunk_features,
        std::vector<int> & chunk_num_frames
    );

    // Overwritten in load_model() from processor_config.json; the literal is the
    // released Gemma4-12B value, used when the checkpoint ships no processor config.
    int image_softtoken_budget = 280;

    void preprocess_image(
      gemma4_12b_image_t &image,
      std::pair<int, int> & patch_element_per_patch,
      uint32_t & valid_patch_size, // the unpadded size per image
      std::vector<bf16> &pixel_values,
      std::vector<int> &image_grid_pairs, // [num_of_position_id][x, y]      
      uint32_t &num_soft_tokens
    );
    std::vector<uint8_t>  aspect_ratio_preserving_resize( 
        gemma4_12b_image_t& image,
        int patch_size,
        int max_patches,
        int pooling_kernel_size
    );
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
        else if (parameter_name == "image_max_tokens") {
            try {
                this->image_softtoken_budget = std::any_cast<int>(value);
                
                if(image_softtoken_budget != 70 && image_softtoken_budget != 140 &&
                    image_softtoken_budget != 280 && image_softtoken_budget != 560 && image_softtoken_budget != 1120) {
                    header_print("WARNING", "Invalid image budget value: " << image_softtoken_budget << ". Supported values are 70, 140, 280, 560, 1120. Using 280...");
                    this->image_softtoken_budget = 280;
                }
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }            
        }

        return false;
    }
};
