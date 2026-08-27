/// \file modeling_qwen3_5_omni.hpp
/// \brief Qwen3_5_Omni driver class
/// \author FastFlowLM Team
/// \note Inherits AutoModel. The underlying engine (qwen3_5_omni) is NOT a
///       causal_lm -- its prefill()/forward() return qwen3_5_omni_thinker_result_t
///       (logits + hidden_states) rather than buffer<bf16>. So this class keeps
///       its own `engine` member (leaving the inherited `lm_engine` null) and
///       overrides the engine-driven methods instead of reusing AutoModel's
///       _shared_* helpers.
#pragma once

#include "AutoModel/automodel.hpp" // base class + shared types
#include "models/qwen3_5_omni/qwen3_5_omni.hpp"

#include "image/image_reader.hpp"
#include "audio/audio_reader.hpp"
#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"
#include "base64.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/************              Qwen3_5_Omni            **************/
class Qwen3_5_Omni : public AutoModel {
public:
    explicit Qwen3_5_Omni(flm_rt::device* npu_device_inst);
    ~Qwen3_5_Omni() override = default;

    /// \brief Load config + weights and set up tokenizer / sampler.
    void load_model(std::string model_path, json model_info, int default_context_length = -1, bool enable_preemption = false) override;

    /// \brief Apply the chat template to the messages.
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;

    /// \brief Encode + preprocess the input (text/image/audio) and prefill.
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;

    /// \brief Decode loop.
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;

    /// \brief insert() followed by generate().
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;

    NonStreamResult parse_nstream_content(const std::string response_text);
    StreamResult parse_stream_content(const std::string content);
    StreamResult parse_stream_content_final(const std::string content) override;

    /// \brief Synthesize speech from the last thinker result (talker path).
    /// \note Talker/codec/WAV-writer are not implemented in the engine yet; this
    ///       surfaces a clear message instead of crashing.
    buffer<bf16> say(std::string wav_out_path = "");

    /// \brief Sampler / context / bookkeeping overrides (engine-driven).
    void set_sampler(sampler_config& sampler_config) override;
    void clear_context() override;
    void set_max_length(unsigned int MAX_L) override;
    int get_current_context_length() override;
    std::string show_model_info() override;
    std::string show_profile() override;
    std::pair<std::string, std::vector<int>> get_history() override;

        /// \brief Configure a parameter with type-erased value
	/// \param parameter_name the name of the parameter
	/// \param value the value to set (can be any type)
	/// \return true if the parameter was configured successfully, false otherwise
	bool configure_parameter(std::string parameter_name, const std::any& value) override{
		if (parameter_name == "system_prompt") {
			try {
				this->user_system_prompt = std::any_cast<std::string>(value);
				this->extra_context["user_system_prompt"] = this->user_system_prompt;
				return true;
			} catch (const std::bad_any_cast&) {
				return false;
			}
		}
        else if (parameter_name == "img_pre_resize") {
            try {
                this->image_pre_resize = std::any_cast<int>(value);
                int target_size;
                if (this->image_pre_resize <= 0) {
                    target_size = 0;
                } else if (this->image_pre_resize == 1) {
                    target_size = 480;
                } else if (this->image_pre_resize <= 2) {
                    target_size = 720;
                } else if (this->image_pre_resize <= 3) {
                    target_size = 1080;
                } else if (this->image_pre_resize <= 4) {
                    target_size = 1440;
                } else if (this->image_pre_resize <= 5) {
                    target_size = 2160;
                } else if (this->image_pre_resize <= 6) {
                    target_size = 2880;
                } else if (this->image_pre_resize <= 7) {
                    target_size = 3240;
                } else if (this->image_pre_resize <= 8) {
                    target_size = 4320;
                } else {
                    this->image_pre_resize = 0;
                    target_size = 0;
                }
                if (this->image_pre_resize > 0) {
                    header_print_r("FLM", "Qwen3_5-Omni pre-resize image height to " + std::to_string(target_size) + " pixels if larger than that");
                }
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
		return false;
	}


private:
    int think_start_id = 248068;
    int think_end_id = 248069;
    // thinker-scope multimodal special token ids (from thinker_config)
    static constexpr int image_token_id       = 248056;
    static constexpr int audio_token_id       = 248076;
    static constexpr int audio_start_token_id = 248070;
    static constexpr int audio_end_token_id   = 248071;

    void setup_tokenizer(std::string model_path);

    // ----- image preprocessing (modeling_qwen3_5_omni_image.cpp) -----
    int image_pre_resize = 0;
    ImageReader image_reader_;
    qwen3_5_omni_image_t load_image(const std::string& filename);
    qwen3_5_omni_image_t load_image_base64(const std::string& base64_string);
    void smart_resize(int height, int width, int& h_bar, int& w_bar, int factor, int min_pixels, int max_pixels);
    void preprocess_image(qwen3_5_omni_image_t& image,
                          std::vector<bf16>& pixel_values,
                          std::vector<int>& image_grid_pair,
                          uint32_t& valid_patch_size,
                          uint32_t& num_soft_tokens);

    // ----- audio preprocessing (modeling_qwen3_5_omni_audio.cpp) -----
    AudioReader audio_reader_;
    audio_data_t load_audio(const std::string& filename, int resample_rate, MonoDownmixMode downmix = MonoDownmixMode::MEAN);
    audio_data_t load_audio_base64(const std::string& base64_str, int resample_rate, MonoDownmixMode downmix = MonoDownmixMode::MEAN);
    std::vector<audio_data_t> clip_audio_length(audio_data_t& audio, double max_duration_second);
    void extract_spectrogram(std::vector<audio_data_t>& audio_inputs, qwen3_5_omni_audio_payload_t& audio_payload);
    std::string get_audio_tokens(int audio_length ) ;
    

    void pad_audio(std::vector<audio_data_t>& audio_inputs);



    // vision config (thinker_config.vision_config), with VL fallbacks
    uint32_t patch_size          = 16;
    uint32_t temporal_patch_size = 2;
    uint32_t merge_size          = 2;
    uint32_t shortest_edge       = 65536;
    uint32_t longest_edge        = 16777216;
    float    vision_rescale_factor = 0.00392156862745098f;
    float    vision_image_mean      = 0.5f;
    float    vision_image_std       = 0.5f;

    // // audio input resample rate
    int audio_sampling_rate = 16000; 
    int audio_downsample_time = 4;
    int audio_downsample_chunk_size = 100;
    int audio_timestamp_interval = 60;
    int audio_hop_length=160;
    int audio_window_attention_length=56;



    inline int _get_feat_extract_output_length(int input_lengths){
        // exact python function
        // Given an input audio length(after pre-processded), it computes the output legnth of the audio encoder( sequence length of the audio prefill)

        int input_lengths_leave = input_lengths % audio_downsample_chunk_size;
        if (input_lengths_leave > 0) {
            for (int i = 0; i < audio_downsample_time; i++){
                input_lengths_leave = (input_lengths_leave - 1) / 2 + 1;
            }
        }

        return input_lengths_leave + (input_lengths / audio_downsample_chunk_size) * std::ceil(100.0f / std::pow(2,audio_downsample_time));



    }
    

    // ----- owned engine (not a causal_lm; inherited lm_engine stays null) -----
    std::unique_ptr<qwen3_5_omni> engine = nullptr;
    qwen3_5_omni_thinker_result_t last_thinker_result;

    StreamResult parse_stream_content_impl(const std::string content, bool is_final);

};
