/// \file lm_config.hpp
/// \brief lm_config class
/// \author FastFlowLM Team
/// \date 2025-08-05
/// \version 0.9.10
/// \note This class is used to store the model configuration.
#pragma once

#include "typedef.hpp"
#include "utils/utils.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>

/// \brief read one parameter out of a config json
/// \note Same semantics as JSON_GET (missing OR null falls back to the default),
///       in expression form. Plain nlohmann .value() throws on a null, and real
///       configs do carry nulls (e.g. "sliding_window": null on qwen3).
template <typename T>
inline T cfg_get(const nlohmann::json& jc, const char* key, T default_value){
    if (jc.contains(key) && !jc[key].is_null()){
        return T(jc[key]);
    }
    return default_value;
}

/// \brief read a nested object out of a config json
/// \note Reference-returning counterpart of cfg_get, for walking nested configs
///       (thinker_config.text_config, ...). Returns a shared empty object when
///       the key is missing or null, so the JSON_GET below it still defaults.
inline const nlohmann::json& cfg_sub(const nlohmann::json& jc, const char* key){
    static const nlohmann::json empty = nlohmann::json::object();
    if (jc.contains(key) && !jc[key].is_null()){
        return jc[key];
    }
    return empty;
}

/// \brief LM_Config class
/// \note Model parameters are NOT cached as members. Everything comes from the
///       model's config.json, so every consumer reads what it needs straight out
///       of _json_config with JSON_GET. from_pretrained() only locates the file
///       and normalizes it, so that all readers share one canonical key set.
/// \note This class is passed by value across the FLM_DLL boundary. Its layout
///       must stay identical to FLM_DLL/include/lm_config.hpp.
class LM_Config{
    public:
        std::string model_path;
        std::string model_name;
        std::string exec_path;
        std::string flm_version;

        nlohmann::json _json_config;

        /// \brief read one model parameter out of config.json
        /// \note Defaults to u32 because most parameters are dimensions:
        ///       config.get("head_dim"), config.get<f32>("rms_norm_eps", 0.0f),
        ///       config.get<std::string>("vision_model_weight", "").
        template <typename T = u32>
        T get(const char* key, T default_value = T(0)) const {
            return cfg_get<T>(this->_json_config, key, default_value);
        }

        /// \brief read a nested config object (vision_config, audio_config, ...)
        /// \return the sub-object, or an empty object when absent/null
        const nlohmann::json& sub(const char* key) const {
            return cfg_sub(this->_json_config, key);
        }

        /// \brief from pretrained
        /// \param model_name the model name
        void from_pretrained(std::string model_name){
            this->_resolve_paths(model_name);
            this->_load_json();
            this->_normalize_multi_modal();
            JSON_GET(this->flm_version, this->_json_config, "flm_version", "0.0.0", std::string);
        }

        std::string _str(){
            return this->_str_from(this->_json_config);
        }
        LM_Config(){}

    protected:
        /// \brief resolve model_path / model_name / exec_path
        void _resolve_paths(const std::string& model_name){
            // #define DEV_BUILD
            #ifdef DEV_BUILD
            #ifdef __WINDOWS__
                this->exec_path = "..\\..\\..\\";
            #else
                this->exec_path = "../../../";
            #endif
            #else
                this->exec_path = utils::find_xclbin_path();
            #endif
            this->model_path = model_name;
            this->model_name = std::filesystem::path(model_name).filename().string();
        }

        /// \brief read model_path/config.json into _json_config
        void _load_json(){
            std::ifstream file(this->model_path + "/config.json");
            if (!file.is_open()){
                std::cerr << "Failed to open file: " << this->model_path << std::endl;
                exit(1);
            }
            // read the json file as a string
            std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            this->_json_config = nlohmann::json::parse(json_str);
            this->_load_processor_json();
        }

        /// \brief read model_path/processor_config.json into _json_config["processor_config"]
        /// \note config.json describes the network; the preprocessing constants
        ///       (rescale_factor, image_mean/std, patch_size, pooling_kernel_size,
        ///       sampling_rate, audio_samples_per_token, the soft token budgets)
        ///       live in the processor config the checkpoint ships alongside it.
        ///       Optional: models without one keep whatever defaults their reader
        ///       carries, so this is never fatal.
        void _load_processor_json(){
            const std::string path = this->model_path + "/processor_config.json";
            std::ifstream file(path);
            if (!file.is_open()){
                return;
            }
            std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            try {
                this->_json_config["processor_config"] = nlohmann::json::parse(json_str);
            } catch (const nlohmann::json::exception& e){
                std::cerr << "Failed to parse " << path << ": " << e.what() << std::endl;
            }
        }

        /// \brief turn the relative weight names into full paths and publish the
        ///        derived is_vlm / is_audio flags, so consumers can read them
        ///        from json like every other parameter.
        void _normalize_multi_modal(){
            std::string vision_model_weight;
            std::string audio_model_weight;
            JSON_GET(vision_model_weight, this->_json_config, "vision_model_weight", "", std::string);
            JSON_GET(audio_model_weight, this->_json_config, "audio_model_weight", "", std::string);

            this->_json_config["is_vlm"] = !vision_model_weight.empty();
            this->_json_config["is_audio"] = !audio_model_weight.empty();
            if (!vision_model_weight.empty()){
                this->_json_config["vision_model_weight"] = this->model_path + "/" + vision_model_weight;
            }
            if (!audio_model_weight.empty()){
                this->_json_config["audio_model_weight"] = this->model_path + "/" + audio_model_weight;
            }
        }

        /// \brief shared pretty printer, reads everything from the normalized json
        std::string _str_from(const nlohmann::json& jc){
            u32 head_dim, hidden_size, intermediate_size;
            u32 num_attention_heads, num_hidden_layers, num_key_value_heads;
            u32 pretraining_tp, sliding_window, sliding_window_pattern;
            f32 rms_norm_eps;
            std::string hidden_act, vision_model_weight;
            bool is_vlm;
            JSON_GET(head_dim, jc, "head_dim", 0, u32);
            JSON_GET(hidden_size, jc, "hidden_size", 0, u32);
            JSON_GET(hidden_act, jc, "hidden_act", "", std::string);
            JSON_GET(intermediate_size, jc, "intermediate_size", 0, u32);
            JSON_GET(num_attention_heads, jc, "num_attention_heads", 0, u32);
            JSON_GET(num_hidden_layers, jc, "num_hidden_layers", 0, u32);
            JSON_GET(num_key_value_heads, jc, "num_key_value_heads", 0, u32);
            JSON_GET(pretraining_tp, jc, "pretraining_tp", 0, u32);
            JSON_GET(rms_norm_eps, jc, "rms_norm_eps", 0.0, f32);
            JSON_GET(sliding_window, jc, "sliding_window", 0, u32);
            JSON_GET(sliding_window_pattern, jc, "sliding_window_pattern", 0, u32);
            JSON_GET(is_vlm, jc, "is_vlm", false, bool);
            JSON_GET(vision_model_weight, jc, "vision_model_weight", "", std::string);

            std::stringstream ss;
            ss << "  Model: "  << std::endl;
            ss << "    model_name:             " << this->model_name << std::endl;
            ss << "    compatible_flm_version: >= " << this->flm_version << std::endl;
            ss << "    head_dim:               " << head_dim << std::endl;
            ss << "    hidden_size:            " << hidden_size << std::endl;
            if (hidden_act != ""){
                ss << "    hidden_act:             " << hidden_act << std::endl;
            }
            ss << "    intermediate_size:      " << intermediate_size << std::endl;
            ss << "    num_attention_heads:    " << num_attention_heads << std::endl;
            ss << "    num_hidden_layers:      " << num_hidden_layers << std::endl;
            ss << "    num_key_value_heads:    " << num_key_value_heads << std::endl;
            ss << "    pretraining_tp:         " << pretraining_tp << std::endl;
            ss << "    rms_norm_eps:           " << rms_norm_eps << std::endl;
            if (sliding_window > 0){
                ss << "    sliding_window:         " << sliding_window << std::endl;
                ss << "    sliding_window_pattern: " << sliding_window_pattern << std::endl;
            }

            if (is_vlm){
                ss << "  Vision: "  << std::endl;
                ss << "    vision_model_weight:    " << vision_model_weight << std::endl;
            }
            return ss.str();
        }
};

class Whisper_Config : public LM_Config{
public:
    /// \brief from pretrained
    /// \param model_name the model name
    void from_pretrained(std::string model_name){
        this->_resolve_paths(model_name);
        this->_load_json();

        // whisper spells the common dimensions differently; republish them under
        // the canonical keys so downstream readers stay model agnostic.
        u32 hidden_size, intermediate_size, vocab_size;
        JSON_GET(hidden_size, this->_json_config, "d_model", 0, u32);
        JSON_GET(intermediate_size, this->_json_config, "decoder_ffn_dim", 0, u32);
        JSON_GET(vocab_size, this->_json_config, "vocab_size", 0, u32);
        this->_json_config["hidden_size"] = hidden_size;
        this->_json_config["intermediate_size"] = intermediate_size;
        this->_json_config["vocab_size"] = (vocab_size + 31) / 32 * 32;

        this->_json_config["is_vlm"] = false;
        this->_json_config["is_audio"] = false;

        JSON_GET(this->flm_version, this->_json_config, "flm_version", "0.0.0", std::string);
    }
    std::string _str(){
        return this->_str_from(this->_json_config);
    }
    Whisper_Config(){}
};
