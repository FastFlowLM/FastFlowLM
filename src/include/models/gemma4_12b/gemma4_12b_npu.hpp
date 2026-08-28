/// \file qwen3vl_npu.hpp
/// \brief qwen3vl_npu class
/// \author FastFlowLM Team
/// \date 2026-01-23
/// \version 0.9.28
/// \note This is a header file for the qwen3vl_npu class
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "modules/embedding.hpp"
#include "modules/lm_head.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"
#if USEAVX2
#include <immintrin.h>  // For AVX intrinsics
#endif

// some helper functions for convenience
constexpr int GEMMA4_12B_IS_GLOBAL_MASK = 0x00000001;

typedef enum :int {
    gemma4_12b_swa_layer = 0,
    gemma4_12b_global_layer = GEMMA4_12B_IS_GLOBAL_MASK,
    gemma4_12b_total_layer_types = 2
} gemma4_12b_layer_type_t;

inline bool is_swa_layer(gemma4_12b_layer_type_t layer) {
    return (layer & GEMMA4_12B_IS_GLOBAL_MASK) == 0;
}

inline bool is_global_layer(gemma4_12b_layer_type_t layer) {
    return (layer & GEMMA4_12B_IS_GLOBAL_MASK) != 0;
}

typedef struct {
    int height;
    int width;
    int height_resized;  // assigned by image preprocessing
    int width_resized;
    bytes _data;
} gemma4_12b_image_t;

typedef struct {
    std::vector<std::pair<int, int>> image_patch__element_per_patch; // [num_of_image][width, height]
    std::vector<uint32_t> valid_patch_size_per_image; // [num_of_image], the unpadded size per image
    std::vector<std::vector<bf16>> pixel_values; // [num_of_image][image_size], where image_size = height_resized * width_resized * 3
    std::vector< std::vector<int>> image_grid_pairs_per_image; // [num_of_image][num_of_position_id][x, y]
    std::vector<unsigned int> num_soft_tokens_per_image; // [num_of_image]
    unsigned int num_images;
}gemma4_12b_image_payload_t;

struct gemma4_12b_audio_payload_t {
    std::vector<std::vector<bf16>> mel_spectrograms;               // [num_audios][frames * bins], row-major
    std::vector<int> mel_spectrogram_frames_per_audio;             // [num_audios]
    std::vector<int> mel_spectrogram_bins_per_audio;               // [num_audios]
    unsigned int num_audios = 0;
    std::vector<unsigned int> num_soft_tokens_per_audio; // [num_audios]
};

typedef struct {
    gemma4_12b_image_payload_t image_payload;
    gemma4_12b_audio_payload_t audio_payload;
} gemma4_12b_multi_modal_payload_t;

class gemma4_12b_npu : public causal_lm{
public:
    /// \brief  initialize the qwen3vl_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    gemma4_12b_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~gemma4_12b_npu();

    /// \brief forward the qwen3vl_npu
    /// \param ids the ids
    /// \return the output tensor
    buffer<bf16> forward(int ids) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief update the max length
    void clear_context() override;

    /// \brief get the k cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the k cache
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the v cache
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief update the max length
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;
    // ---- preprocessing parameters -------------------------------------------
    // These describe the image/audio front end, not the network, so they come from
    // the checkpoint's processor_config.json (image_processor / feature_extractor)
    // rather than config.json. config.json is still honoured as a fallback for
    // checkpoints packaged before processor_config.json was shipped, and the
    // literals below are the released Gemma4-12B values so a checkpoint carrying
    // neither still preprocesses correctly.
    unsigned int GEMMA4_12B_vision_pooling_kernel_size;
    unsigned int GEMMA4_12B_vision_patch_size;
    unsigned int GEMMA4_12B_vision_max_soft_tokens;
    float GEMMA4_12B_vision_rescale_factor;
    float GEMMA4_12B_vision_image_mean;
    float GEMMA4_12B_vision_image_std;
    // parameters for audio preprocessing
    unsigned int GEMMA4_12B_audio_embed_dim;
    unsigned int GEMMA4_12B_audio_samples_per_token;
    unsigned int GEMMA4_12B_audio_max_soft_tokens;
    unsigned int GEMMA4_12B_audio_sampling_rate;

    /// \brief the "image_processor" / "video_processor" / "feature_extractor" block
    ///        of processor_config.json, or an empty object when absent
    inline const nlohmann::json& _processor_sub(LM_Config& config, const char* key){
        return cfg_sub(cfg_sub(config._json_config, "processor_config"), key);
    }

    /// \brief first of `processor[key]`, `fallback[key]`, `default_value` that exists
    template <typename T>
    inline T _preprocess_value(const nlohmann::json& processor, const nlohmann::json& fallback,
                              const char* key, T default_value){
        return cfg_get<T>(processor, key, cfg_get<T>(fallback, key, default_value));
    }

    /// \brief same, for the per-channel means/stds, which the processor config
    ///        writes as a 3-element array and config.json as a scalar
    /// \note  Gemma4 leaves normalization off (mean 0, std 1) and the host kernel
    ///        takes one scalar per statistic, so the channels are required to agree.
    inline float _preprocess_channel_value(const nlohmann::json& processor, const nlohmann::json& fallback,
                                           const char* key, float default_value){
        for (const nlohmann::json* jc : {&processor, &fallback}){
            if (!jc->contains(key) || (*jc)[key].is_null()){
                continue;
            }
            const nlohmann::json& v = (*jc)[key];
            if (!v.is_array()){
                return float(v);
            }
            if (v.empty()){
                continue;
            }
            const float first = float(v[0]);
            for (const auto& c : v){
                if (float(c) != first){
                    header_print("warning", std::string("processor config ") + key
                                 + " is not uniform across channels, using " + std::to_string(first));
                    break;
                }
            }
            return first;
        }
        return default_value;
    }

    inline void load_vision_preprocess_parameters(LM_Config& config){
        // Note: this should be called by Impl:: constructor
        const nlohmann::json& pc = this->_processor_sub(config, "image_processor");
        const nlohmann::json& vc = config.sub("vision_config");
        GEMMA4_12B_vision_pooling_kernel_size = _preprocess_value<unsigned int>(pc, vc, "pooling_kernel_size", 3);
        GEMMA4_12B_vision_patch_size          = _preprocess_value<unsigned int>(pc, vc, "patch_size", 16);
        GEMMA4_12B_vision_max_soft_tokens     = _preprocess_value<unsigned int>(pc, vc, "max_soft_tokens", 280);
        // mean/std make the normalize step the identity, which is why do_normalize is off.
        GEMMA4_12B_vision_rescale_factor      = _preprocess_value<float>(pc, vc, "rescale_factor", 1.0f / 255.0f);
        GEMMA4_12B_vision_image_mean          = _preprocess_channel_value(pc, vc, "image_mean", 0.0f);
        GEMMA4_12B_vision_image_std           = _preprocess_channel_value(pc, vc, "image_std", 1.0f);
    }

    inline void load_audio_preprocess_parameters(LM_Config& config){
        const nlohmann::json& pc = this->_processor_sub(config, "feature_extractor");
        const nlohmann::json& ac = config.sub("audio_config");
        // One soft token is 640 samples at 16 kHz = 40 ms, and the model takes 750
        // of them (30 s). feature_size is the width of one row handed to the audio
        // embedder, which for this encoder free model is the raw sample count.
        GEMMA4_12B_audio_embed_dim          = _preprocess_value<unsigned int>(pc, ac, "feature_size", 640);
        GEMMA4_12B_audio_samples_per_token  = _preprocess_value<unsigned int>(pc, ac, "audio_samples_per_token", 640);
        GEMMA4_12B_audio_sampling_rate      = _preprocess_value<unsigned int>(pc, ac, "sampling_rate", 16000);
        // audio_seq_length sits at the top level of processor_config.json, next to
        // image_seq_length, rather than inside the feature extractor block.
        GEMMA4_12B_audio_max_soft_tokens    = cfg_get<unsigned int>(
            cfg_sub(config._json_config, "processor_config"), "audio_seq_length",
            cfg_get<unsigned int>(ac, "audio_seq_length", 750));
    }

private:
    struct Impl;
    Impl* _impl;
};

