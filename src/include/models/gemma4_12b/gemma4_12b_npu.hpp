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
    // parameters for vision preprocessing in Gemma4e
    unsigned int GEMMA4_12B_vision_pooling_kernel_size;
    unsigned int GEMMA4_12B_vision_patch_size;
    float GEMMA4_12B_vision_rescale_factor;
    float GEMMA4_12B_vision_image_mean;
    float GEMMA4_12B_vision_image_std;
    // parameters for audio preprocessing in Gemma4e
    unsigned int GEMMA4_12B_audio_embed_dim;
    unsigned int GEMMA4_12B_audio_sampling_rate;
    inline void load_vision_preprocess_parameters(LM_Config& config){
        // Note: this should be called by Impl:: constructor
        const nlohmann::json& vc = config.sub("vision_config");
        GEMMA4_12B_vision_pooling_kernel_size = vc.value("pooling_kernel_size", 0);
        GEMMA4_12B_vision_patch_size = vc.value("patch_size", 0);
        // GEMMA4_12B_VISION_MAX_POSITION_EMBEDDINGS = vc.value("GEMMA4_12B_VISION_MAX_POSITION_EMBEDDINGS", -1);
        // // GEMMA4_12B_VISION_PATCH_SIZE          = vc.value("GEMMA4_12B_VISION_PATCH_SIZE", -1);
        // GEMMA4_12B_POSITION_EMBEDDING_SIZE    = vc.value("GEMMA4_12B_POSITION_EMBEDDING_SIZE", -1);
        // GEMMA4_12B_VISION_IMAGE_OUTPUT_SIZE   = vc.value("GEMMA4_12B_VISION_IMAGE_OUTPUT_SIZE", -1);
        GEMMA4_12B_vision_rescale_factor      = vc.value("rescale_factor", -1.0f);
        GEMMA4_12B_vision_image_mean          = vc.value("image_mean", -1.0f);
        GEMMA4_12B_vision_image_std           = vc.value("image_std", -1.0f);
    }

    inline void load_audio_preprocess_parameters(LM_Config& config){
        const nlohmann::json& ac = config.sub("audio_config");
        GEMMA4_12B_audio_embed_dim          = ac.value("audio_embed_dim", 0);
        GEMMA4_12B_audio_sampling_rate      = ac.value("sampling_rate", 0);
    }

private:
    struct Impl;
    Impl* _impl;
};

