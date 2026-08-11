/// \file qwen3_5_omni.hpp
/// \brief qwen3_5_omni class
/// \author FastFlowLM Team
/// \date 2026-01-23
/// \version 0.9.28
/// \note This is a header file for the qwen3_5_omni class
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
#if USEAVX2
#include <immintrin.h>  // For AVX intrinsics
#endif


typedef struct {
    int height;
    int width;
    int height_resized;  // assigned by image preprocessing
    int width_resized;
    bytes _data;

} qwen3_5_omni_image_t;

typedef struct {
    std::vector<bf16> _processed_pixel_values; // [num_of_image][1d array of processed image values]
    std::vector< std::vector<int>> image_grid_h_w;  //[num_of_images][grid_h, grid_w]
    std::vector<unsigned int> num_soft_tokens_per_image; // [num_of_image]
    unsigned int num_images;
}qwen3_5_omni_image_payload_t;


struct qwen3_5_omni_audio_payload_t {
    // per-audio mel spectrogram data
    std::vector<std::vector<bf16>> mel_spectrograms;               // [num_audios][frames * bins], row-major
    std::vector<int> mel_spectrogram_frames_per_audio;             // [num_audios]
    std::vector<int> mel_spectrogram_bins_per_audio;               // [num_audios]
    unsigned int num_audios = 0;


    std::vector<std::string> audio_tokens; // [num_audios, string of tokens ]
};


typedef struct {
    qwen3_5_omni_image_payload_t image_payload;
    qwen3_5_omni_audio_payload_t audio_payload;
} qwen3_5_omni_payload_t;

typedef struct {
    buffer<bf16> logits;
    buffer<bf16> hidden_states;
} qwen3_5_omni_thinker_result_t;

class qwen3_5_omni{
public:
    /// \brief  initialize the qwen3_5_omni
    /// \param config the configuration
    /// \param npu_instance the npu instance
    qwen3_5_omni(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen3_5_omni();

    /// \brief forward the qwen3_5_omni
    /// \param ids the ids
    /// \return the output tensor
    qwen3_5_omni_thinker_result_t forward(int ids);
    buffer<bf16> say(qwen3_5_omni_thinker_result_t thinker_res);
    qwen3_5_omni_thinker_result_t prefill(std::vector<int>& ids, void* payload = nullptr);

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L);

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx);

    /// \brief update the max length
    void clear_context();

    /// \brief update the max length
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L);

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length();
    int checkpoint();
    int restore();
private:
    struct Impl;
    Impl* _impl;
};
