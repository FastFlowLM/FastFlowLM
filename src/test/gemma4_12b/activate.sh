#!/usr/bin/bash

# set up model loading path
export FLM_MODEL_PATH="/scratch/$USER"

# copy src to dst dir only if the contents differ
copy_if_different() {
    local src="$1" dst="$2/$(basename "$1")"
    if cmp -s "$src" "$dst"; then
        echo "unchanged: $dst"
    else
        cp "$src" "$dst" && echo "updated:   $dst"
    fi
}

# copy lib
copy_if_different /scratch/$USER/Projects/FastFlowLM_IRON/FLM_DLL/build/lib/libgemma4_12b_npu.so ../../lib/xrt

# copy xclbins
copy_if_different /scratch/$USER/Projects/FastFlowLM_IRON/FLM_Xclbin/Gemma4_12B_QAT/decoding/build/GEMMA4_12B/xclbins/layer.xclbin ../../xclbins/Gemma4-12B-IT-NPU2
copy_if_different /scratch/$USER/Projects/FastFlowLM_IRON/FLM_Xclbin/Gemma4_12B_QAT/lm_head_npu_bin/build/GEMMA4_12B/xclbins/lm_head.xclbin ../../xclbins/Gemma4-12B-IT-NPU2
copy_if_different /scratch/$USER/Projects/FastFlowLM_IRON/FLM_Xclbin/Gemma4_12B_QAT/dequant_mm/build/xclbins/dequant_mm.xclbin ../../xclbins/Gemma4-12B-IT-NPU2
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Gemma4_12B_QAT/attention_DH_512_prefill/build/xclbins/attn_global.xclbin ../../xclbins/Gemma4-12B-IT-NPU2/
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Gemma4_12B_QAT/attention_DH_256_prefill/build/xclbins/attn_sliding.xclbin ../../xclbins/Gemma4-12B-IT-NPU2/
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Gemma4_12B_QAT/attention_global_image/build/xclbins/attn_global_image.xclbin ../../xclbins/Gemma4-12B-IT-NPU2/
copy_if_different /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Gemma4_12B_QAT/audio_image_mm/build/xclbins/audio_image_mm.xclbin ../../xclbins/Gemma4-12B-IT-NPU2/