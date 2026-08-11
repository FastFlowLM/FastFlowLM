#!/usr/bin/bash

# set up model loading path
export FLM_MODEL_PATH="/scratch/alfxu"

# copy lib
# cp /scratch/alfxu/FastFlowLM_IRON/FLM_DLL/build/lib/libqwen3_6_moe_npu.so ~/public_repo/FastFlowLM/src/lib/

# copy xclbins
cp /scratch/alfxu/FastFlowLM_IRON/FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/build/QWEN3_6_35B_A3B/xclbins/layer.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/
# cp /scratch/alfxu/FastFlowLM/FLM_Xclbin/Dequant_mix_w64x512_chunk_reorder/build/xclbins/dequant.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/
# cp /scratch/alfxu/FastFlowLM/FLM_Xclbin/Qwen3_6/MM/build/xclbins/mm.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/
# cp /scratch/alfxu/FastFlowLM/FLM_Xclbin/Qwen3_6/gated_attention_prefill/build/xclbins/attn.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/
# cp /scratch/alfxu/FastFlowLM/FLM_Xclbin/Qwen3_6/gate_delta_net_prefill/build/xclbins/GateDeltaNet_prefill.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/
# cp /scratch/alfxu/FastFlowLM/FLM_Xclbin/Qwen3_5/lm_head_npu_bin/build/QWEN3_5_2B/xclbins/lm_head.xclbin ../../xclbins/Qwen3.6-35B-A3B-NPU2/

