#!/usr/bin/bash

# set up model loading path
export FLM_MODEL_PATH="/scratch/michyu"

# copy lib
cp /scratch/michyu/Projects/FastFlowLM_IRON/FLM_DLL/build/lib/libgemma4_12b_npu.so ../../lib/xrt

# copy xclbins
cp /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Gemma4_12B_QAT/decoding/build/GEMMA4_12B/xclbins/layer.xclbin ../../xclbins/Gemma4-12B-IT-NPU2/
cp /scratch/michyu/Projects/FastFlowLM_IRON/FLM_Xclbin/Gemma4_12B_QAT/lm_head_npu_bin/build/GEMMA4_12B/xclbins/lm_head.xclbin ../../xclbins/Gemma4-12B-IT-NPU2/