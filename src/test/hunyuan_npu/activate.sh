#!/usr/bin/bash

# Weights live at $FLM_MODEL_PATH/models/Hy-MT2-1.8B-NPU2
export FLM_MODEL_PATH="/scratch/$USER"

FLM_DLL_BUILD="/scratch/$USER/FastFlowLM_IRON/FLM_DLL/build"

# copy src to dst dir only if the contents differ
copy_if_different() {
    local src="$1" dst="$2/$(basename "$1")"
    if cmp -s "$src" "$dst"; then
        echo "unchanged: $dst"
    else
        cp "$src" "$dst" && echo "updated:   $dst"
    fi
}

# The engine .so, plus the three shared modules it was linked against
# (build_hunyuan.sh rebuilds q4nx / gemm / lm_head alongside hunyuan_npu, and a
# stale copy of any of them here silently changes what the engine reads).
copy_if_different $FLM_DLL_BUILD/hunyuan_npu/lib/libhunyuan_npu.so ../../lib/xrt
copy_if_different $FLM_DLL_BUILD/lib/libq4_npu_eXpress.so          ../../lib/xrt
copy_if_different $FLM_DLL_BUILD/lib/libgemm.so                    ../../lib/xrt
copy_if_different $FLM_DLL_BUILD/lib/liblm_head.so                 ../../lib/xrt

# No xclbins: the hunyuan engine's four backend switches all default to 0, so
# the whole graph runs on the host and no xclbin is ever registered.
