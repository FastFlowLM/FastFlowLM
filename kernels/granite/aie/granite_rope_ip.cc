// OpenFFLM -- RoPE epilogue writing back into the accumulator it reads.
// SPDX-License-Identifier: MIT
//
// The fused norm+qkv+RoPE design has 128 B of L1 to spare, so the output cannot
// have a buffer of its own: the float32 accumulator IS the output fifo element,
// and the rotated result narrows into its first half. See the aliasing note in
// granite_qkv_rope.h.
//
// cos|sin sit after x and the norm weight, hence XOFF = 2 * K.
#define GRANITE_QROPE_XOFF 5120
#include "granite_qkv_rope.h"

extern "C" {
void granite_rope_ip(float *acc, const bfloat16 *__restrict xcs,
                     unsigned q_heads, unsigned k_heads, unsigned v_len) {
  granite_qkv_rope_impl(acc, xcs, reinterpret_cast<bfloat16 *>(acc),
                        q_heads, k_heads, v_len);
}
}
