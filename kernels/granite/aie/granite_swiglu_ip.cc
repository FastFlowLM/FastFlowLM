// OpenFFLM -- SwiGLU in place over a combined [gate | up] buffer.
// SPDX-License-Identifier: MIT
//
// ONE pointer, because the fused SwiGLU+down_proj design has two input DMA
// channels per core and both are spoken for -- weights and activation. gate and
// up therefore share the activation element, and the result is written back
// over gate so no destination buffer is needed. See granite_elementwise.h.
#include "granite_elementwise.h"

extern "C" {
void granite_swiglu_ip(bfloat16 *gu, unsigned n) {
  granite_swiglu_impl(gu, gu + n, gu, n);
}
}
