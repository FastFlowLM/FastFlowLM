// OpenFFLM -- RMSNorm in place over a combined [x | weight] buffer.
// SPDX-License-Identifier: MIT
//
// ONE pointer, because a compute tile has only 2 input DMA channels and the
// fused norm+GEMV design spends one of them on weights. The activation and the
// norm weight therefore share a single fifo element: x in the first `cols`,
// the weight in the second. The result is written back over x, which costs no
// extra L1 -- and that is what lets the weight double buffer keep per_call 5.
//
// See granite_rmsnorm.h for why neither pointer is __restrict.
#include "granite_rmsnorm.h"

extern "C" {
void granite_rms_norm_ip(bfloat16 *xw, unsigned cols) {
  granite_rms_norm_impl(xw, xw + cols, xw, cols);
}
}
