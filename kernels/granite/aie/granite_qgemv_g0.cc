// OpenFFLM -- q_proj GEMV group 0, writing at a row offset so a core can
// accumulate its whole slice before the RoPE epilogue runs on it.
// One entry point per translation unit: IRON compiles the source once per
// ExternalFunction, so two here would each define both symbols.
// SPDX-License-Identifier: MIT
#define GRANITE_TILES_PER_CALL 5
#define GRANITE_BATCH 1
#define GRANITE_K 2560
#include "granite_gemv.h"

extern "C" {
void granite_qgemv_g0(const uint8_t *__restrict t, const bfloat16 *__restrict x,
                       float *__restrict y, unsigned row) {
  // `row` selects the 32-float window inside the core's slice; without it every
  // call would write y[0..31] and only the last tile-row would survive.
  gemv_q4_group(t, x, 0, y + row * kRows);
}
}
