#pragma once
//===- granite_rmsnorm.h ------------------------------------*- C++ -*-===//
//
// OpenFFLM -- Llama/Granite RMSNorm on the AIE core.
// SPDX-License-Identifier: MIT
//
//     y[c] = x[c] * rsqrt(mean(x^2) + eps) * w[c]
//
// WHY NOT aie_kernels/aie2p/rms_norm.cc
// -------------------------------------
// That kernel hardcodes `const float gamma = 1.0f` and never applies the
// per-channel weight tensor, and its `epsilon` is a `constexpr`. It normalises
// correctly and then throws away the learned scale -- output of the right
// magnitude and the wrong value, which no shape check catches. `cols` IS a
// runtime argument there, so only the weight was ever the problem.
//
// PRECISION, AND THE TRAP IT AVOIDS
// ---------------------------------
// AIE2P has no fp32 vector multiplier: `aie::mul(vector<float>, vector<float>)`
// compiles and returns **zero**, silently. So `x * w * inv` cannot be done in
// fp32 vectors. Instead:
//
//   * x*w is a bf16 x bf16 product, which lands EXACTLY in an fp32 accumulator;
//   * that fp32 partial is split into two bf16 halves (8 + 8 mantissa bits);
//   * the scalar `inv` is split the same way;
//   * three of the four cross terms are accumulated (hi*hi, lo*hi, hi*lo),
//     giving ~2^-17 relative -- far below the bf16 output's own 2^-9, so the
//     stored result is correctly rounded.
//
// Summing x^2 through an fp32 accumulator rather than a bf16 running sum is the
// other half of it: 2560 bf16 additions would lose ~6 bits.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GRANITE_NORM_EPS
#define GRANITE_NORM_EPS 1e-5f
#endif

static constexpr unsigned kNormVec = 32;

// cols is a RUNTIME argument, so one build serves hidden 2560 and any other
// width the model happens to use.
// `x` and `y` are deliberately NOT __restrict: the fused prologue in
// granite_norm_gemv.py calls this with y == x, normalising in place inside the
// activation fifo's own element. That is what keeps the fused design's fixed L1
// at 10240 B instead of 15360, which is in turn what lets the weight double
// buffer keep per_call = 5 -- and per_call is worth more than the aliasing
// optimisation this gives up. The two passes read x fully before writing y, so
// in-place is correct; it was only ever the promise that was wrong.
// `w` is not __restrict either, for a reason that only shows up on hardware: a
// compute tile has 2 input DMA channels, and the fused design already spends
// one on weights. So x and the norm weight cannot arrive as separate streams --
// they share one fifo element, w sitting at x + cols. Disjoint ranges of one
// object do not alias in practice, but declaring otherwise would be another
// promise this code cannot keep.
__attribute__((noinline)) inline void
granite_rms_norm_impl(const bfloat16 *x, const bfloat16 *w,
                      bfloat16 *y, unsigned cols) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  // sum of x^2, accumulated in fp32.
  aie::accum<accfloat, kNormVec> sq = aie::zeros<accfloat, kNormVec>();
  for (unsigned i = 0; i < cols; i += kNormVec) {
    aie::vector<bfloat16, kNormVec> v = aie::load_v<kNormVec>(x + i);
    sq = aie::mac(sq, v, v);
  }
  const float sum_sq = aie::reduce_add(sq.template to_vector<float>());

  const float inv = aie::invsqrt(sum_sq / (float)cols + GRANITE_NORM_EPS);
  const bfloat16 inv_hi = (bfloat16)inv;
  const bfloat16 inv_lo = (bfloat16)(inv - (float)inv_hi);

  for (unsigned i = 0; i < cols; i += kNormVec) {
    aie::vector<bfloat16, kNormVec> xv = aie::load_v<kNormVec>(x + i);
    aie::vector<bfloat16, kNormVec> wv = aie::load_v<kNormVec>(w + i);

    // x*w exactly, in fp32.
    aie::accum<accfloat, kNormVec> xw = aie::mul(xv, wv);
    aie::vector<bfloat16, kNormVec> hi = xw.template to_vector<bfloat16>();
    aie::vector<bfloat16, kNormVec> lo =
        aie::sub(xw, hi).template to_vector<bfloat16>();

    aie::accum<accfloat, kNormVec> out = aie::zeros<accfloat, kNormVec>();
    out = aie::mac(out, hi, inv_hi);
    out = aie::mac(out, lo, inv_hi);
    out = aie::mac(out, hi, inv_lo);
    aie::store_v(y + i, out.template to_vector<bfloat16>());
  }
  event1();
}

extern "C" {
void granite_rms_norm(const bfloat16 *__restrict x, const bfloat16 *__restrict w,
                      bfloat16 *__restrict y, unsigned cols) {
  granite_rms_norm_impl(x, w, y, cols);
}
}
