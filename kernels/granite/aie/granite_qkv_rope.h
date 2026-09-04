#pragma once
//===- granite_qkv_rope.h -----------------------------------*- C++ -*-===//
//
// OpenFFLM -- epilogue for a fused q/k/v projection: rotate q and k, pass v.
// SPDX-License-Identifier: MIT
//
// q, k and v all consume the same input x, so all three GEMVs share one
// dispatch with no gather. RoPE then applies to q and k but NOT to v, and a
// single core owns whole heads of each, so the whole epilogue is core-local.
//
// The accumulator holds, in order:  q_heads*64 | k_heads*64 | v_len
// and the output has the same shape. Only the layout differs from
// granite_qrope.h, which is why this is a separate kernel rather than three
// calls with pointer offsets -- IRON kernels take whole buffers.
//
// cos/sin ride at the end of the x buffer (GRANITE_QROPE_XOFF): a compute tile
// has 2 input DMA channels, the weights take one and x the other, so a third
// stream for cos/sin does not exist.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "granite_elementwise.h"

#ifndef GRANITE_QROPE_XOFF
#define GRANITE_QROPE_XOFF 2560
#endif

// Rotate `n_heads` consecutive heads starting at `y` into `out`.
static inline void rope_heads(const float *__restrict y,
                              const aie::vector<bfloat16, kHalf> &c,
                              const aie::vector<bfloat16, kHalf> &s,
                              bfloat16 *__restrict out, unsigned n_heads) {
  for (unsigned h = 0; h < n_heads; ++h) {
    const float *__restrict yh = y + h * kHeadDim;
    bfloat16 *__restrict oh = out + h * kHeadDim;
    aie::accum<accfloat, kHalf> alo, ahi;
    alo.from_vector(aie::load_v<kHalf>(yh));
    ahi.from_vector(aie::load_v<kHalf>(yh + kHalf));
    aie::vector<bfloat16, kHalf> lo = alo.template to_vector<bfloat16>();
    aie::vector<bfloat16, kHalf> hi = ahi.template to_vector<bfloat16>();
    aie::store_v(oh, aie::sub(aie::mul(lo, c), aie::mul(hi, s))
                         .template to_vector<bfloat16>());
    aie::store_v(oh + kHalf, aie::add(aie::mul(hi, c), aie::mul(lo, s))
                                 .template to_vector<bfloat16>());
  }
}

// `acc` and `out` are NOT __restrict: granite_rope_ip.cc calls this with the
// output aliasing the accumulator, so the result narrows into the buffer it was
// read from. That is not a micro-optimisation -- it is 512 B of L1, and with a
// 62208 B budget it is the difference between norm+qkv+RoPE fitting in one
// dispatch at per_call 5 and not fitting at all.
//
// Safe because rope_heads loads BOTH halves of a head before storing either,
// and consecutive heads do not overlap: head h reads 64 floats at acc + 64h and
// writes 64 bfloat16 at the same base, i.e. the first half of what it just
// read. Verified by the design's own check against a host reference, not by
// inspection alone.
__attribute__((noinline)) inline void
granite_qkv_rope_impl(const float *acc, const bfloat16 *__restrict xcs,
                      bfloat16 *out,
                      unsigned q_heads, unsigned k_heads, unsigned v_len) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *__restrict cs = xcs + GRANITE_QROPE_XOFF;
  const aie::vector<bfloat16, kHalf> c = aie::load_v<kHalf>(cs);
  const aie::vector<bfloat16, kHalf> s = aie::load_v<kHalf>(cs + kHalf);

  rope_heads(acc, c, s, out, q_heads);
  const unsigned qn = q_heads * kHeadDim;
  rope_heads(acc + qn, c, s, out + qn, k_heads);

  // v is not rotated -- only narrowed.
  const unsigned kn = qn + k_heads * kHeadDim;
  for (unsigned i = 0; i < v_len; i += kHalf) {
    aie::accum<accfloat, kHalf> a;
    a.from_vector(aie::load_v<kHalf>(acc + kn + i));
    aie::store_v(out + kn + i, a.template to_vector<bfloat16>());
  }
  event1();
}

extern "C" {
void granite_qkv_rope(const float *__restrict acc,
                      const bfloat16 *__restrict xcs, bfloat16 *__restrict out,
                      unsigned q_heads, unsigned k_heads, unsigned v_len) {
  granite_qkv_rope_impl(acc, xcs, out, q_heads, k_heads, v_len);
}
}
