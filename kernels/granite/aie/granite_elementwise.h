#pragma once
//===- granite_elementwise.h --------------------------------*- C++ -*-===//
//
// OpenFFLM -- granite's RoPE and SwiGLU on the AIE core.
// SPDX-License-Identifier: MIT
//
// WHY NOT THE aie_kernels/aie2p REFERENCES
// ----------------------------------------
// Both are demos that fail open, in the same way `rms_norm.cc` does (see
// granite_rmsnorm.h):
//
//  * **`rope.cc` uses the interleaved-pair convention.** It pairs element 0 with
//    1, 2 with 3 (`filter_even`/`filter_odd`, GPT-NeoX style). Llama and Granite
//    use **half-split** `rotate_half`, pairing i with i + head_dim/2. The two
//    are different rotations that produce identical magnitudes, so the output
//    looks entirely reasonable and is wrong.
//
//  * **`swiglu.cc`'s entry point hardcodes `input_size = 1024`.** The templated
//    body takes a size, but the `extern "C"` wrapper passes a literal. Granite's
//    intermediate is 8192, so it would compute one eighth of the vector and
//    leave the remaining seven eighths as whatever was already in the buffer --
//    partially correct output, no error, no shape mismatch.
//
// The silu maths is worth keeping though: `sigmoid(x) = (tanh(x/2) + 1) / 2` is
// an identity, not an approximation (tanh(x/2) = 2*sigmoid(x) - 1). Only
// `aie::tanh`'s own implementation is approximate.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// granite-4.2-3B is head_dim 64, so each half is one 32-lane vector. Both halves
// of a head are then a single load, a single multiply and a single store.
#ifndef GRANITE_HEAD_HALF
#define GRANITE_HEAD_HALF 32
#endif
static constexpr unsigned kHalf = GRANITE_HEAD_HALF;
static constexpr unsigned kHeadDim = 2 * kHalf;

// RoPE, HALF-SPLIT (Llama/Granite `rotate_half`):
//
//     y[i]        = x[i]        * cos[i] - x[i + half] * sin[i]
//     y[i + half] = x[i + half] * cos[i] + x[i]        * sin[i]
//
// `cs` holds cos[0..half) then sin[0..half) for this position -- the caller
// owns the position, so this kernel is stateless and the same code serves
// prefill and decode.
__attribute__((noinline)) inline void
granite_rope_impl(const bfloat16 *__restrict x, const bfloat16 *__restrict cs,
                  bfloat16 *__restrict y, unsigned n_heads) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const aie::vector<bfloat16, kHalf> c = aie::load_v<kHalf>(cs);
  const aie::vector<bfloat16, kHalf> s = aie::load_v<kHalf>(cs + kHalf);

  for (unsigned h = 0; h < n_heads; ++h) {
    const bfloat16 *__restrict xh = x + h * kHeadDim;
    bfloat16 *__restrict yh = y + h * kHeadDim;

    aie::vector<bfloat16, kHalf> lo = aie::load_v<kHalf>(xh);
    aie::vector<bfloat16, kHalf> hi = aie::load_v<kHalf>(xh + kHalf);

    // Each product is bf16 x bf16 into an fp32 accumulator, so the rotation
    // carries full precision and only the final store rounds.
    aie::accum<accfloat, kHalf> lo_c = aie::mul(lo, c);
    aie::accum<accfloat, kHalf> hi_s = aie::mul(hi, s);
    aie::accum<accfloat, kHalf> hi_c = aie::mul(hi, c);
    aie::accum<accfloat, kHalf> lo_s = aie::mul(lo, s);

    aie::store_v(yh, aie::sub(lo_c, hi_s).template to_vector<bfloat16>());
    aie::store_v(yh + kHalf, aie::add(hi_c, lo_s).template to_vector<bfloat16>());
  }
  event1();
}

// SwiGLU: y = silu(gate) * up, with `n` a RUNTIME argument.
//
// None of the three pointers is __restrict: granite_swiglu_ip.cc calls this
// with gate, up and y all inside ONE buffer -- up at gate + n, and the result
// written back over gate. That is what lets a fused SwiGLU+down_proj keep its
// activation at 2 x 8192 bf16 with no destination buffer, which is in turn what
// keeps it inside L1. Safe because the op is elementwise at the same index and
// each vector's loads precede its store.
//
//     silu(x) = x * sigmoid(x),  sigmoid(x) = (tanh(x/2) + 1) / 2
__attribute__((noinline)) inline void
granite_swiglu_impl(const bfloat16 *gate,
                    const bfloat16 *up, bfloat16 *y,
                    unsigned n) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  constexpr unsigned V = 32;
  const aie::vector<bfloat16, V> half = aie::broadcast<bfloat16, V>((bfloat16)0.5f);
  const aie::vector<bfloat16, V> one = aie::broadcast<bfloat16, V>((bfloat16)1.0f);

  for (unsigned i = 0; i < n; i += V) {
    aie::vector<bfloat16, V> g = aie::load_v<V>(gate + i);
    aie::vector<bfloat16, V> u = aie::load_v<V>(up + i);

    // Keep x/2 as the fp32 accumulator and feed tanh from there -- rounding it
    // to bf16 first would throw away half the mantissa before the nonlinearity.
    aie::accum<accfloat, V> gh = aie::mul(g, half);
    aie::vector<bfloat16, V> t =
        aie::tanh<bfloat16>(gh.template to_vector<float>());
    aie::vector<bfloat16, V> sig =
        aie::mul(aie::add(t, one), half).template to_vector<bfloat16>();

    aie::vector<bfloat16, V> silu =
        aie::mul(g, sig).template to_vector<bfloat16>();
    aie::store_v(y + i, aie::mul(silu, u).template to_vector<bfloat16>());
  }
  event1();
}

extern "C" {
void granite_rope(const bfloat16 *__restrict x, const bfloat16 *__restrict cs,
                  bfloat16 *__restrict y, unsigned n_heads) {
  granite_rope_impl(x, cs, y, n_heads);
}
void granite_swiglu(const bfloat16 *__restrict gate,
                    const bfloat16 *__restrict up, bfloat16 *__restrict y,
                    unsigned n) {
  granite_swiglu_impl(gate, up, y, n);
}
}
