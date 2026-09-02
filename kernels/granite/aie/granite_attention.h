#pragma once
//===- granite_attention.h ----------------------------------*- C++ -*-===//
//
// OpenFFLM -- granite GQA decode attention on the AIE core, one q head at a
// time, with an online (flash-style) softmax.
// SPDX-License-Identifier: MIT
//
//     s[t]   = scale * dot(q, K[t])          t over the whole KV cache
//     p      = softmax(s)
//     out    = sum_t p[t] * V[t]
//
// WHY ONLINE SOFTMAX AND NOT TWO PASSES
// -------------------------------------
// The KV cache is the one tensor that grows: 8 kv heads x seq x 64 x 2 x 2 B is
// 2 MB at seq 1024, per layer, per token. It cannot live in a 64 KB L1, so it
// has to stream -- and a two-pass softmax would have to stream it twice, once
// for the max and once for the weights. The online form keeps a running max `m`
// and normaliser `l` and rescales the accumulator when the max moves, so the
// cache is read exactly once.
//
//     m_new = max(m, max(s_block))
//     corr  = exp(m - m_new)
//     l     = l*corr + sum(exp(s_block - m_new))
//     acc   = acc*corr + sum_t exp(s[t] - m_new) * V[t]
//
// and `out = acc / l` once, after the last block.
//
// THE SCALE IS NOT OPTIONAL AND IS NOT GRANITE'S MULTIPLIER
// --------------------------------------------------------
// q4nx-build folds `attention_multiplier` into q_proj as
// `q_proj *= attention_multiplier * sqrt(head_dim)`, precisely so that an engine
// applying the standard `head_dim ** -0.5` gets granite's intended result. So
// this kernel MUST still apply `head_dim ** -0.5` -- the fold assumes it. For
// head_dim 64 that is exactly 0.125, a power of two, so it is exact.
//
// TRAP CARRIED OVER FROM aie_kernels/aie2p/softmax.cc
// ---------------------------------------------------
// Its own comment warns: "The multiplication by log2e is very sensitive,
// casting it to bf16 before exponentiation leads to wrong output." `bf16_exp.cc`
// does exactly that (`broadcast<bfloat16>(log2e)` rounds 1.44269504 to
// 1.4453125). Here the log2e scaling stays in fp32 and only the exp result is
// bf16.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GRANITE_ATTN_HEAD_DIM
#define GRANITE_ATTN_HEAD_DIM 64
#endif
#ifndef GRANITE_ATTN_BLOCK
#define GRANITE_ATTN_BLOCK 32
#endif

static constexpr unsigned kHD = GRANITE_ATTN_HEAD_DIM;
static constexpr unsigned kBlk = GRANITE_ATTN_BLOCK;   // KV positions per call
static constexpr unsigned kHDV = 32;                   // vector width over head_dim
static constexpr unsigned kHDVecs = kHD / kHDV;        // 2 for head_dim 64

// head_dim ** -0.5. Exact for any power-of-two head_dim.
#ifndef GRANITE_ATTN_SCALE
#define GRANITE_ATTN_SCALE 0.125f
#endif

// log2(e) is folded straight into the score scale, so every score is already in
// log2 units and the softmax is pure exp2. This is what aie_kernels' softmax.cc
// does ("the max value scaled by log2e"), and it has two payoffs beyond speed:
// there is no per-element `* log2e` needing an fp32 vector multiply (which
// AIE2P does not have), and log2e never passes through bf16 -- the rounding
// that softmax.cc's own comment warns about and that bf16_exp.cc walks into.
static constexpr float kLog2e = 1.4426950408889634f;
static constexpr float kScaleL2 = GRANITE_ATTN_SCALE * kLog2e;

// state layout, all float: acc[kHD] | m | l
//   acc  running unnormalised output
//   m    running max of the scores seen so far
//   l    running sum of exp(s - m)
//
// `kv` holds this block's K then this block's V, each [kBlk][kHD] bf16.
__attribute__((noinline)) inline void
granite_attn_block_impl(const bfloat16 *__restrict q,
                        const bfloat16 *__restrict kv,
                        float *__restrict state, unsigned n_t, unsigned first) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict K = kv;
  const bfloat16 *__restrict V = kv + (unsigned)(kBlk * kHD);

  aie::vector<bfloat16, kHDV> q0 = aie::load_v<kHDV>(q);
  aie::vector<bfloat16, kHDV> q1 = aie::load_v<kHDV>(q + kHDV);

  // 1. scores for this block. Each is a bf16 dot product accumulated in fp32.
  // Padded, because the exponentiation below runs over the whole block: a
  // stale stack value at t >= n_t would exponentiate to garbage and, if large,
  // would poison m_new for every later block.
  float s[kBlk];
  for (unsigned t = n_t; t < kBlk; ++t) s[t] = -3.0e38f;
  float m_blk = -3.0e38f;
  for (unsigned t = 0; t < n_t; ++t) {
    const bfloat16 *__restrict kt = K + t * kHD;
    aie::accum<accfloat, kHDV> a = aie::mul(q0, aie::load_v<kHDV>(kt));
    a = aie::mac(a, q1, aie::load_v<kHDV>(kt + kHDV));
    float v = aie::reduce_add(a.template to_vector<float>()) * kScaleL2;
    s[t] = v;
    if (v > m_blk) m_blk = v;
  }

  // 2. fold this block's max into the running one, and rescale what we have.
  const float m_old = first ? -3.0e38f : state[kHD];
  const float l_old = first ? 0.0f : state[kHD + 1];
  const float m_new = m_blk > m_old ? m_blk : m_old;

  // Scores are already in log2 units, so the correction is a plain exp2 of a
  // difference. aie::exp2 is a vector op with no scalar form, so this evaluates
  // one lane of a broadcast -- once per block, against n_t dot products.
  float corr = 0.0f;
  if (!first) {
    aie::vector<float, kHDV> cv = aie::broadcast<float, kHDV>(m_old - m_new);
    corr = (float)aie::exp2<bfloat16>(cv).get(0);
  }

  // 3. p[t] = exp(s[t] - m_new), and the accumulator rescaled by corr.
  float l_new = l_old * corr;
  aie::accum<accfloat, kHDV> acc[kHDVecs];
  for (unsigned v = 0; v < kHDVecs; ++v) {
    if (first) {
      acc[v] = aie::zeros<accfloat, kHDV>();
    } else {
      // acc *= corr. No fp32 vector multiplier on AIE2P, so this goes through
      // a bf16 hi/lo split of the accumulator, as everywhere else here.
      aie::vector<float, kHDV> a = aie::load_v<kHDV>(state + v * kHDV);
      aie::accum<accfloat, kHDV> t;
      t.from_vector(a);
      aie::vector<bfloat16, kHDV> hi = t.template to_vector<bfloat16>();
      aie::vector<bfloat16, kHDV> lo =
          aie::sub(t, hi).template to_vector<bfloat16>();
      const bfloat16 c_hi = (bfloat16)corr;
      const bfloat16 c_lo = (bfloat16)(corr - (float)c_hi);
      aie::accum<accfloat, kHDV> r = aie::zeros<accfloat, kHDV>();
      r = aie::mac(r, hi, c_hi);
      r = aie::mac(r, lo, c_hi);
      r = aie::mac(r, hi, c_lo);
      acc[v] = r;
    }
  }

  // Exponentiate the whole block at once. kBlk is a multiple of the vector
  // width, so this is one exp2 per kHDV scores rather than one per score.
  bfloat16 pbuf[kBlk];
  for (unsigned t0 = 0; t0 < kBlk; t0 += kHDV) {
    aie::vector<float, kHDV> sv = aie::load_v<kHDV>(s + t0);
    aie::vector<float, kHDV> d =
        aie::sub(sv, aie::broadcast<float, kHDV>(m_new));
    aie::store_v(pbuf + t0, aie::exp2<bfloat16>(d));
  }

  for (unsigned t = 0; t < n_t; ++t) {
    const float p = (float)pbuf[t];
    l_new += p;
    const bfloat16 p_b = pbuf[t];
    const bfloat16 *__restrict vt = V + t * kHD;
    for (unsigned v = 0; v < kHDVecs; ++v) {
      aie::vector<bfloat16, kHDV> vv = aie::load_v<kHDV>(vt + v * kHDV);
      acc[v] = aie::mac(acc[v], vv, p_b);
    }
  }

  for (unsigned v = 0; v < kHDVecs; ++v)
    aie::store_v(state + v * kHDV, acc[v].template to_vector<float>());
  state[kHD] = m_new;
  state[kHD + 1] = l_new;
#ifdef GRANITE_ATTN_DEBUG_SCORES
  // Overwrite the accumulator with the raw scores so they can be compared
  // element by element against numpy. Which scores are wrong, and by how much,
  // is a fact; which stage is at fault has so far only been a guess.
  for (unsigned t = 0; t < kBlk; ++t) state[t] = s[t];
  // and what the kernel actually received for q and K[0] -- which is what
  // distinguishes 'the maths is wrong' from 'the data never arrived'.
  // Via vector loads: scalar bf16 reads at a computed offset make the Peano
  // backend fail with "immediate operand value -120 is not a multiple of 64".
  // q only: state is kHD+2 floats, and writing K at state+kBlk+kHDV ran off
  // the end of the buffer. K was already confirmed to arrive correctly.
  {
    aie::accum<accfloat, kHDV> tq;
    tq.from_vector(aie::load_v<kHDV>(q));
    aie::store_v(state + kBlk, tq.template to_vector<float>());
  }
#endif
  event1();
}

// Divide the accumulator by the normaliser, once, after the last block.
__attribute__((noinline)) inline void
granite_attn_finish_impl(const float *__restrict state,
                         bfloat16 *__restrict out) {
  event0();
  const float inv = 1.0f / state[kHD + 1];
  const bfloat16 i_hi = (bfloat16)inv;
  const bfloat16 i_lo = (bfloat16)(inv - (float)i_hi);
  for (unsigned v = 0; v < kHDVecs; ++v) {
    aie::accum<accfloat, kHDV> t;
    t.from_vector(aie::load_v<kHDV>(state + v * kHDV));
    aie::vector<bfloat16, kHDV> hi = t.template to_vector<bfloat16>();
    aie::vector<bfloat16, kHDV> lo = aie::sub(t, hi).template to_vector<bfloat16>();
    aie::accum<accfloat, kHDV> r = aie::zeros<accfloat, kHDV>();
    r = aie::mac(r, hi, i_hi);
    r = aie::mac(r, lo, i_hi);
    r = aie::mac(r, hi, i_lo);
    aie::store_v(out + v * kHDV, r.template to_vector<bfloat16>());
  }
  event1();
}

// The two entry points are emitted into SEPARATE translation units, selected by
// these macros. IRON compiles the kernel source once per ExternalFunction, so
// pointing two ExternalFunctions at one .cc yields two objects that each define
// BOTH symbols and the link fails on duplicates -- the same trap granite_gemv.h
// records, walked into again from the other direction. The `impl` functions
// above are `inline`, so the shared body still merges into one COMDAT.
#ifdef GRANITE_ATTN_EMIT_BLOCK
extern "C" {
void granite_attn_block(const bfloat16 *__restrict q,
                        const bfloat16 *__restrict kv, float *__restrict state,
                        unsigned n_t, unsigned first) {
  granite_attn_block_impl(q, kv, state, n_t, first);
}
}
#endif

#ifdef GRANITE_ATTN_EMIT_FINISH
extern "C" {
void granite_attn_finish(const float *__restrict state,
                         bfloat16 *__restrict out) {
  granite_attn_finish_impl(state, out);
}
}
#endif
