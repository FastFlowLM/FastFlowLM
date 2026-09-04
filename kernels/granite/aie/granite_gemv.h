#pragma once
//===- granite_gemv.h  --------------------------------------*- C++ -*-===//
//
// OpenFFLM -- W4A16 GEMV for granite-4.2-3B's lm_head, on the AIE core.
// SPDX-License-Identifier: MIT
//
// y[100352] = W[100352, 2560] @ x[2560], W in q4nx's **q4** form.
//
// WHY THIS EXISTS AND NOT lm_head.h
// ---------------------------------
// `lm_head.h` is W8A16: one scale per block, `w = code * scale`. Granite is
// stored Q4_1, which carries a **minimum** as well as a scale:
//
//     w = code * d + m            code 0..15, d and m bf16 per 32-wide K block
//
// and a half-width code. Both differences are in this file; the design around
// it is the same shape.
//
// WHY THE MINIMUM IS ALMOST FREE
// ------------------------------
// Naively `m` is another per-element term. It is not, because it is constant
// across the 32 K of a block, so it factors out of the inner sum:
//
//   y[r] = sum_k (code[k][r]*d[kb][r] + m[kb][r]) * x[k]
//        = sum_kb { d[kb][r] * (sum_{k in kb} code[k][r]*x[k])
//                 + m[kb][r] * (sum_{k in kb} x[k]) }
//
// The second term needs one **scalar** sum of x per K block -- 8 per tile,
// against 8192 MACs. The minimum costs essentially nothing.
//
// THE LAYOUT, AND WHY IT IS STILL CHEAP
// -------------------------------------
// One q4 tile is 32 output rows x 256 K in 5120 bytes, two row-blocks of 16:
//
//   d  [256] bf16 at tile[0    : 512]    index = kb*32 + rb*16 + r
//   m  [256] bf16 at tile[512  : 1024]   same index
//   nib[4096]  B  at tile[1024 : 5120]   weight i = rb*4096 + k*16 + r,
//                                        byte i>>1, low nibble when i is even
//
// The row index is the FASTEST axis, so 16 consecutive nibbles are 16 different
// rows at one k -- they need 16 different scales, and those are constant for a
// whole k-block. Load them once per (rb, kb) and the inner loop is a mask, a
// to_float with a shifted binary point, a zip and a mac. No gather.
//
// TRAPS OBSERVED (NpuEmbeddings CLAUDE.md)
// ----------------------------------------
//  - AIE default rounding is `floor`, a systematic downward bias baked into
//    every weight. Set conv_even.
//  - AIE2P has NO fp32 vector multiplier: `aie::mul(vector<float>,
//    vector<float>)` compiles and returns **zero**, silently. So the fp32
//    partial is split into two bf16 halves and both are scaled; 8 + 8 mantissa
//    bits land exactly in the fp32 accumulator.
//  - `aie::downshift` on uint8 lowers to a deprecated intrinsic the build
//    promotes to an error; mask and use to_float's shift argument instead,
//    which also does the divide by 16 for free.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kRowBlocks = 2;
static constexpr unsigned kRowsPerBlock = 16;
static constexpr unsigned kRows = 32;         // output rows per tile
static constexpr unsigned kKBlocks = 8;       // 32-wide K blocks per tile
static constexpr unsigned kKInBlock = 32;
static constexpr unsigned kTileK = 256;       // K per tile
static constexpr unsigned kMetaEntries = 256; // d entries, and m entries
static constexpr unsigned kDBytes = 512;      // bf16[256]
static constexpr unsigned kMetaBytes = 1024;  // d then m
static constexpr unsigned kTileBytes = 5120;  // whole q4 tile
// K tiles per entry point. Set per shape by the generator in granite_gemv.py:
// the largest divisor of K/256 that L1 can hold, so K = 2560 -> 5 (2 entry
// points) and down_proj's K = 8192 -> 4 (8 entry points).
#ifndef GRANITE_TILES_PER_CALL
#define GRANITE_TILES_PER_CALL 5
#endif
static constexpr unsigned kTilesPerCall = GRANITE_TILES_PER_CALL;

// Tokens processed per weight pass. THIS IS THE ONE KNOB THAT BEATS THE MEMORY
// BOUND: decode reads all 2.13 GB of weights per token, so B tokens sharing one
// pass divide the per-token traffic by B. It only applies to *independent*
// tokens -- concurrent requests, prefill, or speculative decoding -- since in a
// single autoregressive stream token t+1 needs token t.
//
// The cost is B times the arithmetic per byte, and at B = 1 this design is
// already at the DMA bound, so the compute becomes the wall almost immediately.
// See the task notes: useful, but bounded by this kernel's 16-lane MACs.
#ifndef GRANITE_BATCH
#define GRANITE_BATCH 1
#endif
static constexpr unsigned kBatch = GRANITE_BATCH;

// x is [kBatch][K], so token b starts at x + b*GRANITE_K. Only needed to stride
// between tokens; at kBatch == 1 it is never used.
#ifndef GRANITE_K
#define GRANITE_K 0
#endif

// `kt` selects which 256-wide slice of x this tile covers; `first` starts the
// accumulator rather than adding to it, so the K tiles chain without a separate
// zeroing pass over the output.
//
// Both are RUNTIME arguments and this is deliberately **noinline**. As a
// template with compile-time KT, every entry point instantiated its own copy of
// the body -- measured at 4736 bytes of .text each, against a core program
// memory of 16 KB, so even a single entry point plus the runtime overflowed.
// Emitted once, each entry point below is a 208-byte wrapper (measured).
// KT was only ever pointer arithmetic; it never needed to be compile-time.
__attribute__((noinline)) inline void gemv_q4_tile(const uint8_t *__restrict tile,
                                                   const bfloat16 *__restrict x,
                                                   unsigned kt, bool first,
                                                   float *__restrict y) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict dp = (const bfloat16 *)tile;
  const bfloat16 *__restrict mp = (const bfloat16 *)(tile + kDBytes);
  const uint8_t *__restrict nib = tile + kMetaBytes;
  const bfloat16 *__restrict xt = x + kt * kTileK;

  // One accumulator per row-block rather than one 32-lane accumulator for the
  // tile: the two halves are updated independently, and keeping them apart
  // avoids an insert/extract on every (kb, rb) step. Rows 0..15 are row-block 0
  // and 16..31 row-block 1, which is the order the host expects.
  aie::accum<accfloat, kRowsPerBlock> acc[kRowBlocks][kBatch];
#pragma clang loop unroll(full)
  for (unsigned b = 0; b < kBatch; ++b) {
    float *__restrict yb = y + b * kRows;
    if (first) {
      acc[0][b] = aie::zeros<accfloat, kRowsPerBlock>();
      acc[1][b] = aie::zeros<accfloat, kRowsPerBlock>();
    } else {
      acc[0][b].from_vector(aie::load_v<kRowsPerBlock>(yb));
      acc[1][b].from_vector(aie::load_v<kRowsPerBlock>(yb + kRowsPerBlock));
    }
  }

  // Program memory, not speed, is the binding constraint here: the entry
  // points all link into ONE core program, so an unrolled body is multiplied by
  // however many there are. Peano silently drops chess_* pragmas; clang's are
  // honoured.
#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    // sum of x over this K block -- the whole cost of the Q4_1 minimum.
    //
    // Vectorised, and it matters far more than it looks: as 32 scalar adds this
    // was ~256 scalar ops per tile, and scalar work does not overlap the vector
    // pipeline. It is also pure redundancy -- xs depends only on x and kb, never
    // on the weights, yet it is recomputed for every one of the 3136 tile-rows.
    // Summing through an fp32 accumulator rather than in bf16 keeps it exact;
    // a bf16 running sum over 32 terms would lose ~5 bits.
    float xs[kBatch];
#pragma clang loop unroll(full)
    for (unsigned b = 0; b < kBatch; ++b) {
      aie::accum<accfloat, kKInBlock> xa;
      xa.from_vector(aie::load_v<kKInBlock>(xt + b * GRANITE_K + kb * kKInBlock));
      xs[b] = aie::reduce_add(xa.template to_vector<float>());
    }

#pragma clang loop unroll(disable)
    for (unsigned rb = 0; rb < kRowBlocks; ++rb) {
      const unsigned g = kb * kRows + rb * kRowsPerBlock;
      aie::vector<bfloat16, kRowsPerBlock> d16 = aie::load_v<kRowsPerBlock>(dp + g);
      aie::vector<bfloat16, kRowsPerBlock> m16 = aie::load_v<kRowsPerBlock>(mp + g);

      const uint8_t *__restrict src = nib + rb * 2048 + kb * 256;

      // 16 lanes, one per row of this row-block, accumulated over the
      // block's 32 k.
      aie::accum<accfloat, kRowsPerBlock> part[kBatch];
#pragma clang loop unroll(full)
      for (unsigned b = 0; b < kBatch; ++b)
        part[b] = aie::zeros<accfloat, kRowsPerBlock>();

#pragma clang loop unroll(disable)
      for (unsigned kk = 0; kk < kKInBlock; kk += 8) {
        aie::vector<uint8_t, 64> p = aie::load_v<64>(src + kk * 8);

        // High nibble is masked rather than shifted, and to_float's shift
        // argument (the binary point) does the divide by 16 for free.
        aie::vector<bfloat16, 64> flo =
            aie::to_float<bfloat16>(aie::bit_and((uint8_t)0x0F, p), 0);
        aie::vector<bfloat16, 64> fhi =
            aie::to_float<bfloat16>(aie::bit_and((uint8_t)0xF0, p), 4);

        // Low nibble is the even weight index, so zipping at chunk size 1
        // restores weight order: [lo0, hi0, lo1, hi1, ...].
        auto [c0, c1] = aie::interleave_zip(flo, fhi, 1);

        // c0 covers k = kk..kk+3, c1 covers kk+4..kk+7, 16 rows each, so
        // lane group j of c0 is k = kbase + j.
        //
        // `aie::mac` takes a SCALAR third operand, so x needs no broadcast
        // vector -- building 8 broadcast vectors per iteration is pure code.
        // `extract`'s lane index must be a COMPILE-TIME constant; with a loop
        // variable it lowers to a dynamic shuffle chain. Written out, each
        // extract is register selection and costs nothing.
        // The weight vectors are decoded ONCE and reused by every token in the
        // batch -- that reuse is the whole point of batching. Only the scalar
        // activation changes per token.
        const unsigned kbase = kb * kKInBlock + kk;
#pragma clang loop unroll(full)
        for (unsigned b = 0; b < kBatch; ++b) {
          const bfloat16 *__restrict xb = xt + b * GRANITE_K;
          part[b] = aie::mac(part[b], c0.template extract<kRowsPerBlock>(0), xb[kbase + 0]);
          part[b] = aie::mac(part[b], c0.template extract<kRowsPerBlock>(1), xb[kbase + 1]);
          part[b] = aie::mac(part[b], c0.template extract<kRowsPerBlock>(2), xb[kbase + 2]);
          part[b] = aie::mac(part[b], c0.template extract<kRowsPerBlock>(3), xb[kbase + 3]);
          part[b] = aie::mac(part[b], c1.template extract<kRowsPerBlock>(0), xb[kbase + 4]);
          part[b] = aie::mac(part[b], c1.template extract<kRowsPerBlock>(1), xb[kbase + 5]);
          part[b] = aie::mac(part[b], c1.template extract<kRowsPerBlock>(2), xb[kbase + 6]);
          part[b] = aie::mac(part[b], c1.template extract<kRowsPerBlock>(3), xb[kbase + 7]);
        }
      }

#pragma clang loop unroll(full)
      for (unsigned b = 0; b < kBatch; ++b) {
      aie::accum<accfloat, kRowsPerBlock> sum16 = part[b];

      // acc[row] += d*sum + m*xs. Both products must be bf16 x bf16: AIE2P has
      // no fp32 vector multiplier, and aie_api compiles one and returns **zero**
      // in silence. Splitting each fp32 value into two bf16 halves keeps 8 + 8
      // mantissa bits, which land exactly in the fp32 accumulator.
      aie::vector<bfloat16, kRowsPerBlock> hi =
          sum16.template to_vector<bfloat16>();
      aie::vector<bfloat16, kRowsPerBlock> lo =
          aie::sub(sum16, hi).template to_vector<bfloat16>();
      acc[rb][b] = aie::mac(acc[rb][b], hi, d16);
      acc[rb][b] = aie::mac(acc[rb][b], lo, d16);

      // Same split for the block's sum of x, which carries the minimum.
      const bfloat16 xs_hi = (bfloat16)xs[b];
      const bfloat16 xs_lo = (bfloat16)(xs[b] - (float)xs_hi);
      acc[rb][b] = aie::mac(acc[rb][b], m16, xs_hi);
      acc[rb][b] = aie::mac(acc[rb][b], m16, xs_lo);
      }
    }
  }

#pragma clang loop unroll(full)
  for (unsigned b = 0; b < kBatch; ++b) {
    float *__restrict yb = y + b * kRows;
    aie::store_v(yb, acc[0][b].template to_vector<float>());
    aie::store_v(yb + kRowsPerBlock, acc[1][b].template to_vector<float>());
  }
  event1();
}

// Five K tiles per call.
//
// Program memory is the binding constraint, and it was found by measuring, not
// by reasoning: three separate theories (entry-point count, loop unrolling,
// runtime extract indices) were all wrong. Compiling the object directly and
// reading `llvm-objdump -h` settled it in seconds -- the body was 4736 B because
// it was a template on the K-tile index, so EVERY entry point instantiated its
// own copy and even one overflowed the core's 16 KB.
//
// KT was only ever pointer arithmetic (`x + KT*256`); it never needed to be
// compile-time. Runtime + noinline emits the body once per translation unit.
//
// It is `inline`, NOT `static`, and that distinction is what makes wide K
// affordable. `static` gives each entry point's translation unit a private copy;
// `inline` gives the body vague linkage, so the copies land in a COMDAT and the
// linker keeps exactly one (`llvm-objdump -t` shows the symbol as `w`). The cost
// of an entry point drops from a whole body to its 208-byte wrapper -- which is
// what lets down_proj's K = 8192 have its 8 entry points at all.
//
// The other side of the trade is L1: an element is now 5 x 5120 = 25600 B, and
// double-buffered that is 51200 B against the 63 KB budget (64 KB less ~1 KB of
// stack). Adding x (5120 B) and y still fits, and keeping depth 2 matters --
// this kernel is bandwidth-bound, so the DMA must overlap the compute.
static inline void gemv_q4_group(const uint8_t *__restrict tiles,
                                 const bfloat16 *__restrict x, unsigned group,
                                 float *__restrict y) {
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kTilesPerCall; ++i) {
    const unsigned kt = group * kTilesPerCall + i;
    gemv_q4_tile(tiles + i * kTileBytes, x, kt, kt == 0, y);
  }
}

extern "C" {

// One entry point per group of five K tiles. They live in separate translation
// units because IRON compiles the kernel source once per ExternalFunction:
// several functions in one .cc become several objects that each define every
// symbol, and the link fails on duplicates.
//
// Only the group index is needed -- whether to start or continue the accumulator
// follows from it (`kt == 0`), so it is not a second argument that could
// disagree with the first.
// The tiles-per-call variant is part of the SYMBOL, not just the file name.
// down_proj wants 4 tiles per call and everything else wants 5; if both variants
// produced `granite_gemv_k0`, a build cache keyed on anything other than the
// kernel source bytes could hand one shape the other's object -- which is not an
// error, just a different and wrong matmul. Distinct names make that impossible.
// The extra indirection is so GRANITE_TILES_PER_CALL expands before ## pastes.
#define GRANITE_GEMV_ENTRY__(P, B, N)                                            void granite_gemv_p##P##b##B##_k##N(const uint8_t *__restrict t,                                                   const bfloat16 *__restrict x,                                                  float *__restrict y) {                       gemv_q4_group(t, x, N, y);                                                   }
#define GRANITE_GEMV_ENTRY_(P, B, N) GRANITE_GEMV_ENTRY__(P, B, N)
#define GRANITE_GEMV_ENTRY(N)                                                    GRANITE_GEMV_ENTRY_(GRANITE_TILES_PER_CALL, GRANITE_BATCH, N)

}  // extern "C"
