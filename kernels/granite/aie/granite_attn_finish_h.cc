// OpenFFLM -- attention epilogue for one of a core's several query heads.
// SPDX-License-Identifier: MIT
#include "granite_attention.h"

// The per-head state is kHD + 2 = 66 floats, but the stride is 128, not 66 and
// not 72. The kernel stores the accumulator with aie::store_v of 32 floats --
// vectors that are 128 BYTES wide -- so a head's state has to start on a
// 128-byte boundary.
//
// 72 floats = 288 bytes is 32-byte aligned, which is what a first attempt at
// this checked, and it is not enough: head 1's store rounded down to the
// nearest 128-byte boundary, float index 64, which is exactly where head 0
// keeps its softmax max and denominator. Head 1 was overwriting head 0's
// divisor.
//
// The symptom named the cause once it was read properly: every head came out
// with cosine +-1.0 against the reference -- right direction, wrong scale --
// and head 0 was wrong too, which no offset error of head 1's own could do.
// q_per = 1 passed because there was no head 1 to do the overwriting.
static constexpr unsigned kStStride = 128;   // floats: 512 B, 128-B aligned

extern "C" {
void granite_attn_finish_h(const float *__restrict st_all,
                           bfloat16 *__restrict out_all, unsigned h) {
  granite_attn_finish_impl(st_all + h * kStStride,
                           out_all + h * kHD);
}
}
