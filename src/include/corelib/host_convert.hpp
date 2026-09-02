#pragma once

// The only two host conversions FastFlow is permitted to implement
// (design `API-6`). Corelib `e5258d2` removed `ryzenai_corelib_convert`
// and `ryzenai_corelib_convert_strided`: every conversion with a tensor on
// either side now crosses `tensor_write` / `tensor_read`. What remains here
// is data that never touches a tensor.
//
//   1. FP16 -> FP32 widening, for the embedding rows gathered before the
//      host RMSNorm and for an FP16 layer-0 norm. Lossless, so it has no
//      rounding policy and cannot disagree with corelib. It is a scalar
//      loop on purpose: the source is a read-only file mapping, and the
//      naive vectorized widening reads up to 14 bytes past its source,
//      which faults on a page boundary rather than returning garbage.
//
//   2. FP32 -> BF16 round-to-nearest-even, for the SSMLP epsilon / norm0 /
//      norm1 blobs, which are packer inputs rather than tensors. Bit-
//      compatible with the reference driver's `to_bf16`.
//
// There is deliberately no third converter. In particular there is no host
// FP32-to-FP16 narrowing, which is why an FP32 `scales` array is rejected
// rather than converted.

#include <bit>
#include <cstddef>
#include <cstdint>

namespace flm::corelib {

// Exact: every FP16 value is representable in FP32.
inline float WidenFp16(std::uint16_t bits) noexcept {
    const std::uint32_t sign =
        static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent =
        (static_cast<std::uint32_t>(bits) >> 10) & 0x1Fu;
    const std::uint32_t mantissa =
        static_cast<std::uint32_t>(bits) & 0x3FFu;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            return std::bit_cast<float>(sign);
        }
        std::uint32_t significand = mantissa;
        std::uint32_t shift = 0u;
        while ((significand & 0x400u) == 0u) {
            significand <<= 1;
            ++shift;
        }
        significand &= 0x3FFu;
        const std::uint32_t widened_exponent = 127u - 15u - shift + 1u;
        return std::bit_cast<float>(
            sign | (widened_exponent << 23) | (significand << 13));
    }
    if (exponent == 0x1Fu) {
        return std::bit_cast<float>(
            sign | 0x7F800000u | (mantissa << 13));
    }
    return std::bit_cast<float>(
        sign | ((exponent - 15u + 127u) << 23) | (mantissa << 13));
}

// Scalar and tail-guarded: reads exactly `count` halfwords from `source`
// and touches nothing beyond them.
inline void WidenFp16Array(
    const std::uint16_t* source,
    std::size_t count,
    float* destination) noexcept {
    for (std::size_t index = 0; index < count; ++index) {
        destination[index] = WidenFp16(source[index]);
    }
}

// Bit-compatible with the reference driver's `to_bf16`:
//   rounded = bits + 0x7FFF + ((bits >> 16) & 1); return rounded >> 16
// The driver accumulates in uint64 and narrows to uint16; the wrapped
// uint32 sum has the same low 16 bits, so the two agree on every input.
inline std::uint16_t NarrowFp32ToBf16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t rounded =
        bits + 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

inline void NarrowFp32ToBf16Array(
    const float* source,
    std::size_t count,
    std::uint16_t* destination) noexcept {
    for (std::size_t index = 0; index < count; ++index) {
        destination[index] = NarrowFp32ToBf16(source[index]);
    }
}

}  // namespace flm::corelib
