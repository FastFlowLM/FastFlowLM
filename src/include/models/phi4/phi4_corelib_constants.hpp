#pragma once

#include <cstdint>

namespace flm::phi4::constants {

inline constexpr std::int64_t kLayerCount = 32;
inline constexpr std::int64_t kHiddenSize = 3072;
inline constexpr std::int64_t kIntermediateSize = 8192;
inline constexpr std::int64_t kQueryHeadCount = 24;
inline constexpr std::int64_t kKvHeadCount = 8;
inline constexpr std::int64_t kHeadSize = 128;
inline constexpr std::int64_t kQueryDimension = 3072;
inline constexpr std::int64_t kKvDimension = 1024;
inline constexpr std::int64_t kVocabularySize = 200064;
inline constexpr std::uint32_t kGroupSize = 128;
inline constexpr std::int64_t kRopeDimension = 96;
inline constexpr std::int64_t kMaxSequenceLength = 4096;

// The TOKEN attention path supports a smaller KV window than the prefill path.
//
// Measured on the AIE4 target 2026-09-02. A 4096-row prefill at position 0
// dispatches fine -- the boundary sweep runs one -- but a rows=1 step at
// position 4095, whose window is also 4096, is refused by the shipped kernel
// set with "no token attention kernel ships for a 4096-token window". So the
// two paths do not share a bound, and the decode one is `kMaxSequenceLength`
// minus one.
//
// This is not a cosmetic limit. The refusal arrives from `flat_mha` AFTER q, k
// and v have been submitted in the same step, so the failure policy correctly
// classifies it as past the irrevocable boundary and TERMINATES THE PROCESS.
// A generation that walks to position 4095 therefore kills the server, which
// is what it did on 2026-09-02 through a default /api/chat request. Anything
// that can reach a decode step must respect this, not kMaxSequenceLength.
inline constexpr std::int64_t kMaxDecodeWindow = kMaxSequenceLength - 1;

inline constexpr double kRmsEpsilon = 1.0e-5;

static_assert(
    kQueryHeadCount * kHeadSize == kQueryDimension);
static_assert(kQueryDimension == kHiddenSize);
static_assert(kKvHeadCount * kHeadSize == kKvDimension);

}  // namespace flm::phi4::constants
