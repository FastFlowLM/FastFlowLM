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
inline constexpr double kRmsEpsilon = 1.0e-5;

static_assert(
    kQueryHeadCount * kHeadSize == kQueryDimension);
static_assert(kQueryDimension == kHiddenSize);
static_assert(kKvHeadCount * kHeadSize == kKvDimension);

}  // namespace flm::phi4::constants
