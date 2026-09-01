#pragma once

#include <cstddef>
#include <cstdint>

namespace flm::phi4 {

enum class ContinuationRoute {
    Append,
    Reprefill
};

enum class ForcedContinuationRoute {
    Automatic,
    Append,
    Reprefill
};

inline constexpr std::uint32_t kContinuationAppendThreshold = 0;

inline constexpr ContinuationRoute SelectContinuationRoute(
    std::size_t suffix_tokens,
    ForcedContinuationRoute forced) noexcept {
    switch (forced) {
        case ForcedContinuationRoute::Append:
            return ContinuationRoute::Append;
        case ForcedContinuationRoute::Reprefill:
            return ContinuationRoute::Reprefill;
        case ForcedContinuationRoute::Automatic:
            return suffix_tokens <= kContinuationAppendThreshold
                       ? ContinuationRoute::Append
                       : ContinuationRoute::Reprefill;
    }
    return ContinuationRoute::Reprefill;
}

inline constexpr const char* ContinuationRouteName(
    ContinuationRoute route) noexcept {
    return route == ContinuationRoute::Append
               ? "append"
               : "reprefill";
}

}  // namespace flm::phi4
