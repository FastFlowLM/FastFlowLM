#pragma once

#include <biovault_bfloat16.h>
#include <corelib/corelib_api.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace flm::phi4 {

struct VScatterMetrics {
    std::uint64_t read_calls = 0;
    std::uint64_t write_calls = 0;
    std::uint64_t bytes = 0;
    std::uint64_t nanoseconds = 0;
};

// Widens the gathered FP16 rows to FP32 with the `API-6` bounds-safe
// scalar helper. There is no tensor on either side of this conversion:
// the host RMSNorm consumes the result directly.
void GatherEmbedding(
    std::span<const std::uint16_t> embedding_fp16,
    std::span<const int> token_ids,
    std::span<float> output);

void RmsNorm(
    std::span<const float> input,
    std::span<const float> scale,
    std::int64_t rows,
    std::int64_t width,
    float epsilon,
    std::span<float> output);

// Stages only the helper-required initial hidden/residual prefix, in FP32.
// Elements beyond padded_rows * width are intentionally untouched, and the
// FP32-to-BF16 narrowing is corelib's, inside tensor_write.
void StageFp32(
    std::span<const float> input,
    std::int64_t live_rows,
    std::int64_t padded_rows,
    std::int64_t width,
    std::span<float> output);

// Precondition: the caller has successfully synchronized the Stream after
// V projection and before this host read. ScatterV deliberately owns no
// Stream and performs no synchronization.
void ScatterV(
    const corelib::CorelibApi& api,
    ryzenai_corelib_tensor_ptr source,
    ryzenai_corelib_tensor_ptr value_cache,
    std::int64_t rows,
    std::int64_t position,
    std::vector<std::uint16_t>& staging,
    VScatterMetrics& metrics);

int ArgmaxLowest(
    std::span<const biovault::bfloat16_t> logits);

}  // namespace flm::phi4
