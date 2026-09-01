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

void GatherEmbedding(
    const corelib::CorelibApi& api,
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

// Stages only the helper-required initial hidden/residual prefix.
// Elements beyond padded_rows * width are intentionally untouched.
void StageBf16(
    const corelib::CorelibApi& api,
    std::span<const float> input,
    std::int64_t live_rows,
    std::int64_t padded_rows,
    std::int64_t width,
    std::span<std::uint16_t> output);

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
