#include <models/phi4/phi4_corelib_host.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>

#include <corelib/host_convert.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flm::phi4 {
namespace {

std::size_t CheckedExtent(
    std::int64_t rows,
    std::int64_t width,
    std::string_view context) {
    if (rows <= 0 || width <= 0) {
        throw std::invalid_argument(
            std::string(context) +
            " rows and width must be positive");
    }
    const auto row_count = static_cast<std::uint64_t>(rows);
    const auto column_count = static_cast<std::uint64_t>(width);
    if (column_count >
        std::numeric_limits<std::size_t>::max() / row_count) {
        throw std::overflow_error(
            std::string(context) + " extent overflows size_t");
    }
    return static_cast<std::size_t>(row_count * column_count);
}

}  // namespace

void GatherEmbedding(
    std::span<const std::uint16_t> embedding_fp16,
    std::span<const int> token_ids,
    std::span<float> output) {
    constexpr std::size_t width =
        static_cast<std::size_t>(constants::kHiddenSize);
    if (embedding_fp16.empty() ||
        embedding_fp16.size() % width != 0) {
        throw std::invalid_argument(
            "Phi-4 embedding shape must be [vocabulary, 3072]");
    }
    if (token_ids.size() >
        std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error(
            "Phi-4 embedding gather extent overflows size_t");
    }
    const std::size_t output_count = token_ids.size() * width;
    if (output.size() != output_count) {
        throw std::invalid_argument(
            "Phi-4 embedding output shape does not match token IDs");
    }
    if (token_ids.empty()) {
        return;
    }

    // Validate every token before touching the mapping, so an out-of-range
    // ID fails instead of reading a row that is not there.
    const std::size_t vocabulary_rows =
        embedding_fp16.size() / width;
    for (const int token_id : token_ids) {
        if (token_id < 0 ||
            static_cast<std::size_t>(token_id) >= vocabulary_rows) {
            throw std::out_of_range(
                "Phi-4 embedding token ID is outside the mapped table");
        }
    }

    // The gathered rows never reach a tensor before the host RMSNorm
    // consumes them, so this is the `API-6` FP16-to-FP32 widening. It is
    // scalar and per-element on purpose: the embedding table is a
    // read-only file mapping, and a vectorized widening over-reads its
    // source by up to 14 bytes, which faults on a page boundary.
    for (std::size_t row = 0; row < token_ids.size(); ++row) {
        const std::size_t source_offset =
            static_cast<std::size_t>(token_ids[row]) * width;
        corelib::WidenFp16Array(
            embedding_fp16.data() + source_offset,
            width,
            output.data() + row * width);
    }
}

void RmsNorm(
    std::span<const float> input,
    std::span<const float> scale,
    std::int64_t rows,
    std::int64_t width,
    float epsilon,
    std::span<float> output) {
    const std::size_t element_count =
        CheckedExtent(rows, width, "Phi-4 RMSNorm");
    if (input.size() != element_count ||
        output.size() != element_count ||
        scale.size() != static_cast<std::size_t>(width)) {
        throw std::invalid_argument(
            "Phi-4 RMSNorm shape mismatch");
    }
    if (!std::isfinite(epsilon) || epsilon < 0.0f) {
        throw std::invalid_argument(
            "Phi-4 RMSNorm epsilon must be finite and nonnegative");
    }

    const std::size_t row_width = static_cast<std::size_t>(width);
    for (std::size_t row = 0;
         row < static_cast<std::size_t>(rows);
         ++row) {
        const std::size_t base = row * row_width;
        // Accumulated in double, then rounded once.
        //
        // A serial FP32 sum of 3072 squares is the least accurate reduction
        // available here, and the error is not academic: measured on the AIE4
        // target against the corelib reference driver, it moved the layer-0
        // input by up to 4.8e-6 relative, which is enough to land 25 of 58368
        // values on a DIFFERENT BF16 number once `tensor_write` narrows them.
        // Thirty-two layers of BF16 arithmetic amplify those 25 seeds into a
        // logit correlation of 0.9991 against the reference, below the 0.9999
        // design Section 12.4 requires, and into different sampled tokens
        // within six steps.
        //
        // A double accumulator costs one row-length pass per forward pass --
        // this norm runs once per model step, not once per layer, because
        // ssmlp absorbs every other one -- and brings the count of differing
        // BF16 values to zero against the same reference. The rest of the
        // computation stays in FP32 so the result is still exactly what
        // design Section 10.2 specifies the host to produce.
        double sum_of_squares = 0.0;
        for (std::size_t column = 0;
             column < row_width;
             ++column) {
            const double value = input[base + column];
            sum_of_squares += value * value;
        }
        const float mean_square = static_cast<float>(
            sum_of_squares / static_cast<double>(width));
        const float denominator =
            std::sqrt(mean_square + epsilon);
        for (std::size_t column = 0;
             column < row_width;
             ++column) {
            output[base + column] =
                (input[base + column] / denominator) *
                scale[column];
        }
    }
}

void StageFp32(
    std::span<const float> input,
    std::int64_t live_rows,
    std::int64_t padded_rows,
    std::int64_t width,
    std::span<float> output) {
    if (padded_rows < live_rows) {
        throw std::invalid_argument(
            "Phi-4 FP32 staging padded rows are smaller than live rows");
    }
    const std::size_t live_count =
        CheckedExtent(live_rows, width, "Phi-4 FP32 staging");
    const std::size_t padded_count =
        CheckedExtent(padded_rows, width, "Phi-4 FP32 staging");
    if (input.size() != live_count || output.size() < padded_count) {
        throw std::invalid_argument(
            "Phi-4 FP32 staging shape mismatch");
    }

    // Design Section 10.2: the host stays in FP32 and never produces BF16
    // activations. Corelib narrows FP32 to BF16 inside `tensor_write`, so
    // there is exactly one BF16 rounding implementation on this path and
    // FastFlow does not have to match it.
    std::copy(input.begin(), input.end(), output.begin());
    std::fill(
        output.begin() + live_count,
        output.begin() + padded_count,
        0.0f);
}

void ScatterV(
    const corelib::CorelibApi& api,
    ryzenai_corelib_tensor_ptr source,
    ryzenai_corelib_tensor_ptr value_cache,
    std::int64_t rows,
    std::int64_t position,
    std::vector<std::uint16_t>& staging,
    VScatterMetrics& metrics) {
    if (source == nullptr || value_cache == nullptr) {
        throw std::invalid_argument(
            "Phi-4 V scatter requires source and cache tensors");
    }
    if (rows <= 0 ||
        rows > constants::kMaxSequenceLength ||
        position < 0 ||
        position >
            constants::kMaxSequenceLength - rows) {
        throw std::out_of_range(
            "Phi-4 V scatter live rows exceed the cache window");
    }

    constexpr std::size_t head_count =
        static_cast<std::size_t>(constants::kKvHeadCount);
    constexpr std::size_t head_width =
        static_cast<std::size_t>(constants::kHeadSize);
    constexpr std::size_t row_width = head_count * head_width;
    const std::size_t live_rows = static_cast<std::size_t>(rows);
    const std::size_t source_count = live_rows * row_width;
    const std::size_t head_staging_count = live_rows * head_width;
    staging.resize(source_count + head_staging_count);

    const auto started = std::chrono::steady_clock::now();
    // `count` and `offset` are BF16 ELEMENTS of the cache's own dtype, not
    // bytes (`API-7`). Both tensors are BF16 and so is the host staging
    // buffer, so every transfer here is a straight copy.
    api.ReadElements(
        source,
        ryzenai_corelib_data_type_bf16,
        staging.data(),
        source_count,
        0);
    ++metrics.read_calls;
    metrics.bytes += source_count * sizeof(std::uint16_t);

    std::uint16_t* const head_staging =
        staging.data() + source_count;
    for (std::size_t head = 0; head < head_count; ++head) {
        for (std::size_t row = 0; row < live_rows; ++row) {
            const std::size_t source_offset =
                (row * head_count + head) * head_width;
            std::copy_n(
                staging.data() + source_offset,
                head_width,
                head_staging + row * head_width);
        }
        const std::size_t cache_offset =
            ((head *
                  static_cast<std::size_t>(
                      constants::kMaxSequenceLength)) +
             static_cast<std::size_t>(position)) *
            head_width;
        api.WriteElements(
            value_cache,
            ryzenai_corelib_data_type_bf16,
            head_staging,
            head_staging_count,
            cache_offset);
        ++metrics.write_calls;
        metrics.bytes += head_staging_count * sizeof(std::uint16_t);
    }
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
    metrics.nanoseconds +=
        static_cast<std::uint64_t>(elapsed.count());
}

int ArgmaxLowest(
    std::span<const biovault::bfloat16_t> logits) {
    if (logits.empty()) {
        throw std::invalid_argument(
            "Phi-4 argmax cannot consume empty logits");
    }
    if (logits.size() >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "Phi-4 argmax token index exceeds int");
    }

    int best_index = 0;
    float best_value = static_cast<float>(logits.front());
    for (std::size_t index = 1; index < logits.size(); ++index) {
        const float value = static_cast<float>(logits[index]);
        if (value > best_value) {
            best_value = value;
            best_index = static_cast<int>(index);
        }
    }
    return best_index;
}

}  // namespace flm::phi4
