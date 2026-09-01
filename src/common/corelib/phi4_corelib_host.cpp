#include <models/phi4/phi4_corelib_host.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>

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

constexpr std::string_view kConvertCall =
    "ryzenai_corelib_convert";
constexpr std::string_view kTensorReadCall =
    "ryzenai_corelib_tensor_read";
constexpr std::string_view kTensorWriteCall =
    "ryzenai_corelib_tensor_write";

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
    const corelib::CorelibApi& api,
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

    const std::size_t vocabulary_rows =
        embedding_fp16.size() / width;
    thread_local std::vector<std::uint16_t> staging;
    staging.resize(output_count);
    for (std::size_t row = 0; row < token_ids.size(); ++row) {
        const int token_id = token_ids[row];
        if (token_id < 0 ||
            static_cast<std::size_t>(token_id) >= vocabulary_rows) {
            throw std::out_of_range(
                "Phi-4 embedding token ID is outside the mapped table");
        }
        const std::size_t source_offset =
            static_cast<std::size_t>(token_id) * width;
        std::copy_n(
            embedding_fp16.begin() + source_offset,
            width,
            staging.begin() + row * width);
    }

    api.Check(
        api.functions().convert(
            ryzenai_corelib_data_type_fp16,
            staging.data(),
            ryzenai_corelib_data_type_fp32,
            output.data(),
            output_count),
        kConvertCall);
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
        float sum_of_squares = 0.0f;
        for (std::size_t column = 0;
             column < row_width;
             ++column) {
            const float value = input[base + column];
            sum_of_squares += value * value;
        }
        const float mean_square =
            sum_of_squares / static_cast<float>(width);
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

void StageBf16(
    const corelib::CorelibApi& api,
    std::span<const float> input,
    std::int64_t live_rows,
    std::int64_t padded_rows,
    std::int64_t width,
    std::span<std::uint16_t> output) {
    if (padded_rows < live_rows) {
        throw std::invalid_argument(
            "Phi-4 BF16 staging padded rows are smaller than live rows");
    }
    const std::size_t live_count =
        CheckedExtent(live_rows, width, "Phi-4 BF16 staging");
    const std::size_t padded_count =
        CheckedExtent(padded_rows, width, "Phi-4 BF16 staging");
    if (input.size() != live_count || output.size() < padded_count) {
        throw std::invalid_argument(
            "Phi-4 BF16 staging shape mismatch");
    }

    thread_local std::vector<float> staging;
    staging.resize(padded_count);
    std::copy(input.begin(), input.end(), staging.begin());
    std::fill(
        staging.begin() + live_count,
        staging.end(),
        0.0f);
    api.Check(
        api.functions().convert(
            ryzenai_corelib_data_type_fp32,
            staging.data(),
            ryzenai_corelib_data_type_bf16,
            output.data(),
            padded_count),
        kConvertCall);
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
    const std::size_t source_bytes =
        source_count * sizeof(std::uint16_t);
    api.Check(
        api.functions().tensor_read(
            source,
            staging.data(),
            source_bytes,
            0),
        kTensorReadCall);
    ++metrics.read_calls;
    metrics.bytes += source_bytes;

    std::uint16_t* const head_staging =
        staging.data() + source_count;
    const std::size_t head_bytes =
        head_staging_count * sizeof(std::uint16_t);
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
            head_width * sizeof(std::uint16_t);
        api.Check(
            api.functions().tensor_write(
                value_cache,
                head_staging,
                head_bytes,
                cache_offset),
            kTensorWriteCall);
        ++metrics.write_calls;
        metrics.bytes += head_bytes;
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
