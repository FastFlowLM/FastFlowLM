#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <models/phi4/phi4_corelib_constants.hpp>
#include <models/phi4/phi4_corelib_host.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using flm::corelib::CorelibApi;
using flm::phi4::ArgmaxLowest;
using flm::phi4::GatherEmbedding;
using flm::phi4::RmsNorm;
using flm::phi4::ScatterV;
using flm::phi4::StageBf16;
using flm::phi4::VScatterMetrics;
using bf16 = biovault::bfloat16_t;

constexpr std::uint16_t kPoison = 0xDEADu;

struct ConvertCall {
    ryzenai_corelib_data_type source_type;
    const void* source;
    ryzenai_corelib_data_type destination_type;
    void* destination;
    std::size_t count;
    std::vector<std::uint16_t> fp16_source;
    std::vector<std::uint32_t> fp32_source_bits;
};

struct ReadCall {
    ryzenai_corelib_tensor_ptr tensor;
    void* destination;
    std::size_t size;
    std::size_t offset;
};

struct WriteCall {
    ryzenai_corelib_tensor_ptr tensor;
    const void* source;
    std::size_t size;
    std::size_t offset;
    std::vector<std::uint16_t> values;
};

struct RecordingState {
    std::vector<ConvertCall> converts;
    std::vector<ReadCall> reads;
    std::vector<WriteCall> writes;
    std::size_t synchronize_calls = 0;
    std::vector<std::uint16_t> v_source;
    int v_tensor_storage = 0;
    int v_cache_storage = 0;

    ryzenai_corelib_tensor_ptr v_tensor() noexcept {
        return &v_tensor_storage;
    }

    ryzenai_corelib_tensor_ptr v_cache() noexcept {
        return &v_cache_storage;
    }

    void ResetCalls() {
        converts.clear();
        reads.clear();
        writes.clear();
        synchronize_calls = 0;
    }
};

RecordingState g_recording;

bool FixtureFp16ToFp32(std::uint16_t bits, float& value) {
    switch (bits) {
        case 0x0000u:
            value = 0.0f;
            return true;
        case 0x3800u:
            value = 0.5f;
            return true;
        case 0x3C00u:
            value = 1.0f;
            return true;
        case 0x4000u:
            value = 2.0f;
            return true;
        case 0x4200u:
            value = 3.0f;
            return true;
        case 0xBC00u:
            value = -1.0f;
            return true;
        case 0xC000u:
            value = -2.0f;
            return true;
        default:
            return false;
    }
}

bool FixtureFp32ToBf16(std::uint32_t bits, std::uint16_t& value) {
    switch (bits) {
        case 0x00000000u:
            value = 0x0000u;
            return true;
        case 0x3F000000u:
            value = 0x3F00u;
            return true;
        case 0x3F800000u:
            value = 0x3F80u;
            return true;
        case 0x3F808000u:
            value = 0x3F80u;
            return true;
        case 0x3F818000u:
            value = 0x3F82u;
            return true;
        case 0x40000000u:
            value = 0x4000u;
            return true;
        case 0xBF800000u:
            value = 0xBF80u;
            return true;
        case 0xC0000000u:
            value = 0xC000u;
            return true;
        default:
            return false;
    }
}

ryzenai_corelib_status RecordingConvert(
    ryzenai_corelib_data_type source_type,
    const void* source,
    ryzenai_corelib_data_type destination_type,
    void* destination,
    std::size_t count) {
    if ((source == nullptr || destination == nullptr) && count != 0) {
        return ryzenai_corelib_status_bad_argument;
    }

    ConvertCall call{
        source_type,
        source,
        destination_type,
        destination,
        count,
        {},
        {}};
    if (source_type == ryzenai_corelib_data_type_fp16 &&
        destination_type == ryzenai_corelib_data_type_fp32) {
        const auto* input =
            static_cast<const std::uint16_t*>(source);
        auto* output = static_cast<float*>(destination);
        call.fp16_source.assign(input, input + count);
        for (std::size_t index = 0; index < count; ++index) {
            if (!FixtureFp16ToFp32(input[index], output[index])) {
                return ryzenai_corelib_status_bad_argument;
            }
        }
    } else if (
        source_type == ryzenai_corelib_data_type_fp32 &&
        destination_type == ryzenai_corelib_data_type_bf16) {
        const auto* input = static_cast<const float*>(source);
        auto* output = static_cast<std::uint16_t*>(destination);
        call.fp32_source_bits.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto bits = std::bit_cast<std::uint32_t>(input[index]);
            call.fp32_source_bits.push_back(bits);
            if (!FixtureFp32ToBf16(bits, output[index])) {
                return ryzenai_corelib_status_bad_argument;
            }
        }
    } else {
        return ryzenai_corelib_status_bad_argument;
    }
    g_recording.converts.push_back(std::move(call));
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingTensorRead(
    ryzenai_corelib_tensor_ptr tensor,
    void* destination,
    std::size_t size,
    std::size_t offset) {
    if (tensor != g_recording.v_tensor() || destination == nullptr ||
        offset > g_recording.v_source.size() * sizeof(std::uint16_t) ||
        size > g_recording.v_source.size() * sizeof(std::uint16_t) -
                   offset) {
        return ryzenai_corelib_status_bad_argument;
    }
    g_recording.reads.push_back(
        ReadCall{tensor, destination, size, offset});
    std::memcpy(
        destination,
        reinterpret_cast<const std::byte*>(
            g_recording.v_source.data()) +
            offset,
        size);

    const auto tick = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() == tick) {
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingTensorWrite(
    ryzenai_corelib_tensor_ptr tensor,
    const void* source,
    std::size_t size,
    std::size_t offset) {
    if (tensor != g_recording.v_cache() || source == nullptr ||
        size % sizeof(std::uint16_t) != 0) {
        return ryzenai_corelib_status_bad_argument;
    }
    const auto* values = static_cast<const std::uint16_t*>(source);
    g_recording.writes.push_back(WriteCall{
        tensor,
        source,
        size,
        offset,
        std::vector<std::uint16_t>(
            values,
            values + size / sizeof(std::uint16_t))});
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingStreamSynchronize(
    ryzenai_corelib_stream_ptr) {
    ++g_recording.synchronize_calls;
    return ryzenai_corelib_status_success;
}

template <class Function>
void* FunctionAddress(Function function) {
    return reinterpret_cast<void*>(function);
}

std::shared_ptr<CorelibApi> ResolveRecordingCorelib() {
    auto resolver = flm::test::CompleteCorelibResolver();
    resolver["ryzenai_corelib_convert"] =
        FunctionAddress(
            static_cast<decltype(&::ryzenai_corelib_convert)>(
                &RecordingConvert));
    resolver["ryzenai_corelib_tensor_read"] =
        FunctionAddress(
            static_cast<decltype(&::ryzenai_corelib_tensor_read)>(
                &RecordingTensorRead));
    resolver["ryzenai_corelib_tensor_write"] =
        FunctionAddress(
            static_cast<decltype(&::ryzenai_corelib_tensor_write)>(
                &RecordingTensorWrite));
    resolver["ryzenai_corelib_stream_synchronize"] =
        FunctionAddress(
            static_cast<
                decltype(&::ryzenai_corelib_stream_synchronize)>(
                &RecordingStreamSynchronize));
    return CorelibApi::ResolveForTest(
        [resolver = std::move(resolver)](std::string_view name) mutable
            -> void* {
            const auto found = resolver.find(std::string(name));
            return found == resolver.end() ? nullptr : found->second;
        });
}

void TestGatherEmbeddingUsesOneReusableContiguousConversion(
    const std::shared_ptr<CorelibApi>& api) {
    constexpr std::size_t width = static_cast<std::size_t>(
        flm::phi4::constants::kHiddenSize);
    std::vector<std::uint16_t> embedding(3u * width);
    std::fill_n(embedding.begin(), width, 0x3C00u);
    std::fill_n(embedding.begin() + width, width, 0xC000u);
    std::fill_n(embedding.begin() + 2u * width, width, 0x4200u);

    const std::array<int, 2> ids{2, 0};
    std::vector<float> output(ids.size() * width);
    g_recording.ResetCalls();
    GatherEmbedding(*api, embedding, ids, output);

    CHECK(g_recording.converts.size() == 1);
    const auto& first_call = g_recording.converts.front();
    CHECK(
        first_call.source_type ==
        ryzenai_corelib_data_type_fp16);
    CHECK(
        first_call.destination_type ==
        ryzenai_corelib_data_type_fp32);
    CHECK(first_call.count == ids.size() * width);
    CHECK(first_call.fp16_source.size() == ids.size() * width);
    CHECK(std::all_of(
        first_call.fp16_source.begin(),
        first_call.fp16_source.begin() + width,
        [](std::uint16_t value) { return value == 0x4200u; }));
    CHECK(std::all_of(
        first_call.fp16_source.begin() + width,
        first_call.fp16_source.end(),
        [](std::uint16_t value) { return value == 0x3C00u; }));
    CHECK(std::all_of(
        output.begin(),
        output.begin() + width,
        [](float value) { return value == 3.0f; }));
    CHECK(std::all_of(
        output.begin() + width,
        output.end(),
        [](float value) { return value == 1.0f; }));

    const void* first_staging = first_call.source;
    const std::array<int, 1> second_ids{1};
    output.resize(width);
    g_recording.ResetCalls();
    GatherEmbedding(*api, embedding, second_ids, output);
    CHECK(g_recording.converts.size() == 1);
    CHECK(g_recording.converts.front().source == first_staging);
    CHECK(std::all_of(
        output.begin(),
        output.end(),
        [](float value) { return value == -2.0f; }));
}

void TestRmsNormUsesFp32AccumulationAndSharedEpsilon() {
    const std::array<float, 4> input{
        std::bit_cast<float>(0xBE8BBBACu),
        std::bit_cast<float>(0xBCCC9DE0u),
        std::bit_cast<float>(0xBFED682Fu),
        std::bit_cast<float>(0xC2CD01EDu)};
    const std::array<float, 4> scale{
        1.0f,
        1.0f,
        1.0f,
        1.0f};
    std::array<float, 4> output{};

    RmsNorm(
        input,
        scale,
        1,
        4,
        static_cast<float>(
            flm::phi4::constants::kRmsEpsilon),
        output);

    constexpr std::array<std::uint32_t, 4> expected{
        0xBBAE75DBu,
        0xB9FF781Fu,
        0xBD143451u,
        0xBFFFF509u};
    for (std::size_t index = 0; index < output.size(); ++index) {
        CHECK(std::bit_cast<std::uint32_t>(output[index]) ==
              expected[index]);
    }
}

void TestStageBf16UsesRneAndZerosOnlyInitialInputPrefixes(
    const std::shared_ptr<CorelibApi>& api) {
    const std::array<float, 4> normalized{
        std::bit_cast<float>(0x3F800000u),
        std::bit_cast<float>(0x3F808000u),
        std::bit_cast<float>(0x3F818000u),
        std::bit_cast<float>(0xBF800000u)};
    const std::array<float, 4> residual{
        std::bit_cast<float>(0x40000000u),
        std::bit_cast<float>(0xC0000000u),
        std::bit_cast<float>(0x00000000u),
        std::bit_cast<float>(0x3F000000u)};
    std::vector<std::uint16_t> hidden(8, kPoison);
    std::vector<std::uint16_t> residual_device(8, kPoison);

    // Task 7 has no production consumer for q/k/attention/skip-sum/
    // next-hidden padded tails. Task 8's dispatch test must poison those
    // actual persistent buffers and prove its live/helper-authorized
    // regions exclude stale values.

    g_recording.ResetCalls();
    StageBf16(*api, normalized, 2, 3, 2, hidden);
    StageBf16(*api, residual, 2, 3, 2, residual_device);

    CHECK(g_recording.converts.size() == 2);
    CHECK(g_recording.converts[0].count == 6);
    CHECK(g_recording.converts[1].count == 6);
    CHECK(
        g_recording.converts[0].source_type ==
        ryzenai_corelib_data_type_fp32);
    CHECK(
        g_recording.converts[0].destination_type ==
        ryzenai_corelib_data_type_bf16);
    CHECK(
        g_recording.converts[0].source ==
        g_recording.converts[1].source);

    constexpr std::array<std::uint32_t, 6> expected_hidden_source{
        0x3F800000u,
        0x3F808000u,
        0x3F818000u,
        0xBF800000u,
        0x00000000u,
        0x00000000u};
    constexpr std::array<std::uint32_t, 6> expected_residual_source{
        0x40000000u,
        0xC0000000u,
        0x00000000u,
        0x3F000000u,
        0x00000000u,
        0x00000000u};
    CHECK(
        std::equal(
            g_recording.converts[0].fp32_source_bits.begin(),
            g_recording.converts[0].fp32_source_bits.end(),
            expected_hidden_source.begin(),
            expected_hidden_source.end()));
    CHECK(
        std::equal(
            g_recording.converts[1].fp32_source_bits.begin(),
            g_recording.converts[1].fp32_source_bits.end(),
            expected_residual_source.begin(),
            expected_residual_source.end()));

    constexpr std::array<std::uint16_t, 8> expected_hidden{
        0x3F80u,
        0x3F80u,
        0x3F82u,
        0xBF80u,
        0x0000u,
        0x0000u,
        kPoison,
        kPoison};
    constexpr std::array<std::uint16_t, 8> expected_residual{
        0x4000u,
        0xC000u,
        0x0000u,
        0x3F00u,
        0x0000u,
        0x0000u,
        kPoison,
        kPoison};
    CHECK(std::equal(
        hidden.begin(),
        hidden.end(),
        expected_hidden.begin(),
        expected_hidden.end()));
    CHECK(std::equal(
        residual_device.begin(),
        residual_device.end(),
        expected_residual.begin(),
        expected_residual.end()));
}

std::uint16_t VValue(
    std::size_t row,
    std::size_t head,
    std::size_t column) {
    return static_cast<std::uint16_t>(
        row * 4096u + head * 256u + column);
}

void PopulateVSource(std::size_t capacity_rows) {
    constexpr std::size_t heads = 8;
    constexpr std::size_t width = 128;
    g_recording.v_source.assign(
        capacity_rows * heads * width,
        kPoison);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t column = 0; column < width; ++column) {
                const std::size_t index =
                    (row * heads + head) * width + column;
                g_recording.v_source[index] =
                    VValue(row, head, column);
            }
        }
    }
}

void CheckScatterWrites(
    std::int64_t rows,
    std::int64_t position) {
    constexpr std::size_t heads = 8;
    constexpr std::size_t width = 128;
    CHECK(g_recording.writes.size() == heads);
    for (std::size_t head = 0; head < heads; ++head) {
        const auto& write = g_recording.writes[head];
        const std::size_t expected_offset =
            ((head * 4096u) +
             static_cast<std::size_t>(position)) *
            width * sizeof(std::uint16_t);
        CHECK(write.tensor == g_recording.v_cache());
        CHECK(write.offset == expected_offset);
        CHECK(
            write.size ==
            static_cast<std::size_t>(rows) * width *
                sizeof(std::uint16_t));
        CHECK(
            write.values.size() ==
            static_cast<std::size_t>(rows) * width);
        for (std::size_t row = 0;
             row < static_cast<std::size_t>(rows);
             ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                CHECK(
                    write.values[row * width + column] ==
                    VValue(row, head, column));
            }
        }
        CHECK(std::none_of(
            write.values.begin(),
            write.values.end(),
            [](std::uint16_t value) { return value == kPoison; }));
    }
}

void TestScatterVReadsOnlyLiveRowsWithoutHiddenSynchronize(
    const std::shared_ptr<CorelibApi>& api) {
    constexpr std::size_t heads = 8;
    constexpr std::size_t width = 128;
    PopulateVSource(5);
    std::vector<std::uint16_t> staging;
    VScatterMetrics metrics{};

    g_recording.ResetCalls();
    ScatterV(
        *api,
        g_recording.v_tensor(),
        g_recording.v_cache(),
        3,
        17,
        staging,
        metrics);

    const std::size_t read_size =
        3u * heads * width * sizeof(std::uint16_t);
    CHECK(g_recording.reads.size() == 1);
    CHECK(g_recording.reads.front().tensor ==
          g_recording.v_tensor());
    CHECK(g_recording.reads.front().size == read_size);
    CHECK(g_recording.reads.front().offset == 0);
    CHECK(g_recording.reads.front().destination == staging.data());
    CHECK(g_recording.synchronize_calls == 0);
    CheckScatterWrites(3, 17);

    const auto* staging_begin = staging.data();
    const auto* staging_end = staging.data() + staging.size();
    for (const auto& write : g_recording.writes) {
        const auto* source =
            static_cast<const std::uint16_t*>(write.source);
        CHECK(source >= staging_begin);
        CHECK(
            source +
                write.size / sizeof(std::uint16_t) <=
            staging_end);
    }
    CHECK(metrics.read_calls == 1);
    CHECK(metrics.write_calls == 8);
    CHECK(metrics.bytes == 2u * read_size);
    CHECK(metrics.nanoseconds > 0);
    CHECK(std::all_of(
        g_recording.v_source.begin() + 3u * heads * width,
        g_recording.v_source.end(),
        [](std::uint16_t value) { return value == kPoison; }));

    const void* first_data = staging.data();
    const std::size_t first_capacity = staging.capacity();
    const std::uint64_t first_ns = metrics.nanoseconds;
    g_recording.ResetCalls();
    ScatterV(
        *api,
        g_recording.v_tensor(),
        g_recording.v_cache(),
        1,
        20,
        staging,
        metrics);
    CHECK(staging.data() == first_data);
    CHECK(staging.capacity() == first_capacity);
    CHECK(g_recording.reads.size() == 1);
    CHECK(
        g_recording.reads.front().size ==
        heads * width * sizeof(std::uint16_t));
    CHECK(g_recording.synchronize_calls == 0);
    CheckScatterWrites(1, 20);
    CHECK(metrics.read_calls == 2);
    CHECK(metrics.write_calls == 16);
    CHECK(
        metrics.bytes ==
        2u * read_size +
            2u * heads * width * sizeof(std::uint16_t));
    CHECK(metrics.nanoseconds > first_ns);
}

void TestArgmaxLowestChoosesLowestTokenOnTie() {
    std::vector<bf16> logits(12, bf16{0xBF80u, true});
    logits[2] = bf16{0x4000u, true};
    logits[7] = bf16{0x4040u, true};
    logits[9] = bf16{0x4040u, true};
    CHECK(ArgmaxLowest(logits) == 7);

    const std::array<bf16, 3> negative{
        bf16{0xC040u, true},
        bf16{0xC000u, true},
        bf16{0xC000u, true}};
    CHECK(ArgmaxLowest(negative) == 1);
}

void TestInvalidHostArgumentsFailBeforeCorelib(
    const std::shared_ptr<CorelibApi>& api) {
    constexpr std::size_t width = static_cast<std::size_t>(
        flm::phi4::constants::kHiddenSize);
    std::vector<std::uint16_t> embedding(width);
    const std::array<int, 1> invalid_id{1};
    std::vector<float> embedding_output(width);
    std::vector<std::uint16_t> staged(4);
    std::vector<std::uint16_t> scatter_staging;
    VScatterMetrics metrics{};

    g_recording.ResetCalls();
    CheckThrowsContains(
        [&] {
            GatherEmbedding(
                *api,
                embedding,
                invalid_id,
                embedding_output);
        },
        "token ID");
    CheckThrowsContains(
        [&] {
            StageBf16(
                *api,
                std::span<const float>{embedding_output}.first(2),
                2,
                1,
                1,
                staged);
        },
        "padded rows");
    CheckThrowsContains(
        [&] {
            ScatterV(
                *api,
                g_recording.v_tensor(),
                g_recording.v_cache(),
                2,
                4095,
                scatter_staging,
                metrics);
        },
        "cache");
    CheckThrowsContains(
        [&] {
            std::array<float, 1> rms_output{};
            RmsNorm(
                std::span<const float>{embedding_output}.first(1),
                std::span<const float>{embedding_output}.first(1),
                1,
                2,
                static_cast<float>(
                    flm::phi4::constants::kRmsEpsilon),
                rms_output);
        },
        "shape");
    CheckThrowsContains(
        [] {
            const std::span<const bf16> empty;
            (void)ArgmaxLowest(empty);
        },
        "empty");
    CHECK(g_recording.converts.empty());
    CHECK(g_recording.reads.empty());
    CHECK(g_recording.writes.empty());
}

}  // namespace

int main() {
    try {
        const auto api = ResolveRecordingCorelib();
        TestGatherEmbeddingUsesOneReusableContiguousConversion(api);
        TestRmsNormUsesFp32AccumulationAndSharedEpsilon();
        TestStageBf16UsesRneAndZerosOnlyInitialInputPrefixes(api);
        TestScatterVReadsOnlyLiveRowsWithoutHiddenSynchronize(api);
        TestArgmaxLowestChoosesLowestTokenOnTie();
        TestInvalidHostArgumentsFailBeforeCorelib(api);
        std::cout << "phi4 host operation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
