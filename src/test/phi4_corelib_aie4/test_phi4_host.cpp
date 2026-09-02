#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <corelib/host_convert.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>
#include <models/phi4/phi4_corelib_host.hpp>

#include <windows.h>

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
using flm::phi4::StageFp32;
using flm::phi4::VScatterMetrics;
using bf16 = biovault::bfloat16_t;

constexpr std::uint16_t kPoison = 0xDEADu;

struct ReadCall {
    ryzenai_corelib_tensor_ptr tensor;
    ryzenai_corelib_data_type destination_type;
    void* destination;
    std::size_t count;
    std::size_t offset;
};

struct WriteCall {
    ryzenai_corelib_tensor_ptr tensor;
    ryzenai_corelib_data_type source_type;
    const void* source;
    std::size_t count;
    std::size_t offset;
    std::vector<std::uint16_t> values;
};

struct RecordingState {
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
        reads.clear();
        writes.clear();
        synchronize_calls = 0;
    }
};

RecordingState g_recording;

// `count` and `offset` are BF16 ELEMENTS of the V tensor, never bytes. A
// caller that still passes bytes asks for twice the elements and is
// rejected here rather than reading past the source.
ryzenai_corelib_status RecordingTensorRead(
    ryzenai_corelib_tensor_ptr tensor,
    ryzenai_corelib_data_type destination_type,
    void* destination,
    std::size_t count,
    std::size_t offset) {
    if (tensor != g_recording.v_tensor() || destination == nullptr ||
        destination_type != ryzenai_corelib_data_type_bf16 ||
        offset > g_recording.v_source.size() ||
        count > g_recording.v_source.size() - offset) {
        return ryzenai_corelib_status_bad_argument;
    }
    g_recording.reads.push_back(
        ReadCall{tensor, destination_type, destination, count, offset});
    std::memcpy(
        destination,
        g_recording.v_source.data() + offset,
        count * sizeof(std::uint16_t));

    const auto tick = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() == tick) {
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingTensorWrite(
    ryzenai_corelib_tensor_ptr tensor,
    ryzenai_corelib_data_type source_type,
    const void* source,
    std::size_t count,
    std::size_t offset) {
    constexpr std::size_t cache_elements =
        static_cast<std::size_t>(flm::phi4::constants::kKvHeadCount) *
        static_cast<std::size_t>(
            flm::phi4::constants::kMaxSequenceLength) *
        static_cast<std::size_t>(flm::phi4::constants::kHeadSize);
    if (tensor != g_recording.v_cache() || source == nullptr ||
        source_type != ryzenai_corelib_data_type_bf16 ||
        offset > cache_elements || count > cache_elements - offset) {
        return ryzenai_corelib_status_bad_argument;
    }
    const auto* values = static_cast<const std::uint16_t*>(source);
    g_recording.writes.push_back(WriteCall{
        tensor,
        source_type,
        source,
        count,
        offset,
        std::vector<std::uint16_t>(values, values + count)});
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

// The two host conversions design API-6 permits, and only those two.
// This one is lossless, so it is checked against exact values.
void TestWidenFp16IsExactAcrossTheFp16Range() {
    struct Case {
        std::uint16_t bits;
        float value;
    };
    constexpr Case kCases[]{
        {0x0000u, 0.0f},
        {0x8000u, -0.0f},
        {0x3C00u, 1.0f},
        {0xBC00u, -1.0f},
        {0x4000u, 2.0f},
        {0xC000u, -2.0f},
        {0x4200u, 3.0f},
        {0x3800u, 0.5f},
        {0x0001u, 5.9604645e-8f},          // smallest subnormal
        {0x03FFu, 6.0975552e-5f},          // largest subnormal
        {0x0400u, 6.1035156e-5f},          // smallest normal
        {0x7BFFu, 65504.0f},               // largest finite
    };
    for (const auto& item : kCases) {
        CHECK(flm::corelib::WidenFp16(item.bits) == item.value);
    }
    CHECK(std::isinf(flm::corelib::WidenFp16(0x7C00u)));
    CHECK(flm::corelib::WidenFp16(0x7C00u) > 0.0f);
    CHECK(std::isinf(flm::corelib::WidenFp16(0xFC00u)));
    CHECK(std::isnan(flm::corelib::WidenFp16(0x7E00u)));

    // Every representable FP16 round-trips through float, which is what
    // "lossless, so it has no rounding policy" means in practice.
    for (std::uint32_t bits = 0; bits <= 0xFFFFu; ++bits) {
        const auto narrow = static_cast<std::uint16_t>(bits);
        const float widened = flm::corelib::WidenFp16(narrow);
        if (std::isnan(widened)) {
            continue;
        }
        const auto exponent = (narrow >> 10) & 0x1Fu;
        const auto mantissa = narrow & 0x3FFu;
        const bool negative = (narrow & 0x8000u) != 0;
        double expected = 0.0;
        if (exponent == 0u) {
            expected = std::ldexp(static_cast<double>(mantissa), -24);
        } else if (exponent == 0x1Fu) {
            continue;
        } else {
            expected = std::ldexp(
                1.0 + static_cast<double>(mantissa) / 1024.0,
                static_cast<int>(exponent) - 15);
        }
        if (negative) {
            expected = -expected;
        }
        CHECK(static_cast<double>(widened) == expected);
    }
}

// The reference driver's to_bf16, transcribed. FastFlow's helper must
// agree with it bit for bit; there is no second BF16 rounding policy.
std::uint16_t ReferenceToBf16(std::uint32_t bits) {
    const std::uint64_t rounded =
        static_cast<std::uint64_t>(bits) + 0x7FFFull +
        ((static_cast<std::uint64_t>(bits) >> 16) & 1ull);
    return static_cast<std::uint16_t>(rounded >> 16);
}

void TestNarrowFp32ToBf16MatchesTheReferenceDriverBitForBit() {
    constexpr std::uint32_t kExact[]{
        0x00000000u,
        0x80000000u,
        0x3F800000u,
        0x3F808000u,  // exact tie, rounds to even
        0x3F818000u,  // exact tie, rounds up to even
        0xBF800000u,
        0x40000000u,
        0xC0000000u,
        0x3F000000u,
        0x322BCC77u,  // 1e-8f
        0x3727C5ACu,  // 1e-5f, the Phi-4 RMS epsilon
        0x7F800000u,
        0xFF800000u,
        0x7FC00000u,
        0xFFFFFFFFu,  // the wrap the driver hides in uint64
        0xFFFF8000u,
    };
    for (const std::uint32_t bits : kExact) {
        CHECK(
            flm::corelib::NarrowFp32ToBf16(std::bit_cast<float>(bits)) ==
            ReferenceToBf16(bits));
    }

    // A dense sweep of the low mantissa bits, where the tie-breaking
    // actually differs between truncation and round-to-nearest-even.
    for (std::uint32_t low = 0; low < 0x20000u; ++low) {
        const std::uint32_t bits = 0x3F800000u + low;
        CHECK(
            flm::corelib::NarrowFp32ToBf16(std::bit_cast<float>(bits)) ==
            ReferenceToBf16(bits));
    }
    // And a stride across the whole exponent range.
    for (std::uint64_t bits = 0; bits <= 0xFFFFFFFFull; bits += 65413ull) {
        const auto value = static_cast<std::uint32_t>(bits);
        CHECK(
            flm::corelib::NarrowFp32ToBf16(std::bit_cast<float>(value)) ==
            ReferenceToBf16(value));
    }

    CHECK(
        flm::corelib::NarrowFp32ToBf16(
            static_cast<float>(flm::phi4::constants::kRmsEpsilon)) ==
        ReferenceToBf16(std::bit_cast<std::uint32_t>(
            static_cast<float>(flm::phi4::constants::kRmsEpsilon))));
}

void TestGatherEmbeddingWidensWithoutCorelib() {
    constexpr std::size_t width = static_cast<std::size_t>(
        flm::phi4::constants::kHiddenSize);
    std::vector<std::uint16_t> embedding(3u * width);
    std::fill_n(embedding.begin(), width, 0x3C00u);
    std::fill_n(embedding.begin() + width, width, 0xC000u);
    std::fill_n(embedding.begin() + 2u * width, width, 0x4200u);

    const std::array<int, 2> ids{2, 0};
    std::vector<float> output(ids.size() * width);
    g_recording.ResetCalls();
    GatherEmbedding(embedding, ids, output);

    CHECK(std::all_of(
        output.begin(),
        output.begin() + width,
        [](float value) { return value == 3.0f; }));
    CHECK(std::all_of(
        output.begin() + width,
        output.end(),
        [](float value) { return value == 1.0f; }));

    const std::array<int, 1> second_ids{1};
    output.assign(width, 0.0f);
    GatherEmbedding(embedding, second_ids, output);
    CHECK(std::all_of(
        output.begin(),
        output.end(),
        [](float value) { return value == -2.0f; }));

    // No corelib call is made at all: the widening is FastFlow's own.
    CHECK(g_recording.reads.empty());
    CHECK(g_recording.writes.empty());
}

// The embedding table is a read-only file mapping. A vectorized widening
// reads up to 14 bytes past its source, which faults instead of returning
// garbage -- so the gather must touch nothing beyond the last row.
void TestGatherEmbeddingStopsAtAGuardPage() {
    constexpr std::size_t width = static_cast<std::size_t>(
        flm::phi4::constants::kHiddenSize);
    constexpr std::size_t rows = 4;
    constexpr std::size_t table_bytes = rows * width * sizeof(std::uint16_t);
    static_assert(table_bytes % 4096u == 0u);

    auto* base = static_cast<std::byte*>(VirtualAlloc(
        nullptr,
        table_bytes + 4096u,
        MEM_RESERVE,
        PAGE_NOACCESS));
    if (base == nullptr) {
        throw std::runtime_error("failed to reserve guard-page range");
    }
    if (VirtualAlloc(base, table_bytes, MEM_COMMIT, PAGE_READWRITE) ==
        nullptr) {
        VirtualFree(base, 0, MEM_RELEASE);
        throw std::runtime_error("failed to commit guard-page table");
    }

    auto* table = reinterpret_cast<std::uint16_t*>(base);
    for (std::size_t index = 0; index < rows * width; ++index) {
        table[index] = 0x3C00u;
    }
    const std::span<const std::uint16_t> embedding(table, rows * width);

    // The last row ends exactly at the boundary of the reserved,
    // never-committed page that follows.
    const std::array<int, 1> ids{static_cast<int>(rows) - 1};
    std::vector<float> output(width);
    GatherEmbedding(embedding, ids, output);
    const bool all_one = std::all_of(
        output.begin(),
        output.end(),
        [](float value) { return value == 1.0f; });

    VirtualFree(base, 0, MEM_RELEASE);
    CHECK(all_one);
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

void TestStageFp32ZerosOnlyInitialInputPrefixes() {
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
    constexpr float kPoisonFloat = -12345.0f;
    std::vector<float> hidden(8, kPoisonFloat);
    std::vector<float> residual_device(8, kPoisonFloat);

    // Task 7 has no production consumer for q/k/attention/skip-sum/
    // next-hidden padded tails. Task 8's dispatch test must poison those
    // actual persistent buffers and prove its live/helper-authorized
    // regions exclude stale values.

    g_recording.ResetCalls();
    StageFp32(normalized, 2, 3, 2, hidden);
    StageFp32(residual, 2, 3, 2, residual_device);

    // Design 10.2: the host stays in FP32 and never rounds to BF16, so
    // the staged bits are the input bits, unchanged.
    const std::array<float, 8> expected_hidden{
        normalized[0],
        normalized[1],
        normalized[2],
        normalized[3],
        0.0f,
        0.0f,
        kPoisonFloat,
        kPoisonFloat};
    const std::array<float, 8> expected_residual{
        residual[0],
        residual[1],
        residual[2],
        residual[3],
        0.0f,
        0.0f,
        kPoisonFloat,
        kPoisonFloat};
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        CHECK(
            std::bit_cast<std::uint32_t>(hidden[index]) ==
            std::bit_cast<std::uint32_t>(expected_hidden[index]));
        CHECK(
            std::bit_cast<std::uint32_t>(residual_device[index]) ==
            std::bit_cast<std::uint32_t>(expected_residual[index]));
    }

    // Nothing crossed the corelib boundary: the FP32-to-BF16 narrowing on
    // this path is corelib's, inside tensor_write.
    CHECK(g_recording.reads.empty());
    CHECK(g_recording.writes.empty());
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
        // API-7: elements of the cache's BF16 dtype, not bytes. The
        // previous spelling multiplied both by sizeof(uint16_t) and was
        // wrong by exactly 2x.
        const std::size_t expected_offset =
            ((head * 4096u) +
             static_cast<std::size_t>(position)) *
            width;
        CHECK(write.tensor == g_recording.v_cache());
        CHECK(write.source_type == ryzenai_corelib_data_type_bf16);
        CHECK(write.offset == expected_offset);
        CHECK(
            write.count == static_cast<std::size_t>(rows) * width);
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

    const std::size_t read_elements = 3u * heads * width;
    CHECK(g_recording.reads.size() == 1);
    CHECK(g_recording.reads.front().tensor ==
          g_recording.v_tensor());
    CHECK(
        g_recording.reads.front().destination_type ==
        ryzenai_corelib_data_type_bf16);
    CHECK(g_recording.reads.front().count == read_elements);
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
        CHECK(source + write.count <= staging_end);
    }
    CHECK(metrics.read_calls == 1);
    CHECK(metrics.write_calls == 8);
    // Metrics stay in bytes; only the transfer arguments are elements.
    CHECK(
        metrics.bytes ==
        2u * read_elements * sizeof(std::uint16_t));
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
    CHECK(g_recording.reads.front().count == heads * width);
    CHECK(g_recording.synchronize_calls == 0);
    CheckScatterWrites(1, 20);
    CHECK(metrics.read_calls == 2);
    CHECK(metrics.write_calls == 16);
    CHECK(
        metrics.bytes ==
        (2u * read_elements + 2u * heads * width) *
            sizeof(std::uint16_t));
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
    std::vector<float> staged(4);
    std::vector<std::uint16_t> scatter_staging;
    VScatterMetrics metrics{};

    g_recording.ResetCalls();
    CheckThrowsContains(
        [&] {
            GatherEmbedding(embedding, invalid_id, embedding_output);
        },
        "token ID");
    CheckThrowsContains(
        [&] {
            StageFp32(
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
    CHECK(g_recording.reads.empty());
    CHECK(g_recording.writes.empty());
}

}  // namespace

int main() {
    try {
        const auto api = ResolveRecordingCorelib();
        TestWidenFp16IsExactAcrossTheFp16Range();
        TestNarrowFp32ToBf16MatchesTheReferenceDriverBitForBit();
        TestGatherEmbeddingWidensWithoutCorelib();
        TestGatherEmbeddingStopsAtAGuardPage();
        TestRmsNormUsesFp32AccumulationAndSharedEpsilon();
        TestStageFp32ZerosOnlyInitialInputPrefixes();
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
