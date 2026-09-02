#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <models/phi4/phi4_corelib_shape_plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using flm::corelib::CorelibApi;
using flm::phi4::Phi4ShapePlan;
using flm::phi4::RowUse;

struct MatMulCall {
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
    std::uint32_t group_size;
};

struct SsMlpCall {
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
    std::uint32_t group_size;
};

struct AttentionCall {
    std::int64_t m;
    ryzenai_corelib_flat_mha_bf16_desc desc;
};

enum class HelperKind {
    None,
    MatMul,
    SsMlp,
    Attention
};

enum class FaultKind {
    None,
    Unsupported,
    InvalidRows,
    MutateK,
    MutateN
};

struct Fault {
    HelperKind helper = HelperKind::None;
    FaultKind kind = FaultKind::None;
    std::int64_t live_rows = 0;
    std::int64_t matmul_n = 0;
};

struct HelperState {
    std::vector<MatMulCall> matmul_calls;
    std::vector<SsMlpCall> ssmlp_calls;
    std::vector<AttentionCall> attention_calls;
    Fault fault;

    void Reset() {
        matmul_calls.clear();
        ssmlp_calls.clear();
        attention_calls.clear();
        fault = {};
    }

    std::size_t TotalCalls() const noexcept {
        return matmul_calls.size() + ssmlp_calls.size() +
               attention_calls.size();
    }
};

HelperState g_helpers;

bool Matches(
    HelperKind helper,
    std::int64_t live_rows,
    std::int64_t matmul_n = 0) {
    return g_helpers.fault.helper == helper &&
           g_helpers.fault.live_rows == live_rows &&
           (helper != HelperKind::MatMul ||
            g_helpers.fault.matmul_n == matmul_n);
}

std::int64_t QueryProjectionRows(std::int64_t live_rows) {
    if (live_rows == 1) {
        return 1;
    }
    if (live_rows <= 32) {
        return 32;
    }
    if (live_rows <= 512) {
        return 512;
    }
    return 4096;
}

std::int64_t KvProjectionRows(std::int64_t live_rows) {
    if (live_rows == 1) {
        return 1;
    }
    if (live_rows <= 64) {
        return 64;
    }
    if (live_rows <= 1024) {
        return 1024;
    }
    return 4160;
}

std::int64_t SsMlpRows(std::int64_t live_rows) {
    if (live_rows == 1) {
        return 1;
    }
    if (live_rows <= 128) {
        return 128;
    }
    if (live_rows <= 2048) {
        return 2048;
    }
    return 4608;
}

std::int64_t AttentionRows(std::int64_t live_rows) {
    if (live_rows == 1) {
        return 1;
    }
    if (live_rows <= 256) {
        return 256;
    }
    if (live_rows <= 3072) {
        return 3072;
    }
    return 4352;
}

ryzenai_corelib_status RecordingMatMulPadShape(
    std::int64_t* m,
    std::int64_t* k,
    std::int64_t* n,
    std::uint32_t group_size) {
    if (m == nullptr || k == nullptr || n == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }

    const MatMulCall call{*m, *k, *n, group_size};
    g_helpers.matmul_calls.push_back(call);
    if (Matches(HelperKind::MatMul, call.m, call.n)) {
        switch (g_helpers.fault.kind) {
            case FaultKind::Unsupported:
                return ryzenai_corelib_status_unsupported;
            case FaultKind::InvalidRows:
                *m = call.m - 1;
                return ryzenai_corelib_status_success;
            case FaultKind::MutateK:
                ++*k;
                return ryzenai_corelib_status_success;
            case FaultKind::MutateN:
                ++*n;
                return ryzenai_corelib_status_success;
            case FaultKind::None:
                break;
        }
    }

    if (call.k != 3072 || call.group_size != 128) {
        return ryzenai_corelib_status_bad_argument;
    }
    if (call.n == 3072) {
        *m = QueryProjectionRows(call.m);
    } else if (call.n == 1024) {
        *m = KvProjectionRows(call.m);
    } else if (call.n == 200064 && call.m == 1) {
        *m = 1;
    } else {
        return ryzenai_corelib_status_bad_argument;
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingSsMlpPadRows(
    std::int64_t* m,
    std::int64_t k,
    std::int64_t n,
    std::uint32_t group_size) {
    if (m == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }

    const SsMlpCall call{*m, k, n, group_size};
    g_helpers.ssmlp_calls.push_back(call);
    if (Matches(HelperKind::SsMlp, call.m)) {
        if (g_helpers.fault.kind == FaultKind::Unsupported) {
            return ryzenai_corelib_status_unsupported;
        }
        if (g_helpers.fault.kind == FaultKind::InvalidRows) {
            *m = call.m - 1;
            return ryzenai_corelib_status_success;
        }
    }

    if (call.k != 3072 || call.n != 8192 ||
        call.group_size != 128) {
        return ryzenai_corelib_status_bad_argument;
    }
    *m = SsMlpRows(call.m);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingAttentionPadRows(
    std::int64_t* m,
    const ryzenai_corelib_flat_mha_bf16_desc* desc) {
    if (m == nullptr || desc == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }

    const AttentionCall call{*m, *desc};
    g_helpers.attention_calls.push_back(call);
    if (Matches(HelperKind::Attention, call.m)) {
        if (g_helpers.fault.kind == FaultKind::Unsupported) {
            return ryzenai_corelib_status_unsupported;
        }
        if (g_helpers.fault.kind == FaultKind::InvalidRows) {
            *m = call.m - 1;
            return ryzenai_corelib_status_success;
        }
    }

    if (desc->num_heads != 24 || desc->kv_num_heads != 8 ||
        desc->head_size != 128 || desc->max_seq != 4096 ||
        desc->rope_dim != 96) {
        return ryzenai_corelib_status_bad_argument;
    }
    *m = AttentionRows(call.m);
    return ryzenai_corelib_status_success;
}

template <class Function>
void* FunctionAddress(Function function) {
    return reinterpret_cast<void*>(function);
}

std::shared_ptr<CorelibApi> ResolveRecordingCorelib() {
    auto resolver = flm::test::CompleteCorelibResolver();
    resolver["ryzenai_corelib_matmul_bf16_pad_shape"] =
        FunctionAddress(
            static_cast<
                decltype(&::ryzenai_corelib_matmul_bf16_pad_shape)>(
                &RecordingMatMulPadShape));
    resolver["ryzenai_corelib_ssmlp_bf16_pad_rows"] =
        FunctionAddress(
            static_cast<
                decltype(&::ryzenai_corelib_ssmlp_bf16_pad_rows)>(
                &RecordingSsMlpPadRows));
    resolver["ryzenai_corelib_flat_mha_bf16_pad_rows"] =
        FunctionAddress(
            static_cast<
                decltype(&::ryzenai_corelib_flat_mha_bf16_pad_rows)>(
                &RecordingAttentionPadRows));
    return CorelibApi::ResolveForTest(
        [resolver = std::move(resolver)](std::string_view name) mutable
            -> void* {
            const auto found = resolver.find(std::string(name));
            return found == resolver.end() ? nullptr : found->second;
        });
}

void CheckTransitions(
    const std::vector<std::pair<std::int64_t, std::int64_t>>& actual,
    std::initializer_list<
        std::pair<std::int64_t, std::int64_t>> expected) {
    const std::vector<std::pair<std::int64_t, std::int64_t>>
        expected_transitions(expected);
    CHECK(actual == expected_transitions);
}

template <class Function>
void CheckThrowsEquals(Function&& function, std::string_view expected) {
    try {
        function();
    } catch (const std::exception& error) {
        CHECK(std::string_view(error.what()) == expected);
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

void TestCompleteQueriesAndCachedTransitions(
    const std::shared_ptr<CorelibApi>& api) {
    g_helpers.Reset();
    const auto plan = Phi4ShapePlan::Build(api);

    CHECK(g_helpers.matmul_calls.size() == 8193);
    std::array<int, 4097> query_rows{};
    std::array<int, 4097> kv_rows{};
    int lm_head_calls = 0;
    for (const auto& call : g_helpers.matmul_calls) {
        CHECK(call.k == 3072);
        CHECK(call.group_size == 128);
        if (call.n == 3072) {
            CHECK(call.m >= 1 && call.m <= 4096);
            ++query_rows[static_cast<std::size_t>(call.m)];
        } else if (call.n == 1024) {
            CHECK(call.m >= 1 && call.m <= 4096);
            ++kv_rows[static_cast<std::size_t>(call.m)];
        } else {
            CHECK(call.n == 200064);
            CHECK(call.m == 1);
            ++lm_head_calls;
        }
    }
    for (std::int64_t row = 1; row <= 4096; ++row) {
        CHECK(query_rows[static_cast<std::size_t>(row)] == 1);
        CHECK(kv_rows[static_cast<std::size_t>(row)] == 1);
    }
    CHECK(lm_head_calls == 1);

    CHECK(g_helpers.ssmlp_calls.size() == 4096);
    CHECK(g_helpers.attention_calls.size() == 4096);
    for (std::int64_t row = 1; row <= 4096; ++row) {
        const auto index = static_cast<std::size_t>(row - 1);
        const auto& ssmlp = g_helpers.ssmlp_calls[index];
        CHECK(ssmlp.m == row);
        CHECK(ssmlp.k == 3072);
        CHECK(ssmlp.n == 8192);
        CHECK(ssmlp.group_size == 128);

        const auto& attention = g_helpers.attention_calls[index];
        CHECK(attention.m == row);
        CHECK(attention.desc.num_heads == 24);
        CHECK(attention.desc.kv_num_heads == 8);
        CHECK(attention.desc.head_size == 128);
        CHECK(attention.desc.max_seq == 4096);
        CHECK(attention.desc.rope_dim == 96);
    }

    const auto& desc = plan.attention_desc();
    CHECK(desc.num_heads == 24);
    CHECK(desc.kv_num_heads == 8);
    CHECK(desc.head_size == 128);
    CHECK(desc.max_seq == 4096);
    CHECK(desc.rope_dim == 96);

    CHECK(plan.capacities().layer_rows == 4608);
    CHECK(plan.capacities().lm_head_rows == 1);

    CheckTransitions(
        plan.Transitions(RowUse::QueryProjection),
        {{1, 1}, {2, 32}, {33, 512}, {513, 4096}});
    CheckTransitions(
        plan.Transitions(RowUse::OutputProjection),
        {{1, 1}, {2, 32}, {33, 512}, {513, 4096}});
    CheckTransitions(
        plan.Transitions(RowUse::KvProjection),
        {{1, 1}, {2, 64}, {65, 1024}, {1025, 4160}});
    CheckTransitions(
        plan.Transitions(RowUse::SsMlp),
        {{1, 1}, {2, 128}, {129, 2048}, {2049, 4608}});
    CheckTransitions(
        plan.Transitions(RowUse::Attention),
        {{1, 1}, {2, 256}, {257, 3072}, {3073, 4352}});
    CheckTransitions(
        plan.Transitions(RowUse::LmHead),
        {{1, 1}});

    constexpr std::array<RowUse, 6> uses{
        RowUse::QueryProjection,
        RowUse::KvProjection,
        RowUse::Attention,
        RowUse::OutputProjection,
        RowUse::SsMlp,
        RowUse::LmHead};
    for (const auto use : uses) {
        CHECK(plan.RowsFor(use, 1) == 1);
    }

    const std::size_t calls_after_build = g_helpers.TotalCalls();
    CHECK(plan.RowsFor(RowUse::QueryProjection, 2) == 32);
    CHECK(plan.RowsFor(RowUse::QueryProjection, 33) == 512);
    CHECK(plan.RowsFor(RowUse::KvProjection, 4096) == 4160);
    CHECK(plan.RowsFor(RowUse::Attention, 3072) == 3072);
    CHECK(plan.RowsFor(RowUse::SsMlp, 2049) == 4608);
    CHECK(plan.RowsFor(RowUse::LmHead, 1) == 1);
    CHECK(g_helpers.TotalCalls() == calls_after_build);
}

void TestMatMulDimensionMutationsRejectBuild(
    const std::shared_ptr<CorelibApi>& api) {
    struct Case {
        FaultKind kind;
        std::int64_t row;
        std::int64_t n;
        std::string_view message;
    };
    constexpr std::array<Case, 3> cases{{
        {FaultKind::MutateK,
         37,
         3072,
         "query/output projection MatMul K/N mismatch at live row 37: "
         "requested K=3072, N=3072; returned K=3073, N=3072"},
        {FaultKind::MutateN,
         93,
         1024,
         "key/value projection MatMul K/N mismatch at live row 93: "
         "requested K=3072, N=1024; returned K=3072, N=1025"},
        {FaultKind::MutateN,
         1,
         200064,
         "LM head MatMul K/N mismatch at live row 1: "
         "requested K=3072, N=200064; returned K=3072, N=200065"},
    }};

    for (const auto& test_case : cases) {
        g_helpers.Reset();
        g_helpers.fault = {
            HelperKind::MatMul,
            test_case.kind,
            test_case.row,
            test_case.n};
        CheckThrowsEquals(
            [&] {
                (void)Phi4ShapePlan::Build(api);
            },
            test_case.message);
    }
}

void TestUnsupportedRowsRejectBuild(
    const std::shared_ptr<CorelibApi>& api) {
    struct Case {
        HelperKind helper;
        std::int64_t row;
        std::int64_t n;
        std::string_view call;
    };
    constexpr std::array<Case, 3> cases{{
        {HelperKind::MatMul,
         41,
         3072,
         "ryzenai_corelib_matmul_bf16_pad_shape"},
        {HelperKind::SsMlp,
         43,
         0,
         "ryzenai_corelib_ssmlp_bf16_pad_rows"},
        {HelperKind::Attention,
         47,
         0,
         "ryzenai_corelib_flat_mha_bf16_pad_rows"},
    }};

    for (const auto& test_case : cases) {
        g_helpers.Reset();
        g_helpers.fault = {
            test_case.helper,
            FaultKind::Unsupported,
            test_case.row,
            test_case.n};
        CheckThrowsContains(
            [&] {
                (void)Phi4ShapePlan::Build(api);
            },
            test_case.call);
    }
}

void TestInvalidPaddedRowsRejectBuild(
    const std::shared_ptr<CorelibApi>& api) {
    struct Case {
        HelperKind helper;
        std::int64_t row;
        std::int64_t n;
        std::string_view context;
    };
    constexpr std::array<Case, 4> cases{{
        {HelperKind::MatMul, 53, 3072, "query/output projection"},
        {HelperKind::SsMlp, 59, 0, "SSMLP"},
        {HelperKind::Attention, 61, 0, "attention"},
        {HelperKind::MatMul, 1, 200064, "LM head"},
    }};

    for (const auto& test_case : cases) {
        g_helpers.Reset();
        g_helpers.fault = {
            test_case.helper,
            FaultKind::InvalidRows,
            test_case.row,
            test_case.n};
        CheckThrowsContains(
            [&] {
                (void)Phi4ShapePlan::Build(api);
            },
            test_case.context);
    }
}

void TestInvalidInputsRejectWithoutHelperCalls(
    const std::shared_ptr<CorelibApi>& api) {
    g_helpers.Reset();
    CheckThrowsContains(
        [] {
            (void)Phi4ShapePlan::Build(nullptr);
        },
        "CorelibApi");
    CHECK(g_helpers.TotalCalls() == 0);

    const auto plan = Phi4ShapePlan::Build(api);
    const auto calls_after_build = g_helpers.TotalCalls();
    CheckThrowsContains(
        [&] {
            (void)plan.RowsFor(RowUse::QueryProjection, 0);
        },
        "live rows");
    CheckThrowsContains(
        [&] {
            (void)plan.RowsFor(RowUse::Attention, 4097);
        },
        "live rows");
    CheckThrowsContains(
        [&] {
            (void)plan.RowsFor(RowUse::LmHead, 2);
        },
        "live rows");
    CheckThrowsContains(
        [&] {
            (void)plan.Transitions(static_cast<RowUse>(99));
        },
        "RowUse");
    CHECK(g_helpers.TotalCalls() == calls_after_build);
}

// Everything above drives Phi4ShapePlan through a synthetic padding grid,
// which is what lets it test the plan's own logic. This one does the
// opposite: it builds against the UNMODIFIED fake, whose pad helpers encode
// the grid measured from the real e5258d2 library.
//
// It exists because that grid was otherwise consumed only by
// test_real_corelib, which skips without a runtime directory. The two
// together form a chain -- test_real_corelib asserts fake == library, and
// this asserts fake == the numbers written down here -- so a regression in
// either end fails something that runs by default.
std::shared_ptr<CorelibApi> ResolveUnmodifiedFake() {
    auto resolver = flm::test::CompleteCorelibResolver();
    return CorelibApi::ResolveForTest(
        [resolver = std::move(resolver)](std::string_view name) mutable
            -> void* {
            const auto found = resolver.find(std::string(name));
            return found == resolver.end() ? nullptr : found->second;
        });
}

void TestFakeReproducesTheShippedKernelGrid() {
    auto api = ResolveUnmodifiedFake();
    const Phi4ShapePlan plan = Phi4ShapePlan::Build(api);

    const std::vector<std::pair<std::int64_t, std::int64_t>> expected{
        {1, 1},
        {2, 64},
        {65, 128},
        {129, 256},
        {257, 512},
        {513, 1024},
        {1025, 2048},
        {2049, 3072},
        {3073, 4096}};
    for (const RowUse use : {
             RowUse::QueryProjection,
             RowUse::KvProjection,
             RowUse::Attention,
             RowUse::OutputProjection,
             RowUse::SsMlp}) {
        CHECK(plan.Transitions(use) == expected);
    }
    CHECK(
        plan.Transitions(RowUse::LmHead) ==
        (std::vector<std::pair<std::int64_t, std::int64_t>>{{1, 1}}));
    CHECK(plan.capacities().layer_rows == 4096);
    CHECK(plan.capacities().lm_head_rows == 1);

    // Decode stays unpadded on every helper.
    for (const RowUse use : {
             RowUse::QueryProjection,
             RowUse::KvProjection,
             RowUse::Attention,
             RowUse::OutputProjection,
             RowUse::SsMlp,
             RowUse::LmHead}) {
        CHECK(plan.RowsFor(use, 1) == 1);
    }
    CHECK(plan.RowsFor(RowUse::SsMlp, 2) == 64);
    CHECK(plan.RowsFor(RowUse::Attention, 64) == 64);
    CHECK(plan.RowsFor(RowUse::QueryProjection, 65) == 128);
    CHECK(plan.RowsFor(RowUse::QueryProjection, 3072) == 3072);
    CHECK(plan.RowsFor(RowUse::QueryProjection, 4096) == 4096);
}

// The LM head ships M = 1 and M = 128 and refuses anything larger rather
// than rounding up. Phi4ShapePlan never asks for more, so this branch of
// the fake had no other caller -- and an unexercised branch is not a model
// of the library, it is dead code that happens to be written down.
void TestFakeRefusesOutOfGridLmHeadRows() {
    auto api = ResolveUnmodifiedFake();
    const auto& functions = api->functions();

    for (const auto& expectation : std::vector<
             std::pair<std::int64_t, std::int64_t>>{
             {1, 1},
             {2, 128},
             {128, 128}}) {
        std::int64_t m = expectation.first;
        std::int64_t k = 3072;
        std::int64_t n = 200064;
        CHECK(
            functions.matmul_pad_shape(&m, &k, &n, 128) ==
            ryzenai_corelib_status_success);
        CHECK(m == expectation.second);
        // K and N are never padded; MEM-5 rests on that.
        CHECK(k == 3072);
        CHECK(n == 200064);
    }

    for (const std::int64_t rows :
         {std::int64_t{129}, std::int64_t{256}, std::int64_t{4096}}) {
        std::int64_t m = rows;
        std::int64_t k = 3072;
        std::int64_t n = 200064;
        CHECK(
            functions.matmul_pad_shape(&m, &k, &n, 128) ==
            ryzenai_corelib_status_unsupported);
    }

    // The layer shapes have no such ceiling: 4096 is on their grid.
    std::int64_t m = 4096;
    std::int64_t k = 3072;
    std::int64_t n = 3072;
    CHECK(
        functions.matmul_pad_shape(&m, &k, &n, 128) ==
        ryzenai_corelib_status_success);
    CHECK(m == 4096);

    // And nothing beyond the grid is silently accepted anywhere.
    m = 4097;
    CHECK(
        functions.matmul_pad_shape(&m, &k, &n, 128) ==
        ryzenai_corelib_status_unsupported);
    m = 4097;
    CHECK(
        functions.ssmlp_pad_rows(&m, 3072, 8192, 128) ==
        ryzenai_corelib_status_unsupported);
    const ryzenai_corelib_flat_mha_bf16_desc desc{
        24,
        8,
        128,
        4096,
        96};
    m = 4097;
    CHECK(
        functions.flat_mha_pad_rows(&m, &desc) ==
        ryzenai_corelib_status_unsupported);
}

}  // namespace

int main() {
    try {
        auto api = ResolveRecordingCorelib();
        TestCompleteQueriesAndCachedTransitions(api);
        TestMatMulDimensionMutationsRejectBuild(api);
        TestUnsupportedRowsRejectBuild(api);
        TestInvalidPaddedRowsRejectBuild(api);
        TestInvalidInputsRejectWithoutHelperCalls(api);
        TestFakeReproducesTheShippedKernelGrid();
        TestFakeRefusesOutOfGridLmHeadRows();
        std::cout << "test_phi4_shape_plan: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
