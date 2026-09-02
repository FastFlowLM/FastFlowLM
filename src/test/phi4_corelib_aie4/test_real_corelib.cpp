// Validates FastFlow against the REAL ryzenai_corelib, off-hardware.
//
// Everything else in this suite runs against fake_ryzenai_corelib, which is
// FastFlow's own code -- so it validates FastFlow against FastFlow's model
// of corelib, not against corelib. That model stayed green through a full
// ABI break, which is exactly the gap this file closes.
//
// The header documents three things as needing no NPU, and those are what
// runs here:
//
//   1. selftest_dependencies -- "Allocates host memory only, no NPU."
//   2. matmul_bf16_pad_shape / ssmlp_bf16_pad_rows / flat_mha_bf16_pad_rows
//      -- "Needs no NPU: it is a lookup over the shipped kernel set."
//   3. has_device_context -- recorded, not asserted: the development box
//      has an NPU but is not the AIE4 target.
//
// The shape-plan check is the load-bearing one. The whole allocation
// strategy rests on MatMul padded K/N equalling logical K/N, and until now
// that was only "confirmed" against a fake that returns whatever FastFlow
// expects.

#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <corelib/corelib_api.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>
#include <models/phi4/phi4_corelib_shape_plan.hpp>

#include <windows.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace constants = flm::phi4::constants;

using flm::corelib::CorelibApi;
using flm::phi4::Phi4ShapePlan;
using flm::phi4::RowUse;

#if !defined(FLM_REAL_CORELIB_RUNTIME_DIR)
#define FLM_REAL_CORELIB_RUNTIME_DIR ""
#endif

// Semicolon-separated directories holding parts of corelib's dependency
// closure that do not sit beside the DLL. On the dev box that is the conda
// prefix the build linked protobuf from; Task 12 is where the closure
// becomes self-contained.
#if !defined(FLM_REAL_CORELIB_EXTRA_DLL_DIRS)
#define FLM_REAL_CORELIB_EXTRA_DLL_DIRS ""
#endif

// CorelibApi::Load uses LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which honours
// directories added here and deliberately ignores PATH.
void AddExtraDllDirectories(std::string_view directories) {
    std::size_t start = 0;
    while (start <= directories.size()) {
        const std::size_t end = directories.find(';', start);
        const std::string_view entry = directories.substr(
            start,
            end == std::string_view::npos ? std::string_view::npos
                                          : end - start);
        if (!entry.empty()) {
            const std::filesystem::path directory(entry);
            if (!std::filesystem::exists(directory)) {
                throw std::runtime_error(
                    "extra corelib DLL directory does not exist: " +
                    directory.string());
            }
            if (AddDllDirectory(directory.c_str()) == nullptr) {
                throw std::runtime_error(
                    "AddDllDirectory failed for " + directory.string());
            }
            std::cout << "added DLL directory " << directory.string()
                      << '\n';
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
}

struct NamedRowUse {
    RowUse use;
    std::string_view name;
};

constexpr std::array<NamedRowUse, 6> kRowUses{{
    {RowUse::QueryProjection, "query_projection"},
    {RowUse::KvProjection, "kv_projection"},
    {RowUse::Attention, "attention"},
    {RowUse::OutputProjection, "output_projection"},
    {RowUse::SsMlp, "ssmlp"},
    {RowUse::LmHead, "lm_head"},
}};

// The three MatMul shapes a Phi-4 forward pass dispatches.
struct MatMulDescriptor {
    std::int64_t k;
    std::int64_t n;
    std::string_view name;
    // The LM head is only ever dispatched at one row (design 10.5), and
    // the real kernel set has no padded M for it beyond that -- asking is
    // an error, not a larger answer. Phi4ShapePlan queries it at 1 only.
    bool multi_row;
};

constexpr std::array<MatMulDescriptor, 3> kMatMulDescriptors{{
    {constants::kHiddenSize, constants::kQueryDimension, "q/o_proj", true},
    {constants::kHiddenSize, constants::kKvDimension, "k/v_proj", true},
    {constants::kHiddenSize,
     constants::kVocabularySize,
     "lm_head",
     false},
}};

std::shared_ptr<CorelibApi> ResolveFakeCorelib() {
    auto resolver = flm::test::CompleteCorelibResolver();
    return CorelibApi::ResolveForTest(
        [resolver = std::move(resolver)](std::string_view name) mutable
            -> void* {
            const auto found = resolver.find(std::string(name));
            return found == resolver.end() ? nullptr : found->second;
        });
}

void PrintTransitions(
    std::string_view label,
    const std::vector<std::pair<std::int64_t, std::int64_t>>&
        transitions) {
    std::cout << "  " << label << ": " << transitions.size()
              << " transitions";
    const std::size_t shown =
        transitions.size() < 12u ? transitions.size() : 12u;
    for (std::size_t index = 0; index < shown; ++index) {
        std::cout << (index == 0 ? " [" : " ") << transitions[index].first
                  << "->" << transitions[index].second;
    }
    if (shown != 0) {
        std::cout << (shown < transitions.size() ? " ...]" : "]");
    }
    std::cout << " last=" << transitions.back().first << "->"
              << transitions.back().second << '\n';
}

// 1. Dependency self-test, against the real DynamicDispatch and RyzenMM.
void CheckVersionAndSelftest(const std::shared_ptr<CorelibApi>& api) {
    const auto compiled = flm::corelib::CompiledCorelibVersion();
    const auto& runtime = api->runtime_version();
    std::cout << "real corelib version: "
              << flm::corelib::FormatCorelibVersion(runtime)
              << " (compiled against "
              << flm::corelib::FormatCorelibVersion(compiled) << ")\n";
    // Loading at all means the API-5 gate passed; restate it so a future
    // relaxation of the gate cannot make this file vacuous.
    CHECK(flm::corelib::IsCorelibVersionCompatible(compiled, runtime));

    api->Check(
        api->functions().selftest_dependencies(),
        "ryzenai_corelib_selftest_dependencies");
    std::cout << "selftest_dependencies: ok\n";
}

// 3. Missing-device path, recorded rather than asserted.
void RecordDeviceContext(const std::shared_ptr<CorelibApi>& api) {
    const bool has_context = api->functions().has_device_context();
    std::cout << "has_device_context: "
              << (has_context ? "true" : "false")
              << " (recorded, not asserted: this box is not the AIE4 "
                 "target)\n";
    if (!has_context) {
        std::cout
            << "  note: tensors, weights and dispatch will fail with "
               "unsupported here; padding and packing still work.\n";
    }
}

// 2. Shape plan against the real kernel set.
void CheckRealShapePlan(const std::shared_ptr<CorelibApi>& api) {
    // MEM-5 and the Task 5 capacity model rest on this: padded K/N must
    // equal logical K/N for every Phi-4 MatMul, at every live row class.
    // Phi4ShapePlan::Build enforces it over 1..4096 and throws by name if
    // it ever fails, so building it IS the assertion. Restate it directly
    // at the boundaries so the check does not depend on Build's internals.
    for (const auto& descriptor : kMatMulDescriptors) {
        for (const std::int64_t rows :
             {std::int64_t{1},
              std::int64_t{2},
              std::int64_t{63},
              std::int64_t{64},
              std::int64_t{65},
              std::int64_t{2048},
              std::int64_t{4095},
              std::int64_t{4096}}) {
            if (rows != 1 && !descriptor.multi_row) {
                continue;
            }
            std::int64_t m = rows;
            std::int64_t k = descriptor.k;
            std::int64_t n = descriptor.n;
            api->Check(
                api->functions().matmul_pad_shape(
                    &m,
                    &k,
                    &n,
                    constants::kGroupSize),
                "ryzenai_corelib_matmul_bf16_pad_shape");
            if (k != descriptor.k || n != descriptor.n) {
                throw std::runtime_error(
                    "real corelib padded " + std::string(descriptor.name) +
                    " K/N at rows " + std::to_string(rows) + ": K " +
                    std::to_string(descriptor.k) + "->" +
                    std::to_string(k) + ", N " +
                    std::to_string(descriptor.n) + "->" +
                    std::to_string(n) +
                    "; MEM-5 and the capacity model assume K/N are "
                    "unchanged");
            }
            CHECK(m >= rows);
        }
    }
    std::cout << "matmul padded K/N equals logical K/N for all three "
                 "Phi-4 descriptors\n";

    // The LM head ships M = 1 and M = 128 and NOTHING ABOVE, and asking for
    // more is an error rather than a larger answer. This is the constraint
    // that makes Phi4ShapePlan correct: it queries the LM head at m = 1
    // only, and RowsFor(LmHead, n>1) throws. If the real library ever
    // started rounding an out-of-grid M up instead of refusing, that design
    // would be resting on a property the library no longer has -- so assert
    // the refusal here rather than inferring it from the code that avoids
    // it.
    for (const std::int64_t rows :
         {std::int64_t{1}, std::int64_t{2}, std::int64_t{128}}) {
        std::int64_t m = rows;
        std::int64_t k = constants::kHiddenSize;
        std::int64_t n = constants::kVocabularySize;
        api->Check(
            api->functions().matmul_pad_shape(
                &m,
                &k,
                &n,
                constants::kGroupSize),
            "ryzenai_corelib_matmul_bf16_pad_shape");
        CHECK(m == (rows == 1 ? 1 : 128));
    }
    for (const std::int64_t rows :
         {std::int64_t{129},
          std::int64_t{256},
          std::int64_t{2048},
          std::int64_t{4096}}) {
        std::int64_t m = rows;
        std::int64_t k = constants::kHiddenSize;
        std::int64_t n = constants::kVocabularySize;
        const auto status = api->functions().matmul_pad_shape(
            &m,
            &k,
            &n,
            constants::kGroupSize);
        if (status == ryzenai_corelib_status_success) {
            throw std::runtime_error(
                "real corelib accepted LM-head rows " +
                std::to_string(rows) + " and padded to " +
                std::to_string(m) +
                "; Phi4ShapePlan assumes the LM head is single-row and "
                "RowsFor(LmHead) throws above 1, which is only safe while "
                "the library refuses");
        }
    }
    std::cout << "LM head pads 1->1 and 2..128->128, and refuses every M "
                 "above 128\n";

    // rows = 1 must stay 1 on every helper: decode allocates a single row
    // and the header says flat MHA pads its KV window instead.
    for (const auto& descriptor : kMatMulDescriptors) {
        std::int64_t m = 1;
        std::int64_t k = descriptor.k;
        std::int64_t n = descriptor.n;
        api->Check(
            api->functions().matmul_pad_shape(
                &m,
                &k,
                &n,
                constants::kGroupSize),
            "ryzenai_corelib_matmul_bf16_pad_shape");
        std::cout << "  matmul " << descriptor.name << " rows 1 -> " << m
                  << '\n';
        CHECK(m == 1);
    }
    {
        std::int64_t m = 1;
        api->Check(
            api->functions().ssmlp_pad_rows(
                &m,
                constants::kHiddenSize,
                constants::kIntermediateSize,
                constants::kGroupSize),
            "ryzenai_corelib_ssmlp_bf16_pad_rows");
        std::cout << "  ssmlp rows 1 -> " << m << '\n';
        CHECK(m == 1);
    }
    {
        const ryzenai_corelib_flat_mha_bf16_desc desc{
            constants::kQueryHeadCount,
            constants::kKvHeadCount,
            constants::kHeadSize,
            constants::kMaxSequenceLength,
            constants::kRopeDimension};
        std::int64_t m = 1;
        api->Check(
            api->functions().flat_mha_pad_rows(&m, &desc),
            "ryzenai_corelib_flat_mha_bf16_pad_rows");
        std::cout << "  flat_mha rows 1 -> " << m << '\n';
        CHECK(m == 1);
    }

    const Phi4ShapePlan real_plan = Phi4ShapePlan::Build(api);
    std::cout << "real Phi4ShapePlan over rows 1..4096:\n";
    for (const auto& row_use : kRowUses) {
        PrintTransitions(
            row_use.name,
            real_plan.Transitions(row_use.use));
    }
    std::cout << "  capacities: layer_rows="
              << real_plan.capacities().layer_rows
              << " lm_head_rows=" << real_plan.capacities().lm_head_rows
              << '\n';
    CHECK(real_plan.capacities().layer_rows >= constants::kMaxSequenceLength);
    CHECK(real_plan.capacities().lm_head_rows == 1);
    CHECK(real_plan.RowsFor(RowUse::LmHead, 1) == 1);

    // The fake must reproduce the real library's answers exactly. A fake
    // that cannot is not a test double, it is a second implementation of
    // the same guess -- and every other file in this suite trusts it.
    const Phi4ShapePlan fake_plan =
        Phi4ShapePlan::Build(ResolveFakeCorelib());
    for (const auto& row_use : kRowUses) {
        const auto& real_transitions = real_plan.Transitions(row_use.use);
        const auto& fake_transitions = fake_plan.Transitions(row_use.use);
        if (real_transitions != fake_transitions) {
            std::cout << "  DIVERGENCE " << row_use.name
                      << ": real has " << real_transitions.size()
                      << " transitions, fake has "
                      << fake_transitions.size() << '\n';
            PrintTransitions("  real", real_transitions);
            PrintTransitions("  fake", fake_transitions);
            throw std::runtime_error(
                "fake_ryzenai_corelib disagrees with the real shipped "
                "kernel set for " + std::string(row_use.name) +
                "; fix the fake to match, then re-run the host suite");
        }
    }
    CHECK(
        fake_plan.capacities().layer_rows ==
        real_plan.capacities().layer_rows);
    CHECK(
        fake_plan.capacities().lm_head_rows ==
        real_plan.capacities().lm_head_rows);
    std::cout << "fake transition lists match the real ones\n";

    // The engine only ever reads RowsFor(), so the property it depends on
    // is that padding is monotonic and never below the live rows -- which
    // is what makes a single peak allocation safe.
    for (const auto& row_use : kRowUses) {
        if (row_use.use == RowUse::LmHead) {
            continue;
        }
        std::int64_t previous = 0;
        for (std::int64_t rows = 1;
             rows <= constants::kMaxSequenceLength;
             ++rows) {
            const std::int64_t padded =
                real_plan.RowsFor(row_use.use, rows);
            CHECK(padded >= rows);
            CHECK(padded >= previous);
            CHECK(padded <= real_plan.capacities().layer_rows);
            previous = padded;
        }
    }
    std::cout << "real padding is monotonic, never below live rows, and "
                 "never above the planned capacity\n";
}

}  // namespace

// CTest's SKIP_RETURN_CODE. Returning 0 here would report Passed, and a
// green-and-inert check is worse than an absent one because it reads as
// coverage.
constexpr int kCTestSkipReturnCode = 77;

int main() {
    const std::string runtime_dir(FLM_REAL_CORELIB_RUNTIME_DIR);
    if (runtime_dir.empty()) {
        std::cout
            << "test_real_corelib: SKIPPED -- configure with "
               "-DRYZENAI_CORELIB_RUNTIME_DIR=<dir containing "
               "ryzenai_corelib.dll> to run it.\n";
        return kCTestSkipReturnCode;
    }

    try {
        const std::filesystem::path library =
            std::filesystem::absolute(
                std::filesystem::path(runtime_dir) /
                "ryzenai_corelib.dll")
                .lexically_normal();
        if (!std::filesystem::exists(library)) {
            // Configured but absent is a failure, not a skip: a silently
            // skipped ABI check is how a stale DLL stays hidden.
            throw std::runtime_error(
                "RYZENAI_CORELIB_RUNTIME_DIR is set but " +
                library.string() + " does not exist");
        }
        AddExtraDllDirectories(FLM_REAL_CORELIB_EXTRA_DLL_DIRS);
        std::cout << "loading " << library.string() << '\n';

        auto api = CorelibApi::Load(library);
        CheckVersionAndSelftest(api);
        RecordDeviceContext(api);
        CheckRealShapePlan(api);

        api->functions().cleanup();
        std::cout << "test_real_corelib: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_real_corelib: FAIL: " << error.what() << '\n';
        return 1;
    }
}
