// Task 12 Steps 1 and 5: the real corelib, on a real AIE4 device.
//
// test_real_corelib covers the entry points the header documents as needing no
// NPU. Everything here needs one. `has_device_context()` is ASSERTED rather
// than recorded, a Stream and a DeviceTensor are really created on the device,
// and the RoPE upload is a real `tensor_write` rather than a fake that records
// its arguments.
//
// The RoPE case is the one that covers FastFlow rather than corelib. Corelib
// `e5258d2` removed `convert_strided`, so the [4096, 48] slice out of a wider
// table is now FastFlow's own bounds-checked loop over a read-only file
// mapping. The fixture ends the last source row immediately before an
// inaccessible page, so an off-by-one in that loop faults instead of silently
// reading a neighbouring page -- and the gathered rows are then pushed through
// one real FP16-source write into a real FP32 device tensor, which is the pair
// of steps the product actually performs at load.

#include "phi4_package_fixture.hpp"
#include "test_support.hpp"

#include <corelib/corelib_api.hpp>
#include <corelib/corelib_object.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>
#include <models/phi4/phi4_corelib_manifest.hpp>
#include <models/phi4/phi4_corelib_shape_plan.hpp>

#include <windows.h>

#include <algorithm>
#include <array>
#include <cstddef>
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
namespace fixture = flm::test::phi4fixture;

using flm::corelib::CorelibApi;
using flm::phi4::Phi4Package;
using flm::phi4::Phi4ShapePlan;
using flm::phi4::RowUse;

#if !defined(FLM_REAL_CORELIB_RUNTIME_DIR)
#define FLM_REAL_CORELIB_RUNTIME_DIR ""
#endif

#if !defined(FLM_REAL_CORELIB_EXTRA_DLL_DIRS)
#define FLM_REAL_CORELIB_EXTRA_DLL_DIRS ""
#endif

// CorelibApi::Load uses LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which honours
// directories added here and deliberately ignores PATH. Design CLOSURE-2: a
// pass that depended on PATH would certify the build machine, not the closure.
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

constexpr std::array<NamedRowUse, 5> kMultiRowUses{{
    {RowUse::QueryProjection, "query_projection"},
    {RowUse::KvProjection, "kv_projection"},
    {RowUse::Attention, "attention"},
    {RowUse::OutputProjection, "output_projection"},
    {RowUse::SsMlp, "ssmlp"},
}};

// 1a. Version identity. API-5 requires an exact major.minor.patch match while
// the compiled-against major is 0, and `CorelibApi::Load` enforces it before
// resolving any other symbol -- so loading at all means the gate passed.
// Restating it here keeps this file from going vacuous if the gate is ever
// relaxed, and prints both sides so a mismatch is diagnosable from the log.
void CheckVersionIdentity(const std::shared_ptr<CorelibApi>& api) {
    const auto compiled = flm::corelib::CompiledCorelibVersion();
    const auto& runtime = api->runtime_version();
    std::cout << "runtime corelib version "
              << flm::corelib::FormatCorelibVersion(runtime)
              << ", compiled against "
              << flm::corelib::FormatCorelibVersion(compiled) << '\n';
    CHECK(runtime.major == compiled.major);
    CHECK(runtime.minor == compiled.minor);
    CHECK(runtime.patch == compiled.patch);
    CHECK(flm::corelib::IsCorelibVersionCompatible(compiled, runtime));
}

// 1b. Dependency self-test and device context. On this machine the device
// context is REQUIRED, not recorded: the whole point of the target box is that
// there is an AIE4 device, and a run that quietly proceeded without one would
// report every later check as passed while testing nothing.
void CheckDependenciesAndDevice(
    const std::shared_ptr<CorelibApi>& api) {
    api->Check(
        api->functions().selftest_dependencies(),
        "ryzenai_corelib_selftest_dependencies");
    std::cout << "selftest_dependencies: ok\n";

    const bool has_context = api->functions().has_device_context();
    std::cout << "has_device_context: "
              << (has_context ? "true" : "false") << '\n';
    if (!has_context) {
        throw std::runtime_error(
            "ryzenai_corelib_has_device_context reported no AIE4 device "
            "context. This suite must run on the AIE4 target; a pass "
            "without a device would certify nothing. Check that no other "
            "process is holding a device context.");
    }
}

// 1c. A real Stream and a real DeviceTensor, with a bounded element round
// trip. `count` and `offset` are ELEMENTS of the tensor's own dtype (API-7),
// so a BF16 tensor written from an FP32 source consumes half as many source
// bytes as destination bytes -- the asymmetry that makes a byte-taking
// overload dangerous enough that FastFlow does not offer one.
void CheckStreamAndTensorRoundTrip(
    const std::shared_ptr<CorelibApi>& api) {
    ryzenai_corelib_stream_ptr raw_stream = nullptr;
    api->Check(
        api->functions().create_stream(&raw_stream),
        "ryzenai_corelib_create_stream");
    flm::corelib::UniqueStream stream(api, raw_stream);
    CHECK(static_cast<bool>(stream));

    constexpr std::int64_t kRows = 4;
    constexpr std::int64_t kWidth = 128;
    constexpr std::size_t kElements =
        static_cast<std::size_t>(kRows * kWidth);
    const std::array<std::int64_t, 2> shape{kRows, kWidth};

    ryzenai_corelib_tensor_ptr raw_tensor = nullptr;
    api->Check(
        api->functions().create_device_tensor(
            ryzenai_corelib_data_type_bf16,
            shape.data(),
            shape.size(),
            &raw_tensor),
        "ryzenai_corelib_create_device_tensor");
    flm::corelib::UniqueTensor tensor(api, raw_tensor);
    CHECK(static_cast<bool>(tensor));

    std::size_t byte_size = 0;
    api->Check(
        api->functions().tensor_get_byte_size(
            tensor.get(),
            &byte_size),
        "ryzenai_corelib_tensor_get_byte_size");
    CHECK(byte_size == kElements * sizeof(std::uint16_t));

    ryzenai_corelib_data_type data_type{};
    api->Check(
        api->functions().tensor_get_data_type(
            tensor.get(),
            &data_type),
        "ryzenai_corelib_tensor_get_data_type");
    CHECK(data_type == ryzenai_corelib_data_type_bf16);

    // FP32 in, BF16 stored, FP32 back out. The values are chosen to survive
    // BF16 exactly -- powers of two and small sums of them -- so a mismatch
    // means the transfer moved the wrong elements, not that rounding lost a
    // bit. That distinction is why this does not compare with a tolerance.
    std::vector<float> source(kElements);
    for (std::size_t index = 0; index < kElements; ++index) {
        source[index] =
            static_cast<float>((index % 64) + 1) * 0.5f;
    }
    api->WriteElements(
        tensor.get(),
        ryzenai_corelib_data_type_fp32,
        source.data(),
        kElements,
        0);

    std::vector<float> destination(kElements, -1.0f);
    api->ReadElements(
        tensor.get(),
        ryzenai_corelib_data_type_fp32,
        destination.data(),
        kElements,
        0);
    CHECK(destination == source);

    // A bounded interior window, to show that a non-zero element offset lands
    // where it is asked to and leaves its neighbours untouched. Reading the
    // whole tensor back afterwards is what proves the "leaves neighbours
    // untouched" half; checking only the window would pass for a write that
    // clobbered the rest.
    constexpr std::size_t kWindowOffset = kWidth;
    constexpr std::size_t kWindowCount = 16;
    std::vector<float> window(kWindowCount, 8.0f);
    api->WriteElements(
        tensor.get(),
        ryzenai_corelib_data_type_fp32,
        window.data(),
        kWindowCount,
        kWindowOffset);

    std::vector<float> expected = source;
    std::copy(
        window.begin(),
        window.end(),
        expected.begin() + kWindowOffset);
    std::fill(destination.begin(), destination.end(), -1.0f);
    api->ReadElements(
        tensor.get(),
        ryzenai_corelib_data_type_fp32,
        destination.data(),
        kElements,
        0);
    CHECK(destination == expected);

    // Released before the healthy cleanup below. CorelibRuntime refuses to
    // call cleanup() while live objects remain, so a leak here would surface
    // as a shutdown failure rather than as a quiet leak.
    tensor.reset();
    stream.reset();
    CHECK(api->live_object_count() == 0);
    std::cout << "stream and device tensor round trip: ok\n";
}

// 1d. The RoPE gather at a guard page, followed by ONE real tensor_write.
//
// The gather is FastFlow's code and the write is corelib's, and this is the
// only place the two meet on real hardware. The fixture's last source row ends
// exactly at the mapped extent, with PAGE_NOACCESS immediately after, so the
// process dies on an over-read rather than passing with garbage.
void CheckRopeGatherAndRealUpload(
    const std::shared_ptr<CorelibApi>& api) {
    fixture::SyntheticPackage synthetic;
    auto package = Phi4Package::Load(synthetic.path(), api, false);

    const auto& source = package.Require("cos_cache");
    CHECK(source.size == fixture::kRopeBytes);
    CHECK(source.owner->size() == fixture::kRopeMappedBytes);

    constexpr std::size_t kRopeElements =
        static_cast<std::size_t>(constants::kMaxSequenceLength) *
        static_cast<std::size_t>(constants::kRopeDimension / 2);
    static_assert(kRopeElements == 4096u * 48u);

    const std::array<std::int64_t, 2> shape{
        constants::kMaxSequenceLength,
        constants::kRopeDimension / 2};
    ryzenai_corelib_tensor_ptr raw_tensor = nullptr;
    api->Check(
        api->functions().create_device_tensor(
            ryzenai_corelib_data_type_fp32,
            shape.data(),
            shape.size(),
            &raw_tensor),
        "ryzenai_corelib_create_device_tensor");
    flm::corelib::UniqueTensor cos_tensor(api, raw_tensor);

    flm::phi4::RopeSourceView rope{};
    {
        auto* one_past =
            const_cast<std::byte*>(source.data + source.size);
        fixture::NoAccessGuard guard(one_past);
        rope = package.MaterializeRopeGather("cos_cache");
    }
    CHECK(rope.dtype == ryzenai_corelib_data_type_fp16);
    CHECK(rope.count == kRopeElements);

    // One write, in the SOURCE dtype, with `count` in elements of the
    // DESTINATION tensor. tensor_write is the only widening boundary corelib
    // e5258d2 offers, and this is the call the engine makes at load.
    api->WriteElements(
        cos_tensor.get(),
        rope.dtype,
        rope.data,
        kRopeElements,
        0);

    // Read back the three rows the fixture seeded, so the write is shown to
    // have landed rather than merely to have returned success. Row 4095's
    // column 47 is the last element before the guard page, which is the
    // element an over- or under-reading gather would get wrong.
    const auto read_element = [&](std::size_t index) {
        float value = 0.0f;
        api->ReadElements(
            cos_tensor.get(),
            ryzenai_corelib_data_type_fp32,
            &value,
            1,
            index);
        return value;
    };
    CHECK(read_element(0) == 1.0f);
    CHECK(read_element(48) == 2.0f);
    CHECK(read_element(kRopeElements - 1) == 3.0f);

    cos_tensor.reset();
    CHECK(api->live_object_count() == 0);
    std::cout << "RoPE gather at guard page plus real tensor_write: ok\n";
}

// 5. Helper boundaries, discovered rather than transcribed.
//
// The row grid is a property of the SHIPPED kernel set, so the transitions are
// read back out of the running helper table and the boundary rows are derived
// from them. A hard-coded {1, 64, 128, ...} would keep passing against a
// library that had changed its grid, which is exactly the failure this is here
// to catch.
void CheckHelperBoundaries(const std::shared_ptr<CorelibApi>& api) {
    const Phi4ShapePlan plan = Phi4ShapePlan::Build(api);

    for (const auto& row_use : kMultiRowUses) {
        const auto& transitions = plan.Transitions(row_use.use);
        CHECK(!transitions.empty());

        std::vector<std::int64_t> probes{
            1,
            constants::kMaxSequenceLength};
        for (const auto& [live_rows, padded_rows] : transitions) {
            for (const std::int64_t offset : {-1, 0, 1}) {
                const std::int64_t probe = live_rows + offset;
                if (
                    probe >= 1 &&
                    probe <= constants::kMaxSequenceLength) {
                    probes.push_back(probe);
                }
            }
        }
        std::sort(probes.begin(), probes.end());
        probes.erase(
            std::unique(probes.begin(), probes.end()),
            probes.end());

        std::cout << "  " << row_use.name << ": "
                  << transitions.size() << " transitions, "
                  << probes.size() << " boundary probes";

        std::int64_t previous_padded = 0;
        for (const std::int64_t rows : probes) {
            const std::int64_t padded =
                plan.RowsFor(row_use.use, rows);
            // Never below the live rows, never above the single peak
            // allocation, and never decreasing: those three together are
            // what make one capacity-sized tensor safe for every row count.
            CHECK(padded >= rows);
            CHECK(padded <= plan.capacities().layer_rows);
            CHECK(padded >= previous_padded);
            previous_padded = padded;

            // And the plan must still agree with the library it was built
            // from, asked directly at this row count. A plan that had
            // memoised a stale answer would pass every check above.
            std::int64_t m = rows;
            std::int64_t k = constants::kHiddenSize;
            std::int64_t n = row_use.use == RowUse::KvProjection
                                 ? constants::kKvDimension
                                 : constants::kQueryDimension;
            if (row_use.use == RowUse::SsMlp) {
                m = rows;
                api->Check(
                    api->functions().ssmlp_pad_rows(
                        &m,
                        constants::kHiddenSize,
                        constants::kIntermediateSize,
                        constants::kGroupSize),
                    "ryzenai_corelib_ssmlp_bf16_pad_rows");
            } else if (row_use.use == RowUse::Attention) {
                api->Check(
                    api->functions().flat_mha_pad_rows(
                        &m,
                        &plan.attention_desc()),
                    "ryzenai_corelib_flat_mha_bf16_pad_rows");
            } else {
                api->Check(
                    api->functions().matmul_pad_shape(
                        &m,
                        &k,
                        &n,
                        constants::kGroupSize),
                    "ryzenai_corelib_matmul_bf16_pad_shape");
                // MEM-5: the capacity model assumes padding touches M only.
                CHECK(k == constants::kHiddenSize);
            }
            CHECK(m == padded);
        }
        std::cout << ", padded 1 -> " << plan.RowsFor(row_use.use, 1)
                  << ", " << constants::kMaxSequenceLength << " -> "
                  << plan.RowsFor(
                         row_use.use,
                         constants::kMaxSequenceLength)
                  << '\n';

        // Fresh row 1 must stay 1 on every multi-row helper: decode
        // allocates a single row and flat MHA pads its KV window instead.
        CHECK(plan.RowsFor(row_use.use, 1) == 1);
    }

    // The LM head is not a multi-row helper. It ships M in {1, 128} and
    // ERRORS above 128 rather than rounding up, which is the property that
    // makes Phi4ShapePlan's single-row query correct. Assert the refusal
    // rather than inferring it from the code that avoids it.
    CHECK(plan.capacities().lm_head_rows == 1);
    CHECK(plan.RowsFor(RowUse::LmHead, 1) == 1);
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
                "the LM-head shape accepted M=" + std::to_string(rows) +
                " and padded to " + std::to_string(m) +
                ". Phi4ShapePlan queries the LM head at m=1 only because "
                "the shipped kernel set refuses everything above 128; if "
                "that changed, the single-row assumption needs revisiting "
                "rather than this assertion needs relaxing.");
        }
    }
    std::cout << "helper boundaries agree with the running kernel set\n";

    // The peak allocation must really be creatable on the device. The
    // capacity model is only sound if a tensor of that extent exists, and a
    // pad-shape query alone would not show that.
    const std::array<std::int64_t, 2> peak_shape{
        plan.capacities().layer_rows,
        constants::kQueryDimension};
    ryzenai_corelib_tensor_ptr raw_tensor = nullptr;
    api->Check(
        api->functions().create_device_tensor(
            ryzenai_corelib_data_type_bf16,
            peak_shape.data(),
            peak_shape.size(),
            &raw_tensor),
        "ryzenai_corelib_create_device_tensor");
    flm::corelib::UniqueTensor peak(api, raw_tensor);
    CHECK(static_cast<bool>(peak));
    peak.reset();
    CHECK(api->live_object_count() == 0);
    std::cout << "peak layer allocation ("
              << plan.capacities().layer_rows << " x "
              << constants::kQueryDimension
              << " BF16) is creatable on the device\n";
}

}  // namespace

// CTest's SKIP_RETURN_CODE. Returning 0 without a runtime directory would
// report Passed, and a green-and-inert hardware check reads as coverage it
// does not have.
constexpr int kCTestSkipReturnCode = 77;

// Opt-in, and deliberately separate from RYZENAI_CORELIB_RUNTIME_DIR.
//
// A configured runtime directory says which DLL to load; it does not say that
// this machine is the AIE4 target. The development box has an NPU and a real
// corelib but is not AIE4, so keying the hardware run off the directory alone
// would turn every dev-box `ctest` red for a reason that is not a defect.
// Keying it off nothing would be worse: an absent device on the target would
// then read as a skip, and the one machine where this must run would be the
// one machine where a silent skip goes unnoticed. With the flag set, a missing
// device context is a hard failure.
bool HardwareRunRequested() {
    char value[8] = {};
    const DWORD length = GetEnvironmentVariableA(
        "FLM_AIE4_HARDWARE",
        value,
        sizeof(value));
    return length != 0 && length < sizeof(value) &&
           std::string_view(value) == "1";
}

int main() {
    const std::string runtime_dir(FLM_REAL_CORELIB_RUNTIME_DIR);
    if (runtime_dir.empty()) {
        std::cout
            << "test_phi4_hardware: SKIPPED -- configure with "
               "-DRYZENAI_CORELIB_RUNTIME_DIR=<dir containing "
               "ryzenai_corelib.dll> and run on the AIE4 target.\n";
        return kCTestSkipReturnCode;
    }
    if (!HardwareRunRequested()) {
        std::cout
            << "test_phi4_hardware: SKIPPED -- set FLM_AIE4_HARDWARE=1 to "
               "run this on the AIE4 target. Set it only there: on a machine "
               "with no device it turns a skip into a failure.\n";
        return kCTestSkipReturnCode;
    }

    try {
        const std::filesystem::path library =
            std::filesystem::absolute(
                std::filesystem::path(runtime_dir) /
                "ryzenai_corelib.dll")
                .lexically_normal();
        if (!std::filesystem::exists(library)) {
            throw std::runtime_error(
                "RYZENAI_CORELIB_RUNTIME_DIR is set but " +
                library.string() + " does not exist");
        }
        AddExtraDllDirectories(FLM_REAL_CORELIB_EXTRA_DLL_DIRS);
        std::cout << "loading " << library.string() << '\n';

        auto api = CorelibApi::Load(library);
        CheckVersionIdentity(api);
        CheckDependenciesAndDevice(api);
        CheckStreamAndTensorRoundTrip(api);
        CheckRopeGatherAndRealUpload(api);
        CheckHelperBoundaries(api);

        api->functions().cleanup();
        std::cout << "test_phi4_hardware: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_phi4_hardware: FAIL: " << error.what() << '\n';
        return 1;
    }
}
