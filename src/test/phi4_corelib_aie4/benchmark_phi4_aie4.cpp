// Task 13 Steps 4 to 9: the Phi-4 AIE4 performance and memory baseline.
//
// WHAT THIS IS FOR. Design section 4 says performance is explicitly not a
// release blocker for this release, and design section 15.6 says to record
// the baseline "without pass/fail thresholds". So this program's job is not to
// hit numbers. It is to produce an honest, reproducible record of what the
// numbers ARE, with enough identity attached -- machine, CPU and NPU SKU,
// corelib DLL hash and source revision, DynamicDispatch/RyzenMM/XRT versions,
// model hash, FastFlow revision -- that a later change can be compared against
// it.
//
// It therefore reports latency and throughput and asserts NOTHING about them.
// What it DOES assert is the small set of properties that are contracts rather
// than measurements, because a run that quietly violated one would produce a
// baseline describing a different program:
//
//   * 129 synchronizes and 193 dispatches per complete model pass (design
//     10.4), checked per token rather than in aggregate;
//   * 32 V-cache reads and 256 per-head V writes per pass (design 18.5); and
//   * no device tensor and no weight object created after warmup, and no net
//     live corelib object growth, across the 128-token window (design 18.7,
//     design 15.4).
//
// The memory-stability window is the one place a number here is a gate, and
// it is a gate because an unbounded post-warm allocation is a leak, not a slow
// program.
//
// ORDERING IS LOAD-BEARING. Cold TTFT must be the FIRST prefill after load or
// it is not cold -- design 18.6's whole risk is that runtime packing and
// first-use kernel construction dominate it. Warm TTFT must reuse the same
// Stream: design 15.6 says "warmup clears logical sequence state but does not
// rebuild the Stream". And the 128-token stability window runs LAST, after
// every other phase, so that whatever the allocator did during the sweeps has
// already happened and cannot be mistaken for stability.

#include "test_support.hpp"

#include "picosha2.h"

#include <corelib/corelib_runtime.hpp>
#include <models/phi4/phi4_corelib_aie4.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>
#include <models/phi4/phi4_corelib_host.hpp>
#include <models/phi4/phi4_corelib_manifest.hpp>
#include <models/phi4/phi4_corelib_shape_plan.hpp>
#include <lm_config.hpp>
#include <nlohmann/json.hpp>

#include <windows.h>

#include <psapi.h>
#include <setupapi.h>
#include <winver.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace constants = flm::phi4::constants;

using flm::phi4::phi4_corelib_aie4;
using nlohmann::json;

// Design 10.4 and 5.2, as numbers, so a schedule change fails here rather
// than being absorbed into a baseline nobody re-derives.
constexpr std::uint64_t kSynchronizesPerStep =
    static_cast<std::uint64_t>(constants::kLayerCount) * 4u + 1u;
constexpr std::uint64_t kDispatchesPerStep =
    static_cast<std::uint64_t>(constants::kLayerCount) * 6u + 1u;
constexpr std::uint64_t kVReadsPerStep =
    static_cast<std::uint64_t>(constants::kLayerCount);
constexpr std::uint64_t kVWritesPerStep =
    static_cast<std::uint64_t>(constants::kLayerCount) *
    static_cast<std::uint64_t>(constants::kKvHeadCount);

static_assert(kSynchronizesPerStep == 129u);
static_assert(kDispatchesPerStep == 193u);
static_assert(kVReadsPerStep == 32u);
static_assert(kVWritesPerStep == 256u);

// Design 10.1's stated figures, which the run must reproduce rather than
// restate. 64 cache tensors of 8 MiB each.
constexpr std::uint64_t kExpectedKvBytes = 536870912ull;
constexpr std::uint64_t kExpectedEmbeddingBytes = 1229193216ull;

// Step 5: the slope is fitted over tokens 9..128 so initial allocator
// settling is excluded rather than averaged in.
constexpr std::size_t kMemoryWindowTokens = 128;
constexpr std::size_t kMemorySlopeFirstToken = 9;

using Clock = std::chrono::steady_clock;

std::uint64_t ElapsedNs(const Clock::time_point& start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start)
            .count());
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Options {
    std::filesystem::path model_dir;
    std::filesystem::path token_ids_json;
    std::filesystem::path output_json;
    std::string fastflow_revision;
    std::string corelib_source_revision;
    int decode_tokens = static_cast<int>(kMemoryWindowTokens);
    int continuation_samples = 5;
};

[[noreturn]] void Usage(std::string_view problem) {
    throw std::runtime_error(
        std::string(problem) +
        "\nusage: benchmark_phi4_aie4"
        "\n  --model-dir <dir>"
        "\n  --output-json <file>"
        "\n  --token-ids-json <file>"
        "\n  --fastflow-revision <git sha>"
        "\n  --corelib-source-revision <git sha>"
        "\n  [--decode-tokens <n>]"
        "\n  [--continuation-samples <n>]"
        "\n"
        "\n--corelib-source-revision has no default and cannot be derived: "
        "\nryzenai_corelib_get_version reports a hard-coded 0.1.0 that spans "
        "\nthe whole 0.x history, so the DLL cannot identify what it was "
        "\nbuilt from. The caller has to say.");
}

std::string_view RequireValue(
    int argc,
    char** argv,
    int& index,
    std::string_view flag) {
    if (index + 1 >= argc) {
        Usage(std::string(flag) + " requires a value");
    }
    ++index;
    return argv[index];
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--model-dir") {
            options.model_dir = std::filesystem::path(
                RequireValue(argc, argv, index, argument));
        } else if (argument == "--token-ids-json") {
            options.token_ids_json = std::filesystem::path(
                RequireValue(argc, argv, index, argument));
        } else if (argument == "--output-json") {
            options.output_json = std::filesystem::path(
                RequireValue(argc, argv, index, argument));
        } else if (argument == "--fastflow-revision") {
            options.fastflow_revision =
                std::string(RequireValue(argc, argv, index, argument));
        } else if (argument == "--corelib-source-revision") {
            options.corelib_source_revision =
                std::string(RequireValue(argc, argv, index, argument));
        } else if (argument == "--decode-tokens") {
            options.decode_tokens = std::stoi(
                std::string(RequireValue(argc, argv, index, argument)));
        } else if (argument == "--continuation-samples") {
            options.continuation_samples = std::stoi(
                std::string(RequireValue(argc, argv, index, argument)));
        } else {
            Usage("unrecognized argument: " + std::string(argument));
        }
    }
    if (options.model_dir.empty()) {
        Usage("--model-dir is required");
    }
    if (options.output_json.empty()) {
        Usage("--output-json is required");
    }
    if (options.token_ids_json.empty()) {
        Usage("--token-ids-json is required");
    }
    // Design 15.4 and 15.6: the stability window is 128 decode tokens, and
    // design 15.6 wants at least five warm continuation repetitions. Below
    // those the run would produce a document that LOOKS like a baseline and
    // rests on less than the design asks for, which is the failure mode this
    // whole task exists to avoid.
    if (options.decode_tokens < static_cast<int>(kMemoryWindowTokens)) {
        Usage(
            "--decode-tokens must be at least 128: design 15.4 requires "
            "\"at least 128 decode tokens with stable post-warm allocation\"");
    }
    if (options.continuation_samples < 5) {
        Usage(
            "--continuation-samples must be at least 5: design 15.6 requires "
            "\"at least five warm repetitions\"");
    }
    return options;
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

std::string Narrow(const std::wstring& value) {
    if (value.empty()) {
        return {};
    }
    const int size = WideCharToMultiByte(
        CP_UTF8,
        0,
        value.c_str(),
        static_cast<int>(value.size()),
        nullptr,
        0,
        nullptr,
        nullptr);
    std::string result(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(
        CP_UTF8,
        0,
        value.c_str(),
        static_cast<int>(value.size()),
        result.data(),
        size,
        nullptr,
        nullptr);
    return result;
}

std::string MachineName() {
    wchar_t buffer[256];
    DWORD size = static_cast<DWORD>(std::size(buffer));
    if (GetComputerNameExW(ComputerNameDnsHostname, buffer, &size) == 0) {
        throw std::runtime_error(
            "GetComputerNameExW failed (" +
            std::to_string(GetLastError()) + ")");
    }
    return Narrow(std::wstring(buffer, size));
}

std::string RegistryString(
    HKEY root,
    const wchar_t* key,
    const wchar_t* value) {
    wchar_t buffer[1024];
    DWORD bytes = sizeof(buffer);
    const LSTATUS status = RegGetValueW(
        root,
        key,
        value,
        RRF_RT_REG_SZ,
        nullptr,
        buffer,
        &bytes);
    if (status != ERROR_SUCCESS) {
        return {};
    }
    return Narrow(std::wstring(buffer));
}

std::string CpuSku() {
    const std::string name = RegistryString(
        HKEY_LOCAL_MACHINE,
        L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        L"ProcessorNameString");
    if (name.empty()) {
        throw std::runtime_error(
            "could not read ProcessorNameString; the baseline would not say "
            "which CPU produced it");
    }
    // The registry pads this to a fixed width. Trailing spaces in an identity
    // field make two records of the same machine compare unequal.
    const auto end = name.find_last_not_of(" 	");
    return end == std::string::npos ? name : name.substr(0, end + 1);
}

struct NpuIdentity {
    std::string description;
    std::string driver_version;
};

// The NPU is asked of the OS device tree rather than transcribed from a
// session transcript, because a baseline that names the wrong device is worse
// than one that names none.
NpuIdentity DescribeNpu() {
    NpuIdentity identity;
    HDEVINFO set = SetupDiGetClassDevsW(
        nullptr,
        nullptr,
        nullptr,
        DIGCF_PRESENT | DIGCF_ALLCLASSES);
    if (set == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            "SetupDiGetClassDevsW failed (" +
            std::to_string(GetLastError()) + ")");
    }
    SP_DEVINFO_DATA data{};
    data.cbSize = sizeof(data);
    for (DWORD index = 0;
         SetupDiEnumDeviceInfo(set, index, &data) != 0;
         ++index) {
        wchar_t description[512] = {};
        DWORD required = 0;
        if (SetupDiGetDeviceRegistryPropertyW(
                set,
                &data,
                SPDRP_DEVICEDESC,
                nullptr,
                reinterpret_cast<PBYTE>(description),
                static_cast<DWORD>(sizeof(description)),
                &required) == 0) {
            continue;
        }
        const std::wstring text(description);
        if (
            text.find(L"NPU") == std::wstring::npos &&
            text.find(L"XDNA") == std::wstring::npos) {
            continue;
        }
        identity.description = Narrow(text);
        HKEY driver_key = SetupDiOpenDevRegKey(
            set,
            &data,
            DICS_FLAG_GLOBAL,
            0,
            DIREG_DRV,
            KEY_READ);
        if (driver_key != INVALID_HANDLE_VALUE) {
            wchar_t version[256] = {};
            DWORD version_bytes = sizeof(version);
            DWORD type = 0;
            if (RegQueryValueExW(
                    driver_key,
                    L"DriverVersion",
                    nullptr,
                    &type,
                    reinterpret_cast<LPBYTE>(version),
                    &version_bytes) == ERROR_SUCCESS) {
                identity.driver_version = Narrow(std::wstring(version));
            }
            RegCloseKey(driver_key);
        }
        break;
    }
    SetupDiDestroyDeviceInfoList(set);
    if (identity.description.empty()) {
        throw std::runtime_error(
            "no NPU device was found in the OS device tree. This program "
            "measures an AIE4 baseline; running it where there is no NPU "
            "would record numbers for a machine that cannot produce them.");
    }
    return identity;
}

std::string Sha256File(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open for hashing: " + path.string());
    }
    std::vector<unsigned char> digest(picosha2::k_digest_size);
    picosha2::hash256(stream, digest.begin(), digest.end());
    return picosha2::bytes_to_hex_string(digest.begin(), digest.end());
}

// A loaded module's identity: its file version where it has one, and its
// SHA-256 where it does not.
//
// Recording "unknown" for a DLL without a version resource would leave an
// identity field that reads as absent when the binary is in fact perfectly
// identifiable. The hash IS the identity; the version is a convenience. So
// the field always says something true about the exact bytes.
std::string DescribeLoadedModule(const wchar_t* module_name) {
    HMODULE module = GetModuleHandleW(module_name);
    if (module == nullptr) {
        return {};
    }
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        module,
        buffer.data(),
        static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        return {};
    }
    buffer.resize(length);
    const std::filesystem::path path(buffer);

    DWORD ignored = 0;
    const DWORD info_size =
        GetFileVersionInfoSizeW(path.c_str(), &ignored);
    if (info_size != 0) {
        std::vector<std::byte> block(info_size);
        if (GetFileVersionInfoW(
                path.c_str(),
                0,
                info_size,
                block.data()) != 0) {
            VS_FIXEDFILEINFO* fixed = nullptr;
            UINT fixed_size = 0;
            if (
                VerQueryValueW(
                    block.data(),
                    L"\\",
                    reinterpret_cast<LPVOID*>(&fixed),
                    &fixed_size) != 0 &&
                fixed != nullptr) {
                return std::to_string(HIWORD(fixed->dwFileVersionMS)) + "." +
                       std::to_string(LOWORD(fixed->dwFileVersionMS)) + "." +
                       std::to_string(HIWORD(fixed->dwFileVersionLS)) + "." +
                       std::to_string(LOWORD(fixed->dwFileVersionLS)) +
                       " (" + path.string() + ")";
            }
        }
    }
    return "no version resource; sha256 " + Sha256File(path).substr(0, 16) +
           " (" + path.string() + ")";
}

struct LoadedModule {
    std::string path;
    std::string sha256;
};

// Which DLL actually serviced these calls, asked of the OS loader rather than
// of the path we requested. Task 12 added this because every candidate
// explanation for the run-to-run divergence reduced to "the two runs may not
// have been running the same code", and nothing in the artifacts could tell
// them apart.
LoadedModule DescribeLoadedCorelib(const flm::corelib::CorelibApi& api) {
    HMODULE module = nullptr;
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(api.functions().get_version),
            &module) == 0 ||
        module == nullptr) {
        throw std::runtime_error(
            "could not identify the module backing corelib's entry points (" +
            std::to_string(GetLastError()) + ")");
    }
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        module,
        buffer.data(),
        static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        throw std::runtime_error("could not read the loaded corelib path");
    }
    buffer.resize(length);
    const std::filesystem::path resolved(buffer);
    const auto requested = api.library_path().lexically_normal();
    if (
        resolved.lexically_normal() != requested &&
        !std::filesystem::equivalent(resolved, requested)) {
        throw std::runtime_error(
            "the loaded corelib is not the one requested: asked for " +
            requested.string() + ", the loader supplied " +
            resolved.string());
    }
    return LoadedModule{resolved.string(), Sha256File(resolved)};
}

// The model's identity, from the manifest's own recorded per-file hashes.
//
// Those hashes are not taken on trust: the caller loads the package with
// full-hash verification enabled before this runs, so the manifest has already
// been checked against the bytes on disk. Digesting the verified records into
// one value gives a single comparable figure without re-reading 3.2 GB.
struct ModelIdentity {
    std::string combined_sha256;
    json files;
    std::uint64_t embedding_bytes = 0;
};

ModelIdentity DescribeModel(const std::filesystem::path& model_dir) {
    const auto manifest_path = model_dir / "corelib_phi4_manifest.json";
    std::ifstream stream(manifest_path);
    if (!stream) {
        throw std::runtime_error(
            "cannot open " + manifest_path.string());
    }
    json manifest;
    stream >> manifest;

    ModelIdentity identity;
    identity.files = json::object();
    std::string canonical;
    for (const auto& [name, record] : manifest.at("files").items()) {
        const auto hash = record.at("sha256").get<std::string>();
        const auto size = record.at("size").get<std::uint64_t>();
        identity.files[name] = json{{"sha256", hash}, {"size", size}};
        canonical += name + " " + std::to_string(size) + " " + hash + "\n";
    }
    const auto manifest_hash = Sha256File(manifest_path);
    identity.files["corelib_phi4_manifest.json"] =
        json{{"sha256", manifest_hash}};
    canonical += "corelib_phi4_manifest.json " + manifest_hash + "\n";
    identity.combined_sha256 =
        picosha2::hash256_hex_string(canonical.begin(), canonical.end());
    // MEASURED from the package rather than restated from design 10.1. The
    // design says the FP16 embedding is exactly 1,229,193,216 bytes; reading
    // it out of the manifest that was just hash-verified is what turns that
    // sentence into an observation about this package.
    identity.embedding_bytes =
        manifest.at("initializers")
            .at("model.embed_tokens.weight")
            .at("length")
            .get<std::uint64_t>();
    return identity;
}

// ---------------------------------------------------------------------------
// Process memory (Step 5)
// ---------------------------------------------------------------------------

struct MemorySample {
    std::uint64_t private_bytes = 0;
    std::uint64_t working_set_bytes = 0;
    std::uint64_t peak_working_set_bytes = 0;
};

MemorySample SampleMemory() {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
            sizeof(counters)) == 0) {
        throw std::runtime_error(
            "GetProcessMemoryInfo failed (" +
            std::to_string(GetLastError()) + ")");
    }
    return MemorySample{
        static_cast<std::uint64_t>(counters.PrivateUsage),
        static_cast<std::uint64_t>(counters.WorkingSetSize),
        static_cast<std::uint64_t>(counters.PeakWorkingSetSize)};
}

// Ordinary least squares of private bytes against token index, over
// [first, samples.size()). Returned in bytes per token.
//
// The fit starts at token 9 rather than token 1 because the first handful of
// tokens carry the allocator's own settling -- committing pages the earlier
// phases reserved -- and folding that into the slope would report a leak that
// stops on its own.
double PrivateByteSlope(
    const std::vector<MemorySample>& samples,
    std::size_t first) {
    if (samples.size() <= first + 1) {
        throw std::runtime_error(
            "not enough memory samples to fit a slope; the stability window "
            "did not run");
    }
    const std::size_t count = samples.size() - first;
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (std::size_t index = first; index < samples.size(); ++index) {
        sum_x += static_cast<double>(index);
        sum_y += static_cast<double>(samples[index].private_bytes);
    }
    const double mean_x = sum_x / static_cast<double>(count);
    const double mean_y = sum_y / static_cast<double>(count);
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = first; index < samples.size(); ++index) {
        const double dx = static_cast<double>(index) - mean_x;
        numerator +=
            dx * (static_cast<double>(samples[index].private_bytes) - mean_y);
        denominator += dx * dx;
    }
    if (denominator == 0.0) {
        return 0.0;
    }
    return numerator / denominator;
}

// ---------------------------------------------------------------------------
// Latency statistics
// ---------------------------------------------------------------------------

std::uint64_t Percentile(std::vector<std::uint64_t> values, double fraction) {
    if (values.empty()) {
        throw std::runtime_error("percentile of an empty sample");
    }
    std::sort(values.begin(), values.end());
    // Nearest-rank. With five samples the p95 is the largest one, which is
    // the honest answer rather than an interpolation between points that were
    // never observed.
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())));
    const std::size_t index =
        std::min(values.size(), std::max<std::size_t>(rank, 1)) - 1;
    return values[index];
}

json LatencySummary(const std::vector<std::uint64_t>& values) {
    json record;
    record["samples_ns"] = values;
    record["count"] = values.size();
    record["min_ns"] = *std::min_element(values.begin(), values.end());
    record["max_ns"] = *std::max_element(values.begin(), values.end());
    record["mean_ns"] = static_cast<std::uint64_t>(
        std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size()));
    record["p50_ns"] = Percentile(values, 0.50);
    record["p95_ns"] = Percentile(values, 0.95);
    return record;
}

// ---------------------------------------------------------------------------
// Token plans
// ---------------------------------------------------------------------------

std::vector<int> LoadPromptTokens(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error(
            "cannot open --token-ids-json: " + path.string());
    }
    json document;
    stream >> document;
    std::vector<int> prompt;
    if (document.is_array()) {
        prompt = document.get<std::vector<int>>();
    } else {
        prompt = document.at("prefix").get<std::vector<int>>();
        if (document.contains("suffix")) {
            for (const int id : document.at("suffix").get<std::vector<int>>()) {
                prompt.push_back(id);
            }
        }
    }
    if (prompt.empty()) {
        throw std::runtime_error("--token-ids-json holds no tokens");
    }
    return prompt;
}

// Rows are filled by cycling the committed prompt rather than repeating one
// token: a constant ID satisfies every shape check while masking a row
// indexing defect, and it would also give the embedding gather a workload it
// never sees in production.
std::vector<int> MakeTokens(
    const std::vector<int>& sample,
    std::size_t count) {
    std::vector<int> tokens(count);
    for (std::size_t index = 0; index < count; ++index) {
        tokens[index] = sample[index % sample.size()];
    }
    return tokens;
}

// ---------------------------------------------------------------------------
// Schedule assertions
// ---------------------------------------------------------------------------

struct MetricsDelta {
    std::uint64_t dispatches = 0;
    std::uint64_t synchronizes = 0;
    std::uint64_t v_reads = 0;
    std::uint64_t v_writes = 0;
    std::uint64_t v_bytes = 0;
    std::uint64_t v_ns = 0;
};

MetricsDelta Delta(
    const flm::phi4::Phi4Aie4Metrics& before,
    const flm::phi4::Phi4Aie4Metrics& after) {
    return MetricsDelta{
        after.dispatch_count - before.dispatch_count,
        after.synchronize_count - before.synchronize_count,
        after.v_read_calls - before.v_read_calls,
        after.v_write_calls - before.v_write_calls,
        after.v_bytes - before.v_bytes,
        after.v_scatter_ns - before.v_scatter_ns};
}

void RequireOneModelStep(const MetricsDelta& delta, std::uint64_t steps) {
    CHECK(delta.synchronizes == kSynchronizesPerStep * steps);
    CHECK(delta.dispatches == kDispatchesPerStep * steps);
    CHECK(delta.v_reads == kVReadsPerStep * steps);
    CHECK(delta.v_writes == kVWritesPerStep * steps);
}

// ---------------------------------------------------------------------------

int Argmax(const buffer<bf16>& logits) {
    return flm::phi4::ArgmaxLowest(
        std::span<const bf16>(logits.data(), logits.size()));
}

// Prefill `rows` tokens from position zero and leave the engine there.
int EstablishContext(
    phi4_corelib_aie4& engine,
    const std::vector<int>& sample,
    std::size_t rows) {
    engine.clear_context();
    std::vector<int> tokens = MakeTokens(sample, rows);
    const auto logits = engine.prefill(tokens);
    CHECK(
        engine.get_current_context_length() ==
        static_cast<int>(rows));
    return Argmax(logits);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        if (!std::filesystem::is_directory(options.model_dir)) {
            throw std::runtime_error(
                "--model-dir is not a directory: " +
                options.model_dir.string());
        }
        const std::vector<int> prompt =
            LoadPromptTokens(options.token_ids_json);

        const std::filesystem::path executable_dir =
            std::filesystem::absolute(argv[0]).parent_path();
        auto runtime =
            flm::corelib::CorelibRuntime::GetOrCreate(executable_dir);
        const auto api = runtime->api();
        std::cout << "corelib runtime ready: "
                  << api->library_path().string() << '\n';

        json document;

        // -------------------------------------------------------------------
        // Identity
        // -------------------------------------------------------------------
        const LoadedModule corelib = DescribeLoadedCorelib(*api);
        const NpuIdentity npu = DescribeNpu();

        // Verify the package against its own manifest BEFORE recording the
        // model hash. Without this the recorded identity would be a
        // transcription of what the manifest claims rather than a statement
        // about the bytes the run consumed.
        const auto verify_started = Clock::now();
        {
            const auto verified =
                flm::phi4::Phi4Package::Load(options.model_dir, api, true);
            (void)verified;
        }
        const std::uint64_t model_verify_ns = ElapsedNs(verify_started);
        const ModelIdentity model = DescribeModel(options.model_dir);

        json identity;
        identity["machine"] = MachineName();
        identity["cpu_sku"] = CpuSku();
        identity["npu_sku"] = npu.description;
        identity["npu_driver_version"] = npu.driver_version;
        identity["corelib_dll_path"] = corelib.path;
        identity["corelib_dll_sha256"] = corelib.sha256;
        identity["corelib_version"] =
            flm::corelib::FormatCorelibVersion(api->runtime_version());
        identity["corelib_compiled_version"] =
            flm::corelib::FormatCorelibVersion(
                flm::corelib::CompiledCorelibVersion());
        identity["corelib_source_revision"] = options.corelib_source_revision;
        identity["dynamic_dispatch_version"] =
            DescribeLoadedModule(L"dyn_dispatch_core.dll");
        identity["ryzen_mm_version"] = DescribeLoadedModule(L"ryzen_mm.dll");
        identity["xrt_version"] = DescribeLoadedModule(L"xrt_coreutil.dll");
        identity["model_dir"] = options.model_dir.string();
        identity["model_sha256"] = model.combined_sha256;
        identity["model_files"] = model.files;
        identity["model_hash_verified"] = true;
        identity["model_hash_verify_ns"] = model_verify_ns;
        identity["fastflow_revision"] = options.fastflow_revision;
        {
            std::string canonical;
            for (const int id : prompt) {
                canonical += std::to_string(id) + ",";
            }
            identity["prompt_id"] =
                picosha2::hash256_hex_string(
                    canonical.begin(),
                    canonical.end())
                    .substr(0, 16);
        }
        identity["prompt_source"] = options.token_ids_json.string();
        {
            SYSTEMTIME utc{};
            GetSystemTime(&utc);
            char stamp[32];
            std::snprintf(
                stamp,
                sizeof(stamp),
                "%04u-%02u-%02uT%02u:%02u:%02uZ",
                utc.wYear,
                utc.wMonth,
                utc.wDay,
                utc.wHour,
                utc.wMinute,
                utc.wSecond);
            identity["utc"] = stamp;
        }
        document["identity"] = identity;
        std::cout << "identity: " << identity["machine"].get<std::string>()
                  << " / " << npu.description << " / corelib "
                  << corelib.sha256.substr(0, 16) << '\n';

        LM_Config config;
        config._json_config = json::object();
        config.model_path = options.model_dir.string();
        config.model_name = options.model_dir.filename().string();

        const MemorySample before_load = SampleMemory();
        std::uint64_t peak_private = before_load.private_bytes;
        std::uint64_t peak_working_set = before_load.peak_working_set_bytes;
        const auto note_peak = [&](const MemorySample& sample) {
            peak_private = std::max(peak_private, sample.private_bytes);
            peak_working_set =
                std::max(peak_working_set, sample.peak_working_set_bytes);
        };

        {
            // ---------------------------------------------------------------
            // Step 6: model load
            // ---------------------------------------------------------------
            const auto load_started = Clock::now();
            phi4_corelib_aie4 engine(
                config,
                options.model_dir,
                runtime,
                static_cast<std::uint32_t>(constants::kMaxSequenceLength));
            const std::uint64_t wall_load_ns = ElapsedNs(load_started);
            const auto load_metrics = engine.metrics();
            note_peak(SampleMemory());

            json model_load;
            model_load["manifest_map_ns"] = load_metrics.manifest_map_ns;
            model_load["shape_plan_ns"] = load_metrics.shape_plan_ns;
            model_load["weight_pack_ns"] = load_metrics.weight_pack_ns;
            model_load["device_setup_ns"] = load_metrics.device_setup_ns;
            // The remainder, stated rather than left for the reader to
            // subtract. If this is not small, a phase is missing a timer.
            model_load["unaccounted_ns"] =
                wall_load_ns - load_metrics.manifest_map_ns -
                load_metrics.shape_plan_ns - load_metrics.weight_pack_ns -
                load_metrics.device_setup_ns;
            model_load["shape_plan_share"] =
                static_cast<double>(load_metrics.shape_plan_ns) /
                static_cast<double>(wall_load_ns);
            model_load["engine_reported_total_ns"] =
                load_metrics.model_load_ns;
            model_load["total_ns"] = wall_load_ns;
            model_load["weight_objects"] = load_metrics.weight_create_count;
            model_load["device_tensors"] =
                load_metrics.device_tensor_create_count;
            model_load["mapped_source_bytes"] =
                load_metrics.mapped_source_bytes;
            model_load["packed_weight_bytes"] =
                load_metrics.packed_weight_bytes;
            // Design 8.1 and 9.2: 32 layers x 5 objects plus the LM head.
            // A count other than 161 means the package changed shape, which
            // would make every figure below describe a different model.
            CHECK(
                load_metrics.weight_create_count ==
                static_cast<std::uint64_t>(constants::kLayerCount) * 5u + 1u);
            document["model_load"] = model_load;
            std::cout << "model load " << wall_load_ns / 1000000 << " ms ("
                      << load_metrics.weight_create_count
                      << " weight objects)\n";

            const std::uint64_t creates_after_load =
                api->creation_count(flm::corelib::CorelibObjectKind::Tensor);
            const std::uint64_t weights_after_load =
                api->weight_creation_count();
            const std::size_t live_after_load = api->live_object_count();

            // ---------------------------------------------------------------
            // Step 6: cold and warm TTFT
            //
            // Cold is the FIRST prefill on a freshly built Stream, so it
            // carries first-use kernel construction -- design 18.6's risk.
            // Warm is the SAME prompt after clear_context() on the SAME
            // Stream, per design 15.6's "warmup clears logical sequence state
            // but does not rebuild the Stream". The engine is deliberately not
            // rebuilt between them.
            // ---------------------------------------------------------------
            // A SECOND helper interrogation, and it is not free: the
            // engine's own is 86% of model load. It is timed and labelled
            // here so it can never be mistaken for part of the load
            // timeline above, and so the cost of asking twice is visible.
            const auto benchmark_plan_started = Clock::now();
            const auto shape_plan =
                flm::phi4::Phi4ShapePlan::Build(api);
            document["model_load"]["benchmark_second_shape_plan_ns"] =
                ElapsedNs(benchmark_plan_started);
            const std::int64_t prompt_extent = shape_plan.RowsFor(
                flm::phi4::RowUse::Attention,
                static_cast<std::int64_t>(prompt.size()));

            std::vector<int> cold_tokens = prompt;
            const auto cold_before = engine.metrics();
            const auto cold_started = Clock::now();
            auto logits = engine.prefill(cold_tokens);
            const std::uint64_t cold_ns = ElapsedNs(cold_started);
            const auto cold_after = engine.metrics();
            RequireOneModelStep(Delta(cold_before, cold_after), 1);
            const int cold_token = Argmax(logits);
            note_peak(SampleMemory());

            engine.clear_context();
            std::vector<int> warm_tokens = prompt;
            const auto warm_before = engine.metrics();
            const auto warm_started = Clock::now();
            logits = engine.prefill(warm_tokens);
            const std::uint64_t warm_ns = ElapsedNs(warm_started);
            const auto warm_after = engine.metrics();
            RequireOneModelStep(Delta(warm_before, warm_after), 1);
            const int warm_token = Argmax(logits);

            json ttft;
            ttft["prompt_id"] = identity["prompt_id"];
            ttft["prompt_source"] = options.token_ids_json.string();
            ttft["prompt_token_count"] = prompt.size();
            ttft["row_extent"] = prompt_extent;
            ttft["cold_ns"] = cold_ns;
            ttft["warm_ns"] = warm_ns;
            ttft["stream_rebuilt_for_warm"] = false;
            ttft["cold_top1_id"] = cold_token;
            ttft["warm_top1_id"] = warm_token;
            // The same prompt on the same weights must select the same token
            // whether the Stream is cold or warm. If it did not, "warm" would
            // be measuring a different computation.
            CHECK(cold_token == warm_token);
            document["ttft"] = ttft;
            std::cout << "TTFT cold " << cold_ns / 1000000 << " ms, warm "
                      << warm_ns / 1000000 << " ms\n";

            // ---------------------------------------------------------------
            // Step 7: fresh prefill at helper-discovered extents
            // ---------------------------------------------------------------
            std::vector<std::int64_t> prefill_rows{
                1,
                constants::kMaxSequenceLength};
            for (const auto& [live_rows, padded_rows] :
                 shape_plan.Transitions(flm::phi4::RowUse::Attention)) {
                (void)padded_rows;
                if (
                    live_rows >= 1 &&
                    live_rows <= constants::kMaxSequenceLength) {
                    prefill_rows.push_back(live_rows);
                }
            }
            std::sort(prefill_rows.begin(), prefill_rows.end());
            prefill_rows.erase(
                std::unique(prefill_rows.begin(), prefill_rows.end()),
                prefill_rows.end());

            json prefill_points = json::array();
            for (const std::int64_t rows : prefill_rows) {
                engine.clear_context();
                std::vector<int> tokens =
                    MakeTokens(prompt, static_cast<std::size_t>(rows));
                const auto before = engine.metrics();
                const auto started = Clock::now();
                const auto row_logits = engine.prefill(tokens);
                const std::uint64_t ns = ElapsedNs(started);
                const auto after = engine.metrics();
                const auto delta = Delta(before, after);
                RequireOneModelStep(delta, 1);
                CHECK(engine.get_current_context_length() == rows);

                json point;
                point["rows"] = rows;
                point["padded_rows"] = shape_plan.RowsFor(
                    flm::phi4::RowUse::Attention,
                    rows);
                point["ns"] = ns;
                point["tokens_per_second"] =
                    static_cast<double>(rows) * 1e9 / static_cast<double>(ns);
                point["top1_id"] = Argmax(row_logits);
                point["v_bytes"] = delta.v_bytes;
                point["v_scatter_ns"] = delta.v_ns;
                prefill_points.push_back(std::move(point));
                note_peak(SampleMemory());
                std::cout << "  prefill " << rows << " rows: "
                          << ns / 1000000 << " ms\n";
            }
            document["prefill"] = json{{"points", prefill_points}};

            // ---------------------------------------------------------------
            // Step 9: both continuation routes
            //
            // Append restores the logical position to the end of the rendered
            // history and walks the suffix one row at a time. Re-prefill
            // clears and recomputes history+suffix in one call from position
            // zero. Design 10.7's two routes, composed here rather than
            // inferred, so the measurement cannot silently be of whichever one
            // the policy would have picked.
            //
            // checkpoint()/restore() moves the logical position only; the KV
            // rows past it are overwritten by the next pass, which is exactly
            // what the product does after a cancelled turn. Re-prefilling the
            // history between samples instead would have measured the reset,
            // not the route.
            // ---------------------------------------------------------------
            // A fixed origin for the interleaving evidence below. Offsets
            // from one epoch are comparable across points; raw clock values
            // are not portable into JSON.
            const auto benchmark_epoch = Clock::now();
            const std::vector<std::size_t> histories{512, 2048};
            // DENSE ENOUGH TO BRACKET THE CROSSOVER, not just the design's
            // five reporting points.
            //
            // Design 15.6 names 1, 2, 32, 128 and 256 as the lengths to
            // REPORT, and the first version of this benchmark measured only
            // those. On that grid append last won at 2 and first lost at 32,
            // and the report published "the threshold is 2" as a measured
            // answer. It is not. It is the largest point in a sparse grid
            // where append still won, and the true crossover was afterwards
            // bracketed near 9 and 26. Task 14 exists to choose that
            // constant, so publishing a grid artifact as a measurement would
            // have enshrined a value off by a factor of four to thirteen.
            //
            // The design's five points are all still here; the extra ones
            // exist so the answer comes from the data rather than from where
            // the grid happened to stop.
            const std::vector<std::size_t> suffixes{
                1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 256};
            json continuation_points = json::array();
            json crossover = json::object();
            bool prefix_monotonic = true;

            for (const std::size_t history : histories) {
                std::vector<std::uint64_t> append_p50;
                std::vector<std::uint64_t> append_p95;
                std::vector<std::uint64_t> reprefill_p50;
                std::vector<std::uint64_t> reprefill_p95;
                std::vector<std::uint64_t> append_drifts;
                std::vector<std::uint64_t> reprefill_drifts;

                for (const std::size_t suffix : suffixes) {
                    const std::vector<int> suffix_tokens =
                        MakeTokens(prompt, suffix);

                    // INTERLEAVED, one append sample then one re-prefill
                    // sample, and back.
                    //
                    // The first version ran all six append samples and then
                    // all six re-prefill samples. At suffix 256 that is two
                    // minutes of append followed by seven seconds of
                    // re-prefill, and this machine has been measured shifting
                    // regime by a factor of 1.8 between runs. A shift landing
                    // between the two blocks moves one route and not the
                    // other, and NOTHING in the within-point spread can see
                    // it: the spread is computed inside each block, where the
                    // regime was constant.
                    //
                    // Interleaving makes the comparison paired in time, so a
                    // drift moves both routes together. The residual is then
                    // what the within-point spread actually bounds, which is
                    // what makes the decision rule below defensible rather
                    // than merely tighter than the alternative.
                    //
                    // Each repetition re-establishes the history, because a
                    // re-prefill leaves the position past it. That reset is
                    // the same work the append precondition needs anyway.
                    std::vector<std::uint64_t> append_samples;
                    std::vector<std::uint64_t> reprefill_samples;
                    // WHEN each measured sample started, as an offset from a
                    // fixed epoch, in measurement order.
                    //
                    // The document asserts that the samples are interleaved,
                    // and an assertion in prose is worth nothing: re-render
                    // an older baseline and the claim is published over
                    // non-interleaved data. These offsets let the validator
                    // CHECK the alternation instead of taking a boolean's
                    // word for it.
                    std::vector<std::uint64_t> append_starts;
                    std::vector<std::uint64_t> reprefill_starts;
                    for (int repeat = -1;
                         repeat < options.continuation_samples;
                         ++repeat) {
                        // One discarded warm-up repetition, then the measured
                        // ones. Design 15.6 asks for warm repetitions; the
                        // first pass at a new row extent pays for whatever
                        // the library caches on first use at that shape.
                        EstablishContext(engine, prompt, history);
                        engine.checkpoint();

                        engine.restore();
                        {
                            const auto before = engine.metrics();
                            const auto started = Clock::now();
                            for (const int token : suffix_tokens) {
                                logits = engine.forward(token);
                            }
                            const std::uint64_t ns = ElapsedNs(started);
                            const auto after = engine.metrics();
                            // Append is N complete one-row passes, so the
                            // synchronize and V-scatter counts scale
                            // linearly. Design 15.3 states this; it is
                            // checked, not assumed.
                            RequireOneModelStep(
                                Delta(before, after),
                                static_cast<std::uint64_t>(suffix));
                            CHECK(
                                engine.get_current_context_length() ==
                                static_cast<int>(history + suffix));
                            if (repeat >= 0) {
                                append_samples.push_back(ns);
                                append_starts.push_back(
                                    static_cast<std::uint64_t>(
                                        std::chrono::duration_cast<
                                            std::chrono::nanoseconds>(
                                            started - benchmark_epoch)
                                            .count()));
                            }
                        }

                        {
                            engine.clear_context();
                            std::vector<int> full =
                                MakeTokens(prompt, history + suffix);
                            const auto before = engine.metrics();
                            const auto started = Clock::now();
                            logits = engine.prefill(full);
                            const std::uint64_t ns = ElapsedNs(started);
                            const auto after = engine.metrics();
                            // Re-prefill is ONE complete pass regardless of
                            // the row count. That asymmetry against append is
                            // the whole reason the two routes exist.
                            RequireOneModelStep(Delta(before, after), 1);
                            CHECK(
                                engine.get_current_context_length() ==
                                static_cast<int>(history + suffix));
                            if (repeat >= 0) {
                                reprefill_samples.push_back(ns);
                                reprefill_starts.push_back(
                                    static_cast<std::uint64_t>(
                                        std::chrono::duration_cast<
                                            std::chrono::nanoseconds>(
                                            started - benchmark_epoch)
                                            .count()));
                            }
                        }
                    }

                    json append_point = LatencySummary(append_samples);
                    append_point["history_rows"] = history;
                    append_point["suffix"] = suffix;
                    append_point["route"] = "append";
                    append_point["model_steps"] = suffix;
                    append_point["interleaved_with_reprefill"] = true;
                    // DRIFT ACROSS THE POINT: the last measured sample
                    // against the first. Interleaving makes a regime shift
                    // affect both routes; this is what makes it VISIBLE, so
                    // the decision rule can refuse a point where the machine
                    // moved more than the routes differ.
                    const std::uint64_t append_drift =
                        append_samples.back() > append_samples.front()
                            ? append_samples.back() - append_samples.front()
                            : append_samples.front() - append_samples.back();
                    append_point["drift_ns"] = append_drift;
                    append_point["sample_starts_ns"] = append_starts;
                    append_point["tokens_per_second"] =
                        static_cast<double>(suffix) * 1e9 /
                        static_cast<double>(
                            append_point["p50_ns"].get<std::uint64_t>());
                    append_p50.push_back(
                        append_point["p50_ns"].get<std::uint64_t>());
                    append_p95.push_back(
                        append_point["p95_ns"].get<std::uint64_t>());
                    append_drifts.push_back(append_drift);
                    continuation_points.push_back(std::move(append_point));

                    json reprefill_point = LatencySummary(reprefill_samples);
                    reprefill_point["history_rows"] = history;
                    reprefill_point["suffix"] = suffix;
                    reprefill_point["route"] = "reprefill";
                    reprefill_point["model_steps"] = 1;
                    reprefill_point["interleaved_with_append"] = true;
                    const std::uint64_t reprefill_drift =
                        reprefill_samples.back() > reprefill_samples.front()
                            ? reprefill_samples.back() -
                                  reprefill_samples.front()
                            : reprefill_samples.front() -
                                  reprefill_samples.back();
                    reprefill_point["drift_ns"] = reprefill_drift;
                    reprefill_point["sample_starts_ns"] = reprefill_starts;
                    reprefill_point["tokens_per_second"] =
                        static_cast<double>(history + suffix) * 1e9 /
                        static_cast<double>(
                            reprefill_point["p50_ns"].get<std::uint64_t>());
                    reprefill_p50.push_back(
                        reprefill_point["p50_ns"].get<std::uint64_t>());
                    reprefill_p95.push_back(
                        reprefill_point["p95_ns"].get<std::uint64_t>());
                    reprefill_drifts.push_back(reprefill_drift);
                    continuation_points.push_back(std::move(reprefill_point));

                    note_peak(SampleMemory());
                    std::cout << "  continuation history " << history
                              << " suffix " << suffix << ": append p50 "
                              << append_p50.back() / 1000000 << " ms, "
                              << "reprefill p50 "
                              << reprefill_p50.back() / 1000000 << " ms\n";
                }

                // Design 15.6: "assert the append-winning lengths are
                // prefix-monotonic or select threshold zero".
                //
                // A point is only DECIDED when the gap between the two routes
                // exceeds the spread WITHIN a route at that same point. That
                // rule is derived from the data rather than chosen: this
                // machine has been measured moving by a factor of 1.8 between
                // runs, and a route difference smaller than the p50-to-p95
                // spread of the samples that produced it is not a difference
                // anyone can act on. An undecided point widens the crossover
                // bracket instead of silently picking a side.
                //
                // What is reported is a BRACKET, never a single threshold:
                // the largest suffix at which append decisively wins, and the
                // smallest at which it decisively loses. Anything between is
                // unmeasured, and saying so is the point -- Task 14 has to
                // choose the constant and must be able to see how much room
                // the measurement actually leaves it.
                std::size_t append_wins_up_to = 0;
                std::size_t reprefill_wins_from = 0;
                bool still_winning = true;
                bool any_win_after_loss = false;
                json decisions = json::array();
                for (std::size_t index = 0; index < suffixes.size(); ++index) {
                    const std::uint64_t pa = append_p50[index];
                    const std::uint64_t pr = reprefill_p50[index];
                    const std::uint64_t gap = pa > pr ? pa - pr : pr - pa;
                    // TWO sources of uncertainty, and the point must beat
                    // both.
                    //
                    // `noise` is the within-point spread: how much a route
                    // varies across its own five samples. `drift` is how far
                    // the machine moved between the first and last sample of
                    // the point, which is the between-run instability seen
                    // from inside the point. The previous rule used `noise`
                    // alone, and a within-point spread cannot see a regime
                    // shift -- which is exactly the objection the document's
                    // own "below 2x is unresolved" caveat was making.
                    const std::uint64_t noise = std::max(
                        append_p95[index] - append_p50[index],
                        reprefill_p95[index] - reprefill_p50[index]);
                    const std::uint64_t drift = std::max(
                        append_drifts[index], reprefill_drifts[index]);
                    const std::uint64_t uncertainty = std::max(noise, drift);
                    const bool decided = gap > uncertainty;
                    const bool append_wins_here = pa < pr;

                    json decision;
                    decision["suffix"] = suffixes[index];
                    decision["append_p50_ns"] = pa;
                    decision["reprefill_p50_ns"] = pr;
                    decision["gap_ns"] = gap;
                    decision["within_route_spread_ns"] = noise;
                    decision["drift_across_point_ns"] = drift;
                    decision["uncertainty_ns"] = uncertainty;
                    decision["gap_over_uncertainty"] =
                        uncertainty == 0
                            ? 0.0
                            : static_cast<double>(gap) /
                                  static_cast<double>(uncertainty);
                    decision["decided"] = decided;
                    decision["winner"] =
                        !decided ? "undecided"
                                 : (append_wins_here ? "append" : "reprefill");
                    decisions.push_back(std::move(decision));

                    if (!decided) {
                        still_winning = false;
                        continue;
                    }
                    if (append_wins_here) {
                        if (still_winning) {
                            append_wins_up_to = suffixes[index];
                        } else {
                            any_win_after_loss = true;
                        }
                    } else {
                        still_winning = false;
                        if (reprefill_wins_from == 0) {
                            reprefill_wins_from = suffixes[index];
                        }
                    }
                }
                if (any_win_after_loss) {
                    // No single threshold describes the policy. Design 15.6
                    // says select zero rather than the largest winning
                    // length, and that is what an unusable measurement should
                    // produce.
                    prefix_monotonic = false;
                    append_wins_up_to = 0;
                    reprefill_wins_from = 0;
                }
                json entry;
                entry["append_wins_up_to"] = append_wins_up_to;
                entry["reprefill_wins_from"] = reprefill_wins_from;
                entry["crossover_bracket"] =
                    json::array({append_wins_up_to, reprefill_wins_from});
                entry["bracket_is_tight"] =
                    reprefill_wins_from != 0 &&
                    reprefill_wins_from - append_wins_up_to <= 1;
                entry["decisions"] = std::move(decisions);
                crossover[std::to_string(history)] = std::move(entry);
                std::cout << "  crossover at history " << history
                          << ": append decisively wins to "
                          << append_wins_up_to
                          << ", reprefill decisively wins from "
                          << reprefill_wins_from << "\n";
            }
            document["continuation"] = json{
                {"points", continuation_points},
                {"crossover", crossover},
                {"prefix_monotonic", prefix_monotonic},
                {"warm_samples_per_point", options.continuation_samples},
                {"histories", histories},
                {"suffixes", suffixes},
                {"samples_interleaved", true},
                {"decision_rule",
                 "append and re-prefill samples are INTERLEAVED, so a machine "
                 "regime shift moves both routes together; a point is decided "
                 "only when the gap between the two routes' p50 exceeds BOTH "
                 "the larger within-point p50-to-p95 spread AND the larger "
                 "drift between a route's first and last sample at that "
                 "point. The reported figure is a BRACKET, not a threshold."}};

            // ---------------------------------------------------------------
            // Step 7 and Step 8: decode, with the 128-token memory window
            //
            // Three starting contexts, 128 tokens each. The LAST run is the
            // stability window, because by then every other phase has run and
            // whatever the allocator was going to do has already happened. A
            // window measured earlier would report the sweeps' settling as
            // decode's growth.
            // ---------------------------------------------------------------
            const std::vector<std::size_t> decode_contexts{128, 512, 2048};
            json decode_runs = json::array();
            json memory_window;
            for (const std::size_t start_context : decode_contexts) {
                int token = EstablishContext(engine, prompt, start_context);
                note_peak(SampleMemory());

                // Snapshot AFTER the context prefill, so the window measures
                // decode rather than the prefill that preceded it.
                const std::uint64_t tensors_before =
                    api->creation_count(
                        flm::corelib::CorelibObjectKind::Tensor);
                const std::uint64_t weights_before =
                    api->weight_creation_count();
                const std::size_t live_before = api->live_object_count();
                const MemorySample warm_memory = SampleMemory();

                std::vector<std::uint64_t> per_token;
                std::vector<MemorySample> memory_samples;
                const auto run_before = engine.metrics();
                const auto run_started = Clock::now();
                for (int step = 0; step < options.decode_tokens; ++step) {
                    const auto before = engine.metrics();
                    const auto started = Clock::now();
                    logits = engine.forward(token);
                    const std::uint64_t ns = ElapsedNs(started);
                    const auto after = engine.metrics();
                    // Per token, not in aggregate. A total that averages out
                    // to 129 x N would hide a step that dispatched a
                    // different schedule.
                    RequireOneModelStep(Delta(before, after), 1);
                    per_token.push_back(ns);
                    memory_samples.push_back(SampleMemory());
                    note_peak(memory_samples.back());
                    token = Argmax(logits);
                }
                const std::uint64_t run_ns = ElapsedNs(run_started);
                const auto run_after = engine.metrics();
                const auto run_delta = Delta(run_before, run_after);

                const std::uint64_t tensors_created =
                    api->creation_count(
                        flm::corelib::CorelibObjectKind::Tensor) -
                    tensors_before;
                const std::uint64_t weights_created =
                    api->weight_creation_count() - weights_before;
                const std::int64_t live_delta =
                    static_cast<std::int64_t>(api->live_object_count()) -
                    static_cast<std::int64_t>(live_before);

                json run = LatencySummary(per_token);
                run["start_context"] = start_context;
                run["tokens"] = options.decode_tokens;
                run["total_ns"] = run_ns;
                run["tokens_per_second"] =
                    static_cast<double>(options.decode_tokens) * 1e9 /
                    static_cast<double>(run_ns);
                run["synchronizes_per_pass"] =
                    run_delta.synchronizes /
                    static_cast<std::uint64_t>(options.decode_tokens);
                run["dispatches_per_pass"] =
                    run_delta.dispatches /
                    static_cast<std::uint64_t>(options.decode_tokens);
                run["v_bytes"] = run_delta.v_bytes;
                run["v_scatter_ns"] = run_delta.v_ns;
                run["device_tensor_creates_after_warmup"] = tensors_created;
                run["weight_creates_after_warmup"] = weights_created;
                run["live_corelib_object_delta"] = live_delta;
                decode_runs.push_back(run);

                const std::uint64_t growth =
                    memory_samples.back().private_bytes >=
                            warm_memory.private_bytes
                        ? memory_samples.back().private_bytes -
                              warm_memory.private_bytes
                        : 0;
                const double slope = PrivateByteSlope(
                    memory_samples,
                    kMemorySlopeFirstToken - 1);

                json window;
                window["source"] =
                    "decode@" + std::to_string(start_context);
                window["tokens"] = options.decode_tokens;
                window["device_tensor_creates_after_warmup"] =
                    tensors_created;
                window["weight_creates_after_warmup"] = weights_created;
                window["live_corelib_object_delta"] = live_delta;
                window["private_bytes_at_warmup"] = warm_memory.private_bytes;
                window["private_bytes_at_end"] =
                    memory_samples.back().private_bytes;
                window["private_bytes_growth"] = growth;
                // The SIGNED change as well. `growth` is clamped at zero
                // because that is what the stability bound is written
                // against, and a run whose private bytes FELL would otherwise
                // report the same 0 as one that stayed exactly flat -- two
                // different observations reading identically.
                window["private_bytes_delta"] =
                    static_cast<std::int64_t>(
                        memory_samples.back().private_bytes) -
                    static_cast<std::int64_t>(warm_memory.private_bytes);
                // Rounded away from zero, so a fractional slope can never
                // become a reported zero. A leak of 0.6 bytes per token is
                // not a leak, but reporting it as exactly none is a claim the
                // measurement does not support.
                window["private_bytes_slope_per_token"] =
                    static_cast<std::int64_t>(std::ceil(slope));
                window["private_bytes_slope_exact"] = slope;
                window["slope_first_token"] = kMemorySlopeFirstToken;
                json samples = json::array();
                for (const auto& sample : memory_samples) {
                    samples.push_back(json{
                        {"private_bytes", sample.private_bytes},
                        {"working_set_bytes", sample.working_set_bytes}});
                }
                window["samples"] = std::move(samples);
                memory_window = std::move(window);

                std::cout << "  decode @" << start_context << ": "
                          << run["tokens_per_second"].get<double>()
                          << " tok/s, private-byte slope "
                          << static_cast<std::int64_t>(std::ceil(slope))
                          << " B/token\n";
            }
            document["decode"] = json{{"runs", decode_runs}};

            // ---------------------------------------------------------------
            // Step 8: V scatter and memory
            // ---------------------------------------------------------------
            const auto final_metrics = engine.metrics();
            const MemorySample final_memory = SampleMemory();
            note_peak(final_memory);

            json v_scatter;
            // MEASURED QUOTIENTS, not the compile-time constants.
            //
            // These two fields used to be assigned kVReadsPerStep and
            // kVWritesPerStep and then CHECKed against the same constants --
            // a gate that cannot fail, rendered into the benchmark document
            // as though it were a measurement. That is this project's
            // recurring pattern for the eighth time, in the row a reader is
            // most likely to take at face value.
            //
            // They are now the observed totals divided by the observed model
            // step count, and the design contract is asserted against those.
            const std::uint64_t measured_steps =
                final_metrics.synchronize_count / kSynchronizesPerStep;
            CHECK(measured_steps > 0);
            CHECK(final_metrics.v_read_calls % measured_steps == 0);
            CHECK(final_metrics.v_write_calls % measured_steps == 0);
            const std::uint64_t measured_reads_per_step =
                final_metrics.v_read_calls / measured_steps;
            const std::uint64_t measured_writes_per_step =
                final_metrics.v_write_calls / measured_steps;
            v_scatter["reads_per_model_step"] = measured_reads_per_step;
            v_scatter["writes_per_model_step"] = measured_writes_per_step;
            v_scatter["counts_are_measured"] = true;
            v_scatter["reads_per_model_step_source"] =
                "v_read_calls / (synchronize_count / 129)";
            v_scatter["design_18_5_expected_reads"] = kVReadsPerStep;
            v_scatter["design_18_5_expected_writes"] = kVWritesPerStep;
            CHECK(measured_reads_per_step == kVReadsPerStep);
            CHECK(measured_writes_per_step == kVWritesPerStep);
            v_scatter["total_read_calls"] = final_metrics.v_read_calls;
            v_scatter["total_write_calls"] = final_metrics.v_write_calls;
            v_scatter["bytes"] = final_metrics.v_bytes;
            v_scatter["nanoseconds"] = final_metrics.v_scatter_ns;
            v_scatter["model_steps"] =
                final_metrics.synchronize_count / kSynchronizesPerStep;
            // The ratio is a contract, and it is checked over the WHOLE run
            // rather than only per step, because a single miscounted step
            // would otherwise be invisible in the totals.
            CHECK(
                final_metrics.v_write_calls ==
                final_metrics.v_read_calls *
                    static_cast<std::uint64_t>(constants::kKvHeadCount));
            CHECK(
                final_metrics.synchronize_count %
                    kSynchronizesPerStep == 0);
            CHECK(
                final_metrics.v_read_calls ==
                final_metrics.synchronize_count / kSynchronizesPerStep *
                    kVReadsPerStep);
            document["v_scatter"] = v_scatter;

            json memory = memory_window;
            memory["embedding_bytes"] = model.embedding_bytes;
            memory["kv_bytes"] = final_metrics.kv_bytes;
            memory["scratch_bytes"] = final_metrics.scratch_bytes;
            memory["packed_weight_bytes"] =
                final_metrics.packed_weight_bytes;
            memory["mapped_source_bytes"] =
                final_metrics.mapped_source_bytes;
            memory["peak_private_bytes"] = peak_private;
            memory["peak_working_set_bytes"] = peak_working_set;
            memory["private_bytes_before_load"] = before_load.private_bytes;
            memory["device_tensors_created_total"] =
                api->creation_count(flm::corelib::CorelibObjectKind::Tensor);
            memory["weight_objects_created_total"] =
                api->weight_creation_count();
            memory["device_tensors_created_after_load"] =
                api->creation_count(
                    flm::corelib::CorelibObjectKind::Tensor) -
                creates_after_load;
            memory["weight_objects_created_after_load"] =
                api->weight_creation_count() - weights_after_load;
            memory["live_corelib_objects_after_load"] = live_after_load;
            memory["live_corelib_objects_now"] = api->live_object_count();
            // Design 10.1 states these two exactly. Both sides of each
            // comparison are measured -- kv_bytes accumulates the actual
            // device allocations and embedding_bytes comes out of the
            // hash-verified manifest -- so a mismatch is a real change in the
            // memory model rather than a stale comment.
            CHECK(final_metrics.kv_bytes == kExpectedKvBytes);
            CHECK(model.embedding_bytes == kExpectedEmbeddingBytes);
            document["memory"] = memory;

            // NOTHING may have been created between load and here. Every
            // tensor and every weight object is allocated once at load
            // (design 10.1, design 18.7), so the whole post-load period is a
            // stronger statement than the 128-token window alone.
            CHECK(
                memory["device_tensors_created_after_load"]
                    .get<std::uint64_t>() == 0);
            CHECK(
                memory["weight_objects_created_after_load"]
                    .get<std::uint64_t>() == 0);
        }

        // The engine is destroyed before the runtime shuts down, so the
        // healthy path really runs: CorelibRuntime refuses cleanup() while
        // live corelib objects remain, and a leak fails here rather than
        // passing quietly.
        flm::corelib::CorelibRuntime::ShutdownProcess();

        std::ofstream output(options.output_json, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot write --output-json: " +
                options.output_json.string());
        }
        output << document.dump(2);
        if (!output) {
            throw std::runtime_error("failed while writing --output-json");
        }
        output.close();

        std::cout << "benchmark_phi4_aie4: PASS ("
                  << options.output_json.string() << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_phi4_aie4: FAIL: " << error.what() << '\n';
        return 1;
    }
}
