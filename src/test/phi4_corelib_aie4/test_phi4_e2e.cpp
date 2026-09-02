// Task 12 Steps 2, 4 and 5: explicit-token checkpoints from the real engine on
// real hardware, and a real boundary sweep.
//
// This is a harness, not a self-judging test. It takes explicit token IDs and a
// FORCED continuation route, drives `phi4_corelib_aie4` through the real
// corelib on the real device, and writes every checkpoint the Python reference
// comparator needs to a JSON file. It deliberately holds no expected values of
// its own for the numeric checks: the comparison against the reference driver
// belongs in `tools/compare_phi4_corelib_e2e.py`, and duplicating a golden here
// would just be a second copy of the same guess.
//
// The invariants it DOES assert are the ones the reference cannot supply,
// because they are properties of FastFlow's own schedule rather than of the
// model:
//
//   * exactly 129 synchronizes per model step (32 layers x 4, plus the LM
//     head). The reference driver still uses the collapsed two-synchronize
//     schedule that design Section 10.4 no longer considers sound, so this
//     count must never be compared against it;
//   * exactly 193 dispatches per model step (32 x 6, plus the LM head);
//   * 32 V-cache tensor reads and 256 per-head tensor writes per step.
//
// Two processes must never hold AIE4 device contexts at once. The suite script
// runs the Python reference to completion, including corelib.cleanup(), before
// this executable starts.

#include "test_support.hpp"

#include "picosha2.h"

#include <corelib/corelib_runtime.hpp>
#include <models/phi4/phi4_corelib_aie4.hpp>
#include <models/phi4/phi4_corelib_aie4_tuning.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>
#include <models/phi4/phi4_corelib_host.hpp>
#include <models/phi4/phi4_corelib_shape_plan.hpp>
#include <lm_config.hpp>
#include <nlohmann/json.hpp>

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <utility>
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

using flm::phi4::ContinuationRoute;
using flm::phi4::phi4_corelib_aie4;
using nlohmann::json;

// Design Section 10.4's schedule, stated as numbers so a change to it fails
// here rather than being absorbed silently.
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

// Which DLL actually serviced these calls, asked of the OS loader.
//
// Not the configured path and not the path we asked to load: the module that
// backs a resolved function pointer. `GetModuleHandleExW` from an address
// inside `get_version` answers "whose code ran", which is the only form of the
// question that can distinguish two runs that loaded different libraries.
//
// This exists because of an unexplained run-to-run divergence (report §5.1).
// The candidate hypotheses all reduce to "the two runs may not have been
// running the same code", and nothing in the artifacts could tell them apart.
// Recording the resolved path and its SHA-256 makes that testable for free,
// and the self-consistency check refuses to compare two runs that disagree
// on it.
struct LoadedModule {
    std::string path;
    std::string sha256;
};

std::string Sha256Of(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open for hashing: " + path.string());
    }
    std::vector<unsigned char> digest(picosha2::k_digest_size);
    picosha2::hash256(stream, digest.begin(), digest.end());
    return picosha2::bytes_to_hex_string(digest.begin(), digest.end());
}

LoadedModule DescribeLoadedCorelib(const flm::corelib::CorelibApi& api) {
    HMODULE module = nullptr;
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(api.functions().get_version),
            &module) == 0 ||
        module == nullptr) {
        throw std::runtime_error(
            "could not identify the module backing corelib's entry points "
            "(GetModuleHandleExW error " +
            std::to_string(GetLastError()) + ")");
    }

    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        module,
        buffer.data(),
        static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        throw std::runtime_error(
            "could not read the loaded corelib module path "
            "(GetModuleFileNameW error " +
            std::to_string(GetLastError()) + ")");
    }
    buffer.resize(length);
    const std::filesystem::path resolved(buffer);

    // The loader's answer must agree with the path we asked for. If it does
    // not, something on this machine redirected the load and every result
    // below describes a library we did not choose.
    const auto requested = api.library_path().lexically_normal();
    if (
        resolved.lexically_normal() != requested &&
        !std::filesystem::equivalent(resolved, requested)) {
        throw std::runtime_error(
            "the loaded corelib is not the one that was requested: asked "
            "for " + requested.string() + ", the loader supplied " +
            resolved.string());
    }

    std::ifstream stream(resolved, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot open the loaded corelib to hash it: " +
            resolved.string());
    }
    std::vector<unsigned char> digest(picosha2::k_digest_size);
    picosha2::hash256(stream, digest.begin(), digest.end());
    return LoadedModule{
        resolved.string(),
        picosha2::bytes_to_hex_string(digest.begin(), digest.end())};
}

struct Options {
    std::filesystem::path model_dir;
    std::filesystem::path token_ids_json;
    std::filesystem::path output_json;
    int decode_steps = 16;
    ContinuationRoute route = ContinuationRoute::Reprefill;
    bool boundary_sweep = false;
};

[[noreturn]] void Usage(std::string_view problem) {
    throw std::runtime_error(
        std::string(problem) +
        "\nusage: test_phi4_e2e"
        "\n  --model-dir <dir>"
        "\n  --token-ids-json <file>"
        "\n  --decode-steps <n>"
        "\n  --continuation-route force_append|force_reprefill"
        "\n  --output-json <file>"
        "\n  [--boundary-sweep]");
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
    bool route_seen = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--model-dir") {
            options.model_dir =
                std::filesystem::path(
                    RequireValue(argc, argv, index, argument));
        } else if (argument == "--token-ids-json") {
            options.token_ids_json =
                std::filesystem::path(
                    RequireValue(argc, argv, index, argument));
        } else if (argument == "--output-json") {
            options.output_json =
                std::filesystem::path(
                    RequireValue(argc, argv, index, argument));
        } else if (argument == "--decode-steps") {
            options.decode_steps = std::stoi(
                std::string(
                    RequireValue(argc, argv, index, argument)));
        } else if (argument == "--continuation-route") {
            const auto value =
                RequireValue(argc, argv, index, argument);
            if (value == "force_append") {
                options.route = ContinuationRoute::Append;
            } else if (value == "force_reprefill") {
                options.route = ContinuationRoute::Reprefill;
            } else {
                Usage(
                    "--continuation-route must be force_append or "
                    "force_reprefill");
            }
            route_seen = true;
        } else if (argument == "--boundary-sweep") {
            options.boundary_sweep = true;
        } else {
            Usage("unrecognized argument: " + std::string(argument));
        }
    }

    if (options.model_dir.empty()) {
        Usage("--model-dir is required");
    }
    if (options.token_ids_json.empty()) {
        Usage("--token-ids-json is required");
    }
    if (options.output_json.empty()) {
        Usage("--output-json is required");
    }
    // The route is FORCED, never inferred. Design 12.4 wants one golden per
    // route, and a default here would let a run silently produce two goldens
    // for the same route.
    if (!route_seen) {
        Usage("--continuation-route is required");
    }
    if (options.decode_steps < 1) {
        Usage("--decode-steps must be at least 1");
    }
    return options;
}

struct TokenPlan {
    std::vector<int> prefix;
    std::vector<int> suffix;
};

// Accepts either a flat array of IDs (all prefix, no continuation) or an
// object with "prefix" and "suffix". The two-part form is what makes a route
// meaningful: with no suffix there is nothing to append and both routes
// degenerate to the same single prefill.
TokenPlan LoadTokenPlan(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error(
            "cannot open --token-ids-json: " + path.string());
    }
    json document;
    stream >> document;

    TokenPlan plan;
    if (document.is_array()) {
        plan.prefix = document.get<std::vector<int>>();
    } else if (document.is_object()) {
        plan.prefix = document.at("prefix").get<std::vector<int>>();
        if (document.contains("suffix")) {
            plan.suffix = document.at("suffix").get<std::vector<int>>();
        }
    } else {
        throw std::runtime_error(
            "--token-ids-json must hold an array or an object with "
            "\"prefix\" and \"suffix\"");
    }
    if (plan.prefix.empty()) {
        throw std::runtime_error(
            "--token-ids-json prefix must hold at least one token");
    }
    return plan;
}

std::vector<std::uint16_t> LogitBits(const buffer<bf16>& logits) {
    std::vector<std::uint16_t> bits(logits.size());
    const auto* const raw =
        reinterpret_cast<const std::uint16_t*>(logits.data());
    std::copy_n(raw, logits.size(), bits.begin());
    return bits;
}

// Widened here rather than in the comparator only for the ranking below; the
// full BF16 bit pattern is what the JSON carries, so the comparator does its
// own widening and nothing depends on this conversion being reproduced.
float WidenBf16(std::uint16_t bits) {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

struct Ranking {
    std::vector<int> ids;
    std::vector<float> values;
};

// Ties break toward the LOWEST ID, matching flm::phi4::ArgmaxLowest. The
// comparator asserts that behaviour explicitly, so the ordering used to
// produce the top-k here has to be the same one or the check would be
// comparing two different conventions.
Ranking TopK(const std::vector<std::uint16_t>& bits, std::size_t k) {
    std::vector<int> order(bits.size());
    std::iota(order.begin(), order.end(), 0);
    const std::size_t count = std::min(k, order.size());
    std::partial_sort(
        order.begin(),
        order.begin() + static_cast<std::ptrdiff_t>(count),
        order.end(),
        [&](int left, int right) {
            const float left_value =
                WidenBf16(bits[static_cast<std::size_t>(left)]);
            const float right_value =
                WidenBf16(bits[static_cast<std::size_t>(right)]);
            if (left_value != right_value) {
                return left_value > right_value;
            }
            return left < right;
        });
    order.resize(count);

    Ranking ranking;
    ranking.ids = order;
    ranking.values.reserve(count);
    for (const int id : order) {
        ranking.values.push_back(
            WidenBf16(bits[static_cast<std::size_t>(id)]));
    }
    return ranking;
}

json StepRecord(
    const phi4_corelib_aie4& engine,
    const buffer<bf16>& logits,
    const flm::phi4::Phi4Aie4Metrics& before,
    const flm::phi4::Phi4Aie4Metrics& after,
    std::uint64_t model_steps_in_this_call) {
    const auto bits = LogitBits(logits);
    const Ranking top32 = TopK(bits, 32);
    const Ranking top5 = TopK(bits, 5);

    const std::span<const bf16> logit_span(
        logits.data(),
        logits.size());
    const int argmax = flm::phi4::ArgmaxLowest(logit_span);

    // The top-1 from the ranking and ArgmaxLowest must agree. If they ever
    // did not, every downstream comparison would be against a token the
    // engine would not actually have emitted.
    CHECK(!top32.ids.empty());
    CHECK(top32.ids.front() == argmax);

    json record;
    record["top1_id"] = argmax;
    record["top5_ids"] = top5.ids;
    record["top32_ids"] = top32.ids;
    record["top32_values"] = top32.values;
    record["logits_bf16"] = bits;
    // THE LM-HEAD INPUT FOR THIS STEP, captured rather than inferred.
    //
    // DETERM-1 localised the run-to-run divergence to the 3072 x 200064
    // dispatch on the argument "identical LM-head input, non-identical
    // LM-head output". The input half of that was never observed: it was
    // inferred from END-OF-RUN state being identical, at a step that is not
    // the end of the run. With this field the comparator can ask the question
    // directly at the step that actually diverged, and Task 13 measured the
    // answer to be the opposite of the inference.
#ifdef DEV_BUILD
    record["lm_head_input_bf16"] = engine.debug_lm_head_input();
#else
    (void)engine;
#endif
    record["dispatch_delta"] =
        after.dispatch_count - before.dispatch_count;
    record["synchronize_delta"] =
        after.synchronize_count - before.synchronize_count;
    record["v_read_delta"] =
        after.v_read_calls - before.v_read_calls;
    record["v_write_delta"] =
        after.v_write_calls - before.v_write_calls;
    record["model_steps"] = model_steps_in_this_call;

    // Design Section 10.4 and Section 5.2, asserted rather than merely
    // reported. A run that emitted the wrong count and left the judgement to
    // a human reading JSON would be a report, not a test.
    CHECK(
        record["synchronize_delta"].get<std::uint64_t>() ==
        kSynchronizesPerStep * model_steps_in_this_call);
    CHECK(
        record["dispatch_delta"].get<std::uint64_t>() ==
        kDispatchesPerStep * model_steps_in_this_call);
    CHECK(
        record["v_read_delta"].get<std::uint64_t>() ==
        kVReadsPerStep * model_steps_in_this_call);
    CHECK(
        record["v_write_delta"].get<std::uint64_t>() ==
        kVWritesPerStep * model_steps_in_this_call);
    return record;
}

// One "model step" is one RunRows call. Append walks the suffix one token at a
// time, which is one step each; re-prefill recomputes the whole history in a
// single step. That difference is the whole point of the two routes, so the
// harness counts them explicitly rather than inferring them from the metrics
// it is trying to check.
struct RouteResult {
    buffer<bf16> logits;
    std::uint64_t model_steps = 0;
};

RouteResult RunForcedRoute(
    phi4_corelib_aie4& engine,
    const TokenPlan& plan,
    ContinuationRoute route) {
    RouteResult result;
    if (route == ContinuationRoute::Append) {
        // The prefix is already the model's history in the product; here it
        // is established by one prefill, and only the suffix exercises the
        // append path. `prefill` itself walks a multi-token continuation one
        // row at a time, so the suffix is fed through `forward` to keep the
        // step count unambiguous.
        std::vector<int> prefix = plan.prefix;
        result.logits = engine.prefill(prefix);
        ++result.model_steps;
        for (const int token : plan.suffix) {
            result.logits = engine.forward(token);
            ++result.model_steps;
        }
        return result;
    }

    // Re-prefill clears the conversation and recomputes the full rendered
    // history from position zero, which is one step over prefix+suffix rows.
    engine.clear_context();
    std::vector<int> full = plan.prefix;
    full.insert(full.end(), plan.suffix.begin(), plan.suffix.end());
    result.logits = engine.prefill(full);
    ++result.model_steps;
    return result;
}

json SnapshotRecord(const phi4_corelib_aie4& engine) {
#ifdef DEV_BUILD
    const auto snapshot = engine.debug_snapshot();
    json record;
    record["live_rows"] = snapshot.live_rows;
    record["position"] = snapshot.position;
    record["layer0_k"] = snapshot.layer0_k;
    record["layer0_v"] = snapshot.layer0_v;
    record["layer31_k"] = snapshot.layer31_k;
    record["layer31_v"] = snapshot.layer31_v;
    record["last_hidden"] = snapshot.last_hidden;

    // Live K/V only. The caches are allocated at the full 4096-row window but
    // only [0, position) has been written; the rest is uninitialised device
    // memory, and comparing it against the reference would be comparing
    // garbage to garbage and calling it agreement.
    const std::size_t live =
        static_cast<std::size_t>(constants::kKvHeadCount) *
        static_cast<std::size_t>(snapshot.position) *
        static_cast<std::size_t>(constants::kHeadSize);
    CHECK(snapshot.layer0_k.size() == live);
    CHECK(snapshot.layer0_v.size() == live);
    CHECK(snapshot.layer31_k.size() == live);
    CHECK(snapshot.layer31_v.size() == live);
    return record;
#else
    (void)engine;
    throw std::runtime_error(
        "test_phi4_e2e must be built with DEV_BUILD=1; without it "
        "debug_snapshot() does not exist and there are no K/V "
        "checkpoints to emit");
#endif
}

json MetricsRecord(const flm::phi4::Phi4Aie4Metrics& metrics) {
    json record;
    record["dispatch_count"] = metrics.dispatch_count;
    record["synchronize_count"] = metrics.synchronize_count;
    record["v_read_calls"] = metrics.v_read_calls;
    record["v_write_calls"] = metrics.v_write_calls;
    record["v_bytes"] = metrics.v_bytes;
    record["device_tensor_create_count"] =
        metrics.device_tensor_create_count;
    record["weight_create_count"] = metrics.weight_create_count;
    record["padding_write_calls"] = metrics.padding_write_calls;
    record["padding_bytes"] = metrics.padding_bytes;
    record["packed_weight_bytes"] = metrics.packed_weight_bytes;
    record["mapped_source_bytes"] = metrics.mapped_source_bytes;
    record["kv_bytes"] = metrics.kv_bytes;
    record["scratch_bytes"] = metrics.scratch_bytes;
    record["helper_transition_counts"] =
        metrics.helper_transition_counts;
    record["attention_extent_queries"] =
        metrics.attention_extent_queries;
    record["output_projection_extent_queries"] =
        metrics.output_projection_extent_queries;
    record["lm_head_extent_queries"] = metrics.lm_head_extent_queries;
    return record;
}

// Step 5 on real hardware. The probe rows come from the running helper table,
// not from a transcribed grid, and each one is a REAL prefill: a pad-shape
// query alone would not show that the device accepts the padded extent the
// plan chose.
json RunBoundarySweep(
    phi4_corelib_aie4& engine,
    const std::vector<int>& vocabulary_sample,
    const std::vector<std::int64_t>& probe_rows) {
    json sweep = json::array();
    for (const std::int64_t rows : probe_rows) {
        engine.clear_context();
        std::vector<int> tokens(
            static_cast<std::size_t>(rows));
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            tokens[index] =
                vocabulary_sample[index % vocabulary_sample.size()];
        }

        const auto before = engine.metrics();
        const auto logits = engine.prefill(tokens);
        const auto after = engine.metrics();

        const auto bits = LogitBits(logits);
        const Ranking top5 = TopK(bits, 5);
        json record;
        record["rows"] = rows;
        record["top1_id"] = top5.ids.front();
        record["top5_ids"] = top5.ids;
        record["synchronize_delta"] =
            after.synchronize_count - before.synchronize_count;
        record["dispatch_delta"] =
            after.dispatch_count - before.dispatch_count;
        // Every prefill is ONE model step regardless of how many rows it
        // carries, so the schedule counts do not vary with the row count.
        CHECK(
            record["synchronize_delta"].get<std::uint64_t>() ==
            kSynchronizesPerStep);
        CHECK(
            record["dispatch_delta"].get<std::uint64_t>() ==
            kDispatchesPerStep);
        CHECK(engine.get_current_context_length() == rows);
        sweep.push_back(std::move(record));
        std::cout << "  boundary rows " << rows << ": ok\n";
    }
    return sweep;
}

// One below, at, and above every transition the running helper table reports,
// plus a fresh row 1 and the low-level row 4096. The transitions come from
// Phi4ShapePlan, which derived them from the library; nothing here is a
// transcribed grid, so a library that changed its buckets changes the probes
// rather than slipping past them.
std::vector<std::int64_t> BoundaryProbeRows(
    const flm::phi4::Phi4ShapePlan& plan) {
    std::vector<std::int64_t> rows{1, constants::kMaxSequenceLength};
    for (const auto& [live_rows, padded_rows] :
         plan.Transitions(flm::phi4::RowUse::Attention)) {
        (void)padded_rows;
        for (const std::int64_t offset : {-1, 0, 1}) {
            const std::int64_t probe = live_rows + offset;
            if (probe >= 1 && probe <= constants::kMaxSequenceLength) {
                rows.push_back(probe);
            }
        }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
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
        const TokenPlan plan = LoadTokenPlan(options.token_ids_json);

        // RYZENAI_CORELIB_PATH selects the DLL. The suite script points it at
        // the staged closure; there is deliberately no fallback that would
        // load whatever DLL happened to be on PATH.
        const std::filesystem::path executable_dir =
            std::filesystem::absolute(argv[0]).parent_path();
        auto runtime =
            flm::corelib::CorelibRuntime::GetOrCreate(executable_dir);
        std::cout << "corelib runtime ready: "
                  << runtime->api()->library_path().string() << '\n';

        LM_Config config;
        config._json_config = json::object();
        config.model_path = options.model_dir.string();
        config.model_name = options.model_dir.filename().string();

        const LoadedModule loaded =
            DescribeLoadedCorelib(*runtime->api());
        std::cout << "loaded corelib: " << loaded.path << '\n'
                  << "sha256        : " << loaded.sha256 << '\n';

        json document;
        document["model_dir"] = options.model_dir.string();
        document["corelib_loaded_path"] = loaded.path;
        document["corelib_sha256"] = loaded.sha256;
        // I-7: WHICH FASTFLOW BINARY. The artifacts recorded the corelib
        // DLL's hash and nothing at all about the harness that drove it, so
        // two records could describe different FastFlow builds and no check
        // could tell. A determinism baseline pooled across a build tree needs
        // both halves.
        document["harness_path"] =
            std::filesystem::absolute(argv[0]).string();
        document["harness_sha256"] =
            Sha256Of(std::filesystem::absolute(argv[0]));
        document["corelib_library"] =
            runtime->api()->library_path().string();
        document["corelib_version"] =
            flm::corelib::FormatCorelibVersion(
                runtime->api()->runtime_version());
        document["continuation_route"] =
            flm::phi4::ContinuationRouteName(options.route);
        document["prefix_ids"] = plan.prefix;
        document["suffix_ids"] = plan.suffix;
        document["decode_steps_requested"] = options.decode_steps;

        {
            phi4_corelib_aie4 engine(
                config,
                options.model_dir,
                runtime,
                static_cast<std::uint32_t>(
                    constants::kMaxSequenceLength));

            document["load_metrics"] = MetricsRecord(engine.metrics());

            if (options.boundary_sweep) {
                const auto sweep_plan =
                    flm::phi4::Phi4ShapePlan::Build(runtime->api());
                // Rows are sampled out of the prompt so each one carries a
                // real embedding: a constant token would satisfy every shape
                // check while masking a row-indexing defect.
                document["boundary_sweep"] = RunBoundarySweep(
                    engine,
                    plan.prefix,
                    BoundaryProbeRows(sweep_plan));
                engine.clear_context();
            }

            const auto before_route = engine.metrics();
            RouteResult route_result =
                RunForcedRoute(engine, plan, options.route);
            const auto after_route = engine.metrics();
            document["continuation"] = StepRecord(
                engine,
                route_result.logits,
                before_route,
                after_route,
                route_result.model_steps);

            json decode = json::array();
            int token = flm::phi4::ArgmaxLowest(
                std::span<const bf16>(
                    route_result.logits.data(),
                    route_result.logits.size()));
            for (int step = 0; step < options.decode_steps; ++step) {
                const auto before = engine.metrics();
                const auto logits = engine.forward(token);
                const auto after = engine.metrics();
                json record =
                    StepRecord(engine, logits, before, after, 1u);
                record["input_id"] = token;
                token = record["top1_id"].get<int>();
                decode.push_back(std::move(record));
            }
            document["decode"] = std::move(decode);
            document["final_snapshot"] = SnapshotRecord(engine);
            document["final_metrics"] = MetricsRecord(engine.metrics());
            document["final_position"] =
                engine.get_current_context_length();
        }

        // The engine is destroyed before the runtime shuts down, so the
        // healthy path really runs: CorelibRuntime refuses cleanup() while
        // live corelib objects remain, and a leak would fail here rather
        // than pass quietly.
        flm::corelib::CorelibRuntime::ShutdownProcess();

        std::ofstream output(options.output_json, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot write --output-json: " +
                options.output_json.string());
        }
        output << document.dump();
        if (!output) {
            throw std::runtime_error(
                "failed while writing --output-json");
        }
        output.close();

        std::cout << "test_phi4_e2e: PASS ("
                  << options.output_json.string() << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_phi4_e2e: FAIL: " << error.what() << '\n';
        return 1;
    }
}
