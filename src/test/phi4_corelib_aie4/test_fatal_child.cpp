// Task 12 Step 6: the terminal-failure path, in a real process that really
// dies.
//
// Everything before this ran the failure policy in-process against an
// intercepted terminator, so nothing had ever confirmed that a FastFlow process
// actually exits with 0xE0040001, that the detailed record actually reaches
// %LOCALAPPDATA%\FastFlowLM\logs, or that a parent can actually read it back.
// Those are the three things this file establishes, by forking itself.
//
// The record root is deliberately the REAL one. FatalRecordStore resolves it
// with SHGetKnownFolderPath rather than from the environment, so it cannot be
// redirected -- and redirecting it would remove the very precondition design
// Section 12.1 asks about, which is whether that directory is writable on the
// target.
//
// What is injected and what is not, stated plainly because it bounds the
// claim: the FAILURE is real -- a genuine call into the real corelib that the
// real library genuinely refuses, carrying its own status code and its own
// message. The SUBMISSION STATE is synthetic. There is no way to make the
// device fail after a successful submit on demand without corrupting the
// device, so the irrevocable-boundary cases set that flag directly and then
// let the unmodified policy decide. The policy, the record, the exit code and
// the parent-side drain are all the production ones.

#include "test_support.hpp"

#include <corelib/corelib_api.hpp>
#include <corelib/corelib_fatal_record.hpp>
#include <corelib/corelib_object.hpp>
#include <corelib/corelib_runtime.hpp>
#include <models/phi4/phi4_corelib_aie4.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>

#include <windows.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace constants = flm::phi4::constants;

using flm::corelib::CorelibApi;
using flm::corelib::CorelibError;
using flm::corelib::CorelibRuntime;
using flm::corelib::FatalRecordStore;
using flm::corelib::StepSubmissionState;

constexpr unsigned int kFatalExitCode = 0xE0040001u;

// Markers are FILES, not stdout lines, and each one is written, flushed and
// closed before the child proceeds.
//
// The first version of this test looked for the absence of a `std::cout` line.
// That check could not fail: `TerminateProcess` discards buffered stdout, so a
// child that had run cleanup and printed the marker would still show no marker
// once it was killed. "Absent" and "never written" were indistinguishable, and
// the assertion passed for the wrong reason.
//
// A closed file survives `TerminateProcess`, so absence now means the child
// really did not get there. `kReachedPolicy` is the positive control that
// makes the absence meaningful: without it, a child that died during startup
// would satisfy "no cleanup marker" just as well as one that was correctly
// terminated mid-step.
constexpr std::wstring_view kReachedPolicyMarker = L"reached-policy.marker";
constexpr std::wstring_view kCleanupMarker = L"cleanup-ok.marker";
constexpr std::wstring_view kSurvivedMarker = L"survived.marker";

// Where the child drops its markers, passed on the command line so the parent
// and child cannot disagree about it.
std::filesystem::path g_marker_dir;

void WriteMarker(std::wstring_view name, std::string_view contents) {
    const auto path = g_marker_dir / std::filesystem::path(name);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error(
            "child could not write marker " + path.string());
    }
    stream << contents;
    stream.flush();
    stream.close();
    if (!stream) {
        throw std::runtime_error(
            "child could not flush marker " + path.string());
    }
}

#if !defined(FLM_REAL_CORELIB_RUNTIME_DIR)
#define FLM_REAL_CORELIB_RUNTIME_DIR ""
#endif

#if !defined(FLM_REAL_CORELIB_EXTRA_DLL_DIRS)
#define FLM_REAL_CORELIB_EXTRA_DLL_DIRS ""
#endif

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
            if (
                !std::filesystem::exists(directory) ||
                AddDllDirectory(directory.c_str()) == nullptr) {
                throw std::runtime_error(
                    "cannot add corelib DLL directory " +
                    directory.string());
            }
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
}

std::filesystem::path CorelibLibraryPath() {
    const std::string runtime_dir(FLM_REAL_CORELIB_RUNTIME_DIR);
    return std::filesystem::absolute(
               std::filesystem::path(runtime_dir) /
               "ryzenai_corelib.dll")
        .lexically_normal();
}

// A real corelib call that the real library really refuses.
//
// The LM-head MatMul shape ships kernels at M in {1, 128} and ERRORS above
// 128 rather than rounding up -- measured, and asserted separately in
// test_phi4_hardware. Using it here means the CorelibError carries the
// library's own status code and its own last-error message, so the record the
// parent reads back is a real diagnostic rather than a string this file made
// up.
CorelibError RealCorelibFailure(const std::shared_ptr<CorelibApi>& api) {
    std::int64_t m = constants::kMaxSequenceLength;
    std::int64_t k = constants::kHiddenSize;
    std::int64_t n = constants::kVocabularySize;
    try {
        api->Check(
            api->functions().matmul_pad_shape(
                &m,
                &k,
                &n,
                constants::kGroupSize),
            "ryzenai_corelib_matmul_bf16_pad_shape");
    } catch (const CorelibError& error) {
        return error;
    }
    throw std::runtime_error(
        "the real corelib accepted an LM-head M of " +
        std::to_string(constants::kMaxSequenceLength) +
        ", so this file has no genuine failure to drive the policy with. "
        "That is a finding about the library, not a defect here.");
}

// ---------------------------------------------------------------------------
// Child
// ---------------------------------------------------------------------------

int RunChild(std::string_view scenario) {
    AddExtraDllDirectories(FLM_REAL_CORELIB_EXTRA_DLL_DIRS);

    // GetOrCreate builds the production runtime: the real DLL, the real
    // FatalRecordStore rooted in LocalAppData, and the real terminator that
    // calls TerminateProcess. Nothing is substituted.
    const auto library = CorelibLibraryPath();
    SetEnvironmentVariableW(
        L"RYZENAI_CORELIB_PATH",
        library.c_str());
    auto runtime = CorelibRuntime::GetOrCreate(library.parent_path());
    CHECK(
        runtime->state() == flm::corelib::ProcessState::Healthy);

    const CorelibError failure = RealCorelibFailure(runtime->api());
    std::cout << "child scenario=" << scenario
              << " status=" << static_cast<int>(failure.status)
              << " call=" << failure.call << '\n';

    // The library's OWN status and message, handed to the parent so it can
    // check the record against what corelib actually said rather than against
    // a literal this test invented. Written before the policy runs, because
    // two of the three scenarios never return from it.
    WriteMarker(
        L"expected-status.marker",
        std::to_string(static_cast<int>(failure.status)));
    WriteMarker(L"expected-detail.marker", failure.detail);
    WriteMarker(L"expected-call.marker", failure.call);

    StepSubmissionState submission;
    bool synchronize_in_progress = false;
    std::string phase = "qkv";
    if (scenario == "before_submit") {
        // Nothing has been submitted and no synchronize is running, so the
        // policy must RETHROW and leave the process alive.
        phase = "qkv";
    } else if (scenario == "after_submit") {
        // q submitted successfully, k failed. Past the irrevocable boundary.
        submission.MarkSuccessfulSubmit();
        phase = "qkv";
    } else if (scenario == "synchronize") {
        // A failing synchronize is irrevocable even with nothing marked as
        // submitted: the device may already be mid-flight.
        synchronize_in_progress = true;
        phase = "flat_mha";
    } else {
        throw std::runtime_error(
            "unknown child scenario: " + std::string(scenario));
    }

    // The positive control. Written and closed immediately before the policy
    // is entered, so the parent can tell "terminated inside the policy" from
    // "died on the way there" -- both of which leave no cleanup marker, and
    // only one of which is the behaviour under test.
    WriteMarker(kReachedPolicyMarker, std::string(scenario));

    bool rethrown = false;
    try {
        flm::phi4::testing::ApplyCorelibFailurePolicyForTest(
            runtime,
            failure,
            synchronize_in_progress,
            submission,
            phase,
            /*layer=*/7,
            /*rows=*/13,
            /*position=*/29);
    } catch (const CorelibError&) {
        rethrown = true;
    }

    // Only "before_submit" can reach here. The other two scenarios died
    // inside the policy, so if control returns for them the policy did not do
    // what design Section 12.1 requires and the child fails loudly rather
    // than exiting 0.
    if (scenario != "before_submit") {
        std::cerr << "child scenario=" << scenario
                  << " survived an irrevocable failure\n";
        return 2;
    }
    CHECK(rethrown);

    // Admission is still open and the runtime is still healthy.
    CHECK(runtime->admission_open());
    CHECK(runtime->state() == flm::corelib::ProcessState::Healthy);

    // "The session clears" has to mean the session is USABLE again, not just
    // that a flag says Healthy. So take a fresh execution lease and put a real
    // operation through the real library: create a Stream and a DeviceTensor,
    // write and read an element, and release them. If a recoverable failure
    // had left the runtime wedged, this is where it would show.
    {
        auto execution = runtime->AcquireExecution();
        const auto& api = runtime->api();
        ryzenai_corelib_stream_ptr raw_stream = nullptr;
        api->Check(
            api->functions().create_stream(&raw_stream),
            "ryzenai_corelib_create_stream");
        flm::corelib::UniqueStream stream(api, raw_stream);

        const std::array<std::int64_t, 2> shape{1, 64};
        ryzenai_corelib_tensor_ptr raw_tensor = nullptr;
        api->Check(
            api->functions().create_device_tensor(
                ryzenai_corelib_data_type_bf16,
                shape.data(),
                shape.size(),
                &raw_tensor),
            "ryzenai_corelib_create_device_tensor");
        flm::corelib::UniqueTensor tensor(api, raw_tensor);

        const std::array<float, 4> source{1.0f, 2.0f, 4.0f, 8.0f};
        api->WriteElements(
            tensor.get(),
            ryzenai_corelib_data_type_fp32,
            source.data(),
            source.size(),
            0);
        std::array<float, 4> destination{};
        api->ReadElements(
            tensor.get(),
            ryzenai_corelib_data_type_fp32,
            destination.data(),
            destination.size(),
            0);
        CHECK(destination == source);

        api->Check(
            api->functions().stream_synchronize(stream.get()),
            "ryzenai_corelib_stream_synchronize");
        tensor.reset();
        stream.reset();
    }
    CHECK(runtime->api()->live_object_count() == 0);
    WriteMarker(kSurvivedMarker, "session usable after a pre-submit failure");

    CorelibRuntime::ShutdownProcess();
    WriteMarker(kCleanupMarker, "healthy shutdown completed");
    return 0;
}

// ---------------------------------------------------------------------------
// Parent
// ---------------------------------------------------------------------------

class TempDirectory final {
public:
    TempDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("fastflowlm-fatal-child-" +
                 std::to_string(GetCurrentProcessId()));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() noexcept {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

struct ChildResult {
    DWORD exit_code = 0;
    std::string output;
};

ChildResult RunScenario(
    const std::filesystem::path& executable,
    const std::filesystem::path& log_path,
    const std::filesystem::path& marker_dir,
    std::string_view scenario) {
    // Each scenario gets a fresh marker directory, so a marker left by an
    // earlier scenario cannot be read as this one's evidence.
    std::error_code error;
    std::filesystem::remove_all(marker_dir, error);
    std::filesystem::create_directories(marker_dir);

    SECURITY_ATTRIBUTES inheritable{};
    inheritable.nLength = sizeof(inheritable);
    inheritable.bInheritHandle = TRUE;

    HANDLE log = CreateFileW(
        log_path.c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ,
        &inheritable,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (log == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            "cannot create child log " + log_path.string());
    }

    std::wstring command = L"\"" + executable.wstring() +
                           L"\" --child " +
                           std::wstring(
                               scenario.begin(),
                               scenario.end()) +
                           L" \"" + marker_dir.wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = log;
    startup.hStdError = log;
    startup.hStdInput = nullptr;

    PROCESS_INFORMATION process{};
    const BOOL created = CreateProcessW(
        executable.c_str(),
        command.data(),
        nullptr,
        nullptr,
        TRUE,
        0,
        nullptr,
        nullptr,
        &startup,
        &process);
    CloseHandle(log);
    if (created == FALSE) {
        throw std::runtime_error(
            "CreateProcessW failed for scenario " +
            std::string(scenario) + " (error " +
            std::to_string(GetLastError()) + ")");
    }

    ChildResult result;
    // Generous, but bounded: the child loads the real corelib, and a hang
    // must fail the suite rather than wedge it.
    if (WaitForSingleObject(process.hProcess, 300000) != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 1);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        throw std::runtime_error(
            "child scenario " + std::string(scenario) + " did not exit");
    }
    GetExitCodeProcess(process.hProcess, &result.exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);

    std::ifstream stream(log_path, std::ios::binary);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    result.output = buffer.str();
    return result;
}

bool Contains(std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
}

// Field extraction over the record's own on-disk text rather than a JSON
// parser, so the check is against exactly the bytes a support engineer would
// receive. The record writer emits compact JSON with no spaces around the
// colon, which is what makes this reliable.
std::string FieldValue(
    std::string_view record,
    std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\":";
    const auto start = record.find(needle);
    if (start == std::string_view::npos) {
        throw std::runtime_error(
            "fatal record is missing the field \"" + std::string(key) +
            "\": " + std::string(record));
    }
    auto cursor = start + needle.size();
    if (cursor < record.size() && record[cursor] == '"') {
        ++cursor;
        // Unescaped, because the record writer escapes what it stores and the
        // library's own messages carry Windows paths full of backslashes. A
        // raw comparison would fail on `C:\\Users\\...` against `C:\Users\...`
        // and look like a content mismatch rather than an encoding one.
        std::string value;
        while (cursor < record.size() && record[cursor] != '"') {
            if (record[cursor] == '\\' && cursor + 1 < record.size()) {
                ++cursor;
                switch (record[cursor]) {
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    case 'b': value.push_back('\b'); break;
                    case 'f': value.push_back('\f'); break;
                    case 'u': {
                        // The writer only emits \u for control characters.
                        const auto digits = record.substr(cursor + 1, 4);
                        value.push_back(static_cast<char>(
                            std::stoi(std::string(digits), nullptr, 16)));
                        cursor += 4;
                        break;
                    }
                    default: value.push_back(record[cursor]); break;
                }
                ++cursor;
                continue;
            }
            value.push_back(record[cursor]);
            ++cursor;
        }
        return value;
    }
    const auto end = record.find_first_of(",}", cursor);
    return std::string(record.substr(cursor, end - cursor));
}

std::string ReadMarker(
    const std::filesystem::path& marker_dir,
    std::wstring_view name) {
    std::ifstream stream(
        marker_dir / std::filesystem::path(name),
        std::ios::binary);
    if (!stream) {
        return {};
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

bool MarkerExists(
    const std::filesystem::path& marker_dir,
    std::wstring_view name) {
    return std::filesystem::exists(
        marker_dir / std::filesystem::path(name));
}

void CheckDetailedRecord(
    std::string_view record,
    std::string_view expected_phase,
    const std::filesystem::path& marker_dir) {
    // "Complete" means every field design Section 12.1 lists is present and
    // carries the right value -- not merely that a file exists.
    //
    // `status`, `call` and `detail` are checked against what the LIBRARY
    // produced, read out of markers the child wrote from the real
    // CorelibError. Checking them for non-emptiness, as this did before, would
    // pass for a record that carried the wrong diagnostic entirely.
    const auto expected_status =
        ReadMarker(marker_dir, L"expected-status.marker");
    const auto expected_detail =
        ReadMarker(marker_dir, L"expected-detail.marker");
    const auto expected_call =
        ReadMarker(marker_dir, L"expected-call.marker");
    CHECK(!expected_status.empty());
    CHECK(!expected_detail.empty());
    CHECK(!expected_call.empty());

    CHECK(FieldValue(record, "status") == expected_status);
    CHECK(FieldValue(record, "call") == expected_call);
    CHECK(FieldValue(record, "detail") == expected_detail);

    // phase/layer/rows/position are values this test chose, so what they
    // establish is narrower: that FailureContext survives serialisation
    // intact, field for field, including the optional layer. They do NOT
    // establish that the engine supplies the right ones -- reaching the
    // production catch arm needs a fault injection point the engine does not
    // have. See the report's note on I4.
    CHECK(FieldValue(record, "phase") == expected_phase);
    CHECK(FieldValue(record, "layer") == "7");
    CHECK(FieldValue(record, "rows") == "13");
    CHECK(FieldValue(record, "position") == "29");

    CHECK(!FieldValue(record, "pid").empty());
    CHECK(!FieldValue(record, "failure_utc").empty());
    CHECK(!FieldValue(record, "process_start_utc").empty());
}

int RunParent(const std::filesystem::path& executable) {
    TempDirectory logs;

    // Start from a clean slate so a record left by an earlier run cannot be
    // mistaken for this run's evidence.
    std::ostringstream discarded;
    FatalRecordStore::DrainPriorRecords(discarded);

    {
        const auto markers = logs.path() / "before_submit.markers";
        const auto result = RunScenario(
            executable,
            logs.path() / "before_submit.log",
            markers,
            "before_submit");
        std::cout << "before_submit exit=0x" << std::hex
                  << result.exit_code << std::dec << '\n';
        CHECK(result.exit_code == 0);
        CHECK(MarkerExists(markers, kReachedPolicyMarker));
        // The session is usable again: the child put a real Stream, tensor
        // write, read and synchronize through the library after the
        // recoverable failure.
        CHECK(MarkerExists(markers, kSurvivedMarker));
        CHECK(MarkerExists(markers, kCleanupMarker));

        // A recoverable failure writes NO record. The child's healthy
        // shutdown removes its own pending file, so a drain here must come
        // back empty.
        std::ostringstream drained;
        const auto records =
            FatalRecordStore::DrainPriorRecords(drained);
        if (!records.empty()) {
            throw std::runtime_error(
                "a pre-submit failure left a fatal record behind: " +
                records.front());
        }
    }

    for (const auto& [scenario, phase] :
         std::array<std::pair<std::string_view, std::string_view>, 2>{
             {{"after_submit", "qkv"}, {"synchronize", "flat_mha"}}}) {
        const auto markers =
            logs.path() / (std::string(scenario) + ".markers");
        const auto result = RunScenario(
            executable,
            logs.path() / (std::string(scenario) + ".log"),
            markers,
            scenario);
        std::cout << scenario << " exit=0x" << std::hex
                  << result.exit_code << std::dec << '\n';
        CHECK(result.exit_code == kFatalExitCode);

        // The positive control first. Without it, "no cleanup marker" is also
        // satisfied by a child that died before it ever reached the policy,
        // and the absence below would prove nothing.
        CHECK(MarkerExists(markers, kReachedPolicyMarker));
        CHECK(
            ReadMarker(markers, kReachedPolicyMarker) ==
            std::string(scenario));

        // Normal cleanup must NOT have run. These are closed files, so they
        // survive TerminateProcess and their absence is real -- unlike the
        // buffered stdout this check used to read, which TerminateProcess
        // discards, making the assertion pass whether cleanup had run or not.
        CHECK(!MarkerExists(markers, kCleanupMarker));
        CHECK(!MarkerExists(markers, kSurvivedMarker));
        // The terminal diagnostic goes to stderr before the process dies,
        // and it must name the record it wrote.
        CHECK(Contains(result.output, "AIE4 terminal failure"));
        CHECK(Contains(result.output, "AIE4 fatal record:"));

        std::ostringstream drained;
        const auto records =
            FatalRecordStore::DrainPriorRecords(drained);
        if (records.size() != 1) {
            throw std::runtime_error(
                "expected exactly one fatal record from scenario " +
                std::string(scenario) + ", drained " +
                std::to_string(records.size()));
        }
        CheckDetailedRecord(records.front(), phase, markers);
        // The drain reports what it removed, so the record reaches an
        // operator rather than only the filesystem.
        CHECK(Contains(drained.str(), "\"phase\":\"" + std::string(phase)));
        std::cout << "  record: " << records.front();
    }

    std::cout << "test_fatal_child: PASS\n";
    return 0;
}

}  // namespace

constexpr int kCTestSkipReturnCode = 77;

// Same opt-in as test_phi4_hardware, and for the same reason: a configured
// runtime directory names a DLL, it does not assert that this machine is the
// AIE4 target. GetOrCreate refuses to build a runtime without a device
// context, so without the flag this would fail on every development box.
bool HardwareRunRequested() {
    char value[8] = {};
    const DWORD length = GetEnvironmentVariableA(
        "FLM_AIE4_HARDWARE",
        value,
        sizeof(value));
    return length != 0 && length < sizeof(value) &&
           std::string_view(value) == "1";
}

int main(int argc, char** argv) {
    const std::string runtime_dir(FLM_REAL_CORELIB_RUNTIME_DIR);
    if (runtime_dir.empty()) {
        std::cout
            << "test_fatal_child: SKIPPED -- configure with "
               "-DRYZENAI_CORELIB_RUNTIME_DIR and run on the AIE4 "
               "target.\n";
        return kCTestSkipReturnCode;
    }
    if (!HardwareRunRequested()) {
        std::cout
            << "test_fatal_child: SKIPPED -- set FLM_AIE4_HARDWARE=1 to "
               "run this on the AIE4 target.\n";
        return kCTestSkipReturnCode;
    }

    try {
        if (argc >= 4 && std::string_view(argv[1]) == "--child") {
            g_marker_dir = std::filesystem::path(argv[3]);
            if (!std::filesystem::is_directory(g_marker_dir)) {
                throw std::runtime_error(
                    "child marker directory does not exist: " +
                    g_marker_dir.string());
            }
            return RunChild(argv[2]);
        }
        return RunParent(
            std::filesystem::absolute(argv[0]).lexically_normal());
    } catch (const std::exception& error) {
        std::cerr << "test_fatal_child: FAIL: " << error.what() << '\n';
        return 1;
    }
}
