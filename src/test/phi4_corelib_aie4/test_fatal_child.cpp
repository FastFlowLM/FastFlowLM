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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
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
using flm::corelib::FailureContext;
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

// The record-store half of the terminal path, in a real process, with no
// device.
//
// Design 12.4's concurrency properties -- unique files per process, a pending
// file preserved while its owner is still alive, ordered reporting of every
// completed record -- have only ever been checked in ONE process against
// injected PIDs, start times and process probes. That leaves the production
// probe itself unexercised: nothing had ever confirmed that
// `ProbeProcessStart` recognises a genuinely live sibling process, which is
// the single decision that stands between a live process's pending file and
// its deletion.
//
// This child deliberately does NOT load corelib. Two processes holding AIE4
// device contexts at once fail in ways that look like defects, and none of the
// four properties under test needs a device -- so the concurrency is real and
// the device contention is not introduced.
int RunRecordOnlyChild() {
    auto store = FatalRecordStore::ForCurrentProcess();
    store.Prepare();
    WriteMarker(L"pid.marker", std::to_string(GetCurrentProcessId()));
    WriteMarker(L"pending.marker", store.pending_path().string());
    // Written LAST, and empty on purpose.
    //
    // The other scenarios read a child's markers only after it has exited, so
    // they never had to care that `ofstream` CREATES a file before it writes
    // to it. This scenario reads while the child is still alive, and polling
    // `exists("pending.marker")` returns true the instant the stream is
    // opened -- so the parent read a zero-length path, decided the pending
    // file did not exist, and failed. It failed intermittently, which is
    // worse than failing. Existence of this marker is ordered after the other
    // two are flushed and closed, so it means "the markers are complete"
    // rather than "a marker has begun".
    WriteMarker(L"ready.marker", "");

    // Both children hold their pending file open until the parent has drained
    // once with both of them alive. The go file is a sibling of the marker
    // directories so no extra argument has to be threaded through.
    const auto go = g_marker_dir.parent_path() / "go.marker";
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (!std::filesystem::exists(go)) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::cerr << "record_only child timed out waiting for "
                      << go.string() << '\n';
            return 3;
        }
        Sleep(25);
    }

    const FailureContext failure{
        ryzenai_corelib_status_bad_argument,
        "matmul_q",
        "concurrent record child",
        "qkv",
        7,
        13,
        29};
    const auto final_path = store.Persist(failure);
    WriteMarker(L"final.marker", final_path.string());
    return 0;
}

int RunChild(std::string_view scenario) {
    if (scenario == "record_only") {
        return RunRecordOnlyChild();
    }
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
    // PID alone is not unique enough. Windows reuses process IDs freely, and
    // a run that fails can leave child processes polling this directory for
    // up to their timeout; a later run that happens to get the same PID then
    // shares a directory with somebody else's children. That produced an
    // intermittent failure that only ever appeared in the run immediately
    // after a failed one, which is the hardest kind to read. The tick count
    // makes reuse impossible in any window that matters.
    TempDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("fastflowlm-fatal-child-" +
                 std::to_string(GetCurrentProcessId()) + "-" +
                 std::to_string(GetTickCount64()));
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

// A child that has been started but not yet waited for. Needed because the
// concurrency scenario has to have two of them alive at the same moment; every
// other scenario starts one and waits.
struct RunningChild {
    PROCESS_INFORMATION process{};
    std::filesystem::path log;
    std::filesystem::path marker_dir;
    std::string scenario;
};

RunningChild StartChild(
    const std::filesystem::path& executable,
    const std::filesystem::path& log_path,
    const std::filesystem::path& marker_dir,
    std::string_view scenario);

ChildResult AwaitChild(RunningChild& child, DWORD timeout_ms);

ChildResult RunScenario(
    const std::filesystem::path& executable,
    const std::filesystem::path& log_path,
    const std::filesystem::path& marker_dir,
    std::string_view scenario) {
    RunningChild child =
        StartChild(executable, log_path, marker_dir, scenario);
    // Generous, but bounded: the child loads the real corelib, and a hang
    // must fail the suite rather than wedge it.
    return AwaitChild(child, 300000);
}

RunningChild StartChild(
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

    RunningChild child;
    child.process = process;
    child.log = log_path;
    child.marker_dir = marker_dir;
    child.scenario = std::string(scenario);
    return child;
}

ChildResult AwaitChild(RunningChild& child, DWORD timeout_ms) {
    ChildResult result;
    if (
        WaitForSingleObject(child.process.hProcess, timeout_ms) !=
        WAIT_OBJECT_0
    ) {
        TerminateProcess(child.process.hProcess, 1);
        CloseHandle(child.process.hThread);
        CloseHandle(child.process.hProcess);
        child.process = PROCESS_INFORMATION{};
        throw std::runtime_error(
            "child scenario " + child.scenario + " did not exit");
    }
    GetExitCodeProcess(child.process.hProcess, &result.exit_code);
    CloseHandle(child.process.hThread);
    CloseHandle(child.process.hProcess);
    child.process = PROCESS_INFORMATION{};

    std::ifstream stream(child.log, std::ios::binary);
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

// Design 12.4 concurrency, with two processes that are genuinely concurrent.
//
// Four properties, and what each one adds over the in-process tests in
// test_corelib_fatal_record:
//
//  1. Unique files. There the two stores were given PIDs 1001 and 1002 by the
//     test. Here they are whatever Windows assigned, and the start times are
//     whatever the real `GetProcessTimes` reported.
//  2. A live process's pending file is preserved. There the probe was a lambda
//     that returned a chosen answer. Here it is the production
//     `ProbeProcessStart`, asked about a sibling process that really is
//     running -- the only version of this check that can fail if the probe is
//     wrong.
//  3. Every completed record is reported, in order.
//  4. The pending files are gone once their owners have persisted.
//
// What this does NOT add: "safe preservation when the process-start query
// fails". A failing Win32 query cannot be produced on demand, so that case
// stays with the injected-probe test and is not claimed here.
void RunConcurrentRecordScenario(
    const std::filesystem::path& executable,
    TempDirectory& logs) {
    std::cout << "concurrent record children:\n";

    // Every record from the scenarios above has already been drained, but
    // drain again so the counts below describe only these two children.
    std::ostringstream cleared;
    (void)FatalRecordStore::DrainPriorRecords(cleared);

    const auto go = logs.path() / "go.marker";
    std::error_code ignored;
    std::filesystem::remove(go, ignored);

    std::array<RunningChild, 2> children{
        StartChild(
            executable,
            logs.path() / "record_a.log",
            logs.path() / "record_a",
            "record_only"),
        StartChild(
            executable,
            logs.path() / "record_b.log",
            logs.path() / "record_b",
            "record_only")};

    // Nothing here may leave a child behind. A child that outlives a failing
    // parent keeps polling for a release file that will never appear, holds
    // a pending record in the shared root while it does, and is the state
    // that made this scenario's first failure intermittent.
    struct TerminateOnScopeExit {
        std::array<RunningChild, 2>* children;
        ~TerminateOnScopeExit() noexcept {
            for (auto& child : *children) {
                if (child.process.hProcess != nullptr) {
                    TerminateProcess(child.process.hProcess, 1);
                    CloseHandle(child.process.hThread);
                    CloseHandle(child.process.hProcess);
                    child.process = PROCESS_INFORMATION{};
                }
            }
        }
    } terminate_on_scope_exit{&children};

    std::array<std::filesystem::path, 2> pending{};
    std::array<std::string, 2> pids{};
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (std::size_t index = 0; index < children.size(); ++index) {
        const auto marker = children[index].marker_dir / "ready.marker";
        while (!std::filesystem::exists(marker)) {
            if (std::chrono::steady_clock::now() > deadline) {
                for (auto& child : children) {
                    if (child.process.hProcess != nullptr) {
                        TerminateProcess(child.process.hProcess, 1);
                    }
                }
                throw std::runtime_error(
                    "a record_only child never wrote its pending marker");
            }
            Sleep(25);
        }
        pending[index] = std::filesystem::path(
            ReadMarker(children[index].marker_dir, L"pending.marker"));
        pids[index] = ReadMarker(children[index].marker_dir, L"pid.marker");
        // An empty marker means the ordering above broke, not that the store
        // misbehaved. Say which, or the next reader spends an hour on the
        // wrong one -- as this test already cost once.
        if (pending[index].empty() || pids[index].empty()) {
            for (auto& child : children) {
                if (child.process.hProcess != nullptr) {
                    TerminateProcess(child.process.hProcess, 1);
                }
            }
            throw std::runtime_error(
                "a record_only child's markers were readable but empty, so "
                "ready.marker is no longer ordered after them");
        }
    }

    // 1. Two live processes, two distinct pending files.
    CHECK(pending[0] != pending[1]);
    CHECK(pids[0] != pids[1]);
    for (std::size_t index = 0; index < pending.size(); ++index) {
        if (std::filesystem::exists(pending[index])) {
            continue;
        }
        // A bare CHECK here says only "false", and this exact assertion has
        // already been chased once on a wrong theory. Report the state that
        // distinguishes the candidates: whether the child already persisted
        // (a final record carrying its pid), whether the release file it
        // waits on somehow exists, and what is actually in the record root.
        std::ostringstream detail;
        detail << "child " << pids[index]
               << " pending file is absent while the child is still running: "
               << pending[index].string()
               << "\n  go file " << go.string() << " exists: "
               << std::boolalpha << std::filesystem::exists(go)
               << "\n  record root now holds:";
        std::error_code listing;
        for (const auto& entry : std::filesystem::directory_iterator(
                 pending[index].parent_path(), listing)) {
            detail << "\n    " << entry.path().filename().string();
        }
        for (auto& child : children) {
            if (child.process.hProcess != nullptr) {
                DWORD code = 0;
                GetExitCodeProcess(child.process.hProcess, &code);
                detail << "\n  child " << child.scenario << " exit code: "
                       << (code == STILL_ACTIVE
                               ? std::string("still running")
                               : std::to_string(code));
                TerminateProcess(child.process.hProcess, 1);
            }
        }
        throw std::runtime_error(detail.str());
    }

    // 2. Both owners are alive, so the production probe must keep both.
    std::ostringstream while_alive;
    const auto during = FatalRecordStore::DrainPriorRecords(while_alive);
    if (!during.empty()) {
        for (auto& child : children) {
            if (child.process.hProcess != nullptr) {
                TerminateProcess(child.process.hProcess, 1);
            }
        }
        throw std::runtime_error(
            "a drain taken while two children were still running reported " +
            std::to_string(during.size()) + " completed record(s)");
    }
    if (
        !std::filesystem::exists(pending[0]) ||
        !std::filesystem::exists(pending[1])
    ) {
        for (auto& child : children) {
            if (child.process.hProcess != nullptr) {
                TerminateProcess(child.process.hProcess, 1);
            }
        }
        throw std::runtime_error(
            "the drain deleted a pending record whose owner was still alive");
    }
    std::cout << "  both pendings survived a drain taken while both children "
                 "were running\n";

    {
        std::ofstream stream(go, std::ios::binary | std::ios::trunc);
        stream << "go";
    }
    for (auto& child : children) {
        const auto result = AwaitChild(child, 180000);
        CHECK(result.exit_code == 0);
    }

    std::array<std::filesystem::path, 2> finals{
        std::filesystem::path(
            ReadMarker(children[0].marker_dir, L"final.marker")),
        std::filesystem::path(
            ReadMarker(children[1].marker_dir, L"final.marker"))};
    CHECK(finals[0] != finals[1]);
    for (std::size_t index = 0; index < finals.size(); ++index) {
        CHECK(finals[index].filename().string().ends_with(
            "-" + pids[index] + ".json"));
        // 4. Persisting consumed the pending file.
        CHECK(!std::filesystem::exists(pending[index]));
    }

    // 3. Both completed records are reported, and in filename order -- which
    //    is timestamp then PID, so the expected order is derived here rather
    //    than assumed to be the order the children were started in.
    std::vector<std::string> expected{
        finals[0].string(), finals[1].string()};
    std::sort(expected.begin(), expected.end());
    std::ostringstream reported;
    const auto drained = FatalRecordStore::DrainPriorRecords(reported);
    if (drained.size() != 2) {
        throw std::runtime_error(
            "expected 2 completed records from two concurrent children, "
            "drained " + std::to_string(drained.size()));
    }
    std::vector<std::string> drained_paths;
    for (const auto& record : drained) {
        // DrainPriorRecords returns each record's CONTENTS, so identify them
        // by the PID each one carries rather than by a path it never returns.
        CHECK(!record.empty());
        drained_paths.push_back(FieldValue(record, "pid"));
    }
    std::vector<std::string> expected_pids;
    for (const auto& path : expected) {
        const auto name = std::filesystem::path(path).filename().string();
        const auto dash = name.rfind('-');
        expected_pids.push_back(
            name.substr(dash + 1, name.size() - dash - 1 - 5));
    }
    CHECK(drained_paths == expected_pids);
    for (const auto& pid : expected_pids) {
        CHECK(reported.str().find("\"pid\":" + pid) != std::string::npos);
    }
    std::cout << "  2 completed records drained in filename order, pids "
              << expected_pids[0] << " and " << expected_pids[1] << '\n';
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

    RunConcurrentRecordScenario(executable, logs);

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
