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

// Printed by the child ONLY on the healthy shutdown path. Its ABSENCE from a
// hard-exit child's output is an assertion in its own right: a process that
// terminated after an irrevocable failure must not have run normal cleanup,
// and "the record exists" alone would not show that.
constexpr std::string_view kCleanupMarker = "FATAL_CHILD_CLEANUP_OK";
constexpr std::string_view kSurvivedMarker = "FATAL_CHILD_SURVIVED";

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

    // The session clears and the process stays usable: admission is still
    // open and the runtime is still healthy after a recoverable failure.
    CHECK(runtime->admission_open());
    CHECK(runtime->state() == flm::corelib::ProcessState::Healthy);
    std::cout << kSurvivedMarker << '\n';

    CorelibRuntime::ShutdownProcess();
    std::cout << kCleanupMarker << '\n';
    std::cout.flush();
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
    std::string_view scenario) {
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
                               scenario.end());
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
        const auto end = record.find('"', cursor);
        return std::string(record.substr(cursor, end - cursor));
    }
    const auto end = record.find_first_of(",}", cursor);
    return std::string(record.substr(cursor, end - cursor));
}

void CheckDetailedRecord(
    std::string_view record,
    std::string_view expected_phase) {
    // "Complete" means every field design Section 12.1 lists is present and
    // carries the value the child produced -- not merely that a file exists.
    CHECK(FieldValue(record, "phase") == expected_phase);
    CHECK(FieldValue(record, "layer") == "7");
    CHECK(FieldValue(record, "rows") == "13");
    CHECK(FieldValue(record, "position") == "29");
    CHECK(
        FieldValue(record, "call") ==
        "ryzenai_corelib_matmul_bf16_pad_shape");
    CHECK(!FieldValue(record, "status").empty());
    CHECK(!FieldValue(record, "detail").empty());
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
        const auto result = RunScenario(
            executable,
            logs.path() / "before_submit.log",
            "before_submit");
        std::cout << "before_submit exit=0x" << std::hex
                  << result.exit_code << std::dec << '\n';
        CHECK(result.exit_code == 0);
        CHECK(Contains(result.output, kSurvivedMarker));
        CHECK(Contains(result.output, kCleanupMarker));

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
        const auto result = RunScenario(
            executable,
            logs.path() / (std::string(scenario) + ".log"),
            scenario);
        std::cout << scenario << " exit=0x" << std::hex
                  << result.exit_code << std::dec << '\n';
        CHECK(result.exit_code == kFatalExitCode);

        // Normal cleanup must NOT have run. Both markers are printed only on
        // the healthy path, so their absence is the evidence.
        CHECK(!Contains(result.output, kCleanupMarker));
        CHECK(!Contains(result.output, kSurvivedMarker));
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
        CheckDetailedRecord(records.front(), phase);
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
        if (argc >= 3 && std::string_view(argv[1]) == "--child") {
            return RunChild(argv[2]);
        }
        return RunParent(
            std::filesystem::absolute(argv[0]).lexically_normal());
    } catch (const std::exception& error) {
        std::cerr << "test_fatal_child: FAIL: " << error.what() << '\n';
        return 1;
    }
}
