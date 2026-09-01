#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <corelib/corelib_fatal_record.hpp>
#include <corelib/corelib_object.hpp>
#include <corelib/corelib_runtime.hpp>

#include <windows.h>
#include <sddl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;
using flm::corelib::FailureContext;
using flm::corelib::FatalRecordStore;
using flm::corelib::ProcessState;

class TempDirectory final {
public:
    TempDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("fastflowlm-corelib-runtime-" + std::to_string(nonce));
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

std::optional<std::wstring> ReadEnvironment(const wchar_t* name) {
    std::size_t required = 0;
    if (_wgetenv_s(&required, nullptr, 0, name) != 0) {
        throw std::runtime_error("failed to read environment variable");
    }
    if (required == 0) {
        return std::nullopt;
    }

    std::vector<wchar_t> value(required);
    if (_wgetenv_s(&required, value.data(), value.size(), name) != 0) {
        throw std::runtime_error("failed to read environment variable");
    }
    return std::wstring(value.data());
}

class ScopedEnvironment final {
public:
    ScopedEnvironment(
        std::wstring name,
        std::optional<std::wstring> value)
        : name_(std::move(name)),
          original_(ReadEnvironment(name_.c_str())) {
        if (_wputenv_s(
                name_.c_str(),
                value.has_value() ? value->c_str() : L"") != 0) {
            throw std::runtime_error("failed to set environment variable");
        }
    }

    ~ScopedEnvironment() noexcept {
        _wputenv_s(
            name_.c_str(),
            original_.has_value() ? original_->c_str() : L"");
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
    std::wstring name_;
    std::optional<std::wstring> original_;
};

std::chrono::system_clock::time_point KnownStartTime() {
    using namespace std::chrono;
    return sys_days{year{2026} / August / day{31}} + hours{17};
}

std::shared_ptr<CorelibApi> ResolveCompleteCorelib() {
    auto resolver = flm::test::CompleteCorelibResolver();
    return CorelibApi::ResolveForTest(
        [resolver = std::move(resolver)](std::string_view name) mutable
            -> void* {
            const auto found = resolver.find(std::string(name));
            return found == resolver.end() ? nullptr : found->second;
        });
}

void WriteText(
    const std::filesystem::path& path,
    std::string_view contents) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to create test record");
    }
    output.write(
        contents.data(),
        static_cast<std::streamsize>(contents.size()));
    if (!output) {
        throw std::runtime_error("failed to write test record");
    }
}

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open test record");
    }
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

std::vector<std::filesystem::path> MatchingFiles(
    const std::filesystem::path& root,
    std::string_view prefix) {
    std::vector<std::filesystem::path> paths;
    std::error_code error;
    for (std::filesystem::directory_iterator iterator(root, error), end;
         !error && iterator != end;
         iterator.increment(error)) {
        const auto name = iterator->path().filename().string();
        if (name.starts_with(prefix)) {
            paths.push_back(iterator->path());
        }
    }
    if (error) {
        throw std::filesystem::filesystem_error(
            "failed to enumerate test records",
            root,
            error);
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::filesystem::path CurrentExecutablePath() {
    std::wstring buffer(32768, L'\0');
    const DWORD size = GetModuleFileNameW(
        nullptr,
        buffer.data(),
        static_cast<DWORD>(buffer.size()));
    if (size == 0 || size == buffer.size()) {
        throw std::runtime_error("GetModuleFileNameW failed");
    }
    buffer.resize(size);
    return std::filesystem::path(std::move(buffer));
}

void CreateReadOnlyDirectory(const std::filesystem::path& path) {
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            L"D:P(D;;GW;;;WD)(A;;GRGXSD;;;WD)",
            SDDL_REVISION_1,
            &descriptor,
            nullptr)) {
        throw std::runtime_error(
            "failed to construct read-only directory ACL");
    }
    SECURITY_ATTRIBUTES attributes{
        sizeof(SECURITY_ATTRIBUTES),
        descriptor,
        FALSE};
    const BOOL created = CreateDirectoryW(path.c_str(), &attributes);
    const DWORD error = created ? ERROR_SUCCESS : GetLastError();
    LocalFree(descriptor);
    if (!created) {
        throw std::runtime_error(
            "failed to create read-only test directory (error " +
            std::to_string(error) + ")");
    }
}

void TestPendingNamesUseStartTimeAndPid() {
    TempDirectory temp;
    const auto start = KnownStartTime();
    auto probe = [](DWORD)
        -> std::optional<std::chrono::system_clock::time_point> {
        return std::nullopt;
    };
    FatalRecordStore store_a(temp.path(), 1001, start, probe);
    FatalRecordStore store_b(temp.path(), 1002, start, probe);

    store_a.Prepare();
    store_b.Prepare();

    CHECK(store_a.pending_path().filename() ==
          "pending-corelib-fatal-20260831T1700000000000Z-1001.tmp");
    CHECK(store_b.pending_path().filename() ==
          "pending-corelib-fatal-20260831T1700000000000Z-1002.tmp");
    CHECK(store_a.pending_path() != store_b.pending_path());
    CHECK(std::filesystem::exists(store_a.pending_path()));
    CHECK(std::filesystem::exists(store_b.pending_path()));

    const auto pending_a = store_a.pending_path();
    const auto pending_b = store_b.pending_path();
    store_a.RemoveUnusedPending();
    store_b.RemoveUnusedPending();
    CHECK(!std::filesystem::exists(pending_a));
    CHECK(!std::filesystem::exists(pending_b));
}

void TestPersistWritesCompleteUniqueRecords() {
    TempDirectory temp;
    const auto start = KnownStartTime();
    auto probe = [](DWORD)
        -> std::optional<std::chrono::system_clock::time_point> {
        return std::nullopt;
    };
    FatalRecordStore store_a(temp.path(), 1001, start, probe);
    FatalRecordStore store_b(temp.path(), 1002, start, probe);
    store_a.Prepare();
    store_b.Prepare();

    const FailureContext failure{
        ryzenai_corelib_status_bad_argument,
        "matmul_q",
        "quoted \"detail\"\nnext line",
        "qkv",
        7,
        32,
        128};
    const auto final_a = store_a.Persist(failure);
    const auto final_b = store_b.Persist(failure);

    CHECK(final_a != final_b);
    CHECK(final_a.filename().string().starts_with("corelib-fatal-"));
    CHECK(final_a.filename().string().ends_with("-1001.json"));
    CHECK(final_b.filename().string().ends_with("-1002.json"));
    CHECK(std::filesystem::exists(final_a));
    CHECK(std::filesystem::exists(final_b));
    CHECK(!std::filesystem::exists(store_a.pending_path()));
    CHECK(!std::filesystem::exists(store_b.pending_path()));

    const auto record = ReadText(final_a);
    CHECK(record.find("\"status\":2") != std::string::npos);
    CHECK(record.find("\"call\":\"matmul_q\"") != std::string::npos);
    CHECK(record.find("quoted \\\"detail\\\"\\nnext line") !=
          std::string::npos);
    CHECK(record.find("\"phase\":\"qkv\"") != std::string::npos);
    CHECK(record.find("\"layer\":7") != std::string::npos);
    CHECK(record.find("\"rows\":32") != std::string::npos);
    CHECK(record.find("\"position\":128") != std::string::npos);
    CHECK(record.find(
              "\"process_start_utc\":\"20260831T1700000000000Z\"") !=
          std::string::npos);
    CHECK(record.find("\"failure_utc\":\"") != std::string::npos);
    CHECK(record.find("\"pid\":1001") != std::string::npos);
}

void TestDrainOrdersFinalRecordsAndRemovesAfterEmission() {
    TempDirectory temp;
    const auto first =
        temp.path() / "corelib-fatal-20260831T1700000000000Z-1001.json";
    const auto same_time_second =
        temp.path() / "corelib-fatal-20260831T1700000000000Z-1002.json";
    const auto later =
        temp.path() / "corelib-fatal-20260831T1800000000000Z-1000.json";
    WriteText(later, "later\n");
    WriteText(same_time_second, "same-time-second\n");
    WriteText(first, "first\n");
    std::size_t probe_calls = 0;
    std::ostringstream output;

    const auto records = FatalRecordStore::DrainPriorRecords(
        temp.path(),
        [&probe_calls](DWORD)
            -> std::optional<std::chrono::system_clock::time_point> {
            ++probe_calls;
            return std::nullopt;
        },
        output);

    const std::vector<std::string> expected{
        "first\n",
        "same-time-second\n",
        "later\n"};
    CHECK(records == expected);
    CHECK(output.str() == "first\nsame-time-second\nlater\n");
    CHECK(probe_calls == 0);
    CHECK(!std::filesystem::exists(first));
    CHECK(!std::filesystem::exists(same_time_second));
    CHECK(!std::filesystem::exists(later));
}

void TestDrainPreservesLivePendingRecord() {
    TempDirectory temp;
    const auto pending =
        temp.path() /
        "pending-corelib-fatal-20260831T1700000000000Z-1001.tmp";
    WriteText(pending, "pending");
    std::ostringstream output;

    const auto records = FatalRecordStore::DrainPriorRecords(
        temp.path(),
        [](DWORD pid)
            -> std::optional<std::chrono::system_clock::time_point> {
            CHECK(pid == 1001);
            return KnownStartTime();
        },
        output);

    CHECK(records.empty());
    CHECK(output.str().empty());
    CHECK(std::filesystem::exists(pending));
}

void TestDrainPreservesPendingWhenProbeFails() {
    TempDirectory temp;
    const auto pending =
        temp.path() /
        "pending-corelib-fatal-20260831T1700000000000Z-1001.tmp";
    WriteText(pending, "pending");
    std::ostringstream output;

    const auto records = FatalRecordStore::DrainPriorRecords(
        temp.path(),
        [](DWORD)
            -> std::optional<std::chrono::system_clock::time_point> {
            return std::nullopt;
        },
        output);

    CHECK(records.empty());
    CHECK(output.str().empty());
    CHECK(std::filesystem::exists(pending));
}

void TestDrainReportsAndRemovesStalePendingRecord() {
    TempDirectory temp;
    const auto pending =
        temp.path() /
        "pending-corelib-fatal-20260831T1700000000000Z-1001.tmp";
    WriteText(pending, "pending");
    std::ostringstream output;

    const auto records = FatalRecordStore::DrainPriorRecords(
        temp.path(),
        [](DWORD)
            -> std::optional<std::chrono::system_clock::time_point> {
            return KnownStartTime() + std::chrono::seconds{1};
        },
        output);

    CHECK(records.size() == 1);
    CHECK(records.front().find("incomplete corelib fatal record") !=
          std::string::npos);
    CHECK(records.front().find(pending.filename().string()) !=
          std::string::npos);
    CHECK(output.str() == records.front());
    CHECK(!std::filesystem::exists(pending));
}

void TestPrepareRejectsUnwritableRoot() {
    TempDirectory temp;
    const auto read_only = temp.path() / "read-only";
    CreateReadOnlyDirectory(read_only);
    FatalRecordStore records(
        read_only,
        1001,
        KnownStartTime(),
        [](DWORD)
            -> std::optional<std::chrono::system_clock::time_point> {
            return std::nullopt;
        });

    CheckThrowsContains(
        [&] {
            records.Prepare();
        },
        "fatal record");
}

void TestUnusedPendingIsRemovedByDestructor() {
    TempDirectory temp;
    std::filesystem::path pending;
    {
        FatalRecordStore records(
            temp.path(),
            1001,
            KnownStartTime(),
            [](DWORD)
                -> std::optional<std::chrono::system_clock::time_point> {
                return std::nullopt;
            });
        records.Prepare();
        pending = records.pending_path();
        CHECK(std::filesystem::exists(pending));
    }
    CHECK(!std::filesystem::exists(pending));
}

FatalRecordStore MakeTestRecords(
    const std::filesystem::path& root,
    DWORD pid = 1001) {
    return FatalRecordStore(
        root,
        pid,
        KnownStartTime(),
        [](DWORD)
            -> std::optional<std::chrono::system_clock::time_point> {
            return std::nullopt;
        });
}

void TestRuntimePublishesHealthyAndShutsDownOnce() {
    TempDirectory temp;
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();
    bool terminated = false;
    auto runtime = CorelibRuntime::Create(
        api,
        MakeTestRecords(temp.path()),
        [&terminated](unsigned int) {
            terminated = true;
            throw std::runtime_error("unexpected termination");
        });

    CHECK(runtime->state() == ProcessState::Healthy);
    CHECK(runtime->admission_open());
    CHECK(runtime->api().get() == api.get());
    {
        auto lease = runtime->AcquireExecution();
        CHECK(lease.owns_lock());
    }

    runtime->ShutdownHealthy();
    runtime->ShutdownHealthy();
    CHECK(runtime->state() == ProcessState::Shutdown);
    CHECK(!runtime->admission_open());
    CHECK(flm::test::CleanupCount() == 1);
    CHECK(!terminated);
    CHECK(MatchingFiles(
              temp.path(),
              "pending-corelib-fatal-").empty());
}

void TestHealthyShutdownClosesAdmissionBeforeWaiting() {
    TempDirectory temp;
    flm::test::ResetFakeCorelib();
    auto runtime = CorelibRuntime::Create(
        ResolveCompleteCorelib(),
        MakeTestRecords(temp.path()),
        [](unsigned int) {
            throw std::runtime_error("unexpected termination");
        });
    auto active_execution = runtime->AcquireExecution();
    std::atomic<bool> shutdown_started = false;
    std::exception_ptr shutdown_error;
    std::thread shutdown([&] {
        shutdown_started.store(true, std::memory_order_release);
        try {
            runtime->ShutdownHealthy();
        } catch (...) {
            shutdown_error = std::current_exception();
        }
    });
    while (!shutdown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{1};
    while (runtime->admission_open() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool closed_before_active_execution_finished =
        !runtime->admission_open();
    active_execution.unlock();
    shutdown.join();

    if (shutdown_error) {
        std::rethrow_exception(shutdown_error);
    }
    CHECK(closed_before_active_execution_finished);
    CHECK(runtime->state() == ProcessState::Shutdown);
}

void TestInitializationFailureCleansUpAndRemovesPending() {
    TempDirectory temp;
    flm::test::ResetFakeCorelib();
    flm::test::SetLastErrorMessage("dependency probe failed");
    flm::test::SetSelftestStatus(ryzenai_corelib_status_failure);
    auto api = ResolveCompleteCorelib();

    CheckThrowsContains(
        [&] {
            (void)CorelibRuntime::Create(
                api,
                MakeTestRecords(temp.path()),
                [](unsigned int) {
                    throw std::runtime_error("unexpected termination");
                });
        },
        "ryzenai_corelib_selftest_dependencies");

    CHECK(flm::test::CleanupCount() == 1);
    CHECK(MatchingFiles(
              temp.path(),
              "pending-corelib-fatal-").empty());
}

void TestMissingDeviceContextCleansUpAndRemovesPending() {
    TempDirectory temp;
    flm::test::ResetFakeCorelib();
    flm::test::SetHasDeviceContext(false);
    auto api = ResolveCompleteCorelib();

    CheckThrowsContains(
        [&] {
            (void)CorelibRuntime::Create(
                api,
                MakeTestRecords(temp.path()),
                [](unsigned int) {
                    throw std::runtime_error("unexpected termination");
                });
        },
        "device context");

    CHECK(flm::test::CleanupCount() == 1);
    CHECK(MatchingFiles(
              temp.path(),
              "pending-corelib-fatal-").empty());
}

void TestHealthyCleanupFollowsLastObjectRelease() {
    TempDirectory temp;
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();
    auto runtime = CorelibRuntime::Create(
        api,
        MakeTestRecords(temp.path()),
        [](unsigned int) {
            throw std::runtime_error("unexpected termination");
        });
    int storage = 0;
    flm::corelib::UniqueTensor tensor(api, &storage);

    CheckThrowsContains(
        [&] {
            runtime->ShutdownHealthy();
        },
        "live corelib");
    CHECK(runtime->state() == ProcessState::Healthy);
    CHECK(flm::test::CleanupCount() == 0);

    tensor.reset();
    runtime->ShutdownHealthy();

    const std::vector<std::string> expected{"release", "cleanup"};
    CHECK(flm::test::FakeCorelibEvents() == expected);
    CHECK(flm::test::CleanupCount() == 1);
}

struct TerminationIntercept final {};

void TestTerminationClosesAdmissionBeforeTerminator() {
    TempDirectory temp;
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();
    std::shared_ptr<CorelibRuntime> runtime;
    bool terminator_called = false;
    unsigned int termination_code = 0;
    runtime = CorelibRuntime::Create(
        api,
        MakeTestRecords(temp.path()),
        [&](unsigned int code) {
            terminator_called = true;
            termination_code = code;
            CHECK(runtime->state() == ProcessState::Terminating);
            CHECK(!runtime->admission_open());
            throw TerminationIntercept{};
        });
    const FailureContext failure{
        ryzenai_corelib_status_failure,
        "ryzenai_corelib_matmul_bf16",
        "dispatch failed",
        "qkv",
        12,
        4,
        256};

    try {
        runtime->TerminateAfterFailure(failure);
    } catch (const TerminationIntercept&) {
    }

    CHECK(terminator_called);
    CHECK(termination_code == 0xE0040001u);
    CHECK(runtime->state() == ProcessState::Terminating);
    CHECK(!runtime->admission_open());
    CHECK(flm::test::CleanupCount() == 0);
    const auto records = MatchingFiles(temp.path(), "corelib-fatal-");
    CHECK(records.size() == 1);
    const auto contents = ReadText(records.front());
    CHECK(contents.find("\"call\":\"ryzenai_corelib_matmul_bf16\"") !=
          std::string::npos);
    CHECK(contents.find("\"phase\":\"qkv\"") != std::string::npos);
    CHECK(contents.find("\"layer\":12") != std::string::npos);
    CHECK(contents.find("\"rows\":4") != std::string::npos);
    CHECK(contents.find("\"position\":256") != std::string::npos);
}

void TestStepSubmissionStateCrossesIrrevocableBoundary() {
    flm::corelib::StepSubmissionState submission;
    CHECK(!submission.irrevocable());
    submission.MarkSuccessfulSubmit();
    CHECK(submission.irrevocable());
}

void TestGetOrCreateKeepsOneProcessRuntimeUntilExplicitShutdown() {
    TempDirectory temp;
    const auto fake_dll =
        CurrentExecutablePath().parent_path() / "fake_ryzenai_corelib.dll";
    const auto cleanup_marker = temp.path() / "cleanup-marker.txt";
    CHECK(std::filesystem::exists(fake_dll));
    ScopedEnvironment corelib_path{
        L"RYZENAI_CORELIB_PATH",
        fake_dll.wstring()};
    ScopedEnvironment marker_path{
        L"FLM_FAKE_CORELIB_CLEANUP_MARKER",
        cleanup_marker.wstring()};

    auto first = CorelibRuntime::GetOrCreate(
        CurrentExecutablePath().parent_path());
    const auto first_runtime = first.get();
    const auto first_api = first->api().get();
    std::weak_ptr<CorelibRuntime> runtime_weak = first;
    std::weak_ptr<CorelibApi> api_weak = first->api();
    first.reset();

    CHECK(!runtime_weak.expired());
    auto second = CorelibRuntime::GetOrCreate(
        CurrentExecutablePath().parent_path());
    CHECK(second.get() == first_runtime);
    CHECK(second->api().get() == first_api);
    CHECK(!std::filesystem::exists(cleanup_marker));
    second.reset();

    CorelibRuntime::ShutdownProcess();

    CHECK(runtime_weak.expired());
    CHECK(api_weak.expired());
    CHECK(ReadText(cleanup_marker) == "cleanup\n");
}

}  // namespace

int main() {
    try {
        TestPendingNamesUseStartTimeAndPid();
        TestPersistWritesCompleteUniqueRecords();
        TestDrainOrdersFinalRecordsAndRemovesAfterEmission();
        TestDrainPreservesLivePendingRecord();
        TestDrainPreservesPendingWhenProbeFails();
        TestDrainReportsAndRemovesStalePendingRecord();
        TestPrepareRejectsUnwritableRoot();
        TestUnusedPendingIsRemovedByDestructor();
        TestRuntimePublishesHealthyAndShutsDownOnce();
        TestHealthyShutdownClosesAdmissionBeforeWaiting();
        TestInitializationFailureCleansUpAndRemovesPending();
        TestMissingDeviceContextCleansUpAndRemovesPending();
        TestHealthyCleanupFollowsLastObjectRelease();
        TestTerminationClosesAdmissionBeforeTerminator();
        TestStepSubmissionStateCrossesIrrevocableBoundary();
        TestGetOrCreateKeepsOneProcessRuntimeUntilExplicitShutdown();
        std::cout << "test_corelib_fatal_record: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "unexpected non-standard exception\n";
        return 1;
    }
}
