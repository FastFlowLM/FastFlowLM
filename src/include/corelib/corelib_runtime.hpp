#pragma once

#include <corelib/corelib_api.hpp>
#include <corelib/corelib_fatal_record.hpp>

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>

namespace flm::corelib {

enum class ProcessState { Healthy, Terminating, Shutdown };
using ExecutionLease = std::unique_lock<std::mutex>;

class StepSubmissionState {
public:
    void MarkSuccessfulSubmit() noexcept {
        submitted_ = true;
    }

    bool irrevocable() const noexcept {
        return submitted_;
    }

private:
    bool submitted_ = false;
};

class CorelibRuntime final {
public:
    using Terminator = std::function<void(unsigned int)>;

    static std::shared_ptr<CorelibRuntime> GetOrCreate(
        const std::filesystem::path& executable_dir);
    static std::shared_ptr<CorelibRuntime> Create(
        std::shared_ptr<CorelibApi> api,
        FatalRecordStore records,
        Terminator terminator);
    static void ShutdownProcess();

    ~CorelibRuntime() = default;
    CorelibRuntime(const CorelibRuntime&) = delete;
    CorelibRuntime& operator=(const CorelibRuntime&) = delete;
    CorelibRuntime(CorelibRuntime&&) = delete;
    CorelibRuntime& operator=(CorelibRuntime&&) = delete;

    ExecutionLease AcquireExecution();
    bool admission_open() const noexcept;
    ProcessState state() const noexcept;
    const std::shared_ptr<CorelibApi>& api() const noexcept;
    void ShutdownHealthy();
    [[noreturn]] void TerminateAfterFailure(
        const FailureContext& failure);
#if defined(FLM_CORELIB_TESTING)
    void SetBeforeLiveObjectRollbackForTest(
        std::function<void()> hook);
#endif

private:
    CorelibRuntime(
        std::shared_ptr<CorelibApi> api,
        FatalRecordStore records,
        Terminator terminator);

    std::shared_ptr<CorelibApi> api_;
    FatalRecordStore records_;
    Terminator terminator_;
    std::mutex shutdown_mutex_;
    mutable std::mutex execution_mutex_;
    std::atomic<ProcessState> state_{ProcessState::Shutdown};
    bool cleanup_called_ = false;
#if defined(FLM_CORELIB_TESTING)
    std::function<void()> before_live_object_rollback_for_test_;
#endif
};

}  // namespace flm::corelib
