#include <corelib/corelib_runtime.hpp>

#include <windows.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace flm::corelib {
namespace {

constexpr unsigned int kFatalExitCode = 0xE0040001u;

struct ProcessRuntimeSlot {
    std::mutex mutex;
    std::shared_ptr<CorelibRuntime> runtime;
};

ProcessRuntimeSlot& RuntimeSlot() {
    static auto* slot = new ProcessRuntimeSlot;
    return *slot;
}

[[noreturn]] void TerminateCurrentProcess(unsigned int code) {
    TerminateProcess(GetCurrentProcess(), code);
    std::abort();
}

}  // namespace

std::shared_ptr<CorelibRuntime> CorelibRuntime::GetOrCreate(
    const std::filesystem::path& executable_dir) {
    auto& slot = RuntimeSlot();
    std::lock_guard lock(slot.mutex);
    if (slot.runtime) {
        if (slot.runtime->state() != ProcessState::Healthy) {
            throw std::runtime_error(
                "corelib process runtime is not healthy");
        }
        return slot.runtime;
    }

    auto api = CorelibApi::Load(
        CorelibApi::ResolveLibraryPath(executable_dir));
    auto runtime = Create(
        std::move(api),
        FatalRecordStore::ForCurrentProcess(),
        TerminateCurrentProcess);
    slot.runtime = runtime;
    return runtime;
}

std::shared_ptr<CorelibRuntime> CorelibRuntime::Create(
    std::shared_ptr<CorelibApi> api,
    FatalRecordStore records,
    Terminator terminator) {
    if (!api) {
        throw std::invalid_argument(
            "CorelibRuntime::Create requires a CorelibApi");
    }
    if (!terminator) {
        throw std::invalid_argument(
            "CorelibRuntime::Create requires a terminator");
    }

    auto runtime = std::shared_ptr<CorelibRuntime>(
        new CorelibRuntime(
            std::move(api),
            std::move(records),
            std::move(terminator)));
    try {
        runtime->records_.Prepare();
        runtime->api_->Check(
            runtime->api_->functions().selftest_dependencies(),
            "ryzenai_corelib_selftest_dependencies");
        if (!runtime->api_->functions().has_device_context()) {
            throw std::runtime_error(
                "ryzenai_corelib_has_device_context reported no "
                "AIE4 device context");
        }
        runtime->state_.store(
            ProcessState::Healthy,
            std::memory_order_release);
        return runtime;
    } catch (...) {
        runtime->api_->functions().cleanup();
        runtime->cleanup_called_ = true;
        throw;
    }
}

void CorelibRuntime::ShutdownProcess() {
    auto& slot = RuntimeSlot();
    std::shared_ptr<CorelibRuntime> runtime;
    {
        std::lock_guard lock(slot.mutex);
        if (!slot.runtime) {
            return;
        }
        runtime = slot.runtime;
        runtime->ShutdownHealthy();
        slot.runtime.reset();
    }
    runtime.reset();
}

CorelibRuntime::CorelibRuntime(
    std::shared_ptr<CorelibApi> api,
    FatalRecordStore records,
    Terminator terminator)
    : api_(std::move(api)),
      records_(std::move(records)),
      terminator_(std::move(terminator)) {}

ExecutionLease CorelibRuntime::AcquireExecution() {
    ExecutionLease lease(execution_mutex_);
    if (state_.load(std::memory_order_acquire) !=
        ProcessState::Healthy) {
        throw std::runtime_error(
            "corelib process runtime is not accepting execution");
    }
    return lease;
}

bool CorelibRuntime::admission_open() const noexcept {
    return state_.load(std::memory_order_acquire) ==
           ProcessState::Healthy;
}

ProcessState CorelibRuntime::state() const noexcept {
    return state_.load(std::memory_order_acquire);
}

const std::shared_ptr<CorelibApi>& CorelibRuntime::api() const noexcept {
    return api_;
}

void CorelibRuntime::ShutdownHealthy() {
    std::lock_guard shutdown_lock(shutdown_mutex_);
    ProcessState expected = ProcessState::Healthy;
    if (!state_.compare_exchange_strong(
            expected,
            ProcessState::Shutdown,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        if (expected == ProcessState::Shutdown) {
            return;
        }
        throw std::logic_error(
            "cannot clean up a terminating corelib process runtime");
    }

    ExecutionLease lease(execution_mutex_);
    if (state_.load(std::memory_order_acquire) ==
        ProcessState::Terminating) {
        throw std::logic_error(
            "cannot clean up a terminating corelib process runtime");
    }
    if (api_->live_object_count() != 0) {
        state_.store(ProcessState::Healthy, std::memory_order_release);
        throw std::logic_error(
            "cannot clean up corelib while live corelib objects remain");
    }

    if (!cleanup_called_) {
        api_->functions().cleanup();
        cleanup_called_ = true;
    }
    records_.RemoveUnusedPending();
}

[[noreturn]] void CorelibRuntime::TerminateAfterFailure(
    const FailureContext& failure) {
    state_.store(ProcessState::Terminating, std::memory_order_release);

    std::cerr << "AIE4 terminal failure: call=" << failure.call
              << " phase=" << failure.phase
              << " layer=";
    if (failure.layer.has_value()) {
        std::cerr << *failure.layer;
    } else {
        std::cerr << "none";
    }
    std::cerr << " rows=" << failure.rows
              << " position=" << failure.position
              << " detail=" << failure.detail << '\n';
    try {
        const auto path = records_.Persist(failure);
        std::cerr << "AIE4 fatal record: " << path.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "AIE4 fatal record persistence failed: "
                  << error.what() << '\n';
    } catch (...) {
        std::cerr << "AIE4 fatal record persistence failed with an "
                     "unknown error\n";
    }
    std::cerr.flush();

    terminator_(kFatalExitCode);
    std::abort();
}

}  // namespace flm::corelib
