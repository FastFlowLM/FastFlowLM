#pragma once

#include <ryzenai/corelib.h>

#include <windows.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace flm::corelib {

struct FailureContext {
    ryzenai_corelib_status status;
    std::string call;
    std::string detail;
    std::string phase;
    std::optional<int> layer;
    int64_t rows;
    int64_t position;
};

class FatalRecordStore {
public:
    using ProcessProbe =
        std::function<
            std::optional<std::chrono::system_clock::time_point>(DWORD)>;

    FatalRecordStore(
        std::filesystem::path root,
        DWORD pid,
        std::chrono::system_clock::time_point process_start,
        ProcessProbe process_probe);
    FatalRecordStore(FatalRecordStore&& other) noexcept;
    FatalRecordStore& operator=(FatalRecordStore&& other) noexcept;
    ~FatalRecordStore() noexcept;
    FatalRecordStore(const FatalRecordStore&) = delete;
    FatalRecordStore& operator=(const FatalRecordStore&) = delete;

    static FatalRecordStore ForCurrentProcess();

    void Prepare();
    std::filesystem::path Persist(const FailureContext& failure);
    static std::vector<std::string> DrainPriorRecords(
        std::ostream& output);
    static std::vector<std::string> DrainPriorRecords(
        const std::filesystem::path& root,
        ProcessProbe process_probe,
        std::ostream& output);
    void RemoveUnusedPending() noexcept;
    const std::filesystem::path& pending_path() const noexcept;

private:
    void ClosePending() noexcept;

    std::filesystem::path root_;
    DWORD pid_ = 0;
    std::chrono::system_clock::time_point process_start_;
    ProcessProbe process_probe_;
    std::filesystem::path pending_path_;
    HANDLE pending_handle_ = INVALID_HANDLE_VALUE;
    bool persisted_ = false;
};

}  // namespace flm::corelib
