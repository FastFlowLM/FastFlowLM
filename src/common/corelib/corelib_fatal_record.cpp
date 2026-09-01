#include <corelib/corelib_fatal_record.hpp>

#include <ShlObj.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace flm::corelib {
namespace {

using HundredNanoseconds =
    std::chrono::duration<int64_t, std::ratio<1, 10'000'000>>;

constexpr uint64_t kWindowsToUnixEpochTicks = 116'444'736'000'000'000ULL;
constexpr std::string_view kPendingPrefix = "pending-corelib-fatal-";
constexpr std::string_view kPendingSuffix = ".tmp";
constexpr std::string_view kFinalPrefix = "corelib-fatal-";
constexpr std::string_view kFinalSuffix = ".json";

std::runtime_error FatalRecordError(
    std::string_view action,
    const std::filesystem::path& path,
    unsigned long error) {
    return std::runtime_error(
        "AIE4 fatal record " + std::string(action) + " failed for " +
        path.string() + " (error " + std::to_string(error) + ")");
}

std::string FormatUtc(
    std::chrono::system_clock::time_point value) {
    const auto ticks =
        std::chrono::duration_cast<HundredNanoseconds>(
            value.time_since_epoch())
            .count();
    constexpr int64_t ticks_per_second = 10'000'000;
    int64_t seconds = ticks / ticks_per_second;
    int64_t fraction = ticks % ticks_per_second;
    if (fraction < 0) {
        fraction += ticks_per_second;
        --seconds;
    }

    const std::time_t calendar_seconds =
        static_cast<std::time_t>(seconds);
    std::tm utc{};
    if (gmtime_s(&utc, &calendar_seconds) != 0) {
        throw std::runtime_error(
            "AIE4 fatal record timestamp conversion failed");
    }

    std::ostringstream output;
    output << std::put_time(&utc, "%Y%m%dT%H%M%S")
           << std::setfill('0') << std::setw(7) << fraction << 'Z';
    return output.str();
}

std::optional<std::chrono::system_clock::time_point> ParseUtc(
    std::string_view value) {
    if (value.size() != 23 || value[8] != 'T' || value[22] != 'Z') {
        return std::nullopt;
    }

    const auto parse_part =
        [value](std::size_t offset, std::size_t length)
        -> std::optional<int> {
        int result = 0;
        const char* first = value.data() + offset;
        const char* last = first + length;
        const auto parsed = std::from_chars(first, last, result);
        if (parsed.ec != std::errc{} || parsed.ptr != last) {
            return std::nullopt;
        }
        return result;
    };

    const auto year = parse_part(0, 4);
    const auto month = parse_part(4, 2);
    const auto day = parse_part(6, 2);
    const auto hour = parse_part(9, 2);
    const auto minute = parse_part(11, 2);
    const auto second = parse_part(13, 2);
    const auto fraction = parse_part(15, 7);
    if (!year || !month || !day || !hour || !minute || !second ||
        !fraction) {
        return std::nullopt;
    }

    std::tm utc{};
    utc.tm_year = *year - 1900;
    utc.tm_mon = *month - 1;
    utc.tm_mday = *day;
    utc.tm_hour = *hour;
    utc.tm_min = *minute;
    utc.tm_sec = *second;
    const __time64_t seconds = _mkgmtime64(&utc);
    if (seconds == -1) {
        return std::nullopt;
    }

    const HundredNanoseconds duration{
        static_cast<int64_t>(seconds) * 10'000'000 + *fraction};
    const auto result = std::chrono::system_clock::time_point{
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            duration)};
    if (FormatUtc(result) != value) {
        return std::nullopt;
    }
    return result;
}

std::chrono::system_clock::time_point FileTimeToSystemClock(
    const FILETIME& value) {
    ULARGE_INTEGER ticks{};
    ticks.LowPart = value.dwLowDateTime;
    ticks.HighPart = value.dwHighDateTime;
    if (ticks.QuadPart < kWindowsToUnixEpochTicks) {
        throw std::runtime_error(
            "AIE4 fatal record process start time predates Unix epoch");
    }
    const auto unix_ticks = ticks.QuadPart - kWindowsToUnixEpochTicks;
    if (unix_ticks >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error(
            "AIE4 fatal record process start time is out of range");
    }
    return std::chrono::system_clock::time_point{
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            HundredNanoseconds{static_cast<int64_t>(unix_ticks)})};
}

std::chrono::system_clock::time_point QueryProcessStart(HANDLE process) {
    FILETIME creation{};
    FILETIME exit{};
    FILETIME kernel{};
    FILETIME user{};
    if (!GetProcessTimes(
            process,
            &creation,
            &exit,
            &kernel,
            &user)) {
        throw FatalRecordError(
            "process-time query",
            {},
            GetLastError());
    }
    return FileTimeToSystemClock(creation);
}

std::optional<std::chrono::system_clock::time_point> ProbeProcessStart(
    DWORD pid) {
    HANDLE process = OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION,
        FALSE,
        pid);
    if (process == nullptr) {
        if (GetLastError() == ERROR_INVALID_PARAMETER) {
            return std::chrono::system_clock::time_point{};
        }
        return std::nullopt;
    }

    try {
        const auto start = QueryProcessStart(process);
        CloseHandle(process);
        return start;
    } catch (...) {
        CloseHandle(process);
        return std::nullopt;
    }
}

std::filesystem::path LocalAppDataLogRoot() {
    PWSTR raw = nullptr;
    const HRESULT result = SHGetKnownFolderPath(
        FOLDERID_LocalAppData,
        KF_FLAG_DEFAULT,
        nullptr,
        &raw);
    if (FAILED(result)) {
        throw std::runtime_error(
            "AIE4 fatal record LocalAppData resolution failed (HRESULT " +
            std::to_string(static_cast<unsigned long>(result)) + ")");
    }

    std::filesystem::path root(raw);
    CoTaskMemFree(raw);
    return root / "FastFlowLM" / "logs";
}

std::string EscapeJson(std::string_view value) {
    std::ostringstream output;
    output << std::hex << std::uppercase;
    for (const unsigned char character : value) {
        switch (character) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20) {
                    output << "\\u"
                           << std::setfill('0') << std::setw(4)
                           << static_cast<unsigned int>(character);
                } else {
                    output << static_cast<char>(character);
                }
                break;
        }
    }
    return output.str();
}

std::string SerializeFailure(
    const FailureContext& failure,
    std::string_view process_start,
    std::string_view failure_utc,
    DWORD pid) {
    std::ostringstream output;
    output << '{'
           << "\"status\":" << static_cast<int>(failure.status) << ','
           << "\"call\":\"" << EscapeJson(failure.call) << "\","
           << "\"detail\":\"" << EscapeJson(failure.detail) << "\","
           << "\"phase\":\"" << EscapeJson(failure.phase) << "\","
           << "\"layer\":";
    if (failure.layer.has_value()) {
        output << *failure.layer;
    } else {
        output << "null";
    }
    output << ','
           << "\"rows\":" << failure.rows << ','
           << "\"position\":" << failure.position << ','
           << "\"process_start_utc\":\"" << process_start << "\","
           << "\"failure_utc\":\"" << failure_utc << "\","
           << "\"pid\":" << pid
           << "}\n";
    return output.str();
}

void WriteAll(HANDLE file, std::string_view contents) {
    std::size_t offset = 0;
    while (offset < contents.size()) {
        const auto remaining = contents.size() - offset;
        const DWORD chunk = static_cast<DWORD>(
            (std::min)(
                remaining,
                static_cast<std::size_t>(
                    std::numeric_limits<DWORD>::max())));
        DWORD written = 0;
        if (!WriteFile(
                file,
                contents.data() + offset,
                chunk,
                &written,
                nullptr) ||
            written != chunk) {
            throw FatalRecordError(
                "write",
                {},
                GetLastError());
        }
        offset += written;
    }
}

void RewindAndTruncate(HANDLE file) {
    LARGE_INTEGER beginning{};
    if (!SetFilePointerEx(file, beginning, nullptr, FILE_BEGIN) ||
        !SetEndOfFile(file)) {
        throw FatalRecordError(
            "truncate",
            {},
            GetLastError());
    }
}

void EmitRecord(std::ostream& output, std::string_view record) {
    output.write(
        record.data(),
        static_cast<std::streamsize>(record.size()));
    output.flush();
    if (!output) {
        throw std::runtime_error(
            "AIE4 fatal record startup reporting failed");
    }
}

bool HasPrefixAndSuffix(
    std::string_view value,
    std::string_view prefix,
    std::string_view suffix) {
    return value.starts_with(prefix) && value.ends_with(suffix) &&
           value.size() > prefix.size() + suffix.size();
}

struct PendingIdentity {
    std::string timestamp;
    DWORD pid;
    std::chrono::system_clock::time_point start;
};

std::optional<PendingIdentity> ParsePendingIdentity(
    const std::filesystem::path& path) {
    const std::string filename = path.filename().string();
    if (!HasPrefixAndSuffix(
            filename,
            kPendingPrefix,
            kPendingSuffix)) {
        return std::nullopt;
    }

    const std::string_view body(filename.data() + kPendingPrefix.size(),
                                filename.size() - kPendingPrefix.size() -
                                    kPendingSuffix.size());
    const auto separator = body.rfind('-');
    if (separator == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string_view timestamp = body.substr(0, separator);
    const std::string_view pid_text = body.substr(separator + 1);
    unsigned long pid_value = 0;
    const auto parsed = std::from_chars(
        pid_text.data(),
        pid_text.data() + pid_text.size(),
        pid_value);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != pid_text.data() + pid_text.size() ||
        pid_value > std::numeric_limits<DWORD>::max()) {
        return std::nullopt;
    }
    const auto start = ParseUtc(timestamp);
    if (!start.has_value()) {
        return std::nullopt;
    }
    return PendingIdentity{
        std::string(timestamp),
        static_cast<DWORD>(pid_value),
        *start};
}

std::string ReadRecord(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw FatalRecordError("read", path, GetLastError());
    }
    std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    if (input.bad()) {
        throw FatalRecordError("read", path, GetLastError());
    }
    return contents;
}

void EmitDrainWarning(
    std::ostream& output,
    std::string_view operation,
    const std::filesystem::path& path,
    std::string_view detail) noexcept {
    try {
        output << "AIE4 fatal record warning: failed to " << operation
               << ' ' << path.string();
        if (!detail.empty()) {
            output << ": " << detail;
        }
        output << '\n';
    } catch (...) {
    }
}

void RemoveReportedRecord(
    const std::filesystem::path& path,
    std::ostream& output) noexcept {
    try {
        std::error_code error;
        const bool removed = std::filesystem::remove(path, error);
        if (!error && removed) {
            return;
        }
        EmitDrainWarning(
            output,
            "remove",
            path,
            error ? error.message() : "record was not removed");
    } catch (const std::exception& exception) {
        EmitDrainWarning(output, "remove", path, exception.what());
    } catch (...) {
        EmitDrainWarning(output, "remove", path, "unknown error");
    }
}

}  // namespace

FatalRecordStore::FatalRecordStore(
    std::filesystem::path root,
    DWORD pid,
    std::chrono::system_clock::time_point process_start,
    ProcessProbe process_probe)
    : root_(std::move(root)),
      pid_(pid),
      process_start_(process_start),
      process_probe_(std::move(process_probe)) {}

FatalRecordStore::FatalRecordStore(
    FatalRecordStore&& other) noexcept
    : root_(std::move(other.root_)),
      pid_(other.pid_),
      process_start_(other.process_start_),
      process_probe_(std::move(other.process_probe_)),
      pending_path_(std::move(other.pending_path_)),
      pending_handle_(
          std::exchange(other.pending_handle_, INVALID_HANDLE_VALUE)),
      persisted_(other.persisted_) {
    other.pending_path_.clear();
    other.persisted_ = true;
}

FatalRecordStore& FatalRecordStore::operator=(
    FatalRecordStore&& other) noexcept {
    if (this != &other) {
        RemoveUnusedPending();
        root_ = std::move(other.root_);
        pid_ = other.pid_;
        process_start_ = other.process_start_;
        process_probe_ = std::move(other.process_probe_);
        pending_path_ = std::move(other.pending_path_);
        pending_handle_ =
            std::exchange(other.pending_handle_, INVALID_HANDLE_VALUE);
        persisted_ = other.persisted_;
        other.pending_path_.clear();
        other.persisted_ = true;
    }
    return *this;
}

FatalRecordStore::~FatalRecordStore() noexcept {
    RemoveUnusedPending();
}

FatalRecordStore FatalRecordStore::ForCurrentProcess() {
    const DWORD pid = GetCurrentProcessId();
    return FatalRecordStore(
        LocalAppDataLogRoot(),
        pid,
        QueryProcessStart(GetCurrentProcess()),
        ProbeProcessStart);
}

void FatalRecordStore::Prepare() {
    if (pending_handle_ != INVALID_HANDLE_VALUE || persisted_) {
        return;
    }

    std::error_code error;
    std::filesystem::create_directories(root_, error);
    if (error || !std::filesystem::is_directory(root_, error) || error) {
        throw std::runtime_error(
            "AIE4 fatal record directory preparation failed for " +
            root_.string() +
            (error ? ": " + error.message() : ""));
    }

    pending_path_ =
        root_ /
        (std::string(kPendingPrefix) + FormatUtc(process_start_) + "-" +
         std::to_string(pid_) + std::string(kPendingSuffix));
    pending_handle_ = CreateFileW(
        pending_path_.c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ,
        nullptr,
        CREATE_NEW,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH,
        nullptr);
    if (pending_handle_ == INVALID_HANDLE_VALUE) {
        throw FatalRecordError(
            "initialization",
            pending_path_,
            GetLastError());
    }

    try {
        WriteAll(pending_handle_, "P");
        if (!FlushFileBuffers(pending_handle_)) {
            throw FatalRecordError(
                "initial flush",
                pending_path_,
                GetLastError());
        }
        RewindAndTruncate(pending_handle_);
        if (!FlushFileBuffers(pending_handle_)) {
            throw FatalRecordError(
                "initial flush",
                pending_path_,
                GetLastError());
        }
    } catch (...) {
        ClosePending();
        DeleteFileW(pending_path_.c_str());
        throw;
    }
}

std::filesystem::path FatalRecordStore::Persist(
    const FailureContext& failure) {
    if (pending_handle_ == INVALID_HANDLE_VALUE || persisted_) {
        throw std::logic_error(
            "AIE4 fatal record store was not prepared");
    }

    const auto failure_time = std::chrono::system_clock::now();
    const std::string failure_utc = FormatUtc(failure_time);
    const std::string record = SerializeFailure(
        failure,
        FormatUtc(process_start_),
        failure_utc,
        pid_);
    RewindAndTruncate(pending_handle_);
    WriteAll(pending_handle_, record);
    if (!FlushFileBuffers(pending_handle_)) {
        throw FatalRecordError(
            "flush",
            pending_path_,
            GetLastError());
    }
    ClosePending();

    const auto final_path =
        root_ /
        (std::string(kFinalPrefix) + failure_utc + "-" +
         std::to_string(pid_) + std::string(kFinalSuffix));
    if (!MoveFileExW(
            pending_path_.c_str(),
            final_path.c_str(),
            MOVEFILE_WRITE_THROUGH)) {
        throw FatalRecordError(
            "atomic rename",
            final_path,
            GetLastError());
    }
    persisted_ = true;
    return final_path;
}

std::vector<std::string> FatalRecordStore::DrainPriorRecords(
    std::ostream& output) {
    return DrainPriorRecords(
        LocalAppDataLogRoot(),
        ProbeProcessStart,
        output);
}

std::vector<std::string> FatalRecordStore::DrainPriorRecords(
    const std::filesystem::path& root,
    ProcessProbe process_probe,
    std::ostream& output) {
    std::error_code error;
    if (!std::filesystem::exists(root, error)) {
        if (error) {
            EmitDrainWarning(
                output,
                "inspect",
                root,
                error.message());
        }
        return {};
    }
    if (!std::filesystem::is_directory(root, error) || error) {
        EmitDrainWarning(
            output,
            "inspect",
            root,
            error ? error.message() : "not a directory");
        return {};
    }

    std::vector<std::filesystem::path> final_paths;
    std::vector<std::filesystem::path> pending_paths;
    try {
        for (std::filesystem::directory_iterator iterator(root, error), end;
             !error && iterator != end;
             iterator.increment(error)) {
            const auto filename = iterator->path().filename().string();
            if (HasPrefixAndSuffix(
                    filename,
                    kFinalPrefix,
                    kFinalSuffix)) {
                final_paths.push_back(iterator->path());
            } else if (HasPrefixAndSuffix(
                           filename,
                           kPendingPrefix,
                           kPendingSuffix)) {
                pending_paths.push_back(iterator->path());
            }
        }
    } catch (const std::exception& exception) {
        EmitDrainWarning(
            output,
            "enumerate",
            root,
            exception.what());
    } catch (...) {
        EmitDrainWarning(
            output,
            "enumerate",
            root,
            "unknown error");
    }
    if (error) {
        EmitDrainWarning(
            output,
            "enumerate",
            root,
            error.message());
    }
    std::sort(final_paths.begin(), final_paths.end());
    std::sort(pending_paths.begin(), pending_paths.end());

    std::vector<std::string> records;
    records.reserve(final_paths.size() + pending_paths.size());
    for (const auto& path : final_paths) {
        std::string record;
        try {
            record = ReadRecord(path);
        } catch (const std::exception& exception) {
            EmitDrainWarning(
                output,
                "read",
                path,
                exception.what());
            continue;
        } catch (...) {
            EmitDrainWarning(
                output,
                "read",
                path,
                "unknown error");
            continue;
        }
        EmitRecord(output, record);
        records.push_back(std::move(record));
        RemoveReportedRecord(path, output);
    }

    for (const auto& path : pending_paths) {
        const auto identity = ParsePendingIdentity(path);
        if (!identity.has_value() || !process_probe) {
            continue;
        }

        std::optional<std::chrono::system_clock::time_point> current_start;
        try {
            current_start = process_probe(identity->pid);
        } catch (...) {
            continue;
        }
        if (!current_start.has_value() ||
            FormatUtc(*current_start) == identity->timestamp) {
            continue;
        }

        std::string record =
            "incomplete corelib fatal record: " +
            path.filename().string() + "\n";
        EmitRecord(output, record);
        records.push_back(std::move(record));
        RemoveReportedRecord(path, output);
    }
    return records;
}

void FatalRecordStore::RemoveUnusedPending() noexcept {
    if (persisted_) {
        return;
    }
    ClosePending();
    if (!pending_path_.empty()) {
        DeleteFileW(pending_path_.c_str());
    }
}

const std::filesystem::path& FatalRecordStore::pending_path() const noexcept {
    return pending_path_;
}

void FatalRecordStore::ClosePending() noexcept {
    if (pending_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(pending_handle_);
        pending_handle_ = INVALID_HANDLE_VALUE;
    }
}

}  // namespace flm::corelib
