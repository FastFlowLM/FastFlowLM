#include <pull/model_overlay.hpp>

#include "picosha2.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace flm::pull {
namespace {

struct OverlayRecord final {
    std::string target;
    std::filesystem::path source;
    std::uint64_t size;
    std::string sha256;
};

bool IsSha256(std::string_view value) {
    return value.size() == 64 &&
           std::all_of(
               value.begin(),
               value.end(),
               [](unsigned char character) {
                   return std::isxdigit(character) != 0;
               });
}

std::filesystem::path SafeRelativePath(
    const std::string& value,
    std::string_view field) {
    std::filesystem::path path(value);
    if (
        value.empty() ||
        path.is_absolute() ||
        path.has_root_name() ||
        path.has_root_directory()) {
        throw std::invalid_argument(
            std::string(field) + " must be a non-empty relative path");
    }
    for (const auto& component : path) {
        if (component == "..") {
            throw std::invalid_argument(
                std::string(field) + " must not contain path traversal");
        }
    }
    return path.lexically_normal();
}

std::vector<OverlayRecord> ReadOverlayRecords(
    const nlohmann::json& model_info,
    const std::filesystem::path& overlay_root) {
    const auto found = model_info.find("bundled_overlays");
    if (found == model_info.end()) {
        return {};
    }
    if (!found->is_object()) {
        throw std::invalid_argument("bundled_overlays must be an object");
    }

    std::vector<OverlayRecord> records;
    records.reserve(found->size());
    for (const auto& [target, value] : found->items()) {
        if (!value.is_object()) {
            throw std::invalid_argument(
                "bundled overlay record must be an object: " + target);
        }
        if (
            !value.contains("path") ||
            !value.contains("size") ||
            !value.contains("sha256") ||
            value.size() != 3) {
            throw std::invalid_argument(
                "bundled overlay record must contain path, size, and "
                "sha256: " +
                target);
        }
        const auto target_path =
            SafeRelativePath(target, "bundled overlay target");
        if (
            target_path.parent_path() != std::filesystem::path() ||
            target_path.filename().string() != target) {
            throw std::invalid_argument(
                "bundled overlay target must be a model-root filename");
        }
        const auto source_relative = SafeRelativePath(
            value.at("path").get<std::string>(),
            "bundled overlay source");
        const auto size = value.at("size").get<std::uint64_t>();
        std::string sha256 = value.at("sha256").get<std::string>();
        if (!IsSha256(sha256)) {
            throw std::invalid_argument(
                "bundled overlay SHA-256 is invalid: " + target);
        }
        std::transform(
            sha256.begin(),
            sha256.end(),
            sha256.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
        records.push_back(
            OverlayRecord{
                target,
                overlay_root / source_relative,
                size,
                std::move(sha256)});
    }
    return records;
}

void VerifySource(const OverlayRecord& record) {
    std::error_code error;
    const auto size = std::filesystem::file_size(record.source, error);
    if (error) {
        throw std::runtime_error(
            "bundled overlay source is missing or unreadable: " +
            record.source.string());
    }
    if (size != record.size) {
        throw std::runtime_error(
            "bundled overlay size mismatch: " +
            record.source.string());
    }
    if (CalculateFileSha256(record.source) != record.sha256) {
        throw std::runtime_error(
            "bundled overlay SHA-256 mismatch: " +
            record.source.string());
    }
}

bool TargetMatches(
    const OverlayRecord& record,
    const std::filesystem::path& model_dir) {
    const auto target = model_dir / record.target;
    std::error_code error;
    const auto size = std::filesystem::file_size(target, error);
    return !error &&
           size == record.size &&
           CalculateFileSha256(target) == record.sha256;
}

void CopyAtomically(
    const OverlayRecord& record,
    const std::filesystem::path& model_dir) {
    static std::atomic<std::uint64_t> sequence{0};
    const auto target = model_dir / record.target;
    const auto temporary =
        model_dir /
        ("." + record.target + ".flm-overlay-" +
#ifdef _WIN32
         std::to_string(GetCurrentProcessId()) + "-" +
#else
         std::string("process-") +
#endif
         std::to_string(
             sequence.fetch_add(1, std::memory_order_relaxed)) +
         ".tmp");

    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    try {
#ifdef _WIN32
        if (!CopyFileW(
                record.source.c_str(),
                temporary.c_str(),
                TRUE)) {
            throw std::system_error(
                static_cast<int>(GetLastError()),
                std::system_category(),
                "failed to copy bundled overlay to temporary file");
        }
#else
        std::filesystem::copy_file(
            record.source,
            temporary,
            std::filesystem::copy_options::none);
#endif
        OverlayRecord temporary_record = record;
        temporary_record.source = temporary;
        VerifySource(temporary_record);
#ifdef _WIN32
        if (!MoveFileExW(
                temporary.c_str(),
                target.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            throw std::system_error(
                static_cast<int>(GetLastError()),
                std::system_category(),
                "failed to atomically install bundled overlay");
        }
#else
        std::filesystem::rename(temporary, target);
#endif
    } catch (...) {
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

}  // namespace

void RequireSupportedModelSource(
    const nlohmann::json& model_info,
    bool use_modelscope) {
    if (!use_modelscope) {
        return;
    }
    const auto found = model_info.find("modelscope_supported");
    if (
        found != model_info.end() &&
        (!found->is_boolean() || !found->get<bool>())) {
        throw std::invalid_argument(
            "this model does not support --modelscope; use the "
            "Hugging Face source");
    }
}

std::vector<std::string> RemoteModelFiles(
    const nlohmann::json& model_info) {
    const auto overlays = model_info.find("bundled_overlays");
    std::vector<std::string> files;
    for (const auto& value : model_info.at("files")) {
        const std::string filename = value.get<std::string>();
        if (
            overlays == model_info.end() ||
            !overlays->is_object() ||
            !overlays->contains(filename)) {
            files.push_back(filename);
        }
    }
    return files;
}

std::string BuildRemoteFileUrl(
    const nlohmann::json& model_info,
    std::string_view filename) {
    const std::string base_url = model_info.at("url").get<std::string>();
    if (base_url.find("resolve") != std::string::npos) {
        return base_url + "/" + std::string(filename) + "?download=true";
    }
    const std::string revision =
        model_info.value("revision", std::string("main"));
    return base_url + "/resolve/" + revision + "/" +
           std::string(filename) + "?download=true";
}

std::string CalculateFileSha256(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "failed to open file for SHA-256: " + path.string());
    }
    std::vector<unsigned char> hash(picosha2::k_digest_size);
    picosha2::hash256(input, hash.begin(), hash.end());
    return picosha2::bytes_to_hex_string(hash.begin(), hash.end());
}

void StageBundledOverlays(
    const nlohmann::json& model_info,
    const std::filesystem::path& overlay_root,
    const std::filesystem::path& model_dir) {
    const auto records = ReadOverlayRecords(model_info, overlay_root);
    for (const auto& record : records) {
        VerifySource(record);
    }
    if (records.empty()) {
        return;
    }
    std::filesystem::create_directories(model_dir);
    for (const auto& record : records) {
        if (!TargetMatches(record, model_dir)) {
            CopyAtomically(record, model_dir);
        }
    }
}

bool VerifyBundledOverlayTarget(
    const nlohmann::json& model_info,
    std::string_view filename,
    const std::filesystem::path& model_dir) {
    const auto records = ReadOverlayRecords(model_info, {});
    const auto found = std::find_if(
        records.begin(),
        records.end(),
        [&](const OverlayRecord& record) {
            return record.target == filename;
        });
    return found != records.end() && TargetMatches(*found, model_dir);
}

}  // namespace flm::pull
