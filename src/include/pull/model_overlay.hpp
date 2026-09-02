#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace flm::pull {

void RequireSupportedModelSource(
    const nlohmann::json& model_info,
    bool use_modelscope);

std::vector<std::string> RemoteModelFiles(
    const nlohmann::json& model_info);

std::string BuildRemoteFileUrl(
    const nlohmann::json& model_info,
    std::string_view filename);

std::string CalculateFileSha256(const std::filesystem::path& path);

void StageBundledOverlays(
    const nlohmann::json& model_info,
    const std::filesystem::path& overlay_root,
    const std::filesystem::path& model_dir);

bool VerifyBundledOverlayTarget(
    const nlohmann::json& model_info,
    std::string_view filename,
    const std::filesystem::path& model_dir);

}  // namespace flm::pull
