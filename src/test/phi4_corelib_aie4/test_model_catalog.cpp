#include "test_support.hpp"

#include <pull/model_overlay.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <windows.h>

namespace {

using nlohmann::json;

constexpr std::string_view kTag = "phi4-mini-it-aie4:4b";
constexpr std::string_view kCommit =
    "e751fb68c2cfffe6b0d32942118f75ac0a0365bb";
constexpr std::string_view kRepository =
    "https://huggingface.co/amd/phi-4-mini-instruct-oga-dml";

json ReadJson(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    return json::parse(input);
}

std::uint64_t FileSize(const std::filesystem::path& path) {
    return static_cast<std::uint64_t>(
        std::filesystem::file_size(path));
}

std::set<std::string> JsonStringSet(const json& values) {
    std::set<std::string> result;
    for (const auto& value : values) {
        result.insert(value.get<std::string>());
    }
    return result;
}

class TempDirectory final {
public:
    TempDirectory() {
        const auto parent = std::filesystem::temp_directory_path();
        for (int attempt = 0; attempt < 100; ++attempt) {
            path_ = parent /
                    ("flm-model-overlay-" +
                     std::to_string(GetCurrentProcessId()) + "-" +
                     std::to_string(GetTickCount64()) + "-" +
                     std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(path_, error)) {
                return;
            }
        }
        throw std::runtime_error("failed to create temporary directory");
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void TestCatalogContract() {
    const std::filesystem::path source(FLM_TEST_SOURCE_DIR);
    const json catalog = ReadJson(source / "model_list.json");
    const json metadata = ReadJson(source / "model_info.json");
    const auto& model =
        catalog.at("models").at("phi4-mini-it-aie4").at("4b");

    CHECK(model.at("url").get<std::string>() == kRepository);
    CHECK(model.at("revision").get<std::string>() == kCommit);
    const std::string expected_api =
        "https://huggingface.co/api/models/amd/"
        "phi-4-mini-instruct-oga-dml/tree/" +
        std::string(kCommit) +
        "?recursive=true&expand=false";
    CHECK(model.at("file_url") == expected_api);
    CHECK(model.at("flm_min_version") == "1.0.4");
    CHECK(model.at("default_context_length") == 4096);
    CHECK(model.at("max_prefill_len") == 4096);
    CHECK(model.at("vlm") == false);
    CHECK(model.at("modelscope_supported") == false);
    CHECK(!model.contains("ms_url"));
    CHECK(model.at("details").at("family") == "phi4");
    CHECK(
        model.at("details").at("execution_backend") ==
        "corelib_aie4");

    const std::set<std::string> expected_files{
        ".gitattributes",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "corelib_phi4_manifest.json",
        "genai_config.json",
        "merges.txt",
        "model.onnx",
        "model.onnx.data",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    };
    CHECK(JsonStringSet(model.at("files")) == expected_files);

    const auto& overlays = model.at("bundled_overlays");
    CHECK(overlays.size() == 3);
    CHECK(overlays.contains("config.json"));
    CHECK(overlays.contains("corelib_phi4_manifest.json"));
    CHECK(overlays.contains("tokenizer_config.json"));

    const std::filesystem::path overlay_root =
        source / "model_overlays";
    std::uint64_t overlay_size = 0;
    for (const auto& [target, record] : overlays.items()) {
        const auto path = overlay_root / record.at("path").get<std::string>();
        CHECK(std::filesystem::is_regular_file(path));
        CHECK(record.at("size").get<std::uint64_t>() == FileSize(path));
        CHECK(
            record.at("sha256") ==
            flm::pull::CalculateFileSha256(path));
        overlay_size += FileSize(path);
    }

    const auto& remote = metadata.at(kTag);
    CHECK(remote.size() == 11);
    CHECK(
        std::is_sorted(
            remote.begin(),
            remote.end(),
            [](const json& left, const json& right) {
                return left.at("path").get<std::string>() <
                       right.at("path").get<std::string>();
            }));
    const auto find_record = [&](std::string_view path) -> const json& {
        const auto found = std::find_if(
            remote.begin(),
            remote.end(),
            [&](const json& record) {
                return record.at("path").get<std::string>() == path;
            });
        CHECK(found != remote.end());
        return *found;
    };
    CHECK(
        find_record("model.onnx").at("oid") ==
        "df6d4309745d4627b82fd92c615589193ea528db");
    CHECK(
        find_record("model.onnx").at("lfs").at("oid") ==
        "e80b9d83e018784eda6263d09fa2ab7729087722c6073c72123366f2dbec4529");
    CHECK(
        find_record("model.onnx.data").at("lfs").at("size") ==
        UINT64_C(3248488448));
    CHECK(
        find_record("tokenizer.json").at("lfs").at("oid") ==
        "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea");

    constexpr std::uint64_t kRemoteLogicalBytes = UINT64_C(3270726823);
    constexpr std::uint64_t kReplacedTokenizerConfigBytes = 2654;
    const std::uint64_t expected_size =
        kRemoteLogicalBytes - kReplacedTokenizerConfigBytes + overlay_size;
    CHECK(model.at("size").get<std::uint64_t>() == expected_size);
    const double expected_footprint =
        std::round(
            static_cast<double>(expected_size) /
                static_cast<double>(UINT64_C(1024) * 1024 * 1024) *
                100.0) /
        100.0;
    CHECK(model.at("footprint").get<double>() == expected_footprint);
}

void TestSourcePolicyAndPinnedUrls() {
    const std::filesystem::path source(FLM_TEST_SOURCE_DIR);
    const json catalog = ReadJson(source / "model_list.json");
    const auto& model =
        catalog.at("models").at("phi4-mini-it-aie4").at("4b");

    flm::pull::RequireSupportedModelSource(model, false);
    CheckThrowsContains(
        [&] { flm::pull::RequireSupportedModelSource(model, true); },
        "does not support --modelscope");
    CHECK(
        flm::pull::BuildRemoteFileUrl(model, "model.onnx") ==
        std::string(kRepository) + "/resolve/" +
            std::string(kCommit) + "/model.onnx?download=true");

    const auto remote_files = flm::pull::RemoteModelFiles(model);
    CHECK(
        std::find(
            remote_files.begin(),
            remote_files.end(),
            "tokenizer_config.json") == remote_files.end());
    CHECK(
        std::find(
            remote_files.begin(),
            remote_files.end(),
            "model.onnx") != remote_files.end());
}

void TestBundledOverlayCopyAndIntegrity() {
    const std::filesystem::path source(FLM_TEST_SOURCE_DIR);
    const json catalog = ReadJson(source / "model_list.json");
    const auto& model =
        catalog.at("models").at("phi4-mini-it-aie4").at("4b");
    const auto overlay_root = source / "model_overlays";

    TempDirectory temporary;
    const auto model_dir = temporary.path() / "model";
    flm::pull::StageBundledOverlays(model, overlay_root, model_dir);

    for (const auto& [target, record] :
         model.at("bundled_overlays").items()) {
        const auto installed = model_dir / target;
        CHECK(std::filesystem::is_regular_file(installed));
        CHECK(
            flm::pull::CalculateFileSha256(installed) ==
            record.at("sha256").get<std::string>());
    }

    const auto corrupt_root = temporary.path() / "corrupt";
    std::filesystem::create_directories(
        corrupt_root / "phi4-mini-it-aie4");
    for (const auto& [target, record] :
         model.at("bundled_overlays").items()) {
        std::filesystem::copy_file(
            overlay_root / record.at("path").get<std::string>(),
            corrupt_root / record.at("path").get<std::string>());
    }
    {
        std::ofstream corrupt(
            corrupt_root / "phi4-mini-it-aie4" / "config.json",
            std::ios::binary | std::ios::trunc);
        corrupt << std::string(
            model.at("bundled_overlays")
                .at("config.json")
                .at("size")
                .get<std::size_t>(),
            'x');
    }
    CheckThrowsContains(
        [&] {
            flm::pull::StageBundledOverlays(
                model,
                corrupt_root,
                temporary.path() / "rejected");
        },
        "bundled overlay SHA-256 mismatch");
    CHECK(
        !std::filesystem::exists(
            temporary.path() / "rejected" / "config.json"));
}

}  // namespace

int main() {
    try {
        TestCatalogContract();
        TestSourcePolicyAndPinnedUrls();
        TestBundledOverlayCopyAndIntegrity();
        std::cout << "model catalog tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
