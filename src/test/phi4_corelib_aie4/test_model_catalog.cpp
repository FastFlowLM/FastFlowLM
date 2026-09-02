#include "test_support.hpp"

#include <pull/model_overlay.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

    // Design 8.1. `.gitattributes` is a Git repository artifact and
    // `genai_config.json` is excluded by MODEL-2: flm.exe runs no ORT or genai
    // graph, and an unused configuration invites a future reader to believe it
    // is authoritative. `chat_template.jinja` is kept deliberately, because it
    // is the verbatim source of the template the overlay inlines and is what
    // makes the overlay auditable on the target machine.
    const std::set<std::string> expected_files{
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "corelib_phi4_manifest.json",
        "merges.txt",
        "model.onnx",
        "model.onnx.data",
        "provenance.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    };
    CHECK(JsonStringSet(model.at("files")) == expected_files);

    const auto& overlays = model.at("bundled_overlays");
    CHECK(overlays.size() == 4);
    CHECK(overlays.contains("config.json"));
    CHECK(overlays.contains("corelib_phi4_manifest.json"));
    CHECK(overlays.contains("provenance.json"));
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

    // `size` and `footprint` cover the assembled on-disk directory: the
    // upstream files actually downloaded plus the overlay files shipped inside
    // FastFlow. Deriving it from the metadata here, rather than restating a
    // literal, is what keeps the number honest when the file set changes.
    std::uint64_t upstream_size = 0;
    for (const auto& file : model.at("files")) {
        const auto name = file.get<std::string>();
        if (overlays.contains(name)) {
            continue;
        }
        upstream_size += find_record(name).at("size").get<std::uint64_t>();
    }
    const std::uint64_t expected_size = upstream_size + overlay_size;
    CHECK(model.at("size").get<std::uint64_t>() == expected_size);
    const double expected_footprint =
        std::round(
            static_cast<double>(expected_size) /
                static_cast<double>(UINT64_C(1024) * 1024 * 1024) *
                100.0) /
        100.0;
    CHECK(model.at("footprint").get<double>() == expected_footprint);
}

// Design `PACKAGE-1`. The assembled package has two provenances and they are
// checked separately. Requiring every catalog file to carry a Hugging Face
// metadata record cannot hold: the overlay files are FastFlow-authored and do
// not exist upstream by construction. The check that matters is the reverse
// one.
void TestSplitProvenance() {
    const std::filesystem::path source(FLM_TEST_SOURCE_DIR);
    const json catalog = ReadJson(source / "model_list.json");
    const json metadata = ReadJson(source / "model_info.json");
    const auto& model =
        catalog.at("models").at("phi4-mini-it-aie4").at("4b");
    const auto& overlays = model.at("bundled_overlays");
    const auto& remote = metadata.at(kTag);

    std::set<std::string> upstream_paths;
    for (const auto& record : remote) {
        upstream_paths.insert(record.at("path").get<std::string>());
    }

    // Upstream files: every one must have a metadata record to download and
    // hash-check against.
    for (const auto& file : model.at("files")) {
        const auto name = file.get<std::string>();
        if (overlays.contains(name)) {
            continue;
        }
        CHECK(upstream_paths.count(name) == 1);
    }

    // Overlay files: each must be installed beside FastFlow. Three of them
    // must have no upstream record at all; an upstream record for one would
    // mean FastFlow's contract was published to the model repository, which
    // makes the two provenances indistinguishable.
    const std::filesystem::path overlay_root = source / "model_overlays";
    for (const auto& name : {
             std::string("config.json"),
             std::string("corelib_phi4_manifest.json"),
             std::string("provenance.json")}) {
        CHECK(overlays.contains(name));
        CHECK(upstream_paths.count(name) == 0);
        CHECK(
            std::filesystem::is_regular_file(
                overlay_root /
                overlays.at(name).at("path").get<std::string>()));
    }

    // tokenizer_config.json is the one overlay that also exists upstream. The
    // overlay shadows it because the published file carries neither a chat
    // template nor eos_token_id, so its upstream record is expected.
    CHECK(overlays.contains("tokenizer_config.json"));
    CHECK(upstream_paths.count("tokenizer_config.json") == 1);

    // Nothing upstream is silently dropped: every record is either packaged or
    // one of the two files excluded on purpose.
    const std::set<std::string> excluded{".gitattributes", "genai_config.json"};
    const auto packaged = JsonStringSet(model.at("files"));
    for (const auto& path : upstream_paths) {
        CHECK(packaged.count(path) == 1 || excluded.count(path) == 1);
    }
    for (const auto& name : excluded) {
        CHECK(upstream_paths.count(name) == 1);
        CHECK(packaged.count(name) == 0);
    }
}

// `ModelDownloader::check_model_compatibility` compares three versions: the
// overlay config.json's flm_version, the catalog's flm_min_version, and the
// binary's own __FLM_VERSION__. A binary built as 1.0.3 reports Incompatible
// for a 1.0.4 overlay and refuses its own catalog entry, which is a failure
// that only appears at first real load. Pin the relationship here instead.
void TestVersionGateIsSelfConsistent() {
    const std::filesystem::path source(FLM_TEST_SOURCE_DIR);
    const json catalog = ReadJson(source / "model_list.json");
    const auto& model =
        catalog.at("models").at("phi4-mini-it-aie4").at("4b");
    const json overlay_config =
        ReadJson(source / "model_overlays" / "phi4-mini-it-aie4" /
                 "config.json");

    const auto encode = [](const std::string& version) -> std::uint32_t {
        int major = -1;
        int minor = -1;
        int patch = -1;
        CHECK(
            std::sscanf(version.c_str(), "%d.%d.%d", &major, &minor, &patch) ==
            3);
        CHECK(major >= 0 && minor >= 0 && patch >= 0);
        return static_cast<std::uint32_t>(
            major * 1000000 + minor * 1000 + patch);
    };

    const auto minimum = model.at("flm_min_version").get<std::string>();
    const auto packaged = overlay_config.at("flm_version").get<std::string>();
    CHECK(packaged == minimum);
    CHECK(encode(packaged) >= encode(minimum));
    CHECK(encode(std::string(__FLM_VERSION__)) >= encode(packaged));
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
        TestSplitProvenance();
        TestVersionGateIsSelfConsistent();
        TestSourcePolicyAndPinnedUrls();
        TestBundledOverlayCopyAndIntegrity();
        std::cout << "model catalog tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
