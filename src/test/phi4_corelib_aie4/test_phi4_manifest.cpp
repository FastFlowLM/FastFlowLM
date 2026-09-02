#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <models/phi4/phi4_corelib_manifest.hpp>
#include <nlohmann/json.hpp>

#include <windows.h>
#include <winioctl.h>

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using flm::corelib::CorelibApi;
using flm::phi4::InitializerView;
using flm::phi4::MappedFile;
using flm::phi4::Phi4Package;
using flm::phi4::SourceDType;
using flm::phi4::WeightObjectKind;
using nlohmann::json;

constexpr std::string_view kManifestName = "corelib_phi4_manifest.json";
constexpr std::string_view kDataFile = "z-data.bin";
constexpr std::string_view kRopeFile = "z-rope.bin";
constexpr std::uint64_t kDataBytes =
    200064ull * 3072ull * sizeof(std::uint16_t);
constexpr std::size_t kRopeRows = 4096;
constexpr std::size_t kRopeColumns = 64;
constexpr std::uint64_t kRopeBytes =
    kRopeRows * kRopeColumns * sizeof(std::uint16_t);
constexpr std::uint64_t kRopeMappedBytes = kRopeBytes + 4096;
constexpr std::uint64_t kFp32ScaleOffset = 4096;
constexpr std::uint64_t kNormOffset = 1024 * 1024;
constexpr std::uint64_t kSinOffset = 2 * 1024 * 1024;

constexpr std::string_view kFp32Scale =
    "model.layers.0.attn.q_proj.MatMulNBits.scales";
constexpr std::string_view kFp16Scale =
    "model.layers.0.attn.k_proj.MatMulNBits.scales";
constexpr std::string_view kFp32Norm =
    "model.layers.0.post_attention_layernorm.weight";

template <class Function>
void* FunctionAddress(Function function) {
    return reinterpret_cast<void*>(function);
}

std::shared_ptr<CorelibApi> ResolveRecordingCorelib() {
    auto resolver = flm::test::CompleteCorelibResolver();
    return CorelibApi::ResolveForTest(
        [resolver = std::move(resolver)](std::string_view name) mutable
            -> void* {
            const auto found = resolver.find(std::string(name));
            return found == resolver.end() ? nullptr : found->second;
        });
}

class TempDirectory final {
public:
    TempDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("fastflowlm-phi4-manifest-" +
                 std::to_string(GetCurrentProcessId()) + "-" +
                 std::to_string(nonce));
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

void CreateSparseFile(
    const std::filesystem::path& path,
    std::uint64_t size) {
    HANDLE file = CreateFileW(
        path.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("failed to create sparse test file");
    }

    DWORD ignored = 0;
    DeviceIoControl(
        file,
        FSCTL_SET_SPARSE,
        nullptr,
        0,
        nullptr,
        0,
        &ignored,
        nullptr);
    LARGE_INTEGER end{};
    end.QuadPart = static_cast<LONGLONG>(size);
    const bool success =
        SetFilePointerEx(file, end, nullptr, FILE_BEGIN) != FALSE &&
        SetEndOfFile(file) != FALSE;
    CloseHandle(file);
    if (!success) {
        throw std::runtime_error("failed to size sparse test file");
    }
}

template <class T>
void WriteValues(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::span<const T> values) {
    std::fstream file(
        path,
        std::ios::in | std::ios::out | std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open synthetic data file");
    }
    file.seekp(static_cast<std::streamoff>(offset));
    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size_bytes()));
    if (!file) {
        throw std::runtime_error("failed to write synthetic data");
    }
}

std::uint64_t ItemSize(std::string_view dtype) {
    if (dtype == "uint8") {
        return 1;
    }
    if (dtype == "float16") {
        return 2;
    }
    if (dtype == "float32") {
        return 4;
    }
    if (dtype == "int64") {
        return 8;
    }
    throw std::runtime_error("unsupported synthetic dtype");
}

std::uint64_t ByteLength(
    std::string_view dtype,
    const std::vector<std::int64_t>& shape) {
    std::uint64_t elements = 1;
    for (const std::int64_t dimension : shape) {
        elements *= static_cast<std::uint64_t>(dimension);
    }
    return elements * ItemSize(dtype);
}

void AddInitializer(
    json& initializers,
    std::string name,
    std::string dtype,
    std::vector<std::int64_t> shape,
    std::string role,
    std::string file = std::string(kDataFile),
    std::uint64_t offset = 0) {
    CHECK(!initializers.contains(name));
    const std::uint64_t length = ByteLength(dtype, shape);
    initializers[std::move(name)] = {
        {"file", std::move(file)},
        {"offset", offset},
        {"length", length},
        {"dtype", std::move(dtype)},
        {"shape", std::move(shape)},
        {"role", std::move(role)}};
}

void AddMatMul(
    json& manifest,
    const std::string& name,
    std::int64_t k,
    std::int64_t n) {
    json roles = {
        {"qweight", name + ".qweight"},
        {"scales", name + ".scales"},
        {"qzeros", name + ".qzeros"}};
    manifest["weight_objects"].push_back({
        {"name", name},
        {"kind", "matmul"},
        {"descriptor",
         {
             {"k", k},
             {"n", n},
             {"group_size", 128},
             {"has_bias", false},
         }},
        {"roles", roles}});

    auto& initializers = manifest["initializers"];
    AddInitializer(
        initializers,
        roles["qweight"].get<std::string>(),
        "uint8",
        {n, k / 2},
        "matmul.qweight");

    std::string scale_dtype = "float16";
    std::uint64_t scale_offset = 0;
    if (name == "model.layers.0.attn.q_proj.MatMulNBits") {
        scale_dtype = "float32";
        scale_offset = kFp32ScaleOffset;
    }
    AddInitializer(
        initializers,
        roles["scales"].get<std::string>(),
        scale_dtype,
        {n, k / 128},
        "matmul.scales",
        std::string(kDataFile),
        scale_offset);
    AddInitializer(
        initializers,
        roles["qzeros"].get<std::string>(),
        "uint8",
        {n, ((k / 128) + 1) / 2},
        "matmul.qzeros");
}

void AddSsMlp(json& manifest, int layer) {
    const std::string base =
        "model.layers." + std::to_string(layer);
    const std::string object_name = base + ".ssmlp";
    const std::string norm0 =
        base + ".post_attention_layernorm.weight";
    const std::string norm1 =
        layer == 31
            ? "model.layers.32.final_norm_layernorm.weight"
            : "model.layers." + std::to_string(layer + 1) +
                  ".input_layernorm.weight";

    json roles = {
        {"norm0", norm0},
        {"norm1", norm1},
    };
    auto& initializers = manifest["initializers"];
    for (const std::string projection : {"gate", "up", "down"}) {
        const std::int64_t k = projection == "down" ? 8192 : 3072;
        const std::int64_t n = projection == "down" ? 3072 : 8192;
        const std::string prefix =
            base + ".mlp." + projection + "_proj.MatMulNBits";
        for (const std::string component :
             {"qweight", "scales", "qzeros"}) {
            roles[projection + "_" + component] =
                prefix + "." + component;
        }
        AddInitializer(
            initializers,
            prefix + ".qweight",
            "uint8",
            {n, k / 2},
            "ssmlp." + projection + ".qweight");
        AddInitializer(
            initializers,
            prefix + ".scales",
            "float16",
            {n, k / 128},
            "ssmlp." + projection + ".scales");
        AddInitializer(
            initializers,
            prefix + ".qzeros",
            "uint8",
            {n, ((k / 128) + 1) / 2},
            "ssmlp." + projection + ".qzeros");
    }

    AddInitializer(
        initializers,
        norm0,
        layer == 0 ? "float32" : "float16",
        {3072},
        "ssmlp.norm0",
        std::string(kDataFile),
        layer == 0 ? kNormOffset : 0);
    AddInitializer(
        initializers,
        norm1,
        "float16",
        {3072},
        "ssmlp.norm1");

    manifest["weight_objects"].push_back({
        {"name", object_name},
        {"kind", "ssmlp"},
        {"descriptor",
         {
             {"k", 3072},
             {"n", 8192},
             {"group_size", 128},
         }},
        {"roles", std::move(roles)}});
}

json BuildManifest(std::uint64_t model_size) {
    json manifest = {
        {"schema_version", 1},
        {"execution_backend", "corelib_aie4"},
        {"model",
         {
             {"family", "phi4"},
             {"layers", 32},
             {"hidden_size", 3072},
             {"intermediate_size", 8192},
             {"num_heads", 24},
             {"kv_heads", 8},
             {"head_size", 128},
             {"vocab_size", 200064},
             {"group_size", 128},
             {"rope_dim", 96},
             {"rms_epsilon", 0.00001},
         }},
        {"backend", {{"max_seq", 4096}}},
        {"files",
         {
             {"model.onnx", {{"size", model_size}}},
             {std::string(kDataFile), {{"size", kDataBytes}}},
             {std::string(kRopeFile), {{"size", kRopeMappedBytes}}},
         }},
        {"initializers", json::object()},
        {"weight_objects", json::array()},
    };

    for (int layer = 0; layer < 32; ++layer) {
        const std::string base =
            "model.layers." + std::to_string(layer) + ".attn.";
        AddMatMul(
            manifest,
            base + "q_proj.MatMulNBits",
            3072,
            3072);
        AddMatMul(
            manifest,
            base + "k_proj.MatMulNBits",
            3072,
            1024);
        AddMatMul(
            manifest,
            base + "v_proj.MatMulNBits",
            3072,
            1024);
        AddMatMul(
            manifest,
            base + "o_proj.MatMulNBits",
            3072,
            3072);
        AddSsMlp(manifest, layer);
    }
    AddMatMul(manifest, "lm_head.MatMulNBits", 3072, 200064);

    AddInitializer(
        manifest["initializers"],
        "model.embed_tokens.weight",
        "float16",
        {200064, 3072},
        "embedding");
    AddInitializer(
        manifest["initializers"],
        "model.layers.0.input_layernorm.weight",
        "float32",
        {3072},
        "input_norm",
        std::string(kDataFile),
        kNormOffset);
    AddInitializer(
        manifest["initializers"],
        "cos_cache",
        "float16",
        {4096, 64},
        "cos_cache",
        std::string(kRopeFile));
    AddInitializer(
        manifest["initializers"],
        "sin_cache",
        "float32",
        {4096, 48},
        "sin_cache",
        std::string(kDataFile),
        kSinOffset);

    CHECK(manifest["weight_objects"].size() == 161);
    CHECK(manifest["initializers"].size() == 743);
    return manifest;
}

class SyntheticPackage final {
public:
    SyntheticPackage() {
        const auto model_path = temp_.path() / "model.onnx";
        {
            std::ofstream model(model_path, std::ios::binary);
            model << "model";
        }
        CreateSparseFile(temp_.path() / kDataFile, kDataBytes);
        CreateSparseFile(
            temp_.path() / kRopeFile,
            kRopeMappedBytes);

        const std::array<std::uint16_t, 2> half_values{
            0x3c00u,
            0xc000u};
        const std::array<float, 2> float_values{1.0f, -2.0f};
        WriteValues(
            temp_.path() / kDataFile,
            0,
            std::span<const std::uint16_t>(half_values));
        WriteValues(
            temp_.path() / kDataFile,
            kFp32ScaleOffset,
            std::span<const float>(float_values));
        WriteValues(
            temp_.path() / kDataFile,
            kNormOffset,
            std::span<const float>(float_values));
        const std::array<float, 1> sin_value{4.0f};
        WriteValues(
            temp_.path() / kDataFile,
            kSinOffset,
            std::span<const float>(sin_value));

        const std::array<std::uint16_t, 1> one{0x3c00u};
        const std::array<std::uint16_t, 1> two{0x4000u};
        const std::array<std::uint16_t, 1> three{0x4200u};
        WriteValues(
            temp_.path() / kRopeFile,
            0,
            std::span<const std::uint16_t>(one));
        WriteValues(
            temp_.path() / kRopeFile,
            kRopeColumns * sizeof(std::uint16_t),
            std::span<const std::uint16_t>(two));
        WriteValues(
            temp_.path() / kRopeFile,
            ((kRopeRows - 1) * kRopeColumns + 47) *
                sizeof(std::uint16_t),
            std::span<const std::uint16_t>(three));

        manifest_ = BuildManifest(
            std::filesystem::file_size(model_path));
        Write(manifest_);
    }

    const std::filesystem::path& path() const noexcept {
        return temp_.path();
    }

    json manifest() const {
        return manifest_;
    }

    void Write(const json& manifest) const {
        std::ofstream stream(
            temp_.path() / kManifestName,
            std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("failed to write synthetic manifest");
        }
        stream << manifest.dump(2) << '\n';
    }

private:
    TempDirectory temp_;
    json manifest_;
};

void RenameFile(
    json& manifest,
    std::string_view old_name,
    std::string new_name) {
    auto record = manifest["files"].at(std::string(old_name));
    manifest["files"].erase(std::string(old_name));
    manifest["files"][new_name] = std::move(record);
    for (auto& [_, initializer] :
         manifest["initializers"].items()) {
        if (
            initializer["file"].get<std::string>() ==
            std::string(old_name)) {
            initializer["file"] = new_name;
        }
    }
}

template <class Mutation>
void ExpectLoadFailure(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api,
    Mutation&& mutation,
    std::string_view expected,
    bool verify_full_hash = false) {
    json manifest = fixture.manifest();
    std::invoke(
        std::forward<Mutation>(mutation),
        manifest);
    fixture.Write(manifest);
    try {
        (void)Phi4Package::Load(
            fixture.path(),
            api,
            verify_full_hash);
    } catch (const std::exception& error) {
        if (
            std::string_view(error.what()).find(expected) ==
            std::string_view::npos) {
            throw std::runtime_error(
                "expected load failure containing '" +
                std::string(expected) + "', got: " + error.what());
        }
        return;
    }
    throw std::runtime_error(
        "expected package load failure containing '" +
        std::string(expected) + "'");
}

void TestValidMappingAndExplicitRoles(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    fixture.Write(fixture.manifest());

    auto package = Phi4Package::Load(fixture.path(), api, false);
    CHECK(package.weight_objects().size() == 161);
    const auto& first = package.weight_objects().front();
    CHECK(first.name ==
          "model.layers.0.attn.q_proj.MatMulNBits");
    CHECK(first.kind == WeightObjectKind::MatMul);
    CHECK(first.k == 3072);
    CHECK(first.n == 3072);
    CHECK(first.group_size == 128);
    CHECK(!first.has_bias);
    CHECK(
        first.components.at("qweight") ==
        "model.layers.0.attn.q_proj.MatMulNBits.qweight");

    const auto& last = package.weight_objects().back();
    CHECK(last.name == "lm_head.MatMulNBits");
    CHECK(last.n == 200064);

    const auto& embedding =
        package.Require("model.embed_tokens.weight");
    CHECK(embedding.dtype == SourceDType::Float16);
    CHECK(embedding.shape ==
          std::vector<std::int64_t>({200064, 3072}));
    CHECK(embedding.size == kDataBytes);
    CHECK(embedding.data != nullptr);
    CHECK(embedding.owner != nullptr);
    CHECK(embedding.owner->path().filename() == kDataFile);
    CHECK(embedding.owner->size() == kDataBytes);
    CheckThrowsContains(
        [&] {
            (void)package.Require("missing.initializer");
        },
        "missing initializer");
}

void TestOgaQuantizedLayoutsAccepted(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    json manifest = fixture.manifest();
    for (auto& [name, initializer] :
         manifest["initializers"].items()) {
        if (name.ends_with(".qweight")) {
            const auto logical =
                initializer["shape"].get<std::vector<std::int64_t>>();
            CHECK(logical.size() == 2);
            CHECK(logical[1] % 64 == 0);
            initializer["shape"] = {
                logical[0],
                logical[1] / 64,
                64};
        } else if (
            name.ends_with(".scales") ||
            name.ends_with(".qzeros")) {
            const auto logical =
                initializer["shape"].get<std::vector<std::int64_t>>();
            CHECK(logical.size() == 2);
            initializer["shape"] = {logical[0] * logical[1]};
        }
    }
    fixture.Write(manifest);

    auto package = Phi4Package::Load(fixture.path(), api, false);
    CHECK(package.weight_objects().size() == 161);
    CHECK(
        package.Require(
            "model.layers.0.attn.q_proj.MatMulNBits.qweight")
            .shape ==
        std::vector<std::int64_t>({3072, 24, 64}));
}

void TestMappedOwnerOutlivesPackage(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    fixture.Write(fixture.manifest());
    std::optional<InitializerView> retained;
    {
        auto package =
            Phi4Package::Load(fixture.path(), api, false);
        retained = package.Require(kFp16Scale);
        CHECK(retained->owner.use_count() > 1);
    }
    CHECK(retained->owner.use_count() == 1);
    CHECK(retained->data != nullptr);
    CHECK(
        *reinterpret_cast<const std::uint16_t*>(retained->data) ==
        0x3c00u);

    auto direct =
        MappedFile::OpenReadOnly(fixture.path() / "model.onnx");
    CHECK(direct->size() == 5);
    CHECK(direct->path() == fixture.path() / "model.onnx");
    CHECK(direct->data()[0] == std::byte{'m'});
}

void TestPathRangeAndHashRejections(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            RenameFile(manifest, kDataFile, "C:/outside.bin");
        },
        "relative");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            RenameFile(manifest, kDataFile, "../outside.bin");
        },
        "traversal");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            auto& record = manifest["initializers"].at(
                "model.embed_tokens.weight");
            record["offset"] =
                std::numeric_limits<std::uint64_t>::max() - 1;
        },
        "range overflow");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            auto& record =
                manifest["initializers"].at(std::string(kFp16Scale));
            record["length"] =
                record["length"].get<std::uint64_t>() - 1;
        },
        "byte count");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["initializers"]
                    .at(std::string(kFp16Scale))["offset"] = 1;
        },
        "dtype-aligned");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["files"].at(std::string(kDataFile))["size"] =
                kDataBytes - 1;
        },
        "file size");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            for (auto& [_, record] : manifest["files"].items()) {
                record["sha256"] = std::string(64, '0');
            }
        },
        "SHA-256",
        true);
}

void TestWeightObjectRejections(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest.erase("weight_objects");
        },
        "weight_objects");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["weight_objects"] = json::array();
        },
        "weight_objects");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["weight_objects"].erase(
                manifest["weight_objects"].end() - 1);
        },
        "161");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["weight_objects"][1]["name"] =
                manifest["weight_objects"][0]["name"];
        },
        "duplicate weight object");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["weight_objects"][0]["kind"] = "convolution";
        },
        "kind");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["weight_objects"][0]["descriptor"].erase("n");
        },
        "descriptor");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["weight_objects"][0]["descriptor"]["n"] = 1024;
        },
        "descriptor");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["weight_objects"][0]["roles"]["mystery"] =
                "cos_cache";
        },
        "role");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            const std::string target =
                manifest["weight_objects"][0]["roles"]["qweight"];
            manifest["initializers"]["unreferenced-placeholder"] =
                manifest["initializers"].at(target);
            manifest["initializers"].erase(target);
        },
        "unresolved initializer");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            manifest["weight_objects"][0]["roles"]["scales"] =
                manifest["weight_objects"][0]["roles"]["qweight"];
        },
        "duplicate initializer");
}

void TestRoleInitializerIdentityRejections(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            auto& q_role =
                manifest["weight_objects"][0]["roles"]["qweight"];
            auto& o_role =
                manifest["weight_objects"][3]["roles"]["qweight"];
            const std::string q_initializer =
                q_role.get<std::string>();
            q_role = o_role.get<std::string>();
            o_role = q_initializer;
        },
        "model.layers.0.attn.q_proj.MatMulNBits.qweight "
        "(model.layers.0.attn.o_proj.MatMulNBits.qweight)");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            constexpr std::size_t layer5 = 5u * 5u + 4u;
            constexpr std::size_t layer31 = 31u * 5u + 4u;
            auto& layer5_norm =
                manifest["weight_objects"][layer5]["roles"]["norm1"];
            auto& layer31_norm =
                manifest["weight_objects"][layer31]["roles"]["norm1"];
            const std::string initializer =
                layer5_norm.get<std::string>();
            layer5_norm = layer31_norm.get<std::string>();
            layer31_norm = initializer;
        },
        "model.layers.5.ssmlp.norm1 "
        "(model.layers.32.final_norm_layernorm.weight)");
}

void TestExactSourceValidation(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            auto& record = manifest["initializers"].at(
                "model.embed_tokens.weight");
            record["dtype"] = "uint8";
            record["length"] =
                200064ull * 3072ull;
        },
        "embedding");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            auto& record = manifest["initializers"].at(
                "model.layers.0.input_layernorm.weight");
            record["dtype"] = "uint8";
            record["length"] = 3072;
        },
        "input_norm");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            auto& record =
                manifest["initializers"].at("cos_cache");
            record["shape"] = {4095, 64};
            record["length"] =
                4095ull * 64ull * sizeof(std::uint16_t);
        },
        "cos_cache");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            const std::string name =
                manifest["weight_objects"][0]["roles"]["qweight"];
            auto& record = manifest["initializers"].at(name);
            record["shape"] = {3071, 1536};
            record["length"] = 3071ull * 1536ull;
        },
        "qweight");
    ExpectLoadFailure(
        fixture,
        api,
        [](json& manifest) {
            const std::string name =
                manifest["weight_objects"][0]["roles"]["qweight"];
            manifest["initializers"][name]["role"] = "unknown.role";
        },
        "semantic role");
}

void TestComponentDiagnosticsIdentifyWeightObjects(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    std::string failures;
    const auto verify = [&](
                            std::string_view scenario,
                            auto mutation,
                            std::string_view expected) {
        try {
            ExpectLoadFailure(
                fixture,
                api,
                std::move(mutation),
                expected);
        } catch (const std::exception& error) {
            failures += "\n" + std::string(scenario) + ": " + error.what();
        }
    };

    verify(
        "MatMul dtype",
        [](json& manifest) {
            constexpr std::size_t object_index = 7u * 5u + 1u;
            const std::string initializer =
                manifest["weight_objects"][object_index]["roles"]["scales"];
            auto& record = manifest["initializers"].at(initializer);
            record["dtype"] = "uint8";
            record["length"] = 1024u * 24u;
        },
        "model.layers.7.attn.k_proj.MatMulNBits.scales "
        "(model.layers.7.attn.k_proj.MatMulNBits.scales)");
    verify(
        "SSMLP projection shape",
        [](json& manifest) {
            constexpr std::size_t object_index = 19u * 5u + 4u;
            const std::string initializer =
                manifest["weight_objects"][object_index]["roles"]
                        ["up_scales"];
            auto& record = manifest["initializers"].at(initializer);
            record["shape"] = {8191, 24};
            record["length"] =
                8191u * 24u * sizeof(std::uint16_t);
        },
        "model.layers.19.ssmlp.up_scales "
        "(model.layers.19.mlp.up_proj.MatMulNBits.scales)");
    verify(
        "missing MatMul role",
        [](json& manifest) {
            constexpr std::size_t object_index = 12u * 5u + 2u;
            manifest["weight_objects"][object_index]["roles"].erase(
                "qzeros");
        },
        "model.layers.12.attn.v_proj.MatMulNBits.qzeros");
    verify(
        "SSMLP norm shape",
        [](json& manifest) {
            constexpr std::size_t object_index = 27u * 5u + 4u;
            const std::string initializer =
                manifest["weight_objects"][object_index]["roles"]["norm0"];
            auto& record = manifest["initializers"].at(initializer);
            record["shape"] = {3071};
            record["length"] =
                3071u * sizeof(std::uint16_t);
        },
        "model.layers.27.ssmlp.norm0 "
        "(model.layers.27.post_attention_layernorm.weight)");

    if (!failures.empty()) {
        throw std::runtime_error(
            "component diagnostics did not identify their objects:" +
            failures);
    }
}

void TestOwnedScaleAndNormConversions(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    fixture.Write(fixture.manifest());
    auto package = Phi4Package::Load(fixture.path(), api, false);

    // FP16 scales are an element-wise copy into a contiguous model-owned
    // buffer, not a conversion. WEIGHT-2 still holds.
    const auto fp16 = package.MaterializeFp16(kFp16Scale);
    CHECK(fp16.size() == 1024u * 24u);
    CHECK(fp16[0] == 0x3c00u);
    CHECK(fp16[1] == 0xc000u);
    CHECK(
        reinterpret_cast<const void*>(fp16.data()) !=
        reinterpret_cast<const void*>(
            package.Require(kFp16Scale).data));
    const auto* fp16_address = fp16.data();

    const auto same_fp16 = package.MaterializeFp16(kFp16Scale);
    CHECK(same_fp16.data() == fp16_address);

    // Design Section 9.3: an FP32 scales array is REJECTED, not narrowed.
    // Narrowing it would need a host FP32-to-FP16 converter, which API-6
    // does not permit, so admitting one is a spec change.
    CheckThrowsContains(
        [&] { (void)package.MaterializeFp16(kFp32Scale); },
        "FP16");
    CheckThrowsContains(
        [&] { (void)package.MaterializeFp16(kFp32Scale); },
        "rejected");
    CHECK(fp16.data() == fp16_address);

    // Norms are raw BF16 packer blobs, so they keep the API-6 host
    // round-to-nearest-even helper.
    const auto bf16 = package.MaterializeBf16(kFp32Norm);
    CHECK(bf16.size() == 3072u);
    CHECK(bf16[0] == 0x3f80u);
    CHECK(bf16[1] == 0xc000u);

    CheckThrowsContains(
        [&] {
            (void)package.MaterializeFp16(
                "model.layers.0.attn.q_proj.MatMulNBits.qweight");
        },
        "FP16");
    CheckThrowsContains(
        [&] {
            (void)package.MaterializeBf16(
                "model.layers.0.attn.q_proj.MatMulNBits.qweight");
        },
        "floating");
}

class NoAccessGuard final {
public:
    explicit NoAccessGuard(void* address) {
        MEMORY_BASIC_INFORMATION region{};
        if (
            VirtualQuery(address, &region, sizeof(region)) !=
                sizeof(region) ||
            region.State != MEM_COMMIT) {
            throw std::runtime_error("failed to query guard-page address");
        }
        address_ = address;
        if (
            VirtualProtect(
                address,
                4096,
                PAGE_NOACCESS,
                &old_protection_) == FALSE) {
            throw std::runtime_error(
                "failed to protect no-access guard page");
        }
    }

    ~NoAccessGuard() noexcept {
        if (address_ != nullptr) {
            DWORD ignored = 0;
            VirtualProtect(
                address_,
                4096,
                old_protection_,
                &ignored);
        }
    }

    NoAccessGuard(const NoAccessGuard&) = delete;
    NoAccessGuard& operator=(const NoAccessGuard&) = delete;

private:
    void* address_ = nullptr;
    DWORD old_protection_ = 0;
};

// Corelib e5258d2 removed convert_strided, so this slice is now
// FastFlow's own code -- which makes the guard page more important, not
// less. The last source row sits immediately before an inaccessible page,
// and the gather must take its 48 columns without touching the tail.
void TestRopeGatherStaysInSourceDtypeAtGuardPage(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    fixture.Write(fixture.manifest());
    auto package = Phi4Package::Load(fixture.path(), api, false);
    const auto& source = package.Require("cos_cache");
    CHECK(source.size == kRopeBytes);
    CHECK(source.owner->size() == kRopeMappedBytes);
    CHECK(source.data == source.owner->data());

    auto* one_past =
        const_cast<std::byte*>(source.data + source.size);
    NoAccessGuard guard(one_past);

    const auto rope = package.MaterializeRopeGather("cos_cache");
    // The gather preserves the SOURCE dtype: tensor_write does the
    // widening to the FP32 device tensor, and this path performs no
    // conversion of its own.
    CHECK(rope.dtype == ryzenai_corelib_data_type_fp16);
    CHECK(rope.count == 4096u * 48u);
    const auto* elements =
        static_cast<const std::uint16_t*>(rope.data);
    CHECK(elements[0] == 0x3c00u);
    CHECK(elements[48] == 0x4000u);
    CHECK(elements[4096u * 48u - 1u] == 0x4200u);

    const auto again = package.MaterializeRopeGather("cos_cache");
    CHECK(again.data == rope.data);
    CHECK(again.dtype == rope.dtype);
    CHECK(again.count == rope.count);
}

void TestFp32RopeSource(
    const SyntheticPackage& fixture,
    const std::shared_ptr<CorelibApi>& api) {
    fixture.Write(fixture.manifest());
    auto package = Phi4Package::Load(fixture.path(), api, false);

    const auto rope = package.MaterializeRopeGather("sin_cache");
    CHECK(rope.dtype == ryzenai_corelib_data_type_fp32);
    CHECK(rope.count == 196608u);
    CHECK(static_cast<const float*>(rope.data)[0] == 4.0f);
}

static_assert(!std::is_copy_constructible_v<MappedFile>);
static_assert(!std::is_copy_assignable_v<MappedFile>);
static_assert(std::is_nothrow_move_constructible_v<MappedFile>);
static_assert(std::is_nothrow_move_assignable_v<MappedFile>);

}  // namespace

int main() {
    try {
        SyntheticPackage fixture;
        auto api = ResolveRecordingCorelib();
        TestValidMappingAndExplicitRoles(fixture, api);
        TestOgaQuantizedLayoutsAccepted(fixture, api);
        TestMappedOwnerOutlivesPackage(fixture, api);
        TestPathRangeAndHashRejections(fixture, api);
        TestWeightObjectRejections(fixture, api);
        TestRoleInitializerIdentityRejections(fixture, api);
        TestExactSourceValidation(fixture, api);
        TestComponentDiagnosticsIdentifyWeightObjects(fixture, api);
        TestOwnedScaleAndNormConversions(fixture, api);
        TestRopeGatherStaysInSourceDtypeAtGuardPage(fixture, api);
        TestFp32RopeSource(fixture, api);
        std::cout << "test_phi4_manifest: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
