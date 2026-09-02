#pragma once

// Shared on-disk synthetic Phi-4 package fixture. Both the manifest tests
// and the hardware tests build the same package, so the builders live here
// instead of being duplicated per test translation unit.

#include "test_support.hpp"

#include <nlohmann/json.hpp>

#include <windows.h>
#include <winioctl.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace flm::test::phi4fixture {

using nlohmann::json;

inline constexpr std::string_view kManifestName =
    "corelib_phi4_manifest.json";
inline constexpr std::string_view kDataFile = "z-data.bin";
inline constexpr std::string_view kRopeFile = "z-rope.bin";
inline constexpr std::uint64_t kDataBytes =
    200064ull * 3072ull * sizeof(std::uint16_t);
inline constexpr std::size_t kRopeRows = 4096;
inline constexpr std::size_t kRopeColumns = 64;
inline constexpr std::uint64_t kRopeBytes =
    kRopeRows * kRopeColumns * sizeof(std::uint16_t);
inline constexpr std::uint64_t kRopeMappedBytes = kRopeBytes + 4096;
inline constexpr std::uint64_t kFp32ScaleOffset = 4096;
inline constexpr std::uint64_t kNormOffset = 1024 * 1024;
inline constexpr std::uint64_t kSinOffset = 2 * 1024 * 1024;

inline constexpr std::string_view kFp32Scale =
    "model.layers.0.attn.q_proj.MatMulNBits.scales";
inline constexpr std::string_view kFp16Scale =
    "model.layers.0.attn.k_proj.MatMulNBits.scales";
inline constexpr std::string_view kFp32Norm =
    "model.layers.0.post_attention_layernorm.weight";

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

inline void CreateSparseFile(
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

inline std::uint64_t ItemSize(std::string_view dtype) {
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

inline std::uint64_t ByteLength(
    std::string_view dtype,
    const std::vector<std::int64_t>& shape) {
    std::uint64_t elements = 1;
    for (const std::int64_t dimension : shape) {
        elements *= static_cast<std::uint64_t>(dimension);
    }
    return elements * ItemSize(dtype);
}

inline void AddInitializer(
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

inline void AddMatMul(
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

inline void AddSsMlp(json& manifest, int layer) {
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

inline json BuildManifest(std::uint64_t model_size) {
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

}  // namespace flm::test::phi4fixture
