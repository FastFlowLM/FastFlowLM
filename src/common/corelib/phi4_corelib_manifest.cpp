#include <models/phi4/phi4_corelib_manifest.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>

#include "../../pull/picosha2.h"

#include <nlohmann/json.hpp>
#include <windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flm::phi4 {
namespace {

using nlohmann::json;

constexpr std::string_view kManifestName =
    "corelib_phi4_manifest.json";
constexpr std::size_t kExpectedInitializers = 743;
constexpr std::size_t kExpectedWeightObjects = 161;
constexpr std::int64_t kRopeColumns =
    constants::kRopeDimension / 2;

[[noreturn]] void Throw(
    std::string_view context,
    std::string_view detail) {
    throw std::runtime_error(
        std::string(context) + ": " + std::string(detail));
}

[[noreturn]] void ThrowWin32(
    std::string_view operation,
    const std::filesystem::path& path,
    DWORD error) {
    std::ostringstream message;
    message << operation << " failed for " << path.string()
            << " (Win32 error " << error << ")";
    throw std::runtime_error(message.str());
}

void RequireExactKeys(
    const json& value,
    std::initializer_list<std::string_view> expected,
    std::string_view context) {
    if (!value.is_object()) {
        Throw(context, "must be an object");
    }
    if (value.size() != expected.size()) {
        Throw(context, "has an invalid field set");
    }
    for (const std::string_view key : expected) {
        if (!value.contains(std::string(key))) {
            Throw(
                context,
                std::string("is missing field ") + std::string(key));
        }
    }
}

std::string ReadString(
    const json& value,
    std::string_view context) {
    if (!value.is_string()) {
        Throw(context, "must be a string");
    }
    const std::string result = value.get<std::string>();
    if (result.empty() ||
        result.find('\0') != std::string::npos) {
        Throw(context, "must be a non-empty string");
    }
    return result;
}

std::uint64_t ReadU64(
    const json& value,
    std::string_view context) {
    if (!value.is_number_unsigned()) {
        Throw(context, "must be an unsigned integer");
    }
    return value.get<std::uint64_t>();
}

std::int64_t ReadPositiveI64(
    const json& value,
    std::string_view context) {
    std::uint64_t unsigned_value = 0;
    if (value.is_number_unsigned()) {
        unsigned_value = value.get<std::uint64_t>();
    } else if (value.is_number_integer()) {
        const std::int64_t signed_value = value.get<std::int64_t>();
        if (signed_value <= 0) {
            Throw(context, "must be a positive integer");
        }
        return signed_value;
    } else {
        Throw(context, "must be a positive integer");
    }
    if (
        unsigned_value == 0 ||
        unsigned_value >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())) {
        Throw(context, "must fit a positive int64");
    }
    return static_cast<std::int64_t>(unsigned_value);
}

void RequireInteger(
    const json& value,
    std::int64_t expected,
    std::string_view context) {
    if (ReadPositiveI64(value, context) != expected) {
        Throw(context, "does not match the Phi-4 model identity");
    }
}

bool ReadBool(
    const json& value,
    std::string_view context) {
    if (!value.is_boolean()) {
        Throw(context, "must be a boolean");
    }
    return value.get<bool>();
}

std::filesystem::path ParseRelativePath(
    std::string_view raw,
    std::string_view context) {
    if (raw.empty() ||
        raw.find('\0') != std::string_view::npos) {
        Throw(context, "file path must not be empty");
    }

    const auto path = std::filesystem::u8path(
        raw.begin(),
        raw.end());
    if (
        path.is_absolute() || path.has_root_name() ||
        path.has_root_directory()) {
        Throw(context, "file path must be relative");
    }

    bool has_component = false;
    for (const auto& component : path) {
        if (component == "..") {
            Throw(context, "file path contains parent traversal");
        }
        if (component == "." || component.empty()) {
            continue;
        }
        has_component = true;
    }
    if (!has_component) {
        Throw(context, "file path must name a file");
    }
    return path.lexically_normal();
}

bool IsWithin(
    const std::filesystem::path& root,
    const std::filesystem::path& candidate) {
    const auto relative = candidate.lexically_relative(root);
    if (relative.empty() || relative.is_absolute()) {
        return false;
    }
    const auto first = relative.begin();
    return first != relative.end() && *first != "..";
}

std::filesystem::path ResolvePackageFile(
    const std::filesystem::path& root,
    std::string_view raw,
    std::string_view context) {
    const auto relative = ParseRelativePath(raw, context);
    std::error_code error;
    const auto resolved =
        std::filesystem::canonical(root / relative, error);
    if (error) {
        Throw(
            context,
            std::string("package file does not exist: ") +
                std::string(raw));
    }
    if (!IsWithin(root, resolved)) {
        Throw(context, "package file escapes the model directory");
    }
    if (!std::filesystem::is_regular_file(resolved, error) || error) {
        Throw(context, "package path is not a regular file");
    }
    return resolved;
}

std::string CalculateSha256(
    const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        Throw("SHA-256", "failed to open package file");
    }
    std::array<unsigned char, picosha2::k_digest_size> digest{};
    picosha2::hash256(stream, digest.begin(), digest.end());
    if (stream.bad()) {
        Throw("SHA-256", "failed while reading package file");
    }
    return picosha2::bytes_to_hex_string(
        digest.begin(),
        digest.end());
}

bool IsSha256(std::string_view value) {
    return value.size() == picosha2::k_digest_size * 2 &&
           std::all_of(
               value.begin(),
               value.end(),
               [](unsigned char character) {
                   return std::isxdigit(character) != 0;
               });
}

std::string NormalizeSha256(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

SourceDType ParseDType(
    std::string_view value,
    std::string_view context) {
    if (value == "uint8") {
        return SourceDType::UInt8;
    }
    if (value == "float16") {
        return SourceDType::Float16;
    }
    if (value == "float32") {
        return SourceDType::Float32;
    }
    if (value == "int64") {
        return SourceDType::Int64;
    }
    Throw(context, "has an unsupported dtype");
}

std::uint64_t ItemSize(SourceDType dtype) {
    switch (dtype) {
        case SourceDType::UInt8:
            return 1;
        case SourceDType::Float16:
            return 2;
        case SourceDType::Float32:
            return 4;
        case SourceDType::Int64:
            return 8;
    }
    throw std::logic_error("unreachable SourceDType");
}

std::vector<std::int64_t> ParseShape(
    const json& value,
    std::string_view context) {
    if (!value.is_array() || value.empty()) {
        Throw(context, "shape must be a non-empty array");
    }
    std::vector<std::int64_t> shape;
    shape.reserve(value.size());
    for (const auto& dimension : value) {
        shape.push_back(ReadPositiveI64(dimension, context));
    }
    return shape;
}

std::uint64_t CheckedByteCount(
    const std::vector<std::int64_t>& shape,
    SourceDType dtype,
    std::string_view context) {
    std::uint64_t elements = 1;
    for (const std::int64_t dimension : shape) {
        const auto unsigned_dimension =
            static_cast<std::uint64_t>(dimension);
        if (
            elements >
            std::numeric_limits<std::uint64_t>::max() /
                unsigned_dimension) {
            Throw(context, "shape element count overflow");
        }
        elements *= unsigned_dimension;
    }
    const std::uint64_t item_size = ItemSize(dtype);
    if (
        elements >
        std::numeric_limits<std::uint64_t>::max() / item_size) {
        Throw(context, "tensor byte count overflow");
    }
    return elements * item_size;
}

void RequireShape(
    const InitializerView& view,
    std::initializer_list<std::int64_t> expected,
    std::string_view context) {
    if (!std::equal(
            view.shape.begin(),
            view.shape.end(),
            expected.begin(),
            expected.end())) {
        Throw(context, "has an invalid shape");
    }
}

void RequireFloating(
    const InitializerView& view,
    std::string_view context) {
    if (
        view.dtype != SourceDType::Float16 &&
        view.dtype != SourceDType::Float32) {
        Throw(context, "must use a floating FP16 or FP32 source");
    }
}

void RequireSemanticRole(
    const std::map<std::string, std::string, std::less<>>& roles,
    std::string_view initializer,
    std::string_view expected) {
    const auto found = roles.find(initializer);
    if (found == roles.end() || found->second != expected) {
        Throw(
            initializer,
            std::string("semantic role does not match ") +
                std::string(expected));
    }
}

struct ExpectedWeightObject {
    std::string name;
    WeightObjectKind kind;
    std::int64_t k;
    std::int64_t n;
};

const std::vector<ExpectedWeightObject>& ExpectedWeightObjects() {
    static const std::vector<ExpectedWeightObject> expected = [] {
        std::vector<ExpectedWeightObject> values;
        values.reserve(kExpectedWeightObjects);
        for (std::int64_t layer = 0;
             layer < constants::kLayerCount;
             ++layer) {
            const std::string base =
                "model.layers." + std::to_string(layer) + ".attn.";
            values.push_back({
                base + "q_proj.MatMulNBits",
                WeightObjectKind::MatMul,
                constants::kHiddenSize,
                constants::kQueryDimension});
            values.push_back({
                base + "k_proj.MatMulNBits",
                WeightObjectKind::MatMul,
                constants::kHiddenSize,
                constants::kKvDimension});
            values.push_back({
                base + "v_proj.MatMulNBits",
                WeightObjectKind::MatMul,
                constants::kHiddenSize,
                constants::kKvDimension});
            values.push_back({
                base + "o_proj.MatMulNBits",
                WeightObjectKind::MatMul,
                constants::kQueryDimension,
                constants::kHiddenSize});
            values.push_back({
                "model.layers." + std::to_string(layer) + ".ssmlp",
                WeightObjectKind::SsMlp,
                constants::kHiddenSize,
                constants::kIntermediateSize});
        }
        values.push_back({
            "lm_head.MatMulNBits",
            WeightObjectKind::MatMul,
            constants::kHiddenSize,
            constants::kVocabularySize});
        return values;
    }();
    return expected;
}

std::set<std::string> MatMulRoleNames() {
    return {"qweight", "scales", "qzeros"};
}

std::set<std::string> SsMlpRoleNames() {
    return {
        "norm0",
        "norm1",
        "gate_qweight",
        "gate_scales",
        "gate_qzeros",
        "up_qweight",
        "up_scales",
        "up_qzeros",
        "down_qweight",
        "down_scales",
        "down_qzeros"};
}

std::set<std::string> JsonKeys(const json& value) {
    std::set<std::string> keys;
    for (const auto& [key, _] : value.items()) {
        keys.insert(key);
    }
    return keys;
}

std::string ComponentContext(
    std::string_view object_name,
    std::string_view role_name) {
    return std::string(object_name) + "." + std::string(role_name);
}

std::string ComponentContext(
    std::string_view object_name,
    std::string_view role_name,
    std::string_view initializer_name) {
    return ComponentContext(object_name, role_name) + " (" +
           std::string(initializer_name) + ")";
}

void ValidateQuantizedProjection(
    const Phi4Package& package,
    std::string_view object_name,
    const std::map<std::string, std::string, std::less<>>& semantic_roles,
    const std::map<std::string, std::string>& components,
    std::string_view role_prefix,
    std::string_view component_prefix,
    std::int64_t k,
    std::int64_t n) {
    if (k <= 0 || n <= 0 || k % 2 != 0 ||
        k % static_cast<std::int64_t>(
                constants::kGroupSize) != 0) {
        Throw(role_prefix, "has an invalid quantized descriptor");
    }
    const std::int64_t groups =
        k / static_cast<std::int64_t>(
                constants::kGroupSize);

    const auto validate = [&](
                              std::string_view component,
                              SourceDType dtype,
                              std::initializer_list<std::int64_t> shape,
                              std::string semantic) {
        const std::string role_name =
            std::string(component_prefix) + std::string(component);
        const auto found = components.find(role_name);
        if (found == components.end()) {
            Throw(
                ComponentContext(object_name, role_name),
                "is missing a component role");
        }
        const auto& view = package.Require(found->second);
        const std::string context =
            ComponentContext(object_name, role_name, found->second);
        if (view.dtype != dtype) {
            Throw(context, "has an invalid dtype");
        }
        RequireShape(view, shape, context);
        RequireSemanticRole(
            semantic_roles,
            found->second,
            semantic);
    };

    validate(
        "qweight",
        SourceDType::UInt8,
        {n, k / 2},
        std::string(role_prefix) + ".qweight");

    const std::string scales_name =
        std::string(component_prefix) + "scales";
    const auto scales_component = components.find(scales_name);
    if (scales_component == components.end()) {
        Throw(
            ComponentContext(object_name, scales_name),
            "is missing a scales role");
    }
    const auto& scales = package.Require(scales_component->second);
    const std::string scales_context = ComponentContext(
        object_name,
        scales_name,
        scales_component->second);
    RequireFloating(scales, scales_context);
    RequireShape(scales, {n, groups}, scales_context);
    RequireSemanticRole(
        semantic_roles,
        scales_component->second,
        std::string(role_prefix) + ".scales");

    validate(
        "qzeros",
        SourceDType::UInt8,
        {n, (groups + 1) / 2},
        std::string(role_prefix) + ".qzeros");
}

void ValidateHostInitializers(
    const Phi4Package& package,
    const std::map<std::string, std::string, std::less<>>& semantic_roles) {
    const auto& embedding =
        package.Require("model.embed_tokens.weight");
    if (embedding.dtype != SourceDType::Float16) {
        Throw("embedding", "must use an FP16 source");
    }
    RequireShape(
        embedding,
        {constants::kVocabularySize, constants::kHiddenSize},
        "embedding");
    RequireSemanticRole(
        semantic_roles,
        "model.embed_tokens.weight",
        "embedding");

    const auto& input_norm =
        package.Require("model.layers.0.input_layernorm.weight");
    RequireFloating(input_norm, "input_norm");
    RequireShape(
        input_norm,
        {constants::kHiddenSize},
        "input_norm");
    RequireSemanticRole(
        semantic_roles,
        "model.layers.0.input_layernorm.weight",
        "input_norm");

    for (const std::string_view name : {"cos_cache", "sin_cache"}) {
        const auto& rope = package.Require(name);
        RequireFloating(rope, name);
        if (
            rope.shape.size() != 2 ||
            rope.shape[0] < constants::kMaxSequenceLength ||
            rope.shape[1] < kRopeColumns) {
            Throw(name, "must be rank 2 and at least [4096,48]");
        }
        RequireSemanticRole(semantic_roles, name, name);
    }
}

ryzenai_corelib_data_type CorelibDType(
    SourceDType dtype,
    std::string_view context) {
    switch (dtype) {
        case SourceDType::Float16:
            return ryzenai_corelib_data_type_fp16;
        case SourceDType::Float32:
            return ryzenai_corelib_data_type_fp32;
        case SourceDType::UInt8:
        case SourceDType::Int64:
            Throw(context, "requires a floating FP16 or FP32 source");
    }
    throw std::logic_error("unreachable SourceDType");
}

std::size_t ElementCount(
    const InitializerView& view,
    std::string_view context) {
    const std::uint64_t count =
        static_cast<std::uint64_t>(view.size) / ItemSize(view.dtype);
    if (
        count >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        Throw(context, "element count exceeds addressable memory");
    }
    return static_cast<std::size_t>(count);
}

void ValidateModelIdentity(const json& manifest) {
    if (
        manifest.is_object() &&
        !manifest.contains("weight_objects")) {
        Throw("weight_objects", "section is missing");
    }
    RequireExactKeys(
        manifest,
        {
            "schema_version",
            "execution_backend",
            "model",
            "backend",
            "files",
            "initializers",
            "weight_objects",
        },
        "manifest");
    RequireInteger(
        manifest.at("schema_version"),
        1,
        "schema_version");
    if (
        ReadString(
            manifest.at("execution_backend"),
            "execution_backend") != "corelib_aie4") {
        Throw(
            "execution_backend",
            "does not match corelib_aie4");
    }

    const auto& model = manifest.at("model");
    RequireExactKeys(
        model,
        {
            "family",
            "layers",
            "hidden_size",
            "intermediate_size",
            "num_heads",
            "kv_heads",
            "head_size",
            "vocab_size",
            "group_size",
            "rope_dim",
            "rms_epsilon",
        },
        "model");
    if (ReadString(model.at("family"), "model.family") != "phi4") {
        Throw("model.family", "does not match phi4");
    }
    RequireInteger(
        model.at("layers"),
        constants::kLayerCount,
        "model.layers");
    RequireInteger(
        model.at("hidden_size"),
        constants::kHiddenSize,
        "model.hidden_size");
    RequireInteger(
        model.at("intermediate_size"),
        constants::kIntermediateSize,
        "model.intermediate_size");
    RequireInteger(
        model.at("num_heads"),
        constants::kQueryHeadCount,
        "model.num_heads");
    RequireInteger(
        model.at("kv_heads"),
        constants::kKvHeadCount,
        "model.kv_heads");
    RequireInteger(
        model.at("head_size"),
        constants::kHeadSize,
        "model.head_size");
    RequireInteger(
        model.at("vocab_size"),
        constants::kVocabularySize,
        "model.vocab_size");
    RequireInteger(
        model.at("group_size"),
        constants::kGroupSize,
        "model.group_size");
    RequireInteger(
        model.at("rope_dim"),
        constants::kRopeDimension,
        "model.rope_dim");
    if (
        !model.at("rms_epsilon").is_number() ||
        model.at("rms_epsilon").get<double>() !=
            constants::kRmsEpsilon) {
        Throw(
            "model.rms_epsilon",
            "does not match the Phi-4 model identity");
    }

    const auto& backend = manifest.at("backend");
    RequireExactKeys(backend, {"max_seq"}, "backend");
    RequireInteger(
        backend.at("max_seq"),
        constants::kMaxSequenceLength,
        "backend.max_seq");
}

}  // namespace

MappedFile::MappedFile(
    std::filesystem::path path,
    void* file,
    void* mapping,
    const std::byte* data,
    std::uint64_t size) noexcept
    : path_(std::move(path)),
      file_(file),
      mapping_(mapping),
      data_(data),
      size_(size) {}

std::shared_ptr<const MappedFile> MappedFile::OpenReadOnly(
    const std::filesystem::path& path) {
    // Copy before acquiring Win32 resources so an allocation failure cannot
    // strand handles that have not yet been transferred to MappedFile.
    std::filesystem::path owned_path = path;
    HANDLE file = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        ThrowWin32("CreateFileW", path, GetLastError());
    }

    LARGE_INTEGER raw_size{};
    if (GetFileSizeEx(file, &raw_size) == FALSE) {
        const DWORD error = GetLastError();
        CloseHandle(file);
        ThrowWin32("GetFileSizeEx", path, error);
    }
    if (raw_size.QuadPart <= 0) {
        CloseHandle(file);
        Throw(path.string(), "mapped file must not be empty");
    }
    const auto size = static_cast<std::uint64_t>(raw_size.QuadPart);

    HANDLE mapping = CreateFileMappingW(
        file,
        nullptr,
        PAGE_READONLY,
        0,
        0,
        nullptr);
    if (mapping == nullptr) {
        const DWORD error = GetLastError();
        CloseHandle(file);
        ThrowWin32("CreateFileMappingW", path, error);
    }

    const void* raw_data =
        MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (raw_data == nullptr) {
        const DWORD error = GetLastError();
        CloseHandle(mapping);
        CloseHandle(file);
        ThrowWin32("MapViewOfFile", path, error);
    }

    std::unique_ptr<MappedFile> owner;
    try {
        owner.reset(new MappedFile(
            std::move(owned_path),
            file,
            mapping,
            static_cast<const std::byte*>(raw_data),
            size));
    } catch (...) {
        UnmapViewOfFile(raw_data);
        CloseHandle(mapping);
        CloseHandle(file);
        throw;
    }
    return std::shared_ptr<const MappedFile>(std::move(owner));
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : path_(std::move(other.path_)),
      file_(std::exchange(other.file_, nullptr)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        Reset();
        path_ = std::move(other.path_);
        file_ = std::exchange(other.file_, nullptr);
        mapping_ = std::exchange(other.mapping_, nullptr);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

MappedFile::~MappedFile() noexcept {
    Reset();
}

void MappedFile::Reset() noexcept {
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(mapping_));
        mapping_ = nullptr;
    }
    if (file_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(file_));
        file_ = nullptr;
    }
    size_ = 0;
}

const std::byte* MappedFile::data() const noexcept {
    return data_;
}

std::uint64_t MappedFile::size() const noexcept {
    return size_;
}

const std::filesystem::path& MappedFile::path() const noexcept {
    return path_;
}

Phi4Package Phi4Package::Load(
    const std::filesystem::path& model_dir,
    std::shared_ptr<const corelib::CorelibApi> api,
    bool verify_full_hash) {
    if (!api) {
        throw std::invalid_argument(
            "Phi4Package::Load requires a CorelibApi");
    }

    std::error_code error;
    const auto root = std::filesystem::canonical(model_dir, error);
    if (
        error || !std::filesystem::is_directory(root, error) ||
        error) {
        throw std::runtime_error(
            "Phi4 model directory does not exist or is not a directory");
    }

    std::ifstream manifest_stream(
        root / kManifestName,
        std::ios::binary);
    if (!manifest_stream) {
        throw std::runtime_error(
            "Phi-4 package is missing corelib_phi4_manifest.json");
    }

    json manifest;
    try {
        manifest_stream >> manifest;
    } catch (const json::exception& parse_error) {
        throw std::runtime_error(
            std::string("failed to parse corelib_phi4_manifest.json: ") +
            parse_error.what());
    }
    ValidateModelIdentity(manifest);

    Phi4Package package;
    package.api_ = std::move(api);

    const auto& files = manifest.at("files");
    if (!files.is_object() || files.empty()) {
        Throw("files", "must be a non-empty object");
    }
    if (!files.contains("model.onnx")) {
        Throw("files", "is missing model.onnx");
    }
    for (const auto& [name, record] : files.items()) {
        const std::string context = "files." + name;
        if (!record.is_object()) {
            Throw(context, "must be an object");
        }
        const bool has_hash = record.contains("sha256");
        RequireExactKeys(
            record,
            has_hash
                ? std::initializer_list<std::string_view>{
                      "size",
                      "sha256"}
                : std::initializer_list<std::string_view>{"size"},
            context);

        const std::uint64_t expected_size =
            ReadU64(record.at("size"), context + ".size");
        if (expected_size == 0) {
            Throw(context, "file size must be positive");
        }
        const auto resolved =
            ResolvePackageFile(root, name, context);
        auto mapped = MappedFile::OpenReadOnly(resolved);
        if (mapped->size() != expected_size) {
            Throw(context, "file size does not match the manifest");
        }

        if (has_hash) {
            std::string expected_hash = ReadString(
                record.at("sha256"),
                context + ".sha256");
            if (!IsSha256(expected_hash)) {
                Throw(context, "SHA-256 must contain 64 hexadecimal digits");
            }
            expected_hash = NormalizeSha256(std::move(expected_hash));
            if (
                verify_full_hash &&
                CalculateSha256(resolved) != expected_hash) {
                Throw(context, "SHA-256 does not match the package file");
            }
        } else if (verify_full_hash) {
            Throw(context, "SHA-256 is required for full verification");
        }

        package.mapped_files_.emplace(name, std::move(mapped));
    }

    const auto& initializer_records = manifest.at("initializers");
    if (
        !initializer_records.is_object() ||
        initializer_records.size() != kExpectedInitializers) {
        Throw(
            "initializers",
            "must contain exactly 743 initializer records");
    }

    std::map<std::string, std::string, std::less<>>
        semantic_roles;
    for (const auto& [name, record] : initializer_records.items()) {
        const std::string context = "initializers." + name;
        RequireExactKeys(
            record,
            {"file", "offset", "length", "dtype", "shape", "role"},
            context);
        const std::string file_name =
            ReadString(record.at("file"), context + ".file");
        (void)ParseRelativePath(file_name, context + ".file");

        const std::uint64_t offset =
            ReadU64(record.at("offset"), context + ".offset");
        const std::uint64_t length =
            ReadU64(record.at("length"), context + ".length");
        if (length == 0) {
            Throw(context, "initializer length must be positive");
        }
        if (
            offset >
            std::numeric_limits<std::uint64_t>::max() - length) {
            Throw(context, "initializer range overflow");
        }

        const auto owner =
            package.mapped_files_.find(file_name);
        if (owner == package.mapped_files_.end()) {
            Throw(context, "initializer references an unlisted file");
        }
        if (
            offset > owner->second->size() ||
            length > owner->second->size() - offset) {
            Throw(context, "initializer range exceeds file size");
        }

        const std::string dtype_name =
            ReadString(record.at("dtype"), context + ".dtype");
        const SourceDType dtype =
            ParseDType(dtype_name, context + ".dtype");
        if (offset % ItemSize(dtype) != 0) {
            Throw(context, "initializer offset is not dtype-aligned");
        }
        const auto shape =
            ParseShape(record.at("shape"), context + ".shape");
        if (CheckedByteCount(shape, dtype, context) != length) {
            Throw(
                context,
                "dtype/shape byte count does not match length");
        }
        if (
            length >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()) ||
            offset >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            Throw(context, "initializer range is not addressable");
        }

        const std::string semantic_role =
            ReadString(record.at("role"), context + ".role");
        semantic_roles.emplace(name, semantic_role);
        package.initializers_.emplace(
            name,
            InitializerView{
                dtype,
                shape,
                owner->second->data() +
                    static_cast<std::size_t>(offset),
                static_cast<std::size_t>(length),
                owner->second});
    }

    ValidateHostInitializers(package, semantic_roles);

    if (!manifest.contains("weight_objects")) {
        Throw("weight_objects", "section is missing");
    }
    const auto& object_records = manifest.at("weight_objects");
    if (
        !object_records.is_array() ||
        object_records.size() != kExpectedWeightObjects) {
        Throw(
            "weight_objects",
            "must be a non-empty list of exactly 161 entries");
    }

    const auto& expected_objects = ExpectedWeightObjects();
    std::set<std::string> object_names;
    std::set<std::string> referenced_initializers;
    package.weight_objects_.reserve(kExpectedWeightObjects);
    for (std::size_t index = 0; index < object_records.size(); ++index) {
        const auto& record = object_records.at(index);
        const std::string index_context =
            "weight_objects[" + std::to_string(index) + "]";
        RequireExactKeys(
            record,
            {"name", "kind", "descriptor", "roles"},
            index_context);

        const std::string name =
            ReadString(record.at("name"), index_context + ".name");
        if (!object_names.insert(name).second) {
            Throw(name, "duplicate weight object name");
        }

        const std::string kind_name =
            ReadString(record.at("kind"), name + ".kind");
        WeightObjectKind kind;
        if (kind_name == "matmul") {
            kind = WeightObjectKind::MatMul;
        } else if (kind_name == "ssmlp") {
            kind = WeightObjectKind::SsMlp;
        } else {
            Throw(name, "invalid weight object kind");
        }

        const auto& expected = expected_objects.at(index);
        if (name != expected.name) {
            Throw(
                name,
                "weight object name/order does not match the fixed "
                "Phi-4 plan");
        }
        if (kind != expected.kind) {
            Throw(
                name,
                "weight object kind does not match the fixed Phi-4 plan");
        }

        const auto& descriptor = record.at("descriptor");
        RequireExactKeys(
            descriptor,
            kind == WeightObjectKind::MatMul
                ? std::initializer_list<std::string_view>{
                      "k",
                      "n",
                      "group_size",
                      "has_bias"}
                : std::initializer_list<std::string_view>{
                      "k",
                      "n",
                      "group_size"},
            name + ".descriptor");
        const std::int64_t k =
            ReadPositiveI64(descriptor.at("k"), name + ".descriptor.k");
        const std::int64_t n =
            ReadPositiveI64(descriptor.at("n"), name + ".descriptor.n");
        const std::int64_t raw_group_size = ReadPositiveI64(
            descriptor.at("group_size"),
            name + ".descriptor.group_size");
        if (
            raw_group_size >
            static_cast<std::int64_t>(
                std::numeric_limits<std::uint32_t>::max())) {
            Throw(name + ".descriptor", "group_size exceeds uint32");
        }
        const auto group_size =
            static_cast<std::uint32_t>(raw_group_size);
        bool has_bias = false;
        if (kind == WeightObjectKind::MatMul) {
            has_bias = ReadBool(
                descriptor.at("has_bias"),
                name + ".descriptor.has_bias");
            if (has_bias) {
                Throw(
                    name + ".descriptor",
                    "MatMul has_bias must be false");
            }
        }
        if (
            k != expected.k || n != expected.n ||
            group_size != constants::kGroupSize) {
            Throw(
                name + ".descriptor",
                "does not match the fixed Phi-4 descriptor");
        }

        const auto& role_map = record.at("roles");
        if (!role_map.is_object()) {
            Throw(name, "weight object role map must be an object");
        }
        const auto expected_roles =
            kind == WeightObjectKind::MatMul
                ? MatMulRoleNames()
                : SsMlpRoleNames();
        for (const auto& role : expected_roles) {
            if (!role_map.contains(role)) {
                Throw(
                    ComponentContext(name, role),
                    "is missing a component role");
            }
        }
        if (JsonKeys(role_map) != expected_roles) {
            Throw(name, "invalid weight object role map");
        }

        std::map<std::string, std::string> components;
        std::set<std::string> local_references;
        for (const auto& [role, initializer] : role_map.items()) {
            const std::string initializer_name = ReadString(
                initializer,
                name + ".roles." + role);
            if (!local_references.insert(initializer_name).second) {
                Throw(name, "duplicate initializer role reference");
            }
            if (!package.initializers_.contains(initializer_name)) {
                Throw(
                    name,
                    "unresolved initializer " + initializer_name);
            }
            if (!referenced_initializers.insert(initializer_name).second) {
                Throw(
                    name,
                    "duplicate initializer reference across weight objects");
            }
            components.emplace(role, initializer_name);
        }

        if (kind == WeightObjectKind::MatMul) {
            ValidateQuantizedProjection(
                package,
                name,
                semantic_roles,
                components,
                "matmul",
                "",
                k,
                n);
        } else {
            const auto validate_norm = [&](
                                           std::string_view role,
                                           std::string_view semantic) {
                const auto component =
                    components.find(std::string(role));
                const auto& view =
                    package.Require(component->second);
                const std::string context = ComponentContext(
                    name,
                    role,
                    component->second);
                RequireFloating(view, context);
                RequireShape(
                    view,
                    {constants::kHiddenSize},
                    context);
                RequireSemanticRole(
                    semantic_roles,
                    component->second,
                    semantic);
            };
            validate_norm("norm0", "ssmlp.norm0");
            validate_norm("norm1", "ssmlp.norm1");
            ValidateQuantizedProjection(
                package,
                name,
                semantic_roles,
                components,
                "ssmlp.gate",
                "gate_",
                k,
                n);
            ValidateQuantizedProjection(
                package,
                name,
                semantic_roles,
                components,
                "ssmlp.up",
                "up_",
                k,
                n);
            ValidateQuantizedProjection(
                package,
                name,
                semantic_roles,
                components,
                "ssmlp.down",
                "down_",
                n,
                k);
        }

        package.weight_objects_.push_back({
            name,
            kind,
            k,
            n,
            group_size,
            has_bias,
            std::move(components)});
    }

    const std::set<std::string> host_initializers{
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "cos_cache",
        "sin_cache"};
    for (const auto& [name, _] : package.initializers_) {
        const bool is_host = host_initializers.contains(name);
        const bool is_referenced =
            referenced_initializers.contains(name);
        if (is_host == is_referenced) {
            Throw(
                name,
                is_host
                    ? "host initializer must not belong to a weight object"
                    : "initializer is not referenced by weight_objects");
        }
    }

    return package;
}

const InitializerView& Phi4Package::Require(
    std::string_view name) const {
    const auto found = initializers_.find(name);
    if (found == initializers_.end()) {
        throw std::runtime_error(
            "missing initializer: " + std::string(name));
    }
    return found->second;
}

const std::vector<WeightObjectView>&
Phi4Package::weight_objects() const noexcept {
    return weight_objects_;
}

std::span<const std::uint16_t> Phi4Package::MaterializeFp16(
    std::string_view name) {
    if (const auto found = fp16_buffers_.find(name);
        found != fp16_buffers_.end()) {
        return found->second;
    }

    const auto& source = Require(name);
    const auto source_type = CorelibDType(source.dtype, name);
    const std::size_t count = ElementCount(source, name);
    auto [buffer, inserted] =
        fp16_buffers_.try_emplace(std::string(name), count);
    try {
        api_->Check(
            api_->functions().convert(
                source_type,
                source.data,
                ryzenai_corelib_data_type_fp16,
                buffer->second.data(),
                count),
            "ryzenai_corelib_convert");
    } catch (...) {
        if (inserted) {
            fp16_buffers_.erase(buffer);
        }
        throw;
    }
    return buffer->second;
}

std::span<const std::uint16_t> Phi4Package::MaterializeBf16(
    std::string_view name) {
    if (const auto found = bf16_buffers_.find(name);
        found != bf16_buffers_.end()) {
        return found->second;
    }

    const auto& source = Require(name);
    const auto source_type = CorelibDType(source.dtype, name);
    const std::size_t count = ElementCount(source, name);
    auto [buffer, inserted] =
        bf16_buffers_.try_emplace(std::string(name), count);
    try {
        api_->Check(
            api_->functions().convert(
                source_type,
                source.data,
                ryzenai_corelib_data_type_bf16,
                buffer->second.data(),
                count),
            "ryzenai_corelib_convert");
    } catch (...) {
        if (inserted) {
            bf16_buffers_.erase(buffer);
        }
        throw;
    }
    return buffer->second;
}

std::span<const float> Phi4Package::MaterializeRopeFp32(
    std::string_view name) {
    if (const auto found = fp32_buffers_.find(name);
        found != fp32_buffers_.end()) {
        return found->second;
    }

    const auto& source = Require(name);
    const auto source_type = CorelibDType(source.dtype, name);
    if (
        source.shape.size() != 2 ||
        source.shape[0] < constants::kMaxSequenceLength ||
        source.shape[1] < kRopeColumns) {
        Throw(
            name,
            "RoPE source must be rank 2 and at least [4096,48]");
    }
    const auto source_columns =
        static_cast<std::size_t>(source.shape[1]);
    constexpr std::size_t count =
        static_cast<std::size_t>(
            constants::kMaxSequenceLength * kRopeColumns);
    auto [buffer, inserted] =
        fp32_buffers_.try_emplace(std::string(name), count);
    try {
        api_->Check(
            api_->functions().convert_strided(
                source_type,
                source.data,
                source_columns,
                ryzenai_corelib_data_type_fp32,
                buffer->second.data(),
                static_cast<std::size_t>(kRopeColumns),
                count,
                static_cast<std::size_t>(kRopeColumns)),
            "ryzenai_corelib_convert_strided");
    } catch (...) {
        if (inserted) {
            fp32_buffers_.erase(buffer);
        }
        throw;
    }
    return buffer->second;
}

}  // namespace flm::phi4
