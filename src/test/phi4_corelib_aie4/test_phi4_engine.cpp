#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <corelib/corelib_fatal_record.hpp>
#include <corelib/corelib_runtime.hpp>
#include <models/phi4/phi4_corelib_aie4.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>
#include <nlohmann/json.hpp>

#include <windows.h>
#include <winioctl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using flm::corelib::CorelibApi;
using flm::corelib::CorelibError;
using flm::corelib::CorelibRuntime;
using flm::corelib::FatalRecordStore;
using flm::corelib::ProcessState;
using flm::phi4::Phi4Aie4Metrics;
using flm::phi4::Phi4DebugSnapshot;
using flm::phi4::phi4_corelib_aie4;
using nlohmann::json;

namespace constants = flm::phi4::constants;

constexpr std::string_view kManifestName =
    "corelib_phi4_manifest.json";
constexpr std::string_view kDataFile = "weights.bin";
constexpr std::uint64_t kDataBytes =
    200064ull * 3072ull * sizeof(std::uint16_t);
constexpr std::uint16_t kPoison = 0xDEADu;

std::int64_t PaddedRows(std::int64_t rows) {
    return rows == 1 ? 1 : ((rows + 3) / 4) * 4;
}

class TempDirectory final {
public:
    explicit TempDirectory(std::string_view stem) {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                (std::string(stem) + "-" +
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
        throw std::runtime_error(
            "failed to create sparse engine-test model data");
    }

    DWORD ignored = 0;
    if (
        DeviceIoControl(
            file,
            FSCTL_SET_SPARSE,
            nullptr,
            0,
            nullptr,
            0,
            &ignored,
            nullptr) == FALSE) {
        CloseHandle(file);
        throw std::runtime_error(
            "engine-test volume does not support sparse files");
    }

    LARGE_INTEGER end{};
    end.QuadPart = static_cast<LONGLONG>(size);
    const bool success =
        SetFilePointerEx(file, end, nullptr, FILE_BEGIN) != FALSE &&
        SetEndOfFile(file) != FALSE;
    CloseHandle(file);
    if (!success) {
        throw std::runtime_error(
            "failed to size sparse engine-test model data");
    }
}

std::uint64_t ItemSize(std::string_view dtype) {
    if (dtype == "uint8") {
        return 1;
    }
    if (dtype == "float16") {
        return 2;
    }
    if (dtype == "float32" || dtype == "int64") {
        return dtype == "float32" ? 4 : 8;
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
    const std::string& name,
    std::string dtype,
    std::vector<std::int64_t> shape,
    std::string role,
    std::uint64_t offset) {
    CHECK(!initializers.contains(name));
    initializers[name] = {
        {"file", std::string(kDataFile)},
        {"offset", offset},
        {"length", ByteLength(dtype, shape)},
        {"dtype", std::move(dtype)},
        {"shape", std::move(shape)},
        {"role", std::move(role)}};
}

class ManifestBuilder final {
public:
    explicit ManifestBuilder(std::uint64_t model_size)
        : manifest_{
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
               }},
              {"initializers", json::object()},
              {"weight_objects", json::array()}} {}

    void AddMatMul(
        const std::string& name,
        std::int64_t k,
        std::int64_t n) {
        const std::string qweight = name + ".qweight";
        const std::string scales = name + ".scales";
        const std::string qzeros = name + ".qzeros";
        manifest_["weight_objects"].push_back({
            {"name", name},
            {"kind", "matmul"},
            {"descriptor",
             {
                 {"k", k},
                 {"n", n},
                 {"group_size", 128},
                 {"has_bias", false},
             }},
            {"roles",
             {
                 {"qweight", qweight},
                 {"scales", scales},
                 {"qzeros", qzeros},
             }}});

        AddInitializer(
            manifest_["initializers"],
            qweight,
            "uint8",
            {n, k / 2},
            "matmul.qweight",
            NextOffset());
        AddInitializer(
            manifest_["initializers"],
            scales,
            "float16",
            {n, k / 128},
            "matmul.scales",
            NextOffset());
        AddInitializer(
            manifest_["initializers"],
            qzeros,
            "uint8",
            {n, ((k / 128) + 1) / 2},
            "matmul.qzeros",
            NextOffset());
    }

    void AddSsMlp(int layer) {
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

        for (const std::string projection : {"gate", "up", "down"}) {
            const std::int64_t k =
                projection == "down" ? 8192 : 3072;
            const std::int64_t n =
                projection == "down" ? 3072 : 8192;
            const std::string prefix =
                base + ".mlp." + projection +
                "_proj.MatMulNBits";
            const std::string role_prefix =
                "ssmlp." + projection;
            for (const std::string component :
                 {"qweight", "scales", "qzeros"}) {
                roles[projection + "_" + component] =
                    prefix + "." + component;
            }
            AddInitializer(
                manifest_["initializers"],
                prefix + ".qweight",
                "uint8",
                {n, k / 2},
                role_prefix + ".qweight",
                NextOffset());
            AddInitializer(
                manifest_["initializers"],
                prefix + ".scales",
                "float16",
                {n, k / 128},
                role_prefix + ".scales",
                NextOffset());
            AddInitializer(
                manifest_["initializers"],
                prefix + ".qzeros",
                "uint8",
                {n, ((k / 128) + 1) / 2},
                role_prefix + ".qzeros",
                NextOffset());
        }

        AddInitializer(
            manifest_["initializers"],
            norm0,
            "float16",
            {3072},
            "ssmlp.norm0",
            NextOffset());
        AddInitializer(
            manifest_["initializers"],
            norm1,
            "float16",
            {3072},
            "ssmlp.norm1",
            NextOffset());
        manifest_["weight_objects"].push_back({
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

    json Finish() {
        for (int layer = 0; layer < 32; ++layer) {
            const std::string base =
                "model.layers." + std::to_string(layer) +
                ".attn.";
            AddMatMul(
                base + "q_proj.MatMulNBits",
                3072,
                3072);
            AddMatMul(
                base + "k_proj.MatMulNBits",
                3072,
                1024);
            AddMatMul(
                base + "v_proj.MatMulNBits",
                3072,
                1024);
            AddMatMul(
                base + "o_proj.MatMulNBits",
                3072,
                3072);
            AddSsMlp(layer);
        }
        AddMatMul("lm_head.MatMulNBits", 3072, 200064);

        AddInitializer(
            manifest_["initializers"],
            "model.embed_tokens.weight",
            "float16",
            {200064, 3072},
            "embedding",
            0);
        AddInitializer(
            manifest_["initializers"],
            "model.layers.0.input_layernorm.weight",
            "float16",
            {3072},
            "input_norm",
            NextOffset());
        AddInitializer(
            manifest_["initializers"],
            "cos_cache",
            "float16",
            {4096, 48},
            "cos_cache",
            NextOffset());
        AddInitializer(
            manifest_["initializers"],
            "sin_cache",
            "float16",
            {4096, 48},
            "sin_cache",
            NextOffset());

        CHECK(manifest_["weight_objects"].size() == 161);
        CHECK(manifest_["initializers"].size() == 743);
        return std::move(manifest_);
    }

private:
    std::uint64_t NextOffset() {
        const std::uint64_t result = next_offset_;
        next_offset_ += 16;
        return result;
    }

    json manifest_;
    std::uint64_t next_offset_ = 4096;
};

class SyntheticPackage final {
public:
    SyntheticPackage()
        : temp_("fastflowlm-phi4-engine-model") {
        const auto model_path = temp_.path() / "model.onnx";
        {
            std::ofstream model(model_path, std::ios::binary);
            model << "model";
        }
        CreateSparseFile(temp_.path() / kDataFile, kDataBytes);
        ManifestBuilder builder(
            std::filesystem::file_size(model_path));
        std::ofstream manifest(
            temp_.path() / kManifestName,
            std::ios::binary);
        manifest << builder.Finish().dump(2) << '\n';
        if (!manifest) {
            throw std::runtime_error(
                "failed to write synthetic engine manifest");
        }
    }

    const std::filesystem::path& path() const noexcept {
        return temp_.path();
    }

private:
    TempDirectory temp_;
};

float HalfToFloat(std::uint16_t value) {
    const bool negative = (value & 0x8000u) != 0;
    const unsigned exponent = (value >> 10) & 0x1fu;
    const unsigned mantissa = value & 0x03ffu;
    float result = 0.0f;
    if (exponent == 0) {
        result = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        result = mantissa == 0
                     ? std::numeric_limits<float>::infinity()
                     : std::numeric_limits<float>::quiet_NaN();
    } else {
        result = std::ldexp(
            1.0f + static_cast<float>(mantissa) / 1024.0f,
            static_cast<int>(exponent) - 15);
    }
    return negative ? -result : result;
}

std::uint16_t FloatToHalf(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign =
        static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const std::uint32_t source_exponent = (bits >> 23) & 0xffu;
    const std::uint32_t source_mantissa = bits & 0x007fffffu;
    if (source_exponent == 0xffu) {
        return static_cast<std::uint16_t>(
            sign | (source_mantissa == 0 ? 0x7c00u : 0x7e00u));
    }
    const int exponent = static_cast<int>(source_exponent) - 127 + 15;
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        const std::uint32_t mantissa =
            source_mantissa | 0x00800000u;
        const unsigned shift = static_cast<unsigned>(14 - exponent);
        const std::uint32_t rounded =
            (mantissa + (1u << (shift - 1)) - 1u +
             ((mantissa >> shift) & 1u)) >>
            shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t rounded =
        source_mantissa + 0x00000fffu +
        ((source_mantissa >> 13) & 1u);
    if ((rounded & 0x00800000u) != 0) {
        if (exponent + 1 >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        return static_cast<std::uint16_t>(
            sign | (static_cast<unsigned>(exponent + 1) << 10));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<unsigned>(exponent) << 10) |
        (rounded >> 13));
}

std::uint16_t FloatToBf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

float ReadElement(
    ryzenai_corelib_data_type type,
    const void* source,
    std::size_t index) {
    switch (type) {
        case ryzenai_corelib_data_type_fp16:
            return HalfToFloat(
                static_cast<const std::uint16_t*>(source)[index]);
        case ryzenai_corelib_data_type_bf16: {
            const std::uint32_t bits =
                static_cast<std::uint32_t>(
                    static_cast<const std::uint16_t*>(
                        source)[index])
                << 16;
            return std::bit_cast<float>(bits);
        }
        case ryzenai_corelib_data_type_fp32:
            return static_cast<const float*>(source)[index];
        default:
            throw std::runtime_error(
                "engine fake received unsupported conversion source");
    }
}

void WriteElement(
    ryzenai_corelib_data_type type,
    void* destination,
    std::size_t index,
    float value) {
    switch (type) {
        case ryzenai_corelib_data_type_fp16:
            static_cast<std::uint16_t*>(destination)[index] =
                FloatToHalf(value);
            return;
        case ryzenai_corelib_data_type_bf16:
            static_cast<std::uint16_t*>(destination)[index] =
                FloatToBf16(value);
            return;
        case ryzenai_corelib_data_type_fp32:
            static_cast<float*>(destination)[index] = value;
            return;
        default:
            throw std::runtime_error(
                "engine fake received unsupported conversion destination");
    }
}

enum class ObjectKind { Stream, Tensor, MatMulWeight, SsMlpWeight };

struct FakeObject {
    explicit FakeObject(ObjectKind kind_value, std::string label_value)
        : kind(kind_value),
          label(std::move(label_value)) {}
    virtual ~FakeObject() = default;

    ObjectKind kind;
    std::string label;
    bool released = false;
};

struct FakeTensor final : FakeObject {
    static constexpr std::size_t kPageBytes = 4096;
    using Page = std::array<std::byte, kPageBytes>;

    FakeTensor(
        ryzenai_corelib_data_type type,
        std::vector<std::int64_t> dimensions,
        std::size_t bytes,
        std::string label)
        : FakeObject(ObjectKind::Tensor, std::move(label)),
          data_type(type),
          shape(std::move(dimensions)),
          byte_size(bytes) {}

    std::byte DefaultByte(std::size_t offset) const noexcept {
        return std::byte{
            static_cast<unsigned char>(
                offset % 2 == 0 ? 0xADu : 0xDEu)};
    }

    Page& MutablePage(std::size_t page_index) {
        auto [found, inserted] =
            pages.try_emplace(page_index, nullptr);
        if (inserted) {
            found->second = std::make_unique<Page>();
            const std::size_t base = page_index * kPageBytes;
            for (std::size_t index = 0; index < kPageBytes; ++index) {
                (*found->second)[index] = DefaultByte(base + index);
            }
        }
        return *found->second;
    }

    std::byte ReadByte(std::size_t offset) const {
        const std::size_t page_index = offset / kPageBytes;
        const auto found = pages.find(page_index);
        if (found == pages.end()) {
            return DefaultByte(offset);
        }
        return (*found->second)[offset % kPageBytes];
    }

    bool Write(
        const void* source,
        std::size_t size,
        std::size_t offset) {
        if (
            (source == nullptr && size != 0) ||
            offset > byte_size ||
            size > byte_size - offset) {
            return false;
        }
        const auto* bytes = static_cast<const std::byte*>(source);
        for (std::size_t index = 0; index < size; ++index) {
            MutablePage((offset + index) / kPageBytes)
                [(offset + index) % kPageBytes] = bytes[index];
        }
        return true;
    }

    bool Read(
        void* destination,
        std::size_t size,
        std::size_t offset) const {
        if (
            (destination == nullptr && size != 0) ||
            offset > byte_size ||
            size > byte_size - offset) {
            return false;
        }
        auto* bytes = static_cast<std::byte*>(destination);
        for (std::size_t index = 0; index < size; ++index) {
            bytes[index] = ReadByte(offset + index);
        }
        return true;
    }

    std::uint16_t ReadWord(std::size_t word_index) const {
        const std::size_t offset = word_index * sizeof(std::uint16_t);
        const std::array<std::byte, 2> bytes{
            ReadByte(offset),
            ReadByte(offset + 1)};
        std::uint16_t value = 0;
        std::memcpy(&value, bytes.data(), sizeof(value));
        return value;
    }

    void WriteWord(std::size_t word_index, std::uint16_t value) {
        const std::size_t offset = word_index * sizeof(value);
        CHECK(Write(&value, sizeof(value), offset));
    }

    void FillWords(
        std::size_t first,
        std::size_t count,
        std::uint16_t value) {
        for (std::size_t index = 0; index < count; ++index) {
            WriteWord(first + index, value);
        }
    }

    bool ContainsPoisonWords(
        std::size_t first,
        std::size_t count) const {
        for (std::size_t index = 0; index < count; ++index) {
            if (ReadWord(first + index) == kPoison) {
                return true;
            }
        }
        return false;
    }

    ryzenai_corelib_data_type data_type;
    std::vector<std::int64_t> shape;
    std::size_t byte_size;
    std::unordered_map<std::size_t, std::unique_ptr<Page>> pages;
};

struct FakeMatMulWeight final : FakeObject {
    FakeMatMulWeight(
        std::string label,
        ryzenai_corelib_matmul_bf16_weights_desc value)
        : FakeObject(ObjectKind::MatMulWeight, std::move(label)),
          desc(value) {}

    ryzenai_corelib_matmul_bf16_weights_desc desc{};
};

struct FakeSsMlpWeight final : FakeObject {
    FakeSsMlpWeight(
        std::string label,
        ryzenai_corelib_ssmlp_bf16_weights_desc value)
        : FakeObject(ObjectKind::SsMlpWeight, std::move(label)),
          desc(value) {}

    ryzenai_corelib_ssmlp_bf16_weights_desc desc{};
};

struct MatMulCall {
    std::string label;
    FakeTensor* input;
    FakeTensor* output;
    std::int64_t rows;
};

struct SsMlpCall {
    FakeTensor* input;
    FakeTensor* residual;
    FakeTensor* skip_sum;
    FakeTensor* normalized;
    std::int64_t rows;
};

struct MhaCall {
    int layer;
    FakeTensor* query;
    FakeTensor* key;
    FakeTensor* key_cache;
    FakeTensor* value_cache;
    FakeTensor* output;
    std::int64_t rows;
    std::int64_t position;
};

struct TensorWriteCall {
    FakeTensor* tensor;
    std::size_t size;
    std::size_t offset;
};

enum class FailurePoint {
    None,
    FirstQ,
    KAfterQ,
    Synchronize,
    StageBadAlloc,
    ScatterBadAlloc,
    ScatterUnknown
};

struct UnknownFailure final {};

struct RecordingState {
    std::vector<std::unique_ptr<FakeObject>> objects;
    std::unordered_map<void*, FakeObject*> object_index;
    std::vector<FakeTensor*> tensors;
    std::vector<std::string> events;
    std::vector<std::string> release_labels;
    std::vector<MatMulCall> matmul_calls;
    std::vector<SsMlpCall> ssmlp_calls;
    std::vector<MhaCall> mha_calls;
    std::vector<TensorWriteCall> stage_writes;
    std::set<FakeTensor*> q_tensors;
    std::set<FakeTensor*> k_tensors;
    std::set<FakeTensor*> v_tensors;
    std::set<FakeTensor*> attention_tensors;
    std::set<FakeTensor*> skip_sum_tensors;
    std::set<FakeTensor*> normalized_tensors;
    FakeTensor* staged_hidden = nullptr;
    FakeTensor* staged_residual = nullptr;
    FailurePoint failure = FailurePoint::None;
    std::size_t matmul_weight_count = 0;
    std::size_t ssmlp_weight_count = 0;
    std::size_t synchronize_calls = 0;
    int active_layer = -1;
    std::int64_t active_rows = 0;
    bool input_poison_observed = false;
    bool host_read_poison_observed = false;
    bool cache_publish_poison_observed = false;
    bool terminator_called = false;
    unsigned int termination_code = 0;
    std::optional<std::thread::id> load_thread;
    bool load_thread_consistent = true;

    template <class Object, class... Args>
    Object* Create(Args&&... args) {
        auto object =
            std::make_unique<Object>(std::forward<Args>(args)...);
        auto* result = object.get();
        object_index.emplace(result, result);
        objects.push_back(std::move(object));
        return result;
    }

    FakeObject* Object(void* value) const {
        const auto found = object_index.find(value);
        return found == object_index.end() ? nullptr : found->second;
    }

    FakeTensor* Tensor(void* value) const {
        auto* object = Object(value);
        if (object == nullptr || object->kind != ObjectKind::Tensor) {
            return nullptr;
        }
        return static_cast<FakeTensor*>(object);
    }

    void ObserveLoadThread() {
        const auto current = std::this_thread::get_id();
        if (!load_thread.has_value()) {
            load_thread = current;
        } else {
            load_thread_consistent =
                load_thread_consistent && *load_thread == current;
        }
    }

    void ResetExecutionRecords() {
        events.clear();
        matmul_calls.clear();
        ssmlp_calls.clear();
        mha_calls.clear();
        stage_writes.clear();
        q_tensors.clear();
        k_tensors.clear();
        v_tensors.clear();
        attention_tensors.clear();
        skip_sum_tensors.clear();
        normalized_tensors.clear();
        synchronize_calls = 0;
        active_layer = -1;
        active_rows = 0;
        input_poison_observed = false;
        host_read_poison_observed = false;
        cache_publish_poison_observed = false;
    }
};

RecordingState* g_state = nullptr;

RecordingState& State() {
    if (g_state == nullptr) {
        throw std::runtime_error("engine recording fake is not active");
    }
    return *g_state;
}

std::size_t TypeSize(ryzenai_corelib_data_type type) {
    switch (type) {
        case ryzenai_corelib_data_type_fp16:
        case ryzenai_corelib_data_type_bf16:
            return 2;
        case ryzenai_corelib_data_type_fp32:
            return 4;
        default:
            throw std::runtime_error(
                "engine fake cannot size this tensor dtype");
    }
}

std::size_t TensorBytes(
    ryzenai_corelib_data_type type,
    std::span<const std::int64_t> shape) {
    std::size_t elements = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0) {
            throw std::runtime_error(
                "engine fake received non-positive tensor shape");
        }
        elements *= static_cast<std::size_t>(dimension);
    }
    return elements * TypeSize(type);
}

std::string MatMulLabel(std::size_t ordinal) {
    if (ordinal == 128) {
        return "lm_head";
    }
    const std::size_t layer = ordinal / 4;
    constexpr std::array<std::string_view, 4> projections{
        "q",
        "k",
        "v",
        "o"};
    return std::string(projections[ordinal % 4]) + "_" +
           std::to_string(layer);
}

std::string Projection(const std::string& label) {
    if (label == "lm_head") {
        return label;
    }
    const auto separator = label.find('_');
    return separator == std::string::npos
               ? label
               : label.substr(0, separator);
}

int WeightLayer(const std::string& label) {
    if (label == "lm_head") {
        return -1;
    }
    return std::stoi(label.substr(label.find('_') + 1));
}

ryzenai_corelib_status RecordingConvert(
    ryzenai_corelib_data_type source_type,
    const void* source,
    ryzenai_corelib_data_type destination_type,
    void* destination,
    std::size_t count) {
    auto& state = State();
    state.ObserveLoadThread();
    if (
        state.failure == FailurePoint::StageBadAlloc &&
        source_type == ryzenai_corelib_data_type_fp16 &&
        destination_type == ryzenai_corelib_data_type_fp32 &&
        count != static_cast<std::size_t>(constants::kHiddenSize)) {
        throw std::bad_alloc{};
    }
    if ((source == nullptr || destination == nullptr) && count != 0) {
        return ryzenai_corelib_status_bad_argument;
    }
    for (std::size_t index = 0; index < count; ++index) {
        WriteElement(
            destination_type,
            destination,
            index,
            ReadElement(source_type, source, index));
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingConvertStrided(
    ryzenai_corelib_data_type source_type,
    const void* source,
    std::size_t source_stride,
    ryzenai_corelib_data_type destination_type,
    void* destination,
    std::size_t destination_stride,
    std::size_t count,
    std::size_t row) {
    State().ObserveLoadThread();
    if (
        source == nullptr || destination == nullptr || row == 0 ||
        count % row != 0 || source_stride < row ||
        destination_stride < row) {
        return ryzenai_corelib_status_bad_argument;
    }
    const std::size_t rows = count / row;
    for (std::size_t row_index = 0; row_index < rows; ++row_index) {
        for (std::size_t column = 0; column < row; ++column) {
            WriteElement(
                destination_type,
                destination,
                row_index * destination_stride + column,
                ReadElement(
                    source_type,
                    source,
                    row_index * source_stride + column));
        }
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingMatMulPadShape(
    std::int64_t* m,
    std::int64_t*,
    std::int64_t*,
    std::uint32_t) {
    State().ObserveLoadThread();
    if (m == nullptr || *m <= 0) {
        return ryzenai_corelib_status_bad_argument;
    }
    *m = PaddedRows(*m);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingSsMlpPadRows(
    std::int64_t* m,
    std::int64_t,
    std::int64_t,
    std::uint32_t) {
    State().ObserveLoadThread();
    if (m == nullptr || *m <= 0) {
        return ryzenai_corelib_status_bad_argument;
    }
    *m = PaddedRows(*m);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingMhaPadRows(
    std::int64_t* m,
    const ryzenai_corelib_flat_mha_bf16_desc*) {
    State().ObserveLoadThread();
    if (m == nullptr || *m <= 0) {
        return ryzenai_corelib_status_bad_argument;
    }
    *m = PaddedRows(*m);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingCreateStream(
    ryzenai_corelib_stream_ptr* out) {
    auto& state = State();
    state.ObserveLoadThread();
    if (out == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    *out = state.Create<FakeObject>(ObjectKind::Stream, "stream");
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingCreateTensor(
    ryzenai_corelib_data_type data_type,
    const std::int64_t* shape,
    std::size_t shape_len,
    ryzenai_corelib_tensor_ptr* out) {
    auto& state = State();
    state.ObserveLoadThread();
    if (shape == nullptr || shape_len == 0 || out == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    std::vector<std::int64_t> dimensions(shape, shape + shape_len);
    const auto ordinal = state.tensors.size();
    auto* tensor = state.Create<FakeTensor>(
        data_type,
        dimensions,
        TensorBytes(data_type, dimensions),
        "tensor_" + std::to_string(ordinal));
    state.tensors.push_back(tensor);
    *out = tensor;
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingTensorGetByteSize(
    ryzenai_corelib_tensor_ptr tensor,
    std::size_t* out) {
    auto* value = State().Tensor(tensor);
    if (value == nullptr || out == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    *out = value->byte_size;
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingTensorWrite(
    ryzenai_corelib_tensor_ptr tensor,
    const void* source,
    std::size_t size,
    std::size_t offset) {
    auto& state = State();
    auto* value = state.Tensor(tensor);
    if (value == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }

    const bool is_cache =
        value->shape ==
        std::vector<std::int64_t>{8, 4096, 128};
    if (is_cache) {
        if (
            state.failure == FailurePoint::ScatterBadAlloc) {
            // The injected allocation failure belongs to the preceding V read.
            return ryzenai_corelib_status_bad_argument;
        }
        if (value->label.starts_with("tensor_")) {
            value->label =
                "v_cache_" + std::to_string(state.active_layer);
        }
        const auto* words =
            static_cast<const std::uint16_t*>(source);
        const std::size_t word_count = size / sizeof(std::uint16_t);
        state.cache_publish_poison_observed =
            state.cache_publish_poison_observed ||
            std::any_of(
                words,
                words + word_count,
                [](std::uint16_t word) { return word == kPoison; });
        constexpr std::size_t head_pitch_bytes =
            4096u * 128u * sizeof(std::uint16_t);
        const std::size_t head = offset / head_pitch_bytes;
        state.events.push_back(
            "tensor_write_v_head_" + std::to_string(head));
    } else if (
        value->data_type == ryzenai_corelib_data_type_bf16 &&
        value->shape ==
            std::vector<std::int64_t>{4096, 3072}) {
        state.stage_writes.push_back({value, size, offset});
        if (state.stage_writes.size() % 2 == 1) {
            state.staged_hidden = value;
        } else {
            state.staged_residual = value;
        }
    }

    return value->Write(source, size, offset)
               ? ryzenai_corelib_status_success
               : ryzenai_corelib_status_bad_argument;
}

ryzenai_corelib_status RecordingTensorRead(
    ryzenai_corelib_tensor_ptr tensor,
    void* destination,
    std::size_t size,
    std::size_t offset) {
    auto& state = State();
    auto* value = state.Tensor(tensor);
    if (value == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }

    if (state.v_tensors.contains(value)) {
        if (state.failure == FailurePoint::ScatterBadAlloc) {
            throw std::bad_alloc{};
        }
        if (state.failure == FailurePoint::ScatterUnknown) {
            throw UnknownFailure{};
        }
        state.events.emplace_back("tensor_read_v");
        const std::size_t expected =
            static_cast<std::size_t>(state.active_rows) *
            static_cast<std::size_t>(constants::kKvDimension) *
            sizeof(std::uint16_t);
        if (offset != 0 || size != expected) {
            state.host_read_poison_observed = true;
        }
    } else if (
        value->shape ==
        std::vector<std::int64_t>{1, 200064}) {
        state.events.emplace_back("tensor_read_logits");
    }

    state.host_read_poison_observed =
        state.host_read_poison_observed ||
        value->ContainsPoisonWords(
            offset / sizeof(std::uint16_t),
            size / sizeof(std::uint16_t));
    return value->Read(destination, size, offset)
               ? ryzenai_corelib_status_success
               : ryzenai_corelib_status_bad_argument;
}

ryzenai_corelib_status RecordingMatMulWeightsCreate(
    const ryzenai_corelib_matmul_bf16_weights_desc* desc,
    const ryzenai_corelib_matmul_bf16_onnx_weights_components* components,
    ryzenai_corelib_matmul_bf16_weights_ptr* out) {
    auto& state = State();
    state.ObserveLoadThread();
    if (desc == nullptr || components == nullptr || out == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    const auto ordinal = state.matmul_weight_count++;
    *out = state.Create<FakeMatMulWeight>(
        MatMulLabel(ordinal),
        *desc);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingSsMlpWeightsCreate(
    const ryzenai_corelib_ssmlp_bf16_weights_desc* desc,
    const ryzenai_corelib_ssmlp_bf16_onnx_weights_components* components,
    ryzenai_corelib_ssmlp_bf16_weights_ptr* out) {
    auto& state = State();
    state.ObserveLoadThread();
    if (desc == nullptr || components == nullptr || out == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    const auto ordinal = state.ssmlp_weight_count++;
    *out = state.Create<FakeSsMlpWeight>(
        "ssmlp_" + std::to_string(ordinal),
        *desc);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingMatMulWeightsGetData(
    ryzenai_corelib_matmul_bf16_weights_ptr weights,
    const void** data,
    std::size_t* size) {
    if (
        State().Object(weights) == nullptr || data != nullptr ||
        size == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    *size = 17;
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingSsMlpWeightsGetData(
    ryzenai_corelib_ssmlp_bf16_weights_ptr weights,
    const void** data,
    std::size_t* size) {
    if (
        State().Object(weights) == nullptr || data != nullptr ||
        size == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    *size = 29;
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingMatMul(
    ryzenai_corelib_stream_ptr stream,
    ryzenai_corelib_tensor_ptr input,
    std::int64_t rows,
    ryzenai_corelib_matmul_bf16_weights_ptr weights,
    ryzenai_corelib_tensor_ptr output) {
    auto& state = State();
    auto* stream_object = state.Object(stream);
    auto* input_tensor = state.Tensor(input);
    auto* output_tensor = state.Tensor(output);
    auto* weight_object = state.Object(weights);
    if (
        stream_object == nullptr ||
        stream_object->kind != ObjectKind::Stream ||
        input_tensor == nullptr || output_tensor == nullptr ||
        weight_object == nullptr ||
        weight_object->kind != ObjectKind::MatMulWeight ||
        rows <= 0) {
        return ryzenai_corelib_status_bad_argument;
    }
    auto* weight = static_cast<FakeMatMulWeight*>(weight_object);
    const std::string projection = Projection(weight->label);
    const int layer = WeightLayer(weight->label);
    if (layer >= 0) {
        state.active_layer = layer;
    }
    state.active_rows = rows;
    state.events.push_back("matmul_" + projection);
    state.matmul_calls.push_back(
        {weight->label, input_tensor, output_tensor, rows});

    if (
        state.failure == FailurePoint::FirstQ &&
        weight->label == "q_0") {
        flm::test::SetLastErrorMessage("injected first-q failure");
        return ryzenai_corelib_status_failure;
    }
    if (
        state.failure == FailurePoint::KAfterQ &&
        weight->label == "k_0") {
        flm::test::SetLastErrorMessage("injected k failure");
        return ryzenai_corelib_status_failure;
    }

    const std::size_t input_words =
        static_cast<std::size_t>(PaddedRows(rows)) *
        static_cast<std::size_t>(weight->desc.k);
    state.input_poison_observed =
        state.input_poison_observed ||
        input_tensor->ContainsPoisonWords(0, input_words);
    const std::size_t output_words =
        static_cast<std::size_t>(PaddedRows(rows)) *
        static_cast<std::size_t>(weight->desc.n);
    output_tensor->FillWords(
        0,
        output_words,
        projection == "lm_head"
            ? static_cast<std::uint16_t>(0x3F80u)
            : static_cast<std::uint16_t>(
                  0x1000u + static_cast<unsigned>(
                                std::max(layer, 0) * 8 +
                                static_cast<int>(
                                    state.matmul_calls.size() % 8))));

    if (projection == "q") {
        state.q_tensors.insert(output_tensor);
    } else if (projection == "k") {
        state.k_tensors.insert(output_tensor);
    } else if (projection == "v") {
        state.v_tensors.insert(output_tensor);
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingSsMlp(
    ryzenai_corelib_stream_ptr stream,
    ryzenai_corelib_tensor_ptr input,
    ryzenai_corelib_tensor_ptr residual,
    std::int64_t rows,
    ryzenai_corelib_ssmlp_bf16_weights_ptr weights,
    ryzenai_corelib_tensor_ptr skip_sum,
    ryzenai_corelib_tensor_ptr normalized) {
    auto& state = State();
    auto* stream_object = state.Object(stream);
    auto* input_tensor = state.Tensor(input);
    auto* residual_tensor = state.Tensor(residual);
    auto* skip_tensor = state.Tensor(skip_sum);
    auto* normalized_tensor = state.Tensor(normalized);
    auto* weight_object = state.Object(weights);
    if (
        stream_object == nullptr ||
        stream_object->kind != ObjectKind::Stream ||
        input_tensor == nullptr || residual_tensor == nullptr ||
        skip_tensor == nullptr || normalized_tensor == nullptr ||
        weight_object == nullptr ||
        weight_object->kind != ObjectKind::SsMlpWeight ||
        rows <= 0) {
        return ryzenai_corelib_status_bad_argument;
    }
    state.events.emplace_back("ssmlp");
    state.ssmlp_calls.push_back(
        {input_tensor,
         residual_tensor,
         skip_tensor,
         normalized_tensor,
         rows});

    const std::size_t words =
        static_cast<std::size_t>(PaddedRows(rows)) *
        static_cast<std::size_t>(constants::kHiddenSize);
    state.input_poison_observed =
        state.input_poison_observed ||
        input_tensor->ContainsPoisonWords(0, words) ||
        residual_tensor->ContainsPoisonWords(0, words);
    skip_tensor->FillWords(0, words, 0x2200u);
    normalized_tensor->FillWords(0, words, 0x2300u);
    state.skip_sum_tensors.insert(skip_tensor);
    state.normalized_tensors.insert(normalized_tensor);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingFlatMha(
    ryzenai_corelib_stream_ptr stream,
    const ryzenai_corelib_flat_mha_bf16_desc* desc,
    ryzenai_corelib_tensor_ptr query,
    ryzenai_corelib_tensor_ptr key,
    std::int64_t rows,
    std::int64_t position,
    ryzenai_corelib_tensor_ptr cos,
    ryzenai_corelib_tensor_ptr sin,
    ryzenai_corelib_tensor_ptr key_cache,
    ryzenai_corelib_tensor_ptr value_cache,
    ryzenai_corelib_tensor_ptr output) {
    auto& state = State();
    auto* stream_object = state.Object(stream);
    auto* q = state.Tensor(query);
    auto* k = state.Tensor(key);
    auto* cos_tensor = state.Tensor(cos);
    auto* sin_tensor = state.Tensor(sin);
    auto* k_cache = state.Tensor(key_cache);
    auto* v_cache = state.Tensor(value_cache);
    auto* out = state.Tensor(output);
    if (
        stream_object == nullptr ||
        stream_object->kind != ObjectKind::Stream ||
        desc == nullptr || q == nullptr || k == nullptr ||
        cos_tensor == nullptr || sin_tensor == nullptr ||
        k_cache == nullptr || v_cache == nullptr || out == nullptr ||
        rows <= 0 || position < 0 ||
        (rows > 1 && position != 0)) {
        return ryzenai_corelib_status_bad_argument;
    }
    if (k_cache->label.starts_with("tensor_")) {
        k_cache->label =
            "k_cache_" + std::to_string(state.active_layer);
    }
    if (v_cache->label.starts_with("tensor_")) {
        v_cache->label =
            "v_cache_" + std::to_string(state.active_layer);
    }
    state.events.emplace_back("flat_mha");
    state.mha_calls.push_back(
        {state.active_layer,
         q,
         k,
         k_cache,
         v_cache,
         out,
         rows,
         position});

    const std::size_t padded =
        static_cast<std::size_t>(PaddedRows(rows));
    state.input_poison_observed =
        state.input_poison_observed ||
        q->ContainsPoisonWords(
            0,
            padded *
                static_cast<std::size_t>(
                    constants::kQueryDimension)) ||
        k->ContainsPoisonWords(
            0,
            padded *
                static_cast<std::size_t>(
                    constants::kKvDimension));
    for (std::size_t head = 0; head < 8; ++head) {
        for (std::size_t row = 0;
             row < static_cast<std::size_t>(rows);
             ++row) {
            const std::size_t cache_base =
                ((head * 4096u) +
                 static_cast<std::size_t>(position) + row) *
                128u;
            state.input_poison_observed =
                state.input_poison_observed ||
                v_cache->ContainsPoisonWords(cache_base, 128);
            k_cache->FillWords(cache_base, 128, 0x3100u);
        }
    }
    out->FillWords(
        0,
        padded *
            static_cast<std::size_t>(constants::kQueryDimension),
        0x3200u);
    state.attention_tensors.insert(out);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingSynchronize(
    ryzenai_corelib_stream_ptr stream) {
    auto& state = State();
    auto* object = state.Object(stream);
    if (object == nullptr || object->kind != ObjectKind::Stream) {
        return ryzenai_corelib_status_bad_argument;
    }
    ++state.synchronize_calls;
    state.events.emplace_back("synchronize");
    if (state.failure == FailurePoint::Synchronize) {
        flm::test::SetLastErrorMessage("injected synchronize failure");
        return ryzenai_corelib_status_failure;
    }
    return ryzenai_corelib_status_success;
}

void RecordingRelease(ryzenai_corelib_object_ptr object) {
    auto& state = State();
    auto* value = state.Object(object);
    CHECK(value != nullptr);
    CHECK(!value->released);
    value->released = true;
    state.release_labels.push_back(value->label);
}

template <class Function>
void* FunctionAddress(Function function) {
    return reinterpret_cast<void*>(function);
}

std::shared_ptr<CorelibApi> ResolveRecordingCorelib(
    RecordingState& state) {
    g_state = &state;
    auto resolver = flm::test::CompleteCorelibResolver();
    resolver["ryzenai_corelib_object_release"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_object_release)>(
            &RecordingRelease));
    resolver["ryzenai_corelib_create_stream"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_create_stream)>(
            &RecordingCreateStream));
    resolver["ryzenai_corelib_stream_synchronize"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_stream_synchronize)>(
            &RecordingSynchronize));
    resolver["ryzenai_corelib_create_device_tensor"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_create_device_tensor)>(
            &RecordingCreateTensor));
    resolver["ryzenai_corelib_tensor_write"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_tensor_write)>(
            &RecordingTensorWrite));
    resolver["ryzenai_corelib_tensor_read"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_tensor_read)>(
            &RecordingTensorRead));
    resolver["ryzenai_corelib_tensor_get_byte_size"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_tensor_get_byte_size)>(
            &RecordingTensorGetByteSize));
    resolver["ryzenai_corelib_convert"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_convert)>(
            &RecordingConvert));
    resolver["ryzenai_corelib_convert_strided"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_convert_strided)>(
            &RecordingConvertStrided));
    resolver["ryzenai_corelib_matmul_bf16_pad_shape"] =
        FunctionAddress(
            static_cast<
                decltype(&::ryzenai_corelib_matmul_bf16_pad_shape)>(
                &RecordingMatMulPadShape));
    resolver["ryzenai_corelib_ssmlp_bf16_pad_rows"] =
        FunctionAddress(
            static_cast<
                decltype(&::ryzenai_corelib_ssmlp_bf16_pad_rows)>(
                &RecordingSsMlpPadRows));
    resolver["ryzenai_corelib_flat_mha_bf16_pad_rows"] =
        FunctionAddress(
            static_cast<
                decltype(&::ryzenai_corelib_flat_mha_bf16_pad_rows)>(
                &RecordingMhaPadRows));
    resolver
        ["ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components"] =
            FunctionAddress(
                static_cast<decltype(
                    &::ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components)>(
                    &RecordingMatMulWeightsCreate));
    resolver["ryzenai_corelib_matmul_bf16_weights_get_data"] =
        FunctionAddress(
            static_cast<decltype(
                &::ryzenai_corelib_matmul_bf16_weights_get_data)>(
                &RecordingMatMulWeightsGetData));
    resolver
        ["ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components"] =
            FunctionAddress(
                static_cast<decltype(
                    &::ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components)>(
                    &RecordingSsMlpWeightsCreate));
    resolver["ryzenai_corelib_ssmlp_bf16_weights_get_data"] =
        FunctionAddress(
            static_cast<decltype(
                &::ryzenai_corelib_ssmlp_bf16_weights_get_data)>(
                &RecordingSsMlpWeightsGetData));
    resolver["ryzenai_corelib_matmul_bf16"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_matmul_bf16)>(
            &RecordingMatMul));
    resolver["ryzenai_corelib_ssmlp_bf16"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_ssmlp_bf16)>(
            &RecordingSsMlp));
    resolver["ryzenai_corelib_flat_mha_bf16"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_flat_mha_bf16)>(
            &RecordingFlatMha));
    return CorelibApi::ResolveForTest(
        [resolver = std::move(resolver)](std::string_view name) mutable
            -> void* {
            const auto found = resolver.find(std::string(name));
            return found == resolver.end() ? nullptr : found->second;
        });
}

std::chrono::system_clock::time_point KnownStartTime() {
    using namespace std::chrono;
    return sys_days{year{2026} / September / day{1}};
}

FatalRecordStore MakeRecords(const std::filesystem::path& root) {
    return FatalRecordStore(
        root,
        GetCurrentProcessId(),
        KnownStartTime(),
        [](DWORD)
            -> std::optional<std::chrono::system_clock::time_point> {
            return std::nullopt;
        });
}

struct TerminationIntercept final {};

class EngineFixture final {
public:
    explicit EngineFixture(
        const SyntheticPackage& package,
        std::uint32_t max_length = 4096)
        : fatal_root_("fastflowlm-phi4-engine-fatal") {
        flm::test::ResetFakeCorelib();
        api_ = ResolveRecordingCorelib(state);
        runtime = CorelibRuntime::Create(
            api_,
            MakeRecords(fatal_root_.path()),
            [this](unsigned int code) {
                state.terminator_called = true;
                state.termination_code = code;
                throw TerminationIntercept{};
            });
        LM_Config config;
        engine = std::make_unique<phi4_corelib_aie4>(
            std::move(config),
            package.path(),
            runtime,
            max_length);
    }

    ~EngineFixture() noexcept {
        try {
            engine.reset();
            if (runtime && runtime->state() == ProcessState::Healthy) {
                runtime->ShutdownHealthy();
            }
        } catch (...) {
            std::terminate();
        }
        if (g_state == &state) {
            g_state = nullptr;
        }
    }

    void DestroyHealthy() {
        CHECK(runtime->state() == ProcessState::Healthy);
        engine.reset();
        CHECK(api_->live_object_count() == 0);
        runtime->ShutdownHealthy();
    }

    json FatalRecord() const {
        for (const auto& entry :
             std::filesystem::directory_iterator(fatal_root_.path())) {
            const auto name = entry.path().filename().string();
            if (
                name.starts_with("corelib-fatal-") &&
                name.ends_with(".json")) {
                std::ifstream input(entry.path(), std::ios::binary);
                return json::parse(input);
            }
        }
        throw std::runtime_error("expected corelib fatal record");
    }

    RecordingState state;
    std::shared_ptr<CorelibRuntime> runtime;
    std::unique_ptr<phi4_corelib_aie4> engine;

private:
    TempDirectory fatal_root_;
    std::shared_ptr<CorelibApi> api_;
};

void CheckLayerOrder(const std::vector<std::string>& events) {
    const std::array<std::string, 19> expected{
        "matmul_q",
        "matmul_k",
        "matmul_v",
        "synchronize",
        "tensor_read_v",
        "tensor_write_v_head_0",
        "tensor_write_v_head_1",
        "tensor_write_v_head_2",
        "tensor_write_v_head_3",
        "tensor_write_v_head_4",
        "tensor_write_v_head_5",
        "tensor_write_v_head_6",
        "tensor_write_v_head_7",
        "flat_mha",
        "synchronize",
        "matmul_o",
        "synchronize",
        "ssmlp",
        "synchronize"};
    CHECK(events.size() == 32u * expected.size() + 3u);
    for (std::size_t layer = 0; layer < 32; ++layer) {
        CHECK(std::equal(
            expected.begin(),
            expected.end(),
            events.begin() + layer * expected.size()));
    }
    const auto final = events.end() - 3;
    CHECK(final[0] == "matmul_lm_head");
    CHECK(final[1] == "synchronize");
    CHECK(final[2] == "tensor_read_logits");
}

void CheckDistinctAndPingPong(const RecordingState& state) {
    CHECK(state.ssmlp_calls.size() == 32);
    FakeTensor* first_normalized =
        state.ssmlp_calls.front().normalized;
    FakeTensor* second_normalized =
        state.ssmlp_calls.at(1).normalized;
    CHECK(first_normalized != second_normalized);
    for (std::size_t layer = 0;
         layer < state.ssmlp_calls.size();
         ++layer) {
        const auto& call = state.ssmlp_calls[layer];
        CHECK(call.input != call.residual);
        CHECK(call.input != call.skip_sum);
        CHECK(call.input != call.normalized);
        CHECK(call.residual != call.skip_sum);
        CHECK(call.residual != call.normalized);
        CHECK(call.skip_sum != call.normalized);
        CHECK(
            call.normalized ==
            (layer % 2 == 0 ? first_normalized : second_normalized));
        if (layer + 1 < state.ssmlp_calls.size()) {
            CHECK(
                state.ssmlp_calls[layer + 1].input ==
                call.normalized);
            CHECK(
                state.ssmlp_calls[layer + 1].residual ==
                call.skip_sum);
        }
    }
}

void CheckActualPersistentTail(
    const std::set<FakeTensor*>& tensors,
    std::size_t live_prefix_words) {
    CHECK(!tensors.empty());
    for (const auto* tensor : tensors) {
        CHECK(tensor->ReadWord(live_prefix_words) == kPoison);
    }
}

void PoisonActualPersistentTail(
    const std::set<FakeTensor*>& tensors,
    std::size_t first_word,
    std::size_t word_count) {
    for (auto* tensor : tensors) {
        tensor->FillWords(first_word, word_count, kPoison);
    }
}

void CheckMetrics(
    const Phi4Aie4Metrics& metrics,
    std::uint64_t passes) {
    CHECK(metrics.model_load_ns > 0);
    CHECK(metrics.weight_pack_ns > 0);
    CHECK(metrics.packed_weight_bytes ==
          129u * 17u + 32u * 29u);
    CHECK(metrics.mapped_source_bytes == kDataBytes);
    CHECK(metrics.kv_bytes == 536870912u);
    CHECK(metrics.scratch_bytes > 0);
    CHECK(metrics.device_tensor_create_count == 76);
    CHECK(metrics.weight_create_count == 161);
    CHECK(metrics.dispatch_count == passes * 193u);
    CHECK(metrics.synchronize_count == passes * 129u);
    CHECK(metrics.v_read_calls == passes * 32u);
    CHECK(metrics.v_write_calls == passes * 32u * 8u);
    CHECK(metrics.v_bytes > 0);
    CHECK(metrics.v_scatter_ns > 0);
}

void TestOrderBuffersTailsStateAndMetrics(
    const SyntheticPackage& package) {
    EngineFixture fixture(package);
    CHECK(fixture.state.load_thread_consistent);
    CHECK(fixture.state.load_thread.has_value());
    CHECK(fixture.state.matmul_weight_count == 129);
    CHECK(fixture.state.ssmlp_weight_count == 32);
    CHECK(fixture.state.tensors.size() == 76);

    fixture.state.ResetExecutionRecords();
    std::vector<int> prompt{1, 2, 3};
    buffer<bf16> logits = fixture.engine->prefill(prompt);
    CHECK(logits.size() == 200064);
    CHECK(logits.is_owner());
    CHECK(fixture.engine->get_current_context_length() == 3);
    CHECK(fixture.state.synchronize_calls == 129);
    CheckLayerOrder(fixture.state.events);
    CheckDistinctAndPingPong(fixture.state);
    CHECK(fixture.state.stage_writes.size() == 2);
    const std::size_t padded_hidden_bytes =
        static_cast<std::size_t>(PaddedRows(3)) *
        static_cast<std::size_t>(constants::kHiddenSize) *
        sizeof(std::uint16_t);
    CHECK(fixture.state.stage_writes[0].size ==
          padded_hidden_bytes);
    CHECK(fixture.state.stage_writes[1].size ==
          padded_hidden_bytes);
    CHECK(!fixture.state.input_poison_observed);
    CHECK(!fixture.state.host_read_poison_observed);
    CHECK(!fixture.state.cache_publish_poison_observed);

    const std::size_t hidden_tail =
        static_cast<std::size_t>(PaddedRows(3)) *
        static_cast<std::size_t>(constants::kHiddenSize);
    const std::size_t kv_tail =
        static_cast<std::size_t>(PaddedRows(3)) *
        static_cast<std::size_t>(constants::kKvDimension);
    CheckActualPersistentTail(
        fixture.state.q_tensors,
        hidden_tail);
    CheckActualPersistentTail(
        fixture.state.k_tensors,
        kv_tail);
    CheckActualPersistentTail(
        fixture.state.attention_tensors,
        hidden_tail);
    CheckActualPersistentTail(
        fixture.state.skip_sum_tensors,
        hidden_tail);
    CheckActualPersistentTail(
        fixture.state.normalized_tensors,
        hidden_tail);
    CHECK(fixture.state.staged_hidden->ReadWord(hidden_tail) ==
          kPoison);
    CHECK(fixture.state.staged_residual->ReadWord(hidden_tail) ==
          kPoison);

#if defined(DEV_BUILD)
    fixture.state.events.clear();
    const Phi4DebugSnapshot snapshot =
        fixture.engine->debug_snapshot();
    CHECK(snapshot.live_rows == 3);
    CHECK(snapshot.position == 3);
    CHECK(snapshot.layer0_k.size() == 8u * 3u * 128u);
    CHECK(snapshot.layer0_v.size() == 8u * 3u * 128u);
    CHECK(snapshot.layer31_k.size() == 8u * 3u * 128u);
    CHECK(snapshot.layer31_v.size() == 8u * 3u * 128u);
    CHECK(snapshot.last_hidden.size() == 3072);
    CHECK(snapshot.logits.size() == 200064);
    CHECK(std::none_of(
        snapshot.layer0_k.begin(),
        snapshot.layer0_k.end(),
        [](std::uint16_t value) { return value == kPoison; }));
    CHECK(std::none_of(
        snapshot.layer31_v.begin(),
        snapshot.layer31_v.end(),
        [](std::uint16_t value) { return value == kPoison; }));
#endif

    CheckMetrics(fixture.engine->metrics(), 1);

    const std::size_t one_hidden_row =
        static_cast<std::size_t>(constants::kHiddenSize);
    const std::size_t one_kv_row =
        static_cast<std::size_t>(constants::kKvDimension);
    PoisonActualPersistentTail(
        fixture.state.q_tensors,
        one_hidden_row,
        hidden_tail - one_hidden_row);
    PoisonActualPersistentTail(
        fixture.state.k_tensors,
        one_kv_row,
        kv_tail - one_kv_row);
    PoisonActualPersistentTail(
        fixture.state.attention_tensors,
        one_hidden_row,
        hidden_tail - one_hidden_row);
    PoisonActualPersistentTail(
        fixture.state.skip_sum_tensors,
        one_hidden_row,
        hidden_tail - one_hidden_row);
    PoisonActualPersistentTail(
        fixture.state.normalized_tensors,
        one_hidden_row,
        hidden_tail - one_hidden_row);
    fixture.state.staged_hidden->FillWords(
        one_hidden_row,
        hidden_tail - one_hidden_row,
        kPoison);
    fixture.state.staged_residual->FillWords(
        one_hidden_row,
        hidden_tail - one_hidden_row,
        kPoison);

    fixture.state.ResetExecutionRecords();
    std::vector<int> suffix{4, 5, 6};
    buffer<bf16> suffix_logits = fixture.engine->prefill(suffix);
    CHECK(suffix_logits.is_owner());
    CHECK(fixture.engine->get_current_context_length() == 6);
    CHECK(fixture.state.mha_calls.size() == 3u * 32u);
    for (std::size_t pass = 0; pass < 3; ++pass) {
        for (std::size_t layer = 0; layer < 32; ++layer) {
            const auto& call =
                fixture.state.mha_calls[pass * 32 + layer];
            CHECK(call.rows == 1);
            CHECK(call.position == 3 + static_cast<int>(pass));
        }
    }
    CHECK(fixture.state.synchronize_calls == 3u * 129u);
    CHECK(!fixture.state.input_poison_observed);
    CHECK(!fixture.state.host_read_poison_observed);
    CHECK(!fixture.state.cache_publish_poison_observed);
    CheckActualPersistentTail(
        fixture.state.q_tensors,
        one_hidden_row);
    CheckActualPersistentTail(
        fixture.state.k_tensors,
        one_kv_row);
    CheckActualPersistentTail(
        fixture.state.attention_tensors,
        one_hidden_row);
    CheckActualPersistentTail(
        fixture.state.skip_sum_tensors,
        one_hidden_row);
    CheckActualPersistentTail(
        fixture.state.normalized_tensors,
        one_hidden_row);
    CHECK(
        fixture.state.staged_hidden->ReadWord(one_hidden_row) ==
        kPoison);
    CHECK(
        fixture.state.staged_residual->ReadWord(one_hidden_row) ==
        kPoison);

    fixture.state.ResetExecutionRecords();
    buffer<bf16> forward_logits = fixture.engine->forward(7);
    CHECK(forward_logits.is_owner());
    CHECK(fixture.engine->get_current_context_length() == 7);
    CHECK(fixture.state.mha_calls.size() == 32);
    CHECK(std::all_of(
        fixture.state.mha_calls.begin(),
        fixture.state.mha_calls.end(),
        [](const MhaCall& call) {
            return call.rows == 1 && call.position == 6;
        }));

    CHECK(fixture.engine->checkpoint() == 7);
    (void)fixture.engine->forward(8);
    CHECK(fixture.engine->get_current_context_length() == 8);
    CHECK(fixture.engine->restore() == 7);
    CHECK(fixture.engine->get_current_context_length() == 7);
    fixture.engine->set_context_length(7);
    CheckThrowsContains(
        [&] { fixture.engine->set_context_length(6); },
        "current");

    fixture.engine->update_max_length(7);
    CheckThrowsContains(
        [&] { (void)fixture.engine->forward(9); },
        "maximum");
    CheckThrowsContains(
        [&] { fixture.engine->update_max_length(0); },
        "1..4096");
    CheckThrowsContains(
        [&] { fixture.engine->update_max_length(4097); },
        "1..4096");
    CheckThrowsContains(
        [&] { fixture.engine->update_max_length(6); },
        "current");
    fixture.engine->update_max_length(4096);

    CheckThrowsContains(
        [&] { (void)fixture.engine->get_k_cache(0, 0); },
        "unsupported");
    CheckThrowsContains(
        [&] { (void)fixture.engine->get_v_cache(0, 0); },
        "unsupported");
    alignas(Q4NX) std::array<std::byte, sizeof(Q4NX)> q4nx_storage{};
    auto& q4nx =
        *reinterpret_cast<Q4NX*>(q4nx_storage.data());
    CheckThrowsContains(
        [&] { fixture.engine->load_weights(q4nx); },
        "unsupported");

    fixture.engine->clear_context();
    CHECK(fixture.engine->get_current_context_length() == 0);
    CheckThrowsContains(
        [&] { (void)fixture.engine->restore(); },
        "checkpoint");

    fixture.state.ResetExecutionRecords();
    std::vector<int> one_token{10};
    (void)fixture.engine->prefill(one_token);
    CHECK(fixture.engine->get_current_context_length() == 1);
    CHECK(std::all_of(
        fixture.state.mha_calls.begin(),
        fixture.state.mha_calls.end(),
        [](const MhaCall& call) {
            return call.rows == 1 && call.position == 0;
        }));
    CHECK(fixture.state.stage_writes.size() == 2);
    CHECK(
        fixture.state.stage_writes[0].size ==
        static_cast<std::size_t>(constants::kHiddenSize) *
            sizeof(std::uint16_t));

    fixture.engine->clear_context();
    fixture.state.ResetExecutionRecords();
    std::vector<int> too_many(4097, 0);
    CheckThrowsContains(
        [&] { (void)fixture.engine->prefill(too_many); },
        "maximum");
    CHECK(fixture.state.matmul_calls.empty());
    CHECK(fixture.engine->get_current_context_length() == 0);

    fixture.state.ResetExecutionRecords();
    fixture.DestroyHealthy();
    CHECK(fixture.state.synchronize_calls == 1);
    CHECK(!fixture.state.release_labels.empty());
    CHECK(fixture.state.release_labels.front() == "stream");
    const auto first_weight = std::find_if(
        fixture.state.release_labels.begin(),
        fixture.state.release_labels.end(),
        [](const std::string& label) {
            return label.starts_with("q_") ||
                   (label.starts_with("k_") &&
                    !label.starts_with("k_cache_")) ||
                   (label.starts_with("v_") &&
                    !label.starts_with("v_cache_")) ||
                   label.starts_with("o_") ||
                   label.starts_with("ssmlp_") ||
                   label == "lm_head";
        });
    CHECK(first_weight != fixture.state.release_labels.end());
    CHECK(std::all_of(
        fixture.state.release_labels.begin() + 1,
        first_weight,
        [](const std::string& label) {
            return label.starts_with("tensor_") ||
                   label.starts_with("k_cache_") ||
                   label.starts_with("v_cache_");
        }));
    CHECK(std::none_of(
        first_weight,
        fixture.state.release_labels.end(),
        [](const std::string& label) {
            return label.starts_with("tensor_") ||
                   label.starts_with("k_cache_") ||
                   label.starts_with("v_cache_");
        }));
}

void TestRecoverablePreSubmitFailures(
    const SyntheticPackage& package) {
    EngineFixture fixture(package);
    std::vector<int> prompt{1, 2};

    fixture.state.failure = FailurePoint::StageBadAlloc;
    try {
        (void)fixture.engine->prefill(prompt);
    } catch (const std::bad_alloc&) {
    } catch (...) {
        throw std::runtime_error(
            "pre-submit staging bad_alloc changed exception type");
    }
    CHECK(fixture.engine->get_current_context_length() == 0);
    CHECK(fixture.runtime->state() == ProcessState::Healthy);
    CHECK(fixture.state.matmul_calls.empty());

    fixture.state.failure = FailurePoint::FirstQ;
    try {
        (void)fixture.engine->prefill(prompt);
    } catch (const CorelibError& error) {
        CHECK(error.call == "q");
        CHECK(error.detail == "injected first-q failure");
    }
    CHECK(fixture.engine->get_current_context_length() == 0);
    CHECK(fixture.runtime->state() == ProcessState::Healthy);

    fixture.state.failure = FailurePoint::None;
    (void)fixture.engine->prefill(prompt);
    CHECK(fixture.engine->get_current_context_length() == 2);
}

template <class CheckRecord>
void CheckFatalFailure(
    const SyntheticPackage& package,
    FailurePoint failure,
    CheckRecord&& check_record) {
    EngineFixture fixture(package);
    fixture.state.failure = failure;
    std::vector<int> prompt{1, 2};
    try {
        (void)fixture.engine->prefill(prompt);
    } catch (const TerminationIntercept&) {
    }
    CHECK(fixture.state.terminator_called);
    CHECK(fixture.state.termination_code == 0xE0040001u);
    CHECK(fixture.runtime->state() == ProcessState::Terminating);
    CHECK(fixture.engine->get_current_context_length() == 0);
    check_record(fixture.FatalRecord());
}

void TestIrrevocableFailurePolicies(
    const SyntheticPackage& package) {
    CheckFatalFailure(
        package,
        FailurePoint::KAfterQ,
        [](const json& record) {
            CHECK(record.at("status") ==
                  ryzenai_corelib_status_failure);
            CHECK(record.at("call") == "k");
            CHECK(record.at("detail") == "injected k failure");
            CHECK(record.at("phase") == "qkv");
            CHECK(record.at("layer") == 0);
            CHECK(record.at("rows") == 2);
            CHECK(record.at("position") == 0);
        });

    CheckFatalFailure(
        package,
        FailurePoint::Synchronize,
        [](const json& record) {
            CHECK(record.at("status") ==
                  ryzenai_corelib_status_failure);
            CHECK(
                record.at("call") ==
                "ryzenai_corelib_stream_synchronize");
            CHECK(
                record.at("detail") ==
                "injected synchronize failure");
            CHECK(record.at("phase") == "qkv");
            CHECK(record.at("layer") == 0);
            CHECK(record.at("rows") == 2);
            CHECK(record.at("position") == 0);
        });

    CheckFatalFailure(
        package,
        FailurePoint::ScatterBadAlloc,
        [](const json& record) {
            CHECK(record.at("status") ==
                  ryzenai_corelib_status_failure);
            CHECK(record.at("call") == "host_exception");
            CHECK(
                record.at("detail")
                    .get<std::string>()
                    .find("alloc") != std::string::npos);
            CHECK(record.at("phase") == "v_scatter");
            CHECK(record.at("layer") == 0);
            CHECK(record.at("rows") == 2);
            CHECK(record.at("position") == 0);
        });

    CheckFatalFailure(
        package,
        FailurePoint::ScatterUnknown,
        [](const json& record) {
            CHECK(record.at("status") ==
                  ryzenai_corelib_status_failure);
            CHECK(record.at("call") == "unknown_exception");
            CHECK(
                record.at("detail") ==
                "non-standard exception after the irrevocable boundary");
            CHECK(record.at("phase") == "v_scatter");
            CHECK(record.at("layer") == 0);
        });
}

void TestSynchronizeFailureTerminatesWithoutSubmissionFlag() {
    TempDirectory fatal_root("fastflowlm-phi4-policy-fatal");
    RecordingState state;
    flm::test::ResetFakeCorelib();
    auto api = ResolveRecordingCorelib(state);
    auto runtime = CorelibRuntime::Create(
        api,
        MakeRecords(fatal_root.path()),
        [&state](unsigned int code) {
            state.terminator_called = true;
            state.termination_code = code;
            throw TerminationIntercept{};
        });
    flm::corelib::StepSubmissionState submission;
    CHECK(!submission.irrevocable());
    const CorelibError error(
        ryzenai_corelib_status_failure,
        "ryzenai_corelib_stream_synchronize",
        "injected policy synchronize failure",
        "failure");

    try {
        flm::phi4::testing::ApplyCorelibFailurePolicyForTest(
            runtime,
            error,
            true,
            submission,
            "qkv",
            5,
            4,
            17);
    } catch (const TerminationIntercept&) {
    }
    CHECK(state.terminator_called);
    CHECK(state.termination_code == 0xE0040001u);
    CHECK(runtime->state() == ProcessState::Terminating);
    g_state = nullptr;
}

static_assert(std::is_base_of_v<causal_lm, phi4_corelib_aie4>);
static_assert(std::has_virtual_destructor_v<phi4_corelib_aie4>);

}  // namespace

int main() {
    try {
        SyntheticPackage package;
        TestOrderBuffersTailsStateAndMetrics(package);
        TestRecoverablePreSubmitFailures(package);
        TestIrrevocableFailurePolicies(package);
        TestSynchronizeFailureTerminatesWithoutSubmissionFlag();
        std::cout << "test_phi4_engine: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "unexpected non-standard exception\n";
        return 1;
    }
}
