#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <models/phi4/phi4_corelib_constants.hpp>
#include <models/phi4/phi4_corelib_manifest.hpp>
#include <models/phi4/phi4_corelib_weights.hpp>
#include <nlohmann/json.hpp>

#include <windows.h>
#include <winioctl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
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
using flm::phi4::Phi4Package;
using flm::phi4::Phi4Weights;
using flm::phi4::WeightObjectKind;
using nlohmann::json;

namespace constants = flm::phi4::constants;

constexpr std::string_view kManifestName =
    "corelib_phi4_manifest.json";
constexpr std::string_view kDataFile = "weights.bin";
constexpr std::uint64_t kDataBytes =
    200064ull * 3072ull * sizeof(std::uint16_t);
constexpr std::size_t kMatMulPackedBytes = 17;
constexpr std::size_t kSsMlpPackedBytes = 29;
constexpr std::uint16_t kBf16One = 0x3f80u;
constexpr std::uint16_t kBf16Epsilon = 0x3728u;

enum class FailurePoint {
    None,
    MatMulCreate,
    MatMulGetData,
    SsMlpCreate,
    SsMlpGetData
};

struct FakeWeightHandle {
    std::weak_ptr<Phi4Package> package;
};

struct MatMulCreateRecord {
    ryzenai_corelib_matmul_bf16_weights_desc desc{};
    ryzenai_corelib_matmul_bf16_onnx_weights_components components{};
    void* object = nullptr;
    std::thread::id thread;
};

struct SsMlpCreateRecord {
    ryzenai_corelib_ssmlp_bf16_weights_desc desc{};
    ryzenai_corelib_ssmlp_bf16_onnx_weights_components components{};
    void* object = nullptr;
    std::thread::id thread;
};

struct ConvertRecord {
    ryzenai_corelib_data_type source_type =
        ryzenai_corelib_data_type_fp32;
    ryzenai_corelib_data_type destination_type =
        ryzenai_corelib_data_type_fp32;
    const void* source = nullptr;
    void* destination = nullptr;
    std::size_t count = 0;
    float first_source_value = 0.0f;
    std::thread::id thread;
};

struct RecordingState {
    std::mutex mutex;
    std::weak_ptr<Phi4Package> current_package;
    FailurePoint failure = FailurePoint::None;
    std::size_t failure_ordinal = 1;
    std::size_t matmul_create_attempts = 0;
    std::size_t matmul_get_attempts = 0;
    std::size_t ssmlp_create_attempts = 0;
    std::size_t ssmlp_get_attempts = 0;
    std::vector<MatMulCreateRecord> matmul_creates;
    std::vector<SsMlpCreateRecord> ssmlp_creates;
    std::vector<ConvertRecord> converts;
    std::vector<void*> creation_order;
    std::vector<void*> release_order;
    std::vector<bool> package_alive_at_release;
    std::vector<std::string> events;
    std::size_t get_data_calls = 0;
    bool every_get_data_pointer_argument_was_null = true;
    bool every_get_data_size_argument_was_nonnull = true;
};

RecordingState* g_recording = nullptr;

RecordingState& State() {
    if (g_recording == nullptr) {
        throw std::runtime_error("recording corelib is not active");
    }
    return *g_recording;
}

bool ShouldFail(
    FailurePoint configured,
    FailurePoint current,
    std::size_t ordinal) {
    return configured == current &&
           ordinal == State().failure_ordinal;
}

ryzenai_corelib_status RecordingConvert(
    ryzenai_corelib_data_type source_type,
    const void* source,
    ryzenai_corelib_data_type destination_type,
    void* destination,
    std::size_t count) {
    if (source == nullptr || destination == nullptr || count == 0) {
        return ryzenai_corelib_status_bad_argument;
    }

    ConvertRecord record{
        source_type,
        destination_type,
        source,
        destination,
        count,
        0.0f,
        std::this_thread::get_id()};
    if (source_type == ryzenai_corelib_data_type_fp32) {
        record.first_source_value =
            *static_cast<const float*>(source);
    }

    auto& state = State();
    {
        std::lock_guard lock(state.mutex);
        state.converts.push_back(record);
    }

    auto* output = static_cast<std::uint16_t*>(destination);
    if (
        destination_type == ryzenai_corelib_data_type_bf16 &&
        source_type == ryzenai_corelib_data_type_fp32 &&
        count == 1 &&
        std::abs(record.first_source_value - 1.0e-5f) < 1.0e-10f) {
        output[0] = kBf16Epsilon;
    } else if (
        destination_type == ryzenai_corelib_data_type_fp16 ||
        destination_type == ryzenai_corelib_data_type_bf16) {
        output[0] = kBf16One;
    } else {
        return ryzenai_corelib_status_bad_argument;
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingMatMulCreate(
    const ryzenai_corelib_matmul_bf16_weights_desc* desc,
    const ryzenai_corelib_matmul_bf16_onnx_weights_components* components,
    ryzenai_corelib_matmul_bf16_weights_ptr* out) {
    if (desc == nullptr || components == nullptr || out == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    *out = nullptr;

    auto& state = State();
    std::lock_guard lock(state.mutex);
    const std::size_t ordinal = ++state.matmul_create_attempts;
    state.events.emplace_back("matmul_create");
    if (ShouldFail(
            state.failure,
            FailurePoint::MatMulCreate,
            ordinal)) {
        flm::test::SetLastErrorMessage(
            "intentional Task 6 MatMul create failure");
        return ryzenai_corelib_status_unsupported;
    }

    auto* handle = new FakeWeightHandle{state.current_package};
    *out = handle;
    state.matmul_creates.push_back(
        {*desc, *components, handle, std::this_thread::get_id()});
    state.creation_order.push_back(handle);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingMatMulGetData(
    ryzenai_corelib_matmul_bf16_weights_ptr weights,
    const void** data,
    std::size_t* size) {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    const std::size_t ordinal = ++state.matmul_get_attempts;
    state.events.emplace_back("matmul_get_data");
    ++state.get_data_calls;
    state.every_get_data_pointer_argument_was_null =
        state.every_get_data_pointer_argument_was_null &&
        data == nullptr;
    state.every_get_data_size_argument_was_nonnull =
        state.every_get_data_size_argument_was_nonnull &&
        size != nullptr;
    if (
        weights == nullptr || data != nullptr || size == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    if (ShouldFail(
            state.failure,
            FailurePoint::MatMulGetData,
            ordinal)) {
        flm::test::SetLastErrorMessage(
            "intentional Task 6 MatMul get-data failure");
        return ryzenai_corelib_status_unsupported;
    }
    *size = kMatMulPackedBytes;
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingSsMlpCreate(
    const ryzenai_corelib_ssmlp_bf16_weights_desc* desc,
    const ryzenai_corelib_ssmlp_bf16_onnx_weights_components* components,
    ryzenai_corelib_ssmlp_bf16_weights_ptr* out) {
    if (desc == nullptr || components == nullptr || out == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    *out = nullptr;

    auto& state = State();
    std::lock_guard lock(state.mutex);
    const std::size_t ordinal = ++state.ssmlp_create_attempts;
    state.events.emplace_back("ssmlp_create");
    if (ShouldFail(
            state.failure,
            FailurePoint::SsMlpCreate,
            ordinal)) {
        flm::test::SetLastErrorMessage(
            "intentional Task 6 SSMLP create failure");
        return ryzenai_corelib_status_unsupported;
    }

    auto* handle = new FakeWeightHandle{state.current_package};
    *out = handle;
    state.ssmlp_creates.push_back(
        {*desc, *components, handle, std::this_thread::get_id()});
    state.creation_order.push_back(handle);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status RecordingSsMlpGetData(
    ryzenai_corelib_ssmlp_bf16_weights_ptr weights,
    const void** data,
    std::size_t* size) {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    const std::size_t ordinal = ++state.ssmlp_get_attempts;
    state.events.emplace_back("ssmlp_get_data");
    ++state.get_data_calls;
    state.every_get_data_pointer_argument_was_null =
        state.every_get_data_pointer_argument_was_null &&
        data == nullptr;
    state.every_get_data_size_argument_was_nonnull =
        state.every_get_data_size_argument_was_nonnull &&
        size != nullptr;
    if (
        weights == nullptr || data != nullptr || size == nullptr) {
        return ryzenai_corelib_status_bad_argument;
    }
    if (ShouldFail(
            state.failure,
            FailurePoint::SsMlpGetData,
            ordinal)) {
        flm::test::SetLastErrorMessage(
            "intentional Task 6 SSMLP get-data failure");
        return ryzenai_corelib_status_unsupported;
    }
    *size = kSsMlpPackedBytes;
    return ryzenai_corelib_status_success;
}

void RecordingRelease(ryzenai_corelib_object_ptr object) {
    auto* handle = static_cast<FakeWeightHandle*>(object);
    auto& state = State();
    {
        std::lock_guard lock(state.mutex);
        state.release_order.push_back(object);
        state.package_alive_at_release.push_back(
            !handle->package.expired());
        state.events.emplace_back("release");
    }
    delete handle;
}

template <class Function>
void* FunctionAddress(Function function) {
    return reinterpret_cast<void*>(function);
}

std::shared_ptr<CorelibApi> ResolveRecordingCorelib(
    RecordingState& state) {
    g_recording = &state;
    auto resolver = flm::test::CompleteCorelibResolver();
    resolver["ryzenai_corelib_object_release"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_object_release)>(
            &RecordingRelease));
    resolver["ryzenai_corelib_convert"] = FunctionAddress(
        static_cast<decltype(&::ryzenai_corelib_convert)>(
            &RecordingConvert));
    resolver
        ["ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components"] =
            FunctionAddress(
                static_cast<decltype(
                    &::ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components)>(
                    &RecordingMatMulCreate));
    resolver["ryzenai_corelib_matmul_bf16_weights_get_data"] =
        FunctionAddress(
            static_cast<decltype(
                &::ryzenai_corelib_matmul_bf16_weights_get_data)>(
                &RecordingMatMulGetData));
    resolver
        ["ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components"] =
            FunctionAddress(
                static_cast<decltype(
                    &::ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components)>(
                    &RecordingSsMlpCreate));
    resolver["ryzenai_corelib_ssmlp_bf16_weights_get_data"] =
        FunctionAddress(
            static_cast<decltype(
                &::ryzenai_corelib_ssmlp_bf16_weights_get_data)>(
                &RecordingSsMlpGetData));
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
                ("fastflowlm-phi4-weights-" +
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
        throw std::runtime_error("failed to create sparse weight file");
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
            "test volume does not support sparse files");
    }

    LARGE_INTEGER end{};
    end.QuadPart = static_cast<LONGLONG>(size);
    const bool success =
        SetFilePointerEx(file, end, nullptr, FILE_BEGIN) != FALSE &&
        SetEndOfFile(file) != FALSE;
    CloseHandle(file);
    if (!success) {
        throw std::runtime_error("failed to size sparse weight file");
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
        std::int64_t n,
        bool opaque_qweight = false) {
        const std::string qweight =
            opaque_qweight ? "opaque-qweight-from-explicit-role"
                           : name + ".qweight";
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
            name == "model.layers.0.attn.q_proj.MatMulNBits"
                ? "float32"
                : "float16",
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
            layer == 0 ? "float32" : "float16",
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
                3072,
                layer == 0);
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
        AddMatMul(
            "lm_head.MatMulNBits",
            3072,
            200064);

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
            "float32",
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
    SyntheticPackage() {
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
                "failed to write synthetic weight manifest");
        }
    }

    std::shared_ptr<Phi4Package> Load(
        const std::shared_ptr<CorelibApi>& api) const {
        return std::make_shared<Phi4Package>(
            Phi4Package::Load(temp_.path(), api, false));
    }

private:
    TempDirectory temp_;
};

const std::string& Role(
    const flm::phi4::WeightObjectView& object,
    std::string_view role) {
    return object.components.at(std::string(role));
}

void CheckMatMulComponents(
    const MatMulCreateRecord& actual,
    const flm::phi4::WeightObjectView& expected,
    Phi4Package& package) {
    CHECK(expected.kind == WeightObjectKind::MatMul);
    CHECK(actual.desc.k == expected.k);
    CHECK(actual.desc.n == expected.n);
    CHECK(actual.desc.group_size == 128);
    CHECK(actual.desc.has_bias == false);

    const auto& qweight = package.Require(Role(expected, "qweight"));
    const auto scales =
        package.MaterializeFp16(Role(expected, "scales"));
    const auto& qzeros = package.Require(Role(expected, "qzeros"));
    CHECK(actual.components.qweight == qweight.data);
    CHECK(actual.components.scales == scales.data());
    CHECK(actual.components.qzeros == qzeros.data);
    CHECK(actual.components.scales !=
          package.Require(Role(expected, "scales")).data);
    CHECK(
        *static_cast<const std::byte*>(actual.components.qweight) ==
        std::byte{0});
    CHECK(
        *static_cast<const std::byte*>(actual.components.qzeros) ==
        std::byte{0});
    CHECK(
        *static_cast<const std::uint16_t*>(actual.components.scales) ==
        kBf16One);
}

void CheckSsMlpComponents(
    const SsMlpCreateRecord& actual,
    const flm::phi4::WeightObjectView& expected,
    Phi4Package& package) {
    CHECK(expected.kind == WeightObjectKind::SsMlp);
    CHECK(actual.desc.k == 3072);
    CHECK(actual.desc.n == 8192);
    CHECK(actual.desc.group_size == 128);
    CHECK(actual.components.epsilon != nullptr);
    CHECK(
        *static_cast<const std::uint16_t*>(
            actual.components.epsilon) == kBf16Epsilon);

    const auto check_norm = [&](
                                const void* actual_pointer,
                                std::string_view role) {
        const std::string& name = Role(expected, role);
        const auto value = package.MaterializeBf16(name);
        CHECK(actual_pointer == value.data());
        CHECK(actual_pointer != package.Require(name).data);
        CHECK(
            *static_cast<const std::uint16_t*>(actual_pointer) ==
            kBf16One);
    };
    const auto check_projection = [&](
                                      const void* qweight,
                                      const void* scales,
                                      const void* qzeros,
                                      std::string_view prefix) {
        const std::string qweight_role =
            std::string(prefix) + "_qweight";
        const std::string scales_role =
            std::string(prefix) + "_scales";
        const std::string qzeros_role =
            std::string(prefix) + "_qzeros";
        CHECK(qweight ==
              package.Require(Role(expected, qweight_role)).data);
        CHECK(scales ==
              package.MaterializeFp16(
                         Role(expected, scales_role))
                  .data());
        CHECK(qzeros ==
              package.Require(Role(expected, qzeros_role)).data);
        CHECK(scales !=
              package.Require(Role(expected, scales_role)).data);
    };

    check_norm(actual.components.norm0, "norm0");
    check_norm(actual.components.norm1, "norm1");
    check_projection(
        actual.components.gate_qweight,
        actual.components.gate_scales,
        actual.components.gate_qzeros,
        "gate");
    check_projection(
        actual.components.up_qweight,
        actual.components.up_scales,
        actual.components.up_qzeros,
        "up");
    check_projection(
        actual.components.down_qweight,
        actual.components.down_scales,
        actual.components.down_qzeros,
        "down");
}

void CheckImmediateSizeQueries(const RecordingState& state) {
    CHECK(state.events.size() == 322);
    for (std::size_t index = 0; index < state.events.size(); index += 2) {
        const bool matmul =
            state.events[index] == "matmul_create";
        CHECK(matmul || state.events[index] == "ssmlp_create");
        CHECK(
            state.events[index + 1] ==
            (matmul ? "matmul_get_data" : "ssmlp_get_data"));
    }
}

void CheckReleaseOrder(
    const std::vector<void*>& creation_order,
    const std::vector<void*>& release_order) {
    CHECK(creation_order.size() == release_order.size());
    CHECK(std::equal(
        release_order.begin(),
        release_order.end(),
        creation_order.rbegin(),
        creation_order.rend()));
}

void TestExactConstructionAndLifetime(
    const SyntheticPackage& fixture) {
    RecordingState state;
    flm::test::ResetFakeCorelib();
    auto api = ResolveRecordingCorelib(state);
    auto package = fixture.Load(api);
    state.current_package = package;
    const std::weak_ptr<Phi4Package> package_lifetime = package;
    const std::thread::id load_thread = std::this_thread::get_id();

    {
        Phi4Weights weights = Phi4Weights::Load(api, package);
        CHECK(weights.layers().size() == 32);
        CHECK(state.matmul_creates.size() == 32u * 4u + 1u);
        CHECK(state.ssmlp_creates.size() == 32u);
        CHECK(state.get_data_calls == 161u);
        CHECK(state.every_get_data_pointer_argument_was_null);
        CHECK(state.every_get_data_size_argument_was_nonnull);
        CHECK(weights.packed_bytes() ==
              129u * kMatMulPackedBytes +
                  32u * kSsMlpPackedBytes);
        CHECK(api->live_object_count() == 161);
        CheckImmediateSizeQueries(state);

        const auto& objects = package->weight_objects();
        CHECK(objects.size() == 161);
        CHECK(Role(objects.front(), "qweight") ==
              "opaque-qweight-from-explicit-role");
        for (std::size_t layer = 0; layer < 32; ++layer) {
            const std::size_t object_base = layer * 5;
            const std::size_t matmul_base = layer * 4;
            for (std::size_t projection = 0;
                 projection < 4;
                 ++projection) {
                CheckMatMulComponents(
                    state.matmul_creates.at(
                        matmul_base + projection),
                    objects.at(object_base + projection),
                    *package);
            }
            CheckSsMlpComponents(
                state.ssmlp_creates.at(layer),
                objects.at(object_base + 4),
                *package);
            CHECK(weights.layers()[layer].q);
            CHECK(weights.layers()[layer].k);
            CHECK(weights.layers()[layer].v);
            CHECK(weights.layers()[layer].o);
            CHECK(weights.layers()[layer].mlp);
        }
        CheckMatMulComponents(
            state.matmul_creates.back(),
            objects.back(),
            *package);
        CHECK(weights.lm_head());
        CHECK(state.matmul_creates.front().desc.has_bias == false);
        CHECK(state.matmul_creates.back().desc.n == 200064);
        CHECK(
            Role(objects.at(31u * 5u + 4u), "norm1") ==
            "model.layers.32.final_norm_layernorm.weight");
        CHECK(
            state.ssmlp_creates.back().components.norm1 ==
            package
                ->MaterializeBf16(
                    "model.layers.32.final_norm_layernorm.weight")
                .data());

        CHECK(state.converts.size() == 290);
        CHECK(std::count_if(
                  state.converts.begin(),
                  state.converts.end(),
                  [](const ConvertRecord& call) {
                      return call.destination_type ==
                             ryzenai_corelib_data_type_fp16;
                  }) == 225);
        CHECK(std::count_if(
                  state.converts.begin(),
                  state.converts.end(),
                  [](const ConvertRecord& call) {
                      return call.destination_type ==
                             ryzenai_corelib_data_type_bf16;
                  }) == 65);
        CHECK(std::all_of(
            state.converts.begin(),
            state.converts.end(),
            [load_thread](const ConvertRecord& call) {
                return call.thread == load_thread;
            }));
        CHECK(std::all_of(
            state.matmul_creates.begin(),
            state.matmul_creates.end(),
            [load_thread](const MatMulCreateRecord& call) {
                return call.thread == load_thread;
            }));
        CHECK(std::all_of(
            state.ssmlp_creates.begin(),
            state.ssmlp_creates.end(),
            [load_thread](const SsMlpCreateRecord& call) {
                return call.thread == load_thread;
            }));

        const std::vector<void*> creation_order =
            state.creation_order;
        Phi4Weights moved(std::move(weights));
        CHECK(state.release_order.empty());
        CHECK(api->live_object_count() == 161);
        package.reset();
        CHECK(!package_lifetime.expired());

        auto retained = package_lifetime.lock();
        CHECK(retained != nullptr);
        const auto& retained_objects = retained->weight_objects();
        for (std::size_t layer = 0; layer < 32; ++layer) {
            for (std::size_t projection = 0;
                 projection < 4;
                 ++projection) {
                CheckMatMulComponents(
                    state.matmul_creates.at(
                        layer * 4 + projection),
                    retained_objects.at(
                        layer * 5 + projection),
                    *retained);
            }
            CheckSsMlpComponents(
                state.ssmlp_creates.at(layer),
                retained_objects.at(layer * 5 + 4),
                *retained);
        }
        CheckMatMulComponents(
            state.matmul_creates.back(),
            retained_objects.back(),
            *retained);
        retained.reset();
        CHECK(!package_lifetime.expired());
        CHECK(state.converts.size() == 290);
        (void)moved;
        CHECK(creation_order.size() == 161);
    }

    CHECK(state.release_order.size() == 161);
    CHECK(std::all_of(
        state.package_alive_at_release.begin(),
        state.package_alive_at_release.end(),
        [](bool value) { return value; }));
    CheckReleaseOrder(state.creation_order, state.release_order);
    CHECK(package_lifetime.expired());
    CHECK(api->live_object_count() == 0);
    g_recording = nullptr;
}

void TestMoveAssignmentReleasesBeforeOwners(
    const SyntheticPackage& fixture) {
    RecordingState state;
    flm::test::ResetFakeCorelib();
    auto api = ResolveRecordingCorelib(state);
    auto source_package = fixture.Load(api);
    auto destination_package = fixture.Load(api);
    const std::weak_ptr<Phi4Package> source_lifetime = source_package;
    const std::weak_ptr<Phi4Package> destination_lifetime =
        destination_package;

    {
        state.current_package = source_package;
        Phi4Weights source =
            Phi4Weights::Load(api, source_package);
        const std::vector<void*> source_objects =
            state.creation_order;

        state.current_package = destination_package;
        Phi4Weights destination =
            Phi4Weights::Load(api, destination_package);
        const std::vector<void*> destination_objects(
            state.creation_order.begin() + 161,
            state.creation_order.end());
        CHECK(source_objects.size() == 161);
        CHECK(destination_objects.size() == 161);
        CHECK(api->live_object_count() == 322);

        source_package.reset();
        destination_package.reset();
        destination = std::move(source);

        CHECK(state.release_order.size() == 161);
        CheckReleaseOrder(
            destination_objects,
            state.release_order);
        CHECK(destination_lifetime.expired());
        CHECK(!source_lifetime.expired());
        CHECK(api->live_object_count() == 161);
    }

    CHECK(state.release_order.size() == 322);
    const std::vector<void*> source_releases(
        state.release_order.begin() + 161,
        state.release_order.end());
    const std::vector<void*> source_objects(
        state.creation_order.begin(),
        state.creation_order.begin() + 161);
    CheckReleaseOrder(source_objects, source_releases);
    CHECK(std::all_of(
        state.package_alive_at_release.begin(),
        state.package_alive_at_release.end(),
        [](bool value) { return value; }));
    CHECK(source_lifetime.expired());
    CHECK(api->live_object_count() == 0);
    g_recording = nullptr;
}

void CheckLoadFailure(
    RecordingState& state,
    const std::shared_ptr<CorelibApi>& api,
    const std::shared_ptr<Phi4Package>& package,
    FailurePoint failure,
    std::string_view object_name,
    std::string_view call,
    std::string_view detail) {
    state.failure = failure;
    state.failure_ordinal = 1;
    state.matmul_create_attempts = 0;
    state.matmul_get_attempts = 0;
    state.ssmlp_create_attempts = 0;
    state.ssmlp_get_attempts = 0;
    state.matmul_creates.clear();
    state.ssmlp_creates.clear();
    state.converts.clear();
    state.creation_order.clear();
    state.release_order.clear();
    state.package_alive_at_release.clear();
    state.events.clear();
    state.get_data_calls = 0;
    state.every_get_data_pointer_argument_was_null = true;
    state.every_get_data_size_argument_was_nonnull = true;
    state.current_package = package;
    flm::test::SetLastErrorMessage({});

    try {
        (void)Phi4Weights::Load(api, package);
    } catch (const std::exception& error) {
        const std::string_view message(error.what());
        CHECK(message.find(object_name) != std::string_view::npos);
        CHECK(message.find(call) != std::string_view::npos);
        CHECK(message.find(detail) != std::string_view::npos);
        CHECK(api->live_object_count() == 0);
        CHECK(std::all_of(
            state.package_alive_at_release.begin(),
            state.package_alive_at_release.end(),
            [](bool value) { return value; }));
        return;
    }
    throw std::runtime_error(
        "expected Phi4Weights::Load failure was not thrown");
}

void TestActionableFailures(const SyntheticPackage& fixture) {
    RecordingState state;
    flm::test::ResetFakeCorelib();
    auto api = ResolveRecordingCorelib(state);
    auto package = fixture.Load(api);

    CheckThrowsContains(
        [&] {
            (void)Phi4Weights::Load(nullptr, package);
        },
        "CorelibApi");
    CheckThrowsContains(
        [&] {
            (void)Phi4Weights::Load(api, nullptr);
        },
        "Phi4Package");

    CheckLoadFailure(
        state,
        api,
        package,
        FailurePoint::MatMulCreate,
        "model.layers.0.attn.q_proj.MatMulNBits",
        "ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components",
        "intentional Task 6 MatMul create failure");
    CheckLoadFailure(
        state,
        api,
        package,
        FailurePoint::MatMulGetData,
        "model.layers.0.attn.q_proj.MatMulNBits",
        "ryzenai_corelib_matmul_bf16_weights_get_data",
        "intentional Task 6 MatMul get-data failure");
    CheckLoadFailure(
        state,
        api,
        package,
        FailurePoint::SsMlpCreate,
        "model.layers.0.ssmlp",
        "ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components",
        "intentional Task 6 SSMLP create failure");
    CheckLoadFailure(
        state,
        api,
        package,
        FailurePoint::SsMlpGetData,
        "model.layers.0.ssmlp",
        "ryzenai_corelib_ssmlp_bf16_weights_get_data",
        "intentional Task 6 SSMLP get-data failure");
    g_recording = nullptr;
}

static_assert(!std::is_copy_constructible_v<Phi4Weights>);
static_assert(!std::is_copy_assignable_v<Phi4Weights>);
static_assert(std::is_nothrow_move_constructible_v<Phi4Weights>);
static_assert(std::is_nothrow_move_assignable_v<Phi4Weights>);

}  // namespace

int main() {
    try {
        SyntheticPackage fixture;
        TestExactConstructionAndLifetime(fixture);
        TestMoveAssignmentReleasesBeforeOwners(fixture);
        TestActionableFailures(fixture);
        std::cout << "test_phi4_weights: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
