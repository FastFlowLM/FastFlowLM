#pragma once

#include <ryzenai/corelib.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flm::corelib {

// The loaded library's version, and the version this build compiled
// against. Corelib is pre-1.0, and the header states that below 1.0 the
// API may change in any release, so the patch component is load-bearing.
struct CorelibVersion final {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;
};

// API-5. While the compiled-against major is 0 all three components must
// match exactly. From corelib 1.0 the rule relaxes to major equality with
// the runtime minor at least the compiled minor.
bool IsCorelibVersionCompatible(
    const CorelibVersion& compiled,
    const CorelibVersion& runtime) noexcept;

std::string FormatCorelibVersion(const CorelibVersion& value);

std::string FormatCorelibVersionMismatch(
    const CorelibVersion& compiled,
    const CorelibVersion& runtime);

CorelibVersion CompiledCorelibVersion() noexcept;

struct CorelibError final : std::runtime_error {
    CorelibError(
        ryzenai_corelib_status status,
        std::string call,
        std::string detail,
        std::string status_text);

    CorelibError WithContext(std::string_view context) const;

    ryzenai_corelib_status status;
    std::string call;
    std::string detail;

private:
    std::string status_text_;
};

struct CorelibFunctions {
    // Resolved and called first: the version gate runs before any other
    // symbol is looked up, so a mismatched runtime is reported instead of
    // producing a confusing "missing symbol" for a renamed entry point.
    decltype(&::ryzenai_corelib_get_version) get_version;
    decltype(&::ryzenai_corelib_status_to_string) status_to_string;
    decltype(&::ryzenai_corelib_get_last_error_message)
        get_last_error_message;
    decltype(&::ryzenai_corelib_selftest_dependencies)
        selftest_dependencies;
    decltype(&::ryzenai_corelib_has_device_context) has_device_context;
    decltype(&::ryzenai_corelib_object_release) object_release;
    decltype(&::ryzenai_corelib_create_stream) create_stream;
    decltype(&::ryzenai_corelib_stream_synchronize) stream_synchronize;
    decltype(&::ryzenai_corelib_create_device_tensor) create_device_tensor;
    decltype(&::ryzenai_corelib_tensor_write) tensor_write;
    decltype(&::ryzenai_corelib_tensor_read) tensor_read;
    decltype(&::ryzenai_corelib_tensor_get_byte_size) tensor_get_byte_size;
    decltype(&::ryzenai_corelib_tensor_get_data_type) tensor_get_data_type;
    decltype(&::ryzenai_corelib_matmul_bf16_pad_shape) matmul_pad_shape;
    decltype(&::ryzenai_corelib_matmul_bf16_weights_create_onnx)
        matmul_weights_from_onnx;
    decltype(&::ryzenai_corelib_matmul_bf16_weights_get_data)
        matmul_weights_get_data;
    decltype(&::ryzenai_corelib_matmul_bf16) matmul;
    decltype(&::ryzenai_corelib_ssmlp_bf16_pad_rows) ssmlp_pad_rows;
    decltype(&::ryzenai_corelib_ssmlp_bf16_weights_create_onnx)
        ssmlp_weights_from_onnx;
    decltype(&::ryzenai_corelib_ssmlp_bf16_weights_get_data)
        ssmlp_weights_get_data;
    decltype(&::ryzenai_corelib_ssmlp_bf16) ssmlp;
    decltype(&::ryzenai_corelib_flat_mha_bf16_pad_rows) flat_mha_pad_rows;
    decltype(&::ryzenai_corelib_flat_mha_bf16) flat_mha;
    decltype(&::ryzenai_corelib_cleanup) cleanup;
};

// Which kind of corelib object a successful creation call produced.
//
// Task 13 Step 4 needs "no device tensor and no weight object was created
// after warmup" to be a MEASUREMENT rather than a restatement of what the
// code is believed to do. Counting at the RAII wrapper -- the single point
// every corelib object passes through on its way to being owned -- makes the
// answer independent of WHICH code path created it, so an allocation
// introduced anywhere in the decode loop is caught. A counter maintained by
// the engine's own tensor helper can only ever confirm that the helper was
// not called.
//
// Before this existed, `Phi4Aie4Metrics::weight_create_count` was assigned
// the constant `kLayerCount * 5 + 1`. That number is correct, and a constant
// cannot detect a weight object being created after warmup, which is the one
// question the field is read for.
enum class CorelibObjectKind : std::size_t {
    Stream = 0,
    Tensor = 1,
    MatMulWeights = 2,
    SsMlpWeights = 3,
};

inline constexpr std::size_t kCorelibObjectKindCount = 4;

class CorelibApi final {
public:
    using Resolver = std::function<void*(std::string_view)>;

    static std::shared_ptr<CorelibApi> Load(
        const std::filesystem::path& absolute_path);
    static std::shared_ptr<CorelibApi> ResolveForTest(Resolver resolver);
    static std::filesystem::path ResolveLibraryPath(
        const std::filesystem::path& executable_dir);

    ~CorelibApi();

    CorelibApi(const CorelibApi&) = delete;
    CorelibApi& operator=(const CorelibApi&) = delete;
    CorelibApi(CorelibApi&&) = delete;
    CorelibApi& operator=(CorelibApi&&) = delete;

    const CorelibFunctions& functions() const noexcept;
    const CorelibVersion& runtime_version() const noexcept;
    void Check(
        ryzenai_corelib_status status,
        std::string_view call) const;

    // API-7. `count` and `offset` are ELEMENTS of the tensor's own dtype,
    // never bytes. These are the only spellings FastFlow uses; there is
    // deliberately no byte-taking overload, because the byte and element
    // counts differ by 2x when writing FP32 into a BF16 tensor and the
    // wrong one would half-fill or overrun the tensor instead of failing.
    void WriteElements(
        ryzenai_corelib_tensor_ptr tensor,
        ryzenai_corelib_data_type source_type,
        const void* source,
        std::size_t count,
        std::size_t offset) const;

    void ReadElements(
        ryzenai_corelib_tensor_ptr tensor,
        ryzenai_corelib_data_type destination_type,
        void* destination,
        std::size_t count,
        std::size_t offset) const;

    void RegisterObject(CorelibObjectKind kind) const noexcept;
    void Release(void* value) const noexcept;
    std::size_t live_object_count() const noexcept;

    // Cumulative successful creations of `kind` for the process lifetime.
    // Never decremented: this answers "was anything created between these two
    // points", which a live count cannot, because a create-and-release pair
    // leaves the live count where it started.
    std::uint64_t creation_count(CorelibObjectKind kind) const noexcept;

    // MatMul plus SSMLP weight objects. Both kinds are "a weight object" for
    // the purposes of design 18.7's post-warm allocation property, and a
    // caller that summed only one of them would miss half of the model.
    std::uint64_t weight_creation_count() const noexcept;

    const std::filesystem::path& library_path() const noexcept;

private:
    CorelibApi(
        void* module,
        std::filesystem::path library_path,
        CorelibFunctions functions,
        CorelibVersion runtime_version);

    void* module_ = nullptr;
    std::filesystem::path library_path_;
    CorelibFunctions functions_;
    CorelibVersion runtime_version_;
    mutable std::atomic<std::size_t> live_object_count_{0};
    mutable std::array<
        std::atomic<std::uint64_t>,
        kCorelibObjectKindCount>
        creation_counts_{};
};

// The header's "one thread" hint for the ONNX packing entry points.
// Design Section 19 defers concurrent packing; FastFlow does not adopt it.
inline constexpr std::uint32_t kPackingThreads = 0;

}  // namespace flm::corelib
