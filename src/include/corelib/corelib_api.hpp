#pragma once

#include <ryzenai/corelib.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flm::corelib {

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
    decltype(&::ryzenai_corelib_convert) convert;
    decltype(&::ryzenai_corelib_convert_strided) convert_strided;
    decltype(&::ryzenai_corelib_matmul_bf16_pad_shape) matmul_pad_shape;
    decltype(
        &::ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components)
        matmul_weights_from_onnx;
    decltype(&::ryzenai_corelib_matmul_bf16_weights_get_data)
        matmul_weights_get_data;
    decltype(&::ryzenai_corelib_matmul_bf16) matmul;
    decltype(&::ryzenai_corelib_ssmlp_bf16_pad_rows) ssmlp_pad_rows;
    decltype(
        &::ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components)
        ssmlp_weights_from_onnx;
    decltype(&::ryzenai_corelib_ssmlp_bf16_weights_get_data)
        ssmlp_weights_get_data;
    decltype(&::ryzenai_corelib_ssmlp_bf16) ssmlp;
    decltype(&::ryzenai_corelib_flat_mha_bf16_pad_rows) flat_mha_pad_rows;
    decltype(&::ryzenai_corelib_flat_mha_bf16) flat_mha;
    decltype(&::ryzenai_corelib_cleanup) cleanup;
};

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
    void Check(
        ryzenai_corelib_status status,
        std::string_view call) const;
    void RegisterObject() const noexcept;
    void Release(void* value) const noexcept;
    std::size_t live_object_count() const noexcept;
    const std::filesystem::path& library_path() const noexcept;

private:
    CorelibApi(
        void* module,
        std::filesystem::path library_path,
        CorelibFunctions functions);

    void* module_ = nullptr;
    std::filesystem::path library_path_;
    CorelibFunctions functions_;
    mutable std::atomic<std::size_t> live_object_count_{0};
};

}  // namespace flm::corelib
