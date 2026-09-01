#include "fake_corelib.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

namespace flm::test {
namespace {

thread_local std::string g_last_error;
std::unordered_map<void*, std::size_t> g_release_counts;
std::size_t g_release_count = 0;
void* g_last_released_object = nullptr;

const char* FakeStatusToString(ryzenai_corelib_status status) {
    g_last_error = "overwritten by status_to_string";
    switch (status) {
        case ryzenai_corelib_status_success:
            return "success";
        case ryzenai_corelib_status_failure:
            return "failure";
        case ryzenai_corelib_status_bad_argument:
            return "bad argument";
        case ryzenai_corelib_status_unsupported:
            return "unsupported";
    }
    return "unknown";
}

const char* FakeGetLastErrorMessage() {
    return g_last_error.c_str();
}

ryzenai_corelib_status FakeSelftestDependencies() {
    return ryzenai_corelib_status_success;
}

bool FakeHasDeviceContext() {
    return true;
}

void FakeObjectRelease(ryzenai_corelib_object_ptr object) {
    ++g_release_count;
    ++g_release_counts[object];
    g_last_released_object = object;
}

ryzenai_corelib_status FakeCreateStream(ryzenai_corelib_stream_ptr* out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeStreamSynchronize(
    ryzenai_corelib_stream_ptr) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeCreateDeviceTensor(
    ryzenai_corelib_data_type,
    const int64_t*,
    std::size_t,
    ryzenai_corelib_tensor_ptr* out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeTensorWrite(
    ryzenai_corelib_tensor_ptr,
    const void*,
    std::size_t,
    std::size_t) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeTensorRead(
    ryzenai_corelib_tensor_ptr,
    void*,
    std::size_t,
    std::size_t) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeTensorGetByteSize(
    ryzenai_corelib_tensor_ptr,
    std::size_t* out) {
    if (out != nullptr) {
        *out = 0;
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeConvert(
    ryzenai_corelib_data_type,
    const void*,
    ryzenai_corelib_data_type,
    void*,
    std::size_t) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeConvertStrided(
    ryzenai_corelib_data_type,
    const void*,
    std::size_t,
    ryzenai_corelib_data_type,
    void*,
    std::size_t,
    std::size_t,
    std::size_t) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeMatmulPadShape(
    int64_t*,
    int64_t*,
    int64_t*,
    uint32_t) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeMatmulWeightsFromOnnx(
    const ryzenai_corelib_matmul_bf16_weights_desc*,
    const ryzenai_corelib_matmul_bf16_onnx_weights_components*,
    ryzenai_corelib_matmul_bf16_weights_ptr* out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeMatmulWeightsGetData(
    ryzenai_corelib_matmul_bf16_weights_ptr,
    const void** data,
    std::size_t* size) {
    if (data != nullptr) {
        *data = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeMatmul(
    ryzenai_corelib_stream_ptr,
    ryzenai_corelib_tensor_ptr,
    int64_t,
    ryzenai_corelib_matmul_bf16_weights_ptr,
    ryzenai_corelib_tensor_ptr) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeSsmlpPadRows(
    int64_t*,
    int64_t,
    int64_t,
    uint32_t) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeSsmlpWeightsFromOnnx(
    const ryzenai_corelib_ssmlp_bf16_weights_desc*,
    const ryzenai_corelib_ssmlp_bf16_onnx_weights_components*,
    ryzenai_corelib_ssmlp_bf16_weights_ptr* out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeSsmlpWeightsGetData(
    ryzenai_corelib_ssmlp_bf16_weights_ptr,
    const void** data,
    std::size_t* size) {
    if (data != nullptr) {
        *data = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeSsmlp(
    ryzenai_corelib_stream_ptr,
    ryzenai_corelib_tensor_ptr,
    ryzenai_corelib_tensor_ptr,
    int64_t,
    ryzenai_corelib_ssmlp_bf16_weights_ptr,
    ryzenai_corelib_tensor_ptr,
    ryzenai_corelib_tensor_ptr) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeFlatMhaPadRows(
    int64_t*,
    const ryzenai_corelib_flat_mha_bf16_desc*) {
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status FakeFlatMha(
    ryzenai_corelib_stream_ptr,
    const ryzenai_corelib_flat_mha_bf16_desc*,
    ryzenai_corelib_tensor_ptr,
    ryzenai_corelib_tensor_ptr,
    int64_t,
    int64_t,
    ryzenai_corelib_tensor_ptr,
    ryzenai_corelib_tensor_ptr,
    ryzenai_corelib_tensor_ptr,
    ryzenai_corelib_tensor_ptr,
    ryzenai_corelib_tensor_ptr) {
    return ryzenai_corelib_status_success;
}

void FakeCleanup() {}

template <class Function>
void* FunctionAddress(Function function) {
    return reinterpret_cast<void*>(function);
}

#define FLM_FAKE_ENTRY(symbol, replacement)                              \
    {                                                                    \
        #symbol, FunctionAddress(                                        \
                     static_cast<decltype(&::symbol)>(&replacement))     \
    }

}  // namespace

std::unordered_map<std::string, void*> CompleteCorelibResolver() {
    return {
        FLM_FAKE_ENTRY(
            ryzenai_corelib_status_to_string,
            FakeStatusToString),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_get_last_error_message,
            FakeGetLastErrorMessage),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_selftest_dependencies,
            FakeSelftestDependencies),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_has_device_context,
            FakeHasDeviceContext),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_object_release,
            FakeObjectRelease),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_create_stream,
            FakeCreateStream),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_stream_synchronize,
            FakeStreamSynchronize),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_create_device_tensor,
            FakeCreateDeviceTensor),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_tensor_write,
            FakeTensorWrite),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_tensor_read,
            FakeTensorRead),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_tensor_get_byte_size,
            FakeTensorGetByteSize),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_convert,
            FakeConvert),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_convert_strided,
            FakeConvertStrided),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_matmul_bf16_pad_shape,
            FakeMatmulPadShape),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components,
            FakeMatmulWeightsFromOnnx),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_matmul_bf16_weights_get_data,
            FakeMatmulWeightsGetData),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_matmul_bf16,
            FakeMatmul),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_ssmlp_bf16_pad_rows,
            FakeSsmlpPadRows),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components,
            FakeSsmlpWeightsFromOnnx),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_ssmlp_bf16_weights_get_data,
            FakeSsmlpWeightsGetData),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_ssmlp_bf16,
            FakeSsmlp),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_flat_mha_bf16_pad_rows,
            FakeFlatMhaPadRows),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_flat_mha_bf16,
            FakeFlatMha),
        FLM_FAKE_ENTRY(
            ryzenai_corelib_cleanup,
            FakeCleanup),
    };
}

#undef FLM_FAKE_ENTRY

void ResetFakeCorelib() {
    g_last_error.clear();
    g_release_counts.clear();
    g_release_count = 0;
    g_last_released_object = nullptr;
}

void SetLastErrorMessage(std::string message) {
    g_last_error = std::move(message);
}

std::size_t ObjectReleaseCount() noexcept {
    return g_release_count;
}

std::size_t ObjectReleaseCountFor(void* value) noexcept {
    const auto found = g_release_counts.find(value);
    return found == g_release_counts.end() ? 0 : found->second;
}

void* LastReleasedObject() noexcept {
    return g_last_released_object;
}

}  // namespace flm::test
