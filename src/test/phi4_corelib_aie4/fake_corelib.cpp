#include "fake_corelib.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flm::test {
namespace detail {

thread_local std::string g_last_error;
std::unordered_map<void*, std::size_t> g_release_counts;
std::size_t g_release_count = 0;
void* g_last_released_object = nullptr;
ryzenai_corelib_status g_selftest_status =
    ryzenai_corelib_status_success;
bool g_has_device_context = true;
std::size_t g_cleanup_count = 0;
std::vector<std::string> g_events;

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
    return g_selftest_status;
}

bool FakeHasDeviceContext() {
    return g_has_device_context;
}

void FakeObjectRelease(ryzenai_corelib_object_ptr object) {
    ++g_release_count;
    ++g_release_counts[object];
    g_last_released_object = object;
    g_events.emplace_back("release");
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
        *size = 0x4D4D;
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
        *size = 0x5353;
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

void FakeCleanup() {
    ++g_cleanup_count;
    g_events.emplace_back("cleanup");

    std::size_t required = 0;
    if (_wgetenv_s(
            &required,
            nullptr,
            0,
            L"FLM_FAKE_CORELIB_CLEANUP_MARKER") != 0 ||
        required == 0) {
        return;
    }
    std::vector<wchar_t> marker_path(required);
    if (_wgetenv_s(
            &required,
            marker_path.data(),
            marker_path.size(),
            L"FLM_FAKE_CORELIB_CLEANUP_MARKER") != 0) {
        return;
    }
    std::ofstream marker(
        std::filesystem::path(marker_path.data()),
        std::ios::app | std::ios::binary);
    marker << "cleanup\n";
}

}  // namespace detail
}  // namespace flm::test

#if defined(FLM_FAKE_CORELIB_DLL)

const char* ryzenai_corelib_status_to_string(
    ryzenai_corelib_status status) {
    return flm::test::detail::FakeStatusToString(status);
}

const char* ryzenai_corelib_get_last_error_message() {
    return flm::test::detail::FakeGetLastErrorMessage();
}

ryzenai_corelib_status ryzenai_corelib_selftest_dependencies() {
    return flm::test::detail::FakeSelftestDependencies();
}

bool ryzenai_corelib_has_device_context() {
    return flm::test::detail::FakeHasDeviceContext();
}

void ryzenai_corelib_object_release(ryzenai_corelib_object_ptr object) {
    flm::test::detail::FakeObjectRelease(object);
}

ryzenai_corelib_status ryzenai_corelib_create_stream(
    ryzenai_corelib_stream_ptr* out) {
    return flm::test::detail::FakeCreateStream(out);
}

ryzenai_corelib_status ryzenai_corelib_stream_synchronize(
    ryzenai_corelib_stream_ptr stream) {
    return flm::test::detail::FakeStreamSynchronize(stream);
}

ryzenai_corelib_status ryzenai_corelib_create_device_tensor(
    ryzenai_corelib_data_type data_type,
    const int64_t* shape,
    std::size_t shape_len,
    ryzenai_corelib_tensor_ptr* out) {
    return flm::test::detail::FakeCreateDeviceTensor(
        data_type,
        shape,
        shape_len,
        out);
}

ryzenai_corelib_status ryzenai_corelib_tensor_write(
    ryzenai_corelib_tensor_ptr tensor,
    const void* source,
    std::size_t size,
    std::size_t offset) {
    return flm::test::detail::FakeTensorWrite(
        tensor,
        source,
        size,
        offset);
}

ryzenai_corelib_status ryzenai_corelib_tensor_read(
    ryzenai_corelib_tensor_ptr tensor,
    void* destination,
    std::size_t size,
    std::size_t offset) {
    return flm::test::detail::FakeTensorRead(
        tensor,
        destination,
        size,
        offset);
}

ryzenai_corelib_status ryzenai_corelib_tensor_get_byte_size(
    ryzenai_corelib_tensor_ptr tensor,
    std::size_t* out) {
    return flm::test::detail::FakeTensorGetByteSize(tensor, out);
}

ryzenai_corelib_status ryzenai_corelib_convert(
    ryzenai_corelib_data_type source_type,
    const void* source,
    ryzenai_corelib_data_type destination_type,
    void* destination,
    std::size_t count) {
    return flm::test::detail::FakeConvert(
        source_type,
        source,
        destination_type,
        destination,
        count);
}

ryzenai_corelib_status ryzenai_corelib_convert_strided(
    ryzenai_corelib_data_type source_type,
    const void* source,
    std::size_t source_stride,
    ryzenai_corelib_data_type destination_type,
    void* destination,
    std::size_t destination_stride,
    std::size_t count,
    std::size_t row) {
    return flm::test::detail::FakeConvertStrided(
        source_type,
        source,
        source_stride,
        destination_type,
        destination,
        destination_stride,
        count,
        row);
}

ryzenai_corelib_status ryzenai_corelib_matmul_bf16_pad_shape(
    int64_t* m,
    int64_t* k,
    int64_t* n,
    uint32_t group_size) {
    return flm::test::detail::FakeMatmulPadShape(
        m,
        k,
        n,
        group_size);
}

ryzenai_corelib_status
ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components(
    const ryzenai_corelib_matmul_bf16_weights_desc* desc,
    const ryzenai_corelib_matmul_bf16_onnx_weights_components* components,
    ryzenai_corelib_matmul_bf16_weights_ptr* out) {
    return flm::test::detail::FakeMatmulWeightsFromOnnx(
        desc,
        components,
        out);
}

ryzenai_corelib_status ryzenai_corelib_matmul_bf16_weights_get_data(
    ryzenai_corelib_matmul_bf16_weights_ptr weights,
    const void** data,
    std::size_t* size) {
    return flm::test::detail::FakeMatmulWeightsGetData(
        weights,
        data,
        size);
}

ryzenai_corelib_status ryzenai_corelib_matmul_bf16(
    ryzenai_corelib_stream_ptr stream,
    ryzenai_corelib_tensor_ptr input,
    int64_t rows,
    ryzenai_corelib_matmul_bf16_weights_ptr weights,
    ryzenai_corelib_tensor_ptr output) {
    return flm::test::detail::FakeMatmul(
        stream,
        input,
        rows,
        weights,
        output);
}

ryzenai_corelib_status ryzenai_corelib_ssmlp_bf16_pad_rows(
    int64_t* m,
    int64_t k,
    int64_t n,
    uint32_t group_size) {
    return flm::test::detail::FakeSsmlpPadRows(
        m,
        k,
        n,
        group_size);
}

ryzenai_corelib_status
ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components(
    const ryzenai_corelib_ssmlp_bf16_weights_desc* desc,
    const ryzenai_corelib_ssmlp_bf16_onnx_weights_components* components,
    ryzenai_corelib_ssmlp_bf16_weights_ptr* out) {
    return flm::test::detail::FakeSsmlpWeightsFromOnnx(
        desc,
        components,
        out);
}

ryzenai_corelib_status ryzenai_corelib_ssmlp_bf16_weights_get_data(
    ryzenai_corelib_ssmlp_bf16_weights_ptr weights,
    const void** data,
    std::size_t* size) {
    return flm::test::detail::FakeSsmlpWeightsGetData(
        weights,
        data,
        size);
}

ryzenai_corelib_status ryzenai_corelib_ssmlp_bf16(
    ryzenai_corelib_stream_ptr stream,
    ryzenai_corelib_tensor_ptr input,
    ryzenai_corelib_tensor_ptr residual,
    int64_t rows,
    ryzenai_corelib_ssmlp_bf16_weights_ptr weights,
    ryzenai_corelib_tensor_ptr skip_sum,
    ryzenai_corelib_tensor_ptr normalized) {
    return flm::test::detail::FakeSsmlp(
        stream,
        input,
        residual,
        rows,
        weights,
        skip_sum,
        normalized);
}

ryzenai_corelib_status ryzenai_corelib_flat_mha_bf16_pad_rows(
    int64_t* m,
    const ryzenai_corelib_flat_mha_bf16_desc* desc) {
    return flm::test::detail::FakeFlatMhaPadRows(m, desc);
}

ryzenai_corelib_status ryzenai_corelib_flat_mha_bf16(
    ryzenai_corelib_stream_ptr stream,
    const ryzenai_corelib_flat_mha_bf16_desc* desc,
    ryzenai_corelib_tensor_ptr query,
    ryzenai_corelib_tensor_ptr key,
    int64_t rows,
    int64_t position,
    ryzenai_corelib_tensor_ptr cos,
    ryzenai_corelib_tensor_ptr sin,
    ryzenai_corelib_tensor_ptr key_cache,
    ryzenai_corelib_tensor_ptr value_cache,
    ryzenai_corelib_tensor_ptr output) {
    return flm::test::detail::FakeFlatMha(
        stream,
        desc,
        query,
        key,
        rows,
        position,
        cos,
        sin,
        key_cache,
        value_cache,
        output);
}

void ryzenai_corelib_cleanup() {
    flm::test::detail::FakeCleanup();
}

#endif

namespace flm::test {
namespace {

template <class Function>
void* FunctionAddress(Function function) {
    return reinterpret_cast<void*>(function);
}

#define FLM_FAKE_ENTRY(symbol, replacement)                         \
    {                                                                \
        #symbol, FunctionAddress(                                    \
                     static_cast<decltype(&::symbol)>(               \
                         &detail::replacement))                      \
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
    detail::g_last_error.clear();
    detail::g_release_counts.clear();
    detail::g_release_count = 0;
    detail::g_last_released_object = nullptr;
    detail::g_selftest_status = ryzenai_corelib_status_success;
    detail::g_has_device_context = true;
    detail::g_cleanup_count = 0;
    detail::g_events.clear();
}

void SetLastErrorMessage(std::string message) {
    detail::g_last_error = std::move(message);
}

void SetSelftestStatus(ryzenai_corelib_status status) noexcept {
    detail::g_selftest_status = status;
}

void SetHasDeviceContext(bool value) noexcept {
    detail::g_has_device_context = value;
}

std::size_t ObjectReleaseCount() noexcept {
    return detail::g_release_count;
}

std::size_t ObjectReleaseCountFor(void* value) noexcept {
    const auto found = detail::g_release_counts.find(value);
    return found == detail::g_release_counts.end() ? 0 : found->second;
}

void* LastReleasedObject() noexcept {
    return detail::g_last_released_object;
}

std::size_t CleanupCount() noexcept {
    return detail::g_cleanup_count;
}

std::vector<std::string> FakeCorelibEvents() {
    return detail::g_events;
}

}  // namespace flm::test
