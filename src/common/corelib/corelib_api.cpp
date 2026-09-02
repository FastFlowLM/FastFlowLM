#include <corelib/corelib_api.hpp>

#include <windows.h>

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace flm::corelib {
namespace {

constexpr wchar_t kCorelibPathEnvironment[] = L"RYZENAI_CORELIB_PATH";
constexpr wchar_t kCorelibFilename[] = L"ryzenai_corelib.dll";

std::string FormatCorelibError(
    std::string_view call,
    std::string_view status_text,
    std::string_view detail) {
    std::string message(call);
    message += " failed: ";
    message += status_text;
    if (!detail.empty()) {
        message += ": ";
        message += detail;
    }
    return message;
}

std::optional<std::wstring> ReadWideEnvironment(const wchar_t* name) {
    SetLastError(ERROR_SUCCESS);
    const DWORD required = GetEnvironmentVariableW(name, nullptr, 0);
    if (required == 0) {
        const DWORD error = GetLastError();
        if (error == ERROR_SUCCESS || error == ERROR_ENVVAR_NOT_FOUND) {
            return std::nullopt;
        }
        throw std::system_error(
            static_cast<int>(error),
            std::system_category(),
            "GetEnvironmentVariableW failed");
    }

    std::wstring value(required, L'\0');
    const DWORD written =
        GetEnvironmentVariableW(name, value.data(), required);
    if (written == 0 || written >= required) {
        throw std::system_error(
            static_cast<int>(GetLastError()),
            std::system_category(),
            "GetEnvironmentVariableW failed");
    }
    value.resize(written);
    return value;
}

std::filesystem::path MakeAbsolute(std::filesystem::path path) {
    if (!path.is_absolute()) {
        path = std::filesystem::absolute(path);
    }
    return path.lexically_normal();
}

template <class Function>
Function ResolveRequired(
    const CorelibApi::Resolver& resolver,
    std::string_view name) {
    void* address = resolver(name);
    if (address == nullptr) {
        throw std::runtime_error(
            "missing required ryzenai-corelib symbol: " +
            std::string(name));
    }
    return reinterpret_cast<Function>(address);
}

// Resolves and calls the version entry point BEFORE any other symbol is
// looked up. A runtime built from a different corelib revision renames and
// removes entry points, so resolving the rest first would report a missing
// symbol instead of the version skew that actually caused it.
CorelibVersion GateVersion(
    const CorelibApi::Resolver& resolver,
    decltype(&::ryzenai_corelib_get_version)& out_get_version) {
    out_get_version =
        ResolveRequired<decltype(&::ryzenai_corelib_get_version)>(
            resolver,
            "ryzenai_corelib_get_version");

    CorelibVersion runtime{};
    out_get_version(&runtime.major, &runtime.minor, &runtime.patch);

    const auto compiled = CompiledCorelibVersion();
    if (!IsCorelibVersionCompatible(compiled, runtime)) {
        throw std::runtime_error(
            FormatCorelibVersionMismatch(compiled, runtime));
    }
    return runtime;
}

CorelibFunctions ResolveFunctions(
    const CorelibApi::Resolver& resolver,
    CorelibVersion& out_runtime_version) {
    decltype(&::ryzenai_corelib_get_version) get_version = nullptr;
    out_runtime_version = GateVersion(resolver, get_version);

    return CorelibFunctions{
        get_version,
        ResolveRequired<
            decltype(&::ryzenai_corelib_status_to_string)>(
            resolver,
            "ryzenai_corelib_status_to_string"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_get_last_error_message)>(
            resolver,
            "ryzenai_corelib_get_last_error_message"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_selftest_dependencies)>(
            resolver,
            "ryzenai_corelib_selftest_dependencies"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_has_device_context)>(
            resolver,
            "ryzenai_corelib_has_device_context"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_object_release)>(
            resolver,
            "ryzenai_corelib_object_release"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_create_stream)>(
            resolver,
            "ryzenai_corelib_create_stream"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_stream_synchronize)>(
            resolver,
            "ryzenai_corelib_stream_synchronize"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_create_device_tensor)>(
            resolver,
            "ryzenai_corelib_create_device_tensor"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_tensor_write)>(
            resolver,
            "ryzenai_corelib_tensor_write"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_tensor_read)>(
            resolver,
            "ryzenai_corelib_tensor_read"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_tensor_get_byte_size)>(
            resolver,
            "ryzenai_corelib_tensor_get_byte_size"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_tensor_get_data_type)>(
            resolver,
            "ryzenai_corelib_tensor_get_data_type"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_matmul_bf16_pad_shape)>(
            resolver,
            "ryzenai_corelib_matmul_bf16_pad_shape"),
        ResolveRequired<decltype(
            &::ryzenai_corelib_matmul_bf16_weights_create_onnx)>(
            resolver,
            "ryzenai_corelib_matmul_bf16_weights_create_onnx"),
        ResolveRequired<decltype(
            &::ryzenai_corelib_matmul_bf16_weights_get_data)>(
            resolver,
            "ryzenai_corelib_matmul_bf16_weights_get_data"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_matmul_bf16)>(
            resolver,
            "ryzenai_corelib_matmul_bf16"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_ssmlp_bf16_pad_rows)>(
            resolver,
            "ryzenai_corelib_ssmlp_bf16_pad_rows"),
        ResolveRequired<decltype(
            &::ryzenai_corelib_ssmlp_bf16_weights_create_onnx)>(
            resolver,
            "ryzenai_corelib_ssmlp_bf16_weights_create_onnx"),
        ResolveRequired<decltype(
            &::ryzenai_corelib_ssmlp_bf16_weights_get_data)>(
            resolver,
            "ryzenai_corelib_ssmlp_bf16_weights_get_data"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_ssmlp_bf16)>(
            resolver,
            "ryzenai_corelib_ssmlp_bf16"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_flat_mha_bf16_pad_rows)>(
            resolver,
            "ryzenai_corelib_flat_mha_bf16_pad_rows"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_flat_mha_bf16)>(
            resolver,
            "ryzenai_corelib_flat_mha_bf16"),
        ResolveRequired<
            decltype(&::ryzenai_corelib_cleanup)>(
            resolver,
            "ryzenai_corelib_cleanup"),
    };
}

std::string LoadFailureMessage(
    const std::filesystem::path& path,
    DWORD error) {
    return "LoadLibraryExW failed for " + path.string() +
           " (Win32 error " + std::to_string(error) + ")";
}

}  // namespace

bool IsCorelibVersionCompatible(
    const CorelibVersion& compiled,
    const CorelibVersion& runtime) noexcept {
    if (compiled.major == 0) {
        // Pre-1.0 corelib may change the API in any release, including a
        // patch one, so every component is part of the contract.
        return compiled.major == runtime.major &&
               compiled.minor == runtime.minor &&
               compiled.patch == runtime.patch;
    }
    return compiled.major == runtime.major &&
           runtime.minor >= compiled.minor;
}

std::string FormatCorelibVersion(const CorelibVersion& value) {
    return std::to_string(value.major) + '.' +
           std::to_string(value.minor) + '.' +
           std::to_string(value.patch);
}

std::string FormatCorelibVersionMismatch(
    const CorelibVersion& compiled,
    const CorelibVersion& runtime) {
    return "ryzenai-corelib version mismatch: the loaded runtime reports "
           "version " +
           FormatCorelibVersion(runtime) +
           " but FastFlowLM was compiled against version " +
           FormatCorelibVersion(compiled) +
           "; install the matching ryzenai_corelib.dll";
}

CorelibVersion CompiledCorelibVersion() noexcept {
    return CorelibVersion{
        static_cast<std::uint32_t>(RYZENAI_CORELIB_VERSION_MAJOR),
        static_cast<std::uint32_t>(RYZENAI_CORELIB_VERSION_MINOR),
        static_cast<std::uint32_t>(RYZENAI_CORELIB_VERSION_PATCH)};
}

CorelibError::CorelibError(
    ryzenai_corelib_status status_value,
    std::string call_value,
    std::string detail_value,
    std::string status_text)
    : std::runtime_error(FormatCorelibError(
          call_value,
          status_text,
          detail_value)),
      status(status_value),
      call(std::move(call_value)),
      detail(std::move(detail_value)),
      status_text_(std::move(status_text)) {}

CorelibError CorelibError::WithContext(
    std::string_view context) const {
    std::string enriched_detail(context);
    if (!enriched_detail.empty() && !detail.empty()) {
        enriched_detail += ": ";
    }
    enriched_detail += detail;
    return CorelibError{
        status,
        call,
        std::move(enriched_detail),
        status_text_};
}

std::shared_ptr<CorelibApi> CorelibApi::Load(
    const std::filesystem::path& absolute_path) {
    if (!absolute_path.is_absolute()) {
        throw std::invalid_argument(
            "CorelibApi::Load requires an absolute DLL path");
    }

    HMODULE module = LoadLibraryExW(
        absolute_path.c_str(),
        nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (module == nullptr) {
        const DWORD error = GetLastError();
        throw std::runtime_error(
            LoadFailureMessage(absolute_path, error));
    }

    try {
        Resolver resolver = [module](std::string_view name) -> void* {
            const std::string symbol(name);
            return reinterpret_cast<void*>(
                GetProcAddress(module, symbol.c_str()));
        };
        CorelibVersion runtime_version{};
        auto functions = ResolveFunctions(resolver, runtime_version);
        return std::shared_ptr<CorelibApi>(
            new CorelibApi(
                module,
                absolute_path.lexically_normal(),
                std::move(functions),
                runtime_version));
    } catch (...) {
        FreeLibrary(module);
        throw;
    }
}

std::shared_ptr<CorelibApi> CorelibApi::ResolveForTest(
    Resolver resolver) {
    CorelibVersion runtime_version{};
    auto functions = ResolveFunctions(resolver, runtime_version);
    return std::shared_ptr<CorelibApi>(
        new CorelibApi(
            nullptr,
            {},
            std::move(functions),
            runtime_version));
}

std::filesystem::path CorelibApi::ResolveLibraryPath(
    const std::filesystem::path& executable_dir) {
    if (const auto configured =
            ReadWideEnvironment(kCorelibPathEnvironment)) {
        std::filesystem::path path(*configured);
        if (!path.is_absolute()) {
            throw std::invalid_argument(
                "RYZENAI_CORELIB_PATH must be an absolute file or "
                "directory path");
        }
        path = path.lexically_normal();
        std::error_code error;
        if (std::filesystem::is_directory(path, error)) {
            path /= kCorelibFilename;
        }
        return path.lexically_normal();
    }

    return MakeAbsolute(
        executable_dir / "aie4" / kCorelibFilename);
}

CorelibApi::CorelibApi(
    void* module,
    std::filesystem::path library_path,
    CorelibFunctions functions,
    CorelibVersion runtime_version)
    : module_(module),
      library_path_(std::move(library_path)),
      functions_(std::move(functions)),
      runtime_version_(runtime_version) {}

CorelibApi::~CorelibApi() {
    if (module_ != nullptr) {
        FreeLibrary(static_cast<HMODULE>(module_));
    }
}

const CorelibFunctions& CorelibApi::functions() const noexcept {
    return functions_;
}

const CorelibVersion& CorelibApi::runtime_version() const noexcept {
    return runtime_version_;
}

void CorelibApi::WriteElements(
    ryzenai_corelib_tensor_ptr tensor,
    ryzenai_corelib_data_type source_type,
    const void* source,
    std::size_t count,
    std::size_t offset) const {
    Check(
        functions_.tensor_write(
            tensor,
            source_type,
            source,
            count,
            offset),
        "ryzenai_corelib_tensor_write");
}

void CorelibApi::ReadElements(
    ryzenai_corelib_tensor_ptr tensor,
    ryzenai_corelib_data_type destination_type,
    void* destination,
    std::size_t count,
    std::size_t offset) const {
    Check(
        functions_.tensor_read(
            tensor,
            destination_type,
            destination,
            count,
            offset),
        "ryzenai_corelib_tensor_read");
}

void CorelibApi::Check(
    ryzenai_corelib_status status,
    std::string_view call) const {
    if (status == ryzenai_corelib_status_success) {
        return;
    }
    const char* raw = functions_.get_last_error_message();
    std::string detail = raw == nullptr ? std::string() : std::string(raw);
    const char* status_text = functions_.status_to_string(status);
    throw CorelibError{
        status,
        std::string(call),
        std::move(detail),
        status_text == nullptr ? "corelib failure" : status_text};
}

void CorelibApi::RegisterObject() const noexcept {
    live_object_count_.fetch_add(1, std::memory_order_relaxed);
}

void CorelibApi::Release(void* value) const noexcept {
    if (value == nullptr) {
        return;
    }

    functions_.object_release(value);
    const auto previous =
        live_object_count_.fetch_sub(1, std::memory_order_acq_rel);
    assert(previous > 0);
}

std::size_t CorelibApi::live_object_count() const noexcept {
    return live_object_count_.load(std::memory_order_acquire);
}

const std::filesystem::path& CorelibApi::library_path() const noexcept {
    return library_path_;
}

}  // namespace flm::corelib
