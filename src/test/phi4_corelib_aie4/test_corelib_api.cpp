#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <corelib/corelib_api.hpp>
#include <corelib/corelib_object.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using flm::corelib::CorelibApi;

constexpr std::array<std::string_view, 24> kRequiredSymbols{
    "ryzenai_corelib_status_to_string",
    "ryzenai_corelib_get_last_error_message",
    "ryzenai_corelib_selftest_dependencies",
    "ryzenai_corelib_has_device_context",
    "ryzenai_corelib_object_release",
    "ryzenai_corelib_create_stream",
    "ryzenai_corelib_stream_synchronize",
    "ryzenai_corelib_create_device_tensor",
    "ryzenai_corelib_tensor_write",
    "ryzenai_corelib_tensor_read",
    "ryzenai_corelib_tensor_get_byte_size",
    "ryzenai_corelib_convert",
    "ryzenai_corelib_convert_strided",
    "ryzenai_corelib_matmul_bf16_pad_shape",
    "ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components",
    "ryzenai_corelib_matmul_bf16_weights_get_data",
    "ryzenai_corelib_matmul_bf16",
    "ryzenai_corelib_ssmlp_bf16_pad_rows",
    "ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components",
    "ryzenai_corelib_ssmlp_bf16_weights_get_data",
    "ryzenai_corelib_ssmlp_bf16",
    "ryzenai_corelib_flat_mha_bf16_pad_rows",
    "ryzenai_corelib_flat_mha_bf16",
    "ryzenai_corelib_cleanup",
};

std::shared_ptr<CorelibApi> ResolveCompleteCorelib() {
    auto resolver = flm::test::CompleteCorelibResolver();
    return CorelibApi::ResolveForTest(
        [resolver = std::move(resolver)](std::string_view name) mutable
            -> void* {
            const auto found = resolver.find(std::string(name));
            return found == resolver.end() ? nullptr : found->second;
        });
}

std::optional<std::wstring> ReadEnvironment(const wchar_t* name) {
    std::size_t required = 0;
    if (_wgetenv_s(&required, nullptr, 0, name) != 0) {
        throw std::runtime_error("failed to read environment variable");
    }
    if (required == 0) {
        return std::nullopt;
    }

    std::vector<wchar_t> value(required);
    if (_wgetenv_s(&required, value.data(), value.size(), name) != 0) {
        throw std::runtime_error("failed to read environment variable");
    }
    return std::wstring(value.data());
}

class ScopedEnvironment final {
public:
    ScopedEnvironment(
        std::wstring name,
        std::optional<std::wstring> value)
        : name_(std::move(name)),
          original_(ReadEnvironment(name_.c_str())) {
        Set(value);
    }

    ~ScopedEnvironment() noexcept {
        _wputenv_s(
            name_.c_str(),
            original_.has_value() ? original_->c_str() : L"");
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
    void Set(const std::optional<std::wstring>& value) {
        if (_wputenv_s(
                name_.c_str(),
                value.has_value() ? value->c_str() : L"") != 0) {
            throw std::runtime_error("failed to set environment variable");
        }
    }

    std::wstring name_;
    std::optional<std::wstring> original_;
};

class ScopedCurrentPath final {
public:
    explicit ScopedCurrentPath(const std::filesystem::path& value)
        : original_(std::filesystem::current_path()) {
        std::filesystem::current_path(value);
    }

    ~ScopedCurrentPath() noexcept {
        std::error_code error;
        std::filesystem::current_path(original_, error);
    }

    ScopedCurrentPath(const ScopedCurrentPath&) = delete;
    ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

private:
    std::filesystem::path original_;
};

class TempDirectory final {
public:
    TempDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("fastflowlm-corelib-api-" + std::to_string(nonce));
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

void Touch(const std::filesystem::path& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to create test file");
    }
}

void TestCompleteResolution() {
    auto resolver = flm::test::CompleteCorelibResolver();
    CHECK(resolver.size() == kRequiredSymbols.size());
    for (const auto name : kRequiredSymbols) {
        CHECK(resolver.contains(std::string(name)));
    }

    std::vector<std::string> requested;
    auto api = CorelibApi::ResolveForTest(
        [&resolver, &requested](std::string_view name) -> void* {
            requested.emplace_back(name);
            const auto found = resolver.find(std::string(name));
            return found == resolver.end() ? nullptr : found->second;
        });

    CHECK(api != nullptr);
    CHECK(requested.size() == kRequiredSymbols.size());
    for (const auto name : kRequiredSymbols) {
        CHECK(std::count(requested.begin(), requested.end(), name) == 1);
    }

    const auto& functions = api->functions();
    CHECK(functions.status_to_string != nullptr);
    CHECK(functions.get_last_error_message != nullptr);
    CHECK(functions.selftest_dependencies != nullptr);
    CHECK(functions.has_device_context != nullptr);
    CHECK(functions.object_release != nullptr);
    CHECK(functions.create_stream != nullptr);
    CHECK(functions.stream_synchronize != nullptr);
    CHECK(functions.create_device_tensor != nullptr);
    CHECK(functions.tensor_write != nullptr);
    CHECK(functions.tensor_read != nullptr);
    CHECK(functions.tensor_get_byte_size != nullptr);
    CHECK(functions.convert != nullptr);
    CHECK(functions.convert_strided != nullptr);
    CHECK(functions.matmul_pad_shape != nullptr);
    CHECK(functions.matmul_weights_from_onnx != nullptr);
    CHECK(functions.matmul_weights_get_data != nullptr);
    CHECK(functions.matmul != nullptr);
    CHECK(functions.ssmlp_pad_rows != nullptr);
    CHECK(functions.ssmlp_weights_from_onnx != nullptr);
    CHECK(functions.ssmlp_weights_get_data != nullptr);
    CHECK(functions.ssmlp != nullptr);
    CHECK(functions.flat_mha_pad_rows != nullptr);
    CHECK(functions.flat_mha != nullptr);
    CHECK(functions.cleanup != nullptr);
}

void TestMissingSymbolFailsAtomically() {
    auto resolver = flm::test::CompleteCorelibResolver();
    resolver.erase("ryzenai_corelib_flat_mha_bf16");
    CheckThrowsContains(
        [&] {
            flm::corelib::CorelibApi::ResolveForTest(
            [resolver](std::string_view name) mutable -> void* {
                const auto found = resolver.find(std::string(name));
                return found == resolver.end() ? nullptr : found->second;
            });
        },
        "ryzenai_corelib_flat_mha_bf16");
}

void TestErrorDetailSurvivesStatusConversion() {
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();
    const std::string expected_detail =
        "durable detail copied before the next corelib call";
    flm::test::SetLastErrorMessage(expected_detail);

    try {
        api->Check(
            ryzenai_corelib_status_bad_argument,
            "ryzenai_corelib_test_call");
    } catch (const flm::corelib::CorelibError& error) {
        CHECK(error.status == ryzenai_corelib_status_bad_argument);
        CHECK(error.call == "ryzenai_corelib_test_call");
        CHECK(error.detail == expected_detail);
        CHECK(std::string_view(error.what()).find(
                  "ryzenai_corelib_test_call") != std::string_view::npos);
        CHECK(std::string_view(error.what()).find("bad argument") !=
              std::string_view::npos);
        CHECK(std::string_view(error.what()).find(expected_detail) !=
              std::string_view::npos);
        return;
    }
    throw std::runtime_error("expected CorelibError was not thrown");
}

void TestSuccessfulStatusDoesNotReadErrorState() {
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();
    flm::test::SetLastErrorMessage("must remain untouched");

    api->Check(ryzenai_corelib_status_success, "successful_call");

    CHECK(std::string_view(
              api->functions().get_last_error_message()) ==
          "must remain untouched");
}

void TestMoveConstructionReleasesExactlyOnce() {
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();
    int storage = 0;
    void* handle = &storage;

    {
        flm::corelib::UniqueStream original(api, handle);
        CHECK(api->live_object_count() == 1);

        flm::corelib::UniqueStream moved(std::move(original));
        CHECK(!original);
        CHECK(moved.get() == handle);
        CHECK(api->live_object_count() == 1);
        CHECK(flm::test::ObjectReleaseCount() == 0);
    }

    CHECK(flm::test::ObjectReleaseCount() == 1);
    CHECK(flm::test::ObjectReleaseCountFor(handle) == 1);
    CHECK(flm::test::LastReleasedObject() == handle);
    CHECK(api->live_object_count() == 0);
}

void TestMoveAssignmentReleasesEachObjectOnce() {
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();
    int source_storage = 0;
    int target_storage = 0;
    void* source_handle = &source_storage;
    void* target_handle = &target_storage;

    {
        flm::corelib::UniqueTensor source(api, source_handle);
        flm::corelib::UniqueTensor target(api, target_handle);
        CHECK(api->live_object_count() == 2);

        target = std::move(source);
        CHECK(!source);
        CHECK(target.get() == source_handle);
        CHECK(flm::test::ObjectReleaseCountFor(target_handle) == 1);
        CHECK(flm::test::ObjectReleaseCountFor(source_handle) == 0);
        CHECK(api->live_object_count() == 1);
    }

    CHECK(flm::test::ObjectReleaseCount() == 2);
    CHECK(flm::test::ObjectReleaseCountFor(target_handle) == 1);
    CHECK(flm::test::ObjectReleaseCountFor(source_handle) == 1);
    CHECK(api->live_object_count() == 0);
}

void TestNullResetDoesNotRelease() {
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();

    flm::corelib::UniqueMatMulWeights empty;
    empty.reset();
    flm::corelib::UniqueSsMlpWeights null_value(api, nullptr);
    null_value.reset();

    CHECK(flm::test::ObjectReleaseCount() == 0);
    CHECK(api->live_object_count() == 0);
}

void TestExplicitCorelibFileWins() {
    TempDirectory temp;
    const auto executable_dir = temp.path() / "bin";
    const auto explicit_file = temp.path() / "chosen-corelib.dll";
    std::filesystem::create_directories(executable_dir);
    Touch(explicit_file);
    ScopedEnvironment override_path{
        L"RYZENAI_CORELIB_PATH",
        explicit_file.wstring()};

    CHECK(CorelibApi::ResolveLibraryPath(executable_dir) ==
          explicit_file.lexically_normal());
}

void TestExplicitCorelibDirectoryWins() {
    TempDirectory temp;
    const auto executable_dir = temp.path() / "bin";
    const auto explicit_directory = temp.path() / "runtime";
    std::filesystem::create_directories(executable_dir);
    std::filesystem::create_directories(explicit_directory);
    ScopedEnvironment override_path{
        L"RYZENAI_CORELIB_PATH",
        explicit_directory.wstring()};

    CHECK(CorelibApi::ResolveLibraryPath(executable_dir) ==
          (explicit_directory / "ryzenai_corelib.dll").lexically_normal());
}

void TestFallbackIgnoresCurrentDirectoryAndPath() {
    TempDirectory temp;
    const auto executable_dir = temp.path() / "application";
    const auto trap_directory = temp.path() / "trap";
    std::filesystem::create_directories(executable_dir);
    std::filesystem::create_directories(trap_directory);
    Touch(trap_directory / "ryzenai_corelib.dll");

    ScopedEnvironment no_override{
        L"RYZENAI_CORELIB_PATH",
        std::nullopt};
    ScopedEnvironment trap_path{
        L"PATH",
        trap_directory.wstring()};
    ScopedCurrentPath trap_current_directory{trap_directory};

    CHECK(CorelibApi::ResolveLibraryPath(executable_dir) ==
          (executable_dir / "aie4" / "ryzenai_corelib.dll")
              .lexically_normal());
}

static_assert(
    !std::is_copy_constructible_v<flm::corelib::UniqueStream>);
static_assert(
    !std::is_copy_assignable_v<flm::corelib::UniqueStream>);
static_assert(
    std::is_nothrow_move_constructible_v<flm::corelib::UniqueStream>);
static_assert(
    std::is_nothrow_move_assignable_v<flm::corelib::UniqueStream>);

}  // namespace

int main() {
    try {
        TestCompleteResolution();
        TestMissingSymbolFailsAtomically();
        TestErrorDetailSurvivesStatusConversion();
        TestSuccessfulStatusDoesNotReadErrorState();
        TestMoveConstructionReleasesExactlyOnce();
        TestMoveAssignmentReleasesEachObjectOnce();
        TestNullResetDoesNotRelease();
        TestExplicitCorelibFileWins();
        TestExplicitCorelibDirectoryWins();
        TestFallbackIgnoresCurrentDirectoryAndPath();
        std::cout << "test_corelib_api: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
