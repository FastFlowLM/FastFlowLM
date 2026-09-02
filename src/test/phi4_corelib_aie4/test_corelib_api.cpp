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
    "ryzenai_corelib_get_version",
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
    "ryzenai_corelib_tensor_get_data_type",
    "ryzenai_corelib_matmul_bf16_pad_shape",
    "ryzenai_corelib_matmul_bf16_weights_create_onnx",
    "ryzenai_corelib_matmul_bf16_weights_get_data",
    "ryzenai_corelib_matmul_bf16",
    "ryzenai_corelib_ssmlp_bf16_pad_rows",
    "ryzenai_corelib_ssmlp_bf16_weights_create_onnx",
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

void CheckGetDataIdentities(
    const CorelibApi& api,
    const std::unordered_map<std::string, void*>& expected) {
    const auto& functions = api.functions();
    CHECK(reinterpret_cast<void*>(functions.matmul_weights_get_data) ==
          expected.at(
              "ryzenai_corelib_matmul_bf16_weights_get_data"));
    CHECK(reinterpret_cast<void*>(functions.ssmlp_weights_get_data) ==
          expected.at(
              "ryzenai_corelib_ssmlp_bf16_weights_get_data"));
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
#define CHECK_MEMBER_IDENTITY(member, symbol)                          \
    CHECK(reinterpret_cast<void*>(functions.member) ==                \
          resolver.at(#symbol))
    CHECK_MEMBER_IDENTITY(get_version, ryzenai_corelib_get_version);
    CHECK_MEMBER_IDENTITY(
        status_to_string,
        ryzenai_corelib_status_to_string);
    CHECK_MEMBER_IDENTITY(
        get_last_error_message,
        ryzenai_corelib_get_last_error_message);
    CHECK_MEMBER_IDENTITY(
        selftest_dependencies,
        ryzenai_corelib_selftest_dependencies);
    CHECK_MEMBER_IDENTITY(
        has_device_context,
        ryzenai_corelib_has_device_context);
    CHECK_MEMBER_IDENTITY(
        object_release,
        ryzenai_corelib_object_release);
    CHECK_MEMBER_IDENTITY(create_stream, ryzenai_corelib_create_stream);
    CHECK_MEMBER_IDENTITY(
        stream_synchronize,
        ryzenai_corelib_stream_synchronize);
    CHECK_MEMBER_IDENTITY(
        create_device_tensor,
        ryzenai_corelib_create_device_tensor);
    CHECK_MEMBER_IDENTITY(tensor_write, ryzenai_corelib_tensor_write);
    CHECK_MEMBER_IDENTITY(tensor_read, ryzenai_corelib_tensor_read);
    CHECK_MEMBER_IDENTITY(
        tensor_get_byte_size,
        ryzenai_corelib_tensor_get_byte_size);
    CHECK_MEMBER_IDENTITY(
        tensor_get_data_type,
        ryzenai_corelib_tensor_get_data_type);
    CHECK_MEMBER_IDENTITY(
        matmul_pad_shape,
        ryzenai_corelib_matmul_bf16_pad_shape);
    CHECK_MEMBER_IDENTITY(
        matmul_weights_from_onnx,
        ryzenai_corelib_matmul_bf16_weights_create_onnx);
    CHECK_MEMBER_IDENTITY(
        matmul_weights_get_data,
        ryzenai_corelib_matmul_bf16_weights_get_data);
    CHECK_MEMBER_IDENTITY(matmul, ryzenai_corelib_matmul_bf16);
    CHECK_MEMBER_IDENTITY(
        ssmlp_pad_rows,
        ryzenai_corelib_ssmlp_bf16_pad_rows);
    CHECK_MEMBER_IDENTITY(
        ssmlp_weights_from_onnx,
        ryzenai_corelib_ssmlp_bf16_weights_create_onnx);
    CHECK_MEMBER_IDENTITY(
        ssmlp_weights_get_data,
        ryzenai_corelib_ssmlp_bf16_weights_get_data);
    CHECK_MEMBER_IDENTITY(ssmlp, ryzenai_corelib_ssmlp_bf16);
    CHECK_MEMBER_IDENTITY(
        flat_mha_pad_rows,
        ryzenai_corelib_flat_mha_bf16_pad_rows);
    CHECK_MEMBER_IDENTITY(flat_mha, ryzenai_corelib_flat_mha_bf16);
    CHECK_MEMBER_IDENTITY(cleanup, ryzenai_corelib_cleanup);
#undef CHECK_MEMBER_IDENTITY
}

void TestMatchingVersionLoadsAndIsRecorded() {
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();
    CHECK(api != nullptr);
    const auto compiled = flm::corelib::CompiledCorelibVersion();
    CHECK(api->runtime_version().major == compiled.major);
    CHECK(api->runtime_version().minor == compiled.minor);
    CHECK(api->runtime_version().patch == compiled.patch);
}

void TestMinorMismatchFailsNamingBothVersions() {
    flm::test::ResetFakeCorelib();
    const auto compiled = flm::corelib::CompiledCorelibVersion();
    flm::test::SetFakeCorelibVersion(
        compiled.major,
        compiled.minor + 1,
        compiled.patch);

    try {
        (void)ResolveCompleteCorelib();
    } catch (const std::exception& error) {
        const std::string_view message(error.what());
        CHECK(message.find(
                  flm::corelib::FormatCorelibVersion(compiled)) !=
              std::string_view::npos);
        CHECK(message.find(
                  flm::corelib::FormatCorelibVersion(
                      flm::corelib::CorelibVersion{
                          compiled.major,
                          compiled.minor + 1,
                          compiled.patch})) != std::string_view::npos);
        CHECK(message.find("version") != std::string_view::npos);
        flm::test::ResetFakeCorelib();
        return;
    }
    flm::test::ResetFakeCorelib();
    throw std::runtime_error(
        "a corelib minor-version mismatch must fail the load");
}

void TestPatchMismatchFailsWhileCorelibIsPreOneDotZero() {
    flm::test::ResetFakeCorelib();
    const auto compiled = flm::corelib::CompiledCorelibVersion();
    if (compiled.major != 0) {
        return;
    }
    flm::test::SetFakeCorelibVersion(
        compiled.major,
        compiled.minor,
        compiled.patch + 1);
    CheckThrowsContains(
        [&] { (void)ResolveCompleteCorelib(); },
        "version");
    flm::test::ResetFakeCorelib();
}

void TestVersionIsCheckedBeforeAnyOtherSymbolIsResolved() {
    flm::test::ResetFakeCorelib();
    auto resolver = flm::test::CompleteCorelibResolver();
    std::vector<std::string> requested;
    const auto resolve = [&resolver, &requested](
                             std::string_view name) -> void* {
        requested.emplace_back(name);
        const auto found = resolver.find(std::string(name));
        return found == resolver.end() ? nullptr : found->second;
    };

    (void)CorelibApi::ResolveForTest(resolve);
    CHECK(!requested.empty());
    CHECK(requested.front() == "ryzenai_corelib_get_version");

    // With an incompatible runtime the gate must stop there. Resolving the
    // rest first would report a missing renamed symbol and hide the skew
    // that actually caused it.
    const auto compiled = flm::corelib::CompiledCorelibVersion();
    flm::test::SetFakeCorelibVersion(
        compiled.major + 1,
        compiled.minor,
        compiled.patch);
    requested.clear();
    CheckThrowsContains(
        [&] { (void)CorelibApi::ResolveForTest(resolve); },
        "version");
    CHECK(requested.size() == 1);
    CHECK(requested.front() == "ryzenai_corelib_get_version");
    flm::test::ResetFakeCorelib();
}

void TestVersionCompatibilityRule() {
    using flm::corelib::CorelibVersion;
    using flm::corelib::IsCorelibVersionCompatible;

    // Below 1.0 the header says the API may change in any release, so all
    // three components are part of the contract.
    const CorelibVersion pre{0, 1, 0};
    CHECK(IsCorelibVersionCompatible(pre, CorelibVersion{0, 1, 0}));
    CHECK(!IsCorelibVersionCompatible(pre, CorelibVersion{0, 1, 1}));
    CHECK(!IsCorelibVersionCompatible(pre, CorelibVersion{0, 2, 0}));
    CHECK(!IsCorelibVersionCompatible(pre, CorelibVersion{1, 1, 0}));

    // From 1.0 the C ABI is additive within a major version.
    const CorelibVersion stable{1, 3, 2};
    CHECK(IsCorelibVersionCompatible(stable, CorelibVersion{1, 3, 2}));
    CHECK(IsCorelibVersionCompatible(stable, CorelibVersion{1, 3, 0}));
    CHECK(IsCorelibVersionCompatible(stable, CorelibVersion{1, 4, 0}));
    CHECK(!IsCorelibVersionCompatible(stable, CorelibVersion{1, 2, 9}));
    CHECK(!IsCorelibVersionCompatible(stable, CorelibVersion{2, 3, 2}));
}

// API-7: the wrapper takes elements, and there is no byte-taking overload
// to reach for by mistake. A count in bytes is 2x too large for a BF16
// tensor and the fake rejects it rather than moving twice the data.
void TestElementCountsAreBoundedByTheTensorNotItsBytes() {
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();

    const std::array<std::int64_t, 2> shape{4, 8};
    ryzenai_corelib_tensor_ptr raw = nullptr;
    api->Check(
        api->functions().create_device_tensor(
            ryzenai_corelib_data_type_bf16,
            shape.data(),
            shape.size(),
            &raw),
        "ryzenai_corelib_create_device_tensor");
    flm::corelib::UniqueTensor tensor(api, raw);

    std::size_t byte_size = 0;
    api->Check(
        api->functions().tensor_get_byte_size(tensor.get(), &byte_size),
        "ryzenai_corelib_tensor_get_byte_size");
    CHECK(byte_size == 32u * sizeof(std::uint16_t));

    ryzenai_corelib_data_type data_type{};
    api->Check(
        api->functions().tensor_get_data_type(tensor.get(), &data_type),
        "ryzenai_corelib_tensor_get_data_type");
    CHECK(data_type == ryzenai_corelib_data_type_bf16);

    std::vector<float> source(32, 0.0f);
    api->WriteElements(
        tensor.get(),
        ryzenai_corelib_data_type_fp32,
        source.data(),
        32,
        0);
    api->WriteElements(
        tensor.get(),
        ryzenai_corelib_data_type_fp32,
        source.data(),
        8,
        24);

    // The old spelling: 32 BF16 elements is 64 bytes, and 64 must fail.
    CheckThrowsContains(
        [&] {
            api->WriteElements(
                tensor.get(),
                ryzenai_corelib_data_type_fp32,
                source.data(),
                byte_size,
                0);
        },
        "ELEMENTS");
    CheckThrowsContains(
        [&] {
            api->ReadElements(
                tensor.get(),
                ryzenai_corelib_data_type_bf16,
                source.data(),
                32,
                1);
        },
        "ELEMENTS");
}

void TestTypeIdenticalGetDataSymbolsCannotBeSwapped() {
    auto expected = flm::test::CompleteCorelibResolver();
    const auto matmul_name =
        "ryzenai_corelib_matmul_bf16_weights_get_data";
    const auto ssmlp_name =
        "ryzenai_corelib_ssmlp_bf16_weights_get_data";
    CHECK(expected.at(matmul_name) != expected.at(ssmlp_name));

    auto api = CorelibApi::ResolveForTest(
        [&expected](std::string_view name) -> void* {
            return expected.at(std::string(name));
        });
    std::size_t matmul_size = 0;
    std::size_t ssmlp_size = 0;
    CHECK(api->functions().matmul_weights_get_data(
              nullptr,
              nullptr,
              &matmul_size) == ryzenai_corelib_status_success);
    CHECK(api->functions().ssmlp_weights_get_data(
              nullptr,
              nullptr,
              &ssmlp_size) == ryzenai_corelib_status_success);
    CHECK(matmul_size == 0x4D4D);
    CHECK(ssmlp_size == 0x5353);

    auto swapped = expected;
    std::swap(swapped.at(matmul_name), swapped.at(ssmlp_name));
    auto swapped_api = CorelibApi::ResolveForTest(
        [&swapped](std::string_view name) -> void* {
            return swapped.at(std::string(name));
        });
    CheckThrowsContains(
        [&] {
            CheckGetDataIdentities(*swapped_api, expected);
        },
        "matmul_weights_get_data");
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

// Task 13 Step 4. The post-warm allocation measurement rests on one property
// that `live_object_count()` cannot supply: a create followed by a release
// leaves the live count exactly where it started, so a decode loop that
// allocated and freed a device tensor per token would read as perfectly
// stable. The cumulative creation counters are what make that visible, and
// this is the test that they are cumulative rather than another live count.
void TestCreationCountsAreCumulativePerKind() {
    flm::test::ResetFakeCorelib();
    auto api = ResolveCompleteCorelib();
    int storage = 0;
    void* handle = &storage;

    using flm::corelib::CorelibObjectKind;
    CHECK(api->creation_count(CorelibObjectKind::Tensor) == 0);
    CHECK(api->weight_creation_count() == 0);

    {
        flm::corelib::UniqueStream stream(api, handle);
        flm::corelib::UniqueTensor first(api, handle);
        flm::corelib::UniqueMatMulWeights matmul(api, handle);
        flm::corelib::UniqueSsMlpWeights ssmlp(api, handle);
        CHECK(api->live_object_count() == 4);
        CHECK(api->creation_count(CorelibObjectKind::Stream) == 1);
        CHECK(api->creation_count(CorelibObjectKind::Tensor) == 1);
        CHECK(api->creation_count(CorelibObjectKind::MatMulWeights) == 1);
        CHECK(api->creation_count(CorelibObjectKind::SsMlpWeights) == 1);
        // Both weight kinds, because both are "a weight object" for design
        // 18.7's purposes and summing one of them would miss half the model.
        CHECK(api->weight_creation_count() == 2);
    }

    // Everything released: live is back to zero, creations are not.
    CHECK(api->live_object_count() == 0);
    CHECK(api->creation_count(CorelibObjectKind::Tensor) == 1);
    CHECK(api->weight_creation_count() == 2);

    // The churn case: a create/release pair inside a measurement window.
    const auto tensors_before =
        api->creation_count(CorelibObjectKind::Tensor);
    const auto live_before = api->live_object_count();
    {
        flm::corelib::UniqueTensor churn(api, handle);
    }
    CHECK(api->live_object_count() == live_before);
    CHECK(
        api->creation_count(CorelibObjectKind::Tensor) ==
        tensors_before + 1);

    // A null object is not a creation. Otherwise a failed create would count
    // toward a stability window it never allocated in.
    const auto before_null =
        api->creation_count(CorelibObjectKind::Tensor);
    flm::corelib::UniqueTensor null_value(api, nullptr);
    CHECK(
        api->creation_count(CorelibObjectKind::Tensor) == before_null);

    // Moving an object does not create a second one.
    const auto before_move =
        api->creation_count(CorelibObjectKind::Stream);
    {
        flm::corelib::UniqueStream original(api, handle);
        flm::corelib::UniqueStream moved(std::move(original));
        CHECK(
            api->creation_count(CorelibObjectKind::Stream) ==
            before_move + 1);
    }
    CHECK(
        api->creation_count(CorelibObjectKind::Stream) == before_move + 1);
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

void TestRelativeCorelibFileOverrideIsRejected() {
    TempDirectory temp;
    const auto executable_dir = temp.path() / "bin";
    std::filesystem::create_directories(executable_dir);
    Touch(temp.path() / "relative-corelib.dll");
    ScopedCurrentPath current_path{temp.path()};
    ScopedEnvironment override_path{
        L"RYZENAI_CORELIB_PATH",
        L"relative-corelib.dll"};

    CheckThrowsContains(
        [&] {
            (void)CorelibApi::ResolveLibraryPath(executable_dir);
        },
        "absolute");
}

void TestRelativeCorelibDirectoryOverrideIsRejected() {
    TempDirectory temp;
    const auto executable_dir = temp.path() / "bin";
    std::filesystem::create_directories(executable_dir);
    std::filesystem::create_directories(temp.path() / "relative-runtime");
    ScopedCurrentPath current_path{temp.path()};
    ScopedEnvironment override_path{
        L"RYZENAI_CORELIB_PATH",
        L"relative-runtime"};

    CheckThrowsContains(
        [&] {
            (void)CorelibApi::ResolveLibraryPath(executable_dir);
        },
        "absolute");
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
        TestMatchingVersionLoadsAndIsRecorded();
        TestMinorMismatchFailsNamingBothVersions();
        TestPatchMismatchFailsWhileCorelibIsPreOneDotZero();
        TestVersionIsCheckedBeforeAnyOtherSymbolIsResolved();
        TestVersionCompatibilityRule();
        TestElementCountsAreBoundedByTheTensorNotItsBytes();
        TestTypeIdenticalGetDataSymbolsCannotBeSwapped();
        TestMissingSymbolFailsAtomically();
        TestErrorDetailSurvivesStatusConversion();
        TestSuccessfulStatusDoesNotReadErrorState();
        TestMoveConstructionReleasesExactlyOnce();
        TestMoveAssignmentReleasesEachObjectOnce();
        TestNullResetDoesNotRelease();
        TestCreationCountsAreCumulativePerKind();
        TestExplicitCorelibFileWins();
        TestExplicitCorelibDirectoryWins();
        TestRelativeCorelibFileOverrideIsRejected();
        TestRelativeCorelibDirectoryOverrideIsRejected();
        TestFallbackIgnoresCurrentDirectoryAndPath();
        std::cout << "test_corelib_api: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
