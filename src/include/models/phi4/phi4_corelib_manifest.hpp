#pragma once

#include <corelib/corelib_api.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace flm::phi4 {

class MappedFile final {
public:
    static std::shared_ptr<const MappedFile> OpenReadOnly(
        const std::filesystem::path& path);

    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;
    ~MappedFile() noexcept;

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept;
    std::uint64_t size() const noexcept;
    const std::filesystem::path& path() const noexcept;

private:
    MappedFile(
        std::filesystem::path path,
        void* file,
        void* mapping,
        const std::byte* data,
        std::uint64_t size) noexcept;

    void Reset() noexcept;

    std::filesystem::path path_;
    void* file_ = nullptr;
    void* mapping_ = nullptr;
    const std::byte* data_ = nullptr;
    std::uint64_t size_ = 0;
};

enum class SourceDType {
    UInt8,
    Float16,
    Float32,
    Int64
};

struct InitializerView {
    SourceDType dtype;
    std::vector<std::int64_t> shape;
    const std::byte* data;
    std::size_t size;
    std::shared_ptr<const MappedFile> owner;
};

enum class WeightObjectKind {
    MatMul,
    SsMlp
};

struct WeightObjectView {
    std::string name;
    WeightObjectKind kind;
    std::int64_t k;
    std::int64_t n;
    std::uint32_t group_size;
    bool has_bias;
    std::map<std::string, std::string> components;
};

// A contiguous [4096, 48] slice of a RoPE table, kept in the SOURCE dtype.
// `tensor_write` performs the widening to the FP32 device tensor, which is
// the only conversion boundary corelib `e5258d2` offers.
struct RopeSourceView {
    ryzenai_corelib_data_type dtype;
    const void* data;
    std::size_t count;
};

class Phi4Package final {
public:
    static Phi4Package Load(
        const std::filesystem::path& model_dir,
        std::shared_ptr<const corelib::CorelibApi> api,
        bool verify_full_hash);

    Phi4Package(Phi4Package&&) noexcept = default;
    Phi4Package& operator=(Phi4Package&&) noexcept = default;
    ~Phi4Package() = default;

    Phi4Package(const Phi4Package&) = delete;
    Phi4Package& operator=(const Phi4Package&) = delete;

    const InitializerView& Require(std::string_view name) const;
    const std::vector<WeightObjectView>& weight_objects() const noexcept;
    std::span<const std::uint16_t> MaterializeFp16(
        std::string_view name);
    std::span<const std::uint16_t> MaterializeBf16(
        std::string_view name);
    RopeSourceView MaterializeRopeGather(std::string_view name);

private:
    Phi4Package() = default;

    std::shared_ptr<const corelib::CorelibApi> api_;

    // Declare every owner before the non-owning views. Reverse member
    // destruction then releases views before their mapped/derived storage.
    std::map<
        std::string,
        std::shared_ptr<const MappedFile>,
        std::less<>>
        mapped_files_;
    std::map<std::string, std::vector<std::uint16_t>, std::less<>>
        fp16_buffers_;
    std::map<std::string, std::vector<std::uint16_t>, std::less<>>
        bf16_buffers_;
    std::map<std::string, std::vector<std::byte>, std::less<>>
        rope_buffers_;

    std::map<std::string, InitializerView, std::less<>>
        initializers_;
    std::vector<WeightObjectView> weight_objects_;
};

}  // namespace flm::phi4
