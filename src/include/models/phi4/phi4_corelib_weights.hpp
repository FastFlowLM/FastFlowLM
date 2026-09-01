#pragma once

#include <corelib/corelib_object.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>
#include <models/phi4/phi4_corelib_manifest.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace flm::phi4 {

struct LayerWeights {
    corelib::UniqueMatMulWeights q;
    corelib::UniqueMatMulWeights k;
    corelib::UniqueMatMulWeights v;
    corelib::UniqueMatMulWeights o;
    corelib::UniqueSsMlpWeights mlp;
};

class Phi4Weights final {
public:
    static Phi4Weights Load(
        std::shared_ptr<const corelib::CorelibApi> api,
        std::shared_ptr<Phi4Package> package);

    Phi4Weights(Phi4Weights&&) noexcept = default;
    Phi4Weights& operator=(Phi4Weights&& other) noexcept;

    Phi4Weights(const Phi4Weights&) = delete;
    Phi4Weights& operator=(const Phi4Weights&) = delete;

    const std::array<LayerWeights, constants::kLayerCount>&
    layers() const noexcept;
    const corelib::UniqueMatMulWeights& lm_head() const noexcept;
    std::size_t packed_bytes() const noexcept;

private:
    Phi4Weights() = default;

    void ResetWeightObjects() noexcept;

    // Owners precede every weight that receives their addresses. Reverse
    // destruction therefore releases corelib objects before source storage.
    std::shared_ptr<Phi4Package> package_;
    std::shared_ptr<const std::uint16_t> epsilon_bf16_;
    std::array<LayerWeights, constants::kLayerCount> layers_;
    corelib::UniqueMatMulWeights lm_head_;
    std::size_t packed_bytes_ = 0;
};

}  // namespace flm::phi4
