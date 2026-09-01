#pragma once

#include <corelib/corelib_api.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace flm::phi4 {

enum class RowUse {
    QueryProjection,
    KvProjection,
    Attention,
    OutputProjection,
    SsMlp,
    LmHead
};

struct MatMulPaddedShape {
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
};

struct Phi4Capacities {
    std::int64_t layer_rows;
    std::int64_t lm_head_rows;
};

class Phi4ShapePlan final {
public:
    static Phi4ShapePlan Build(
        std::shared_ptr<const corelib::CorelibApi> api);

    std::int64_t RowsFor(RowUse use, std::int64_t live_rows) const;
    const std::vector<std::pair<std::int64_t, std::int64_t>>&
    Transitions(RowUse use) const;
    const Phi4Capacities& capacities() const noexcept;
    const ryzenai_corelib_flat_mha_bf16_desc& attention_desc()
        const noexcept;

private:
    static constexpr std::size_t kRowUseCount = 6;

    Phi4ShapePlan() = default;

    std::array<
        std::vector<std::pair<std::int64_t, std::int64_t>>,
        kRowUseCount>
        transitions_;
    Phi4Capacities capacities_{};
    ryzenai_corelib_flat_mha_bf16_desc attention_desc_{};
};

}  // namespace flm::phi4
