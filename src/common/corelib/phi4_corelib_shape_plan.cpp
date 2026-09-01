#include <models/phi4/phi4_corelib_shape_plan.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flm::phi4 {
namespace {

constexpr std::int64_t kShapePrecomputeMaxLiveRows = 4096;
static_assert(
    kShapePrecomputeMaxLiveRows ==
        constants::kMaxSequenceLength,
    "Phi4ShapePlan v1 queries every live row through the "
    "physical attention window");

constexpr ryzenai_corelib_flat_mha_bf16_desc kAttentionDesc{
    constants::kQueryHeadCount,
    constants::kKvHeadCount,
    constants::kHeadSize,
    constants::kMaxSequenceLength,
    constants::kRopeDimension};

std::size_t RowUseIndex(RowUse use) {
    switch (use) {
        case RowUse::QueryProjection:
            return 0;
        case RowUse::KvProjection:
            return 1;
        case RowUse::Attention:
            return 2;
        case RowUse::OutputProjection:
            return 3;
        case RowUse::SsMlp:
            return 4;
        case RowUse::LmHead:
            return 5;
    }
    throw std::invalid_argument("unknown Phi-4 RowUse");
}

void ValidatePaddedRows(
    std::string_view context,
    std::int64_t live_rows,
    std::int64_t padded_rows) {
    if (padded_rows < live_rows) {
        throw std::runtime_error(
            std::string(context) +
            " returned invalid padded rows: live=" +
            std::to_string(live_rows) +
            ", padded=" + std::to_string(padded_rows));
    }
}

MatMulPaddedShape QueryMatMul(
    const corelib::CorelibApi& api,
    std::int64_t live_rows,
    std::int64_t k,
    std::int64_t n,
    std::string_view context) {
    MatMulPaddedShape shape{live_rows, k, n};
    api.Check(
        api.functions().matmul_pad_shape(
            &shape.m,
            &shape.k,
            &shape.n,
            constants::kGroupSize),
        "ryzenai_corelib_matmul_bf16_pad_shape");
    if (shape.k != k || shape.n != n) {
        throw std::runtime_error(
            std::string(context) +
            " MatMul K/N mismatch at live row " +
            std::to_string(live_rows) +
            ": requested K=" + std::to_string(k) +
            ", N=" + std::to_string(n) +
            "; returned K=" + std::to_string(shape.k) +
            ", N=" + std::to_string(shape.n));
    }
    ValidatePaddedRows(context, live_rows, shape.m);
    return shape;
}

std::int64_t QuerySsMlp(
    const corelib::CorelibApi& api,
    std::int64_t live_rows) {
    std::int64_t padded_rows = live_rows;
    api.Check(
        api.functions().ssmlp_pad_rows(
            &padded_rows,
            constants::kHiddenSize,
            constants::kIntermediateSize,
            constants::kGroupSize),
        "ryzenai_corelib_ssmlp_bf16_pad_rows");
    ValidatePaddedRows("SSMLP", live_rows, padded_rows);
    return padded_rows;
}

std::int64_t QueryAttention(
    const corelib::CorelibApi& api,
    std::int64_t live_rows,
    const ryzenai_corelib_flat_mha_bf16_desc& desc) {
    std::int64_t padded_rows = live_rows;
    api.Check(
        api.functions().flat_mha_pad_rows(&padded_rows, &desc),
        "ryzenai_corelib_flat_mha_bf16_pad_rows");
    ValidatePaddedRows("attention", live_rows, padded_rows);
    return padded_rows;
}

void AppendTransition(
    std::vector<std::pair<std::int64_t, std::int64_t>>& transitions,
    std::int64_t live_rows,
    std::int64_t padded_rows) {
    if (transitions.empty() ||
        transitions.back().second != padded_rows) {
        transitions.emplace_back(live_rows, padded_rows);
    }
}

}  // namespace

Phi4ShapePlan Phi4ShapePlan::Build(
    std::shared_ptr<const corelib::CorelibApi> api) {
    if (!api) {
        throw std::invalid_argument(
            "Phi4ShapePlan::Build requires a CorelibApi");
    }

    Phi4ShapePlan plan;
    plan.attention_desc_ = kAttentionDesc;

    auto& query_transitions =
        plan.transitions_[RowUseIndex(RowUse::QueryProjection)];
    auto& kv_transitions =
        plan.transitions_[RowUseIndex(RowUse::KvProjection)];
    auto& attention_transitions =
        plan.transitions_[RowUseIndex(RowUse::Attention)];
    auto& output_transitions =
        plan.transitions_[RowUseIndex(RowUse::OutputProjection)];
    auto& ssmlp_transitions =
        plan.transitions_[RowUseIndex(RowUse::SsMlp)];

    for (std::int64_t live_rows = 1;
         live_rows <= kShapePrecomputeMaxLiveRows;
         ++live_rows) {
        const auto query_shape = QueryMatMul(
            *api,
            live_rows,
            constants::kHiddenSize,
            constants::kQueryDimension,
            "query/output projection");
        const auto kv_shape = QueryMatMul(
            *api,
            live_rows,
            constants::kHiddenSize,
            constants::kKvDimension,
            "key/value projection");
        const auto ssmlp_rows = QuerySsMlp(*api, live_rows);
        const auto attention_rows = QueryAttention(
            *api,
            live_rows,
            plan.attention_desc_);

        AppendTransition(
            query_transitions,
            live_rows,
            query_shape.m);
        AppendTransition(
            output_transitions,
            live_rows,
            query_shape.m);
        AppendTransition(kv_transitions, live_rows, kv_shape.m);
        AppendTransition(
            ssmlp_transitions,
            live_rows,
            ssmlp_rows);
        AppendTransition(
            attention_transitions,
            live_rows,
            attention_rows);

        plan.capacities_.layer_rows = std::max(
            {plan.capacities_.layer_rows,
             query_shape.m,
             kv_shape.m,
             ssmlp_rows,
             attention_rows});
    }

    const auto lm_head_shape = QueryMatMul(
        *api,
        1,
        constants::kHiddenSize,
        constants::kVocabularySize,
        "LM head");
    AppendTransition(
        plan.transitions_[RowUseIndex(RowUse::LmHead)],
        1,
        lm_head_shape.m);
    plan.capacities_.lm_head_rows = lm_head_shape.m;

    return plan;
}

std::int64_t Phi4ShapePlan::RowsFor(
    RowUse use,
    std::int64_t live_rows) const {
    const auto& transitions = Transitions(use);
    const std::int64_t max_live_rows =
        use == RowUse::LmHead
            ? 1
            : kShapePrecomputeMaxLiveRows;
    if (live_rows < 1 || live_rows > max_live_rows) {
        throw std::out_of_range(
            "Phi4ShapePlan live rows are outside the cached range");
    }

    const auto next = std::upper_bound(
        transitions.begin(),
        transitions.end(),
        live_rows,
        [](std::int64_t value, const auto& transition) {
            return value < transition.first;
        });
    if (next == transitions.begin()) {
        throw std::logic_error(
            "Phi4ShapePlan has no transition for live rows");
    }
    return std::prev(next)->second;
}

const std::vector<std::pair<std::int64_t, std::int64_t>>&
Phi4ShapePlan::Transitions(RowUse use) const {
    return transitions_[RowUseIndex(use)];
}

const Phi4Capacities& Phi4ShapePlan::capacities() const noexcept {
    return capacities_;
}

const ryzenai_corelib_flat_mha_bf16_desc&
Phi4ShapePlan::attention_desc() const noexcept {
    return attention_desc_;
}

}  // namespace flm::phi4
