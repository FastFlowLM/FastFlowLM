#include <models/phi4/phi4_corelib_aie4.hpp>

#include <corelib/corelib_object.hpp>
#include <corelib/host_convert.hpp>
#include <models/phi4/phi4_corelib_constants.hpp>
#include <models/phi4/phi4_corelib_host.hpp>
#include <models/phi4/phi4_corelib_manifest.hpp>
#include <models/phi4/phi4_corelib_shape_plan.hpp>
#include <models/phi4/phi4_corelib_weights.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace flm::phi4 {
namespace {

constexpr std::string_view kEmbeddingName =
    "model.embed_tokens.weight";
constexpr std::string_view kInputNormName =
    "model.layers.0.input_layernorm.weight";
constexpr std::string_view kCosName = "cos_cache";
constexpr std::string_view kSinName = "sin_cache";
constexpr std::string_view kCreateStreamCall =
    "ryzenai_corelib_create_stream";
constexpr std::string_view kCreateTensorCall =
    "ryzenai_corelib_create_device_tensor";
constexpr std::string_view kTensorByteSizeCall =
    "ryzenai_corelib_tensor_get_byte_size";
constexpr std::string_view kSynchronizeCall =
    "ryzenai_corelib_stream_synchronize";
constexpr std::string_view kTensorDataTypeCall =
    "ryzenai_corelib_tensor_get_data_type";

static_assert(sizeof(bf16) == sizeof(std::uint16_t));

std::uint32_t ValidateMaxLength(std::uint32_t max_length) {
    if (
        max_length == 0 ||
        max_length >
            static_cast<std::uint32_t>(
                constants::kMaxSequenceLength)) {
        throw std::out_of_range(
            "Phi-4 AIE4 maximum length must be in 1..4096");
    }
    return max_length;
}

void ValidateOptionalIntegerIdentity(
    const LM_Config& config,
    std::string_view key,
    std::int64_t expected) {
    const auto found = config._json_config.find(std::string(key));
    if (
        found == config._json_config.end() ||
        found->is_null()) {
        return;
    }
    if (!found->is_number_integer()) {
        throw std::invalid_argument(
            "Phi-4 AIE4 LM_Config field '" +
            std::string(key) + "' must be an integer");
    }
    const std::int64_t actual = found->get<std::int64_t>();
    if (actual != expected) {
        throw std::invalid_argument(
            "Phi-4 AIE4 LM_Config field '" +
            std::string(key) +
            "' does not match the package identity");
    }
}

void ValidateOptionalFloatingIdentity(
    const LM_Config& config,
    std::string_view key,
    double expected) {
    const auto found = config._json_config.find(std::string(key));
    if (
        found == config._json_config.end() ||
        found->is_null()) {
        return;
    }
    if (!found->is_number()) {
        throw std::invalid_argument(
            "Phi-4 AIE4 LM_Config field '" +
            std::string(key) + "' must be numeric");
    }
    const double actual = found->get<double>();
    if (
        !std::isfinite(actual) ||
        std::abs(actual - expected) >
            std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(expected)) * 8.0) {
        throw std::invalid_argument(
            "Phi-4 AIE4 LM_Config field '" +
            std::string(key) +
            "' does not match the package identity");
    }
}

void ValidateConfigIdentity(const LM_Config& config) {
    // Legacy LM_Config files do not guarantee every fixed field. Validate
    // each field they do provide; the package manifest independently remains
    // the authority for absent fields.
    ValidateOptionalIntegerIdentity(
        config,
        "num_hidden_layers",
        constants::kLayerCount);
    ValidateOptionalIntegerIdentity(
        config,
        "hidden_size",
        constants::kHiddenSize);
    ValidateOptionalIntegerIdentity(
        config,
        "intermediate_size",
        constants::kIntermediateSize);
    ValidateOptionalIntegerIdentity(
        config,
        "num_attention_heads",
        constants::kQueryHeadCount);
    ValidateOptionalIntegerIdentity(
        config,
        "num_key_value_heads",
        constants::kKvHeadCount);
    ValidateOptionalIntegerIdentity(
        config,
        "head_dim",
        constants::kHeadSize);
    ValidateOptionalIntegerIdentity(
        config,
        "vocab_size",
        constants::kVocabularySize);
    ValidateOptionalFloatingIdentity(
        config,
        "rms_norm_eps",
        constants::kRmsEpsilon);
}

std::size_t CheckedElements(
    std::int64_t rows,
    std::int64_t width,
    std::string_view context) {
    if (rows <= 0 || width <= 0) {
        throw std::invalid_argument(
            std::string(context) +
            " rows and width must be positive");
    }
    const auto row_count = static_cast<std::uint64_t>(rows);
    const auto column_count = static_cast<std::uint64_t>(width);
    if (
        row_count >
        std::numeric_limits<std::size_t>::max() / column_count) {
        throw std::overflow_error(
            std::string(context) + " extent overflows size_t");
    }
    return static_cast<std::size_t>(row_count * column_count);
}

std::size_t TensorByteCount(
    ryzenai_corelib_data_type data_type,
    std::span<const std::int64_t> shape) {
    if (shape.empty()) {
        throw std::invalid_argument(
            "Phi-4 AIE4 tensor shape cannot be empty");
    }
    std::size_t elements = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0) {
            throw std::invalid_argument(
                "Phi-4 AIE4 tensor dimensions must be positive");
        }
        const auto value = static_cast<std::size_t>(dimension);
        if (
            elements >
            std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error(
                "Phi-4 AIE4 tensor extent overflows size_t");
        }
        elements *= value;
    }

    std::size_t element_size = 0;
    switch (data_type) {
        case ryzenai_corelib_data_type_bf16:
        case ryzenai_corelib_data_type_fp16:
            element_size = sizeof(std::uint16_t);
            break;
        case ryzenai_corelib_data_type_fp32:
            element_size = sizeof(float);
            break;
        default:
            throw std::invalid_argument(
                "Phi-4 AIE4 requested an unsupported tensor dtype");
    }
    if (
        elements >
        std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error(
            "Phi-4 AIE4 tensor byte size overflows size_t");
    }
    return elements * element_size;
}

ryzenai_corelib_data_type SourceDataType(
    SourceDType data_type,
    std::string_view context) {
    switch (data_type) {
        case SourceDType::Float16:
            return ryzenai_corelib_data_type_fp16;
        case SourceDType::Float32:
            return ryzenai_corelib_data_type_fp32;
        case SourceDType::UInt8:
        case SourceDType::Int64:
            throw std::invalid_argument(
                std::string(context) +
                " requires an FP16 or FP32 source");
    }
    throw std::logic_error("unreachable Phi-4 source dtype");
}

std::uint64_t MappedSourceBytes(const Phi4Package& package) {
    std::unordered_set<const MappedFile*> owners;
    std::uint64_t total = 0;
    const auto add = [&](const InitializerView& view) {
        if (!view.owner || !owners.insert(view.owner.get()).second) {
            return;
        }
        if (
            view.owner->size() >
            std::numeric_limits<std::uint64_t>::max() - total) {
            throw std::overflow_error(
                "Phi-4 mapped-source byte total overflows uint64_t");
        }
        total += view.owner->size();
    };

    add(package.Require(kEmbeddingName));
    add(package.Require(kInputNormName));
    add(package.Require(kCosName));
    add(package.Require(kSinName));
    for (const auto& object : package.weight_objects()) {
        for (const auto& [_, initializer] : object.components) {
            add(package.Require(initializer));
        }
    }
    return total;
}

std::uint64_t ElapsedNanoseconds(
    std::chrono::steady_clock::time_point started) {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
    return static_cast<std::uint64_t>(
        std::max<std::int64_t>(elapsed.count(), 0));
}

bool RecoverableBeforeSubmit(
    bool synchronize_in_progress,
    const corelib::StepSubmissionState& submission) noexcept {
    return !synchronize_in_progress && !submission.irrevocable();
}

[[noreturn]] void TerminateCorelibFailure(
    const std::shared_ptr<corelib::CorelibRuntime>& runtime,
    const corelib::CorelibError& error,
    std::string phase,
    std::optional<int> layer,
    std::int64_t rows,
    std::int64_t position) {
    runtime->TerminateAfterFailure(corelib::FailureContext{
        error.status,
        error.call,
        error.detail,
        std::move(phase),
        layer,
        rows,
        position});
}

[[noreturn]] void TerminateHostFailure(
    const std::shared_ptr<corelib::CorelibRuntime>& runtime,
    const std::exception& error,
    std::string phase,
    std::optional<int> layer,
    std::int64_t rows,
    std::int64_t position) {
    runtime->TerminateAfterFailure(corelib::FailureContext{
        ryzenai_corelib_status_failure,
        "host_exception",
        error.what(),
        std::move(phase),
        layer,
        rows,
        position});
}

[[noreturn]] void TerminateUnknownFailure(
    const std::shared_ptr<corelib::CorelibRuntime>& runtime,
    std::string phase,
    std::optional<int> layer,
    std::int64_t rows,
    std::int64_t position) {
    runtime->TerminateAfterFailure(corelib::FailureContext{
        ryzenai_corelib_status_failure,
        "unknown_exception",
        "non-standard exception after the irrevocable boundary",
        std::move(phase),
        layer,
        rows,
        position});
}

}  // namespace

struct phi4_corelib_aie4::Impl final {
    Impl(
        LM_Config config,
        std::filesystem::path model_path,
        std::shared_ptr<corelib::CorelibRuntime> supplied_runtime,
        std::uint32_t requested_max_length)
        : runtime(std::move(supplied_runtime)),
          max_length(ValidateMaxLength(requested_max_length)) {
        ValidateConfigIdentity(config);
        if (!runtime) {
            throw std::invalid_argument(
                "phi4_corelib_aie4 requires a CorelibRuntime");
        }
        api = runtime->api();
        if (!api) {
            throw std::invalid_argument(
                "phi4_corelib_aie4 runtime has no CorelibApi");
        }
        if (model_path.empty()) {
            throw std::invalid_argument(
                "phi4_corelib_aie4 requires a model path");
        }

        const auto model_load_started =
            std::chrono::steady_clock::now();
        auto execution = runtime->AcquireExecution();

        package = std::make_shared<Phi4Package>(
            Phi4Package::Load(model_path, api, false));
        metrics.mapped_source_bytes =
            MappedSourceBytes(*package);
        shape_plan.emplace(Phi4ShapePlan::Build(api));
        metrics.helper_transition_counts = {
            static_cast<std::uint32_t>(
                shape_plan->Transitions(
                    RowUse::QueryProjection).size()),
            static_cast<std::uint32_t>(
                shape_plan->Transitions(
                    RowUse::KvProjection).size()),
            static_cast<std::uint32_t>(
                shape_plan->Transitions(
                    RowUse::Attention).size()),
            static_cast<std::uint32_t>(
                shape_plan->Transitions(
                    RowUse::OutputProjection).size()),
            static_cast<std::uint32_t>(
                shape_plan->Transitions(
                    RowUse::SsMlp).size()),
            static_cast<std::uint32_t>(
                shape_plan->Transitions(
                    RowUse::LmHead).size()),
        };

        const auto& embedding_view =
            package->Require(kEmbeddingName);
        embedding = std::span<const std::uint16_t>(
            reinterpret_cast<const std::uint16_t*>(
                embedding_view.data),
            embedding_view.size / sizeof(std::uint16_t));

        // The layer-0 norm feeds the host RMSNorm directly and never
        // reaches a tensor, so widening it is the `API-6` FP16-to-FP32
        // helper; an FP32 source is a plain copy.
        const auto& norm_view = package->Require(kInputNormName);
        input_norm.resize(
            static_cast<std::size_t>(constants::kHiddenSize));
        if (
            SourceDataType(norm_view.dtype, kInputNormName) ==
            ryzenai_corelib_data_type_fp32) {
            std::copy_n(
                reinterpret_cast<const float*>(norm_view.data),
                input_norm.size(),
                input_norm.begin());
        } else {
            corelib::WidenFp16Array(
                reinterpret_cast<const std::uint16_t*>(norm_view.data),
                input_norm.size(),
                input_norm.data());
        }

        const auto cos_host =
            package->MaterializeRopeGather(kCosName);
        const auto sin_host =
            package->MaterializeRopeGather(kSinName);

        const auto weight_pack_started =
            std::chrono::steady_clock::now();
        weights.emplace(Phi4Weights::Load(api, package));
        metrics.weight_pack_ns =
            ElapsedNanoseconds(weight_pack_started);
        metrics.packed_weight_bytes =
            static_cast<std::uint64_t>(weights->packed_bytes());
        metrics.weight_create_count =
            static_cast<std::uint64_t>(
                constants::kLayerCount * 5 + 1);

        const auto& capacities = shape_plan->capacities();
        const std::size_t layer_elements = CheckedElements(
            capacities.layer_rows,
            constants::kHiddenSize,
            "Phi-4 host layer staging");
        embedding_fp32.resize(layer_elements);
        normalized_fp32.resize(layer_elements);
        fp32_staging.resize(layer_elements);
        const std::size_t lm_head_elements = CheckedElements(
            capacities.lm_head_rows,
            constants::kHiddenSize,
            "Phi-4 LM-head padding staging");
        padding_zero_staging.resize(
            std::max(layer_elements, lm_head_elements),
            0);
        v_staging.reserve(
            CheckedElements(
                capacities.layer_rows,
                constants::kKvDimension + constants::kHeadSize,
                "Phi-4 V staging"));
        last_hidden_staging.resize(
            static_cast<std::size_t>(constants::kHiddenSize));

        stream = CreateStream();
        hidden_tensors[0] = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.layer_rows, constants::kHiddenSize},
            false);
        hidden_tensors[1] = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.layer_rows, constants::kHiddenSize},
            false);
        residual_tensor = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.layer_rows, constants::kHiddenSize},
            false);
        skip_sum_tensor = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.layer_rows, constants::kHiddenSize},
            false);
        query_tensor = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.layer_rows, constants::kQueryDimension},
            false);
        key_tensor = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.layer_rows, constants::kKvDimension},
            false);
        value_tensor = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.layer_rows, constants::kKvDimension},
            false);
        attention_tensor = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.layer_rows, constants::kQueryDimension},
            false);
        lm_input_tensor = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.lm_head_rows, constants::kHiddenSize},
            false);
        lm_output_tensor = CreateTensor(
            ryzenai_corelib_data_type_bf16,
            {capacities.lm_head_rows, constants::kVocabularySize},
            false);
        cos_tensor = CreateTensor(
            ryzenai_corelib_data_type_fp32,
            {constants::kMaxSequenceLength,
             constants::kRopeDimension / 2},
            false);
        sin_tensor = CreateTensor(
            ryzenai_corelib_data_type_fp32,
            {constants::kMaxSequenceLength,
             constants::kRopeDimension / 2},
            false);

        for (std::size_t layer = 0;
             layer <
             static_cast<std::size_t>(constants::kLayerCount);
             ++layer) {
            key_caches[layer] = CreateTensor(
                ryzenai_corelib_data_type_bf16,
                {constants::kKvHeadCount,
                 constants::kMaxSequenceLength,
                 constants::kHeadSize},
                true);
            value_caches[layer] = CreateTensor(
                ryzenai_corelib_data_type_bf16,
                {constants::kKvHeadCount,
                 constants::kMaxSequenceLength,
                 constants::kHeadSize},
                true);
        }

        // One write per table, in the source dtype, with `count` in FP32
        // elements of the destination tensor. tensor_write is the only
        // conversion boundary corelib now offers.
        const std::size_t rope_elements = CheckedElements(
            constants::kMaxSequenceLength,
            constants::kRopeDimension / 2,
            "Phi-4 RoPE upload");
        if (
            cos_host.count != rope_elements ||
            sin_host.count != rope_elements) {
            throw std::logic_error(
                "Phi-4 RoPE gather produced the wrong element count");
        }
        api->WriteElements(
            cos_tensor.get(),
            cos_host.dtype,
            cos_host.data,
            rope_elements,
            0);
        api->WriteElements(
            sin_tensor.get(),
            sin_host.dtype,
            sin_host.data,
            rope_elements,
            0);

        current_hidden = &hidden_tensors[0];
        next_hidden = &hidden_tensors[1];
        current_residual = &residual_tensor;
        next_skip_sum = &skip_sum_tensor;
        metrics.model_load_ns =
            ElapsedNanoseconds(model_load_started);
    }

    ~Impl() noexcept {
        if (!stream) {
            ReleaseResourcesWithoutSynchronization();
            return;
        }

        try {
            auto execution = runtime->AcquireExecution();
            api->Check(
                api->functions().stream_synchronize(stream.get()),
                kSynchronizeCall);
            stream.reset();
            ReleaseTensorsWeightsAndPackage();
        } catch (const corelib::CorelibError& error) {
            TerminateCorelibFailure(
                runtime,
                error,
                "destruction",
                std::nullopt,
                last_live_rows,
                position);
        } catch (const std::exception& error) {
            TerminateHostFailure(
                runtime,
                error,
                "destruction",
                std::nullopt,
                last_live_rows,
                position);
        } catch (...) {
            TerminateUnknownFailure(
                runtime,
                "destruction",
                std::nullopt,
                last_live_rows,
                position);
        }
    }

    corelib::UniqueStream CreateStream() {
        ryzenai_corelib_stream_ptr raw = nullptr;
        const auto status =
            api->functions().create_stream(&raw);
        corelib::UniqueStream result(api, raw);
        api->Check(status, kCreateStreamCall);
        if (!result) {
            throw std::runtime_error(
                "ryzenai_corelib_create_stream succeeded with a "
                "null object");
        }
        return result;
    }

    corelib::UniqueTensor CreateTensor(
        ryzenai_corelib_data_type data_type,
        std::initializer_list<std::int64_t> shape_values,
        bool is_kv) {
        const std::vector<std::int64_t> shape(shape_values);
        ryzenai_corelib_tensor_ptr raw = nullptr;
        const auto status =
            api->functions().create_device_tensor(
                data_type,
                shape.data(),
                shape.size(),
                &raw);
        corelib::UniqueTensor result(api, raw);
        api->Check(status, kCreateTensorCall);
        if (!result) {
            throw std::runtime_error(
                "ryzenai_corelib_create_device_tensor succeeded "
                "with a null object");
        }

        std::size_t actual_bytes = 0;
        api->Check(
            api->functions().tensor_get_byte_size(
                result.get(),
                &actual_bytes),
            kTensorByteSizeCall);
        const std::size_t expected_bytes =
            TensorByteCount(data_type, shape);
        if (actual_bytes != expected_bytes) {
            throw std::runtime_error(
                "corelib device tensor byte size does not match "
                "the requested Phi-4 shape");
        }

        // Every subsequent write and read counts in elements of THIS
        // dtype, so confirm the tensor holds what was asked for rather
        // than inferring it from the byte size, which FP16 and BF16 share.
        ryzenai_corelib_data_type actual_type{};
        api->Check(
            api->functions().tensor_get_data_type(
                result.get(),
                &actual_type),
            kTensorDataTypeCall);
        if (actual_type != data_type) {
            throw std::runtime_error(
                "corelib device tensor dtype does not match the "
                "requested Phi-4 dtype");
        }
        ++metrics.device_tensor_create_count;
        if (is_kv) {
            metrics.kv_bytes += actual_bytes;
        } else {
            metrics.scratch_bytes += actual_bytes;
        }
        return result;
    }

    void ReleaseResourcesWithoutSynchronization() noexcept {
        stream.reset();
        ReleaseTensorsWeightsAndPackage();
    }

    void ReleaseTensorsWeightsAndPackage() noexcept {
        for (auto layer = value_caches.rbegin();
             layer != value_caches.rend();
             ++layer) {
            layer->reset();
        }
        for (auto layer = key_caches.rbegin();
             layer != key_caches.rend();
             ++layer) {
            layer->reset();
        }
        sin_tensor.reset();
        cos_tensor.reset();
        lm_output_tensor.reset();
        lm_input_tensor.reset();
        attention_tensor.reset();
        value_tensor.reset();
        key_tensor.reset();
        query_tensor.reset();
        skip_sum_tensor.reset();
        residual_tensor.reset();
        hidden_tensors[1].reset();
        hidden_tensors[0].reset();
        weights.reset();
        embedding = {};
        package.reset();
    }

    void ValidateTokens(std::span<const int> token_ids) const {
        for (const int token_id : token_ids) {
            if (
                token_id < 0 ||
                token_id >= constants::kVocabularySize) {
                throw std::out_of_range(
                    "Phi-4 token ID is outside the vocabulary");
            }
        }
    }

    void EnsureCapacity(std::size_t token_count) const {
        if (token_count == 0) {
            throw std::invalid_argument(
                "Phi-4 prefill requires at least one token");
        }
        const auto remaining =
            static_cast<std::uint64_t>(max_length) -
            static_cast<std::uint64_t>(position);
        if (token_count > remaining) {
            throw std::out_of_range(
                "Phi-4 AIE4 maximum context length would be "
                "exceeded");
        }
    }

    struct RunRowExtents {
        std::int64_t query_projection;
        std::int64_t kv_projection;
        std::int64_t attention;
        std::int64_t output_projection;
        std::int64_t ssmlp;
        std::int64_t lm_head;

        std::int64_t ProjectionInput() const noexcept {
            return std::max(query_projection, kv_projection);
        }
    };

    RunRowExtents RowsForRun(std::int64_t rows) {
        RunRowExtents extents{
            shape_plan->RowsFor(RowUse::QueryProjection, rows),
            shape_plan->RowsFor(RowUse::KvProjection, rows),
            shape_plan->RowsFor(RowUse::Attention, rows),
            shape_plan->RowsFor(RowUse::OutputProjection, rows),
            shape_plan->RowsFor(RowUse::SsMlp, rows),
            shape_plan->RowsFor(RowUse::LmHead, 1)};
        ++metrics.attention_extent_queries;
        ++metrics.output_projection_extent_queries;
        ++metrics.lm_head_extent_queries;
        return extents;
    }

    void WriteZeroRows(
        corelib::UniqueTensor& tensor,
        std::int64_t first_row,
        std::int64_t row_count,
        std::int64_t width,
        std::string_view context) {
        if (row_count <= 0) {
            return;
        }
        if (first_row < 0) {
            throw std::out_of_range(
                "Phi-4 padding write has a negative row offset");
        }
        // Offsets and counts are BF16 elements of the destination tensor,
        // not bytes; the staging buffer is BF16 too, so this is a copy.
        const std::size_t offset =
            first_row == 0
                ? 0
                : CheckedElements(first_row, width, context);
        const std::size_t words = CheckedElements(
            row_count,
            width,
            context);
        if (words > padding_zero_staging.size()) {
            throw std::out_of_range(
                "Phi-4 padding write exceeds host staging capacity");
        }
        api->WriteElements(
            tensor.get(),
            ryzenai_corelib_data_type_bf16,
            padding_zero_staging.data(),
            words,
            offset);
        ++metrics.padding_write_calls;
        metrics.padding_bytes += words * sizeof(std::uint16_t);
    }

    void BridgePadding(
        corelib::UniqueTensor& tensor,
        std::int64_t producer_rows,
        std::int64_t consumer_rows,
        std::int64_t width,
        std::string_view context) {
        if (consumer_rows > producer_rows) {
            WriteZeroRows(
                tensor,
                producer_rows,
                consumer_rows - producer_rows,
                width,
                context);
        }
    }

    void StageInput(
        std::span<const int> token_ids,
        const RunRowExtents& extents) {
        const auto rows =
            static_cast<std::int64_t>(token_ids.size());
        const std::size_t live_elements = CheckedElements(
            rows,
            constants::kHiddenSize,
            "Phi-4 input staging");
        if (
            live_elements > embedding_fp32.size() ||
            live_elements > normalized_fp32.size()) {
            throw std::out_of_range(
                "Phi-4 input exceeds the planned host capacity");
        }

        const auto embedding_output =
            std::span<float>(embedding_fp32).first(live_elements);
        const auto normalized_output =
            std::span<float>(normalized_fp32).first(live_elements);
        GatherEmbedding(
            embedding,
            token_ids,
            embedding_output);
        RmsNorm(
            embedding_output,
            input_norm,
            rows,
            constants::kHiddenSize,
            static_cast<float>(constants::kRmsEpsilon),
            normalized_output);

        // Design Section 10.2: the host stays in FP32 and writes FP32 into
        // the BF16 tensors. Corelib narrows inside tensor_write, so there
        // is one BF16 rounding implementation on this path, not two that
        // have to agree. `count` is in BF16 elements of the destination.
        const std::int64_t hidden_rows =
            extents.ProjectionInput();
        StageFp32(
            normalized_output,
            rows,
            hidden_rows,
            constants::kHiddenSize,
            fp32_staging);
        const std::size_t hidden_elements = CheckedElements(
            hidden_rows,
            constants::kHiddenSize,
            "Phi-4 hidden staging");
        api->WriteElements(
            current_hidden->get(),
            ryzenai_corelib_data_type_fp32,
            fp32_staging.data(),
            hidden_elements,
            0);

        const std::int64_t residual_rows = extents.ssmlp;
        StageFp32(
            embedding_output,
            rows,
            residual_rows,
            constants::kHiddenSize,
            fp32_staging);
        const std::size_t residual_elements = CheckedElements(
            residual_rows,
            constants::kHiddenSize,
            "Phi-4 residual staging");
        api->WriteElements(
            current_residual->get(),
            ryzenai_corelib_data_type_fp32,
            fp32_staging.data(),
            residual_elements,
            0);
    }

    ryzenai_corelib_status SubmitMatMul(
        const corelib::UniqueTensor& input,
        const corelib::UniqueMatMulWeights& operation_weights,
        corelib::UniqueTensor& output,
        std::int64_t rows) {
        return api->functions().matmul(
            stream.get(),
            input.get(),
            rows,
            operation_weights.get(),
            output.get());
    }

    ryzenai_corelib_status SubmitMha(
        std::size_t layer,
        std::int64_t rows,
        std::int64_t current_position) {
        return api->functions().flat_mha(
            stream.get(),
            &shape_plan->attention_desc(),
            query_tensor.get(),
            key_tensor.get(),
            rows,
            current_position,
            cos_tensor.get(),
            sin_tensor.get(),
            key_caches[layer].get(),
            value_caches[layer].get(),
            attention_tensor.get());
    }

    ryzenai_corelib_status SubmitSsMlp(
        const corelib::UniqueTensor& input,
        const corelib::UniqueTensor& residual,
        const corelib::UniqueSsMlpWeights& operation_weights,
        corelib::UniqueTensor& skip_sum,
        corelib::UniqueTensor& normalized,
        std::int64_t rows) {
        return api->functions().ssmlp(
            stream.get(),
            input.get(),
            residual.get(),
            rows,
            operation_weights.get(),
            skip_sum.get(),
            normalized.get());
    }

    void ScatterValue(
        std::size_t layer,
        std::int64_t rows,
        std::int64_t current_position) {
        flm::phi4::ScatterV(
            *api,
            value_tensor.get(),
            value_caches[layer].get(),
            rows,
            current_position,
            v_staging,
            v_metrics);
        metrics.v_read_calls = v_metrics.read_calls;
        metrics.v_write_calls = v_metrics.write_calls;
        metrics.v_bytes = v_metrics.bytes;
        metrics.v_scatter_ns = v_metrics.nanoseconds;
    }

    void PrepareLastHidden(
        std::int64_t rows,
        std::int64_t lm_head_rows) {
        // Design Section 10.5: both tensors are BF16 and so is the caller
        // dtype, so this is a straight copy with no FP32 round trip.
        const std::size_t row_elements = CheckedElements(
            1,
            constants::kHiddenSize,
            "Phi-4 last hidden");
        const std::size_t source_offset =
            static_cast<std::size_t>(rows - 1) * row_elements;
        WriteZeroRows(
            lm_input_tensor,
            0,
            lm_head_rows,
            constants::kHiddenSize,
            "Phi-4 LM-head input initialization");
        api->ReadElements(
            current_hidden->get(),
            ryzenai_corelib_data_type_bf16,
            last_hidden_staging.data(),
            row_elements,
            source_offset);
        api->WriteElements(
            lm_input_tensor.get(),
            ryzenai_corelib_data_type_bf16,
            last_hidden_staging.data(),
            row_elements,
            0);
    }

    void ReadLogits(buffer<bf16>& output) {
        constexpr std::size_t logits_elements =
            static_cast<std::size_t>(constants::kVocabularySize);
        if (output.size() != logits_elements) {
            throw std::logic_error(
                "Phi-4 logits buffer has the wrong size");
        }
        // Straight into the returned buffer<bf16>: the return type is
        // already BF16, so widening and narrowing again would be a round
        // trip for nothing.
        api->ReadElements(
            lm_output_tensor.get(),
            ryzenai_corelib_data_type_bf16,
            output.data(),
            logits_elements,
            0);
    }

    buffer<bf16> RunRows(std::span<const int> token_ids) {
        const std::int64_t rows =
            static_cast<std::int64_t>(token_ids.size());
        auto execution = runtime->AcquireExecution();
        corelib::StepSubmissionState submission;
        std::optional<int> active_layer;
        std::string active_phase = "stage_input";
        bool synchronize_in_progress = false;
        buffer<bf16> logits(
            static_cast<std::size_t>(constants::kVocabularySize));

        try {
            auto checked_submit = [&](
                                      ryzenai_corelib_status status,
                                      std::string_view call) {
                api->Check(status, call);
                submission.MarkSuccessfulSubmit();
                ++metrics.dispatch_count;
            };
            auto checked_synchronize = [&] {
                synchronize_in_progress = true;
                api->Check(
                    api->functions().stream_synchronize(
                        stream.get()),
                    kSynchronizeCall);
                synchronize_in_progress = false;
                ++metrics.synchronize_count;
            };

            const RunRowExtents extents = RowsForRun(rows);
            StageInput(token_ids, extents);

            for (int layer = 0;
                 layer < constants::kLayerCount;
                 ++layer) {
                active_layer = layer;
                active_phase = "qkv";
                const auto& layer_weights =
                    weights->layers()[static_cast<std::size_t>(layer)];
                checked_submit(
                    SubmitMatMul(
                        *current_hidden,
                        layer_weights.q,
                        query_tensor,
                        rows),
                    "q");
                checked_submit(
                    SubmitMatMul(
                        *current_hidden,
                        layer_weights.k,
                        key_tensor,
                        rows),
                    "k");
                checked_submit(
                    SubmitMatMul(
                        *current_hidden,
                        layer_weights.v,
                        value_tensor,
                        rows),
                    "v");
                checked_synchronize();

                active_phase = "v_scatter";
                ScatterValue(
                    static_cast<std::size_t>(layer),
                    rows,
                    position);

                active_phase = "flat_mha";
                BridgePadding(
                    query_tensor,
                    extents.query_projection,
                    extents.attention,
                    constants::kQueryDimension,
                    "Phi-4 query-to-attention padding");
                BridgePadding(
                    key_tensor,
                    extents.kv_projection,
                    extents.attention,
                    constants::kKvDimension,
                    "Phi-4 key-to-attention padding");
                checked_submit(
                    SubmitMha(
                        static_cast<std::size_t>(layer),
                        rows,
                        position),
                    "flat_mha");
                checked_synchronize();

                active_phase = "o";
                BridgePadding(
                    attention_tensor,
                    extents.attention,
                    extents.output_projection,
                    constants::kQueryDimension,
                    "Phi-4 attention-to-output padding");
                checked_submit(
                    SubmitMatMul(
                        attention_tensor,
                        layer_weights.o,
                        *current_hidden,
                        rows),
                    "o");
                checked_synchronize();

                active_phase = "ssmlp";
                BridgePadding(
                    *current_hidden,
                    extents.output_projection,
                    extents.ssmlp,
                    constants::kHiddenSize,
                    "Phi-4 output-to-SSMLP padding");
                checked_submit(
                    SubmitSsMlp(
                        *current_hidden,
                        *current_residual,
                        layer_weights.mlp,
                        *next_skip_sum,
                        *next_hidden,
                        rows),
                    "ssmlp");
                checked_synchronize();
                if (layer + 1 < constants::kLayerCount) {
                    active_phase = "next_layer_padding";
                    BridgePadding(
                        *next_hidden,
                        extents.ssmlp,
                        extents.ProjectionInput(),
                        constants::kHiddenSize,
                        "Phi-4 SSMLP-to-projection padding");
                }
                std::swap(current_hidden, next_hidden);
                std::swap(current_residual, next_skip_sum);
            }

            active_layer.reset();
            active_phase = "lm_head";
            PrepareLastHidden(rows, extents.lm_head);
            checked_submit(
                SubmitMatMul(
                    lm_input_tensor,
                    weights->lm_head(),
                    lm_output_tensor,
                    1),
                "lm_head");
            checked_synchronize();
            ReadLogits(logits);
        } catch (const corelib::CorelibError& error) {
            if (RecoverableBeforeSubmit(
                    synchronize_in_progress,
                    submission)) {
                throw;
            }
            TerminateCorelibFailure(
                runtime,
                error,
                active_phase,
                active_layer,
                rows,
                position);
        } catch (const std::exception& error) {
            if (RecoverableBeforeSubmit(
                    synchronize_in_progress,
                    submission)) {
                throw;
            }
            TerminateHostFailure(
                runtime,
                error,
                active_phase,
                active_layer,
                rows,
                position);
        } catch (...) {
            if (RecoverableBeforeSubmit(
                    synchronize_in_progress,
                    submission)) {
                throw;
            }
            TerminateUnknownFailure(
                runtime,
                active_phase,
                active_layer,
                rows,
                position);
        }

        position += rows;
        last_live_rows = rows;
        return logits;
    }

    std::vector<std::uint16_t> ReadLiveCache(
        const corelib::UniqueTensor& cache) const {
        if (position == 0) {
            return {};
        }
        const std::size_t live_rows =
            static_cast<std::size_t>(position);
        const std::size_t head_width =
            static_cast<std::size_t>(constants::kHeadSize);
        std::vector<std::uint16_t> result(
            static_cast<std::size_t>(constants::kKvHeadCount) *
            live_rows * head_width);
        const std::size_t elements_per_head = live_rows * head_width;
        for (std::size_t head = 0;
             head <
             static_cast<std::size_t>(constants::kKvHeadCount);
             ++head) {
            const std::size_t source_offset =
                head *
                static_cast<std::size_t>(
                    constants::kMaxSequenceLength) *
                head_width;
            api->ReadElements(
                cache.get(),
                ryzenai_corelib_data_type_bf16,
                result.data() + head * elements_per_head,
                elements_per_head,
                source_offset);
        }
        return result;
    }

#ifdef DEV_BUILD
    Phi4DebugSnapshot DebugSnapshot() const {
        auto execution = runtime->AcquireExecution();
        Phi4DebugSnapshot snapshot;
        snapshot.live_rows = last_live_rows;
        snapshot.position = position;
        snapshot.layer0_k = ReadLiveCache(key_caches.front());
        snapshot.layer0_v = ReadLiveCache(value_caches.front());
        snapshot.layer31_k = ReadLiveCache(key_caches.back());
        snapshot.layer31_v = ReadLiveCache(value_caches.back());
        snapshot.last_hidden.resize(
            static_cast<std::size_t>(constants::kHiddenSize));
        snapshot.logits.resize(
            static_cast<std::size_t>(constants::kVocabularySize));
        api->ReadElements(
            lm_input_tensor.get(),
            ryzenai_corelib_data_type_bf16,
            snapshot.last_hidden.data(),
            snapshot.last_hidden.size(),
            0);
        api->ReadElements(
            lm_output_tensor.get(),
            ryzenai_corelib_data_type_bf16,
            snapshot.logits.data(),
            snapshot.logits.size(),
            0);
        return snapshot;
    }
#endif

    std::shared_ptr<corelib::CorelibRuntime> runtime;
    std::shared_ptr<corelib::CorelibApi> api;
    std::optional<Phi4ShapePlan> shape_plan;
    std::shared_ptr<Phi4Package> package;
    std::optional<Phi4Weights> weights;

    std::span<const std::uint16_t> embedding;
    std::vector<float> input_norm;
    std::vector<float> embedding_fp32;
    std::vector<float> normalized_fp32;
    std::vector<float> fp32_staging;
    std::vector<std::uint16_t> padding_zero_staging;
    std::vector<std::uint16_t> v_staging;
    std::vector<std::uint16_t> last_hidden_staging;

    std::array<corelib::UniqueTensor, 2> hidden_tensors;
    corelib::UniqueTensor residual_tensor;
    corelib::UniqueTensor skip_sum_tensor;
    corelib::UniqueTensor query_tensor;
    corelib::UniqueTensor key_tensor;
    corelib::UniqueTensor value_tensor;
    corelib::UniqueTensor attention_tensor;
    corelib::UniqueTensor lm_input_tensor;
    corelib::UniqueTensor lm_output_tensor;
    corelib::UniqueTensor cos_tensor;
    corelib::UniqueTensor sin_tensor;
    std::array<
        corelib::UniqueTensor,
        static_cast<std::size_t>(constants::kLayerCount)>
        key_caches;
    std::array<
        corelib::UniqueTensor,
        static_cast<std::size_t>(constants::kLayerCount)>
        value_caches;
    corelib::UniqueTensor* current_hidden = nullptr;
    corelib::UniqueTensor* next_hidden = nullptr;
    corelib::UniqueTensor* current_residual = nullptr;
    corelib::UniqueTensor* next_skip_sum = nullptr;

    Phi4Aie4Metrics metrics;
    VScatterMetrics v_metrics;
    std::uint32_t max_length;
    std::int64_t position = 0;
    std::int64_t last_live_rows = 0;
    std::optional<std::int64_t> checkpoint_position;

    // Declared last so constructor rollback also releases the Stream before
    // tensors, weights, and package storage.
    corelib::UniqueStream stream;
};

phi4_corelib_aie4::phi4_corelib_aie4(
    LM_Config config,
    std::filesystem::path model_path,
    std::shared_ptr<corelib::CorelibRuntime> runtime,
    std::uint32_t max_length)
    : impl_(std::make_unique<Impl>(
          std::move(config),
          std::move(model_path),
          std::move(runtime),
          max_length)) {}

phi4_corelib_aie4::~phi4_corelib_aie4() = default;

buffer<bf16> phi4_corelib_aie4::forward(int id) {
    const std::array<int, 1> token{id};
    impl_->ValidateTokens(token);
    impl_->EnsureCapacity(token.size());
    return impl_->RunRows(token);
}

buffer<bf16> phi4_corelib_aie4::prefill(
    std::vector<int>& ids,
    void* payload) {
    (void)payload;
    const std::span<const int> token_ids(ids);
    impl_->ValidateTokens(token_ids);
    impl_->EnsureCapacity(token_ids.size());
    if (impl_->position == 0 || token_ids.size() == 1) {
        return impl_->RunRows(token_ids);
    }

    buffer<bf16> logits;
    for (const int token_id : token_ids) {
        const std::array<int, 1> token{token_id};
        logits = impl_->RunRows(token);
    }
    return logits;
}

void phi4_corelib_aie4::set_context_length(int length) {
    if (length != impl_->position) {
        throw std::invalid_argument(
            "phi4_corelib_aie4 set_context_length accepts only "
            "the current logical position");
    }
}

void phi4_corelib_aie4::load_weights(Q4NX& q4nx) {
    (void)q4nx;
    throw std::runtime_error(
        "Q4NX weight loading is unsupported by phi4_corelib_aie4");
}

void phi4_corelib_aie4::update_max_length(
    std::uint32_t max_length) {
    const std::uint32_t validated =
        ValidateMaxLength(max_length);
    if (
        validated <
        static_cast<std::uint32_t>(impl_->position)) {
        throw std::out_of_range(
            "Phi-4 AIE4 maximum length cannot be below the "
            "current logical position");
    }
    impl_->max_length = validated;
}

void phi4_corelib_aie4::clear_context() {
    impl_->position = 0;
    impl_->last_live_rows = 0;
    impl_->checkpoint_position.reset();
}

buffer<bf16> phi4_corelib_aie4::get_k_cache(
    int layer,
    int index) {
    (void)layer;
    (void)index;
    throw std::runtime_error(
        "K-cache getters are unsupported by phi4_corelib_aie4");
}

buffer<bf16> phi4_corelib_aie4::get_v_cache(
    int layer,
    int index) {
    (void)layer;
    (void)index;
    throw std::runtime_error(
        "V-cache getters are unsupported by phi4_corelib_aie4");
}

int phi4_corelib_aie4::get_current_context_length() {
    return static_cast<int>(impl_->position);
}

int phi4_corelib_aie4::checkpoint() {
    impl_->checkpoint_position = impl_->position;
    return static_cast<int>(impl_->position);
}

int phi4_corelib_aie4::restore() {
    if (!impl_->checkpoint_position.has_value()) {
        throw std::logic_error(
            "phi4_corelib_aie4 has no checkpoint to restore");
    }
    impl_->position = *impl_->checkpoint_position;
    impl_->last_live_rows = 0;
    return static_cast<int>(impl_->position);
}

const Phi4Aie4Metrics&
phi4_corelib_aie4::metrics() const noexcept {
    return impl_->metrics;
}

#ifdef DEV_BUILD
Phi4DebugSnapshot phi4_corelib_aie4::debug_snapshot() const {
    return impl_->DebugSnapshot();
}
#endif

#if defined(FLM_CORELIB_TESTING)
namespace testing {

[[noreturn]] void ApplyCorelibFailurePolicyForTest(
    const std::shared_ptr<corelib::CorelibRuntime>& runtime,
    const corelib::CorelibError& error,
    bool synchronize_in_progress,
    const corelib::StepSubmissionState& submission,
    std::string phase,
    std::optional<int> layer,
    std::int64_t rows,
    std::int64_t position) {
    if (RecoverableBeforeSubmit(
            synchronize_in_progress,
            submission)) {
        throw error;
    }
    TerminateCorelibFailure(
        runtime,
        error,
        std::move(phase),
        layer,
        rows,
        position);
}

}  // namespace testing
#endif

}  // namespace flm::phi4
