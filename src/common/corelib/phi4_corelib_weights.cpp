#include <models/phi4/phi4_corelib_weights.hpp>

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flm::phi4 {
namespace {

constexpr std::string_view kConvertCall =
    "ryzenai_corelib_convert";
constexpr std::string_view kMatMulCreateCall =
    "ryzenai_corelib_matmul_bf16_weights_create_from_onnx_components";
constexpr std::string_view kMatMulGetDataCall =
    "ryzenai_corelib_matmul_bf16_weights_get_data";
constexpr std::string_view kSsMlpCreateCall =
    "ryzenai_corelib_ssmlp_bf16_weights_create_from_onnx_components";
constexpr std::string_view kSsMlpGetDataCall =
    "ryzenai_corelib_ssmlp_bf16_weights_get_data";

const std::string& ComponentName(
    const WeightObjectView& object,
    std::string_view role) {
    const auto component = object.components.find(std::string(role));
    if (component == object.components.end()) {
        throw std::runtime_error(
            "validated component role is missing: " +
            std::string(role));
    }
    return component->second;
}

void AddPackedBytes(
    std::size_t packed_size,
    std::size_t& total) {
    if (
        packed_size >
        std::numeric_limits<std::size_t>::max() - total) {
        throw std::overflow_error(
            "packed weight byte total overflows size_t");
    }
    total += packed_size;
}

[[noreturn]] void ThrowObjectError(
    const WeightObjectView& object,
    const corelib::CorelibError& error) {
    throw error.WithContext(
        "Phi-4 weight object '" + object.name + "'");
}

[[noreturn]] void ThrowObjectError(
    const WeightObjectView& object,
    const std::exception& error) {
    throw std::runtime_error(
        "failed to load Phi-4 weight object '" + object.name +
        "': " + error.what());
}

corelib::UniqueMatMulWeights CreateMatMul(
    const std::shared_ptr<const corelib::CorelibApi>& api,
    Phi4Package& package,
    const WeightObjectView& object,
    std::size_t& packed_bytes) {
    try {
        if (object.kind != WeightObjectKind::MatMul) {
            throw std::runtime_error(
                "validated weight object has the wrong kind");
        }

        const auto& qweight =
            package.Require(ComponentName(object, "qweight"));
        const auto scales =
            package.MaterializeFp16(
                ComponentName(object, "scales"));
        const auto& qzeros =
            package.Require(ComponentName(object, "qzeros"));

        const ryzenai_corelib_matmul_bf16_weights_desc descriptor{
            object.k,
            object.n,
            constants::kGroupSize,
            false};
        const ryzenai_corelib_matmul_bf16_onnx_weights_components
            components{
                qweight.data,
                scales.data(),
                qzeros.data};

        ryzenai_corelib_matmul_bf16_weights_ptr raw = nullptr;
        const auto status =
            api->functions().matmul_weights_from_onnx(
                &descriptor,
                &components,
                &raw);
        corelib::UniqueMatMulWeights weights(api, raw);
        api->Check(status, kMatMulCreateCall);
        if (!weights) {
            throw std::runtime_error(
                std::string(kMatMulCreateCall) +
                " succeeded with a null object");
        }

        std::size_t packed_size = 0;
        api->Check(
            api->functions().matmul_weights_get_data(
                weights.get(),
                nullptr,
                &packed_size),
            kMatMulGetDataCall);
        AddPackedBytes(packed_size, packed_bytes);
        return weights;
    } catch (const corelib::CorelibError& error) {
        ThrowObjectError(object, error);
    } catch (const std::exception& error) {
        ThrowObjectError(object, error);
    }
}

corelib::UniqueSsMlpWeights CreateSsMlp(
    const std::shared_ptr<const corelib::CorelibApi>& api,
    Phi4Package& package,
    const std::uint16_t* epsilon,
    const WeightObjectView& object,
    std::size_t& packed_bytes) {
    try {
        if (object.kind != WeightObjectKind::SsMlp) {
            throw std::runtime_error(
                "validated weight object has the wrong kind");
        }

        const auto norm0 =
            package.MaterializeBf16(
                ComponentName(object, "norm0"));
        const auto norm1 =
            package.MaterializeBf16(
                ComponentName(object, "norm1"));

        const auto& gate_qweight =
            package.Require(
                ComponentName(object, "gate_qweight"));
        const auto gate_scales =
            package.MaterializeFp16(
                ComponentName(object, "gate_scales"));
        const auto& gate_qzeros =
            package.Require(
                ComponentName(object, "gate_qzeros"));

        const auto& up_qweight =
            package.Require(
                ComponentName(object, "up_qweight"));
        const auto up_scales =
            package.MaterializeFp16(
                ComponentName(object, "up_scales"));
        const auto& up_qzeros =
            package.Require(
                ComponentName(object, "up_qzeros"));

        const auto& down_qweight =
            package.Require(
                ComponentName(object, "down_qweight"));
        const auto down_scales =
            package.MaterializeFp16(
                ComponentName(object, "down_scales"));
        const auto& down_qzeros =
            package.Require(
                ComponentName(object, "down_qzeros"));

        const ryzenai_corelib_ssmlp_bf16_weights_desc descriptor{
            object.k,
            object.n,
            constants::kGroupSize};
        const ryzenai_corelib_ssmlp_bf16_onnx_weights_components
            components{
                epsilon,
                norm0.data(),
                norm1.data(),
                gate_qweight.data,
                gate_scales.data(),
                gate_qzeros.data,
                up_qweight.data,
                up_scales.data(),
                up_qzeros.data,
                down_qweight.data,
                down_scales.data(),
                down_qzeros.data};

        ryzenai_corelib_ssmlp_bf16_weights_ptr raw = nullptr;
        const auto status =
            api->functions().ssmlp_weights_from_onnx(
                &descriptor,
                &components,
                &raw);
        corelib::UniqueSsMlpWeights weights(api, raw);
        api->Check(status, kSsMlpCreateCall);
        if (!weights) {
            throw std::runtime_error(
                std::string(kSsMlpCreateCall) +
                " succeeded with a null object");
        }

        std::size_t packed_size = 0;
        api->Check(
            api->functions().ssmlp_weights_get_data(
                weights.get(),
                nullptr,
                &packed_size),
            kSsMlpGetDataCall);
        AddPackedBytes(packed_size, packed_bytes);
        return weights;
    } catch (const corelib::CorelibError& error) {
        ThrowObjectError(object, error);
    } catch (const std::exception& error) {
        ThrowObjectError(object, error);
    }
}

}  // namespace

Phi4Weights Phi4Weights::Load(
    std::shared_ptr<const corelib::CorelibApi> api,
    std::shared_ptr<Phi4Package> package) {
    if (!api) {
        throw std::invalid_argument(
            "Phi4Weights::Load requires a CorelibApi");
    }
    if (!package) {
        throw std::invalid_argument(
            "Phi4Weights::Load requires a Phi4Package");
    }

    Phi4Weights result;
    result.package_ = std::move(package);

    auto epsilon = std::make_shared<std::uint16_t>();
    const float epsilon_fp32 =
        static_cast<float>(constants::kRmsEpsilon);
    try {
        api->Check(
            api->functions().convert(
                ryzenai_corelib_data_type_fp32,
                &epsilon_fp32,
                ryzenai_corelib_data_type_bf16,
                epsilon.get(),
                1),
            kConvertCall);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "failed to materialize Phi-4 RMS epsilon: " +
            std::string(error.what()));
    }
    result.epsilon_bf16_ = std::move(epsilon);

    const auto& objects = result.package_->weight_objects();
    constexpr std::size_t objects_per_layer = 5;
    constexpr std::size_t expected_objects =
        static_cast<std::size_t>(constants::kLayerCount) *
            objects_per_layer +
        1;
    if (objects.size() != expected_objects) {
        throw std::runtime_error(
            "Phi4Package must contain exactly 161 validated "
            "weight objects");
    }

    for (std::size_t layer = 0;
         layer <
         static_cast<std::size_t>(constants::kLayerCount);
         ++layer) {
        const std::size_t base = layer * objects_per_layer;
        auto& destination = result.layers_[layer];
        destination.q = CreateMatMul(
            api,
            *result.package_,
            objects[base],
            result.packed_bytes_);
        destination.k = CreateMatMul(
            api,
            *result.package_,
            objects[base + 1],
            result.packed_bytes_);
        destination.v = CreateMatMul(
            api,
            *result.package_,
            objects[base + 2],
            result.packed_bytes_);
        destination.o = CreateMatMul(
            api,
            *result.package_,
            objects[base + 3],
            result.packed_bytes_);
        destination.mlp = CreateSsMlp(
            api,
            *result.package_,
            result.epsilon_bf16_.get(),
            objects[base + 4],
            result.packed_bytes_);
    }

    result.lm_head_ = CreateMatMul(
        api,
        *result.package_,
        objects.back(),
        result.packed_bytes_);
    return result;
}

Phi4Weights& Phi4Weights::operator=(Phi4Weights&& other) noexcept {
    if (this != &other) {
        ResetWeightObjects();
        epsilon_bf16_.reset();
        package_.reset();

        package_ = std::move(other.package_);
        epsilon_bf16_ = std::move(other.epsilon_bf16_);
        layers_ = std::move(other.layers_);
        lm_head_ = std::move(other.lm_head_);
        packed_bytes_ = std::exchange(other.packed_bytes_, 0);
    }
    return *this;
}

const std::array<LayerWeights, constants::kLayerCount>&
Phi4Weights::layers() const noexcept {
    return layers_;
}

const corelib::UniqueMatMulWeights&
Phi4Weights::lm_head() const noexcept {
    return lm_head_;
}

std::size_t Phi4Weights::packed_bytes() const noexcept {
    return packed_bytes_;
}

void Phi4Weights::ResetWeightObjects() noexcept {
    lm_head_.reset();
    for (auto layer = layers_.rbegin();
         layer != layers_.rend();
         ++layer) {
        layer->mlp.reset();
        layer->o.reset();
        layer->v.reset();
        layer->k.reset();
        layer->q.reset();
    }
}

}  // namespace flm::phi4
