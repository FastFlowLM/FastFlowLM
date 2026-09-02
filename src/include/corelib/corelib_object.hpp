#pragma once

#include <corelib/corelib_api.hpp>

#include <cassert>
#include <memory>
#include <utility>

namespace flm::corelib {

template <class Tag>
class UniqueObject {
public:
    UniqueObject() noexcept = default;

    UniqueObject(
        std::shared_ptr<const CorelibApi> api,
        void* value) noexcept
        : api_(std::move(api)),
          value_(value) {
        if (value_ != nullptr) {
            assert(api_ != nullptr);
            api_->RegisterObject(Tag::kKind);
        }
    }

    UniqueObject(UniqueObject&& other) noexcept
        : api_(std::move(other.api_)),
          value_(std::exchange(other.value_, nullptr)) {}

    UniqueObject& operator=(UniqueObject&& other) noexcept {
        if (this != &other) {
            reset();
            api_ = std::move(other.api_);
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }

    ~UniqueObject() noexcept {
        reset();
    }

    UniqueObject(const UniqueObject&) = delete;
    UniqueObject& operator=(const UniqueObject&) = delete;

    void* get() const noexcept {
        return value_;
    }

    explicit operator bool() const noexcept {
        return value_ != nullptr;
    }

    void reset() noexcept {
        if (value_ != nullptr) {
            api_->Release(std::exchange(value_, nullptr));
        }
        api_.reset();
    }

private:
    std::shared_ptr<const CorelibApi> api_;
    void* value_ = nullptr;
};

// Each tag carries the kind it counts as. Defined rather than forward
// declared so `Tag::kKind` above resolves, and so adding a new corelib object
// type is a compile error here until it is given a kind, rather than a silent
// omission from the post-warm allocation measurement.
struct StreamTag {
    static constexpr CorelibObjectKind kKind = CorelibObjectKind::Stream;
};
struct TensorTag {
    static constexpr CorelibObjectKind kKind = CorelibObjectKind::Tensor;
};
struct MatMulWeightsTag {
    static constexpr CorelibObjectKind kKind =
        CorelibObjectKind::MatMulWeights;
};
struct SsMlpWeightsTag {
    static constexpr CorelibObjectKind kKind =
        CorelibObjectKind::SsMlpWeights;
};

using UniqueStream = UniqueObject<StreamTag>;
using UniqueTensor = UniqueObject<TensorTag>;
using UniqueMatMulWeights = UniqueObject<MatMulWeightsTag>;
using UniqueSsMlpWeights = UniqueObject<SsMlpWeightsTag>;

}  // namespace flm::corelib
