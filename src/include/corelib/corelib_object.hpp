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
            api_->RegisterObject();
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

struct StreamTag;
struct TensorTag;
struct MatMulWeightsTag;
struct SsMlpWeightsTag;

using UniqueStream = UniqueObject<StreamTag>;
using UniqueTensor = UniqueObject<TensorTag>;
using UniqueMatMulWeights = UniqueObject<MatMulWeightsTag>;
using UniqueSsMlpWeights = UniqueObject<SsMlpWeightsTag>;

}  // namespace flm::corelib
