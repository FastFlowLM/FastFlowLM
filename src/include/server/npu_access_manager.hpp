#pragma once

#include <functional>
#include <string>
#include <utility>

class NPUAccessManager {
public:
    static bool try_acquire_npu_access();
    static void release_npu_access();
    static bool is_npu_available();
    static int get_active_npu_requests();
};

class NPURequestCompletionGuard final {
public:
    explicit NPURequestCompletionGuard(
        std::function<void()> completion)
        : completion_(std::move(completion)) {}

    ~NPURequestCompletionGuard() noexcept {
        try {
            if (completion_) {
                completion_();
            }
        } catch (...) {
            if (!NPUAccessManager::is_npu_available()) {
                NPUAccessManager::release_npu_access();
            }
        }
    }

    NPURequestCompletionGuard(
        const NPURequestCompletionGuard&) = delete;
    NPURequestCompletionGuard& operator=(
        const NPURequestCompletionGuard&) = delete;

private:
    std::function<void()> completion_;
};

bool requires_npu_access(
    const std::string& method,
    const std::string& path);
