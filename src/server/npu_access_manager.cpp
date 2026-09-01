#include <server/npu_access_manager.hpp>

#include <atomic>
#include <mutex>

namespace {

std::mutex g_npu_access_mutex;
std::atomic<bool> g_npu_in_use{false};
std::atomic<int> g_npu_active_requests{0};

}  // namespace

bool NPUAccessManager::try_acquire_npu_access() {
    std::lock_guard<std::mutex> lock(g_npu_access_mutex);
    if (g_npu_in_use.load(std::memory_order_relaxed)) {
        return false;
    }
    g_npu_in_use.store(true, std::memory_order_release);
    g_npu_active_requests.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void NPUAccessManager::release_npu_access() {
    std::lock_guard<std::mutex> lock(g_npu_access_mutex);
    g_npu_in_use.store(false, std::memory_order_release);
    g_npu_active_requests.fetch_sub(1, std::memory_order_relaxed);
}

bool NPUAccessManager::is_npu_available() {
    return !g_npu_in_use.load(std::memory_order_acquire);
}

int NPUAccessManager::get_active_npu_requests() {
    return g_npu_active_requests.load(std::memory_order_relaxed);
}

bool requires_npu_access(
    const std::string& method,
    const std::string& path) {
    if (method != "POST") {
        return false;
    }
    return path == "/api/generate" ||
           path == "/api/chat" ||
           path == "/v1/chat/completions" ||
           path == "/v1/completions" ||
           path == "/v1/audio/transcriptions" ||
           path == "/v1/embeddings";
}
