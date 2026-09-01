#pragma once

#include <string>

class NPUAccessManager {
public:
    static bool try_acquire_npu_access();
    static void release_npu_access();
    static bool is_npu_available();
    static int get_active_npu_requests();
};

bool requires_npu_access(
    const std::string& method,
    const std::string& path);
