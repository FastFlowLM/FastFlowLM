/// \file utils.cpp
/// \brief utils class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// 
/// \note This file contains some utility functions for the FastFlowLM project.
#include "utils/utils.hpp"
#include <filesystem>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace utils {

std::string find_model_list() {
    std::string install_prefix = CMAKE_INSTALL_PREFIX;

    // 1. Check FLM_CONFIG_PATH environment variable
    const char* env_path = std::getenv("FLM_CONFIG_PATH");
    if (env_path && *env_path) {
        if (std::filesystem::exists(env_path)) {
            std::cerr << "[FLM]  Using custom model list path: " << env_path << std::endl;
            return env_path;
        }
    }

    // 2. Check for install prefix
    std::string installed_path = install_prefix + "/share/flm/model_list.json";
    if (std::filesystem::exists(installed_path)) {
        return installed_path;
    }

    // 3. Check relative to executable
    std::string exe_dir = get_executable_directory();
    std::string exe_relative_path = exe_dir + "/model_list.json";
    if (std::filesystem::exists(exe_relative_path)) {
        return exe_relative_path;
    }

    // 4. Check current working directory
    if (std::filesystem::exists("model_list.json")) {
        return "model_list.json";
    }

    // If not found, throw an error
    throw std::runtime_error("model_list.json not found. Please set FLM_CONFIG_PATH or place it next to the executable.");
}

std::string find_xclbin_path() {
    std::string xclbin_prefix = CMAKE_XCLBIN_PREFIX;

    // 1. Check FLM_CONFIG_PATH environment variable
    const char* env_path = std::getenv("FLM_XCLBIN_PATH");
    if (env_path && *env_path) {
        // Remove possible "/xclbins" or "/xclbins/" from the end of the path
        std::string path(env_path);
        if (path.size() > 9 && path.substr(path.size() - 9) == "/xclbins/") {
            path = path.substr(0, path.size() - 9);
        } else if (path.size() > 9 && path.substr(path.size() - 9) == "\\xclbins\\") {
            path = path.substr(0, path.size() - 9);
        } else if (path.size() > 8 && path.substr(path.size() - 8) == "/xclbins") {
            path = path.substr(0, path.size() - 8);
        } else if (path.size() > 8 && path.substr(path.size() - 8) == "\\xclbins") {
            path = path.substr(0, path.size() - 8);
        }

        if (std::filesystem::exists(path + "/xclbins")) {
            return path;
        }
    }

    // 2. Check for install prefix
    std::string installed_path = xclbin_prefix;
    if (std::filesystem::exists(installed_path)) {
        return installed_path;
    }

    // 3. Check relative to executable
    std::string exe_dir = get_executable_directory();
    std::string exe_relative_path = exe_dir;
    if (std::filesystem::exists(exe_relative_path)) {
        return exe_relative_path;
    }

    // 4. Check current working directory
    if (std::filesystem::exists("xclbins")) {
        return "xclbins";
    }

    // If not found, throw an error
    throw std::runtime_error("xclbins not found. Please set FLM_XCLBIN_PATH or place it next to the executable.");
}

std::string get_executable_directory() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string exe_path(buffer);
    size_t last_slash = exe_path.find_last_of("\\");
    if (last_slash != std::string::npos) {
        return exe_path.substr(0, last_slash);
    }
    return ".";
#else
    char buffer[PATH_MAX] = {0};
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len > 0) {
        buffer[len] = '\0';
        std::string exe_path(buffer);
        size_t last_slash = exe_path.find_last_of("/");
        if (last_slash != std::string::npos) {
            return exe_path.substr(0, last_slash);
        }
    }
    return ".";
#endif
}

std::string get_user_documents_directory() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_PERSONAL, NULL, 0, buffer))) {
        return std::string(buffer);
    }
    // Fallback to current directory if Documents folder cannot be found
    return ".";
#else
    const char* home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.config";
    }
    return ".";
#endif
}

std::string get_cache_directory() {
#ifdef _WIN32
    char* local_app_data = nullptr;
    size_t len = 0;
    if (_dupenv_s(&local_app_data, &len, "LOCALAPPDATA") == 0 && local_app_data != nullptr) {
        std::string result(local_app_data);
        free(local_app_data);
        return result;
    }
    return ".";
#else
    const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
    if (xdg_cache && *xdg_cache) {
        return std::string(xdg_cache);
    }

    const char* home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.cache";
    }
    return ".";
#endif
}

int get_server_port(int user_port) {
    if (user_port > 0 && user_port <= 65535) {
        return user_port;
    }
    else {
#ifdef _WIN32
        char* port_env = nullptr;
        size_t len = 0;
        if (_dupenv_s(&port_env, &len, "FLM_SERVE_PORT") == 0 && port_env != nullptr) {
            try {
                int port = std::stoi(port_env);
                free(port_env);
                if (port > 0 && port <= 65535) {
                    return port;
                }
            }
            catch (const std::exception&) {
                free(port_env);
                // Invalid port number, use default
            }
        }
#else
        const char* port_env = std::getenv("FLM_SERVE_PORT");
        if (port_env && *port_env) {
            try {
                int port = std::stoi(port_env);
                if (port > 0 && port <= 65535) {
                    return port;
                }
            }
            catch (const std::exception&) {
                // Invalid port number, use default
            }
        }
#endif
    }

    return 52625; // Default port
}

std::string get_models_directory() {
#ifdef _WIN32
    char* model_path_env = nullptr;
    size_t len = 0;
    if (_dupenv_s(&model_path_env, &len, "FLM_MODEL_PATH") == 0 && model_path_env != nullptr) {
        std::string custom_path(model_path_env);
        free(model_path_env);
        if (!custom_path.empty()) {
            return custom_path;
        }
    }
#else
    const char* model_path_env = std::getenv("FLM_MODEL_PATH");
    if (model_path_env && *model_path_env) {
        return std::string(model_path_env);
    }
#endif
    // Fallback to Documents/flm/models on Windows or ~/.cache/flm on Linux if environment variable is not set
    std::string documents_dir = get_user_documents_directory();
    std::string cache_dir = get_cache_directory();
#ifdef _WIN32
    return documents_dir + "/flm/models";
#else
    // Check for deprecated location ~/.config/flm/models and warn user
    std::string deprecated_path = documents_dir + "/flm/models";
    if (std::filesystem::exists(deprecated_path)) {
        std::cerr << "[FLM] WARNING: Found models in deprecated location: " << deprecated_path << std::endl;
        std::cerr << "[FLM] Please move your models to: " << cache_dir << "/flm/models" << std::endl;
        std::cerr << "[FLM] Alternatively, set FLM_MODEL_PATH environment variable to specify a custom location." << std::endl;
        std::cerr << "[FLM] The deprecated location may be removed in future versions." << std::endl;
        return deprecated_path;
    }
    return cache_dir + "/flm";
#endif
}

} // end of namespace utils
