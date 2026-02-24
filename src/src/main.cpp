/// \file main.cpp
/// \brief Main entry point for the FLM application
/// \author FastFlowLM Team
/// \date 2025-08-05
/// \version 0.9.24
/// \note This is a source file for the main entry point
#pragma once
#include "runner.hpp"
#include "server.hpp"
#include "model_list.hpp"
#include "model_downloader.hpp"
#include "update.hpp"
#include "utils/utils.hpp"
#include "minja/chat-template.hpp"
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <filesystem>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <shellapi.h>
#endif
#include "utils/vm_args.hpp"
#include <boost/program_options.hpp>
#include "benchmarking.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <libdrm/drm.h>
#include <sys/utsname.h>
#include <cerrno>
#include "npu_utils/amdxdna_accel.h"
#endif

#include "AutoModel/automodel.hpp"

#ifndef _WIN32
#include <dlfcn.h>
#endif

// Global variables
///@brief should_exit is used to control the server thread
std::atomic<bool> should_exit(false);

#ifndef _WIN32
///@brief Preload critical XRT libraries from the executable directory
///@details This ensures that dlopen() calls within libraries find the bundled versions
///@note Only on Linux/Unix; Windows handles DLL loading differently
void preload_bundled_libraries() {
    std::string exe_dir = utils::get_executable_directory();

    const std::vector<std::string> libraries = {
        "libxrt_core.so.2",           // Core - no dependencies
        "libxrt_coreutil.so.2",       // Depends on core
        "libxrt_driver_xdna.so.2",    // Driver
    };

    // Try to load the library with RTLD_GLOBAL so symbols are available to dependent libraries
    // Don't care about failures
    for (const auto& lib : libraries) {
        std::string lib_path = exe_dir + "/" + lib;
        void* handle = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    }
}
#endif



#ifdef _WIN32
///@brief get_unicode_command_line_args gets Unicode command line arguments
///@param argc_out reference to store argument count
///@return vector of UTF-8 encoded argument strings
std::vector<std::string> get_unicode_command_line_args(int& argc_out) {
    std::vector<std::string> args;
    
    // Get the Unicode command line
    LPWSTR* szArglist;
    int nArgs;
    
    szArglist = CommandLineToArgvW(GetCommandLineW(), &nArgs);
    if (szArglist == nullptr) {
        argc_out = 0;
        return args;
    }
    
    argc_out = nArgs;
    
    // Convert each argument from wide string to UTF-8
    for (int i = 0; i < nArgs; i++) {
        std::wstring warg(szArglist[i]);
        
        // Convert to UTF-8
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, warg.c_str(), (int)warg.size(), nullptr, 0, nullptr, nullptr);
        if (size_needed > 0) {
            std::string utf8_arg(size_needed, 0);
            WideCharToMultiByte(CP_UTF8, 0, warg.c_str(), (int)warg.size(), &utf8_arg[0], size_needed, nullptr, nullptr);
            args.push_back(utf8_arg);
        } else {
            args.push_back(""); // fallback for conversion error
        }
    }
    
    // Free memory allocated by CommandLineToArgvW
    LocalFree(szArglist);
    
    return args;
}
#endif

///@brief get_user_documents_directory gets the user's Documents directory
///@return the user's Documents directory path
std::string get_user_documents_directory() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_PERSONAL, NULL, 0, buffer))) {
        return std::string(buffer);
    }
    // Fallback to executable directory if Documents folder cannot be found
    return utils::get_executable_directory();
#else
    return utils::get_user_documents_directory();
#endif
}

///@brief ensure_models_directory creates the models directory if it doesn't exist
///@param exe_dir the executable directory
void ensure_models_directory(const std::string& exe_dir) {
    // Use Documents/flm/models directory on Windows or ~/.config/flm on Linux for models instead of executable directory
    std::string models_dir = utils::get_models_directory();
    if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
    }
}

///@brief handle_user_input is used to handle the user input
void handle_user_input() {
    std::string input;
    header_print("FLM", "Enter 'exit' to stop the server: ");
    while (!should_exit) {
        std::getline(std::cin, input);
        if (input == "exit") {
            should_exit = true;
            break;
        }
    }
}

///@brief create_lm_server is used to create the ollama server
///@param models the model list
///@param default_tag the default tag
///@param port the port to listen on, default is 52625, same with the ollama server
///@return the server
std::unique_ptr<WebServer> create_lm_server(model_list& models, ModelDownloader& downloader, const std::string& default_tag, bool asr, bool embed, std::string host, int port, int ctx_length, int img_pre_resize, bool cors, bool preemption);


///@brief get_server_port gets the server port from environment variable FLM_SERVE_PORT
///@return the server port, default is 52625 if environment variable is not set
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

///@brief get_models_directory gets the models directory from environment variable or defaults to Documents/flm/models on Windows or ~/.config/flm on Linux
///@return the models directory path
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
    // Fallback to Documents/flm/models on Windows or ~/.config/flm on Linux if environment variable is not set
    std::string documents_dir = get_user_documents_directory();
#ifdef _WIN32
    return documents_dir + "/flm/models";
#else
    return documents_dir + "/flm";
#endif
}

static bool sanity_check_npu_stack(bool quiet) {
#ifndef _WIN32
    // Check kernel version
    struct utsname u_name;
    if (uname(&u_name) != 0) {
        if (!quiet)
            perror("Failed to get kernel version");
        return false;
    }
    int major, minor;
    sscanf(u_name.release, "%d.%d", &major, &minor);
    bool kernel_ok = (major > 6) || (major == 6 && minor >= 14);
    if (!kernel_ok) {
        header_print("ERROR", "Kernel version incompatible with this version of FLM. Please update your kernel!");
        return false;
    }
    if (!quiet) {
        header_print("Linux", "Kernel Version: " << u_name.release);
    }

    // Check firmware version of all AMD devices
    bool all_fw_ok = true;
    bool amd_device_found = false;

    for (int i = 0; i < 16; ++i) {
        std::string dev_name = "/dev/accel/accel" + std::to_string(i);
        int fd = open(dev_name.c_str(), O_RDWR);

        if (fd < 0) {
            if (errno == ENOENT)
                break;
            continue;
        }

        amdxdna_drm_query_firmware_version query_fw_version;
        amdxdna_drm_get_info get_info = {
            .param = DRM_AMDXDNA_QUERY_FIRMWARE_VERSION,
            .buffer_size = sizeof(amdxdna_drm_query_firmware_version),
            .buffer = (unsigned long)&query_fw_version,
        };

        int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
        close(fd);
        if (ret < 0) {
            // ENOTTY means it's not an AMD device, just ignore it.
            if (errno != ENOTTY) {
                 if (!quiet) {
                    std::cout << "Error code: " << ret << " on " << dev_name << std::endl;
                    perror("Failed to get firmware version");
                 }
            }
            continue;
        }

        amd_device_found = true;

        if (!quiet) {
            header_print("Linux", "Found AMD NPU at " + dev_name);
            header_print("Linux", "NPU FW Version: " << query_fw_version.major << "." << query_fw_version.minor << "." << query_fw_version.patch << "." << query_fw_version.build);
        }

        bool fw_ok = (query_fw_version.major > 1 || (query_fw_version.major == 1 && query_fw_version.minor >= 1));
        if (!fw_ok) {
            all_fw_ok = false;
            header_print("ERROR", "NPU firmware version on " + dev_name + " is incompatible. Please update NPU firmware!");
        }
    }

    if (!amd_device_found && !quiet) {
        header_print("ERROR", "No AMD NPU device found.");
    }

    return amd_device_found && kernel_ok && all_fw_ok;
#else
    return true;
#endif
}

///@brief main function
///@param argc the number of arguments
///@param argv the arguments
///@return the exit code
int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#else
    // Preload bundled libraries from executable directory
    preload_bundled_libraries();
#endif
    
    // Parse command line arguments using Boost Program Options
    arg_utils::ParsedArgs parsed_args;
    if (!arg_utils::parse_options(argc, argv, parsed_args)) {
        return 1; // Help was already printed by Boost Program Options
    }

    
    // Get the command, model tag, and force flag
    std::string exe_dir = utils::get_executable_directory();
    std::string config_path = utils::find_model_list();
    // Get the models directory from environment variable or default
    std::string models_dir = utils::get_models_directory();

    
    model_list availble_models(config_path, models_dir);
    
    // Handle version requests
    
    if (parsed_args.version_requested) {
        std::cout << "FLM v" << __FLM_VERSION__ << std::endl;
        return 0;
    }

    if (parsed_args.port_requested) {
        std::cout << "Default server port: " << utils::get_server_port(-1) << std::endl;
        return 0;
    }

    
    // Extract parsed values
    std::string command = parsed_args.command;
    std::string tag = parsed_args.model_tag;
    bool force_redownload = parsed_args.force_redownload;
    std::string power_mode = parsed_args.power_mode;
    bool got_power_mode = (power_mode != "performance"); // Check if user explicitly set power mode
    int ctx_length = parsed_args.ctx_length;
    bool preemption = parsed_args.preemption;
    size_t max_socket_connections = parsed_args.max_socket_connections;
    size_t max_npu_queue = parsed_args.max_npu_queue;
    int user_port = parsed_args.port;
    int img_pre_resize = parsed_args.img_pre_resize;
    std::string user_host = parsed_args.host;
    bool quiet_list = parsed_args.quiet_list;
    std::string list_filter = parsed_args.list_filter;
    bool cors = parsed_args.cors;
    bool asr = parsed_args.asr;
    bool embed = parsed_args.embed;
    bool stable_stack = false;

    // Set process priority to high for better performance
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    
    // Handle special case for serve command - use default tag if none provided
    if (command == "serve" && tag.empty()) {
        tag = "model-faker"; // Use default tag
    }
    
    if (command == "run" || command == "serve" || command == "pull" || command == "remove" || command == "bench") {
      if (tag != "model-faker" && (!availble_models.is_model_supported(tag))) {
            header_print("ERROR", "Model not found: " << tag << "; Please check with `flm list` and try again.");
            return 1;
        }
    }
  
    if (command == "serve" || command == "run" || command == "bench"){
        // Configure AMD XRT for the specified power mode
        if (power_mode == "default" || power_mode == "powersaver" || power_mode == "balanced" || 
            power_mode == "performance" || power_mode == "turbo") {
#ifdef _WIN32
            std::string xrt_cmd = "cd \"C:\\Windows\\System32\\AMD\" && .\\xrt-smi.exe configure --pmode " + power_mode + " > NUL 2>&1";
            header_print("FLM", "Configuring NPU Power Mode to " + power_mode + (got_power_mode ? "" : " (flm default)"));
            (void)system(xrt_cmd.c_str());
#endif
        }
        else{
            std::cout << "Invalid power mode: " << power_mode << std::endl;
            std::cout << "Valid power modes: default, powersaver, balanced, performance, turbo" << std::endl;
            return 1;
        }
    }

    if (preemption){
        header_print("FLM", "Allowing high priority tasks to preempt FLM!");
    }

    stable_stack = sanity_check_npu_stack(command != "validate");
    if (command == "validate")
        return stable_stack ? 0 : 1;

    try {

        // Load the model list with the models directory as the base
        ModelDownloader downloader(availble_models);

        // Ensure models directory exists
        if (!std::filesystem::exists(models_dir)) {
            std::filesystem::create_directories(models_dir);
        }

        if (command == "bench") {
            benchmarking::BenchmarkResults_t results = benchmarking::run_benchmarks(tag, parsed_args.input_file_name, availble_models);
        }
        else if (command == "run") {
            check_and_notify_new_version();
            Runner runner(availble_models, downloader, tag, asr, embed, ctx_length, img_pre_resize, preemption);
            runner.run();

        } else if (command == "serve") {
            check_and_notify_new_version();
            // Create the server
            int port = utils::get_server_port(user_port);
            auto server = create_lm_server(availble_models, downloader, tag, asr, embed, user_host, port, ctx_length, img_pre_resize, cors, preemption);
            server->set_max_connections(max_socket_connections);           // Allow up to 10 concurrent connections
            server->set_io_threads(10);          // Allow up to 5 io threads
            server->set_npu_queue_length(max_npu_queue);           // Allow up to 10 concurrent queue
            server->set_request_timeout(std::chrono::seconds(600)); // 10 minute timeout for long requests
            // Start the server
            header_print("FLM", "Starting server on port " << port << "...");
            server->start();

            // Start a thread to handle user input, this thread will be used to handle the user input
            std::thread input_thread(handle_user_input);

            // Wait for exit command, this thread will be used to wait for the user to exit the server
            while (!should_exit) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // Cleanup, this will be used to stop the server
            header_print("FLM", "Stopping server...");
            server->stop();
            input_thread.join();
        }
        else if (command == "pull") {
            // Check if the model is already downloaded, if true, the model will not be downloaded
            // Check if model is already downloaded
            if (!force_redownload && downloader.is_model_downloaded(tag)) {
                header_print("FLM", "Model is already downloaded.");
                // Show missing files if any, this will be used to show the missing files
                auto missing_files = downloader.get_missing_files(tag);
                if (!missing_files.empty()) {
                    header_print("FLM", "Missing files:");
                    for (const auto& file : missing_files) {
                        std::cout << "  - " << file << std::endl;
                    }
                } else {
                    header_print("FLM", "All required files are present.");
                }
            } else {
                // Download the model, this will be used to download the model
                bool success = downloader.pull_model(tag, force_redownload);
                if (!success) {
                    header_print("ERROR", "Failed to pull model: " + tag);
                    return 1;
                }
            }
        }
        else if (command == "remove") {
            // Remove the model, this will be used to remove the model
            downloader.remove_model(tag);
        }
        else if (command == "list") {
            // List the models, this will be used to list the models
            if (list_filter == "installed" || list_filter == "not-installed" || list_filter == "all") {
                std::cout << "Models:" << std::endl;
                nlohmann::json models = availble_models.get_all_models();
                for (const auto& model : models["models"]) {
                    bool is_present = downloader.is_model_downloaded(model["name"].get<std::string>(), quiet_list);
                    if ((list_filter == "installed") == is_present || list_filter == "all") {
                        std::cout << "  - " << model["name"].get<std::string>();
                        if (!quiet_list) {
                            std::cout << (is_present ? " ✅" : " ⏬");
                        }
                        std::cout << std::endl;
                    }
                }
            }
            else
                header_print("Error", "Invalid filter: please use 'all', 'installed', or 'not-installed'");


        }
        else {
            // Invalid command, this will be used to show the invalid command
            std::cerr << "Invalid command: " << command << std::endl;
            std::cerr << "Use --help for usage information" << std::endl;
            return 1;
        }
        // Return 0 if the command is valid
        return 0;
    } catch (const std::exception& e) {
        // If an error occurs, this will be used to show the error
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}