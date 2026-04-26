/// \file model_downloader.hpp
/// \brief Model downloader class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to download models from the huggingface
#pragma once

#include "model_list.hpp"
#include "lm_config.hpp"
#include "download_model.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>

struct ModelDownloadSize {
    double total_mb = 0.0;
    double missing_mb = 0.0;
    size_t total_files = 0;
    size_t missing_files = 0;
    bool exact = false;
};

class ModelDownloader {
public:
    ModelDownloader(model_list& models);
    
    // Check if model is already downloaded
    bool is_model_downloaded(const std::string& model_tag, bool sub_process_mode=0);
    
    // Download model files if not present
    bool pull_model(const std::string& model_tag, bool force_redownload = false);
    
    // Get total and missing download size information
    ModelDownloadSize get_download_size_info(const std::string& model_tag);

    // Get list of missing files for a model
    std::vector<std::string> get_missing_files(const std::string& model_tag);
    
    // Get list of present files for a model
    std::vector<std::string> get_present_files(const std::string& model_tag);
    
    // Remove a model and all its files
    bool remove_model(const std::string& model_tag, bool sub_process_mode=0);
    
    bool check_model(const std::string& model_tag, bool sub_process_mode=0);

    // Get download progress callback
    std::function<void(size_t, size_t)> get_progress_callback();

    void model_not_found(const std::string& model_tag);

private:
    model_list& supported_models;
    download_utils::CurlInitializer curl_init;
    
    // Check if a specific file exists
    bool file_exists(const std::string& file_path);
    
    // Get the full path for a model file
    std::string get_model_file_path(const std::string& model_path, const std::string& filename);
    
    // Build download URLs for model files
    std::pair<nlohmann::json, float> build_download_list(const std::string& model_tag);

    // bool check_model_compatibility(const std::string& model_tag);
    bool check_model_compatibility(const std::string& model_tag, bool sub_process_mode=0);
}; 