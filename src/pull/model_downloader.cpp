/// \file model_downloader.cpp
/// \brief Model downloader class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to download models from the huggingface
#include "model_downloader.hpp"
#include "utils/utils.hpp"
#include "download_model.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <sstream>

namespace {
struct HfRepoSpec {
    std::string repo;
    std::string tag;
};

HfRepoSpec parse_hf_repo_spec(const std::string& spec) {
    HfRepoSpec parsed;
    auto colon = spec.rfind(':');
    if (colon == std::string::npos) {
        parsed.repo = spec;
        parsed.tag = "main";
        return parsed;
    }
    parsed.repo = spec.substr(0, colon);
    parsed.tag = spec.substr(colon + 1);
    if (parsed.tag.empty()) {
        parsed.tag = "main";
    }
    return parsed;
}

std::string repo_basename(const std::string& repo) {
    auto slash = repo.rfind('/');
    return slash == std::string::npos ? repo : repo.substr(slash + 1);
}

std::string make_registry_tag(const std::string& tag) {
    std::string out = tag;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        if (c == ' ' || c == '-') return '_';
        return static_cast<char>(std::tolower(c));
    });
    return out.empty() ? std::string("default") : out;
}

std::string model_tree_url(const std::string& repo, const std::string& tag) {
    return "https://huggingface.co/api/models/" + repo + "/tree/" + tag;
}

std::string resolve_url(const std::string& repo, const std::string& tag, const std::string& file) {
    return "https://huggingface.co/" + repo + "/resolve/" + tag + "/" + file + "?download=true";
}

std::string get_remote_oid(const nlohmann::json& file_entry, bool is_lfs) {
    if (is_lfs && file_entry.contains("lfs") && file_entry["lfs"].is_object() && file_entry["lfs"].contains("oid")) {
        return file_entry["lfs"]["oid"].get<std::string>();
    }
    return file_entry.value("oid", "");
}

bool local_file_matches_oid(const std::string& local_path, bool is_lfs, const std::string& oid) {
    if (oid.empty() || !std::filesystem::exists(local_path) || !std::filesystem::is_regular_file(local_path)) {
        return false;
    }
    const std::string local_oid = is_lfs
        ? download_utils::calculate_file_sha256(local_path)
        : download_utils::calculate_git_blob_oid(local_path);
    return local_oid == oid;
}

/// \brief Detect the FLM dispatch family string from a downloaded model directory.
/// Uses the model_type field from config.json (with text_config promotion) plus
/// name-based hints for families that share a single model_type (e.g. all Llama
/// derivatives, Nanbeige, DeepSeek; LFM2 vs LFM2.5-TK; Gemma3 vs Gemma3-Text).
std::string detect_model_family(const std::filesystem::path& model_dir,
                                 const std::string& model_name_hint) {
    nlohmann::json config;
    {
        std::ifstream f(model_dir / "config.json");
        if (f.is_open()) {
            try { config = nlohmann::json::parse(f); } catch (...) {}
        }
    }
    // Promote text_config so nested fields are accessible
    if (config.contains("text_config") && config["text_config"].is_object()) {
        for (auto& [k, v] : config["text_config"].items()) {
            if (!config.contains(k)) config[k] = v;
        }
    }

    const std::string model_type = config.value("model_type", "");
    const bool has_vision_weight = config.value("vision_model_weight", "") != "";
    const bool has_vision_file   = std::filesystem::exists(model_dir / "vision_weight.q4nx");
    const bool is_vlm            = has_vision_weight || has_vision_file || config.contains("vision_config");

    // Lower-case hint for substring matching
    std::string hint = model_name_hint;
    std::transform(hint.begin(), hint.end(), hint.begin(), ::tolower);
    auto has = [&](const std::string& s) { return hint.find(s) != std::string::npos; };
    auto mt  = [&](const std::string& prefix) {
        return model_type.rfind(prefix, 0) == 0 || model_type == prefix;
    };

    if (mt("qwen3_5") || mt("qwen3.5"))         return "qwen3.5";
    if (mt("qwen3_vl") || mt("qwen3vl"))         return "qwen3vl";
    if (mt("qwen3")) {
        if (is_vlm)                              return "qwen3vl";
        if (has("thinking") || has("-tk"))        return "qwen3-tk";
        if (has("instruct")  || has("-it"))       return "qwen3-it";
                                                  return "qwen3";
    }
    if (mt("qwen2_5_vl") || mt("qwen2vl"))       return "qwen2vl";
    if (mt("qwen2")) {
        if (is_vlm)                              return "qwen2vl";
                                                  return "qwen2";
    }
    if (mt("gemma4"))                             return "gemma4e";
    if (mt("gemma3")) {
        if (has("embed"))                         return "embed-gemma";
        if (is_vlm)                              return "gemma3";
        // Larger gemma3 models (medgemma, translategemma, etc.) are VLMs but
        // vision_weight may not be present until fully downloaded; treat any
        // named variant that matches known VLM names as gemma3.
        if (has("medgemma") || has("translategemma") || has("gemma3-4b") ||
            has("gemma3_4b") || has("gemma4"))    return "gemma3";
                                                  return "gemma3-text";
    }
    if (mt("llama")) {
        if (has("nanbeige"))                      return "nanbeige";
        if (has("deepseek")) {
            if (has("0528"))                      return "deepseek-r1-0528";
                                                  return "deepseek-r1";
        }
                                                  return "llama3";
    }
    if (mt("lfm2") || mt("lfm")) {
        if (has("thinking") || has("-tk") || has("2.5"))  return "lfm2.5-tk";
                                                          return "lfm2";
    }
    if (mt("phi"))                                return "phi4";
    if (mt("nanbeige"))                           return "nanbeige";
    if (mt("gpt_oss") || mt("gpt-oss"))           return "gpt-oss";
    if (mt("whisper"))                            return "whisper-v3";

    return ""; // unknown
}

/// \brief Try to find a compatible installed xclbin directory for a model.
/// Resolution order:
///   1. Explicit "flm_xclbin_dir" field in the model's config.json
///   2. Case-insensitive normalized name matching: strip "-NPU2"/"-NPU1" suffix,
///      then check if any installed dir name is a prefix of the model dir name
///      (covers fine-tunes that extend the base model name, e.g. medgemma-4b-it)
///   3. Architecture fingerprint table (model_type + hidden_size + num_hidden_layers)
/// Returns the matching installed directory name, or "" if none found.
std::string detect_xclbin_base(const std::filesystem::path& model_dir,
                                const std::string& xclbin_root) {
    nlohmann::json config;
    {
        std::ifstream f(model_dir / "config.json");
        if (f.is_open()) {
            try { config = nlohmann::json::parse(f); } catch (...) {}
        }
    }

    // 1. Explicit override
    if (config.contains("flm_xclbin_dir") && config["flm_xclbin_dir"].is_string()) {
        return config["flm_xclbin_dir"].get<std::string>();
    }

    if (!std::filesystem::is_directory(xclbin_root)) return "";

    // Helper: normalize a name for fuzzy matching
    //   - lowercase, replace '.' and '_' with '-'
    //   - strip trailing "-npu2", "-npu1", "-npu" suffixes
    auto normalize = [](std::string s) -> std::string {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            if (c == '.' || c == '_') return '-';
            return static_cast<char>(std::tolower(c));
        });
        for (const char* sfx : {"-npu2", "-npu1", "-npu"}) {
            std::string sx(sfx);
            if (s.size() > sx.size() && s.substr(s.size() - sx.size()) == sx) {
                s = s.substr(0, s.size() - sx.size());
                break;
            }
        }
        return s;
    };

    auto compact_alnum = [](const std::string& s) -> std::string {
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s) {
            if (std::isalnum(c)) out.push_back(static_cast<char>(std::tolower(c)));
        }
        return out;
    };

    auto strip_variant_suffix = [](std::string s) -> std::string {
        for (const char* sfx : {
                "-instruct", "-thinking", "-tool", "-transcript", "-it", "-tk", "-sg"
            }) {
            std::string sx(sfx);
            if (s.size() > sx.size() && s.substr(s.size() - sx.size()) == sx) {
                s = s.substr(0, s.size() - sx.size());
                break;
            }
        }
        return s;
    };

    // 2. Case-insensitive normalized name matching
    //    Check model_dir name first, then config.json model_name field
    std::vector<std::string> name_hints;
    name_hints.push_back(model_dir.filename().string());
    if (config.contains("model_name") && config["model_name"].is_string()) {
        std::string mn = config["model_name"].get<std::string>();
        auto slash = mn.rfind('/');
        if (slash != std::string::npos) mn = mn.substr(slash + 1);
        if (!mn.empty()) name_hints.push_back(mn);
    }

    std::string best_match;
    size_t best_score = 0;
    for (const auto& entry : std::filesystem::directory_iterator(xclbin_root)) {
        if (!entry.is_directory()) continue;
        const std::string dir_name = entry.path().filename().string();
        const std::string norm_dir = normalize(dir_name);
        const std::string norm_dir_core = strip_variant_suffix(norm_dir);
        const std::string compact_dir = compact_alnum(norm_dir);
        const std::string compact_dir_core = compact_alnum(norm_dir_core);
        for (const auto& hint : name_hints) {
            const std::string norm_hint = normalize(hint);
            const std::string compact_hint = compact_alnum(norm_hint);

            // Installed dir name is a prefix of (or equal to) the model hint.
            if (norm_hint.rfind(norm_dir, 0) == 0 && norm_dir.size() > best_score) {
                best_score = norm_dir.size();
                best_match = dir_name;
            }

            // Core name without variant suffix (e.g. "-it") as prefix.
            if (norm_hint.rfind(norm_dir_core, 0) == 0 && norm_dir_core.size() > best_score) {
                best_score = norm_dir_core.size();
                best_match = dir_name;
            }

            // Compact alnum fallback: handles punctuation differences
            // like "Gemma-4-E4B" vs "Gemma4-E4B".
            if (compact_hint.rfind(compact_dir, 0) == 0 && compact_dir.size() > best_score) {
                best_score = compact_dir.size();
                best_match = dir_name;
            }
            if (compact_hint.rfind(compact_dir_core, 0) == 0 && compact_dir_core.size() > best_score) {
                best_score = compact_dir_core.size();
                best_match = dir_name;
            }
        }
    }
    if (!best_match.empty()) return best_match;

    // 3. Architecture fingerprint table
    nlohmann::json eff = config;
    if (eff.contains("text_config") && eff["text_config"].is_object()) {
        for (auto& [k, v] : eff["text_config"].items()) {
            if (!eff.contains(k)) eff[k] = v;
        }
    }
    const std::string model_type = eff.value("model_type", "");
    const int hidden_size        = eff.value("hidden_size", 0);
    const int num_layers         = eff.value("num_hidden_layers", 0);

    // Gemma4 fine-tunes can change layer count; use size token + model_type
    // to select the correct base xclbin directory.
    std::string hint_joined;
    for (const auto& h : name_hints) {
        if (!hint_joined.empty()) hint_joined += " ";
        hint_joined += normalize(h);
    }
    if (model_type.rfind("gemma4", 0) == 0) {
        if ((hint_joined.find("e4b") != std::string::npos || hidden_size == 2560) &&
            std::filesystem::is_directory(xclbin_root + "/Gemma4-E4B-IT-NPU2")) {
            return "Gemma4-E4B-IT-NPU2";
        }
        if ((hint_joined.find("e2b") != std::string::npos || hidden_size == 2048) &&
            std::filesystem::is_directory(xclbin_root + "/Gemma4-E2B-IT-NPU2")) {
            return "Gemma4-E2B-IT-NPU2";
        }
    }

    struct ArchEntry { std::string model_type_prefix; int hidden_size; int num_layers; std::string xclbin_dir; };
    static const std::vector<ArchEntry> arch_table = {
        // Qwen3.5 family
        {"qwen3_5",    4096, 32, "Qwen3.5-9B-NPU2"},
        {"qwen3_5",    2560, 36, "Qwen3.5-4B-NPU2"},
        {"qwen3_5",    1536, 28, "Qwen3.5-2B-NPU2"},
        {"qwen3_5",     896, 24, "Qwen3.5-0.8B-NPU2"},
        // Qwen3 family
        {"qwen3",      4096, 36, "Qwen3-8B-NPU2"},
        {"qwen3",      2048, 36, "Qwen3-4B-NPU2"},
        {"qwen3",      1536, 28, "Qwen3-1.7B-NPU2"},
        {"qwen3",       896, 28, "Qwen3-0.6B-NPU2"},
        // Qwen2 / Qwen2.5 family
        {"qwen2",      2048, 36, "Qwen2.5-3B-Instruct-NPU2"},
        {"qwen2_5_vl", 2048, 36, "Qwen2.5-VL-3B-Instruct-NPU2"},
        // Llama family (name-based disambiguation preferred; fallback to first match)
        {"llama",      4096, 32, "Llama-3.1-8B-NPU2"},
        {"llama",      2048, 28, "Llama-3.2-3B-NPU2"},
        {"llama",      2048, 16, "Llama-3.2-1B-NPU2"},
        {"llama",      2560, 32, "Nanbeige4.1-3B-NPU2"},
        // Gemma3 / Gemma4 family
        {"gemma4",     2560, 34, "Gemma4-E4B-IT-NPU2"},
        {"gemma3",     2560, 34, "Medgemma-4B-NPU2"},
        {"gemma3",      768, 18, "Gemma3-1B-NPU2"},
        // LFM family
        {"lfm2",       2048, 16, "LFM2.5-1.2B-NPU2"},
        {"lfm2",       2048, 24, "LFM2-2.6B-NPU2"},
    };
    for (const auto& ae : arch_table) {
        if (model_type.rfind(ae.model_type_prefix, 0) == 0 &&
            hidden_size == ae.hidden_size &&
            num_layers  == ae.num_layers) {
            if (std::filesystem::is_directory(xclbin_root + "/" + ae.xclbin_dir)) {
                return ae.xclbin_dir;
            }
        }
    }
    return "";
}

} // namespace

/// \brief Constructor
/// \param models the model list
/// \return the model downloader
ModelDownloader::ModelDownloader(model_list& models) 
    : supported_models(models), curl_init() {
}

std::string ModelDownloader::pull_hf_repo(const std::string& hf_repo_with_tag, bool force_redownload) {
    const auto spec = parse_hf_repo_spec(hf_repo_with_tag);
    const std::string repo_id = spec.repo;
    const std::string revision = spec.tag;
    const std::string tag_key = make_registry_tag(spec.tag);
    const std::string model_type = repo_basename(repo_id);

    const std::filesystem::path model_root = std::filesystem::path(utils::get_models_directory()) / "models";
    const std::filesystem::path model_dir = model_root / model_type;
    std::filesystem::create_directories(model_root);
    std::filesystem::create_directories(model_dir);

    const std::vector<std::string> required_files = {
        "config.json",
        "model.q4nx",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vision_weight.q4nx",
        "audio_weight.q4nx"
    };

    const std::string tree_json = download_utils::download_string(model_tree_url(repo_id, revision));
    if (tree_json.empty()) {
        throw std::runtime_error("failed to query Hugging Face repo tree for " + hf_repo_with_tag);
    }

    const nlohmann::json repo_tree = nlohmann::json::parse(tree_json, nullptr, false);
    if (repo_tree.is_discarded() || !repo_tree.is_array()) {
        throw std::runtime_error("failed to parse Hugging Face repo tree for " + hf_repo_with_tag);
    }

    // Collect matching repo-tree entries, then sort by size so small files
    // are verified/downloaded first and large .q4nx files are last.
    struct FileEntry {
        std::string filename;
        nlohmann::json tree_entry;
        uint64_t size;
    };
    std::vector<FileEntry> file_entries;
    for (const auto& filename : required_files) {
        auto it = std::find_if(repo_tree.begin(), repo_tree.end(), [&](const nlohmann::json& entry) {
            return entry.contains("path") && entry["path"] == filename;
        });
        if (it == repo_tree.end()) continue;
        file_entries.push_back({filename, *it, it->value("size", static_cast<uint64_t>(0))});
    }
    std::stable_sort(file_entries.begin(), file_entries.end(),
        [](const FileEntry& a, const FileEntry& b) { return a.size < b.size; });

    nlohmann::json downloads = nlohmann::json::array();
    std::vector<std::string> downloaded_files;
    for (const auto& fe : file_entries) {
        const std::string& filename = fe.filename;
        const auto& tree_it = fe.tree_entry;

        const std::string local_path = (model_dir / filename).string();
        const bool is_lfs = tree_it.contains("lfs") && tree_it["lfs"].is_object() && tree_it["lfs"].contains("oid");
        const std::string oid = get_remote_oid(tree_it, is_lfs);
        const uint64_t remote_size = fe.size;

        if (!force_redownload && std::filesystem::exists(local_path) && std::filesystem::is_regular_file(local_path)) {
            header_print("FLM", "Checking local file: " + filename);
            if (local_file_matches_oid(local_path, is_lfs, oid)) {
                header_print("FLM", "Unchanged: " + filename);
                downloaded_files.push_back(filename);
                continue;
            }

            header_print("FLM", "Changed/corrupt, will re-download: " + filename);

            const uint64_t local_size = std::filesystem::file_size(local_path);
            if (remote_size > 0 && local_size >= remote_size) {
                // Existing file is full-sized but stale/corrupt: restart from scratch.
                std::filesystem::remove(local_path);
            }
        }

        downloads.push_back({
            {"file", filename},
            {"size", remote_size},
            {"url", resolve_url(repo_id, revision, filename)},
            {"localpath", local_path},
            {"oid", oid},
            {"is_lfs", is_lfs}
        });
    } // end file_entries loop

    if (!downloads.empty() && !download_utils::download_multiple_files(downloads)) {
        throw std::runtime_error("failed to download Hugging Face files for " + hf_repo_with_tag);
    }

    for (const auto& item : downloads) {
        downloaded_files.push_back(item["file"].get<std::string>());
    }

    if (downloaded_files.empty()) {
        throw std::runtime_error("no downloadable FastFlowLM artifact files found in " + hf_repo_with_tag);
    }

    // ---- xclbin setup -------------------------------------------------
    // Case 1: repo ships its own xclbins (entries like "xclbins/attn.xclbin")
    std::string xclbin_root;
    try { xclbin_root = utils::find_xclbin_path() + "/xclbins"; } catch (...) {}

    std::string resolved_xclbin_dir; // will be stored in registry
    if (!xclbin_root.empty()) {
        const std::string model_xclbin_dir = xclbin_root + "/" + model_type;

        // Collect any xclbin files the repo ships
        nlohmann::json xclbin_downloads = nlohmann::json::array();
        for (const auto& entry : repo_tree) {
            if (!entry.contains("path")) continue;
            const std::string path = entry["path"].get<std::string>();
            if (path.rfind("xclbins/", 0) == 0 && path.size() > 8 &&
                path.substr(path.size() - 7) == ".xclbin") {
                const std::string fname = std::filesystem::path(path).filename().string();
                const std::string local_path = model_xclbin_dir + "/" + fname;
                const bool is_lfs = entry.contains("lfs") && entry["lfs"].is_object() && entry["lfs"].contains("oid");
                const std::string oid = get_remote_oid(entry, is_lfs);
                if (!std::filesystem::exists(local_path) || !local_file_matches_oid(local_path, is_lfs, oid)) {
                    xclbin_downloads.push_back({
                        {"file", path},
                        {"size", entry.value("size", static_cast<uint64_t>(0))},
                        {"url", resolve_url(repo_id, revision, path)},
                        {"localpath", local_path},
                        {"oid", oid},
                        {"is_lfs", is_lfs}
                    });
                }
            }
        }
        if (!xclbin_downloads.empty()) {
            std::filesystem::create_directories(model_xclbin_dir);
            if (!download_utils::download_multiple_files(xclbin_downloads)) {
                header_print("FLM", "Warning: some xclbin files failed to download for " + model_type);
            } else {
                header_print("FLM", "xclbins installed for " + model_type);
            }
            resolved_xclbin_dir = model_type;
        }

        // Case 2: fine-tune — create a symlink to the base model's xclbin dir
        if (resolved_xclbin_dir.empty() && !std::filesystem::exists(model_xclbin_dir)) {
            const std::string base = detect_xclbin_base(model_dir, xclbin_root);
            if (!base.empty()) {
                try {
                    // Relative symlink: xclbins/<model_type> → <base>
                    std::filesystem::create_directory_symlink(base, model_xclbin_dir);
                    header_print("FLM", "xclbins linked: " + model_type + " → " + base);
                    resolved_xclbin_dir = base;
                } catch (const std::exception& e) {
                    header_print("FLM", "Warning: could not create xclbin symlink: " + std::string(e.what()));
                }
            } else {
                header_print("FLM", "Warning: no compatible installed xclbin directory found for " + model_type
                             + ". Add \"flm_xclbin_dir\" to config.json to specify one explicitly.");
            }
        } else if (resolved_xclbin_dir.empty()) {
            // xclbin dir already exists (previously linked or installed)
            resolved_xclbin_dir = model_type;
        }
    }
    // -------------------------------------------------------------------

    // Detect model family from the downloaded config.json
    const std::string detected_family = detect_model_family(model_dir, model_type);
    if (detected_family.empty()) {
        header_print("FLM", "Warning: could not determine model family for " + model_type
                     + ". Add \"flm_family\" to config.json to specify one explicitly.");
    } else {
        header_print("FLM", "Detected model family: " + detected_family);
    }

    // Extract parameter size hint from model_type name (e.g. "9B", "4B")
    std::string param_size = "?";
    {
        std::string mt_upper = model_type;
        std::transform(mt_upper.begin(), mt_upper.end(), mt_upper.begin(), ::toupper);
        for (const char* sz : {"0.5B","0.6B","0.8B","1B","1.2B","1.5B","1.7B","2B","2.6B","3B","4B","7B","8B","9B","11B","14B","20B","27B","32B","70B"}) {
            if (mt_upper.find(sz) != std::string::npos) { param_size = sz; break; }
        }
    }

    const bool is_vlm_model = std::filesystem::exists(model_dir / "vision_weight.q4nx");
    const bool is_think     = [&]() {
        std::string h = model_type; std::transform(h.begin(),h.end(),h.begin(),::tolower);
        return h.find("thinking") != std::string::npos || h.find("-tk") != std::string::npos
            || detected_family == "deepseek-r1" || detected_family == "deepseek-r1-0528"
            || detected_family == "lfm2.5-tk" || detected_family == "qwen3-tk";
    }();

    // For HF shortcuts, derive minimum compatible FLM version from the model's
    // own config when available. Converter outputs often omit this field; in
    // that case keep it permissive to avoid false incompatibility rejections.
    std::string flm_min_version = "0.0.0";
    try {
        std::ifstream cfg_in(model_dir / "config.json");
        if (cfg_in.is_open()) {
            nlohmann::json cfg_json = nlohmann::json::parse(cfg_in, nullptr, false);
            if (!cfg_json.is_discarded() && cfg_json.contains("flm_version") && cfg_json["flm_version"].is_string()) {
                const std::string cfg_ver = cfg_json["flm_version"].get<std::string>();
                if (!cfg_ver.empty()) {
                    flm_min_version = cfg_ver;
                }
            }
        }
    } catch (...) {
        // Keep permissive default if config parsing fails.
    }

    nlohmann::json model_entry = {
        {"name", model_type},
        {"url", "https://huggingface.co/" + repo_id},
        {"file_url", model_tree_url(repo_id, revision)},
        {"flm_min_version", flm_min_version},
        {"files", downloaded_files},
        {"xclbin_dir", resolved_xclbin_dir},
        {"vlm", is_vlm_model},
        {"default_context_length", 32768},
        {"max_prefill_len", 4096},
        {"details", {
            {"format", "NPU2"},
            {"family", detected_family.empty() ? "qwen3.5" : detected_family},
            {"think", is_think},
            {"parameter_size", param_size},
            {"quantization_level", "Q4_1"}
        }},
        {"label", nlohmann::json::array()},
        {"footprint", 0.0}
    };

    std::filesystem::path overlay_path = model_root / "hf_model_list.json";
    nlohmann::json overlay_json;
    if (std::filesystem::exists(overlay_path)) {
        std::ifstream overlay_in(overlay_path);
        overlay_json = nlohmann::json::parse(overlay_in, nullptr, false);
    }
    if (overlay_json.is_discarded() || !overlay_json.is_object()) {
        overlay_json = nlohmann::json::object();
    }
    if (!overlay_json.contains("models") || !overlay_json["models"].is_object()) {
        overlay_json["models"] = nlohmann::json::object();
    }
    if (!overlay_json["models"].contains(model_type) || !overlay_json["models"][model_type].is_object()) {
        overlay_json["models"][model_type] = nlohmann::json::object();
    }
    overlay_json["models"][model_type][tag_key] = model_entry;

    std::ofstream overlay_file(overlay_path);
    overlay_file << overlay_json.dump(4) << std::endl;

    return model_type + ":" + tag_key;
}

/// \brief Check if the model is downloaded
/// \param model_tag the model tag
/// \return true if the model is downloaded, false otherwise
bool ModelDownloader::is_model_downloaded(const std::string& model_tag, bool sub_process_mode, bool fast_check) {
    auto missing_files = get_missing_files(model_tag);
    bool is_config_file_missing = std::find(missing_files.begin(), missing_files.end(), "config.json") != missing_files.end();
    if (!is_config_file_missing) {
        if (!check_model_compatibility(model_tag, sub_process_mode)) {
            if (!sub_process_mode)
                header_print("FLM", "Model " + model_tag + " is not compatible with the current FLM version. ");
            // Fast path: skip the expensive HF metadata fetch + per-file hash
            // verification. Caller (e.g. `flm list`) only needs the boolean
            // status; cleanup can happen later on `pull` / `check`.
            if (fast_check) {
                return false;
            }
            // Instead of removing all files wholesale, verify each file against
            // HuggingFace metadata and remove only corrupted ones (same logic
            // as check_model). Files that pass verification can be reused.
            verify_and_clean_files(model_tag, sub_process_mode);
            return false;
        }
    }
    return missing_files.empty();
}

/// \brief Check if the model is compatible with the current FLM version
/// \param model_tag the model tag
/// \return true if the model is compatible, false otherwise
bool ModelDownloader::check_model_compatibility(const std::string& model_tag, bool sub_process_mode) {
    auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    LM_Config config;
    config.from_pretrained(this->supported_models.get_model_path(new_model_tag));
    std::string flm_version = config.flm_version;
    std::string flm_min_version = "0.0.0";
    if (model_info.contains("flm_min_version") && model_info["flm_min_version"].is_string()) {
        flm_min_version = model_info["flm_min_version"].get<std::string>();
    }
    int l_l = 0, m_l = 0, r_l = 0; //left, middle, right on local version
    int l_r = 0, m_r = 0, r_r = 0; //left, middle, right on requried version
    int l_f = 0, m_f = 0, r_f = 0; //left, middle, right on flm version
    sscanf(__FLM_VERSION__, "%d.%d.%d", &l_f, &m_f, &r_f);
    sscanf(flm_version.c_str(), "%d.%d.%d", &l_l, &m_l, &r_l);
    sscanf(flm_min_version.c_str(), "%d.%d.%d", &l_r, &m_r, &r_r);
    uint32_t local_version_u32 = l_l * 1000000 + m_l * 1000 + r_l;
    uint32_t required_version_u32 = l_r * 1000000 + m_r * 1000 + r_r;
    uint32_t flm_version_u32 = l_f * 1000000 + m_f * 1000 + r_f;

    // Converter-generated models may omit flm_version; LM_Config defaults that
    // to 0.0.0. Treat this as "unknown" and do not reject compatibility solely
    // based on missing version metadata.
    if (local_version_u32 == 0) {
        if (!sub_process_mode) {
            header_print("WARNING", "Model " + model_tag + " has no flm_version metadata (0.0.0); skipping strict version gate.");
        }
        return true;
    }

    bool is_future_version = false;
    if (local_version_u32 > flm_version_u32) {
        is_future_version = true;
    }
    bool is_compatible = true;
    if (local_version_u32 < required_version_u32) {
        is_compatible = false;
    }
    if (!sub_process_mode) {
        if (is_future_version) {
            header_print("WARNING", "Local model version: " + flm_version + " > " + __FLM_VERSION__);
            header_print("WARNING", "This model may not be compatible with the current FLM version.");
            header_print("WARNING", "Please update FLM to the latest version.");
            exit(0);
        }
        if (!is_compatible) {
            header_print("FLM", "Local model " + model_tag + " version: " + flm_version + " < " + flm_min_version);
            return false;
        }
    }
    return is_compatible;
}
/// \brief Pull the model
/// \param model_tag the model tag
/// \param force_redownload true if the model should be downloaded even if it is already downloaded
/// \return true if the model is downloaded, false otherwise
bool ModelDownloader::pull_model(const std::string& model_tag, bool force_redownload) {
    try {
        // Get model info
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string model_name = model_info["name"];
        std::string base_url = model_info["url"];
        
        header_print("FLM", "Model: " + new_model_tag);
        header_print("FLM", "Name: " + model_name);
        
        // Check if model is already downloaded
        if (!force_redownload && is_model_downloaded(new_model_tag)) {
            header_print("FLM", "Model already downloaded. Use --force to re-download.");
            return true;
        }

        // If force, verify existing files and remove only corrupted ones,
        // instead of wiping the whole model directory. Files that pass
        // verification will be reused; missing/corrupted ones get re-downloaded.
        if (force_redownload) {
            verify_and_clean_files(new_model_tag);
        }
        
        // Get missing files
        auto missing_files = get_missing_files(new_model_tag);
        if (missing_files.empty() && !force_redownload) {
            header_print("FLM", "All files already present.");
            return true;
        }
        
        if (!missing_files.empty()) {
            header_print("FLM", "Missing files (" + std::to_string(missing_files.size()) + "):");
            for (const auto& file : missing_files) {
                std::cout << "  - " << file << std::endl;
            }
        } else {
            header_print("FLM", "All required files are present.");
        }
        
        // Show present files if any
        auto present_files = get_present_files(new_model_tag);
        if (!present_files.empty()) {
            header_print("FLM", "Present files (" + std::to_string(present_files.size()) + "):");
            for (const auto& file : present_files) {
                std::cout << "  - " << file << std::endl;
            }
        }
        
        // Build download list
        auto download_list = build_download_list(new_model_tag);
        auto downloads = download_list.first;
        float sum_fize_size = download_list.second;
        if (downloads.empty()) {
            header_print("FLM", "No files to download for model: " + new_model_tag);
            return true; // Return true since all files are already present
        }
        
        header_print("FLM", "Downloading " + std::to_string(downloads.size()) + " missing files...");

        header_print("FLM", "Files to download (" << std::fixed << std::setprecision(2) << sum_fize_size << " MB): ");
        for (const auto& download : downloads) {
            float file_size = download["size"];
            std::string filename = download["file"];
            std::cout << "  - " << filename << " ("
                << std::fixed << std::setprecision(2) << file_size << " MB)"
                << std::endl;
        }
        
        // Download files with progress
        bool success = download_utils::download_multiple_files(downloads, get_progress_callback());

        if (success) {
            header_print("FLM", "Model downloaded successfully!");
            
            // Verify download
            auto final_missing = get_missing_files(new_model_tag);
            if (final_missing.empty()) {
                header_print("FLM", "All files verified successfully.");
            } else {
                header_print("WARNING", "Some files may be missing after download:");
                for (const auto& file : final_missing) {
                    std::cout << "  - " << file << std::endl;
                }
            }
            return true;
        } else {
            header_print("ERROR", "Failed to download model files.");
            return false;
        }
        
    } catch (const std::exception& e) {
        header_print("ERROR", "Exception during download: " + std::string(e.what()));
        return false;
    }
}

/// \brief Model not found
/// \param model_tag the model tag
void ModelDownloader::model_not_found(const std::string& model_tag) {
    header_print("ERROR", "Model not found: " + model_tag);
    header_print("ERROR", "Supported models: ");
    nlohmann::json models = supported_models.get_all_models();
    for (const auto& model : models["models"]) {
        header_print("ERROR", "  - " + model["name"].get<std::string>());
    }
}

/// \brief Get missing files
/// \param model_tag the model tag
/// \return the missing files
std::vector<std::string> ModelDownloader::get_missing_files(const std::string& model_tag) {
    std::vector<std::string> missing_files;

    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string model_name = model_info["name"];
        std::string model_path = supported_models.get_model_path(new_model_tag);
        std::vector<std::string> model_files = model_info["files"];

        // Check if this is a VLM model (default to false if key doesn't exist)

        // Check each required model file
        for (int i = 0; i < model_files.size(); ++i) {
            std::string filename = model_files[i];
            std::string file_path = get_model_file_path(model_path, filename);
            if (!file_exists(file_path)) {
                missing_files.push_back(filename);
            }
        }
    } catch (const std::exception& e) {
        header_print("ERROR", "Error checking missing files: " + std::string(e.what()));
    }

    return missing_files;
}

/// \brief Get present files
/// \param model_tag the model tag
/// \return the present files
std::vector<std::string> ModelDownloader::get_present_files(const std::string& model_tag) {
    std::vector<std::string> present_files;
    
    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string model_name = model_info["name"];
        std::string model_path = supported_models.get_model_path(new_model_tag);
        std::vector<std::string> model_files = model_info["files"];

        // Check if this is a VLM model (default to false if key doesn't exist)
        
        // Check each required model file
        for (int i = 0; i < model_files.size(); ++i) {
            std::string filename = model_files[i];
            std::string file_path = get_model_file_path(model_path, filename);
            if (file_exists(file_path)) {
                present_files.push_back(filename);
            }
        }     
    } catch (const std::exception& e) {
        header_print("ERROR", "Error checking present files: " + std::string(e.what()));
    }
    
    return present_files;
}

/// \brief Get progress callback
/// \return the progress callback
std::function<void(size_t, size_t)> ModelDownloader::get_progress_callback() {
    return [](size_t completed, size_t total) {
        if (total > 0) {
            double percentage = (static_cast<double>(completed) / total) * 100.0;
            std::cout << "\r[FLM]  Overall progress:  " << completed << "/" << total << " files" << std::flush;
            
            std::cout << std::endl;
        }
    };
}

/// \brief Check if the file exists
/// \param file_path the file path
/// \return true if the file exists, false otherwise
bool ModelDownloader::file_exists(const std::string& file_path) {
    return std::filesystem::exists(file_path) && std::filesystem::is_regular_file(file_path);
}

/// \brief Get the model file path
/// \param model_path the model path
/// \param filename the filename
/// \return the model file path
std::string ModelDownloader::get_model_file_path(const std::string& model_path, const std::string& filename) {
    std::filesystem::path full_path = std::filesystem::path(model_path) / filename;
    return full_path.string();
}

/// \brief Build the download list
/// \param model_tag the model tag
/// \return the download list
std::pair<nlohmann::json, float> ModelDownloader::build_download_list(const std::string& model_tag) {
    
    nlohmann::json downloads = nlohmann::json::array();
    float sum_file_size = 0;

    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string base_url = model_info["url"];
        std::string model_name = model_info["name"];
        std::string file_url = model_info["file_url"];
        std::vector<std::string> model_files = model_info["files"];
        
        // Create model directory
        std::string model_path = supported_models.get_model_path(new_model_tag);
        std::filesystem::create_directories(model_path);
        
        // GET HF api/models
        std::string hf_response = download_utils::download_string(file_url);
        nlohmann::json hf_model_infos = nlohmann::json::parse(hf_response);

        // Collect + sort by size (small files first, large .q4nx last)
        struct FileEntry2 { std::string filename; nlohmann::json info; uint64_t size; };
        std::vector<FileEntry2> sorted_files;
        for (const auto& filename : model_files) {
            auto it = std::find_if(hf_model_infos.begin(), hf_model_infos.end(),
                [&](const nlohmann::json& f) { return f["path"] == filename; });
            if (it == hf_model_infos.end()) continue;
            sorted_files.push_back({filename, *it, it->value("size", static_cast<uint64_t>(0))});
        }
        std::stable_sort(sorted_files.begin(), sorted_files.end(),
            [](const FileEntry2& a, const FileEntry2& b) { return a.size < b.size; });

        for (const auto& fe : sorted_files) {
            const std::string& filename = fe.filename;
            const auto& file = fe.info;
            std::string local_path = get_model_file_path(model_path, filename);
            bool is_lfs = file.contains("lfs") && file["lfs"].is_object() && file["lfs"].contains("oid");
            std::string oid = get_remote_oid(file, is_lfs);
            uint64_t remote_size = fe.size;

            if (file_exists(local_path)) {
                header_print("FLM", "Checking local file: " + filename);
                if (local_file_matches_oid(local_path, is_lfs, oid)) {
                    header_print("FLM", "Unchanged: " + filename);
                    continue;
                }

                header_print("FLM", "Changed/corrupt, will re-download: " + filename);

                uint64_t local_size = std::filesystem::file_size(local_path);
                if (remote_size > 0 && local_size >= remote_size) {
                    std::filesystem::remove(local_path);
                }
            }

            std::string url;
            if (std::string(base_url).find("resolve") != std::string::npos) { // resolve provided , may from a specific branch
                url = base_url + "/" + filename + "?download=true";
            }
            else {
                url = base_url + "/resolve/main/" + filename + "?download=true";
            }
            float file_size = static_cast<float>(remote_size) / 1024 / 1024;
            sum_file_size += file_size;

            nlohmann::json entry = {
                {"file", filename},
                {"size", file_size},
                {"url", url},
                {"localpath", local_path},
                {"oid", oid},
                {"is_lfs", is_lfs},
            };
            downloads.push_back(entry);

        }
    } 
    catch (const std::exception& e) {
        header_print("ERROR", "Error building download list: " + std::string(e.what()));
    }

    return std::make_pair(downloads, sum_file_size);
}

/// \brief Remove a model and all its files
/// \param model_tag the model tag
/// \return true if the model was successfully removed, false otherwise
bool ModelDownloader::remove_model(const std::string& model_tag, bool sub_process_mode) {
    try {
        // Check if model exists in supported models by trying to get its info
        try {
            supported_models.get_model_info(model_tag);
        } catch (const std::exception& e) {
            header_print("ERROR", "Model not found: " + model_tag);
            model_not_found(model_tag);
            return false;
        }
        
        // Get model path
        std::string model_path = supported_models.get_model_path(model_tag);
        
        // Check if model directory exists
        if (!std::filesystem::exists(model_path)) {
            header_print("FLM", "Model directory does not exist: " + model_path);
            return true; // Consider it already removed
        }

        if (!sub_process_mode) {
            header_print("FLM", "Removing model: " + model_tag);
            header_print("FLM", "Path: " + model_path);
        }
        
        // Remove all files in the model directory
        size_t removed_files = 0;
        for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
            if (entry.is_regular_file()) {
                std::filesystem::remove(entry.path());
                removed_files++;
            }
        }
        
        // Remove the model directory itself
        if (std::filesystem::remove(model_path)) {
            if(!sub_process_mode)
                header_print("FLM", "Successfully removed " + std::to_string(removed_files) + " files and model directory.");
            return true;
        } else {
            header_print("ERROR", "Failed to remove model directory: " + model_path);
            return false;
        }
        
    } catch (const std::exception& e) {
        header_print("ERROR", "Exception during model removal: " + std::string(e.what()));
        return false;
    }
}

/// \brief Check hash of model files
/// \param model_tag the model tag
/// \return true if all files are present and compatible, false otherwise
bool ModelDownloader::check_model(const std::string& model_tag, bool sub_process_mode) {
    auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    header_print("FLM", "Checking model: " + new_model_tag + "...\n");

    if (!is_model_downloaded(new_model_tag, sub_process_mode)) {
        header_print("FLM", "Model not exist or not compatible: " + new_model_tag);
        header_print("FLM", "Please use `flm pull " + new_model_tag + "` to download the model.");
        return true;
    }
    else {
        bool ok = verify_and_clean_files(new_model_tag, sub_process_mode);
        if (!ok) {
            header_print("FLM", "Model check completed with errors. Please use `flm pull " + new_model_tag + "` to re-download corrupted files.");
        }
        else {
            header_print("FLM", "Model check completed successfully. All files are present and compatible.");
        }
    }
    return true;
}

/// \brief Verify each model file's hash against HuggingFace metadata and
///        remove any corrupted files. Files that pass verification are kept.
/// \param model_tag the model tag
/// \param sub_process_mode if true, suppress informational logging
/// \return true if all files passed verification, false otherwise
bool ModelDownloader::verify_and_clean_files(const std::string& model_tag, bool sub_process_mode) {
    bool any_error = false;
    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::vector<std::string> model_files = model_info["files"];
        std::string model_path = supported_models.get_model_path(new_model_tag);
        std::string file_url = model_info["file_url"];

        // GET HF api/models
        std::string hf_response = download_utils::download_string(file_url);
        nlohmann::json hf_model_infos = nlohmann::json::parse(hf_response);

        for (const auto& filename : model_files) {
            if (!sub_process_mode) {
                header_print("FLM", "Checking file: " + filename + "...");
            }

            auto it = std::find_if(
                hf_model_infos.begin(),
                hf_model_infos.end(),
                [&](const nlohmann::json& f) {
                    return f["path"] == filename;
                }
            );
            if (it == hf_model_infos.end()) {
                continue;
            }
            const auto& file = *it;
            std::string local_path = get_model_file_path(model_path, filename);

            // If the file isn't present locally, there's nothing to verify or
            // remove; treat as an error so the caller knows a re-pull is needed.
            if (!file_exists(local_path)) {
                any_error = true;
                header_print("FLM", "File missing: " + filename);
                continue;
            }

            bool is_lfs = file.contains("lfs");
            std::string oid_ref = is_lfs ? file["lfs"]["oid"] : file["oid"];
            std::string local_oid = is_lfs ? download_utils::calculate_file_sha256(local_path) : download_utils::calculate_git_blob_oid(local_path);

            if (local_oid == oid_ref) {
                if (!sub_process_mode) {
                    header_print("FLM", "Success!");
                }
            }
            else {
                if (!sub_process_mode) {
                    header_print("FLM", "Fail!");
                    header_print("FLM", "Removing corrupted file: " + filename + "...");
                }

                if (std::filesystem::remove(local_path)) {
                    if (!sub_process_mode) {
                        header_print("FLM", "Successfully removed " + filename + "!");
                    }
                }
                else {
                    header_print("ERROR", "Failed to remove corrupted file: " + filename);
                }
                any_error = true;
            }
        }
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Exception during file verification: " + std::string(e.what()));
        any_error = true;
    }
    return !any_error;
}