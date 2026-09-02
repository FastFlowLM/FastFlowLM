/// \file bench_prefill.cpp
/// \brief prompt-length sweep over the hunyuan-dense mm (batched) prefill path
/// \author FastFlowLM Team
/// \date 2026-09-02
/// \note The engine picks the prefill path by prompt length: at most
///       HUNYUAN_MV_PREFILL_MAX_TOKENS tokens go through the decode overlay one
///       token at a time, anything longer takes the batched matmul path. The
///       danmaku benchmark's turns are all short, so it exercises the mm path
///       exactly once (the system prompt); this driver walks a range of lengths
///       instead, so the mm path's cost as a function of L is visible.
/// \note Build the engine with PREFILL_PROFILE=1 to also get the per-call
///       embedding / layers / lm_head split printed from inside the engine.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#define NOMINMAX
#ifdef __WINDOWS__
#include <windows.h>
#endif
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_hunyuan.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

/// \brief the lengths swept, in tokens
/// \note 8 sits under the mv ceiling and is included as the contrast; the rest
///       all take the mm path. 512 is the chunking boundary -- past it
///       _chunked_insert splits the prompt, so the 1000-token row is two engine
///       calls, not one.
static const int kLengths[] = {8, 16, 32, 64, 128, 256, 500, 1000};

/// \brief repeats a filler word until the prompt tokenizes to about n tokens
/// \note exactness is not needed: the measured length is read back from the
///       engine's context length, which is what the table reports.
static std::string make_prompt(int n_tokens) {
    std::string s;
    for (int i = 0; i < n_tokens; i++) {
        s += "word ";
    }
    return s;
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("hunyuan:1.8b"), "Model tag");
    desc.add_options()("repeat,r", arg_utils::po::value<int>()->default_value(5), "Timed calls per length");
    desc.add_options()("warmup,w", arg_utils::po::value<int>()->default_value(2), "Discarded calls per length");
    desc.add_options()("context,L", arg_utils::po::value<int>()->default_value(2048), "Context length the engine is loaded with");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    int repeat = vm["repeat"].as<int>();
    int warmup = vm["warmup"].as<int>();
    int context_length = vm["context"].as<int>();

    std::string exe_dir = utils::get_executable_directory();
    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = exe_dir + "/model_list.json";
    model_list model_list(model_list_path, model_dir);

    std::string model_path = model_list.get_model_path(tag);
    nlohmann::json model_info = model_list.get_model_info(tag).second;
    if (model_info["details"]["family"] != "hunyuan") {
        std::cout << "Tag '" << tag << "' is not a hunyuan-dense model" << std::endl;
        return 1;
    }

    std::unique_ptr<AutoModel> chat = std::make_unique<Hunyuan>(&npu_device_global);
    npu_device_global = flm_rt::device(0);  // the device comes up after the engine object
    // the default context is 512, which the longest sweep point would overrun
    chat->load_model(model_path, model_info, context_length, false);
    chat->set_topk(1);

    // no prefix pinning and no system prompt: every call below prefills its own
    // prompt from an empty context, which is the thing being measured
    static_cast<Hunyuan*>(chat.get())->set_prefix_pinning(false);

    std::cout << std::endl
              << std::setw(10) << "tokens" << std::setw(12) << "mean(ms)"
              << std::setw(12) << "min(ms)" << std::setw(12) << "max(ms)"
              << std::setw(14) << "tok/s" << std::setw(12) << "us/tok" << std::endl;

    for (int target : kLengths) {
        std::string text = make_prompt(target);
        double sum = 0.0, lo = 1e30, hi = 0.0;
        int measured_tokens = 0;

        for (int i = 0; i < warmup + repeat; i++) {
            chat->clear_context();

            nlohmann::ordered_json messages = nlohmann::ordered_json::array();
            messages.push_back({ {"role", "user"}, {"content", text} });
            lm_uniform_input_t input;
            input.messages = messages;
            chat_meta_info_t meta_info;
            meta_info.restore_allowed = false;

            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            bool ok = chat->insert(meta_info, input);
            double ms = ms_since(t0);
            if (!ok) {
                std::cout << "prompt rejected at target length " << target << std::endl;
                return 1;
            }
            if (i < warmup) {
                continue;  // the first calls pay for lazy weight / kernel paging
            }
            measured_tokens = chat->get_current_context_length();
            sum += ms;
            lo = std::min(lo, ms);
            hi = std::max(hi, ms);
        }

        double mean = sum / repeat;
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << measured_tokens
                  << std::setw(12) << mean
                  << std::setw(12) << lo
                  << std::setw(12) << hi
                  << std::setw(14) << (measured_tokens * 1000.0 / mean)
                  << std::setw(12) << (mean * 1000.0 / measured_tokens) << std::endl;
    }
    std::cout << std::endl;
    return 0;
}
