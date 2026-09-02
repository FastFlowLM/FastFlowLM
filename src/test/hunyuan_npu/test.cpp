/// \file test.cpp
/// \brief standalone harness for the hunyuan-dense engine (Hy-MT2-1.8B)
/// \author FastFlowLM Team
/// \date 2026-09-01
/// \note Three phases:
///         1. a first prompt on a clean context
///         2. a follow-up carrying the accumulated history, with
///            restore_allowed set (exercises the prompt cache and the
///            checkpoint + restore path in Hunyuan::insert)
///         3. clear_context() then a fresh prompt
///       -s 0 reads the prompt from ../../../../prompt.txt instead (long-prompt
///       prefill check) and only inserts it -- no generation.

#include <iostream>
#include <fstream>
#include <cmath>
#define NOMINMAX
#ifdef __WINDOWS__
#include <windows.h>
#endif
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_hunyuan.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

/// \brief run one turn against the accumulated history and dump the profile
/// \param messages grows in place: the user turn goes in, the reply comes back out
/// \param restore_allowed true to let insert() restore the previous checkpoint
///        instead of re-prefilling the whole conversation
static void run_turn(std::unique_ptr<AutoModel>& chat,
                     nlohmann::ordered_json& messages, const std::string& prompt,
                     int length_limit, bool restore_allowed) {
    chat_meta_info_t meta_info;
    messages.push_back({ {"role", "user"}, {"content", prompt} });

    lm_uniform_input_t input;
    input.messages = messages;
    meta_info.restore_allowed = restore_allowed;

    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Response: " << std::endl;
    chat->start_total_timer();
    std::string response = chat->generate_with_prompt(meta_info, input, length_limit, std::cout);
    chat->stop_total_timer();
    std::cout << std::endl << std::endl;
    std::cout << chat->show_profile() << std::endl;
    chat->clear_context();

    messages.push_back({ {"role", "assistant"}, {"content", response} });
}

int main(int argc, char* argv[]) {
    #ifdef __WINDOWS__
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    #endif

    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("hunyuan:1.8b"), "Model tag");
    desc.add_options()("Short,s", arg_utils::po::value<bool>()->default_value(true), "Short Prompt");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    desc.add_options()("Length,l", arg_utils::po::value<int>()->default_value(512), "Max generated tokens");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    bool short_prompt = vm["Short"].as<bool>();
    bool preemption = vm["Preemption"].as<bool>();
    int length_limit = vm["Length"].as<int>();
    std::cout << "Model: " << tag << std::endl;

    std::string exe_dir = utils::get_executable_directory();
    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = exe_dir + "/model_list.json";
    model_list model_list(model_list_path, model_dir);

    header_print("info", "Initializing chat model...");
    std::string model_path = model_list.get_model_path(tag);
    std::pair<std::string, nlohmann::json> model_info_pair = model_list.get_model_info(tag);
    nlohmann::json model_info = model_info_pair.second;
    std::cout << "Model path: " << model_path << std::endl;

    if (model_info["details"]["family"] != "hunyuan") {
        std::cout << "Tag '" << tag << "' is not a hunyuan-dense model" << std::endl;
        return 1;
    }

    std::unique_ptr<AutoModel> chat = std::make_unique<Hunyuan>(&npu_device_global);

    // the device must come up after the engine object exists
    npu_device_global = flm_rt::device(0);

    chat->load_model(model_path, model_info, -1, preemption);
    std::cout << chat->show_model_info() << std::endl;

    chat->set_topk(1);   // greedy: deterministic output for testing

    if (short_prompt) {
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();

        // 1. clean context
        run_turn(chat, messages,
                 "Translate into English: 今天天气很好，我们一起去公园散步吧。", length_limit, false);

        // 2. follow-up carrying the history -- prompt cache / checkpoint+restore.
        //    A "System prompt changed! Clearing context..." here means the
        //    restore desynchronized from token_history.
        run_turn(chat, messages,
                 "Now translate the same sentence into French.", length_limit, true);

        // 3. fresh context
        header_print("info", "Clearing context...");
        chat->clear_context();
        messages = nlohmann::ordered_json::array();
        run_turn(chat, messages,
                 "Translate into Chinese: The quick brown fox jumps over the lazy dog.", length_limit, false);
    }
    std::cout << std::endl;
    return 0;
}
