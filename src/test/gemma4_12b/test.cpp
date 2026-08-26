#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_gemma4_12b.hpp"
#include "model_list.hpp"

flm_rt::device npu_device_global;

int main(int argc, char* argv[]) {
    #ifdef __WINDOWS__
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    // Set thread priority to low
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    #endif

    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("Short,s", arg_utils::po::value<bool>()->default_value(true), "Short Prompt");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    desc.add_options()("Think,t", arg_utils::po::value<bool>()->default_value(false), "Enable thinking");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    bool short_prompt = vm["Short"].as<bool>();
    bool preemption = vm["Preemption"].as<bool>();
    bool enable_think = vm["Think"].as<bool>();

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

    std::unique_ptr<AutoModel> chat = std::make_unique<Gemma4_12B>(&npu_device_global);
    std::cout << "Chat model initialized" << std::endl;
    npu_device_global = flm_rt::device(0);
    std::cout << "NPU Device initialized: " << npu_device_global.get_info<flm_rt::info::device::name>() << std::endl;
    chat->load_model(model_path, model_info, -1, preemption);
    header_print("info", "Model loaded");

    chat_meta_info_t meta_info;
    lm_uniform_input_t uniformed_input;
    chat->set_topk(1);
    chat->configure_parameter("enable_think", enable_think);

    if (short_prompt) {
        std::string response;

        // Phase 0: plain text generation
        uniformed_input.prompt = "Hello, introduce yourself briefly.";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: " << std::endl;
        chat->start_total_timer();
        response = chat->generate_with_prompt(meta_info, uniformed_input, 8192, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl << std::endl;
        std::cout << chat->show_profile() << std::endl;

        // Phase 1: follow-up turn on the same context (exercises the prompt cache)
        uniformed_input.prompt = "Now, name three cities in the United States.";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: " << std::endl;
        chat->start_total_timer();
        response = chat->generate_with_prompt(meta_info, uniformed_input, 8192, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl << std::endl;
        std::cout << chat->show_profile() << std::endl;
        chat->clear_context();

        // Phase 2: a fresh context after clear_context()
        uniformed_input.prompt = "Write a haiku about an NPU running a large language model.";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: " << std::endl;
        chat->start_total_timer();
        response = chat->generate_with_prompt(meta_info, uniformed_input, 8192, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl << std::endl;
        std::cout << chat->show_profile() << std::endl;
        chat->clear_context();
    }
    else {
        std::ifstream file("../../../../prompt.txt", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open prompt file" << std::endl;
            return 1;
        }
        uniformed_input.prompt = "";
        file.seekg(0, std::ios::end);
        uniformed_input.prompt.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(uniformed_input.prompt.data(), uniformed_input.prompt.size());
        file.close();
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        chat->insert(meta_info, uniformed_input);
        chat->stop_total_timer();
        std::cout << std::endl << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }

    std::pair<std::string, std::vector<int>> history = chat->get_history();
    std::cout << "History length: " << history.second.size() << std::endl;
    std::cout << std::endl;

    return 0;
}
