#include <iostream>
#include <cmath>
#define NOMINMAX
#ifdef __WINDOWS__
#include <windows.h>
#endif
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_qwen3_6_moe.hpp"
#include "model_list.hpp"
#include "utils/vm_args.hpp"

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
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    bool short_prompt = vm["Short"].as<bool>();
    bool preemption = vm["Preemption"].as<bool>();
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

    std::unique_ptr<AutoModel> chat = std::make_unique<Qwen3_6_MOE>(&npu_device_global);
    npu_device_global = flm_rt::device(0);


    chat->load_model(model_path, model_info, -1, preemption);
    chat_meta_info_t meta_info;
    lm_uniform_input_t uniformed_input;
    chat->set_topk(1);
    chat->configure_parameter("enable_think", false);

    if (short_prompt) {
        // uniformed_input.prompt = "Hello";
        uniformed_input.prompt = "What are these?";
        uniformed_input.images.push_back("../../../tb_files/panda.png");
        uniformed_input.images.push_back("../../../tb_files/puppy.png");
        uniformed_input.images.push_back("../../../tb_files/mj_icon.jpg");
        uniformed_input.images.push_back("../../../tb_files/google_icon.png");
        uniformed_input.images.push_back("../../../tb_files/pcb.jpg");
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        chat->insert(meta_info, uniformed_input);
        std::cout << "Prefill Done!" << std::endl;
        std::string response = chat->generate(meta_info, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;

        std::cout << chat->show_profile() << std::endl;
        chat->clear_context(); 

        return 0;

        meta_info.restore_allowed = true;
        uniformed_input.prompt = "How far is the previous city from Beijing?";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: " << std::endl;
        chat->start_total_timer();
        chat->insert(meta_info, uniformed_input);
        response = chat->generate(meta_info, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
        chat->clear_context();

        uniformed_input.prompt = "Is it possible to go to the capital of France from Beijing by bicycle?";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: " << std::endl;
        chat->start_total_timer();
        chat->insert(meta_info, uniformed_input);
        response = chat->generate(meta_info, 1024, std::cout);
        chat->stop_total_timer();
      
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
      chat->clear_context();            
    }
    else{
        std::ifstream file("/scratch/alfxu/FastFlowLM/src/tb_files/8k.txt", std::ios::binary);
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
        chat->insert(meta_info, uniformed_input);
        chat->generate(meta_info, 32, std::cout);
        chat->stop_total_timer();

    }

    std::cout << std::endl;

    return 0;
}
