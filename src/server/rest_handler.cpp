/*!
 *  Copyright (c) 2023 by Contributors
 * \file rest_handler.cpp
 * \brief RestHandler class and related declarations
 * \author FastFlowLM Team
 * \date 2025-08-05
 *  \version 0.9.24
 */
#include "rest_handler.hpp"
#include "wstream_buf.hpp"
#include "streaming_ostream.hpp"
#include "streaming_ostream_openai.hpp"
#include "streaming_ostream_anthropic.hpp"
#include "image/image_reader.hpp"
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <locale>
#include <random>
#include "server.hpp"

///@brief Normalize messages by merging consecutive user messages (like Ollama does)
///@param messages the original messages
///@return normalized messages with consecutive user messages merged
static json normalize_messages(json messages) {
    if (messages.empty()) return messages;

    json normalized = nlohmann::ordered_json::array();

    for (size_t i = 0; i < messages.size(); i++) {
        auto current_msg = messages[i];
        std::string role = current_msg.value("role", "");

        if (role == "user") {
            json merged_content_array = json::array();

            // Merge all consecutive user messages into array format
            while (i < messages.size() && messages[i].value("role", "") == "user") {
                if (messages[i].contains("content")) {
                    if (messages[i]["content"].is_array()) {
                        for (auto& item : messages[i]["content"]) {
                            merged_content_array.push_back(item);
                        }
                    }
                    else if (messages[i]["content"].is_string()) {
                        std::string text = messages[i]["content"].get<std::string>();
                        if (!text.empty()) {
                            nlohmann::ordered_json text_item;
                            text_item["type"] = "text";
                            text_item["text"] = text;
                            merged_content_array.push_back(text_item);
                        }
                    }
                }
                if (i + 1 < messages.size() && messages[i + 1].value("role", "") == "user") i++;
                else break;
            }

            current_msg["content"] = merged_content_array;
        }
        else if (role == "system") {
            json merged_content_array = json::array();

            // Merge all consecutive system messages into array format
            while (i < messages.size() && messages[i].value("role", "") == "system") {
                if (messages[i].contains("content")) {
                    if (messages[i]["content"].is_array()) {
                        for (auto& item : messages[i]["content"]) {
                            merged_content_array.push_back(item);
                        }
                    }
                    else if (messages[i]["content"].is_string()) {
                        std::string text = messages[i]["content"].get<std::string>();
                        if (!text.empty()) {
                            json text_item;
                            text_item["type"] = "text";
                            text_item["text"] = text;
                            merged_content_array.push_back(text_item);
                        }
                    }
                }
                if (i + 1 < messages.size() && messages[i + 1].value("role", "") == "system") i++;
                else break;
            }

            current_msg["content"] = merged_content_array;
        }
        normalized.push_back(current_msg);
    }

    return normalized;
}

static json normalize_template(json messages) {
    json template_message = json::array();

    for (auto& message : messages) {
        json new_message = message;
        std::string merged_text;
        nlohmann::ordered_json::array_t merged_images;

        if (message["content"].is_string()) {
            // Simple format: just text
            merged_text = message["content"].get<std::string>();
        }
        else if (message["content"].is_array()) {
            // Structured format: extract text and image URLs
            for (auto& contentItem : message["content"]) {
                if (contentItem.contains("type") && contentItem["type"] == "text") {
                    merged_text += contentItem["text"].get<std::string>();
                }
                else if (contentItem.contains("type") && contentItem["type"] == "image_url") {
                    std::string image_url = contentItem["image_url"]["url"].get<std::string>();
                    const std::vector<std::string> prefixes = {
                        "data:image/png;base64,",
                        "data:image/jpeg;base64,",
                        "data:image/jpg;base64,"
                    };
                    for (const auto& prefix : prefixes) {
                        if (image_url.substr(0, prefix.length()) == prefix) {
                            image_url = image_url.substr(prefix.length());
                            break;
                        }
                    }
                    merged_images.push_back(image_url);
                }
            }
        }

        //new_message["role"] = message["role"];
        new_message["content"] = merged_text;
        if (!merged_images.empty()) {
            new_message["images"] = merged_images;
        }

        template_message.push_back(new_message);
    }

    return template_message;
}

static uint64_t checksum(const void* p, size_t len, uint64_t sum = 0) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(p);
    uint64_t _sum = sum;

    const uint64_t* p64 = reinterpret_cast<const uint64_t*>(data);
    size_t blocks = len / sizeof(uint64_t);
    for (size_t i = 0; i < blocks; ++i) {
        _sum += p64[i];
    }

    const uint8_t* p8 = data + blocks * sizeof(uint64_t);
    size_t remain = len % sizeof(uint64_t);
    for (size_t i = 0; i < remain; ++i) {
        _sum += p8[i];
    }

    return _sum;
}


///@brief RestHandler constructor
///@param models the model list
///@param downloader the downloader
///@param default_tag the default tag
///@return the rest handler
RestHandler::RestHandler(model_list& models, ModelDownloader& downloader, const std::string& default_tag, bool asr, bool embed, int ctx_length, bool preemption)
    : supported_models(models), downloader(downloader), default_model_tag(default_tag), current_model_tag(""), asr(asr), embed(embed), preemption(preemption){
    this->npu_device_inst = xrt::device(0);

    if (ctx_length != -1) {
        this->ctx_length = ctx_length >= 512 ? ctx_length : 512;
    } else {
        this->ctx_length = -1;
    }
    // Initialize chat bot with default model
    if (this->asr) {
        std::string whisper_tag = "whisper-v3:turbo";
        ensure_asr_model_loaded(whisper_tag);
    }
    if (this->embed) {
        std::string embed_tag = "embed-gemma:300m";
        ensure_embed_model_loaded(embed_tag);
    }

    if (default_model_tag != "model-faker") {
        if (!supported_models.is_model_supported(default_model_tag)) {
            header_print("Warning", "Default model tag '" << default_model_tag << "' is not supported. Falling back to 'llama3.2:1b'.");
            this->default_model_tag = "llama3.2:1b";
        }
        ensure_model_loaded(default_model_tag);
    }
    else {
        this->current_model_tag = "model-faker";
    }
}

///@brief RestHandler destructor
///@return the rest handler
RestHandler::~RestHandler() = default;

///@brief Ensure the model is loaded
///@param model_tag the model tag
void RestHandler::ensure_model_loaded(const std::string& model_tag) {
    std::string ensure_tag = model_tag;
    if (current_model_tag != ensure_tag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (auto_chat_engine != nullptr) {
            auto_chat_engine.reset();
        }
        std::pair<std::string, std::unique_ptr<AutoModel>> auto_model = get_auto_model(ensure_tag, this->supported_models, &this->npu_device_inst);
        auto_chat_engine = std::move(auto_model.second);
        ensure_tag = auto_model.first;
        if (!downloader.is_model_downloaded(ensure_tag)) {
            downloader.pull_model(ensure_tag);
        }
        auto [new_ensure_tag, model_info] = supported_models.get_model_info(ensure_tag);
        auto_chat_engine->load_model(supported_models.get_model_path(new_ensure_tag), model_info, ctx_length, preemption);
        current_model_tag = ensure_tag;
    }
}

///@brief Ensure the asr model is loaded
///@param model_tag the model tag
void RestHandler::ensure_asr_model_loaded(const std::string& model_tag) {
    std::string ensure_tag = model_tag;
    if (!downloader.is_model_downloaded(ensure_tag)) {
        downloader.pull_model(ensure_tag);
    }
    this->whisper_engine = std::make_unique<Whisper>(&this->npu_device_inst);
    auto [new_ensure_tag, whisper_model_info] = this->supported_models.get_model_info(ensure_tag);
    std::string whisper_model_path = this->supported_models.get_model_path(new_ensure_tag);
    this->whisper_engine->load_model(whisper_model_path, whisper_model_info, this->preemption);
}

///@brief Ensure the embed model is loaded
///@param model_tag the model tag
void RestHandler::ensure_embed_model_loaded(const std::string& model_tag) {
    std::string ensure_tag = model_tag;
    if (!this->downloader.is_model_downloaded(ensure_tag)) {
        this->downloader.pull_model(ensure_tag);
    }
    auto [embedding_model_tag, auto_embedding_engine] = get_auto_embedding_model(ensure_tag, &this->npu_device_inst);
    this->auto_embedding_engine = std::move(auto_embedding_engine);
    auto [new_embedding_model_tag, embedding_model_info] = this->supported_models.get_model_info(embedding_model_tag);
    std::string embedding_model_path = this->supported_models.get_model_path(new_embedding_model_tag);
    this->auto_embedding_engine->load_model(embedding_model_path, embedding_model_info, this->preemption);
}

///@brief Configure chat engine parameters from options and request
///@param options the options JSON object
///@param request the request JSON object
void RestHandler::configure_chat_engine_parameters(const json& options, const json& request) {
    if (options.contains("temperature")) {
        float temperature = options["temperature"];
        auto_chat_engine->set_temperature(temperature);
    }
    if (options.contains("top_p")) {
        float top_p = options["top_p"];
        auto_chat_engine->set_topp(top_p);
    }
    if (options.contains("top_k")) {
        int top_k = options["top_k"];
        auto_chat_engine->set_topk(top_k);
    }
    if (options.contains("frequency_penalty")) {
        float frequency_penalty = options["frequency_penalty"];
        auto_chat_engine->set_frequency_penalty(frequency_penalty);
    }
    if (options.contains("repetition_penalty")) {
        float repetition_penalty = options["repetition_penalty"];
        auto_chat_engine->set_repetition_penalty(repetition_penalty);
    }
    if (request.contains("think")) {
        bool enable_thinking = request["think"];
        auto_chat_engine->configure_parameter("enable_think", enable_thinking);
    }
    if (request.contains("reasoning_effort")) {
        std::string reasoning_effort = request["reasoning_effort"];
        auto_chat_engine->configure_parameter("reasoning_effort", reasoning_effort);
    }
}

json RestHandler::build_nstream_response(std::string response_text) {
    // Get tool info
    auto tool_info = auto_chat_engine->parse_nstream_content(response_text);

    json message;
    message["role"] = "assistant";

    bool is_tool_call = !tool_info.first.empty();

    if (is_tool_call) {
        message["content"] = nullptr;
        message["tool_calls"] = json::array({
            {
                {"index", 0},
                {"id", "call_" + tool_info.first}, // Generating a pseudo ID based on name or random
                {"type", "function"},
                {"function", {
                    {"name", tool_info.first},
                    {"arguments", tool_info.second}
                }}
            }
        });
    }
    else {
        // Standard text response
        message["content"] = response_text;
    }

    // Construct the final choice object
    return json::array({
        {
            {"index", 0},
            {"message", message},
            {"logprobs", nullptr},
            {"finish_reason", is_tool_call ? "tool_calling" : "stop"}
        }
    });
}

///@brief Handle the show request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_show(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response) {
    try {
        std::string model = request["model"];
        json info = {
            {"modelfile", ""},
            {"parameters", ""},
            {"template", ""},
            {"details", {
                {"parent_model", ""},
                {"format", ""},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
                }
            },
            {"model_info", {
                {"general.architecture", "flm" }
            }},
            {"capabilities", {"chat", "vision", "completion"}}
        };


        send_response(info);
    }
    catch (const std::exception& e) {
        json error_response = { {"error", e.what()} };
        send_response(error_response);
    }
}

///@brief Handle the generate request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_generate(const json& request,
                                 std::function<void(const json&)> send_response,
                                 StreamResponseCallback send_streaming_response,
                                 std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        std::string prompt = request["prompt"];
        bool stream = request.value("stream", true);
        std::string model = request.value("model", current_model_tag);
        json options = request.value("options", json::object());
       
        int length_limit = request.value("max_tokens", 4096);
        auto load_start_time = time_utils::now();
        // TODO: Use Another Check Function avoid loading again
        ensure_model_loaded(model);
        auto load_end_time = time_utils::now();
      
        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        header_print("FLM", "Start generating...");
        
        if (stream) {
            // Streaming response using streaming_ostream
            auto total_start_time = time_utils::now();
            streaming_ostream ostream(model, send_streaming_response, false);
            uniformed_input.prompt = prompt;
            bool success = auto_chat_engine->insert(meta_info, uniformed_input);
            if (!success){
                json error_response = {{"error", "Max length reached"}};
                send_response(error_response);
                return;
            }
            auto_chat_engine->generate(meta_info, length_limit, ostream);
            auto total_end_time = time_utils::now();
            auto history = this->auto_chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            ostream.finalize_generate(meta_info, history.second);
        } else {
            // Non-streaming response
            std::stringstream ss;
            wstream_buf obuf(ss);
            std::ostream ostream(&obuf);
            uniformed_input.prompt = prompt;
            bool success = auto_chat_engine->insert(meta_info, uniformed_input);
            if (!success){
                json error_response = {{"error", "Max length reached"}};
                send_response(error_response);
                return;
            }
            auto_chat_engine->generate(meta_info, length_limit, ostream);
            std::string response_text = ss.str();
            auto history = this->auto_chat_engine->get_history();
            json response = {
                {"model", model},
                {"response", response_text},
                {"context", history.second},
                {"done", true},
                {"prompt_eval_count", meta_info.prompt_tokens},
                {"eval_count", meta_info.generated_tokens},
                {"total_duration", meta_info.total_duration},
                {"load_duration", meta_info.load_duration},
                {"prompt_eval_duration", meta_info.prefill_duration},
                {"eval_duration", meta_info.decoding_duration},
                {"done_reason", stop_reason_to_string(meta_info.stop_reason)}
            };
            // std::cout << "history: " << history.first << std::endl;
            send_response(response);
        }
    } catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the chat request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_chat(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response,
                             std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        nlohmann::ordered_json messages = request["messages"];
        bool stream = request.value("stream", false);
        std::string model = request.value("model", current_model_tag);
        json options = request.value("options", json::object());
        int length_limit = options.value("num_predict", 4096);

        auto load_start_time = time_utils::now();
        ensure_model_loaded(model);
        auto load_end_time = time_utils::now();
       
        configure_chat_engine_parameters(options, request);

        // messages = normalize_messages(messages);
        
        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        header_print("FLM", "Start generating...");
        if (stream) {
            // Streaming response using streaming_ostream
            auto total_start_time = time_utils::now();
            streaming_ostream ostream(model, send_streaming_response, true);  // true for chat format
            uniformed_input.messages = messages;
            bool success = auto_chat_engine->insert(meta_info, uniformed_input);
            if (!success){
                json error_response = {{"error", "Max length reached"}};
                send_response(error_response);
                return;
            }
            auto_chat_engine->generate(meta_info, length_limit, ostream);
            auto total_end_time = time_utils::now();
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            
            ostream.finalize_chat(meta_info);
            // auto history = this->chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            this->auto_chat_engine->clear_context();
        } else {
            // Non-streaming response
            uniformed_input.messages = messages;
            auto total_start_time = time_utils::now();
            nullstream nstream;
            //std::string response_text = auto_chat_engine->generate_with_prompt(meta_info, uniformed_input, length_limit, std::cout);
            std::string response_text = auto_chat_engine->generate_with_prompt(meta_info, uniformed_input, length_limit, nstream);
            //std::string response_text = chat_engine->generate_with_prompt(meta_info, prompts, length_limit, std::cout, payload);
            auto total_end_time = time_utils::now();
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            
            json response = {
                {"model", model},
                {"message", {
                    {"role", "assistant"},
                    {"content", response_text},
                    {"images", nullptr}
                }},
                {"done", true},
                {"prompt_eval_count", meta_info.prompt_tokens},
                {"eval_count", meta_info.generated_tokens},
                {"total_duration", meta_info.total_duration},
                {"load_duration", meta_info.load_duration},
                {"prompt_eval_duration", meta_info.prefill_duration},
                {"eval_duration", meta_info.decoding_duration},
                {"done_reason", stop_reason_to_string(meta_info.stop_reason)}
            };
            send_response(response);
            
            // auto history = this->chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            this->auto_chat_engine->clear_context();
        }
    } catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the embeddings request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_embeddings(const json& request,
                                   std::function<void(const json&)> send_response,
                                   StreamResponseCallback send_streaming_response) {
    try {
        std::string model = request["model"];
        //std::string input = request["input"];
        std::string input;
        if (request["input"].is_string()) {
            input = request["input"];
        }
        else if (request["input"].is_array() && !request["input"].empty()) {
            input = request["input"][0];
        }

        //std::string encoding_format = request["encoding_format"];

        
        json response;
        if (this->embed) {
            std::cout << "Embedding input: " << input << std::endl;
            std::vector<float> embedding_result = this->auto_embedding_engine->embed(input, embedding_task_type_t::task_query);
            
            response = {
                {"object", "list"},
                {"data", json::array({
                    {
                        {"object", "embedding"},
                        {"embedding", embedding_result},
                        {"index", 0}
                    }
                })},
                {"model", model},
                {"usage", {
                    {"prompt_tokens", 0},
                    {"total_tokens", 0}
                }}
            };
        }
        else {
            header_print("Warning", "No embedding model loaded");
        }
        send_response(response);
    } 
    catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the models request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_models(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    try {
        json models = supported_models.get_all_models_ollama();
        send_response(models);
    } catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the models request (open ai)
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_models_openai(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response) {
    try {
        json models = supported_models.get_all_models_openai();
        send_response(models);
    }
    catch (const std::exception& e) {
        json error_response = { {"error", e.what()} };
        send_response(error_response);
    }
}

///@brief Handle the ps request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_ps(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    try {
        // Generate expires_at timestamp (1 hour from now)
        auto now = std::chrono::system_clock::now();
        auto expires_time = now + std::chrono::hours(1);
        auto expires_time_t = std::chrono::system_clock::to_time_t(expires_time);
        auto expires_tp = std::chrono::system_clock::from_time_t(expires_time_t);
        auto fractional_seconds = std::chrono::duration_cast<std::chrono::microseconds>(expires_time - expires_tp).count();
        
        // Get local time and timezone offset
        std::tm* local_tm = std::localtime(&expires_time_t);
        std::tm* utc_tm = std::gmtime(&expires_time_t);
        
        // Calculate timezone offset in minutes
        int offset_minutes = (local_tm->tm_hour - utc_tm->tm_hour) * 60 + (local_tm->tm_min - utc_tm->tm_min);
        if (local_tm->tm_mday != utc_tm->tm_mday) {
            offset_minutes += (local_tm->tm_mday > utc_tm->tm_mday) ? 1440 : -1440;
        }
        
        std::stringstream expires_ss;
        expires_ss.imbue(std::locale::classic()); // Use C locale to avoid commas
        expires_ss << std::put_time(local_tm, "%Y-%m-%dT%H:%M:%S");
        expires_ss << "." << std::setfill('0') << std::setw(5) << (fractional_seconds / 40); // 5 decimal places
        
        // Format timezone offset
        int offset_hours = offset_minutes / 60;
        int offset_mins = abs(offset_minutes % 60);
        expires_ss << (offset_minutes >= 0 ? "+" : "-") 
                  << std::setfill('0') << std::setw(2) << abs(offset_hours)
                  << ":" << std::setfill('0') << std::setw(2) << offset_mins;
        
        std::string expires_at = expires_ss.str();
        
        auto [new_current_model_tag, model_info] = supported_models.get_model_info(current_model_tag);
        json response = {
            {"models", json::array({
                {
                    {"name", current_model_tag},
                    {"model", current_model_tag},
                    {"size", model_info["size"]},
                    {"details", model_info["details"]},
                    {"expires_at", expires_at},
                }
            })}
        };
        // std::cout << "response: " << response.dump(4) << std::endl;
        send_response(response);
    } catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the version request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_version(const json& request,
                                std::function<void(const json&)> send_response,
                                StreamResponseCallback send_streaming_response) {
    json response = {{"version", "1.0.0"}};
    send_response(response);
}

///@brief Handle the pull request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_pull(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Pull operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the push request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_push(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Push operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the delete request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_delete(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Delete operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the copy request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_copy(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Copy operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the create request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_create(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Create operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the openai chat completion request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_chat_completion(const json& request,
                                               std::function<void(const json&)> send_response,
                                               StreamResponseCallback send_streaming_response,
                                               std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        // Extract OpenAI-style parameters
        json current_messages = request["messages"];
        std::string model = request.value("model", current_model_tag);
        std::string reasoning_effort = request.value("reasoning_effort", "medium");
        bool stream = request.value("stream", false);
        int length_limit = request.value("max_tokens", 4096);
        json tools = request.value("tools", json::array());
        json options = request.value("options", json::object());

        auto load_start_time = time_utils::now();
        ensure_model_loaded(model);
        auto load_end_time = time_utils::now();

        current_messages = normalize_messages(current_messages);
        current_messages = normalize_template(current_messages);
        
        static uint64_t sum_old = 0;
        static std::string current_model=model;
        uint64_t sum_new1 = 0;
        uint64_t sum_new2 = 0;
        json messages;

        // kv-cache
        if (model != current_model) { // switch models will clear context
            sum_old++;
            current_model = model;
        }
        if (current_messages.size() > 2) {
            for (size_t i = 0; i < current_messages.size(); ++i) {
                const auto& msg = current_messages[i];
                const std::string role = msg.value("role", "");
                const std::string content = msg.value("content", "");

                bool has_tool_call = msg.contains("tool_calls");
                bool skip_checksum = (role == "tool") || has_tool_call;

                if (skip_checksum) continue;
                    
                if(i < current_messages.size()-2)
                    sum_new1 = checksum(content.data(), content.size(), sum_new1);
                sum_new2 = checksum(content.data(), content.size(), sum_new2);
            }

            if (sum_old == sum_new1) {
                header_print("FLM", "Use cached prompt...");
                messages.push_back(current_messages.back());
            }
            else {
                sum_old = 0;
                messages = current_messages;
                this->auto_chat_engine->clear_context();
            }
            sum_old = sum_new2;
        }
        else {
            sum_old = 0;
            this->auto_chat_engine->clear_context();
            messages = current_messages;
            for (const auto& msg : current_messages) {
                std::string value = msg.value("content", "");
                sum_old = checksum(value.data(), value.size(), sum_old);
            }
        }

        // OpenAI API doesn't put the parameters into options
        if (request.contains("temperature")) {
            float temperature = request["temperature"];
            auto_chat_engine->set_temperature(temperature);
        }
        if (request.contains("top_p")) {
            float top_p = request["top_p"];
            auto_chat_engine->set_topp(top_p);
        }
        if (request.contains("top_k")) {
            int top_k = request["top_k"];
            auto_chat_engine->set_topk(top_k);
        }
        if (request.contains("frequency_penalty")) {
            float frequency_penalty = request["frequency_penalty"];
            auto_chat_engine->set_frequency_penalty(frequency_penalty);
        }
        if (request.contains("repetition_penalty")) {
            float repetition_penalty = request["repetition_penalty"];
            auto_chat_engine->set_repetition_penalty(repetition_penalty);
        }
        configure_chat_engine_parameters(options, request);

        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        uniformed_input.messages = messages;
        uniformed_input.tools = tools;
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        if (stream){
            // Create a wrapper callback that passes the pre-formatted SSE string directly
            cancellation_token->reset();
            auto openai_stream_callback = [&send_streaming_response](const std::string& data, bool is_final) {
                json data_json = data;
                send_streaming_response(data_json, is_final);
                };
            streaming_ostream_openai_chat ostream(model, auto_chat_engine.get(), openai_stream_callback);  // streaming in chat completion format

            header_print("FLM", "Start prefill...");
            bool success = auto_chat_engine->insert(meta_info, uniformed_input);
            if (!success) {
                json error_response = { {"error", "Max length reached"} };
                send_response(error_response);
                return;
            }
            header_print("FLM", "Start generating...");
            auto_chat_engine->generate(meta_info, length_limit, ostream, [&] { return cancellation_token->cancelled(); });
            ostream.finalize(meta_info);

            if (meta_info.stop_reason == CANCEL_DETECTED) {
                header_print("FLM", "Generation Cancelled!");
                sum_old++; // make sure checksum doesn't match
            }
        }
        else {
            nullstream nstream;
            std::string response_text = auto_chat_engine->generate_with_prompt(meta_info, uniformed_input, length_limit, nstream);

            // check response_text
            json choices = build_nstream_response(response_text);
            json response = {
                {"id", "fastflowlm-chat-completion"},
                {"object", "chat.completion"},
                {"created", static_cast<long long>(std::time(nullptr))},
                {"model", model},
                {"choices", choices},
                {"usage", {
                    {"prompt_tokens", meta_info.prompt_tokens},
                    {"completion_tokens", meta_info.generated_tokens},
                    {"total_tokens", meta_info.prompt_tokens + meta_info.generated_tokens},
                    {"load_duration", static_cast<double>(meta_info.load_duration) / 1'000'000'000},
                    {"prefill_duration_ttft", static_cast<double>(meta_info.prefill_duration) / 1'000'000'000},
                    {"decoding_duration", static_cast<double>(meta_info.decoding_duration) / 1'000'000'000},
                    {"prefill_speed_tps", static_cast<double>(meta_info.prompt_tokens) / static_cast<double>(meta_info.prefill_duration) * 1'000'000'000},
                    {"decoding_speed_tps", static_cast<double>(meta_info.generated_tokens) / static_cast<double>(meta_info.decoding_duration) * 1'000'000'000},
                    //{"prompt_tokens_details", json::array({
                    //    {
                    //        {"cached_tokens", 0},
                    //        {"audio_tokens", 0}
                    //    }
                    //})},
                    //{"completion_tokens_details", json::array({
                    //    {
                    //        {"reasoning_tokens", 0},
                    //        {"audio_tokens", 0},
                    //        {"accepted_prediction_tokens", 0},
                    //        {"rejected_prediction_tokens", 0}
                    //    }
                    //})}
                }},
                {"service_tier", "default"}
            };
            send_response(response);
            this->auto_chat_engine->clear_context();
        }

    } catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}

///@brief Handle the openai audio transcriptions request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_audio_transcriptions(const json& request,
                                        std::function<void(const json&)> send_response,
                                        StreamResponseCallback send_streaming_response,
                                        std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        std::string model = request["model"];
        std::string file_content = request["file"].get<std::string>();
        std::vector<uint8_t> audio_raw(file_content.begin(), file_content.end());
        bool stream = request.value("stream", false);
        json response;
        if (this->asr) {
            this->whisper_engine->load_audio(audio_raw);
            header_print("FLM", "Transforming audio to text...");
            // Show text 
            std::cout << "Audio content: " << std::flush;
            std::pair<std::string, std::string> audio_result = this->whisper_engine->generate(Whisper::whisper_task_type_t::e_transcribe, true, false, std::cout);
            std::string audio_context = audio_result.first;
            std::cout << std::endl;

            response = {
                {"model", model},
                {"text", audio_context}
                //{"usage", {
                //    {"type", "tokens"},
                //    {"input_tokens", 0},
                //    {"input_tokens_details", json::array({
                //        {
                //            {"text_tokens", 0},
                //            {"audio_tokens", 0}
                //        }
                //    })},
                //    {"output_tokens", 0},
                //    {"total_tokens", 0}
                //}}
            };
        }
        else {
            header_print("Warning", "No asr model loaded, cannot load audio file");
        }
        send_response(response);
        //this->whisper_engine->clear_context();
    }
    catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}

///@brief Handle the openai completion request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_completion(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response,
    std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        // Extract OpenAI-style parameters
        std::string prompt = request["prompt"];
        std::string model = request.value("model", current_model_tag);
        std::string reasoning_effort = request.value("reasoning_effort", "medium");
        bool stream = request.value("stream", false);
        json options = request.value("options", json::object());

        // direct return if model not supported
        if (!supported_models.is_model_supported(model)) {
            throw std::runtime_error("Model " + model + " is not supported.");
        }
       
        int length_limit = request.value("max_tokens", 4096);

        ensure_model_loaded(model);

        configure_chat_engine_parameters(options, request);

        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        header_print("FLM", "Start generating...");

        if (stream) {
            // Create a wrapper callback that passes the pre-formatted SSE string directly
            auto openai_stream_callback = [&send_streaming_response](const std::string& data, bool is_final) {
                json data_json = data;
                send_streaming_response(data_json, is_final);
                };
            streaming_ostream_openai ostream(model, openai_stream_callback);  // streaming in completion format
            uniformed_input.prompt = prompt;
            bool success = auto_chat_engine->insert(meta_info, uniformed_input);
            if (!success) {
                json error_response = { {"error", "Max length reached"} };
                send_response(error_response);
                return;
            }
            auto_chat_engine->generate(meta_info, length_limit, ostream);
            ostream.finalize(meta_info);

            this->auto_chat_engine->clear_context();
        }
        else {
            std::stringstream ss;
            wstream_buf obuf(ss);
            std::ostream ostream(&obuf);
            uniformed_input.prompt = prompt;
            bool success = auto_chat_engine->insert(meta_info, uniformed_input);
            if (!success) {
                json error_response = { {"error", "Max length reached"} };
                send_response(error_response);
                return;
            }
            auto_chat_engine->generate(meta_info, length_limit, ostream);
            std::string response_text = ss.str();
            auto history = this->auto_chat_engine->get_history();

            json response = {
                {"id", "fastflowlm-chat-completion"},
                {"object", "text_completion"},
                {"created", (int)std::time(nullptr)},
                {"model", model},
                {"choices", json::array({
                    {
                        {"text", response_text},
                        {"index", 0},
                        {"logprobs", nullptr},
                        {"finish_reason", "stop"}
                    }
                })},
                {"usage", {
                    {"prompt_tokens", meta_info.prompt_tokens},
                    {"completion_tokens", meta_info.generated_tokens},
                    {"total_tokens", meta_info.prompt_tokens + meta_info.generated_tokens}
                }}
            };
            send_response(response);
        }
    }
    catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}

///@brief Handle the Anthropic messages request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_anthropic_messages(const json& request,
                                            std::function<void(const json&)> send_response,
                                            StreamResponseCallback send_streaming_response,
                                            std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        // Extract Anthropic-style parameters
        std::string model = request.value("model", current_model_tag);
        int max_tokens = request.value("max_tokens", 4096);
        bool stream = request.value("stream", false);
        float temperature = request.value("temperature", 1.0f);
        json tools = request.value("tools", json::array());
        
        // Anthropic puts system prompt at top level, not in messages
        std::string system_prompt = request.value("system", "");
        json anthropic_messages = request["messages"];
        
        // Convert Anthropic messages format to internal format
        // Anthropic uses the same role/content structure but system is separate
        json messages = json::array();
        
        // Add system message if provided
        if (!system_prompt.empty()) {
            messages.push_back({
                {"role", "system"},
                {"content", system_prompt}
            });
        }
        
        // Process Anthropic messages - handle content blocks if present
        for (const auto& msg : anthropic_messages) {
            json converted_msg;
            converted_msg["role"] = msg["role"];
            
            // Anthropic content can be a string or array of content blocks
            if (msg["content"].is_string()) {
                converted_msg["content"] = msg["content"];
            } else if (msg["content"].is_array()) {
                // Convert content blocks to string (combine text blocks)
                std::string combined_content;
                for (const auto& block : msg["content"]) {
                    if (block["type"] == "text") {
                        if (!combined_content.empty()) combined_content += "\n";
                        combined_content += block["text"].get<std::string>();
                    }
                    // TODO: Handle image blocks if needed
                }
                converted_msg["content"] = combined_content;
            }
            messages.push_back(converted_msg);
        }
        
        auto load_start_time = time_utils::now();
        ensure_model_loaded(model);
        auto load_end_time = time_utils::now();
        
        // Configure parameters
        if (request.contains("temperature")) {
            auto_chat_engine->set_temperature(temperature);
        }
        if (request.contains("top_p")) {
            float top_p = request["top_p"];
            auto_chat_engine->set_topp(top_p);
        }
        if (request.contains("top_k")) {
            int top_k = request["top_k"];
            auto_chat_engine->set_topk(top_k);
        }
        
        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        uniformed_input.messages = messages;
        uniformed_input.tools = tools;
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        
        if (stream) {
            // Streaming response using Anthropic format
            cancellation_token->reset();
            auto anthropic_stream_callback = [&send_streaming_response](const std::string& data, bool is_final) {
                json data_json = data;
                send_streaming_response(data_json, is_final);
            };
            streaming_ostream_anthropic ostream(model, auto_chat_engine.get(), anthropic_stream_callback);
            
            header_print("FLM", "Start prefill...");
            bool success = auto_chat_engine->insert(meta_info, uniformed_input);
            if (!success) {
                json error_response = {
                    {"type", "error"},
                    {"error", {
                        {"type", "invalid_request_error"},
                        {"message", "Max context length reached"}
                    }}
                };
                send_response(error_response);
                return;
            }
            
            // Send message_start event
            ostream.send_message_start(meta_info.prompt_tokens);
            
            header_print("FLM", "Start generating...");
            auto_chat_engine->generate(meta_info, max_tokens, ostream, [&] { return cancellation_token->cancelled(); });
            ostream.finalize(meta_info);
            
            if (meta_info.stop_reason == CANCEL_DETECTED) {
                header_print("FLM", "Generation Cancelled!");
            }
            
            this->auto_chat_engine->clear_context();
        } else {
            // Non-streaming response
            nullstream nstream;
            std::string response_text = auto_chat_engine->generate_with_prompt(meta_info, uniformed_input, max_tokens, nstream);
            
            // Determine stop reason
            std::string stop_reason = "end_turn";
            if (meta_info.stop_reason == MAX_LENGTH_REACHED) {
                stop_reason = "max_tokens";
            }
            
            // Build Anthropic response format
            json response = {
                {"id", "msg_fastflowlm_" + std::to_string(std::time(nullptr))},
                {"type", "message"},
                {"role", "assistant"},
                {"content", json::array({
                    {
                        {"type", "text"},
                        {"text", response_text}
                    }
                })},
                {"model", model},
                {"stop_reason", stop_reason},
                {"stop_sequence", nullptr},
                {"usage", {
                    {"input_tokens", meta_info.prompt_tokens},
                    {"output_tokens", meta_info.generated_tokens}
                }}
            };
            
            send_response(response);
            this->auto_chat_engine->clear_context();
        }
        
    } catch (const std::exception& e) {
        // Anthropic error format
        json error_response = {
            {"type", "error"},
            {"error", {
                {"type", "api_error"},
                {"message", e.what()}
            }}
        };
        send_response(error_response);
    }
}
