/*!
 *  Copyright (c) 2023 by Contributors
 * \file rest_handler.hpp
 * \brief RestHandler class and related declarations
 * \author FastFlowLM Team
 * \date 2025-06-24
 *  \version 0.9.24
 */
#pragma once

#include "AutoModel/all_models.hpp"
#include "whisper/modeling_whisper.hpp"
#include "AutoEmbeddingModel/all_embedding_model.hpp"
#include "model_list.hpp"
#include "model_downloader.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <functional>

using json = nlohmann::ordered_json;

// Forward declaration
struct CancellationToken;

///@brief Stream callback type for sending streaming responses
using StreamResponseCallback = std::function<void(const json&, bool)>; // data, is_final

class RestHandler {
public:
    RestHandler(model_list& models, ModelDownloader& downloader, const std::string& default_tag, bool asr, bool embed, int ctx_length = -1, bool preemption = false);
    ~RestHandler();

    void handle_show(const json& request,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response);

    void handle_generate(const json& request, 
                        std::function<void(const json&)> send_response,
                        StreamResponseCallback send_streaming_response,
                        std::shared_ptr<CancellationToken> cancellation_token = nullptr);

    void handle_chat(const json& request,
                    std::function<void(const json&)> send_response, 
                    StreamResponseCallback send_streaming_response,
                    std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    

    void handle_embeddings(const json& request,
                          std::function<void(const json&)> send_response,
                          StreamResponseCallback send_streaming_response);
    

    void handle_models(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);
    
    void handle_models_openai(const json& request,
                            std::function<void(const json&)> send_response,
                            StreamResponseCallback send_streaming_response);

    void handle_ps(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_version(const json& request,
                       std::function<void(const json&)> send_response,
                       StreamResponseCallback send_streaming_response);
    
    // Placeholder handlers for unimplemented endpoints
    void handle_pull(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_push(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_delete(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);
    
    void handle_copy(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_create(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);

    void handle_openai_chat_completion(const json& request,
                                      std::function<void(const json&)> send_response,
                                      StreamResponseCallback send_streaming_response,
                                      std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    void handle_openai_audio_transcriptions(const json& request,
                                      std::function<void(const json&)> send_response,
                                      StreamResponseCallback send_streaming_response,
                                      std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    void handle_openai_completion(const json& request,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response,
        std::shared_ptr<CancellationToken> cancellation_token = nullptr);

    // Anthropic API handlers
    void handle_anthropic_messages(const json& request,
                                   std::function<void(const json&)> send_response,
                                   StreamResponseCallback send_streaming_response,
                                   std::shared_ptr<CancellationToken> cancellation_token = nullptr);

private:
    void ensure_model_loaded(const std::string& model_tag);
    void ensure_asr_model_loaded(const std::string& model_tag);
    void ensure_embed_model_loaded(const std::string& model_tag);
    void configure_chat_engine_parameters(const json& options, const json& request);
    json build_nstream_response(std::string response_text);


    std::unique_ptr<AutoModel> auto_chat_engine;
    std::unique_ptr<Whisper> whisper_engine;
    std::unique_ptr<AutoEmbeddingModel> auto_embedding_engine;
    xrt::device npu_device_inst;
    model_list& supported_models;
    ModelDownloader& downloader;
    std::string current_model_tag;
    std::string default_model_tag;
    bool asr;
    bool embed;
    int generate_context_id;
    int chat_context_id;
    int ctx_length;
    std::string last_question;
    bool preemption;
}; 