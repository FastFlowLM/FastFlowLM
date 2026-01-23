/*!
 *  Copyright (c) 2023 by Contributors
 * \file streaming_ostream_anthropic.hpp
 * \brief Custom ostream for Anthropic API streaming format
 * \author FastFlowLM Team
 * \date 2025-01-23
 *  \version 0.9.24
 */
#pragma once

#include <ostream>
#include <streambuf>
#include <functional>
#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include "AutoModel/automodel.hpp"

using json = nlohmann::ordered_json;

class AutoModel;

// Anthropic /v1/messages streaming format
///@brief Custom streambuf that captures tokens and sends them in Anthropic SSE format
///@param model the model
///@param callback the callback
///@return the streaming buf
///@note Anthropic uses event: type\ndata: json format for SSE
class streaming_buf_anthropic : public std::streambuf {
public:
    ///@brief StreamCallback
    using StreamCallback = std::function<void(const std::string&, bool)>;
    
    streaming_buf_anthropic(const std::string& model, AutoModel* auto_chat_engine, StreamCallback callback)
        : model_name(model), auto_chat_engine(auto_chat_engine), stream_callback(callback), 
          content_block_started(false), content_index(0) {
        // Generate a unique ID for this message
        generate_message_id();
    }

protected:
    ///@brief Called when buffer is full or flush is requested
    ///@param ch the character
    ///@return the character
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            buffer += static_cast<char>(ch);
        }
        return ch;
    }
    
    ///@brief Called when stream is flushed
    ///@return 0
    int sync() override {
        flush_complete_utf8_sequences(false);
        return 0;
    }

public:
    ///@brief Send the initial message_start event
    void send_message_start(int input_tokens) {
        json message_start = {
            {"type", "message_start"},
            {"message", {
                {"id", message_id},
                {"type", "message"},
                {"role", "assistant"},
                {"content", json::array()},
                {"model", model_name},
                {"stop_reason", nullptr},
                {"stop_sequence", nullptr},
                {"usage", {
                    {"input_tokens", input_tokens},
                    {"output_tokens", 0}
                }}
            }}
        };
        stream_callback("event: message_start\ndata: " + message_start.dump() + "\n\n", false);
    }

    ///@brief Call this when generation is complete
    void finalize(chat_meta_info_t& meta_info) {
        // Send all remaining content, including incomplete sequences
        if (!buffer.empty()) {
            send_content_delta(buffer);
            buffer.clear();
        }
        
        // Close the content block if one was started
        if (content_block_started) {
            send_content_block_stop();
        }
        
        send_message_delta(meta_info);
        send_message_stop();
    }

private:
    ///@brief Generate a unique message ID
    void generate_message_id() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 15);
        
        std::stringstream ss;
        ss << "msg_";
        for (int i = 0; i < 24; ++i) {
            ss << std::hex << dis(gen);
        }
        message_id = ss.str();
    }

    ///@brief Get UTF-8 sequence length from first byte
    ///@param first_byte the first byte
    ///@return sequence length, or 0 if invalid
    size_t get_utf8_sequence_length(unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) {
            return 1; // Single byte sequence
        } else if ((first_byte & 0xE0) == 0xC0) {
            return 2; // Two byte sequence
        } else if ((first_byte & 0xF0) == 0xE0) {
            return 3; // Three byte sequence
        } else if ((first_byte & 0xF8) == 0xF0) {
            return 4; // Four byte sequence
        }
        return 0; // Invalid UTF-8 start byte
    }
    
    ///@brief Flush only complete UTF-8 sequences
    ///@param is_final the is final
    void flush_complete_utf8_sequences(bool is_final) {
        if (buffer.empty()) return;
        
        std::string complete_content;
        size_t pos = 0;
        
        // Process complete UTF-8 sequences
        while (pos < buffer.size()) {
            unsigned char first = static_cast<unsigned char>(buffer[pos]);
            size_t seq_len = get_utf8_sequence_length(first);
            
            if (seq_len == 0) {
                // Invalid UTF-8 start byte, skip it
                pos++;
                continue;
            }
            
            // Check if we have a complete sequence
            if (pos + seq_len > buffer.size()) {
                // Incomplete sequence, stop here
                break;
            }
            
            // Add complete sequence to output
            complete_content.append(buffer, pos, seq_len);
            pos += seq_len;
        }
        
        // Send complete sequences if any
        if (!complete_content.empty()) {
            // Start content block if not started
            if (!content_block_started) {
                send_content_block_start();
            }
            send_content_delta(complete_content);
        }
        
        // Remove processed bytes from buffer
        if (pos > 0) {
            buffer.erase(0, pos);
        }
    }

    ///@brief Send content_block_start event
    void send_content_block_start() {
        json event = {
            {"type", "content_block_start"},
            {"index", content_index},
            {"content_block", {
                {"type", "text"},
                {"text", ""}
            }}
        };
        stream_callback("event: content_block_start\ndata: " + event.dump() + "\n\n", false);
        content_block_started = true;
    }

    ///@brief Send content_block_delta event
    ///@param content the content to send
    void send_content_delta(const std::string& content) {
        // Parse stream content for tool calls or reasoning (similar to OpenAI)
        StreamResult result = auto_chat_engine->parse_stream_content(content);
        
        if (result.type == StreamEventType::WAITING) {
            return;
        }
        
        json event;
        if (result.type == StreamEventType::TOOL_DONE) {
            // Handle tool use - Anthropic uses a different format
            // First close any text content block
            if (content_block_started) {
                send_content_block_stop();
                content_index++;
            }
            
            // Start tool use content block
            json tool_start = {
                {"type", "content_block_start"},
                {"index", content_index},
                {"content_block", {
                    {"type", "tool_use"},
                    {"id", result.tool_id},
                    {"name", result.tool_name},
                    {"input", {}}
                }}
            };
            stream_callback("event: content_block_start\ndata: " + tool_start.dump() + "\n\n", false);
            
            // Send tool input as delta
            json tool_delta = {
                {"type", "content_block_delta"},
                {"index", content_index},
                {"delta", {
                    {"type", "input_json_delta"},
                    {"partial_json", result.tool_args_str}
                }}
            };
            stream_callback("event: content_block_delta\ndata: " + tool_delta.dump() + "\n\n", false);
            
            // Close tool content block
            json tool_stop = {
                {"type", "content_block_stop"},
                {"index", content_index}
            };
            stream_callback("event: content_block_stop\ndata: " + tool_stop.dump() + "\n\n", false);
            content_index++;
            content_block_started = false;
        }
        else if (result.type == StreamEventType::REASONING || result.type == StreamEventType::CONTENT) {
            std::string text_content = result.content;
            if (text_content.empty()) return;
            
            event = {
                {"type", "content_block_delta"},
                {"index", content_index},
                {"delta", {
                    {"type", "text_delta"},
                    {"text", text_content}
                }}
            };
            stream_callback("event: content_block_delta\ndata: " + event.dump() + "\n\n", false);
        }
    }

    ///@brief Send content_block_stop event
    void send_content_block_stop() {
        json event = {
            {"type", "content_block_stop"},
            {"index", content_index}
        };
        stream_callback("event: content_block_stop\ndata: " + event.dump() + "\n\n", false);
        content_block_started = false;
    }

    ///@brief Send message_delta event with final usage info
    void send_message_delta(chat_meta_info_t& meta_info) {
        std::string stop_reason = "end_turn";
        if (meta_info.stop_reason == MAX_LENGTH_REACHED) {
            stop_reason = "max_tokens";
        } else if (meta_info.stop_reason == CANCEL_DETECTED) {
            stop_reason = "end_turn";
        }
        
        json event = {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", nullptr}
            }},
            {"usage", {
                {"output_tokens", meta_info.generated_tokens}
            }}
        };
        stream_callback("event: message_delta\ndata: " + event.dump() + "\n\n", false);
    }

    ///@brief Send message_stop event
    void send_message_stop() {
        json event = {
            {"type", "message_stop"}
        };
        stream_callback("event: message_stop\ndata: " + event.dump() + "\n\n", true);
    }

    ///@brief Buffer
    std::string buffer;
    ///@brief Model name
    std::string model_name;
    ///@brief auto_chat_engine 
    AutoModel* auto_chat_engine;
    ///@brief Stream callback
    StreamCallback stream_callback;
    ///@brief Message ID
    std::string message_id;
    ///@brief Content block started flag
    bool content_block_started;
    ///@brief Current content block index
    int content_index;
};

///@brief Custom ostream for Anthropic streaming
///@param model the model
///@param auto_chat_engine the auto chat engine
///@param callback the callback
///@return the streaming ostream
class streaming_ostream_anthropic : public std::ostream {
public:
    streaming_ostream_anthropic(const std::string& model, AutoModel* auto_chat_engine, 
                                 streaming_buf_anthropic::StreamCallback callback)
        : std::ostream(&buf), buf(model, auto_chat_engine, callback) {}
    
    ///@brief Send message_start event (call before generating)
    void send_message_start(int input_tokens) {
        buf.send_message_start(input_tokens);
    }
    
    ///@brief Finalize the message
    void finalize(chat_meta_info_t& meta_info) {
        buf.finalize(meta_info);
    }

private:
    ///@brief Buffer
    streaming_buf_anthropic buf;
};
