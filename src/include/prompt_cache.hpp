#pragma once
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <locale>
#include <random>
#include <utility>
#include <vector>
#include <algorithm>
#include "nlohmann/json.hpp"
#include "AutoModel/automodel.hpp"

using json = nlohmann::ordered_json;

class PromptCache {
private:
    std::vector<uint64_t> message_checksums_;
    std::vector<uint64_t> tool_checksums_;

    uint64_t _calculate_single_message_checksum(const json& message) {
        // Hash only the content-bearing fields so messages produced by
        // different backends (e.g. local vs cloud) compare equal as long
        // as their semantic payload matches.
        uint64_t sum = 0;
        auto mix = [&](const char* key) {
            if (message.contains(key)) {
                const std::string s = message[key].dump();
                sum = _calculate_checksum(s.data(), s.size(), sum);
            }
        };
        mix("role");
        mix("content");
        mix("tool_calls");
        mix("tool_call_id");
        mix("name");
        return sum;
    }

    std::vector<uint64_t> _calculate_message_checksums(const json& messages, size_t end) {
        const size_t message_count = std::min(end, messages.size());
        std::vector<uint64_t> checksums;
        checksums.reserve(message_count);
        for (size_t i = 0; i < message_count; ++i) {
            checksums.push_back(_calculate_single_message_checksum(messages[i]));
        }
        return checksums;
    }

    std::vector<uint64_t> _calculate_tool_checksums(const json& tools) {
        std::vector<uint64_t> tool_checksums;
        tool_checksums.reserve(tools.size());
        for (const auto& tool : tools) {
            const std::string tool_string = tool.dump();
            tool_checksums.push_back(_calculate_checksum(tool_string.data(), tool_string.size()));
        }
        return tool_checksums;
    }

    uint64_t _calculate_checksum(const void* p, size_t len, uint64_t sum = 0) {
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
    
public:
    PromptCache() : message_checksums_(), tool_checksums_() {}

    bool can_use_tool_cache(json& tools) {
        std::vector<uint64_t> new_tool_checksums = _calculate_tool_checksums(tools);

        if (tool_checksums_.size() == new_tool_checksums.size()){
            for (size_t i = 0; i < tool_checksums_.size(); ++i) {
                if (tool_checksums_[i] != new_tool_checksums[i]) {
                    tool_checksums_ = std::move(new_tool_checksums);
                    return false;
                }
            }
            return true;
        }
        else {
            tool_checksums_ = std::move(new_tool_checksums);
            return false;
        }
    }

    void reset_tool_checksum() {
        tool_checksums_.clear();
    }

    void update_tool_checksum(json& tools) {
        if (!tools.is_array() || tools.empty()) {
            reset_tool_checksum();
            return;
        }

        tool_checksums_ = _calculate_tool_checksums(tools);
    }


    bool can_use_message_cache(json& messages, chat_template_type_t template_type) {
        (void)template_type;
        if (messages.size() <= 2) {
            return false;
        }

        std::vector<uint64_t> new_checksums = _calculate_message_checksums(messages, messages.size());
        const size_t prefix_len = messages.size() - 2;
        const bool prefix_match =
            message_checksums_.size() <= prefix_len &&
            std::equal(message_checksums_.begin(), message_checksums_.end(), new_checksums.begin());
        message_checksums_ = std::move(new_checksums);
        return prefix_match;
    }

    void update_message_checksum(json& messages) {
        message_checksums_ = _calculate_message_checksums(messages, messages.size());
    }


    bool can_use_cache(json& messages, chat_template_type_t template_type, json& tools) {
        (void)template_type;
        if (messages.size() <= 2) {
            update_message_checksum(messages);
            update_tool_checksum(tools);
            return false;
        }

        std::vector<uint64_t> new_checksums = _calculate_message_checksums(messages, messages.size());
        std::vector<uint64_t> new_tool_checksums = _calculate_tool_checksums(tools);

        // Cache is reusable when every previously seen message still appears
        // (in order) at the start of the new conversation, allowing rounds
        // produced by other backends (cloud) to be appended without
        // invalidating the locally-built KV cache prefix.
        const size_t prefix_len = messages.size() - 2;
        const bool can_use_message =
            message_checksums_.size() <= prefix_len &&
            std::equal(message_checksums_.begin(), message_checksums_.end(), new_checksums.begin());
        const bool can_use_tools = tool_checksums_ == new_tool_checksums;

        message_checksums_ = std::move(new_checksums);
        tool_checksums_ = std::move(new_tool_checksums);

        return can_use_message && can_use_tools;
    }

    /// @brief Reset the checksum to force cache miss
    /// @note Clears the stored checksums so the next can_use_cache call misses.
    void reset() {
        message_checksums_.clear();
        reset_tool_checksum();
    }
};