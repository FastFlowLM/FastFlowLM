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
#include "nlohmann/json.hpp"
#include "AutoModel/automodel.hpp"

using json = nlohmann::ordered_json;

class PromptCache {
private:
    uint64_t checksum_;
    std::vector<uint64_t> tool_checksums_;
    bool can_use_tool_cache_;
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
    PromptCache() : checksum_(0), can_use_tool_cache_(true) {}

    json tool_checksum(json& tools) {
        if (!tools.is_array() || tools.empty()) {
            can_use_tool_cache_ = tool_checksums_.empty();
            tool_checksums_.clear();
            return tools;
        }

        std::vector<uint64_t> new_tool_checksums = _calculate_tool_checksums(tools);

        can_use_tool_cache_ = tool_checksums_.size() <= new_tool_checksums.size();
        for (size_t i = 0; can_use_tool_cache_ && i < tool_checksums_.size(); ++i) {
            can_use_tool_cache_ = tool_checksums_[i] == new_tool_checksums[i];
        }

        json tools_to_insert = json::array();
        if (can_use_tool_cache_) {
            for (size_t i = tool_checksums_.size(); i < tools.size(); ++i) {
                tools_to_insert.push_back(tools[i]);
            }
        }
        else {
            tools_to_insert = tools;
        }

        tool_checksums_ = std::move(new_tool_checksums);
        return tools_to_insert;
    }

    bool can_use_tool_cache() const {
        return can_use_tool_cache_;
    }

    void reset_tool_checksum() {
        tool_checksums_.clear();
        can_use_tool_cache_ = true;
    }

    void update_tool_checksum(json& tools) {
        if (!tools.is_array() || tools.empty()) {
            reset_tool_checksum();
            return;
        }

        tool_checksums_ = _calculate_tool_checksums(tools);
        can_use_tool_cache_ = true;
    }


    bool can_use_cache(json& messages, chat_template_type_t template_type) {
        uint64_t check_sum_to_compare = 0;
        uint64_t new_checksum = 0;
        if (messages.size() > 2) {
            for (size_t i = 0; i < messages.size(); ++i) {
                const auto& msg = messages[i];
                const std::string role = msg.value("role", "");
                const std::string content = msg.value("content", "");

                bool has_tool_call = msg.contains("tool_calls");
                bool skip_this_message = (role == "tool") || has_tool_call;
                if (template_type == chat_template_type_t::harmony) {
                    skip_this_message = false;
                }

                if (skip_this_message) continue;

                if(i < messages.size() - 2)
                    check_sum_to_compare = _calculate_checksum(content.data(), content.size(), check_sum_to_compare);
                new_checksum = _calculate_checksum(content.data(), content.size(), new_checksum);
            }

            if (checksum_ == check_sum_to_compare) {
                checksum_ = new_checksum;
                return true;
            }
            else {
                checksum_ = new_checksum;
                return false;
            }
        }
        else {
            return false;
        }

    }

    void update_checksum(json& messages) {
        uint64_t new_checksum = 0;
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto& msg = messages[i];
            const std::string role = msg.value("role", "");
            const std::string content = msg.value("content", "");
            bool has_tool_call = msg.contains("tool_calls");
            bool skip_this_message = (role == "tool") || has_tool_call;
            if (skip_this_message) continue;
            new_checksum = _calculate_checksum(content.data(), content.size(), new_checksum);
        }
        checksum_ = new_checksum;
    }

    void reset() {
        checksum_ = checksum_ + 1;
    }
};
