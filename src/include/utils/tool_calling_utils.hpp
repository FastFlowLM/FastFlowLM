#pragma once

#include <cctype>
#include <cstdlib>
#include <string>
#include <nlohmann/json.hpp>

inline std::string trim_tool_argument_token(const std::string& text) {
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        start++;
    }

    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        end--;
    }

    return text.substr(start, end - start);
}

inline nlohmann::ordered_json normalize_tool_argument_value(const std::string& value_text) {
    std::string trimmed = trim_tool_argument_token(value_text);
    if (trimmed.empty()) {
        return "";
    }

    if (trimmed.size() >= 2 && trimmed.front() == '\'' && trimmed.back() == '\'') {
        return trimmed.substr(1, trimmed.size() - 2);
    }

    try {
        return nlohmann::ordered_json::parse(trimmed);
    }
    catch (...) {
        return trimmed;
    }
}

inline nlohmann::ordered_json normalize_tool_arguments(const std::string& arguments_text) {
    try {
        return nlohmann::ordered_json::parse(arguments_text);
    }
    catch (...) {
        return arguments_text;
    }
}