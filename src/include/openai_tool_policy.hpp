#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

namespace openai_tools {

using json = nlohmann::ordered_json;

enum class ToolChoiceMode {
    none,
    auto_select,
    required,
    named,
};

class ToolPolicyError : public std::runtime_error {
public:
    ToolPolicyError(std::string message, std::string type, std::string param, int status_code)
        : std::runtime_error(std::move(message)),
          type(std::move(type)),
          param(std::move(param)),
          status_code(status_code) {}

    std::string type;
    std::string param;
    int status_code;
};

struct ToolPolicy {
    ToolChoiceMode mode = ToolChoiceMode::none;
    std::string required_function;
    bool parallel_tool_calls = true;
    bool has_tool_contract = false;
    json tools = json::array();

    bool requires_buffered_validation() const {
        return has_tool_contract;
    }
};

inline ToolPolicyError invalid_request(const std::string& message, const std::string& param) {
    return ToolPolicyError(message, "invalid_request_error", param, 400);
}

inline ToolPolicyError model_contract_error(const std::string& message) {
    return ToolPolicyError(message, "model_error", "tool_choice", 500);
}

inline bool valid_function_name(const std::string& name) {
    return !name.empty() && name.size() <= 64 && std::all_of(name.begin(), name.end(), [](unsigned char value) {
        return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
            (value >= '0' && value <= '9') || value == '_' || value == '-';
    });
}

inline std::string function_name(const json& tool, std::size_t index) {
    const std::string param = "tools[" + std::to_string(index) + "]";
    if (!tool.is_object() || !tool.contains("type") || !tool["type"].is_string() ||
        tool["type"] != "function") {
        throw invalid_request(param + " must be a function tool", param);
    }
    if (!tool.contains("function") || !tool["function"].is_object() ||
        !tool["function"].contains("name") || !tool["function"]["name"].is_string() ||
        !valid_function_name(tool["function"]["name"].get<std::string>())) {
        throw invalid_request(
            param + ".function.name must contain at most 64 letters, digits, underscores, or hyphens",
            param + ".function.name");
    }
    if (tool["function"].contains("strict") && !tool["function"]["strict"].is_boolean()) {
        throw invalid_request(param + ".function.strict must be a boolean", param + ".function.strict");
    }
    if (tool["function"].value("strict", false)) {
        throw invalid_request(
            param + ".function.strict is not supported because constrained JSON-schema decoding is unavailable",
            param + ".function.strict");
    }
    return tool["function"]["name"].get<std::string>();
}

inline ToolPolicy parse_tool_policy(const json& request) {
    ToolPolicy policy;
    if (request.contains("tools") && !request["tools"].is_array()) {
        throw invalid_request("tools must be an array", "tools");
    }
    policy.tools = request.contains("tools") ? request["tools"] : json::array();
    policy.has_tool_contract = !policy.tools.empty() || request.contains("tool_choice") ||
        request.contains("parallel_tool_calls");

    std::unordered_set<std::string> available_names;
    for (std::size_t i = 0; i < policy.tools.size(); ++i) {
        const auto name = function_name(policy.tools[i], i);
        if (!available_names.insert(name).second) {
            throw invalid_request("tool function names must be unique: " + name, "tools");
        }
    }

    if (request.contains("parallel_tool_calls")) {
        if (!request["parallel_tool_calls"].is_boolean()) {
            throw invalid_request("parallel_tool_calls must be a boolean", "parallel_tool_calls");
        }
        policy.parallel_tool_calls = request["parallel_tool_calls"].get<bool>();
    }

    if (!request.contains("tool_choice")) {
        policy.mode = policy.tools.empty() ? ToolChoiceMode::none : ToolChoiceMode::auto_select;
        return policy;
    }

    const auto& choice = request["tool_choice"];
    if (choice.is_string()) {
        const auto value = choice.get<std::string>();
        if (value == "none") {
            policy.mode = ToolChoiceMode::none;
            policy.tools = json::array();
        } else if (value == "auto") {
            policy.mode = ToolChoiceMode::auto_select;
        } else if (value == "required") {
            policy.mode = ToolChoiceMode::required;
        } else {
            throw invalid_request("tool_choice must be 'none', 'auto', 'required', or a named function", "tool_choice");
        }
    } else if (choice.is_object() && choice.contains("type") && choice["type"] == "function") {
        if (!choice.contains("function") ||
            !choice["function"].is_object() || !choice["function"].contains("name") ||
            !choice["function"]["name"].is_string() ||
            !valid_function_name(choice["function"]["name"].get<std::string>())) {
            throw invalid_request(
                "named tool_choice must have the shape {type:'function', function:{name:'...'}}",
                "tool_choice");
        }
        policy.mode = ToolChoiceMode::named;
        policy.required_function = choice["function"]["name"].get<std::string>();
        if (!available_names.contains(policy.required_function)) {
            throw invalid_request(
                "tool_choice names a function that is not present in tools: " + policy.required_function,
                "tool_choice.function.name");
        }
        json selected = json::array();
        for (const auto& tool : policy.tools) {
            if (tool["function"]["name"] == policy.required_function) {
                selected.push_back(tool);
            }
        }
        policy.tools = std::move(selected);
    } else if (choice.is_object() && choice.contains("type") && choice["type"] == "allowed_tools") {
        if (!choice.contains("allowed_tools") || !choice["allowed_tools"].is_object()) {
            throw invalid_request("allowed_tools tool_choice must contain an allowed_tools object", "tool_choice.allowed_tools");
        }
        const auto& allowed = choice["allowed_tools"];
        if (!allowed.contains("mode") || !allowed["mode"].is_string() ||
            (allowed["mode"] != "auto" && allowed["mode"] != "required")) {
            throw invalid_request("allowed_tools.mode must be 'auto' or 'required'", "tool_choice.allowed_tools.mode");
        }
        if (!allowed.contains("tools") || !allowed["tools"].is_array() || allowed["tools"].empty()) {
            throw invalid_request("allowed_tools.tools must be a non-empty array", "tool_choice.allowed_tools.tools");
        }

        std::unordered_set<std::string> allowed_names;
        for (std::size_t i = 0; i < allowed["tools"].size(); ++i) {
            const auto& selected = allowed["tools"][i];
            const auto param = "tool_choice.allowed_tools.tools[" + std::to_string(i) + "]";
            if (!selected.is_object() || !selected.contains("type") || selected["type"] != "function" ||
                !selected.contains("function") || !selected["function"].is_object() ||
                !selected["function"].contains("name") || !selected["function"]["name"].is_string()) {
                throw invalid_request(param + " must name a function", param);
            }
            const auto name = selected["function"]["name"].get<std::string>();
            if (!available_names.contains(name)) {
                throw invalid_request("allowed_tools names a function that is not present in tools: " + name, param);
            }
            allowed_names.insert(name);
        }

        policy.mode = allowed["mode"] == "required" ? ToolChoiceMode::required : ToolChoiceMode::auto_select;
        json selected_tools = json::array();
        for (const auto& tool : policy.tools) {
            if (allowed_names.contains(tool["function"]["name"].get<std::string>())) {
                selected_tools.push_back(tool);
            }
        }
        policy.tools = std::move(selected_tools);
    } else {
        throw invalid_request("tool_choice must be a supported string or function-tool object", "tool_choice");
    }

    if ((policy.mode == ToolChoiceMode::required || policy.mode == ToolChoiceMode::named) && policy.tools.empty()) {
        throw invalid_request("tool_choice requires at least one function in tools", "tool_choice");
    }
    return policy;
}

inline json apply_policy_prompt(json messages, const ToolPolicy& policy) {
    if (!messages.is_array()) {
        throw invalid_request("messages must be an array", "messages");
    }

    std::string directive;
    if (policy.mode == ToolChoiceMode::required) {
        directive = "Tool selection policy: call one or more of the provided functions before replying.";
    } else if (policy.mode == ToolChoiceMode::named) {
        directive = "Tool selection policy: call the function named '" + policy.required_function + "' before replying.";
    }
    if (!policy.parallel_tool_calls && policy.mode != ToolChoiceMode::none) {
        if (!directive.empty()) directive += " ";
        directive += "Emit no more than one function call in this response.";
    }

    if (!directive.empty()) {
        const json policy_message = {{"role", "system"}, {"content", directive}};
        messages.insert(messages.begin(), policy_message);
    }
    return messages;
}

inline void validate_tool_calls(const json& choices, const ToolPolicy& policy) {
    if (!choices.is_array() || choices.empty() || !choices[0].is_object() ||
        !choices[0].contains("message") || !choices[0]["message"].is_object()) {
        throw model_contract_error("model response is missing choices[0].message");
    }

    const auto& message = choices[0]["message"];
    if (message.contains("tool_calls") && !message["tool_calls"].is_null() &&
        !message["tool_calls"].is_array()) {
        throw model_contract_error("choices[0].message.tool_calls must be an array");
    }
    const bool has_calls = message.contains("tool_calls") && message["tool_calls"].is_array() &&
        !message["tool_calls"].empty();
    const std::size_t call_count = has_calls ? message["tool_calls"].size() : 0;

    if (!choices[0].contains("finish_reason") || !choices[0]["finish_reason"].is_string()) {
        throw model_contract_error("model response is missing choices[0].finish_reason");
    }
    const auto finish_reason = choices[0]["finish_reason"].get<std::string>();
    if ((has_calls && finish_reason != "tool_calls") || (!has_calls && finish_reason == "tool_calls")) {
        throw model_contract_error("finish_reason does not match the model's tool calls");
    }

    if (policy.mode == ToolChoiceMode::none && has_calls) {
        throw model_contract_error("model emitted a tool call while tool_choice was 'none'");
    }
    if ((policy.mode == ToolChoiceMode::required || policy.mode == ToolChoiceMode::named) && !has_calls) {
        throw model_contract_error("model did not emit a tool call required by tool_choice");
    }
    if (!policy.parallel_tool_calls && call_count > 1) {
        throw model_contract_error("model emitted parallel tool calls while parallel_tool_calls was false");
    }

    std::unordered_set<std::string> available_names;
    for (const auto& tool : policy.tools) {
        available_names.insert(tool["function"]["name"].get<std::string>());
    }

    if (!has_calls) return;
    for (std::size_t i = 0; i < call_count; ++i) {
        const auto& call = message["tool_calls"][i];
        const auto prefix = "choices[0].message.tool_calls[" + std::to_string(i) + "]";
        if (!call.is_object() || !call.contains("type") || !call["type"].is_string() ||
            call["type"] != "function" ||
            !call.contains("id") || !call["id"].is_string() || call["id"].get<std::string>().empty() ||
            !call.contains("function") || !call["function"].is_object() ||
            !call["function"].contains("name") || !call["function"]["name"].is_string()) {
            throw model_contract_error(prefix + " is not a valid function tool call");
        }
        const auto name = call["function"]["name"].get<std::string>();
        if (!available_names.contains(name)) {
            throw model_contract_error("model called an unavailable function: " + name);
        }
        if (policy.mode == ToolChoiceMode::named && name != policy.required_function) {
            throw model_contract_error("model called '" + name + "' instead of required function '" + policy.required_function + "'");
        }
        if (!call["function"].contains("arguments") || !call["function"]["arguments"].is_string()) {
            throw model_contract_error(prefix + ".function.arguments must be a JSON-encoded string");
        }
        try {
            const auto arguments = json::parse(call["function"]["arguments"].get<std::string>());
            if (!arguments.is_object()) {
                throw model_contract_error(prefix + ".function.arguments must encode a JSON object");
            }
        } catch (const nlohmann::json::exception&) {
            throw model_contract_error(prefix + ".function.arguments is not valid JSON");
        }
    }
}

inline json error_response(const ToolPolicyError& error) {
    return {{"error", {
        {"message", error.what()},
        {"type", error.type},
        {"param", error.param},
        {"code", error.status_code},
    }}};
}

} // namespace openai_tools
