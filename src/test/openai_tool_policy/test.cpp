#include "openai_tool_policy.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>

using openai_tools::ToolChoiceMode;
using openai_tools::ToolPolicyError;
using openai_tools::json;

namespace {

json tools() {
    return json::array({
        {{"type", "function"}, {"function", {
            {"name", "read_file"},
            {"description", "Read a file"},
            {"parameters", {{"type", "object"}}},
        }}},
        {{"type", "function"}, {"function", {
            {"name", "write_file"},
            {"description", "Write a file"},
            {"parameters", {{"type", "object"}}},
        }}},
    });
}

json minimal_tool() {
    return json::array({
        {{"type", "function"}, {"function", {{"name", "ping"}}}},
    });
}

json choices_with_calls(std::initializer_list<std::string> names) {
    json calls = json::array();
    for (const auto& name : names) {
        calls.push_back({
            {"id", "call_1"},
            {"type", "function"},
            {"function", {{"name", name}, {"arguments", R"({"path":"README.md"})"}}},
        });
    }
    return json::array({{
        {"message", {{"role", "assistant"}, {"tool_calls", calls}}},
        {"finish_reason", "tool_calls"},
    }});
}

json choices_with_text() {
    return json::array({{
        {"message", {{"role", "assistant"}, {"content", "done"}}},
        {"finish_reason", "stop"},
    }});
}

json buffered_response() {
    return {
        {"id", "chatcmpl-test"},
        {"object", "chat.completion"},
        {"created", 123},
        {"model", "test-model"},
        {"choices", choices_with_calls({"read_file"})},
        {"usage", {
            {"prompt_tokens", 10},
            {"completion_tokens", 5},
            {"total_tokens", 15},
        }},
    };
}

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("contract assertion failed");
    }
}

void expect_error(const std::function<void()>& action, int status_code) {
    try {
        action();
    } catch (const ToolPolicyError& error) {
        require(error.status_code == status_code);
        return;
    }
    throw std::runtime_error("expected ToolPolicyError");
}

} // namespace

int main() {
    const auto automatic = openai_tools::parse_tool_policy({{"tools", tools()}});
    require(automatic.mode == ToolChoiceMode::auto_select);
    require(automatic.tools.size() == 2);
    require(automatic.requires_buffered_validation());

    const auto plain_chat = openai_tools::parse_tool_policy(json::object());
    require(plain_chat.mode == ToolChoiceMode::none);
    require(!plain_chat.requires_buffered_validation());

    const auto none = openai_tools::parse_tool_policy({{"tools", tools()}, {"tool_choice", "none"}});
    require(none.mode == ToolChoiceMode::none);
    require(none.tools.empty());
    openai_tools::validate_tool_calls(choices_with_text(), none);
    expect_error([&] { openai_tools::validate_tool_calls(choices_with_calls({"read_file"}), none); }, 500);

    const auto required = openai_tools::parse_tool_policy({{"tools", tools()}, {"tool_choice", "required"}});
    openai_tools::validate_tool_calls(choices_with_calls({"read_file"}), required);
    expect_error([&] { openai_tools::validate_tool_calls(choices_with_text(), required); }, 500);

    const json named_choice = {{"type", "function"}, {"function", {{"name", "write_file"}}}};
    const auto named = openai_tools::parse_tool_policy({{"tools", tools()}, {"tool_choice", named_choice}});
    require(named.mode == ToolChoiceMode::named);
    require(named.tools.size() == 1);
    require(named.tools[0]["function"]["name"] == "write_file");
    openai_tools::validate_tool_calls(choices_with_calls({"write_file"}), named);
    expect_error([&] { openai_tools::validate_tool_calls(choices_with_calls({"read_file"}), named); }, 500);
    expect_error([&] {
        openai_tools::validate_tool_calls(choices_with_calls({"write_file", "write_file"}), named);
    }, 500);

    const json allowed_choice = {
        {"type", "allowed_tools"},
        {"allowed_tools", {
            {"mode", "required"},
            {"tools", json::array({
                {{"type", "function"}, {"function", {{"name", "read_file"}}}},
            })},
        }},
    };
    const auto allowed = openai_tools::parse_tool_policy({{"tools", tools()}, {"tool_choice", allowed_choice}});
    require(allowed.mode == ToolChoiceMode::required);
    require(allowed.tools.size() == 1);
    require(allowed.tools[0]["function"]["name"] == "read_file");
    openai_tools::validate_tool_calls(choices_with_calls({"read_file"}), allowed);

    const auto serial = openai_tools::parse_tool_policy({
        {"tools", tools()}, {"tool_choice", "required"}, {"parallel_tool_calls", false},
    });
    expect_error([&] {
        openai_tools::validate_tool_calls(choices_with_calls({"read_file", "write_file"}), serial);
    }, 500);

    expect_error([&] { openai_tools::parse_tool_policy({{"tool_choice", "required"}}); }, 400);
    expect_error([&] {
        openai_tools::parse_tool_policy({{"tools", tools()}, {"tool_choice", "sometimes"}});
    }, 400);
    expect_error([&] {
        openai_tools::parse_tool_policy({{"tools", tools()}, {"parallel_tool_calls", "false"}});
    }, 400);
    expect_error([&] { openai_tools::parse_tool_policy({{"tools", nullptr}}); }, 400);
    expect_error([&] {
        openai_tools::parse_tool_policy({{"tools", tools()}, {"parallel_tool_calls", nullptr}});
    }, 400);
    expect_error([&] {
        openai_tools::parse_tool_policy({
            {"tools", tools()},
            {"tool_choice", {{"type", "function"}, {"function", {{"name", "missing"}}}}},
        });
    }, 400);
    expect_error([&] {
        auto unavailable = allowed_choice;
        unavailable["allowed_tools"]["tools"][0]["function"]["name"] = "missing";
        openai_tools::parse_tool_policy({{"tools", tools()}, {"tool_choice", unavailable}});
    }, 400);

    auto malformed = choices_with_calls({"read_file"});
    malformed[0]["message"]["tool_calls"][0]["function"]["arguments"] = "{{bad";
    expect_error([&] { openai_tools::validate_tool_calls(malformed, required); }, 500);

    auto wrong_finish_reason = choices_with_calls({"read_file"});
    wrong_finish_reason[0]["finish_reason"] = "stop";
    expect_error([&] { openai_tools::validate_tool_calls(wrong_finish_reason, required); }, 500);

    auto strict_tools = tools();
    strict_tools[0]["function"]["strict"] = true;
    expect_error([&] { openai_tools::parse_tool_policy({{"tools", strict_tools}}); }, 400);

    auto invalid_description = tools();
    invalid_description[0]["function"]["description"] = json::array();
    expect_error([&] {
        openai_tools::parse_tool_policy({{"tools", invalid_description}});
    }, 400);

    auto invalid_parameters = tools();
    invalid_parameters[0]["function"]["parameters"] = "not a schema";
    expect_error([&] {
        openai_tools::parse_tool_policy({{"tools", invalid_parameters}});
    }, 400);

    auto invalid_strict = tools();
    invalid_strict[0]["function"]["strict"] = "false";
    expect_error([&] {
        openai_tools::parse_tool_policy({{"tools", invalid_strict}});
    }, 400);

    const auto minimal = openai_tools::parse_tool_policy({{"tools", minimal_tool()}});
    require(minimal.tools.size() == 1);

    auto unicode_description = minimal_tool();
    unicode_description[0]["function"]["description"] = "Meteo pour Sao Paulo, Tokyo, and \U0001f30d";
    const auto unicode = openai_tools::parse_tool_policy({{"tools", unicode_description}});
    require(unicode.tools[0]["function"]["description"] ==
        unicode_description[0]["function"]["description"]);

    const auto default_stream = openai_tools::parse_stream_options(json::object());
    require(!default_stream.include_usage);
    const auto chunks_without_usage =
        openai_tools::build_buffered_stream_chunks(buffered_response(), default_stream);
    require(chunks_without_usage.size() == 2);
    require(!chunks_without_usage[0].contains("usage"));
    require(!chunks_without_usage[1].contains("usage"));
    require(chunks_without_usage[1]["choices"][0]["finish_reason"] == "tool_calls");

    const auto usage_stream = openai_tools::parse_stream_options({
        {"stream_options", {{"include_usage", true}}},
    });
    require(usage_stream.include_usage);
    const auto chunks_with_usage =
        openai_tools::build_buffered_stream_chunks(buffered_response(), usage_stream);
    require(chunks_with_usage.size() == 3);
    require(chunks_with_usage[0]["usage"].is_null());
    require(chunks_with_usage[1]["usage"].is_null());
    require(chunks_with_usage[2]["choices"].empty());
    require(chunks_with_usage[2]["usage"]["total_tokens"] == 15);

    expect_error([&] {
        openai_tools::parse_stream_options({{"stream_options", json::array()}});
    }, 400);
    expect_error([&] {
        openai_tools::parse_stream_options({
            {"stream_options", {{"include_usage", "true"}}},
        });
    }, 400);

    const auto prompted = openai_tools::apply_policy_prompt(
        json::array({{{"role", "user"}, {"content", "write it"}}}), named);
    require(prompted.size() == 2);
    require(prompted[0]["role"] == "system");
    require(prompted[0]["content"].get<std::string>().find("write_file") != std::string::npos);

    std::cout << "openai_tool_policy contract tests passed\n";
}
