#!/usr/bin/env python3

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request


ENDPOINT = "http://127.0.0.1:52625/v1/chat/completions"
REQUEST_TIMEOUT = 300.0

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}

CHAIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_dataset",
            "description": "Read a dataset by identifier",
            "parameters": {
                "type": "object",
                "properties": {"dataset_id": {"type": "string"}},
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total",
            "description": "Calculate the total of a list of integers",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["values"],
            },
        },
    },
]

WORKFLOW_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at a path",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file at a path",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def post(payload):
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def post_sse(payload):
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        events = []
        for raw_line in response.read().decode("utf-8").splitlines():
            if not raw_line.startswith("data: "):
                continue
            data = raw_line[6:]
            events.append(data if data == "[DONE]" else json.loads(data))
        return response.status, events


def base_request(model, messages, tools, tool_choice="auto", max_tokens=256):
    return {
        "model": model,
        "messages": copy.deepcopy(messages),
        "tools": copy.deepcopy(tools),
        "tool_choice": copy.deepcopy(tool_choice),
        "parallel_tool_calls": False,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }


def response_message(status, body):
    if status != 200:
        raise AssertionError(f"HTTP {status}: {body}")
    return body["choices"][0]["message"], body["choices"][0]["finish_reason"]


def single_call(message, finish_reason, expected_name):
    calls = message.get("tool_calls", [])
    if finish_reason != "tool_calls" or len(calls) != 1:
        raise AssertionError(
            f"expected one {expected_name} call; finish={finish_reason}, "
            f"calls={calls}, content={message.get('content')}"
        )
    call = calls[0]
    if call["function"]["name"] != expected_name:
        raise AssertionError(f"expected {expected_name}, got {call['function']['name']}")
    arguments = json.loads(call["function"]["arguments"])
    if not isinstance(arguments, dict):
        raise AssertionError(f"arguments are not an object: {arguments!r}")
    return call, arguments


def expect_request_error(model, mutate, expected_param):
    payload = base_request(
        model,
        [{"role": "user", "content": "Get the weather for Boston."}],
        [WEATHER_TOOL],
    )
    mutate(payload)
    status, body = post(payload)
    error = body.get("error", {})
    if status != 400 or error.get("type") != "invalid_request_error":
        raise AssertionError(f"expected request error, got HTTP {status}: {body}")
    if error.get("param") != expected_param:
        raise AssertionError(f"expected param {expected_param}, got {error.get('param')}: {body}")


def test_request_validation(model):
    expect_request_error(
        model,
        lambda payload: payload["tools"][0]["function"].update({"description": []}),
        "tools[0].function.description",
    )
    expect_request_error(
        model,
        lambda payload: payload["tools"][0]["function"].update({"parameters": "bad"}),
        "tools[0].function.parameters",
    )
    expect_request_error(
        model,
        lambda payload: payload["tools"][0]["function"].update({"strict": True}),
        "tools[0].function.strict",
    )
    expect_request_error(
        model,
        lambda payload: payload.update(
            {"stream": True, "stream_options": {"include_usage": "true"}}
        ),
        "stream_options.include_usage",
    )


def test_none(model):
    status, body = post(
        base_request(
            model,
            [{"role": "user", "content": "Call get_weather for Boston."}],
            [WEATHER_TOOL],
            "none",
            max_tokens=128,
        )
    )
    message, finish = response_message(status, body)
    if message.get("tool_calls") or finish == "tool_calls":
        raise AssertionError(f"tool_choice none returned calls: {body}")


def test_required(model):
    status, body = post(
        base_request(
            model,
            [{"role": "user", "content": "Get the weather for Boston using the function."}],
            [WEATHER_TOOL],
            "required",
        )
    )
    message, finish = response_message(status, body)
    _, arguments = single_call(message, finish, "get_weather")
    if not isinstance(arguments.get("location"), str):
        raise AssertionError(f"location is not a string: {arguments!r}")


def test_named(model):
    choice = {"type": "function", "function": {"name": "get_weather"}}
    status, body = post(
        base_request(
            model,
            [{"role": "user", "content": "Get the weather for Tokyo using a function."}],
            [WEATHER_TOOL, TIME_TOOL],
            choice,
        )
    )
    message, finish = response_message(status, body)
    single_call(message, finish, "get_weather")


def test_allowed_tools(model):
    choice = {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "required",
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        },
    }
    status, body = post(
        base_request(
            model,
            [{"role": "user", "content": "Get the weather for Paris using a function."}],
            [WEATHER_TOOL, TIME_TOOL],
            choice,
        )
    )
    message, finish = response_message(status, body)
    single_call(message, finish, "get_weather")


def test_fail_closed_and_recovery(model):
    status, body = post(
        base_request(
            model,
            [{"role": "user", "content": "Get the weather for Miami using the function."}],
            [WEATHER_TOOL],
            "required",
            max_tokens=1,
        )
    )
    if status < 500 or body.get("error", {}).get("type") != "model_error":
        raise AssertionError(f"truncated call escaped as HTTP {status}: {body}")
    test_required(model)


def test_chain(model):
    messages = [
        {
            "role": "system",
            "content": "Complete the workflow using functions. Do not guess tool results.",
        },
        {
            "role": "user",
            "content": (
                "Read dataset alpha. After receiving it, calculate the total of its values. "
                "After receiving the total, answer with the final total."
            ),
        },
    ]
    status, body = post(base_request(model, messages, CHAIN_TOOLS))
    first_message, first_finish = response_message(status, body)
    first_call, first_args = single_call(first_message, first_finish, "read_dataset")
    if first_args.get("dataset_id") != "alpha":
        raise AssertionError(f"wrong dataset arguments: {first_args!r}")

    messages.extend(
        [
            first_message,
            {
                "role": "tool",
                "tool_call_id": first_call["id"],
                "name": "read_dataset",
                "content": json.dumps({"dataset_id": "alpha", "values": [2, 3, 5]}),
            },
        ]
    )
    status, body = post(base_request(model, messages, CHAIN_TOOLS))
    second_message, second_finish = response_message(status, body)
    second_call, second_args = single_call(second_message, second_finish, "calculate_total")
    if second_args.get("values") != [2, 3, 5]:
        raise AssertionError(f"tool result not carried into second call: {second_args!r}")

    messages.extend(
        [
            second_message,
            {
                "role": "tool",
                "tool_call_id": second_call["id"],
                "name": "calculate_total",
                "content": json.dumps({"total": 10}),
            },
        ]
    )
    status, body = post(base_request(model, messages, CHAIN_TOOLS))
    final_message, final_finish = response_message(status, body)
    content = final_message.get("content") or ""
    if final_message.get("tool_calls") or final_finish != "stop" or "10" not in content:
        raise AssertionError(f"wrong final response: {body}")


def test_stream_usage(model):
    for include_usage in (False, True):
        payload = base_request(
            model,
            [{"role": "user", "content": "Get the weather for Boston using the function."}],
            [WEATHER_TOOL],
            "required",
        )
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": include_usage}
        status, events = post_sse(payload)
        if status != 200 or not events or events[-1] != "[DONE]":
            raise AssertionError(f"invalid SSE response: HTTP {status}: {events}")
        chunks = events[:-1]
        if include_usage:
            if len(chunks) != 3:
                raise AssertionError(f"expected three usage chunks: {chunks}")
            if any(chunk.get("usage", "missing") is not None for chunk in chunks[:-1]):
                raise AssertionError(f"ordinary chunks must contain usage null: {chunks}")
            if chunks[-1].get("choices") != [] or not isinstance(chunks[-1].get("usage"), dict):
                raise AssertionError(f"invalid final usage chunk: {chunks[-1]}")
        elif any("usage" in chunk for chunk in chunks):
            raise AssertionError(f"usage appeared without include_usage: {chunks}")


def test_skill_workflow(model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous tool-using agent. Follow every step in order: "
                "list .skills, read .skills/task-manager/SKILL.md, read tasks.md, write "
                "the updated tasks.md, then answer. Never guess results or repeat a step."
            ),
        },
        {"role": "user", "content": "Add 'Buy bread, eggs and milk' to my todo list."},
    ]
    read_skill = False
    read_tasks = False
    seen = set()
    write_message = None
    write_call = None

    for _ in range(8):
        status, body = post(base_request(model, messages, WORKFLOW_TOOLS, max_tokens=768))
        message, finish = response_message(status, body)
        calls = message.get("tool_calls", [])
        if finish != "tool_calls" or len(calls) != 1:
            raise AssertionError(f"workflow stopped before write: {body}")
        call = calls[0]
        name = call["function"]["name"]
        arguments = json.loads(call["function"]["arguments"])
        signature = (name, json.dumps(arguments, sort_keys=True))
        if signature in seen:
            raise AssertionError(f"workflow repeated a call: {signature!r}")
        seen.add(signature)
        path = arguments.get("path")

        if name == "list_directory" and path in (".", ".skills"):
            result = (
                {"subdirectories": [".skills"], "files": ["tasks.md"]}
                if path == "."
                else {"subdirectories": ["task-manager"], "files": []}
            )
        elif name == "list_directory" and path == ".skills/task-manager":
            result = {"subdirectories": [], "files": ["SKILL.md"]}
        elif name == "read_file" and path == ".skills/task-manager/SKILL.md":
            read_skill = True
            result = {
                "content": "Read tasks.md, preserve it, and append the new unchecked task with write_file."
            }
        elif name == "read_file" and path == "tasks.md":
            read_tasks = True
            result = {"content": "# My Tasks\n- [ ] Call the dentist"}
        elif name == "write_file":
            content = arguments.get("content", "")
            if not read_skill or not read_tasks:
                raise AssertionError("workflow wrote before reading its skill and current tasks")
            if path != "tasks.md" or "Call the dentist" not in content or "bread" not in content.lower():
                raise AssertionError(f"invalid write call: {arguments!r}")
            write_message = message
            write_call = call
            break
        else:
            raise AssertionError(f"unexpected workflow call: {name}({arguments!r})")

        messages.extend(
            [
                message,
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(result),
                },
            ]
        )

    if write_call is None:
        raise AssertionError(f"workflow did not reach write_file: {seen!r}")

    messages.extend(
        [
            write_message,
            {
                "role": "tool",
                "tool_call_id": write_call["id"],
                "name": "write_file",
                "content": json.dumps({"status": "success", "path": "tasks.md"}),
            },
        ]
    )
    status, body = post(base_request(model, messages, WORKFLOW_TOOLS, max_tokens=768))
    message, finish = response_message(status, body)
    if message.get("tool_calls") or finish != "stop" or "bread" not in (message.get("content") or "").lower():
        raise AssertionError(f"workflow did not finish correctly: {body}")


def run(model, include_workflow):
    tests = [
        ("request_validation", test_request_validation),
        ("none", test_none),
        ("required", test_required),
        ("named", test_named),
        ("allowed_tools", test_allowed_tools),
        ("fail_closed_and_recovery", test_fail_closed_and_recovery),
        ("chain", test_chain),
        ("stream_usage", test_stream_usage),
    ]
    if include_workflow:
        tests.append(("skill_workflow", test_skill_workflow))

    results = {}
    for name, test in tests:
        try:
            test(model)
            results[name] = {"status": "pass"}
        except Exception as error:
            results[name] = {"status": "fail", "error": str(error)}
    return results


def main():
    global ENDPOINT, REQUEST_TIMEOUT

    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenAI tool-calling contract tests against a live "
            "FastFlowLM server."
        )
    )
    parser.add_argument("model", help="Model tag loaded by the server")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:52625/v1",
        help="OpenAI-compatible API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="Also run a multi-turn synthetic skill workflow",
    )
    args = parser.parse_args()
    ENDPOINT = f"{args.base_url.rstrip('/')}/chat/completions"
    REQUEST_TIMEOUT = args.timeout
    report = {args.model: run(args.model, args.workflow)}
    print(json.dumps(report, indent=2))
    if any(result["status"] != "pass" for result in report[args.model].values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
