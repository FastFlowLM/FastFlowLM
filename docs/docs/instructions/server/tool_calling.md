---
layout: docs
title: Tool Calling
nav_order: 8
parent: Local Server (Server Mode)
---

# 🔧 Tool Calling

**Tool calling** (also known as **function calling**) provides a powerful and flexible way for FLM models to interface with external systems and access data outside their training data. This guide shows how you can connect a model to data and actions provided by your application. 

## ⚙️ How Tool Calling works

Let's begin by understanding a few key terms about tool calling. After we have a shared vocabulary for tool calling, we'll show you how it's done with some practical examples.

1. Tools

   A tool refers in the abstract to a piece of functionality that we tell the model it has access to. You are essentially telling the model, "If you need to do $X$, here is the definition of a tool you can use."

   For example, you might define a tool called `get_temperature` and give the model access to it.
   When we make an API request to the model with a prompt, we can include the tools the model could consider using. For example, if you ask the model the temperature of any city, the model will call the `get_temperature` tool that takes `city` as an argument.

2. Tool calls

   A **tool call** is the model's response when it decides it needs external help. The model does not execute the tool itself. Instead, it pauses generating text and returns a structured output.

   For example, if the model receives a prompt like `what is the temperature in New York?`, it could respond to that prompt with a **tool call** for the `get_temperature` tool, with `New York` as the `city` argument.

3. Tool call outputs

   A **tool call output** refers to the response a tool generates using the input from a model's tool call. The **tool call output** can either be structured JSON or plain text. 
   
   For example:
   - The model has access to a `get_temperature` tool that takes `city` as an argument
   - In response to a prompt like `what is the temperature in New York?` the model returns a tool call that contains a tool name with a value of `get_temperature` and a `city` argument with a value of `New York`
   - The **tool call output** might return a JSON object (e.g., `{"temperature": "25°C", "city": "New York"}`, indicating a current temperature of 25 degrees)

4. The full workflow

   Let's walk through the full lifecycle of the `get_temperature` example:

   - Step 1:

      In response to a prompt like "what's the temperature in New York?" the model returns a **tool call** that contains a tool name with a value of `get_temperature` and a `city` argument with a value of `New York`
   
   - Step 2:

      After receiving the **tool name** and **argument**, you can either   
      - Invoke the tool manually using a [Python script](#-how-to-use-tool-calling-with-flm-via-python-script), or

      - Let a client that supports tool calling (e.g., [Open WebUI](#️-how-to-use-tool-calling-with-flm-in-open-webui)) automatically execute the tool.

      The **tool call output** might return a JSON object (e.g., `{"temperature": "25°C", "city": "New York"}`, indicating a current temperature of 25 degrees).

   - Step 3:

      Finally, the messages containing **tool definition**, the **original prompt**, the model's **tool call**, and the **tool call output**, are combined either by [Python script](#-how-to-use-tool-calling-with-flm-via-python-script) or a client like [Open WebUI](#️-how-to-use-tool-calling-with-flm-in-open-webui) (as in Step 2) and sent back to the model.

      The model then produces a natural-language response such as:
      ```
      The temperature in New York today is 25°C.
      ```

> 📘 For more information, you can check out OpenAI tool-calling [docs](https://platform.openai.com/docs/guides/function-calling).

## 🤖 Tool calling supported models

Tool calling has been supported since `v0.9.26`.

Please refer to the [model card](https://fastflowlm.com/docs/models/) to see whether tool calling is supported for each model.

## 🐍 How to use tool calling with FLM via Python script

This section walks through an end‑to‑end tool‑calling flow in both streaming and non‑streaming modes. 
> ⚠️ This simulates a tool‑calling scenario; no real tools are executed.


### 🌡️ Example: `get_temperature`
A streaming‑mode example demonstrating how the `get_temperature` tool retrieves the current temperature.

```python
from openai import OpenAI
import json

# Connect to local FastFlowLM server
client = OpenAI(
    base_url="http://127.0.0.1:52625/v1",
    api_key="flm",
)

# OpenAI Chat Completions Tool definition
tools = [
{
    "type": "function",
    "function": {
            "name": "get_temperature",
            "description": "Get the current temperature for a city",
            "parameters": {
                "type": "object",
                "required": ["city"],
                "properties": {
                    "city": {"type": "string", "description": "The name of the city"}
                }
            }
        }
    }
]

# Tool implementation
def get_temperature(city: str) -> str:
    return {"New York":"22°C","London":"15°C","Tokyo":"18°C"}.get(city, "Unknown")

# Step 1:
## Original user message
messages = [
    {"role": "user", "content": "What is the temperature in New York?"}
]

## Ask model; it should emit a tool call
tool_calls = []
assistant_content = ""
for chunk in client.chat.completions.create(
    model="qwen3-it:4b",
    messages=messages,
    tools=tools,
    stream=True,
):
    delta = chunk.choices[0].delta

    # Collect assistant text (if any)
    if hasattr(delta, "content") and delta.content is not None:
        assistant_content += delta.content

    # Collect tool call tokens
    if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
        for tc in delta.tool_calls:
            tool_calls.append(tc)


# Step 2:
## Add assistant message (with tool calls) to messages
messages.append(
    {
        "role": "assistant",
        "content": assistant_content,      # may be None
        "tool_calls": tool_calls # important: keep tool_calls
    }
)

## Execute tool calls and add tool call outputs back
if tool_calls:
    for tc in tool_calls:
        if tc.function.name == "get_temperature":
            args = json.loads(tc.function.arguments)
            temp = get_temperature(args["city"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"temperature": temp}),
                }
            )

print("Messages so far:")
print(messages)

# Step 3:
## Combine all the messages and ask again
for chunk in client.chat.completions.create(
    model="qwen3-it:4b",
    messages=messages,
    tools=tools,
    stream=True,
):
    delta = chunk.choices[0].delta
    if hasattr(delta, "content") and delta.content is not None:
        print(delta.content, end="", flush=True)

```

### 🔮 Example: `get_horoscope`
A non‑streaming example showing how the `get_horoscope` tool returns a daily horoscope for a given astrological sign.

```python
from openai import OpenAI
import json

# Connect to local FastFlowLM server
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="flm",
)

# OpenAI Chat Completions Tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_horoscope",
            "description": "Get today's horoscope for an astrological sign.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sign": {
                        "type": "string",
                        "description": "An astrological sign like Taurus or Aquarius",
                    },
                },
                "required": ["sign"],
            },
        },
    },
]

# Tool implementation
def get_horoscope(sign: str) -> str:
    return f"{sign}: Next Tuesday you will befriend a baby otter."

# Step 1:
## Original user message
messages = [
    {"role": "user", "content": "What is my horoscope? I am an Aquarius."}
]

## Ask model; it should emit a tool call
resp1 = client.chat.completions.create(
    model="qwen3-it:4b",
    messages=messages,
    tools=tools,
)
assistant_msg = resp1.choices[0].message

# Step 2:
## Add assistant message (with tool calls) to messages
messages.append(
    {
        "role": "assistant",
        "content": assistant_msg.content,      # may be None
        "tool_calls": assistant_msg.tool_calls # tool_calls
    }
)

## Execute tool calls and add tool call outputs back
if assistant_msg.tool_calls:
    for tc in assistant_msg.tool_calls:
        if tc.function.name == "get_horoscope":
            args = json.loads(tc.function.arguments)  # {"sign": "..."}
            horoscope = get_horoscope(args["sign"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"horoscope": horoscope}),
                }
            )

print("Messages so far:")
print(messages)

# Step 3:
## Combine all the messages and ask again
resp2 = client.chat.completions.create(
    model="qwen3-it:4b",
    messages=messages,
    tools=tools,
)

print("\nFinal output:")
print(resp2.choices[0].message.content)

```

## 🕸️ How to use tool calling with FLM in Open WebUI (Stream Mode Only)
### 🛠️ How to set up and use a tool

1. Follow the quick Open WebUI setup guide [here](https://fastflowlm.com/docs/instructions/server/webui/#-run-open-webui-with-fastflowlm).

2. Disable Builtin Tools (These may include functionality that could interfere with your expected results.)
    - In the **bottom-left corner** of the Open WebUI page, click the `User` icon, then select `Settings`.
    - In the **bottom panel**, open `Admin Settings`.
    - From the **left sidebar**, click `Models`.
    - Locate the tool‑calling model you want to use (e.g. `qwen3-it:4b`).
    - Under `Capabilities`, uncheck `Builtin Tools`.
    - Click `Save & Update`.

3. Go to the Open WebUI community (you may need to sign up or log in).
    - In the **left sidebar** of the Open WebUI page, navigate to `Workspace`.
    - In the **top bar**, open `Tools`.
    - Click `Discover a tool` to find interesting tools.

4. Get a tool in your Open WebUI.
    - You can scroll through the list to find something interesting, or use the search box to look up a tool. (Keep in mind that not every tool works perfectly, and many require an API key.)
    - For example, try searching for `arXiv Search Tool` — it has worked well in our tests and doesn’t require an API key.
    - Click on the tool and navigate to the tool page.
    - Click `Get` and then `Import`.
    - Configure the tool as guided on the tool page if necessary (for example, provide an API key; in the `arXiv Search Tool` case, it's **not** needed).
    - Click `Save` in the **bottom-right corner**.

5. Use `arXiv Search Tool` with your FLM tool-calling models.
    - Start a new chat.
    - Select a tool-calling model, such as `qwen3-it:4b` (One of the best-performing models in our tool‑calling tests).
    - Under the chat box, click the `Integration Icon` > `Tools` > activate `arXiv Search Tool`.
    - In the **top-right**, click `Controls` and change `Function Calling` from `Default` to `Native`.

        (In `Default` mode, the tool definition and tool call output are treated as regular user content, meaning the model must interpret them purely through its own comprehension rather than using actual tool‑calling capabilities.)
    - Ask the model what tools it supports by entering a prompt like `What tools do you have?`. 
    - For specific tasks, enter a direct request, for example: `Search for papers about Mamba LLM`. FLM together with Open WebUI will behave as follows:
        - FLM receives the prompt and generates the **tool call** tokens, which are sent back to Open WebUI (Step 1).
        - Open WebUI invokes the tool according to the **tool call** tokens and gets **tool call outputs** (Step 2).
        - Open WebUI then combines all messages and sends them back to FLM, which produces the final answer containing the relevant list of papers (Step 3).


### ⭐ Recommended tools
Here we recommend some useful tools and walk you through their step‑by‑step setup.
#### ▶️ Tool: Play YouTube Video
1. Get a YouTube API key:
   - Go to [Google Cloud Console API Library](https://console.cloud.google.com/apis/library) (You may need to sign up or log in at this page).
   - Search for `YouTube Data API v3` and then select it to open the API details page.
   - Click `Enable`. You will then be automatically redirected to the [Google Cloud Console Dashboard](https://console.cloud.google.com/apis/dashboard).
   - In the **left sidebar**, choose `Credentials`.
   - In the **top** bar, click `Create credentials` -> `API key`.
   - Give the key a name, for example: `YouTube API v3 key`.
   - Under `Application restrictions`, select `None`.
   - Under `API restrictions`, choose `Restrict key`, then select `YouTube Data API v3`.
   - Save your changes and copy the generated API key.


2. Refer to the [tool-setup-guide](#how-to-set-up-and-use-a-tool) to add and activate the `Play YouTube Video` tool.

3. Use your `YouTube Data API v3` key:
    - In the **left sidebar** of Open WebUI, navigate to `Workspace`.
    - In the **top bar**, open `Tools`.
    - Find `Play YouTube Video` and click ⚙️.
    - Change the `YouTube API Key` field from `Default` to `Custom`.
    - Paste your API key into the input box and save.

4. Start chatting! Choose a tool calling model such as `qwen3-it:4b` and test the tool by asking the model to search for content. You can try prompts like: 
    - "What tools do you have?" 
    - "Search for YouTube videos about FastFlowLM."
    - "Play the first video."

#### 🔍 Tool: Google PSE Search

#### 🤔 Other tools