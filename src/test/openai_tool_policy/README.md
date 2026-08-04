# OpenAI Tool-Calling Contract Tests

This directory contains two layers of contract coverage:

- `test.cpp` verifies the policy and validation logic through CTest.
- `test_server_contract.py` exercises the OpenAI-compatible HTTP and SSE
  boundaries against a live FastFlowLM server.

The Python harness is intentionally manual. It requires an NPU, a downloaded
model, and a running FastFlowLM server, so it is not part of ordinary CI.

## Run the live contract test

Start one model at a time:

```bash
flm serve gemma4-it:e4b --ctx-len 65536 --port 52625
```

In another terminal, run the baseline contract and optional multi-turn
workflow:

```bash
python3 src/test/openai_tool_policy/test_server_contract.py \
  gemma4-it:e4b --workflow
```

The model argument must match the model loaded by the server. To test another
model, stop the server, start that model, and run the harness again. Keep this
serial because concurrent model loading can exhaust NPU memory.

Use a different endpoint or timeout when needed:

```bash
python3 src/test/openai_tool_policy/test_server_contract.py \
  qwen3-it:4b \
  --base-url http://127.0.0.1:1234/v1 \
  --timeout 600
```

## Coverage

The live harness checks request validation; `none`, `required`, named, and
allowed-tool selection; fail-closed behavior and recovery; multi-turn tool
result chaining; and streamed usage accounting. `--workflow` adds a synthetic
read/edit/write skill flow to probe whether the loaded model can sustain a
tool-use conversation.

The harness does not execute real tools or modify files. It supplies synthetic
tool results in memory and exits nonzero when any check fails. Very small token
limits are used deliberately in one negative-path check, so a logged
`model_error` can be expected while that check is running.
