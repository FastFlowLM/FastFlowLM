---
layout: docs
title: Granite
nav_order: 14
parent: Models
---

## 🧩 Model Card: [ibm-granite/granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b)

- **Type:** Text-to-Text
- **Think:** Yes
- **Tool Calling Support:** Yes
- **Base Model:** [ibm-granite/granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens
- **Default Context Length:** 8k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:

```shell
flm run granite:3b
```

Granite 4.2 is a reasoning model: the chat template opens a `<think>` block in
the generation prompt, so the model emits a reasoning trace, closes it with
`</think>`, and then answers.

---
