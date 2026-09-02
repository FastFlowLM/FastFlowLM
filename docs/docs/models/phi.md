---
layout: docs
title: Phi
nav_order: 10
parent: Models
---

## 🧩 Model Card: [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No
- **Base Model:** [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens 
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run phi4-mini-it:4b
```

---

## Phi-4 mini on Ryzen AI AIE4

The `phi4-mini-it-aie4:4b` tag uses the optional `corelib_aie4`
backend and the pinned AMD
[OGA DML package](https://huggingface.co/amd/phi-4-mini-instruct-oga-dml).
It has a 4096-token default and maximum prefill length and requires
FastFlowLM 1.0.4 or newer.

Select **Phi-4 AIE4 corelib runtime** in the Windows installer, then use:

```shell
flm pull phi4-mini-it-aie4:4b
flm run phi4-mini-it-aie4:4b
```

This package is hosted only on Hugging Face. `--modelscope` is rejected
before any download starts.

The assembled model directory has two provenances. The weights, tokenizer and
vocabulary are downloaded from the pinned upstream revision and hash-checked
against its published metadata. Four files -- `config.json`,
`tokenizer_config.json`, `corelib_phi4_manifest.json` and `provenance.json` --
are authored by FastFlowLM and installed with the product, because the upstream
repository ships no `config.json` and its tokenizer configuration carries
neither a chat template nor the EOS token IDs this backend needs. Those files
restate the model contract for FastFlowLM's existing readers; they never
override it, and a model whose weights disagree with them fails to load.
