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
against its published metadata. Four further files are shipped inside
FastFlowLM rather than fetched, and they are not all the same kind of file:

- **Authored, with no upstream counterpart.** `config.json`,
  `corelib_phi4_manifest.json` and `provenance.json` do not exist in the
  upstream repository at all. The upstream package ships no `config.json`, and
  the other two describe FastFlowLM's own packaging.
- **Shadowing an upstream file.** `tokenizer_config.json` *does* exist
  upstream, and the shipped copy replaces it, because the published one carries
  neither a chat template nor the EOS token IDs this backend requires.

The distinction is not cosmetic: an upstream metadata record is expected for
the shadowing file and is a provenance error for the authored ones, since it
would mean FastFlowLM's own package contract had been published to the model
repository.

These files restate the model contract for FastFlowLM's existing readers; they
never override it, and a model whose weights disagree with them fails to load.
That last sentence is checked rather than asserted: editing a single value in
the shipped `config.json` makes `flm run phi4-mini-it-aie4:4b` refuse to load,
naming the field, the value it found and the value the ONNX initializers
require.

### What has been verified on hardware

This backend has been run end to end against the published model on an AMD
Ryzen AI NPU. The signed-off record — machine and driver identity, the
FastFlowLM commit, corelib DLL hashes, every measured timing, the verbatim
prompts and completions, and a line-by-line result against the design's
acceptance criteria — is at
[Phi-4 AIE4 acceptance](../benchmarks/phi4_aie4_acceptance.html).

Read that document before relying on any claim here. It states plainly which
criteria were met, which were not, and which were **not tested at all**, and
the last category is large.

### Known limitations

- **Multi-turn conversation in `flm run` does not retain history.** Each turn
  is answered as though it were the first: after telling the model a fact and
  asking about it on the next turn, the model does not have it. `/history`
  confirms it — the token count tracks the current turn instead of growing, so
  the previous turn was replaced rather than appended. This is **not specific
  to this model or to the AIE4 backend**: the CLI builds a single-message
  prompt from the current input, and every text frontend does the same. Supply
  the whole conversation yourself — the `/api/chat` and `/v1/chat/completions`
  endpoints take a full `messages` array and count it in full — until this is
  fixed.
- **Generation can run to the context cap, and output degrades when it does.**
  A request with no generation limit is bounded only by the model emitting an
  end token. When it does not, generation continues to the cap — and the text
  falls apart before it gets there. In the acceptance run, two turns ran to the
  4095-token cap taking **186.9 s** and **179.9 s**; one repeated a single
  phrase for the remainder, and the other degenerated into meaningless
  characters. **Set an explicit generation limit.** It is the difference
  between a bounded reply and three minutes of noise.
- **The limit is 4095, not 4096**, for the prompt and the generated tokens
  together, counted over the complete rendered conversation. The KV window is
  4096 rows; the usable cap is one less because no token-attention kernel
  ships for a 4096-token window. Requests over the cap are refused before any
  work is submitted, with HTTP 400 on the server and a message naming the cap
  and the rendered prompt length.
