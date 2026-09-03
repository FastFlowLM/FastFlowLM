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

**Windows and AIE4 only.** The backend is compiled in only when FastFlowLM is
built with `FLM_ENABLE_CORELIB_AIE4=ON`, the packaging step is Windows-only,
and the tag runs on an AIE4 NPU. There is no Linux build of it and it does not
run on XDNA2 hardware.

**The existing NPU2 tag is unchanged.** `phi4-mini-it:4b` still resolves to the
legacy `phi4_npu` engine, still uses the `Q4_1` `model.q4nx` package with a
32k default context, and is unaffected by any of this. The two tags are
separate catalog entries with separate model packages; installing one does not
change the other.

Select **Phi-4 AIE4 corelib runtime** in the Windows installer, then use:

```shell
flm pull phi4-mini-it-aie4:4b
flm run phi4-mini-it-aie4:4b
```

This package is hosted only on Hugging Face. `--modelscope` is rejected
before any download starts.

### The runtime this tag needs, and what happens without it

The AIE4 feature is an optional installer component. It installs
`ryzenai_corelib.dll` and its derived dependency closure into an `aie4`
directory beside `flm.exe`, together with `aie4-closure.txt` — a manifest
naming the corelib version, the corelib DLL's SHA-256, and the name and
SHA-256 of every DLL staged with it. If a result ever looks wrong, that file
is what identifies the runtime that produced it.

`flm.exe` never links `ryzenai_corelib.lib`. The DLL is resolved at run time by
absolute path, so a FastFlowLM without the AIE4 component installed starts and
runs normally; only this one tag is unavailable.

**There is no fallback.** If the backend is not there, the tag fails — it does
not quietly run on something else, and it does not silently produce answers
from a different engine:

- A build without `FLM_ENABLE_CORELIB_AIE4` refuses the tag outright with
  `This binary was built without Phi-4 AIE4 corelib support`, and
  `flm validate --json` reports `corelib_aie4.available_in_build: false`.
- A build with the feature but without the runtime directory installed fails
  to load the library and says so, naming the path it tried and the Win32
  error.

`flm validate --json` is the way to check before running: it reports the
loader, the dependency closure, the device context, whether the fatal-record
directory is writable, and clean shutdown, each separately.

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

### What it costs to run

Two measured runs on the same machine (`xcomedusad-43`), a day apart, are shown
side by side rather than one being picked. This machine has been measured
moving by up to 1.8x on decode throughput with no code change, so a single
column would read as more precise than the measurement is. Treat a difference
below roughly 2x against any figure here as unresolved.

| | 2026-09-02 | 2026-09-03 |
| --- | ---: | ---: |
| model load, total | 13,043.9 ms | 12,796.3 ms |
| — of which the 1..4096 helper interrogation | 11,282.6 ms (86%) | 11,061.6 ms (86%) |
| — of which weight pack and upload (161 objects) | 1,603.4 ms | 1,596.0 ms |
| cold TTFT (19-token prompt) | 3,169.4 ms | 3,061.7 ms |
| warm TTFT (same stream, context cleared) | 49.1 ms | 53.5 ms |
| prefill, 4096 rows | 1,746.3 ms / 2,345.5 tok/s | 1,748.6 ms / 2,342.5 tok/s |
| decode from 128 tokens | 20.04 tok/s | 24.89 tok/s |
| decode from 512 tokens | 21.16 tok/s | 24.38 tok/s |
| decode from 2048 tokens | 20.68 tok/s | 22.68 tok/s |
| peak host private bytes | 7.67 GiB | 7.68 GiB |
| peak host working set | 9.52 GiB | 9.60 GiB |
| V-cache reads per model step | 32 | 32 |
| per-head V writes per model step | 256 | 256 |
| V bytes transferred | 44.60 GiB | 44.60 GiB |
| V scatter wall time | 3,315.9 ms | 3,081.7 ms |

**Model load is dominated by one thing.** 86% of it is interrogating the
corelib helpers for every row extent from 1 to 4096. That is a fixed startup
cost, paid once per process, and it is why a cold `flm run` takes about 13
seconds before the first token.

**Accounted device-side memory**, identical in both runs: FP16 embedding
1.14 GiB, KV cache 512 MiB, corelib packed weights 1.95 GiB, and the ONNX
source mapped at 3.03 GiB. Those account for the bulk of the host peak above.

**Memory is stable once warm.** Over a 128-token decode window after warmup,
both runs created **0** new device tensors, **0** new weight objects, and
recorded a net change of **0** live corelib objects and **0 bytes** of
private-byte growth. The least-squares slope over that window was negative in
every window measured — memory did not grow.

**Continuation: append or re-prefill.** When you continue a conversation, the
backend either appends the new tokens to the existing KV cache or clears it and
re-prefills the whole rendered history. The choice is a fixed release constant,
not a runtime measurement: **append when the new suffix is at most 4 tokens,
otherwise re-prefill.** It does not drift with load or thermal state.

The cost difference is large and runs the opposite way at each end. From the
2026-09-03 run, at a 2048-token history: appending 1 token took **54.2 ms**
against **1,150.1 ms** to re-prefill — 21x in append's favour — while appending
256 tokens took **10,422.5 ms** against the same **1,151.2 ms**. Append cost
grows with the suffix; re-prefill cost does not.

The constant is deliberately set at the low end. It sits inside the
append-wins region on every run measured, whereas a higher value has not been
supported by every run — on one run the rule would have chosen 8, and on two
earlier ones it would not. The full per-suffix tables, and that disagreement,
are in
[the Phi-4 benchmarks]({{ site.baseurl }}/docs/benchmarks/phi4_results.html).

**A terminal device failure restarts the process, by design.** If a corelib
call fails after work has already been submitted to the device, the state of
that work is unknowable, so FastFlowLM does not try to continue: it writes a
diagnostic record and terminates the process with exit code `0xE0040001`. Under
`flm serve` that ends the server and you restart it; under `flm run` the
session ends. A failure *before* anything was submitted is recoverable — the
session is cleared, an error is returned, and the process keeps running.

**Where the record goes.** `%LOCALAPPDATA%\FastFlowLM\logs`, as
`corelib-fatal-<timestamp>-<pid>.json`, carrying the corelib status code, the
call that failed, the library's own message, the phase, layer, row count and
position. The next `flm` process on that machine prints any records it finds
and removes them, so the diagnostic reaches you on the next run rather than
only on disk. A record belonging to a process that is still alive is left
alone.

### Known limitations

- **Multi-turn conversation in `flm run` does not retain history.** Each turn
  is answered as though it were the first: after telling the model a fact and
  asking about it on the next turn, the model does not have it. `/history`
  confirms it — the conversation it prints contains only the current exchange,
  with the previous turn's message and reply absent, so each turn replaces the
  history rather than appending to it. This is **not specific to this model or
  to the AIE4 backend, and it is not an AIE4 defect.** `Phi4::insert`
  (`common/AutoModel/modeling_phi4.cpp:559`) wraps `input.prompt` in a fresh
  single-message array whenever `input.messages` is empty, which is every CLI
  turn, and that happens at lines 569–573 — *above* the
  `#if defined(FLM_ENABLE_CORELIB_AIE4)` branch that starts at line 579. The
  history is already gone before any AIE4 code runs. The identical construction
  is in the Llama 3, Qwen 2, Qwen 3, Gemma 3, LFM2, GPT-OSS and Nanbeige
  frontends: it is the shared CLI path for every text model. Supply the whole
  conversation yourself — the `/api/chat` and `/v1/chat/completions` endpoints
  take a full `messages` array and count it in full — until this is fixed.
- **Long, open-ended generations can collapse. Set an explicit generation
  limit.** A request with no limit is bounded only by the model emitting an end
  token. When it does not, generation runs to the cap. In the acceptance run,
  two open-ended turns did exactly that, taking **186.9 s** and **179.9 s**;
  neither emitted a stop token and both ended on *"Max length reached, stopping
  generation"*. Their prompts were tiny — 13 and 9 tokens — so the 4,095 tokens
  were output alone. A limit is the difference between a bounded reply and
  three minutes of unusable text.

  **Normal-length output is sound, and that is the larger part of the
  evidence.** Five separate single-turn prompts in the same acceptance run —
  arithmetic, a list, a Shakespeare quotation, a definition, a factual
  question — all returned correct, well-formed answers in about ten seconds
  each. The collapse has been seen in long generations only.

  Two distinct things happen, and only the first is ordinary:

  - **Both** long turns entered a **phrase-repetition loop**, early. One locked
    on about 3% of the way in and repeated a single clause 375 times; the other
    looped on a similar phrase. This is ordinary greedy-decode degeneration and
    is what a length limit is for.
  - **One** of them additionally, and abruptly, stopped producing language at
    all: about 85% of the way through that reply — roughly decode step 3,470 —
    it broke **mid-word** and emitted high-entropy punctuation and digits
    (`,7&)6E-G.()$G%D:*DC#C1<$25DFC70D1)9%=!C4$+…`) to the cap. The other turn
    never did this; its output stayed alphabetic throughout.

  A language model losing the thread repeats, drifts or confabulates. It does
  not emit uniform random punctuation. So the second event is a different kind
  of failure from the first, and it is recorded as a **known open issue**, not
  investigated on this branch.

  **A hypothesis, from operator experience and not verified here:** this
  signature has come up several times before on this class of hardware and has
  essentially always turned out to be the attention kernel. The entry point a
  future investigation should start from is `ryzenai_corelib_flat_mha_bf16`
  (bound at `common/corelib/corelib_api.cpp:195-197`, called at
  `common/corelib/phi4_corelib_aie4.cpp:941`, with `active_phase = "flat_mha"`
  set at `:1107`). Nothing in this project's testing confirms that, and it
  should not be read as a finding.

  **This is a single observation.** Whether it reproduces at all, whether the
  onset is stable near that step, and whether it depends on the prompt are all
  unknown, and no run has been made to find out.

  The acceptance record itself describes both turns with one reason, *"the
  reply is dominated by a repeated token"*, which is right about the repetition
  and silent about the collapse. The record is not edited after the fact; the
  correction, with the derivation from the record's own data, is in
  [Phi-4 AIE4 acceptance provenance](../benchmarks/phi4_aie4_acceptance_provenance.html).
- **The limit is 4095 tokens** for the prompt and the generated tokens
  together, counted over the complete rendered conversation. This is **by
  design, not a shortfall**: the AIE4 operators support at most about 4k input.
  The KV window is 4096 rows and the usable cap is one less, because no
  token-attention kernel ships for a 4096-token window. Requests over the cap
  are refused before any work is submitted, with HTTP 400 on the server and a
  message naming the cap and the rendered prompt length. The acceptance run
  verifies the enforcement exactly: the remaining capacity is admitted and one
  token more is refused.
