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
It requires FastFlowLM 1.0.4 or newer.

**Size prompts against 4,094 tokens, not 4,096.** The catalog entry lists a
4096-token default and maximum prefill length, and the backend is stricter than
its catalog entry: the prompt and the generated tokens together are capped at
**4,095**, and a prompt is refused once it reaches that figure on its own, so
**the largest prompt this tag will accept is 4,094 tokens** — it needs at least
one token of room to answer in. Sizing a prompt to the advertised 4,096 gets
HTTP 400 before any work is submitted. The rule is
`Phi4::validate_aie4_capacity` (`src/common/AutoModel/modeling_phi4.cpp:420`),
which refuses when the rendered prompt length reaches the cap; see
[Known limitations](#known-limitations) for why the cap is what it is.

**Windows and AIE4 only.** The backend is compiled in only when FastFlowLM is
built with `FLM_ENABLE_CORELIB_AIE4=ON`, the packaging step is Windows-only,
and the tag runs on an AIE4 NPU. There is no Linux build of it and it does not
run on XDNA2 hardware.

**Building it against a shared dependency prefix also needs
`-DFLM_WIN_STATIC_DEPS=OFF`.** That option defaults to `ON`, which is the static
Boost and curl layout CI and the MSI ship; a conda or other shared prefix — the
usual way the AIE4 work is built — has to opt out, or the Boost linkage macros
and the library name the build picks describe two different Boosts. Getting it
wrong is a configure-time hard failure naming both halves, not a mystery link
error at the end.

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
Ryzen AI NPU: a 20-step acceptance pass whose tally was **16 steps met, 1 not
met, 3 not exercised**, and **12 of 52 criteria met, 12 partial, 2 not met, 26
not exercised**.

Read that last number before relying on any claim here — half the criteria were
**not tested at all**. What was and was not established is summarised in
[Phi-4 AIE4 Implementation](/docs/phi4_aie4_implementation/); every known defect
and limit is in [Phi-4 AIE4 Caveats](/docs/phi4_aie4_caveats/). The step-by-step
record itself was an artifact of that one campaign and is not carried in the
repository — see caveat 11.

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
| prefill, 4096 rows † | 1,746.3 ms / 2,345.5 tok/s | 1,748.6 ms / 2,342.5 tok/s |
| decode from 128 tokens | 20.04 tok/s | 24.89 tok/s |
| decode from 512 tokens | 21.16 tok/s | 24.38 tok/s |
| decode from 2048 tokens | 20.68 tok/s | 22.68 tok/s |
| peak host private bytes | 7.67 GiB | 7.68 GiB |
| peak host working set | 9.52 GiB | 9.60 GiB |
| V-cache reads per model step | 32 | 32 |
| per-head V writes per model step | 256 | 256 |
| V-cache read calls, whole run | 231,264 | 231,264 |
| per-head V write calls, whole run | 1,850,112 | 1,850,112 |
| model steps those calls span | 7,227 | 7,227 |
| V bytes transferred | 44.60 GiB | 44.60 GiB |
| V scatter wall time | 3,315.9 ms | 3,081.7 ms |

† The 4096-row prefill is measured by driving the engine directly. It is **not**
a prompt size you can submit — see the cap above: the largest admissible prompt
is 4,094 tokens.

The three V-scatter call and step figures are counted by the engine, not inferred
from a rate: both baselines record `v_scatter.counts_are_measured: true`, and the
per-step figures in the two rows above them are those counts divided by
`model_steps`. They are identical in both runs because the workload is; only the
wall time moved.

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
[the Phi-4 benchmarks](/docs/benchmarks/phi4_results/).

**A terminal device failure restarts the process, by design.** If a corelib
call fails while work may still be outstanding on the device — or if a
synchronize itself fails — the state of that work is unknowable, so FastFlowLM
does not try to continue: it writes a diagnostic record and terminates the
process with exit code `0xE0040001`. Under `flm serve` that ends the server and
you restart it; under `flm run` the session ends.

A failure at a point where nothing is outstanding is recoverable: the session
is cleared, an error is returned, and the process keeps running. That covers
both a failure before anything was submitted *and* a failure after a
synchronize has completed and drained the device, so a host-side error between
two dispatch groups no longer costs you the server — only the conversation.

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
  (`common/AutoModel/modeling_phi4.cpp:576`) wraps `input.prompt` in a fresh
  single-message array whenever `input.messages` is empty, which is every CLI
  turn, and that happens at line 589 — *above* the
  `#if defined(FLM_ENABLE_CORELIB_AIE4)` branch that starts at line 596. The
  history is already gone before any AIE4 code runs. The identical construction
  is in the Llama 3, Qwen 2, Qwen 3, Gemma 3, LFM2, GPT-OSS and Nanbeige
  frontends: it is the shared CLI path for every text model. Supply the whole
  conversation yourself — the `/api/chat` and `/v1/chat/completions` endpoints
  take a full `messages` array and count it in full — until this is fixed.
- **Two runs of the same binary, on the same input, have produced different
  text.** This backend is **not deterministic run to run**, and the difference
  is not a rounding difference in the last bits — it is a different answer. In
  the committed determinism campaign, **3 of the 150 recorded run-pair
  comparisons** ended with the two runs emitting different token sequences. One
  pair parts at the **8th** emitted token, the other two at the 14th. Two
  measures of severity are recorded and **they do not rank the three records
  the same way**, so each figure below is given against the record it came
  from rather than rolled into one "worst": the largest absolute logit
  difference is **49.25**, while the largest number of logits moving at a
  single step is **196,835 of 200,064** — and those are different records.
  Each comparison was a JSON record written by the campaign. The 150 records
  are not carried in this branch — they are the output of one campaign on one
  machine, and nothing in the tree reads them — so the three that diverged are
  summarised here rather than cited by path:

  - `determ1-force_append-010` — max abs diff **48.34**; up to **196,154** of
    200,064 logits differing at one step (`decode[13]`) by more than the
    2-BF16-ULP bound the suite gated on; sequences part at token 8
  - `determ1-force_append-022` — **49.25**; up to **196,748** at `decode[15]`;
    part at token 14
  - `determ1-force_append-026` — **49.21**; up to **196,835** at `decode[15]`;
    part at token 14

  All three are on the **append** continuation route. Two re-prefill records
  also carry a logit difference, but in both the emitted text still matched.
  The pooled rates, the two runs that breached a `DETERM-2` hard gate, and the
  measurement showing that the divergence enters the **model body** rather than
  the LM-head dispatch are published in
  [the Phi-4 benchmarks](/docs/benchmarks/phi4_results/), which reports them
  honestly and refuses to call that window a settled baseline. Nothing here is
  a fix: the cause is not known, and no run has been made to find it.

  **What this means for you:** do not assume the same prompt gives the same
  answer, do not use output equality as a test oracle against this backend, and
  do not cache or diff on the assumption that a repeat is a repeat.
- **Long, open-ended generations can collapse. Set an explicit generation
  limit — and know which interface gives you one.** There is **no command-line
  flag**: nothing in `src/include/utils/vm_args.hpp` takes a generation limit,
  so `flm run phi4-mini-it-aie4:4b` cannot be launched with one. Two mechanisms
  exist, and they cover both ways of using this tag:
  - **In the `flm run` session, type `/set gen-lim <value>`** before the prompt
    you want bounded (`src/runner/runner.cpp:643`). It sets the per-round token
    limit for the rest of the session, and the AIE4 admission check reads it
    (`CliRequestedMaxNewTokens`, `src/server/generation_limit.cpp:176-182`).
    It is a REPL command, not a startup option, so it has to be typed each
    session; `/set` on its own prints the list it appears in
    (`src/runner/runner.cpp:572-586`).
  - **Over HTTP, send `max_tokens` (OpenAI and `/api/generate`) or
    `options.num_predict` (`/api/chat`)** — `src/server/generation_limit.cpp:65,73`.

  A request with no limit is bounded only by the model emitting an end
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

  The acceptance record described both turns with one reason, *"the reply is
  dominated by a repeated token"*, which was right about the repetition and
  silent about the collapse. The paragraph above is the correction, derived from
  that record's own data.
- **The limit is 4095 tokens** for the prompt and the generated tokens
  together, counted over the complete rendered conversation, **so the largest
  prompt you can submit is 4,094.** This is **by design, not a shortfall**: the
  AIE4 operators support at most about 4k input. The KV window is 4096 rows and
  the usable cap is one less, because no token-attention kernel ships for a
  4096-token window; and a prompt that fills the cap exactly leaves no room to
  answer, so `Phi4::validate_aie4_capacity`
  (`src/common/AutoModel/modeling_phi4.cpp:420`) refuses at 4,095 rendered
  tokens rather than at 4,096. Requests over the cap are refused before any work
  is submitted, with HTTP 400 on the server and a message naming the cap and the
  rendered prompt length. The acceptance run verifies the enforcement exactly:
  the remaining capacity is admitted and one token more is refused.

  **The catalog says 4096, and it is describing a different thing.**
  `src/model_list.json` gives `phi4-mini-it-aie4:4b` a `default_context_length`
  and `max_prefill_len` of 4096, which is the KV window the engine allocates.
  The admission gate is one token tighter than the window and two tighter than
  a full-window prompt; the figure to size against is **4,094**.
