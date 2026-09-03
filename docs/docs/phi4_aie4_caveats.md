---
layout: docs
title: Phi-4 AIE4 Caveats
nav_order: 6
---

# Phi-4 on AIE4 — what to watch out for

Every item below is known, recorded, and either measured or explicitly marked
as unverified. Nothing here is speculation dressed as fact; where something is
a hypothesis it says so.

For what was built, read
[Phi-4 AIE4 Implementation](../phi4_aie4_implementation/).

---

## 1. The MSI is not self-contained. It will not start on a clean machine.

**Symptom.** `flm.exe` exits immediately with `0xC0000135`
(`STATUS_DLL_NOT_FOUND`). No message, no log.

**Verified by execution**, not inferred: on a freshly installed machine with a
conda prefix removed from `PATH`, the shipped binary cannot start.

**Cause.** The installer stages `src/lib/*.dll`, which carries FFmpeg
`avformat-61 / avcodec-61 / avutil-59 / swscale-8 / swresample-5`. This
`flm.exe` links `avformat-63 / avcodec-63 / avutil-61 / swscale-10 /
swresample-7`. Five direct imports are missing, and completing them from a
conda prefix pulls in 54 further transitive DLLs. The version families line up
one for one, so this is **version skew between the vendored `src/lib` set and
a conda-linked build** — an environment artifact.

A sixth missing import, `boost_program_options.dll`, had a different cause and
**is fixed**: this work had moved the Windows Boost linkage from static to
shared without staging a DLL for it. The static default is restored.

**Workaround.** Put a matching FFmpeg on `PATH` before running — a conda
prefix's `Library\bin` works.

**Not fixed here**, because it is a packaging-list problem rather than an AIE4
one, and the scope for this work was narrow. It is real and it is on the path
of anyone who installs the MSI on a clean box.

---

## 2. `Aie4Feature` is not installed by default

`Aie4Feature` carries `Level="1000"`. A plain `msiexec /i` — or a double-click
— produces a FastFlowLM **without the AIE4 runtime**, and the failure surfaces
later as a confusing model-load error rather than at install time.

```
msiexec /i flm-setup-with-aie4.msi ADDLOCAL=MainFeature,Aie4Feature
```

Confirm with `/status` in the REPL: it must report `Engine: corelib_aie4`.

---

## 3. Multi-turn conversation in `flm run` does not retain history

Each turn is answered as though it were the first. `/history` prints only the
current exchange.

**This is not an AIE4 defect and not specific to Phi-4.** `Phi4::insert`
(`src/common/AutoModel/modeling_phi4.cpp`, line 576 at the time of writing)
wraps `input.prompt` in a fresh single-message array whenever `input.messages`
is empty — which is every CLI turn. That `messages.push_back` sits at line 589,
*above* the `#if defined(FLM_ENABLE_CORELIB_AIE4)` branch at line 596: the
history is gone before any AIE4 code runs. The identical construction is in the
Llama 3, Qwen 2, Qwen 3, Gemma 3, LFM2, GPT-OSS and Nanbeige frontends.

Line numbers in this file have already moved once during this work. If they no
longer match, search for the `messages.push_back` inside `Phi4::insert` and
compare its position to the `#if` — the *ordering* is the claim, not the
numbers.

**Workaround.** Use `/api/chat` or `/v1/chat/completions` and supply the whole
conversation yourself; those endpoints take a full `messages` array and count
it in full.

---

## 4. The same binary, on the same input, has produced different text

This backend is **not deterministic run to run**, and the difference is not a
rounding difference in the last bits — it is a different answer.

In the committed determinism campaign, **3 of 150 recorded run-pair
comparisons** ended with the two runs emitting different token sequences. One
pair parts at the 8th emitted token, the other two at the 14th.

Two severity measures are recorded and **they do not rank the three records the
same way**, so each figure belongs to its own record rather than to a single
"worst":

| Record | Max absolute logit difference | Most logits moving at one step | Gate breaches |
|---|---:|---:|---:|
| `…-010.json` | 48.34 | 196,154 of 200,064 | 17 |
| `…-022.json` | **49.25** | 196,748 | 8 |
| `…-026.json` | 49.21 | **196,835** | 10 |

All three are on the **append** continuation route. All three **breach** the
2-BF16-ULP bound the suite gates on — the bound is not merely approached.

**Two things follow, and only the second is obvious.** The per-value bound did
not hold. And even inside that bound the effect would not be bounded, because
greedy decoding turns one flipped argmax into an unrelated continuation.
Read "bounded" in the acceptance record as a statement about a number, not
about what a user sees.

**Current understanding.** The LM head is exonerated — against an FP64 host
reference, 200,059 of 200,064 logits fall within half a BF16 ULP. The
divergence is localised to the 32-layer model body; the specific layer is
unknown. The search is for an operator whose accumulation or work-partitioning
**order** is unpinned, not for the least accurate one: deterministic rounding
of deterministic inputs is reproducible, so varying results additionally
require varying order.

Carried as a known defect (`DETERM-5`), not a release blocker.

---

## 5. Long, open-ended generations can collapse

Set an explicit generation limit. In the REPL:

```
/set gen-lim 200
```

**What was observed.** In the multi-turn acceptance step, two of four turns ran
to the 4095-token cap — 186.9 s and 179.9 s — because no stop token was ever
emitted. Both began coherently. Both then entered a phrase-repetition loop,
which is ordinary language-model degeneration.

**One of them did something else.** At roughly decode step 3,470 — about 85 %
of the way through, breaking mid-word immediately after "foot" — the output
switched from English to high-entropy random ASCII and stayed there:

```
...want you to scratch Fluffy on Fluffy's foot C(.;%>%A9G<:,&>-@G<*9H.",3;2=A,EC2#6F3F3...
```

A language model losing coherence repeats, drifts or confabulates. It does not
emit uniform random punctuation and digits. That signature suggests the logits
themselves became garbage.

**Status: one observation.** Whether it reproduces, whether the onset is stable
near step 3,470, and whether it is prompt-dependent are all **unknown**.

**Hypothesis, attributed to operator experience and not verified here**: this
signature has previously turned out to be the attention kernel. Entry point for
anyone investigating:

```
ryzenai_corelib_flat_mha_bf16
  bound   src/common/corelib/corelib_api.cpp        :197
  called  src/common/corelib/phi4_corelib_aie4.cpp  :966   api->functions().flat_mha(
  phase   src/common/corelib/phi4_corelib_aie4.cpp  :1178  active_phase = "flat_mha"
```

Search by symbol rather than by line if these have drifted.

Not investigated here by decision, not by oversight.

---

## 6. The limit is 4095, not 4096

`kMaxDecodeWindow` is `kMaxSequenceLength - 1`. A 4096-row prefill at position
0 dispatches, but a single-row decode step at position 4095 — whose window is
also 4096 — is refused: no token-attention kernel ships for a 4096-token
window. The prefill and decode paths do not share a bound.

The ~4k ceiling itself is **by design** — the operators support at most 4k
input. Raising `/set ctx-len` beyond it prints an error rather than crashing.

---

## 7. Building against a shared or conda prefix needs a flag

`FLM_WIN_STATIC_DEPS` defaults to **`ON`**, which is the static layout CI and
the MSI ship. Building against a shared or conda prefix requires:

```
-DFLM_WIN_STATIC_DEPS=OFF
```

A mismatch is a **configure-time hard failure naming both halves**, not a link
error at the end.

**One open question, settled by a single command on the CI runner.** A probe
established that `target_link_directories` contributes nothing to
`find_library`: a stage directory known only that way yields
`PROBE-NOTFOUND`; on `LIB` or `CMAKE_PREFIX_PATH` it resolves. Nothing in the
repository arranges for the Boost stage directory to be findable, and the
workflow sets no environment. So either the runner has an ambient `LIB` that
covers it — meaning this work introduced an **undeclared dependency on runner
environment** that the pre-branch raw-name link never needed — or the shipping
preset fails at configure. Run `echo $env:LIB` on the runner to find out.
Explicit `PATHS` have been added, which is correct either way.

---

## 8. Nothing automated compiles this backend or runs any of its tests

No CI job runs any test, and the release preset never compiles the AIE4 path.
After merge, none of these tests run unless a human runs them.

**This is worse than an absence.** `debian-portable.yml:122` defines a job
named `test-summary` whose single step echoes four hard-coded ✅ lines into the
step summary, unconditionally, `if: always()`. The absence of testing is
currently reported as four green ticks.

Out of scope for this work by decision; recorded so it is not mistaken for
coverage.

---

## 9. Two things are recorded as unverified, on purpose

- **The acceptance harness's REPL-hang fix** is verified by parse, by 72
  surviving offline assertions, and by being the same bounded console call the
  other steps use — **not by execution**. Confirming it needs AIE4 hardware, a
  model and a console, and re-running the acceptance harness was ruled out.
- **The legacy NPU2 path** was compiled and linked, not run. No such hardware
  was available.

---

## 10. The determinism baseline is frozen

`DETERM-3` forbids pooling measurements across harness binaries, and the
committed records now span three. The published baseline is therefore frozen at
its original run, later measurements are committed separately, and the
stability sentences in that block are **hand-corrected rather than
re-rendered** — labelled as such, so a reader can tell the two apart. If the
pooling question is resolved, that block should be re-rendered and the label
deleted.

---

## A note on how to read the records

The acceptance record has a companion
[provenance page](../benchmarks/phi4_aie4_acceptance_provenance/) that
documents where the record itself is wrong: two failed turns described
inaccurately, a source revision claiming a tree that was not pristine, and the
word "bounded" doing more work than the data supports.

That page exists because the recurring failure in this work was **a record that
reads better than the run it describes** — a check reported as passing for work
it did not do, a retraction reaching the code but not the rendered artifact, a
test agreeing with the implementation about which case exists. Roughly twenty
instances were found and fixed across the effort, several of them by reviews
catching earlier reviews.

Trust the committed JSON and the cited line numbers over any prose summary,
including this page.
