---
layout: docs
title: Phi-4 AIE4 acceptance provenance
nav_order: 21
parent: Benchmarks
---

# Provenance of the Phi-4 AIE4 acceptance record

Hand-written, unlike `phi4_aie4_acceptance.md` and `.json`, which are generated
by `src/test/phi4_corelib_aie4/run_real_model_acceptance.ps1` and overwritten on
every run. This file exists to answer one question a careful reader will
otherwise be unable to answer from the repository alone.

## If you recompute `harness_sha256`, it will not match. Here is why.

`phi4_aie4_acceptance.json` records the SHA-256 of the two scripts that
produced it:

| Field | Value in the record |
| --- | --- |
| `harness_sha256` | `631a271b…` — `run_real_model_acceptance.ps1` |
| `driver_sha256` | `cdf901b3…` — `drive_flm_console.ps1` |
| `checkout_revision` | `81a01f7ba83ad1a177601a5f55b08befb5f767ce` |
| `binary_revision` | `f67ed0054c1da14035de51b94b96d793693f7605` |

Both hashes are of the files **as checked out**, not of the git blobs —
`.gitattributes` normalises line endings on the way out of the object store, so
`git show <rev>:<path>` piped to a hasher will not reproduce them. Hash the
working-tree file.

`drive_flm_console.ps1` still matches. **`run_real_model_acceptance.ps1` does
not, and that is deliberate.** Review rounds after that run changed the harness
without authorizing a new run, so the current harness is strictly newer than
the record it sits beside. The changes were:

- Step 8's history-accumulation gate was replaced. It had asserted that
  `/history`'s **token count** grows across turns, which is a proxy that can
  pass on a conversation that is demonstrably not accumulating. It now asserts
  that turn N's history **contains** turn N−1's exchange.
- The document readback gained a content check, and then a fix: its first
  version used `@(Get-Field …).Count -gt 0` to ask whether a step had per-turn
  evidence, and `@($null).Count` is `1` in PowerShell, so it matched every step
  and would have failed an otherwise healthy run.
- Two Section 17 criteria, `C01` and `C31`, gained declared coverage gaps and
  will render **PARTIAL** rather than **MET** on the next run.
- The shared guard logic moved into `acceptance_guards.ps1`, dot-sourced by
  both the harness and `verify_acceptance_guards.ps1`.
- **Review round 4 fixed four more things, three of which would have damaged
  the next run.** In order of consequence:
  - The Step 8 call site read `@(Test-HistoryContainment $perTurn)`. The
    function returns a comma-wrapped array, `@(f …)` preserves the wrapper, and
    the accumulation gate could therefore **never fire** — on the recorded data
    the correct form finds 3 missing turns and the shipped form found 0. Had a
    run happened between round 3 and round 4, this branch's headline finding
    would have vanished from the record while Step 8 still failed for other
    reasons. `verify_acceptance_guards.ps1` now lints the suite's call sites for
    the pattern and has a positive control proving the lint fires.
  - `Get-Field` threw on a property-less object (`{}` reloaded from JSON), which
    is what a step recorded with no evidence becomes — and the harness reloads
    the record from disk before rendering. **Any partial run would have ended in
    `DOCUMENT NOT RENDERED` and exit 1.** The committed `-Steps all` run escaped
    it only because every selected step supplied evidence.
  - `Add-UnselectedStep` threw on a fresh record (`@($null | Where-Object { $_.id … })`
    under strict mode), so a `-Steps <list>` run whose first step was not
    selected died in seconds having recorded nothing. This one is **not** a
    round-3 regression; it is as old as the function.
  - The renderer emitted `### Step 7` twice for any step carrying both `probes`
    and `prompt_verbatim` — which step 7 does.
- **`C20` is now declared the other way round.** Project-owner ruling: the AIE4
  operators support at most about 4k input, so the 4095 cap is expected
  behaviour and the harness asserting against 4095 is correct. The *criterion
  text* saying 4096 is the defect. It stays PARTIAL rather than being upgraded,
  because nothing in a run can make that sentence true — the correction belongs
  in design Section 17.

## Correction: the record describes Step 8's two failed turns wrongly

Task 15, 2026-09-03. The record gives **both** long turns of Step 8 the single
reason *"the reply is dominated by a repeated token"*. That is accurate for one
of them and wrong for the other, and the wrong wording hides the more alarming
of the two signatures. The record is not edited — it is the account of a run
that happened — so the correction lives here, and the classifier that produced
the wrong wording has been fixed for future runs.

**What the record's own data shows**, re-derived from
`steps[id=8].evidence.per_turn[*].reply_verbatim`:

| | turn 1 | turn 3 |
| --- | --- | --- |
| prompt length | 13 tokens | 9 tokens |
| reply, whitespace-normalised | 32,004 chars | 23,802 chars |
| ended on | *Max length reached* — no stop token | *Max length reached* — no stop token |
| phrase-repetition loop | **yes**, early | **yes**, early |
| letters, whole reply | 74.2% | 71.6% |
| collapse into high-entropy ASCII | **no** — alphabetic throughout | **yes**, from character 10,200 of 23,802 |

So the 4,095-token totals on those turns are **output alone**; the inputs were
tiny. Reaching the cap is a consequence of a reply that will not stop, not a
cause of one.

**Repetition is common to both turns** and is ordinary greedy-decode
degeneration. What is not ordinary is the second event, and it happened once:
turn 3 produced language for about 85% of its reply, then broke **mid-word**
and emitted uniform punctuation and digits to the cap. A language model losing
the thread repeats, drifts or confabulates; it does not emit random
punctuation.

Note that `reply_verbatim` holds each reply **twice** — the streamed copy, then
the `[FLM]  Model RAW Output:` echo — so any character offset must be read
within one copy. The figures above are over the normalised whole field, which
is why the offset reads 10,200 of 23,802.

**Why the harness missed it.** `Test-LooksLikeEnglish` had a whole-text check —
*fewer than half the characters are letters* — and turn 3 is 71.6% letters,
because the clean 85% carries the average. The function now also scans in
200-character windows and reports the first window that stops being language
after an earlier one that was, with its offset. Driven over this record it
fires on turn 3 at character 10,200 (22% letters) and is silent on turns 1, 2
and 4; `verify_acceptance_guards.ps1` asserts exactly that, against the shipped
text rather than a fixture.

**Hypothesis, attributed and unverified.** From the project owner's direct
experience of this class of hardware: this signature has come up several times
before and has essentially always turned out to be the attention kernel. A
future investigation should start at `ryzenai_corelib_flat_mha_bf16` — bound at
`src/common/corelib/corelib_api.cpp:195-197`, called at
`src/common/corelib/phi4_corelib_aie4.cpp:941`, with `active_phase = "flat_mha"`
set at `:1107`. **Nothing in this project's testing confirms that**, and it must
not be read as a finding of ours.

**Status: known open issue, accepted, not investigated on this branch.**
Functionality and accuracy are accepted as sound for normal-length output — the
five single-turn probes in the same run were all coherent and correct in about
ten seconds each. This is a **single observation** in a ~4,000-token
generation: whether it reproduces, whether the onset is stable near that point
(roughly decode step 3,470), and whether it depends on the prompt are all
unknown, and no run has been made to find out.

## Correction: the record's `corelib_source_revision` claims a tree that was not pristine

Fix round after Task 15, 2026-09-03. `phi4_aie4_acceptance.json` records
`identity.corelib_source_revision` as the bare SHA
`e5258d29b5cb979d4a538994409b90ceff6e6e7a`, rendered in the record's identity
table. Every other committed artifact describing the **same** corelib checkout
records that SHA with a suffix:

| artifact | value |
| --- | --- |
| `phi4_aie4_acceptance.json` → `identity.corelib_source_revision` | `e5258d29…` |
| `phi4_aie4_baseline.json` | `e5258d29…-untracked-only` |
| `phi4_aie4_baseline_task15_rerun.json` | `e5258d29…-untracked-only` |
| `phi4_results.md`, both identity tables | `e5258d29…-untracked-only` |

`-untracked-only` is **derived** — `run_hardware_suite.ps1` runs `git rev-parse`
and a dirty check and appends what it finds, with the comment that a bare SHA
*"would claim the baseline describes a committed revision when it does not, and
that is a claim nobody can check later."* The acceptance harness, at the time of
this run, took the value as a plain optional string and wrote it into the record
unchecked. So the figure the record publishes is **operator-typed free text**,
and it asserts a pristine tree that this project's own other tooling recorded as
not pristine. **`-untracked-only` is the correct label for that checkout**; the
bare SHA is not.

Two things bound the consequence.

1. **The identifier that actually pins the binary is derived and is not in
   doubt.** The record's own `identity.corelib_dlls` gives
   `ryzenai_corelib.dll` SHA-256 `a523b238…` — byte-identical to the DLL both
   baselines record. Which binary produced these numbers is answerable; which
   source tree produced that binary is answerable only to a commit plus
   "untracked files were present".
2. **It cannot recur.** `run_real_model_acceptance.ps1` now derives the corelib
   revision itself, with the same shared `Get-GitRevision` that
   `run_hardware_suite.ps1` uses (from `acceptance_guards.ps1`); it records
   `corelib_source_revision_derived` beside the value so a reader can tell a
   checked SHA from a typed one; it aborts the run if an operator-supplied
   `-CorelibSourceRevision` contradicts the checkout; and where there is no
   checkout to check against it labels the value
   `-operator-supplied-unverified` rather than passing it off as derived. No
   line numbers are given here on purpose — the file is under active edit, and
   a line number in prose is the kind of citation that rots silently.

As everywhere else on this page, the record is not edited after the fact, which
is why the correction lives here.

## On determinism: the record's word "bounded" is true of a number, not of what a user sees

`phi4_aie4_acceptance.json`'s `determinism_scope.policy` reads *"DETERM-5:
run-to-run divergence is a known, measured, **bounded** defect carried with this
release"*, and the same object's `sharpened_hypothesis` explains the mechanism
with *"Floating-point addition is not associative, so varying order varies the
last bits."* Both sentences are about the difference between two numbers. The
observable defect is not that difference, and on this record's own terms the
word does not survive either half of the check:

- **The per-value bound is what `DETERM-2` specifies, and in the divergent runs
  it did not hold.** The gate is two relative BF16 ULP — the records carry it as
  `determ2_bound_ulps: 2`, `determ2_bound_kind: relative_bf16_ulp`. The three
  records whose runs emitted different text carry **17, 8 and 10** separate
  gate-breach lines, with absolute logit differences up to **49.25**.
  `phi4_results.md` counts two of the three as `DETERM-2` HARD GATE breaches
  rather than all three, because the earliest carries no harness SHA-256 and is
  one of the 64 records excluded from the pooled window — not because it was
  inside the bound.
- **Inside the bound it would still not be bounded in effect.** Decoding here is
  greedy, so a difference large enough to flip a single argmax changes the
  emitted token, and from that token onward the two continuations are unrelated.
  Nothing bounds how far apart the resulting text can then be. Three committed
  records show exactly that: the sequences part at the 8th emitted token in one
  and the 14th in the other two.

**The rendered record is not silent about this and must not be read as though it
were.** `phi4_aie4_acceptance.md` carries a paragraph headed *"What diverges is
not only the last bits"*, which states that two runs of the same binary have
produced different emitted tokens, tells the reader not to take the
non-associativity sentence above it as meaning the divergence is confined to
rounding, and points at `phi4_results.md` and `phi.md`. That paragraph is written
by the renderer rather than read from the record, because the record is not
edited after the fact — the same rule the rest of this page follows. What
remains, and what this note is about, is the wording of the record's own
`policy` and `sharpened_hypothesis` fields: accurate about the arithmetic, and
on their own an understatement of the defect.

Counts, magnitudes, the route split and the localisation are in
[the Phi-4 benchmarks](/docs/benchmarks/phi4_results/); the user-facing statement
is the fourth bullet of
[Phi's known limitations](/docs/models/phi/#known-limitations).

## What that means for the record

**The record is not stale and has not been edited.** It is an accurate,
immutable account of the run identified by `run_stamp 20260903T051041Z`,
produced by the harness as it stood at `81a01f7b`. Nothing in it was
regenerated or corrected by hand; the fixes above are in the harness only.

Two consequences worth stating plainly:

1. **The current harness has never run on hardware.** Its changes are verified
   offline by `src/test/phi4_corelib_aie4/verify_acceptance_guards.ps1`, which
   drives the real guard functions against this record. On this data the old
   gate and the new one reach the same verdict for Step 8 — both fire — so the
   recorded result stands. That is not the same as having been observed. Round 4
   also drove the harness itself offline, with no NPU, no `flm.exe` and no
   model: it completes all 20 steps, the Section 17 roll-up, the reload, the
   render and the readback. That exercises the pipeline, not the measurements.
2. **`C01` and `C31` read `met` here and will read `partial` next run.** The
   record is not wrong about what it measured; the coverage claim beside it
   was too generous, and only the claim changed.
3. **The Section 17 counts will move, and the movement is now measured rather
   than predicted.** Replaying this record's step results through the current
   harness offline — every step carried forward, nothing re-measured — the
   roll-up comes out **10 met, 14 partial, 2 not met, 26 not exercised** against
   the **12 / 12 / 2 / 26** in the record. The two that moved are `C01` and
   `C31`. Steps are unchanged at 16 met, 1 not met, 3 not exercised.

The next hardware run will emit a record whose `harness_sha256` matches the
harness beside it again, and this note should be updated or removed when it
does.

## Known gap: nothing automated compiles this backend or runs any of its tests

Stated as a gap, not as a plan. It is recorded here because this is the page
whose subject is what was and was not verified, and because everything else on
this page describes evidence that only exists if somebody runs something.

**No continuous-integration job builds the AIE4 path, and no job runs any test
of any kind.** Verified against the tree at the time of writing:

- The branch did not touch CI at all: `git diff --stat 2d6c4838..6a516f7f -- .github/`
  is empty.
- No workflow under `.github/workflows/` invokes `ctest`, `pytest` or
  `unittest` — grep over the whole directory returns nothing. The four
  workflows (`windows-build.yml`, `ubuntu-build.yml`, `debian-portable.yml`,
  `build-container.yml`) build and package; none of them tests.
- `windows-build.yml` configures with `cmake --preset windows-vs18`
  (`:34`, `:88`). That preset sets `CMAKE_BUILD_TYPE` and nothing else, and
  `FLM_ENABLE_CORELIB_AIE4` is set by no preset and by no workflow — so the
  AIE4 product C++ behind the `#if` is **never compiled** in CI. A change that
  breaks it compiles green.
- `src/test/phi4_corelib_aie4/CMakeLists.txt` is a standalone CMake project.
  `src/CMakeLists.txt` never `add_subdirectory`s it (its only
  `add_subdirectory` is `third_party/tokenizers-cpp`), so the host test suite
  is not part of the product build's `ctest` either.
- `src/test/phi4_corelib_aie4/verify_acceptance_guards.ps1` — the offline
  verifier that drives the real guard functions against this record, including
  the AST call-site lint that caught the round-4 accumulation-gate bug — is
  registered nowhere. It runs when a human remembers.

**And one job is named as though it tests.** `debian-portable.yml:122` defines
a job called **`test-summary`**. It runs no test: its single step
(`:127-133`) echoes `# Portable Build Test Results` and four hard-coded `✅`
lines — *built successfully*, *FFmpeg statically linked*, *XRT and XDNA plugin
bundled*, *FFTW bundled* — into `$GITHUB_STEP_SUMMARY`, unconditionally, with
`if: always()`. They are restatements of what the build job was configured to
do, emitted whether or not it did it. A maintainer scanning job names, or
reading a green run's summary, has every reason to conclude that tests ran.
This is pre-existing upstream FastFlow — the branch did not touch `.github/` —
so it is recorded here rather than fixed, but it is why the gap above is worse
than an absence: **the absence of testing is currently reported as four green
ticks.**

The consequence is worth stating in one sentence: **the evidence on this page,
and in the acceptance record beside it, was produced by hand on one machine and
nothing will reproduce it automatically.** Re-running it is a manual act. Until
that changes, treat a green build as saying nothing about this backend.

CI was ruled out of scope for the round in which this note was written. The gap
is recorded so it is inherited deliberately rather than by accident.

## Verifying the chain yourself

Run these under **`pwsh`** (PowerShell 7), not `powershell.exe`. On at least one
development box `Get-FileHash` is missing from Windows PowerShell 5.1 because
`$env:PSModulePath` puts `…\PowerShell\7\Modules` ahead of
`…\WindowsPowerShell\v1.0\Modules`, so 5.1 auto-loads PowerShell 7's
`Microsoft.PowerShell.Utility` manifest and gets a subset without it. That is an
environment quirk, not a missing feature — `pwsh` has it.

```powershell
# The record's own account of what produced it.
$r = Get-Content docs/docs/benchmarks/phi4_aie4_acceptance.json -Raw | ConvertFrom-Json
$r.identity | Select-Object binary_revision, checkout_revision, harness_sha256, driver_sha256

# The driver still matches; the harness intentionally does not.
Get-FileHash src/test/phi4_corelib_aie4/drive_flm_console.ps1        -Algorithm SHA256
Get-FileHash src/test/phi4_corelib_aie4/run_real_model_acceptance.ps1 -Algorithm SHA256

# Everything under the suite that changed since the recorded checkout.
# As of review round 4 this lists exactly three files -- the harness, the new
# shared acceptance_guards.ps1, and the offline verifier. drive_flm_console.ps1
# is not among them, which is why its hash still matches. Read the list rather
# than trusting a count written in prose somewhere.
git diff --name-only 81a01f7b..HEAD -- src/test/phi4_corelib_aie4/

# The guards the harness gained since, exercised against this record.
pwsh -File src/test/phi4_corelib_aie4/verify_acceptance_guards.ps1
```
