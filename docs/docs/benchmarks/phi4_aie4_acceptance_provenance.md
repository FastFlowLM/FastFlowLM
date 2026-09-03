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
