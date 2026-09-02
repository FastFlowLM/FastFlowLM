# Run-to-run determinism records (`DETERM-4`)

Committed evidence for the `DETERM-3` bit-identity baseline and for every
`DETERM-1`/`DETERM-2` gate breach observed on the AIE4 target.

**Why these are in the source tree.** They are produced under
`src/build/phi4-hardware/artifacts/`, which is gitignored. That has now cost
this effort its divergence evidence twice: once when a fixed artifact
directory overwrote the first 16/17 event before anyone could interrogate it,
and once when a finding that overturned a design position was written up from
records that no longer existed by the time a reviewer tried to check it.
Design `DETERM-4` requires retention; `run_hardware_suite.ps1` copies every
record here, and they are committed.

**Every record, not only the breaching ones.** A rate is only auditable if the
clean runs behind it are present too. `tools/report_phi4_corelib_baseline.py`
computes the published baseline straight from this tree:

```powershell
python tools/report_phi4_corelib_baseline.py `
    --input src/build/phi4-hardware/phi4_aie4_baseline.json `
    --determinism-glob 'src/test/phi4_corelib_aie4/determinism_records/*/determ1-*.json' `
    --markdown docs/docs/benchmarks/phi4_results.md
```

## Layout

One directory per suite run, named `<UTC stamp>-<PID>` — the same stamp the
run's artifact directory uses, so a record can be traced back to its run.

| file | what it is |
| --- | --- |
| `determ1-<route>.json` | the run-to-run comparison from that run's numeric golden |
| `determ1-<route>-<NNN>.json` | one sample of the `DETERM-3` campaign |
| `compare-summary-<route>.json` | the cross-implementation comparison against the corelib reference driver |
| `lmhead-<route>-<NNN>.json` | Step 9b's FP64 host LM-head analysis, written automatically for any pair that was not bit-identical |

Each `determ1` record carries the corelib DLL SHA-256, the FastFlow harness
SHA-256, the bit-identity counts, the observed maximum absolute difference,
the first diverging step, the verbatim gate-failure messages, and — for runs
made after the per-step capture existed — the measured `localisation`.

## Reading a breach

`localisation.source` is the answer `DETERM-1` inferred and this project
measured. At the step whose logits first differ, the harness records the exact
3072-element row that was fed to the LM head in each run:

* `lm_head` — the two runs fed the LM head an identical row and it produced
  different logits. This is what `DETERM-1` assumed.
* `model_body` — the two runs fed the LM head **different** rows, so the
  divergence entered before the LM head and `DETERM-1`'s localisation to the
  `3072 x 200064` dispatch does not hold for that event.

Records from before that capture existed have `localisation.measured: false`
and say why. They are still committed — they are the evidence for the
model-state signature in the task report — but they are **excluded** from the
pooled rate, because a record with no harness hash cannot be shown to describe
the same binary as the rest. The report tool states the excluded count and
lists the paths rather than dropping them silently.
