---
layout: docs
title: Phi4
parent: Benchmarks
nav_order: 6
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.30.
> - Under FLM's default NPU power mode (Performance)   
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models. 

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/phi4_mini_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/phi4_mini_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Phi-4-mini-instruct**  | NPU (FLM)    | 21.8	| 21.2	| 19.9	| 18.1	| 14.9	| 11.2|

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Phi-4-mini-instruct**  | NPU (FLM)    | 643	| 787	| 857	| 809	| 644	| 447 | 


---

<!-- BEGIN phi4-aie4-baseline -->

## Phi-4 mini instruct on AIE4 (corelib backend) — recorded baseline

Recorded per design section 15.6, **without pass/fail thresholds**. Design section 4 makes performance explicitly not a release blocker for this release; these figures exist so a later change can be compared against a measured starting point, not so a number can be defended.

> **Every latency and throughput figure here comes from ONE run on a shared lab machine.** Task 13 ran this same benchmark three times against the same binary and the same model within two hours and measured decode throughput of 22.5, 22.4 and 12.4 tokens/s at context 128 — a factor of 1.8, with no code change. Within a single run, per-token append latency stepped from 76 ms to 46 ms partway through the continuation sweep and stayed there. The machine runs corporate endpoint agents whose scans are not under this project's control, and the host share of a decode token is large enough for CPU contention to show.

> **Two different comparisons, and they do not share a rule.** Comparing a figure here against one from a DIFFERENT run — a later revision, another machine, this document a month from now — is subject to that 1.8x instability, so treat a difference below roughly 2x as unresolved unless it is reproduced across runs. The append-versus-re-prefill decision AT EACH POINT is not that comparison: its samples are **interleaved within a single point**, so a regime shift moves both routes together, and each point additionally has to beat the drift measured across it. That is why a single point can resolve differences the 2x rule could not.

> **The exemption is for a single point, not for the bracket.** Whether a given suffix can be decided depends on how quiet the machine was during that point, so the SET of decided points — and therefore the bracket's width — is subject to the same run-to-run instability as everything else here. The run-to-run table under "Where append stops winning" shows how far it has actually moved.

### Identity

| | |
| --- | --- |
| machine | `xcomedusad-43` |
| CPU | `AMD Eng Sample: 100-000001713-33_N` |
| NPU | `AMD XDNA(TM) NPU` |
| NPU driver | `32.0.20214.4161` |
| corelib DLL | `C:\Users\chiz\work\aie4-runtime-mirrored\ryzenai_corelib.dll` |
| corelib SHA-256 | `a523b23837c9bea3b06f8f8ae74e1e8292092eb73aaa93791022c448b865ecdb` |
| corelib version | `0.1.0` |
| corelib source revision | `e5258d29b5cb979d4a538994409b90ceff6e6e7a-untracked-only` |
| DynamicDispatch | `9999.0.0.0 (C:\Users\chiz\work\aie4-runtime-mirrored\dyn_dispatch_core.dll)` |
| RyzenMM | `0.10.2.0 (C:\Users\chiz\work\aie4-runtime-mirrored\ryzen_mm.dll)` |
| XRT | `32.0.20214.4161 (C:\Windows\SYSTEM32\xrt_coreutil.dll)` |
| model directory | `C:\Users\chiz\work\FastFlowLM\src\build\phi4-hardware\model` |
| model SHA-256 | `d6f503a9ea142c8b6320313d6ae341a88049b1b8ef01e641b2313fe42cdc7309` |
| FastFlow revision | `a02a2cf7e6cda62ab21e1383de2674b933ed2b74-untracked-only` |
| measured (UTC) | `2026-09-02T16:32:18Z` |

### Model load and TTFT

| | |
| --- | --- |
| manifest parse and file mapping | 4.5 ms |
| **1..4096 helper interrogation (`Phi4ShapePlan::Build`)** | **11282.6 ms** — **86% of load** |
| weight pack/upload (161 objects) | 1603.4 ms |
| stream, 76 device tensors, RoPE upload | 142.3 ms |
| unaccounted | 11.1 ms |
| total model load | 13043.9 ms |
| cold TTFT (19 prompt tokens, row extent 64) | 3169.4 ms |
| warm TTFT, same Stream after `clear_context()` | 49.1 ms |

### Fresh prefill

| rows | padded rows | wall time | tokens/s |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 144.7 ms | 6.9 |
| 2 | 64 | 50.5 ms | 39.6 |
| 65 | 128 | 70.6 ms | 920.6 |
| 129 | 256 | 92.7 ms | 1,392.2 |
| 257 | 512 | 136.4 ms | 1,884.7 |
| 513 | 1024 | 289.4 ms | 1,772.5 |
| 1025 | 2048 | 731.3 ms | 1,401.6 |
| 2049 | 3072 | 1171.7 ms | 1,748.7 |
| 3073 | 4096 | 1717.8 ms | 1,788.9 |
| 4096 | 4096 | 1746.3 ms | 2,345.5 |

### Decode

| starting context | tokens | tokens/s | p50 | p95 | synchronizes per pass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 20.04 | 41.1 ms | 91.8 ms | 129 |
| 512 | 128 | 21.16 | 41.7 ms | 54.8 ms | 129 |
| 2048 | 128 | 20.68 | 44.2 ms | 55.8 ms | 129 |

### Continuation routes

| history rows | suffix | route | samples | p50 | p95 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 512 | 1 | append | 5 | 48.9 ms | 52.9 ms |
| 512 | 1 | reprefill | 5 | 315.7 ms | 365.7 ms |
| 512 | 2 | append | 5 | 93.9 ms | 100.9 ms |
| 512 | 2 | reprefill | 5 | 332.0 ms | 343.5 ms |
| 512 | 4 | append | 5 | 182.5 ms | 192.3 ms |
| 512 | 4 | reprefill | 5 | 311.1 ms | 350.1 ms |
| 512 | 8 | append | 5 | 352.6 ms | 381.9 ms |
| 512 | 8 | reprefill | 5 | 334.1 ms | 336.2 ms |
| 512 | 12 | append | 5 | 513.2 ms | 530.7 ms |
| 512 | 12 | reprefill | 5 | 298.6 ms | 335.6 ms |
| 512 | 16 | append | 5 | 702.2 ms | 727.5 ms |
| 512 | 16 | reprefill | 5 | 318.0 ms | 391.0 ms |
| 512 | 24 | append | 5 | 1006.9 ms | 1048.9 ms |
| 512 | 24 | reprefill | 5 | 296.9 ms | 340.6 ms |
| 512 | 32 | append | 5 | 1429.6 ms | 1509.0 ms |
| 512 | 32 | reprefill | 5 | 308.3 ms | 335.4 ms |
| 512 | 64 | append | 5 | 2671.5 ms | 2750.8 ms |
| 512 | 64 | reprefill | 5 | 300.7 ms | 342.8 ms |
| 512 | 128 | append | 5 | 5301.3 ms | 5356.2 ms |
| 512 | 128 | reprefill | 5 | 311.2 ms | 344.9 ms |
| 512 | 256 | append | 5 | 10624.5 ms | 10841.4 ms |
| 512 | 256 | reprefill | 5 | 294.8 ms | 344.2 ms |
| 2048 | 1 | append | 5 | 54.7 ms | 61.7 ms |
| 2048 | 1 | reprefill | 5 | 1161.0 ms | 1176.0 ms |
| 2048 | 2 | append | 5 | 104.8 ms | 113.8 ms |
| 2048 | 2 | reprefill | 5 | 1152.3 ms | 1173.2 ms |
| 2048 | 4 | append | 5 | 196.4 ms | 203.7 ms |
| 2048 | 4 | reprefill | 5 | 1156.1 ms | 1162.1 ms |
| 2048 | 8 | append | 5 | 370.5 ms | 386.5 ms |
| 2048 | 8 | reprefill | 5 | 1152.1 ms | 1157.7 ms |
| 2048 | 12 | append | 5 | 547.3 ms | 589.4 ms |
| 2048 | 12 | reprefill | 5 | 1148.7 ms | 1156.0 ms |
| 2048 | 16 | append | 5 | 783.4 ms | 1536.1 ms |
| 2048 | 16 | reprefill | 5 | 1158.3 ms | 1190.8 ms |
| 2048 | 24 | append | 5 | 1229.6 ms | 2382.6 ms |
| 2048 | 24 | reprefill | 5 | 1147.3 ms | 1160.7 ms |
| 2048 | 32 | append | 5 | 1545.3 ms | 2234.5 ms |
| 2048 | 32 | reprefill | 5 | 1146.1 ms | 1173.4 ms |
| 2048 | 64 | append | 5 | 4158.2 ms | 4624.3 ms |
| 2048 | 64 | reprefill | 5 | 1156.3 ms | 1175.9 ms |
| 2048 | 128 | append | 5 | 5775.1 ms | 6003.7 ms |
| 2048 | 128 | reprefill | 5 | 1150.6 ms | 1180.5 ms |
| 2048 | 256 | append | 5 | 11376.5 ms | 13055.3 ms |
| 2048 | 256 | reprefill | 5 | 1160.7 ms | 1245.2 ms |

#### Where append stops winning

**This is a BRACKET, not a threshold.** Append and re-prefill samples are interleaved within each point, so a machine regime shift moves both together rather than one. A point counts as decided only when the gap between the routes exceeds **both** the larger within-point p50-to-p95 spread **and** the larger drift between a route's first and last sample there; anything else widens the bracket. Task 14 has to choose this constant, so what it needs to see is how much room the measurement leaves — not a number picked because it was the last grid point where append happened to win.

| rendered history | append decisively wins up to | re-prefill decisively wins from | crossover lies in | margin at each bracket edge |
| ---: | ---: | ---: | :--- | :--- |
| 2048 | 12 | 64 | `(12, 64]` — not resolved further | 12: 14.3x, 64: 6.4x |
| 512 | 4 | 12 | `(4, 12]` — not resolved further | 4: 3.3x, 12: 5.8x |

18 of 22 sweep points were decided; 4 were not and widen the brackets above. The margin column is the gap between the routes divided by the uncertainty it had to beat at that suffix.

Undecided points, which is where the bracket's width comes from: history 2048 at suffix 16, 24, 32; history 512 at suffix 8.

##### The same measurement, run to run

**The bracket's upper edge is not stable and its lower edge is.** This table is every render of this document that recorded a crossover, from the committed baseline artifacts.

| measured (UTC) | interleaved | grid | history 2048 | history 512 | decided | note |
| --- | :---: | ---: | ---: | ---: | ---: | :--- |
| 2026-09-02T13:12:13Z | no | 5 points | threshold `2` | threshold `2` | n/a | sparse five-point grid; reported a single threshold, retracted as a grid artifact (report section 12). Not comparable. |
| 2026-09-02T14:55:44Z | no | 11 points | `(12, 16]` | `(4, 8]` | 21/22 |  |
| 2026-09-02T15:34:01Z | yes | 11 points | `(12, 16]` | `(4, 8]` | 22/22 |  |
| 2026-09-02T16:32:18Z | yes | 11 points | `(12, 64]` | `(4, 12]` | 18/22 |  |

> **Read the lower edge as measured and the upper edge as an upper bound.** Across the 2 interleaved runs the lower edge has been 12 at history 2048; 4 at history 512 every time, while the upper edge has taken 16, 64 at history 2048; 8, 12 at history 512 on the same binary and the same model. The upper edge moves with how quiet the machine was, because that is what decides how many points near the crossover can be called at all.

2 of the 4 rows are non-interleaved and are excluded from that comparison, because they measured the routes in blocks rather than paired in time. They are shown for provenance.

Prefix-monotonic: `True`. Decision rule: append and re-prefill samples are INTERLEAVED, so a machine regime shift moves both routes together; a point is decided only when the gap between the two routes' p50 exceeds BOTH the larger within-point p50-to-p95 spread AND the larger drift between a route's first and last sample at that point. The reported figure is a BRACKET, not a threshold..

### V scatter, memory and synchronization

| | |
| --- | --- |
| V cache reads per model step | 32 |
| per-head V writes per model step | 256 |
| V bytes transferred | 44.60 GiB |
| V scatter wall time | 3315.9 ms |
| FP16 embedding | 1.14 GiB |
| KV cache | 512.00 MiB |
| corelib packed weights | 1.95 GiB |
| scratch and device tensors | 161.89 MiB |
| mapped ONNX source | 3.03 GiB |
| peak host private bytes | 7.67 GiB |
| peak host working set | 9.52 GiB |

128-token stability window: device tensors created after warmup **0**, weight objects **0**, net live corelib objects **0**, private-byte growth **0 B**, least-squares private-byte slope over tokens 9..128 **-50.21 KiB/token**.

### DETERM-3 — run-to-run logit bit-identity baseline

Two runs of the same binary, same device, same explicit token IDs, proven by recorded SHA-256 to have loaded the same corelib DLL. Per `DETERM-3` the routes are reported separately and never averaged: append and re-prefill drive different row extents through the same LM-head shape.

| route | runs (n) | step bit-identity | run bit-identity | max abs diff: max / median | nonzero runs | ≥ 20 runs? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| append | 41 | 97.42% (n = 697) | 95.12% (n = 41) | 49.25 / 0.0 | 2 | yes |
| reprefill | 41 | 100.00% (n = 697) | 100.00% (n = 41) | 0.0 / 0.0 | 0 | yes |

**This window is not a settled baseline.** The figures above are what has been measured; the problems listed below say why they cannot yet be cited as the floor `DETERM-2`'s "degrades from the recorded baseline" clause needs.

Observed maximum absolute difference, `append`: `49.2148` × 1, `49.25` × 1.

#### Where the divergence enters — measured

At the step whose logits first differ, the harness records the exact row that was fed to the LM head in each run, so this is an observation and not an inference. **2 event(s): `model_body`**.

> **The instrument perturbs what it measures.** These 2 localisation(s) were measured by capturing the LM-head input after every model step, which adds a host tensor read and a stream acquisition between steps — changing the timing of exactly the window a race would occupy. The phenomenon survives the instrumentation: it still reproduces, and with the same coarse signature. What this data cannot rule out is that the capture shifts the rate, or which step is reached first — the campaigns are far too small to detect either.

In the **2** `model_body` event(s) the two runs fed the LM head **different rows** — 2,571, 2,754 of 3,072 elements differing, at `decode[6]`, `decode[8]`. **In those events the divergence enters the model body, not the LM-head dispatch.** Each run's LM head was separately measured to be correctly rounded against its own input, so it is faithfully transforming inputs that already differ.

**The layer at which it enters is not known.** Layer 0's K and V have been bit-identical in the events where the emitted tokens matched, and layer 31's have not, which bounds it to somewhere above layer 0. Narrowing it further needs per-layer capture, and until then this is an open question rather than a characterised one.

> **2 run(s) in this window breached a `DETERM-2` HARD GATE.** They are counted in the rates above -- dropping them would bias the figures upward, and silently -- and they are not a wobble within the recorded tolerance. Read the run records before citing anything here as a settled baseline.

The two routes' observed rates differ, but **the n does not support calling the rate route-dependent** (Fisher exact p = 0.494 against 0.05). They are still reported separately, per `DETERM-3`, because pooling them would hide a difference that a larger campaign might resolve — not because this one resolved it.

- 64 record(s) carry no FastFlow harness SHA-256 and were EXCLUDED from the pooled figures: they cannot be shown to describe the same binary as the rest. They remain committed as evidence and are listed under `excluded_sources`.
- route 'append': 2 run(s) recorded a DETERM-2 gate failure. They ARE counted in the rate above -- dropping them would bias it upward -- but a baseline should not be declared over a window containing a hard-gate failure.

<!-- END phi4-aie4-baseline -->

<!-- BEGIN phi4-continuation-threshold -->

## Phi-4 continuation routing — the release-fixed threshold

Design Section 10.7 fixes ONE integer for the release and applies it at every history length: append when the suffix is at most the threshold, clear and re-prefill when it is larger. It is a FastFlow backend constant — not model-package data, not a runtime calibration — so route choice does not drift with thermal state or load.

```cpp
inline constexpr std::uint32_t kContinuationAppendThreshold = 4;
```

**Selected: 4.** Suffix lengths [1, 2, 4] are the sampled lengths whose append p95 is lower than re-prefill p95 at BOTH measured history lengths, and they form a prefix of the sampled set [1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 256], so the largest of them is the threshold.

### Which sampled lengths append wins, per history

| history rows | append p95 wins at | last winning length |
| ---: | :--- | ---: |
| 512 | [1, 2, 4] | 4 |
| 2048 | [1, 2, 4, 8, 12] | 12 |

The rule is a conjunction, so the constant is the smaller of those ceilings. It is NOT a per-history policy: Section 10.7 specifies one integer, and one integer cannot be optimal at two history lengths whose crossovers differ.

### What the single constant gives up

**Worst case: history 2048, suffix 8 — 3.0x slower, +771.2 ms.** Every conceded point follows, ordered by slowdown. The widest conceded suffix is not the most expensive one, because append cost grows with suffix length while re-prefill cost does not — so the band's cost must be read off its narrow end.

| history rows | suffix | append p95 | re-prefill p95 | route taken | penalty | slowdown |
| ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| 2048 | 8 | 386.5 ms | 1,157.7 ms | re-prefill | +771.2 ms | 3.0x |
| 2048 | 12 | 589.4 ms | 1,156.0 ms | re-prefill | +566.6 ms | 2.0x |
| 512 | none | — | — | — | — | — |

Those are measured losses taken deliberately. The alternative — a threshold above the smaller ceiling — is optimal for the longer history and WRONG for the shorter one, where it would append past the point at which append has already lost. Conceding measured throughput at one history is the cheaper error than routing against the measurement at the other.

There is a second reason to prefer the smaller ceiling, and it is about confidence rather than cost. Task 13 measured this crossover three times and found the bracket's LOWER edge stable across runs and its UPPER edge not: the upper edge moves with how quiet the machine was, because that is what decides how many points near the crossover can be called at all. The threshold selected here sits at or below the lower edge at every measured history, so it is inside the append-wins region on every run. A threshold chosen from the upper edge would be supported by some runs and not others.

### The measurement behind it

| history rows | suffix | append p50 | append p95 | re-prefill p50 | re-prefill p95 | append p95 wins |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 512 | 1 | 48.9 ms | 52.9 ms | 315.7 ms | 365.7 ms | yes |
| 512 | 2 | 93.9 ms | 100.9 ms | 332.0 ms | 343.5 ms | yes |
| 512 | 4 | 182.5 ms | 192.3 ms | 311.1 ms | 350.1 ms | yes |
| 512 | 8 | 352.6 ms | 381.9 ms | 334.1 ms | 336.2 ms | no |
| 512 | 12 | 513.2 ms | 530.7 ms | 298.6 ms | 335.6 ms | no |
| 512 | 16 | 702.2 ms | 727.5 ms | 318.0 ms | 391.0 ms | no |
| 512 | 24 | 1,006.9 ms | 1,048.9 ms | 296.9 ms | 340.6 ms | no |
| 512 | 32 | 1,429.6 ms | 1,509.0 ms | 308.3 ms | 335.4 ms | no |
| 512 | 64 | 2,671.5 ms | 2,750.8 ms | 300.7 ms | 342.8 ms | no |
| 512 | 128 | 5,301.3 ms | 5,356.2 ms | 311.2 ms | 344.9 ms | no |
| 512 | 256 | 10,624.5 ms | 10,841.4 ms | 294.8 ms | 344.2 ms | no |
| 2048 | 1 | 54.7 ms | 61.7 ms | 1,161.0 ms | 1,176.0 ms | yes |
| 2048 | 2 | 104.8 ms | 113.8 ms | 1,152.3 ms | 1,173.2 ms | yes |
| 2048 | 4 | 196.4 ms | 203.7 ms | 1,156.1 ms | 1,162.1 ms | yes |
| 2048 | 8 | 370.5 ms | 386.5 ms | 1,152.1 ms | 1,157.7 ms | yes |
| 2048 | 12 | 547.3 ms | 589.4 ms | 1,148.7 ms | 1,156.0 ms | yes |
| 2048 | 16 | 783.4 ms | 1,536.1 ms | 1,158.3 ms | 1,190.8 ms | no |
| 2048 | 24 | 1,229.6 ms | 2,382.6 ms | 1,147.3 ms | 1,160.7 ms | no |
| 2048 | 32 | 1,545.3 ms | 2,234.5 ms | 1,146.1 ms | 1,173.4 ms | no |
| 2048 | 64 | 4,158.2 ms | 4,624.3 ms | 1,156.3 ms | 1,175.9 ms | no |
| 2048 | 128 | 5,775.1 ms | 6,003.7 ms | 1,150.6 ms | 1,180.5 ms | no |
| 2048 | 256 | 11,376.5 ms | 13,055.3 ms | 1,160.7 ms | 1,245.2 ms | no |

At least 5 warm samples per route per point, append and re-prefill interleaved WITHIN each point. p50 and p95 are nearest-rank and were recomputed from the raw samples by the calibrator, not copied.

### Do the earlier runs permit this constant?

Not: would they have selected it. Each recorded crossover run carries, per history, the bracket `[lower, upper]`, and the lower edge is that run's per-history winner CEILING. The minimum of those ceilings is an **upper bound** on what Section 10.7 would have selected from that run, and in general only an upper bound: the rule takes the largest suffix in the INTERSECTION of the winner sets, which equals the minimum of the ceilings only when each winner set is downward-closed. A crossover record stores the edges and never the winner sets, so that property cannot be checked from it. What holds with no assumption is the inequality — any suffix winning at every history is at most every ceiling — so that is what is enforced here.

Task 13 measured the lower edges to be stable and the upper edges not. This bound touches only the lower edges, which is why an unstable upper edge cannot move it.

| measured (UTC) | interleaved | bounds the threshold at | permits 4 | meets the bound exactly |
| --- | :---: | ---: | :---: | :---: |
| 2026-09-02T13:12:13Z | no | (2) | excluded — routes not measured against the same machine state | — |
| 2026-09-02T14:55:44Z | no | (4) | excluded — routes not measured against the same machine state | — |
| 2026-09-02T15:34:01Z | yes | 4 | yes | yes |
| 2026-09-02T16:32:18Z | yes | 4 | yes | yes |

Every interleaved run on record permits 4, and 2 of 2 bound it there exactly. An exactly-met bound is what "that run would have selected the same constant" would need, but only under the downward-closure assumption above, which its record does not attest.

This run's own per-history winner sets ARE downward-closed at every history, checked directly from the sampled points rather than assumed. That is only a property of this run; it says nothing about the earlier ones, whose winner sets were never recorded.

The non-interleaved rows are shown for provenance and excluded from the verdict, because their two routes were measured in separate blocks and a machine regime shift there lands on one route and not the other. The first of them bounds the constant lower for a reason that has nothing to do with the machine: it swept only the five suffix lengths Section 10.7 names, and on `{1, 2, 32, 128, 256}` the last winning sampled length is 2 whether the true crossover is at 3 or at 31. **Two earlier answers are withdrawn and must not be reused: the threshold `2` from that sparse grid, and the extrapolated figures `≈9 at history 512, ≈26 at history 2048` from a contended non-interleaved run.**

**The selection RULE is Section 10.7's, verbatim and unqualified; the measurement GRID is wider than Section 10.7 specifies, deliberately.** Those are two different things and only one of them changed. The rule — largest suffix whose append p95 is lower at both history lengths, after asserting the winners are prefix-contiguous — was applied as written. The grid was not: Section 10.7 names five suffix lengths, and on those five alone this same measurement yields 2, the answer since retracted as a grid artifact. Locating a crossover needs sample points near it, so the five named lengths are treated as a floor and every additional measured length is used. The measurement plan was corrected; the decision rule was not touched.

### Identity of the run this constant came from

| | |
| --- | --- |
| machine | `xcomedusad-43` |
| CPU | `AMD Eng Sample: 100-000001713-33_N` |
| NPU | `AMD XDNA(TM) NPU` |
| NPU driver | `32.0.20214.4161` |
| corelib SHA-256 | `a523b23837c9bea3b06f8f8ae74e1e8292092eb73aaa93791022c448b865ecdb` |
| corelib source revision | `e5258d29b5cb979d4a538994409b90ceff6e6e7a-untracked-only` |
| model SHA-256 | `d6f503a9ea142c8b6320313d6ae341a88049b1b8ef01e641b2313fe42cdc7309` |
| FastFlow revision | `a02a2cf7e6cda62ab21e1383de2674b933ed2b74-untracked-only` |
| measured (UTC) | `2026-09-02T16:32:18Z` |

**"Release-fixed" does not mean hardware-independent.** Every measurement behind this constant comes from the single machine, corelib build and model named above. Nothing here establishes where the crossover sits on different silicon, on a corelib whose append or prefill path changed, or on a different model. Section 10.7 fixes the constant for the release rather than calibrating at run time, so that scope is intended — but a port to other hardware needs this re-measured, not inherited.

The timing table above is published here and is NOT shipped in the model package: Section 10.7 carries the threshold and no cost table.

<!-- END phi4-continuation-threshold -->
