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

> **Two different comparisons, and they do not share a rule.** Comparing a figure here against one from a DIFFERENT run — a later revision, another machine, this document a month from now — is subject to that 1.8x instability, so treat a difference below roughly 2x as unresolved unless it is reproduced across runs. The append-versus-re-prefill comparison below is not that comparison: its samples are **interleaved within a single point**, so a regime shift moves both routes together, and each point additionally has to beat the drift measured across it. That is why it can resolve differences the 2x rule could not — the instability it would be guarding against has been measured and subtracted rather than assumed away.

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
| FastFlow revision | `26f2dcb5d2d8d8b7cd2915f5a56db94e5a68fb7d-untracked-only` |
| measured (UTC) | `2026-09-02T15:34:01Z` |

### Model load and TTFT

| | |
| --- | --- |
| manifest parse and file mapping | 8.0 ms |
| **1..4096 helper interrogation (`Phi4ShapePlan::Build`)** | **15739.6 ms** — **90% of load** |
| weight pack/upload (161 objects) | 1622.3 ms |
| stream, 76 device tensors, RoPE upload | 146.3 ms |
| unaccounted | 11.0 ms |
| total model load | 17527.3 ms |
| cold TTFT (19 prompt tokens, row extent 64) | 3667.9 ms |
| warm TTFT, same Stream after `clear_context()` | 149.7 ms |

### Fresh prefill

| rows | padded rows | wall time | tokens/s |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 193.4 ms | 5.2 |
| 2 | 64 | 89.1 ms | 22.5 |
| 65 | 128 | 138.3 ms | 469.9 |
| 129 | 256 | 173.1 ms | 745.1 |
| 257 | 512 | 226.4 ms | 1,135.0 |
| 513 | 1024 | 342.4 ms | 1,498.2 |
| 1025 | 2048 | 715.9 ms | 1,431.7 |
| 2049 | 3072 | 1187.0 ms | 1,726.2 |
| 3073 | 4096 | 1657.5 ms | 1,854.0 |
| 4096 | 4096 | 1699.2 ms | 2,410.6 |

### Decode

| starting context | tokens | tokens/s | p50 | p95 | synchronizes per pass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 22.27 | 41.9 ms | 51.4 ms | 129 |
| 512 | 128 | 21.88 | 42.4 ms | 55.3 ms | 129 |
| 2048 | 128 | 20.67 | 45.6 ms | 55.3 ms | 129 |

### Continuation routes

| history rows | suffix | route | samples | p50 | p95 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 512 | 1 | append | 5 | 43.9 ms | 66.7 ms |
| 512 | 1 | reprefill | 5 | 360.6 ms | 364.1 ms |
| 512 | 2 | append | 5 | 115.8 ms | 243.6 ms |
| 512 | 2 | reprefill | 5 | 348.4 ms | 440.3 ms |
| 512 | 4 | append | 5 | 316.5 ms | 323.9 ms |
| 512 | 4 | reprefill | 5 | 380.9 ms | 396.9 ms |
| 512 | 8 | append | 5 | 654.1 ms | 668.3 ms |
| 512 | 8 | reprefill | 5 | 380.8 ms | 390.3 ms |
| 512 | 12 | append | 5 | 1006.7 ms | 1115.1 ms |
| 512 | 12 | reprefill | 5 | 377.9 ms | 386.3 ms |
| 512 | 16 | append | 5 | 1270.1 ms | 1311.7 ms |
| 512 | 16 | reprefill | 5 | 384.4 ms | 385.1 ms |
| 512 | 24 | append | 5 | 1914.4 ms | 1964.4 ms |
| 512 | 24 | reprefill | 5 | 384.2 ms | 386.0 ms |
| 512 | 32 | append | 5 | 2505.8 ms | 2591.3 ms |
| 512 | 32 | reprefill | 5 | 383.0 ms | 385.3 ms |
| 512 | 64 | append | 5 | 4977.2 ms | 5095.7 ms |
| 512 | 64 | reprefill | 5 | 385.4 ms | 388.8 ms |
| 512 | 128 | append | 5 | 9744.2 ms | 10185.0 ms |
| 512 | 128 | reprefill | 5 | 384.9 ms | 405.2 ms |
| 512 | 256 | append | 5 | 19725.7 ms | 20580.1 ms |
| 512 | 256 | reprefill | 5 | 387.5 ms | 389.7 ms |
| 2048 | 1 | append | 5 | 80.7 ms | 99.7 ms |
| 2048 | 1 | reprefill | 5 | 1202.0 ms | 1204.7 ms |
| 2048 | 2 | append | 5 | 171.9 ms | 180.0 ms |
| 2048 | 2 | reprefill | 5 | 1189.0 ms | 1207.7 ms |
| 2048 | 4 | append | 5 | 361.5 ms | 370.1 ms |
| 2048 | 4 | reprefill | 5 | 1186.0 ms | 1212.8 ms |
| 2048 | 8 | append | 5 | 638.2 ms | 664.7 ms |
| 2048 | 8 | reprefill | 5 | 1191.9 ms | 1201.6 ms |
| 2048 | 12 | append | 5 | 945.2 ms | 993.8 ms |
| 2048 | 12 | reprefill | 5 | 1197.2 ms | 1205.5 ms |
| 2048 | 16 | append | 5 | 1319.2 ms | 1350.4 ms |
| 2048 | 16 | reprefill | 5 | 1185.4 ms | 1197.1 ms |
| 2048 | 24 | append | 5 | 1969.3 ms | 1978.8 ms |
| 2048 | 24 | reprefill | 5 | 1194.7 ms | 1205.4 ms |
| 2048 | 32 | append | 5 | 2763.3 ms | 2896.6 ms |
| 2048 | 32 | reprefill | 5 | 1194.7 ms | 1212.1 ms |
| 2048 | 64 | append | 5 | 2942.3 ms | 3617.0 ms |
| 2048 | 64 | reprefill | 5 | 1149.0 ms | 1155.6 ms |
| 2048 | 128 | append | 5 | 6159.0 ms | 7009.8 ms |
| 2048 | 128 | reprefill | 5 | 1147.7 ms | 1193.5 ms |
| 2048 | 256 | append | 5 | 13732.9 ms | 14008.4 ms |
| 2048 | 256 | reprefill | 5 | 1156.3 ms | 1171.1 ms |

#### Where append stops winning

**This is a BRACKET, not a threshold.** Append and re-prefill samples are interleaved within each point, so a machine regime shift moves both together rather than one. A point counts as decided only when the gap between the routes exceeds **both** the larger within-point p50-to-p95 spread **and** the larger drift between a route's first and last sample there; anything else widens the bracket. Task 14 has to choose this constant, so what it needs to see is how much room the measurement leaves — not a number picked because it was the last grid point where append happened to win.

| rendered history | append decisively wins up to | re-prefill decisively wins from | crossover lies in | margin at each bracket edge |
| ---: | ---: | ---: | :--- | :--- |
| 2048 | 12 | 16 | `(12, 16]` — not resolved further | 12: 5.2x, 16: 4.3x |
| 512 | 4 | 8 | `(4, 8]` — not resolved further | 4: 3.9x, 8: 19.3x |

22 of 22 sweep points were decided; 0 were not and widen the brackets above. The margin column is the gap between the routes divided by the uncertainty it had to beat at that suffix.

Prefix-monotonic: `True`. Decision rule: append and re-prefill samples are INTERLEAVED, so a machine regime shift moves both routes together; a point is decided only when the gap between the two routes' p50 exceeds BOTH the larger within-point p50-to-p95 spread AND the larger drift between a route's first and last sample at that point. The reported figure is a BRACKET, not a threshold..

### V scatter, memory and synchronization

| | |
| --- | --- |
| V cache reads per model step | 32 |
| per-head V writes per model step | 256 |
| V bytes transferred | 44.60 GiB |
| V scatter wall time | 4355.6 ms |
| FP16 embedding | 1.14 GiB |
| KV cache | 512.00 MiB |
| corelib packed weights | 1.95 GiB |
| scratch and device tensors | 161.89 MiB |
| mapped ONNX source | 3.03 GiB |
| peak host private bytes | 7.63 GiB |
| peak host working set | 9.55 GiB |

128-token stability window: device tensors created after warmup **0**, weight objects **0**, net live corelib objects **0**, private-byte growth **0 B**, least-squares private-byte slope over tokens 9..128 **-234.96 KiB/token**.

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

In every measured event the two runs fed the LM head **different rows** — 2,571, 2,754 of 3,072 elements differing, at `decode[6]`, `decode[8]`. **The divergence therefore enters the model body, not the LM-head dispatch.** Each run's LM head was separately measured to be correctly rounded against its own input, so it is faithfully transforming inputs that already differ.

**The layer at which it enters is not known.** Layer 0's K and V have been bit-identical in the events where the emitted tokens matched, and layer 31's have not, which bounds it to somewhere above layer 0. Narrowing it further needs per-layer capture, and until then this is an open question rather than a characterised one.

> **2 run(s) in this window breached a `DETERM-2` HARD GATE.** They are counted in the rates above -- dropping them would bias the figures upward, and silently -- and they are not a wobble within the recorded tolerance. Read the run records before citing anything here as a settled baseline.

The two routes' observed rates differ, but **the n does not support calling the rate route-dependent** (Fisher exact p = 0.494 against 0.05). They are still reported separately, per `DETERM-3`, because pooling them would hide a difference that a larger campaign might resolve — not because this one resolved it.

- 64 record(s) carry no FastFlow harness SHA-256 and were EXCLUDED from the pooled figures: they cannot be shown to describe the same binary as the rest. They remain committed as evidence and are listed under `excluded_sources`.
- route 'append': 2 run(s) recorded a DETERM-2 gate failure. They ARE counted in the rate above -- dropping them would bias it upward -- but a baseline should not be declared over a window containing a hard-gate failure.

<!-- END phi4-aie4-baseline -->
