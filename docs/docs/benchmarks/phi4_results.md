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

> **Every latency and throughput figure here comes from ONE run on a shared lab machine.** Task 13 ran this same benchmark three times against the same binary and the same model within two hours and measured decode throughput of 22.5, 22.4 and 12.4 tokens/s at context 128 — a factor of 1.8, with no code change. Within a single run, per-token append latency stepped from 76 ms to 46 ms partway through the continuation sweep and stayed there. The machine runs corporate endpoint agents whose scans are not under this project's control, and the host share of a decode token is large enough for CPU contention to show. Treat a difference below roughly 2x as unresolved unless it is reproduced across runs.

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
| FastFlow revision | `21ad125ffdea96341d7bf898348dcccc1c26e0ad-untracked-only` |
| measured (UTC) | `2026-09-02T14:55:44Z` |

### Model load and TTFT

| | |
| --- | --- |
| manifest parse and file mapping | 4.6 ms |
| **1..4096 helper interrogation (`Phi4ShapePlan::Build`)** | **11044.2 ms** — **86% of load** |
| weight pack/upload (161 objects) | 1606.7 ms |
| stream, 76 device tensors, RoPE upload | 124.5 ms |
| unaccounted | 11.2 ms |
| total model load | 12791.1 ms |
| cold TTFT (19 prompt tokens, row extent 64) | 3252.0 ms |
| warm TTFT, same Stream after `clear_context()` | 150.3 ms |

### Fresh prefill

| rows | padded rows | wall time | tokens/s |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 244.9 ms | 4.1 |
| 2 | 64 | 150.8 ms | 13.3 |
| 65 | 128 | 173.5 ms | 374.5 |
| 129 | 256 | 191.5 ms | 673.5 |
| 257 | 512 | 229.2 ms | 1,121.2 |
| 513 | 1024 | 381.4 ms | 1,345.0 |
| 1025 | 2048 | 742.3 ms | 1,380.9 |
| 2049 | 3072 | 1182.0 ms | 1,733.6 |
| 3073 | 4096 | 1750.9 ms | 1,755.1 |
| 4096 | 4096 | 1746.1 ms | 2,345.7 |

### Decode

| starting context | tokens | tokens/s | p50 | p95 | synchronizes per pass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 21.08 | 42.2 ms | 74.2 ms | 129 |
| 512 | 128 | 21.74 | 42.4 ms | 54.6 ms | 129 |
| 2048 | 128 | 20.77 | 44.6 ms | 54.0 ms | 129 |

### Continuation routes

| history rows | suffix | route | samples | p50 | p95 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 512 | 1 | append | 5 | 73.5 ms | 95.8 ms |
| 512 | 1 | reprefill | 5 | 381.6 ms | 386.0 ms |
| 512 | 2 | append | 5 | 155.4 ms | 179.3 ms |
| 512 | 2 | reprefill | 5 | 383.3 ms | 385.8 ms |
| 512 | 4 | append | 5 | 312.1 ms | 334.0 ms |
| 512 | 4 | reprefill | 5 | 383.6 ms | 393.3 ms |
| 512 | 8 | append | 5 | 615.5 ms | 644.8 ms |
| 512 | 8 | reprefill | 5 | 384.4 ms | 386.7 ms |
| 512 | 12 | append | 5 | 949.3 ms | 1154.9 ms |
| 512 | 12 | reprefill | 5 | 384.1 ms | 385.1 ms |
| 512 | 16 | append | 5 | 1224.6 ms | 1248.3 ms |
| 512 | 16 | reprefill | 5 | 386.2 ms | 388.0 ms |
| 512 | 24 | append | 5 | 1871.6 ms | 1888.8 ms |
| 512 | 24 | reprefill | 5 | 383.6 ms | 393.7 ms |
| 512 | 32 | append | 5 | 2590.0 ms | 2677.0 ms |
| 512 | 32 | reprefill | 5 | 383.2 ms | 385.9 ms |
| 512 | 64 | append | 5 | 5150.2 ms | 5345.3 ms |
| 512 | 64 | reprefill | 5 | 384.9 ms | 386.4 ms |
| 512 | 128 | append | 5 | 10033.3 ms | 10188.2 ms |
| 512 | 128 | reprefill | 5 | 391.9 ms | 405.8 ms |
| 512 | 256 | append | 5 | 19788.3 ms | 20785.0 ms |
| 512 | 256 | reprefill | 5 | 386.2 ms | 395.5 ms |
| 2048 | 1 | append | 5 | 69.0 ms | 107.1 ms |
| 2048 | 1 | reprefill | 5 | 1196.9 ms | 1286.5 ms |
| 2048 | 2 | append | 5 | 158.7 ms | 180.9 ms |
| 2048 | 2 | reprefill | 5 | 1196.8 ms | 1206.0 ms |
| 2048 | 4 | append | 5 | 300.1 ms | 315.7 ms |
| 2048 | 4 | reprefill | 5 | 1195.3 ms | 1202.4 ms |
| 2048 | 8 | append | 5 | 623.8 ms | 654.4 ms |
| 2048 | 8 | reprefill | 5 | 1191.3 ms | 1200.3 ms |
| 2048 | 12 | append | 5 | 943.0 ms | 982.1 ms |
| 2048 | 12 | reprefill | 5 | 1198.4 ms | 1212.4 ms |
| 2048 | 16 | append | 5 | 1227.7 ms | 1248.8 ms |
| 2048 | 16 | reprefill | 5 | 1191.2 ms | 1219.7 ms |
| 2048 | 24 | append | 5 | 1873.7 ms | 1944.5 ms |
| 2048 | 24 | reprefill | 5 | 1189.0 ms | 1199.6 ms |
| 2048 | 32 | append | 5 | 1444.6 ms | 2501.7 ms |
| 2048 | 32 | reprefill | 5 | 1170.4 ms | 1175.9 ms |
| 2048 | 64 | append | 5 | 2864.5 ms | 2889.7 ms |
| 2048 | 64 | reprefill | 5 | 1175.4 ms | 1186.3 ms |
| 2048 | 128 | append | 5 | 5753.3 ms | 6113.8 ms |
| 2048 | 128 | reprefill | 5 | 1171.6 ms | 1175.7 ms |
| 2048 | 256 | append | 5 | 11609.4 ms | 11703.6 ms |
| 2048 | 256 | reprefill | 5 | 1173.4 ms | 1186.9 ms |

#### Where append stops winning

**This is a BRACKET, not a threshold.** A point counts as decided only when the gap between the two routes exceeds the larger of their own p50-to-p95 spreads at that point; anything else widens the bracket. Task 14 has to choose this constant, so what it needs to see is how much room the measurement leaves — not a number picked because it was the last grid point where append happened to win.

| rendered history | append decisively wins up to | re-prefill decisively wins from | crossover lies in |
| ---: | ---: | ---: | :--- |
| 2048 | 12 | 16 | `(12, 16]` — not resolved further |
| 512 | 4 | 8 | `(4, 8]` — not resolved further |

Prefix-monotonic: `True`. Decision rule: a point is decided only when the gap between the two routes' p50 exceeds the larger of the two routes' own p50-to-p95 spread at that point; the reported figure is a BRACKET, not a threshold.

### V scatter, memory and synchronization

| | |
| --- | --- |
| V cache reads per model step | 32 |
| per-head V writes per model step | 256 |
| V bytes transferred | 27.41 GiB |
| V scatter wall time | 2765.5 ms |
| FP16 embedding | 1.14 GiB |
| KV cache | 512.00 MiB |
| corelib packed weights | 1.95 GiB |
| scratch and device tensors | 161.89 MiB |
| mapped ONNX source | 3.03 GiB |
| peak host private bytes | 7.65 GiB |
| peak host working set | 9.54 GiB |

128-token stability window: device tensors created after warmup **0**, weight objects **0**, net live corelib objects **0**, private-byte growth **0 B**, least-squares private-byte slope over tokens 9..128 **-258.27 KiB/token**.

### DETERM-3 — run-to-run logit bit-identity baseline

Two runs of the same binary, same device, same explicit token IDs, proven by recorded SHA-256 to have loaded the same corelib DLL. Per `DETERM-3` the routes are reported separately and never averaged: append and re-prefill drive different row extents through the same LM-head shape.

| route | runs (n) | step bit-identity | run bit-identity | max abs diff: max / median | nonzero runs | ≥ 20 runs? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| append | 41 | 97.42% (n = 697) | 95.12% (n = 41) | 49.25 / 0.0 | 2 | yes |
| reprefill | 41 | 100.00% (n = 697) | 100.00% (n = 41) | 0.0 / 0.0 | 0 | yes |

**This window is not a settled baseline.** The figures above are what has been measured; the problems listed below say why they cannot yet be cited as the floor `DETERM-2`'s "degrades from the recorded baseline" clause needs.

Observed maximum absolute difference, `append`: `49.2148` × 1, `49.25` × 1.

> **2 run(s) in this window breached a `DETERM-2` HARD GATE.** They are counted in the rates above -- dropping them would bias the figures upward, and silently -- and they are not a wobble within the recorded tolerance. Read the run records before citing anything here as a settled baseline.

The two routes' observed rates differ, but **the n does not support calling the rate route-dependent** (Fisher exact p = 0.494 against 0.05). They are still reported separately, per `DETERM-3`, because pooling them would hide a difference that a larger campaign might resolve — not because this one resolved it.

- 64 record(s) carry no FastFlow harness SHA-256 and were EXCLUDED from the pooled figures: they cannot be shown to describe the same binary as the rest. They remain committed as evidence and are listed under `excluded_sources`.
- route 'append': 2 run(s) recorded a DETERM-2 gate failure. They ARE counted in the rate above -- dropping them would bias it upward -- but a baseline should not be declared over a window containing a hard-gate failure.

<!-- END phi4-aie4-baseline -->
