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
| FastFlow revision | `00332b995ce2da95b2bf0309f1d06615fc58d681` |
| measured (UTC) | `2026-09-02T13:12:13Z` |

### Model load and TTFT

| | |
| --- | --- |
| manifest map and mapping | 4.8 ms |
| weight pack/upload (161 objects) | 1600.1 ms |
| total model load | 12765.2 ms |
| cold TTFT (19 prompt tokens, row extent 64) | 3117.0 ms |
| warm TTFT, same Stream after `clear_context()` | 69.7 ms |

### Fresh prefill

| rows | padded rows | wall time | tokens/s |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 174.3 ms | 5.7 |
| 2 | 64 | 76.0 ms | 26.3 |
| 65 | 128 | 131.5 ms | 494.1 |
| 129 | 256 | 179.4 ms | 719.0 |
| 257 | 512 | 221.4 ms | 1,160.5 |
| 513 | 1024 | 375.2 ms | 1,367.1 |
| 1025 | 2048 | 727.1 ms | 1,409.7 |
| 2049 | 3072 | 1174.0 ms | 1,745.3 |
| 3073 | 4096 | 1697.1 ms | 1,810.7 |
| 4096 | 4096 | 1732.8 ms | 2,363.9 |

### Decode

| starting context | tokens | tokens/s | p50 | p95 | synchronizes per pass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 12.36 | 75.4 ms | 92.1 ms | 129 |
| 512 | 128 | 12.21 | 77.9 ms | 96.0 ms | 129 |
| 2048 | 128 | 11.61 | 78.6 ms | 112.7 ms | 129 |

### Continuation routes

| history rows | suffix | route | samples | p50 | p95 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 512 | 1 | append | 5 | 44.4 ms | 50.0 ms |
| 512 | 1 | reprefill | 5 | 380.1 ms | 399.7 ms |
| 512 | 2 | append | 5 | 78.9 ms | 88.3 ms |
| 512 | 2 | reprefill | 5 | 375.7 ms | 380.0 ms |
| 512 | 32 | append | 5 | 1350.4 ms | 1361.5 ms |
| 512 | 32 | reprefill | 5 | 374.9 ms | 375.4 ms |
| 512 | 128 | append | 5 | 5368.3 ms | 5538.9 ms |
| 512 | 128 | reprefill | 5 | 375.5 ms | 376.4 ms |
| 512 | 256 | append | 5 | 10714.6 ms | 10792.4 ms |
| 512 | 256 | reprefill | 5 | 318.9 ms | 362.5 ms |
| 2048 | 1 | append | 5 | 50.0 ms | 50.7 ms |
| 2048 | 1 | reprefill | 5 | 1166.5 ms | 1195.2 ms |
| 2048 | 2 | append | 5 | 90.2 ms | 99.8 ms |
| 2048 | 2 | reprefill | 5 | 1167.4 ms | 1180.5 ms |
| 2048 | 32 | append | 5 | 1414.7 ms | 1427.5 ms |
| 2048 | 32 | reprefill | 5 | 1172.5 ms | 1174.8 ms |
| 2048 | 128 | append | 5 | 9948.7 ms | 10379.6 ms |
| 2048 | 128 | reprefill | 5 | 1214.9 ms | 1231.8 ms |
| 2048 | 256 | append | 5 | 20393.9 ms | 20583.5 ms |
| 2048 | 256 | reprefill | 5 | 1214.7 ms | 1261.2 ms |

Longest suffix at which append beats re-prefill, per rendered history: `2048` → 2, `512` → 2. Prefix-monotonic: `True`.

### V scatter, memory and synchronization

| | |
| --- | --- |
| V cache reads per model step | 32 |
| per-head V writes per model step | 256 |
| V bytes transferred | 13.91 GiB |
| V scatter wall time | 1409.5 ms |
| FP16 embedding | 1.14 GiB |
| KV cache | 512.00 MiB |
| corelib packed weights | 1.95 GiB |
| scratch and device tensors | 161.89 MiB |
| mapped ONNX source | 3.03 GiB |
| peak host private bytes | 7.64 GiB |
| peak host working set | 9.54 GiB |

128-token stability window: device tensors created after warmup **0**, weight objects **0**, net live corelib objects **0**, private-byte growth **0 B**, least-squares private-byte slope over tokens 9..128 **-157.08 KiB/token**.

### DETERM-3 — run-to-run logit bit-identity baseline

Two runs of the same binary, same device, same explicit token IDs, proven by recorded SHA-256 to have loaded the same corelib DLL. Per `DETERM-3` the routes are reported separately and never averaged: append and re-prefill drive different row extents through the same LM-head shape.

| route | runs (n) | step bit-identity | run bit-identity | max abs diff: max / median | nonzero runs | baseline? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| append | 33 | 96.97% (n = 561) | 96.97% (n = 33) | 48.34375 / 0.0 | 1 | yes |
| reprefill | 33 | 98.22% (n = 561) | 93.94% (n = 33) | 0.3125 / 0.0 | 2 | yes |

Observed maximum absolute difference, `append`: `48.3438` × 1.
Observed maximum absolute difference, `reprefill`: `0.25` × 1, `0.3125` × 1.

> **3 run(s) in this window breached a `DETERM-2` HARD GATE.** They are counted in the rates above -- dropping them would bias the figures upward, and silently -- and they are not a wobble within the recorded tolerance. Read the run records before citing anything here as a settled baseline.

**The rate is route-dependent.** The two routes are reported separately above rather than pooled.

- route 'append': 1 run(s) recorded a DETERM-2 gate failure. They ARE counted in the rate above -- dropping them would bias it upward -- but a baseline should not be declared over a window containing a hard-gate failure.
- route 'reprefill': 2 run(s) recorded a DETERM-2 gate failure. They ARE counted in the rate above -- dropping them would bias it upward -- but a baseline should not be declared over a window containing a hard-gate failure.

<!-- END phi4-aie4-baseline -->
