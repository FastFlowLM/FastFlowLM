---
layout: docs
title: Qwen2.5
parent: Benchmarks
nav_order: 8
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.33.
> - Under FLM's default NPU power mode (Performance)   
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models. 

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/qwen25_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/qwen25_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Qwen2.5-3B-Instruct**  | NPU (FLM)    | 23.5	| 22.5	| 19.8	| 16.8	| 12.5	| 8.4|
| **Qwen2.5-VL-3B-Instruct**  | NPU (FLM)    | 23.5	| 22.5	| 19.8	| 16.8	| 12.5	| 8.4|

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Qwen2.5-3B-Instruct**  | NPU (FLM)    | 660	| 809	| 899	| 891	| 741	| 532 | 
| **Qwen2.5-VL-3B-Instruct**  | NPU (FLM)    | 660	| 809	| 899	| 891	| 741	| 532 | 

---

### 🚀 Prefill TTFT with Image Input (Seconds)

Prefill time-to-first-token (TTFT) for Qwen2.5-VL-3B-Instruct on NPU (FastFlowLM) with different image resolutions.

**Mid Resolution Images:**

| Model        | HW  | 720p (1280×720) | 1080p (1920×1080) | 
|--------------|-----------|----------------:|------------------:|
| Qwen2.5-VL-3B-Instruct  | NPU (FLM) |            4.3 |               7.9 |


**High Resolution Images:**

| Model        | HW  | 2K (2560×1440) | 4K (3840×2160) |
|--------------|-----------|---------------:|---------------:|
| Qwen2.5-VL-3B-Instruct  | NPU (FLM) |           13.3 |             36.4 |

> This test uses a short prompt: “Describe this image.”