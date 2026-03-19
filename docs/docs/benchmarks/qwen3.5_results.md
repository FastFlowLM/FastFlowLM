---
layout: docs
title: Qwen 3.5
parent: Benchmarks
nav_order: 7
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance of Qwen 3.5 on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.36.  
> - Under FLM's default NPU power mode (Performance)    
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models.   

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/qwen35_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/qwen35_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Qwen 3.5 4B**    | NPU (FLM)    | 15.0 | 14.6 | 14.2 | 13.3 | 11.8 | 9.6| 

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Qwen 3.5 4B**    | NPU (FLM)    | 378 | 440 | 479 | 493 | 487 | 450|

---

### 🚀 Prefill TTFT with Image Input (Seconds)

Prefill time-to-first-token (TTFT) for Qwen3.5-4B on NPU (FastFlowLM) with different image resolutions.

**Mid Resolution Images:**

| Model        | HW  | 720p (1280×720) | 1080p (1920×1080) | 
|--------------|-----------|----------------:|------------------:|
| Qwen3.5-4B  | NPU (FLM) |            3.3 |               7.4 |


**High Resolution Images:**

| Model        | HW  | 2K (2560×1440) | 4K (3840×2160) |
|--------------|-----------|---------------:|---------------:|
| Qwen3.5-4B  | NPU (FLM) |           13.7 |             41.2 |

> This test uses a short prompt: “Describe this image.”