---
layout: docs
title: Qwen 3.6
parent: Benchmarks
nav_order: 10
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance of Qwen 3.6 on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.45.  
> - Under FLM's default NPU power mode (Performance)    
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models.   

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 96 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/qwen36_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/qwen36_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Qwen3.6-35B-A3B**    | NPU (FLM)    | 17.48 | 17.16 | 16.59 | 15.6 | 13.76 | 11.19 | 

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Qwen3.6-35B-A3B**    | NPU (FLM)    | 102.45 | 144.8 | 202.45 | 245.99 | 277.09 | 280.97 | 

---

### 🚀 Prefill TTFT with Image Input (Seconds)

Prefill time-to-first-token (TTFT) for Qwen3.6-35B-A3B on NPU (FastFlowLM) with different image resolutions.

**Mid Resolution Images:**

| Model        | HW  | 720p (1280×720) | 1080p (1920×1080) | 
|--------------|-----------|----------------:|------------------:|
| **Qwen3.6-35B-A3B**  | NPU (FLM) |       11.3      |      17.8      |


**High Resolution Images:**

| Model        | HW  | 2K (2560×1440) | 4K (3840×2160) |
|--------------|-----------|---------------:|---------------:|
| **Qwen3.6-35B-A3B**  | NPU (FLM) |      28.5       |    59.2            |


> This test uses a short prompt: “Describe this image.”
