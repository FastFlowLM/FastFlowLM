---
layout: docs
title: gpt-oss
parent: Benchmarks
nav_order: 5
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.32.
> - Under FLM's default NPU power mode (Performance)  
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models. 

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/gpt-oss_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/gpt-oss_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**64k** | **128k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|
| **gpt-oss-20b**  | NPU (FLM)    | 18.2|	18.0|	17.4|	16.3|	14.5|	12.0|	8.7|	5.7|

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **gpt-oss-20b**  | NPU (FLM)    | 221|	326|	415|	477|	473|	391|
