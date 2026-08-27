---
layout: docs
title: SmolVLA
parent: Benchmarks
nav_order: 12
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v1.0.0.
> - Under FLM's default NPU power mode (Performance)  
> - Newer versions may deliver improved performance.
> - SmolVLA is a Vision-Language-Action (robotics policy) model. It is measured as end-to-end latency per inference (camera images + instruction → action chunk), not in tokens per second.

---

### **Test System:** 

AMD Ryzen™ AI 9 370 (Strix Point) with 32 GB DRAM; performance is comparable to other Strix Point and Strix Halo systems.

---

### 🚀 Inference Latency (ms per inference, with different camera input counts)

| **Model**        | **HW**       | **1 image** | **2 images** | **3 images** |
|------------------|--------------------|--------:|--------:|--------:|
| **SmolVLA**  | NPU (FLM)    | 298	| 363	| 430 |

---

### 📦 Package and Model Download

SmolVLA ships as a self-contained package. The model can be downloaded here:

- HuggingFace: [FastFlowLM/smolvla-NPU2](https://huggingface.co/FastFlowLM/smolvla-NPU2)
- ModelScope: [amd/smolvla](https://modelscope.cn/models/amd/smolvla)

For the model card and usage notes, see [SmolVLA](/docs/models/smolvla/).

---
