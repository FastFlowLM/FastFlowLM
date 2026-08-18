---
layout: docs
title: LLaMA 3.x
parent: Benchmarks
nav_order: 1
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance of LLaMA 3.x on NPU with FastFlowLM (FLM).

> **Note:** 
> - Results are based on FastFlowLM v0.9.30.  
> - Under FLM's default NPU power mode (Performance)   
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models.   

---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/llama3_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/llama3_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW** | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** | **64k** | **128k** |
|------------------|--------------|-------:|-------:|-------:|-------:|--------:|--------:|--------:|---------:|
| **LLaMA 3.2 1B** | NPU (FLM)    |64.5	|	62.2	|	58.9	|	53.9	|	45.5	|	35.0	|	24.1	|	13.6	|
| **LLaMA 3.2 3B** | NPU (FLM)    | 26.3	| 25.5	| 24.1	| 21.7	| 18.0	| 13.6	| 9.0   | OOM      |
| **LLaMA 3.1 8B** | NPU (FLM)    | 12.8   |	12.6   |	12.2   |	11.5   |	10.2   |	8.5   | OOM     | OOM      |

<!-- | **Model**        | **Hardware** | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** | **64k** | **128k** | **Hardware** | **Model**        |
|------------------|--------------|-------:|-------:|-------:|-------:|--------:|--------:|--------:|---------:|--------------|------------------|
| **LLaMA 3.2 1B** | NPU (FLM)    | 41.5   | 40.6   | 38.1   | 33.2   | 25.6    | 18.6    | 12.2    | 8.9      | NPU (FLM)    | **LLaMA 3.2 1B** |
|                  | NPU (RAI)    | 18.6   | 14.9   | *NA*   | *NA*   | *NA*    | *NA*    | *NA*    | *NA*     | NPU (RAI)    |                  |
|                  | iGPU         | 28.7   | 19.0   | 10.9   | 6.0    | 3.2     | 1.6     | 0.8     | OOM      | iGPU         |                  |
|                  | CPU          | 54.6   | 52.6   | 42.3   | 34.1   | 24.4    | 14.8    | 8.4     | OOM      | CPU          |                  |

| **Model**        | **Hardware** | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** | **64k** | **128k** | **Hardware** | **Model**        |
|------------------|--------------|-------:|-------:|-------:|-------:|--------:|--------:|--------:|---------:|--------------|------------------|
| **LLaMA 3.2 3B** | NPU (FLM)    | 18.3   | 17.8   | 15.9   | 13.6   | 10.5    | 7.3     | 6.3     | OOM      | NPU (FLM)    | **LLaMA 3.2 3B** |
|                  | NPU (RAI)    | 9.0    | 6.1    | *NA*   | *NA*   | *NA*    | *NA*    | *NA*    | *NA*     | NPU (RAI)    |                  |
|                  | iGPU         | 23.2   | 18.8   | 14.0   | 9.2    | 5.5     | 3.0     | OOM     | OOM      | iGPU         |                  |
|                  | CPU          | 22.6   | 21.3   | 17.5   | 14.1   | 9.4     | 6.1     | OOM     | OOM      | CPU          |                  |

| **Model**        | **Hardware** | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** | **64k** | **128k** | **Hardware** | **Model**        |
|------------------|--------------|-------:|-------:|-------:|-------:|--------:|--------:|--------:|---------:|--------------|------------------|
| **LLaMA 3.1 8B** | NPU (FLM)    | 9.1    | 9.0    | 8.3    | 7.5    | 6.2     | 4.6     | OOM     | OOM      | NPU (FLM)    | **LLaMA 3.1 8B** |
|                  | NPU (RAI)    | 6.3    | 4.6    | *NA*   | *NA*   | *NA*    | *NA*    | *NA*    | *NA*     | NPU (RAI)    |                  |
|                  | iGPU         | 11.3   | 9.9    | 7.7    | 5.4    | 3.4     | OOM     | OOM     | OOM      | iGPU         |                  |
|                  | CPU          | 10.3   | 7.7    | 7.6    | 6.7    | 5.8     | OOM     | OOM     | OOM      | CPU          |                  | -->
<!-- > **Note:** 
- The official release of Ryzen™ AI Software limits context length to 2,048 tokens, thus "*NA*" is used in the table (NPU-only mode). 
- The hybrid mode of Ryzen™ AI Software uses iGPU for decoding. Its performance is similar to iGPU (LM Studio). Also, it limits context length to 2,048, thus, we did not include hybrid mode for comparison.  -->
> OOM: Out Of Memory  
> Only <50% system DRAM can be accessed by NPU  
> On systems with more than 32 GB DRAM, longer context lengths are supported. FLM supports the full context length available for each model.  

---
<!-- 
### 🔋 Average Power Consumption (Watts) during decoding

| **Method**         | **CPU** | **NPU** | **iGPU** | **Total Power (W)** | **Efficiency Gain** |
|--------------------|--------:|--------:|--------:|---------------------:|----------------------:|
| NPU (FLM)          | 0.5     | 1.6      | 0         | **2.1**           | –                 |
| NPU (RAI)          | 11    | 2.0       | 0          | 13                 | 6.1×                |
| iGPU               | 11.3    | 0       | 9.00       | 20.3               | 9.7×               |
| CPU                | 18      | 0       | 0          | 18.0               | 8.6×             |

--- -->

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **LLaMA 3.2 1B** | NPU (FLM)    |1686	|2136	|2339	|2212	|1706	|1157|
| **LLaMA 3.2 3B**  | NPU (FLM)    | 766	|910	|991	|933	|721	|500|
| **LLaMA 3.1 8B**   | NPU (FLM)    | 403|	472|	495|	467|	381|	281|
