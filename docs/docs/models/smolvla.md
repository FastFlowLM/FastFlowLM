---
layout: docs
title: SmolVLA
nav_order: 13
parent: Models
---

## 🧩 Model Card: [SmolVLA](https://huggingface.co/lerobot/smolvla_base)

- **Type:** Vision-Language-Action
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- **Quantization:** bf16
- **Max Context Length:** 48 tokens  
- **Max Camera Input:** 3 images

📝 **Note:**

- SmolVLA is a robotics policy model that maps camera images and language instructions directly to robot actions — it does not run in FLM's standard CLI or Server chat modes.
- For detailed usage instructions, please refer to:
    - HuggingFace: [FastFlowLM/smolvla-NPU2](https://huggingface.co/FastFlowLM/smolvla-NPU2)
    - ModelScope: [amd/smolvla](https://modelscope.cn/models/amd/smolvla)
- For measured inference latency on Strix Point, see [SmolVLA benchmarks](/docs/benchmarks/smolvla_results/).

---