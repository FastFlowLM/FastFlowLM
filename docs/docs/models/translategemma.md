---
layout: docs
title: TranslateGemma
nav_order: 12
parent: Models
---

## 🧩 Model Card: [translategemma:4b](https://huggingface.co/google/translategemma-4b-it)

- **Type:** Image-Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens 
- **Default Context Length:** 64k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run translategemma:4b
```

📖 Prompt Guide

**Prompt Format**

TranslateGemma expects a single **user message** with this structure:

    You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
    Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:


    {TEXT}

**Important**: There are two blank lines before the text to translate.

🦙Refer to the Ollama model library for detailed usage [examples](https://ollama.com/library/translategemma#:~:text=text%20to%20translate.-,Examples,-English%20to%20Spanish).

- [Research Blog](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/)
- [Technical Report](https://arxiv.org/pdf/2601.09012)