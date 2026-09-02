---
layout: docs
title: Gemma
nav_order: 4
parent: Models
---

## 🧩 Model Card: [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gemma3:1b
```

---

## 🧩 Model Card: [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

- **Type:** Image-Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 64k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gemma3:4b
```

📝 **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

---

## 🧩 Model Card: [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)

- **Type:** Any-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 64k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gemma4-it:e2b
```

🖼️ **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

🗣️ **Note:** In CLI mode, attach an audio with:

```shell
/input "file/to/audio.mp3" summarize this audio.
```

📝 **Note:** 

- In server mode, Gemma 4 supports multimodal input with text, images, and audio. See the [OpenAI API multimodal example](https://fastflowlm.com/docs/instructions/server/openapi/#%EF%B8%8F-example-multi-modal-input). 

- Change the visual token budget for images with the `image-max-tokens` parameter for different tasks. For more details, see the [Open WebUI custom parameters example](https://fastflowlm.com/docs/instructions/server/webui/#%EF%B8%8F-example-add-flm-custom-parameters).

---

## 🧩 Model Card: [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)

- **Type:** Any-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 64k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gemma4-it:e4b
```

🖼️ **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

🗣️ **Note:** In CLI mode, attach an audio with:

```shell
/input "file/to/audio.mp3" summarize this audio.
```

📝 **Note:** 

- In server mode, Gemma 4 supports multimodal input with text, images, and audio. See the [OpenAI API multimodal example](https://fastflowlm.com/docs/instructions/server/openapi/#%EF%B8%8F-example-multi-modal-input). 

- Change the visual token budget for images with the `image-max-tokens` parameter for different tasks. For more details, see the [Open WebUI custom parameters example](https://fastflowlm.com/docs/instructions/server/webui/#%EF%B8%8F-example-add-flm-custom-parameters).

---

## 🧩 Model Card: [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)

- **Type:** Any-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)
- **Quantization:** Q4_0
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 64k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run gemma4-it:12b
```

🖼️ **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

🗣️ **Note:** In CLI mode, attach an audio with:

```shell
/input "file/to/audio.mp3" summarize this audio.
```

📝 **Note:** 

- In server mode, Gemma 4 supports multimodal input with text, images, and audio. See the [OpenAI API multimodal example](https://fastflowlm.com/docs/instructions/server/openapi/#%EF%B8%8F-example-multi-modal-input). 

- Change the visual token budget for images with the `image-max-tokens` parameter for different tasks. For more details, see the [Open WebUI custom parameters example](https://fastflowlm.com/docs/instructions/server/webui/#%EF%B8%8F-example-add-flm-custom-parameters).