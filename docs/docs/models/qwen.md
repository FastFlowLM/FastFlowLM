---
layout: docs
title: Qwen
nav_order: 3
parent: Models
---

## 🧩 Model Card: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

- **Type:** Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** No  
- **Base Model:** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:0.6b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

- **Type:** Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** No  
- **Base Model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:1.7b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

- **Type:** Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:4b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

- **Type:** Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 16k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3:8b
```

📝 **Note:**

- **CLI**: Type `/think` to toggle on/off interactively.  
- **Server Mode**: Set the `"think"` flag in the request payload.

---

## 🧩 Model Card: [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)

- **Type:** Text-to-Text
- **Think:** Yes
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- **Quantization:** Q4_1
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3-tk:4b
```

---

## 🧩 Model Card: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Quantization:** Q4_1
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3-it:4b
```

---

## 🧩 Model Card: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

- **Type:** Image-Text-to-Text
- **Think:** No
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3vl-it:4b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3vl-it:3b -r 1
```

```shell
flm serve qwen3vl-it:3b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.

---

## 🧩 Model Card: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen2.5-it:3b
```

---

## 🧩 Model Card: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

- **Type:** Image-Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- **Quantization:** Q4_1
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen2.5vl-it:3b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen2.5vl-it:3b -r 1
```

```shell
flm serve qwen2.5vl-it:3b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.

---

## 🧩 Model Card: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** No  
- **Base Model:** [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.5:0.8b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.5:0.8b -r 1
```

```shell
flm serve qwen3.5:0.8b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B#using-qwen35-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.

---

## 🧩 Model Card: [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.5:2b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.5:2b -r 1
```

```shell
flm serve qwen3.5:2b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B#using-qwen35-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.

---

## 🧩 Model Card: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.5:4b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.5:4b -r 1
```

```shell
flm serve qwen3.5:4b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px
- 4: height = 1440 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B#using-qwen35-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.

---

## 🧩 Model Card: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.5:9b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.5:9b -r 1
```

```shell
flm serve qwen3.5:9b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px 
- 4: height = 1440 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B#using-qwen35-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.

---

## 🧩 Model Card: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

- **Type:** Image-Text-to-Text
- **Think:** Toggleable
- **Tool Calling Support:** Yes  
- **Base Model:** [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Quantization:** Q4_K
- **Max Context Length:** 256k tokens  
- **Default Context Length:** 32k tokens ([change default](https://fastflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://fastflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with FastFlowLM in PowerShell:  

```shell
flm run qwen3.6-moe:35b-a3b
```

▶️ Image Resize Options

You can control image resizing when running or serving the model using the `--img-pre-resize` flag or simply `-r`:

```shell
flm run qwen3.6-moe:35b-a3b -r 1
```

```shell
flm run qwen3.6-moe:35b-a3b -r 1
```

The `-r` option determines the image's height:

- 0: original size 
- 1: height = 480 px 
- 2: height = 720 px (default)
- 3: height = 1080 px 
- 4: height = 1440 px 
                 
> Don't worry—if your image is already smaller than the setup, it keeps its original resolution! ✨

📝 **Note**

- Optimal sampling parameters for generation vary depending on the task. Check the [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B#using-qwen36-via-the-chat-completions-api) for details.
- Image understanding adapts to image size. Image TTFT can range from under 1 second to ~200 seconds depending on resolution. Use lower-resolution images (720p or below) unless high resolution is required (e.g. OCR on small text).
- Video understanding is not supported yet.