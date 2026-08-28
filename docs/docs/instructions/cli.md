---
layout: docs
title: System Command and CLI Mode
nav_order: 1
parent: Instructions
---

# ⚡ CLI Mode

FLM CLI mode offers a familiar terminal-based interactive experience, fully offline and accelerated exclusively on AMD NPUs. Here are detailed descriptions of commands and setup for CLI mode usage. It includes:

- **[🔧 Pre-Run PowerShell Commands (System)](#-pre-run-powershell-commands)**
- **[💻 Commands Inside CLI Mode](#-commands-inside-cli-mode)**
- **[🗂️ Others](#️-others)**

---

## 🔧 Pre-Run PowerShell Commands (System)

### 🖥️ System Compatibility Check

Verify that your hardware meets the necessary requirements to run FastFlowLM:

```shell
flm validate
```

Output the validation results as a JSON object:

```shell
flm validate --json
```

---

### 🆘 Show Help

```shell
flm help
```

---

### 🚀 Run a Model

Run a model interactively from the terminal:

```shell
flm run llama3.2:1b
```

> `flm` is short for FastFlowLM. If the model isn't available locally, it will be downloaded automatically. This launches FastFlowLM in CLI mode.

> **Linux note:** `flm validate` checks the kernel DRM device, while `flm run` opens the NPU through XRT. If validation succeeds but `flm run` fails with `No such device with index '0'`, confirm XRT can see the NPU:
> ```shell
> xrt-smi examine
> ```
> On Arch Linux, install `xrt-plugin-amdxdna` in addition to `xrt` and the `amdxdna` driver. See the [Linux install guide](/docs/install_lin/) for the full driver and firmware checklist.

---

### ⬇️ Pull a Model (Download Only)

Download a model from HuggingFace without launching it:

```shell
flm pull llama3.2:3b
```

This code forces a re-download of the model, overwriting the current version.

```shell
flm pull llama3.2:3b --force
```

> ⚠️ Use `--force` **only if the model file is corrupted** (e.g., incomplete download). Proceed with caution.

#### 📁 Default Model Storage Location

| Platform | Default Path |
|----------|-------------|
| Windows  | `C:\Users\<USER>\.flm\models` |
| Linux    | `~/.config/flm/models` |

#### 🔧 Changing the Model Storage Path

You can override the default location by setting the `FLM_MODEL_PATH` environment variable.

**Windows** — Update the existing system environment variable:
1. Open **Start** and search for **"Edit the system environment variables"**.
2. Click **Environment Variables…**.
3. Under **System variables**, find `FLM_MODEL_PATH`, select it, and click **Edit…**.
4. Update the value to your desired path (e.g., `D:\models\flm`).
5. Click **OK** and restart any open terminals for the change to take effect.

**Linux** — Set temporarily for the current shell session:
```shell
export FLM_MODEL_PATH="/your/custom/path"
```

To make the change permanent, add the line above to your `~/.bashrc`, then reload it:
```shell
echo 'export FLM_MODEL_PATH="/your/custom/path"' >> ~/.bashrc
source ~/.bashrc
```

---

### 📦 List Supported and Downloaded Models

Display all available models and locally downloaded models:

```shell
flm list
```

Output as JSON:

```shell
flm list --json
```


Filters flag:

```shell
# Show everything
flm list --filter all

# Only models already installed
flm list --filter installed

# Only models not yet installed
flm list --filter not-installed
```

Quiet mode:

```shell
# Default view (pretty, with icons)
flm list

# Quiet view (no emoji / minimal)
flm list --quiet

# Show everything
flm list --filter all --quiet

# Only models already installed
flm list --filter installed --quiet

# Only models not yet installed
flm list --filter not-installed --quiet
```

---

### ❌ Remove a Downloaded Model

Delete a model from local storage:

```shell
flm remove llama3.2:3b
```

### ✅ Check a Downloaded Model

Verify the file hashes for a downloaded model:

```shell
flm check llama3.2:3b
```

---

### 🚀 Start Server Mode (Local)

Launch FastFlowLM as a local REST API server (also supports the OpenAI API):

```shell
flm serve llama3.2:1b
```

---

### 🔌 Show Server Port 

Show current FLM port (default) in PowerShell:  
  
```shell
flm port
```

---

### ⚡ NPU Power Mode

By default, **FLM runs in `performance` NPU power mode**. You can switch to other NPU power modes (`powersaver`, `balanced`, or `turbo`) using the `--pmode` flag:

For **CLI mode**:
```shell
flm run gemma3:4b --pmode balanced
```

For **Server mode**:
```shell
flm serve gemma3:4b --pmode balanced
```

---

### 📏 Set Context Length at Launch

The default context length for each model can be found [here](https://fastflowlm.com/docs/models/).   

Set the context length with `--ctx-len` (or `-c`).  

In PowerShell, run:

For **CLI mode**:
```shell
flm run llama3.2:1b --ctx-len 8192
```

For **Server mode**:
```shell
flm serve llama3.2:1b --ctx-len 8192
```

> - Internally, FLM enforces a minimum context length of 512. If you specify a smaller value, it will automatically be adjusted up to 512.  
> - If you enter a context length that is not a power of 2, FLM automatically rounds it up to the nearest power of 2. For example: input `8000` → adjusted to `8192`.

---

### 🖧 Set Server Port at Launch

Set a custom port at launch:

  ```shell
  flm serve llama3.2:1b --port 8000
  flm serve llama3.2:1b -p 8000
  ```

> ⚠️ `--port` (`-p`) only affects the **current run**; it won’t change the default port.

---

### 🛠️ Set Host at Launch

Specify a custom host address when starting the server:

```powershell
flm serve llama3.2:1b --host 127.0.0.1
```

⚠️ Note: --host applies only to the current session. It does not modify the default host configuration (default: `127.0.0.1`).

---

### 🌐 Cross-Origin Resource Sharing (CORS)

CORS lets browser apps hosted on a different origin call your FLM server safely.

- Enable CORS

```shell
flm serve --cors 1
```
- Disable CORS

```shell
flm serve --cors 0
```

> ⚠️ **Default:** CORS is **enabled**.  
> 🔒 **Security tip:** Disable CORS (or restrict at your proxy) if your server is exposed beyond localhost (127.0.0.1).

---

### ⏸️ Preemption

Preemption allows high-priority tasks to interrupt ongoing NPU jobs, improving responsiveness for critical workloads. To enable preemption:

For **CLI mode**:
```shell
flm run llama3.2:1b --preemption 1
```

For **Server mode**:
```shell
flm serve llama3.2:1b --preemption 1
```

> ⚠️ Note: Preemption is for **engineering testing/optimization** only. It requires a special driver + toolkit and is **not for public use**.


---

### 🧩 Change Prefill Chunk Size at Launch

The `--prefill-chunk-len` flag controls how many tokens are processed per chunk during the prefill phase of inference (default: 4096).

For **CLI mode**:

```shell
flm run llama3.2:1b --prefill-chunk-len 8192
```

For **Server mode**:

```shell
flm serve llama3.2:1b --prefill-chunk-len 8192
```

---

### 🎙️ ASR (Automatic Speech Recognition)

**Requirement:** The ASR model (e.g., `whisper-large-v3-turbo`) must run **with an LLM loaded concurrently**. Enabling `--asr 1` starts Whisper in the background **while** your chosen LLM loads.

#### CLI mode
```shell
flm run gemma3:4b --asr 1  # Load Whisper (whisper-large-v3-turbo) in the background and load the LLM (gemma3:4b) concurrently.
```

#### Server mode
```shell
flm serve gemma3:4b --asr 1  # Background-load Whisper and initialize the LLM (gemma3:4b) concurrently.
```

> **Note:** ASR alone isn’t supported—an LLM must be present for end-to-end voice→text→LLM workflows.

See the ASR guide [here](https://fastflowlm.com/docs/models/whisper/)

---

## 💻 Commands Inside CLI Mode

Once inside the CLI, use the following commands. System commands always start with `/` (e.g., `/help`).

---

### 🆘 Help

```text
/?
```

> Displays all available interactive system commands. Highly recommended for first-time users.

---

### 🪪 Model Info

```text
/show
```

> View model architecture, size, **max context size (Adjustable – see bottom)** and more.

---

### 🔄 Change Model

```text
/load [model_name]
```

> Unload the current model and load a new one. KV cache will be cleared.

---

### 💾 Save Conversation

```text
/save
```

> Save the current conversation history to disk.

---

### 🧹 Clear Memory

```text
/clear
```

> Clear the KV cache (model memory) for a fresh start.

---

### 📊 Show Runtime Stats

```text
/status
```

> Display runtime statistics like token count, throughput, etc.

---

### 🕰️ Show History

```text
/history
```

> Review the current session's conversation history.

---

### 🔍 Toggle Verbose Mode

```text
/verbose
```

> Enable detailed performance metrics per turn. Run again to disable.

---

### 📦 List Models

Display all available models and locally downloaded models:

```text
/list
```

---

### 👋 Quit CLI Mode

```text
/bye
```

> Exit the CLI.

---

### 🧠 Think Mode Toggle

Type `/think` to toggle Think Mode on or off interactively in the CLI.

> 💡 **Note**: This feature is only supported on certain models, such as **Qwen3**.

---

### 📂 Load a Local Text File in CLI Mode

Use any file that can be opened in Notepad (like `.txt`, `.json`, `.csv`, etc.).

Format (in CLI mode):

```shell
/input "<file_path>" prompt
```

Example:

```shell
/input "C:\Users\Public\Desktop\alice_in_wonderland.txt" Summarize it into 200 words
```

> Notes:

* Use quotes **only around the file path**
* **No quotes** around the prompt
* File must be plain text (readable in Notepad)

👉 [Download a sample prompt (around 40k tokens)](https://github.com/ROCm/FastFlowLM/blob/main/assets/alice_in_wonderland.txt)  

> ⚠️ **Caution:** a model’s supported context length is limited by available DRAM capacity. For example, with **32 GB** of DRAM, **LLaMA 3.1:8B** cannot run beyond a **32K** context length. For the full **128K** context, we recommend a larger memory system.

If DRAM is heavily used by other programs while running **FastFlowLM**, you may encounter errors due to insufficient memory, such as:

```error
[XRT] ERROR: Failed to submit the command to the hw queue (0xc01e0200):
Even after the video memory manager split the DMA buffer, the video memory manager
could not page-in all of the required allocations into video memory at the same time.
The device is unable to continue.
```

> 🤔 Interested in checking the DRAM usage?

<!-- **Method 1 – Task Manager (Quick View)**   -->
1. Press **Ctrl + Shift + Esc** (or **Ctrl + Alt + Del** and select **Task Manager**).  
2. Go to the **Performance** tab.  
3. Click **Memory** to see total, used, and available DRAM, as well as usage percentage.  

<!-- **Method 2 – Resource Monitor (Detailed View)**  
1. Press **Windows + R**.  
2. Type:  ```resmon```
3. Press **Enter**.  
4. Go to the **Memory** tab to view detailed DRAM usage and a per-process breakdown. -->

---

### 🌄 Loading Images in CLI Mode (for VLMs only, e.g. gemma3:4b)

Supports **.png** and **.jpg** formats.  

```shell
/input "<image_path>" prompt
```

Example:

```shell
/input "C:\Users\Public\Desktop\cat.jpg" describe this image
```

> Notes:

* Make sure the model you are using is a **vision model (VLM)** (e.g., gemma3:4b) 
* Put quotes **only around the file path**  
* Do **not** use quotes around the prompt  
* Image must be in **.jpg** or **.png** format  

---

### ⚙️ Set Variables

```text
/set
```

> Customize decoding parameters like `top_k`, `top_p`, `temperature`, `context length (max)`, `generate limit`, etc.

> ⚠️ **Note:** Providing invalid or extreme hyperparameter values may cause inference errors.
> `generate limit` sets an upper limit on the number of tokens that can be generated for each response. Example:

```text
/set gen-lim 128
```

---

## 📊 Benchmarking Tool

Use the FLM benchmarking tool to measure a model's performance across different context lengths.

Each benchmark tests context lengths from `1k` to `32k`, running `2` iterations at each length.

```shell
flm bench llama3.2:1b
```

Change the iteration times by `bench-iterations`:

```shell
flm bench llama3.2:1b --bench-iterations 4
```

FLM prints the results in your terminal and also saves them as a CSV file in the current folder for later reference.

```text
[FLM]  === Benchmark Results ===

 Context Length |              TTFT (s) |      Prefill Speed (tok/s) |     Decoding Speed (tok/s)
----------------------------------------------------------------------------------------------------
             1k |       0.815 ±   0.012 |        1233.57 ±     17.91 |       61.39 ±      0.46
             2k |       1.144 ±   0.035 |        1728.16 ±     55.68 |       59.04 ±      0.36
             4k |       1.955 ±   0.009 |        2001.55 ±      9.22 |       54.32 ±      0.47
             8k |       3.821 ±   0.010 |        2037.62 ±      5.40 |       46.89 ±      0.19
            16k |       9.144 ±   0.011 |        1698.20 ±      2.08 |       37.20 ±      0.14
            32k |      26.346 ±   0.037 |        1177.16 ±      1.53 |       26.38 ±      0.07
----------------------------------------------------------------------------------------------------
```

---

## 🗂️ Others

### 🛠 Change Default Context Length (max)

You can find more information about available models here:  

```
C:\Program Files\flm\model_list.json
```

You can also change the `default_context_length` setting.

> ⚠️ **Note:** Be cautious! The system reserves DRAM space based on the context length you set.  
> Setting a longer default context length may cause errors on systems with smaller DRAM.
> Also, each model has its own context length limit (examples below).  
> - **qwen3-tk:4b** → up to **256k** tokens  
> - **gemma3:4b** → up to **128k** tokens
> - **gemma3:1b** → up to **32k** tokens
> - **llama3.x** → up to **128k** tokens

