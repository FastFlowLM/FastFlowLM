<p align="center">
  <a href="https://www.fastflowlm.com" target="_blank">
    <img src="assets/logo.png" alt="FastFlowLM Logo" width="200"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NPU-Optimized-red" />
</p>

## ⚡ FastFlowLM (FLM) — Unlock Ryzen™ AI NPUs

Run large language models — now with **Vision**, **Audio**, **Embedding** and **MoE** support — on **AMD Ryzen™ AI NPUs** in minutes.  
**No GPU required. Faster and over 10× more power-efficient. Supports context lengths up to 256k tokens. Ultra-Lightweight (16 MB). Installs within 20 seconds.**

📦 **The only out-of-box, NPU-first runtime built exclusively for Ryzen™ AI.**  
🤝 **Think Ollama — but deeply optimized for NPUs.**  
✨ **From Idle Silicon to Instant Power — FastFlowLM Makes Ryzen™ AI Shine.**

> FastFlowLM (FLM) supports all Ryzen™ AI Series chips with XDNA2 NPUs (Strix, Strix Halo, Kraken, and Gorgon Point).

---

## 🔗 Quick Links

  🔽 **[Download](https://github.com/FastFlowLM/FastFlowLM/releases/latest/download/flm-setup.exe)** | 📊 **[Benchmarks](https://fastflowlm.com/docs/benchmarks/)** | 📦 **[Model List](https://fastflowlm.com/docs/models/)**  

  🐧 **[Linux Getting Started Guide](https://fastflowlm.com/docs/install_lin/)**

  📖 **[Docs](https://fastflowlm.com/docs)** | 📺 **[Demos](https://www.youtube.com/@FastFlowLM-YT/playlists)** | 🧪 **[Test Drive](https://fastflowlm.com/docs/#-test-drive-remote-demo)** | 💬 **[Discord](https://discord.gg/z24t23HsHF)** 

---

## 🚀 Quick Start

A packaged FLM Windows installer is available here: [**flm-setup.exe**](https://github.com/FastFlowLM/FastFlowLM/releases/latest/download/flm-setup.exe). For more details, see the [release notes](https://github.com/FastFlowLM/FastFlowLM/releases/).

📺 [**Watch the quick start video (Windows)**](https://www.youtube.com/watch?v=mYOfDNkyBII)

> [!IMPORTANT]  
> ⚠️ Ensure NPU driver version is **>= 32.0.203.304** (`.304` is the minimum requirement but `.311` is recommended; check via Task Manager→Performance→NPU or Device Manager).  
> ⚙️ **Tip:**
>   * **RECOMMENDED**: Try running **Windows Update** or **[Driver Download](https://www.amd.com/en/support)**.
>   * **[Official AMD Install Doc](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)** *(AMD account required)*.
>   * **[Unofficial forum downloads](https://www.elevenforum.com/t/drivers-amd-npu-ryzen-8xxx-9xxx-apu.24220/)** *(CAUTION, we do not hold responsible for what you download here)*.

After installation, open **PowerShell** (`Win + X → I`). To run a model in terminal (**CLI Mode**):
```powershell
flm run llama3.2:1b
```
> **Notes:**
> - Internet access to HuggingFace is required to download the optimized model kernels.
> - Sometimes downloads from HuggingFace may get corrupted. If this happens, run `flm pull <model_tag> --force` (e.g. `flm pull llama3.2:1b --force`) to re-download and fix them.
> - By default, models are stored in:
>   - **Windows**: `C:\Users\<USER>\Documents\flm\models\`
>   - **Linux**: `~/.config/flm/`
> - During installation on Windows, you can select a different base folder (e.g., if you choose `C:\Users\<USER>\flm`, models will be saved under `C:\Users\<USER>\flm\models\`).
> - On Linux, you can override the default location by setting the `FLM_MODEL_PATH` environment variable.
> - ⚠️ If HuggingFace is not accessible in your region, manually download the model ([check this issue](https://github.com/FastFlowLM/FastFlowLM/issues/2)) and place it in the chosen directory.   

🎉🚀 FastFlowLM (FLM) is ready — your NPU is unlocked and you can start chatting with models right away!

Open **Task Manager** (`Ctrl + Shift + Esc`). Go to the **Performance** tab → click **NPU** to monitor usage.  

> **⚡ Quick Tips:**  
> - Use `/verbose` during a session to turn on performance reporting (toggle off with `/verbose` again).   
> - Type `/bye` to exit a conversation.  
> - Run `flm list` in PowerShell to show all available models.  

To start the local server (**Server Mode**):
```powershell
flm serve llama3.2:1b
```
> The model tag (e.g., `llama3.2:1b`) sets the initial model, which is optional. If another model is requested, FastFlowLM will automatically switch to it. Local server is on port 52625 (default).  

**[![FastFlowLM Docs](https://img.shields.io/badge/FastFlowLM-Detailed%20Instructions-red?style=flat&logo=readthedocs)](https://fastflowlm.com/docs/instructions/)**

---

## 📰 In the News

- 10/01/2025 🎉 FLM was integrated into AMD's **[Lemonade Server](https://lemonade-server.ai/)** 🍋. Watch this **[short demo](https://www.youtube.com/watch?v=w0Tb3h4WUnE)** about using FLM in Lemonade.

---

## 🧠 Local AI on NPU

FLM makes it easy to run cutting-edge **LLMs** (and now **VLMs**) locally with:
- ⚡ Fast and low power
- 🧰 Simple CLI and API (REST and OpenAI API)
- 🔐 Fully private and offline

No model rewrites, no tuning — it just works.

---

## ✅ Highlights

- **Runs fully on AMD Ryzen™ AI NPU** — no GPU or CPU load
- **Lightweight runtime (16 MB)** — installs within **20 seconds**, easy to integrate    
- **Developer-first flow** — like Ollama, but optimized for NPU  
- **Support for long context windows** — up to 256k tokens (e.g., Qwen3-4B-Thinking-2507)  
- **No low-level tuning required** — You focus on your app, we handle the rest

---

## 📄 License

- All orchestration code and CLI tools are open-source under the [MIT License](./LICENSE_RUNTIME.txt).  
- NPU-accelerated kernels are **proprietary binaries**, free for **commercial use up to USD 10 million in annual company revenue**.
- Companies exceeding this threshold (**USD 10 million**) must obtain a commercial license. See [LICENSE_BINARY.txt](./LICENSE_BINARY.txt) and [TERMS.md](./TERMS.md) for full details.
- **Free-tier users:** Please acknowledge FastFlowLM in your README/project page (or product) as follows:
  ```
  Powered by [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM)
  ```
For commercial licensing inquiries, email us: info@fastflowlm.com

---

💬 Have **feedback/issues** or want **early access** to our new releases? [Open an issue](https://github.com/fastflowlm/fastflowlm/issues/new) or [Join our Discord community](https://discord.gg/z24t23HsHF)

---

## 🙏 Acknowledgements

- Powered by the advanced **AMD Ryzen™ AI NPU architecture**
- Inspired by the widely adopted [llama.cpp](https://github.com/ggml-org/llama.cpp) and [Ollama](https://github.com/ollama/ollama)
- Tokenization accelerated with [MLC-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)
- Chat formatting via [Google/minja](https://github.com/google/minja)
- Low-level kernels optimized using the powerful [IRON](https://github.com/amd/iron)+[AIE-MLIR](https://github.com/Xilinx/mlir-aie)

---

## 🛠️ Building from Source

For developers who want to build FastFlowLM from source, we provide CMake presets for a convenient and consistent build experience.

### Prerequisites

- Git
- CMake (version 3.22 or higher)
- A C++20 compatible compiler (e.g., GCC, Clang, MSVC)
- Ninja (recommended)

### Build Instructions

More details on the exact procedure, with dependencies to be installed, for linux can be found in [linux-getting-started.md](docs/linux-getting-started.md).

1.  **Clone the repository:**

    ```bash
    git clone --recursive https://github.com/FastFlowLM/FastFlowLM.git
    cd FastFlowLM/src
    ```

2.  **Configure CMake using presets:**

    -   **For Linux:**

        ```bash
        cmake --preset linux-default
        ```

        This will configure the build to install to `/opt/fastflowlm`.

    -   **For Windows (in a developer command prompt):**

        ```bash
        cmake --preset windows-default
        ```

3.  **Build the project:**

    ```bash
    cmake --build build
    ```

4.  **Install the project (optional):**

    -   **For Linux:**

        ```bash
        sudo cmake --install build
        ```

    -   **For Windows (with administrator privileges):**

        ```bash
        cmake --install build
        ```

