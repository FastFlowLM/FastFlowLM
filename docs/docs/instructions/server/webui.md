---
layout: docs
title: Open WebUI + FLM
nav_order: 2
parent: Local Server (Server Mode)
---

# 📑 Table of Contents

- **[🧩 Run Open WebUI with FastFlowLM](#-run-open-webui-with-fastflowlm)**
  - [ Option 1: Use Open WebUI Desktop](#option-1-use-open-webui-desktop)  
  - [ Option 2: Use Open WebUI in Docker](#option-2-use-open-webui-in-docker)  
- **[🧪 More Examples](#-more-examples)**
  - [ Example: Multi Models Comparison Enabled by FLM Queuing](#-example-multi-models-comparison-enabled-by-flm-queuing)  
  - [ Example: Agentic AI Web Search with FastFlowLM](#-example-agentic-ai-web-search-with-fastflowlm)  
  - [ Example: Local Private Database with RAG + FastFlowLM](#️-example-local-private-database-with-rag--fastflowlm)
  - [ Example: Add FLM Custom Parameters](#️-example-add-flm-custom-parameters)

---

# 🧩 Run Open WebUI with FastFlowLM

## Option 1: Use Open WebUI Desktop

Set up **Open WebUI Desktop** to talk to a local **FastFlowLM** instance on Windows.

---

### 📥 Step 1: Download Open WebUI Desktop
1. Go to the [releases page](https://github.com/open-webui/desktop/releases).
2. Download **`windows-latest-x64.exe`**.
3. Double-click the installer and complete setup.

---

### 🧑‍💻 Step 2: Create Your Open WebUI Account
Enter **Name**, **Email**, and **Password** to create a local account.

---

### 🔌 Step 3: Connect Open WebUI to FastFlowLM
1. In Open WebUI: **user icon → Admin Panel → Settings → Connections**.
2. **Deactivate** the Ollama API.
3. Under **Manage OpenAI API Connections**, click **+** and fill in:
   - **Name:** FastFlowLM
   - **Base URL:** `http://127.0.0.1:52625/v1`
   - **API Key:** `DUMMY` (any non-empty value)
4. Click **Save**.

> ⚠️ Disable `Title Generation`, `Follow Up Generation`, `Tags Generation`, `Retrieval Query Generation`, and `Web Search Query Generation` under **Settings->Interface** for best performance. 

---

### 🚀 Step 4: Serve FastFlowLM with a Model
Open **PowerShell** and start the server:

```shell
flm serve llama3.2:1b
```

You can now chat in Open WebUI using FastFlowLM.

> Note: Switching models may take time while the new model loads into memory.

---

## Option 2: Use Open WebUI in Docker

This guide walks you through using `docker-compose.yaml` to run Open WebUI connected to a local FastFlowLM instance on Windows.

---

### ✅ Prerequisites

1. [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
   - During installation, enable **WSL2 backend**
   - Reboot if prompted

2. [FastFlowLM](https://fastflowlm.com/docs/install/)

---

### 📁 Step 1: Create Project Folder

Open PowerShell and run:

```shell
mkdir open-webui && cd open-webui
```

This creates a clean workspace for your Docker setup.

---

### 📝 Step 2: Create `docker-compose.yaml`

Launch Notepad:

```shell
notepad docker-compose.yaml
```

Paste the following:

```yaml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    volumes:
      - open-webui-data:/app/backend/data
    environment:
      # Point WebUI to FLM's OpenAI-compatible server
      - OPENAI_API_BASE_URL=http://host.docker.internal:52625/v1
      - OPENAI_API_KEY=dummy-key

      # Explicitly disable Ollama auto-connect
      - OLLAMA_BASE_URL=

      # WebUI settings
      - WEBUI_AUTH=false
      - WEBUI_SECRET_KEY=dummysecretkey
      - ENABLE_TITLE_GENERATION=false
      - ENABLE_FOLLOW_UP_GENERATION=false
      - ENABLE_TAGS_GENERATION=false
      - ENABLE_RETRIEVAL_QUERY_GENERATION=false
      - ENABLE_IMAGE_PROMPT_GENERATION=false
      - ENABLE_WEB_SEARCH=false
      - ENABLE_SEARCH_QUERY_GENERATION=false
    restart: unless-stopped

volumes:
  open-webui-data:
```

---

### ▶️ Step 3: Launch the Open WebUI Container (in PowerShell)

```shell
docker compose up -d
```
> It could take up to 1 min before you can access Open WebUI.

This starts the container in detached mode.  
You can check logs with:

```shell
docker logs -f open-webui
```

---

### 🌐 Step 4: Access the WebUI (in Browser)

Open browser and go to:  
**http://127.0.0.1:3000**

You should now see the Open WebUI interface.

---

### 🧪 Step 5: Serve FastFlowLM with Model

```shell
flm serve llama3.2:1b
```

You can now use `FastFlowLM` directly in Open WebUI.
> When switching models, it may take a longer time to replace the model in memory.

---

### 🧼 Step 6: Stop or Clean Up (in PowerShell)

```shell
docker compose stop
```

To **remove** it completely:

```shell
docker compose down
```

This also removes the container but keeps persistent volume data.

or 

```shell
docker compose down -v
```

This removes the container and persistent volume data.

---

### 🧼 Step 7: Update Open WebUI

```shell
docker compose pull
```

---

### 🧠 Notes

- Want login? Set `WEBUI_AUTH=true`
- You must keep FastFlowLM server running
- For persistent chat history, the volume `open-webui-data` stores user data

---

> **Note (When using Open WebUI):**  
> The **Open WebUI** sends multiple background requests to the **server**.  
> To improve stability and performance, you can disable these in **Settings → Chat**:
> - **Title Auto-Generation**
> - **Follow-Up Auto-Generation**
> - **Chat Tags Auto-Generation**
> 
> Toggle them **off**, then refresh the page.

---

# 🧪 More Examples

Well done 🎉 — now let’s explore more apps together!

---

## 🤖 Example: Multi Models Comparison Enabled by FLM Queuing

A step-by-step guide to launching FastFlowLM and interacting with multiple models via Open WebUI.

[🎬 Watch the Teaser Video](https://www.youtube.com/watch?v=vUyt2MZFDm0)

---

### 🌐 Step 1: Run Open WebUI with FastFlowLM

Follow the quick setup [here](https://fastflowlm.com/docs/instructions/server/webui/).

---

### 🧩 Step 2:  Select and Add Models

At the top-right corner of the WebUI:

- Choose a model to begin (e.g., `llama3.2:1b`)
- Click **➕** to add other models, e.g.:
	- `qwen3:0.6b`
	- `gemma3:1b`

	You’ll now see several models listed. That means each one can answer your prompt.

---

### 💬 Step 3: Interact with Models

Type anything you're curious about in the input box.

⚠️ Please note:

- Each model will reply in sequence (not all at once).
- The flm server dynamically loads each model based on your selection.

---

### 🎯 Step 4: Select or Merge

After receiving replies from multiple models, choose how you'd like to continue:

- ✅ **Use the Best Response**  
  Select the answer that best meets your expectations. That response will become the active context for your next question.

- 🔗 **Merge All Responses**  
  Combine insights from all models and continue the conversation using your preferred model. This lets you synthesize multiple perspectives into a unified thread.

---

## 🌐 Example: Agentic AI Web Search with FastFlowLM

Step-by-step guide to powering Agentic AI web search in Open WebUI — NPU-only, lightning-fast, with Google PSE + FLM.

[🎬 Watch the Teaser Video](https://www.youtube.com/watch?v=wHO8ektTlik)

---

### 🛠️ Step 1: Set Up Google PSE

1. Go to [Google Programmable Search Engine](https://developers.google.com/custom-search) and sign in or create an account. Click `create a search engine`. Review the *Overview* page.
2. Visit the [Control Panel](https://programmablesearchengine.google.com/controlpanel/all) and click the `Add` button.
3. Fill in:
	- A **name** for your search engine (e.g., flm-search)
	- **What to search?** (e.g., select `Search the entire web`)
	- **Search settings** (e.g., enable `Image search`)
	- Verify you’re not a robot
	- Then click **`Create`**
4. After creation, click **`Customize`**.
5. Copy and save your **Search Engine ID** (you’ll need it later).
6. Scroll down to **Programmatic Access** → click **Get started**.
7. Find **Programmable Search Engine (free edition) users** → click **Get a Key**.
8. Select `Create a project` → Enter new project name (e.g., owbui-search) → click next → click `SHOW KEY` to reveal your **API key** → copy and save it (you'll need it later).

---

### 🌐 Step 2: Run Open WebUI with FastFlowLM

Follow the quick setup guide **[here](#-run-open-webui-with-fastflowlm)**.

---
### 🧩 Step 3: Enable Web Search in Open WebUI

With your **API Key** and **Search Engine ID** from Step 1, follow these steps:

1. In the **bottom-left corner**, click **`User`** icon, then select **`Settings`**.
2. In the **bottom panel**, open **`Admin Settings`**.
3. From the **left sidebar**, click **`Web Search`**.
4. Under `General`, toggle **`Web Search`** to enable web search function.
5. Set **`Web Search Engine`** as **`google_pse`**.
6. Enter your saved:
    - **Google PSE API Key**
    - **Google PSE Engine ID**
7. Under `Loader`, set `Concurrent Requests` to 10 or more (optional).
8. Click **`Save`**.

---

### 💬 Step 4: Start Using Web Search

1. Start a new chat and select your preferred model (e.g., qwen3-it:4b).
> ⚠️ **Note:** not all models handle web search well.
2. Under the chat input box, Click `integrations`, and toggle **Web Search** to activate it.
- 🔄 You’ll need to activate this **every time you start a new chat**. 
3. Ask anything you're curious about—real-time search will enhance your answers!

---

## 🗄️ Example: Local Private Database with RAG + FastFlowLM  

This example walks you through setting up a **local, private knowledge base** using **Retrieval-Augmented Generation (RAG)** powered by FastFlowLM.  

RAG combines two steps:  
1. **Retrieval** – fetch the most relevant information from your knowledge base (e.g., `.md` docs).  
2. **Generation** – use an AI model to create accurate, context-aware answers based on that retrieved data.  

In this example, the knowledge base is the **Open WebUI documentation**. With FastFlowLM running on the **NPU**, you get fast, efficient, and secure responses — all without sending your data to the cloud.  

[🎬 Watch the Teaser Video](https://youtu.be/GAzPj6QbfKk?si=5FDkpjlVDI64oIol)

### 📝 Step 1: Download the Documentation

1. Download the latest `Open WebUI` **[documentation](https://github.com/open-webui/docs/archive/refs/heads/main.zip)**.
2. Extract the `docs-main.zip` file to get all documentation files.
3. In the extracted folder, locate all files with `.md` and `.mdx` extensions (tip: `Ctrl+F` and search for `*.md*`).

---

### 🌐 Step 2: Run Open WebUI with FastFlowLM 

Follow the quick setup guide **[here](#-run-open-webui-with-fastflowlm)**.

---
### 🧠 Step 3: Create a Knowledge Base

1. In Open WebUI, from the **top-left** menu, navigate to **Workspace** > **Knowledge** (top bar) > Click `+` symbol on the right side to **Create a Knowledge Base**.
2. Enter `What are you working on?` → `Open WebUI Documentation`
3. Enter `What are you trying to achieve?`→ `Assistance`.
4. Click on **`Create Knowledge`**.
5. In the extracted folder, press `Ctrl+A`, then drag and drop the `.md` and `.mdx` files from the extracted folder into the `Open WebUI Documentation` knowledge base. (159 files in total as of 09/22/2025)

---

### 🧩 Step 4: Create and Configure the Model

1. Go to the **top-left** menu, navigate to **Workspace** > **Models** (top bar) > Click `+` symbol on the right side to **Add New Model**
2. Configure the Model:
	- **Model Name**: Enter a name, e.g. `FLM_RAG`
	- **Base Model**: Choose from the available list, e.g., gemma3:4b
	- **Knowledge**: Select `Open WebUI Documentation` from the dropdown
	- **Capabilities**: Check the options you need (e.g. enable **citation** to show sources)
3. Save  & Create.

---

### 💬 Step 5: Examples and Usage

1. Start a New Chat:
    - Navigate to **New Chat** and select the `FLM_RAG` model.
2. Example Queries:

🧑 User: "Introduce Open WebUI."  
🤖 Assistant: *Based on the knowledge base `Open WebUI Documentation`, here’s an introduction...*  

🧑 User: "How to use Open WebUI with Docker?"  
🤖 Assistant: *Here are the steps from the knowledge base `Open WebUI Documentation`...*  

---

## 🛠️ Example: Add FLM Custom Parameters

This example shows how to add FastFlowLM custom parameters in Open WebUI. We use `gemma4-it:e4b` as the example model.

Gemma4 supports a configurable visual token budget, which controls how many tokens are used to represent an image. FLM uses a custom parameter `image-max-tokens` to support this feature. By passing this value through FLM, you can adjust the image token budget for each query in Open WebUI and balance image detail against response speed. For more details, see [Variable Image Resolution](https://huggingface.co/google/gemma-4-E4B-it#5-variable-image-resolution).


### 🌐 Step 1: Run Open WebUI with FastFlowLM

Follow the quick setup guide **[here](#-run-open-webui-with-fastflowlm)** to start Open WebUI and connect it to FastFlowLM.

### ➕ Step 2: Add a Custom Parameter

1. In the top-left model selector, choose `gemma4-it:e4b`.
2. In the top-right corner of Open WebUI, click the `Control` icon to open the parameter list.
3. Scroll to the bottom of the parameter list and click `Add Custom Parameter`.
4. Fill in the custom parameter details:
   - **custom_param_name**: `image-max-tokens`
   - **custom_param_value**: `560` (see [Variable Image Resolution](https://huggingface.co/google/gemma-4-E4B-it#5-variable-image-resolution) for details)

### 🖼️ Step 3: Chat with an Image

Send a message with an image attachment in Open WebUI. 

Use a higher visual token budget for tasks that need more image detail, such as OCR, or a lower visual token budget for faster responses, such as a quick image description.

---