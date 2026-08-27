---
layout: docs
title: Obsidian + FLM
nav_order: 3
parent: Local Server (Server Mode)
---

# 🧠 Run Obsidian with FastFlowLM — A Faster, Smarter Second Brain

This guide walks you through setting up **Obsidian** (with *AI Providers* and *Local GPT* plug-ins) and **FastFlowLM (FLM)** to work seamlessly together.

---
## 📑 Prerequisites

Before starting, ensure you have the following installed:
- **[Obsidian](https://obsidian.md/)**
- **[FastFlowLM](https://fastflowlm.com/docs/install/)**

---
## 🚀 Quick Start

### Install plugins in Obsidian

1. Open your **Obsidian vault**.
2. If the sidebar is collapsed, click **Expand** in the top-left corner.
3. In the bottom-right corner of the sidebar, click **⚙️** to open the setting panel.
4. Under the **Options** section (left), click **Community Plugins** .
5. Click **Browse**, then search for **AI Providers**.
6. Install and enable **AI Providers**.
7. Repeat the same steps to install and enable **Local GPT**.

### Start FLM server

Run the following command in PowerShell to launch the FastFlowLM server:

`flm serve llama3.2:1b`

### Configure AI Providers 

#### Option A: OpenAI API

1. Click **⚙️** to open the setting panel.
2. Under **Community Plugins** section (left), click **AI Providers**.
3. Click **➕** to create a new provider.
4. Set **Provider type** to **OpenAI**.
5. Enter the **Provider URL**: `http://127.0.0.1:52625/v1`.
6. Enter a **Provider name** (e.g., `FLM_OpenAI`).
7. Enter any **API key** (e.g., `flm`).
8. On the **Model** line, click the **🔄** icon to load available models, then select your preferred model (e.g. llama3.2:1b).
9. Click **Save**.

#### Option B: Ollama API

1. Click **⚙️** to open the setting panel.
2. Under **Community Plugins** section (left), click **AI Providers**.
3. Click **➕** to create a new provider.
4. Set **Provider type** to **Ollama**.
5. Enter the **Provider URL**: `http://127.0.0.1:52625`.
6. Enter a **Provider name** (e.g., `FLM_Ollama`).
7. Enter any **API key** (e.g., `flm`).
8. On the **Model** line, click the **🔄** icon to load available models, then select your preferred one (e.g. llama3.2:1b).
9. Click **Save** to apply your **AI Provider**.

### Configure Local GPT

#### Pick models

1. Click **⚙️** to open the setting panel.
2. Under **Community Plugins** section (left), click **Local GPT**.
3. Set **Main AI Provider** to the provider you configured earlier (e.g., `FLM_OpenAI ~ llama3.2:1b`).
4. Review the **Action list** to see what **Local GPT** can do.

#### Set up hotkeys

1. On setting panel (Click **⚙️** to open the setting panel if you are not on setting panel).
2. Under the **Options** section (left), click **Community Plugins**.
3. Under **Installed Plugins** (right), find the **Local GPT** row, then click **⊕** to add Hotkeys.
4. For each action, click **⊕** on the right side and assign a hotkey.

For example:  
- `Alt + H` → *General Help*  
- `Alt + C` → *Continue Writing*  
- `Alt + S` → *Summarize*  
- `Alt + G` → *Fix Spelling and Grammar*  
- `Alt + F` → *Find Action Items*  

---

## 💡 Start Creating Smarter

🎉 Ready to roll… let’s give it a try!

### ✨ General Help

📌 *Scenario*: You wrote some short notes in Obsidian, but they look too simple.  
👉 Highlight your notes, press the hotkey you set for **General Help** (e.g., `Alt + H`), and the LLM will expand them into a full explanation.


### ✍️ Continue Writing
 
📌 *Scenario*: You started a paragraph but don’t know how to continue.  
👉 Highlight the unfinished text, press your **Continue Writing** hotkey (e.g., `Alt + C`), and the LLM will keep writing naturally for you.


### 🔍 Summarize

📌 *Scenario*: You have a long article or meeting notes that take too much time to read.  
👉 Highlight the long text, press your **Summarize** hotkey (e.g., `Alt + S`), and the LLM will give you the key points in a short summary.

### 📘 Fix Spelling and Grammar

📌 *Scenario*: You typed something fast and it’s full of mistakes.  
👉 Highlight your text, press your **Fix Spelling and Grammar** hotkey (e.g., `Alt + G`), and the AI will clean it up and make it easy to read.

