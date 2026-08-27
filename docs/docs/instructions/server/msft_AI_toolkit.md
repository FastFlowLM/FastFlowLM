---
layout: docs
title: Microsoft AI Toolkit + FLM
nav_order: 4
parent: Local Server (Server Mode)
---

# 🧠 Using Microsoft AI Toolkit with FastFlowLM in VS Code

This guide explains how to run **FastFlowLM locally** on Windows and connect it to **Microsoft AI Toolkit** in **Visual Studio Code**.

---

## ✅ 1. Install Visual Studio Code (Windows)

1. Go to: https://code.visualstudio.com  
2. Download the **User Installer for Windows**  
3. Run the installer:
   - ✅ Check **"Add to PATH"**
   - ✅ (Optional) Create a desktop icon
   - ✅ Accept the license agreement  
4. Complete the installation

---

## 📦 2. Install AI Toolkit Extension

1. Launch **VS Code**
2. Open the **Extensions panel**: `Ctrl + Shift + X`
3. Search: `AI Toolkit for Visual Studio Code`
4. Click **Install**

You’ll now see the **AI Toolkit** icon on the sidebar.

---

## 🧠 3. Install FastFlowLM & Pull a Model

1. Download & install [FastFlowLM](../../install_win/)

2. Open **PowerShell**

3. Pull a base model:

```shell
flm pull llama3.2:1b
```

4. Confirm it's installed:

```shell
flm list
```

You should see models like `llama3.2:1b` listed.

---

## 🧩 4. Add FastFlowLM Model in AI Toolkit via Custom Endpoint

1. In **VS Code**, open the **AI Toolkit** panel  
2. Navigate to **Models → Catalog**  
3. Click **➕ Add Your Own Model**  
4. In the top bar, select **Add Custom Model**
5. Enter OpenAI compatible chat completion endpoint URI:
```
http://127.0.0.1:52625/v1/chat/completions
```
Click **Enter**
6. Enter the exact model name as in the API:
```
llama3.2:1b
```
Click **Enter**
7. Enter display model name:
```
flm-llama3.2:1b
```
Click **Enter**
8. Enter API key:
```
dummy
```
Click **Enter**
9. You will now see the model under **My Models**

---

## 📡 5. Activate Server Mode with a Model

Open PowerShell, enter

```shell
flm serve llama3.2:1b
```

---

## 💬 6. Use the Model in Playground

1. Switch to the **Playground** tab in AI Toolkit  
2. Choose your custom FastFlowLM model from the dropdown  
3. Type a prompt, e.g.:

```
What are the benefits of local inference?
```

4. Click **Send**
5. The model response will stream from your local FastFlowLM instance

---

## 🗑️ 7. Remove a Model from "My Models" in AI Toolkit

To remove a previously added model from the **My Models** section in the AI Toolkit:

1. Navigate to **AI Toolkit → My Models**
2. Expand the **Custom** section
3. Right-click the model you wish to remove (e.g., `llama3.2:1b`)
4. Select **Delete**

> 🧹 This action removes the model's reference from the AI Toolkit interface but does **not** delete the model files from local disk.

---