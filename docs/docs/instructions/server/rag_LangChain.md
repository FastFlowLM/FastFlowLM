---
layout: docs
title: Build RAG with LangChain
nav_order: 5
parent: Local Server (Server Mode)
---

# ⚡ RAG with LangChain + FastFlowLM + FAISS (Windows, Offline)

This guide walks you through building a **Retrieval-Augmented Generation (RAG)** system using:

- ✅ **LangChain** for orchestration  
- ✅ **FastFlowLM** as a local LLM backend (drop-in replacement for Ollama)  
- ✅ **HuggingFace Embeddings** via `sentence-transformers`  
- ✅ **FAISS** for fast vector search  
- ✅ **Multiple .txt files** loaded from a folder  
- ✅ Fully **offline**, **privacy-preserving**, and **Windows-compatible**

---

## 📦 Prerequisites

### ✅ System Requirements

- Python 3.10+
- Windows OS
- VS Code (recommended)
- PowerShell

### ✅ Install FastFlowLM (local server)

Download from GitHub Releases:  
👉 https://github.com/ROCm/FastFlowLM

Launch the server:

```shell
flm serve llama3.2:3b
```

> This exposes a REST API at `http://127.0.0.1:52625` by default.

---

## 🛠️ Environment Setup

### 1. Create a virtual environment

```shell
mkdir rag_with_flm
cd rag_with_flm
python -m venv rag-env
.\rag-env\Scripts\activate
```

> If script execution is blocked, run:
> ```shell
> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### 2. Install Python dependencies

```bash
pip install -U langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu tiktoken ollama langchain-ollama
```

> ✅ We still install the `ollama` Python package because **LangChain’s `OllamaLLM` class can be pointed at any local REST backend like FastFlowLM**.  
> 🔁 `Ollama` and `FastFlowLM` use the same base URL `base_url="http://127.0.0.1:52625"`. Thus, they are interchangeable.

---

## 📂 Project Structure

```
rag_with_flm/
├── docs/
│   ├── un_history.txt
│   ├── peacekeeping.txt
│   └── founding_charter.txt
├── rag_demo.py
├── rag-env/
```

---

## 📄 Example Documents

### `docs/un_history.txt`

```text
The United Nations (UN) was officially established on 24 October 1945 after the ratification of its Charter. It replaced the League of Nations.
```

### `docs/peacekeeping.txt`

```text
UN Peacekeeping began in 1948. Since then, more than 70 operations have been deployed to help maintain peace and security.
```

### `docs/founding_charter.txt`

```text
The Charter of the United Nations was signed on 26 June 1945 in San Francisco, at the conclusion of the United Nations Conference on International Organization.
```

---

## 🧠 `rag_demo.py`

```python
# rag_demo.py

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM  # still used to wrap FastFlowLM!
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load .txt files from folder
loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

# Smart chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)

# Embed with sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Prompt template with fallback
prompt_template = """
You are an expert assistant. Answer using ONLY the context provided.
If the answer is not present, reply: "I don't know based on the given information."

Context:
{context}

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# ✅ FastFlowLM via OllamaLLM (optional: custom base_url)
llm = OllamaLLM(model="llama3.2:3b", base_url="http://127.0.0.1:52625")

# Build Retrieval-Augmented QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Ask a question
query = "When was the United Nations founded and what replaced the League of Nations?"
result = qa_chain.invoke(query)

# Display answer and sources
print("\n" + "="*80)
print("🧠 Answer:\n", result["result"])
print("="*80)

print("📚 Top Sources:")
for i, doc in enumerate(result["source_documents"]):
    print(f"\n--- Source {i+1} ---\n{doc.page_content}")
```

---

## ▶️ Run It

Start your FastFlowLM server in one terminal:

```bash
flm serve llama3.2:3b
```

Then in another terminal (with env activated):

```bash
python rag_demo.py
```

---

## ✅ Output Example

```
🧠 Answer:
 The United Nations was founded on 24 October 1945 and it replaced the League of Nations.

📚 Top Sources:
--- Source 1 ---
The United Nations (UN) was officially established on 24 October 1945...
```
