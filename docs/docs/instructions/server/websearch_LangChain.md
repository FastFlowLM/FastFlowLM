---
layout: docs
title: Web Search with LangChain
nav_order: 6
parent: Local Server (Server Mode)
---

# 🔍 RAG with Live Web Search + FastFlowLM Summarizer

This project demonstrates how to build a lightweight Retrieval-Augmented Generation (RAG) pipeline that:
- Performs live web search using `ddgs` (DuckDuckGo)
- Summarizes the results using a local LLM via **FastFlowLM**
- Runs fully offline except for the web search

---

## ✅ Prerequisites

- Windows machine
- Python 3.9 or later
- [FastFlowLM](https://github.com/ROCm/FastFlowLM) installed
- FastFlowLM model served (e.g., `llama3.2:1b`)

---

## 🛠️ Step-by-Step Setup

### 1. 🧪 Create the Project Folder

```shell
mkdir rag_websearch_flm
cd rag_websearch_flm
```

### 2. 🐍 Create a Virtual Environment

```shell
python -m venv rag_websearch-env
.\rag_websearch-env\Scripts\activate
```

> 💡 If PowerShell blocks script execution:
```shell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 3. 📦 Install Required Packages

```bash
pip install -U langchain langchain-community langchainhub langchain-ollama langchain-huggingface sentence-transformers ddgs beautifulsoup4 requests ollama
```

---

### 4. 🚀 Launch FastFlowLM

In another terminal:

```bash
flm serve llama3.2:1b
```

This starts your FastFlowLM API at: `http://127.0.0.1:52625`

---

### 5. 📝 Create the Script: `websearch_rag.py`

```python
# websearch_rag.py

import warnings
import os
from ddgs import DDGS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=ResourceWarning)

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "llama3.2:1b"
BASE_URL = "http://127.0.0.1:52625"
MAX_RESULTS = 5


# ----------------------------
# Step 1: DuckDuckGo Search
# ----------------------------
def run_web_search(query: str, max_results: int = 5) -> str:
    print(f"\n🔍 Running web search for: '{query}'")
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "No title")
                body = r.get("body", "No description")
                link = r.get("href", "No link")
                results.append(f"• {title}\n{body}\n🔗 {link}\n")
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return ""

    return "\n".join(results)


# ----------------------------
# Step 2: FastFlowLM Summarization
# ----------------------------
def summarize_with_fastflowlm(search_results: str, model_name=MODEL_NAME) -> str:
    if not search_results.strip():
        return "⚠️ No search results to summarize."

    llm = OllamaLLM(model=model_name, base_url=BASE_URL)

    prompt = PromptTemplate.from_template("""
You are a factual and concise research assistant. Summarize the following web search results clearly and accurately.

Search Results:
{search_results}

Summary:
""")

    try:
        summary = (prompt | llm).invoke({"search_results": search_results})
    except Exception as e:
        return f"❌ Error generating summary: {e}"

    return summary.strip()


# ----------------------------
# Step 3: Convert Summary to RAG Docs
# ----------------------------
def build_retriever_from_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# ----------------------------
# Step 4: Ask Follow-up with RAG
# ----------------------------
def ask_rag_question(question: str, retriever, model_name=MODEL_NAME):
    llm = OllamaLLM(model=model_name, base_url=BASE_URL, temperature=0.0)

    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the information in the context below to answer the question.

If the context does not contain the answer, say: "I don't know."

Context:
{context}

Question: {question}
Answer:""")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa.invoke(question)
    print("\n🤖 RAG Answer:\n", result["result"])

    if result.get("source_documents"):
        print("\n📄 Context Source:\n", result["source_documents"][0].page_content)
    else:
        print("\n📄 No context used.")


# ----------------------------
# MAIN
# ----------------------------
def main():
    search_query = "Recent developments in AMD Ryzen AI chips"
    follow_up_question = "What new features do AMD Ryzen AI chips have in 2025?"

    # Web Search
    search_text = run_web_search(search_query, max_results=MAX_RESULTS)
    print("\n🌐 Raw Web Search Output:\n")
    print(search_text)

    # Summarization
    summary = summarize_with_fastflowlm(search_text)
    print("\n🧠 Summary:\n")
    print(summary)

    # Build RAG Vector DB
    retriever = build_retriever_from_text(summary)

    # RAG Follow-up
    ask_rag_question(follow_up_question, retriever)


if __name__ == "__main__":
    main()
```

---

## ▶️ Run the Script

Make sure FastFlowLM is running, then:

```bash
python websearch_rag.py
```

---

## 📤 Expected Output

```
🔍 Running web search for: 'Recent developments in AMD Ryzen AI chips'

🌐 Raw Web Search Output:

• AMD Introduces New Radeon Graphics Cards and Ryzen ...
May 20, 2025 — AMD Introduces New Radeon Graphics Cards and Ryzen Threadripper Processors at COMPUTEX 2025 · AMD Powers Next-Gen Gaming Infused with AI · Pricing ...
🔗 https://ir.amd.com/news-events/press-releases/detail/1253/amd-introduces-new-radeon-graphics-cards-and-ryzen-threadripper-processors-at-computex-2025

• AMD Ryzen™ AI - Windows PCs with AI Built In
A new era of AI PCs begins with AMD and Windows. AMD Ryzen™ AI 300 Series processor powered Copilot+ PCs deliver new, transformative AI experiences for your ...
🔗 https://www.amd.com/en/products/processors/consumer/ryzen-ai.html

• AMD launches Ryzen AI 300 and 200 series chips for laptops
Jan 6, 2025 — AMD has launched its Ryzen AI 300 and Ryzen 200 series of mobile processors at CES 2025 at Las Vegas, debuting a total of 15 new models.
🔗 https://www.tomshardware.com/pc-components/cpus/amd-launches-ryzen-ai-300-and-200-series-chips-for-laptops

• AMD Announces Expanded Consumer and Commercial AI ...
Jan 6, 2025 — The new Ryzen AI 300 Series processors, feature up to 8 “Zen 5” CPU cores and the latest RDNA 3.5 graphics architecture. With an industry- ...
🔗 https://www.amd.com/en/newsroom/press-releases/2025-1-6-amd-announces-expanded-consumer-and-commercial-ai-.html

• AMD Ryzen™ AI PCs: A Portfolio Built for the Future
May 29, 2025 — Coming Soon. Exciting new systems featuring AMD Ryzen™ AI processors are on the horizon, offering a glimpse into the next generation of ...
🔗 https://www.amd.com/en/blogs/2025/available-amd-ai-pcs.html


🧠 Summary:

The AMD Ryzen AI series has been launched, offering new features and enhancements in:

* New Radeon Graphics Cards with up to 8 "Zen" CPU cores
* Ryzen Threadripper Processors
* Ryzen AI PCs, featuring the latest "Zen" CPU architecture

🤖 RAG Answer:
 The new Ryzen AI chips offer several features, including:

* New Radeon Graphics Cards
* Ryzen Threadripper Processors

📄 Context Source:
 The AMD Ryzen AI series has been launched, offering new features and enhancements in:

* New Radeon Graphics Cards with up to 8 "Zen" CPU cores
* Ryzen Threadripper Processors
* Ryzen AI PCs, featuring the latest "Zen" CPU architecture
```

---

## 🧠 What’s Happening Behind the Scenes

This Python script combines real-time web search with a local FastFlowLM model for Retrieval-Augmented Generation (RAG). Here's how each step works:

---

### 🔍 1. Web Search (via DuckDuckGo API)

We use the `ddgs` package to send live search queries to DuckDuckGo. The results include:
- Page titles
- Snippets (summaries)
- URLs

These are printed and also compiled into a large raw text block for downstream use.

---

### 🕸️ 2. Optional: Full Page Scraping

If full-text RAG is enabled, each URL is downloaded via `requests` and parsed using `BeautifulSoup`, skipping:
- `<script>`, `<style>`, `<nav>`, `<footer>`, etc.

This step pulls **full paragraphs and sentences**, improving retrieval quality.

---

### 🧩 3. Chunking and Embedding

The retrieved or scraped content is broken into chunks using `RecursiveCharacterTextSplitter`. Each chunk:
- Is around 1000 characters (with overlap)
- Gets embedded using a sentence transformer model via `HuggingFaceEmbeddings`
- Is stored in a `FAISS` vector store for fast similarity search

---

### 🤖 4. RAG Retrieval with FastFlowLM

A local FastFlowLM model is accessed via `OllamaLLM`, pointing to `http://127.0.0.1:52625`. We build a LangChain `RetrievalQA` pipeline:
- Queries are matched to relevant chunks from the vector store
- A prompt template supplies those chunks as **context**
- FastFlowLM generates an answer based strictly on that context

---

### 🧠 5. Output

You get:
- ✅ Final answer (streamed or batched)
- 📄 Source snippet (for transparency)
- 🔗 Search links (for traceability)

---

### ⚙️ Tools Behind the Curtain

| Module | Purpose |
|--------|---------|
| `ddgs` | Live DuckDuckGo search |
| `requests` + `BeautifulSoup` | Scrape full HTML page |
| `langchain.text_splitter` | Chunk large documents |
| `langchain_huggingface` | Create vector embeddings |
| `langchain_community.vectorstores.FAISS` | Store & search context chunks |
| `langchain_ollama.OllamaLLM` | Call local FastFlowLM models |
| `RetrievalQA` | Combine retriever + LLM in RAG pipeline |

---


---
