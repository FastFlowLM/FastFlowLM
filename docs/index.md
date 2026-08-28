---
layout: page
title: "FastFlowLM"
permalink: /
description: "NPU-native runtime that unlocks AMD Ryzen AI NPUs with instant installs, rich models, and matching docs."
sections:
  - type: hero
    id: top
    kicker: "NPU-first runtime"
    title: "The fastest, most efficient LLM inference on NPUs"
    body: |
      FastFlowLM (FLM) delivers a familiar single-command developer experience built exclusively for AMD Ryzen™ AI NPUs. Install in seconds, stream tokens instantly, and run context windows up to 256k — all with dramatically better efficiency than GPU-first stacks.
      
      The FastFlowLM team is now part of AMD.
    ctas:
      - label: "Download FastFlowLM (Windows)"
        href: "https://github.com/ROCm/FastFlowLM/releases/latest/download/flm-setup.msi"
        style: primary
        external: true
      - label: "GitHub"
        href: "https://github.com/ROCm/FastFlowLM"
        style: ghost
        external: true
      - label: "Models"
        href: "/models/"
        style: ghost
      - label: "Benchmarks"
        href: "/benchmarks/"
        style: ghost
      # - label: "Demos"
      #   href: "/demos/"
      #   style: ghost
      - label: "Discord"
        href: "https://discord.com/invite/jtWZdMJ8ee?utm_source=site"
        style: ghost
        external: true
    stats:
      - label: "Runtime size"
        value: "~17 MB"
      - label: "Context"
        value: "Up to 256k tokens"
      - label: "Supported chips"
        value: "Ryzen™ AI (Strix, Strix Halo, Kraken, Gorgon Point)"
    right:
      carousel:
        slides:
          - title: "NPU‑First Architecture"
            description: "Built exclusively for AMD Ryzen™ AI NPUs with optimized kernels for maximum efficiency."
            visual:
              type: "npu_orbit"
              title: "Ryzen™ AI"
              subtitle: "FastFlowLM Runtime"
              metrics:
                - "Over 10× power efficiency"
                - "256k ctx"
                - "Vision · Audio · OMNI · Text"
          - title: "Familiar Developer Flow"
            description: "Same simple CLI & API patterns you already know — deeply optimized for NPU performance."
            visual:
              type: "code"
              title: "PowerShell"
              content: |
                flm run qwen3.5:4b
                flm list
                flm serve qwen3.5:4b
          - title: "Unmatched Performance and Efficiency"
            description: "Ultra‑lightweight runtime with exceptional power efficiency and massive context windows."
            visual:
              type: "metrics"
              metrics:
                - value: "Over 10×"
                  label: "Power Efficiency"
                - value: "256k"
                  label: "Context Tokens"
                - value: "~17 MB"
                  label: "Runtime Size"
          - title: "Multi‑Modal AI Support"
            description: "Run language models, vision models, audio processing, embeddings, and support Mixture‑of‑Experts (MoE) architectures."
            visual:
              type: "pills"
              pills:
                - "LLMs"
                - "Vision"
                - "Audio"
                - "Reasoning"
                - "Embeddings"
                - "MoE Support"

  - type: media
    variant: alt
    kicker: "Qwen3.6-MoE on NPU"
    title: "Qwen3.6-35B-A3B streaming fully on the Ryzen™ AI NPU"
    media:
      src: "/assets/qwen3.6-demo.gif"
      alt: "Qwen3.6-35B-A3B MoE model running locally on the Ryzen AI NPU"
      href: "https://youtu.be/TT7WZOeqxQY?si=DUpdyw_xNBYfEZNz"
      kicker: "Qwen3.6-MoE on NPU"
      body: |
        The fastest Qwen3.6-35B-A3B MoE implementation on any NPU—built for both Windows and Linux (Linux in this demo).

  - type: media
    variant: alt
    kicker: "GPT-OSS on NPU"
    title: "GPT-OSS-20B streaming fully on the Ryzen™ AI NPU"
    media:
      src: "/assets/gpt-oss-demo.gif"
      alt: "GPT-OSS 20B running locally on the Ryzen AI NPU"
      href: "https://youtu.be/sZt1WyNoL2U?si=7U3z6u6E9KF6G_Dd"
      kicker: "GPT-OSS on NPU"
      body: |
        Runs GPT-OSS-20B at 19 TPS (tokens per second) with 10× GPU efficiency — the fastest MoE on any NPU.

  - type: media
    variant: alt
    kicker: "Gemma3 (Vision) on NPU"
    title: "Gemma3 (Vision) understands and describes the image"
    media:
      src: "/assets/gemma3_4b.gif"
      alt: "Gemma3 vision model understanding an image"
      href: "https://youtu.be/BLd2j9dPw40?si=5GzOW2FlZgGVp8Xi"
      kicker: "Gemma3 (Vision) on NPU"
      body: |
        Understand and describe images instantly — FastFlowLM runs Google Gemma3 fully on the NPU for fast, private, and efficient vision inference.

  - type: media
    variant: alt
    kicker: "Whisper on-device"
    title: "Transcribe and summarize long-form audio locally"
    media:
      src: "/assets/cramer2.gif"
      alt: "Whisper transcription and summarization demo"
      href: "https://youtu.be/0t8ijUPg4A0?si=ETKdvig6lYiZb1Q_"
      kicker: "Whisper on-device"
      body: |
        Transcribe hours of audio locally — FLM runs OpenAI Whisper fully on the NPU — fast, private, and efficient.

  - type: media
    variant: alt
    kicker: "Llama 3.2 on NPU"
    title: "Interact with Llama 3.2-3B via Open WebUI"
    media:
      src: "/assets/llama-demo.gif"
      alt: "Llama 3.2 chat demo running in the FastFlowLM WebUI"
      href: "https://youtu.be/mPrr9FLd8ps?si=vsyHkmtrBjP4s-dq"
      kicker: "Llama 3.2 on NPU"
      body: |
        Runs Meta Llama 3.2-3B at 28 TPS with over 10× GPU efficiency — the fastest on any NPU.

  - type: two_column
    id: install
    left:
      kicker: "Install"
      title: "From download to first token in under a minute"
      body: |
        FastFlowLM ships as a 17 MB runtime with a familiar single-command CLI and a server compatible with the OpenAI API.
        No drivers, no guesswork—just run the installer, pull a model, and start chatting.
      items:
        - heading: "Zero-conf installer"
          body: "Signed installers for every Ryzen™ AI 300 laptop — download, run, done."
        - heading: "Drop-in APIs"
          body: "OpenAI-compatible APIs — plug in your existing tools instantly."
        - heading: "Secure by default"
          body: "On-device security: local tokens and full offline mode."
    right:
      kicker: "Quickstart"
      code_blocks:
        - title: "CLI"
          content: |
            Invoke-WebRequest https://github.com/ROCm/FastFlowLM/releases/latest/download/flm-setup.msi `
              -OutFile flm-setup.msi
            Start-Process .\flm-setup.msi -Wait
            flm pull llama3.2:3b
            flm run llama3.2:3b --ctx-len 131072
            
  - type: cards
    id: models
    kicker: "Models"
    title: "One runtime, every Ryzen-ready model"
    body: |
      Pull curated FastFlowLM recipes.
      The runtime streams tokens via an OpenAI-compatible API, so existing apps work without rewrites.
    cards:
      - label: "Flagship reasoning"
        title: "GPT-OSS · DeepSeek-R1 · Qwen3"
        body: "Optimized kernels with smart context reuse."
      - label: "Vision & speech"
        title: "Gemma3 · Qwen3-VL · Whisper"
        body: "VLM and audio pipelines run on the NPU, enabling private multimodal assistants."
      - label: "Local private edge database"
        title: "Retrieval-Augmented Generation (RAG) · Embedding Model"
        body: "Build and run a complete RAG workflow fully on the NPU, without relying on the CPU or GPU."
    ctas:
      - label: "Browse models"
        href: "/models/"
        style: ghost
      - label: "Model docs"
        href: "/docs/models/"
        style: ghost

  - type: two_column
    id: benchmarks
    variant: alt
    left:
      kicker: "Benchmarks"
      title: "Proof on silicon, not slides"
      body: |
        FastFlowLM is tuned on real Ryzen™ AI hardware with synthetic and application-level workloads.
        Expect steady 20–80 tok/s on models at < 2 W (CPU+NPU), plus deterministic latency for agentic chains.
      items:
        - heading: "Full-stack telemetry"
          body: "See exactly where compute goes with NPU, CPU, and memory counters."
        - heading: "Scenario-driven suites"
          body: "Instruction tuning, RAG, chat, and multimodal tests with real workloads."
      ctas:
        - label: "Benchmark details"
          href: "/benchmarks/"
          style: ghost
        - label: "View raw results"
          href: "/docs/benchmarks/"
          style: ghost
    right:
      metric_cards:
        - label: "Llama3.2 1B @ Q4_1 (4-bit with bias)"
          value: "66 tok/s"
          desc: "Ryzen™ AI 7 HX 350 · ms-level latency"
        - label: "Gemma 3 Vision"
          value: "~2 sec"
          desc: "Image understanding on XDNA2 NPU"
        - label: "Power draw (CPU + NPU)"
          value: "< 2 W"
          desc: "Full assistant stack vs ~25 W GPU baseline"

  - type: two_column
    id: docs
    left:
      title: "Dive into the docs"
      body: |
        Everything you need to install FastFlowLM, pick a model, and start streaming on your Ryzen™ AI NPU.

        For inquiries: [Discord](https://discord.com/invite/jtWZdMJ8ee?utm_source=site)
    right:
      ctas:
        - label: "Dive into docs"
          href: "/docs/"
          style: primary
        - label: "Installation guide"
          href: "/docs/install/"
          style: ghost
---

