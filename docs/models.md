---
layout: page
title: "Models"
permalink: /models/
description: "Curated Ryzen AI-ready model catalog with first-class FastFlowLM recipes."
sections:
  - type: hero
    kicker: "Model catalog"
    title: "Choose a recipe"
    body: |
      FastFlowLM curates the most requested families and publishes tuned manifests under `flm pull <model>`.
      We validate every build on Ryzen™ AI laptops and provide matching cards—see them [here](/docs/models/).
      The catalog already covers models spanning Qwen3.6 (MoE), Qwen3.5 (Vision), Gemma4 (omni), Qwen3, GPT-OSS-20B (MoE), DeepSeek-R1, Whisper
      (Audio), MedGemma (Medical), and other leading open families.
    ctas:
      - label: "View model docs"
        href: "/docs/models/"
        style: primary
    right:
      title: "Supported families"
      body: |
        Each manifest describes quantization, context window, tokenizer, and recommended memory,
        so there are no surprises after download.
      pills:
        - "Qwen3.6 (Vision and MoE)"
        - "Gemma 4 (omni)"
        - "Qwen3.5 (Vision)"
        - "Qwen 3 (Vision)"
        - "GPT-OSS (MoE)"
        - "LFM2 (Liquid AI)"
        - "EmbeddingGemma (RAG)"
        - "Whisper (Audio)"
        - "MedGemma (Medical)"
        - "DeepSeek R-1"
        - "LLaMA 3.x"
       
        # - "xLAM-2 (Tool-calling, Salesforce), and their corresponding Chat Templates and major API (e.g., OpenAI-compatible API)"

  - type: two_column
    left:
      kicker: "Roadmap"
      title: "Upcoming drops"
      body: |
        Follow the GitHub project for rolling releases. We publish nightly builds for community testing and
        stable channels every few weeks.
      items:
        - heading: "Fill this in"
          body: "Coming soon!"
    right:
      title: "Stay in the loop"
      ctas:
        - label: "Release notes"
          href: "https://github.com/ROCm/FastFlowLM/releases"
          style: primary
          external: true
        - label: "Discord"
          href: "https://discord.com/invite/jtWZdMJ8ee?utm_source=site"
          style: ghost
          external: true
---

