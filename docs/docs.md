---
layout: page
title: "Documentation Hub"
permalink: /docs/landing/
description: "Overview of FastFlowLM documentation, organized for builders."
sections:
  - type: hero
    kicker: "Docs"
    title: "Everything lives in one Jekyll site"
    body: |
      Marketing pages, guides, and deep reference content all share the same layouts and includes.
      Browse the highlights below or jump straight into the markdown-driven docs.
    ctas:
      - label: "Go to docs"
        href: "/docs/"
        style: primary
      - label: "Installation guide"
        href: "/docs/install/"
        style: ghost
    right:
      title: "Collections"
      items:
        - heading: "Install"
          body: "Windows, Linux, and WSL walkthroughs with troubleshooting tips."
        - heading: "Instructions"
          body: "Guides for CLI, server mode, Open WebUI, LangChain, Obsidian, and web search flows."
        - heading: "Models & Benchmarks"
          body: "Model cards and reproducible benchmark data for every release."

  - type: cards
    variant: alt
    kicker: "Structure"
    title: "Modular layouts + Markdown content"
    body: |
      The docs site uses the `docs` layout plus shared components (hero blocks, feature lists, pill lists).
      Everything inherits the same typography and navigation as the landing pages.
    cards:
      - label: "Layouts"
        title: "`_layouts/docs.html`"
        body: |
          Wraps content with the shared header/footer and a sticky sidebar generated from `site.docs_nav`.
      - label: "Includes"
        title: "`_includes/sections/*`"
        body: |
          Reusable sections for heroes, model teasers, CTA bars, and more.
      - label: "Styling"
        title: "`assets/styles.css`"
        body: |
          Dark theme shared by every page, with component classes for sections, grids, and docs content.

  - type: two_column
    left:
      kicker: "Contribute"
      title: "Docs are just Markdown files"
      body: |
        Edit anything in `/docs`, add front matter (`layout: docs`) and open a PR.
        Liquid shortcodes + markdown tables keep everything portable.
      ctas:
        - label: "Edit on GitHub"
          href: "https://github.com/ROCm/FastFlowLM/tree/main/docs"
          style: ghost
          external: true
    right:
      title: "Looking for something else?"
      body: |
        The docs hub is the best starting point, but you can also reach us via Discord or
        [info@fastflowlm.com](mailto:info@fastflowlm.com) for bespoke support.
---


