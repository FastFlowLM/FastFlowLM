---
layout: page
title: "Team"
permalink: /team/
description: "Researchers, engineers, and community leads building FastFlowLM."
sections:
  - type: hero
    kicker: "Team"
    title: "Architecture + systems + community"
    body: |
      FastFlowLM is a collaboration between academic researchers, software engineers,
      and community maintainers. The core group includes four PhDs and three
      B.S. graduates in Electrical and Computer Engineering with deep experience in LLM internals, parallel
      processing, and architecture-specific software optimization.
    ctas:
      - label: "Contact us"
        href: "mailto:info@fastflowlm.com"
        style: primary
      - label: "Join Discord"
        href: "https://discord.com/invite/jtWZdMJ8ee?utm_source=site"
        style: ghost
        external: true
    right:
      title: "What we focus on"
      items:
        - heading: "Kernel research"
          body: "Co-designing fused attention + MoE operators with AMD."
        - heading: "Developer experience"
          body: "Shipping ergonomic CLI, APIs, and docs inspired by llama.cpp and Ollama."
        - heading: "Open community"
          body: "Weekly office hours, demos, and benchmarking nights."

  - type: people
    kicker: "Core collaborators"
    title: "Hardware + runtime expertise"
    people:
      - name: "Tao Wei"
        role: "Co-Founder"
        image: "/assets/weitao.jpg"
        bio: |
          Professor of Electrical & Computer Engineering · Clemson University. Leads the NEXT Lab focused on domain-specific accelerators, reconfigurable computing, and applied ML.
          Guides FastFlowLM kernel strategy and academic collaborations.
        links:
          - label: "Clemson profile"
            url: "https://www.clemson.edu/cecas/departments/ece/faculty_staff/faculty/twei.html"
          # - label: "Scholar"
          #   url: "https://scholar.google.com/citations?user=SsgrItsAAAAJ&hl=en"
      - name: "Ken Qing Yang"
        role: "Co-Founder"
        image: "/assets/kenqingyang.png"
        bio: |
          Distinguished Engineering Professor · University of Rhode Island. With more than 30 years of experience in computer architecture and parallel processing, he is a serial entrepreneur who has successfully built four high-tech startups rooted in his research innovations—including VeloBit (acquired by Western Digital) and DapuStor (recently IPOed). 
        links:
          - label: "URI profile"
            url: "https://web.uri.edu/ecbe/meet/qing-ken-yang/"
          - label: "VeloBit"
            url: "https://web.uri.edu/hpcl/from-innovation-to-successful-company/"
          - label: "DapuStor"
            url: "https://en.dapustor.com/"
      - name: "Zhenyu (Alfred) Xu"
        role: "Co-Founder"
        image: "/assets/zhenyu.png"
        bio: |
          Research Assistant Professor · Clemson University. Focused on domain-specific accelerator design, reconfigurable computing, and efficient on-device AI inference.
          Brings deep experience in hardware–software co-optimization across FPGA, CGRA, and emerging AI accelerator architectures.
        links:
          - label: "Scholar"
            url: "https://scholar.google.com/citations?hl=en&user=iuIZeGYAAAAJ&view_op=list_works"

  - type: two_column
    variant: alt
    left:
      kicker: "Advisors & contributors"
      title: "Builder network"
      body: |
        FastFlowLM thrives because of community engineers who maintain connectors, polish docs, and stress-test nightly builds.
    right:
      kicker: "Press & partnerships"
      title: "Let’s make Ryzen AI shine"
      body: |
        Hardware vendors, ISVs, and research labs collaborate with FastFlowLM for demos, co-marketing,
        and silicon feedback loops. Drop us a line to get started.
      ctas:
        - label: "Email the team"
          href: "mailto:info@fastflowlm.com"
          style: primary
        - label: "Open an issue"
          href: "https://github.com/ROCm/FastFlowLM/issues/new"
          style: ghost
          external: true
---

