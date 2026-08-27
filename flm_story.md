# 📜 The Story of FastFlowLM (FLM)

From a 2025 university research project to the NPU-first runtime for AMD Ryzen™ AI — now part of AMD.

---

## 🎓 2025 — A University Project

Modern AMD Ryzen™ AI laptops all shipped with a capable, power-efficient NPU. Almost nobody could run an LLM on it with good performance and long context length.

FastFlowLM started as an **NSF-funded university research project** aimed squarely at that gap. The bet: closing it required co-designing kernels around the NPU's dataflow architecture from the ground up — not porting an existing runtime onto it.

What began as federally funded lab work became **FastFlowLM, Inc.** in 2025, and the runtime was released publicly as open source that June.

### Founders

- **[Tao Wei](https://www.linkedin.com/in/tao-wei-734a121b/)** — Professor of ECE, Clemson University; directs the NEXT Lab (domain-specific accelerators, reconfigurable computing, applied ML). Leads FLM kernel strategy.
- **[Ken Qing Yang](https://www.linkedin.com/in/ken-qing-yang-3b30331)** — Distinguished Engineering Professor, University of Rhode Island. 30+ years in computer architecture; serial entrepreneur behind four deep-tech startups, including VeloBit (acquired by Western Digital) and [DapuStor](https://en.dapustor.com/).
- **[Zhenyu (Alfred) Xu](https://www.linkedin.com/in/zhenyu-xu-327bb71b4)** — Research Assistant Professor, Clemson University. Accelerator design and on-device AI inference, with hardware–software co-optimization across FPGA, CGRA, and AI accelerators.

---

## 🚀 2025–2026 — Building in the Open

First public commit landed **June 16, 2025**; **v0.1.0** nine days later. The design principle was borrowed from tools developers already liked — *"Think Ollama, but deeply optimized for NPUs."* Install in seconds, `flm run <model>`, done.

FLM was **MIT-licensed and open source from the start**, built on **[IRON](https://github.com/amd/iron)**, the open-source NPU compiler technology developed and released by AMD's **Research and Advanced Development (RAD) Group**. That open foundation, plus close alignment with **[Lemonade](https://lemonade-server.ai/)** (AMD's open-source inference initiative), drove adoption across developers and ISVs.

| Date | Milestone |
|---|---|
| **Jun 16, 2025** | First public commit |
| **Jun 25, 2025** | v0.1.0 — first tagged release |
| **Oct 1, 2025** | Integrated into AMD's Lemonade Server 🍋 |
| **Mar 11, 2026** | Linux support launches 🐧 |
| **Jul 17, 2026** | **FastFlowLM joins AMD** 🎉 |
| **Aug 7, 2026** | Repo moves to the **ROCm** organization |
| **Aug 10, 2026** | **v1.0.0** 🎊 |

Over that span: 60+ releases, 1,500+ commits, ~38 contributors. The runtime grew past text to **Vision, Audio, Embedding, and MoE** models with context up to **256k tokens** — in a **17 MB** package.

---

## 🤝 July 17, 2026 — Joining AMD

The FastFlowLM team [joined AMD](https://www.amd.com/en/blogs/2026/fastflowlm-joins-amd-to-advance-ai-inference.html), joining the **AMD Artificial Intelligence Group** to accelerate the client and workstation AI software stack and Day-0 enablement of new models. AMD called it:

> "another key step in our strategy to advance AI performance and efficiency across the stack."

And on the open ecosystem:

> "We remain committed to investing in this open ecosystem, and we're thrilled to build the future of on-device AI together."

Joining AMD did not close the project. FLM stays **open source under the MIT license**, and its NPU kernels remain **free for any use, including commercial use**.

---

## 🏠 August 2026 — Home in ROCm

On **August 7, 2026** the repo moved from `FastFlowLM/FastFlowLM` to **[ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM)**, alongside AMD's open-source ROCm organization. **v1.0.0** shipped three days later.

The mission is unchanged: **turn idle NPU silicon into instant, private, power-efficient AI.**

---

## 📚 References

- **[AMD Blog — FastFlowLM joins AMD to advance AI inference](https://www.amd.com/en/blogs/2026/fastflowlm-joins-amd-to-advance-ai-inference.html)** *(primary source)*
- [Phoronix — FastFlowLM Developers Join AMD](https://www.phoronix.com/news/FastFlowLM-Joins-AMD)
- [fastflowlm.com](https://fastflowlm.com/) · [Docs](https://fastflowlm.com/docs) · [Team](https://fastflowlm.com/team/)
