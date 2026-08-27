---
layout: docs
title: Get Started (Linux)
nav_order: 2
has_children: false
---

# Linux NPU Support

This article will teach you how to run LLMs on your **AMD XDNA2 NPU** on Linux using **FastFlowLM**.  
Get set up and then show us what you build!

**Date:** March 5, 2026  
**Authors:** [Lemonade-server🍋](https://lemonade-server.ai/) and FastFlowLM contributors

## 📢 FastFlowLM Linux Support

[FastFlowLM](https://github.com/ROCm/FastFlowLM) is a lightweight LLM runtime optimized for **AMD NPUs**.  
Today, FastFlowLM is adding support for **Ubuntu, Arch, and other distros** to enable **fast, low-power LLMs** on **Ryzen™ AI PCs that run Linux**.

This article will help you:

- Understand **Linux NPU support status** and required platform versions
- Install the **FLM + driver stack** for your distribution
- Validate your setup with `flm validate`
- Fix common **firmware, driver, and memlock issues**

---

## ⚙️ Hardware Requirements

### Supported processors

FastFlowLM on Linux requires an **AMD XDNA2 NPU**.

| Ryzen AI family | Codename | Status |
|---|---|---|
| Max 300-series | Strix Halo | Supported |
| 300-series | Kraken Point, Strix Point | Supported |
| 400-series | Gorgon Point | Supported |
| Z2 Extreme | Handheld devices | Supported |

> **Note:** Ryzen AI 7000 / 8000 / 200-series chips have **XDNA1**, which is **not supported**.

---

## 🧰 Software Requirements

### Runtime stack

The NPU requires specific firmware, kernel version, driver, and runtime software to function.  
The quickstart guide below will help you install these requirements.

| Item | Requirement |
|---|---|
| NPU firmware | Version 1.1.0.0 or later |
| Kernel + driver | Kernel **7.0+** with `amdxdna`, or `amdxdna-dkms` |
| Runtime | FastFlowLM installed |
| Memlock limit | Must be high enough for NPU execution |

---

## 🚀 Quickstart

## Supported Distributions
- Ubuntu 24.04 LTS
- Ubuntu 25.10
- Ubuntu 26.04
- Arch Linux
- Other (Generic Linux)

---

## 1. Prerequisites
- `amdxdna` driver (included in kernel 7.0+, or via `amdxdna-dkms`)
- NPU firmware version 1.1.0.0 or later
- Python 3.8+
- XRT stack from AMD

---

## 2. System Preparation

### Ubuntu (24.04, 25.10)

#### 1. Add the AMD XRT PPA (Required for NPU/XDNA)
The AMD XRT stack is a prerequisite for NPU support. Add AMD's PPA:
```sh
sudo add-apt-repository ppa:lemonade-team/stable
sudo apt update
```
See [lemonade-team/stable PPA](https://launchpad.net/~lemonade-team/+archive/ubuntu/stable) for details.

#### 2. Install XRT and NPU Drivers
```sh
sudo apt install libxrt-npu2 amdxdna-dkms
```

#### 3. Reboot
```sh
sudo reboot
```

#### 4. Install FastFlowLM
- Download the latest `.deb` package from the [Releases page](https://github.com/ROCm/FastFlowLM/releases):

```sh
sudo apt install ./fastflowlm*.deb
```

#### 5. (NPU) Check memlock limit
- Run:
   ```sh
   ulimit -l
   ```
- If not `unlimited`, add to `/etc/security/limits.conf`:

   `*    soft    memlock    unlimited`   
   `*    hard    memlock    unlimited`
- Reboot system

---

### Ubuntu 26.04, Arch, and Others

For Ubuntu 26.04 and other distributions, check this [Linux NPU setup guide](https://lemonade-server.ai/flm_npu_linux.html).

#### Arch Linux

Arch users need the kernel driver, matching kernel headers, XRT, and the AMD XDNA XRT plugin:

```sh
sudo pacman -Syu
sudo pacman -S linux-headers linux-firmware-other xrt xrt-plugin-amdxdna
```

Install `amdxdna-dkms` from the AUR using your preferred AUR workflow, then reboot. If needed, rebuild the DKMS module for the running kernel:

```sh
sudo dkms autoinstall -k "$(uname -r)"
sudo depmod -a
sudo reboot
```

If you need to rebuild a specific DKMS version, check `dkms status` and use that version explicitly.

After rebooting, confirm the DKMS module is selected:

```sh
modinfo -F filename amdxdna
```

The path should contain `updates/dkms`. If it points under `kernel/drivers/accel/amdxdna/`, the stock in-tree driver is still being used.

Then confirm XRT can see the NPU:

```sh
xrt-smi examine
```

If `flm validate` passes but `flm run` fails with `No such device with index '0'`, XRT does not see a device. Make sure `xrt-plugin-amdxdna` is installed and `xrt-smi examine` lists the NPU.

> **Arch firmware note:** Some `linux-firmware-other` versions ship both 1.0 and 1.1 NPU firmware for `17f0_10`. On stock Linux 6.19, forcing `npu.sbin.zst` to 1.1 firmware can make the NPU disappear because the in-tree driver expects the older firmware protocol. Use `amdxdna-dkms` or a newer kernel that supports the protocol-7 firmware path, then verify `flm validate` reports firmware `1.1.x`.

---

### Building from Source

1. Clone the repository:
   ```sh
   git clone https://github.com/ROCm/FastFlowLM.git
   cd FastFlowLM
   ```
2. Build:
   ```sh
   cd src
   cmake --preset linux-default
   cmake --build --preset linux-default -j$(nproc)
   cmake --install --preset linux-default
   ```

#### Advanced Build Options

**Static Build with Bundled XRT/XDNA**

To build a fully static binary that bundles XRT and the XDNA driver (no system XRT required):

```sh
cd src
cmake --preset linux-static
cmake --build build -j$(nproc)
sudo cmake --install build
```

This will automatically fetch and build XRT (v2.21.75) and the XDNA driver from source if not found on your system. The resulting binary is fully self-contained and more portable.

---

## 3. Validating NPU Setup

To validate your NPU setup, run:
```sh
flm validate
```
You should see output similar to:
```
[Linux]  Kernel: 7.0.0-rc1-00052-g27936bfca73d
[Linux]  NPU: /dev/accel/accel0
[Linux]  NPU FW Version: 1.1.2.64
[Linux]  Memlock Limit: infinity
```

On Linux, `flm validate` checks the kernel DRM path. `flm run` uses XRT. If validation succeeds but running a model fails with `No such device with index '0'`, run `xrt-smi examine` and install the XRT AMD XDNA plugin for your distribution.

---

## 📚 Additional resources

- [Lemonade-server🍋](https://lemonade-server.ai/)
- [Lemonade GitHub issues](https://github.com/lemonade-ai/lemonade/issues)
- [Lemonade Discord](https://discord.gg)

---