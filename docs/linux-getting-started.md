# Linux Getting Started Guide

This guide will help you get started with FastFlowLM on Linux, including setup for various distributions and NPU (Neural Processing Unit) support.

## Supported Distributions
- Ubuntu 24.04 LTS
- Ubuntu 25.10
- Arch Linux
- Other (Generic Linux)

---

## Prerequisites
- `amdxdna` driver (included in kernel 7.0+, or via `amdxdna-dkms`)
- NPU firmware version 1.1.0.0 or later
- Python 3.8+
- XRT stack from AMD

---

## System Preparation


### Ubuntu (24.04, 25.10)

#### 1. Add the AMD XRT PPA (Required for NPU/XDNA)
The AMD XRT stack is a prerequisite for NPU support. Add AMD's PPA:
```sh
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update
```
See [amd-team/xrt PPA](https://launchpad.net/~amd-team/+archive/ubuntu/xrt) for details.

#### 2. Install XRT and NPU Drivers
```sh
sudo apt install libxrt-npu2 amdxdna-dkms
```

#### 3. Reboot
```sh
sudo reboot
```

#### 4. Install FastFlowLM
- Download the latest `.deb` package from the [Releases page](https://github.com/FastFlowLM/FastFlowLM/releases):

```sh
sudo apt install ./fastflowlm*.deb
```

#### 5. (NPU) Check memlock limit
- Run:
   ```sh
   ulimit -l
   ```
- If not `unlimited`, add to `/etc/security/limits.conf`:
   ```
   *    soft    memlock    unlimited
   *    hard    memlock    unlimited
   ```
- Reboot system

---

### Building from Source

1. Ensure all required development packages are installed:
   ```sh
   sudo apt ninja
   sudo apt install libavformat-dev  libavutil-dev libavcodec-dev libswresample-dev libswscale-dev libxrt-dev uuid-dev libdrm-dev
   ```

2. Clone the repository and pull all submodules:
   ```sh
   git clone https://github.com/FastFlowLM/FastFlowLM.git
   cd FastFlowLM
   git submodule init
   git submodule update --recursive --init
   ```
3. Build:
   ```sh
   cd src
   cmake --preset linux-default
   cd build
   cmake --build . -j$(nproc)
   sudo cmake --install .
   ```

---

## Validating NPU Setup

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