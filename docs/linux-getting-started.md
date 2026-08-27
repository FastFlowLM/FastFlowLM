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
   ```
   *    soft    memlock    unlimited
   *    hard    memlock    unlimited
   ```
- Reboot system

---

### Arch Linux

Arch requires both the kernel-side `amdxdna` driver and the XRT userspace plugin. `flm validate` can see the NPU through `/dev/accel/accel0`, but `flm run` uses XRT (`xrt::device(0)`), so both layers must be working.

#### 1. Install the runtime packages

```sh
sudo pacman -Syu
sudo pacman -S linux-headers linux-firmware-other xrt xrt-plugin-amdxdna
```

Install `amdxdna-dkms` from the AUR using your preferred AUR workflow, then rebuild the module for the running kernel if needed:

```sh
sudo dkms autoinstall -k "$(uname -r)"
sudo depmod -a
sudo reboot
```

If you need to rebuild a specific DKMS version, check `dkms status` and use that version explicitly.

If you run a non-default kernel, install the matching headers package instead, for example `linux-zen-headers` for `linux-zen`.

#### 2. Confirm the DKMS driver is selected

After rebooting, make sure `modinfo` resolves to the DKMS module rather than the in-tree kernel module:

```sh
modinfo -F filename amdxdna
```

Expected output should contain `updates/dkms`, for example:

```text
/lib/modules/6.19.13-arch1-1/updates/dkms/amdxdna.ko.zst
```

If it points under `kernel/drivers/accel/amdxdna/`, the stock kernel driver is still being used. Recheck matching headers, the DKMS build status, `sudo depmod -a`, and reboot.

#### 3. Confirm XRT sees the NPU

```sh
xrt-smi examine
```

The output should list an NPU under the device table. If `flm validate` succeeds but `flm run` fails with `No such device with index '0'`, XRT usually cannot see the NPU. Confirm `xrt-plugin-amdxdna` is installed and that `xrt-smi examine` lists the device before trying `flm run` again.

#### 4. Firmware note for Linux 6.19

Some Arch `linux-firmware-other` versions include both `npu.sbin.1.0.0.63.zst` and `npu.sbin.1.1.2.64.zst` for `17f0_10`. On the stock 6.19 in-tree `amdxdna` driver, forcing `npu.sbin.zst` to the 1.1 firmware can make the NPU disappear because the driver expects the older firmware protocol. The DKMS driver can use the protocol-7 firmware through `npu_7.sbin.zst`.

In short: if 1.1 firmware breaks probing on stock 6.19, do not keep forcing the `npu.sbin.zst` symlink. Use `amdxdna-dkms` or a kernel with the newer `amdxdna` driver, then verify `flm validate` reports firmware version `1.1.x`.

---

### Building from Source

1. Ensure all required development packages are installed:
   ```sh
   sudo apt install ninja
   sudo apt install libavformat-dev  libavutil-dev libavcodec-dev libswresample-dev libswscale-dev libxrt-dev uuid-dev libdrm-dev
   ```

2. Clone the repository and pull all submodules:
   ```sh
   git clone --recursive https://github.com/ROCm/FastFlowLM.git
   cd FastFlowLM
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

If validation passes but running a model fails, check XRT separately:

```sh
xrt-smi examine
```

`flm validate` uses the kernel DRM device directly, while `flm run` uses XRT. On Linux, a passing validation does not guarantee XRT can open device index `0`.
