# NPU kernel source

Source for AIE kernels, built on request. See [CONVENTION.md](CONVENTION.md)
for what this directory is and the rules it follows.

**Nothing here is built by default.** `FLM_BUILD_KERNELS` is `OFF`, and with it
off the `flm` build is identical to one in which this directory does not exist.

## State

| | |
|---|---|
| kernel source in tree | yes |
| builds from source | yes, `-DFLM_BUILD_KERNELS=ON` |
| validated against a host reference on NPU2 | yes — see `granite/README.md` |
| **dispatched by `flm`** | **not yet** — see *The gap* below |

## Prerequisites

The [mlir-aie](https://github.com/Xilinx/mlir-aie) / IRON toolchain, which
supplies the `aie` Python package, the `aiecc` driver and the Peano (LLVM-AIE)
backend. Versions are pinned in [requirements.txt](requirements.txt) and
recorded again in every build's `manifest.json`.

Activate that environment first — the build does not try to activate it for
you, because activation does not survive a subprocess.

## Build

```shell
# with the IRON environment active
python kernels/build_kernels.py --check-toolchain
python kernels/build_kernels.py --family granite --out build/kernels
```

or through CMake:

```shell
cmake -S src -B src/build -DFLM_BUILD_KERNELS=ON
cmake --build src/build --config Release
```

If the toolchain is not usable, configuring fails with one sentence naming both
fixes, rather than an `aiecc` traceback halfway through a build. Pass
`-DFLM_KERNELS_PYTHON=<path>` to point at a specific interpreter.

Output mirrors `src/xclbins/<Model>/` so it installs through the rule that
already exists:

```
build/kernels/Granite-4.2-3B-NPU2/
    norm_qkv_rope.xclbin   norm_qkv_rope.insts.bin   norm_qkv_rope.json
    attn_o.xclbin          ...
    manifest.json          toolchain versions + sha256 per artefact
```

## Validate

```shell
python kernels/build_kernels.py --family granite --validate
```

Runs each design against a host reference built from the same bytes. **This is
a hardware test**: it needs an NPU and the model weights, and CI is not asked to
run it.

## The gap

These artefacts are complete and numerically validated, but `flm` cannot
dispatch them yet, and the reason is specific.

`src/include/npu_utils/npu_utils_xrt.hpp` builds its ELF from a control
sequence assembled **on the host** by `npu_sequence`
(`src/include/npu_utils/npu_instr_utils.hpp`). IRON emits that same control code
as a prebuilt `insts.bin` at build time. Bridging the two is roughly one
function: load `insts.bin` and hand it to `aiebu_assembler_get_elf` in place of
the assembled sequence.

That is deliberately not here. **This is a build and convention change, not a
runtime change** — the two are worth deciding separately, and the runtime side
follows only if the convention is wanted.
