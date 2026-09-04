---
layout: docs
title: Phi-4 AIE4 Implementation
nav_order: 5
---

# Phi-4 on AIE4 — what was built

FastFlowLM had no AIE4 backend. This work adds one: Phi-4-mini runs on an AMD
AIE4 NPU by calling `ryzenai-corelib`'s C ABI. Everything below describes what
ships, not what was attempted.

For what to watch out for, read [Phi-4 AIE4 Caveats](../phi4_aie4_caveats/).

---

## The shape of it

```
flm.exe ──(runtime, absolute path)──► ryzenai_corelib.dll ──► AIE4 NPU
   │
   └── phi4_corelib_aie4  ── the backend added by this work
```

`flm.exe` **never links** `ryzenai_corelib.lib`. The DLL is resolved at run
time by absolute path, so a build without the AIE4 component installed starts
normally and reports the backend as unavailable. **There is no silent
fallback** — a Phi-4 AIE4 tag either runs on the NPU or fails loudly.

## The corelib surface actually used

Pinned to corelib `e5258d2`, ABI version **0.1.0**, gated on an exact
`major.minor.patch` match while the major version is 0.

| Group | Entry points |
|---|---|
| Compute | `matmul_bf16`, `flat_mha_bf16`, `ssmlp_bf16` |
| Padding | `matmul_bf16_pad_shape`, `flat_mha_bf16_pad_rows`, `ssmlp_bf16_pad_rows` |
| Weights | `matmul_bf16_weights_create_onnx` / `_get_data`, `ssmlp_bf16_weights_create_onnx` / `_get_data` |
| Tensors | `create_device_tensor`, `tensor_write`, `tensor_read`, `tensor_get_byte_size`, `tensor_get_data_type` |
| Stream | `create_stream`, `stream_synchronize` |
| Lifecycle | `get_version`, `has_device_context`, `selftest_dependencies`, `object_release`, `cleanup` |
| Errors | `get_last_error_message`, `status_to_string` |

Two conventions that bite if forgotten:

- **Counts and offsets are elements, never bytes.** `tensor_write` and
  `tensor_read` take element counts and convert dtype on the way.
- **Exactly two host-side conversions exist.** FP16 → FP32 widening (lossless,
  bounds-checked), and FP32 → BF16 round-to-nearest-even for the SSMLP epsilon
  and the two norm weights. Everything else crosses the boundary unconverted.

## Model constants

From `src/include/models/phi4/phi4_corelib_constants.hpp`, all validated
against the published ONNX:

| | |
|---|---|
| Layers | 32 |
| Hidden / intermediate | 3072 / 8192 |
| Query heads / KV heads | 24 / 8 |
| Head size | 128 |
| Query dim / KV dim | 3072 / 1024 |
| Vocabulary | 200064 |
| Quantisation group | 128 |
| RoPE dimension | 96 |
| RMS epsilon | 1e-5 |
| Max sequence | 4096 |
| **Max decode window** | **4095** (`kMaxSequenceLength - 1`) |

The last row is not a typo — see the caveats page.

## What ships

**Model tag** `phi4-mini-it-aie4:4b`, built from
[`amd/phi-4-mini-instruct-oga-dml`](https://huggingface.co/amd/phi-4-mini-instruct-oga-dml)
at revision `e751fb68`, plus four overlay files whose SHA-256 values are pinned
in the catalog.

**Installer**: the AIE4 runtime is an **optional MSI feature**
(`Aie4Feature`). It stages `ryzenai_corelib.dll`, DynamicDispatch and RyzenMM
into an `aie4` directory beside `flm.exe`, together with `aie4-closure.txt` —
a manifest naming the corelib version, the corelib DLL's SHA-256, and the name
and SHA-256 of every DLL staged with it.

## Execution notes

- **Four stream synchronizes per layer.** AIE4 completion is not ordered by
  submission, so a dispatch whose input another dispatch is still writing must
  be separated by an explicit synchronize.
- **RoPE is applied host-side** as a strided gather followed by a single
  `tensor_write`. FP32 scale tensors are rejected rather than silently narrowed.
- **The host stays FP32**; corelib narrows on write.
- **Unbounded generation is capped.** `kMaxDecodeWindow` bounds every decode
  path, which closed a defect where an ungated generation could exhaust the
  server.

## What was verified

**On `xcomedusad-43`** — a 20-step acceptance run against the real model in one
uninterrupted pass. Five single-turn probes, five coherent and factually
correct, roughly ten seconds each after a ~20 s model load. Recorded step by
step and criterion by criterion in the
[acceptance record](../benchmarks/phi4_aie4_acceptance/), with its own
corrections in the
[provenance page](../benchmarks/phi4_aie4_acceptance_provenance/).

Final acceptance tally: **16 steps met, 1 not met, 3 not exercised**;
**12 of 52 criteria met, 12 partial, 2 not met, 26 not exercised**.

**On `xcomedusad-44`** — a machine that took no part in development, given the
shipped MSI, the model package, and a conda environment reproduced from an
explicit package list. The model loads and generates. Read the caveats page
before concluding what this does and does not prove.

**The instruments that produced both are not in this branch.** They ran only on
a machine with an AIE4 device and the real model, so the harnesses, the
PowerShell campaign and its raw records were removed rather than merged; the
paths cited in the two pages above resolve at commit `5c93aadb`. The
consequence — that these figures are reports rather than artifacts you can
re-derive — is
[caveat 11](../phi4_aie4_caveats/#11-the-hardware-campaign-is-not-in-this-branch).

## Scale

| | |
|---|---|
| Files changed | 110 (**32 of them tests**, carrying 23,912 of the added lines) |
| Lines | +49,635 / −289 |
| `ctest` on `xcomedusad-43` | **13 passed, 3 skipped, 0 failed** (16 registered) |
| Python tooling | **312 passed** |

The test share is the point rather than an accident: most real defects in this
work were found by tests and review, not by the compiler.

The three skips are `test_phi4_hardware` and `test_fatal_child` (they want
`FLM_AIE4_HARDWARE=1`) and `test_packaged_runtime` (it wants a staged
installer). `test_real_corelib` is not among them — it loaded the real
`ryzenai_corelib.dll` and passed in 11 s.

### Configuring this suite on the AIE4 target

**You must pass `-DFLM_PYTHON_EXECUTABLE` there.** On that machine `PATH`
resolves `python3` to a Cygwin symlink Windows cannot execute, so the probe
finds no usable interpreter and `test_phi4_continuation_calibration` is
deliberately registered as a *failing* test rather than silently skipped. The
script that used to supply this path was part of the hardware campaign and is
no longer in the branch, so it is now the caller's job:

```
cmake -S src/test/phi4_corelib_aie4 -B <build> -G "Visual Studio 17 2022" -A x64 \
  -DRYZENAI_CORELIB_INCLUDE_DIR=<corelib>/install-mirrored/include \
  -DRYZENAI_CORELIB_RUNTIME_DIR=<staged aie4 runtime> \
  -DXRT_INCLUDE_DIR=<xrt>/include -DXRT_LIB_DIR=<xrt>/lib \
  -DBOOST_INCLUDE_DIR=<conda>/Library/include \
  -DFLM_PYTHON_EXECUTABLE=<conda>/python.exe
cmake --build <build> --config Release
ctest -C Release
```

Omit that last flag and you get 15 passed, 1 failed — and the failure is the
configure, not the code. CMake says so at configure time; the warning is worth
reading.
