# A convention for NPU kernel source in this repository

This document proposes how contributed AIE kernel *source* lives in
FastFlowLM. It is deliberately separable from the kernels that come with it:
you can accept or reject the convention on its own terms, and `kernels/granite`
is just the first worked example.

## What this is, and what it is not

`kernels/` holds **source**. `src/xclbins/` continues to hold the shipped
binaries, and nothing here changes, moves, re-derives or replaces any of them.

**No compiled artefact is ever committed under `kernels/`.** Artefacts are build
outputs. They land in the build directory and are installed from there, exactly
as the checked-in ones are.

## Why source and not a binary

An `.xclbin` is only valid for the toolchain that built it. A committed binary
carries no record of what produced it and cannot be rebuilt when the toolchain
moves; it rots silently, and the rot is invisible until a user hits it. Source
plus a recorded toolchain fingerprint is the only representation that survives
a version bump.

That is the whole argument for this directory. The granite kernels are the
occasion, not the point.

## Layout

One directory per model family, named to match `src/xclbins/<Model>/`:

```
kernels/
├── CONVENTION.md          this file
├── README.md              how to build
├── LICENSE                MIT
├── requirements.txt       pinned toolchain packages
├── build_kernels.py       the only entry point
├── common/                shared host-side helpers
└── <family>/
    ├── README.md          geometry, measured numbers, what was rejected
    ├── geometry.json      model dimensions -- so the build needs no weights
    ├── iron/              IRON Python: placement and data movement
    └── aie/               C++ that runs on the AIE cores
```

**`iron/` and `aie/` are split because they are two different review surfaces.**
`aie/` is device C++: it is what actually executes, and it is the surface that
matters for provenance and for correctness. `iron/` is host Python that only
describes where things are placed and how data moves. **Read `aie/` first.**

## Rules

**Never a build dependency.** No family may become a dependency of `flm`. With
`FLM_BUILD_KERNELS=OFF` — the default — the build must be byte-identical to one
in which `kernels/` does not exist.

**The build must not need model weights**, network access, or any path outside
the repository other than the toolchain itself. Shapes come from
`geometry.json`. This is what makes the option safe to enable in a container,
and it is why `geometry.json` exists at all.

**No generated file is committed.** Generators live in `iron/`; their output
goes to the build directory.

**The toolchain is declared, not assumed.** Every build writes a
`manifest.json` recording the exact package versions, compiler flags and a
sha256 per artefact. A consumer that finds a fingerprint mismatch must refuse
rather than dispatch a mismatched pair to the NPU.

**Licence.** Everything under `kernels/` is MIT, matching
`LICENSE_RUNTIME.txt`, with `SPDX-License-Identifier: MIT` in every file.

## Provenance

The kernels here were written from the public MLIR-AIE/IRON examples and
published AIE2P documentation. **No shipped `.xclbin` was disassembled and no
closed component was reverse-engineered.** The q4nx container layout was
derived by inverting a published packer and cross-checked against published
model files. (The companion granite PR documents the same derivation in
`src/include/models/granite/q4nx_host.hpp`, if it has landed; this PR does not
depend on it.)

## What a new family must ship

A `README.md` with measured numbers against a host reference, a `geometry.json`,
and a `validate` path that can be run on hardware. A kernel with no reference
comparison is not a contribution; it is a claim.
