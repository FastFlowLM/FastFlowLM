#!/usr/bin/env python3
"""Build FastFlowLM NPU kernels from source. See CONVENTION.md.

    python kernels/build_kernels.py --check-toolchain
    python kernels/build_kernels.py --list
    python kernels/build_kernels.py --family granite --out build/kernels
    python kernels/build_kernels.py --family granite --validate

Runs standalone, with no CMake. `src/CMakeLists.txt` shells out to it when
FLM_BUILD_KERNELS=ON, and to `--check-toolchain` at configure time so a missing
toolchain is one sentence rather than an aiecc traceback halfway through a
build.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
JIT_CACHE = Path.home() / ".npu" / "cache"


@dataclass
class Design:
    """One artefact: an IRON design plus the arguments that select its shape."""
    out: str
    module: str
    fn: str
    kwargs: dict = field(default_factory=dict)


FAMILIES: dict[str, list[Design]] = {
    "granite": [
        Design("norm_qkv_rope", "granite_qkv_wide", "granite_qkv_wide"),
        Design("attn_o", "granite_attn_o", "granite_attn_o", {"seq": 64}),
        Design("norm_qkv", "granite_norm_gemv", "granite_norm_gemv", {"tensor": "qkv"}),
        Design("norm_gate_up", "granite_norm_gemv", "granite_norm_gemv",
               {"tensor": "gate_up"}),
        Design("swiglu_down", "granite_swiglu_down", "granite_swiglu_down"),
    ],
}

# The directory name must match src/xclbins/<Model>/ so the built artefacts
# install through the rule that already exists for the shipped ones.
MODEL_DIR = {"granite": "Granite-4.2-3B-NPU2"}


# ---------------------------------------------------------------- toolchain

def probe_toolchain() -> tuple[bool, str]:
    """Report whether this interpreter can build, and say precisely why not.

    Deliberately does not try to activate anything. `iron_env.cmd` is
    Windows-only and machine-specific, and activating it in a subprocess would
    not affect this process anyway. Activation is the caller's job; saying
    exactly what is missing is ours -- that is what makes CMake's error message
    useful.
    """
    try:
        import aie.iron  # noqa: F401
    except ImportError as e:
        return False, (
            f"the `aie` package is not importable from {sys.executable} ({e}). "
            "Activate the mlir-aie/IRON environment before configuring, or pass "
            "-DFLM_KERNELS_PYTHON=<path to the IRON python>.")
    try:
        import importlib.metadata as md
        ver = md.version("mlir_aie")
    except Exception:
        ver = "unknown"
    try:
        from aie.utils import config
        hdr = Path(config.cxx_header_path())
        if not (hdr / "aie_kernels").is_dir():
            return False, (f"`aie` {ver} is importable but its kernel headers are "
                           f"missing at {hdr / 'aie_kernels'}.")
    except Exception as e:
        return False, f"`aie` {ver} is importable but unusable: {e!r}"
    return True, f"mlir-aie {ver}, python {sys.version.split()[0]}"


def toolchain_manifest() -> dict:
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except Exception:
            return None

    return {
        "mlir_aie": ver("mlir_aie"),
        "ml_dtypes": ver("ml_dtypes"),
        "numpy": ver("numpy"),
        "python": sys.version.split()[0],
        "aiecc_flags": ["--alloc-scheme=basic-sequential"],
    }


# ------------------------------------------------------------------- build

def _snapshot() -> set[Path]:
    return set(JIT_CACHE.iterdir()) if JIT_CACHE.is_dir() else set()


def _new_cache_dir(before: set[Path]) -> Path | None:
    """Find what the JIT just produced, by difference.

    Not by matching strings in the emitted MLIR: that is what
    LLMNpuTest's export_design.py does, and its own docstring calls it fragile.
    A set difference cannot be wrong, and it does not depend on IRON internals
    that change between versions. If the design was already cached no directory
    appears, and the caller falls back to the newest matching one.
    """
    after = _snapshot()
    fresh = [p for p in after - before if (p / "final.xclbin").is_file()]
    if fresh:
        return max(fresh, key=lambda p: (p / "final.xclbin").stat().st_mtime)
    cached = [p for p in after if (p / "final.xclbin").is_file()]
    return max(cached, key=lambda p: (p / "final.xclbin").stat().st_mtime) if cached else None


def build_family(family: str, out_root: Path, validate: bool) -> int:
    designs = FAMILIES[family]
    fam_dir = HERE / family
    geometry = json.loads((fam_dir / "geometry.json").read_text(encoding="utf-8"))

    sys.path.insert(0, str(fam_dir / "iron"))
    sys.path.insert(0, str(HERE / "common"))
    # Generated entry points belong in the build tree, never beside tracked
    # source. granite_gemv.py reads this.
    gen = out_root / "_generated"
    os.environ["GRANITE_GEN_DIR"] = str(gen)

    import aie.iron as iron
    from aie.iron.device import from_name
    iron.set_current_device(from_name("npu2", n_cols=None))

    dest = out_root / MODEL_DIR[family]
    dest.mkdir(parents=True, exist_ok=True)
    entries, failures = [], 0

    for d in designs:
        mod = __import__(d.module)
        builder = getattr(mod, "build_artifact", None)
        if builder is None:
            print(f"  SKIP {d.out}: {d.module} has no build_artifact(geometry, **kw); "
                  f"see CONVENTION.md -- the build must not need model weights")
            failures += 1
            continue
        print(f"  building {d.out} ...", flush=True)
        before = _snapshot()
        try:
            builder(geometry, **d.kwargs)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  FAIL {d.out}: {e.__class__.__name__}: {e}")
            failures += 1
            continue
        src = _new_cache_dir(before)
        if src is None:
            print(f"  FAIL {d.out}: no final.xclbin appeared in {JIT_CACHE}")
            failures += 1
            continue
        rec = {"design": d.out, "module": d.module, "kwargs": d.kwargs, "files": {}}
        for name, suffix in (("final.xclbin", ".xclbin"), ("insts.bin", ".insts.bin")):
            s = src / name
            if not s.is_file():
                continue
            t = dest / f"{d.out}{suffix}"
            shutil.copy2(s, t)
            rec["files"][t.name] = {
                "bytes": t.stat().st_size,
                "sha256": hashlib.sha256(t.read_bytes()).hexdigest(),
            }
        (dest / f"{d.out}.json").write_text(json.dumps(rec, indent=2) + "\n",
                                            encoding="utf-8")
        entries.append(rec)
        print(f"    -> {', '.join(rec['files'])}")

        if validate:
            checker = getattr(mod, "validate_artifact", None)
            if checker is None:
                print(f"    (no validate_artifact in {d.module})")
            else:
                ok = checker(geometry, **d.kwargs)
                print(f"    validate: {'PASS' if ok else 'FAIL'}")
                failures += 0 if ok else 1

    geom_hash = hashlib.sha256(
        json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:16]
    (dest / "manifest.json").write_text(json.dumps({
        "family": family,
        "model_dir": MODEL_DIR[family],
        "geometry_sha256_16": geom_hash,
        "toolchain": toolchain_manifest(),
        "artifacts": entries,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  manifest -> {dest / 'manifest.json'}")
    return failures


# -------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check-toolchain", action="store_true",
                    help="report whether this interpreter can build, and exit")
    ap.add_argument("--list", action="store_true", help="print the design map")
    ap.add_argument("--family", choices=sorted(FAMILIES))
    ap.add_argument("--out", type=Path, default=Path("build/kernels"))
    ap.add_argument("--validate", action="store_true",
                    help="also check each design against a host reference "
                         "(needs an NPU and the model weights)")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if a.check_toolchain:
        ok, msg = probe_toolchain()
        print(msg)
        return 0 if ok else 1

    if a.list:
        for fam, ds in FAMILIES.items():
            print(f"{fam} -> {MODEL_DIR[fam]}/")
            for d in ds:
                extra = f"  {d.kwargs}" if d.kwargs else ""
                print(f"  {d.out:16} {d.module}.{d.fn}{extra}")
        return 0

    if not a.family:
        ap.error("one of --check-toolchain, --list or --family is required")

    ok, msg = probe_toolchain()
    if not ok:
        print(f"toolchain unusable: {msg}", file=sys.stderr)
        return 1
    print(f"toolchain: {msg}")
    failures = build_family(a.family, a.out.resolve(), a.validate)
    print("OK" if failures == 0 else f"{failures} failure(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
