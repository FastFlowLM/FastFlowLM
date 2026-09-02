r"""Any granite-4.2-3B projection on the NPU: y[N] = W[N, K] @ x[K], W in q4nx q4.

Every matmul in granite is the same kernel at a different (N, K) -- the seven
projections plus lm_head differ only in shape, so one design covers the lot:

    q_proj    2560 x 2560     gate_proj  8192 x 2560     lm_head 100352 x 2560
    k_proj     512 x 2560     up_proj    8192 x 2560
    v_proj     512 x 2560     down_proj  2560 x 8192   <- the only K != 2560
    o_proj    2560 x 2560

**(N, K) cannot be read off the file.** q4nx stores a tiled shape
`[N/32 * K/256, 5120]`, and `gate_proj` and `down_proj` are BOTH `[2560, 5120]`
-- 256 tile-rows x 10 K-tiles against 80 x 32. The two factor differently and a
GEMV against the wrong factoring is silently a different (wrong) matmul, not an
error. So (N, K) comes from `config.json` and the product is checked against the
stored shape.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_gemv.py --tensor lm_head.weight
    python designs\granite_gemv\granite_gemv.py --all          :: every shape
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,
                      TaskGroup, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D
from aie.utils import config
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
AIE = HERE.parent / "aie"              # the device C++ this design compiles
# Generated entry points go to a build directory, never beside tracked source.
# GRANITE_GEN_DIR is set by build_kernels.py; the fallback keeps a bare
# `python granite_gemv.py` working for development.
GEN = Path(os.environ.get("GRANITE_GEN_DIR", "")) or (HERE.parent / "_generated")
sys.path.insert(0, str(HERE.parent.parent / "common"))
import q4nx  # noqa: E402

TILE_BYTES = q4nx.Q4_TILE_BYTES     # 5120
ROWS_PER_TILE = q4nx.TILE_ROWS      # 32
TILE_K = q4nx.TILE_K                # 256

# L1 is the cap on how many K tiles one call may hold: an element is
# TILES_PER_CALL * 5120 B and it is double-buffered, against a 63 KB budget
# (64 KB less ~1 KB of stack). 5 * 5120 * 2 = 51200 B fits; 6 would not.
MAX_TILES_PER_CALL = 5

# one result object per token: 32 floats, double-buffered
TILE_ROWS_BYTES = 32 * 4 * 2

MODEL = Path.home() / ".cache" / "openfflm" / "Granite-4.2-3B-NPU2"


# L1 less the worker stack (0xD00). Set from evidence, not from the 63 KB rule
# of thumb: batch 2 with 5 tiles per call needs 61952 B of buffers and builds and
# runs, which a 63 KB - stack budget (61184) would have wrongly rejected -- and
# rejecting it is not a safe error, it silently halves the DMA element and cost
# ~20% throughput before the cause was spotted.
L1_BUDGET = 64 * 1024 - 0xD00        # 62208, verified against 61952


def tiles_per_call(k_tiles: int, batch: int = 1, k: int = 0) -> int:
    """Largest divisor of k_tiles whose double-buffered weights fit in L1.

    Entry points are nearly free since the kernel body has vague linkage (see
    granite_gemv.h), so this maximises the DMA element: K = 2560 -> 5 tiles x 2
    entry points, K = 8192 -> 4 x 8.

    **Batching competes for the same L1.** The activation buffer is batch*K*2
    bytes and grows with the batch while the weights do not, so a batch that is
    free in DMA terms still shrinks the weight element. At K = 2560 a batch of 4
    takes 20 KB of activations, which pushes 5 tiles (51200 B double-buffered)
    over the budget -- measured as `allocated buffers exceeded available memory`,
    at MLIR level, not as anything the kernel could report.
    """
    fixed = batch * k * 2 + batch * TILE_ROWS_BYTES  # activations + results
    for d in range(min(MAX_TILES_PER_CALL, k_tiles), 0, -1):
        if k_tiles % d == 0 and d * TILE_BYTES * 2 + fixed <= L1_BUDGET:
            return d
    return 1


def projection_shape(name: str, cfg: dict) -> tuple[int, int]:
    """(N, K) for a tensor name, from the config -- NOT from the stored shape."""
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    hd = cfg.get("head_dim") or h // cfg["num_attention_heads"]
    q = cfg["num_attention_heads"] * hd
    kv = cfg["num_key_value_heads"] * hd
    if name == "lm_head.weight":
        return cfg["vocab_size"], h
    leaf = name.rsplit(".", 2)[-2] if name.endswith(".weight") else name
    return {
        "q_proj": (q, h), "k_proj": (kv, h), "v_proj": (kv, h), "o_proj": (h, q),
        "gate_proj": (i, h), "up_proj": (i, h), "down_proj": (h, i),
    }[leaf]


NULL_SRC = """// OpenFFLM -- DMA probe, group {i} (batch {batch}). GENERATED.
// No arithmetic: measures what the weight stream alone sustains, which
// separates 'the kernel is slow' from 'the memory path is slow'.
// SPDX-License-Identifier: MIT
#include "granite_gemv.h"

extern "C" {{
void granite_gemv_p{per_call}b{batch}_k{i}(const uint8_t *__restrict t,
                                   float *__restrict y) {{
  // One byte, so the load is not elided; the DMA has already moved the
  // whole object by the time this runs. No activation argument: without an
  // x stream the design spends all 16 shim MM2S channels on weights, which
  // is what lets it reach 16 cores where the GEMV caps at 8.
  y[0] += (float)t[0];
}}
}}
"""


def ensure_entry_points(n_entry: int, per_call: int,
                        null: bool = False, batch: int = 1,
                        k: int = 0) -> list[Path]:
    """Write one .cc per entry point.

    They must be separate translation units: IRON compiles the kernel source
    once per ExternalFunction, so several entry points in one .cc become several
    objects that each define every symbol and the link fails on duplicates.
    The shared body costs nothing extra per TU -- it is `inline`, so the copies
    merge into one COMDAT at link time.
    """
    out = []
    for i in range(n_entry):
        stem = "granite_null" if null else "granite_gemv"
        GEN.mkdir(parents=True, exist_ok=True)
        p = GEN / f"{stem}_p{per_call}b{batch}_k{i}.cc"
        lo, hi = i * per_call, (i + 1) * per_call - 1
        if null:
            src = NULL_SRC.format(i=i, per_call=per_call, batch=batch)
            if not p.is_file() or p.read_text(encoding='utf-8') != src:
                p.write_text(src, encoding='utf-8', newline=chr(10))
            out.append(p)
            continue
        src = (
            f"// OpenFFLM -- K tile group {i} (tiles {lo}..{hi}) of a granite "
            f"q4nx GEMV.\n"
            f"// GENERATED by granite_gemv.py -- edit the generator, not this.\n"
            f"// See granite_gemv.h.\n"
            f"// SPDX-License-Identifier: MIT\n"
            f"#define GRANITE_TILES_PER_CALL {per_call}\n"
            f"#define GRANITE_BATCH {batch}\n"
            f"#define GRANITE_K {k}\n"
            f'#include "granite_gemv.h"\n\n'
            f'extern "C" {{\n'
            f"GRANITE_GEMV_ENTRY({i})\n"
            f"}}\n"
        )
        # Only rewrite on change: IRON caches compiled kernels by source hash,
        # and rewriting identical bytes would still invalidate mtime-based caches.
        if not p.is_file() or p.read_text(encoding="utf-8") != src:
            p.write_text(src, encoding="utf-8", newline="\n")
        out.append(p)
    return out


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    # This tree's own device headers. Needed because the generated entry points
    # live in a build directory, so their quoted #include cannot resolve
    # relative to themselves.
    inc.append(str(AIE))
    return inc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_gemv(w: In, x: In, y: Out, *, tile_rows: CompileTime[int],
                 k: CompileTime[int], n_cores: CompileTime[int] = 1,
                 null: CompileTime[bool] = False,
                 per_call: CompileTime[int] = 5):
    # Explicit, not derived: see the note in granite_gemv32.py -- iron.jit's
    # cache key never sees a value computed inside the generator.
    k_tiles = k // TILE_K
    n_entry = k_tiles // per_call
    call_bytes = per_call * TILE_BYTES
    row_bytes = k_tiles * TILE_BYTES
    per_core = tile_rows // n_cores

    srcs = ensure_entry_points(n_entry, per_call, null, 1, k)
    tile_ty = np.ndarray[(call_bytes,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(k,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(ROWS_PER_TILE,), np.dtype[np.float32]]
    w_ty = np.ndarray[(tile_rows * row_bytes,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(tile_rows * ROWS_PER_TILE,), np.dtype[np.float32]]

    kernels = [
        ExternalFunction(
            f"granite_gemv_p{per_call}b1_k{i}",
            source_file=str(srcs[i]),
            arg_types=([tile_ty, acc_ty] if null else [tile_ty, x_ty, acc_ty]),
            include_dirs=_include_dirs(),
        )
        for i in range(n_entry)
    ]

    of_w = [ObjectFifo(tile_ty, name=f"w{c}", depth=2) for c in range(n_cores)]
    of_y = [ObjectFifo(acc_ty, name=f"y{c}", depth=2) for c in range(n_cores)]
    # One activation, broadcast. Private copies would want one shim MM2S channel
    # each and there are only 16 device-wide -- the weights need those.
    of_x = ObjectFifo(x_ty, name="x", depth=1)

    def core_body(win, xin, yout, *ks):
        # x is acquired once and held: the same activation feeds every tile-row.
        xe = None if null else xin.acquire(1)
        for _ in range_(per_core):
            ye = yout.acquire(1)
            for fn in ks:
                we = win.acquire(1)
                if null:
                    fn(we, ye)
                else:
                    fn(we, xe, ye)
                win.release(1)
            yout.release(1)
        if not null:
            xin.release(1)

    workers = [
        Worker(core_body,
               fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), *kernels],
               stack_size=0xD00)
        for c in range(n_cores)
    ]

    w_taps = TensorTiler2D.simple_tiler(
        (1, tile_rows * row_bytes), (1, per_core * row_bytes))
    y_taps = TensorTiler2D.simple_tiler(
        (1, tile_rows * ROWS_PER_TILE), (1, per_core * ROWS_PER_TILE))

    def sequence(a_w, a_x, c_y, w_prods, x_prod, y_conss):
        tg = TaskGroup()
        x_prod.fill(a_x, group=tg)
        for c in range(n_cores):
            w_prods[c].fill(a_w, tap=w_taps[c], group=tg)
            y_conss[c].drain(c_y, tap=y_taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence,
                 [w_ty, x_ty, y_ty,
                  [f.prod() for f in of_w], of_x.prod(), [f.cons() for f in of_y]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def reference(raw: bytes, x: np.ndarray, tile_rows: int, k_tiles: int,
              chunk: int = 256) -> np.ndarray:
    """The same GEMV on the host, from the same bytes, in float32.

    Chunked over tile-rows: lm_head dequantised to float32 in one piece would be
    100352 x 2560 x 4 B = 1.0 GB and pointless to hold at once.
    """
    xf = x.astype(np.float32)
    out = np.empty(tile_rows * ROWS_PER_TILE, np.float32)
    b_all = np.frombuffer(raw, dtype=np.uint8).reshape(tile_rows * k_tiles, TILE_BYTES)
    for lo in range(0, tile_rows, chunk):
        n = min(chunk, tile_rows - lo)
        b = b_all[lo * k_tiles:(lo + n) * k_tiles]
        w = q4nx._untile(q4nx._q4_tiles(b), n, k_tiles)
        out[lo * ROWS_PER_TILE:(lo + n) * ROWS_PER_TILE] = w.astype(np.float32) @ xf
    return out


def run_one(f, cfg, name: str, cores: int, limit_rows: int | None,
            xmode: str, iters: int, null: bool = False) -> bool:
    n, k = projection_shape(name, cfg)
    k_tiles = k // TILE_K
    per_call = tiles_per_call(k_tiles)
    row_bytes = k_tiles * TILE_BYTES
    n_rows_all = n // ROWS_PER_TILE

    # Cross-check the config-derived factoring against what the file stores.
    # gate_proj and down_proj share a stored shape and differ only here.
    stored = f.header[name]["shape"]
    if stored[0] != n_rows_all * k_tiles or stored[1] != TILE_BYTES:
        print(f"  [SKIP] {name}: config says N={n} K={k} -> "
              f"{n_rows_all * k_tiles} x {TILE_BYTES}, file has {stored}")
        return False

    tile_rows = n_rows_all if limit_rows is None else min(limit_rows, n_rows_all)
    while cores > 1 and tile_rows % cores:
        cores -= 1

    first, _ = f.header[name]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + first)
        raw = fh.read(tile_rows * row_bytes)

    if xmode == "ones":
        x = np.ones(k, np.float32).astype(bfloat16)
    elif xmode.startswith("onehot"):
        x = np.zeros(k, np.float32)
        x[int(xmode.split(":")[1]) if ":" in xmode else 0] = 1.0
        x = x.astype(bfloat16)
    else:
        x = np.random.default_rng(0).standard_normal(k).astype(np.float32).astype(bfloat16)

    a_w = iron.tensor(np.frombuffer(raw, dtype=np.uint8).copy(), dtype=np.uint8, device="npu")
    a_x = iron.tensor(x, dtype=bfloat16, device="npu")
    c_y = iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu")
    bench = run_iters(granite_gemv, a_w, a_x, c_y, tile_rows=tile_rows, k=k,
                      n_cores=cores, null=null, per_call=per_call,
                      warmup=1, iters=iters)
    mb = tile_rows * row_bytes / 1e6
    if null:
        # No arithmetic ran, so there is nothing to check: bandwidth only.
        us = bench.npu.avg_us
        print(f"  DMA   {name:44} {mb:6.1f}MB cores={cores:2} "
              f"{us / 1000:7.2f}ms {mb / us * 1e3:5.1f}GB/s"
              f"  <- weight stream, no compute")
        return True

    got = c_y.numpy().copy()
    ref = reference(raw, x, tile_rows, k_tiles)

    # float64: both of these are reductions over up to 100352 terms, and in
    # float32 the dot and the two norms accumulate in different orders, so the
    # ratio drifts from 1 even when the vectors are bit-identical. Measured:
    # x = ones reproduced the reference exactly (max rel err 0.0) and the float32
    # cosine still read 0.99999988 -- the metric's own rounding, reported as if
    # it were the kernel's error.
    g64, r64 = got.astype(np.float64), ref.astype(np.float64)
    rel = np.abs(g64 - r64).max() / (np.abs(r64).max() + 1e-30)
    cos = float(g64 @ r64 / (np.linalg.norm(g64) * np.linalg.norm(r64) + 1e-30))
    ok = cos > 0.9999999 and rel < 1e-4

    us = bench.npu.avg_us if bench.npu is not None else float("nan")
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44} N={n:6} K={k:5} "
          f"kt={k_tiles:2}/{per_call} rows={tile_rows * ROWS_PER_TILE:6} "
          f"cores={cores:2} {mb:6.1f}MB {us / 1000:7.2f}ms "
          f"{mb / us * 1e3:5.1f}GB/s cos={cos:.8f} rel={rel:.2e}")
    return ok


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="lm_head.weight")
    ap.add_argument("--all", action="store_true",
                    help="every distinct projection shape, layer 0 + lm_head")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--tile-rows", type=int, default=None,
                    help="cap tile-rows (default: the whole tensor)")
    ap.add_argument("--x", default="random", help="random | ones | onehot:<k>")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--null", action="store_true",
                    help="DMA probe: stream the weights, do no arithmetic")
    a = ap.parse_args(argv[1:])

    if not (MODEL / "model.q4nx").is_file():
        raise SystemExit(f"model not found: {MODEL / 'model.q4nx'}")
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    f = q4nx.Q4NX(MODEL / "model.q4nx")

    # Trap: without this IRON silently falls back to aie2/NPU1 -- no error, wrong
    # mac_dims, halved shim DMA burst.
    iron.set_current_device(from_name("npu2", n_cols=None))

    if a.all:
        names = [f"model.layers.0.self_attn.{p}.weight"
                 for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        names += [f"model.layers.0.mlp.{p}.weight"
                  for p in ("gate_proj", "up_proj", "down_proj")]
        names.append("lm_head.weight")
    else:
        names = [a.tensor]

    print(f"granite q4nx GEMV on NPU  ({a.x} activation)")
    results = [run_one(f, cfg, n, a.cores, a.tile_rows, a.x, a.iters, a.null)
               for n in names]
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} shapes match the host GEMV")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
