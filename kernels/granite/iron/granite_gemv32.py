r"""The same granite q4nx GEMV, but on all 32 cores via the memtile leg.

WHY THIS EXISTS
---------------
`granite_gemv.py` gives every core its own shim stream, which costs n+1 MM2S and
n S2MM channels. There are 16 of each device-wide, so it tops out at **8 cores**
and 16 dies at placement with `no ShimNOCTile has sufficient DMA capacity`.
npu2 is 8 columns x 6 rows (row 0 shim, row 1 memtile, rows 2-5 compute) = **32
compute cores**, so three quarters of the array is unreachable that way.

Measured, and this is the reason to bother:

    NPU DMA alone, no arithmetic   3.47 ms   46.3 GB/s
    NPU GEMV, 8 cores              8.08 ms   19.9 GB/s
    per-core compute               ~2.5 GB/s, perfectly linear

The weight path sustains 46.3 GB/s and the kernel uses 19.9 of them, so this is
compute-bound and ~18 cores saturates it. The bandwidth is available long before
the cores are.

THE SHAPE OF THE FIX
--------------------
One shim stream per **column** into the memtile, split four ways to that column's
compute cores, and joined back on the way out:

    shim MM2S x1  ->  memtile  ->  split -> 4 cores        (per column)
    shim S2MM x1  <-  memtile  <-  join  <- 4 cores

That is 8 + 1 MM2S (weights + broadcast x) and 8 S2MM for 32 cores, comfortably
inside 16/16.

WHY THE WEIGHTS ARE PERMUTED ON THE HOST
----------------------------------------
`split()` hands child i the i-th slice of each parent object, so the DDR stream
must arrive as [core0 chunk k][core1 chunk k][core2 chunk k][core3 chunk k].
With each core owning a contiguous block of tile-rows, that is a strided 3-D tap
whose innermost run is 25600 B -- close enough to the BD size limits to be a
liability. Permuting host-side makes each column's stream plain contiguous
instead, and it is a **one-time cost at model load**, not per token: the weights
are uploaded once and streamed for every token thereafter.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_gemv32.py --cols 8        :: 32 cores
    python designs\granite_gemv\granite_gemv32.py --cols 8 --null :: DMA only
"""

from __future__ import annotations

import json
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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "common"))

import q4nx  # noqa: E402
from granite_gemv import (MODEL, ROWS_PER_TILE, TILE_BYTES, TILE_K,  # noqa: E402
                          _include_dirs, ensure_entry_points, projection_shape,
                          reference, tiles_per_call)

ROWS_PER_COL = 4  # compute rows per column on npu2 (array rows 2..5)


def permute_weights(raw: bytes, n_cols: int, per_core: int, n_entry: int,
                    call_bytes: int) -> np.ndarray:
    """Reorder so each column's stream is contiguous in the order split() wants.

    In: core-major, each core's tile-rows contiguous.
    Out: per column, chunk-major then core -- [k][r] -- which is exactly the
    layout `split()` consumes, one parent object per k.
    """
    a = np.frombuffer(raw, dtype=np.uint8)
    # (col, row_in_col, chunk, bytes) -> (col, chunk, row_in_col, bytes)
    a = a.reshape(n_cols, ROWS_PER_COL, per_core * n_entry, call_bytes)
    return np.ascontiguousarray(a.transpose(0, 2, 1, 3)).reshape(-1)


def unpermute_y(y: np.ndarray, n_cols: int, per_core: int,
                batch: int = 1) -> np.ndarray:
    """Inverse of the above for the joined output.

    On the wire it is [col][t][row_in_col][token][32]; the caller wants one
    contiguous result vector per token, so the token axis comes out front.
    Returns (batch, tile_rows * 32).
    """
    a = y.reshape(n_cols, per_core, ROWS_PER_COL, batch, ROWS_PER_TILE)
    a = a.transpose(3, 0, 2, 1, 4)          # [token][col][row][t][32]
    return np.ascontiguousarray(a).reshape(batch, -1)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_gemv32(w: In, x: In, y: Out, *, tile_rows: CompileTime[int],
                   k: CompileTime[int], n_cols: CompileTime[int] = 8,
                   null: CompileTime[bool] = False,
                   batch: CompileTime[int] = 1,
                   per_call: CompileTime[int] = 5):
    # per_call MUST be an explicit argument, not derived here. iron.jit's cache
    # key hashes the call's arguments and nothing else (trap 7d), so a derived
    # per_call is invisible to it: two runs that differ only in per_call collide
    # on one cache entry and the second silently gets the first one's xclbin,
    # while the host permutes for its own value. That is not a crash -- it
    # returned cosine 0.208 against the reference, a plausible-looking wrong
    # answer, and it cost a confusing debug cycle to find.
    k_tiles = k // TILE_K
    n_entry = k_tiles // per_call
    call_bytes = per_call * TILE_BYTES
    n_cores = n_cols * ROWS_PER_COL
    per_core = tile_rows // n_cores
    chunks = per_core * n_entry           # parent objects per column

    srcs = ensure_entry_points(n_entry, per_call, null, batch, k)

    w_l1_ty = np.ndarray[(call_bytes,), np.dtype[np.uint8]]
    w_l2_ty = np.ndarray[(ROWS_PER_COL * call_bytes,), np.dtype[np.uint8]]
    # One weight pass serves `batch` tokens, so every activation and result
    # buffer carries the token axis; the weights do not change size at all --
    # that asymmetry is the entire point of batching.
    y_l1_ty = np.ndarray[(batch * ROWS_PER_TILE,), np.dtype[np.float32]]
    y_l2_ty = np.ndarray[(ROWS_PER_COL * batch * ROWS_PER_TILE,),
                         np.dtype[np.float32]]
    x_ty = np.ndarray[(batch * k,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(tile_rows * k_tiles * TILE_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(tile_rows * batch * ROWS_PER_TILE,), np.dtype[np.float32]]

    kernels = [
        ExternalFunction(
            f"granite_gemv_p{per_call}b{batch}_k{i}",
            source_file=str(srcs[i]),
            arg_types=[w_l1_ty, x_ty, y_l1_ty],
            include_dirs=_include_dirs(),
        )
        for i in range(n_entry)
    ]

    # One activation for the whole array. 32 private copies are impossible --
    # there are only 16 shim MM2S channels and the weights need 8 of them.
    of_x = ObjectFifo(x_ty, name="x", depth=1)

    w_l3l2, y_l2l3, w_cores, y_cores = [], [], [], []
    for c in range(n_cols):
        wf = ObjectFifo(w_l2_ty, name=f"wL2_{c}", depth=2)
        w_l3l2.append(wf)
        w_cores.append(wf.cons().split(
            [r * call_bytes for r in range(ROWS_PER_COL)],
            obj_types=[w_l1_ty] * ROWS_PER_COL,
            names=[f"w_{c}_{r}" for r in range(ROWS_PER_COL)],
        ))
        yf = ObjectFifo(y_l2_ty, name=f"yL2_{c}", depth=2)
        y_l2l3.append(yf)
        y_cores.append(yf.prod().join(
            [r * batch * ROWS_PER_TILE for r in range(ROWS_PER_COL)],
            obj_types=[y_l1_ty] * ROWS_PER_COL,
            names=[f"y_{c}_{r}" for r in range(ROWS_PER_COL)],
        ))

    def core_body(win, xin, yout, *ks):
        xe = xin.acquire(1)
        for _ in range_(per_core):
            ye = yout.acquire(1)
            for fn in ks:
                we = win.acquire(1)
                fn(we, xe, ye)
                win.release(1)
            yout.release(1)
        xin.release(1)

    workers = [
        Worker(core_body,
               fn_args=[w_cores[c][r].cons(), of_x.cons(),
                        y_cores[c][r].prod(), *kernels],
               stack_size=0xD00)
        for c in range(n_cols) for r in range(ROWS_PER_COL)
    ]

    # After the host-side permutation each column's weights are one contiguous
    # run, so this is a plain split rather than a strided gather.
    col_w = chunks * ROWS_PER_COL * call_bytes
    col_y = per_core * ROWS_PER_COL * batch * ROWS_PER_TILE
    w_taps = TensorTiler2D.simple_tiler((1, n_cols * col_w), (1, col_w))
    y_taps = TensorTiler2D.simple_tiler((1, n_cols * col_y), (1, col_y))

    def sequence(a_w, a_x, c_y, w_prods, x_prod, y_conss):
        tg = TaskGroup()
        x_prod.fill(a_x, group=tg)
        for c in range(n_cols):
            w_prods[c].fill(a_w, tap=w_taps[c], group=tg)
            y_conss[c].drain(c_y, tap=y_taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence,
                 [w_ty, x_ty, y_ty,
                  [f.prod() for f in w_l3l2], of_x.prod(),
                  [f.cons() for f in y_l2l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main(argv: list[str]) -> int:
    import argparse
    from aie.utils.benchmark import run_iters

    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="lm_head.weight")
    ap.add_argument("--cols", type=int, default=8, help="columns; cores = 4x")
    ap.add_argument("--tile-rows", type=int, default=None)
    ap.add_argument("--x", default="random")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1,
                    help="tokens per weight pass (independent tokens only)")
    ap.add_argument("--null", action="store_true",
                    help="DMA probe: stream the weights, do no arithmetic")
    a = ap.parse_args(argv[1:])

    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    f = q4nx.Q4NX(MODEL / "model.q4nx")
    n, k = projection_shape(a.tensor, cfg)
    k_tiles = k // TILE_K
    per_call = tiles_per_call(k_tiles, a.batch, k)  # passed in explicitly
    n_entry = k_tiles // per_call
    call_bytes = per_call * TILE_BYTES
    row_bytes = k_tiles * TILE_BYTES

    n_cores = a.cols * ROWS_PER_COL
    tile_rows = n // ROWS_PER_TILE if a.tile_rows is None else a.tile_rows
    if tile_rows % n_cores:
        tile_rows -= tile_rows % n_cores
    per_core = tile_rows // n_cores

    first, _ = f.header[a.tensor]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + first)
        raw = fh.read(tile_rows * row_bytes)

    if a.x == "ones":
        x = np.ones(k, np.float32).astype(bfloat16)
    elif a.x.startswith("onehot"):
        x = np.zeros(k, np.float32)
        x[int(a.x.split(":")[1]) if ":" in a.x else 0] = 1.0
        x = x.astype(bfloat16)
    else:
        x = np.random.default_rng(0).standard_normal(
            a.batch * k).astype(np.float32).astype(bfloat16)
    if a.batch > 1 and x.size == k:
        x = np.tile(x, a.batch)

    iron.set_current_device(from_name("npu2", n_cols=None))
    w_perm = permute_weights(raw, a.cols, per_core, n_entry, call_bytes)

    a_w = iron.tensor(w_perm, dtype=np.uint8, device="npu")
    a_x = iron.tensor(x, dtype=bfloat16, device="npu")
    c_y = iron.zeros(tile_rows * a.batch * ROWS_PER_TILE, dtype=np.float32,
                     device="npu")
    bench = run_iters(granite_gemv32, a_w, a_x, c_y, tile_rows=tile_rows, k=k,
                      n_cols=a.cols, null=a.null, batch=a.batch,
                      per_call=per_call, warmup=1, iters=a.iters)

    mb = tile_rows * row_bytes / 1e6
    us = bench.npu.avg_us
    head = (f"{a.tensor}  N={n} K={k}  {tile_rows * ROWS_PER_TILE} rows  "
            f"{a.cols} cols x {ROWS_PER_COL} = {n_cores} cores  {mb:.1f} MB"
            + (f"  batch {a.batch}" if a.batch > 1 else ""))
    if a.null:
        print(f"DMA   {head}\n      {us / 1000:.2f} ms  {mb / us * 1e3:.1f} GB/s"
              f"   <- weight stream, no compute")
        return 0

    got = unpermute_y(c_y.numpy().copy(), a.cols, per_core, a.batch)
    # Check EVERY token, not just the first: a batched kernel that ignored the
    # token index would still reproduce token 0 perfectly.
    rel, cos, ok = 0.0, 1.0, True
    for b in range(a.batch):
        ref = reference(raw, x[b * k:(b + 1) * k], tile_rows, k_tiles)
        g64, r64 = got[b].astype(np.float64), ref.astype(np.float64)
        rel = max(rel, np.abs(g64 - r64).max() / (np.abs(r64).max() + 1e-30))
        cos = min(cos, float(g64 @ r64 /
                             (np.linalg.norm(g64) * np.linalg.norm(r64) + 1e-30)))
    ok = cos > 0.9999999 and rel < 1e-4
    print(f"{'PASS' if ok else 'FAIL'}  {head}\n"
          f"      {us / 1000:.2f} ms  {mb / us * 1e3:.1f} GB/s  "
          f"({2.0 * a.batch * tile_rows * ROWS_PER_TILE * k / us / 1e3:.1f} GFLOP/s)  "
          f"cos={cos:.8f} rel={rel:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
