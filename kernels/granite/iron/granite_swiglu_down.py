r"""SwiGLU folded into down_proj as a per-core prologue. ONE dispatch.

WHY
---
0148 measured the per-dispatch floor at ~200 us and SwiGLU at 205 us for 8192
values -- almost pure floor, since it moves no weights at all. Folding it into
the dispatch that consumes its output removes that entirely.

THE COST, WHICH IS REAL
-----------------------
down_proj has K = 8192, so its activation is 8192 bf16. Taking gate and up
instead doubles that to 32768 B, and the L1 budget is 62208:

    per_call 1   10240 + 32768 = 43008   fits
    per_call 2   20480 + 32768 = 53248   fits
    per_call 4   40960 + 32768 = 73728   over

Unfused, down_proj runs at per_call 4. Fused it must drop to 2, which halves the
DMA element and costs bandwidth. The trade is one whole dispatch against that,
and it is worth measuring rather than assuming -- granite_mlp_wide.py is the
case where exactly this trade came out NEGATIVE.

IN PLACE, AND WHY THAT IS FORCED
--------------------------------
A core has two input DMA channels and both are taken: weights and activation.
gate and up therefore share one element, up at gate + 8192, and the SwiGLU
result is written back over gate so the GEMV can read it as x with no third
buffer. See granite_swiglu_ip.cc.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_swiglu_down.py
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
from granite_gemv import (AIE, MODEL, ROWS_PER_TILE, TILE_BYTES, TILE_K,  # noqa: E402
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



ROWS_PER_COL = 4      # compute rows per column on npu2 (array rows 2..5)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_swiglu_down(w: In, xw: In, y: Out, *,
                      tile_rows: CompileTime[int], k: CompileTime[int],
                      n_cols: CompileTime[int] = 8,
                      per_call: CompileTime[int] = 5):
    # per_call is an explicit argument for the reason granite_gemv32 records:
    # iron.jit's cache key hashes the call's arguments and nothing else, so a
    # value derived in here is invisible to it and two configs collide.
    k_tiles = k // TILE_K
    n_entry = k_tiles // per_call
    call_bytes = per_call * TILE_BYTES
    n_cores = n_cols * ROWS_PER_COL
    per_core = tile_rows // n_cores
    chunks = per_core * n_entry

    srcs = ensure_entry_points(n_entry, per_call, False, 1, k)

    w_l1_ty = np.ndarray[(call_bytes,), np.dtype[np.uint8]]
    w_l2_ty = np.ndarray[(ROWS_PER_COL * call_bytes,), np.dtype[np.uint8]]
    y_l1_ty = np.ndarray[(ROWS_PER_TILE,), np.dtype[np.float32]]
    y_l2_ty = np.ndarray[(ROWS_PER_COL * ROWS_PER_TILE,), np.dtype[np.float32]]
    # x and the norm weight share ONE fifo: a compute tile has 2 input DMA
    # channels and the weights already take one, so a second activation stream
    # is not available at any L1 cost. Two 5120 B buffers and one 10240 B buffer
    # cost the same anyway.
    xw_ty = np.ndarray[(2 * k,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(tile_rows * k_tiles * TILE_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(tile_rows * ROWS_PER_TILE,), np.dtype[np.float32]]

    kernels = [ExternalFunction(f"granite_gemv_p{per_call}b1_k{i}",
                                source_file=str(srcs[i]),
                                arg_types=[w_l1_ty, xw_ty, y_l1_ty],
                                include_dirs=_include_dirs())
               for i in range(n_entry)]
    swiglu = ExternalFunction("granite_swiglu_ip",
                              source_file=str(AIE / "granite_swiglu_ip.cc"),
                              arg_types=[xw_ty, np.int32],
                              include_dirs=_include_dirs())

    of_x = ObjectFifo(xw_ty, name="nx", depth=1)

    w_l3l2, y_l2l3, w_cores, y_cores = [], [], [], []
    for c in range(n_cols):
        wf = ObjectFifo(w_l2_ty, name=f"nwL2_{c}", depth=2)
        w_l3l2.append(wf)
        w_cores.append(wf.cons().split(
            [r * call_bytes for r in range(ROWS_PER_COL)],
            obj_types=[w_l1_ty] * ROWS_PER_COL,
            names=[f"nw_{c}_{r}" for r in range(ROWS_PER_COL)]))
        yf = ObjectFifo(y_l2_ty, name=f"nyL2_{c}", depth=2)
        y_l2l3.append(yf)
        y_cores.append(yf.prod().join(
            [r * ROWS_PER_TILE for r in range(ROWS_PER_COL)],
            obj_types=[y_l1_ty] * ROWS_PER_COL,
            names=[f"ny_{c}_{r}" for r in range(ROWS_PER_COL)]))

    def core_body(win, xin, yout, sw, *ks):
        xe = xin.acquire(1)
        # Redundantly on every core, in place. down_proj's K is the whole
        # intermediate, so every core needs all 8192 values anyway -- computing
        # them per core needs no barrier and no communication.
        sw(xe, k)
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
                        y_cores[c][r].prod(), swiglu, *kernels],
               stack_size=0xD00)
        for c in range(n_cols) for r in range(ROWS_PER_COL)
    ]

    col_w = chunks * ROWS_PER_COL * call_bytes
    col_y = per_core * ROWS_PER_COL * ROWS_PER_TILE
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
                 [w_ty, xw_ty, y_ty,
                  [f.prod() for f in w_l3l2], of_x.prod(),
                  [f.cons() for f in y_l2l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main(argv: list[str] | None = None) -> int:
    import argparse
    from aie.utils.benchmark import run_iters
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args((argv or sys.argv)[1:])

    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    name = "model.layers.0.mlp.down_proj.weight"
    n, k = projection_shape(name, cfg)          # 2560 x 8192
    tile_rows, k_tiles = n // ROWS_PER_TILE, k // TILE_K

    f = q4nx.Q4NX(MODEL / "model.q4nx")
    off, _ = f.header[name]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + off)
        raw = fh.read(tile_rows * k_tiles * TILE_BYTES)

    cols = a.cols or max(c for c in range(1, 9)
                         if tile_rows % (c * ROWS_PER_COL) == 0)
    n_cores = cols * ROWS_PER_COL
    per_core = tile_rows // n_cores

    # per_call from the ACTUAL fixed cost, which is 2*k*2 here because the
    # activation carries gate and up: tiles_per_call() assumes k*2 and would
    # return 4, which does not fit.
    L1 = 64 * 1024 - 0xD00
    fixed = 2 * k * 2
    per_call = max(d for d in range(1, k_tiles + 1)
                   if k_tiles % d == 0 and d * TILE_BYTES * 2 + fixed <= L1)
    n_entry = k_tiles // per_call
    w = permute_weights(raw, cols, per_core, n_entry, per_call * TILE_BYTES)

    rng = np.random.default_rng(0)
    gate = rng.standard_normal(k).astype(np.float32).astype(bfloat16)
    up = rng.standard_normal(k).astype(np.float32).astype(bfloat16)
    gu = np.concatenate([gate, up])

    iron.set_current_device(from_name("npu2", n_cols=None))
    c_y = iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu")
    b = run_iters(granite_swiglu_down,
                  iron.tensor(w, dtype=np.uint8, device="npu"),
                  iron.tensor(gu, dtype=bfloat16, device="npu"), c_y,
                  tile_rows=tile_rows, k=k, n_cols=cols, per_call=per_call,
                  warmup=1, iters=a.iters)
    got = unpermute_y(c_y.numpy().copy(), cols, per_core).astype(np.float64)

    # Reference: SwiGLU then the GEMV, together. The kernel narrows the SwiGLU
    # result to bf16 before the matmul because that is what the buffer holds, so
    # the reference must too -- comparing against an fp64 intermediate would
    # charge the kernel for a rounding the storage format dictates.
    g64 = gate.astype(np.float64)
    h = ((g64 / (1.0 + np.exp(-g64))) * up.astype(np.float64))
    h_bf = h.astype(np.float32).astype(bfloat16)
    ref = reference(raw, h_bf, tile_rows, k_tiles).astype(np.float64)

    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    g1, r1 = got.ravel(), ref.ravel()
    cos = float(g1 @ r1 / (np.linalg.norm(g1) * np.linalg.norm(r1) + 1e-30))
    mb = len(raw) / 1e6
    us = b.npu.avg_us
    # aie::tanh sets the floor, as granite_mlp.py records: ~8e-3 on the SwiGLU
    # alone, and down_proj adds little on top.
    ok = cos > 0.999 and rel < 5e-2
    print(f"SwiGLU + down_proj, ONE dispatch   {cols} cols x {ROWS_PER_COL} = "
          f"{n_cores} cores, per_call {per_call}")
    print(f"  {n} rows  K={k}  {mb:.1f} MB")
    print(f"  cosine {cos:.8f}   max rel err {rel:.3e}")
    print(f"  {us:.1f} us device   {b.e2e.avg_us:.1f} us wall   "
          f"{mb / us * 1e3:.1f} GB/s")
    print(f"  separate: SwiGLU 205 + down 440 = 645 us  "
          f"(down alone runs per_call 4, this must use {per_call})")
    print(f"  [fused == SwiGLU then down on the host]  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# build_artifact: produce the xclbin WITHOUT model weights.
#
# iron.jit keys its cache on argument shapes and dtypes, not contents, so an
# artefact built from zeros is bit-identical to one built from real weights.
# That is what lets the in-tree build run from a clean checkout with nothing
# but the toolchain -- see kernels/CONVENTION.md. main() below is unchanged and
# remains the developer path: it needs the model and checks the result.
# --------------------------------------------------------------------------


def build_artifact(geometry: dict) -> None:
    n, k = projection_shape("down_proj", geometry)
    tile_rows, k_tiles = n // ROWS_PER_TILE, k // TILE_K
    cols = max(c for c in range(1, 9) if tile_rows % (c * ROWS_PER_COL) == 0)
    L1 = 64 * 1024 - 0xD00
    fixed = 2 * k * 2
    per_call = max(d for d in range(1, k_tiles + 1)
                   if k_tiles % d == 0 and d * TILE_BYTES * 2 + fixed <= L1)
    w_bytes = tile_rows * k_tiles * TILE_BYTES

    iron.set_current_device(from_name("npu2", n_cols=None))
    granite_swiglu_down(
        iron.zeros(w_bytes, dtype=np.uint8, device="npu"),
        iron.zeros(2 * k, dtype=bfloat16, device="npu"),
        iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu"),
        tile_rows=tile_rows, k=k, n_cols=cols, per_call=per_call)


if __name__ == "__main__":
    sys.exit(main())
