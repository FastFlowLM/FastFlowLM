r"""RMSNorm folded into a GEMV as a per-core prologue. ONE dispatch, full width.

WHY THIS AND NOT MORE FUSION OF THE BIG OPS
-------------------------------------------
0148 measured where a granite layer's time actually goes: the four GEMV groups
cost 1.65 ms and move 49.2 MB, while five small ops (2x RMSNorm, RoPE, SwiGLU,
attention) cost 1.20 ms and move almost nothing. ~1.0 ms of that is pure
per-dispatch floor -- about 200 us each, paid regardless of size.

Fusing the BIG ops does not help: even a perfect one-dispatch MLP at gate_up's
39.7 GB/s is 1.19 ms against 1.31 ms for the same three ops unfused across
20-32 cores. Measured, in granite_mlp_wide.py. The big ops are already work,
not waiting.

So the lever is the small ones, and RMSNorm is the easiest because it is
elementwise and has no head structure to keep core-local.

THE TRICK: REDUNDANT, NOT DISTRIBUTED
-------------------------------------
Every core's GEMV consumes the WHOLE activation, so every core needs the whole
normalised vector. Computing the norm once and broadcasting it would need a
cross-core barrier inside the dispatch. Computing it **redundantly on every
core** needs nothing: it is 2560 multiply-accumulates against 8192 MACs per
weight tile, and it removes the dispatch outright.

AND IN PLACE, WHICH IS THE PART THAT DECIDES IT
-----------------------------------------------
The activation fifo carries `x` then the norm weight, 2 x 2560 bf16. The
prologue normalises into the first half. A separate destination buffer would
cost another 5120 B of L1, pushing the fixed cost from 10240 to 15360 -- and
with a 62208 B budget that drops the weight double buffer from per_call 5 to
per_call 2 (the only divisors of 10 tiles), which costs more bandwidth than the
saved dispatch is worth. See the __restrict note in granite_rmsnorm.h.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_norm_gemv.py
    python designs\granite_gemv\granite_norm_gemv.py --tensor gate_up
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
def granite_norm_gemv(w: In, xw: In, y: Out, *,
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
    rmsnorm = ExternalFunction("granite_rms_norm_ip",
                               source_file=str(AIE / "granite_rmsnorm_ip.cc"),
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

    def core_body(win, xin, yout, norm, *ks):
        xe = xin.acquire(1)
        # Redundantly on every core, in place. Every core's GEMV reads the whole
        # activation anyway, so a broadcast would need a barrier inside the
        # dispatch; this needs nothing.
        norm(xe, k)
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
                        y_cores[c][r].prod(), rmsnorm, *kernels],
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
    ap.add_argument("--tensor", default="qkv",
                    choices=("qkv", "gate_up", "o"))
    ap.add_argument("--cols", type=int, default=0,
                    help="0 = widest that divides the tile-rows")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args((argv or sys.argv)[1:])

    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    groups = {"qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
              "gate_up": ["mlp.gate_proj", "mlp.up_proj"],
              "o": ["self_attn.o_proj"]}[a.tensor]
    # o_proj takes the attention output, not a normalised hidden state -- it is
    # here only to show the norm prologue is shape-agnostic, and its reference
    # normalises too so the comparison stays honest.
    norm_name = ("model.layers.0.post_attention_layernorm.weight"
                 if a.tensor == "gate_up"
                 else "model.layers.0.input_layernorm.weight")

    f = q4nx.Q4NX(MODEL / "model.q4nx")
    raws, tile_rows, k = [], 0, None
    for g in groups:
        nm = f"model.layers.0.{g}.weight"
        n, kk = projection_shape(nm, cfg)
        k = kk
        rows = n // ROWS_PER_TILE
        off, _ = f.header[nm]["data_offsets"]
        with f.path.open("rb") as fh:
            fh.seek(f._data_start + off)
            raws.append(fh.read(rows * (kk // TILE_K) * TILE_BYTES))
        tile_rows += rows
    raw = b"".join(raws)
    k_tiles = k // TILE_K

    off, _ = f.header[norm_name]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + off)
        nwv = np.frombuffer(fh.read(k * 2), dtype=bfloat16)

    # The array has 8 columns of 4, but a group only fills the ones its
    # tile-rows divide over: qkv is 112 rows = 16 x 7, so 7 columns and not 8.
    # This is the same cap that pins o_proj and down_proj (80 rows) to 20 cores.
    cols = a.cols or max(c for c in range(1, 9) if tile_rows % (c * ROWS_PER_COL) == 0)
    n_cores = cols * ROWS_PER_COL
    per_call = tiles_per_call(k_tiles, 1, k)
    # The norm weight adds a second k-wide fifo, so the fixed L1 is 2*k*2 rather
    # than k*2. Check it here rather than discover it as a build failure.
    fixed = 2 * k * 2   # the combined [x | norm weight] element
    assert per_call * TILE_BYTES * 2 + fixed <= 64 * 1024 - 0xD00, (
        f"per_call {per_call} plus the norm weight overflows L1")
    n_entry = k_tiles // per_call
    per_core = tile_rows // n_cores
    w = permute_weights(raw, cols, per_core, n_entry, per_call * TILE_BYTES)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(k).astype(np.float32).astype(bfloat16)

    iron.set_current_device(from_name("npu2", n_cols=None))
    c_y = iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu")
    b = run_iters(granite_norm_gemv,
                  iron.tensor(w, dtype=np.uint8, device="npu"),
                  iron.tensor(np.concatenate([x, nwv]), dtype=bfloat16,
                              device="npu"),
                  c_y, tile_rows=tile_rows, k=k, n_cols=cols,
                  per_call=per_call, warmup=1, iters=a.iters)
    got = unpermute_y(c_y.numpy().copy(), cols, per_core).astype(np.float64)

    # Reference: normalise, THEN the GEMV -- both together. Checking the two
    # halves separately is what let a wrong composition pass in 0147.
    xf = x.astype(np.float32)
    inv = 1.0 / np.sqrt((xf * xf).mean() + cfg["rms_norm_eps"])
    h = (xf * inv * nwv.astype(np.float32)).astype(bfloat16)
    ref = reference(raw, h, tile_rows, k_tiles).astype(np.float64)

    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    g1, r1 = got.ravel(), ref.ravel()
    cos = float(g1 @ r1 / (np.linalg.norm(g1) * np.linalg.norm(r1) + 1e-30))
    mb = len(raw) / 1e6
    us = b.npu.avg_us
    ok = cos > 0.9999 and rel < 8e-3
    print(f"RMSNorm + {a.tensor} GEMV, ONE dispatch   "
          f"{cols} cols x {ROWS_PER_COL} = {n_cores} cores, per_call {per_call}")
    print(f"  {tile_rows * ROWS_PER_TILE} rows  K={k}  {mb:.1f} MB")
    print(f"  cosine {cos:.8f}   max rel err {rel:.3e}")
    print(f"  {us:.1f} us device   {b.e2e.avg_us:.1f} us wall   "
          f"{mb / us * 1e3:.1f} GB/s")
    print(f"  separate would be RMSNorm 244 us + GEMV -> compare against 0148")
    print(f"  [fused == normalise then GEMV on the host]  "
          f"{'PASS' if ok else 'FAIL'}")
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


def build_artifact(geometry: dict, tensor: str = "qkv") -> None:
    groups = {"qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
              "gate_up": ["mlp.gate_proj", "mlp.up_proj"],
              "o": ["self_attn.o_proj"]}[tensor]
    tile_rows, k = 0, None
    for g in groups:
        n, kk = projection_shape(g.rsplit(".", 1)[-1], geometry)
        k = kk
        tile_rows += n // ROWS_PER_TILE
    k_tiles = k // TILE_K
    cols = max(c for c in range(1, 9) if tile_rows % (c * ROWS_PER_COL) == 0)
    per_call = tiles_per_call(k_tiles, 1, k)
    n_cores = cols * ROWS_PER_COL
    per_core = tile_rows // n_cores
    w_bytes = tile_rows * k_tiles * TILE_BYTES

    iron.set_current_device(from_name("npu2", n_cols=None))
    granite_norm_gemv(
        iron.zeros(w_bytes, dtype=np.uint8, device="npu"),
        iron.zeros(2 * k, dtype=bfloat16, device="npu"),
        iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu"),
        tile_rows=tile_rows, k=k, n_cols=cols, per_call=per_call)


if __name__ == "__main__":
    sys.exit(main())
