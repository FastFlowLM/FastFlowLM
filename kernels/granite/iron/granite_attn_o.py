r"""attention (40 heads) + o_proj in ONE dispatch, on disjoint cores.

WHY
---
0148 measured the per-dispatch floor at ~200 us: RMSNorm on 2560 values costs
244 us, SwiGLU on 8192 costs 205, and RoPE cost NOTHING when it moved inside an
existing dispatch. A granite layer is 6 dispatches and 2.337 ms, so about 1.2 ms
of it is the cost of asking. Removing a dispatch is worth ~200 us wherever it
can be done without losing cores.

WHY DISJOINT CORES, AND WHY THE WHOLE BLOCK DOES NOT FIT
--------------------------------------------------------
Fusing all of norm+qkv+RoPE, attention and o_proj would want 7 columns for qkv,
2 for attention and 5 for o_proj -- 14 of the 8 the array has. Cores would then
have to serve several phases, which means one ObjectFifo carrying different
payloads in different phases (the granite_mlp_full.py pattern) against a much
tighter L1.

This pair fits disjointly:

    columns 0-4   o_proj      20 cores, 4 tile-rows each
    columns 5-6   attention    8 cores, one kv head each
    column  7     unused

so no core does both, no fifo is reused, and the shim budget is 10 MM2S and
7 S2MM against 16 of each.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_attn_o.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (Buffer, CompileTime, In, ObjectFifo, Out, Program,
                      Runtime, TaskGroup, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import run_iters

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "common"))
import q4nx  # noqa: E402
from granite_gemv import (AIE, MODEL, ROWS_PER_TILE, TILE_BYTES, TILE_K,  # noqa: E402
                          _include_dirs, ensure_entry_points,
                          projection_shape, reference, tiles_per_call)
from granite_gemv32 import (ROWS_PER_COL, permute_weights,  # noqa: E402
                            unpermute_y)

SRC_BLOCK_H = str(AIE / "granite_attn_block_h.cc")
SRC_FINISH_H = str(AIE / "granite_attn_finish_h.cc")
HD = 64                # granite head_dim
BLK = 32               # KV positions per call, matches GRANITE_ATTN_BLOCK
ST_STRIDE = 128        # floats; 512 B, 128-byte aligned. granite_attn_block_h.cc
O_COLS = 4             # columns 0..3 -> o_proj, 16 cores, 5 tile-rows each
A_COLS = 4             # columns 4..7 -> attention
A_ROWS = 2             # attention cores per column; 4 x 2 = 8 kv heads

# Why 4 x 2 and not 2 x 4: a memtile has about 6 DMA channels each way, and an
# attention column needs a split for q (1 in, 4 out), a split for kv (1 in,
# 4 out) and a join for the result (4 in, 1 out) -- 6 in and 9 out at four cores
# per column, which does not place:
#
#   error: no MemTile has sufficient DMA capacity for 1 input/4 output channels
#
# At two cores per column it is 4 in and 5 out. o_proj gives up a column for it
# (20 cores -> 16), which costs little because o_proj is floor-dominated: 4.1 MB
# at 15.8 GB/s is nearly all fixed cost already.


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_attn_o(a_q: In, a_kv: In, scratch_o: Out, scratch_i: In,
                   a_w: In, c_y: Out, *, seq: CompileTime[int],
                   tile_rows: CompileTime[int], k: CompileTime[int],
                   per_call: CompileTime[int],
                   q_per: CompileTime[int] = 5,
                   o_cols: CompileTime[int] = O_COLS,
                   a_cols: CompileTime[int] = A_COLS,
                   a_rows: CompileTime[int] = A_ROWS):
    n_blk = (seq + BLK - 1) // BLK
    n_kv = a_cols * a_rows
    o_cores = o_cols * ROWS_PER_COL
    per_core = tile_rows // o_cores
    k_tiles = k // TILE_K
    n_entry = k_tiles // per_call
    call_bytes = per_call * TILE_BYTES
    chunks = per_core * n_entry

    srcs = ensure_entry_points(n_entry, per_call, False, 1, k)

    # ---- attention side ----
    # No Q_PAD: 320 bf16 is 640 B, clear of the 128-byte transfer that arrives
    # as zeros, and an unpadded output makes the scratch the plain head-ordered
    # 2560-wide vector o_proj wants, with no holes to skip.
    qc_ty = np.ndarray[(q_per * HD,), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(2 * BLK * HD,), np.dtype[bfloat16]]
    st_ty = np.ndarray[(q_per * ST_STRIDE,), np.dtype[np.float32]]
    ao_ty = np.ndarray[(q_per * HD,), np.dtype[bfloat16]]

    # ---- o_proj side ----
    w_l1_ty = np.ndarray[(call_bytes,), np.dtype[np.uint8]]
    w_l2_ty = np.ndarray[(ROWS_PER_COL * call_bytes,), np.dtype[np.uint8]]
    y_l1_ty = np.ndarray[(ROWS_PER_TILE,), np.dtype[np.float32]]
    y_l2_ty = np.ndarray[(ROWS_PER_COL * ROWS_PER_TILE,), np.dtype[np.float32]]
    x_ty = np.ndarray[(k,), np.dtype[bfloat16]]

    all_q_ty = np.ndarray[(n_kv * q_per * HD,), np.dtype[bfloat16]]
    all_kv_ty = np.ndarray[(n_kv * n_blk * 2 * BLK * HD,), np.dtype[bfloat16]]
    attn_ty = np.ndarray[(k,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(tile_rows * k_tiles * TILE_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(tile_rows * ROWS_PER_TILE,), np.dtype[np.float32]]

    blk = ExternalFunction("granite_attn_block_h", source_file=SRC_BLOCK_H,
                           arg_types=[qc_ty, kv_ty, st_ty, np.int32, np.int32,
                                      np.int32],
                           include_dirs=_include_dirs())
    fin = ExternalFunction("granite_attn_finish_h", source_file=SRC_FINISH_H,
                           arg_types=[st_ty, ao_ty, np.int32],
                           include_dirs=_include_dirs())
    gemv = [ExternalFunction(f"granite_gemv_p{per_call}b1_k{i}",
                             source_file=str(srcs[i]),
                             arg_types=[w_l1_ty, x_ty, y_l1_ty],
                             include_dirs=_include_dirs())
            for i in range(n_entry)]

    # Per COLUMN, not per core. Eight cores own eight distinct kv heads, so
    # there is nothing to broadcast -- but giving each its own shim stream costs
    # 8 q + 8 kv + 5 weights + 1 activation = 22 MM2S against the 16 the device
    # has, and the placer says so:
    #
    #   error: no ShimNOCTile has sufficient DMA capacity ...
    #
    # Through the memtile it is 2 + 2 + 5 + 1 = 10 MM2S and 2 + 5 = 7 S2MM.
    q_l2_ty = np.ndarray[(a_rows * q_per * HD,), np.dtype[bfloat16]]
    kv_l2_ty = np.ndarray[(a_rows * 2 * BLK * HD,), np.dtype[bfloat16]]
    ao_l2_ty = np.ndarray[(a_rows * q_per * HD,), np.dtype[bfloat16]]
    of_x = ObjectFifo(x_ty, name="aox", depth=1)
    q_l3l2, kv_l3l2, ao_l2l3 = [], [], []
    q_cores, kv_cores, ao_cores = [], [], []
    for c in range(a_cols):
        qf = ObjectFifo(q_l2_ty, name=f"aoqL2_{c}", depth=2)
        q_l3l2.append(qf)
        q_cores.append(qf.cons().split(
            [r * q_per * HD for r in range(a_rows)],
            obj_types=[qc_ty] * a_rows,
            names=[f"aoq_{c}_{r}" for r in range(a_rows)]))
        kf = ObjectFifo(kv_l2_ty, name=f"aokvL2_{c}", depth=2)
        kv_l3l2.append(kf)
        kv_cores.append(kf.cons().split(
            [r * 2 * BLK * HD for r in range(a_rows)],
            obj_types=[kv_ty] * a_rows,
            names=[f"aokv_{c}_{r}" for r in range(a_rows)]))
        af = ObjectFifo(ao_l2_ty, name=f"aoaL2_{c}", depth=2)
        ao_l2l3.append(af)
        ao_cores.append(af.prod().join(
            [r * q_per * HD for r in range(a_rows)],
            depths=[1] * a_rows,
            obj_types=[ao_ty] * a_rows,
            names=[f"aoa_{c}_{r}" for r in range(a_rows)]))

    w_l3l2, y_l2l3, w_cores, y_cores = [], [], [], []
    for c in range(o_cols):
        wf = ObjectFifo(w_l2_ty, name=f"aowL2_{c}", depth=2)
        w_l3l2.append(wf)
        w_cores.append(wf.cons().split(
            [r * call_bytes for r in range(ROWS_PER_COL)],
            obj_types=[w_l1_ty] * ROWS_PER_COL,
            names=[f"aow_{c}_{r}" for r in range(ROWS_PER_COL)]))
        yf = ObjectFifo(y_l2_ty, name=f"aoyL2_{c}", depth=2)
        y_l2l3.append(yf)
        y_cores.append(yf.prod().join(
            [r * ROWS_PER_TILE for r in range(ROWS_PER_COL)],
            obj_types=[y_l1_ty] * ROWS_PER_COL,
            names=[f"aoy_{c}_{r}" for r in range(ROWS_PER_COL)]))

    def attn_body(qi, kvi, oo, state, kb, fn):
        qe = qi.acquire(1)
        ke = kvi.acquire(1)
        # Python range, not range_: the head loop is unrolled at build time.
        for h in range(q_per):
            kb(qe, ke, state, BLK, 1, h)
        kvi.release(1)
        if n_blk > 1:
            for _ in range_(n_blk - 1):
                ke = kvi.acquire(1)
                for h in range(q_per):
                    kb(qe, ke, state, BLK, 0, h)
                kvi.release(1)
        oe = oo.acquire(1)
        for h in range(q_per):
            fn(state, oe, h)
        oo.release(1)
        qi.release(1)

    def gemv_body(win, xin, yout, *ks):
        xe = xin.acquire(1)
        for _ in range_(per_core):
            ye = yout.acquire(1)
            for fn in ks:
                we = win.acquire(1)
                fn(we, xe, ye)
                win.release(1)
            yout.release(1)
        xin.release(1)

    workers = []
    for c in range(o_cols):
        for r in range(ROWS_PER_COL):
            workers.append(Worker(
                gemv_body,
                fn_args=[w_cores[c][r].cons(), of_x.cons(),
                         y_cores[c][r].prod(), *gemv],
                stack_size=0xD00))
    for c in range(a_cols):
        for r in range(a_rows):
            state = Buffer(np.ndarray[(q_per * ST_STRIDE,), np.dtype[np.float32]],
                           name=f"aost{c}_{r}")
            workers.append(Worker(
                attn_body,
                fn_args=[q_cores[c][r].cons(), kv_cores[c][r].cons(),
                         ao_cores[c][r].prod(), state, blk, fin],
                stack_size=0xD00))

    col_w = chunks * ROWS_PER_COL * call_bytes
    col_y = per_core * ROWS_PER_COL * ROWS_PER_TILE
    w_taps = TensorTiler2D.simple_tiler((1, o_cols * col_w), (1, col_w))
    y_taps = TensorTiler2D.simple_tiler((1, o_cols * col_y), (1, col_y))
    col_q = a_rows * q_per * HD
    col_kv = a_rows * n_blk * 2 * BLK * HD
    q_taps = TensorTiler2D.simple_tiler((1, a_cols * col_q), (1, col_q))
    kv_taps = TensorTiler2D.simple_tiler((1, a_cols * col_kv), (1, col_kv))
    a_taps = TensorTiler2D.simple_tiler((1, a_cols * col_q), (1, col_q))

    def sequence(t_q, t_kv, t_so, t_si, t_w, t_y, qp, kvp, aoc, wp, xp, yc):
        # Phase 1: attention. The o_proj cores have nothing to do and are simply
        # not mentioned -- a worker with no traffic in a phase costs nothing.
        tg1 = TaskGroup()
        for c in range(a_cols):
            qp[c].fill(t_q, tap=q_taps[c], group=tg1)
            kvp[c].fill(t_kv, tap=kv_taps[c], group=tg1)
            aoc[c].drain(t_so, tap=a_taps[c], wait=True, group=tg1)
        tg1.finish()
        # Phase 2: o_proj over the gathered attention output. A SEPARATE task
        # group, for the reason granite_mlp_full.py records: one group spanning
        # both phases cannot complete, because finish() would be waiting on a
        # fill the core cannot consume until finish() returns. Deadlock, not an
        # error.
        tg2 = TaskGroup()
        xp.fill(t_si, group=tg2)
        for c in range(o_cols):
            wp[c].fill(t_w, tap=w_taps[c], group=tg2)
            yc[c].drain(t_y, tap=y_taps[c], wait=True, group=tg2)
        tg2.finish()

    rt = Runtime(sequence, [all_q_ty, all_kv_ty, attn_ty, attn_ty, w_ty, y_ty,
                            [f.prod() for f in q_l3l2],
                            [f.prod() for f in kv_l3l2],
                            [f.cons() for f in ao_l2l3],
                            [f.prod() for f in w_l3l2], of_x.prod(),
                            [f.cons() for f in y_l2l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args(sys.argv[1:])
    seq, n_blk = a.seq, (a.seq + BLK - 1) // BLK

    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    n_kv_model = cfg["num_key_value_heads"]
    q_per = cfg["num_attention_heads"] // n_kv_model
    scale = cfg["attention_multiplier"]
    name = "model.layers.0.self_attn.o_proj.weight"
    n, k = projection_shape(name, cfg)
    tile_rows, k_tiles = n // ROWS_PER_TILE, k // TILE_K
    n_kv = A_COLS * A_ROWS
    assert n_kv == n_kv_model, (
        f"{A_COLS} x {A_ROWS} gives {n_kv} cores, model has {n_kv_model} kv heads")
    o_cores = O_COLS * ROWS_PER_COL
    assert tile_rows % o_cores == 0
    per_core = tile_rows // o_cores
    per_call = tiles_per_call(k_tiles, 1, k)
    n_entry = k_tiles // per_call

    f = q4nx.Q4NX(MODEL / "model.q4nx")
    off, _ = f.header[name]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + off)
        raw = fh.read(tile_rows * k_tiles * TILE_BYTES)
    w = permute_weights(raw, O_COLS, per_core, n_entry, per_call * TILE_BYTES)

    rng = np.random.default_rng(0)
    q = rng.standard_normal((n_kv, q_per, HD)).astype(np.float32).astype(bfloat16)
    K = rng.standard_normal((n_kv, n_blk * BLK, HD)).astype(np.float32).astype(bfloat16)
    V = rng.standard_normal((n_kv, n_blk * BLK, HD)).astype(np.float32).astype(bfloat16)
    # A column's memtile object holds its FOUR cores' block i back to back, and
    # split() hands each core its own 4096 bf16. So the layout is
    # column-major, then block, then core -- not core-major as it is when every
    # core has its own shim stream.
    kv = np.concatenate([
        np.concatenate([
            np.concatenate([
                np.concatenate([K[col * A_ROWS + r, i * BLK:(i + 1) * BLK].reshape(-1),
                                V[col * A_ROWS + r, i * BLK:(i + 1) * BLK].reshape(-1)])
                for r in range(A_ROWS)])
            for i in range(n_blk)])
        for col in range(A_COLS)])

    iron.set_current_device(from_name("npu2", n_cols=None))
    scratch = iron.zeros(k, dtype=bfloat16, device="npu")
    c_y = iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu")
    b = run_iters(granite_attn_o,
                  iron.tensor(q.reshape(-1), dtype=bfloat16, device="npu"),
                  iron.tensor(kv, dtype=bfloat16, device="npu"),
                  scratch, scratch,
                  iron.tensor(w, dtype=np.uint8, device="npu"), c_y,
                  seq=seq, tile_rows=tile_rows, k=k, per_call=per_call,
                  q_per=q_per, o_cols=O_COLS, a_cols=A_COLS, a_rows=A_ROWS,
                  warmup=1, iters=a.iters)
    got = unpermute_y(c_y.numpy().copy(), O_COLS, per_core).astype(np.float64)

    # Reference: attention for all 40 heads, THEN o_proj on the concatenation.
    # Both together, because checking the halves separately is what let a wrong
    # composition pass earlier in this workstream.
    attn = np.empty(n_kv * q_per * HD, np.float64)
    for c in range(n_kv):
        Kc, Vc = K[c, :seq].astype(np.float64), V[c, :seq].astype(np.float64)
        for h in range(q_per):
            s = (Kc @ q[c, h].astype(np.float64)) * scale
            s -= s.max()
            e = np.exp(s)
            attn[(c * q_per + h) * HD:(c * q_per + h + 1) * HD] = (e / e.sum()) @ Vc
    ref = reference(raw, attn.astype(np.float32).astype(bfloat16),
                    tile_rows, k_tiles).astype(np.float64)

    sc = scratch.numpy().astype(np.float64)
    a_rel = np.abs(sc - attn).max() / (np.abs(attn).max() + 1e-30)
    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    g1, r1 = got.ravel(), ref.ravel()
    cos = float(g1 @ r1 / (np.linalg.norm(g1) * np.linalg.norm(r1) + 1e-30))
    ok = cos > 0.999 and rel < 8e-2 and a_rel < 8e-2
    mb = len(raw) / 1e6
    us = b.npu.avg_us
    print(f"attention (40 heads) + o_proj, ONE dispatch   seq {seq}")
    print(f"  {O_COLS} cols o_proj ({o_cores} cores) + {A_COLS}x{A_ROWS} "
          f"attention ({n_kv} cores), per_call {per_call}, {mb:.1f} MB")
    print(f"  cosine {cos:.8f}   max rel err {rel:.3e}")
    # The intermediate separately: if the scratch matches, phase 1 is right and
    # any error is phase 2's. Guessing which half is at fault cost three builds
    # on the standalone attention design.
    print(f"  attention scratch vs reference: {a_rel:.3e}  "
          f"{'(phase 1 OK)' if a_rel < 8e-2 else '(PHASE 1 WRONG)'}")
    print(f"  {us:.1f} us device   {b.e2e.avg_us:.1f} us wall")
    print(f"  separate: attention 278 + o_proj 260 = 538 us")
    print(f"  [fused == attention then o_proj on the host]  "
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


def build_artifact(geometry: dict, seq: int = 64) -> None:
    n_blk = (seq + BLK - 1) // BLK
    n_kv_model = geometry["num_key_value_heads"]
    q_per = geometry["num_attention_heads"] // n_kv_model
    n, k = projection_shape("o_proj", geometry)
    tile_rows, k_tiles = n // ROWS_PER_TILE, k // TILE_K
    n_kv = A_COLS * A_ROWS
    per_call = tiles_per_call(k_tiles, 1, k)
    w_bytes = tile_rows * k_tiles * TILE_BYTES

    iron.set_current_device(from_name("npu2", n_cols=None))
    scratch = iron.zeros(k, dtype=bfloat16, device="npu")
    granite_attn_o(
        iron.zeros(n_kv * q_per * HD, dtype=bfloat16, device="npu"),
        iron.zeros(n_kv * n_blk * 2 * BLK * HD, dtype=bfloat16, device="npu"),
        scratch, scratch,
        iron.zeros(w_bytes, dtype=np.uint8, device="npu"),
        iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu"),
        seq=seq, tile_rows=tile_rows, k=k, per_call=per_call,
        q_per=q_per, o_cols=O_COLS, a_cols=A_COLS, a_rows=A_ROWS)


if __name__ == "__main__":
    sys.exit(main())
