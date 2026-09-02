r"""RMSNorm + q + k + v + RoPE. FOUR ops, ONE dispatch, 28 cores.

WHY THIS BREAKS THE 8-CORE CAP granite_qkv.py HIT
--------------------------------------------------
granite_qkv.py fuses the same four ops and is stuck at 8 cores. The reason
looked structural: granite has 8 kv heads, RoPE is head-local, so a core must
own whole heads -- and there are only 8 to go round.

But that follows from ITS layout, where every core gets a slice of q AND k AND
v. Concatenating the three along N instead gives each core a slice of exactly
one of them, and then the arithmetic is different: k is 16 tile-rows over 4
dedicated cores = 4 tile-rows = 2 whole heads each. The cap was in the design,
not in the model.

    112 tile-rows = 80 (q) + 16 (k) + 16 (v), over 7 columns x 4 rows:
      columns 0-4   ->  q, 2 heads per core, rotated
      column  5     ->  k, 2 heads per core, rotated
      column  6     ->  v, 128 values per core, NOT rotated, only narrowed

Three roles, so three core bodies, chosen per worker by column. The RoPE kernel
already takes (q_heads, k_heads, v_len) and does the right thing for each.

L1, WHICH IS WHAT DECIDES THE SHAPE OF THIS
-------------------------------------------
    weights, per_call 5, double buffered      51200
    activation [x | norm weight | cos,sin]    10368
    accumulator                                 512
                                              -----
                                              62080   against a 62208 budget

128 bytes spare. That is why the float32 accumulator IS the output fifo element
and the rotated result narrows into its first half (granite_rope_ip.cc): a
separate 256 B output element would put it 128 B over, and dropping per_call to
2 -- the only other divisor of 10 K-tiles -- would cost more bandwidth than all
three fused ops save.

    call c:\dev\mlir-aie\iron_env.cmd
    python designs\granite_gemv\granite_qkv_wide.py
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



ROWS_PER_COL = 4
HD = 64                 # head_dim
PER_CALL = 5


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def granite_qkv_wide(w: In, xw: In, out: Out, *,
                     q_rows: CompileTime[int], kv_rows: CompileTime[int],
                     k: CompileTime[int], n_cols: CompileTime[int] = 7):
    k_tiles = k // TILE_K
    n_entry = k_tiles // PER_CALL
    call_bytes = PER_CALL * TILE_BYTES
    n_cores = n_cols * ROWS_PER_COL
    tile_rows = q_rows + 2 * kv_rows
    per_core = tile_rows // n_cores
    slice_f = per_core * ROWS_PER_TILE          # 128 floats = 2 heads
    heads_per_core = slice_f // HD
    chunks = per_core * n_entry

    # Which columns own q, which own k, which own v. Whole columns, because the
    # memtile split hands a column's stream to its own four cores.
    q_cols = q_rows // (per_core * ROWS_PER_COL)
    k_cols = kv_rows // (per_core * ROWS_PER_COL)

    w_l1_ty = np.ndarray[(call_bytes,), np.dtype[np.uint8]]
    w_l2_ty = np.ndarray[(ROWS_PER_COL * call_bytes,), np.dtype[np.uint8]]
    # The output element is float32 and doubles as the accumulator; the RoPE
    # epilogue narrows into its first half. See granite_rope_ip.cc.
    y_l1_ty = np.ndarray[(slice_f,), np.dtype[np.float32]]
    y_l2_ty = np.ndarray[(ROWS_PER_COL * slice_f,), np.dtype[np.float32]]
    act_ty = np.ndarray[(2 * k + 2 * (HD // 2),), np.dtype[bfloat16]]
    w_ty = np.ndarray[(tile_rows * k_tiles * TILE_BYTES,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(tile_rows * ROWS_PER_TILE,), np.dtype[np.float32]]

    gemv = [ExternalFunction(f"granite_qgemv_g{i}",
                             source_file=str(AIE / f"granite_qgemv_g{i}.cc"),
                             arg_types=[w_l1_ty, act_ty, y_l1_ty, np.int32],
                             include_dirs=_include_dirs())
            for i in range(n_entry)]
    rmsnorm = ExternalFunction("granite_rms_norm_ip",
                               source_file=str(AIE / "granite_rmsnorm_ip.cc"),
                               arg_types=[act_ty, np.int32],
                               include_dirs=_include_dirs())
    rope = ExternalFunction("granite_rope_ip",
                            source_file=str(AIE / "granite_rope_ip.cc"),
                            arg_types=[y_l1_ty, act_ty, np.int32, np.int32,
                                       np.int32],
                            include_dirs=_include_dirs())

    of_x = ObjectFifo(act_ty, name="qx", depth=1)

    w_l3l2, y_l2l3, w_cores, y_cores = [], [], [], []
    for c in range(n_cols):
        wf = ObjectFifo(w_l2_ty, name=f"qwL2_{c}", depth=2)
        w_l3l2.append(wf)
        w_cores.append(wf.cons().split(
            [r * call_bytes for r in range(ROWS_PER_COL)],
            obj_types=[w_l1_ty] * ROWS_PER_COL,
            names=[f"qw_{c}_{r}" for r in range(ROWS_PER_COL)]))
        yf = ObjectFifo(y_l2_ty, name=f"qyL2_{c}", depth=2)
        y_l2l3.append(yf)
        # depths=1: a core produces exactly ONE output element per dispatch, so
        # the default double buffer buys no overlap and costs 512 B of L1 --
        # which is more than the 128 B this design has spare. Measured as a
        # build failure, not guessed:
        #   qy_6_3_buff_0 and _1, 512 bytes each -> Basic sequential allocation failed
        y_cores.append(yf.prod().join(
            [r * slice_f for r in range(ROWS_PER_COL)],
            depths=[1] * ROWS_PER_COL,
            obj_types=[y_l1_ty] * ROWS_PER_COL,
            names=[f"qy_{c}_{r}" for r in range(ROWS_PER_COL)]))

    def make_body(qh, kh, vl):
        """One body per role. The counts are baked per worker, not branched on."""
        def body(win, xin, yout, norm, rp, *ks):
            xe = xin.acquire(1)
            norm(xe, k)                      # redundant, in place, on every core
            ye = yout.acquire(1)
            for r in range_(per_core):
                for fn in ks:
                    we = win.acquire(1)
                    fn(we, xe, ye, r)
                    win.release(1)
            rp(ye, xe, qh, kh, vl)           # narrows into ye's first half
            yout.release(1)
            xin.release(1)
        return body

    workers = []
    for c in range(n_cols):
        if c < q_cols:
            body = make_body(heads_per_core, 0, 0)
        elif c < q_cols + k_cols:
            body = make_body(0, heads_per_core, 0)
        else:
            body = make_body(0, 0, slice_f)   # v is narrowed, never rotated
        for r in range(ROWS_PER_COL):
            workers.append(Worker(
                body,
                fn_args=[w_cores[c][r].cons(), of_x.cons(),
                         y_cores[c][r].prod(), rmsnorm, rope, *gemv],
                stack_size=0xD00))

    col_w = chunks * ROWS_PER_COL * call_bytes
    col_y = ROWS_PER_COL * slice_f
    w_taps = TensorTiler2D.simple_tiler((1, n_cols * col_w), (1, col_w))
    y_taps = TensorTiler2D.simple_tiler((1, n_cols * col_y), (1, col_y))

    def sequence(a_w, a_x, c_y, w_prods, x_prod, y_conss):
        tg = TaskGroup()
        x_prod.fill(a_x, group=tg)
        for c in range(n_cols):
            w_prods[c].fill(a_w, tap=w_taps[c], group=tg)
            y_conss[c].drain(c_y, tap=y_taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [w_ty, act_ty, out_ty,
                            [f.prod() for f in w_l3l2], of_x.prod(),
                            [f.cons() for f in y_l2l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main(argv: list[str] | None = None) -> int:
    import argparse
    from aie.utils.benchmark import run_iters
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pos", type=int, default=17)
    a = ap.parse_args((argv or sys.argv)[1:])

    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    names = [f"model.layers.0.self_attn.{p}.weight"
             for p in ("q_proj", "k_proj", "v_proj")]
    nq, k = projection_shape(names[0], cfg)
    nkv, _ = projection_shape(names[1], cfg)
    k_tiles = k // TILE_K
    q_rows, kv_rows = nq // ROWS_PER_TILE, nkv // ROWS_PER_TILE
    tile_rows = q_rows + 2 * kv_rows

    f = q4nx.Q4NX(MODEL / "model.q4nx")
    raws = []
    for nm, rows in zip(names, (q_rows, kv_rows, kv_rows)):
        off, _ = f.header[nm]["data_offsets"]
        with f.path.open("rb") as fh:
            fh.seek(f._data_start + off)
            raws.append(fh.read(rows * k_tiles * TILE_BYTES))
    raw = b"".join(raws)

    off, _ = f.header["model.layers.0.input_layernorm.weight"]["data_offsets"]
    with f.path.open("rb") as fh:
        fh.seek(f._data_start + off)
        nwv = np.frombuffer(fh.read(k * 2), dtype=bfloat16)

    n_cols = 7
    n_cores = n_cols * ROWS_PER_COL
    assert tile_rows % n_cores == 0
    per_core = tile_rows // n_cores
    n_entry = k_tiles // PER_CALL
    slice_f = per_core * ROWS_PER_TILE
    w = permute_weights(raw, n_cols, per_core, n_entry, PER_CALL * TILE_BYTES)

    half = HD // 2
    theta = float(cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"])
    inv_f = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / HD))
    ang = a.pos * inv_f
    rng = np.random.default_rng(0)
    act = np.zeros(2 * k + HD, np.float32)
    act[:k] = rng.standard_normal(k)
    act[k:2 * k] = nwv.astype(np.float32)
    act[2 * k:2 * k + half] = np.cos(ang)
    act[2 * k + half:2 * k + HD] = np.sin(ang)
    act_bf = act.astype(bfloat16)

    iron.set_current_device(from_name("npu2", n_cols=None))
    c_y = iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu")
    b = run_iters(granite_qkv_wide,
                  iron.tensor(w, dtype=np.uint8, device="npu"),
                  iron.tensor(act_bf, dtype=bfloat16, device="npu"), c_y,
                  q_rows=q_rows, kv_rows=kv_rows, k=k, n_cols=n_cols,
                  warmup=1, iters=a.iters)
    # Each core's 512 B element holds its 128 bf16 results in the first 256 B --
    # the accumulator it narrowed into. Read the bytes, not the floats.
    raw_y = c_y.numpy().tobytes()
    got = np.concatenate([
        np.frombuffer(raw_y[i * slice_f * 4: i * slice_f * 4 + slice_f * 2],
                      dtype=bfloat16) for i in range(n_cores)]).astype(np.float64)

    xf = act[:k]
    invn = 1.0 / np.sqrt((xf * xf).mean() + cfg["rms_norm_eps"])
    h = (xf * invn * nwv.astype(np.float32)).astype(bfloat16)
    co, si = np.cos(ang), np.sin(ang)

    def rope(y):
        hh = y.reshape(-1, HD)
        r = np.empty_like(hh)
        r[:, :half] = hh[:, :half] * co - hh[:, half:] * si
        r[:, half:] = hh[:, half:] * co + hh[:, :half] * si
        return r.reshape(-1)

    q = reference(raws[0], h, q_rows, k_tiles).astype(np.float64)
    kk = reference(raws[1], h, kv_rows, k_tiles).astype(np.float64)
    v = reference(raws[2], h, kv_rows, k_tiles).astype(np.float64)
    ref = np.concatenate([rope(q), rope(kk), v])

    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    cos_ = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    # v must NOT be rotated: checked separately, or a kernel that rotated
    # everything would still pass on the q and k majority.
    vn = 2 * kv_rows * ROWS_PER_TILE
    v_rel = np.abs(got[-vn // 2:] - ref[-vn // 2:]).max() / (
        np.abs(ref[-vn // 2:]).max() + 1e-30)
    mb = len(raw) / 1e6
    us = b.npu.avg_us
    ok = cos_ > 0.9999 and rel < 8e-3
    print(f"RMSNorm + q + k + v + RoPE, ONE dispatch   "
          f"{n_cols} cols x {ROWS_PER_COL} = {n_cores} cores, per_call {PER_CALL}")
    print(f"  {q_rows * ROWS_PER_TILE} q + {kv_rows * ROWS_PER_TILE} k + "
          f"{kv_rows * ROWS_PER_TILE} v   K={k}   {mb:.1f} MB")
    print(f"  cosine {cos_:.8f}   max rel err {rel:.3e}   "
          f"v (unrotated) {v_rel:.3e}")
    print(f"  {us:.1f} us device   {b.e2e.avg_us:.1f} us wall   "
          f"{mb / us * 1e3:.1f} GB/s")
    print(f"  separate: RMSNorm 244 + qkv 290 + RoPE 226 = 760 us   "
          f"(norm+qkv fused was 365 + 226 = 591)")
    print(f"  [fused == norm, then three GEMVs, then RoPE on q and k]  "
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


def build_artifact(geometry: dict) -> None:
    nq, k = projection_shape("q_proj", geometry)
    nkv, _ = projection_shape("k_proj", geometry)
    q_rows, kv_rows = nq // ROWS_PER_TILE, nkv // ROWS_PER_TILE
    tile_rows = q_rows + 2 * kv_rows
    k_tiles = k // TILE_K
    n_cols = 7
    w_bytes = tile_rows * k_tiles * TILE_BYTES

    iron.set_current_device(from_name("npu2", n_cols=None))
    granite_qkv_wide(
        iron.zeros(w_bytes, dtype=np.uint8, device="npu"),
        iron.zeros(2 * k + HD, dtype=bfloat16, device="npu"),
        iron.zeros(tile_rows * ROWS_PER_TILE, dtype=np.float32, device="npu"),
        q_rows=q_rows, kv_rows=kv_rows, k=k, n_cols=n_cols)


if __name__ == "__main__":
    sys.exit(main())
