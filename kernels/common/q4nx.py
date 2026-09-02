"""FastFlowLM `q4nx` container: reader and host-side reference dequantiser.

`.q4nx` is a plain safetensors file (u64 header length, JSON header, data). The
weight tensors inside are pre-tiled for the NPU, shaped `[N/32][K/256][bytes]`
-- one tile is 32 output rows x 256 K, split into two row-blocks of 16.

    q4  5120 B  [512 B bf16 d][512 B bf16 m][4096 B packed nibbles]
    q8  8704 B  [512 B bf16 scale][8192 B int8]

Within a tile, with rb = row block (0..1), r = row in block (0..15),
k = 0..255, kb = k // 32:

    metadata index  = kb * 32 + rb * 16 + r        (256 entries per plane)
    weight index i  = rb * 4096 + k * 16 + r       (8192 weights)
    q4 byte, nibble = i >> 1, low nibble when i is even

    q4:  w = code * d + m        (GGUF Q4_1 semantics -- scale and minimum)
    q8:  w = code * scale        (symmetric)

This was not guessed. The q8 form was solved against ground truth: layer 0's
`ssm_alpha_proj` is stored twice in the same file, once quantised and once as
bf16, and under this mapping all 8192 codes of all four tiles reproduce the bf16
weights to within one quantisation step. The q4 form then follows from three
independent checks -- `d` is everywhere positive, `m` everywhere negative,
mean(m/d) = -7.48 (the Q4_1 signature for a symmetric weight distribution), and
every group's codes span 0..15.

The nibble parity -- whether the low nibble is the even or the odd weight index
-- could not be settled from the file alone, since it only swaps adjacent output
rows and every group stays a valid Q4_1 group either way. It is settled now, by
diffing against the upstream bf16 checkpoint the model was quantised from:
LOW_NIBBLE_IS_EVEN = True scores cosine 0.9975, False scores 0.0298. See
reference/check_weights.py.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
from ml_dtypes import bfloat16

GROUP = 32  # weights per quantisation group
TILE_ROWS = 32  # output rows per tile
TILE_K = 256  # K per tile
ROW_BLOCK = 16  # rows per row-block; a tile holds two
N_META = TILE_ROWS * (TILE_K // GROUP)  # 256 scale entries per plane
N_WEIGHTS = TILE_ROWS * TILE_K  # 8192

Q4_TILE_BYTES = 2 * 2 * N_META + N_WEIGHTS // 2  # 5120
Q8_TILE_BYTES = 2 * N_META + N_WEIGHTS  # 8704

LOW_NIBBLE_IS_EVEN = True  # measured, not assumed: see the module docstring


def read_header(f: BinaryIO) -> dict:
    """Parse the safetensors header. Reads only the header, not the data."""
    (n,) = struct.unpack("<Q", f.read(8))
    return json.loads(f.read(n))


def split_q4_tile(raw: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One 5120-byte q4 tile -> (nibbles uint8[4096], d bf16[256], m bf16[256]).

    Already planar in the file, so this is three slices. Convenient, because a
    core has only 2 input DMA streams (trap 3b): d and m ride together in the
    first 1024 bytes and the nibbles take the other stream.
    """
    if len(raw) != Q4_TILE_BYTES:
        raise ValueError(f"{len(raw)} bytes, expected a {Q4_TILE_BYTES} B q4 tile")
    b = np.frombuffer(raw, dtype=np.uint8)
    d = b[0:512].copy().view(bfloat16)
    m = b[512:1024].copy().view(bfloat16)
    return b[1024:].copy(), d, m


def unpack_nibbles(nib: np.ndarray) -> np.ndarray:
    """uint8[4096] -> uint8[8192] raw 0..15 codes, in weight-index order."""
    out = np.empty(N_WEIGHTS, dtype=np.uint8)
    lo, hi = nib & 0x0F, nib >> 4
    first, second = (lo, hi) if LOW_NIBBLE_IS_EVEN else (hi, lo)
    out[0::2], out[1::2] = first, second
    return out


def meta_index() -> np.ndarray:
    """Metadata index for each of the 8192 weight positions: kb*32 + rb*16 + r."""
    i = np.arange(N_WEIGHTS)
    rb, rest = divmod(i, 4096)
    k, r = divmod(rest, ROW_BLOCK)
    return (k // GROUP) * TILE_ROWS + rb * ROW_BLOCK + r


def dequant_q4(nib: np.ndarray, d: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The reference: w = code * d + m, accumulated in float32, returned bf16."""
    g = meta_index()
    x = unpack_nibbles(nib).astype(np.float32)
    return (x * d.astype(np.float32)[g] + m.astype(np.float32)[g]).astype(bfloat16)


def dequant_q8(raw: bytes) -> np.ndarray:
    """One 8704-byte q8 tile -> bf16[8192] in weight-index order."""
    if len(raw) != Q8_TILE_BYTES:
        raise ValueError(f"{len(raw)} bytes, expected a {Q8_TILE_BYTES} B q8 tile")
    b = np.frombuffer(raw, dtype=np.uint8)
    s = b[0:512].copy().view(bfloat16).astype(np.float32)
    c = b[512:].view(np.int8).astype(np.float32)
    return (c * s[meta_index()]).astype(bfloat16)


def to_matrix(flat: np.ndarray) -> np.ndarray:
    """Weight-index order -> [32 rows][256 K], the natural view of a tile."""
    return flat.reshape(2, TILE_K, ROW_BLOCK).transpose(0, 2, 1).reshape(TILE_ROWS, TILE_K)


def pack_meta(d: np.ndarray, m: np.ndarray) -> np.ndarray:
    """d and m as one bf16 buffer -- one input stream, not two (trap 3b)."""
    return np.concatenate([np.asarray(d), np.asarray(m)]).astype(bfloat16)


# ---------------------------------------------------------------- whole file


def _q4_tiles(b: np.ndarray) -> np.ndarray:
    """[T, 5120] uint8 -> [T, 8192] float32, weight-index order."""
    d = b[:, 0:512].copy().view(bfloat16).astype(np.float32)
    m = b[:, 512:1024].copy().view(bfloat16).astype(np.float32)
    nib = b[:, 1024:]
    codes = np.empty((b.shape[0], N_WEIGHTS), dtype=np.uint8)
    lo, hi = nib & 0x0F, nib >> 4
    first, second = (lo, hi) if LOW_NIBBLE_IS_EVEN else (hi, lo)
    codes[:, 0::2], codes[:, 1::2] = first, second
    g = meta_index()
    return codes.astype(np.float32) * d[:, g] + m[:, g]


def _q8_tiles(b: np.ndarray) -> np.ndarray:
    """[T, 8704] uint8 -> [T, 8192] float32, weight-index order."""
    s = b[:, 0:512].copy().view(bfloat16).astype(np.float32)
    return b[:, 512:].view(np.int8).astype(np.float32) * s[:, meta_index()]


def _untile(flat: np.ndarray, n_t: int, k_t: int) -> np.ndarray:
    """[n_t*k_t, 8192] -> [n_t*32, k_t*256], undoing the row-block interleave."""
    m = flat.reshape(n_t, k_t, 2, TILE_K, ROW_BLOCK).transpose(0, 2, 4, 1, 3)
    return m.reshape(n_t * TILE_ROWS, k_t * TILE_K)


class Q4NX:
    """A `.q4nx` model file. Tensors come back dequantised and un-tiled."""

    _DT = {"BF16": bfloat16, "F32": np.float32, "F16": np.float16, "I8": np.int8}

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            self.header = read_header(f)
            self._data_start = f.tell()
        self.header.pop("__metadata__", None)

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def names(self) -> list[str]:
        return sorted(self.header)

    def raw(self, name: str) -> bytes:
        first, last = self.header[name]["data_offsets"]
        with self.path.open("rb") as f:
            f.seek(self._data_start + first)
            return f.read(last - first)

    def tensor(self, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
        """Dequantised and un-tiled. `shape` trims the tile padding.

        A tile is 32 rows x 256 K, so a tensor whose N or K is not a multiple of
        those is stored padded -- N=16 becomes 32 rows, the upper half a copy of
        the lower. Pass the logical shape and the padding is dropped.
        """
        e = self.header[name]
        if e["dtype"] != "I8":  # stored as-is: embeddings, norms, conv1d, biases
            v = np.frombuffer(self.raw(name), dtype=self._DT[e["dtype"]])
            return v.reshape(e["shape"])

        n_t, k_t, tile_bytes = e["shape"]
        b = np.frombuffer(self.raw(name), dtype=np.uint8).reshape(n_t * k_t, tile_bytes)
        if tile_bytes == Q4_TILE_BYTES:
            flat = _q4_tiles(b)
        elif tile_bytes == Q8_TILE_BYTES:
            flat = _q8_tiles(b)
        else:
            raise ValueError(f"{name}: {tile_bytes} B/tile is neither q4 nor q8")
        w = _untile(flat, n_t, k_t)
        return w if shape is None else w[: shape[0], : shape[1]]
