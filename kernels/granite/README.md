# granite kernels

AIE kernels for IBM Granite 4.2 3B. A companion pull request adds a host
engine that runs the same model on the CPU; these kernels run its arithmetic on
the NPU instead. **Neither PR depends on the other** — this one adds no code
that `flm` links, and the engine needs nothing from here.

## Geometry

Hidden 2560, 40 layers, 40 query heads over 8 kv heads, head_dim 64,
intermediate 8192, vocab 100352, RoPE theta 1e7 with the half-split
(`rotate_half`) convention. All of it in [geometry.json](geometry.json), which
is what the build reads — **no model weights are needed to produce an
artefact.**

Granite needs head_dim 64 at hidden 2560. Every shipped design at hidden >= 2560
is head_dim 128, and head_dim cannot be padded, so nothing existing could be
reused for it.

## The layer, in four dispatches

| dispatch | ops | cores | device time |
|---|---|---|---|
| `norm_qkv_rope` | RMSNorm, q, k, v, RoPE | 28 | 286.8 µs |
| `attn_o` | attention (all 40 heads), o_proj | 8 + 16 | 330.2 µs |
| `norm_gate_up` | RMSNorm, gate, up | 32 | 669.2 µs |
| `swiglu_down` | SwiGLU, down | 20 | 458.5 µs |
| | | **layer** | **1744.7 µs** |

×40 layers plus lm_head (3.66 ms, 43.9 GB/s) is **73.5 ms/token = 13.6 tok/s**
of device time, against 8.5 tok/s for the same work in nine dispatches, and
8.7 tok/s end to end for the CPU host engine.

Measured on a Ryzen AI 9 HX 370, medians of repeated runs at `--iters 200`.
Every design checks against a host reference built from the same bytes; cosines
are in each design's own output (1.00000000 for the GEMV groups under a one-hot
activation, > 0.9996 for the fused blocks).

## Why fuse the small ops and not the big ones

The per-dispatch floor is about 200 µs regardless of size: RMSNorm on 2560
values costs 244 µs, SwiGLU on 8192 costs 205, and RoPE cost **nothing** when it
moved inside a dispatch that was already running. Four GEMV groups move 49.2 MB
and cost 1.65 ms — nearly all real work. Five small ops move almost nothing and
cost 1.20 ms, of which ~1.0 ms is floor.

So the fusions here all absorb a *small* op into a dispatch that was already
paying for bandwidth.

## What was tried and rejected

Worth recording, because it is the evidence for the shape above.

**Fusing the whole MLP into one dispatch is a net loss.** Measured 2.32 ms at 8
cores and 1.73 ms at 16, against 1.31 ms for the same three ops unfused across
20–32 cores. Even a perfect one-dispatch MLP at gate_up's 39.7 GB/s would be
1.19 ms. Fusion saves 0.40 ms of dispatch floor and costs more than that in
width, because fusion and width compete for the same scarce resources.

**Two independent costs make wide fusion expensive.** Shim DMA channels: the
device has 16 each way, and private per-core streams at 16 cores want 17 in and
32 out. Routing through the memtile (one stream per column, `split()`/`join()`)
costs 5 and 8. And `PER_CALL` must divide the K-tile count of *every* matrix in
a dispatch, so fusing gate/up (10 tiles) with down (32) pins it at gcd = 2 and
halves the DMA element.

**The same trade goes the other way for SwiGLU + down**, which is why it is in
the table above. Taking gate and up as the activation forces `per_call` from 4
to 2, but down_proj has 32 K-tiles so that is still a long stream: 29.8 GB/s
becomes 28.8, a 3% cost against a 205 µs dispatch that moves no weights at all.
**The same decision, opposite outcomes, decided by K.**

## Four hardware limits, in the order they bind

None of these is derivable from bandwidth and L1 alone; each appeared only when
the placer ran. They are recorded in the sources at the point where they bind.

| resource | budget | what it forces |
|---|---|---|
| shim MM2S / S2MM | 16 / 16 device-wide | streams per column, not per core |
| memtile DMA | ~6 in / 6 out per column | at most two split/join structures of four |
| compute tile DMA | 2 in / 2 out | at most two input streams per core |
| L1 | 62208 B | sets `per_call`, and so bandwidth |

The compute-tile limit is why several kernels write **in place**: with weights
and one activation already using both input channels, a fused op has no third
stream, so the norm weight rides in the activation buffer and the result is
written back over its input.

One more, which cost the most to find: **`aie::store_v` of 32 floats is a
128-byte operation**, so a per-head state array must be 128-byte aligned. A
72-float stride is 32-byte aligned and not enough — the second head's store
rounds down onto the first head's softmax denominator, and every head comes out
with the right direction and the wrong scale.
