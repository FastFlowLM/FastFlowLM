#!/usr/bin/env python3
"""Task 12 Step 3: compare FastFlow's Phi-4 AIE4 engine against the corelib
reference driver, on the same explicit token IDs and the same forced route.

Two modes, and they are separate PROCESSES on purpose:

  emit-reference   loads the reference driver, runs it, writes JSON, and
                   calls corelib.cleanup() before exiting.
  compare          reads two JSON files and holds no device context at all.

Nothing here ever runs while the C++ harness is running. Two processes holding
AIE4 device contexts at once fail in ways that look like defects, so the suite
script serialises them and this file makes that easy to honour rather than
something to remember.

Three things about the `e5258d2` reference are worth stating, because each one
would otherwise look like a bug in this comparator:

  * `Phi4.forward` and `Phi4.logits_for` return FP32 arrays directly. Corelib
    widens on `tensor_read`, so there is no `from_bf16` unpacking left to
    mirror. It is FASTFLOW's BF16 logits that get widened here.
  * The reference driver still uses the collapsed two-synchronize-per-layer
    schedule. Design Section 10.4 no longer considers that sound and FastFlow
    deliberately uses four. Its VALUES are the reference; its SCHEDULE is not,
    so synchronize counts are never compared.
  * The driver is read-only. No `--continuation-route` option is added to it
    and that repository is not modified; the routes are composed here out of
    `Phi4.forward` calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Thresholds from design Section 12.4. They are named rather than inlined so a
# report can quote the number that actually ran.
MIN_CORRELATION = 0.9999
MAX_TOP32_ABS_DIFF = 0.25
MIN_DECODE_STEPS = 16
TOP_K = 32
TOP_5 = 5


def _import_reference():
    """Import the corelib reference driver from RYZENAI_CORELIB_SOURCE.

    Imported by path rather than vendored: a copy would drift, and the whole
    value of this comparison is that the reference is the corelib repository's
    own driver rather than a second transcription of the same guess.
    """
    source = os.environ.get("RYZENAI_CORELIB_SOURCE")
    if not source:
        raise SystemExit(
            "RYZENAI_CORELIB_SOURCE is not set. Point it at the "
            "ryzenai-corelib checkout whose python/ holds phi4_driver.py."
        )
    python_dir = Path(source) / "python"
    if not (python_dir / "phi4_driver.py").is_file():
        raise SystemExit(f"no phi4_driver.py under {python_dir}")
    sys.path.insert(0, str(python_dir))
    import phi4_driver  # noqa: E402
    import ryzenai_corelib as corelib  # noqa: E402

    return phi4_driver, corelib


def _argmax_lowest(values) -> int:
    """The lowest ID among ties.

    `np.argmax` already returns the first maximal index, which for a dense
    logit vector IS the lowest ID. It is spelled out because FastFlow's
    `ArgmaxLowest` promises the same thing and the comparison below is only
    meaningful if both sides use one convention.
    """
    return int(np.argmax(np.asarray(values)))


def _widen_bf16(bits) -> np.ndarray:
    """FastFlow emits raw BF16 bit patterns; the reference is already FP32.

    BF16 to FP32 is a 16-bit left shift and nothing else, so this is exact and
    introduces no error of its own into the comparison below.
    """
    raw = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return np.ascontiguousarray(raw).view(np.float32)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def _read_live_cache(tensor, position: int, driver) -> list[int]:
    """Live rows only, in FastFlow's [head][row][head_size] order.

    The cache is allocated at the full 4096-row window but only [0, position)
    has ever been written. Comparing the tail would be comparing uninitialised
    device memory on both sides and calling the agreement a result.
    """
    if position <= 0:
        return []
    head_size = driver.HEAD_SIZE
    max_seq = driver.MAX_SEQ
    out = []
    for head in range(driver.KV_HEADS):
        offset = (head * max_seq) * head_size
        raw = tensor.read(position * head_size, offset, driver.DataType.BF16)
        out.extend(np.frombuffer(raw, dtype=np.uint16).tolist())
    return out


def emit_reference(args) -> int:
    driver, corelib = _import_reference()

    if not corelib.load_library().ryzenai_corelib_has_device_context():
        raise SystemExit(
            "no AIE4 device context for the reference driver. Check that "
            "nothing else is holding one; the C++ harness must not be "
            "running concurrently."
        )

    plan = json.loads(Path(args.token_ids_json).read_text(encoding="utf-8"))
    if isinstance(plan, list):
        prefix, suffix = list(plan), []
    else:
        prefix = list(plan["prefix"])
        suffix = list(plan.get("suffix", []))

    model_onnx = Path(args.model_dir) / "model.onnx"
    if not model_onnx.is_file():
        raise SystemExit(f"no model.onnx under {args.model_dir}")

    record = {
        "continuation_route": args.continuation_route,
        "prefix_ids": prefix,
        "suffix_ids": suffix,
    }
    try:
        model = driver.Phi4(model_onnx)

        # Routes composed here, out of Phi4.forward calls, exactly as design
        # Section 12.4 describes them.
        if args.continuation_route == "force_append":
            hidden = model.forward(
                model.embed_rows(prefix), len(prefix), 0
            )
            position = len(prefix)
            for token in suffix:
                hidden = model.forward(
                    model.embed_rows([token]), 1, position
                )
                position += 1
        elif args.continuation_route == "force_reprefill":
            # A fresh process is a cleared reference state, and the full
            # rendered history is recomputed from position zero in one call.
            full = prefix + suffix
            hidden = model.forward(model.embed_rows(full), len(full), 0)
            position = len(full)
        else:
            raise SystemExit(
                "--continuation-route must be force_append or "
                "force_reprefill"
            )

        logits = model.logits_for(hidden)
        record["continuation"] = {
            "logits": logits.tolist(),
            "top1_id": _argmax_lowest(logits),
        }

        decode = []
        token = _argmax_lowest(logits)
        for _ in range(args.decode_steps):
            hidden = model.forward(model.embed_rows([token]), 1, position)
            position += 1
            logits = model.logits_for(hidden)
            step_top1 = _argmax_lowest(logits)
            decode.append(
                {
                    "input_id": token,
                    "logits": logits.tolist(),
                    "top1_id": step_top1,
                }
            )
            token = step_top1
        record["decode"] = decode
        record["final_position"] = position
        record["final_snapshot"] = {
            "position": position,
            "layer0_k": _read_live_cache(model.k_cache[0], position, driver),
            "layer0_v": _read_live_cache(model.v_cache[0], position, driver),
            "layer31_k": _read_live_cache(model.k_cache[-1], position, driver),
            "layer31_v": _read_live_cache(model.v_cache[-1], position, driver),
        }
    finally:
        # Released before the C++ harness starts. The suite runs this process
        # to completion for exactly this reason.
        corelib.cleanup()

    Path(args.output_json).write_text(json.dumps(record), encoding="utf-8")
    print(f"reference written to {args.output_json}")
    return 0


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


class Failures:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.messages.append(message)

    def report(self, label: str) -> int:
        if not self.messages:
            print(f"{label}: PASS")
            return 0
        print(f"{label}: FAIL")
        for message in self.messages:
            print(f"  - {message}")
        return 1


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _compare_step(
    failures: Failures,
    label: str,
    mine_bits,
    theirs,
) -> None:
    mine = _widen_bf16(mine_bits)
    reference = np.asarray(theirs, dtype=np.float32)
    if mine.shape != reference.shape:
        failures.check(
            False,
            f"{label}: logit vector length {mine.shape} vs "
            f"{reference.shape}",
        )
        return

    if not np.all(np.isfinite(mine)) or not np.all(np.isfinite(reference)):
        failures.check(False, f"{label}: non-finite logits")
        return

    correlation = _correlation(mine, reference)
    failures.check(
        correlation >= MIN_CORRELATION,
        f"{label}: correlation {correlation:.8f} < {MIN_CORRELATION}",
    )

    mine_top1 = _argmax_lowest(mine)
    reference_top1 = _argmax_lowest(reference)
    failures.check(
        mine_top1 == reference_top1,
        f"{label}: top-1 {mine_top1} vs {reference_top1}",
    )

    # Ties resolve to the lowest ID on both sides. Asserting it directly
    # matters because a tie is exactly where two argmax conventions diverge,
    # and BF16 logits tie far more often than FP32 ones.
    for name, values, chosen in (
        ("fastflow", mine, mine_top1),
        ("reference", reference, reference_top1),
    ):
        tied = np.flatnonzero(values == values[chosen])
        failures.check(
            int(tied[0]) == chosen,
            f"{label}: {name} argmax {chosen} is not the lowest tied ID "
            f"{int(tied[0])}",
        )

    mine_top5 = set(np.argsort(-mine, kind="stable")[:TOP_5].tolist())
    reference_top5 = set(
        np.argsort(-reference, kind="stable")[:TOP_5].tolist()
    )
    failures.check(
        mine_top5 == reference_top5,
        f"{label}: top-5 {sorted(mine_top5)} vs {sorted(reference_top5)}",
    )

    union = sorted(
        set(np.argsort(-mine, kind="stable")[:TOP_K].tolist())
        | set(np.argsort(-reference, kind="stable")[:TOP_K].tolist())
    )
    index = np.asarray(union, dtype=np.int64)
    max_abs = float(np.max(np.abs(mine[index] - reference[index])))
    failures.check(
        max_abs <= MAX_TOP32_ABS_DIFF,
        f"{label}: max |diff| over the union top-{TOP_K} is {max_abs:.4f} "
        f"> {MAX_TOP32_ABS_DIFF}",
    )


def compare(args) -> int:
    mine = json.loads(Path(args.fastflow_json).read_text(encoding="utf-8"))
    theirs = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))
    failures = Failures()

    failures.check(
        mine["continuation_route"].replace("force_", "")
        == theirs["continuation_route"].replace("force_", ""),
        f"route {mine['continuation_route']} vs "
        f"{theirs['continuation_route']}",
    )
    failures.check(
        mine["prefix_ids"] == theirs["prefix_ids"]
        and mine["suffix_ids"] == theirs["suffix_ids"],
        "the two runs did not use the same explicit token IDs",
    )

    _compare_step(
        failures,
        "continuation",
        mine["continuation"]["logits_bf16"],
        theirs["continuation"]["logits"],
    )

    steps = min(len(mine["decode"]), len(theirs["decode"]))
    failures.check(
        steps >= MIN_DECODE_STEPS,
        f"only {steps} decode steps compared; design Section 12.4 requires "
        f"at least {MIN_DECODE_STEPS}",
    )
    for index in range(steps):
        mine_step = mine["decode"][index]
        reference_step = theirs["decode"][index]
        failures.check(
            mine_step["input_id"] == reference_step["input_id"],
            f"decode[{index}]: fed {mine_step['input_id']} vs "
            f"{reference_step['input_id']}",
        )
        _compare_step(
            failures,
            f"decode[{index}]",
            mine_step["logits_bf16"],
            reference_step["logits"],
        )

    # Live K/V only, and only where both sides reached the same position.
    # Comparing beyond `position` would compare uninitialised device memory.
    mine_snapshot = mine["final_snapshot"]
    reference_snapshot = theirs["final_snapshot"]
    failures.check(
        mine_snapshot["position"] == reference_snapshot["position"],
        f"final position {mine_snapshot['position']} vs "
        f"{reference_snapshot['position']}",
    )
    for name in ("layer0_k", "layer0_v", "layer31_k", "layer31_v"):
        left = _widen_bf16(mine_snapshot[name])
        right = _widen_bf16(reference_snapshot[name])
        if left.shape != right.shape:
            failures.check(
                False,
                f"{name}: live extent {left.shape} vs {right.shape}",
            )
            continue
        if left.size == 0:
            failures.check(False, f"{name}: no live rows to compare")
            continue
        correlation = _correlation(left, right)
        failures.check(
            correlation >= MIN_CORRELATION,
            f"{name}: live-cache correlation {correlation:.8f} < "
            f"{MIN_CORRELATION}",
        )

    # Deliberately NOT compared: synchronize counts. FastFlow uses four
    # synchronizes per layer by design and the reference still uses two, so
    # equality there would mean FastFlow had regressed to a schedule design
    # Section 10.4 rejected. The 129-per-step count is asserted inside the
    # C++ harness against FastFlow's own contract instead.
    if "final_metrics" in mine:
        print(
            "fastflow synchronize_count="
            f"{mine['final_metrics']['synchronize_count']} "
            "(not compared against the reference: different schedules)"
        )

    return failures.report(f"compare[{mine['continuation_route']}]")


def self_consistency(args) -> int:
    """Two runs of the same binary on the same input must agree bit for bit.

    This is a stronger and much cheaper signal than the comparison against the
    reference, and it is not the same question. A systematic numeric difference
    from the reference is a question about tolerances and about which
    implementation is right. A difference between two runs of the SAME
    implementation on the SAME input is neither: it means something is reading
    state that the inputs do not determine, and no tolerance makes that
    acceptable. Measured on the AIE4 target, FastFlow fails this while the
    corelib reference driver passes it.

    Bit-exact, deliberately. There is no tolerance under which "the same
    program, twice, on the same data" is allowed to disagree.
    """
    left = json.loads(Path(args.a).read_text(encoding="utf-8"))
    right = json.loads(Path(args.b).read_text(encoding="utf-8"))
    failures = Failures()

    def logit_steps(document):
        steps = [("continuation", document["continuation"]["logits_bf16"])]
        for index, step in enumerate(document["decode"]):
            steps.append((f"decode[{index}]", step["logits_bf16"]))
        return steps

    a_steps = logit_steps(left)
    b_steps = logit_steps(right)
    failures.check(
        len(a_steps) == len(b_steps),
        f"step counts differ: {len(a_steps)} vs {len(b_steps)}",
    )

    first_divergence = None
    for (label, a_bits), (_, b_bits) in zip(a_steps, b_steps):
        if a_bits != b_bits:
            differing = sum(1 for x, y in zip(a_bits, b_bits) if x != y)
            mine = _widen_bf16(a_bits)
            theirs = _widen_bf16(b_bits)
            failures.check(
                False,
                f"{label}: two runs of the same binary on the same input "
                f"disagree on {differing}/{len(a_bits)} logits, "
                f"max |diff| {float(np.max(np.abs(mine - theirs))):.6g}",
            )
            if first_divergence is None:
                first_divergence = label

    a_tokens = [left["continuation"]["top1_id"]] + [
        step["top1_id"] for step in left["decode"]
    ]
    b_tokens = [right["continuation"]["top1_id"]] + [
        step["top1_id"] for step in right["decode"]
    ]
    failures.check(
        a_tokens == b_tokens,
        f"emitted token sequences differ:\n      {a_tokens}\n      {b_tokens}",
    )
    if first_divergence is not None:
        print(f"first divergence at {first_divergence}")

    return failures.report(
        f"self-consistency[{left['continuation_route']}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    emit = subparsers.add_parser("emit-reference")
    emit.add_argument("--model-dir", required=True)
    emit.add_argument("--token-ids-json", required=True)
    emit.add_argument("--decode-steps", type=int, default=MIN_DECODE_STEPS)
    emit.add_argument("--continuation-route", required=True)
    emit.add_argument("--output-json", required=True)
    emit.set_defaults(handler=emit_reference)

    check = subparsers.add_parser("compare")
    check.add_argument("--fastflow-json", required=True)
    check.add_argument("--reference-json", required=True)
    check.set_defaults(handler=compare)

    repeat = subparsers.add_parser("self-consistency")
    repeat.add_argument("--a", required=True)
    repeat.add_argument("--b", required=True)
    repeat.set_defaults(handler=self_consistency)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
