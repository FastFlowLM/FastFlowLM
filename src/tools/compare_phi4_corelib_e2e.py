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


def _narrow_bf16(values) -> list:
    """FP32 -> raw BF16 bits, round-to-nearest-even.

    Mirrors `phi4_driver.to_bf16`, reimplemented here so `compare` stays a pure
    JSON-to-JSON operation that imports neither the driver nor the corelib
    bindings and therefore cannot open a device context.
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits.astype(np.uint64) + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16).tolist()


def _sha256_file(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    # Which DLL the PYTHON side resolved.
    #
    # The C++ harness and this script find the library by different fallbacks,
    # so "both used the same corelib" was an assumption rather than a recorded
    # fact. Recording it on both sides makes the comparison state which two
    # binaries it actually compared.
    library = corelib.load_library()
    library_path = str(getattr(library, "_dll", None)._name)
    record = {
        "continuation_route": args.continuation_route,
        "prefix_ids": prefix,
        "suffix_ids": suffix,
        "corelib_loaded_path": library_path,
        "corelib_sha256": _sha256_file(library_path),
    }
    print(f"reference corelib: {library_path}")
    print(f"sha256           : {record['corelib_sha256']}")
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
            # Design 15.3 lists the final hidden state as a checkpoint. It is
            # narrowed to BF16 here because that is what the device stores:
            # FastFlow reads `lm_input_tensor` back as raw BF16, and the
            # reference's FP32 array is the value corelib narrowed on the way
            # in. Comparing the FP32 against the BF16 would be comparing two
            # different things and would need a tolerance to hide it.
            "last_hidden": _narrow_bf16(hidden),
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

    # Which two binaries are actually being compared. Reported always; if both
    # sides recorded a hash they must agree, because a comparison between two
    # different corelib builds is not the comparison anyone thinks it is.
    mine_sha = mine.get("corelib_sha256")
    theirs_sha = theirs.get("corelib_sha256")
    if mine_sha and theirs_sha:
        print(f"fastflow  corelib {mine_sha}  {mine.get('corelib_loaded_path')}")
        print(f"reference corelib {theirs_sha}  {theirs.get('corelib_loaded_path')}")
        failures.check(
            mine_sha == theirs_sha,
            "the two sides loaded DIFFERENT corelib builds:\n"
            f"      fastflow  {mine_sha} {mine.get('corelib_loaded_path')}\n"
            f"      reference {theirs_sha} {theirs.get('corelib_loaded_path')}",
        )
    else:
        failures.check(
            False,
            "one side recorded no corelib SHA-256, so this comparison cannot "
            "say which two binaries it compared",
        )

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

    # Bit-exactness is tracked as its own property, separate from the design
    # 12.4 thresholds.
    #
    # The thresholds are the release gate. Bit-exactness is the claim the
    # report makes, and until now no check produced it -- it was computed by
    # hand from the artifacts and asserted in prose, which is exactly the kind
    # of claim that quietly stops being true. `--require-bit-exact` makes it
    # fail the run instead.
    exact = {"logits": [], "kv": [], "last_hidden": None}

    _compare_step(
        failures,
        "continuation",
        mine["continuation"]["logits_bf16"],
        theirs["continuation"]["logits"],
    )
    exact["logits"].append(
        mine["continuation"]["logits_bf16"]
        == _narrow_bf16(theirs["continuation"]["logits"])
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
        exact["logits"].append(
            mine_step["logits_bf16"] == _narrow_bf16(reference_step["logits"])
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
            exact["kv"].append(False)
            continue
        if left.size == 0:
            failures.check(False, f"{name}: no live rows to compare")
            exact["kv"].append(False)
            continue
        correlation = _correlation(left, right)
        failures.check(
            correlation >= MIN_CORRELATION,
            f"{name}: live-cache correlation {correlation:.8f} < "
            f"{MIN_CORRELATION}",
        )
        identical = int(np.sum(left == right))
        exact["kv"].append(identical == left.size)
        print(
            f"{name}: correlation {correlation:.8f}, "
            f"{identical}/{left.size} elements identical"
        )

    # Design 15.3 lists the final hidden state as a checkpoint. It was emitted
    # by the harness from the first version of this suite and never compared,
    # which made it a payload rather than a check.
    if "last_hidden" in reference_snapshot:
        mine_hidden = mine_snapshot["last_hidden"]
        reference_hidden = reference_snapshot["last_hidden"]
        if len(mine_hidden) != len(reference_hidden):
            failures.check(
                False,
                f"last_hidden: length {len(mine_hidden)} vs "
                f"{len(reference_hidden)}",
            )
            exact["last_hidden"] = False
        else:
            widened_mine = _widen_bf16(mine_hidden)
            widened_reference = _widen_bf16(reference_hidden)
            correlation = _correlation(widened_mine, widened_reference)
            failures.check(
                correlation >= MIN_CORRELATION,
                f"last_hidden: correlation {correlation:.8f} < "
                f"{MIN_CORRELATION}",
            )
            identical = int(np.sum(widened_mine == widened_reference))
            exact["last_hidden"] = identical == len(mine_hidden)
            print(
                f"last_hidden: correlation {correlation:.8f}, "
                f"{identical}/{len(mine_hidden)} elements identical"
            )
    else:
        failures.check(
            False,
            "the reference emitted no last_hidden, so the design 15.3 final "
            "hidden checkpoint was not compared",
        )

    # The route's expected token sequence, from a file committed to the
    # repository. Without this the golden is re-derived from the reference on
    # every run, so the pair could drift together -- a corelib change that
    # moved both sides identically would pass every check above.
    if args.expected_tokens:
        expected_document = json.loads(
            Path(args.expected_tokens).read_text(encoding="utf-8")
        )
        route_key = mine["continuation_route"]
        if route_key not in expected_document:
            failures.check(
                False,
                f"no expected token sequence recorded for route "
                f"'{route_key}' in {args.expected_tokens}",
            )
        else:
            expected = expected_document[route_key]
            observed = [mine["continuation"]["top1_id"]] + [
                step["top1_id"] for step in mine["decode"]
            ]
            failures.check(
                observed == expected,
                f"route '{route_key}' emitted a different token sequence "
                f"from the one recorded in {args.expected_tokens}:\n"
                f"      expected {expected}\n"
                f"      observed {observed}",
            )

    logits_exact = all(exact["logits"])
    kv_exact = bool(exact["kv"]) and all(exact["kv"])
    hidden_exact = exact["last_hidden"] is True
    all_exact = logits_exact and kv_exact and hidden_exact
    print(
        "bit-exact vs the reference: logits "
        f"{sum(1 for v in exact['logits'] if v)}/{len(exact['logits'])} steps, "
        f"K/V {sum(1 for v in exact['kv'] if v)}/{len(exact['kv'])} tensors, "
        f"last_hidden {hidden_exact}"
    )
    if args.require_bit_exact:
        failures.check(
            all_exact,
            "--require-bit-exact was given and the result is not "
            "bit-identical to the reference. Any report describing it as "
            "bit-identical is now wrong.",
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

    # Same library, first. Without this the check cannot tell "the same binary
    # twice" from "two runs that loaded different DLLs", and the second is a
    # live hypothesis for the divergence in report section 5.1 -- so a pass
    # here would have been evidence for a claim it never tested.
    for field, label in (
        ("corelib_sha256", "corelib SHA-256"),
        ("corelib_loaded_path", "loaded corelib path"),
    ):
        a_value = left.get(field)
        b_value = right.get(field)
        failures.check(
            a_value is not None and b_value is not None,
            f"a run recorded no {label}, so this check cannot show the two "
            f"runs used the same library",
        )
        if a_value is not None and b_value is not None:
            failures.check(
                a_value == b_value,
                f"the two runs used a different {label}:\n"
                f"      A {a_value}\n      B {b_value}",
            )
    if left.get("corelib_sha256"):
        print(f"both runs loaded {left['corelib_sha256']}")

    failures.check(
        left.get("prefix_ids") == right.get("prefix_ids")
        and left.get("suffix_ids") == right.get("suffix_ids")
        and left.get("continuation_route") == right.get("continuation_route"),
        "the two runs were not given the same input",
    )

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

    # The snapshot, not just the logits. The logits are one row out of the last
    # LM-head dispatch; the K/V caches are the accumulated state of all 32
    # layers over every step, so they are where an intermittent divergence
    # shows up earliest and most visibly.
    a_snapshot = left.get("final_snapshot", {})
    b_snapshot = right.get("final_snapshot", {})
    for name in (
        "position",
        "live_rows",
        "last_hidden",
        "layer0_k",
        "layer0_v",
        "layer31_k",
        "layer31_v",
    ):
        a_value = a_snapshot.get(name)
        b_value = b_snapshot.get(name)
        if a_value is None or b_value is None:
            failures.check(False, f"snapshot field {name} is missing")
            continue
        if a_value != b_value:
            if isinstance(a_value, list) and len(a_value) == len(b_value):
                differing = sum(1 for x, y in zip(a_value, b_value) if x != y)
                failures.check(
                    False,
                    f"snapshot {name}: {differing}/{len(a_value)} elements "
                    f"differ between two runs of the same binary",
                )
            else:
                failures.check(
                    False,
                    f"snapshot {name}: {a_value!r} vs {b_value!r}",
                )

    # The counts too. These are the design 5.2 / 10.4 contract, and two runs
    # that dispatched a different number of times are not the same run even if
    # the numbers happened to land in the same place.
    a_metrics = left.get("final_metrics", {})
    b_metrics = right.get("final_metrics", {})
    failures.check(
        bool(a_metrics) and bool(b_metrics),
        "a run recorded no final_metrics",
    )
    for key in sorted(set(a_metrics) | set(b_metrics)):
        # Wall-clock fields are not reproducible and are not part of the
        # contract; everything else is a count or an extent and must match.
        if key.endswith("_ns"):
            continue
        failures.check(
            a_metrics.get(key) == b_metrics.get(key),
            f"metric {key}: {a_metrics.get(key)} vs {b_metrics.get(key)}",
        )

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
    check.add_argument(
        "--expected-tokens",
        help="JSON file of route-keyed expected top-1 token sequences, "
        "committed to the repository so the golden is not re-derived from "
        "the reference on every run",
    )
    check.add_argument(
        "--require-bit-exact",
        action="store_true",
        help="fail unless every logit vector, every live K/V tensor and the "
        "final hidden state are bit-identical to the reference",
    )
    check.set_defaults(handler=compare)

    repeat = subparsers.add_parser("self-consistency")
    repeat.add_argument("--a", required=True)
    repeat.add_argument("--b", required=True)
    repeat.set_defaults(handler=self_consistency)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
