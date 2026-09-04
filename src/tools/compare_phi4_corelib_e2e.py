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
# report can quote the number that actually ran, and they are SEALED below --
# see `_DETERM2_SEALED`. Do not change either number here alone.
MIN_CORRELATION = 0.9999
MIN_DECODE_STEPS = 16
TOP_K = 32
TOP_5 = 5

# DETERM-2. Two bounds, deliberately in DIFFERENT UNITS.
#
# CROSS-IMPLEMENTATION (design 12.4): absolute 0.25 over the union top-32.
# FastFlow and the reference driver can differ for reasons other than
# accumulation order, so an absolute ceiling is the right shape there.
#
# RUN-TO-RUN (design 15.3, DETERM-1): 2 BF16 ULP of the larger of the two
# compared values. The phenomenon is ULP-relative, not absolute. The one
# observed event sat at exactly 0.25, which is 2 ULP for logits in [16, 32) --
# an absolute 0.25 bound would therefore have hard-failed the identical 2-ULP
# wobble on a logit in [32, 64), where 2 ULP is 0.5, for no reason but
# magnitude. This corrects the UNITS; it does not widen the tolerance.
#
# Both are OBSERVED CEILINGS WITH NO MARGIN, not tolerances. A run that exceeds
# either has done something not previously seen, and the answer is to measure
# and amend the spec -- never to widen the number so a red run goes green.
#
# TASK 13 MEASURED IT, AND THE 2-ULP FIGURE IS NOT AN UPPER BOUND. Two events
# exceeded it during that baseline: a reprefill pair differing by up to 0.3125
# on a logit of 11.125 -- about 2.5 ULP, with 17512 logits over the bound --
# and an append pair differing by up to 48.34 after the two runs emitted
# DIFFERENT TOKENS from decode step 7. So this constant no longer describes a
# ceiling. It describes the largest run-to-run difference that has ever been
# benign.
#
# THAT IS DELIBERATELY LEFT AS IS. DETERM-2 says a logit difference above the
# bound is a failure and not a wider tolerance, and both events were caught by
# this gate doing exactly its job. Raising the number to accommodate them is
# the move DETERM-2 exists to forbid, and it would convert a measured device
# property into an unbounded one. The right response is the design decision
# Task 13 report asks for, not an edit here.
MAX_TOP32_ABS_DIFF = 0.25
RUN_TO_RUN_MAX_ULPS = 2

# The seal, and it covers EVERY name a failure message can hand you.
#
# The previous version sealed one private constant and then exposed unsealed
# public aliases -- including the very name its own error message advertised.
# Grepping the name from the message landed on an alias that could be edited
# freely, and the tool started normally and went green. The bump was off to the
# side of the road.
#
# So: every constant below is cross-checked against a literal that exists for
# no other purpose, and the runtime failure messages point at this block by
# name so that grepping what the message says leads HERE.
#
# IT NOW COVERS THE DESIGN 12.4 ACCEPTANCE THRESHOLDS TOO, and not because
# 12.4 asked for it. `MIN_CORRELATION` and `MIN_DECODE_STEPS` sat outside the
# seal while their values were printed verbatim in four failure messages --
# the exact grep-the-message-and-edit path the seal exists to block, forty
# lines above the constants it did block. Sealing the instance and leaving the
# class open is the mistake this file has already paid for once, in the alias
# incident described above; two lines close the class.
#
# Each entry carries the reason its own number cannot be moved, because the
# reasons are NOT the same -- two are measured ceilings and two are the size
# of the experiment -- and one message that fitted both would be true of
# neither.
_DETERM2_SEALED = {
    "MAX_TOP32_ABS_DIFF": (
        0.25,
        "the largest cross-implementation top-32 logit difference ever "
        "measured on this hardware, recorded with no margin (design 12.4)",
    ),
    "RUN_TO_RUN_MAX_ULPS": (
        2,
        "in BF16 ULP, the largest run-to-run difference that has ever been "
        "benign; Task 13 measured two events above it and they are defects, "
        "not headroom (design 15.3, DETERM-1)",
    ),
    "MIN_CORRELATION": (
        0.9999,
        "the correlation FastFlow must reach against the corelib reference "
        "before the two are called the same computation; lowering it does "
        "not make them agree, it stops the comparison asking (design 12.4)",
    ),
    "MIN_DECODE_STEPS": (
        16,
        "the shortest decode run design 12.4 accepts as evidence; fewer "
        "steps is a smaller experiment, not a passing one",
    ),
    # TOP_K was the third one left outside, and it is not a threshold: it is
    # the SIZE OF THE WINDOW the MAX_TOP32_ABS_DIFF threshold is measured
    # over. Shrinking it trips no bound, it just stops the comparison looking
    # at the logits where the two implementations disagree -- so it is the
    # cheapest way to make a red run green, and its value is interpolated
    # verbatim into the very message that reports the failure.
    "TOP_K": (
        32,
        "the width of the union top-k window MAX_TOP32_ABS_DIFF is measured "
        "over; design 12.4's '0.25 over the union top-32' is one claim, not "
        "two independent numbers, so narrowing k narrows the experiment while "
        "every bound still reads as met",
    ),
}
for _sealed_name, (_sealed_value, _sealed_reason) in _DETERM2_SEALED.items():
    _actual = globals()[_sealed_name]
    if _actual != _sealed_value:
        raise SystemExit(
            f"DETERM-2: {_sealed_name} has been changed from its sealed "
            f"value {_sealed_value} to {_actual}.\n"
            "\n"
            f"{_sealed_name} is {_sealed_reason}. Moving it does not make a "
            "failing run acceptable; it makes the suite stop reporting a "
            "change in behaviour that nobody has looked at.\n"
            "\n"
            "If a run genuinely misses one of these: measure it, characterise "
            "it the way report section 14.2 characterises the run-to-run "
            "case, and amend the design section the entry names. Only then "
            "update the constant AND its entry in _DETERM2_SEALED together."
        )


def _two_bf16_ulp(magnitude: np.ndarray) -> np.ndarray:
    """`RUN_TO_RUN_MAX_ULPS` ULP at each magnitude, exactly.

    BF16 keeps 8 significand bits, so ULP at a value with exponent e is
    2**(e - 7). The exponent is read out of the FP32 bit pattern rather than
    via log2: these values were widened from BF16 so the field is exact, and
    log2 of a value a hair under a power of two is exactly the case that would
    round the wrong way and silently shift the bound by a factor of two.

    Zero gets a bound of zero -- two runs that both produced zero are already
    equal, and anything else there is a real difference.
    """
    magnitude = np.asarray(magnitude, dtype=np.float32)
    bits = np.ascontiguousarray(magnitude).view(np.uint32)
    exponent = ((bits >> 23) & 0xFF).astype(np.int32) - 127
    # N * 2**(e - 7), written out rather than folded into the exponent: the
    # folded form is only correct while N is a power of two, and a later change
    # to RUN_TO_RUN_MAX_ULPS would silently compute the wrong bound.
    ulp = np.exp2((exponent - 7).astype(np.float64))
    return np.where(magnitude == 0.0, 0.0, RUN_TO_RUN_MAX_ULPS * ulp)


# The ULP bound is checked at import, at the boundaries that matter.
#
# The whole correction in DETERM-2 is that the bound scales with magnitude, so
# a bound that silently failed to step at a power of two would reintroduce the
# absolute-ceiling bug while looking relative. 20 and 31.5 must give 0.25 (the
# observed event), 32 and 40 must give 0.5, and the step must land exactly on
# the binade edge -- which is why the exponent comes from the bit pattern and
# not from log2.
_ULP_SELF_CHECK = ((16.0, 0.25), (20.0, 0.25), (31.5, 0.25),
                   (32.0, 0.5), (40.0, 0.5), (64.0, 1.0), (0.0, 0.0))
if RUN_TO_RUN_MAX_ULPS == 2:
    _probe = _two_bf16_ulp(
        np.array([v for v, _ in _ULP_SELF_CHECK], dtype=np.float32))
    for _index, (_value, _expected) in enumerate(_ULP_SELF_CHECK):
        if float(_probe[_index]) != _expected:
            raise SystemExit(
                f"DETERM-2: the ULP bound is miscomputed. 2 ULP of {_value} "
                f"should be {_expected}, got {float(_probe[_index])}. The "
                "run-to-run gate would be applying the wrong tolerance."
            )


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
        # The artifact records the route as ContinuationRouteName() spells it
        # ("append"/"reprefill"); the suite and the golden file use the forced
        # spelling ("force_append"/"force_reprefill"). Accept either, so a
        # naming difference cannot be mistaken for a missing golden.
        route_key = mine["continuation_route"]
        if route_key not in expected_document:
            alternate = (
                route_key[len("force_") :]
                if route_key.startswith("force_")
                else "force_" + route_key
            )
            if alternate in expected_document:
                route_key = alternate
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

    # MODEL STATE must be bit-identical; the LM head's output need not be.
    #
    # This split is measured, not a convenience. Across every CROSS-
    # IMPLEMENTATION run recorded to date the live K/V for layers 0 and 31 and
    # the final hidden state have been bit-identical to the reference -- the
    # whole 32-layer computation agreeing exactly -- while some runs had a
    # single logit vector out of 17 differ, with every design 12.4 threshold
    # still met and the same top-1.
    #
    # THE RATE HERE READ "one run in three" AND WAS ALREADY STALE WHEN WRITTEN.
    # Task 13 counted the surviving artifacts: the append route differed from
    # the reference on 2 of the 4 comparisons whose compare-summary records
    # exist, the reprefill route on 0 of 4. Stated with its n rather than as a
    # fraction, because n=4 does not support a rate anyone should rely on. It
    # is enough to say the effect is route-dependent and real. Task 12 reported
    # two further append divergences whose artifacts were overwritten before
    # run-scoped directories existed; they are not counted here.
    #
    # NOTE, and it matters: this comment is about the CROSS-IMPLEMENTATION
    # comparison, where the model state HAS always matched. It is not evidence
    # about the RUN-TO-RUN case, where Task 13 measured model-state divergence
    # at layer 31 with layer 0 identical. See the self_consistency docstring.
    #
    # So state bit-identity is enforced: a regression there would mean the
    # model computed something different. Logit bit-identity is reported and
    # not enforced, because enforcing a property that is not reliably true
    # produces a red suite for something that is not a defect -- and a gate
    # nobody trusts is worse than a number everybody reads.
    failures.check(
        kv_exact,
        "live K/V is NOT bit-identical to the reference. The 32-layer "
        "computation itself has diverged, which is a different and more "
        "serious thing than a logit difference within tolerance.",
    )
    failures.check(
        hidden_exact,
        "the final hidden state is NOT bit-identical to the reference "
        "(design 15.3 checkpoint)",
    )
    if args.require_bit_exact:
        failures.check(
            all_exact,
            "--require-bit-exact was given and the LOGITS are not "
            "bit-identical. State and last_hidden are checked "
            "unconditionally; this flag additionally demands the LM-head "
            "output match, which is a diagnostic rather than a release gate.",
        )

    # A reported property has to outlive stdout.
    #
    # These counts were prints and nothing else: absent from every artifact and
    # from the suite summary, and never aggregated, so a rising rate of
    # LM-head divergence would have been invisible -- the suite's last line
    # reads the same at 17/17 and at 16/17. Writing them where the runner can
    # pick them up is what makes "reported rather than gated" an actual
    # position rather than a way of not looking.
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(
                {
                    "route": mine["continuation_route"],
                    "corelib_sha256": mine_sha,
                    "logits_bit_exact_steps": sum(
                        1 for value in exact["logits"] if value
                    ),
                    "logits_total_steps": len(exact["logits"]),
                    "kv_bit_exact_tensors": sum(
                        1 for value in exact["kv"] if value
                    ),
                    "kv_total_tensors": len(exact["kv"]),
                    "last_hidden_bit_exact": hidden_exact,
                    "all_bit_exact": all_exact,
                    "failures": failures.messages,
                },
                indent=2,
            ),
            encoding="utf-8",
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
    """Two runs of the same binary on the same input, per design 15.3.

    DETERM-1. The product does NOT claim run-to-run logit bit-identity, and
    this check is written to that claim rather than to a stronger one nobody
    can honour. What must hold exactly:

      * live K/V, last_hidden and every non-timing metric, bit-identical;
      * the emitted token-ID sequence, identical;
      * every logit within RUN_TO_RUN_MAX_ULPS BF16 ULP of the larger of
        the two compared values -- a RELATIVE bound, because the
        phenomenon is.

    What is recorded but not gated: the logit bit-identity rate and the
    observed maximum difference.

    The reason for the split is measured, not conceded: run-to-run logit
    bit-identity is not reliably true on this hardware, so gating on it would
    make the suite intermittently red for a property the product does not
    promise.

    CORRECTED BY TASK 13 BASELINE MEASUREMENT, 2026-09-02. This docstring used
    to say the non-determinism was "confined to" the 3072 x 200064 LM-head
    dispatch and "changes no decision", on the strength of ONE observed event:
    100389 of 200064 logits differing at decode[13] by at most 0.25, with
    state, metrics and tokens identical. Two of those three claims are now
    known false.

    IT IS NOT CONFINED TO THE LM HEAD. A reprefill event measured on
    2026-09-02 left layer 0 K and V bit-identical while layer 31 K and V and
    last_hidden all differed, with the emitted tokens unchanged. The LM head
    cannot write a layer-31 K cache, so the divergence entered the model body
    at some layer above 0. Separately, an FP64 host reference computed from the
    same ONNX components corelib packs from found the LM head to be a correctly
    rounded function of its own input in every run measured: 200059 and 200060
    of 200064 logits within half a BF16 ULP of truth, mean 0.2497 ULP, signs
    balanced to four parts in 200064. A correctly rounded function of identical
    input cannot produce different output, so the earlier localisation -- an
    INFERENCE from end-of-run state identity, never an observation of the
    LM-head input at the diverging step -- does not hold.

    IT CAN CHANGE A DECISION. An append event in the same campaign produced
    DIFFERENT EMITTED TOKEN SEQUENCES from two runs of the same binary on the
    same input, diverging at decode step 7.

    What survives: both runs provably load the same DLL, and the phenomenon is
    intermittent. For the rate, the spread and the n they rest on, see the
    DETERM-3 baseline in docs/docs/benchmarks/phi4_results.md.

    The gates below are unchanged and they are what caught both events. Do not
    weaken them.
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
        # I-7. The corelib DLL was pinned and the FastFlow binary that drove
        # it was not, so two records could describe different FastFlow builds
        # with nothing able to tell. A determinism record pooled across a
        # build tree needs both halves of "the same binary twice".
        ("harness_sha256", "harness SHA-256"),
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

    # THE LOCALISATION, MEASURED. Same step index as logit_steps, so the two
    # lists line up.
    def lm_head_inputs(document):
        rows = [document["continuation"].get("lm_head_input_bf16")]
        for step in document["decode"]:
            rows.append(step.get("lm_head_input_bf16"))
        return rows

    a_inputs = lm_head_inputs(left)
    b_inputs = lm_head_inputs(right)

    first_divergence = None
    first_divergence_index = None
    bit_exact_steps = 0
    observed_max_diff = 0.0
    for index, ((label, a_bits), (_, b_bits)) in enumerate(
        zip(a_steps, b_steps)
    ):
        if a_bits == b_bits:
            bit_exact_steps += 1
            continue
        differing = sum(1 for x, y in zip(a_bits, b_bits) if x != y)
        mine = _widen_bf16(a_bits)
        theirs = _widen_bf16(b_bits)
        if mine.shape != theirs.shape:
            failures.check(
                False,
                f"{label}: logit vector lengths differ, "
                f"{mine.shape} vs {theirs.shape}",
            )
            continue
        difference = np.abs(mine - theirs)
        step_max = float(np.max(difference))
        observed_max_diff = max(observed_max_diff, step_max)
        if first_divergence is None:
            first_divergence = label
            first_divergence_index = index

        # GATED in ULP, RECORDED for the fact of differing.
        #
        # DETERM-1 does not promise bit-identity here, but it does bound the
        # size of the difference -- and the bound is relative, because the
        # phenomenon is. Comparing each element against 2 ULP of the larger of
        # the two values treats a 2-ULP wobble the same whether the logit is 20
        # or 40; an absolute ceiling would have failed the second for nothing
        # but magnitude.
        allowed = _two_bf16_ulp(np.maximum(np.abs(mine), np.abs(theirs)))
        # NaN compares false against everything, so a non-finite result on
        # either side lands here rather than slipping through the <= .
        over = ~(difference <= allowed)
        over_count = int(np.count_nonzero(over))
        if over_count:
            worst = int(np.argmax(np.where(over, difference, 0.0)))
            failures.check(
                False,
                f"{label}: {over_count}/{len(a_bits)} logits differ by more "
                f"than {RUN_TO_RUN_MAX_ULPS} BF16 ULP between two runs of the "
                f"same binary. Worst at index {worst}: {mine[worst]:.9g} vs "
                f"{theirs[worst]:.9g}, difference {difference[worst]:.6g}, "
                f"allowed {allowed[worst]:.6g}. Do NOT widen the bound -- see "
                f"_DETERM2_SEALED in this file; measure what changed.",
            )
        print(
            f"  {label}: {differing}/{len(a_bits)} logits differ, "
            f"max |diff| {step_max:.6g}, "
            f"{'ALL' if not over_count else f'{len(a_bits) - over_count}/{len(a_bits)}'} "
            f"within {RUN_TO_RUN_MAX_ULPS} ULP"
        )

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

    # DETERM-1 requires the rate and the observed maximum to be recorded every
    # run, in the artifact and in the suite summary -- not merely printed. The
    # gate above answers "is this within what we have measured"; this record is
    # what makes "is the rate degrading from baseline" answerable at all, and
    # that question cannot be asked of a number that only ever existed on
    # stdout.
    print(
        f"run-to-run: logits bit-identical "
        f"{bit_exact_steps}/{len(a_steps)} steps, "
        f"max |diff| {observed_max_diff:.6g} "
        f"(bound {RUN_TO_RUN_MAX_ULPS} BF16 ULP, relative)"
    )
    if first_divergence is not None:
        print(f"first divergence at {first_divergence}")
    # WHERE THE DIVERGENCE ENTERED, asked of the recorded LM-head input at the
    # step that actually diverged rather than inferred from end-of-run state.
    #
    # DETERM-1's argument was "identical LM-head input with non-identical
    # LM-head output localises this to the 3072 x 200064 dispatch". The input
    # half of that was never observed. This computes it: if the two runs fed
    # the LM head the same 3072-element row and still produced different
    # logits, the LM head is the source; if the rows differ, the divergence
    # entered the model body and the LM head is faithfully transforming
    # different inputs.
    localisation = {
        "measured": False,
        "reason": "the runs recorded no per-step LM-head input",
    }
    if first_divergence_index is not None:
        a_row = a_inputs[first_divergence_index]
        b_row = b_inputs[first_divergence_index]
        if a_row is None or b_row is None:
            localisation["reason"] = (
                "a run predates the per-step LM-head input capture; rebuild "
                "the end-to-end harness with DEV_BUILD and re-run"
            )
        elif len(a_row) != len(b_row):
            localisation["reason"] = (
                f"LM-head input lengths differ, {len(a_row)} vs {len(b_row)}"
            )
        else:
            differing = sum(1 for x, y in zip(a_row, b_row) if x != y)
            localisation = {
                "measured": True,
                "step": first_divergence,
                "lm_head_input_elements": len(a_row),
                "lm_head_input_differing_elements": differing,
                "source": "lm_head" if differing == 0 else "model_body",
                # THE INSTRUMENT, recorded with the result.
                #
                # Capturing the LM-head input after every step adds a host
                # tensor read and a stream acquisition between model steps,
                # which changes the timing of exactly the window a race would
                # live in. That is a caveat on every localisation measured
                # this way, and a reader of a downstream document should not
                # have to find it in a task report. It travels with the
                # record so the document can state it from the data.
                "measured_by": "per_step_lm_head_input_capture",
                "instrumentation_effect": (
                    "the capture adds a host tensor read and a stream "
                    "acquisition between every model step, which perturbs "
                    "the timing of the window a race would occupy"
                ),
                "reason": (
                    "the two runs fed the LM head an identical row and it "
                    "produced different logits"
                    if differing == 0
                    else (
                        "the two runs fed the LM head DIFFERENT rows, so the "
                        "divergence entered before the LM head; DETERM-1's "
                        "localisation to the 3072 x 200064 dispatch does not "
                        "hold for this event"
                    )
                ),
            }
        print(
            "localisation at "
            f"{first_divergence}: {localisation.get('source', 'unmeasured')}"
            f" -- {localisation['reason']}"
        )
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(
                {
                    "route": left.get("continuation_route"),
                    "corelib_sha256": left.get("corelib_sha256"),
                    "harness_sha256": left.get("harness_sha256"),
                    "a": str(args.a),
                    "b": str(args.b),
                    "logits_bit_exact_steps": bit_exact_steps,
                    "logits_total_steps": len(a_steps),
                    "observed_max_abs_diff": observed_max_diff,
                    "determ2_bound_ulps": RUN_TO_RUN_MAX_ULPS,
                    "determ2_bound_kind": "relative_bf16_ulp",
                    "first_divergence": first_divergence,
                    "localisation": localisation,
                    "failures": failures.messages,
                },
                indent=2,
            ),
            encoding="utf-8",
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
        "--summary-json",
        help="write the bit-exactness counts here, so the reported (ungated) "
        "half of the comparison survives the run and can be aggregated",
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
    repeat.add_argument(
        "--summary-json",
        help="write the DETERM-1 record here: the run-to-run logit "
        "bit-identity rate and the observed maximum difference",
    )
    repeat.set_defaults(handler=self_consistency)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
