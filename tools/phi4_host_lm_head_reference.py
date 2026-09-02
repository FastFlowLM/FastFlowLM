"""Host LM-head reference for the `DETERM-1` diagnostic. Task 13 Step 9b.

THIS IS A DIAGNOSTIC, NOT A PRODUCT PATH. Nothing here is imported by
`phi4_corelib_aie4`, no host island is added, and no production decode is
routed through it. Design section 10.5 keeps the LM head on AIE4; moving it to
the host would cost 614M MACs per token and roughly 1.2 GB of dequantised
weights to fix something that changes no emitted token. Human decision
2026-09-02: build the reference, keep the product as it is.

WHAT IT ANSWERS. `DETERM-1` records that the AIE4 LM-head MatMul is not
bit-deterministic across runs, and it localises that **by inference**:
identical LM-head input, identical model state, identical emitted tokens,
non-identical LM-head output, therefore the difference is inside the
`3072 x 200064` dispatch. That inference is sound but it compares two NPU runs
against each other, with no third point. This script supplies the third point
-- ground truth -- computed in FP64 from the SAME ONNX components corelib packs
from: `lm_head.MatMulNBits` qweight, scales and qzeros, at the byte offsets the
package manifest records, dequantised.

The question worth answering is not "do they differ" but which of three:

  1. both NPU runs sit within rounding distance of the true value, straddling
     it -- the wobble is benign accumulation-order nondeterminism;
  2. one run is systematically further from truth than the other -- something
     is wrong beyond rounding; or
  3. both are offset from truth in the same direction -- a bias, not noise.

(2) or (3) is a DIFFERENT finding from the one `DETERM-1` accepts and would
reopen the product decision, so the classification is emitted explicitly and
the exit code carries it.

THE UNIT THAT MATTERS IS THE ULP, NOT THE ABSOLUTE. The NPU logits are BF16,
so the very best any implementation can do is the correctly rounded BF16
neighbour of the true value -- half a BF16 ULP away, which at a logit of 20 is
0.0625 and at a logit of 40 is 0.125. Judging in absolute units would call the
larger logit worse for nothing but magnitude, which is the units error design
section 15.3 already had to correct once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ONNX MatMulNBits with 4-bit weights and a 128-element group.
BITS = 4
GROUP_SIZE = 128
HIDDEN = 3072
VOCAB = 200064
LM_HEAD_OBJECT = "lm_head.MatMulNBits"

# Rows of the [200064, 3072] weight matrix dequantised at a time. 8192 rows is
# 100 MB in FP32 and 200 MB promoted to FP64; the full matrix would be 4.9 GB
# in FP64, which is the reason this is chunked at all.
CHUNK_ROWS = 8192

# Classification thresholds, in BF16 ULP of the value being compared, stated
# here rather than buried in the code that applies them.
#
# A correctly rounded BF16 result is at most 0.5 ULP from truth. A mean signed
# deviation is a BIAS if it is a substantial fraction of that: an unbiased
# rounding error averages to roughly zero over 200,064 values, so anything
# above a quarter ULP of consistent offset is not rounding.
BIAS_MEAN_SIGNED_ULP = 0.25
# One run is "systematically further" if its mean absolute deviation exceeds
# the other's by more than a quarter of the larger. Two runs of the same
# kernel differing only in accumulation order should be indistinguishable on
# this measure.
FURTHER_RELATIVE = 0.25
# Above this mean absolute deviation the two sides are not computing the same
# thing at all, and the classification below would be meaningless -- worse
# than meaningless, since "one run further" or "a bias" read as findings about
# the device when the likeliest cause is a mistake in the dequantisation here.
# A BF16 dot product over K=3072 accumulates well inside a few ULP of the
# exact result; tens of ULP means a layout or zero-point convention is wrong.
SANITY_MAX_MEAN_ULP = 8.0


def widen_bf16(bits) -> np.ndarray:
    """BF16 bit patterns to FP32. Exact and lossless; BF16 is the FP32 top
    half, so this reconstructs the value the device actually holds."""
    raw = (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16)
    return raw.view(np.float32)


def bf16_ulp(magnitude: np.ndarray) -> np.ndarray:
    """One BF16 ULP at each magnitude.

    The exponent is read from the FP32 BIT PATTERN, not via log2. These values
    were widened from BF16 so the exponent field is exact, and log2 of a value
    a hair under a power of two rounds the wrong way and shifts the answer by
    a factor of two -- the same trap design section 15.3's implementation note
    calls out.
    """
    magnitude = np.abs(np.asarray(magnitude, dtype=np.float32))
    raw = magnitude.view(np.uint32)
    exponent = ((raw >> 23) & 0xFF).astype(np.int32) - 127
    ulp = np.ldexp(np.ones_like(exponent, dtype=np.float64), exponent - 7)
    # Subnormal or zero: the smallest normal BF16 ULP is the right floor, and
    # a zero exponent field would otherwise produce a meaningless bound.
    ulp[(raw & 0x7F800000) == 0] = np.ldexp(1.0, -126 - 7)
    return ulp


def read_initializer(model_dir: Path, manifest: dict, name: str) -> np.ndarray:
    record = manifest["initializers"][name]
    path = model_dir / record["file"]
    dtype = {
        "uint8": np.uint8,
        "float16": np.float16,
        "float32": np.float32,
        "int64": np.int64,
    }[record["dtype"]]
    count = record["length"] // np.dtype(dtype).itemsize
    with path.open("rb") as handle:
        handle.seek(record["offset"])
        raw = handle.read(record["length"])
    if len(raw) != record["length"]:
        raise RuntimeError(
            f"{name}: read {len(raw)} bytes at offset {record['offset']} of "
            f"{path}, expected {record['length']}"
        )
    return np.frombuffer(raw, dtype=dtype, count=count)


def dequantise_rows(
    qweight: np.ndarray,
    scales: np.ndarray,
    qzeros: np.ndarray,
    first: int,
    count: int,
) -> np.ndarray:
    """Dequantise output rows [first, first + count) of the LM head.

    ONNX MatMulNBits packs two 4-bit weights per byte, low nibble first, in
    groups of `GROUP_SIZE` along K. The dequantised value is
    `(q - zero) * scale`, which is exact in FP32: `q - zero` is a small
    integer and `scale` is an FP16 value, so the product has at most 4 + 11
    significant bits.
    """
    groups = HIDDEN // GROUP_SIZE
    bytes_per_group = GROUP_SIZE // 2
    zero_bytes_per_row = groups // 2

    block = qweight.reshape(VOCAB, groups, bytes_per_group)[
        first : first + count
    ]
    low = (block & 0x0F).astype(np.int16)
    high = (block >> 4).astype(np.int16)
    # Interleave so element j of the group comes from nibble j: even j in the
    # low nibble, odd j in the high nibble.
    quantised = np.empty((count, groups, GROUP_SIZE), dtype=np.int16)
    quantised[:, :, 0::2] = low
    quantised[:, :, 1::2] = high

    zero_block = qzeros.reshape(VOCAB, zero_bytes_per_row)[
        first : first + count
    ]
    zero_low = (zero_block & 0x0F).astype(np.int16)
    zero_high = (zero_block >> 4).astype(np.int16)
    zeros = np.empty((count, groups), dtype=np.int16)
    zeros[:, 0::2] = zero_low
    zeros[:, 1::2] = zero_high

    scale_block = (
        scales.reshape(VOCAB, groups)[first : first + count]
    ).astype(np.float32)

    centred = (quantised - zeros[:, :, None]).astype(np.float32)
    weights = centred * scale_block[:, :, None]
    return weights.reshape(count, HIDDEN)


def host_lm_head(model_dir: Path, hidden: np.ndarray) -> np.ndarray:
    """FP64 logits for one LM-head row, from the packed ONNX components."""
    manifest = json.loads(
        (model_dir / "corelib_phi4_manifest.json").read_text(encoding="utf-8")
    )
    descriptor = next(
        entry
        for entry in manifest["weight_objects"]
        if entry["name"] == LM_HEAD_OBJECT
    )
    if (
        descriptor["descriptor"]["k"] != HIDDEN
        or descriptor["descriptor"]["n"] != VOCAB
        or descriptor["descriptor"]["group_size"] != GROUP_SIZE
    ):
        raise RuntimeError(
            f"the manifest's LM head is {descriptor['descriptor']}, not the "
            f"K={HIDDEN} N={VOCAB} group={GROUP_SIZE} this reference "
            f"implements"
        )
    roles = descriptor["roles"]
    qweight = read_initializer(model_dir, manifest, roles["qweight"])
    scales = read_initializer(model_dir, manifest, roles["scales"])
    qzeros = read_initializer(model_dir, manifest, roles["qzeros"])

    x = np.asarray(hidden, dtype=np.float64)
    logits = np.empty(VOCAB, dtype=np.float64)
    for first in range(0, VOCAB, CHUNK_ROWS):
        count = min(CHUNK_ROWS, VOCAB - first)
        weights = dequantise_rows(qweight, scales, qzeros, first, count)
        logits[first : first + count] = weights.astype(np.float64) @ x
    return logits


def bin_counts(indices: np.ndarray, bins: int = 32) -> list[int]:
    edges = np.linspace(0, VOCAB, bins + 1)
    counts, _ = np.histogram(indices, bins=edges)
    return [int(value) for value in counts]


def describe_run(
    label: str,
    npu: np.ndarray,
    reference: np.ndarray,
    top32: np.ndarray,
) -> dict:
    deviation = npu.astype(np.float64) - reference
    absolute = np.abs(deviation)
    ulp = bf16_ulp(np.maximum(np.abs(npu), np.abs(reference).astype(np.float32)))
    in_ulp = absolute / ulp

    # "Within half a BF16 ULP of truth" rather than "equal to the correctly
    # rounded truth". The two mean the same thing away from tie points, and
    # this form avoids a double-rounding artefact: reducing the FP64 reference
    # to FP32 and then to BF16 disagrees with a direct FP64-to-BF16 rounding
    # for values within 2^-24 of a BF16 midpoint, which over 200,064 logits
    # would silently misclassify a handful of them.
    within_half_ulp = absolute <= 0.5 * ulp
    positive = int(np.count_nonzero(deviation > 0))
    negative = int(np.count_nonzero(deviation < 0))
    zero = int(np.count_nonzero(deviation == 0))

    top_dev = absolute[top32]
    top_ulp = in_ulp[top32]

    # Uniform across the vocabulary, or concentrated? The differing logits are
    # binned into 32 equal ID ranges; a uniform mechanism puts roughly 1/32 of
    # them in each. The max/mean ratio is the single number that says which,
    # and the counts are emitted so nobody has to take the ratio on trust.
    differing = np.flatnonzero(~within_half_ulp)
    bins = bin_counts(differing)
    mean_bin = float(np.mean(bins)) if bins else 0.0

    return {
        "label": label,
        "max_abs_deviation": float(np.max(absolute)),
        "mean_abs_deviation": float(np.mean(absolute)),
        "mean_signed_deviation": float(np.mean(deviation)),
        "max_deviation_ulp": float(np.max(in_ulp)),
        "mean_abs_deviation_ulp": float(np.mean(in_ulp)),
        "mean_signed_deviation_ulp": float(np.mean(deviation / ulp)),
        "within_half_ulp": int(np.count_nonzero(within_half_ulp)),
        "total": int(npu.size),
        "within_half_ulp_fraction": float(
            np.count_nonzero(within_half_ulp) / npu.size
        ),
        "sign_distribution": {
            "npu_above_reference": positive,
            "npu_below_reference": negative,
            "exactly_equal": zero,
        },
        "top32": {
            "count": int(top32.size),
            "max_abs_deviation": float(np.max(top_dev)),
            "mean_abs_deviation": float(np.mean(top_dev)),
            "max_deviation_ulp": float(np.max(top_ulp)),
            "within_half_ulp": int(
                np.count_nonzero(within_half_ulp[top32])
            ),
        },
        "differing_index_distribution": {
            "bins": 32,
            "counts": bins,
            "max_over_mean": float(max(bins) / mean_bin) if mean_bin else 0.0,
        },
    }


def classify(runs: list[dict], straddle: dict | None) -> tuple[str, list[str]]:
    """Which of `DETERM-1`'s three possibilities the evidence supports."""
    notes: list[str] = []

    # Before any of the three: is the reference even describing the same
    # computation? A wrong nibble order or zero-point convention here would
    # produce a confident "common bias" about a device that is fine.
    worst = max(run["mean_abs_deviation_ulp"] for run in runs) if runs else 0.0
    if worst > SANITY_MAX_MEAN_ULP:
        notes.append(
            f"mean absolute deviation is {worst:.1f} BF16 ULP, far beyond the "
            f"{SANITY_MAX_MEAN_ULP} ULP a BF16 dot product over K=3072 can "
            f"accumulate. Either this script's dequantisation of "
            f"lm_head.MatMulNBits is wrong -- the likelier explanation, and "
            f"the first thing to check -- or the device is not computing the "
            f"LM head this package describes. Either way, the three-way "
            f"classification below would be meaningless and is not offered."
        )
        return "gross_disagreement", notes

    biased = [
        run
        for run in runs
        if abs(run["mean_signed_deviation_ulp"]) > BIAS_MEAN_SIGNED_ULP
    ]
    if len(biased) == len(runs) and runs:
        signs = {
            run["mean_signed_deviation_ulp"] > 0 for run in runs
        }
        if len(signs) == 1:
            notes.append(
                "every run's mean signed deviation exceeds "
                f"{BIAS_MEAN_SIGNED_ULP} BF16 ULP in the SAME direction: "
                + ", ".join(
                    f"{run['label']} {run['mean_signed_deviation_ulp']:+.4f} ULP"
                    for run in runs
                )
            )
            return "common_bias", notes

    if len(runs) == 2:
        left, right = runs
        larger = max(
            left["mean_abs_deviation_ulp"], right["mean_abs_deviation_ulp"]
        )
        gap = abs(
            left["mean_abs_deviation_ulp"] - right["mean_abs_deviation_ulp"]
        )
        if larger > 0 and gap / larger > FURTHER_RELATIVE:
            notes.append(
                f"the two runs are not equidistant from truth: "
                f"{left['label']} {left['mean_abs_deviation_ulp']:.4f} ULP "
                f"vs {right['label']} "
                f"{right['mean_abs_deviation_ulp']:.4f} ULP, a relative gap "
                f"of {gap / larger:.2%} against a {FURTHER_RELATIVE:.0%} "
                f"threshold"
            )
            return "one_run_further", notes

    if straddle is not None and straddle["differing_logits"] == 0:
        notes.append(
            "the two runs produced bit-identical logits for this step, so "
            "there is no run-to-run divergence in this sample to attribute. "
            "The deviation-from-truth figures below still stand, and they "
            "describe the kernel rather than the divergence."
        )
        return "benign_no_divergence_in_sample", notes

    if straddle is not None:
        notes.append(
            f"{straddle['reference_between']}/{straddle['differing_logits']} "
            f"of the logits where the two runs disagree have the true value "
            f"between them, and "
            f"{straddle['both_within_half_ulp']}/"
            f"{straddle['differing_logits']} have both runs within half a "
            f"BF16 ULP of truth"
        )
    return "benign_accumulation_order", notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare NPU LM-head logits against an FP64 host reference "
            "computed from the same ONNX components corelib packs from. "
            "Diagnostic only; see the module docstring."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--run-json",
        action="append",
        required=True,
        help=(
            "a test_phi4_e2e output document. Pass twice to analyse a "
            "run-to-run pair."
        ),
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    documents = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.run_json
    ]
    if not documents:
        print("no run documents supplied", file=sys.stderr)
        return 2

    # Every document must describe the same LM-head input, or the reference
    # computed from the first would be truth for a different question.
    hidden_bits = documents[0]["final_snapshot"]["last_hidden"]
    for path, document in zip(args.run_json[1:], documents[1:]):
        if document["final_snapshot"]["last_hidden"] != hidden_bits:
            print(
                f"{path} has a different LM-head input from the first "
                f"document. DETERM-1's whole claim is IDENTICAL input with "
                f"non-identical output; comparing these two would answer a "
                f"different question.",
                file=sys.stderr,
            )
            return 2
    hashes = {document.get("corelib_sha256") for document in documents}
    if len(hashes) != 1:
        print(
            f"the run documents loaded different corelib DLLs {hashes}; the "
            f"result would not be about one binary",
            file=sys.stderr,
        )
        return 2

    hidden = widen_bf16(hidden_bits)
    if hidden.size != HIDDEN:
        raise RuntimeError(
            f"last_hidden holds {hidden.size} values, expected {HIDDEN}"
        )

    model_dir = Path(args.model_dir)
    reference = host_lm_head(model_dir, hidden)

    # The last decode step's logits ARE the contents of lm_output_tensor at
    # the moment final_snapshot was taken, which is what makes them the output
    # of the dispatch whose input is last_hidden.
    npu_runs = [
        widen_bf16(document["decode"][-1]["logits_bf16"])
        for document in documents
    ]
    for values in npu_runs:
        if values.size != VOCAB:
            raise RuntimeError(
                f"logits hold {values.size} values, expected {VOCAB}"
            )

    reference_order = np.argsort(-reference, kind="stable")[:32]
    union_top32 = set(int(index) for index in reference_order)
    for values in npu_runs:
        union_top32.update(
            int(index) for index in np.argsort(-values, kind="stable")[:32]
        )
    top32 = np.array(sorted(union_top32), dtype=np.int64)

    runs = [
        describe_run(f"run{index}", values, reference, top32)
        for index, values in enumerate(npu_runs)
    ]

    straddle = None
    if len(npu_runs) == 2:
        left, right = npu_runs
        differing = np.flatnonzero(left != right)
        if differing.size:
            low = np.minimum(left[differing], right[differing]).astype(
                np.float64
            )
            high = np.maximum(left[differing], right[differing]).astype(
                np.float64
            )
            truth = reference[differing]
            between = int(
                np.count_nonzero((truth >= low) & (truth <= high))
            )
            half_ulp = 0.5 * bf16_ulp(
                np.maximum(
                    np.abs(left[differing]), np.abs(right[differing])
                )
            )
            both_close = int(
                np.count_nonzero(
                    (np.abs(left[differing] - truth) <= half_ulp)
                    & (np.abs(right[differing] - truth) <= half_ulp)
                )
            )
        else:
            between = 0
            both_close = 0
        straddle = {
            "differing_logits": int(differing.size),
            "reference_between": between,
            "both_within_half_ulp": both_close,
            "differing_index_distribution": {
                "bins": 32,
                "counts": bin_counts(differing),
            },
        }

    verdict, notes = classify(runs, straddle)

    result = {
        "diagnostic": "phi4 host LM-head reference (Task 13 Step 9b)",
        "product_path": False,
        "model_dir": str(model_dir),
        "corelib_sha256": documents[0].get("corelib_sha256"),
        "corelib_loaded_path": documents[0].get("corelib_loaded_path"),
        "continuation_route": documents[0].get("continuation_route"),
        "prefix_ids": documents[0].get("prefix_ids"),
        "suffix_ids": documents[0].get("suffix_ids"),
        "decode_step_index": len(documents[0]["decode"]) - 1,
        "decode_input_id": documents[0]["decode"][-1].get("input_id"),
        "reference": {
            "implementation": (
                "FP64 dense matmul over lm_head.MatMulNBits dequantised as "
                "(q - zero) * scale, 4-bit, group 128, read at the byte "
                "offsets the package manifest records"
            ),
            "k": HIDDEN,
            "n": VOCAB,
            "group_size": GROUP_SIZE,
            "accumulation": "float64",
            "top1_id": int(reference_order[0]),
            "top5_ids": [int(index) for index in reference_order[:5]],
        },
        "npu_top1_ids": [
            int(np.argsort(-values, kind="stable")[0]) for values in npu_runs
        ],
        "runs": runs,
        "run_to_run": straddle,
        "thresholds": {
            "bias_mean_signed_ulp": BIAS_MEAN_SIGNED_ULP,
            "one_run_further_relative": FURTHER_RELATIVE,
        },
        "verdict": verdict,
        "notes": notes,
    }
    Path(args.output_json).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    for run in runs:
        print(
            f"{run['label']}: max |dev| {run['max_abs_deviation']:.6g} "
            f"({run['max_deviation_ulp']:.3f} ULP), mean |dev| "
            f"{run['mean_abs_deviation']:.6g} "
            f"({run['mean_abs_deviation_ulp']:.4f} ULP), within half a ULP "
            f"{run['within_half_ulp']}/{run['total']}, signed mean "
            f"{run['mean_signed_deviation_ulp']:+.4f} ULP"
        )
    if straddle is not None:
        print(
            f"run-to-run: {straddle['differing_logits']} logits differ; "
            f"truth between the two in {straddle['reference_between']}, "
            f"both within half a ULP in {straddle['both_within_half_ulp']}"
        )
    for note in notes:
        print(f"note: {note}")
    print(f"verdict: {verdict}")

    # (2) and (3) are a different finding from the one DETERM-1 accepts and
    # they reopen the product decision, so they reach the exit code rather
    # than sitting in a JSON file nobody reads.
    if verdict in ("one_run_further", "common_bias"):
        print(
            "STOP AND REPORT: this is not the benign accumulation-order "
            "nondeterminism DETERM-1 accepts.",
            file=sys.stderr,
        )
        return 1
    if verdict == "gross_disagreement":
        print(
            "STOP: the reference and the device disagree by far more than "
            "rounding. Check this script's dequantisation before reading "
            "anything into the numbers.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
