"""Tests for the `DETERM-1` host LM-head reference. Task 13 Step 9b.

The script itself can only run against a 3.2 GB model on the AIE4 target, so
what is testable here is everything that decides what the numbers MEAN:

  * the ONNX `MatMulNBits` unpacking, against a hand-built block whose answer
    is worked out by hand rather than by a second copy of the same code;
  * the BF16 ULP function at the binade edges, which is where a `log2`-based
    implementation would silently be off by a factor of two;
  * the step selection, which must land on the FIRST diverging step and not
    the last one — analysing the last step of a pair that has already emitted
    different tokens answers a different question; and
  * the three-way classification, including the refusal to classify at all
    when the reference and the device disagree by more than rounding.

A review found the earlier version of this script wired into nothing and
covered by nothing, which is why this file exists.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.phi4_host_lm_head_reference import (
    BIAS_MEAN_SIGNED_ULP,
    SANITY_MAX_MEAN_ULP,
    bf16_ulp,
    classify,
    dequantise_rows,
    first_diverging_step,
    lm_head_input,
    step_labels,
    step_logits,
    widen_bf16,
)


def _bf16_bits(value: float) -> int:
    """The BF16 pattern of an exactly representable float."""
    raw = np.float32(value).view(np.uint32)
    assert raw & 0xFFFF == 0, f"{value} is not exactly representable in BF16"
    return int(raw >> 16)


class DequantisationTests(unittest.TestCase):
    """corelib.h: `qweight` is UINT4 codes [N, K/2], two per byte, LOW NIBBLE
    FIRST; `qzeros` is UINT4 codes, two per byte; the value is
    `(q - zero) * scale`."""

    def test_nibble_order_and_zero_point(self):
        # One output row, K = 8, two groups of 4. Codes chosen so every nibble
        # is distinguishable and the expected answer is arithmetic anyone can
        # check: group 0 codes 1..4 with zero 8 and scale 2, group 1 codes
        # 9..12 with zero 3 and scale 0.5.
        k, n, group = 8, 1, 4
        qweight = np.array(
            [
                # group 0: elements 0,1 then 2,3 -> low nibble first
                (2 << 4) | 1,
                (4 << 4) | 3,
                # group 1
                (10 << 4) | 9,
                (12 << 4) | 11,
            ],
            dtype=np.uint8,
        )
        # Two groups, so one byte of zero points: group 0 low, group 1 high.
        qzeros = np.array([(3 << 4) | 8], dtype=np.uint8)
        scales = np.array([2.0, 0.5], dtype=np.float16)

        weights = dequantise_rows(
            qweight, scales, qzeros, 0, 1, k=k, n=n, group_size=group
        )
        expected = np.array(
            [
                (1 - 8) * 2.0,
                (2 - 8) * 2.0,
                (3 - 8) * 2.0,
                (4 - 8) * 2.0,
                (9 - 3) * 0.5,
                (10 - 3) * 0.5,
                (11 - 3) * 0.5,
                (12 - 3) * 0.5,
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(weights[0], expected)

    def test_a_swapped_nibble_order_would_not_pass(self):
        # The guard the previous test is really providing: if the unpacking
        # took the HIGH nibble first, element 0 would be 2 rather than 1.
        k, n, group = 4, 1, 4
        qweight = np.array([(2 << 4) | 1, (4 << 4) | 3], dtype=np.uint8)
        qzeros = np.array([0], dtype=np.uint8)
        scales = np.array([1.0], dtype=np.float16)
        weights = dequantise_rows(
            qweight, scales, qzeros, 0, 1, k=k, n=n, group_size=group
        )
        self.assertEqual(weights[0][0], 1.0)
        self.assertEqual(weights[0][1], 2.0)

    def test_multiple_output_rows_are_sliced_independently(self):
        k, n, group = 4, 3, 4
        qweight = np.array(
            [(2 << 4) | 1, (4 << 4) | 3] * 3, dtype=np.uint8
        )
        qweight[2] = (6 << 4) | 5
        qzeros = np.array([0, 0, 0], dtype=np.uint8)
        scales = np.array([1.0, 1.0, 1.0], dtype=np.float16)
        rows = dequantise_rows(
            qweight, scales, qzeros, 1, 2, k=k, n=n, group_size=group
        )
        self.assertEqual(rows.shape, (2, 4))
        self.assertEqual(rows[0][0], 5.0)


class Bf16UlpTests(unittest.TestCase):
    def test_binade_edges(self):
        # BF16 has 8 significand bits, so one ULP at magnitude m in [2^e,
        # 2^(e+1)) is 2^(e-7).
        for magnitude, expected in (
            (1.0, 2.0**-7),
            (1.9921875, 2.0**-7),
            (2.0, 2.0**-6),
            (16.0, 2.0**-3),
            (31.75, 2.0**-3),
            (32.0, 2.0**-2),
            (40.0, 2.0**-2),
        ):
            self.assertEqual(
                float(bf16_ulp(np.float32([magnitude]))[0]),
                expected,
                f"magnitude {magnitude}",
            )

    def test_the_units_error_the_design_had_to_correct(self):
        # Design 15.3: an absolute 0.25 bound is 2 ULP at magnitude 20 and
        # only 1 ULP at 40, which is the whole reason the bound is stated in
        # ULP. If this function were wrong that correction would be undone.
        self.assertEqual(float(bf16_ulp(np.float32([20.0]))[0]) * 2, 0.25)
        self.assertEqual(float(bf16_ulp(np.float32([40.0]))[0]) * 2, 0.5)

    def test_zero_does_not_produce_a_meaningless_bound(self):
        self.assertGreater(float(bf16_ulp(np.float32([0.0]))[0]), 0.0)

    def test_widen_bf16_round_trips(self):
        bits = [_bf16_bits(1.5), _bf16_bits(-20.0), _bf16_bits(0.0)]
        np.testing.assert_array_equal(
            widen_bf16(bits), np.float32([1.5, -20.0, 0.0])
        )


def _document(logits_per_step, inputs_per_step=None):
    steps = len(logits_per_step)
    inputs_per_step = inputs_per_step or [[0] * 4] * steps
    document = {
        "continuation": {
            "logits_bf16": logits_per_step[0],
            "lm_head_input_bf16": inputs_per_step[0],
        },
        "decode": [
            {
                "logits_bf16": logits_per_step[index],
                "lm_head_input_bf16": inputs_per_step[index],
            }
            for index in range(1, steps)
        ],
    }
    return document


class StepSelectionTests(unittest.TestCase):
    """The bug this replaces: the tool always analysed the LAST decode step.

    For a pair that has already emitted different tokens, the last step's
    hidden states legitimately differ and the comparison answers nothing. The
    question lives at the FIRST diverging step.
    """

    def test_labels_and_accessors_line_up(self):
        document = _document([[1], [2], [3]])
        self.assertEqual(
            step_labels(document), ["continuation", "decode[0]", "decode[1]"]
        )
        self.assertEqual(step_logits(document, 0), [1])
        self.assertEqual(step_logits(document, 2), [3])
        self.assertEqual(lm_head_input(document, 2), [0, 0, 0, 0])

    def test_the_first_diverging_step_is_found_not_the_last(self):
        left = _document([[1], [2], [3], [4]])
        right = _document([[1], [9], [9], [9]])
        self.assertEqual(first_diverging_step([left, right]), 1)

    def test_divergence_in_the_continuation_step_is_index_zero(self):
        left = _document([[1], [2]])
        right = _document([[7], [2]])
        self.assertEqual(first_diverging_step([left, right]), 0)

    def test_identical_runs_have_no_diverging_step(self):
        left = _document([[1], [2]])
        self.assertIsNone(first_diverging_step([left, _document([[1], [2]])]))

    def test_a_single_document_has_no_diverging_step(self):
        self.assertIsNone(first_diverging_step([_document([[1]])]))

    def test_a_missing_input_reads_as_absent_rather_than_empty(self):
        document = _document([[1], [2]])
        del document["decode"][0]["lm_head_input_bf16"]
        self.assertIsNone(lm_head_input(document, 1))


def _run(label, mean_signed_ulp=0.0, mean_abs_ulp=0.25):
    return {
        "label": label,
        "mean_signed_deviation_ulp": mean_signed_ulp,
        "mean_abs_deviation_ulp": mean_abs_ulp,
    }


class ClassificationTests(unittest.TestCase):
    def test_benign_accumulation_order(self):
        straddle = {
            "applicable": True,
            "differing_logits": 100,
            "reference_between": 100,
            "both_within_half_ulp": 100,
        }
        verdict, _ = classify([_run("a"), _run("b")], straddle)
        self.assertEqual(verdict, "benign_accumulation_order")

    def test_one_run_systematically_further(self):
        verdict, notes = classify(
            [_run("a", 0.0, 0.25), _run("b", 0.0, 0.9)], None
        )
        self.assertEqual(verdict, "one_run_further")
        self.assertTrue(notes)

    def test_a_common_bias(self):
        offset = BIAS_MEAN_SIGNED_ULP + 0.1
        verdict, notes = classify(
            [_run("a", offset, 0.3), _run("b", offset, 0.3)], None
        )
        self.assertEqual(verdict, "common_bias")
        self.assertTrue(notes)

    def test_opposite_offsets_are_not_a_common_bias(self):
        offset = BIAS_MEAN_SIGNED_ULP + 0.1
        verdict, _ = classify(
            [_run("a", offset, 0.3), _run("b", -offset, 0.3)], None
        )
        self.assertNotEqual(verdict, "common_bias")

    def test_gross_disagreement_refuses_to_classify(self):
        # A wrong nibble order in this script would land here, and reporting
        # "common bias" about the device instead would be a false alarm about
        # hardware caused by a bug in the analysis.
        worst = SANITY_MAX_MEAN_ULP + 1
        verdict, notes = classify(
            [_run("a", 5.0, worst), _run("b", 5.0, worst)], None
        )
        self.assertEqual(verdict, "gross_disagreement")
        self.assertIn("dequantisation", notes[0])

    def test_different_lm_head_inputs_are_a_model_body_divergence(self):
        straddle = {
            "applicable": False,
            "lm_head_input_differing_elements": 2587,
            "reason": "the two runs fed the LM head DIFFERENT rows",
        }
        verdict, notes = classify([_run("a"), _run("b")], straddle)
        self.assertEqual(verdict, "model_body_divergence")
        self.assertTrue(notes)

    def test_a_bit_identical_pair_says_there_is_nothing_to_attribute(self):
        straddle = {
            "applicable": True,
            "differing_logits": 0,
            "reference_between": 0,
            "both_within_half_ulp": 0,
        }
        verdict, notes = classify([_run("a"), _run("b")], straddle)
        self.assertEqual(verdict, "benign_no_divergence_in_sample")
        self.assertIn("bit-identical", notes[0])


if __name__ == "__main__":
    unittest.main()
