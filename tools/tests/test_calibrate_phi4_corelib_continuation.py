"""Tests for the Phi-4 continuation threshold calibrator. Task 14 Steps 1-5.

Design Section 10.7 fixes ONE integer for the release, chosen as "the largest
suffix length whose append p95 is lower at BOTH history lengths, but only after
asserting that the set of winning sampled lengths is a prefix-contiguous set".
Everything below tests that sentence and the ways it can go wrong -- not the
one dataset that happens to be checked in today.

Three defects in Task 13 survived because a test and its implementation shared
an assumption, so the test could not see the implementation being wrong about
it. The rule followed here is therefore to enumerate the cases the CODE
BRANCHES ON rather than the case that exists:

  * `select_threshold` branches on empty / no-winner / prefix-winner /
    non-monotonic, on strict-versus-tied comparison, and on the two accepted
    entry shapes -- all four verdicts and both shapes are exercised;
  * ingestion branches on every validation it performs, so every rejection has
    a test that constructs the specific malformed record;
  * the grid width is a branch that MATTERS: the retracted answer "2" came from
    honouring only the five spec-named suffixes. There is a test that the same
    routes measured on a denser grid select a different, larger threshold, so
    an implementation that silently drops the extra points fails;
  * the header writer branches on bootstrap-versus-regenerate and must be
    idempotent, so it is applied twice;
  * the cross-run check branches on permits / contradicts / permits-without-
    confirming / nothing comparable, and the contradicting case must reach a
    non-zero exit; and
  * the published concession is ordered by COST, and the cheapest member of a
    conceded band is its widest suffix -- so there is a test that the reported
    worst case is not the widest one.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import tempfile
import unittest

from tools.report_phi4_corelib_baseline import crossover_entry
from tools.calibrate_phi4_corelib_continuation import (
    CalibrationError,
    MIN_WARM_SAMPLES,
    REQUIRED_HISTORIES,
    REQUIRED_SUFFIXES,
    Selection,
    apply_section_to_document,
    apply_threshold_to_header,
    build_document_section,
    conceded_points,
    history_agreement,
    load_continuation_samples,
    lower_edge_confidence,
    main,
    percentile_ns,
    recorded_lower_edges,
    select_threshold,
    select_threshold_detailed,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BASELINE = _REPO_ROOT / "docs" / "docs" / "benchmarks" / "phi4_aie4_baseline.json"
_HISTORY = (
    _REPO_ROOT / "docs" / "docs" / "benchmarks" / "phi4_aie4_crossover_history.json"
)
_HEADER = (
    _REPO_ROOT
    / "src"
    / "include"
    / "models"
    / "phi4"
    / "phi4_corelib_aie4_tuning.hpp"
)
_DOCUMENT = _REPO_ROOT / "docs" / "docs" / "benchmarks" / "phi4_results.md"

# The task report is a plan artifact rather than a shipped file, so it sits
# in the SDD working area beside the repository and is not committed here.
# The guard reads it when it is present: a retracted claim survived two
# review rounds in that file precisely because nothing read it back.
_REPORT = (
    _REPO_ROOT
    / ".superpowers"
    / "sdd"
    / "2026-08-31-phi4-aie4-corelib-fastflow"
    / "task-14-report.md"
)


def _flat(append: float, reprefill: float) -> dict:
    return {"append_p95": append, "reprefill_p95": reprefill}


def _per_history(**by_history: tuple) -> dict:
    return {
        "per_history": {
            int(name.lstrip("h")): _flat(*pair)
            for name, pair in by_history.items()
        }
    }


class SelectThresholdTest(unittest.TestCase):
    """The Section 10.7 selection rule, one branch at a time."""

    def test_non_monotonic_winners_select_zero(self):
        samples = {
            1: {"append_p95": 4.0, "reprefill_p95": 5.0},
            2: {"append_p95": 7.0, "reprefill_p95": 5.0},
            32: {"append_p95": 4.0, "reprefill_p95": 5.0},
        }
        self.assertEqual(select_threshold(samples), 0)

    def test_non_monotonic_emits_a_diagnostic_naming_the_offender(self):
        samples = {
            1: _flat(4.0, 5.0),
            2: _flat(7.0, 5.0),
            32: _flat(4.0, 5.0),
        }
        selection = select_threshold_detailed(samples)
        self.assertIsInstance(selection, Selection)
        self.assertEqual(selection.threshold, 0)
        self.assertEqual(selection.verdict, "non_monotonic")
        self.assertEqual(selection.winners, (1, 32))
        self.assertTrue(selection.diagnostics)
        joined = " ".join(selection.diagnostics)
        self.assertIn("32", joined)
        self.assertIn("2", joined)

    def test_no_winner_selects_zero(self):
        samples = {
            1: _flat(9.0, 5.0),
            2: _flat(9.0, 5.0),
            32: _flat(9.0, 5.0),
        }
        selection = select_threshold_detailed(samples)
        self.assertEqual(selection.threshold, 0)
        self.assertEqual(selection.verdict, "no_winner")
        self.assertEqual(selection.winners, ())

    def test_prefix_contiguous_winners_select_the_largest_winner(self):
        samples = {
            1: _flat(4.0, 5.0),
            2: _flat(4.0, 5.0),
            32: _flat(9.0, 5.0),
            128: _flat(9.0, 5.0),
        }
        selection = select_threshold_detailed(samples)
        self.assertEqual(selection.threshold, 2)
        self.assertEqual(selection.verdict, "selected")
        self.assertEqual(selection.winners, (1, 2))
        self.assertEqual(selection.diagnostics, ())

    def test_every_sampled_length_winning_selects_the_largest_sampled_length(self):
        samples = {1: _flat(1.0, 5.0), 2: _flat(1.0, 5.0), 32: _flat(1.0, 5.0)}
        selection = select_threshold_detailed(samples)
        self.assertEqual(selection.threshold, 32)
        self.assertEqual(selection.verdict, "selected")

    def test_empty_sample_set_selects_zero_rather_than_raising(self):
        selection = select_threshold_detailed({})
        self.assertEqual(selection.threshold, 0)
        self.assertEqual(selection.verdict, "empty")

    def test_a_tie_is_not_a_win(self):
        """Section 10.7 says append p95 must be LOWER, not "not higher"."""
        samples = {1: _flat(5.0, 5.0), 2: _flat(9.0, 5.0)}
        selection = select_threshold_detailed(samples)
        self.assertEqual(selection.threshold, 0)
        self.assertEqual(selection.verdict, "no_winner")

    def test_insertion_order_does_not_change_the_answer(self):
        forwards = {1: _flat(1.0, 5.0), 2: _flat(1.0, 5.0), 32: _flat(9.0, 5.0)}
        backwards = {32: _flat(9.0, 5.0), 2: _flat(1.0, 5.0), 1: _flat(1.0, 5.0)}
        self.assertEqual(select_threshold(forwards), 2)
        self.assertEqual(select_threshold(backwards), 2)

    def test_the_return_value_is_a_plain_int(self):
        value = select_threshold({1: _flat(1.0, 5.0)})
        self.assertIs(type(value), int)

    def test_a_win_at_one_history_and_a_loss_at_the_other_is_not_a_win(self):
        samples = {
            1: _per_history(h512=(1.0, 5.0), h2048=(1.0, 5.0)),
            2: _per_history(h512=(9.0, 5.0), h2048=(1.0, 5.0)),
            4: _per_history(h512=(9.0, 5.0), h2048=(9.0, 5.0)),
        }
        selection = select_threshold_detailed(samples)
        self.assertEqual(selection.threshold, 1)
        self.assertEqual(selection.winners, (1,))

    def test_a_win_at_every_history_is_a_win(self):
        samples = {
            1: _per_history(h512=(1.0, 5.0), h2048=(1.0, 5.0)),
            2: _per_history(h512=(1.0, 5.0), h2048=(1.0, 5.0)),
        }
        self.assertEqual(select_threshold(samples), 2)

    def test_the_conjunction_is_not_a_worst_case_reduction(self):
        """max(append) < min(reprefill) is STRICTER than a win at each history.

        A reduction that collapses the histories to one worst-case pair before
        comparing would reject suffix 1 here, because max append 10 is not
        below min reprefill 2 -- even though append wins at BOTH histories.
        """
        samples = {1: _per_history(h512=(10.0, 20.0), h2048=(1.0, 2.0))}
        self.assertEqual(select_threshold(samples), 1)

    def test_an_empty_per_history_map_is_rejected(self):
        with self.assertRaises(CalibrationError):
            select_threshold({1: {"per_history": {}}})

    def test_mixing_the_two_entry_shapes_in_one_entry_is_rejected(self):
        entry = _flat(1.0, 5.0)
        entry["per_history"] = {512: _flat(1.0, 5.0)}
        with self.assertRaises(CalibrationError):
            select_threshold({1: entry})

    def test_an_entry_with_neither_shape_is_rejected(self):
        with self.assertRaises(CalibrationError):
            select_threshold({1: {"append_p50": 1.0}})

    def test_a_non_numeric_measurement_is_rejected(self):
        with self.assertRaises(CalibrationError):
            select_threshold({1: _flat("fast", 5.0)})

    def test_a_non_positive_suffix_length_is_rejected(self):
        with self.assertRaises(CalibrationError):
            select_threshold({0: _flat(1.0, 5.0)})
        with self.assertRaises(CalibrationError):
            select_threshold({-4: _flat(1.0, 5.0)})

    def test_a_non_integer_suffix_length_is_rejected(self):
        with self.assertRaises(CalibrationError):
            select_threshold({1.5: _flat(1.0, 5.0)})

    def test_a_boolean_suffix_length_is_rejected(self):
        """`True == 1` in Python; a bool key must not pass as suffix 1."""
        with self.assertRaises(CalibrationError):
            select_threshold({True: _flat(1.0, 5.0)})


class PercentileTest(unittest.TestCase):
    """Nearest-rank, because that is what the recorded p50/p95 used."""

    def test_median_of_an_odd_sample_count(self):
        self.assertEqual(percentile_ns([5, 1, 3, 2, 4], 0.5), 3)

    def test_median_of_an_even_sample_count_does_not_interpolate(self):
        self.assertEqual(percentile_ns([1, 2, 3, 4], 0.5), 2)

    def test_p95_of_five_samples_is_the_largest(self):
        self.assertEqual(percentile_ns([5, 1, 3, 2, 4], 0.95), 5)

    def test_p95_of_twenty_samples_is_the_nineteenth(self):
        self.assertEqual(percentile_ns(list(range(1, 21)), 0.95), 19)

    def test_a_single_sample_is_both_percentiles(self):
        self.assertEqual(percentile_ns([7], 0.5), 7)
        self.assertEqual(percentile_ns([7], 0.95), 7)

    def test_an_empty_sample_list_is_rejected(self):
        with self.assertRaises(CalibrationError):
            percentile_ns([], 0.5)


def _point(history: int, suffix: int, route: str, base_ns: int) -> dict:
    samples = [base_ns + step * 1000 for step in range(MIN_WARM_SAMPLES)]
    ordered = sorted(samples)
    return {
        "history_rows": history,
        "suffix": suffix,
        "route": route,
        "count": len(samples),
        "samples_ns": samples,
        "p50_ns": ordered[2],
        "p95_ns": ordered[4],
        "interleaved_with_reprefill": True,
    }


def _synthetic_document(
    suffixes=(1, 2, 32, 128, 256),
    append_ns=None,
    reprefill_ns=500_000_000,
) -> dict:
    """A minimal but VALID document; each test breaks exactly one thing.

    Append costs 50 ms per suffix token, so at the default re-prefill cost of
    500 ms append wins at suffix 1, 2, 4 and 8 and loses from 32 up.
    `reprefill_ns` may be one number or a per-history mapping, which is how a
    test makes the crossover sit at a DIFFERENT suffix for each history.
    """
    append_ns = append_ns or {}
    if not isinstance(reprefill_ns, dict):
        reprefill_ns = {history: reprefill_ns for history in REQUIRED_HISTORIES}
    points = []
    for history in REQUIRED_HISTORIES:
        for suffix in suffixes:
            points.append(
                _point(
                    history,
                    suffix,
                    "append",
                    append_ns.get((history, suffix), 50_000_000 * suffix),
                )
            )
            points.append(
                _point(history, suffix, "reprefill", reprefill_ns[history])
            )
    return {
        "identity": {
            "machine": "xcomedusad-43",
            "utc": "2026-09-02T16:32:18Z",
            "fastflow_revision": "deadbeef",
            "corelib_dll_sha256": "0" * 64,
            "model_sha256": "1" * 64,
            "cpu_sku": "AMD Eng Sample",
            "npu_sku": "AMD XDNA(TM) NPU",
            "npu_driver_version": "32.0.20214.4161",
        },
        "continuation": {
            "histories": list(REQUIRED_HISTORIES),
            "suffixes": list(suffixes),
            "warm_samples_per_point": MIN_WARM_SAMPLES,
            "samples_interleaved": True,
            "prefix_monotonic": True,
            "points": points,
        },
    }


def _unclosed_document() -> dict:
    """A record whose history-512 winner set is NOT downward-closed.

    `W512 = {1, 8}` and `W2048 = {1, 2, 4}`: append loses at 512/2 and 512/4 and
    wins again at 512/8. The intersection is `{1}`, which IS a prefix, so
    Section 10.7's monotonicity assertion passes and the rule returns 1 -- while
    `min` of the ceilings (8 and 4) would say 4. This is the review's I-1
    counter-example as a loadable document.

    It is also the only shape in which concessions can appear at BOTH histories.
    With downward-closed winner sets the ceilings are nested, so everything
    above the threshold belongs to whichever history has the larger ceiling and
    the conceded band lives at exactly one history. Two-history ordering is
    therefore only reachable through a record like this one.
    """
    append = {
        (512, 1): 10_000_000,
        (512, 2): 2_000_000_000,
        (512, 4): 2_000_000_000,
        (512, 8): 20_000_000,
        (512, 32): 5_000_000_000,
        (512, 128): 5_000_000_000,
        (512, 256): 5_000_000_000,
        (2048, 1): 10_000_000,
        (2048, 2): 500_000_000,
        (2048, 4): 800_000_000,
        (2048, 8): 2_000_000_000,
        (2048, 32): 5_000_000_000,
        (2048, 128): 5_000_000_000,
        (2048, 256): 5_000_000_000,
    }
    return _synthetic_document(
        suffixes=(1, 2, 4, 8, 32, 128, 256),
        append_ns=append,
        reprefill_ns={512: 1_000_000_000, 2048: 1_000_000_000},
    )


class IngestionTest(unittest.TestCase):
    """Every rejection ingestion performs gets the record that triggers it."""

    def test_a_valid_synthetic_document_loads(self):
        loaded = load_continuation_samples(_synthetic_document())
        self.assertEqual(loaded.histories, tuple(sorted(REQUIRED_HISTORIES)))
        self.assertEqual(loaded.suffixes, (1, 2, 32, 128, 256))
        self.assertEqual(len(loaded.points), 2 * 5)

    def test_the_spec_grid_alone_selects_two_and_a_denser_grid_selects_more(self):
        """The retracted "2" was a property of the grid, not of the machine.

        Same routes, same crossover: on `{1, 2, 32, 128, 256}` the last winning
        sampled length is 2, and on a grid that also contains 4 and 8 it is 8.
        An implementation that honoured only the five spec-named suffixes would
        pass every other test in this file and still reproduce the retracted
        answer, so this is the test that separates them.
        """
        sparse = load_continuation_samples(_synthetic_document())
        self.assertEqual(select_threshold(sparse.samples()), 2)

        dense = load_continuation_samples(
            _synthetic_document(suffixes=(1, 2, 4, 8, 32, 128, 256))
        )
        self.assertEqual(dense.suffixes, (1, 2, 4, 8, 32, 128, 256))
        self.assertEqual(select_threshold(dense.samples()), 8)

    def test_a_missing_required_history_is_rejected(self):
        for dropped in REQUIRED_HISTORIES:
            with self.subTest(history=dropped):
                document = _synthetic_document()
                block = document["continuation"]
                block["points"] = [
                    point
                    for point in block["points"]
                    if point["history_rows"] != dropped
                ]
                block["histories"] = [
                    h for h in block["histories"] if h != dropped
                ]
                with self.assertRaises(CalibrationError) as caught:
                    load_continuation_samples(document)
                self.assertIn(str(dropped), str(caught.exception))

    def test_an_unexpected_history_is_rejected(self):
        """The conjunction is OVER the histories, so its domain is fixed."""
        document = _synthetic_document()
        block = document["continuation"]
        block["points"].append(_point(1024, 1, "append", 1_000_000))
        block["points"].append(_point(1024, 1, "reprefill", 2_000_000))
        block["histories"].append(1024)
        with self.assertRaises(CalibrationError) as caught:
            load_continuation_samples(document)
        self.assertIn("1024", str(caught.exception))

    def test_a_missing_required_suffix_is_rejected(self):
        for dropped in REQUIRED_SUFFIXES:
            with self.subTest(suffix=dropped):
                document = _synthetic_document()
                block = document["continuation"]
                block["points"] = [
                    point for point in block["points"] if point["suffix"] != dropped
                ]
                block["suffixes"] = [
                    s for s in block["suffixes"] if s != dropped
                ]
                with self.assertRaises(CalibrationError) as caught:
                    load_continuation_samples(document)
                self.assertIn(str(dropped), str(caught.exception))

    def test_too_few_warm_samples_at_one_point_is_rejected(self):
        document = _synthetic_document()
        victim = document["continuation"]["points"][3]
        victim["samples_ns"] = victim["samples_ns"][: MIN_WARM_SAMPLES - 1]
        victim["count"] = len(victim["samples_ns"])
        ordered = sorted(victim["samples_ns"])
        victim["p50_ns"] = ordered[len(ordered) // 2]
        victim["p95_ns"] = ordered[-1]
        with self.assertRaises(CalibrationError) as caught:
            load_continuation_samples(document)
        self.assertIn("warm", str(caught.exception).lower())

    def test_a_declared_warm_sample_floor_below_five_is_rejected(self):
        document = _synthetic_document()
        document["continuation"]["warm_samples_per_point"] = MIN_WARM_SAMPLES - 1
        with self.assertRaises(CalibrationError):
            load_continuation_samples(document)

    def test_a_point_missing_one_route_is_rejected(self):
        for route in ("append", "reprefill"):
            with self.subTest(route=route):
                document = _synthetic_document()
                block = document["continuation"]
                index = next(
                    i
                    for i, point in enumerate(block["points"])
                    if point["route"] == route and point["suffix"] == 32
                )
                del block["points"][index]
                with self.assertRaises(CalibrationError) as caught:
                    load_continuation_samples(document)
                self.assertIn(route, str(caught.exception))

    def test_a_duplicate_point_is_rejected(self):
        document = _synthetic_document()
        block = document["continuation"]
        block["points"].append(copy.deepcopy(block["points"][0]))
        with self.assertRaises(CalibrationError) as caught:
            load_continuation_samples(document)
        self.assertIn("duplicate", str(caught.exception).lower())

    def test_an_unknown_route_name_is_rejected(self):
        document = _synthetic_document()
        document["continuation"]["points"][0]["route"] = "appendix"
        with self.assertRaises(CalibrationError):
            load_continuation_samples(document)

    def test_a_non_interleaved_record_is_refused(self):
        """The first, non-interleaved round is the one that was retracted."""
        for flag in (False, None):
            with self.subTest(flag=flag):
                document = _synthetic_document()
                document["continuation"]["samples_interleaved"] = flag
                with self.assertRaises(CalibrationError) as caught:
                    load_continuation_samples(document)
                self.assertIn("interleav", str(caught.exception).lower())

    def test_a_missing_interleaving_flag_is_refused(self):
        document = _synthetic_document()
        del document["continuation"]["samples_interleaved"]
        with self.assertRaises(CalibrationError):
            load_continuation_samples(document)

    def test_a_recorded_percentile_that_disagrees_with_the_samples_is_rejected(self):
        for field in ("p50_ns", "p95_ns"):
            with self.subTest(field=field):
                document = _synthetic_document()
                document["continuation"]["points"][0][field] += 1
                with self.assertRaises(CalibrationError) as caught:
                    load_continuation_samples(document)
                self.assertIn(field, str(caught.exception))

    def test_a_count_that_disagrees_with_the_samples_is_rejected(self):
        document = _synthetic_document()
        document["continuation"]["points"][0]["count"] += 1
        with self.assertRaises(CalibrationError) as caught:
            load_continuation_samples(document)
        self.assertIn("count", str(caught.exception).lower())

    def test_a_missing_continuation_block_is_rejected(self):
        with self.assertRaises(CalibrationError):
            load_continuation_samples({"identity": {}})

    def test_a_non_positive_sample_is_rejected(self):
        document = _synthetic_document()
        document["continuation"]["points"][0]["samples_ns"][0] = 0
        with self.assertRaises(CalibrationError):
            load_continuation_samples(document)

    def test_the_recorded_crossover_is_cross_checked_when_present(self):
        """A recorded `append_wins_up_to` that the p95 rule contradicts."""
        document = _synthetic_document()
        document["continuation"]["crossover"] = {
            "512": {"append_wins_up_to": 128},
            "2048": {"append_wins_up_to": 2},
        }
        loaded = load_continuation_samples(document)
        selection = select_threshold_detailed(loaded.samples())
        diagnostics = loaded.crossover_disagreements(selection)
        self.assertTrue(diagnostics)
        self.assertIn("512", " ".join(diagnostics))

    def test_an_agreeing_recorded_crossover_produces_no_diagnostic(self):
        document = _synthetic_document()
        document["continuation"]["crossover"] = {
            "512": {"append_wins_up_to": 2},
            "2048": {"append_wins_up_to": 2},
        }
        loaded = load_continuation_samples(document)
        selection = select_threshold_detailed(loaded.samples())
        self.assertEqual(loaded.crossover_disagreements(selection), ())


class MinIdentityTest(unittest.TestCase):
    """`min` over per-history ceilings is an UPPER BOUND, not the rule.

    Section 10.7 selects the largest suffix in the INTERSECTION of the
    per-history winner sets. `min` of the ceilings equals that only when each
    winner set is downward-closed, and a crossover record stores edges rather
    than winner sets, so it cannot attest that. These tests pin the gap so a
    later simplification back to `min` cannot pass silently.
    """

    def test_min_of_ceilings_can_exceed_what_the_rule_selects(self):
        """W512 = {1, 8}, W2048 = {1, 2, 4}: rule gives 1, `min` gives 4."""
        samples = {
            1: _per_history(h512=(1.0, 5.0), h2048=(1.0, 5.0)),
            2: _per_history(h512=(9.0, 5.0), h2048=(1.0, 5.0)),
            4: _per_history(h512=(9.0, 5.0), h2048=(1.0, 5.0)),
            8: _per_history(h512=(1.0, 5.0), h2048=(9.0, 5.0)),
        }
        selection = select_threshold_detailed(samples)
        # The intersection is {1}, which IS a prefix, so the Section 10.7
        # monotonicity assertion passes and the rule returns a threshold.
        self.assertEqual(selection.verdict, "selected")
        self.assertEqual(selection.winners, (1,))
        self.assertEqual(selection.threshold, 1)
        # Ceilings are 8 and 4; `min` would have said 4.
        ceilings = {512: 8, 2048: 4}
        self.assertEqual(min(ceilings.values()), 4)
        self.assertNotEqual(selection.threshold, min(ceilings.values()))

    def test_the_loader_reports_a_history_whose_winners_are_not_closed(self):
        document = _synthetic_document(
            suffixes=(1, 2, 4, 8, 32, 128, 256),
            reprefill_ns={512: 500_000_000, 2048: 500_000_000},
        )
        # Make append LOSE at 512/2 and keep winning at 512/4, so history 512's
        # winner set is {1, 4, 8} -- not downward-closed.
        for point in document["continuation"]["points"]:
            if (
                point["history_rows"] == 512
                and point["suffix"] == 2
                and point["route"] == "append"
            ):
                point["samples_ns"] = [900_000_000 + i for i in range(5)]
                point["count"] = 5
                point["p50_ns"] = 900_000_002
                point["p95_ns"] = 900_000_004
        loaded = load_continuation_samples(document)
        self.assertEqual(loaded.winners_by_history()[512], (1, 4, 8))
        unclosed = loaded.unclosed_histories()
        self.assertTrue(unclosed)
        self.assertIn("512", " ".join(unclosed))
        self.assertNotIn("2048", " ".join(unclosed))

    def test_closed_winner_sets_report_nothing(self):
        loaded = load_continuation_samples(_synthetic_document())
        self.assertEqual(loaded.unclosed_histories(), ())

    def test_the_committed_measurement_has_closed_winner_sets(self):
        loaded = load_continuation_samples(
            json.loads(_BASELINE.read_text(encoding="utf-8"))
        )
        self.assertEqual(loaded.unclosed_histories(), ())


class UnclosedWinnersConsumersTest(unittest.TestCase):
    """The three things that CONSUME `unclosed_histories()`.

    The function itself was tested when it was written; its consumers were not,
    and a review pointed out that the write-up claimed otherwise. Each of these
    is a separate place the finding has to survive to: the exit code, the
    rendered document, and the machine-readable summary. A finding that reaches
    only one of them is the failure mode this whole branch keeps hitting.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = pathlib.Path(self._tmp.name)
        self.baseline = self.directory / "baseline.json"
        self.baseline.write_text(
            json.dumps(_unclosed_document()), encoding="utf-8"
        )
        self.header = self.directory / "tuning.hpp"
        self.header.write_text(_BARE_HEADER, encoding="utf-8")
        self.document = self.directory / "results.md"
        self.document.write_text("# heading\n", encoding="utf-8")
        self.summary = self.directory / "summary.json"

    def _run(self, extra=()):
        return main(
            [
                "--baseline",
                str(self.baseline),
                "--crossover-history",
                str(_HISTORY),
                "--header",
                str(self.header),
                "--document",
                str(self.document),
                "--json",
                str(self.summary),
                *extra,
            ]
        )

    def test_it_reaches_the_exit_code(self):
        self.assertNotEqual(self._run(), 0)

    def test_it_refuses_to_write_the_header_or_the_document(self):
        self._run()
        self.assertEqual(self.header.read_text(encoding="utf-8"), _BARE_HEADER)
        self.assertEqual(self.document.read_text(encoding="utf-8"), "# heading\n")

    def test_it_reaches_the_json_summary(self):
        self._run()
        payload = json.loads(self.summary.read_text(encoding="utf-8"))
        self.assertFalse(payload["winners_downward_closed"])
        self.assertTrue(
            any("not a prefix" in line for line in payload["diagnostics"]),
            payload["diagnostics"],
        )

    def test_it_reaches_the_rendered_document(self):
        loaded = load_continuation_samples(_unclosed_document())
        selection = select_threshold_detailed(loaded.samples())
        agreement = history_agreement(
            json.loads(_HISTORY.read_text(encoding="utf-8")), selection.threshold
        )
        section = build_document_section(loaded, selection, agreement)
        self.assertIn("not all downward-closed", section)
        self.assertIn("won again at a longer suffix", section)
        self.assertNotIn("ARE downward-closed", section)

    def test_a_closed_record_takes_the_other_branch_everywhere(self):
        """Otherwise the four assertions above could pass for every input."""
        self.baseline.write_text(
            json.dumps(_synthetic_document()), encoding="utf-8"
        )
        self.assertEqual(self._run(), 0)
        self.assertNotEqual(self.header.read_text(encoding="utf-8"), _BARE_HEADER)
        payload = json.loads(self.summary.read_text(encoding="utf-8"))
        self.assertTrue(payload["winners_downward_closed"])
        self.assertIn(
            "ARE downward-closed", self.document.read_text(encoding="utf-8")
        )


class ConcessionTest(unittest.TestCase):
    """What the single constant gives up, and which member of it is worst.

    The penalty at a conceded point is `reprefill - append`. Append cost grows
    with suffix length and re-prefill cost does not, so within a conceded band
    the NARROWEST suffix is the most expensive and the widest is the cheapest.
    Naming the widest -- the obvious thing to do, and the thing the first
    version of this did -- understates the cost.
    """

    def _conceded_document(self):
        # Append 50 ms/token; re-prefill 500 ms at 512 and 1,700 ms at 2048.
        # Winners: {1,2,4,8} at 512, {1,2,4,8,32} at 2048. Threshold 8; the
        # conceded band is 2048 at suffix 32 only, so widen it by also making
        # 512 lose earlier.
        return _synthetic_document(
            suffixes=(1, 2, 4, 8, 32, 128, 256),
            reprefill_ns={512: 300_000_000, 2048: 1_700_000_000},
        )

    def test_the_worst_conceded_point_is_the_narrowest_not_the_widest(self):
        loaded = load_continuation_samples(self._conceded_document())
        selection = select_threshold_detailed(loaded.samples())
        # 512 re-prefill is 300 ms, so append wins only at 1, 2, 4.
        self.assertEqual(loaded.winners_by_history()[512], (1, 2, 4))
        self.assertEqual(loaded.winners_by_history()[2048], (1, 2, 4, 8, 32))
        self.assertEqual(selection.threshold, 4)

        rows = conceded_points(loaded, selection.threshold)
        self.assertEqual([(row[0], row[1]) for row in rows], [(2048, 8), (2048, 32)])
        # Ordered worst-first, and the worst is the NARROWER suffix.
        self.assertGreater(rows[0][3], rows[1][3])
        self.assertLess(rows[0][1], rows[1][1])
        # The widest conceded suffix would have been the cheaper quote.
        widest = max(rows, key=lambda row: row[1])
        self.assertLess(widest[3], rows[0][3])

    def test_the_section_quotes_the_worst_case_not_the_widest_suffix(self):
        loaded = load_continuation_samples(self._conceded_document())
        selection = select_threshold_detailed(loaded.samples())
        rows = conceded_points(loaded, selection.threshold)
        agreement = history_agreement(
            [
                {
                    "utc": "2026-09-02T15:34:01Z",
                    "samples_interleaved": True,
                    "edges": {"512": [4, 8], "2048": [32, 128]},
                }
            ],
            selection.threshold,
        )
        section = build_document_section(loaded, selection, agreement)
        self.assertIn(f"suffix {rows[0][1]} — {rows[0][3]:,.1f}x slower", section)
        # Every conceded point appears, not just the worst.
        for _, suffix, _, _ in rows:
            self.assertIn(f"| 2048 | {suffix} |", section)

    def test_no_concession_when_nothing_is_given_up(self):
        loaded = load_continuation_samples(_synthetic_document())
        selection = select_threshold_detailed(loaded.samples())
        self.assertEqual(conceded_points(loaded, selection.threshold), ())

    def test_the_ordering_is_by_cost_across_histories_not_by_suffix(self):
        """The one case a single-history fixture cannot see.

        Within one history, slowdown-descending and suffix-ascending coincide,
        so every other test here would still pass if the sort key reverted to
        the suffix. With concessions at TWO histories they diverge: ordered by
        cost the band is [(512, 8), (2048, 2), (2048, 4)] and ordered by suffix
        it is [(2048, 2), (2048, 4), (512, 8)], which disagree on the very
        first element -- the one the document quotes as the worst case.
        """
        loaded = load_continuation_samples(_unclosed_document())
        selection = select_threshold_detailed(loaded.samples())
        self.assertEqual(selection.threshold, 1)

        rows = conceded_points(loaded, selection.threshold)
        self.assertEqual(
            [(row[0], row[1]) for row in rows],
            [(512, 8), (2048, 2), (2048, 4)],
        )
        # Both histories are represented, which is what makes the orderings
        # distinguishable at all.
        self.assertEqual({row[0] for row in rows}, {512, 2048})

        by_suffix = sorted(rows, key=lambda row: (row[1], row[0]))
        self.assertNotEqual(rows, by_suffix)
        self.assertNotEqual(rows[0], by_suffix[0])
        # And not the narrowest suffix overall either: a naive "min suffix"
        # fix would pick (2048, 2), which is 2.0x rather than 50.0x.
        self.assertEqual(min(rows, key=lambda row: row[1])[:2], (2048, 2))
        self.assertGreater(rows[0][3], 40.0)

    def test_the_committed_measurement_concedes_2048_at_8_worst(self):
        loaded = load_continuation_samples(
            json.loads(_BASELINE.read_text(encoding="utf-8"))
        )
        rows = conceded_points(loaded, 4)
        self.assertEqual([(row[0], row[1]) for row in rows], [(2048, 8), (2048, 12)])
        self.assertAlmostEqual(rows[0][3], 2.995, places=2)
        self.assertAlmostEqual(rows[1][3], 1.961, places=2)


class HistoryAgreementTest(unittest.TestCase):
    """Whether earlier interleaved runs PERMIT the constant.

    Not whether they would have selected it: see `MinIdentityTest` for why the
    stronger claim is not available from a crossover record.
    """

    @staticmethod
    def _record(interleaved, edges, utc="2026-09-02T15:34:01Z", note=None):
        record = {
            "utc": utc,
            "samples_interleaved": interleaved,
            "edges": {str(k): list(v) for k, v in edges.items()},
        }
        if note:
            record["note"] = note
        return record

    def test_interleaved_runs_that_permit_it_produce_no_diagnostic(self):
        records = [
            self._record(True, {512: (4, 8), 2048: (12, 16)}),
            self._record(True, {512: (4, 12), 2048: (12, 64)}),
        ]
        report = history_agreement(records, 4)
        self.assertTrue(report.comparable)
        self.assertTrue(report.permits)
        self.assertEqual(report.diagnostics, ())
        self.assertEqual(
            [row.upper_bound for row in report.rows if row.comparable], [4, 4]
        )
        self.assertEqual(len(report.tight), 2)

    def test_a_run_that_contradicts_the_threshold_is_a_diagnostic(self):
        records = [
            self._record(True, {512: (4, 8), 2048: (12, 16)}),
            self._record(True, {512: (2, 12), 2048: (12, 64)}),
        ]
        report = history_agreement(records, 4)
        self.assertTrue(report.comparable)
        self.assertFalse(report.permits)
        self.assertTrue(report.diagnostics)
        self.assertIn("bounds it at 2", " ".join(report.diagnostics))

    def test_a_run_whose_bound_is_higher_permits_without_confirming(self):
        """8 > 4: that run cannot contradict 4, and cannot vouch for it."""
        records = [self._record(True, {512: (8, 12), 2048: (12, 64)})]
        report = history_agreement(records, 4)
        self.assertTrue(report.permits)
        self.assertEqual(report.diagnostics, ())
        self.assertEqual(report.tight, ())
        self.assertEqual(report.rows[0].upper_bound, 8)
        self.assertFalse(report.rows[0].tight)

    def test_non_interleaved_runs_are_excluded_from_the_verdict(self):
        records = [
            self._record(None, {512: (2, None), 2048: (2, None)}, note="retracted"),
            self._record(True, {512: (4, 8), 2048: (12, 16)}),
        ]
        report = history_agreement(records, 4)
        self.assertTrue(report.permits)
        self.assertEqual(sum(1 for row in report.rows if not row.comparable), 1)

    def test_a_non_interleaved_run_cannot_contradict_the_threshold(self):
        """Its bound is 2, below 4 -- and it must still not fail the check."""
        records = [
            self._record(None, {512: (2, None), 2048: (2, None)}),
            self._record(True, {512: (4, 8), 2048: (12, 16)}),
        ]
        report = history_agreement(records, 4)
        self.assertEqual(report.rows[0].upper_bound, 2)
        self.assertFalse(report.rows[0].permits)
        self.assertTrue(report.permits)
        self.assertEqual(report.diagnostics, ())

    def test_only_non_interleaved_runs_means_nothing_is_comparable(self):
        records = [self._record(False, {512: (2, None), 2048: (2, None)})]
        report = history_agreement(records, 4)
        self.assertFalse(report.comparable)
        self.assertFalse(report.permits)
        self.assertTrue(report.diagnostics)

    def test_a_null_lower_edge_bounds_at_zero(self):
        records = [self._record(True, {512: (None, None), 2048: (12, 16)})]
        report = history_agreement(records, 0)
        self.assertTrue(report.permits)
        self.assertEqual(report.rows[0].upper_bound, 0)
        self.assertTrue(report.rows[0].tight)

    def test_a_null_lower_edge_contradicts_a_positive_threshold(self):
        records = [self._record(True, {512: (None, None), 2048: (12, 16)})]
        report = history_agreement(records, 4)
        self.assertFalse(report.permits)
        self.assertTrue(report.diagnostics)

    def test_an_empty_history_file_means_nothing_is_comparable(self):
        report = history_agreement([], 4)
        self.assertFalse(report.comparable)
        self.assertTrue(report.diagnostics)

    def test_a_record_missing_a_required_history_edge_is_rejected(self):
        records = [self._record(True, {512: (4, 8)})]
        with self.assertRaises(CalibrationError):
            history_agreement(records, 4)


_BARE_HEADER = """#pragma once

#include <cstdint>

namespace flm::phi4 {

inline constexpr std::uint32_t kContinuationAppendThreshold = 0;

}  // namespace flm::phi4
"""


class HeaderWriterTest(unittest.TestCase):
    """Bootstrap, regenerate, and refuse -- and stay byte-identical."""

    _PROVENANCE = ("source: phi4_aie4_baseline.json", "measured: 2026-09-02T16:32:18Z")

    def test_bootstrapping_a_bare_declaration_wraps_it_in_markers(self):
        written = apply_threshold_to_header(_BARE_HEADER, 4, self._PROVENANCE)
        self.assertIn(
            "inline constexpr std::uint32_t kContinuationAppendThreshold = 4;",
            written,
        )
        self.assertIn("BEGIN generated", written)
        self.assertIn("END generated", written)
        self.assertIn("source: phi4_aie4_baseline.json", written)

    def test_regenerating_is_idempotent(self):
        once = apply_threshold_to_header(_BARE_HEADER, 4, self._PROVENANCE)
        twice = apply_threshold_to_header(once, 4, self._PROVENANCE)
        self.assertEqual(once, twice)

    def test_regenerating_replaces_the_previous_block_rather_than_nesting(self):
        four = apply_threshold_to_header(_BARE_HEADER, 4, self._PROVENANCE)
        twelve = apply_threshold_to_header(four, 12, self._PROVENANCE)
        self.assertEqual(twelve.count("BEGIN generated"), 1)
        self.assertNotIn("= 4;", twelve)
        self.assertIn(
            "inline constexpr std::uint32_t kContinuationAppendThreshold = 12;",
            twelve,
        )

    def test_the_surrounding_hand_written_code_survives(self):
        source = _HEADER.read_text(encoding="utf-8")
        written = apply_threshold_to_header(source, 4, self._PROVENANCE)
        for kept in (
            "SelectContinuationRoute",
            "ContinuationRouteName",
            "enum class ForcedContinuationRoute",
        ):
            self.assertIn(kept, written)

    def test_a_header_without_the_declaration_is_refused(self):
        with self.assertRaises(CalibrationError):
            apply_threshold_to_header("#pragma once\n", 4, self._PROVENANCE)

    def test_two_declarations_are_refused(self):
        doubled = _BARE_HEADER + (
            "inline constexpr std::uint32_t kContinuationAppendThreshold = 1;\n"
        )
        with self.assertRaises(CalibrationError):
            apply_threshold_to_header(doubled, 4, self._PROVENANCE)

    def test_a_negative_threshold_is_refused(self):
        with self.assertRaises(CalibrationError):
            apply_threshold_to_header(_BARE_HEADER, -1, self._PROVENANCE)


class DocumentSectionTest(unittest.TestCase):
    """The published evidence, and its idempotence."""

    def _section_inputs(self):
        loaded = load_continuation_samples(_synthetic_document())
        selection = select_threshold_detailed(loaded.samples())
        records = [
            {
                "utc": "2026-09-02T15:34:01Z",
                "samples_interleaved": True,
                "edges": {"512": [2, 32], "2048": [2, 32]},
            }
        ]
        return loaded, selection, history_agreement(records, selection.threshold)

    def test_the_section_carries_the_threshold_and_the_identity(self):
        loaded, selection, agreement = self._section_inputs()
        section = build_document_section(loaded, selection, agreement)
        self.assertIn("xcomedusad-43", section)
        self.assertIn("kContinuationAppendThreshold", section)
        self.assertIn(str(selection.threshold), section)
        self.assertIn("2026-09-02T16:32:18Z", section)

    def test_the_section_names_what_the_single_constant_gives_up(self):
        """A longer history whose winners run past the chosen constant.

        Re-prefill costs 500 ms at history 512 and 1,700 ms at history 2048, so
        append wins up to 8 at 512 and up to 32 at 2048. The conjunction gives
        8, and suffix 32 at history 2048 is what that constant gives up.
        """
        document = _synthetic_document(
            suffixes=(1, 2, 4, 8, 32, 128, 256),
            reprefill_ns={512: 500_000_000, 2048: 1_700_000_000},
        )
        loaded = load_continuation_samples(document)
        selection = select_threshold_detailed(loaded.samples())
        self.assertEqual(selection.threshold, 8)
        self.assertEqual(
            loaded.winners_by_history(),
            {512: (1, 2, 4, 8), 2048: (1, 2, 4, 8, 32)},
        )
        agreement = history_agreement(
            [
                {
                    "utc": "2026-09-02T15:34:01Z",
                    "samples_interleaved": True,
                    "edges": {"512": [8, 32], "2048": [16, 32]},
                }
            ],
            selection.threshold,
        )
        section = build_document_section(loaded, selection, agreement)
        self.assertIn("gives up", section.lower())
        self.assertIn("2048", section)

    def test_the_baseline_block_is_left_alone(self):
        """Two generators write into `phi4_results.md`; neither owns the file.

        `report_phi4_corelib_baseline.py` replaces everything between its
        `phi4-aie4-baseline` markers and this script replaces everything
        between its own. If the sections ever nested or overlapped, one tool
        would silently eat the other's evidence the next time it ran.
        """
        loaded, selection, agreement = self._section_inputs()
        section = build_document_section(loaded, selection, agreement)
        original = _DOCUMENT.read_text(encoding="utf-8")
        baseline_begin = original.index("<!-- BEGIN phi4-aie4-baseline -->")
        baseline_end = original.index("<!-- END phi4-aie4-baseline -->")
        baseline_block = original[
            baseline_begin : baseline_end + len("<!-- END phi4-aie4-baseline -->")
        ]

        rewritten = apply_section_to_document(original, section)
        self.assertIn(baseline_block, rewritten)
        self.assertEqual(rewritten.count("<!-- BEGIN phi4-aie4-baseline -->"), 1)
        self.assertEqual(
            rewritten.count("<!-- BEGIN phi4-continuation-threshold -->"), 1
        )
        self.assertGreater(
            rewritten.index("<!-- BEGIN phi4-continuation-threshold -->"),
            rewritten.index("<!-- END phi4-aie4-baseline -->"),
        )
        self.assertEqual(rewritten, apply_section_to_document(rewritten, section))

    def test_appending_to_a_document_without_the_section_is_idempotent(self):
        loaded, selection, agreement = self._section_inputs()
        section = build_document_section(loaded, selection, agreement)
        once = apply_section_to_document("# heading\n", section)
        self.assertEqual(once, apply_section_to_document(once, section))

    def test_an_unterminated_section_is_refused(self):
        loaded, selection, agreement = self._section_inputs()
        section = build_document_section(loaded, selection, agreement)
        with self.assertRaises(CalibrationError):
            apply_section_to_document(
                "# heading\n<!-- BEGIN phi4-continuation-threshold -->\n", section
            )

    def test_the_section_is_deterministic(self):
        loaded, selection, agreement = self._section_inputs()
        self.assertEqual(
            build_document_section(loaded, selection, agreement),
            build_document_section(loaded, selection, agreement),
        )

    @staticmethod
    def _records(*edges_by_run, interleaved=True):
        return [
            {
                "utc": f"2026-09-0{index + 2}T00:00:00Z",
                "samples_interleaved": interleaved,
                "edges": {str(h): [low, None] for h, low in edges.items()},
            }
            for index, edges in enumerate(edges_by_run)
        ]

    def test_a_moved_lower_edge_is_never_called_stable(self):
        """The paragraph that went false, at the branch that made it false.

        It was a constant string asserting the lower edge was stable, so it
        could not respond to a third run recording 24 where the first two
        recorded 12. This is that exact shape.
        """
        agreement = history_agreement(
            self._records(
                {512: 4, 2048: 12}, {512: 4, 2048: 12}, {512: 4, 2048: 24}
            ),
            4,
        )
        self.assertEqual(recorded_lower_edges(agreement), {512: (4,), 2048: (12, 24)})
        text = lower_edge_confidence(Selection(4, "selected", (1, 2, 4), (), ()), agreement)
        self.assertIn("12, 24 at history 2048", text)
        self.assertIn("NOT a measured constant", text)
        self.assertNotIn("stable across runs", text)
        # And the inequality, which does still hold, is what carries it.
        self.assertIn("at or below EVERY recorded lower edge", text)

    def test_an_unmoved_lower_edge_is_reported_without_overclaiming(self):
        agreement = history_agreement(
            self._records({512: 4, 2048: 12}, {512: 4, 2048: 12}), 4
        )
        self.assertEqual(recorded_lower_edges(agreement), {512: (4,), 2048: (12,)})
        text = lower_edge_confidence(Selection(4, "selected", (1, 2, 4), (), ()), agreement)
        self.assertIn("has not moved between runs", text)
        # Two runs agreeing is not a guarantee, and the sentence has to say so
        # -- this is the wording whose absence let one run falsify the claim.
        self.assertIn("not a guarantee about the next one", text)

    def test_a_non_interleaved_run_does_not_vote_on_the_edge(self):
        agreement = history_agreement(
            self._records({512: 4, 2048: 24}, interleaved=False), 4
        )
        self.assertEqual(recorded_lower_edges(agreement), {})
        text = lower_edge_confidence(Selection(4, "selected", (1, 2, 4), (), ()), agreement)
        self.assertIn("no interleaved run is on record", text)

    def test_a_contradicted_threshold_does_not_claim_the_inequality(self):
        agreement = history_agreement(self._records({512: 4, 2048: 12}), 8)
        self.assertFalse(agreement.permits)
        text = lower_edge_confidence(Selection(8, "selected", (), (), ()), agreement)
        self.assertIn("is NOT at or below every recorded lower edge", text)

    def test_the_rendered_section_prints_the_lower_edges_it_reasons_from(self):
        """The claim and the table it sits beside read the same field.

        They used to be independent: the table printed `min(lowers)` and the
        prose asserted a stability result from nowhere, so nothing made them
        agree. Against the committed history the section must show the run
        that moved the edge, in the table AND in the sentence.
        """
        loaded = load_continuation_samples(
            json.loads(_BASELINE.read_text(encoding="utf-8"))
        )
        selection = select_threshold_detailed(loaded.samples())
        agreement = history_agreement(
            json.loads(_HISTORY.read_text(encoding="utf-8")), selection.threshold
        )
        section = build_document_section(loaded, selection, agreement)
        self.assertIn("| 2026-09-03T08:20:08Z | yes | 512: 4, 2048: 24 |", section)
        self.assertIn("12, 24 at history 2048", section)
        self.assertIn("NOT a measured constant", section)

    def test_the_section_says_so_when_no_length_wins(self):
        document = _synthetic_document(reprefill_ns=1_000_000)
        loaded = load_continuation_samples(document)
        selection = select_threshold_detailed(loaded.samples())
        self.assertEqual(selection.threshold, 0)
        agreement = history_agreement(
            [
                {
                    "utc": "2026-09-02T15:34:01Z",
                    "samples_interleaved": True,
                    "edges": {"512": [None, 1], "2048": [None, 1]},
                }
            ],
            0,
        )
        section = build_document_section(loaded, selection, agreement)
        self.assertIn("every prefix hit re-prefills", section)


class CommandLineTest(unittest.TestCase):
    """Exit codes, `--check`, and the skipped-work rule."""

    def _run(self, extra, header_text=None, document_text=None):
        directory = pathlib.Path(self._tmp.name)
        header = directory / "tuning.hpp"
        header.write_text(
            header_text
            if header_text is not None
            else _HEADER.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        document = directory / "results.md"
        document.write_text(
            document_text
            if document_text is not None
            else _DOCUMENT.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        argv = [
            "--baseline",
            str(_BASELINE),
            "--crossover-history",
            str(_HISTORY),
            "--header",
            str(header),
            "--document",
            str(document),
            *extra,
        ]
        return main(argv), header, document

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_writing_then_checking_succeeds_and_is_stable(self):
        code, header, document = self._run([])
        self.assertEqual(code, 0)
        first_header = header.read_text(encoding="utf-8")
        first_document = document.read_text(encoding="utf-8")

        self.assertEqual(
            main(
                [
                    "--baseline",
                    str(_BASELINE),
                    "--crossover-history",
                    str(_HISTORY),
                    "--header",
                    str(header),
                    "--document",
                    str(document),
                    "--check",
                ]
            ),
            0,
        )
        code, _, _ = self._run([], first_header, first_document)
        self.assertEqual(code, 0)
        self.assertEqual(header.read_text(encoding="utf-8"), first_header)
        self.assertEqual(document.read_text(encoding="utf-8"), first_document)

    def test_check_fails_when_the_header_is_stale(self):
        stale = _BARE_HEADER.replace("= 0;", "= 7;")
        code, _, _ = self._run(["--check"], header_text=stale)
        self.assertNotEqual(code, 0)

    def test_check_fails_when_the_document_section_is_missing(self):
        code, _, _ = self._run(["--check"], document_text="# nothing here\n")
        self.assertNotEqual(code, 0)

    def test_a_malformed_baseline_exits_non_zero(self):
        directory = pathlib.Path(self._tmp.name)
        broken = directory / "broken.json"
        document = json.loads(_BASELINE.read_text(encoding="utf-8"))
        document["continuation"]["samples_interleaved"] = False
        broken.write_text(json.dumps(document), encoding="utf-8")
        header = directory / "tuning.hpp"
        header.write_text(_BARE_HEADER, encoding="utf-8")
        code = main(
            [
                "--baseline",
                str(broken),
                "--crossover-history",
                str(_HISTORY),
                "--header",
                str(header),
                "--document",
                str(directory / "out.md"),
            ]
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(header.read_text(encoding="utf-8"), _BARE_HEADER)

    def test_a_missing_crossover_history_exits_non_zero(self):
        directory = pathlib.Path(self._tmp.name)
        code = main(
            [
                "--baseline",
                str(_BASELINE),
                "--crossover-history",
                str(directory / "absent.json"),
                "--header",
                str(directory / "tuning.hpp"),
                "--document",
                str(directory / "out.md"),
            ]
        )
        self.assertNotEqual(code, 0)

    def test_the_json_summary_records_the_threshold_and_the_verdict(self):
        directory = pathlib.Path(self._tmp.name)
        summary = directory / "summary.json"
        code, _, _ = self._run(["--json", str(summary)])
        self.assertEqual(code, 0)
        payload = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual(payload["verdict"], "selected")
        self.assertEqual(payload["threshold"], 4)
        self.assertTrue(payload["history_agreement"]["permits"])
        self.assertTrue(payload["winners_downward_closed"])


def _markdown_blocks(text):
    """Split markdown into `(collapsed_text, is_quoted)` blocks.

    A block is a run of consecutive non-blank lines, whitespace-collapsed so a
    claim broken across a line wrap is still found -- checking line by line
    would miss exactly the occurrences a text reflow creates.

    `is_quoted` is true when every line of the block is a blockquote, which is
    how a retracted claim is legitimately cited: inside a marked correction
    that quotes the old wording in order to say it was wrong.
    """
    for raw in re.split(r"\n\s*\n", text):
        lines = [line for line in raw.splitlines() if line.strip()]
        if not lines:
            continue
        quoted = all(line.lstrip().startswith(">") for line in lines)
        yield " ".join(raw.split()), quoted


class RetractedClaimsTest(unittest.TestCase):
    """Retractions must reach the RENDERED artifacts, not just the code.

    Four times on this branch a claim was retracted in the source while the
    rendered document went on asserting it, because nothing read the rendered
    output back. A fifth time it survived in the task REPORT -- the very file
    documenting the retraction -- because the guard read the document and not
    the report. So both are read back here.

    The claims are matched by TEXT, not by line number: a line number stops
    meaning anything the moment the prose moves, and would then pass vacuously.
    """

    # Each entry is (substring, why it is wrong). The substrings are the exact
    # retracted wording, not a paraphrase and not a keyword: "would have
    # selected", for instance, appears LEGITIMATELY in the corrected prose,
    # which says the bound is not what that run would have selected. A matcher
    # loose enough to catch the retracted claim by keyword would forbid its own
    # correction. Likewise "at the widest conceded suffix" rather than "widest
    # conceded suffix", because the correction has to be able to say that the
    # widest conceded suffix is the cheapest one.
    _RETRACTED = (
        (
            "the constant is the smaller of those ceilings",
            "Section 10.7 takes the largest suffix in the INTERSECTION of the "
            "per-history winner sets. That coincides with the smaller ceiling "
            "only when the sets are downward-closed, which is a property of "
            "the data and not of the rule.",
        ),
        (
            "recomputable from every recorded run",
            "a crossover record stores edges and never winner sets, so it "
            "BOUNDS what Section 10.7 would have selected and cannot "
            "reproduce it.",
        ),
        (
            "at the widest conceded suffix",
            "the widest suffix in a conceded band is its CHEAPEST member. The "
            "penalty is reprefill - append, append grows with suffix length "
            "and re-prefill does not, so the cost is read off the narrow end.",
        ),
        # Task 15's re-measurement (2026-09-03T08:20:08Z, committed at
        # `phi4_aie4_baseline_task15_rerun.json` and now the fifth record in
        # `phi4_aie4_crossover_history.json`) measured the lower edge at
        # history 2048 to be 24. Two claims in the baseline block asserted the
        # opposite in the present tense, and the correction reached only the
        # hand-written section 200 lines below them.
        (
            "the lower edge has been 12 at history 2048",
            "the third interleaved run on record measured it at 24. Across "
            "the three interleaved runs in the committed crossover history "
            "the lower edge at history 2048 has taken 12 and 24, so it is not "
            "a measured constant.",
        ),
        (
            "upper edge is not stable and its lower edge is",
            "neither edge is stable. `crossover_edge_stability` over the "
            "committed crossover history reports lower_is_stable false once "
            "the 2026-09-03 run is included, and its own narrative for that "
            "case reads \"Neither edge of the bracket is stable between "
            "runs.\"",
        ),
        # The same falsification, in the two paragraphs the CALIBRATOR emitted.
        # Both were constant strings: they could not change when the data did,
        # and one of them said "three times" above a table of two comparable
        # runs. `build_document_section` now derives both from
        # `HistoryRow.lower_edges`, so the claim and the table it sits beside
        # cannot disagree. Attributed past-tense wording about what Task 13
        # measured is NOT retracted -- only the unattributed present-tense form
        # that says the edges are stable.
        (
            "found the bracket's LOWER edge stable across runs",
            "the lower edge at history 2048 took 12 on the first two "
            "interleaved runs and 24 on the third. The renderer now prints "
            "the per-run lower edges and derives whether they held still.",
        ),
        (
            "measured the lower edges to be stable",
            "they are not stable: history 2048 recorded 12 and 24 across the "
            "three interleaved runs. What the bound actually relies on is "
            "that it touches only the lower edges, which is a property of the "
            "construction and is what the paragraph says now.",
        ),
    )

    @staticmethod
    def _collapse(text):
        """Whitespace-collapse, so a line wrap cannot hide a claim.

        The report matcher collapsed and the DOCUMENT matcher did not, which
        made the strict check score zero on any wrapped occurrence. It was
        harmless only because `build_document_section` emits one-line
        paragraphs -- and it went vacuous the moment the check was widened to
        the whole committed document, which contains hand-wrapped prose.
        """
        return " ".join(text.split())

    @classmethod
    def _asserts(cls, claim, text):
        """Does `text` contain `claim` as a phrase, not inside a longer word?

        Plain `in` was wrong in a way that took a failing run to see: "at the
        widest conceded suffix" is a substring of "th-AT THE widest conceded
        suffix", so the matcher flagged the very sentence explaining that the
        widest conceded suffix is the cheapest. Third variant of the same trap
        -- a matcher loose enough to catch the claim also catches its
        correction -- so the left edge is anchored to a word boundary.

        Both sides are whitespace-collapsed. Without that, a claim broken over
        a line wrap scored zero here while the report matcher (which reads
        collapsed blocks) found it -- see
        `test_the_strict_matcher_also_finds_a_claim_split_across_a_line_wrap`,
        which fails against the uncollapsed version.
        """
        return (
            re.search(
                r"(?<!\w)" + re.escape(cls._collapse(claim)), cls._collapse(text)
            )
            is not None
        )

    def _violations(self, text):
        return [
            f"asserts a retracted claim ({claim!r}): {why}"
            for claim, why in self._RETRACTED
            if self._asserts(claim, text)
        ]

    def _assert_clean(self, text, where):
        violations = self._violations(text)
        self.assertEqual(violations, [], f"{where} " + "; ".join(violations))

    # Inline citation: "like this" or `like this`. Bounded so an unbalanced
    # delimiter cannot swallow the rest of a block and exempt everything after
    # it -- an exemption that grows without limit is an exemption that hides
    # what it was built to catch.
    _INLINE_QUOTE = re.compile(r"\"[^\"]{0,400}\"|`[^`]{0,400}`")

    def _collect_unquoted(self, text, blockquote_is_citation=True):
        """Retracted claims ASSERTED rather than cited.

        Prose that retracts a claim has to be able to quote it, so a claim is
        treated as cited, and allowed, when it sits inside an inline
        quotation. The inline case is not a nicety: every citation in fix
        round 3's own prose is inline, and a blockquote-only rule failed on
        the round that introduced it. Assert the claim in your own unquoted
        voice and it is still caught.

        `blockquote_is_citation` is FALSE for the published document and true
        for the task report, and the difference is not a preference. In a
        report a blockquote is how a human marks a quotation. In
        `phi4_results.md` a blockquote is a paragraph the RENDERER emits --
        `crossover_stability_narrative` returns its guidance as one -- so
        treating it as a citation exempted the generator's own voice. Measured:
        with the exemption on, the falsified sentence "the lower edge has been
        12 at history 2048" sat unflagged in the committed document because it
        was rendered behind a `>`.
        """
        violations = []
        for block, quoted in _markdown_blocks(text):
            if quoted and blockquote_is_citation:
                continue
            cited = [
                match.span() for match in self._INLINE_QUOTE.finditer(block)
            ]
            for claim, why in self._RETRACTED:
                for match in re.finditer(
                    r"(?<!\w)" + re.escape(claim), block
                ):
                    if any(
                        start <= match.start() < end for start, end in cited
                    ):
                        continue
                    violations.append(
                        f"asserts a retracted claim ({claim!r}) outside a "
                        f"quotation: {why}"
                    )
                    break
        return violations

    def _assert_clean_unless_quoted(self, text, where, blockquote_is_citation=True):
        violations = self._collect_unquoted(text, blockquote_is_citation)
        self.assertEqual(violations, [], f"{where} " + "; ".join(violations))

    def _task14_section(self):
        """Only the block this task owns.

        `phi4_results.md` also carries Task 13's baseline block. Grepping the
        whole file would let this test go red on prose Task 14 neither wrote
        nor may edit.
        """
        document = _DOCUMENT.read_text(encoding="utf-8")
        begin = document.find("<!-- BEGIN phi4-continuation-threshold -->")
        end = document.find("<!-- END phi4-continuation-threshold -->")
        self.assertNotEqual(begin, -1, "the Task 14 section is missing")
        self.assertNotEqual(end, -1, "the Task 14 section is unterminated")
        self.assertLess(begin, end)
        return document[begin:end]

    def test_the_committed_document_asserts_no_retracted_claim(self):
        """The WHOLE committed document, not one block.

        This used to read `self._task14_section()`, on the reasoning that
        Task 14 may not edit prose it does not own. The cost of that reasoning
        was I4/I9: Task 15 retracted a stability claim, the retraction reached
        a hand-written section 200 lines below the claim, and the guard built
        over three fix rounds to stop exactly that could not see it -- the
        claim lives in the Task 13 baseline block, which was out of scope.

        Scope is now the file. A retracted claim is not allowed to survive
        anywhere in the published document, whoever generated the paragraph
        it sits in. The citation rule is the report's, not the strict one,
        because the document now carries retraction prose of its own and
        prose that retracts a claim has to be able to quote it.
        """
        self._assert_clean_unless_quoted(
            _DOCUMENT.read_text(encoding="utf-8"),
            str(_DOCUMENT),
            blockquote_is_citation=False,
        )

    def test_the_committed_report_asserts_no_retracted_claim(self):
        """The file that documents the retractions is not exempt from them.

        Two retracted claims sat unmarked in it for two review rounds, in the
        sections describing the very fixes that retracted them, because the
        guard read the rendered document and stopped there.
        """
        if not _REPORT.exists():  # pragma: no cover - report lives beside the plan
            self.skipTest(f"{_REPORT} is not present in this checkout")
        self._assert_clean_unless_quoted(
            _REPORT.read_text(encoding="utf-8"), str(_REPORT)
        )

    def test_a_freshly_rendered_section_asserts_no_retracted_claim(self):
        loaded = load_continuation_samples(
            json.loads(_BASELINE.read_text(encoding="utf-8"))
        )
        selection = select_threshold_detailed(loaded.samples())
        agreement = history_agreement(
            json.loads(_HISTORY.read_text(encoding="utf-8")),
            selection.threshold,
        )
        self._assert_clean(
            build_document_section(loaded, selection, agreement),
            "a freshly rendered section",
        )

    def test_the_guard_would_notice_each_claim_coming_back(self):
        """The matcher must actually match; otherwise the two tests above
        pass for every possible document, including one that reinstates the
        claim verbatim."""
        for claim, _ in self._RETRACTED:
            with self.subTest(claim=claim):
                self.assertEqual(
                    len(self._violations(f"prose prose {claim} prose")), 1
                )
                with self.assertRaises(AssertionError):
                    self._assert_clean(
                        f"prose prose {claim} prose", "a synthetic document"
                    )

    def test_a_claim_inside_a_longer_word_is_not_a_claim(self):
        """"that the widest conceded suffix is the cheapest" is the
        CORRECTION, and it contains "at the widest conceded suffix" as a bare
        substring. The guard found this by failing on it."""
        correction = (
            "carries an inline note explaining that the widest conceded "
            "suffix is the cheapest"
        )
        self.assertEqual(self._violations(correction), [])
        # The same words with a real word boundary in front ARE the claim.
        self.assertEqual(
            len(self._violations("the p95 at the widest conceded suffix")), 1
        )

    def test_the_quotation_exemption_is_not_a_blanket_pass(self):
        """Otherwise the report check passes for any report at all."""
        claim = self._RETRACTED[0][0]
        # Asserted as the author's own prose: caught.
        self.assertEqual(
            len(self._collect_unquoted(f"Some prose. {claim}. More prose.\n")), 1
        )
        # Quoted inside a marked correction: allowed.
        self.assertEqual(
            self._collect_unquoted(
                f"> **Corrected.** It originally read {claim}.\n"
            ),
            [],
        )
        # A blockquote glued to unquoted prose does NOT launder it: the block
        # is mixed, so it is treated as the author speaking.
        self.assertEqual(
            len(self._collect_unquoted(f"Prose {claim}.\n> quoted tail\n")), 1
        )
        # Cited inline, in the author's own paragraph: allowed. Every citation
        # in fix round 3's prose takes this form, and a blockquote-only rule
        # went red on the round that introduced it.
        self.assertEqual(
            self._collect_unquoted(f'It originally read "{claim}", which is wrong.\n'),
            [],
        )
        self.assertEqual(
            self._collect_unquoted(f"It originally read `{claim}`.\n"), []
        )
        # A claim in the middle of a longer quoted span is still cited.
        self.assertEqual(
            self._collect_unquoted(f'It read "so {claim} without re-measuring".\n'),
            [],
        )

    def test_an_unbalanced_quote_cannot_exempt_the_rest_of_a_block(self):
        """A runaway quoted span would silently disable the guard downstream."""
        claim = self._RETRACTED[0][0]
        block = 'He said "' + ("filler " * 90) + f". Then {claim}.\n"
        self.assertGreater(len(block.split('"')[1]), 400)
        self.assertEqual(len(self._collect_unquoted(block)), 1)

    def test_a_claim_split_across_a_line_wrap_is_still_found(self):
        """Line-by-line matching would miss every reflowed occurrence."""
        claim = self._RETRACTED[1][0]
        head, tail = claim.split(" ", 1)
        self.assertEqual(len(self._collect_unquoted(f"Prose {head}\n{tail}.\n")), 1)

    def test_the_strict_matcher_also_finds_a_claim_split_across_a_line_wrap(self):
        """The parked defect: only the REPORT matcher collapsed whitespace.

        `_collect_unquoted` reads whitespace-collapsed blocks; `_asserts` read
        the raw text, so a claim broken over a line wrap scored ZERO under the
        strict check that guards the rendered section. It was harmless only
        because `build_document_section` emits one-line paragraphs, and it goes
        vacuous the day that generator wraps prose -- or, as here, the day the
        check is pointed at a hand-wrapped file. This case failed before the
        collapse was added to `_asserts`.
        """
        for claim, _ in self._RETRACTED:
            head, tail = claim.split(" ", 1)
            wrapped = f"prose prose {head}\n{tail} prose"
            with self.subTest(claim=claim):
                self.assertEqual(len(self._violations(wrapped)), 1)
                with self.assertRaises(AssertionError):
                    self._assert_clean(wrapped, "a wrapped synthetic document")

    def test_a_blockquote_does_not_launder_a_claim_in_the_document(self):
        """`crossover_stability_narrative` returns its guidance AS a
        blockquote, so in the published document a leading `>` is the
        renderer's voice and not a human marking a citation. Measured: with
        blockquotes exempt, the falsified "the lower edge has been 12 at
        history 2048" sat unflagged in the committed document."""
        claim = "the lower edge has been 12 at history 2048"
        self.assertIn(claim, [entry[0] for entry in self._RETRACTED])
        quoted_block = f"> **Read the lower edge as measured.** Across runs {claim}.\n"
        # Report rules: a blockquote is a citation.
        self.assertEqual(self._collect_unquoted(quoted_block), [])
        # Document rules: it is not.
        self.assertEqual(
            len(self._collect_unquoted(quoted_block, blockquote_is_citation=False)),
            1,
        )
        # An inline citation is still allowed under document rules, because
        # the document now carries its own retraction prose.
        self.assertEqual(
            self._collect_unquoted(
                f'The table concluded "{claim}", which the 2026-09-03 run '
                f"falsified.\n",
                blockquote_is_citation=False,
            ),
            [],
        )

    def test_the_corrected_wording_is_not_itself_flagged(self):
        """"would have selected" appears in the correction. A matcher that
        forbade it would make the fix unwritable."""
        corrected = (
            "The minimum of those ceilings is an upper bound on what Section "
            "10.7 would have selected from that run, and in general only an "
            "upper bound."
        )
        self.assertEqual(self._violations(corrected), [])

    def test_the_intersection_is_stated_where_the_reader_meets_it(self):
        """Removing the wrong claim is not enough; the right one must be
        there, and above the paragraph that explains it."""
        section = self._task14_section()
        # assertIn first: `str.index` on a missing phrase raises ValueError,
        # which reports as an ERROR and hides which phrase went missing.
        self.assertIn("INTERSECTION of those winner sets", section)
        self.assertIn("only when each winner set is downward-closed", section)
        self.assertLess(
            section.index("INTERSECTION of those winner sets"),
            section.index("only when each winner set is downward-closed"),
        )


class CheckedInArtefactsTest(unittest.TestCase):
    """The committed constant and document must match the committed data.

    This is the check that would catch the header, the document and the
    measurement drifting apart -- the failure mode that no unit test over
    synthetic records can see.
    """

    def test_the_committed_measurement_selects_the_committed_constant(self):
        loaded = load_continuation_samples(
            json.loads(_BASELINE.read_text(encoding="utf-8"))
        )
        selection = select_threshold_detailed(loaded.samples())
        self.assertEqual(selection.verdict, "selected")
        self.assertEqual(
            loaded.winners_by_history(),
            {512: (1, 2, 4), 2048: (1, 2, 4, 8, 12)},
        )
        self.assertEqual(selection.winners, (1, 2, 4))
        self.assertEqual(selection.threshold, 4)
        self.assertEqual(loaded.crossover_disagreements(selection), ())
        self.assertIn(
            f"kContinuationAppendThreshold = {selection.threshold};",
            _HEADER.read_text(encoding="utf-8"),
        )

    def test_the_committed_artefacts_are_already_up_to_date(self):
        self.assertEqual(
            main(
                [
                    "--baseline",
                    str(_BASELINE),
                    "--crossover-history",
                    str(_HISTORY),
                    "--header",
                    str(_HEADER),
                    "--document",
                    str(_DOCUMENT),
                    "--check",
                ]
            ),
            0,
        )

    def test_every_interleaved_run_on_record_supports_the_committed_constant(self):
        records = json.loads(_HISTORY.read_text(encoding="utf-8"))
        report = history_agreement(records, 4)
        self.assertTrue(report.comparable)
        self.assertTrue(report.permits)

    def test_the_run_that_moved_the_lower_edge_reached_the_history_file(self):
        """The falsification has to reach the file the TOOLING reads.

        Task 15 measured a third interleaved run whose lower edge at history
        2048 is 24, wrote it up in prose, and never appended it to
        `phi4_aie4_crossover_history.json` -- so `--crossover-history` kept
        feeding `history_agreement` the two runs that agree, and the calibrator
        kept rendering "2 of 2 bound it there exactly" from data that had been
        falsified. Prose is not an input to anything.

        Every figure below is derived, not typed: it is what
        `report_phi4_corelib_baseline.crossover_entry()` produces from the
        committed `phi4_aie4_baseline_task15_rerun.json`.
        """
        rerun = (
            _REPO_ROOT
            / "docs"
            / "docs"
            / "benchmarks"
            / "phi4_aie4_baseline_task15_rerun.json"
        )
        derived = crossover_entry(
            json.loads(rerun.read_text(encoding="utf-8")), source=""
        )
        records = json.loads(_HISTORY.read_text(encoding="utf-8"))
        by_utc = {record.get("utc"): record for record in records}
        self.assertIn(
            derived["utc"],
            by_utc,
            f"{rerun.name} is committed but its run is absent from "
            f"{_HISTORY.name}, so nothing that reads the history can see it",
        )
        committed = by_utc[derived["utc"]]
        for field in (
            "edges",
            "samples_interleaved",
            "suffix_grid",
            "points_decided",
            "points_total",
            "undecided_suffixes",
            "fastflow_revision",
            "corelib_dll_sha256",
        ):
            self.assertEqual(committed[field], derived[field], field)

        # And the property that makes it matter: across the interleaved runs
        # the lower edge at history 2048 is NOT a single value.
        lowers = sorted(
            {
                record["edges"]["2048"][0]
                for record in records
                if record.get("samples_interleaved")
            }
        )
        self.assertEqual(lowers, [12, 24])


if __name__ == "__main__":
    unittest.main()
