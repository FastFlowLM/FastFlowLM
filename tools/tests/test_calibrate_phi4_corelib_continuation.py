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
import tempfile
import unittest

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
    main,
    percentile_ns,
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


if __name__ == "__main__":
    unittest.main()
