"""Tests for the Phi-4 AIE4 performance/memory baseline validator.

Task 13 Steps 1, 2 and 10.

The validator's job is narrow and worth stating, because it is easy to build
the wrong thing here. Design section 4 makes performance explicitly NOT a
release blocker, and design section 15.6 says to record the numbers "without
pass/fail thresholds". So this module must never gate on a latency or a
throughput figure. What it DOES gate on:

  * every required section and identity field is present -- a baseline whose
    machine, corelib DLL hash or model hash is missing cannot be compared
    against anything later, which is the only reason the baseline exists;
  * the 128-token memory window is stable, which IS a correctness property
    (design 18.7, and design 15.4's "at least 128 decode tokens with stable
    post-warm allocation"); and
  * the `DETERM-3` bit-identity baseline states its rate, its spread and the n
    it rests on, with at least the 20 runs per route design `DETERM-3`
    requires.

That last one is the reason this file is careful about a shape that looks
harmless: a "baseline" that reports a floor without the sample size behind it.
`DETERM-2` names "a bit-identity rate that degrades from the recorded
baseline" as a failure condition, and `DETERM-3` says in as many words that a
floor derived from a handful of runs is the unmeasured number `DETERM-2`
forbids. A validator that accepted such a record would make the absence
invisible, which is worse than the absence.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools import report_phi4_corelib_baseline as report
from tools.report_phi4_corelib_baseline import (
    MIN_DETERMINISM_RUNS_PER_ROUTE,
    ROUTE_DEPENDENCE_ALPHA,
    _fisher_exact_two_sided,
    REQUIRED_IDENTITY_FIELDS,
    REQUIRED_SECTIONS,
    determinism_baseline,
    memory_is_stable,
    render_markdown,
    validate,
)


def _identity() -> dict:
    return {
        "machine": "xcomedusad-43",
        "cpu_sku": "AMD Eng Sample: 100-000001713-33_N",
        "npu_sku": "AMD XDNA(TM) NPU",
        "npu_driver_version": "32.0.20214.4161",
        "corelib_dll_path": "C:/Users/chiz/work/aie4-runtime-mirrored/"
        "ryzenai_corelib.dll",
        "corelib_dll_sha256": "a523b238" + "0" * 56,
        "corelib_version": "0.1.0",
        "corelib_source_revision": "e5258d29b5cb979d4a538994409b90ceff6e6e7a",
        "dynamic_dispatch_version": "1.2.3.4",
        "ryzen_mm_version": "1.0.0.0",
        "xrt_version": "2.19.0",
        "model_dir": "C:/models/phi-4",
        "model_sha256": "c48fd647" + "0" * 56,
        "fastflow_revision": "084e2c4d",
        "utc": "2026-09-02T12:00:00Z",
    }


def _memory(**overrides) -> dict:
    record = {
        "device_tensor_creates_after_warmup": 0,
        "weight_creates_after_warmup": 0,
        "live_corelib_object_delta": 0,
        "private_bytes_growth": 2 * 1024 * 1024,
        "private_bytes_slope_per_token": 1024,
        "peak_private_bytes": 6 * 1024**3,
        "peak_working_set_bytes": 5 * 1024**3,
        "embedding_bytes": 1229193216,
        "kv_bytes": 536870912,
        "packed_weight_bytes": 2 * 1024**3,
        "scratch_bytes": 1024**3,
        "mapped_source_bytes": 3248488448,
        "samples": [],
    }
    record.update(overrides)
    return record


def _baseline(**overrides) -> dict:
    document = {
        "identity": _identity(),
        "model_load": {
            "manifest_map_ns": 1,
            "shape_plan_ns": 90,
            "weight_pack_ns": 2,
            "device_setup_ns": 5,
            "unaccounted_ns": 2,
            "shape_plan_share": 0.9,
            "total_ns": 100,
            "weight_objects": 161,
            "device_tensors": 76,
        },
        "ttft": {
            "cold_ns": 10,
            "warm_ns": 5,
            "prompt_token_count": 12,
            "prompt_id": "phi4_tokens.prefix",
            "row_extent": 12,
            "stream_rebuilt_for_warm": False,
        },
        "prefill": {"points": [{"rows": 1, "ns": 100, "tokens_per_second": 10.0}]},
        "continuation": {
            "points": [
                {
                    "history_rows": 512,
                    "suffix": 1,
                    "route": "append",
                    "samples_ns": [1, 2, 3, 4, 5],
                    "p50_ns": 3,
                    "p95_ns": 5,
                }
            ],
            "crossover": {
                "512": {
                    "append_wins_up_to": 8,
                    "reprefill_wins_from": 12,
                    "crossover_bracket": [8, 12],
                    "bracket_is_tight": False,
                    "decisions": [],
                },
                "2048": {
                    "append_wins_up_to": 24,
                    "reprefill_wins_from": 32,
                    "crossover_bracket": [24, 32],
                    "bracket_is_tight": False,
                    "decisions": [],
                },
            },
            "decision_rule": "gap must exceed the within-route spread",
            "prefix_monotonic": True,
        },
        "decode": {
            "runs": [
                {
                    "start_context": 128,
                    "tokens": 128,
                    "tokens_per_second": 20.0,
                    "synchronizes_per_pass": 129,
                    "per_token_ns": [1] * 128,
                    "p50_ns": 1,
                    "p95_ns": 1,
                }
            ]
        },
        "memory": _memory(),
        "v_scatter": {
            "reads_per_model_step": 32,
            "writes_per_model_step": 256,
            "counts_are_measured": True,
            "bytes": 1024,
            "nanoseconds": 2048,
        },
    }
    document.update(overrides)
    return document


def _determ(route: str, bit_exact: int, total: int, max_diff: float) -> dict:
    return {
        "route": route,
        "corelib_sha256": "a523b238" + "0" * 56,
        "harness_sha256": "beefcafe" + "0" * 56,
        "logits_bit_exact_steps": bit_exact,
        "logits_total_steps": total,
        "observed_max_abs_diff": max_diff,
        "determ2_bound_ulps": 2,
        "determ2_bound_kind": "relative_bf16_ulp",
        "first_divergence": None if bit_exact == total else "decode[13]",
        "failures": [],
    }


def _determ_runs(route: str, count: int) -> list[dict]:
    return [_determ(route, 17, 17, 0.0) for _ in range(count)]


class MemoryStabilityTests(unittest.TestCase):
    """Step 2. The one gate in this file that is about correctness."""

    def test_a_bounded_allocator_fluctuation_is_accepted(self):
        self.assertTrue(memory_is_stable(_memory()))

    def test_a_400_kib_per_token_leak_is_rejected(self):
        # 400 KiB/token over the 120-token measurement window is ~47 MiB, so
        # both the slope and the total growth are out of bounds. A validator
        # that only checked one of them would accept a leak that stayed under
        # the other, so the test drives both from the same leak rather than
        # constructing two independent violations.
        leak_per_token = 400 * 1024
        self.assertFalse(
            memory_is_stable(
                _memory(
                    private_bytes_slope_per_token=leak_per_token,
                    private_bytes_growth=leak_per_token * 120,
                )
            )
        )

    def test_the_slope_alone_rejects(self):
        self.assertFalse(
            memory_is_stable(_memory(private_bytes_slope_per_token=400 * 1024))
        )

    def test_the_growth_alone_rejects(self):
        self.assertFalse(
            memory_is_stable(_memory(private_bytes_growth=9 * 1024 * 1024))
        )

    def test_a_new_device_tensor_after_warmup_rejects(self):
        self.assertFalse(
            memory_is_stable(_memory(device_tensor_creates_after_warmup=1))
        )

    def test_a_new_weight_object_after_warmup_rejects(self):
        self.assertFalse(memory_is_stable(_memory(weight_creates_after_warmup=1)))

    def test_a_live_object_increase_rejects(self):
        self.assertFalse(memory_is_stable(_memory(live_corelib_object_delta=1)))

    def test_the_boundary_values_are_accepted(self):
        # Design 15.4 wants "stable", and the bounds are the ones Step 2 of the
        # task brief fixes. Exactly at the bound is stable; one byte over is
        # not. Pinning this stops a later ">=" edit from silently loosening it.
        self.assertTrue(
            memory_is_stable(
                _memory(
                    private_bytes_growth=8 * 1024 * 1024,
                    private_bytes_slope_per_token=64 * 1024,
                )
            )
        )
        self.assertFalse(
            memory_is_stable(_memory(private_bytes_growth=8 * 1024 * 1024 + 1))
        )
        self.assertFalse(
            memory_is_stable(_memory(private_bytes_slope_per_token=64 * 1024 + 1))
        )

    def test_a_missing_field_is_not_silently_stable(self):
        record = _memory()
        del record["live_corelib_object_delta"]
        # An absent counter must not read as zero. That is the shape this
        # project has now found seven times: a check that reports success for
        # work it did not do.
        with self.assertRaises(KeyError):
            memory_is_stable(record)


class SchemaTests(unittest.TestCase):
    """Step 1."""

    def test_a_complete_document_validates(self):
        self.assertEqual(validate(_baseline()), [])

    def test_every_required_section_is_required(self):
        for section in sorted(REQUIRED_SECTIONS):
            document = _baseline()
            del document[section]
            problems = validate(document)
            self.assertTrue(
                any(section in problem for problem in problems),
                f"removing {section!r} produced {problems!r}",
            )

    def test_the_required_section_set_is_the_one_the_brief_fixes(self):
        self.assertEqual(
            REQUIRED_SECTIONS,
            {
                "identity",
                "model_load",
                "ttft",
                "prefill",
                "continuation",
                "decode",
                "memory",
                "v_scatter",
            },
        )

    def test_every_required_identity_field_is_required(self):
        for field in sorted(REQUIRED_IDENTITY_FIELDS):
            document = _baseline()
            del document["identity"][field]
            problems = validate(document)
            self.assertTrue(
                any(field in problem for problem in problems),
                f"removing identity.{field} produced {problems!r}",
            )

    def test_an_empty_identity_value_is_not_a_value(self):
        # "" and None are what a collector writes when it could not find the
        # thing. Accepting them would record a baseline that names a machine
        # nobody can identify.
        for empty in ("", None):
            document = _baseline()
            document["identity"]["corelib_dll_sha256"] = empty
            self.assertTrue(
                any(
                    "corelib_dll_sha256" in problem
                    for problem in validate(document)
                ),
                f"an identity value of {empty!r} was accepted",
            )

    def test_the_identity_set_covers_what_the_brief_names(self):
        for field in (
            "machine",
            "cpu_sku",
            "npu_sku",
            "corelib_dll_sha256",
            "dynamic_dispatch_version",
            "ryzen_mm_version",
            "xrt_version",
            "model_sha256",
            "fastflow_revision",
        ):
            self.assertIn(field, REQUIRED_IDENTITY_FIELDS)

    def test_an_unstable_memory_window_fails_validation(self):
        document = _baseline()
        document["memory"]["private_bytes_slope_per_token"] = 400 * 1024
        problems = validate(document)
        self.assertTrue(any("memory" in problem for problem in problems))

    def test_latency_and_throughput_are_recorded_not_gated(self):
        # Design section 4: performance is not a release blocker. A validator
        # that failed on a slow number would convert a recorded figure into a
        # release threshold, which is precisely what 15.6 says not to do.
        document = _baseline()
        document["decode"]["runs"][0]["tokens_per_second"] = 0.001
        document["prefill"]["points"][0]["tokens_per_second"] = 0.001
        document["ttft"]["cold_ns"] = 10**12
        self.assertEqual(validate(document), [])

    def test_the_v_scatter_counts_are_the_design_contract(self):
        # 32 reads and 256 writes per model step is design 10.3/18.5, and it
        # IS a contract rather than a measurement.
        document = _baseline()
        document["v_scatter"]["writes_per_model_step"] = 255
        self.assertTrue(any("v_scatter" in p for p in validate(document)))

    def test_the_synchronize_count_is_the_design_contract(self):
        document = _baseline()
        document["decode"]["runs"][0]["synchronizes_per_pass"] = 2
        self.assertTrue(any("synchronize" in p for p in validate(document)))


class DeterminismBaselineTests(unittest.TestCase):
    """`DETERM-3`. The highest-value part of this task, and the easiest to
    get wrong by stating a confident number."""

    def test_twenty_clean_runs_per_route_produce_a_baseline(self):
        records = _determ_runs("append", 20) + _determ_runs("reprefill", 20)
        baseline = determinism_baseline(records)
        self.assertEqual(baseline["problems"], [])
        for route in ("append", "reprefill"):
            entry = baseline["routes"][route]
            self.assertEqual(entry["runs"], 20)
            self.assertEqual(entry["bit_identical_runs"], 20)
            self.assertEqual(entry["step_bit_identity_rate"], 1.0)
            self.assertEqual(entry["observed_max_abs_diff"]["max"], 0.0)

    def test_fewer_than_twenty_runs_is_reported_as_no_baseline(self):
        records = _determ_runs("append", 3) + _determ_runs("reprefill", 20)
        baseline = determinism_baseline(records)
        self.assertTrue(
            any("append" in problem for problem in baseline["problems"]),
            baseline["problems"],
        )
        # And it must not quietly publish a rate for the short route as if it
        # were a baseline.
        self.assertFalse(baseline["routes"]["append"]["is_baseline"])
        self.assertTrue(baseline["routes"]["reprefill"]["is_baseline"])

    def test_a_route_with_no_runs_at_all_is_a_problem(self):
        baseline = determinism_baseline(_determ_runs("append", 20))
        self.assertTrue(
            any("reprefill" in problem for problem in baseline["problems"]),
            baseline["problems"],
        )

    def test_the_minimum_is_the_one_determ3_states(self):
        self.assertEqual(MIN_DETERMINISM_RUNS_PER_ROUTE, 20)

    def test_every_route_figure_carries_its_n(self):
        records = _determ_runs("append", 20) + _determ_runs("reprefill", 21)
        baseline = determinism_baseline(records)
        for route, entry in baseline["routes"].items():
            self.assertIn("runs", entry, route)
            self.assertIn("steps", entry, route)
            self.assertIn("step_bit_identity_rate", entry, route)
            self.assertIn("observed_max_abs_diff", entry, route)

    def test_routes_are_reported_separately_and_never_averaged(self):
        # DETERM-3: "If the rate turns out to be route-dependent, say so
        # rather than averaging". A pooled figure would hide exactly the
        # mechanism difference the two routes exist to expose.
        records = [_determ("append", 16, 17, 0.25) for _ in range(20)]
        records += _determ_runs("reprefill", 20)
        baseline = determinism_baseline(records)
        self.assertNotIn("step_bit_identity_rate", baseline)
        self.assertLess(
            baseline["routes"]["append"]["step_bit_identity_rate"],
            baseline["routes"]["reprefill"]["step_bit_identity_rate"],
        )
        self.assertTrue(baseline["route_dependent"])

    def test_identical_rates_are_not_reported_as_route_dependent(self):
        records = _determ_runs("append", 20) + _determ_runs("reprefill", 20)
        self.assertFalse(determinism_baseline(records)["route_dependent"])

    def test_the_max_abs_diff_distribution_is_reported_not_just_its_peak(self):
        records = [_determ("append", 16, 17, 0.25) for _ in range(10)]
        records += [_determ("append", 16, 17, 0.5) for _ in range(10)]
        records += _determ_runs("reprefill", 20)
        entry = determinism_baseline(records)["routes"]["append"]
        spread = entry["observed_max_abs_diff"]
        self.assertEqual(spread["max"], 0.5)
        self.assertEqual(spread["min"], 0.25)
        self.assertEqual(spread["histogram"], {"0.25": 10, "0.5": 10})
        self.assertEqual(spread["nonzero_runs"], 20)

    def test_a_run_that_recorded_failures_is_counted_and_flagged(self):
        # A run whose self-consistency check FAILED is exactly the run a rate
        # baseline most needs to count. Dropping it would bias the figure
        # upward, and silently.
        failed = _determ("append", 16, 17, 4.0)
        failed["failures"] = ["decode[3]: 12 logits differ by more than 2 ULP"]
        records = [failed] + _determ_runs("append", 19) + _determ_runs(
            "reprefill", 20
        )
        baseline = determinism_baseline(records)
        entry = baseline["routes"]["append"]
        self.assertEqual(entry["runs"], 20)
        self.assertEqual(entry["runs_with_gate_failures"], 1)
        self.assertTrue(
            any("gate" in problem.lower() for problem in baseline["problems"]),
            baseline["problems"],
        )
        # A gate failure is a FINDING, not a structural defect in the
        # baseline. It must not block the record from being written; see the
        # blocking/non-blocking split in determinism_baseline.
        self.assertEqual(baseline["blocking_problems"], [])

    def test_runs_that_loaded_different_libraries_are_flagged(self):
        odd = _determ("append", 17, 17, 0.0)
        odd["corelib_sha256"] = "deadbeef" + "0" * 56
        records = [odd] + _determ_runs("append", 19) + _determ_runs(
            "reprefill", 20
        )
        baseline = determinism_baseline(records)
        self.assertTrue(
            any("sha-256" in problem.lower() for problem in baseline["problems"]),
            baseline["problems"],
        )

    def test_no_records_at_all_is_a_problem_not_an_empty_success(self):
        baseline = determinism_baseline([])
        self.assertTrue(baseline["problems"])
        self.assertFalse(baseline["is_baseline"])
        # A zero-run route is still listed, so the report shows what is
        # missing by name rather than by omission -- but nothing about it may
        # read as a measured figure.
        for route, entry in baseline["routes"].items():
            self.assertFalse(entry["is_baseline"], route)
            self.assertEqual(entry["runs"], 0, route)
            self.assertIsNone(entry["step_bit_identity_rate"], route)


class RenderingTests(unittest.TestCase):
    def test_the_markdown_states_the_rate_the_spread_and_the_n(self):
        document = _baseline()
        document["determinism"] = determinism_baseline(
            _determ_runs("append", 20) + _determ_runs("reprefill", 20)
        )
        text = render_markdown(document)
        self.assertIn("DETERM-3", text)
        self.assertIn("n = 20", text)
        self.assertIn("xcomedusad-43", text)
        self.assertIn("a523b238", text)

    def test_the_markdown_never_states_a_floor_without_its_n(self):
        document = _baseline()
        document["determinism"] = determinism_baseline(
            _determ_runs("append", 3) + _determ_runs("reprefill", 3)
        )
        text = render_markdown(document)
        self.assertIn("no baseline", text.lower())
        self.assertIn("3", text)


class MainTests(unittest.TestCase):
    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = report.main(argv)
        return code, out.getvalue() + err.getvalue()

    def _fixture(self, directory: Path, runs: int) -> tuple[Path, str, Path]:
        baseline = directory / "phi4_aie4_baseline.json"
        baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
        artifacts = directory / "artifacts"
        for index in range(runs):
            run_dir = artifacts / f"20260902T00{index:04d}Z-1"
            run_dir.mkdir(parents=True)
            for route in ("append", "reprefill"):
                name = "force_append" if route == "append" else "force_reprefill"
                (run_dir / f"determ1-{name}.json").write_text(
                    json.dumps(_determ(route, 17, 17, 0.0)), encoding="utf-8"
                )
        markdown = directory / "phi4_results.md"
        markdown.write_text(
            "---\ntitle: Phi4\n---\n\n## Existing content\n", encoding="utf-8"
        )
        return baseline, str(artifacts / "*" / "determ1-*.json"), markdown

    def test_a_complete_run_writes_the_markdown_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            baseline, glob, markdown = self._fixture(directory, 20)
            code, output = self._run(
                [
                    "--input",
                    str(baseline),
                    "--determinism-glob",
                    glob,
                    "--markdown",
                    str(markdown),
                ]
            )
            self.assertEqual(code, 0, output)
            text = markdown.read_text(encoding="utf-8")
            self.assertIn("## Existing content", text)
            self.assertIn("DETERM-3", text)
            self.assertIn("n = 20", text)

    def test_rendering_twice_does_not_duplicate_the_section(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            baseline, glob, markdown = self._fixture(directory, 20)
            argv = [
                "--input",
                str(baseline),
                "--determinism-glob",
                glob,
                "--markdown",
                str(markdown),
            ]
            self.assertEqual(self._run(argv)[0], 0)
            first = markdown.read_text(encoding="utf-8")
            self.assertEqual(self._run(argv)[0], 0)
            second = markdown.read_text(encoding="utf-8")
            self.assertEqual(first, second)

    def test_an_incomplete_determinism_campaign_fails_the_exit_code(self):
        # Skipped work must reach the exit code. Three runs per route is the
        # state DETERM-3 describes as "no baseline exists", and a tool that
        # rendered a document anyway would publish the thing DETERM-3 forbids.
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            baseline, glob, markdown = self._fixture(directory, 3)
            before = markdown.read_text(encoding="utf-8")
            code, output = self._run(
                [
                    "--input",
                    str(baseline),
                    "--determinism-glob",
                    glob,
                    "--markdown",
                    str(markdown),
                ]
            )
            self.assertNotEqual(code, 0)
            self.assertIn("DETERM-3", output)
            self.assertEqual(markdown.read_text(encoding="utf-8"), before)

    def test_a_gate_failure_is_published_and_still_fails_the_exit_code(self):
        # The distinction this test pins: "there is no baseline" blocks the
        # render, "the baseline contains a hard-gate failure" does not.
        # Suppressing the whole record because the campaign found something
        # would discard the measurement and leave only an exit code, which is
        # the opposite of what this baseline is for.
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            baseline, glob, markdown = self._fixture(directory, 20)
            broken = _determ("append", 0, 17, 48.34375)
            broken["failures"] = ["decode[7]: emitted token sequences differ"]
            first = sorted((directory / "artifacts").glob("*"))[0]
            (first / "determ1-force_append.json").write_text(
                json.dumps(broken), encoding="utf-8"
            )
            code, output = self._run(
                [
                    "--input",
                    str(baseline),
                    "--determinism-glob",
                    glob,
                    "--markdown",
                    str(markdown),
                ]
            )
            self.assertNotEqual(code, 0)
            self.assertIn("gate", output.lower())
            text = markdown.read_text(encoding="utf-8")
            self.assertIn("DETERM-3", text)
            self.assertIn("HARD GATE", text)
            self.assertIn("n = 20", text)

    def test_an_unstable_memory_window_fails_the_exit_code(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            baseline, glob, markdown = self._fixture(directory, 20)
            document = json.loads(baseline.read_text(encoding="utf-8"))
            document["memory"]["private_bytes_slope_per_token"] = 400 * 1024
            baseline.write_text(json.dumps(document), encoding="utf-8")
            code, output = self._run(
                [
                    "--input",
                    str(baseline),
                    "--determinism-glob",
                    glob,
                    "--markdown",
                    str(markdown),
                ]
            )
            self.assertNotEqual(code, 0)
            self.assertIn("memory", output)

    def test_a_determinism_glob_matching_nothing_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            baseline, _, markdown = self._fixture(directory, 20)
            code, output = self._run(
                [
                    "--input",
                    str(baseline),
                    "--determinism-glob",
                    str(directory / "nowhere" / "*.json"),
                    "--markdown",
                    str(markdown),
                ]
            )
            self.assertNotEqual(code, 0)
            self.assertIn("no DETERM-1 records", output)

    def test_the_merged_json_carries_the_determinism_section(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            baseline, glob, markdown = self._fixture(directory, 20)
            merged = directory / "merged.json"
            code, output = self._run(
                [
                    "--input",
                    str(baseline),
                    "--determinism-glob",
                    glob,
                    "--markdown",
                    str(markdown),
                    "--output-json",
                    str(merged),
                ]
            )
            self.assertEqual(code, 0, output)
            document = json.loads(merged.read_text(encoding="utf-8"))
            self.assertEqual(
                document["determinism"]["routes"]["append"]["runs"], 20
            )


class RouteDependenceTests(unittest.TestCase):
    """I-6. "The rate is route-dependent" is a claim, and it needs a test.

    It was previously a float inequality on two ratios, published in bold from
    1 divergent run of 33 against 2 of 33 -- the same n-limited overclaim
    DETERM-3 exists to prevent, made about DETERM-3's own output.
    """

    def test_fisher_matches_known_values(self):
        # Symmetric table: nothing to distinguish, p is exactly 1.
        self.assertEqual(_fisher_exact_two_sided((2, 31), (2, 31)), 1.0)
        # A table with an empty margin cannot be extreme.
        self.assertEqual(_fisher_exact_two_sided((0, 5), (0, 5)), 1.0)
        # Textbook 2x2: ((1,9),(8,2)) has two-sided p = 0.0055 to 2 s.f.
        self.assertAlmostEqual(
            _fisher_exact_two_sided((1, 9), (8, 2)), 0.005477, places=5
        )
        self.assertLessEqual(_fisher_exact_two_sided((3, 30), (0, 33)), 1.0)

    def test_one_versus_two_divergent_runs_of_33_is_not_route_dependence(self):
        records = _determ_runs("append", 32) + [_determ("append", 0, 17, 48.3)]
        records += _determ_runs("reprefill", 31) + [
            _determ("reprefill", 15, 17, 0.3125),
            _determ("reprefill", 9, 17, 0.25),
        ]
        baseline = determinism_baseline(records)
        self.assertTrue(baseline["observed_rates_differ"])
        self.assertFalse(baseline["route_dependent"])
        self.assertGreaterEqual(
            baseline["route_dependence_p"], ROUTE_DEPENDENCE_ALPHA
        )

    def test_a_large_separation_is_route_dependence(self):
        records = _determ_runs("append", 33)
        records += _determ_runs("reprefill", 21) + [
            _determ("reprefill", 9, 17, 0.25) for _ in range(12)
        ]
        baseline = determinism_baseline(records)
        self.assertTrue(baseline["route_dependent"])
        self.assertLess(
            baseline["route_dependence_p"], ROUTE_DEPENDENCE_ALPHA
        )

    def test_the_document_says_the_n_does_not_support_the_claim(self):
        document = _baseline()
        records = _determ_runs("append", 32) + [_determ("append", 0, 17, 48.3)]
        records += _determ_runs("reprefill", 33)
        document["determinism"] = determinism_baseline(records)
        text = render_markdown(document)
        self.assertIn("does not support", text)
        self.assertNotIn("**The rate is route-dependent**", text)


class BaselineFlagTests(unittest.TestCase):
    """I-3. `is_baseline` is the machine-readable form of the sentence four
    lines below it, and it used to contradict it."""

    def test_a_gate_failure_makes_is_baseline_false(self):
        broken = _determ("append", 0, 17, 48.34375)
        broken["failures"] = ["decode[7]: emitted token sequences differ"]
        records = [broken] + _determ_runs("append", 32) + _determ_runs(
            "reprefill", 33
        )
        baseline = determinism_baseline(records)
        self.assertEqual(baseline["gate_failures"], 1)
        self.assertFalse(baseline["is_baseline"])
        # Per-route run counts are still met, so the ROUTE flag stays true;
        # it answers "does this route have enough runs", not "is this a
        # settled baseline".
        self.assertTrue(baseline["routes"]["append"]["is_baseline"])

    def test_a_clean_window_is_a_baseline(self):
        records = _determ_runs("append", 33) + _determ_runs("reprefill", 33)
        baseline = determinism_baseline(records)
        self.assertEqual(baseline["gate_failures"], 0)
        self.assertTrue(baseline["is_baseline"])


class BinaryIdentityTests(unittest.TestCase):
    """I-7. Both halves of "the same binary", and an auditable source list."""

    def test_a_differing_harness_hash_blocks_the_baseline(self):
        odd = _determ("append", 17, 17, 0.0)
        odd["harness_sha256"] = "deadbeef" + "0" * 56
        records = [odd] + _determ_runs("append", 32) + _determ_runs(
            "reprefill", 33
        )
        baseline = determinism_baseline(records)
        self.assertTrue(
            any(
                "harness" in problem.lower()
                for problem in baseline["blocking_problems"]
            ),
            baseline["blocking_problems"],
        )

    def test_a_record_with_no_harness_hash_is_excluded_and_counted(self):
        # A record predating the field cannot be shown to belong to the pool,
        # and silently including it is how a glob over a build tree turns into
        # a baseline nobody can reproduce. It is dropped, not blocked:
        # refusing to produce a baseline at all while such a record exists
        # anywhere under the glob would make the committed DETERM-4 evidence
        # permanently unusable.
        stale = _determ("append", 0, 17, 48.0)
        stale["_source"] = "old/determ1-force_append-010.json"
        del stale["harness_sha256"]
        records = [stale] + _determ_runs("append", 33) + _determ_runs(
            "reprefill", 33
        )
        baseline = determinism_baseline(records)
        self.assertEqual(baseline["blocking_problems"], [])
        self.assertTrue(
            any(
                "EXCLUDED" in problem for problem in baseline["problems"]
            ),
            baseline["problems"],
        )
        # And it really is excluded: 33 append runs counted, not 34, and the
        # divergent one is not dragged into the rate.
        self.assertEqual(baseline["routes"]["append"]["runs"], 33)
        self.assertEqual(baseline["routes"]["append"]["run_bit_identity_rate"], 1.0)
        self.assertEqual(
            baseline["excluded_sources"],
            ["old/determ1-force_append-010.json"],
        )

    def test_the_sources_are_recorded(self):
        records = _determ_runs("append", 33) + _determ_runs("reprefill", 33)
        for index, record in enumerate(records):
            record["_source"] = f"artifacts/run/determ1-{index}.json"
        baseline = determinism_baseline(records)
        self.assertEqual(len(baseline["sources"]), 66)
        self.assertIn("artifacts/run/determ1-0.json", baseline["sources"])


class CrossoverRenderingTests(unittest.TestCase):
    """C-1. The document published a grid artifact as a measured threshold."""

    def test_the_document_publishes_a_bracket_and_says_so(self):
        text = render_markdown(_baseline())
        self.assertIn("BRACKET, not a threshold", text)
        self.assertIn("(8, 12]", text)
        self.assertIn("(24, 32]", text)
        # And never the old phrasing, which read as a measured answer.
        self.assertNotIn("Longest suffix at which append beats", text)

    def test_an_unbracketed_crossover_says_so_rather_than_naming_a_number(self):
        document = _baseline()
        document["continuation"]["crossover"]["512"][
            "reprefill_wins_from"
        ] = 0
        text = render_markdown(document)
        self.assertIn("not bracketed by this grid", text)

    def test_the_load_breakdown_names_the_shape_plan(self):
        text = render_markdown(_baseline())
        self.assertIn("Phi4ShapePlan::Build", text)
        self.assertIn("90% of load", text)
        self.assertIn("unaccounted", text)


class VScatterProvenanceTests(unittest.TestCase):
    """I-4. The per-step counts must be measured, not restated constants."""

    def test_a_record_that_does_not_claim_measurement_is_rejected(self):
        document = _baseline()
        del document["v_scatter"]["counts_are_measured"]
        self.assertTrue(
            any("counts_are_measured" in p for p in validate(document)),
            validate(document),
        )

    def test_a_record_that_claims_measurement_falsely_is_rejected(self):
        document = _baseline()
        document["v_scatter"]["counts_are_measured"] = False
        self.assertTrue(
            any("counts_are_measured" in p for p in validate(document))
        )


def _determ_with_localisation(route, step, differing):
    record = _determ(route, 9, 17, 49.25)
    record["failures"] = [f"{step}: logits differ by more than 2 BF16 ULP"]
    record["first_divergence"] = step
    record["localisation"] = {
        "measured": True,
        "step": step,
        "lm_head_input_elements": 3072,
        "lm_head_input_differing_elements": differing,
        "source": "model_body",
        "reason": "the two runs fed the LM head DIFFERENT rows",
    }
    return record


class LocalisationPublicationTests(unittest.TestCase):
    """The finding, not just the rate.

    A reader of the benchmark document was learning that 2 runs in 41 were not
    bit-identical, and nothing about the thing that matters: the two runs fed
    the LM head different rows, so the divergence is upstream of it. That was
    in the task report, the records README and the design spec, and in none of
    the places a reader of the benchmarks would look.
    """

    def _document(self):
        records = _determ_runs("append", 39) + [
            _determ_with_localisation("append", "decode[8]", 2571),
            _determ_with_localisation("append", "decode[6]", 2754),
        ]
        records += _determ_runs("reprefill", 41)
        document = _baseline()
        document["determinism"] = determinism_baseline(records)
        return document

    def test_the_localisation_is_aggregated_per_route(self):
        entry = self._document()["determinism"]["routes"]["append"][
            "localisation"
        ]
        self.assertEqual(entry["measured_runs"], 2)
        self.assertEqual(entry["by_source"], {"model_body": 2})
        self.assertEqual(
            entry["lm_head_input_differing_elements"], [2571, 2754]
        )
        self.assertEqual(entry["steps"], ["decode[6]", "decode[8]"])

    def test_an_unmeasured_localisation_is_not_counted(self):
        # Records predating the per-step capture carry no localisation. They
        # must not be counted as evidence of anything.
        records = _determ_runs("append", 41) + _determ_runs("reprefill", 41)
        entry = determinism_baseline(records)["routes"]["append"][
            "localisation"
        ]
        self.assertEqual(entry["measured_runs"], 0)
        self.assertEqual(entry["by_source"], {})

    def test_the_document_states_the_finding(self):
        text = render_markdown(self._document())
        self.assertIn("Where the divergence enters", text)
        self.assertIn("model_body", text)
        self.assertIn("2,571", text)
        self.assertIn("2,754", text)
        self.assertIn("divergence therefore enters the model body", text)

    def test_the_document_says_the_layer_is_unknown(self):
        text = render_markdown(self._document())
        self.assertIn("layer at which it enters is not known", text)

    def test_a_document_with_no_measured_localisation_says_nothing(self):
        document = _baseline()
        document["determinism"] = determinism_baseline(
            _determ_runs("append", 41) + _determ_runs("reprefill", 41)
        )
        text = render_markdown(document)
        self.assertNotIn("Where the divergence enters", text)


class DecisionRuleReconciliationTests(unittest.TestCase):
    """Two live rules, one contradicting the other, is C-1's failure mode in a
    subtler form: the reader cannot tell which number to act on."""

    def test_the_2x_rule_is_scoped_to_cross_run_comparisons(self):
        text = render_markdown(_baseline())
        self.assertIn("Two different comparisons", text)
        self.assertIn("roughly 2x as unresolved", text)
        # And it must be visibly scoped, not stated unconditionally.
        self.assertIn("DIFFERENT run", text)

    def test_the_crossover_blurb_names_the_rule_it_actually_uses(self):
        text = render_markdown(_baseline())
        self.assertIn("interleaved within each point", text)
        self.assertIn("drift between a route's first and last", text)


if __name__ == "__main__":
    unittest.main()
