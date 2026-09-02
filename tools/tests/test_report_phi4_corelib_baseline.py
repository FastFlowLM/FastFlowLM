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
            "weight_pack_ns": 2,
            "total_ns": 3,
            "weight_objects": 161,
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
            "append_wins": {"512": 32, "2048": 0},
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


if __name__ == "__main__":
    unittest.main()
