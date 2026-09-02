"""Validate and render the Phi-4 AIE4 performance and memory baseline.

Task 13 Step 10.

WHAT THIS TOOL IS FOR, because it is easy to build the wrong thing.

Design section 4 says performance is explicitly not a release blocker for this
release, and design section 15.6 says to record the baseline "without pass/fail
thresholds". So no latency or throughput number in here is ever a gate. What
this tool produces is an honest, reproducible RECORD with enough identity
attached that a later change can be compared against it, and what it gates on
is only the three things whose absence would make the record worthless or
misleading:

  1. every required section and identity field is present. A baseline that
     does not say which machine, which corelib DLL and which model produced it
     is not a baseline, it is a set of numbers;
  2. the 128-token memory window is stable. That is not a performance figure --
     design 18.7 and design 15.4 make bounded post-warm allocation a
     correctness property, and an unbounded one is a leak; and
  3. the `DETERM-3` bit-identity baseline states its rate, the spread of the
     observed maximum absolute difference, and the n those rest on, over at
     least the 20 runs per route `DETERM-3` requires.

Gate 3 deserves its own paragraph. `DETERM-2` names "a bit-identity rate that
degrades from the recorded baseline" as a failure condition. As of 2026-09-02
no such baseline existed, so that clause enforced nothing, and `DETERM-3` says
so explicitly rather than letting it look enforced. This tool is what closes
that -- and the failure mode it must avoid is not "no baseline" but "a
confident number derived from too few runs", which is harder to notice than the
absence it would replace. So a short campaign fails the exit code and renders
nothing, rather than publishing a floor.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import math
import statistics
import sys
from pathlib import Path

# Step 1 of the task brief fixes this set.
REQUIRED_SECTIONS = {
    "identity",
    "model_load",
    "ttft",
    "prefill",
    "continuation",
    "decode",
    "memory",
    "v_scatter",
}

# Design section 14's telemetry list, reduced to the fields without which two
# baselines cannot be compared at all. `corelib_source_revision` rather than
# only the version: `ryzenai_corelib_get_version` is a hard-coded 0.1.0 in
# corelib's own source and spans the whole 0.x history, so it cannot identify a
# revision (Task 12 report section 1). The DLL SHA-256 identifies the exact
# binary; the source revision identifies what it was built from. Both, because
# MSVC RelWithDebInfo is not byte-reproducible and the hash alone changes on a
# rebuild of identical source.
REQUIRED_IDENTITY_FIELDS = {
    "machine",
    "cpu_sku",
    "npu_sku",
    "npu_driver_version",
    "corelib_dll_path",
    "corelib_dll_sha256",
    "corelib_version",
    "corelib_source_revision",
    "dynamic_dispatch_version",
    "ryzen_mm_version",
    "xrt_version",
    "model_dir",
    "model_sha256",
    "fastflow_revision",
}

# Step 2's bounds, and the design contracts that are contracts rather than
# measurements.
MAX_PRIVATE_BYTES_GROWTH = 8 * 1024 * 1024
MAX_PRIVATE_BYTES_SLOPE_PER_TOKEN = 64 * 1024
SYNCHRONIZES_PER_MODEL_STEP = 129
V_READS_PER_MODEL_STEP = 32
V_WRITES_PER_MODEL_STEP = 256

# Design `DETERM-3`: "it requires at least 20 runs per route".
MIN_DETERMINISM_RUNS_PER_ROUTE = 20

# I-6. The significance level for the route-dependence claim. 0.05 is the
# conventional value and is named here rather than inlined, because the claim
# it gates -- "the rate is route-dependent" -- was previously published in
# bold from a float inequality on 1-vs-2 divergent runs of 33.
ROUTE_DEPENDENCE_ALPHA = 0.05

# Design 5.1. The LM-head input row is one hidden state, and the localisation
# counts are reported against it.
HIDDEN_SIZE = 3072

# The routes `DETERM-3` requires separately. Append and re-prefill drive
# different row extents through the same LM-head shape, so a difference
# between them is informative about the mechanism and must not be averaged
# away.
DETERMINISM_ROUTES = ("append", "reprefill")

_MARKDOWN_BEGIN = "<!-- BEGIN phi4-aie4-baseline -->"
_MARKDOWN_END = "<!-- END phi4-aie4-baseline -->"


def _fisher_exact_two_sided(
    left: tuple[int, int],
    right: tuple[int, int],
) -> float:
    """Two-sided Fisher exact p for a 2x2 table of counts.

    Exact and dependency-free: `math.comb` is all this needs, and pulling in
    SciPy for one hypothesis test would add a dependency to a validator that
    otherwise runs anywhere Python does.

    Sums the hypergeometric probability of every table at least as extreme as
    the observed one, which is the standard two-sided construction.
    """
    a, b = left
    c, d = right
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2
    if total == 0 or row1 == 0 or row2 == 0 or col1 == 0 or col1 == total:
        return 1.0

    def probability(value: int) -> float:
        return (
            math.comb(row1, value)
            * math.comb(row2, col1 - value)
            / math.comb(total, col1)
        )

    observed = probability(a)
    low = max(0, col1 - row2)
    high = min(col1, row1)
    # 1e-9 relative slack: the equal-probability tables belong in the tail, and
    # floating-point evaluation of equal binomial ratios does not reproduce
    # bit-identical values.
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(low, high + 1)
            if probability(value) <= observed * (1 + 1e-9)
        ),
    )


def memory_is_stable(record: dict[str, object]) -> bool:
    """Step 2's gate, verbatim from the task brief.

    Indexing rather than `.get`: an absent counter must raise, not read as
    zero. A validator that treats a missing measurement as a passing one is
    the defect this project has now found seven times.
    """
    return (
        record["device_tensor_creates_after_warmup"] == 0
        and record["weight_creates_after_warmup"] == 0
        and record["live_corelib_object_delta"] == 0
        and record["private_bytes_growth"] <= 8 * 1024 * 1024
        and record["private_bytes_slope_per_token"] <= 64 * 1024
    )


def _missing_identity(identity: object) -> list[str]:
    problems: list[str] = []
    if not isinstance(identity, dict):
        return ["identity is not an object"]
    for field in sorted(REQUIRED_IDENTITY_FIELDS):
        if field not in identity:
            problems.append(f"identity.{field} is absent")
            continue
        value = identity[field]
        # "" and None are what a collector writes when it could not find the
        # thing, and recording them would name a machine nobody can identify.
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"identity.{field} is empty")
    return problems


def validate(document: dict[str, object]) -> list[str]:
    """Return every problem found, rather than the first.

    A caller re-running a two-hour hardware campaign for one missing field at
    a time is a worse outcome than a long list.
    """
    problems: list[str] = []
    for section in sorted(REQUIRED_SECTIONS):
        if section not in document:
            problems.append(f"section {section!r} is absent")
    problems.extend(_missing_identity(document.get("identity")))

    memory = document.get("memory")
    if isinstance(memory, dict):
        try:
            stable = memory_is_stable(memory)
        except KeyError as error:
            problems.append(
                f"memory is missing the counter {error.args[0]!r}, so the "
                f"128-token stability window cannot be judged"
            )
        else:
            if not stable:
                problems.append(
                    "memory: the 128-token stability window is NOT stable "
                    f"(device tensors created after warmup "
                    f"{memory.get('device_tensor_creates_after_warmup')}, "
                    f"weight objects "
                    f"{memory.get('weight_creates_after_warmup')}, "
                    f"live-object delta "
                    f"{memory.get('live_corelib_object_delta')}, "
                    f"private-byte growth "
                    f"{memory.get('private_bytes_growth')} > "
                    f"{MAX_PRIVATE_BYTES_GROWTH}, slope "
                    f"{memory.get('private_bytes_slope_per_token')} > "
                    f"{MAX_PRIVATE_BYTES_SLOPE_PER_TOKEN})"
                )

    scatter = document.get("v_scatter")
    if isinstance(scatter, dict):
        # Design 10.3 and 18.5. These two are a CONTRACT, not a measurement:
        # one complete model pass copies 32 V caches out and 256 per-head
        # slices back. A number other than those means the schedule changed.
        for field, expected in (
            ("reads_per_model_step", V_READS_PER_MODEL_STEP),
            ("writes_per_model_step", V_WRITES_PER_MODEL_STEP),
        ):
            if field not in scatter:
                problems.append(f"v_scatter.{field} is absent")
            elif scatter[field] != expected:
                problems.append(
                    f"v_scatter.{field} is {scatter[field]}, design 18.5 "
                    f"requires {expected}"
                )
        for field in ("bytes", "nanoseconds"):
            if field not in scatter:
                problems.append(f"v_scatter.{field} is absent")
        # I-4. The two counts above are only worth reading if they were
        # DIVIDED OUT of the observed totals rather than copied from the same
        # constants they are checked against. A benchmark that restates the
        # constants produces a row that cannot fail and reads as measured, so
        # the document refuses to render a record that does not assert the
        # provenance.
        if scatter.get("counts_are_measured") is not True:
            problems.append(
                "v_scatter.counts_are_measured is not true: the per-step "
                "read and write counts must be the observed totals divided "
                "by the observed model-step count, not the design constants "
                "restated. A gate that compares a constant against itself "
                "cannot fail and must not be rendered as a measurement."
            )

    decode = document.get("decode")
    if isinstance(decode, dict):
        runs = decode.get("runs")
        if not isinstance(runs, list) or not runs:
            problems.append("decode.runs is absent or empty")
        else:
            for run in runs:
                count = run.get("synchronizes_per_pass")
                if count != SYNCHRONIZES_PER_MODEL_STEP:
                    problems.append(
                        f"decode run at context {run.get('start_context')}: "
                        f"synchronizes_per_pass is {count}, design 10.4 "
                        f"requires {SYNCHRONIZES_PER_MODEL_STEP}"
                    )

    for section in ("prefill", "continuation"):
        body = document.get(section)
        if isinstance(body, dict) and not body.get("points"):
            problems.append(f"{section}.points is absent or empty")

    return problems


def _format_number(value: float) -> str:
    # Histogram keys have to survive a JSON round trip and stay comparable
    # between runs, so they are formatted rather than repr'd.
    if value == int(value):
        return str(int(value)) if abs(value) < 1e16 else repr(value)
    return f"{value:.6g}"


def determinism_baseline(records: list[dict]) -> dict:
    """Aggregate `determ1-<route>.json` records into the `DETERM-3` baseline.

    Reports per route, never pooled. Design `DETERM-3`: "If the rate turns out
    to be route-dependent, say so rather than averaging: append and re-prefill
    drive different row extents through the same LM-head shape, and a
    difference between them would itself be informative about the mechanism."

    Every figure is emitted with the n it rests on, and a route that has not
    reached `MIN_DETERMINISM_RUNS_PER_ROUTE` is marked `is_baseline: false`
    and produces a problem. That combination is the point: the numbers are
    still visible, so the campaign's progress is legible, but nothing
    downstream can cite them as a baseline.

    Two kinds of problem, kept apart on purpose.

    `blocking_problems` are the ones that mean there is NO baseline: too few
    runs, no records at all, or records from more than one binary. Those stop
    the report being rendered, because a document that implies a baseline
    exists when it does not is the exact defect `DETERM-3` names.

    `problems` also includes hard-gate failures, and those must NOT block
    rendering. A run that breached `DETERM-2` is a finding, and suppressing
    the whole record because the campaign found something is the opposite of
    what this task is for. It is counted in the rate, called out in the
    document, and it reaches the exit code.
    """
    problems: list[str] = []
    blocking: list[str] = []
    routes: dict[str, dict] = {}
    # I-7. Which files this baseline was actually built from, so a pooled glob
    # is auditable rather than trusted.
    # A record that cannot be shown to come from the same FastFlow binary is
    # dropped here, before anything is counted, rather than quietly averaged
    # in with the rest.
    excluded = [
        record for record in records if not record.get("harness_sha256")
    ]
    excluded_sources = sorted(
        str(record.get("_source", "<unknown>")) for record in excluded
    )
    records = [record for record in records if record.get("harness_sha256")]
    sources = sorted(
        str(record.get("_source", "<unknown>")) for record in records
    )

    by_route: dict[str, list[dict]] = {}
    for record in records:
        route = record.get("route")
        if route not in DETERMINISM_ROUTES:
            blocking.append(
                f"a DETERM-1 record names an unknown route {route!r}; it is "
                f"not counted toward any baseline"
            )
            continue
        by_route.setdefault(route, []).append(record)

    # I-7. BOTH halves of "the same binary". The corelib DLL was pinned and
    # the FastFlow harness that drove it was not, so a glob spanning a build
    # tree could pool records from two different FastFlow builds with nothing
    # able to notice. A record that predates the harness-hash field is itself
    # a reason to refuse: it cannot be shown to belong.
    for field, label in (
        ("corelib_sha256", "corelib SHA-256"),
        ("harness_sha256", "FastFlow harness SHA-256"),
    ):
        hashes = {
            record.get(field) for record in records if record.get(field)
        }
        if len(hashes) > 1:
            blocking.append(
                f"the records do not all share one {label} "
                f"({len(hashes)} distinct values), so they do not describe "
                f"one binary and cannot be pooled into a baseline: "
                + ", ".join(sorted(value[:16] for value in hashes))
            )
    if excluded:
        # EXCLUDED, NOT BLOCKED, and counted.
        #
        # A record predating the harness-hash field cannot be shown to belong
        # to the pool, so it must not be counted. But refusing to produce any
        # baseline while such a record exists anywhere under the glob would
        # make the committed DETERM-4 evidence permanently unusable, which is
        # the opposite of why it is committed. It is dropped, the count and
        # the reason are stated, and `sources` lists exactly what WAS used.
        problems.append(
            f"{len(excluded)} record(s) carry no FastFlow harness SHA-256 "
            f"and were EXCLUDED from the pooled figures: they cannot be shown "
            f"to describe the same binary as the rest. They remain committed "
            f"as evidence and are listed under `excluded_sources`."
        )
    if not records:
        blocking.append("no DETERM-1 records were supplied")

    for route in DETERMINISM_ROUTES:
        entries = by_route.get(route, [])
        if not entries:
            blocking.append(
                f"route {route!r}: no DETERM-1 records at all, so DETERM-3's "
                f"minimum of {MIN_DETERMINISM_RUNS_PER_ROUTE} runs is not met"
            )
            routes[route] = {
                "runs": 0,
                "steps": 0,
                "bit_identical_runs": 0,
                "step_bit_identity_rate": None,
                "run_bit_identity_rate": None,
                "observed_max_abs_diff": {
                    "max": None,
                    "min": None,
                    "median": None,
                    "nonzero_runs": 0,
                    "histogram": {},
                },
                "runs_with_gate_failures": 0,
                "first_divergences": {},
                "localisation": {
                    "measured_runs": 0,
                    "by_source": {},
                    "lm_head_input_differing_elements": [],
                    "lm_head_input_elements": HIDDEN_SIZE,
                    "steps": [],
                },
                "is_baseline": False,
            }
            continue

        runs = len(entries)
        total_steps = sum(int(entry["logits_total_steps"]) for entry in entries)
        exact_steps = sum(
            int(entry["logits_bit_exact_steps"]) for entry in entries
        )
        bit_identical_runs = sum(
            1
            for entry in entries
            if int(entry["logits_bit_exact_steps"])
            == int(entry["logits_total_steps"])
        )
        maxima = [
            float(entry.get("observed_max_abs_diff") or 0.0) for entry in entries
        ]
        histogram: dict[str, int] = {}
        for value in maxima:
            if value == 0.0:
                continue
            key = _format_number(value)
            histogram[key] = histogram.get(key, 0) + 1
        nonzero = [value for value in maxima if value != 0.0]
        failing = [entry for entry in entries if entry.get("failures")]
        divergences: dict[str, int] = {}
        for entry in entries:
            label = entry.get("first_divergence")
            if label:
                divergences[label] = divergences.get(label, 0) + 1

        # WHERE THE DIVERGENCE ENTERED, aggregated so the published document
        # carries the finding and not only the rate.
        #
        # A reader of the benchmark document was learning that 2 runs in 41
        # were not bit-identical, and nothing about the thing that actually
        # matters: that the two runs fed the LM head DIFFERENT rows, so the
        # divergence is upstream of it. That was in the task report, the
        # records README and the design spec, and in none of the places a
        # reader of the benchmarks would look.
        localisations: dict[str, int] = {}
        localisation_elements: list[int] = []
        localisation_steps: list[str] = []
        for entry in entries:
            record = entry.get("localisation") or {}
            if not record.get("measured"):
                continue
            source = str(record.get("source", "unknown"))
            localisations[source] = localisations.get(source, 0) + 1
            if record.get("lm_head_input_differing_elements") is not None:
                localisation_elements.append(
                    int(record["lm_head_input_differing_elements"])
                )
            if record.get("step"):
                localisation_steps.append(str(record["step"]))

        routes[route] = {
            "runs": runs,
            "steps": total_steps,
            "bit_identical_steps": exact_steps,
            "bit_identical_runs": bit_identical_runs,
            # Two rates, because they answer different questions and reporting
            # only one invites the reader to assume the other. The STEP rate is
            # the per-logit-vector property DETERM-1 records; the RUN rate is
            # the probability that a whole 17-step comparison comes back clean,
            # which is what a suite operator actually observes.
            "step_bit_identity_rate": exact_steps / total_steps
            if total_steps
            else None,
            "run_bit_identity_rate": bit_identical_runs / runs,
            "observed_max_abs_diff": {
                "max": max(maxima),
                "min": min(nonzero) if nonzero else 0.0,
                "median": statistics.median(maxima),
                "nonzero_runs": len(nonzero),
                "histogram": dict(
                    sorted(histogram.items(), key=lambda item: float(item[0]))
                ),
            },
            "runs_with_gate_failures": len(failing),
            "first_divergences": dict(sorted(divergences.items())),
            "localisation": {
                "measured_runs": sum(localisations.values()),
                "by_source": dict(sorted(localisations.items())),
                "lm_head_input_differing_elements": sorted(
                    localisation_elements
                ),
                "lm_head_input_elements": HIDDEN_SIZE,
                "steps": sorted(localisation_steps),
            },
            "is_baseline": runs >= MIN_DETERMINISM_RUNS_PER_ROUTE,
        }
        if runs < MIN_DETERMINISM_RUNS_PER_ROUTE:
            blocking.append(
                f"route {route!r}: {runs} run(s), below DETERM-3's minimum of "
                f"{MIN_DETERMINISM_RUNS_PER_ROUTE}. No baseline is established "
                f"for this route; the figures above are what has been measured "
                f"so far and must not be cited as a floor."
            )
        if failing:
            problems.append(
                f"route {route!r}: {len(failing)} run(s) recorded a DETERM-2 "
                f"gate failure. They ARE counted in the rate above -- dropping "
                f"them would bias it upward -- but a baseline should not be "
                f"declared over a window containing a hard-gate failure."
            )

    # I-6. ROUTE DEPENDENCE IS A CLAIM, AND IT NEEDS A TEST.
    #
    # This used to be `len(set(rates)) > 1` -- a float inequality on two
    # ratios. With 1 divergent run of 33 against 2 of 33 that is true, and the
    # renderer published "the rate is route-dependent" in bold. It is the same
    # n-limited overclaim DETERM-3 exists to prevent, made about DETERM-3's own
    # output.
    #
    # A two-sided Fisher exact test on the 2x2 table of (divergent, clean) per
    # route is the standard answer and introduces no tunable of its own. 1-of-33
    # against 2-of-33 gives p = 1.0; the observed counts differ and the
    # difference is not supported. Both facts are reported, separately, because
    # they are different statements and collapsing them is what went wrong.
    counts = [
        (entry["runs"] - entry["bit_identical_runs"], entry["bit_identical_runs"])
        for entry in routes.values()
        if entry["runs"] > 0
    ]
    observed_rates_differ = (
        len(
            {
                entry["run_bit_identity_rate"]
                for entry in routes.values()
                if entry["run_bit_identity_rate"] is not None
            }
        )
        > 1
    )
    route_dependence_p = (
        _fisher_exact_two_sided(counts[0], counts[1])
        if len(counts) == 2
        else None
    )
    route_dependent = (
        route_dependence_p is not None
        and route_dependence_p < ROUTE_DEPENDENCE_ALPHA
    )

    gate_failures = sum(
        entry["runs_with_gate_failures"] for entry in routes.values()
    )
    return {
        "min_runs_per_route": MIN_DETERMINISM_RUNS_PER_ROUTE,
        "routes": routes,
        "observed_rates_differ": observed_rates_differ,
        "route_dependence_p": route_dependence_p,
        "route_dependence_alpha": ROUTE_DEPENDENCE_ALPHA,
        "route_dependent": route_dependent,
        "gate_failures": gate_failures,
        # I-3. A window containing a DETERM-2 hard-gate failure is NOT a
        # baseline, and this flag used to say it was -- four lines below the
        # problem text saying it must not. The flag is the machine-readable
        # form of that sentence, so it has to agree with it.
        "is_baseline": not blocking
        and bool(routes)
        and gate_failures == 0
        and all(entry["is_baseline"] for entry in routes.values()),
        "sources": sources,
        "excluded_sources": excluded_sources,
        "blocking_problems": blocking,
        "problems": blocking + problems,
    }


def load_determinism_records(pattern: str) -> list[dict]:
    records = []
    for path in sorted(globlib.glob(pattern)):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        # I-7. Where each record came from, so a pooled glob can be audited
        # rather than trusted. Prefixed so it cannot collide with a field the
        # comparator writes.
        record["_source"] = path
        records.append(record)
    return records


def _ns(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) / 1e6:.1f} ms"


def _bytes(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024 or unit == "GiB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.2f} {unit}"
        value /= 1024
    return f"{value} B"


def _rate(value: object, total: object) -> str:
    if value is None:
        return "not measured"
    return f"{float(value) * 100:.2f}% (n = {total})"


def render_markdown(document: dict) -> str:
    identity = document.get("identity", {})
    load = document.get("model_load", {})
    ttft = document.get("ttft", {})
    memory = document.get("memory", {})
    scatter = document.get("v_scatter", {})
    determinism = document.get("determinism", {})

    lines: list[str] = []
    add = lines.append
    add(_MARKDOWN_BEGIN)
    add("")
    add("## Phi-4 mini instruct on AIE4 (corelib backend) — recorded baseline")
    add("")
    add(
        "Recorded per design section 15.6, **without pass/fail thresholds**. "
        "Design section 4 makes performance explicitly not a release blocker "
        "for this release; these figures exist so a later change can be "
        "compared against a measured starting point, not so a number can be "
        "defended."
    )
    add("")
    # A caveat that is always present, because it applies to every latency
    # figure below and a reader who does not know it will read ordinary
    # machine noise as a regression.
    add(
        "> **Every latency and throughput figure here comes from ONE run on a "
        "shared lab machine.** Task 13 ran this same benchmark three times "
        "against the same binary and the same model within two hours and "
        "measured decode throughput of 22.5, 22.4 and 12.4 tokens/s at "
        "context 128 — a factor of 1.8, with no code change. Within a single "
        "run, per-token append latency stepped from 76 ms to 46 ms partway "
        "through the continuation sweep and stayed there. The machine runs "
        "corporate endpoint agents whose scans are not under this project's "
        "control, and the host share of a decode token is large enough for "
        "CPU contention to show."
    )
    add("")
    # TWO COMPARISONS, TWO RULES, AND WHICH IS WHICH.
    #
    # This caveat used to end "treat a difference below roughly 2x as
    # unresolved", stated unconditionally -- and thirty lines below it the
    # crossover table published a bracket decided on differences far smaller
    # than 2x. Two live rules, one contradicting the other, is the same defect
    # as publishing a grid artifact as a measurement: the reader cannot tell
    # which number to act on.
    add(
        "> **Two different comparisons, and they do not share a rule.** "
        "Comparing a figure here against one from a DIFFERENT run — a later "
        "revision, another machine, this document a month from now — is "
        "subject to that 1.8x instability, so treat a difference below "
        "roughly 2x as unresolved unless it is reproduced across runs. The "
        "append-versus-re-prefill comparison below is not that comparison: "
        "its samples are **interleaved within a single point**, so a regime "
        "shift moves both routes together, and each point additionally has "
        "to beat the drift measured across it. That is why it can resolve "
        "differences the 2x rule could not — the instability it would be "
        "guarding against has been measured and subtracted rather than "
        "assumed away."
    )
    add("")
    add("### Identity")
    add("")
    add("| | |")
    add("| --- | --- |")
    for label, key in (
        ("machine", "machine"),
        ("CPU", "cpu_sku"),
        ("NPU", "npu_sku"),
        ("NPU driver", "npu_driver_version"),
        ("corelib DLL", "corelib_dll_path"),
        ("corelib SHA-256", "corelib_dll_sha256"),
        ("corelib version", "corelib_version"),
        ("corelib source revision", "corelib_source_revision"),
        ("DynamicDispatch", "dynamic_dispatch_version"),
        ("RyzenMM", "ryzen_mm_version"),
        ("XRT", "xrt_version"),
        ("model directory", "model_dir"),
        ("model SHA-256", "model_sha256"),
        ("FastFlow revision", "fastflow_revision"),
        ("measured (UTC)", "utc"),
    ):
        add(f"| {label} | `{identity.get(key, 'n/a')}` |")
    add("")

    add("### Model load and TTFT")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| manifest parse and file mapping | {_ns(load.get('manifest_map_ns'))} |")
    # I-5. The shape plan is 86% of model load, and the first rendered version
    # of this table omitted it entirely -- leaving 11.2 s of a 12.8 s load
    # unexplained in the document while the finding sat only in a task report.
    # Every phase is listed, and the remainder is stated rather than left for
    # the reader to subtract.
    share = load.get("shape_plan_share")
    share_text = f" — **{share * 100:.0f}% of load**" if isinstance(
        share, (int, float)
    ) else ""
    add(
        f"| **1..4096 helper interrogation (`Phi4ShapePlan::Build`)** | "
        f"**{_ns(load.get('shape_plan_ns'))}**{share_text} |"
    )
    add(
        f"| weight pack/upload ({load.get('weight_objects', 'n/a')} objects) "
        f"| {_ns(load.get('weight_pack_ns'))} |"
    )
    add(
        f"| stream, {load.get('device_tensors', 'n/a')} device tensors, RoPE "
        f"upload | {_ns(load.get('device_setup_ns'))} |"
    )
    add(f"| unaccounted | {_ns(load.get('unaccounted_ns'))} |")
    add(f"| total model load | {_ns(load.get('total_ns'))} |")
    add(
        f"| cold TTFT ({ttft.get('prompt_token_count', 'n/a')} prompt tokens, "
        f"row extent {ttft.get('row_extent', 'n/a')}) "
        f"| {_ns(ttft.get('cold_ns'))} |"
    )
    add(
        f"| warm TTFT, same Stream after `clear_context()` "
        f"| {_ns(ttft.get('warm_ns'))} |"
    )
    add("")

    points = document.get("prefill", {}).get("points", [])
    if points:
        add("### Fresh prefill")
        add("")
        add("| rows | padded rows | wall time | tokens/s |")
        add("| ---: | ---: | ---: | ---: |")
        for point in points:
            add(
                f"| {point.get('rows')} | {point.get('padded_rows', 'n/a')} | "
                f"{_ns(point.get('ns'))} | "
                f"{point.get('tokens_per_second', float('nan')):,.1f} |"
            )
        add("")

    runs = document.get("decode", {}).get("runs", [])
    if runs:
        add("### Decode")
        add("")
        add(
            "| starting context | tokens | tokens/s | p50 | p95 | "
            "synchronizes per pass |"
        )
        add("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for run in runs:
            add(
                f"| {run.get('start_context')} | {run.get('tokens')} | "
                f"{run.get('tokens_per_second', float('nan')):,.2f} | "
                f"{_ns(run.get('p50_ns'))} | {_ns(run.get('p95_ns'))} | "
                f"{run.get('synchronizes_per_pass')} |"
            )
        add("")

    continuation = document.get("continuation", {})
    if continuation.get("points"):
        add("### Continuation routes")
        add("")
        add("| history rows | suffix | route | samples | p50 | p95 |")
        add("| ---: | ---: | --- | ---: | ---: | ---: |")
        for point in continuation["points"]:
            add(
                f"| {point.get('history_rows')} | {point.get('suffix')} | "
                f"{point.get('route')} | "
                f"{len(point.get('samples_ns', []))} | "
                f"{_ns(point.get('p50_ns'))} | {_ns(point.get('p95_ns'))} |"
            )
        add("")
        crossover = continuation.get("crossover", {})
        if crossover:
            add("#### Where append stops winning")
            add("")
            add(
                "**This is a BRACKET, not a threshold.** Append and "
                "re-prefill samples are interleaved within each point, so a "
                "machine regime shift moves both together rather than one. A "
                "point counts as decided only when the gap between the routes "
                "exceeds **both** the larger within-point p50-to-p95 spread "
                "**and** the larger drift between a route's first and last "
                "sample there; anything else widens the bracket. Task 14 has "
                "to choose this constant, so what it needs to see is how much "
                "room the measurement leaves — not a number picked because it "
                "was the last grid point where append happened to win."
            )
            add("")
            add(
                "| rendered history | append decisively wins up to | "
                "re-prefill decisively wins from | crossover lies in | "
                "margin at each bracket edge |"
            )
            add("| ---: | ---: | ---: | :--- | :--- |")
            for history, entry in crossover.items():
                lower = entry.get("append_wins_up_to")
                upper = entry.get("reprefill_wins_from")
                if not upper:
                    span = "**not bracketed by this grid**"
                elif entry.get("bracket_is_tight"):
                    span = f"exactly {upper}"
                else:
                    span = f"`({lower}, {upper}]` — not resolved further"
                # HOW FIRM EACH EDGE IS, as the gap over the uncertainty that
                # had to be beaten. Without it "decisively" is a word rather
                # than a number, and the bracket's edges are exactly where a
                # reader should be allowed to check the strength of the call.
                margins = []
                for decision in entry.get("decisions", []):
                    if decision.get("suffix") in (lower, upper):
                        ratio = decision.get("gap_over_uncertainty")
                        if isinstance(ratio, (int, float)):
                            margins.append(
                                f"{decision['suffix']}: {ratio:.1f}x"
                            )
                add(
                    f"| {history} | {lower} | {upper} | {span} | "
                    f"{', '.join(margins) if margins else 'n/a'} |"
                )
            add("")
            undecided = sum(
                1
                for entry in crossover.values()
                for decision in entry.get("decisions", [])
                if not decision.get("decided")
            )
            total = sum(
                len(entry.get("decisions", [])) for entry in crossover.values()
            )
            add(
                f"{total - undecided} of {total} sweep points were decided; "
                f"{undecided} were not and widen the brackets above. The "
                f"margin column is the gap between the routes divided by the "
                f"uncertainty it had to beat at that suffix."
            )
            add("")
            add(
                f"Prefix-monotonic: "
                f"`{continuation.get('prefix_monotonic')}`. Decision rule: "
                f"{continuation.get('decision_rule', 'n/a')}."
            )
            add("")

    add("### V scatter, memory and synchronization")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(
        f"| V cache reads per model step | "
        f"{scatter.get('reads_per_model_step')} |"
    )
    add(
        f"| per-head V writes per model step | "
        f"{scatter.get('writes_per_model_step')} |"
    )
    add(f"| V bytes transferred | {_bytes(scatter.get('bytes'))} |")
    add(f"| V scatter wall time | {_ns(scatter.get('nanoseconds'))} |")
    add(f"| FP16 embedding | {_bytes(memory.get('embedding_bytes'))} |")
    add(f"| KV cache | {_bytes(memory.get('kv_bytes'))} |")
    add(f"| corelib packed weights | {_bytes(memory.get('packed_weight_bytes'))} |")
    add(f"| scratch and device tensors | {_bytes(memory.get('scratch_bytes'))} |")
    add(f"| mapped ONNX source | {_bytes(memory.get('mapped_source_bytes'))} |")
    add(f"| peak host private bytes | {_bytes(memory.get('peak_private_bytes'))} |")
    add(
        f"| peak host working set | "
        f"{_bytes(memory.get('peak_working_set_bytes'))} |"
    )
    add("")
    add(
        "128-token stability window: device tensors created after warmup "
        f"**{memory.get('device_tensor_creates_after_warmup')}**, weight "
        f"objects **{memory.get('weight_creates_after_warmup')}**, net live "
        f"corelib objects **{memory.get('live_corelib_object_delta')}**, "
        f"private-byte growth **{_bytes(memory.get('private_bytes_growth'))}**, "
        "least-squares private-byte slope over tokens 9..128 "
        f"**{_bytes(memory.get('private_bytes_slope_per_token'))}/token**."
    )
    add("")

    add("### DETERM-3 — run-to-run logit bit-identity baseline")
    add("")
    if determinism:
        add(
            "Two runs of the same binary, same device, same explicit token "
            "IDs, proven by recorded SHA-256 to have loaded the same corelib "
            "DLL. Per `DETERM-3` the routes are reported separately and never "
            "averaged: append and re-prefill drive different row extents "
            "through the same LM-head shape."
        )
        add("")
        add(
            "| route | runs (n) | step bit-identity | run bit-identity | "
            "max abs diff: max / median | nonzero runs | "
            f"≥ {MIN_DETERMINISM_RUNS_PER_ROUTE} runs? |"
        )
        # The last column is the run-count precondition, NOT "is this a
        # settled baseline". Labelling it "baseline?" put a `yes` beside a
        # route whose window contained a hard-gate failure, two lines above
        # the blockquote saying exactly that must not be read as a baseline.
        add("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for route, entry in determinism.get("routes", {}).items():
            spread = entry.get("observed_max_abs_diff", {})
            add(
                f"| {route} | {entry.get('runs')} | "
                f"{_rate(entry.get('step_bit_identity_rate'), entry.get('steps'))}"
                f" | "
                f"{_rate(entry.get('run_bit_identity_rate'), entry.get('runs'))}"
                f" | {spread.get('max')} / {spread.get('median')} | "
                f"{spread.get('nonzero_runs')} | "
                f"{'yes' if entry.get('is_baseline') else '**no — too few runs**'} |"
            )
        add("")
        if not determinism.get("is_baseline"):
            add(
                "**This window is not a settled baseline.** The figures above "
                "are what has been measured; the problems listed below say "
                "why they cannot yet be cited as the floor `DETERM-2`'s "
                '"degrades from the recorded baseline" clause needs.'
            )
            add("")
        for route, entry in determinism.get("routes", {}).items():
            histogram = entry.get("observed_max_abs_diff", {}).get("histogram")
            if histogram:
                add(
                    f"Observed maximum absolute difference, `{route}`: "
                    + ", ".join(
                        f"`{key}` × {count}" for key, count in histogram.items()
                    )
                    + "."
                )
        if any(
            entry.get("observed_max_abs_diff", {}).get("histogram")
            for entry in determinism.get("routes", {}).values()
        ):
            add("")
        by_source: dict[str, int] = {}
        elements: list[int] = []
        steps: list[str] = []
        for entry in determinism.get("routes", {}).values():
            record = entry.get("localisation") or {}
            for source, count in (record.get("by_source") or {}).items():
                by_source[source] = by_source.get(source, 0) + count
            elements.extend(
                record.get("lm_head_input_differing_elements") or []
            )
            steps.extend(record.get("steps") or [])
        if by_source:
            add("#### Where the divergence enters — measured")
            add("")
            add(
                "At the step whose logits first differ, the harness records "
                "the exact row that was fed to the LM head in each run, so "
                "this is an observation and not an inference. "
                + ", ".join(
                    f"**{count} event(s): `{source}`**"
                    for source, count in sorted(by_source.items())
                )
                + "."
            )
            add("")
            if by_source.get("model_body"):
                width = (
                    determinism.get("routes", {})
                    .get("append", {})
                    .get("localisation", {})
                    .get("lm_head_input_elements", HIDDEN_SIZE)
                )
                add(
                    f"In every measured event the two runs fed the LM head "
                    f"**different rows** — "
                    + ", ".join(f"{value:,}" for value in sorted(elements))
                    + f" of {width:,} elements differing, at "
                    + ", ".join(f"`{step}`" for step in sorted(set(steps)))
                    + ". **The divergence therefore enters the model body, "
                    "not the LM-head dispatch.** Each run's LM head was "
                    "separately measured to be correctly rounded against its "
                    "own input, so it is faithfully transforming inputs that "
                    "already differ."
                )
                add("")
                add(
                    "**The layer at which it enters is not known.** Layer 0's "
                    "K and V have been bit-identical in the events where the "
                    "emitted tokens matched, and layer 31's have not, which "
                    "bounds it to somewhere above layer 0. Narrowing it "
                    "further needs per-layer capture, and until then this is "
                    "an open question rather than a characterised one."
                )
                add("")
        gate_failures = sum(
            entry.get("runs_with_gate_failures", 0)
            for entry in determinism.get("routes", {}).values()
        )
        if gate_failures:
            add(
                f"> **{gate_failures} run(s) in this window breached a "
                f"`DETERM-2` HARD GATE.** They are counted in the rates above "
                f"-- dropping them would bias the figures upward, and "
                f"silently -- and they are not a wobble within the recorded "
                f"tolerance. Read the run records before citing anything here "
                f"as a settled baseline."
            )
            add("")
        p_value = determinism.get("route_dependence_p")
        if determinism.get("route_dependent"):
            add(
                "**The rate is route-dependent** (Fisher exact "
                f"p = {p_value:.3g} < "
                f"{determinism.get('route_dependence_alpha')}). The two "
                "routes are reported separately above rather than pooled."
            )
            add("")
        elif determinism.get("observed_rates_differ"):
            add(
                "The two routes' observed rates differ, but **the n does not "
                "support calling the rate route-dependent** (Fisher exact "
                f"p = {p_value:.3g} against "
                f"{determinism.get('route_dependence_alpha')}). They are "
                "still reported separately, per `DETERM-3`, because pooling "
                "them would hide a difference that a larger campaign might "
                "resolve — not because this one resolved it."
            )
            add("")
        for problem in determinism.get("problems", []):
            add(f"- {problem}")
        if determinism.get("problems"):
            add("")
    else:
        add(
            "No `DETERM-1` records were supplied, so **no baseline** exists. "
            "`DETERM-2`'s \"degrades from the recorded baseline\" clause "
            "continues to enforce nothing, per `DETERM-3`."
        )
        add("")
    add(_MARKDOWN_END)
    return "\n".join(lines) + "\n"


def _splice(existing: str, section: str) -> str:
    begin = existing.find(_MARKDOWN_BEGIN)
    end = existing.find(_MARKDOWN_END)
    if begin != -1 and end != -1:
        return existing[:begin] + section + existing[end + len(_MARKDOWN_END) + 1 :]
    separator = "" if existing.endswith("\n\n") else "\n"
    return existing + separator + "\n---\n\n" + section


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Phi-4 AIE4 baseline and render it into the "
            "benchmark document."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="phi4_aie4_baseline.json written by benchmark_phi4_aie4.exe",
    )
    parser.add_argument(
        "--determinism-glob",
        required=True,
        help=(
            "glob matching the run-scoped determ1-<route>.json artifacts, for "
            "example src/build/phi4-hardware/artifacts/*/determ1-*.json. "
            "Required, not optional: DETERM-3's baseline is the point of this "
            "task, and a tool that silently rendered a report without it "
            "would reproduce the absence it exists to close."
        ),
    )
    parser.add_argument(
        "--markdown",
        required=True,
        help="benchmark document to splice the rendered section into",
    )
    parser.add_argument(
        "--output-json",
        help="write the validated document, with the determinism baseline "
        "merged in, to this path",
    )
    parser.add_argument(
        "--allow-incomplete-determinism",
        action="store_true",
        help=(
            "render even when a route has fewer than "
            f"{MIN_DETERMINISM_RUNS_PER_ROUTE} runs. The rendered section then "
            "says NO BASELINE for that route and the exit code is still "
            "non-zero; this only exists so a partial campaign can be inspected."
        ),
    )
    args = parser.parse_args(argv)

    document = json.loads(Path(args.input).read_text(encoding="utf-8"))
    problems = validate(document)

    records = load_determinism_records(args.determinism_glob)
    if not records:
        print(
            f"no DETERM-1 records matched {args.determinism_glob!r}. "
            f"DETERM-3's baseline cannot be established, and this tool will "
            f"not render a report that implies one exists.",
            file=sys.stderr,
        )
        problems.append("DETERM-3: no DETERM-1 records were found")
    determinism = determinism_baseline(records)
    document["determinism"] = determinism
    for problem in determinism["problems"]:
        problems.append(f"DETERM-3: {problem}")

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(document, indent=2), encoding="utf-8"
        )

    # Rendering is blocked by a structural problem with the baseline itself,
    # never by a finding IN it. A DETERM-2 hard-gate failure among the samples
    # is a result this document exists to carry; refusing to write the
    # document because the campaign found something would discard the
    # measurement and leave only an exit code.
    determinism_incomplete = bool(determinism["blocking_problems"]) or not records
    render_blocked = bool(
        [
            problem
            for problem in problems
            if not problem.startswith("DETERM-3:")
        ]
    ) or (determinism_incomplete and not args.allow_incomplete_determinism)

    if not render_blocked:
        target = Path(args.markdown)
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        target.write_text(
            _splice(existing, render_markdown(document)), encoding="utf-8"
        )
        print(f"rendered the baseline into {target}")

    if problems:
        print(
            "report_phi4_corelib_baseline: FAIL — "
            f"{len(problems)} problem(s):",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        if render_blocked:
            print(
                "  the benchmark document was NOT modified.",
                file=sys.stderr,
            )
        return 1

    print("report_phi4_corelib_baseline: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
