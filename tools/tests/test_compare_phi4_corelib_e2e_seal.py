"""The `_DETERM2_SEALED` import-time seal, exercised without hardware.

Nothing else in the repository does. `src/tools/compare_phi4_corelib_e2e.py`
is invoked only from `run_hardware_suite.ps1`, which needs an AIE4 device and
the real model, so until now the seal was verified by hand and by review and
by nothing that runs. A seal whose failure path is never executed is a comment.

Each test edits the constant the way somebody would after grepping it out of a
failure message -- in a copy, then executing the real file -- and requires the
import to die. The seal has to hold for the design 12.4 acceptance thresholds
as well as the DETERM-2 bounds: those two were outside it while their values
were printed verbatim in four failure messages, which is the entire path the
seal exists to block.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


COMPARATOR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tools"
    / "compare_phi4_corelib_e2e.py"
)

# name -> (source line as committed, the same line with the number moved)
#
# The "after" values are the plausible edits, not absurd ones: loosening the
# correlation, shortening the decode run, doubling either bound.
TAMPERS = {
    "MAX_TOP32_ABS_DIFF": (
        "MAX_TOP32_ABS_DIFF = 0.25",
        "MAX_TOP32_ABS_DIFF = 0.5",
    ),
    "RUN_TO_RUN_MAX_ULPS": (
        "RUN_TO_RUN_MAX_ULPS = 2",
        "RUN_TO_RUN_MAX_ULPS = 4",
    ),
    "MIN_CORRELATION": (
        "MIN_CORRELATION = 0.9999",
        "MIN_CORRELATION = 0.99",
    ),
    "MIN_DECODE_STEPS": (
        "MIN_DECODE_STEPS = 16",
        "MIN_DECODE_STEPS = 4",
    ),
    "TOP_K": (
        "TOP_K = 32",
        "TOP_K = 2",
    ),
}

# Every `{NAME}` an f-string in the comparator interpolates, where NAME looks
# like a module constant. DERIVED, never typed.
#
# The previous version of the test below compared the seal against `TAMPERS`,
# a literal in this file, and then claimed to be "the assertion that would have
# caught this omission in the first place". It was not: the universe and the
# expectation were the same hand-maintained list, so an unsealed constant was
# invisible to it by construction -- and there was one in the tree, `TOP_K`,
# whose value is interpolated into the union-top-k failure message and which
# could be dropped from 32 to 2 with the whole module green. Deriving the
# universe from the source is the only shape in which this assertion means
# what it says.
_INTERPOLATED_CONSTANT = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")


def _execute(source: str) -> dict[str, object]:
    """Run the comparator's module body, as import would.

    `__name__` is not `__main__`, so the `main()` entry point at the bottom
    does not fire and nothing touches a device: only the import-time checks
    run, which is exactly what is under test.
    """
    namespace: dict[str, object] = {"__name__": "_determ2_seal_probe"}
    exec(compile(source, str(COMPARATOR), "exec"), namespace)
    return namespace


class Determ2SealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Newlines normalised, because the tampers below are `\n`-joined
        # literals. `core.autocrlf` is true by default on Windows and
        # `.gitattributes` exempts only src/model_overlays, so a fresh clone
        # gets this file with CRLF -- and every tamper silently stops matching,
        # turning the whole seal suite red for a reason that has nothing to do
        # with the seal. Observed on this tree.
        cls.source = (
            COMPARATOR.read_bytes()
            .decode("utf-8")
            .replace("\r\n", "\n")
        )

    def test_the_committed_comparator_imports(self):
        namespace = _execute(self.source)
        sealed = namespace["_DETERM2_SEALED"]
        for name, (value, reason) in sealed.items():
            self.assertEqual(namespace[name], value)
            self.assertTrue(reason, f"{name} has no sealed reason")

    def test_every_threshold_quoted_in_a_failure_message_is_sealed(self):
        # The alias incident, generalised: a constant whose value reaches a
        # failure message is a constant somebody will grep and edit, so it
        # must be in the dict.
        #
        # The universe is ENUMERATED OUT OF THE SOURCE, not compared against a
        # list in this file. A literal expectation can only ever agree with
        # itself, which is how `TOP_K` sat unsealed underneath a test that
        # claimed to forbid exactly that.
        namespace = _execute(self.source)
        sealed = set(namespace["_DETERM2_SEALED"])
        quoted = {
            name
            for name in _INTERPOLATED_CONSTANT.findall(self.source)
            if isinstance(namespace.get(name), (int, float))
            and not isinstance(namespace.get(name), bool)
        }
        # Non-vacuity: a regex that stopped matching would make the next
        # assertion pass by finding nothing to require.
        self.assertGreaterEqual(
            len(quoted),
            len(TAMPERS),
            f"only {sorted(quoted)} were found interpolated into the "
            "comparator's messages; the scan is not reading the file",
        )
        self.assertEqual(
            quoted - sealed,
            set(),
            f"{sorted(quoted - sealed)} reach a failure message but are not "
            "in _DETERM2_SEALED. That is the grep-the-message-and-edit path "
            "the seal exists to block: add each to _DETERM2_SEALED with the "
            "reason its number cannot move, and to TAMPERS here.",
        )
        # And every sealed constant must have a tamper case below, or the
        # seal's failure path is unexercised for it.
        self.assertEqual(sealed, set(TAMPERS))
        for name in TAMPERS:
            self.assertIn(
                f"{{{name}}}",
                self.source,
                f"{name} is sealed but no message quotes it; if it stopped "
                "being user-visible, say so rather than leaving this stale",
            )

    def test_moving_a_sealed_constant_stops_the_tool(self):
        for name, (before, after) in TAMPERS.items():
            with self.subTest(constant=name):
                self.assertEqual(
                    self.source.count(before),
                    1,
                    f"{before!r} is not a unique line; the tamper would "
                    "not be testing what it claims to",
                )
                with self.assertRaises(SystemExit) as caught:
                    _execute(self.source.replace(before, after))
                message = str(caught.exception)
                self.assertIn(f"DETERM-2: {name} has been changed", message)
                # The message must lead back to the seal, not to the constant.
                self.assertIn("_DETERM2_SEALED", message)

    def test_editing_the_seal_to_match_is_still_caught_at_the_other_end(self):
        # Editing BOTH halves is a deliberate act and the seal lets it
        # through by design -- but only when they agree. Moving the entry
        # without the constant is the half-finished version of that act and
        # it must not start.
        source = self.source.replace(
            '        0.25,\n        "the largest cross-implementation',
            '        0.5,\n        "the largest cross-implementation',
        )
        self.assertNotEqual(source, self.source)
        with self.assertRaises(SystemExit) as caught:
            _execute(source)
        self.assertIn("MAX_TOP32_ABS_DIFF", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
