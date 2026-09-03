"""The Windows non-vcpkg Boost linkage decision, read back out of src/CMakeLists.txt.

Why this file exists rather than a comment.

Before the AIE4 branch, `src/CMakeLists.txt` made ONE decision about Boost on
the `WIN32 AND NOT VCPKG_TOOLCHAIN` path and made it in one place: it defined
`CURL_STATICLIB` / `BOOST_ALL_NO_LIB` / `BOOST_ALL_STATIC_LINK` unconditionally
and linked `libboost_program_options-vc143-mt-x64-1_88` by name. b2 encodes
linkage in the file name -- a `lib` prefix is the static archive, a bare
`boost_` name is the import library for a DLL -- so the macros and the name
said the same thing, and the shipped flm.exe had no Boost DLL dependency at
all. That is why `src/lib` vendors no `boost_program_options.dll` and
`src/wix/get_files.bat` stages none.

Commit 0db2c3ac split that one decision into two independent ones -- an
`option(FLM_WIN_STATIC_DEPS ... OFF)` choosing the macros, and a search list
choosing the name -- and set them to disagree with the shipping configuration:
the option is set ON nowhere in the tree, no preset sets a `toolchainFile` (so
`VCPKG_TOOLCHAIN` is false on every Windows preset), and CI runs
`cmake --preset windows-vs18` bare. The shipping binary's Boost linkage flipped
from static to shared with no diff anyone read as saying so, and the resulting
MSI was missing a DLL nothing in the repository produces.

Nothing offline could have caught that, because the defect is not in any single
line: every line was defensible, and the pair was wrong. So the assertions here
are about the RELATIONSHIP between the two halves --

  * the option's default is the configuration CI and the MSI actually ship;
  * for EACH value of the option, the macro set and the first-searched library
    name describe the same linkage; and
  * a configure-time `FATAL_ERROR` exists to catch the residual case the source
    cannot rule out, where the search lands on the other variant anyway.

-- plus a non-vacuity test, because a parser that quietly finds nothing would
make every one of those pass.

Set FLM_TEST_WIN_LINKAGE_CMAKELISTS to point the parse at a scratch copy; that
is how these assertions are themselves verified against mutated inputs.
FLM_TEST_WIN_LINKAGE_PRESETS does the same for src/CMakePresets.json.

WHY THE PRESET FILE IS READ HERE TOO
------------------------------------
The assertions above are about `option(FLM_WIN_STATIC_DEPS ... ON)`, and a
default is only the shipping value while nothing overrides it. CI runs
`cmake --preset windows-vs18`, so a single `"FLM_WIN_STATIC_DEPS": "OFF"` in a
`cacheVariables` block -- on that preset or on anything it inherits from --
re-breaks CRIT-3 exactly, with the option line untouched and every assertion
above still green. That vector was demonstrated and this file did not see it,
because it never opened CMakePresets.json. It does now. The same goes for a
later `set(FLM_WIN_STATIC_DEPS OFF ...)` in CMakeLists.txt, which reaches the
same end without touching the `option()` call either.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_CMAKELISTS = _REPO_ROOT / "src" / "CMakeLists.txt"
_DEFAULT_PRESETS = _REPO_ROOT / "src" / "CMakePresets.json"

# CMake's false constants, which is what a re-break would spell.
_CMAKE_FALSE = {"OFF", "FALSE", "N", "NO", "0", "", "IGNORE", "NOTFOUND"}

# b2's spellings. A `lib` prefix is the static archive; the bare name is the
# import library for boost_program_options.dll.
_STATIC_NAME = re.compile(r"^libboost_program_options")
_SHARED_NAME = re.compile(r"^boost_program_options$")

_STATIC_MACROS = {"CURL_STATICLIB", "BOOST_ALL_NO_LIB", "BOOST_ALL_STATIC_LINK"}
_SHARED_MACROS = {"BOOST_ALL_NO_LIB", "BOOST_ALL_DYN_LINK"}


class CMakeParseError(AssertionError):
    """Raised when the file no longer has the shape this test reads.

    Deliberately an AssertionError: a restructure that defeats the parse is a
    test failure, not a silent pass.
    """


def _strip_comments(text: str) -> str:
    """Drop whole-line CMake comments.

    Needed because the shared branch explains itself by NAMING the macros and
    library spellings it is not using; a token scan over the raw text would
    read those explanations as code.
    """
    out = []
    for line in text.splitlines():
        out.append("" if line.lstrip().startswith("#") else line)
    return "\n".join(out)


def _if_else_bodies(code: str, header: str, start: int = 0) -> tuple[str, str, int]:
    """Split the `if(<header>) ... else() ... endif()` starting at/after `start`.

    Returns (if_body, else_body, index_of_endif). Counts nesting so an inner
    `if()` cannot terminate the outer block.
    """
    lines = code.splitlines(keepends=True)
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    begin = None
    for idx, line in enumerate(lines):
        if offsets[idx] < start:
            continue
        if line.strip() == header:
            begin = idx
            break
    if begin is None:
        raise CMakeParseError(f"no line reading exactly {header!r} found")

    depth = 0
    in_else = False
    if_body: list[str] = []
    else_body: list[str] = []
    for idx in range(begin, len(lines)):
        stripped = lines[idx].strip()
        if idx == begin:
            depth = 1
            continue
        if stripped == "endif()":
            depth -= 1
            if depth == 0:
                return "".join(if_body), "".join(else_body), offsets[idx]
            (else_body if in_else else if_body).append(lines[idx])
            continue
        if stripped.startswith("if("):
            depth += 1
        if depth == 1 and stripped == "else()":
            in_else = True
            continue
        (else_body if in_else else if_body).append(lines[idx])
    raise CMakeParseError(f"{header} is never closed by endif()")


def _matching_endforeach(code: str, start: int) -> int:
    depth = 0
    pos = start
    for match in re.finditer(r"^\s*(foreach\(|endforeach\(\))", code[start:], re.M):
        if match.group(1) == "foreach(":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return start + match.end()
    raise CMakeParseError("the Windows dependency foreach is never closed")


class WindowsBoostLinkage:
    """What src/CMakeLists.txt decides about Boost, per FLM_WIN_STATIC_DEPS."""

    def __init__(self, text: str) -> None:
        self.raw = text
        code = _strip_comments(text)
        self.code = code

        option = re.search(
            r"option\(\s*FLM_WIN_STATIC_DEPS\s+"
            r'"(?P<doc>[^"]*)"\s+(?P<default>[A-Za-z0-9_]+)\s*\)',
            code,
        )
        if option is None:
            raise CMakeParseError(
                "option(FLM_WIN_STATIC_DEPS \"...\" <default>) not found"
            )
        self.option_default = option.group("default")
        self.option_doc = option.group("doc")

        # Half one: the linkage macros. The `if(WIN32 AND NOT VCPKG_TOOLCHAIN)`
        # wrapper is required, because these macros are only correct there.
        guard = code.find("if(WIN32 AND NOT VCPKG_TOOLCHAIN)")
        if guard == -1:
            raise CMakeParseError("no if(WIN32 AND NOT VCPKG_TOOLCHAIN) block")
        static_body, shared_body, _ = _if_else_bodies(
            code, "if(FLM_WIN_STATIC_DEPS)", start=guard
        )
        self.static_macros = self._definitions(static_body, "static")
        self.shared_macros = self._definitions(shared_body, "shared")

        # Half two: the searched library names, per branch.
        entry = re.search(
            r'"required\|boost_program_options\|(?P<names>[^"]*)"', code
        )
        if entry is None:
            raise CMakeParseError(
                "no 'required|boost_program_options|...' dependency entry"
            )
        self.dependency_entry_names = entry.group("names")
        self.static_names = self._names_for(True)
        self.shared_names = self._names_for(False)

        self._locate_guard()

    @staticmethod
    def _definitions(body: str, which: str) -> frozenset[str]:
        call = re.search(
            r"target_compile_definitions\(\s*flm\s+PUBLIC(?P<args>[^)]*)\)", body
        )
        if call is None:
            raise CMakeParseError(
                f"the {which} branch defines no macros on target flm"
            )
        return frozenset(call.group("args").split())

    def _variables(self) -> dict[str, str]:
        found = {}
        for name in ("_flm_boost_po_static", "_flm_boost_po_shared"):
            match = re.search(
                rf'set\(\s*{name}\s+"(?P<value>[^"]*)"\s*\)', self.code
            )
            if match is not None:
                found[name] = match.group("value")
        return found

    def _names_for(self, static: bool) -> list[str]:
        """The ordered spellings searched when FLM_WIN_STATIC_DEPS is/isn't ON.

        Handles both shapes a regression could take: the list derived from the
        option (today), and a hard-coded literal list (the pre-fix shape, which
        must then read the same for both branches and so fail the agreement
        test rather than the parse).
        """
        raw = self.dependency_entry_names
        if "${_flm_boost_po_names}" not in raw:
            return [part for part in raw.split(";") if part]

        static_body, shared_body, _ = _if_else_bodies(
            self.code, "if(FLM_WIN_STATIC_DEPS)", start=self.code.find(
                "_flm_boost_po_static"
            )
        )
        body = static_body if static else shared_body
        assign = re.search(
            r'set\(\s*_flm_boost_po_names\s+"(?P<value>[^"]*)"\s*\)', body
        )
        if assign is None:
            raise CMakeParseError(
                "_flm_boost_po_names is not assigned in the "
                f"{'static' if static else 'shared'} branch"
            )
        value = assign.group("value")
        for name, expansion in self._variables().items():
            value = value.replace("${" + name + "}", expansion)
        if "${" in value:
            raise CMakeParseError(
                f"unresolved variable in the boost name list: {value!r}"
            )
        return [part for part in value.split(";") if part]

    def _locate_guard(self) -> None:
        """Find the configure-time consistency guard and record what it says."""
        loop = self.code.find("foreach(_flm_dep IN ITEMS")
        if loop == -1:
            raise CMakeParseError("the Windows dependency foreach is missing")
        after_loop = _matching_endforeach(self.code, loop)
        link = self.code.find(
            "target_link_libraries(flm PUBLIC ${_flm_win_libs})", after_loop
        )
        if link == -1:
            raise CMakeParseError(
                "the resolved Windows libraries are never linked"
            )
        region = self.code[after_loop:link]
        self.guard_region = region
        self.guard_reads_resolved_library = (
            "FLM_LIB_boost_program_options" in region
        )
        self.guard_fatal_messages = [
            match.group("body")
            for match in re.finditer(
                r"message\(\s*FATAL_ERROR(?P<body>.*?)\)\s*$",
                region,
                re.S | re.M,
            )
        ]
        self.guard_warning_count = len(
            re.findall(r"message\(\s*WARNING", region)
        )


def _preset_overrides() -> list[tuple[str, str]]:
    """Every (preset, value) that sets FLM_WIN_STATIC_DEPS in CMakePresets.json.

    Both spellings of a cache variable are handled: the plain string form and
    the `{"type": ..., "value": ...}` form.
    """
    path = pathlib.Path(
        os.environ.get("FLM_TEST_WIN_LINKAGE_PRESETS", _DEFAULT_PRESETS)
    )
    if not path.is_file():
        raise CMakeParseError(f"{path} does not exist")
    document = json.loads(path.read_text(encoding="utf-8"))
    found = []
    for preset in document.get("configurePresets", []):
        variables = preset.get("cacheVariables") or {}
        if "FLM_WIN_STATIC_DEPS" not in variables:
            continue
        entry = variables["FLM_WIN_STATIC_DEPS"]
        value = entry.get("value") if isinstance(entry, dict) else entry
        found.append((str(preset.get("name", "<unnamed>")), str(value)))
    return found


def _preset_names() -> list[str]:
    path = pathlib.Path(
        os.environ.get("FLM_TEST_WIN_LINKAGE_PRESETS", _DEFAULT_PRESETS)
    )
    if not path.is_file():
        raise CMakeParseError(f"{path} does not exist")
    document = json.loads(path.read_text(encoding="utf-8"))
    return [
        str(preset.get("name", ""))
        for preset in document.get("configurePresets", [])
    ]


def _load() -> WindowsBoostLinkage:
    path = pathlib.Path(
        os.environ.get("FLM_TEST_WIN_LINKAGE_CMAKELISTS", _DEFAULT_CMAKELISTS)
    )
    if not path.is_file():
        raise CMakeParseError(f"{path} does not exist")
    return WindowsBoostLinkage(path.read_text(encoding="utf-8"))


class WindowsDependencyLinkageParseTest(unittest.TestCase):
    """Non-vacuity. Every other test here is a claim about parsed content."""

    def test_every_part_of_the_decision_was_actually_found(self) -> None:
        parsed = _load()
        self.assertIn(
            parsed.option_default,
            {"ON", "OFF"},
            "FLM_WIN_STATIC_DEPS has no boolean default; the parse below is "
            "reading something else",
        )
        self.assertTrue(parsed.static_macros, "static branch macros not parsed")
        self.assertTrue(parsed.shared_macros, "shared branch macros not parsed")
        self.assertTrue(parsed.static_names, "static boost name list not parsed")
        self.assertTrue(parsed.shared_names, "shared boost name list not parsed")
        self.assertTrue(
            parsed.guard_region.strip(),
            "nothing at all sits between the dependency loop and the link "
            "call; the consistency guard cannot be there",
        )

    def test_the_two_branches_are_not_the_same_branch(self) -> None:
        # A "fix" that makes both arms identical would satisfy several of the
        # assertions below while restoring the defect.
        parsed = _load()
        self.assertNotEqual(
            parsed.static_macros,
            parsed.shared_macros,
            "the static and shared arms define the same macros",
        )
        self.assertNotEqual(
            parsed.static_names,
            parsed.shared_names,
            "the static and shared arms search the same name order, so the "
            "option no longer selects a linkage",
        )


class WindowsDependencyLinkageDefaultTest(unittest.TestCase):
    def test_static_is_the_default(self) -> None:
        parsed = _load()
        self.assertEqual(
            parsed.option_default,
            "ON",
            "FLM_WIN_STATIC_DEPS must default ON. It is set ON nowhere in the "
            "tree, no CMake preset sets a toolchainFile (so VCPKG_TOOLCHAIN is "
            "false on every Windows preset), and CI runs "
            "`cmake --preset windows-vs18` bare -- so this default IS the "
            "linkage of the shipped binary and the MSI. OFF makes flm.exe "
            "depend on boost_program_options.dll, which src/lib does not "
            "vendor and src/wix/get_files.bat does not stage.",
        )

    def test_the_default_configuration_links_boost_statically(self) -> None:
        # Stated as the consequence rather than the setting: this is what the
        # packaging in src/lib and src/wix/get_files.bat assumes.
        parsed = _load()
        default_static = parsed.option_default == "ON"
        macros = parsed.static_macros if default_static else parsed.shared_macros
        names = parsed.static_names if default_static else parsed.shared_names
        self.assertIn(
            "BOOST_ALL_STATIC_LINK",
            macros,
            "the default configure does not declare static Boost linkage",
        )
        self.assertRegex(
            names[0],
            _STATIC_NAME,
            "the default configure prefers a Boost import library over the "
            "static archive, so the shipped flm.exe would import "
            "boost_program_options.dll",
        )


class WindowsDependencyLinkageAgreementTest(unittest.TestCase):
    """The macros and the name preference must describe the same linkage."""

    def test_static_branch(self) -> None:
        parsed = _load()
        self.assertEqual(
            parsed.static_macros,
            _STATIC_MACROS,
            "FLM_WIN_STATIC_DEPS=ON must define exactly CURL_STATICLIB, "
            "BOOST_ALL_NO_LIB and BOOST_ALL_STATIC_LINK",
        )
        self.assertRegex(
            parsed.static_names[0],
            _STATIC_NAME,
            "FLM_WIN_STATIC_DEPS=ON declares static Boost linkage but searches "
            f"{parsed.static_names[0]!r} first -- b2's bare name is the import "
            "library for boost_program_options.dll, not an archive",
        )

    def test_shared_branch(self) -> None:
        parsed = _load()
        self.assertEqual(
            parsed.shared_macros,
            _SHARED_MACROS,
            "FLM_WIN_STATIC_DEPS=OFF must define exactly BOOST_ALL_NO_LIB and "
            "BOOST_ALL_DYN_LINK (NO_LIB stays: MSVC auto-linking would "
            "otherwise ask for the decorated static name)",
        )
        self.assertNotIn(
            "CURL_STATICLIB",
            parsed.shared_macros,
            "CURL_STATICLIB against an import library removes the "
            "__declspec(dllimport) that library expects",
        )
        self.assertRegex(
            parsed.shared_names[0],
            _SHARED_NAME,
            "FLM_WIN_STATIC_DEPS=OFF declares BOOST_ALL_DYN_LINK but searches "
            f"{parsed.shared_names[0]!r} first -- a 'lib'-prefixed archive, "
            "which exports no data symbols and fails the link on "
            "boost::program_options::arg",
        )

    def test_both_spellings_remain_reachable_as_fallbacks(self) -> None:
        # Order carries the preference; dropping the other spelling would turn
        # a prefix that ships only one variant into a configure failure, and
        # would make the FATAL_ERROR guard unreachable.
        parsed = _load()
        for label, names in (
            ("static", parsed.static_names),
            ("shared", parsed.shared_names),
        ):
            with self.subTest(branch=label):
                self.assertTrue(
                    any(_STATIC_NAME.match(name) for name in names),
                    f"{label} branch never searches the static archive name",
                )
                self.assertTrue(
                    any(_SHARED_NAME.match(name) for name in names),
                    f"{label} branch never searches the import library name",
                )


class WindowsDependencyLinkageGuardTest(unittest.TestCase):
    """The configure-time check that the resolved library matches the macros."""

    def test_guard_is_present_and_fatal(self) -> None:
        parsed = _load()
        self.assertTrue(
            parsed.guard_reads_resolved_library,
            "nothing between the dependency loop and the link call inspects "
            "FLM_LIB_boost_program_options, so a search that resolves to the "
            "wrong variant -- or a stale cache entry, which find_library never "
            "re-searches -- is not detected",
        )
        self.assertGreaterEqual(
            len(parsed.guard_fatal_messages),
            2,
            "the guard must FATAL_ERROR on both contradictions: a "
            "'lib'-prefixed archive under BOOST_ALL_DYN_LINK, and a bare "
            f"'boost_' import library under BOOST_ALL_STATIC_LINK. Found "
            f"{len(parsed.guard_fatal_messages)} FATAL_ERROR message(s) and "
            f"{parsed.guard_warning_count} WARNING(s) -- a warning in a "
            "thousand-line configure log is what was missed the first time.",
        )

    def test_guard_names_both_halves_and_the_way_out(self) -> None:
        parsed = _load()
        joined = " ".join(parsed.guard_fatal_messages)
        for token in (
            "FLM_WIN_STATIC_DEPS",
            "BOOST_ALL_STATIC_LINK",
            "BOOST_ALL_DYN_LINK",
        ):
            with self.subTest(token=token):
                self.assertIn(
                    token,
                    joined,
                    f"the guard's diagnostics never mention {token}; the "
                    "reader cannot tell which of the two halves to change",
                )
        self.assertIn(
            "${FLM_LIB_boost_program_options}",
            joined,
            "the guard does not report which library it actually resolved",
        )


class WindowsDependencyLinkageEffectiveValueTest(unittest.TestCase):
    """Nothing between the `option()` line and the configure may flip it OFF.

    `test_static_is_the_default` reads the option's default and stops there. A
    default is only the shipping value while nothing overrides it, and the two
    cheapest overrides -- a preset cache variable and a later `set()` -- leave
    the option line byte-identical. Both were applied to this tree and the
    suite stayed green; these are the assertions that turn them red.
    """

    def test_no_preset_turns_the_option_off(self) -> None:
        offenders = [
            f"{name}={value}"
            for name, value in _preset_overrides()
            if value.strip().upper() in _CMAKE_FALSE
        ]
        self.assertEqual(
            offenders,
            [],
            "a CMake preset sets FLM_WIN_STATIC_DEPS to a false value "
            f"({', '.join(offenders)}). CI runs `cmake --preset windows-vs18` "
            "bare, so a preset cache variable IS the shipping linkage: this "
            "makes flm.exe import boost_program_options.dll, which src/lib "
            "does not vendor and src/wix/get_files.bat does not stage. "
            "cacheVariables are inherited, so a false value on ANY preset in "
            "this file is reported here.",
        )

    def test_the_preset_file_was_actually_read(self) -> None:
        # Non-vacuity. A missing, empty or restructured preset file would make
        # the assertion above pass by finding nothing -- which is precisely
        # how this file failed to see the preset re-break before.
        names = _preset_names()
        self.assertIn(
            "windows-vs18",
            names,
            "src/CMakePresets.json has no `windows-vs18` configure preset, so "
            "the preset scan is not reading the file CI invokes. Found: "
            f"{names}",
        )

    def test_cmakelists_does_not_reassign_the_option(self) -> None:
        # `set(FLM_WIN_STATIC_DEPS OFF CACHE BOOL "" FORCE)` after the option,
        # or a plain `set(FLM_WIN_STATIC_DEPS OFF)` before it (CMP0077 makes
        # `option()` honour an existing normal variable), both reach the same
        # place without editing the `option()` call.
        parsed = _load()
        assignments = re.findall(
            r"^\s*set\(\s*FLM_WIN_STATIC_DEPS\b[^)]*\)",
            parsed.code,
            re.M,
        )
        self.assertEqual(
            assignments,
            [],
            "src/CMakeLists.txt assigns FLM_WIN_STATIC_DEPS outside its "
            f"option() declaration: {assignments}. Under CMP0077 that "
            "overrides the default the tests above read, so the option line "
            "no longer tells you the shipping linkage.",
        )


if __name__ == "__main__":
    unittest.main()
