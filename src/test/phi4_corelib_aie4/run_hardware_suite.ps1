# Task 12 Step 8: the single entry point for the AIE4 hardware acceptance run.
#
# Everything this suite needs to do on the target is here rather than in a
# session transcript, because a sequence that only exists in somebody's shell
# history is not reproducible and cannot be reviewed.
#
# The ordering is load-bearing, not stylistic. Two processes holding AIE4
# device contexts at once fail in ways that look like defects, so the Python
# reference is run to completion -- including corelib.cleanup() -- before the
# C++ harness starts, and the harness finishes before the next reference run
# begins. Nothing here runs in parallel.
#
# Skipped work is reported as SKIPPED, never as passed, and exit code 77 is
# CTest's SKIP_RETURN_CODE so a partial run reads as Skipped rather than
# Passed.

[CmdletBinding()]
param(
    # The accepted OGA DML Phi-4 package. Without it the numeric and boundary
    # blocks cannot run; the device smoke and the fatal child still can.
    [string]$ModelDir,

    # Directory holding the ryzenai_corelib.dll under test, with its derived
    # runtime closure staged beside it. Design CLOSURE-1: derive that closure
    # with cmake -P cmake/StageAie4Runtime.cmake against the exact shipped
    # DLL. Never transcribe it.
    [string]$CorelibRuntimeDir,

    # Additional directories completing the closure, if the staged directory
    # is not self-contained. Semicolon-separated.
    [string]$DependencyDir = "",

    # The ryzenai-corelib checkout whose python/ holds phi4_driver.py. Read
    # only: no --continuation-route option is added to that driver and that
    # repository is never modified.
    [string]$CorelibSource = $env:RYZENAI_CORELIB_SOURCE,

    # Build inputs. These have no usable defaults on the AIE4 target: the
    # suite CMakeLists falls back to C:/dev paths that exist on the original
    # development box and nowhere else, and its find_path calls are REQUIRED,
    # so a wrong guess fails configure loudly rather than building something
    # subtly different.
    [string]$XrtDir,
    [string]$CorelibIncludeDir,
    [string]$BoostIncludeDir,

    # The product binary. When supplied, Step 7's four generation endpoints run
    # against a real `flm serve`; when absent, Step 7 is reported as skipped
    # rather than quietly omitted.
    [string]$FlmExe,

    [string]$BuildDir,
    [string]$Cmake,
    [string]$Python = "python",
    [int]$DecodeSteps = 16,

    # Rows the boundary sweep runs REAL prefills at. Off by default because a
    # 4096-row prefill is the slowest thing in the suite; the helper-table
    # half of Step 5 runs unconditionally inside test_phi4_hardware.
    [switch]$BoundarySweep
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$failures = @()
$suiteDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Nested rather than a three-argument Join-Path: Windows PowerShell 5.1 is the
# shell on the AIE4 target and its Join-Path takes only -Path and -ChildPath,
# so the multi-segment form binds the extra segments positionally and fails
# with a parameter-binding error before the script has done anything.
$sourceDir = (Resolve-Path (Join-Path (Join-Path $suiteDir '..') '..')).Path
$ran = @()
$skipped = @()

function Write-Section {
    param([string]$Name)
    Write-Output ''
    Write-Output "=== $Name ==="
}

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$Exe,
        [string[]]$Arguments
    )
    Write-Output "> $Exe $($Arguments -join ' ')"
    & $Exe @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "$Label failed with exit code $code"
    }
}

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

if (-not $Cmake) {
    # The Cygwin cmake on PATH cannot drive a Visual Studio generator. Prefer
    # the one Visual Studio ships, and only fall back to PATH if it is absent.
    $bundled = Join-Path $env:ProgramFiles ('Microsoft Visual Studio\2022\Community\Common7\IDE\' +
        'CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe')
    if (Test-Path $bundled) {
        $Cmake = $bundled
    } else {
        $Cmake = 'cmake'
    }
}
if (-not $BuildDir) {
    $BuildDir = Join-Path $sourceDir 'build/phi4-hardware'
}

if (-not $CorelibRuntimeDir) {
    Write-Output 'run_hardware_suite: SKIPPED -- -CorelibRuntimeDir is required.'
    Write-Output '  Stage the derived closure first, for example:'
    Write-Output '    cmake -DFLM_AIE4_CORELIB_DIR=<corelib install bin> \'
    Write-Output '          -DFLM_AIE4_XRT_DIR=<xrt> \'
    Write-Output '          -DFLM_AIE4_EXTRA_DIRS=<dd install bin> \'
    Write-Output '          -DFLM_AIE4_DESTINATION=<staged dir> \'
    Write-Output '          -P src/cmake/StageAie4Runtime.cmake'
    exit 77
}
$CorelibRuntimeDir = (Resolve-Path $CorelibRuntimeDir).Path
$corelibDll = Join-Path $CorelibRuntimeDir 'ryzenai_corelib.dll'
if (-not (Test-Path $corelibDll)) {
    throw "no ryzenai_corelib.dll in $CorelibRuntimeDir"
}

$hash = (Get-FileHash -Algorithm SHA256 -Path $corelibDll).Hash
Write-Section 'Runtime under test'
Write-Output "corelib   : $corelibDll"
Write-Output "sha256    : $hash"
Write-Output "size      : $((Get-Item $corelibDll).Length) bytes"
Get-ChildItem $CorelibRuntimeDir -Filter *.dll |
    Sort-Object Name |
    ForEach-Object { Write-Output ("staged    : {0} ({1} bytes)" -f $_.Name, $_.Length) }

# ---------------------------------------------------------------------------
# Configure and build
# ---------------------------------------------------------------------------

Write-Section 'Configure and build'
$configureArgs = @(
    '-S', $suiteDir,
    '-B', $BuildDir,
    '-G', 'Visual Studio 17 2022',
    '-A', 'x64',
    "-DRYZENAI_CORELIB_RUNTIME_DIR=$CorelibRuntimeDir"
)
if ($DependencyDir) {
    $configureArgs += "-DRYZENAI_CORELIB_EXTRA_DLL_DIRS=$DependencyDir"
}
if ($XrtDir) {
    $XrtDir = (Resolve-Path $XrtDir).Path
    $configureArgs += "-DXRT_INCLUDE_DIR=$XrtDir/include"
    $configureArgs += "-DXRT_LIB_DIR=$XrtDir/lib"
}
if ($CorelibIncludeDir) {
    $configureArgs += "-DRYZENAI_CORELIB_INCLUDE_DIR=$((Resolve-Path $CorelibIncludeDir).Path)"
}
if ($BoostIncludeDir) {
    $configureArgs += "-DBOOST_INCLUDE_DIR=$((Resolve-Path $BoostIncludeDir).Path)"
}
Invoke-Checked 'configure' $Cmake $configureArgs
Invoke-Checked 'build' $Cmake @('--build', $BuildDir, '--config', 'Release')

$binDir = Join-Path $BuildDir 'Release'
$e2eExe = Join-Path $binDir 'test_phi4_e2e.exe'

# ---------------------------------------------------------------------------
# Standalone suite, then the device smoke, then the fatal child
# ---------------------------------------------------------------------------

Write-Section 'CTest suite (host tests, real-corelib check, device smoke, fatal child)'
# The hardware tests skip unless this is set, so that a development box with a
# configured runtime directory but no AIE4 device does not go red for a reason
# that is not a defect. With it set, an absent device context is a hard
# failure -- which is the behaviour that matters on the one machine where this
# script is meant to run.
$env:FLM_AIE4_HARDWARE = '1'
# The ctest beside the chosen cmake, never the one on PATH. On the AIE4 target
# PATH finds Cygwin's ctest, which reads a Windows -DVALUE --test-dir as a
# POSIX path, silently reports "Test project /cygdrive/c/Users/chiz" and then
# "No tests were found" -- a result that looks like an empty suite rather than
# like a wrong tool.
$ctest = Join-Path (Split-Path -Parent $Cmake) 'ctest.exe'
if (-not (Test-Path $ctest)) { $ctest = 'ctest' }
# Serial on purpose: several of these hold an AIE4 device context.
#
# A ctest failure is recorded and the run continues to the numeric goldens.
# Those are the most expensive and least reproducible part of the acceptance,
# and aborting before them because an unrelated host test regressed would cost
# a whole hardware session for no reason.
$ctestLog = Join-Path $BuildDir 'ctest-output.txt'
# -V, not --output-on-failure.
#
# ctest shows a test's stdout only when it FAILS, and test_packaged_runtime
# reports SKIPPED rather than failing. Its RAN/SKIPPED lines -- the only place
# that says whether CLOSURE-1 and CLOSURE-2 executed -- were therefore never
# captured. Verbose output goes to the log; the console gets the result lines
# and, on failure, the log path.
try {
    & $ctest --test-dir $BuildDir -C Release -V --no-tests=error *>&1 |
        Out-File -FilePath $ctestLog -Encoding utf8
    $ctestExit = $LASTEXITCODE
    Get-Content $ctestLog |
        Where-Object { $_ -match '^\s*\d+/\d+ Test|tests passed|The following tests' } |
        ForEach-Object { Write-Output $_ }
    if ($ctestExit -ne 0) {
        throw "ctest failed with exit code $ctestExit; full log at $ctestLog"
    }
    $ran += 'ctest suite'
} catch {
    Write-Output "CTEST FAILED: $($_.Exception.Message)"
    $failures += 'ctest suite'
}

# A green ctest is not the same as "the hardware tests ran".
#
# test_phi4_hardware and test_fatal_child report SKIPPED via exit code 77 when
# their preconditions are absent, and ctest counts a skip as not-failed. So the
# suite could report 100% passed on a machine where nothing touched the NPU.
# Require them by name.
$ctestText = if (Test-Path $ctestLog) { Get-Content $ctestLog -Raw } else { '' }
foreach ($required in @('test_phi4_hardware', 'test_fatal_child')) {
    if ($ctestText -match [regex]::Escape($required) + '\s*\.+\s*\*\*\*Skipped') {
        Write-Output "REQUIRED TEST SKIPPED: $required"
        $failures += "$required was skipped, so no hardware check ran"
    } elseif ($ctestText -match [regex]::Escape($required) + '\s*\.+\s*Passed') {
        $ran += "$required (on the AIE4 device)"
    } else {
        Write-Output "REQUIRED TEST DID NOT REPORT: $required"
        $failures += "$required produced no recognizable ctest result"
    }
}

# CLOSURE-1 and CLOSURE-2 live inside test_packaged_runtime, which legitimately
# reports SKIPPED overall because its flm.exe block needs an executable it may
# not have been given. That skip hides which of its blocks DID run, so the
# closure derivation and the negative control were invisible in the suite's
# output -- present or absent looked identical. Require the block by name from
# the RAN lines it prints.
if ($ctestText -match 'real-closure \(CLOSURE-1/2') {
    $ran += 'CLOSURE-1/CLOSURE-2 (derived closure and negative control)'
} else {
    Write-Output 'CLOSURE-1/CLOSURE-2 DID NOT RUN inside test_packaged_runtime.'
    Write-Output '  Re-run ctest with --verbose to see which blocks it skipped.'
    $failures += 'CLOSURE-1/CLOSURE-2 did not run'
}

# ---------------------------------------------------------------------------
# Numeric goldens, one per forced continuation route
# ---------------------------------------------------------------------------

$comparator = Join-Path $sourceDir 'tools/compare_phi4_corelib_e2e.py'

# A bare Hugging Face download is not a Phi-4 AIE4 package.
#
# `flm pull` writes the four bundled overlays -- config.json,
# corelib_phi4_manifest.json, provenance.json and tokenizer_config.json -- over
# the downloaded files, and the engine requires the manifest. flm.exe cannot be
# built here (see Step 7 below), so the same composition is done directly, from
# the same src/model_overlays the installer ships.
#
# The downloaded files are HARDLINKED rather than copied: model.onnx.data is
# 3.2 GB, and a copy would be three minutes and three gigabytes for no
# additional coverage. The overlays are real copies, because they are the files
# being overlaid.
function New-ComposedModelDir {
    param([string]$Source, [string]$Destination, [string]$OverlayDir)

    if (Test-Path $Destination) {
        Remove-Item $Destination -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    foreach ($file in Get-ChildItem -Path $Source -File) {
        New-Item -ItemType HardLink `
            -Path (Join-Path $Destination $file.Name) `
            -Target $file.FullName | Out-Null
    }
    foreach ($overlay in Get-ChildItem -Path $OverlayDir -File) {
        Copy-Item -Path $overlay.FullName `
            -Destination (Join-Path $Destination $overlay.Name) -Force
    }
    return $Destination
}

if (-not $ModelDir) {
    $skipped += 'numeric goldens and boundary sweep (-ModelDir not supplied)'
} elseif (-not $CorelibSource) {
    $skipped += 'numeric goldens (-CorelibSource / RYZENAI_CORELIB_SOURCE not set)'
} else {
    $ModelDir = (Resolve-Path $ModelDir).Path
    $CorelibSource = (Resolve-Path $CorelibSource).Path
    $artifacts = Join-Path $BuildDir 'artifacts'
    New-Item -ItemType Directory -Force -Path $artifacts | Out-Null

    if (-not (Test-Path (Join-Path $ModelDir 'corelib_phi4_manifest.json'))) {
        $overlayDir = Join-Path $sourceDir 'model_overlays/phi4-mini-it-aie4'
        $composed = Join-Path $BuildDir 'model'
        Write-Output "composing an AIE4 package from $ModelDir + $overlayDir"
        $ModelDir = New-ComposedModelDir $ModelDir $composed $overlayDir
        Write-Output "model dir: $ModelDir"
        $ran += 'model package composed from the shipped overlays'
    }

    # Explicit token IDs, from a file committed to the repository rather than
    # written here. An expected OUTPUT sequence is only a golden if the INPUT
    # that produced it is fixed too, and a plan generated by the runner can be
    # edited in the same commit that "fixes" a failing expectation without
    # anybody noticing.
    $tokenPlan = Join-Path $suiteDir 'phi4_tokens.json'
    if (-not (Test-Path $tokenPlan)) {
        throw "missing committed token plan: $tokenPlan"
    }

    # Route-keyed expected top-1 sequences, committed to the repository.
    #
    # Without them the golden is re-derived from the reference on every run, so
    # a corelib change that moved FastFlow and the reference identically would
    # pass every comparison in this suite. These are the design 12.4
    # "route-specific expected IDs".
    $expectedTokens = Join-Path $suiteDir 'phi4_expected_tokens.json'

    $env:RYZENAI_CORELIB_SOURCE = $CorelibSource
    # The harness resolves the DLL through this, so it loads the staged
    # closure rather than whatever happens to be on PATH.
    $env:RYZENAI_CORELIB_PATH = $corelibDll

    # Both routes always run. Design 12.4 wants one golden per route, and
    # stopping at the first failing one would leave the other route with no
    # result at all -- which reads as "not tested" and is indistinguishable
    # from "tested and fine" once the log scrolls past.
    foreach ($route in @('force_reprefill', 'force_append')) {
      try {
        Write-Section "Numeric golden: $route"
        $referenceJson = Join-Path $artifacts "reference-$route.json"
        $fastflowJson = Join-Path $artifacts "fastflow-$route.json"

        # 1. Reference FIRST, run to completion including corelib.cleanup().
        Invoke-Checked "reference ($route)" $Python @(
            $comparator, 'emit-reference',
            '--model-dir', $ModelDir,
            '--token-ids-json', $tokenPlan,
            '--decode-steps', "$DecodeSteps",
            '--continuation-route', $route,
            '--output-json', $referenceJson
        )

        # 2. Only then the C++ harness. Never both at once.
        $e2eArgs = @(
            '--model-dir', $ModelDir,
            '--token-ids-json', $tokenPlan,
            '--decode-steps', "$DecodeSteps",
            '--continuation-route', $route,
            '--output-json', $fastflowJson
        )
        Invoke-Checked "fastflow ($route)" $e2eExe $e2eArgs

        # 3. The same binary, again, on the same input. Run before the
        #    comparison because it is the sharper question: a difference from
        #    the reference is about tolerances and about which implementation
        #    is right, whereas a difference between two runs of the same
        #    implementation on the same input means something is reading state
        #    the inputs do not determine. No tolerance makes that acceptable,
        #    so this comparison is bit-exact.
        $repeatJson = Join-Path $artifacts "fastflow-$route-repeat.json"
        $repeatArgs = @(
            '--model-dir', $ModelDir,
            '--token-ids-json', $tokenPlan,
            '--decode-steps', "$DecodeSteps",
            '--continuation-route', $route,
            '--output-json', $repeatJson
        )
        Invoke-Checked "fastflow repeat ($route)" $e2eExe $repeatArgs

        # The two comparisons answer different questions, so one failing must
        # not hide the other's result.
        try {
            Invoke-Checked "self-consistency ($route)" $Python @(
                $comparator, 'self-consistency',
                '--a', $fastflowJson,
                '--b', $repeatJson
            )
            $ran += "self-consistency $route"
        } catch {
            Write-Output "SELF-CONSISTENCY FAILED ($route)"
            $failures += "self-consistency $route"
        }

        # 4. Comparison holds no device context at all.
        try {
            $compareArgs = @(
                $comparator, 'compare',
                '--fastflow-json', $fastflowJson,
                '--reference-json', $referenceJson,
                # The design 12.4 thresholds are the gate; bit-exactness is
                # the claim the report makes about the result. Asserting it
                # here is what stops that claim being recomputed by hand from
                # the artifacts and then quietly going stale.
                '--require-bit-exact'
            )
            if (Test-Path $expectedTokens) {
                $compareArgs += @('--expected-tokens', $expectedTokens)
            }
            Invoke-Checked "compare ($route)" $Python $compareArgs
            $ran += "numeric golden $route"
        } catch {
            Write-Output "REFERENCE COMPARISON FAILED ($route)"
            $failures += "numeric golden $route"
        }
      } catch {
        Write-Output "NUMERIC GOLDEN FAILED ($route): $($_.Exception.Message)"
        $failures += "numeric golden $route"
      }
    }
    # The boundary sweep runs on its OWN invocation, never attached to a golden.
    #
    # It was previously folded into the reprefill golden's first run, which made
    # that run and its repeat structurally different: the swept run performed 25
    # extra prefills, so self-consistency correctly reported the counts
    # disagreeing (8106 dispatches against 3281) and looked like a
    # non-determinism finding when it was a harness defect. A run that is
    # compared against a repeat must be argument-identical to it.
    if ($BoundarySweep) {
        try {
            Invoke-Checked 'boundary sweep' $e2eExe @(
                '--model-dir', $ModelDir,
                '--token-ids-json', $tokenPlan,
                '--decode-steps', '1',
                '--continuation-route', 'force_reprefill',
                '--output-json', (Join-Path $artifacts 'boundary-sweep.json'),
                '--boundary-sweep'
            )
            $ran += 'boundary sweep (real prefills at every helper transition)'
        } catch {
            Write-Output "BOUNDARY SWEEP FAILED: $($_.Exception.Message)"
            $failures += 'boundary sweep'
        }
    } else {
        $skipped += 'boundary sweep real prefills (-BoundarySweep not passed; the helper-table half still ran inside test_phi4_hardware)'
    }
}

# ---------------------------------------------------------------------------
# Step 7 -- CLI and server endpoints
# ---------------------------------------------------------------------------

Write-Section 'CLI and server endpoints (Step 7)'
if ($FlmExe -and (Test-Path $FlmExe)) {
    try {
        Invoke-Checked 'server endpoints' $PSHOME\powershell.exe @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            (Join-Path $suiteDir 'run_server_endpoints.ps1'),
            '-FlmExe', $FlmExe,
            '-WorkDir', $BuildDir
        )
        $ran += 'Step 7 server endpoints (four generation endpoints, default and over-limit)'
    } catch {
        Write-Output "STEP 7 FAILED: $($_.Exception.Message)"
        $failures += 'Step 7 server endpoints'
    }
    # `flm run` is NOT driven here. On Windows the CLI reads through
    # ReadConsoleInput, a console-only API that cannot see redirected stdin, so
    # the REPL cannot be scripted: it prints its prompt and exits on the first
    # read. What is verifiable non-interactively -- that `flm run <tag>` loads
    # the AIE4 model and enters the REPL, and that `flm validate` reports the
    # corelib backend ready -- is recorded in the task report rather than
    # asserted here, because a check that cannot fail is worse than none.
    $skipped += 'Step 7 interactive `flm run` generation (CLI reads via ReadConsoleInput; not scriptable)'
} else {
    Write-Output 'SKIPPED: pass -FlmExe <path to flm.exe> to run the CLI and'
    Write-Output '  server endpoint checks.'
    $skipped += 'Step 7 CLI and server endpoints (-FlmExe not supplied)'
}

# ---------------------------------------------------------------------------

Write-Section 'Summary'
foreach ($item in $ran) { Write-Output "RAN     : $item" }
foreach ($item in $skipped) { Write-Output "SKIPPED : $item" }
foreach ($item in $failures) { Write-Output "FAILED  : $item" }

# Skipped work is visible in the EXIT CODE, not only in the log.
#
# This script previously printed PASS and exited 0 for runs where the numeric
# goldens, the self-consistency checks and the boundary sweep had never run --
# `$skipped` was printed and then ignored. A caller, or a human skimming the
# last line, could not tell a full acceptance run from one that built the tests
# and stopped. That is the fifth time this project has shipped a green result
# for work that did not happen, so it is fixed the way Task 11 fixed it: 77 is
# CTest's SKIP_RETURN_CODE and means INCOMPLETE, and only a run with nothing
# skipped is allowed to say PASS.
#
# Step 7 is currently always skipped, so this script is expected to report
# INCOMPLETE until flm.exe can be built and exercised. That is the honest
# result and it should stay visible rather than being special-cased away.
if ($failures.Count -gt 0) {
    Write-Output "run_hardware_suite: FAIL ($($failures.Count) block(s) failed, $($skipped.Count) skipped)"
    exit 1
}
if ($ran.Count -eq 0) {
    Write-Output 'run_hardware_suite: SKIPPED -- nothing ran.'
    exit 77
}
if ($skipped.Count -gt 0) {
    Write-Output "run_hardware_suite: INCOMPLETE -- $($ran.Count) block(s) ran, $($skipped.Count) skipped"
    exit 77
}
Write-Output "run_hardware_suite: PASS -- $($ran.Count) block(s) ran, none skipped"
exit 0
