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

    # Must match the build's FLM_RUNTIME_NAME. The engine DLLs flm.exe imports
    # live in lib/<backend>, and getting this wrong produces a
    # STATUS_DLL_NOT_FOUND with no output at all.
    [ValidateSet('xrt', 'hrx')]
    [string]$BackendName = 'xrt',

    [string]$BuildDir,
    [string]$Cmake,
    [string]$Python = "python",
    [int]$DecodeSteps = 16,

    # Rows the boundary sweep runs REAL prefills at. Off by default because a
    # 4096-row prefill is the slowest thing in the suite; the helper-table
    # half of Step 5 runs unconditionally inside test_phi4_hardware.
    [switch]$BoundarySweep,

    # Task 13 / design DETERM-3. Extra run-to-run determinism samples per
    # route, on top of the one each numeric golden already produces.
    #
    # DETERM-2 names "a bit-identity rate that degrades from the recorded
    # baseline" as a failure condition, and until a baseline exists that
    # clause enforces nothing. DETERM-3 requires at least 20 runs per route.
    # Zero by default, because 20 samples per route is roughly 25 minutes of
    # device time and the acceptance suite should not silently carry that.
    [int]$DeterminismRuns = 0,

    # Task 13. Run benchmark_phi4_aie4 and render the baseline document.
    [switch]$Baseline,

    # The revision the corelib under test was BUILT from. Derived from
    # -CorelibSource when that is a git checkout, because
    # ryzenai_corelib_get_version reports a hard-coded 0.1.0 spanning the
    # whole 0.x history and therefore cannot identify a revision.
    [string]$CorelibSourceRevision,

    # Large per-run artifacts (reference-*.json is 28 MB of FP32 logits as
    # JSON text, each fastflow-*.json is 21 MB) are deleted once the
    # comparisons that consume them have PASSED. A failing comparison keeps
    # its inputs, because that is the only run whose evidence anyone will
    # want. Pass this to keep everything.
    [switch]$KeepArtifacts
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$failures = @()
$bitExactNotes = @()
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

# Delete artifacts whose consumers have already succeeded.
#
# A run retains reference-<route>.json -- 17 x 200,064 FP32 logits as JSON
# text -- plus two 21 MB FastFlow documents, roughly 90 MB per route. At the
# >= 20 runs per route DETERM-3 needs that is several gigabytes, and the step
# would succeed while leaving the build tree unusable. Only determ1-*.json and
# compare-summary-*.json are needed afterwards.
#
# Called only on the success path, deliberately. The one run whose 21 MB of
# logits anybody will ever want to read is the run that diverged.
function Remove-ConsumedArtifact {
    param([string[]]$Paths)
    if ($KeepArtifacts) { return }
    foreach ($path in $Paths) {
        if ($path -and (Test-Path $path)) {
            Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        }
    }
}

# DETERM-4. Evidence for a determinism result must OUTLIVE the build tree.
#
# The build directory is gitignored, so every determ1 record this project has
# ever produced has lived only on one machine until it was overwritten or
# cleaned. That has now cost this effort its divergence evidence twice: once
# when a fixed artifact directory overwrote the 16/17 event, and once when a
# finding that overturned a design position was written up from records that
# no longer existed by the time anyone tried to check it.
#
# Every record is copied out to a committed directory, not just the breaching
# ones: a rate is only auditable if the clean runs it rests on are there too.
function Save-DeterminismRecord {
    param([string]$Path, [string]$Destination)
    if (-not (Test-Path $Path)) { return $null }
    if (-not (Test-Path $Destination)) {
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    }
    $target = Join-Path $Destination (Split-Path -Leaf $Path)
    Copy-Item -LiteralPath $Path -Destination $target -Force
    return $target
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
# Anchored on RAN, and self-checked.
#
# test_packaged_runtime prints the SAME block name on both its RAN line and its
# SKIPPED line, so an unanchored match records the block as having run in
# precisely the case this guard exists to catch: an empty
# RYZENAI_CORELIB_RUNTIME_DIR, where the closure work is skipped. Matching a
# substring of the summary is not the same question as "did it run".
#
# ctest -V prefixes each line with the test number, hence the leading `\d+:`.
function Test-ClosureRan {
    param([string]$CtestOutput)
    return [bool]($CtestOutput -match
        '(?m)^\s*\d+:\s*RAN\s+:\s*real-closure \(CLOSURE-1/2')
}

# The pattern this file is fixing has now shipped six times in this project,
# once inside the fix for the fifth. So the matcher is checked against both
# lines it has to tell apart, before it is trusted with a real result. The
# CMakeLists in this directory guards its argv construction the same way.
$closureRanProbe = "6: RAN     : real-closure (CLOSURE-1/2, 3 load-bearing DLLs proven)"
$closureSkippedProbe = "6: SKIPPED : real-closure (CLOSURE-1/2): pass -CorelibRuntimeDir, and -DependencyDir where the corelib's own dependencies live"
if (-not (Test-ClosureRan $closureRanProbe)) {
    throw ('The CLOSURE-1/2 guard fails to recognise a RAN line, so it would ' +
           'report the closure work as missing on every run.')
}
if (Test-ClosureRan $closureSkippedProbe) {
    throw ('The CLOSURE-1/2 guard accepts a SKIPPED line as evidence the ' +
           'closure ran. That is the defect this guard exists to prevent, ' +
           'and it would fire in exactly the stale-cache case I6 covers.')
}

if (Test-ClosureRan $ctestText) {
    $ran += 'CLOSURE-1/CLOSURE-2 (derived closure and negative control)'
} else {
    Write-Output 'CLOSURE-1/CLOSURE-2 DID NOT RUN inside test_packaged_runtime.'
    Write-Output ('  It prints the same block name whether it ran or skipped; ' +
                  'this guard requires the RAN line.')
    Write-Output "  Full ctest log: $ctestLog"
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
    # RUN-SCOPED, so history accumulates.
    #
    # A fixed directory meant determ1-<route>.json was overwritten every run
    # and nothing could be computed across runs. That was a tolerable annoyance
    # until DETERM-3: Task 13 has to establish the bit-identity baseline over at
    # least 20 runs per route, and a baseline assembled from whichever files
    # happened to survive, or from someone remembering to copy them off between
    # runs, is not a baseline. It also cost this task the evidence for the
    # 16/17 event, which was overwritten before it could be interrogated.
    #
    # Sortable UTC stamp plus the PID: two runs started in the same second on
    # the same machine still get their own directory.
    $runStamp = '{0}-{1}' -f (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'), $PID
    $artifacts = Join-Path (Join-Path $BuildDir 'artifacts') $runStamp
    New-Item -ItemType Directory -Force -Path $artifacts | Out-Null
    Write-Output "artifacts for this run: $artifacts"
    Write-Output ("  DETERM-3 baseline input: " +
        (Join-Path (Join-Path $BuildDir 'artifacts') '*/determ1-*.json'))

    # DETERM-4: the committed home for this run's determinism evidence. Inside
    # the source tree, NOT the build tree, because the build tree is
    # gitignored and the whole point is that these survive.
    $recordDir = Join-Path (Join-Path $suiteDir 'determinism_records') $runStamp
    New-Item -ItemType Directory -Force -Path $recordDir | Out-Null
    Write-Output "DETERM-4 records (COMMIT THESE): $recordDir"
    $lmHeadTool = Join-Path (Split-Path -Parent $sourceDir) `
        'tools/phi4_host_lm_head_reference.py'
    $lmHeadRuns = 0

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
    # Absent means FAIL, not "run without it".
    #
    # This was appended only `if (Test-Path)`, so deleting the golden produced
    # a full-green run with the strongest check in the suite silently not
    # applied -- the same shape as the CLOSURE guard above. The token plan two
    # lines up already throws on exactly this condition.
    if (-not (Test-Path $expectedTokens)) {
        throw "missing committed expected-token golden: $expectedTokens"
    }

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
        $failuresAtRouteStart = $failures.Count
        $routeDiverged = $false
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
        #    comparison because it asks a different and sharper question: a
        #    difference from the reference is about which implementation is
        #    right, whereas a difference between two runs of the same
        #    implementation is about whether anything outside the inputs is
        #    being read.
        #
        #    Per DETERM-1 this is NOT a bit-exact comparison. State, metrics
        #    and the emitted tokens must match exactly; the logits must agree
        #    within 2 BF16 ULP, and their bit-identity rate is recorded rather
        #    than gated. An earlier version of this comment said the opposite,
        #    twenty lines above the check that had already stopped doing it.
        $repeatJson = Join-Path $artifacts "fastflow-$route-repeat.json"
        $summaryJson = Join-Path $artifacts "compare-summary-$route.json"
        $determJson = Join-Path $artifacts "determ1-$route.json"
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
                '--b', $repeatJson,
                '--summary-json', $determJson
            )
            $ran += "self-consistency $route"
            # DETERM-1 requires the run-to-run rate and the observed
            # maximum to be recorded every run, not merely printed.
            if (-not (Test-Path $determJson)) {
                throw "self-consistency ($route) exited 0 but wrote " +
                      "no DETERM-1 record at $determJson"
            }
            $d = Get-Content $determJson -Raw | ConvertFrom-Json
            Save-DeterminismRecord $determJson $recordDir | Out-Null
            $routeDiverged =
                ($d.logits_bit_exact_steps -lt $d.logits_total_steps)
            $dnote = ("run-to-run [$route]: logits bit-identical " +
                "$($d.logits_bit_exact_steps)/$($d.logits_total_steps), " +
                "max |diff| $($d.observed_max_abs_diff) " +
                "(DETERM-2 bound $($d.determ2_bound_ulps) BF16 ULP)")
            $ran += $dnote
            $bitExactNotes += $dnote
        } catch {
            Write-Output "SELF-CONSISTENCY FAILED ($route)"
            $failures += "self-consistency $route"
            # The record goes to the summary on the FAILURE path too.
            # DETERM-1 says every run, and a red run is precisely the one
            # a baseline most needs to count -- it was previously written
            # only on the success path, so the artifact survived and the
            # summary line did not.
            if (Test-Path $determJson) {
                $d = Get-Content $determJson -Raw | ConvertFrom-Json
                Save-DeterminismRecord $determJson $recordDir | Out-Null
                $routeDiverged = $true
                $bitExactNotes += ("run-to-run [$route] (FAILED): logits " +
                    "bit-identical " +
                    "$($d.logits_bit_exact_steps)/$($d.logits_total_steps), " +
                    "max |diff| $($d.observed_max_abs_diff), " +
                    "first divergence $($d.first_divergence)")
            } else {
                $bitExactNotes +=
                    "run-to-run [$route] (FAILED): no DETERM-1 record written"
            }
        }

        # 4. Comparison holds no device context at all.
        try {
            $compareArgs = @(
                $comparator, 'compare',
                '--fastflow-json', $fastflowJson,
                '--reference-json', $referenceJson
                # --require-bit-exact is deliberately NOT passed.
                #
                # The comparator already enforces the part that is reliably
                # true and load-bearing: live K/V and the final hidden state
                # must be bit-identical, which is the 32-layer computation
                # agreeing exactly. Logit bit-identity is reported but not
                # gated, because some runs have a single logit vector out
                # of 17 differ from the reference while every design 12.4
                # threshold is still met and the top-1 matches.
                #
                # The rate here read "one measured run in three" and was
                # stale. Task 13 counted the surviving artifacts: the APPEND
                # route differed on 2 of the 4 comparisons whose records
                # exist, the REPREFILL route on 0 of 4. Stated with its n,
                # because n=4 does not support a rate anyone should rely on.
                # Gating on it would turn that into a red suite for something
                # that is not a defect. Pass the flag by hand when
                # investigating.
            )
            $compareArgs += @('--expected-tokens', $expectedTokens)
            $compareArgs += @('--summary-json', $summaryJson)
            Invoke-Checked "compare ($route)" $Python $compareArgs
            $ran += "numeric golden $route (expected-token golden applied)"
            # The "reported" half of the gate split, made durable.
            #
            # The bit-exact counts were stdout prints: not in any artifact, not
            # in the summary, never aggregated. The suite's last line read
            # identically at 17/17 and 16/17, so a rising rate of LM-head
            # divergence was undetectable. It is a reported property, so it has
            # to survive the run.
            # compare has just exited 0 with --summary-json passed, so an
            # absent file means it did not do what it reported. Skipping
            # here is the optional-guard shape again.
            if (-not (Test-Path $summaryJson)) {
                throw "compare ($route) exited 0 but wrote no " +
                      "--summary-json at $summaryJson"
            }
            $s = Get-Content $summaryJson -Raw | ConvertFrom-Json
            $note = ("bit-exact vs reference [$route]: logits " +
                "$($s.logits_bit_exact_steps)/$($s.logits_total_steps), " +
                "K/V $($s.kv_bit_exact_tensors)/$($s.kv_total_tensors), " +
                "last_hidden $($s.last_hidden_bit_exact)")
            $ran += $note
            $bitExactNotes += $note
            if ($s.logits_bit_exact_steps -lt $s.logits_total_steps) {
                # Not a failure -- see the comparator's note on why logit
                # bit-identity is reported rather than gated -- but it must
                # be impossible to miss in the summary.
                $bitExactNotes +=
                    ("  NOTE [$route]: $($s.logits_total_steps - $s.logits_bit_exact_steps) " +
                     "logit vector(s) differed from the reference inside " +
                     "the LM head, within all design 12.4 thresholds.")
            }
        } catch {
            Write-Output "REFERENCE COMPARISON FAILED ($route)"
            $failures += "numeric golden $route"
        }

        # Both comparisons are done with these three documents. They are kept
        # when something failed, and also when the two runs were not
        # bit-identical -- a within-2-ULP divergence passes every DETERM-1
        # gate by design and is exactly the sample Step 9b's host LM-head
        # reference needs, so pruning on failure alone would throw away every
        # usable one.
        if ($failures.Count -eq $failuresAtRouteStart -and -not $routeDiverged) {
            Remove-ConsumedArtifact @(
                $referenceJson, $fastflowJson, $repeatJson)
        } else {
            Write-Output ("keeping the raw artifacts for $route " +
                "(failed=$($failures.Count -ne $failuresAtRouteStart), " +
                "diverged=$routeDiverged): $artifacts")
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

    # -----------------------------------------------------------------------
    # DETERM-3: the run-to-run bit-identity baseline
    #
    # Each sample is the SAME comparison the numeric golden already performs
    # -- two processes, same binary, same explicit token IDs, compared by
    # `self-consistency` -- repeated. Keeping the methodology identical is the
    # point: a baseline assembled from a differently shaped measurement cannot
    # be compared against the three post-instrumentation runs that preceded
    # it, or against Task 16's acceptance runs.
    #
    # The reference driver is NOT run here. It contributes nothing to a
    # run-to-run question and each invocation writes 28 MB of FP32 logits as
    # JSON text.
    #
    # A sample whose hard gates fail is COUNTED and reported, not skipped.
    # Dropping it would bias the rate upward, and silently, which is the exact
    # species of error DETERM-2 exists to prevent.
    # -----------------------------------------------------------------------
    if ($DeterminismRuns -gt 0) {
        Write-Section "DETERM-3 baseline: $DeterminismRuns extra sample(s) per route"
        $determFailures = 0
        $determCompleted = @{}
        foreach ($route in @('force_reprefill', 'force_append')) {
            $determCompleted[$route] = 0
            for ($sample = 1; $sample -le $DeterminismRuns; $sample++) {
                $tag = '{0}-{1:d3}' -f $route, $sample
                $aJson = Join-Path $artifacts "determ-a-$tag.json"
                $bJson = Join-Path $artifacts "determ-b-$tag.json"
                $recordJson = Join-Path $artifacts "determ1-$tag.json"
                $baseArgs = @(
                    '--model-dir', $ModelDir,
                    '--token-ids-json', $tokenPlan,
                    '--decode-steps', "$DecodeSteps",
                    '--continuation-route', $route
                )
                try {
                    Invoke-Checked "determinism A ($tag)" $e2eExe `
                        ($baseArgs + @('--output-json', $aJson))
                    Invoke-Checked "determinism B ($tag)" $e2eExe `
                        ($baseArgs + @('--output-json', $bJson))
                } catch {
                    Write-Output "DETERMINISM SAMPLE FAILED TO RUN ($tag): $($_.Exception.Message)"
                    $failures += "DETERM-3 sample $tag did not produce two runs"
                    continue
                }

                $gateFailed = $false
                try {
                    Invoke-Checked "self-consistency ($tag)" $Python @(
                        $comparator, 'self-consistency',
                        '--a', $aJson,
                        '--b', $bJson,
                        '--summary-json', $recordJson
                    )
                } catch {
                    # A DETERM-2 hard-gate failure. Real, and it must reach
                    # the exit code -- but the sample still counts toward the
                    # rate, so the campaign continues.
                    $gateFailed = $true
                    $determFailures++
                    Write-Output "DETERM-2 GATE FAILED on sample $tag"
                }

                if (-not (Test-Path $recordJson)) {
                    # The record is the entire product of this sample. Its
                    # absence is not a skip to note, it is a sample that did
                    # not happen.
                    $failures += "DETERM-3 sample $tag wrote no record"
                    continue
                }
                $determCompleted[$route]++
                $d = Get-Content $recordJson -Raw | ConvertFrom-Json
                Save-DeterminismRecord $recordJson $recordDir | Out-Null
                Write-Output ("  ${tag}: logits bit-identical " +
                    "$($d.logits_bit_exact_steps)/$($d.logits_total_steps), " +
                    "max |diff| $($d.observed_max_abs_diff)")
                if ($d.localisation -and $d.localisation.measured) {
                    Write-Output ("    localisation: " +
                        "$($d.localisation.source) at " +
                        "$($d.localisation.step)")
                }
                # Keep the raw pair whenever the two runs were NOT
                # bit-identical, not only when a gate failed.
                #
                # A divergence within 2 ULP passes every DETERM-1 gate by
                # design, and it is precisely the event Step 9b's host
                # LM-head reference needs as input. Pruning on gate status
                # alone would discard every usable sample and leave only the
                # ones with nothing to look at.
                $diverged = ($gateFailed -or
                    ($d.logits_bit_exact_steps -lt $d.logits_total_steps))
                if ($diverged) {
                    Write-Output ("  keeping ${tag}'s two run documents: " +
                        "the runs were not bit-identical, which is the " +
                        "sample worth reading")
                    # STEP 9b, RUN AUTOMATICALLY ON THE EVENT THAT NEEDS IT.
                    #
                    # This used to be a script somebody could remember to run
                    # by hand, which meant the one campaign that produced a
                    # divergence had no host-reference analysis of it until
                    # long afterwards, from artifacts that were nearly gone.
                    # A divergent pair is exactly and only when this question
                    # is answerable, so the suite asks it there and then.
                    $lmHeadJson = Join-Path $recordDir "lmhead-$tag.json"
                    try {
                        Invoke-Checked "host LM-head reference ($tag)" `
                            $Python @(
                                $lmHeadTool,
                                '--model-dir', $ModelDir,
                                '--run-json', $aJson,
                                '--run-json', $bJson,
                                '--output-json', $lmHeadJson)
                        $lmHeadRuns++
                    } catch {
                        # A NON-ZERO EXIT HERE IS A FINDING, AND SOME FINDINGS
                        # MUST REACH THE SUITE'S EXIT CODE.
                        #
                        # This block used to downgrade every non-zero exit
                        # that still wrote a record to a summary note. That
                        # swallowed `common_bias` and `one_run_further` --
                        # which are precisely diagnostic answers (2) and (3),
                        # the two the human asked to be TOLD about -- and
                        # `gross_disagreement`, which says the analysis itself
                        # is not trustworthy. A campaign could exit 0 while
                        # Step 9b had reported a systematic LM-head bias.
                        # That is the ninth instance of this project's
                        # recurring pattern.
                        #
                        # The split, and which is which:
                        #
                        #   common_bias / one_run_further  -> FAILURE. Beyond
                        #     rounding; nothing else in the suite looks for
                        #     them.
                        #   gross_disagreement             -> FAILURE. The
                        #     reference and the device disagree by more than
                        #     rounding, so every other number in the record
                        #     is suspect.
                        #   model_body_divergence          -> a note ONLY
                        #     when this same sample already failed the
                        #     DETERM-2 gate above, because then it is already
                        #     counted. When the gate PASSED, no other check
                        #     in this suite would notice, so it fails too.
                        if (Test-Path $lmHeadJson) {
                            $lm = Get-Content $lmHeadJson -Raw |
                                ConvertFrom-Json
                            $lmHeadRuns++
                            Write-Output ("  Step 9b verdict [$tag]: " +
                                "$($lm.verdict)")
                            $note = ("Step 9b [$tag]: $($lm.verdict); " +
                                "LM-head inputs identical: " +
                                "$($lm.lm_head_inputs_identical)")
                            $bitExactNotes += $note
                            $beyondRounding = @(
                                'common_bias',
                                'one_run_further',
                                'gross_disagreement')
                            if ($beyondRounding -contains $lm.verdict) {
                                Write-Output ("  STEP 9b REPORTS SOMETHING " +
                                    "BEYOND ROUNDING ($tag): " +
                                    "$($lm.verdict)")
                                $failures += ("Step 9b [$tag]: " +
                                    "$($lm.verdict) -- not the benign " +
                                    "accumulation-order nondeterminism " +
                                    "DETERM-1 accepts")
                            } elseif (
                                $lm.verdict -eq 'model_body_divergence' -and
                                -not $gateFailed) {
                                Write-Output ("  STEP 9b FOUND A MODEL-BODY " +
                                    "DIVERGENCE THAT NO GATE CAUGHT ($tag)")
                                $failures += ("Step 9b [$tag]: model-body " +
                                    "divergence in a sample that PASSED the " +
                                    "DETERM-2 gates, so nothing else here " +
                                    "would have reported it")
                            }
                        } else {
                            Write-Output ("STEP 9b FAILED TO PRODUCE A " +
                                "RECORD ($tag): $($_.Exception.Message)")
                            $failures += "Step 9b produced no record for $tag"
                        }
                    }
                    if (Test-Path $lmHeadJson) {
                        $lm = Get-Content $lmHeadJson -Raw | ConvertFrom-Json
                        Write-Output ("  Step 9b [$tag]: verdict " +
                            "$($lm.verdict), LM-head inputs identical " +
                            "$($lm.lm_head_inputs_identical) at " +
                            "$($lm.analysed_step)")
                    }
                } else {
                    Remove-ConsumedArtifact @($aJson, $bJson)
                }
            }
        }
        if ($determFailures -gt 0) {
            $failures += ("DETERM-2 hard gate failed on $determFailures " +
                "determinism sample(s); the records are kept and counted")
        }
        foreach ($route in @('force_reprefill', 'force_append')) {
            $ran += ("DETERM-3 samples [$route]: " +
                "$($determCompleted[$route])/$DeterminismRuns completed")
        }
        $ran += ("DETERM-4 records written to $recordDir " +
            "(commit them; the build tree is gitignored)")
        if ($lmHeadRuns -gt 0) {
            $ran += ("Step 9b host LM-head reference: $lmHeadRuns " +
                "divergent pair(s) analysed")
        } else {
            # NOT a skip to be quiet about, and not a failure either: with no
            # divergent pair there is nothing for Step 9b to analyse. Saying
            # so keeps "it did not fire" distinguishable from "nobody ran it".
            $ran += ('Step 9b host LM-head reference: no divergent pair ' +
                'occurred in this campaign, so there was nothing to analyse')
        }
    } else {
        $skipped += ('DETERM-3 determinism campaign (-DeterminismRuns not ' +
            'passed; only the one sample per route from the numeric goldens ' +
            'was recorded)')
    }

    # -----------------------------------------------------------------------
    # Task 13: the performance and memory baseline, and its report
    # -----------------------------------------------------------------------
    if ($Baseline) {
        Write-Section 'Performance and memory baseline (Task 13)'
        $benchmarkExe = Join-Path $binDir 'benchmark_phi4_aie4.exe'
        if (-not (Test-Path $benchmarkExe)) {
            throw "benchmark_phi4_aie4.exe was not built: $benchmarkExe"
        }
        # Both revisions are DERIVED from the checkouts under test rather than
        # typed. A transcribed SHA in a baseline is a claim about a tree
        # nobody can go back and check.
        $repoRoot = Split-Path -Parent $sourceDir
        function Get-GitRevision {
            param([string]$Directory)
            $revision = (& git -C $Directory rev-parse HEAD 2>$null)
            if ($LASTEXITCODE -ne 0 -or -not $revision) { return '' }
            $revision = ([string]$revision).Trim()
            # A tree that is not pristine is recorded as such. A SHA on its
            # own would claim the baseline describes a committed revision when
            # it does not, and that is a claim nobody can check later.
            #
            # Tracked modifications and untracked files are labelled
            # DIFFERENTLY, because they mean different things and a single
            # `-dirty` makes the harmless case look like the serious one. The
            # corelib checkout on the AIE4 target carries four untracked build
            # output directories and no tracked change at all; calling that
            # "dirty" would tell a later reader the source had been edited.
            $status = @(
                (& git -C $Directory status --porcelain 2>$null) |
                    Where-Object { $_ })
            if ($LASTEXITCODE -eq 0 -and $status.Count -gt 0) {
                $tracked = @(
                    $status | Where-Object { -not $_.StartsWith('??') })
                if ($tracked.Count -gt 0) {
                    $revision = "$revision-dirty"
                } else {
                    $revision = "$revision-untracked-only"
                }
            }
            return $revision
        }
        $fastflowRevision = Get-GitRevision $repoRoot
        if (-not $fastflowRevision) {
            throw ("cannot identify the FastFlow revision from $repoRoot; " +
                'a baseline that does not say which tree produced it cannot ' +
                'be compared against anything later')
        }
        if (-not $CorelibSourceRevision) {
            $CorelibSourceRevision = Get-GitRevision $CorelibSource
        }
        if (-not $CorelibSourceRevision) {
            throw ('cannot identify the corelib source revision. Pass ' +
                '-CorelibSourceRevision: the DLL cannot supply it, because ' +
                'ryzenai_corelib_get_version is a hard-coded 0.1.0 spanning ' +
                'the whole 0.x history.')
        }
        $baselineJson = Join-Path $BuildDir 'phi4_aie4_baseline.json'
        try {
            Invoke-Checked 'baseline benchmark' $benchmarkExe @(
                '--model-dir', $ModelDir,
                '--token-ids-json', $tokenPlan,
                '--output-json', $baselineJson,
                '--fastflow-revision', $fastflowRevision,
                '--corelib-source-revision', $CorelibSourceRevision
            )
            $ran += 'baseline benchmark (load, TTFT, prefill, continuation, decode, memory, V scatter)'
        } catch {
            Write-Output "BASELINE BENCHMARK FAILED: $($_.Exception.Message)"
            $failures += 'baseline benchmark'
        }

        if (Test-Path $baselineJson) {
            $reportTool = Join-Path $repoRoot 'tools/report_phi4_corelib_baseline.py'
            $determGlob = Join-Path (Join-Path $BuildDir 'artifacts') '*/determ1-*.json'
            try {
                Invoke-Checked 'baseline report' $Python @(
                    $reportTool,
                    '--input', $baselineJson,
                    '--determinism-glob', $determGlob,
                    '--markdown', (Join-Path $repoRoot 'docs/docs/benchmarks/phi4_results.md'),
                    '--output-json', (Join-Path $BuildDir 'phi4_aie4_baseline_merged.json')
                )
                $ran += 'baseline report rendered (validated, DETERM-3 baseline established)'
            } catch {
                # The tool fails when a route has fewer than DETERM-3's 20
                # runs. That is not a tooling problem, it is the campaign not
                # having happened, and it must reach the exit code.
                Write-Output "BASELINE REPORT FAILED: $($_.Exception.Message)"
                $failures += 'baseline report (validation or DETERM-3 minimum)'
            }
        }
    } else {
        $skipped += 'Task 13 performance and memory baseline (-Baseline not passed)'
    }
}

# ---------------------------------------------------------------------------
# Step 7 -- CLI and server endpoints
# ---------------------------------------------------------------------------

Write-Section 'CLI and server endpoints (Step 7)'
if ($FlmExe -and (Test-Path $FlmExe)) {
    # The product's OWN runtime libraries, on PATH.
    #
    # flm.exe imports the per-backend engine DLLs from lib/<backend> and the
    # vendored libfftw3f-3.dll from lib/. Neither is beside the executable in a
    # build tree. Missing them makes the process die with STATUS_DLL_NOT_FOUND
    # before it writes a single byte to stdout or stderr, which presents as
    # "flm serve exited during startup" with no diagnostic at all -- observed
    # exactly once, the first time the FFTW linkage was corrected to use the
    # in-tree import library its header already matched.
    # lib/<backend>, resolved rather than hard-coded: src/lib/hrx exists
    # too, and an HRX build would die with the same no-diagnostic
    # STATUS_DLL_NOT_FOUND this block was added to prevent.
    $backendLib = Join-Path $sourceDir "lib/$BackendName"
    if (-not (Test-Path $backendLib)) {
        throw "no engine library directory at $backendLib; pass " +
              "-BackendName to match the build's FLM_RUNTIME_NAME"
    }
    $env:PATH = (
        (Join-Path $sourceDir 'lib'),
        $backendLib,
        $env:PATH
    ) -join ';'
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
foreach ($item in $bitExactNotes) { Write-Output "BITEXACT: $item" }

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
