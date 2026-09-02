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
try {
    Invoke-Checked 'ctest' $ctest @(
        '--test-dir', $BuildDir,
        '-C', 'Release',
        '--output-on-failure',
        '--no-tests=error'
    )
    $ran += 'ctest suite (includes test_phi4_hardware and test_fatal_child)'
} catch {
    Write-Output "CTEST FAILED: $($_.Exception.Message)"
    $failures += 'ctest suite'
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

    # Explicit token IDs, never a tokenizer. The point of design Section 12.4's
    # explicit-token checkpoints is that both sides consume the SAME integers,
    # and a tokenizer in the loop would make a mismatch ambiguous between the
    # model and the encoder. The prefix/suffix split is what makes a
    # continuation route meaningful: with no suffix both routes degenerate to
    # one prefill.
    $tokenPlan = Join-Path $artifacts 'tokens.json'
    @{
        prefix = @(200022, 882, 200024, 3923, 374, 279, 6864, 315, 9822, 30, 200021, 200022, 78191, 200024)
        suffix = @(791, 6864, 315, 9822, 374)
    } | ConvertTo-Json -Compress | Set-Content -Path $tokenPlan -Encoding ascii

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
        if ($BoundarySweep -and $route -eq 'force_reprefill') {
            # Once, not once per route: the sweep is route-independent and a
            # 4096-row prefill is the slowest thing in the suite.
            $e2eArgs += '--boundary-sweep'
        }
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
            Invoke-Checked "compare ($route)" $Python @(
                $comparator, 'compare',
                '--fastflow-json', $fastflowJson,
                '--reference-json', $referenceJson
            )
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
    if ($BoundarySweep) {
        $ran += 'boundary sweep (real prefills at every helper transition)'
    } else {
        $skipped += 'boundary sweep real prefills (-BoundarySweep not passed; the helper-table half still ran inside test_phi4_hardware)'
    }
}

# ---------------------------------------------------------------------------
# Step 7 -- CLI and server endpoints
# ---------------------------------------------------------------------------

Write-Section 'CLI and server endpoints (Step 7)'
# BLOCKED, and reported as blocked rather than skipped-and-forgotten.
# FastFlow depends on tokenizers-cpp, which needs Cargo, and neither cargo nor
# rustc exists on the development box or on the AIE4 target. flm.exe therefore
# cannot be built anywhere in this environment, so `flm run`, /api/generate,
# /api/chat, /v1/chat/completions and /v1/completions are unverified. Installing
# a Rust toolchain on a shared lab box is a human decision, not something this
# script should take.
Write-Output 'BLOCKED: flm.exe cannot be built (no Rust toolchain for tokenizers-cpp).'
Write-Output '  Unverified: flm run phi4-mini-it-aie4:4b, /api/generate, /api/chat,'
Write-Output '              /v1/chat/completions, /v1/completions.'
$skipped += 'Step 7 CLI and server endpoints (BLOCKED: no Rust toolchain, flm.exe unbuildable)'

# ---------------------------------------------------------------------------

Write-Section 'Summary'
foreach ($item in $ran) { Write-Output "ran    : $item" }
foreach ($item in $skipped) { Write-Output "skipped: $item" }
foreach ($item in $failures) { Write-Output "FAILED : $item" }
if ($failures.Count -gt 0) {
    Write-Output "run_hardware_suite: FAIL ($($failures.Count) block(s))"
    exit 1
}
if ($ran.Count -eq 0) {
    Write-Output 'run_hardware_suite: SKIPPED -- nothing ran.'
    exit 77
}
Write-Output 'run_hardware_suite: PASS'
exit 0
