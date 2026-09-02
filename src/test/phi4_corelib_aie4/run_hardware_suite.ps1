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
# Serial on purpose: several of these hold an AIE4 device context.
Invoke-Checked 'ctest' 'ctest' @(
    '--test-dir', $BuildDir,
    '-C', 'Release',
    '--output-on-failure',
    '--no-tests=error'
)
$ran += 'ctest suite (includes test_phi4_hardware and test_fatal_child)'

# ---------------------------------------------------------------------------
# Numeric goldens, one per forced continuation route
# ---------------------------------------------------------------------------

$comparator = Join-Path $sourceDir 'tools/compare_phi4_corelib_e2e.py'

if (-not $ModelDir) {
    $skipped += 'numeric goldens and boundary sweep (-ModelDir not supplied)'
} elseif (-not $CorelibSource) {
    $skipped += 'numeric goldens (-CorelibSource / RYZENAI_CORELIB_SOURCE not set)'
} else {
    $ModelDir = (Resolve-Path $ModelDir).Path
    $CorelibSource = (Resolve-Path $CorelibSource).Path
    $artifacts = Join-Path $BuildDir 'artifacts'
    New-Item -ItemType Directory -Force -Path $artifacts | Out-Null

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

    foreach ($route in @('force_reprefill', 'force_append')) {
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

        # 3. Comparison holds no device context at all.
        Invoke-Checked "compare ($route)" $Python @(
            $comparator, 'compare',
            '--fastflow-json', $fastflowJson,
            '--reference-json', $referenceJson
        )
        $ran += "numeric golden $route"
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
if ($ran.Count -eq 0) {
    Write-Output 'run_hardware_suite: SKIPPED -- nothing ran.'
    exit 77
}
Write-Output 'run_hardware_suite: PASS'
exit 0
