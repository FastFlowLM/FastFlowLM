# Provisions the AIE4 target with a self-contained build/runtime environment
# and builds ryzenai_corelib from a pinned revision.
#
# This exists because the identity of the corelib under test was previously
# narrative: the report said which revision was built and the artifacts said
# nothing. A recipe in a session transcript is not reproducible and cannot be
# reviewed. Running this file, and recording the SHA-256 it prints, is what
# makes "we tested e5258d2" a checkable statement.
#
# WHY A PRIVATE PREFIX. The AIE4 dependencies (DynamicDispatch, ryzen_mm,
# aie_codegen, AIEBU, cpptrace, XRT) exist on the lab machine only inside
# another user's tree. That tree is read-only to us AND it moves: its
# ryzenai_corelib.dll was rebuilt from an uncommitted WIP patch mid-task, with
# .base/.new A/B variants beside it. Building against it would make our results
# depend on someone else's working directory. So the pieces are copied out once
# into a prefix we own, and everything after that points only at our own paths.
#
# `ryzenai_corelib_get_version` reports a hard-coded 0.1.0 for every 0.x
# revision, so the version symbol cannot identify a build. The SHA-256 can, and
# test_phi4_e2e records the hash of the DLL the loader actually served into
# every result artifact.
#
# Idempotent: re-running with the same inputs reproduces the same DLL. Measured
# 2026-09-01: building the same source against this prefix and against the
# original produced byte-identical DLLs, SHA-256
# bc12a285af7bb92a5053fb2431abcfb3986fa1595c3d628a40e9f316a157fd1c.

[CmdletBinding()]
param(
    # The read-only reference tree to copy the AIE4 dependencies out of. Only
    # read from; never written to.
    [string]$ReferenceInstall = 'C:\Users\akholodn\hybrid-llm\install',

    # Our own prefix. Mirrors the reference's shape so a corelib configure can
    # point at one directory, which is what the reference build does.
    [string]$Prefix = 'C:\Users\chiz\work\hybrid-llm\install',

    # A clean checkout of ryzenai-corelib at the pinned revision, with the
    # src/common submodule initialised.
    [string]$CorelibSource = 'C:\Users\chiz\work\corelib-e5258d2',
    [string]$CorelibRevision = 'e5258d29b5cb979d4a538994409b90ceff6e6e7a',
    [string]$CommonRevision = '91da5c76a66ac0fbdf46c6b1abf5685e7319caaf',
    [string]$CorelibInstall = 'C:\Users\chiz\work\corelib-e5258d2\install-mirrored',
    [string]$CorelibBuild = 'C:\Users\chiz\work\corelib-e5258d2\build-mirrored',

    # Supplies spdlog, fmt, protobuf, abseil, nlohmann_json, zlib and boost.
    [string]$CondaPrefix = 'C:\Users\chiz\.conda\envs\hybrid-llm\Library',

    [string]$Cmake,
    [switch]$SkipCopy
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $Cmake) {
    # The `cmake` on the target's PATH is Cygwin's 3.14.5, which cannot drive a
    # Visual Studio generator and is below the 3.24 the closure derivation
    # needs. Prefer the one Visual Studio ships.
    $Cmake = Join-Path $env:ProgramFiles ('Microsoft Visual Studio\2022\Community\Common7\IDE\' +
        'CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe')
}
if (-not (Test-Path $Cmake)) { throw "no usable cmake at $Cmake" }

# ---------------------------------------------------------------------------
# 1. Copy the dependency subset
# ---------------------------------------------------------------------------

if (-not $SkipCopy) {
    Write-Output "=== copying AIE4 dependencies from $ReferenceInstall ==="
    # A SUBSET, deliberately. The reference install is ~79 GB, almost all of it
    # ONNX Runtime build output that nothing here uses. These five directories
    # plus six DLLs are ~2.9 GB and are what a corelib build and its runtime
    # closure actually need.
    foreach ($dir in @('lib', 'include', 'share', 'xrt_package', 'aiebu')) {
        $source = Join-Path $ReferenceInstall $dir
        if (-not (Test-Path $source)) { continue }
        # robocopy returns <8 for success; 1 and 3 mean files were copied.
        & robocopy $source (Join-Path $Prefix $dir) /E /NFL /NDL /NJH /NJS /NP /MT | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy $dir failed ($LASTEXITCODE)" }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $Prefix 'bin') | Out-Null
    foreach ($dll in @('dyn_dispatch_core.dll', 'dyn_bins.dll', 'ryzen_mm.dll',
                       'spdlog.dll', 'fmt.dll', 'zlib.dll')) {
        Copy-Item (Join-Path $ReferenceInstall "bin\$dll") `
                  (Join-Path $Prefix "bin\$dll") -Force
    }
}

# A copied CMake package config that still names the source tree would defeat
# the entire exercise silently: the build would succeed and quietly depend on
# the tree we were trying to stop depending on. Fail loudly instead of
# rewriting, so a new absolute path is a decision rather than a guess.
Write-Output '=== checking copied CMake configs for foreign absolute paths ==='
$leaked = @(
    Get-ChildItem -Recurse -File -Path `
        (Join-Path $Prefix 'lib\cmake'), (Join-Path $Prefix 'share'), `
        (Join-Path $Prefix 'xrt_package') -Filter *.cmake -ErrorAction SilentlyContinue |
    Where-Object { (Get-Content $_.FullName -Raw) -match 'akholodn' }
)
if ($leaked.Count -gt 0) {
    $leaked | ForEach-Object { Write-Output "  leaked: $($_.FullName)" }
    throw ("$($leaked.Count) copied CMake config(s) still reference the source " +
           'tree. Rewrite or drop those entries before building, or the build ' +
           'depends on a directory we do not control.')
}
Write-Output '  none'

# ---------------------------------------------------------------------------
# 2. Verify the pinned source
# ---------------------------------------------------------------------------

Write-Output '=== verifying the pinned corelib checkout ==='
$head = (& git -C $CorelibSource rev-parse HEAD).Trim()
if ($head -ne $CorelibRevision) {
    throw "corelib HEAD is $head, expected the pinned $CorelibRevision"
}
$common = (& git -C (Join-Path $CorelibSource 'src\common') rev-parse HEAD).Trim()
if ($common -ne $CommonRevision) {
    throw "src/common is $common, expected the pinned $CommonRevision"
}
# Tracked files only. Untracked build and install directories are expected
# here and are not source drift.
$dirty = @(& git -C $CorelibSource status --porcelain --untracked-files=no)
if ($dirty.Count -gt 0) {
    $dirty | ForEach-Object { Write-Output "  dirty: $_" }
    throw 'the corelib checkout has modified tracked files; refusing to build'
}
Write-Output "  HEAD $head, src/common $common, no modified tracked files"

# ---------------------------------------------------------------------------
# 3. Build corelib
# ---------------------------------------------------------------------------

Write-Output '=== configuring corelib ==='
$p = $Prefix -replace '\\', '/'
$c = $CondaPrefix -replace '\\', '/'
# spdlog_DIR and fmt_DIR are load-bearing. Without them the link fails
# LNK2019/LNK1120 on spdlog::logger::log out of
# dyn_dispatch_core.lib(logging.obj).
& $Cmake -S $CorelibSource -B $CorelibBuild -G 'Visual Studio 17 2022' -A x64 `
    "-DCMAKE_INSTALL_PREFIX=$($CorelibInstall -replace '\\','/')" `
    "-DCMAKE_PREFIX_PATH=$c;$p" `
    "-DXRT_DIR=$p/xrt_package/xrt/share/cmake/XRT" `
    "-DDynamicDispatch_DIR=$p/lib/cmake/DynamicDispatch" `
    "-Dryzen_mm_DIR=$p/lib/cmake/ryzen_mm" `
    "-Daie_codegen_DIR=$p/lib/cmake/aie_codegen" `
    "-Dcpptrace_DIR=$p/lib/cmake/cpptrace" `
    "-DAIEBU_DIR=$p/share/cmake/AIEBU" `
    "-Dspdlog_DIR=$c/lib/cmake/spdlog" `
    "-Dfmt_DIR=$c/lib/cmake/fmt" `
    "-Dabsl_DIR=$c/lib/cmake/absl" `
    "-Dnlohmann_json_DIR=$c/share/cmake/nlohmann_json"
if ($LASTEXITCODE -ne 0) { throw "corelib configure failed ($LASTEXITCODE)" }

Write-Output '=== building corelib ==='
# RelWithDebInfo to match the DynamicDispatch package, which ships only that
# configuration.
& $Cmake --build $CorelibBuild --config RelWithDebInfo --target install -- /m
if ($LASTEXITCODE -ne 0) { throw "corelib build failed ($LASTEXITCODE)" }

$dll = Join-Path $CorelibInstall 'bin\ryzenai_corelib.dll'
$hash = (Get-FileHash -Algorithm SHA256 -Path $dll).Hash.ToLower()
Write-Output ''
Write-Output '=== corelib under test ==='
Write-Output "path   : $dll"
Write-Output "size   : $((Get-Item $dll).Length) bytes"
Write-Output "sha256 : $hash"
Write-Output "source : $CorelibRevision (src/common $CommonRevision)"
Write-Output ''
Write-Output 'Next: derive the runtime closure with cmake/StageAie4Runtime.cmake'
Write-Output 'against THIS DLL, then run run_hardware_suite.ps1 against the'
Write-Output 'staged directory. Do not transcribe the closure (CLOSURE-1).'
