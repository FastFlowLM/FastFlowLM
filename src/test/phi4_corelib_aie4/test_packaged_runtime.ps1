# Packaging and clean-environment tests for the optional Phi-4 AIE4 feature.
#
# `-CorelibRuntimeDir` (with `-DependencyDir`) enables the checks that matter
# most, and they are the ones that cannot be faked: the closure is derived from
# the real ryzenai_corelib.dll, loaded from the staged directory with
# development paths removed, and then re-loaded with one staged DLL at a time
# hidden. Without the negative control a green result proves nothing, because
# an ambient conda or toolchain prefix supplies the missing DLL on precisely
# the machine that built the binary.

param(
    [string]$FlmExe = "",
    [string]$CorelibRuntimeDir = "",
    [string]$XrtRuntimeDir = "",
    [string]$DependencyDir = "",
    # The CMake to configure the fixture with. CMakeLists passes
    # ${CMAKE_COMMAND}, so the guard runs against the same CMake that
    # configured the suite. The default is only for a hand-run.
    #
    # Bare `cmake` is not good enough. On the AIE4 target PATH finds Cygwin's
    # CMake 3.14.5, which is below the 3.24 that StageAie4Runtime.cmake
    # requires for file(GET_RUNTIME_DEPENDENCIES) -- so the closure guard
    # failed for a reason that had nothing to do with the closure.
    [string]$CmakeExe = "cmake",
    [switch]$RunAie4ModelLoad
)

$ErrorActionPreference = "Stop"

# Without this, a terminating error inside the script's try/finally exits 1
# with NOTHING on either stream when the script is launched with `pwsh -File`.
# That is how the CMake version problem above stayed invisible: the packaging
# guard reported failure and said nothing about it, which is the worst thing a
# guard can do.
trap {
    [Console]::Error.WriteLine("test_packaged_runtime: $($_.Exception.Message)")
    [Console]::Error.WriteLine($_.ScriptStackTrace)
    exit 1
}

# Skipped work is reported as skipped, never as passed. A script that prints a
# success line for blocks it never entered reads as coverage it does not have,
# which is the defect test_real_corelib had to fix in Task 10R. Exit code 77 is
# CTest's SKIP_RETURN_CODE, so a partial run shows as Skipped rather than
# Passed.
$ran = @()
$skipped = @()
$SKIP_EXIT = 77

$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$modulePath = Join-Path $sourceRoot "cmake/ConfigureAie4Runtime.cmake"
$stageScript = Join-Path $sourceRoot "cmake/StageAie4Runtime.cmake"
$temporary = Join-Path ([System.IO.Path]::GetTempPath()) (
    "flm-aie4-package-{0}-{1}" -f $PID, [DateTime]::UtcNow.Ticks)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class FlmAie4Loader {
    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr LoadLibraryEx(
        string path, IntPtr file, uint flags);
    [DllImport("kernel32", SetLastError = true)]
    public static extern bool FreeLibrary(IntPtr module);
}
"@

# Get-FileHash is not reliably resolvable in the constrained -NoProfile host
# CTest launches, so hash through .NET directly rather than depending on module
# autoloading.
function Get-Sha256Hex {
    param([string]$Path)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            $bytes = $algorithm.ComputeHash($stream)
        } finally {
            $stream.Dispose()
        }
    } finally {
        $algorithm.Dispose()
    }
    return (
        -join ($bytes | ForEach-Object { $_.ToString("x2") })
    )
}

function Invoke-Configure {
    param(
        [string]$Build,
        [bool]$Enabled,
        [string]$Corelib = "",
        [string]$Xrt = "",
        [string]$Dependency = "",
        [bool]$ExpectSuccess = $true
    )
    $arguments = @(
        "-S", (Join-Path $temporary "fixture"),
        "-B", $Build,
        "-DFLM_ENABLE_CORELIB_AIE4=$(
            if ($Enabled) { "ON" } else { "OFF" })"
    )
    if ($Corelib) {
        $arguments += "-DRYZENAI_CORELIB_RUNTIME_DIR=$Corelib"
    }
    if ($Xrt) {
        $arguments += "-DXRT_RUNTIME_DIR=$Xrt"
    }
    if ($Dependency) {
        $arguments += "-DFLM_AIE4_DEPENDENCY_DIRS=$Dependency"
    }
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = (& $CmakeExe @arguments 2>&1 |
        ForEach-Object { $_.ToString() } |
        Out-String)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if (($exitCode -eq 0) -ne $ExpectSuccess) {
        throw "Unexpected CMake configure result.`n$output"
    }
    return $output
}

function Invoke-Install {
    param(
        [string]$Build,
        [string]$Prefix,
        [bool]$ExpectSuccess = $true
    )
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = (& $CmakeExe --install $Build --prefix $Prefix 2>&1 |
        ForEach-Object { $_.ToString() } |
        Out-String)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if (($exitCode -eq 0) -ne $ExpectSuccess) {
        throw "Unexpected CMake install result.`n$output"
    }
    return $output
}

# Loads the corelib DLL by absolute path from $Dir with development paths
# removed. LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
# resolves dependencies from the staged directory and the approved system
# directories only; PATH is not consulted at all, which is the product's own
# search behaviour and the point of design CLOSURE-2.
function Invoke-CleanEnvironmentLoad {
    param([string]$Dir)
    $savedPath = $env:PATH
    $savedCorelib = $env:RYZENAI_CORELIB_PATH
    $savedXrt = $env:XILINX_XRT
    try {
        $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
        Remove-Item Env:RYZENAI_CORELIB_PATH -ErrorAction SilentlyContinue
        Remove-Item Env:XILINX_XRT -ErrorAction SilentlyContinue
        $module = [FlmAie4Loader]::LoadLibraryEx(
            (Join-Path $Dir "ryzenai_corelib.dll"),
            [IntPtr]::Zero,
            0x00000100 -bor 0x00001000)
        if ($module -eq [IntPtr]::Zero) {
            return [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        }
        [FlmAie4Loader]::FreeLibrary($module) | Out-Null
        return 0
    } finally {
        $env:PATH = $savedPath
        if ($null -eq $savedCorelib) {
            Remove-Item Env:RYZENAI_CORELIB_PATH -ErrorAction SilentlyContinue
        } else {
            $env:RYZENAI_CORELIB_PATH = $savedCorelib
        }
        if ($null -eq $savedXrt) {
            Remove-Item Env:XILINX_XRT -ErrorAction SilentlyContinue
        } else {
            $env:XILINX_XRT = $savedXrt
        }
    }
}

try {
    $inno = Get-Content (Join-Path $sourceRoot "inno/flm.iss") -Raw
    foreach ($required in @(
        'AppVersion=1.0.4',
        'Name: "aie4runtime"',
        'corelib_phi4_manifest.json',
        'tokenizer_config.json',
        'config.json',
        'provenance.json'
    )) {
        if ($inno -notmatch [regex]::Escape($required)) {
            throw "Inno manifest is missing: $required"
        }
    }
    $wix = Get-Content (Join-Path $sourceRoot "wix/flm.wxs") -Raw
    foreach ($required in @(
        'Version="1.0.4"',
        'Feature Id="Aie4Feature"',
        'ComponentGroup Id="Aie4OverlayComponents"',
        'corelib_phi4_manifest.json',
        'tokenizer_config.json',
        'config.json',
        'provenance.json'
    )) {
        if ($wix -notmatch [regex]::Escape($required)) {
            throw "WiX manifest is missing: $required"
        }
    }
    [xml]$parsedWix = $wix
    $ran += "installer-manifests"

    # "The main installer must still build with no AIE4 closure present" was
    # asserted here by forbidding the substring `exit /b 1` anywhere in either
    # get_files.bat. That guard is retired, deliberately.
    #
    # It never checked the property it was named for. It checked that the
    # scripts contain no error handling at all -- so the moment a DIFFERENT
    # failure legitimately needed reporting (the 2026-09-01 decision to ship
    # the closure report, which must fail loudly when the report is absent)
    # the guard fired on correct code while still being unable to notice a
    # regression that failed the no-closure path some other way, such as
    # `goto :eof` after an echo, or a non-zero errorlevel left by the last
    # command in the else branch.
    #
    # The property is now proven by RUNNING both shipped scripts with no
    # closure staged and requiring exit 0 -- see the installer-staging block
    # below, which reports under this same name.

    # flm.exe must never gain a link-time dependency on ryzenai_corelib. The
    # DLL is resolved at run time by absolute path precisely so that a binary
    # without the AIE4 runtime installed still starts. dumpbin on the packaged
    # exe is the end check, but it only runs where an exe exists; this catches
    # the regression at its source.
    #
    # The scan reads whole balanced-paren command invocations, not lines. Every
    # link call in this tree spans multiple lines -- the AIE4 target's own does
    # -- so a line-at-a-time match would miss the exact form the regression
    # would take and report success for a check it never performed.
    $cmakeFiles = @(
        Get-ChildItem -Path $sourceRoot -Recurse -File -Include @(
            "CMakeLists.txt", "*.cmake"
        ) | Where-Object {
            $_.FullName -notmatch "[\\/](build|out|third_party)[^\\/]*[\\/]"
        }
    )
    if ($cmakeFiles.Count -lt 3) {
        throw "The link-guard scan found only $($cmakeFiles.Count) CMake files"
    }
    $scannedLinkCalls = 0
    foreach ($file in $cmakeFiles) {
        $text = Get-Content $file.FullName -Raw
        if ($null -eq $text) { continue }
        # Strip line comments so a commented-out example cannot trip the guard
        # and, more importantly, cannot hide a real call behind an unbalanced
        # parenthesis inside a comment.
        $text = [regex]::Replace($text, '(?m)#.*$', '')
        foreach ($match in [regex]::Matches(
            $text, '(?i)\b(target_link_libraries|link_libraries)\s*\(')) {
            $depth = 1
            $index = $match.Index + $match.Length
            while ($index -lt $text.Length -and $depth -gt 0) {
                if ($text[$index] -eq '(') { $depth++ }
                elseif ($text[$index] -eq ')') { $depth-- }
                $index++
            }
            if ($depth -ne 0) {
                throw "Unbalanced $($match.Value) in $($file.FullName)"
            }
            $scannedLinkCalls++
            $body = $text.Substring(
                $match.Index, $index - $match.Index)
            if ($body -match '(?i)\bryzenai_corelib\b') {
                $relative = $file.FullName.Substring($sourceRoot.Length + 1)
                throw (
                    "$relative links ryzenai_corelib: " +
                    ($body -replace '\s+', ' '))
            }
        }
    }
    if ($scannedLinkCalls -lt 1) {
        throw "The link-guard scan matched no link calls; it is inert"
    }
    $ran += "no-corelib-import-guard ($scannedLinkCalls link calls)"

    New-Item -ItemType Directory -Path $temporary | Out-Null
    $fixture = Join-Path $temporary "fixture"
    New-Item -ItemType Directory -Path $fixture | Out-Null
    @"
cmake_minimum_required(VERSION 3.24)
project(flm_aie4_packaging NONE)
option(FLM_ENABLE_CORELIB_AIE4 "" OFF)
include("$($modulePath.Replace('\', '/'))")
flm_aie4_warn_if_unstageable()
flm_aie4_install_runtime(DESTINATION bin/aie4 COMPONENT AIE4)
"@ | Set-Content -Path (Join-Path $fixture "CMakeLists.txt") -Encoding utf8

    # Feature OFF: no corelib runtime is required and nothing is staged.
    $offBuild = Join-Path $temporary "off"
    Invoke-Configure -Build $offBuild -Enabled $false | Out-Null
    Invoke-Install -Build $offBuild -Prefix (Join-Path $temporary "off-stage") |
        Out-Null
    if (Test-Path (Join-Path $temporary "off-stage/bin/aie4")) {
        throw "Feature OFF unexpectedly staged an AIE4 directory"
    }

    # Feature ON without a runtime directory: configuring must SUCCEED, because
    # flm.exe resolves the corelib DLL at run time by absolute path and never
    # links its import library. Only the install step needs the runtime.
    $devBuild = Join-Path $temporary "dev"
    $devOutput = Invoke-Configure -Build $devBuild -Enabled $true
    if ($devOutput -notmatch "RYZENAI_CORELIB_RUNTIME_DIR") {
        throw "Feature-ON configure did not warn about the missing runtime"
    }
    $devInstall = Invoke-Install `
        -Build $devBuild `
        -Prefix (Join-Path $temporary "dev-stage") `
        -ExpectSuccess $false
    if ($devInstall -notmatch "RYZENAI_CORELIB_RUNTIME_DIR") {
        throw "Install-time failure was not actionable.`n$devInstall"
    }

    # A configured but wrong runtime directory must fail at install, loudly.
    $badBuild = Join-Path $temporary "bad"
    Invoke-Configure `
        -Build $badBuild `
        -Enabled $true `
        -Corelib (Join-Path $temporary "does-not-exist") | Out-Null
    $badInstall = Invoke-Install `
        -Build $badBuild `
        -Prefix (Join-Path $temporary "bad-stage") `
        -ExpectSuccess $false
    if ($badInstall -notmatch "RYZENAI_CORELIB_RUNTIME_DIR") {
        throw "Missing runtime directory was not reported.`n$badInstall"
    }

    # A directory that exists but holds no ryzenai_corelib.dll is the mistake a
    # packager is most likely to make, so it gets its own named failure rather
    # than an unresolved-import message later.
    $emptyCorelib = Join-Path $temporary "empty-corelib"
    New-Item -ItemType Directory -Path $emptyCorelib | Out-Null
    $emptyBuild = Join-Path $temporary "empty"
    Invoke-Configure `
        -Build $emptyBuild `
        -Enabled $true `
        -Corelib $emptyCorelib | Out-Null
    $emptyInstall = Invoke-Install `
        -Build $emptyBuild `
        -Prefix (Join-Path $temporary "empty-stage") `
        -ExpectSuccess $false
    if ($emptyInstall -notmatch "no ryzenai_corelib\.dll") {
        throw "Empty runtime directory was not reported.`n$emptyInstall"
    }
    $ran += "packaging-configure-and-install-contract"

    if ($CorelibRuntimeDir) {
        $resolvedCorelib = (Resolve-Path $CorelibRuntimeDir).Path
        # -DependencyDir is a semicolon-separated list, matching the CMake
        # cache variable it feeds. Resolving each entry separately is also what
        # exercises the multi-entry path end to end: a single interpolated
        # string silently loses everything after the first `;`.
        $resolvedDependencies = @(
            $DependencyDir -split ';' |
                Where-Object { $_ } |
                ForEach-Object { (Resolve-Path $_).Path }
        )
        $realBuild = Join-Path $temporary "real"
        Invoke-Configure `
            -Build $realBuild `
            -Enabled $true `
            -Corelib $resolvedCorelib `
            -Xrt $(if ($XrtRuntimeDir) {
                (Resolve-Path $XrtRuntimeDir).Path } else { "" }) `
            -Dependency ($resolvedDependencies -join ';') | Out-Null
        $realStage = Join-Path $temporary "real-stage"
        Invoke-Install -Build $realBuild -Prefix $realStage | Out-Null
        $stagedDir = Join-Path $realStage "bin/aie4"

        # CLOSURE-1: the staged set is whatever the walker derived from this
        # exact binary, so the test asserts properties of the derivation rather
        # than a transcribed list that would silently disagree with the other
        # DynamicDispatch linkage.
        $report = Join-Path $stagedDir "aie4-closure.txt"
        if (-not (Test-Path $report)) {
            throw "The install did not record a derived closure report"
        }
        $derived = @(
            Get-Content $report |
                Where-Object { $_ -like "staged`t*" } |
                ForEach-Object { ($_ -split "`t")[1] }
        )
        if ($derived -notcontains "ryzenai_corelib.dll") {
            throw "The derived closure does not contain ryzenai_corelib.dll"
        }
        $staged = @(
            Get-ChildItem $stagedDir -File |
                Where-Object { $_.Extension -eq ".dll" } |
                ForEach-Object Name
        )
        if (Compare-Object ($derived | Sort-Object) ($staged | Sort-Object)) {
            throw "Staged AIE4 files do not match the derived closure"
        }
        if ($derived -contains "msvcp140.dll") {
            throw "The closure staged a build-machine Visual C++ runtime"
        }
        # The report ships inside bin/aie4, so it must not carry absolute
        # paths from the machine that built it. The audit record that does
        # name them stays in the build tree.
        foreach ($line in Get-Content $report) {
            if ($line -match '(?i)[a-z]:[\\/]') {
                throw "The shipped closure report leaks a build path: $line"
            }
        }
        # The corelib's own identity, ahead of the file list. Human decision,
        # 2026-09-01: the shipped artifact must answer "which runtime produced
        # this" without reference to a build tree that travelled nowhere.
        #
        # Asserted by key, not by position, and asserted to be PRESENT --
        # dropping the identity silently is the failure mode that matters,
        # because the report would still parse and still verify every hash.
        $identity = @{}
        foreach ($line in Get-Content $report) {
            $fields = $line -split "`t"
            if ($fields[0] -eq 'staged') {
                if (
                    $fields.Count -ne 3 -or
                    $fields[2] -notmatch '^[0-9a-f]{64}$'
                ) {
                    throw "The shipped closure report lacks a SHA-256: $line"
                }
                $actual = Get-Sha256Hex (Join-Path $stagedDir $fields[1])
                if ($actual -ne $fields[2]) {
                    throw (
                        "Staged $($fields[1]) does not match its recorded hash")
                }
            } elseif ($fields.Count -eq 2) {
                $identity[$fields[0]] = $fields[1]
            } else {
                throw "Unrecognised line in the shipped closure report: $line"
            }
        }
        foreach ($key in @('corelib_version', 'corelib_sha256')) {
            if (-not $identity.ContainsKey($key)) {
                throw (
                    "The shipped closure report carries no $key. A field " +
                    "report about a wrong number cannot then say which " +
                    "runtime produced it.")
            }
        }
        if ($identity['corelib_version'] -notmatch '^\d+\.\d+\.\d+$') {
            throw (
                "corelib_version is not MAJOR.MINOR.PATCH: " +
                $identity['corelib_version'])
        }
        # Two independent things, both checked: the recorded hash is the hash
        # of the file that shipped, and it is the same hash the file list
        # already recorded for that file. Either one alone can agree with a
        # wrong value.
        $rootHash = Get-Sha256Hex (Join-Path $stagedDir 'ryzenai_corelib.dll')
        if ($identity['corelib_sha256'] -ne $rootHash) {
            throw (
                "corelib_sha256 does not match the staged ryzenai_corelib.dll")
        }
        $rootLine = @(
            Get-Content $report |
                Where-Object { $_ -like "staged`tryzenai_corelib.dll`t*" }
        )
        if ($rootLine.Count -ne 1) {
            throw "The file list does not name ryzenai_corelib.dll exactly once"
        }
        if (($rootLine[0] -split "`t")[2] -ne $identity['corelib_sha256']) {
            throw "corelib_sha256 disagrees with the file list's own entry"
        }

        # CLOSURE-2, positive control.
        $code = Invoke-CleanEnvironmentLoad -Dir $stagedDir
        if ($code -ne 0) {
            throw "Clean-environment corelib load failed: Win32 error $code"
        }

        # CLOSURE-2, negative control. Every derived import must be
        # load-bearing from the staged directory. If hiding one still loads,
        # the environment supplied it and the positive control certified
        # nothing. `dyn_bins.dll` is exempt: it is opened by name at run time
        # rather than imported, so it is discovered by presence, not by the
        # walker, and its absence does not break LoadLibrary.
        $exempt = @("dyn_bins.dll")
        $proved = 0
        foreach ($name in $derived) {
            if ($exempt -contains $name) { continue }
            $path = Join-Path $stagedDir $name
            $hidden = "$path.hidden"
            Rename-Item -Path $path -NewName "$name.hidden"
            try {
                $missingCode = Invoke-CleanEnvironmentLoad -Dir $stagedDir
            } finally {
                Rename-Item -Path $hidden -NewName $name
            }
            if ($missingCode -eq 0) {
                throw (
                    "Removing $name from the staged closure still loaded. " +
                    "The load resolved it from outside the staged directory, " +
                    "so this closure is not proven.")
            }
            $proved += 1
        }
        if ($proved -lt 1) {
            throw "No staged dependency was proven load-bearing"
        }
        $ran += "real-closure (CLOSURE-1/2, $proved load-bearing DLLs proven)"
    } else {
        $skipped += (
            "real-closure (CLOSURE-1/2): pass -CorelibRuntimeDir, and " +
            "-DependencyDir where the corelib's own dependencies live")
    }

    # ------------------------------------------------------------------
    # The closure report must survive the INSTALLER staging, not just the
    # CMake install.
    #
    # Human decision, 2026-09-01: ship the provenance with the artifact. Both
    # get_files.bat scripts copy `..\build\aie4\*` wholesale, so the report was
    # already arriving -- incidentally. An incidental dependency is exactly the
    # kind that breaks silently: narrow that copy to `*.dll` and the installed
    # runtime loses the only thing in it that says which corelib produced a
    # result, with nothing failing.
    #
    # Driven through the REAL shipped .bat files against a synthetic tree, so
    # this cannot agree with a copy of the logic that has drifted from the one
    # that ships. Needs nothing but cmd.exe, so it always runs.
    # ------------------------------------------------------------------
    $installerRoot = Join-Path $temporary "installer"
    $fakeSrc = Join-Path $installerRoot "src"
    foreach ($relative in @(
        "build/aie4", "lib/xrt", "wix", "inno")) {
        New-Item -ItemType Directory -Force `
            -Path (Join-Path $fakeSrc $relative) | Out-Null
    }
    Set-Content -Path (Join-Path $fakeSrc "build/flm.exe") -Value "not an exe"
    Set-Content -Path (Join-Path $fakeSrc "lib/placeholder.dll") -Value "dll"
    Set-Content -Path (Join-Path $fakeSrc "lib/xrt/placeholder_xrt.dll") `
        -Value "dll"
    Set-Content -Path (Join-Path $fakeSrc "model_list.json") -Value "{}"
    Set-Content -Path (Join-Path $fakeSrc "model_info.json") -Value "{}"
    Set-Content -Path (Join-Path $fakeSrc "inno/logo.ico") -Value "ico"
    Set-Content -Path (Join-Path $fakeSrc "inno/terms.rtf") -Value "rtf"
    Copy-Item (Join-Path $sourceRoot "wix/get_files.bat") `
        (Join-Path $fakeSrc "wix/get_files.bat")
    Copy-Item (Join-Path $sourceRoot "inno/get_files.bat") `
        (Join-Path $fakeSrc "inno/get_files.bat")

    $fakeAie4 = Join-Path $fakeSrc "build/aie4"
    $fakeCorelib = Join-Path $fakeAie4 "ryzenai_corelib.dll"
    $fakeReport = Join-Path $fakeAie4 "aie4-closure.txt"
    Set-Content -Path $fakeCorelib -Value "not a dll"
    # Shaped like the real report, including the identity header the shipped
    # artifact now carries.
    $fakeReportText = (
        "corelib_version`t0.1.0`n" +
        "corelib_sha256`t$('a' * 64)`n" +
        "staged`tryzenai_corelib.dll`t$('a' * 64)`n")
    [System.IO.File]::WriteAllText($fakeReport, $fakeReportText)

    function Invoke-GetFiles {
        param([string]$Directory)
        $previous = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $output = (& cmd.exe /c (
            "cd /d `"$Directory`" && get_files.bat") 2>&1 |
            ForEach-Object { $_.ToString() } | Out-String)
        $code = $LASTEXITCODE
        $ErrorActionPreference = $previous
        return [pscustomobject]@{ ExitCode = $code; Output = $output }
    }

    $installerCases = @(
        [pscustomobject]@{
            Name = "wix"
            Directory = (Join-Path $fakeSrc "wix")
            Staged = (Join-Path $fakeSrc "wix/package/aie4")
        },
        [pscustomobject]@{
            Name = "inno"
            Directory = (Join-Path $fakeSrc "inno")
            Staged = (Join-Path $fakeSrc "inno/aie4")
        }
    )

    foreach ($case in $installerCases) {
        # 1. With a complete closure the report must arrive, unmodified and
        #    still free of build-machine paths.
        $result = Invoke-GetFiles -Directory $case.Directory
        if ($result.ExitCode -ne 0) {
            throw (
                "$($case.Name)/get_files.bat failed with a complete closure " +
                "(exit $($result.ExitCode)).`n$($result.Output)")
        }
        $stagedReport = Join-Path $case.Staged "aie4-closure.txt"
        if (-not (Test-Path -LiteralPath $stagedReport)) {
            throw (
                "$($case.Name)/get_files.bat did not stage aie4-closure.txt " +
                "beside the AIE4 runtime")
        }
        $stagedText = [System.IO.File]::ReadAllText($stagedReport)
        if ($stagedText -ne $fakeReportText) {
            throw "$($case.Name) staged a modified aie4-closure.txt"
        }
        foreach ($line in $stagedText -split "`r?`n") {
            if ($line -match '(?i)[a-z]:[\\/]') {
                throw (
                    "The $($case.Name)-staged closure report leaks a build " +
                    "path: $line")
            }
        }

        # 2. RED proof. Remove the report and the staging must FAIL, because a
        #    guard that cannot fail is not a guard. This is the case the
        #    wholesale xcopy would pass silently.
        Remove-Item -LiteralPath $fakeReport -Force
        Remove-Item -LiteralPath $stagedReport -Force
        $result = Invoke-GetFiles -Directory $case.Directory
        if ($result.ExitCode -eq 0) {
            throw (
                "$($case.Name)/get_files.bat reported success while staging " +
                "an AIE4 runtime with no aie4-closure.txt.`n$($result.Output)")
        }
        # A non-zero exit for some unrelated reason would satisfy the check
        # above while proving nothing, so the diagnostic has to name the file.
        if ($result.Output -notmatch 'aie4-closure\.txt') {
            throw (
                "$($case.Name)/get_files.bat failed without saying the " +
                "closure report was missing.`n$($result.Output)")
        }
        [System.IO.File]::WriteAllText($fakeReport, $fakeReportText)

        # 3. The optional feature stays optional. With no closure staged at
        #    all, the ordinary NPU2 installer must still build -- the
        #    regression dc15d66f introduced and Task 11 fixed.
        Rename-Item -LiteralPath $fakeCorelib -NewName "ryzenai_corelib.hidden"
        try {
            $result = Invoke-GetFiles -Directory $case.Directory
        } finally {
            Rename-Item `
                -LiteralPath (Join-Path $fakeAie4 "ryzenai_corelib.hidden") `
                -NewName "ryzenai_corelib.dll"
        }
        if ($result.ExitCode -ne 0) {
            throw (
                "$($case.Name)/get_files.bat failed with no AIE4 closure " +
                "staged, making an optional feature a precondition for " +
                "shipping.`n$($result.Output)")
        }
    }

    # The wix branch additionally emits the include file WiX compiles. Assert
    # both shapes, because the empty ComponentGroup is the one no build has
    # ever exercised.
    $wxi = Join-Path $fakeSrc "wix/package/aie4.wxi"
    if (-not (Test-Path -LiteralPath $wxi)) {
        throw "wix/get_files.bat emitted no package\aie4.wxi"
    }
    $wxiText = [System.IO.File]::ReadAllText($wxi)
    if ($wxiText -notmatch '<ComponentGroup Id="Aie4RuntimeComponents"[^>]*/>') {
        throw (
            "With no closure staged, aie4.wxi must define an EMPTY " +
            "Aie4RuntimeComponents group.`n$wxiText")
    }
    # The namespace is asserted on BOTH shapes, and this one first.
    #
    # get_files.bat emits the `<Include xmlns=...>` line twice, once per
    # branch, and until 2026-09-03 only the closure-PRESENT shape was checked
    # (below, after the rerun). Dropping it from the else branch would have
    # sailed past this guard -- and the else branch is the one that produces
    # the ordinary non-AIE4 MSI, which is the build WIX0200 actually broke.
    # The empty group is still a ComponentGroup, so it lands in the empty
    # namespace and is rejected in exactly the same way.
    if ($wxiText -notmatch '<Include xmlns="http://wixtoolset\.org/schemas/v4/wxs">') {
        throw (
            "With no closure staged, aie4.wxi must still declare the WiX " +
            "namespace or WiX rejects the ordinary non-AIE4 MSI with " +
            "WIX0200.`n$wxiText")
    }
    $result = Invoke-GetFiles -Directory (Join-Path $fakeSrc "wix")
    if ($result.ExitCode -ne 0) {
        throw "wix/get_files.bat failed on the closure-present rerun"
    }
    $wxiText = [System.IO.File]::ReadAllText($wxi)
    # Two defects a real `wix build` found on 2026-09-03, both of which had
    # been asserted-not-demonstrated since Task 11:
    #
    #  1. The generated <Include> declared no namespace, so every element
    #     inside it landed in the empty namespace and WiX rejected the file
    #     with WIX0200. That failed BOTH MSI builds -- including the ordinary
    #     non-AIE4 one, which had nothing to do with this feature.
    #  2. `Include="package\aie4\**"` doubled the path. The Files element
    #     resolves against the directory of the file CONTAINING it, and
    #     aie4.wxi is generated into package\, so WiX looked in
    #     package\package\aie4, found nothing, and emitted WIX8601 -- a
    #     WARNING. The build succeeded and produced an MSI with no AIE4
    #     runtime in it at all, byte-for-byte the same size as the no-closure
    #     MSI. A silent empty feature is exactly the outcome an "optional"
    #     feature must not have.
    foreach ($shape in @(
        @{
            Pattern = '<Include xmlns="http://wixtoolset\.org/schemas/v4/wxs">'
            Why = ("aie4.wxi must declare the WiX namespace or WiX rejects " +
                   "ComponentGroup with WIX0200")
        },
        @{
            Pattern = '<Files Include="aie4\\\*\*" />'
            Why = ("the Files path is relative to aie4.wxi's own directory; " +
                   "prefixing it with package\ makes WiX harvest nothing and " +
                   "only warn")
        }
    )) {
        if ($wxiText -notmatch $shape.Pattern) {
            throw "$($shape.Why).`n$wxiText"
        }
    }
    if ($wxiText -match 'Include="package\\aie4') {
        throw (
            "aie4.wxi doubles the staged path: WiX resolves it against " +
            "package\, so this harvests package\package\aie4 and ships an " +
            "empty AIE4 feature with a warning.`n$wxiText")
    }
    $ran += (
        "installer-staging (aie4-closure.txt ships, RED-proven, both " +
        "get_files.bat, both aie4.wxi shapes)")
    $ran += "optional-feature-packaging (no-closure staging exits 0, executed)"

    # ------------------------------------------------------------------
    # cmake/ReadCorelibVersion.ps1 must survive a build environment's LIB.
    #
    # Design 5.4's probe calls Add-Type; PowerShell compiles that C# with
    # warnings-as-errors; the C# compiler reads LIB and INCLUDE. Under MSBuild
    # those hold the MSVC toolchain's paths, one of which is RELATIVE
    # (`lib\um\x64`), and the compiler refuses a relative search path:
    #
    #   Warning as Error: Invalid search path 'lib\um\x64' specified in
    #   'LIB environment variable'
    #
    # The staging step that runs the probe is a POST_BUILD command, so that
    # took the entire flm build down -- and it passed every test beforehand
    # because every test ran it from a plain shell where LIB is unset. The fix
    # (clear LIB/INCLUDE around Add-Type, restore afterwards) was then verified
    # in exactly one environment, once, by hand: no test referenced the script
    # and no test set LIB. The installer-staging block above cannot help --
    # its `ryzenai_corelib.dll` contains the text "not a dll" and never reaches
    # the probe.
    #
    # WHAT THIS COVERS: that with a relative entry in LIB the script gets past
    # Add-Type and past LoadLibraryEx, and then fails where it should -- at the
    # missing export -- with its own diagnostic rather than a compiler error.
    # That is the whole of the regression, and it is able to fail: reverting
    # the save/restore turns the output into the compiler error above.
    #
    # WHAT THIS DOES NOT COVER: reading a real version number. That needs a
    # real ryzenai_corelib.dll, which a machine without the AIE4 runtime does
    # not have. The success path is asserted in the real-closure block above,
    # via the staged report's `corelib_version` -- which StageAie4Runtime.cmake
    # obtained from this same script.
    #
    # Launched through powershell.exe by absolute path, and that is
    # load-bearing rather than incidental: Windows PowerShell 5.1 shells out to
    # the .NET Framework csc.exe, which reads LIB, while pwsh 7 compiles
    # in-process with Roslyn and never looks at it. Under pwsh this check
    # cannot fail, so inheriting the host would make it vacuous on exactly the
    # machines that run pwsh. 5.1 is also what StageAie4Runtime.cmake invokes.
    # ------------------------------------------------------------------
    $probeScript = Join-Path $sourceRoot "cmake/ReadCorelibVersion.ps1"
    if (-not (Test-Path -LiteralPath $probeScript -PathType Leaf)) {
        throw "the corelib version probe is missing: $probeScript"
    }
    # A real PE, in an approved system directory, that certainly does not
    # export ryzenai_corelib_get_version.
    $probeDll = Join-Path $env:SystemRoot "System32\version.dll"
    $windowsPowerShell = Join-Path $env:SystemRoot `
        "System32\WindowsPowerShell\v1.0\powershell.exe"
    foreach ($needed in @($probeDll, $windowsPowerShell)) {
        if (-not (Test-Path -LiteralPath $needed -PathType Leaf)) {
            throw (
                "the corelib-version-probe regression check needs $needed " +
                "and it is absent, so the check would pass vacuously")
        }
    }
    $savedProbeLib = $env:LIB
    $savedProbeInclude = $env:INCLUDE
    $previousPreference = $ErrorActionPreference
    $probeOutput = ""
    $probeExit = $null
    try {
        # The shape that broke it: a relative entry beside an absolute one,
        # which is how vcvars64 leaves LIB.
        $env:LIB = "lib\um\x64;$env:SystemRoot\System32"
        $env:INCLUDE = "include\um"
        $ErrorActionPreference = "Continue"
        $probeOutput = (& $windowsPowerShell -NoProfile -NonInteractive `
            -ExecutionPolicy Bypass -File $probeScript -Dll $probeDll 2>&1 |
            ForEach-Object { $_.ToString() } | Out-String)
        $probeExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
        if ($null -eq $savedProbeLib) {
            Remove-Item Env:LIB -ErrorAction SilentlyContinue
        } else {
            $env:LIB = $savedProbeLib
        }
        if ($null -eq $savedProbeInclude) {
            Remove-Item Env:INCLUDE -ErrorAction SilentlyContinue
        } else {
            $env:INCLUDE = $savedProbeInclude
        }
    }
    if (
        $probeOutput -match 'LIB environment variable' -or
        $probeOutput -match '(?i)Cannot add type'
    ) {
        throw (
            "ReadCorelibVersion.ps1 is environment-sensitive again: an " +
            "MSBuild-shaped LIB reaches its Add-Type. This is the defect " +
            "that took the flm build down, because the staging step runs " +
            "from a POST_BUILD command.`n$probeOutput")
    }
    # Positive requirement, so that a probe which failed EARLIER for some other
    # reason cannot satisfy the check above by never compiling at all.
    if ($probeOutput -notmatch 'exports no ryzenai_corelib_get_version') {
        throw (
            "the probe did not reach GetProcAddress (exit $probeExit), so " +
            "nothing was proven about Add-Type surviving LIB.`n$probeOutput")
    }
    if ($probeExit -eq 0) {
        throw (
            "the probe reported success for a DLL that is not corelib.`n" +
            $probeOutput)
    }
    $ran += (
        "corelib-version-probe (5.4 LIB regression, run under Windows " +
        "PowerShell 5.1 against a non-corelib PE)")

    if ($FlmExe) {
        $resolvedFlm = (Resolve-Path $FlmExe).Path
        # -FlmExe is the first positional parameter, so a stray argument from a
        # mis-quoted argument vector lands here. Refusing a non-file keeps that
        # mistake from entering this block and reporting a run it never made.
        if (-not (Test-Path -LiteralPath $resolvedFlm -PathType Leaf)) {
            throw "-FlmExe is not a file: $resolvedFlm"
        }

        # dumpbin's exit code is load-bearing. If it runs but errors -- wrong
        # architecture, missing tool, unreadable image -- its output simply
        # fails to match the import pattern, and the Step 8 guard passes
        # vacuously while checking nothing.
        $imports = (& dumpbin /nologo /dependents $resolvedFlm 2>&1 |
            Out-String)
        if ($LASTEXITCODE -ne 0) {
            throw "dumpbin failed with exit code ${LASTEXITCODE}.`n$imports"
        }
        if ($imports -notmatch "(?i)Image has the following dependencies") {
            throw (
                "dumpbin produced no dependency listing, so the no-import " +
                "check would pass vacuously.`n$imports")
        }
        if ($imports -match "(?im)^\s*ryzenai_corelib\.dll\s*$") {
            throw "flm.exe has an unexpected ryzenai_corelib import"
        }

        $savedPath = $env:PATH
        $savedCorelib = $env:RYZENAI_CORELIB_PATH
        $savedXrt = $env:XILINX_XRT
        try {
            Remove-Item Env:RYZENAI_CORELIB_PATH -ErrorAction SilentlyContinue
            Remove-Item Env:XILINX_XRT -ErrorAction SilentlyContinue
            $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
            $validation = (& $resolvedFlm validate --json 2>&1 | Out-String)
            if ($LASTEXITCODE -ne 0) {
                throw "Clean-environment flm validate failed.`n$validation"
            }
            if ($RunAie4ModelLoad) {
                # Driven through drive_flm_console.ps1, NOT by piping into
                # `flm run`.
                #
                # This block used to be `"/bye" | & $resolvedFlm run <tag>`.
                # `flm run` reads keystrokes with ReadConsoleInput, a
                # console-only API that cannot see a redirected pipe -- Task 16
                # established that against the real binary -- so the prompt was
                # never delivered and the block would have sat in the REPL
                # until the CTest timeout killed it. It had never executed:
                # nothing in this tree passes -RunAie4ModelLoad, so a latent
                # hang shipped for five tasks under a line that read
                # "none skipped".
                #
                # The driver injects KEY_EVENT records into the console input
                # buffer the REPL is actually reading, so this exercises the
                # product's own interactive path. `/status` rather than a
                # generation turn: it is the REPL command wired to
                # AutoModel::show_profile, so it proves the AIE4 model loaded
                # and names the corelib DLL that answered, without spending a
                # decode on a packaging test.
                $driver = Join-Path $PSScriptRoot "drive_flm_console.ps1"
                if (-not (Test-Path -LiteralPath $driver)) {
                    throw "the console driver is missing: $driver"
                }
                $smokeJson = Join-Path $temporary "aie4-model-load.json"
                $smokeScreen = Join-Path $temporary "aie4-model-load-screen.txt"
                # The driver AllocConsole()s and takes its own stdout with it,
                # so it has to be a separate process and its streams must be
                # left alone -- piping them makes flm.exe inherit a pipe and
                # spin on a read it can never satisfy. So: a separate
                # PowerShell with untouched streams, and the result read back
                # from the JSON file the driver writes rather than from stdout.
                $psExe = (Get-Process -Id $PID).Path
                & $psExe -NoProfile -ExecutionPolicy Bypass -File $driver `
                    -FlmExe $resolvedFlm `
                    -ModelTag 'phi4-mini-it-aie4:4b' `
                    -Turns '/status' `
                    -OutJson $smokeJson `
                    -ScreenLog $smokeScreen
                $driverExit = $LASTEXITCODE
                if (-not (Test-Path -LiteralPath $smokeJson)) {
                    throw (
                        "the console driver exited $driverExit and wrote no " +
                        "result, so nothing is known about the model load")
                }
                $smoke = Get-Content $smokeJson -Raw | ConvertFrom-Json
                if ($driverExit -ne 0 -or -not $smoke.reached_prompt) {
                    throw (
                        "Clean-environment AIE4 model load failed (driver " +
                        "exit $driverExit): $($smoke.error)`n" +
                        "$($smoke.screen_final)")
                }
                # The screen has to name the tag that was loaded. Reaching a
                # prompt alone would also be true of a build that fell back to
                # some other model.
                if ("$($smoke.screen_final)" -notmatch 'phi4-mini-it-aie4') {
                    throw (
                        "the REPL reached its prompt but never named " +
                        "phi4-mini-it-aie4.`n$($smoke.screen_final)")
                }
                $ran += "flm-exe AIE4 model load (console-driven, /status)"
            } else {
                # Reported, because it is the block this test is named for and
                # the one that needs AIE4 hardware. It had no else at all: with
                # -FlmExe supplied the run printed "none skipped" having never
                # loaded a model, which is this branch's recurring defect --
                # a record that reads better than the run it describes.
                $skipped += (
                    "flm-exe AIE4 model load (-RunAie4ModelLoad not " +
                    "supplied; needs AIE4 hardware and the packaged model)")
            }
        } finally {
            $env:PATH = $savedPath
            if ($null -eq $savedCorelib) {
                Remove-Item Env:RYZENAI_CORELIB_PATH `
                    -ErrorAction SilentlyContinue
            } else {
                $env:RYZENAI_CORELIB_PATH = $savedCorelib
            }
            if ($null -eq $savedXrt) {
                Remove-Item Env:XILINX_XRT -ErrorAction SilentlyContinue
            } else {
                $env:XILINX_XRT = $savedXrt
            }
        }

        $portable = Join-Path $temporary "without-aie4"
        Copy-Item `
            -Path (Split-Path $resolvedFlm -Parent) `
            -Destination $portable `
            -Recurse
        Remove-Item (Join-Path $portable "aie4") `
            -Recurse `
            -Force `
            -ErrorAction SilentlyContinue
        & (Join-Path $portable (Split-Path $resolvedFlm -Leaf)) list |
            Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Non-AIE4 command failed without the AIE4 directory"
        }
        $ran += "flm-exe (Step 7 clean environment, Step 8 dumpbin)"
    } else {
        # Names only what this branch skips. It used to also invite
        # -RunAie4ModelLoad, which is (a) irrelevant here, since that flag does
        # nothing without -FlmExe, and (b) an invitation to a hang, back when
        # the model-load block piped a prompt into a REPL that reads through
        # ReadConsoleInput. The model-load skip is now reported where it
        # belongs: on the -FlmExe branch, where the flag is actually consulted.
        $skipped += (
            "flm-exe (Step 7 clean environment, Step 8 dumpbin): pass -FlmExe")
    }

    foreach ($item in $ran) {
        Write-Output "RAN     : $item"
    }
    foreach ($item in $skipped) {
        Write-Output "SKIPPED : $item"
    }
    if ($skipped.Count -gt 0) {
        Write-Output (
            "packaged runtime tests INCOMPLETE: " +
            "$($ran.Count) block(s) ran, $($skipped.Count) skipped")
        $script:exitCode = $SKIP_EXIT
    } else {
        Write-Output (
            "packaged runtime tests passed: $($ran.Count) block(s), " +
            "none skipped")
        $script:exitCode = 0
    }
} finally {
    if (Test-Path $temporary) {
        Remove-Item $temporary -Recurse -Force -ErrorAction SilentlyContinue
    }
}
exit $script:exitCode
