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
    [switch]$RunAie4ModelLoad
)

$ErrorActionPreference = "Stop"
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
    $output = (& cmake @arguments 2>&1 |
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
    $output = (& cmake --install $Build --prefix $Prefix 2>&1 |
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

    # The main installer must still build with no AIE4 closure present.
    # Hard-failing here would make the AIE4 feature a precondition of shipping
    # the ordinary NPU2 product, which is the opposite of optional.
    foreach ($script in @("wix/get_files.bat", "inno/get_files.bat")) {
        $text = Get-Content (Join-Path $sourceRoot $script) -Raw
        if ($text -match "exit /b 1") {
            throw "$script still fails the non-AIE4 package build"
        }
    }

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

    if ($CorelibRuntimeDir) {
        $resolvedCorelib = (Resolve-Path $CorelibRuntimeDir).Path
        $realBuild = Join-Path $temporary "real"
        Invoke-Configure `
            -Build $realBuild `
            -Enabled $true `
            -Corelib $resolvedCorelib `
            -Xrt $(if ($XrtRuntimeDir) {
                (Resolve-Path $XrtRuntimeDir).Path } else { "" }) `
            -Dependency $(if ($DependencyDir) {
                (Resolve-Path $DependencyDir).Path } else { "" }) | Out-Null
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
    }

    if ($FlmExe) {
        $resolvedFlm = (Resolve-Path $FlmExe).Path
        $imports = (& dumpbin /nologo /dependents $resolvedFlm | Out-String)
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
                $smoke = "/bye`r`n" |
                    & $resolvedFlm run phi4-mini-it-aie4:4b 2>&1 |
                    Out-String
                if (
                    $LASTEXITCODE -ne 0 -or
                    $smoke -notmatch "phi4-mini-it-aie4"
                ) {
                    throw "Clean-environment AIE4 model load failed.`n$smoke"
                }
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
    }

    Write-Output "packaged runtime tests passed"
} finally {
    if (Test-Path $temporary) {
        Remove-Item $temporary -Recurse -Force -ErrorAction SilentlyContinue
    }
}
