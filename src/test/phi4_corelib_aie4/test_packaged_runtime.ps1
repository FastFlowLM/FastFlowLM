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
$temporary = Join-Path ([System.IO.Path]::GetTempPath()) (
    "flm-aie4-package-{0}-{1}" -f $PID, [DateTime]::UtcNow.Ticks)

function Write-Bytes {
    param([string]$Path, [int]$Count)
    [System.IO.File]::WriteAllBytes($Path, [byte[]]::new($Count))
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
    $output = (& cmake @arguments 2>&1 |
        ForEach-Object { $_.ToString() } |
        Out-String)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    $succeeded = $exitCode -eq 0
    if ($succeeded -ne $ExpectSuccess) {
        throw "Unexpected CMake result.`n$output"
    }
    return $output
}

try {
    $inno = Get-Content (Join-Path $sourceRoot "inno/flm.iss") -Raw
    foreach ($required in @(
        'AppVersion=1.0.4',
        'Name: "aie4runtime"',
        'Source: "aie4\*"',
        'corelib_phi4_manifest.json',
        'tokenizer_config.json',
        'config.json'
    )) {
        if ($inno -notmatch [regex]::Escape($required)) {
            throw "Inno manifest is missing: $required"
        }
    }
    $wix = Get-Content (Join-Path $sourceRoot "wix/flm.wxs") -Raw
    foreach ($required in @(
        'Version="1.0.4"',
        'Feature Id="Aie4Feature"',
        'ComponentGroup Id="Aie4RuntimeComponents"',
        'ComponentGroup Id="Aie4OverlayComponents"',
        'corelib_phi4_manifest.json',
        'tokenizer_config.json',
        'config.json'
    )) {
        if ($wix -notmatch [regex]::Escape($required)) {
            throw "WiX manifest is missing: $required"
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
flm_collect_aie4_runtime_files(_flm_aie4_files)
install(FILES `${_flm_aie4_files} DESTINATION bin/aie4)
"@ | Set-Content -Path (Join-Path $fixture "CMakeLists.txt") -Encoding utf8

    $offBuild = Join-Path $temporary "off"
    Invoke-Configure -Build $offBuild -Enabled $false | Out-Null
    & cmake --install $offBuild --prefix (Join-Path $temporary "off-stage")
    if ($LASTEXITCODE -ne 0) {
        throw "Feature-OFF install failed"
    }
    if (Test-Path (Join-Path $temporary "off-stage/bin/aie4")) {
        throw "Feature OFF unexpectedly staged an AIE4 directory"
    }

    $missingOutput = Invoke-Configure `
        -Build (Join-Path $temporary "missing") `
        -Enabled $true `
        -ExpectSuccess $false
    if ($missingOutput -notmatch "RYZENAI_CORELIB_RUNTIME_DIR") {
        throw "Missing runtime path failure was not actionable"
    }

    $corelib = Join-Path $temporary "corelib"
    $xrt = Join-Path $temporary "xrt"
    New-Item -ItemType Directory -Path $corelib, $xrt | Out-Null
    foreach ($name in @(
        "ryzenai_corelib.dll",
        "ryzen_mm.dll",
        "dyn_bins.dll",
        "spdlog.dll",
        "libprotobuf.dll",
        "fmt.dll",
        "zlib.dll",
        "zlib1.dll",
        "libutf8_validity.dll",
        "abseil_dll.dll"
    )) {
        Write-Bytes -Path (Join-Path $corelib $name) -Count 17
    }
    foreach ($name in @(
        "xrt_coreutil.dll",
        "xrt_core.dll",
        "xrt_umddml.dll",
        "xdp_native_plugin.dll"
    )) {
        Write-Bytes -Path (Join-Path $xrt $name) -Count 19
    }

    $onBuild = Join-Path $temporary "on"
    Invoke-Configure `
        -Build $onBuild `
        -Enabled $true `
        -Corelib $corelib `
        -Xrt $xrt | Out-Null
    $stage = Join-Path $temporary "on-stage"
    & cmake --install $onBuild --prefix $stage
    if ($LASTEXITCODE -ne 0) {
        throw "Feature-ON install failed"
    }
    $actual = @(
        Get-ChildItem (Join-Path $stage "bin/aie4") -File |
            ForEach-Object Name |
            Sort-Object
    )
    $expected = @(
        "dyn_bins.dll",
        "abseil_dll.dll",
        "fmt.dll",
        "libprotobuf.dll",
        "libutf8_validity.dll",
        "ryzen_mm.dll",
        "ryzenai_corelib.dll",
        "spdlog.dll",
        "xdp_native_plugin.dll",
        "xrt_core.dll",
        "xrt_coreutil.dll",
        "xrt_umddml.dll",
        "zlib.dll",
        "zlib1.dll"
    ) | Sort-Object
    if (Compare-Object $expected $actual) {
        throw "Installed AIE4 closure did not match the collected files"
    }

    if ($CorelibRuntimeDir -or $XrtRuntimeDir -or $DependencyDir) {
        if (
            -not $CorelibRuntimeDir -or
            -not $XrtRuntimeDir -or
            -not $DependencyDir
        ) {
            throw "Real closure check requires all three runtime directories"
        }
        $realBuild = Join-Path $temporary "real"
        Invoke-Configure `
            -Build $realBuild `
            -Enabled $true `
            -Corelib (Resolve-Path $CorelibRuntimeDir).Path `
            -Xrt (Resolve-Path $XrtRuntimeDir).Path `
            -Dependency (Resolve-Path $DependencyDir).Path | Out-Null
        $realStage = Join-Path $temporary "real-stage"
        & cmake --install $realBuild --prefix $realStage
        if ($LASTEXITCODE -ne 0) {
            throw "Real AIE4 closure staging failed"
        }

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
        $savedPathForLoad = $env:PATH
        $savedCorelibForLoad = $env:RYZENAI_CORELIB_PATH
        $savedXrtForLoad = $env:XILINX_XRT
        try {
            $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
            Remove-Item Env:RYZENAI_CORELIB_PATH `
                -ErrorAction SilentlyContinue
            Remove-Item Env:XILINX_XRT -ErrorAction SilentlyContinue
            $library = Join-Path `
                $realStage `
                "bin/aie4/ryzenai_corelib.dll"
            $module = [FlmAie4Loader]::LoadLibraryEx(
                $library,
                [IntPtr]::Zero,
                0x00000100 -bor 0x00001000)
            if ($module -eq [IntPtr]::Zero) {
                $errorCode = [Runtime.InteropServices.Marshal]::
                    GetLastWin32Error()
                throw "Clean-environment corelib load failed: $errorCode"
            }
            [FlmAie4Loader]::FreeLibrary($module) | Out-Null
        } finally {
            $env:PATH = $savedPathForLoad
            if ($null -eq $savedCorelibForLoad) {
                Remove-Item Env:RYZENAI_CORELIB_PATH `
                    -ErrorAction SilentlyContinue
            } else {
                $env:RYZENAI_CORELIB_PATH = $savedCorelibForLoad
            }
            if ($null -eq $savedXrtForLoad) {
                Remove-Item Env:XILINX_XRT `
                    -ErrorAction SilentlyContinue
            } else {
                $env:XILINX_XRT = $savedXrtForLoad
            }
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
        Remove-Item $temporary -Recurse -Force
    }
}
