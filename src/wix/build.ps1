<#
.SYNOPSIS
    Builds the flm MSI installer with WiX v5.

.DESCRIPTION
    Runs `wix build` against flm.wxs and drops the resulting flm-setup.msi
    into <repo root>/output/. Requires WiX Toolset v5 (wix.exe) on PATH.

.PARAMETER OutputDir
    Directory to write flm-setup.msi into. Defaults to <repo root>/output.

.EXAMPLE
    ./build.ps1
    ./build.ps1 -OutputDir C:\artifacts
#>

[CmdletBinding()]
param(
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $OutputDir) {
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
    $OutputDir = Join-Path $RepoRoot "src/wix/output"
}

if (-not (Get-Command wix -ErrorAction SilentlyContinue)) {
    throw "wix.exe not found on PATH. Install WiX Toolset v5 (https://wixtoolset.org/) first."
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
$OutputDir = (Resolve-Path $OutputDir).Path
$msiPath = Join-Path $OutputDir "flm-setup.msi"

Push-Location $ScriptDir
try {
    Write-Host "Running get_files.bat to stage package\"
    cmd.exe /c "$ScriptDir\get_files.bat"
    if ($LASTEXITCODE -ne 0) {
        throw "get_files.bat failed with exit code $LASTEXITCODE"
    }

    Write-Host "Building flm.wxs -> $msiPath"
    wix build flm.wxs -arch x64 -ext WixToolset.UI.wixext -out $msiPath
    if ($LASTEXITCODE -ne 0) {
        throw "wix build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Built $msiPath"
