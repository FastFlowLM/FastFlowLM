<#
.SYNOPSIS
  Fetch and verify the pinned HRX amdxdna *Windows public package* release.

.DESCRIPTION
  Windows counterpart of fetch-hrx-release.sh. Downloads the Windows .zip pinned
  in hrx-release.env (HRX_RELEASE_ASSET_WINDOWS / HRX_RELEASE_SHA256_WINDOWS),
  verifies its checksum, extracts it, locates the HRX CMake package config, and
  prints the package prefix to feed find_package(hrx) via CMAKE_PREFIX_PATH.

  After XADX removal, FLM consumes HRX through find_package(hrx CONFIG REQUIRED)
  from the public package (CMake package config + hrx.dll/import lib + public
  headers), not the former HRX_DIR/HRX_BUILD source+build tree. So configure the
  FLM build with:

      -DCMAKE_PREFIX_PATH=<the HRX_CMAKE_PREFIX printed below>

  The archive is extracted into <script-dir>\.hrx-release\<artifact-root>.

  When running in GitHub Actions, also writes `prefix=<package-prefix>` to
  $env:GITHUB_OUTPUT.

.PARAMETER OutDir
  Optional output directory (defaults to <script-dir>\.hrx-release).
#>
[CmdletBinding()]
param(
    [string]$OutDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$here       = $PSScriptRoot
$releaseEnv = Join-Path $here 'hrx-release.env'
if (-not (Test-Path -LiteralPath $releaseEnv)) {
    throw "missing $releaseEnv"
}

# Parse the shell-style .env (KEY=VALUE or KEY="VALUE"; ignore comments/blanks).
$env_vars = @{}
foreach ($line in Get-Content -LiteralPath $releaseEnv) {
    $trimmed = $line.Trim()
    if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }
    if ($trimmed -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        $key = $Matches[1]
        $val = $Matches[2].Trim().Trim('"').Trim("'")
        $env_vars[$key] = $val
    }
}

function Require-Var([string]$name) {
    if (-not $env_vars.ContainsKey($name) -or [string]::IsNullOrWhiteSpace($env_vars[$name])) {
        throw "missing $name in $releaseEnv"
    }
    return $env_vars[$name]
}

$repo   = Require-Var 'HRX_RELEASE_REPO'
$tag    = Require-Var 'HRX_RELEASE_TAG'
$asset  = Require-Var 'HRX_RELEASE_ASSET_WINDOWS'
$sha256 = (Require-Var 'HRX_RELEASE_SHA256_WINDOWS').ToLower()

if ($tag -like 'TODO*' -or $asset -like '*TODO*' -or $sha256 -like 'todo*') {
    throw "hrx-release.env still has TODO placeholders. The Linux agent must " +
          "publish the HRX public package and fill in REPO/TAG/ASSET/SHA256 first."
}

if (-not $OutDir) { $OutDir = Join-Path $here '.hrx-release' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$baseUrl = "https://github.com/$repo/releases/download/$tag"
$zip     = Join-Path $OutDir $asset
$shaFile = "$zip.sha256"

function Download-File([string]$url, [string]$path) {
    Write-Host "download: $url"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing
            return
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds 2
        }
    }
}

Download-File "$baseUrl/$asset" $zip
Download-File "$baseUrl/$asset.sha256" $shaFile

$remoteSha = ((Get-Content -LiteralPath $shaFile | Select-Object -First 1) -split '\s+')[0].ToLower()
if ([string]::IsNullOrWhiteSpace($remoteSha)) { throw "empty remote sha256 file: $shaFile" }
if ($remoteSha -ne $sha256) {
    throw "remote sha256 mismatch: expected $sha256, got $remoteSha"
}

$actualSha = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLower()
if ($actualSha -ne $sha256) {
    throw "zip sha256 mismatch: expected $sha256, got $actualSha"
}

# The archive root directory mirrors the asset name without the .zip suffix.
$artifactRootName = [System.IO.Path]::GetFileNameWithoutExtension($asset)
if ([string]::IsNullOrWhiteSpace($artifactRootName)) { throw "could not determine zip root" }

$artifactRoot = Join-Path $OutDir $artifactRootName
if (Test-Path -LiteralPath $artifactRoot) {
    Remove-Item -LiteralPath $artifactRoot -Recurse -Force
}

Expand-Archive -LiteralPath $zip -DestinationPath $OutDir -Force

# Some archives contain the package files at the top level rather than under a
# directory named after the asset; fall back to the extraction dir in that case.
if (-not (Test-Path -LiteralPath $artifactRoot)) {
    $artifactRoot = $OutDir
}

# Locate the HRX CMake package config (hrx-config.cmake / hrxConfig.cmake). Its
# containing dir is what find_package(hrx CONFIG) needs on CMAKE_PREFIX_PATH; we
# report the package prefix (two levels up from lib/cmake/hrx when present).
$configFile = Get-ChildItem -LiteralPath $artifactRoot -Recurse -File `
    -Include 'hrx-config.cmake', 'hrxConfig.cmake' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $configFile) {
    throw "no HRX CMake package config (hrx-config.cmake/hrxConfig.cmake) found " +
          "under $artifactRoot -- is this a public HRX package built with the " +
          "CMake packaging flow and IREE_HAL_DRIVER_AMDXDNA=ON?"
}

$configDir = Split-Path -Parent $configFile.FullName
# Prefer the install prefix (…/<prefix>/lib/cmake/hrx -> <prefix>); else use the
# config dir directly. Both are valid on CMAKE_PREFIX_PATH.
$prefix = $configDir
if ((Split-Path -Leaf $configDir) -ieq 'hrx') {
    $cmakeDir = Split-Path -Parent $configDir
    if ((Split-Path -Leaf $cmakeDir) -ieq 'cmake') {
        $libDir = Split-Path -Parent $cmakeDir
        $prefix = Split-Path -Parent $libDir
    }
}

Write-Host "HRX_ARTIFACT_ROOT=$artifactRoot"
Write-Host "HRX_CMAKE_CONFIG=$($configFile.FullName)"
Write-Host "HRX_CMAKE_PREFIX=$prefix"
Write-Host ""
Write-Host "Configure FLM with: -DCMAKE_PREFIX_PATH=`"$prefix`""
if ($env:GITHUB_OUTPUT) {
    # Append without a BOM: Windows PowerShell 5.1's `Out-File -Encoding utf8`
    # emits a UTF-8 BOM, which would corrupt the output key name and make
    # steps.<id>.outputs.prefix resolve to empty.
    [System.IO.File]::AppendAllText(
        $env:GITHUB_OUTPUT,
        "prefix=$prefix`n",
        (New-Object System.Text.UTF8Encoding($false)))
}
