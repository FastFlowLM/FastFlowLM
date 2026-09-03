# Re-run the acceptance harness's markdown renderer, offline, against a
# committed record.
#
# WHY THIS EXISTS
#
# The acceptance document is generated from `phi4_aie4_acceptance.json` by a
# renderer that sits at the end of a ninety-minute hardware harness, so until
# now the only way to run it was to run the harness. That meant every change to
# the renderer -- and every hand-edit of the document -- went unchecked, and it
# is exactly how the document came to ship with a UTF-8 BOM: the harness's own
# read-back guard proves the file is fresh and populated, and cannot see that
# it is not a page.
#
# NOTHING HERE IS A COPY OF THE RENDERER. The render block is located in
# `run_real_model_acceptance.ps1` by its own anchors and executed as it stands,
# and the two helper functions it needs are lifted out of the harness's source
# by AST. A copy would agree with the original by construction, which is the
# failure this suite has already paid for twice.
#
# `verify_acceptance_guards.ps1` drives this and compares the result with the
# committed document, so a renderer change that is not reflected in the
# committed artifact -- or a hand-edit that no run would reproduce -- goes red
# offline.
param(
    [Parameter(Mandatory = $true)][string]$Harness,
    [Parameter(Mandatory = $true)][string]$RecordPath,
    [Parameter(Mandatory = $true)][string]$Out
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# NOT `$record`: PowerShell variable names are case-insensitive, so a parameter
# called $Record would silently shadow the harness's `$script:record` and the
# render block would try to index a string.
$suiteDir = Split-Path -Parent (Resolve-Path $Harness).Path
$script:suiteDir = $suiteDir
. (Join-Path $suiteDir 'acceptance_guards.ps1')

$parsed = Get-Content $RecordPath -Raw | ConvertFrom-Json
$script:record = [ordered]@{}
foreach ($p in $parsed.PSObject.Properties) { $script:record[$p.Name] = $p.Value }

$script:runStamp = [string]$script:record['run_stamp']
$Markdown = $Out
$OutJson = $RecordPath
# The harness's own definitions, minus the disk writes: this script must not be
# able to modify the record it is rendering.
function Set-Field { param([string]$Name, $Value) $script:record[$Name] = $Value }
function Save-Record { }

$tok = $null
$perr = $null
$harnessPath = (Resolve-Path $Harness).Path
$hast = [System.Management.Automation.Language.Parser]::ParseFile($harnessPath, [ref]$tok, [ref]$perr)
if (@($perr).Count -gt 0) { throw ("parse error in {0}: {1}" -f $harnessPath, $perr[0].Message) }
$lifted = @()
foreach ($fn in $hast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    if ($fn.Name -in @('Get-RecordSteps', 'ConvertTo-PlainData')) {
        Invoke-Expression $fn.Extent.Text
        $lifted += $fn.Name
    }
}
if ($lifted.Count -lt 2) {
    throw ("expected to lift Get-RecordSteps and ConvertTo-PlainData out of the harness, got: " +
        ($lifted -join ', ') + '. The renderer needs both; a rename here would silently render nothing.')
}

$lines = Get-Content -LiteralPath $harnessPath
$start = -1
$end = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($start -lt 0 -and $lines[$i] -eq '$script:renderError = $null') { $start = $i }
    if ($start -ge 0 -and $lines[$i] -eq '# Exit code. Skipped work reaches it.') { $end = $i - 2; break }
}
if ($start -lt 0 -or $end -lt 0) {
    throw ("could not locate the render block in $harnessPath. It runs from the line " +
        '`$script:renderError = $null` to just before the exit-code section; if either anchor ' +
        'moved, fix this locator rather than deleting the check.')
}
$block = ($lines[$start..$end] -join "`n")
Write-Output ("render block: {0} lines {1}..{2}" -f (Split-Path -Leaf $harnessPath), ($start + 1), ($end + 1))
Invoke-Expression $block
if ($script:renderError) { throw "render failed: $script:renderError" }
Write-Output "wrote $Out"
