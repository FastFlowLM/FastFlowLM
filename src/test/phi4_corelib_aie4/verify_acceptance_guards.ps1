# Verify the acceptance harness's own guards, offline, against the committed
# record.
#
# Two of the checks in run_real_model_acceptance.ps1 exist because a specific
# thing went wrong, and neither can be exercised without an hour and a half of
# hardware:
#
#   Step 8's history-accumulation gate, which replaced a token-count proxy that
#   could pass on a conversation that was demonstrably not accumulating; and
#
#   the document readback, which replaced a stamp-only check that would accept
#   a freshly written but entirely empty set of verbatim blocks.
#
# A guard that has only ever been reasoned about is not a guard. This runs both
# against the real data in docs/docs/benchmarks/phi4_aie4_acceptance.json --
# including the defect they were written for -- plus controls they must NOT
# fire on. It needs no hardware and no model; run it after touching either
# check.
$ErrorActionPreference = 'Stop'
$json = Join-Path $PSScriptRoot '..\..\..\docs\docs\benchmarks\phi4_aie4_acceptance.json'
$record = Get-Content $json -Raw | ConvertFrom-Json
$step8 = @($record.steps | Where-Object { $_.id -eq '8' })[0]
$perTurn = @($step8.evidence.per_turn)

function Remove-AllWhitespace { param([string]$T) return ($T -replace '\s', '') }

function Test-Containment {
    param($Turns)
    $missing = @()
    for ($k = 1; $k -lt $Turns.Count; $k++) {
        $needle = Remove-AllWhitespace ([string]$Turns[$k - 1].prompt)
        $hay = Remove-AllWhitespace ([string]$Turns[$k].history_verbatim)
        if (-not ($needle.Length -gt 0 -and $hay.Contains($needle))) { $missing += $Turns[$k].index }
    }
    return $missing
}

Write-Output "record run_stamp: $($record.run_stamp)"
Write-Output "turns captured:   $($perTurn.Count)"
Write-Output ''

Write-Output '--- 1. the real, defective conversation ---'
$missing = Test-Containment $perTurn
Write-Output "turns whose history lost the previous turn: $($missing -join ', ')"
Write-Output "gate fires: $($missing.Count -gt 0)"
if ($missing.Count -ne ($perTurn.Count - 1)) {
    Write-Output "UNEXPECTED: expected every one of turns 2..$($perTurn.Count) to have lost its predecessor"
}

Write-Output ''
Write-Output '--- 2. the old token-count proxy, on the same data ---'
$counts = @($perTurn | ForEach-Object { $_.history_tokens })
$growing = $true
for ($k = 1; $k -lt $counts.Count; $k++) {
    if ($counts[$k] -le $counts[$k-1]) { $growing = $false }
}
Write-Output "counts: $($counts -join ', ')  -> monotonic = $growing (old gate fires: $(-not $growing))"

Write-Output ''
Write-Output '--- 3. the near-miss the old proxy would have PASSED ---'
# The token counts an earlier run of this same step actually recorded, with a
# long final turn substituted. The conversation is the same broken one.
$nearMiss = @(76, 81, 94, 200)
$g2 = $true
for ($k = 1; $k -lt $nearMiss.Count; $k++) { if ($nearMiss[$k] -le $nearMiss[$k-1]) { $g2 = $false } }
Write-Output "counts: $($nearMiss -join ', ')  -> monotonic = $g2 (old gate fires: $(-not $g2))"
Write-Output 'containment is unaffected by reply length, so it still fires on this conversation.'

Write-Output ''
Write-Output '--- 4. a HEALTHY accumulating conversation must NOT be flagged ---'
# Built the way an accumulating frontend would render it, and deliberately
# wrapped mid-word to reproduce what the console screen buffer does to a long
# line -- the case that would defeat a whitespace-normalising match.
$t1 = 'My favourite colour is teal. Please remember it.'
$t2 = 'What is my favourite colour?'
$h1 = "History:`n<|user|>$t1<|end|><|assistant|>Noted.<|end|>`nTokens: 36"
$wrapped = "<|user|>My favourite colour is teal. Please remem`nber it.<|end|><|assistant|>Noted.<|end|><|user|>$t2<|end|><|assistant|>Teal.<|end|>"
$h2 = "History:`n$wrapped`nTokens: 74"
$healthy = @(
    [pscustomobject]@{ index = 1; prompt = $t1; history_verbatim = $h1; history_tokens = 36 },
    [pscustomobject]@{ index = 2; prompt = $t2; history_verbatim = $h2; history_tokens = 74 }
)
$missingHealthy = Test-Containment $healthy
Write-Output "turns flagged: $(if ($missingHealthy.Count -eq 0) { '(none)' } else { $missingHealthy -join ', ' })"
Write-Output "gate fires: $($missingHealthy.Count -gt 0)  <- must be False, including across the mid-word wrap"

Write-Output ''
$script:n2ok = ($missing.Count -gt 0 -and $missingHealthy.Count -eq 0)
if ($script:n2ok) {
    Write-Output 'N2 GATE VERIFIED: fires on the recorded defect, silent on an accumulating conversation.'
} else {
    Write-Output 'N2 GATE NOT VERIFIED.'
}

# --- appended: the N5 "document is populated" guard ---
#
# Same discipline as above: the guard is exercised against the shape that
# caused the incident, not merely reasoned about. The incident shape is a
# per_turn row whose fields the renderer cannot find, which Get-Field turns
# into $null instead of an exception.
Write-Output ''
Write-Output '=== N5: the populated-document guard ==='

function Get-Field {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    if ($Object -is [System.Collections.IDictionary]) {
        if ($Object.Contains($Name)) { return $Object[$Name] }
        return $null
    }
    if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
    return $null
}

# Exactly the harness's checks, run over supplied inputs.
function Test-RenderGuard {
    param($RenderedTurnPrompts, [string]$Written, $StepsWithTurns)
    $blank = @($RenderedTurnPrompts | Where-Object { -not $_.prompt -or $_.prompt.Trim().Length -eq 0 })
    if ($blank.Count -gt 0) { return "blank prompt(s) for step(s) $(@($blank | ForEach-Object { $_.step } | Sort-Object -Unique) -join ', ')" }
    foreach ($rt in $RenderedTurnPrompts) {
        if ($Written -notmatch [regex]::Escape($rt.prompt)) { return "step $($rt.step): prompt rendered but absent from the document" }
    }
    foreach ($swt in $StepsWithTurns) {
        $got = @($RenderedTurnPrompts | Where-Object { $_.step -eq $swt.id }).Count
        if ($got -ne $swt.count) { return "step $($swt.id): $($swt.count) turn(s) of evidence but $got rendered" }
    }
    return $null
}

$good = @($perTurn | ForEach-Object { [pscustomobject]@{ step = '8'; prompt = [string]$_.prompt } })
$doc = "run-stamp x`n" + (($good | ForEach-Object { "> $($_.prompt)" }) -join "`n")
$expect = @([pscustomobject]@{ id = '8'; count = $perTurn.Count })

$r = Test-RenderGuard $good $doc $expect
Write-Output "healthy render        -> $(if ($r) { "FLAGGED: $r" } else { 'passes (correct)' })"

# The incident shape: Get-Field found nothing, so every prompt is $null.
$blankRows = @($perTurn | ForEach-Object { [pscustomobject]@{ step = '8'; prompt = (Get-Field $null 'prompt') } })
$r2 = Test-RenderGuard $blankRows $doc $expect
Write-Output "blank prompts         -> $(if ($r2) { "FLAGGED (correct): $r2" } else { 'PASSED -- guard is useless' })"

# Rendered text that never reached the file.
$r3 = Test-RenderGuard $good "run-stamp x`nnothing here" $expect
Write-Output "prompts missing from doc -> $(if ($r3) { "FLAGGED (correct): $r3" } else { 'PASSED -- guard is useless' })"

# Fewer turns rendered than the evidence holds.
$r4 = Test-RenderGuard @($good[0]) $doc $expect
Write-Output "turn dropped          -> $(if ($r4) { "FLAGGED (correct): $r4" } else { 'PASSED -- guard is useless' })"

$n5ok = ((-not $r) -and $r2 -and $r3 -and $r4)
if ($n5ok) {
    Write-Output 'N5 GUARD VERIFIED: silent on a good render, fires on each way it can go wrong.'
} else {
    Write-Output 'N5 GUARD NOT VERIFIED.'
}

Write-Output ''
if ($script:n2ok -and $n5ok) { Write-Output 'ALL OFFLINE GUARD CHECKS PASSED.'; exit 0 }
Write-Output 'OFFLINE GUARD CHECKS FAILED.'
exit 1
