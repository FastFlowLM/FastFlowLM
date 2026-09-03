# Exercise the acceptance harness's guards offline, against the committed
# record.
#
# WHY, AND WHAT THIS SCRIPT IS NOT ALLOWED TO DO
#
# Two checks in run_real_model_acceptance.ps1 exist because a specific thing
# went wrong, and neither can be exercised without ninety minutes of hardware:
# Step 8's history-accumulation gate, and the readback that decides whether the
# rendered document actually contains its verbatim evidence.
#
# The first version of this script RE-IMPLEMENTED both. That is not a test. A
# copy of the implementation agrees with the implementation by construction,
# and nothing fails when the two drift. It cost what you would expect: the
# render guard shipped labelled "verified offline" while containing
# `@(Get-Field ...).Count -gt 0`, which selects every step because
# `@($null).Count` is 1 -- and this script's copy could not see it, because the
# one value it hand-built was the derivation the bug lived in.
#
# So: this script DOT-SOURCES acceptance_guards.ps1 and calls the same
# functions the harness calls, on the real record loaded the way the harness
# loads it. Nothing here may restate a check. If a case needs a value the
# harness derives, derive it with the harness's function.
#
# Controls matter as much as the positive cases. A guard that fires on
# everything is as useless as one that fires on nothing, so every check below
# has an input it must stay silent on.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'acceptance_guards.ps1')

$json = Join-Path $PSScriptRoot '..\..\..\docs\docs\benchmarks\phi4_aie4_acceptance.json'
if (-not (Test-Path $json)) { Write-Output "no committed record at $json"; exit 1 }

# Loaded exactly as the harness reloads it before rendering: ConvertFrom-Json,
# then an [ordered] top level. Shape differences between a live run and a
# reloaded record are themselves a source of bugs, so the test takes the shape
# the guarded code actually sees.
$parsed = Get-Content $json -Raw | ConvertFrom-Json
$record = [ordered]@{}
foreach ($prop in $parsed.PSObject.Properties) { $record[$prop.Name] = $prop.Value }

$step8 = @(@(Get-Field $record 'steps') | Where-Object { (Get-Field $_ 'id') -eq '8' })[0]
# Assigned, then used -- never `@(Get-FieldArray ...)`. See the note on
# Get-FieldArray: `@()` around the call would preserve the comma-wrap and
# report 1. The verifier is not exempt from the trap it exists to catch.
$perTurn = Get-FieldArray (Get-Field $step8 'evidence') 'per_turn'

$failures = @()
function Check {
    param([string]$Name, [bool]$Expected, [bool]$Actual, [string]$Detail = '')
    $ok = ($Expected -eq $Actual)
    Write-Output ("  {0,-58} {1}{2}" -f $Name, $(if ($ok) { 'ok' } else { 'FAILED' }), $(if ($Detail) { "  ($Detail)" } else { '' }))
    if (-not $ok) { $script:failures += $Name }
}

Write-Output "record run_stamp: $(Get-Field $record 'run_stamp')"
Write-Output "turns captured:   $($perTurn.Count)"

# ---------------------------------------------------------------------------
# Get-FieldArray -- the helper the render bug was an absence of
# ---------------------------------------------------------------------------
Write-Output ''
Write-Output '=== Get-FieldArray: absence must be empty, not a one-element array ==='
$ev8 = Get-Field $step8 'evidence'
Check 'Get-FieldCount: missing field -> 0' $true ((Get-FieldCount $ev8 'no_such_field') -eq 0)
Check 'Get-FieldCount: present field -> its length' $true ((Get-FieldCount $ev8 'per_turn') -eq 4)
$absent = Get-FieldArray $ev8 'no_such_field'
Check 'Get-FieldArray: missing field -> empty, not null' $true ($null -ne $absent -and $absent.Count -eq 0)
$present = Get-FieldArray $ev8 'per_turn'
Check 'Get-FieldArray: present field -> its length' $true ($present.Count -eq 4)
# The three shapes that made this subtle, kept as regression witnesses so the
# next person can see why a plain @() is not enough.
Check 'witness: @(Get-Field ...) says 1 for an absent field' $true (@(Get-Field $ev8 'no_such_field').Count -eq 1) 'the shipped bug'
Check 'witness: @(Get-FieldArray ...) says 1 for 4 items' $true (@(Get-FieldArray $ev8 'per_turn').Count -eq 1) 'why counts use Get-FieldCount'

# ---------------------------------------------------------------------------
# Test-HistoryContainment -- Step 8's accumulation gate
# ---------------------------------------------------------------------------
Write-Output ''
Write-Output '=== Test-HistoryContainment ==='
$rows = Test-HistoryContainment $perTurn
$lost = @($rows | Where-Object { -not (Get-Field $_ 'history_contains_previous_turn') })
Write-Output "  recorded defect: turns losing the previous turn = $(@($lost | ForEach-Object { Get-Field $_ 'turn' }) -join ', ')"
Check 'fires on the recorded defective run' $true ($lost.Count -gt 0)
Check 'fires on every comparable turn' $true ($lost.Count -eq ($perTurn.Count - 1))

# Control: a healthy accumulating conversation, built the way an accumulating
# frontend renders one, and deliberately wrapped MID-WORD to reproduce what the
# console screen buffer does to a long line -- the case that defeats a
# whitespace-normalising match.
$t1 = 'My favourite colour is teal. Please remember it.'
$t2 = 'What is my favourite colour?'
$h1 = "History:`n<|user|>$t1<|end|><|assistant|>Noted, teal it is.<|end|>`nTokens: 36"
$h2 = "History:`n<|user|>My favourite colour is teal. Please remem`nber it.<|end|><|assistant|>Noted, teal it i`ns.<|end|><|user|>$t2<|end|><|assistant|>Teal.<|end|>`nTokens: 74"
$healthy = @(
    [pscustomobject]@{ index = 1; prompt = $t1; history_verbatim = $h1 },
    [pscustomobject]@{ index = 2; prompt = $t2; history_verbatim = $h2 }
)
$healthyRows = Test-HistoryContainment $healthy
$healthyLost = @($healthyRows | Where-Object { -not (Get-Field $_ 'history_contains_previous_turn') })
Check 'silent on a healthy conversation wrapped mid-word' $true ($healthyLost.Count -eq 0)

# Control: the near-miss the retired token-count proxy would have passed. Same
# broken conversation; only the reply lengths differ.
$counts = @($perTurn | ForEach-Object { Get-Field $_ 'history_tokens' })
$nearMiss = @(76, 81, 94, 200)
$proxyFires = $false
for ($k = 1; $k -lt $nearMiss.Count; $k++) { if ($nearMiss[$k] -le $nearMiss[$k - 1]) { $proxyFires = $true } }
Write-Output "  retired proxy on $($nearMiss -join ', ') -> fires = $proxyFires (recorded run was $($counts -join ', '))"
Check 'the retired proxy would have MISSED the near-miss' $true (-not $proxyFires) 'why it was replaced'

# Dropping only the assistant reply must still fail: C21 is about the COMPLETE
# rendered history, and a prompt-only check would pass this.
$repliesDropped = @(
    [pscustomobject]@{ index = 1; prompt = $t1; history_verbatim = $h1 },
    [pscustomobject]@{ index = 2; prompt = $t2
        history_verbatim = "History:`n<|user|>$t1<|end|><|user|>$t2<|end|><|assistant|>Teal.<|end|>`nTokens: 40" }
)
$droppedRows = Test-HistoryContainment $repliesDropped
$droppedLost = @($droppedRows | Where-Object { -not (Get-Field $_ 'history_contains_previous_turn') })
Check 'fires when replies are dropped but prompts kept' $true ($droppedLost.Count -gt 0)

# ---------------------------------------------------------------------------
# Get-StepsWithTurns -- the derivation the render bug lived in
# ---------------------------------------------------------------------------
Write-Output ''
Write-Output '=== Get-StepsWithTurns ==='
$swt = Get-StepsWithTurns $record
Write-Output "  steps with per-turn evidence: $(@($swt | ForEach-Object { $_.id }) -join ', ') (of $(@(Get-Field $record 'steps').Count) steps)"
Check 'selects only steps that have turns' $true ($swt.Count -eq 1 -and $swt[0].id -eq '8')
Check 'reports the right turn count' $true ($swt[0].count -eq $perTurn.Count)

# ---------------------------------------------------------------------------
# Test-RenderedDocument -- driven with what the renderer would have produced
# ---------------------------------------------------------------------------
Write-Output ''
Write-Output '=== Test-RenderedDocument ==='
$good = @(@($perTurn) | ForEach-Object {
    [pscustomobject]@{ step = '8'; prompt = [string](Get-Field $_ 'prompt') } })
$doc = "<!-- run-stamp: x -->`n" + ((@($good) | ForEach-Object { "> $($_.prompt)" }) -join "`n")

Check 'silent on a good render' $true ($null -eq (Test-RenderedDocument -Record $record -RenderedTurnPrompts $good -Written $doc))

# The incident shape: Get-Field found nothing, so every prompt is $null.
$blankRows = @($perTurn | ForEach-Object { [pscustomobject]@{ step = '8'; prompt = $null } })
Check 'fires on blank prompts' $true ($null -ne (Test-RenderedDocument -Record $record -RenderedTurnPrompts $blankRows -Written $doc))
Check 'fires when the text never reached the file' $true ($null -ne (Test-RenderedDocument -Record $record -RenderedTurnPrompts $good -Written "<!-- run-stamp: x -->`nnothing here"))
Check 'fires when a turn is dropped' $true ($null -ne (Test-RenderedDocument -Record $record -RenderedTurnPrompts @($good[0]) -Written $doc))

# The regression that motivated this file: with the old presence test, a
# HEALTHY render threw on the first step that has no per_turn evidence. If that
# ever comes back, 'silent on a good render' above goes red -- but name it
# explicitly so the reason is legible in the output.
$firstProblem = Test-RenderedDocument -Record $record -RenderedTurnPrompts $good -Written $doc
Check 'does not throw on steps that simply have no turns' $true ($null -eq $firstProblem) $(if ($firstProblem) { $firstProblem } else { 'all 20 steps considered' })

Write-Output ''
if ($failures.Count -eq 0) {
    Write-Output 'ALL OFFLINE GUARD CHECKS PASSED.'
    exit 0
}
Write-Output "OFFLINE GUARD CHECKS FAILED: $($failures -join '; ')"
exit 1
