# Exercise the acceptance harness's guards offline, against the committed
# record -- and lint the harness's CALL SITES, which is where the last two bugs
# actually were.
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
# THAT WAS STILL NOT ENOUGH, AND THE SECOND BUG PROVED IT. Extracting the
# guards into shared functions moved the untested surface; it did not remove
# it. The very commit that shared them wrote the Step 8 call site as
# `@(Test-HistoryContainment $perTurn)` -- the comma-wrap trap that
# acceptance_guards.ps1 documents in its own comments -- and the accumulation
# gate could no longer fire at all. Every function-level check here passed,
# because this script assigns and the harness wrapped. A verifier that reads
# only guard bodies is compatible with a harness that does not work.
#
# So there is a third section below: an AST lint that reads the harness's
# STATEMENTS. It derives which guard functions return a comma-wrapped array
# (from their source, not from a list somebody has to maintain), finds every
# call to one of them, and fails on the forms that silently collapse the
# result. It has its own positive control, because a lint that finds nothing
# looks exactly like a lint that passes.
#
# Controls matter as much as the positive cases. A guard that fires on
# everything is as useless as one that fires on nothing, so every check below
# has an input it must stay silent on.
#
# COUNTING: `Check` is coverage -- it can go red because shipped code is wrong.
# `Note` is documentation -- it pins a PowerShell behaviour or a retired
# mechanism, it is legible and cheap, and it is NOT evidence about this
# harness. The summary reports the two separately so "N checks passed" is never
# read as N units of coverage.

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
# report 1. The verifier is not exempt from the trap it exists to catch, and
# the lint at the bottom of this file reads this file too.
$perTurn = Get-FieldArray (Get-Field $step8 'evidence') 'per_turn'

$failures = @()
$script:coverageCount = 0
$script:noteCount = 0

function Check {
    param([string]$Name, [bool]$Expected, [bool]$Actual, [string]$Detail = '')
    $ok = ($Expected -eq $Actual)
    $script:coverageCount++
    Write-Output ("  {0,-58} {1}{2}" -f $Name, $(if ($ok) { 'ok' } else { 'FAILED' }), $(if ($Detail) { "  ($Detail)" } else { '' }))
    if (-not $ok) { $script:failures += $Name }
}

# Same assertion strength, different claim. A Note that goes red is still a
# failure -- it just cannot go red for a reason that says anything about this
# harness, so it is not counted as coverage.
function Note {
    param([string]$Name, [bool]$Expected, [bool]$Actual, [string]$Detail = '')
    $ok = ($Expected -eq $Actual)
    $script:noteCount++
    Write-Output ("  {0,-58} {1}{2}" -f $Name, $(if ($ok) { 'witness' } else { 'FAILED' }), $(if ($Detail) { "  ($Detail)" } else { '' }))
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

# A step recorded with no evidence gets the default `@{}`, which serialises as
# `{}` and reloads as a PSCustomObject with ZERO properties -- and the harness
# reloads the record from disk before it renders. Get-Field used to throw on
# exactly that object, killing the document with "The property 'Name' cannot be
# found on this object", which is how a partial run ends in DOCUMENT NOT
# RENDERED. Found by driving the real renderer offline over a seeded record;
# these three cases are what stop it coming back.
$propertyless = '{}' | ConvertFrom-Json
$threw = $false
$got = 'unset'
try { $got = Get-Field $propertyless 'anything' } catch { $threw = $true }
Check 'Get-Field: property-less object -> $null, never a throw' $true ((-not $threw) -and ($null -eq $got))
Check 'Get-FieldCount: property-less object -> 0' $true ((Get-FieldCount $propertyless 'anything') -eq 0)
$populated = '{"a":1,"b":null}' | ConvertFrom-Json
Check 'Get-Field: still reads a present field, and a present null' $true `
    ((Get-Field $populated 'a') -eq 1 -and $null -eq (Get-Field $populated 'b') -and $null -eq (Get-Field $populated 'zz'))

Check 'Get-FieldCount: missing field -> 0' $true ((Get-FieldCount $ev8 'no_such_field') -eq 0)
Check 'Get-FieldCount: present field -> its length' $true ((Get-FieldCount $ev8 'per_turn') -eq 4)
$absent = Get-FieldArray $ev8 'no_such_field'
Check 'Get-FieldArray: missing field -> empty, not null' $true ($null -ne $absent -and $absent.Count -eq 0)
$present = Get-FieldArray $ev8 'per_turn'
Check 'Get-FieldArray: present field -> its length' $true ($present.Count -eq 4)
# Documentation, not coverage. This asserts a PowerShell language behaviour
# that will not change, so it can never catch a defect in this suite; it is
# here so the next reader can see in one line why the helpers exist. The
# behaviour it describes is now POLICED by the AST lint below, which can fail
# for a reason that matters.
Note 'witness: @($null).Count is 1, which is why presence tests moved' $true (@(Get-Field $ev8 'no_such_field').Count -eq 1) 'the shipped bug, in one expression'

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
#
# Documentation, not coverage: the proxy it models was deleted in round 2, so
# this re-implements a mechanism nothing ships and asserts on hand-built
# numbers. It is kept because "why was the gate replaced" is the first question
# a reader of Test-HistoryContainment asks.
$counts = @($perTurn | ForEach-Object { Get-Field $_ 'history_tokens' })
$nearMiss = @(76, 81, 94, 200)
$proxyFires = $false
for ($k = 1; $k -lt $nearMiss.Count; $k++) { if ($nearMiss[$k] -le $nearMiss[$k - 1]) { $proxyFires = $true } }
Write-Output "  retired proxy on $($nearMiss -join ', ') -> fires = $proxyFires (recorded run was $($counts -join ', '))"
Note 'witness: the retired proxy would have MISSED the near-miss' $true (-not $proxyFires) 'why it was replaced'

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

# The reply probe is coupled to Phi-4's chat template. If the markers are not
# there -- another model, or a /history format change -- "reply missing" is the
# WRONG reading: the reply is not checkable, and scoring it as dropped would
# fail a healthy conversation for a rendering reason. The row must say which
# happened, and the harness reports the uncheckable case as its own problem
# rather than degrading silently to a prompt-only gate.
$noMarkers = @(
    [pscustomobject]@{ index = 1; prompt = $t1; history_verbatim = "History:`nUser: $t1`nAssistant: Noted, teal it is.`nTokens: 36" },
    [pscustomobject]@{ index = 2; prompt = $t2
        history_verbatim = "History:`nUser: My favourite colour is teal. Please remember it.`nAssistant: Noted, teal it is.`nUser: $t2`nAssistant: Teal.`nTokens: 74" }
)
$noMarkerRows = Test-HistoryContainment $noMarkers
Check 'marks the reply half UNCHECKABLE when the template markers are absent' `
    $true (-not (Get-Field $noMarkerRows[0] 'previous_reply_checkable'))
Check 'does not fail an accumulating history just for missing markers' `
    $true ([bool](Get-Field $noMarkerRows[0] 'history_contains_previous_turn'))
Check 'the recorded run IS checkable, so its verdict is the strong one' `
    $true (@($rows | Where-Object { -not (Get-Field $_ 'previous_reply_checkable') }).Count -eq 0)

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

# ---------------------------------------------------------------------------
# Test-LooksLikeEnglish, driven against the two real degenerate replies
# ---------------------------------------------------------------------------
#
# The committed record contains two long turns that failed, and the harness
# gave BOTH of them the single reason "the reply is dominated by a repeated
# token". That is accurate for one of them and wrong for the other, and the
# wrong wording hid the more alarming signature: a reply that produced hundreds
# of clean tokens and then broke mid-word into high-entropy punctuation. The
# whole-text letter ratio could not see it, because the clean 85% carried the
# average above the floor.
#
# These cases run the SHIPPED function over the SHIPPED text. No fixture
# restates what the collapse looks like -- if it did, it would agree with the
# implementation by construction, which is the failure this verifier exists to
# avoid.
Write-Output ''
Write-Output '=== Test-LooksLikeEnglish against the recorded replies ==='

$replies = @{}
foreach ($turn in $perTurn) {
    $replies[[string](Get-Field $turn 'index')] = [string](Get-Field $turn 'reply_verbatim')
}
Write-Output ("  turns with a recorded reply: " + (($replies.Keys | Sort-Object) -join ', '))
Check 'all four recorded turns have reply text to judge' $true `
    (@($replies.Values | Where-Object { $_.Length -gt 0 }).Count -eq 4)

$collapse = 'stops producing language'
$verdicts = @{}
foreach ($index in @('1', '2', '3', '4')) {
    $verdicts[$index] = Test-LooksLikeEnglish -Text $replies[$index]
}
foreach ($index in @('1', '2', '3', '4')) {
    Write-Output ("  turn {0}: {1}" -f $index,
        $(if ($verdicts[$index].Ok) { 'reads as English' }
          else { ($verdicts[$index].Reasons) -join '; ' }))
}

# The turn that collapsed, and only that turn.
Check 'the collapse is reported for the turn that collapsed' $true `
    (@($verdicts['3'].Reasons | Where-Object { $_ -match $collapse }).Count -eq 1)
Check 'and NOT for the turn that only repeated itself' $true `
    (@($verdicts['1'].Reasons | Where-Object { $_ -match $collapse }).Count -eq 0)
Check 'and not for either healthy short reply' $true `
    ((@($verdicts['2'].Reasons | Where-Object { $_ -match $collapse }).Count -eq 0) -and
     (@($verdicts['4'].Reasons | Where-Object { $_ -match $collapse }).Count -eq 0))

# Repetition is common to BOTH long turns. The record's original wording was
# not wrong about that -- it was incomplete.
Check 'repetition is still reported for both long turns' $true `
    ((@($verdicts['1'].Reasons | Where-Object { $_ -match 'repeated token or phrase' }).Count -eq 1) -and
     (@($verdicts['3'].Reasons | Where-Object { $_ -match 'repeated token or phrase' }).Count -eq 1))

# Controls: the two short turns are the healthy case, and they must pass
# outright. A coherence check that fails everything says nothing.
Check 'the two healthy short replies pass' $true `
    ($verdicts['2'].Ok -and $verdicts['4'].Ok) `
    $(if (-not ($verdicts['2'].Ok -and $verdicts['4'].Ok)) {
        (($verdicts['2'].Reasons + $verdicts['4'].Reasons) -join '; ') } else { '' })

# The offset is the point of the message: it is the only precise entry point a
# later investigation has, so require it to be a number inside the text rather
# than prose.
$offsetReason = @($verdicts['3'].Reasons | Where-Object { $_ -match $collapse })[0]
$offsetOk = $offsetReason -match 'at character (\d+) of (\d+)'
$offset = if ($offsetOk) { [int]$Matches[1] } else { -1 }
$length = if ($offsetOk) { [int]$Matches[2] } else { -1 }
Check 'the collapse is reported with a usable character offset' $true `
    ($offsetOk -and $offset -gt 0 -and $offset -lt $length) `
    $(if ($offsetOk) { "char $offset of $length" } else { $offsetReason })

# A negative control for the segment scan itself: language throughout, of the
# same length as the real replies, must stay silent. Without this, a scan that
# fired on any long text would pass every case above.
$longHealthy = (('The quick brown fox jumps over the lazy dog and then considers ' +
                 'the matter carefully before continuing on its way. ') * 300)
$healthyVerdict = Test-LooksLikeEnglish -Text $longHealthy
Check 'the segment scan is silent on a long, wholly alphabetic reply' $true `
    (@($healthyVerdict.Reasons | Where-Object { $_ -match $collapse }).Count -eq 0) `
    $("$($longHealthy.Length) characters")

# ---------------------------------------------------------------------------
# The rollup must fail CLOSED on a status nobody recognises
# ---------------------------------------------------------------------------
#
# The acceptance verdict failed OPEN: `$rank['<unknown>']` is $null, `$null -gt 0`
# is $false, so an unrecognised step status rolled up to `met` and reached none
# of the counters that feed the exit ladder. `RESULT: ACCEPTED`, exit 0.
#
# These cases drive Get-UnrecognisedStepStatus, the function the harness itself
# calls -- not a copy of it. The harness's use of it is separately checked by
# the AST section below and by the explicit call-site assertion at the end.
Write-Output ''
Write-Output '=== Get-UnrecognisedStepStatus: an unknown status must be a hard failure ==='

# Assigned, never `@(Get-FieldArray ...)`. Same trap as $perTurn above.
$recordedStepsForCheck = Get-FieldArray $record 'steps'
$rankTable = Get-StepStatusRank
Check 'the rank table has exactly the four known statuses' $true `
    ($rankTable.Count -eq 4 -and $rankTable.ContainsKey('met') -and $rankTable.ContainsKey('partial') -and
     $rankTable.ContainsKey('not_exercised') -and $rankTable.ContainsKey('not_met'))

# Documentation, not coverage: the language behaviour the bug was made of.
Note 'witness: $rank[<unknown>] is $null and $null -gt 0 is $false' $true `
    (($null -eq $rankTable['bogus']) -and (-not ($rankTable['bogus'] -gt $rankTable['met']))) `
    'the whole defect, in one expression'

# Control first. The committed record must be silent, or every positive case
# below is meaningless.
$realUnknown = Get-UnrecognisedStepStatus $recordedStepsForCheck
Check 'silent on the committed record' $true ($realUnknown.Count -eq 0) `
    $(if ($realUnknown.Count) { $realUnknown -join '; ' } else { "$($recordedStepsForCheck.Count) steps" })

# The four statuses the harness may legitimately produce, plus the wrong-case
# spelling. PowerShell hashtable lookup is case-insensitive, so 'MET' IS
# recognised and correctly ranks as met -- asserting otherwise would be a
# check that agrees with a wrong belief about the language.
$legal = @('met', 'partial', 'not_exercised', 'not_met', 'MET') | ForEach-Object {
    [pscustomobject]@{ id = "legal-$_"; status = $_ } }
$legalUnknown = Get-UnrecognisedStepStatus $legal
Check 'silent on every legal status, including wrong case' $true ($legalUnknown.Count -eq 0) `
    $(if ($legalUnknown.Count) { $legalUnknown -join '; ' } else { 'met, partial, not_exercised, not_met, MET' })

# The reachable shapes: a status from an older schema, a truncated record, a
# hand-edited one. `-Append` merges $OutJson wholesale and Add-UnselectedStep
# carries a prior step forward verbatim, so none of these is hypothetical.
$illegal = @(
    [pscustomobject]@{ id = '1'; status = 'failed' },
    [pscustomobject]@{ id = '2'; status = '' },
    [pscustomobject]@{ id = '3'; status = 'bogus' },
    [pscustomobject]@{ id = '4'; status = 'met' }
)
$illegalUnknown = Get-UnrecognisedStepStatus $illegal
Check 'fires on every unrecognised status and only those' $true `
    ($illegalUnknown.Count -eq 3) $($illegalUnknown -join '; ')
Check 'the diagnostic names the step and the status it did not recognise' $true `
    ((@($illegalUnknown | Where-Object { $_ -match "^step 1: 'failed'$" }).Count -eq 1) -and
     (@($illegalUnknown | Where-Object { $_ -match "^step 3: 'bogus'$" }).Count -eq 1))

# A step with no `status` field at all -- the shape a truncated record has.
# Get-Field returns $null, [string]$null is '', and '' is not in the table.
$statusless = @('{"id":"9"}' | ConvertFrom-Json)
$statuslessUnknown = Get-UnrecognisedStepStatus $statusless
Check 'fires on a step carrying no status field at all' $true ($statuslessUnknown.Count -eq 1) `
    $($statuslessUnknown -join '; ')

# ---------------------------------------------------------------------------
# Publishability: a UTF-8 BOM stops a Jekyll page being a page
# ---------------------------------------------------------------------------
#
# The signed-off acceptance record shipped with a BOM and was therefore not a
# page: no front matter, no layout, no nav entry, no pretty URL, and no entry
# in search.json. `Set-Content -Encoding utf8` means UTF-8 WITH BOM under
# Windows PowerShell 5.1, and the harness's read-back stamp check cannot see it
# because every text reader strips a BOM.
Write-Output ''
Write-Output '=== UTF-8 BOM: no .md under docs/ may carry one ==='

$docsRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\docs')).Path
$mdCount = @(Get-ChildItem -LiteralPath $docsRoot -Recurse -File -Filter '*.md').Count
# Non-vacuity: a scan that found no files would pass for the wrong reason.
Check 'the BOM scan found markdown to scan' $true ($mdCount -ge 40) "$mdCount .md files under docs/"
$bomFiles = Get-BomMarkdownFile $docsRoot
Check 'no .md under docs/ carries a UTF-8 BOM' $true ($bomFiles.Count -eq 0) `
    $(if ($bomFiles.Count) { ($bomFiles | ForEach-Object { Split-Path -Leaf $_ }) -join ', ' } else { "$mdCount files clean" })

# Positive control: the detector must actually detect. Written with the exact
# cmdlet and parameter that caused the incident, so this case also pins the
# claim that `Set-Content -Encoding utf8` emits a BOM on this host.
$bomProbe = Join-Path ([System.IO.Path]::GetTempPath()) ("bomprobe-$PID-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $bomProbe | Out-Null
try {
    # The probe file is written with EXPLICIT BOM BYTES, not with
    # `Set-Content -Encoding utf8`.
    #
    # That cmdlet is where the defect came from, but it is not a reliable way
    # to PRODUCE one: `-Encoding utf8` means UTF-8-with-BOM in Windows
    # PowerShell 5.1 and UTF-8-without-BOM in PowerShell 7. Building the
    # positive control out of it made the detector's own test pass vacuously
    # under pwsh 7 -- a check that cannot fail, in the file whose subject is
    # checks that cannot fail. The host difference is reported separately
    # below, as the documentation it is.
    $withBom = Join-Path $bomProbe 'explicit-bom.md'
    [System.IO.File]::WriteAllBytes($withBom,
        ([byte[]](0xEF, 0xBB, 0xBF) + [System.Text.Encoding]::UTF8.GetBytes("---`nlayout: docs`n---`n")))
    Check 'Test-Utf8Bom detects an explicit BOM' $true (Test-Utf8Bom $withBom)

    # Documentation, not coverage: WHY the defect shipped, and why it would not
    # have been caught by running the harness under pwsh. Asserted against the
    # host actually running, so it stays true on both rather than pinning one.
    $setContentProbe = Join-Path $bomProbe 'set-content-utf8.txt'
    "x" | Set-Content -Path $setContentProbe -Encoding utf8
    $setContentBom = Test-Utf8Bom $setContentProbe
    $isDesktop = ($PSVersionTable.PSEdition -ne 'Core')
    Note 'witness: Set-Content -Encoding utf8 emits a BOM only on Windows PowerShell' $true `
        ($setContentBom -eq $isDesktop) `
        "$($PSVersionTable.PSEdition) $($PSVersionTable.PSVersion): BOM=$setContentBom"

    $withoutBom = Join-Path $bomProbe 'write-utf8-nobom.md'
    Write-Utf8NoBom -Path $withoutBom -Text "---`nlayout: docs`n---`n"
    Check 'Write-Utf8NoBom writes no BOM' $true (-not (Test-Utf8Bom $withoutBom))

    $probeHits = Get-BomMarkdownFile $bomProbe
    Check 'the scan finds a BOM when there is one, and only that file' $true `
        ($probeHits.Count -eq 1 -and $probeHits[0] -eq $withBom) `
        "$($probeHits.Count) of 2 files flagged"
} finally {
    Remove-Item -LiteralPath $bomProbe -Recurse -Force -ErrorAction SilentlyContinue
}

# Source lint: the statement that caused it must not come back.
#
# Scoped to the PUBLISHED DOCUMENT rather than banning the cmdlet outright.
# The harness's other `Set-Content -Encoding utf8` calls write run artifacts
# and HTTP request bodies under $script:artifactDir -- a BOM there is
# tolerated (nlohmann's lexer skips one) and none of them is parsed by Jekyll.
# The one that mattered is the one that writes $Markdown, and that write must
# go through Write-Utf8NoBom.
$docWrittenBySetContent = @()
$docWrittenByGuardedWriter = 0
$otherSetContentUtf8 = @()
foreach ($p in @((Join-Path $PSScriptRoot 'run_real_model_acceptance.ps1'),
                 (Join-Path $PSScriptRoot 'acceptance_guards.ps1'))) {
    # Parsed directly: Get-Ast is defined with the call-site lint further down,
    # and moving it above this section would put the lint's helper a long way
    # from the lint.
    $lintTokens = $null
    $lintErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($p, [ref]$lintTokens, [ref]$lintErrors)
    if (@($lintErrors).Count -gt 0) { throw ("parse error in {0}: {1}" -f $p, $lintErrors[0].Message) }
    foreach ($cmd in $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true)) {
        $cn = $cmd.GetCommandName()
        $where = "{0}:{1}" -f (Split-Path -Leaf $p), $cmd.Extent.StartLineNumber
        $touchesDoc = ($cmd.Extent.Text -match '\$Markdown\b')
        if ($cn -eq 'Set-Content') {
            if ($touchesDoc) { $docWrittenBySetContent += $where }
            elseif ($cmd.Extent.Text -match '-Encoding\s+utf8') { $otherSetContentUtf8 += $where }
        } elseif ($cn -eq 'Write-Utf8NoBom' -and $touchesDoc) {
            $docWrittenByGuardedWriter++
        }
    }
}
Check 'the acceptance document is never written with Set-Content' $true `
    ($docWrittenBySetContent.Count -eq 0) $($docWrittenBySetContent -join ', ')
# Non-vacuity: the check above is also satisfied by a harness that writes the
# document nowhere at all.
Check 'the acceptance document is written through Write-Utf8NoBom' $true `
    ($docWrittenByGuardedWriter -eq 1) "$docWrittenByGuardedWriter call site(s)"
Note 'witness: other Set-Content -Encoding utf8 sites, all run artifacts' $true `
    ($otherSetContentUtf8.Count -gt 0) ($otherSetContentUtf8 -join ', ')

# ---------------------------------------------------------------------------
# The links out of this branch's pages must resolve
# ---------------------------------------------------------------------------
#
# `docs/_config.yml` sets `permalink: pretty` and `baseurl: ""`, so a page
# builds to `<name>/index.html` and is served from a DIRECTORY. Two link forms
# are wrong under that scheme and both shipped on this branch: a `.html`
# suffix, which no built page has; and a relative link from a non-index page,
# which resolves one level too deep because the page IS a directory. The
# acceptance record's new link to the provenance page -- the page carrying the
# correction to a wrong failure reason in the record itself -- was written the
# second way in this very fix round and would have 404'd.
#
# Scoped to the three pages this branch owns. A repository-wide checker also
# flags one pre-existing upstream page, and turning an upstream defect into a
# red gate here is not this round's business.
Write-Output ''
Write-Output '=== Internal links out of the branch pages resolve to real pages ==='

$docsForLinks = @('docs\docs\benchmarks\phi4_aie4_acceptance.md',
                  'docs\docs\benchmarks\phi4_aie4_acceptance_provenance.md',
                  'docs\docs\models\phi.md')
$repoRootForLinks = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path

# Every page's built URL, derived the way Jekyll derives it under
# `permalink: pretty` -- an index page maps to its directory, any other page to
# a directory of its own name.
$pageUrls = @{}
foreach ($f in @(Get-ChildItem -LiteralPath $docsRoot -Recurse -File -Filter '*.md')) {
    $rel = $f.FullName.Substring($docsRoot.Length).TrimStart('\').Replace('\', '/')
    $url = $(if ($rel -match '(^|/)index\.md$') {
        '/' + ($rel -replace '(^|/)index\.md$', '') + '/'
    } else {
        '/' + ($rel -replace '\.md$', '') + '/'
    })
    $pageUrls[($url -replace '/+', '/')] = $rel
}
Check 'the link checker enumerated the site pages' $true ($pageUrls.Count -ge 40) "$($pageUrls.Count) pages"

$linkProblems = @()
$linksChecked = 0
foreach ($rel in $docsForLinks) {
    $full = Join-Path $repoRootForLinks $rel
    if (-not (Test-Path $full)) { $linkProblems += "$rel : missing"; continue }
    $relPosix = ($rel -replace '\\', '/') -replace '^docs/', ''
    $base = $(if ($relPosix -match '(^|/)index\.md$') {
        '/' + ($relPosix -replace '(^|/)index\.md$', '') + '/'
    } else {
        '/' + ($relPosix -replace '\.md$', '') + '/'
    }) -replace '/+', '/'
    foreach ($m in [regex]::Matches((Get-Content -LiteralPath $full -Raw), '\]\(([^)\s]+)\)')) {
        $target = ($m.Groups[1].Value -split '#')[0]
        if (-not $target -or $target -match '^(https?:|mailto:|#)') { continue }
        $linksChecked++
        if ($target -match '\{\{') { $linkProblems += "$rel -> $target (Liquid in an internal link)"; continue }
        if ($target -match '\.html$') { $linkProblems += "$rel -> $target (.html suffix; permalink is pretty)"; continue }
        $joined = $(if ($target.StartsWith('/')) { $target } else { $base + $target })
        $parts = New-Object System.Collections.Generic.List[string]
        foreach ($seg in ($joined -split '/')) {
            if ($seg -eq '' -or $seg -eq '.') { continue }
            if ($seg -eq '..') { if ($parts.Count -gt 0) { $parts.RemoveAt($parts.Count - 1) }; continue }
            $parts.Add($seg)
        }
        $resolved = '/' + ($parts -join '/') + '/'
        if (-not $pageUrls.ContainsKey($resolved)) {
            $linkProblems += "$rel -> $target (resolves to $resolved, no such page)"
        }
    }
}
# Non-vacuity: pages with no links would pass for the wrong reason.
Check 'the branch pages contain internal links to check' $true ($linksChecked -ge 4) "$linksChecked links"
Check 'every internal link out of the branch pages resolves' $true ($linkProblems.Count -eq 0) `
    $(if ($linkProblems.Count) { $linkProblems -join ' | ' } else { "$linksChecked links against $($pageUrls.Count) pages" })

# ---------------------------------------------------------------------------
# The committed document must be what the renderer produces
# ---------------------------------------------------------------------------
#
# Until now the renderer could only be run by running ninety minutes of
# hardware, so nothing checked that the committed document and the committed
# record still agreed -- which is how a BOM, and before it a stale Step 9b
# claim, reached the published artifact. rerender_acceptance_document.ps1
# executes the harness's own render block against the committed record and
# this compares the result with the file in docs/.
#
# Only the "Generated <clock> from <path>" line may differ: it is render-time
# provenance, not content. Exactly one line is allowed to differ, and that is
# asserted rather than assumed.
Write-Output ''
Write-Output '=== The committed acceptance document re-renders from the committed record ==='

$rerenderScript = Join-Path $PSScriptRoot 'rerender_acceptance_document.ps1'
$committedMd = Join-Path $PSScriptRoot '..\..\..\docs\docs\benchmarks\phi4_aie4_acceptance.md'
$rerenderOut = Join-Path ([System.IO.Path]::GetTempPath()) ("rerender-$PID-" + [guid]::NewGuid().ToString('N') + '.md')
try {
    # Re-invoked in the SAME host that is running this verifier, not a
    # hard-coded `powershell`. The renderer's output is host-sensitive in
    # exactly the way that caused the BOM, so checking it under 5.1 while the
    # operator runs pwsh 7 would be checking the wrong thing.
    $psHost = (Get-Process -Id $PID).Path
    $rerenderLog = & $psHost -NoProfile -ExecutionPolicy Bypass -File $rerenderScript `
        -Harness (Join-Path $PSScriptRoot 'run_real_model_acceptance.ps1') `
        -RecordPath $json -Out $rerenderOut 2>&1
    $rendered = Test-Path $rerenderOut
    Check 'the renderer runs offline against the committed record' $true $rendered `
        $(if ($rendered) { ($rerenderLog | Select-Object -First 1) } else { ($rerenderLog -join ' | ') })
    if ($rendered) {
        Check 'the re-rendered document carries no BOM' $true (-not (Test-Utf8Bom $rerenderOut))
        $committedLines = @(Get-Content -LiteralPath $committedMd)
        $renderedLines = @(Get-Content -LiteralPath $rerenderOut)
        # Non-vacuity: comparing two empty files passes for the wrong reason.
        Check 'the re-render produced a full document' $true `
            ($renderedLines.Count -gt 500) "$($renderedLines.Count) lines"
        Check 'the committed document has the same number of lines' $true `
            ($committedLines.Count -eq $renderedLines.Count) `
            "committed $($committedLines.Count), re-rendered $($renderedLines.Count)"
        $differing = @()
        for ($i = 0; $i -lt [Math]::Min($committedLines.Count, $renderedLines.Count); $i++) {
            if ($committedLines[$i] -cne $renderedLines[$i]) { $differing += ($i + 1) }
        }
        Check 'exactly one line differs, and it is the render-time stamp' $true `
            ($differing.Count -eq 1 -and $committedLines[$differing[0] - 1] -match '^Generated .* from `') `
            $(if ($differing.Count -eq 1) { "line $($differing[0])" } else { "lines $($differing -join ', ')" })
    }
} finally {
    Remove-Item -LiteralPath $rerenderOut -Force -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# Get-GitRevision: a revision that is typed is not a revision that is known
# ---------------------------------------------------------------------------
#
# The acceptance record's corelib_source_revision was an operator-typed string
# written straight to disk with no git call and no dirty check, and it asserted
# a pristine tree that every sibling artifact recorded as `-untracked-only`.
# The derivation is now shared with run_hardware_suite.ps1 and driven here
# against real repositories in all three states.
Write-Output ''
Write-Output '=== Get-GitRevision ==='

$gitProbe = Join-Path ([System.IO.Path]::GetTempPath()) ("gitprobe-$PID-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $gitProbe | Out-Null
try {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & git -C $gitProbe init --quiet 2>$null | Out-Null
    & git -C $gitProbe config user.email 'probe@example.invalid' 2>$null | Out-Null
    & git -C $gitProbe config user.name 'probe' 2>$null | Out-Null
    Set-Content -LiteralPath (Join-Path $gitProbe 'a.txt') -Value 'one' -NoNewline
    & git -C $gitProbe add -A 2>$null | Out-Null
    & git -C $gitProbe commit -m 'probe' --quiet 2>$null | Out-Null
    $head = ([string](& git -C $gitProbe rev-parse HEAD 2>$null)).Trim()
    $ErrorActionPreference = $prevEap

    Check 'a clean tree gives the bare SHA with no suffix' $true `
        ((Get-GitRevision $gitProbe) -eq $head) "$head"

    Set-Content -LiteralPath (Join-Path $gitProbe 'untracked.txt') -Value 'x' -NoNewline
    Check 'untracked files only are labelled -untracked-only, not -dirty' $true `
        ((Get-GitRevision $gitProbe) -eq "$head-untracked-only") (Get-GitRevision $gitProbe)

    Set-Content -LiteralPath (Join-Path $gitProbe 'a.txt') -Value 'two' -NoNewline
    Check 'a tracked modification is labelled -dirty, and outranks untracked' $true `
        ((Get-GitRevision $gitProbe) -eq "$head-dirty") (Get-GitRevision $gitProbe)

    # Control: a directory that is not a repository must return empty, so the
    # caller can tell "not derivable" from "clean".
    $notARepo = Join-Path $gitProbe 'not-a-repo-parent'
    New-Item -ItemType Directory -Force -Path $notARepo | Out-Null
    Check 'a path that does not exist yields empty, not a throw' $true `
        ((Get-GitRevision (Join-Path $gitProbe 'no-such-directory')) -eq '')

    # One definition, or the cases above only cover one of the two callers.
    # run_hardware_suite.ps1 held a private copy while the acceptance harness
    # had none at all -- which is exactly how the two artifacts came to
    # disagree about the same corelib checkout.
    $privateCopies = @()
    foreach ($p in @(Get-ChildItem -LiteralPath $PSScriptRoot -File -Filter '*.ps1')) {
        if ($p.Name -eq 'acceptance_guards.ps1') { continue }
        $t2 = $null
        $e2 = $null
        $a2 = [System.Management.Automation.Language.Parser]::ParseFile($p.FullName, [ref]$t2, [ref]$e2)
        if (@($e2).Count -gt 0) { throw ("parse error in {0}: {1}" -f $p.Name, $e2[0].Message) }
        foreach ($fn in $a2.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
            if ($fn.Name -eq 'Get-GitRevision') { $privateCopies += ("{0}:{1}" -f $p.Name, $fn.Extent.StartLineNumber) }
        }
    }
    Check 'no script in the suite keeps a private Get-GitRevision' $true `
        ($privateCopies.Count -eq 0) $($privateCopies -join ', ')
    # ...and the suite that used to own it now takes the shared one.
    $suiteText = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'run_hardware_suite.ps1') -Raw
    Check 'run_hardware_suite.ps1 dot-sources the shared guards' $true `
        ($suiteText -match "\.\s+\(Join-Path \`$suiteDir 'acceptance_guards\.ps1'\)")
} finally {
    Remove-Item -LiteralPath $gitProbe -Recurse -Force -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# CALL SITES. Everything above tests guard BODIES; both shipped bugs were in
# how the harness used them.
#
# This is a lint, not a unit test, and it is deliberate: driving every harness
# statement offline would mean running the harness, which needs hardware. What
# CAN be done without hardware is read the statements. Two forms silently
# collapse a comma-wrapped return, and both have now shipped:
#
#   @(f ...)                 keeps the wrapper; .Count is 1 for any length
#   foreach ($x in f ...)    iterates the wrapper once, giving the whole array
#
# Measured on this box, PowerShell 5.1 and 7, for `function f { return , @(1,2,3,4) }`:
#   $x = f -> 4     @(f) -> 1     (f) -> 4     foreach in (f) -> 4     foreach in f -> 1
#
# Piping a bare call is the same mistake wearing different clothes: the
# pipeline sees one object.
#
# The set of comma-wrapped functions is DERIVED from the source, so a new guard
# that adopts the convention is policed the day it is written and nobody has to
# remember to add it to a list here.
# ---------------------------------------------------------------------------
Write-Output ''
Write-Output '=== Call sites: the comma-wrap trap, read out of the source ==='

$AstNs = 'System.Management.Automation.Language'
$suiteFiles = @(
    (Join-Path $PSScriptRoot 'acceptance_guards.ps1'),
    (Join-Path $PSScriptRoot 'run_real_model_acceptance.ps1'),
    (Join-Path $PSScriptRoot 'verify_acceptance_guards.ps1')
)

function Get-Ast {
    param([string]$Path, [string]$Text)
    $tokens = $null
    $errors = $null
    $ast = $(if ($Path) {
        [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    } else {
        [System.Management.Automation.Language.Parser]::ParseInput($Text, [ref]$tokens, [ref]$errors)
    })
    if (@($errors).Count -gt 0) {
        throw ("parse error in {0}: {1} (line {2})" -f $(if ($Path) { $Path } else { '<snippet>' }), $errors[0].Message, $errors[0].Extent.StartLineNumber)
    }
    return $ast
}

# A function is comma-wrapped if any of its `return` statements returns a
# unary-comma array literal -- `return , @(...)`. That is the convention that
# makes an empty result survive, and the same convention that makes `@(f)`
# lie.
function Get-CommaWrappedFunctionNames {
    param([string[]]$Paths)
    $names = @()
    foreach ($p in $Paths) {
        $ast = Get-Ast -Path $p
        $fns = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)
        foreach ($fn in $fns) {
            $rets = $fn.Body.FindAll({ param($n) $n -is [System.Management.Automation.Language.ReturnStatementAst] }, $true)
            foreach ($r in $rets) {
                if ($null -eq $r.Pipeline) { continue }
                if ($r.Pipeline -isnot [System.Management.Automation.Language.PipelineAst]) { continue }
                if ($r.Pipeline.PipelineElements.Count -ne 1) { continue }
                $el = $r.Pipeline.PipelineElements[0]
                if ($el -isnot [System.Management.Automation.Language.CommandExpressionAst]) { continue }
                if ($el.Expression -is [System.Management.Automation.Language.ArrayLiteralAst]) {
                    $names += $fn.Name
                    break
                }
            }
        }
    }
    $unique = @($names | Sort-Object -Unique)
    return , $unique
}

# Returns one row per offending call site, and one row per call site inspected,
# so the lint can prove it looked at something.
function Get-GuardCallSites {
    param($Ast, [string[]]$Names, [string]$Label)
    $rows = @()
    $calls = $Ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true)
    foreach ($cmd in $calls) {
        $name = $cmd.GetCommandName()
        if (-not $name -or ($Names -notcontains $name)) { continue }
        $why = ''
        $pipe = $cmd.Parent
        if ($pipe -is [System.Management.Automation.Language.PipelineAst]) {
            if ($pipe.PipelineElements.Count -gt 1 -and $pipe.PipelineElements[0] -eq $cmd) {
                $why = 'piped from a bare call: the pipeline sees the comma-wrapper as a single object'
            } else {
                $up = $pipe.Parent
                if ($up -is [System.Management.Automation.Language.StatementBlockAst] -and
                    $up.Parent -is [System.Management.Automation.Language.ArrayExpressionAst]) {
                    $why = '@(...) around the call keeps the comma-wrapper and reports 1'
                } elseif ($up -is [System.Management.Automation.Language.ForEachStatementAst] -and $up.Condition -eq $pipe) {
                    $why = 'foreach over a bare call iterates the comma-wrapper once'
                }
            }
        }
        $rows += [pscustomobject]@{
            file = $Label
            line = $cmd.Extent.StartLineNumber
            call = $name
            text = $cmd.Extent.Text
            problem = $why
        }
    }
    return , @($rows)
}

$commaWrapped = Get-CommaWrappedFunctionNames -Paths $suiteFiles
Write-Output "  comma-wrapped functions found in the suite: $($commaWrapped -join ', ')"

# Non-vacuity, part 1: if a rename or a refactor empties this set, every call
# site check below becomes trivially true. Fail instead.
Check 'the lint derived a non-empty set of comma-wrapped functions' $true ($commaWrapped.Count -ge 3)
Check 'Test-HistoryContainment is in that set' $true ($commaWrapped -contains 'Test-HistoryContainment')

$allSites = @()
foreach ($p in $suiteFiles) {
    $ast = Get-Ast -Path $p
    $sites = Get-GuardCallSites -Ast $ast -Names $commaWrapped -Label (Split-Path -Leaf $p)
    $allSites += $sites
}
$bad = @($allSites | Where-Object { $_.problem })
Write-Output "  call sites inspected: $($allSites.Count) across $($suiteFiles.Count) files; offending: $($bad.Count)"
foreach ($b in $bad) { Write-Output ("    {0}:{1}  {2}" -f $b.file, $b.line, $b.text); Write-Output ("        {0}" -f $b.problem) }

# Non-vacuity, part 2: the harness must actually contain call sites. If it does
# not, the scan below is passing on an empty set.
$harnessSites = @($allSites | Where-Object { $_.file -eq 'run_real_model_acceptance.ps1' })
Check 'the harness contains guard call sites to inspect' $true ($harnessSites.Count -ge 5) "$($harnessSites.Count) found"
Check 'no call site collapses a comma-wrapped return' $true ($bad.Count -eq 0) $(if ($bad.Count) { "$($bad[0].file):$($bad[0].line)" } else { 'harness, guards and verifier' })

# The guards above prove the FUNCTIONS work. This proves the harness still
# uses them -- which is the half that was missing when the verdict failed open
# and when the document shipped with a BOM. Deleting either call restores the
# original defect and leaves every functional check above green, so the call
# has to be asserted by name.
$harnessAst = Get-Ast -Path (Join-Path $PSScriptRoot 'run_real_model_acceptance.ps1')
$harnessCalls = @($harnessAst.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true) |
    ForEach-Object { $_.GetCommandName() } | Where-Object { $_ })
foreach ($required in @('Get-UnrecognisedStepStatus', 'Write-Utf8NoBom', 'Test-Utf8Bom', 'Get-GitRevision')) {
    Check "the harness calls $required" $true ($harnessCalls -contains $required)
}
# ...and that the rollup's rank table is the shared one, not a fresh literal
# beside the guard. A local `$rank = @{...}` would drift from the table
# Get-UnrecognisedStepStatus checks against, and the two disagreeing is the
# whole failure mode.
Check 'the harness takes its rank table from Get-StepStatusRank' $true `
    ($harnessCalls -contains 'Get-StepStatusRank')

# Non-vacuity, part 3: the positive control. A lint that finds nothing is
# indistinguishable from a lint that passes, so run the detector over source
# that is known to be wrong -- including the exact statement that shipped -- and
# require it to find every one, and to leave the correct forms alone.
$controlSource = @'
$containment = @(Test-HistoryContainment $perTurn)
foreach ($s in Get-StepsWithTurns $record) { $s }
$flat = Get-FieldArray $o "f" | Where-Object { $_ }
$okAssigned = Test-HistoryContainment $perTurn
$okCounted = @($okAssigned | Where-Object { $_ })
foreach ($s in (Get-StepsWithTurns $record)) { $s }
$okParen = (Get-FieldArray $o "f").Count
'@
$controlAst = Get-Ast -Text $controlSource
$controlSites = Get-GuardCallSites -Ast $controlAst -Names $commaWrapped -Label '<control>'
$controlBad = @($controlSites | Where-Object { $_.problem })
Write-Output "  positive control: $($controlSites.Count) call sites, $($controlBad.Count) flagged (3 wrong forms, 3 right ones)"
Check 'the lint flags the exact statement that shipped' $true `
    (@($controlBad | Where-Object { $_.text -like '*Test-HistoryContainment*' }).Count -eq 1)
Check 'the lint flags all three broken forms and no correct one' $true ($controlBad.Count -eq 3) `
    $(if ($controlBad.Count -ne 3) { ($controlBad | ForEach-Object { $_.text }) -join ' | ' } else { '@(f), foreach in f, f | ...' })

Write-Output ''
$total = $script:coverageCount + $script:noteCount
if ($failures.Count -eq 0) {
    Write-Output "ALL OFFLINE GUARD CHECKS PASSED: $total assertions -- $($script:coverageCount) coverage, $($script:noteCount) documentation witnesses."
    Write-Output 'Coverage here means guard bodies against the committed record, plus a source-level lint of their call sites.'
    Write-Output 'It does NOT mean the harness has been run. No version after 81a01f7b has run on hardware.'
    exit 0
}
Write-Output "OFFLINE GUARD CHECKS FAILED: $($failures -join '; ')"
exit 1
