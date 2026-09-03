# Guard logic shared by the acceptance harness and its offline verifier.
#
# WHY THIS FILE EXISTS
#
# `verify_acceptance_guards.ps1` used to RE-IMPLEMENT the checks it claimed to
# verify. That is not a test: a copy of the implementation agrees with the
# implementation by construction, and nothing fails when the two drift apart.
# It cost exactly what you would expect. A guard shipped as "verified offline"
# used `@(Get-Field ...).Count -gt 0` to test whether a step had per-turn
# evidence -- and in PowerShell `@($null).Count` is 1, not 0, so it selected
# every step and would have thrown on the first one without turns. A healthy
# acceptance run would have ended in `DOCUMENT NOT RENDERED` and `exit 1` after
# ninety minutes of hardware time. The verifier passed, because the shape the
# bug lived in was the one thing it hand-built rather than derived.
#
# So the logic lives here, once. The harness dot-sources it and so does the
# verifier, and the verifier drives it with the real committed record. There is
# no second copy to keep in step.
#
# This file must stay side-effect free: dot-sourcing it defines functions and
# does nothing else.

# Read one field from either shape a record node can take: an ordered
# dictionary built in-session, or a PSCustomObject loaded from the JSON.
#
# `$Object.PSObject.Properties[$Name]` -- an indexer lookup -- NOT
# `$Object.PSObject.Properties.Name -contains $Name`, which is member
# enumeration and THROWS on a property-less object under
# Set-StrictMode -Version Latest: "The property 'Name' cannot be found on this
# object."
#
# That is not hypothetical. A step recorded with no evidence gets the default
# `@{}`, which serialises as `{}` and reloads as a PSCustomObject with zero
# properties -- and the harness reloads the record from disk before rendering.
# Any run that leaves a step `not_exercised` through Add-UnselectedStep
# therefore produced `{}`, and the renderer died in the one helper written to
# make it degrade instead. Reproduced offline: a seeded partial run rendered a
# complete 80 KB document and then reported DOCUMENT NOT RENDERED and exit 1.
# The committed `-Steps all` run escaped it only because every step supplied
# evidence. The renderer already documents this exact trap for a different
# expression a hundred lines away; the shared helper had it too.
function Get-Field {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    if ($Object -is [System.Collections.IDictionary]) {
        if ($Object.Contains($Name)) { return $Object[$Name] }
        return $null
    }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -ne $prop) { return $prop.Value }
    return $null
}

# How many of this field are there? Returns an [int], never a collection.
#
# COUNTING IS A SEPARATE FUNCTION ON PURPOSE. Three different PowerShell
# behaviours conspire here and any two of them look consistent:
#
#   @($null).Count  is 1, so `@(...)` cannot be used to test presence -- this
#                   is the bug that shipped: the render guard selected all 20
#                   steps and threw on the first one without turns;
#   return @()      UNROLLS to nothing, so the caller gets $null and `.Count`
#                   throws under Set-StrictMode;
#   return , @(...) survives assignment ($x = f; @($x).Count is right) but NOT
#                   `@(f ...)`, which keeps the wrapper and reports 1.
#
# So the answer is not a better array convention -- it is to stop passing
# arrays across the boundary when the question is "how many". An [int] is a
# scalar; nothing unrolls it, nothing wraps it, and it reads the same at every
# call site. Every presence test goes through this.
function Get-FieldCount {
    param($Object, [string]$Name)
    $value = Get-Field $Object $Name
    if ($null -eq $value) { return [int]0 }
    return [int](@($value).Count)
}

# Coerce a field to a real array, treating "absent" as empty.
#
# Use `$x = Get-FieldArray ...` and then `$x`. Do NOT write
# `@(Get-FieldArray ...)`: the comma-wrap below is what stops an empty result
# unrolling to $null, and that same wrap is preserved by `@(...)` around a
# call, which would report 1 for a 4-element result. When you want a count,
# use Get-FieldCount instead and avoid the question.
function Get-FieldArray {
    param($Object, [string]$Name)
    $value = Get-Field $Object $Name
    if ($null -eq $value) { return , @() }
    return , @($value)
}

# Whitespace deleted, not normalised.
#
# /history is read back out of a console screen buffer, so a long line is
# wrapped at the buffer width -- and a wrap splits a word with NO separator.
# Collapsing runs of whitespace would leave "remem ber" and fail a history that
# is perfectly correct. Deleting it entirely makes the comparison immune to
# however the console chose to lay the text out.
function Remove-AllWhitespace {
    param([string]$Text)
    if ($null -eq $Text) { return '' }
    return ($Text -replace '\s', '')
}

# Does each turn's /history still contain the turn before it?
#
# The question C21 actually asks is whether the COMPLETE rendered history is
# carried forward, so both halves of the previous exchange are required: its
# user message and the assistant reply to it. Checking the prompt alone would
# be satisfied by a frontend that appended user turns and dropped every reply.
#
# Returns one row per comparison; the caller decides what to do with them.
function Test-HistoryContainment {
    param($Turns)
    $rows = @()
    $turnList = @($Turns)
    for ($k = 1; $k -lt $turnList.Count; $k++) {
        $prev = $turnList[$k - 1]
        $thisHistory = Remove-AllWhitespace ([string](Get-Field $turnList[$k] 'history_verbatim'))

        $prevPrompt = [string](Get-Field $prev 'prompt')
        $promptNeedle = Remove-AllWhitespace $prevPrompt
        $hasPrompt = ($promptNeedle.Length -gt 0 -and $thisHistory.Contains($promptNeedle))

        # The reply is taken from the previous turn's OWN /history, which is
        # where the frontend rendered it, rather than from the screen text of
        # the reply -- the screen carries the banner lines and the echoed
        # prompt as well, and would never match verbatim.
        #
        # THE MARKERS ARE PHI-4'S CHAT TEMPLATE, and that is a coupling, not a
        # constant. A /history format change, or the same probe pointed at
        # another model, leaves no marker to find. "No marker" must therefore
        # mean NOT CHECKABLE, not "the reply was dropped": scoring it as
        # dropped would fail a perfectly healthy conversation for a rendering
        # reason, which is the same mistake the whitespace handling above
        # exists to avoid. `previous_reply_checkable` says which happened, and
        # the caller is expected to report an uncheckable reply rather than
        # silently degrade to a prompt-only gate.
        $prevHistory = Remove-AllWhitespace ([string](Get-Field $prev 'history_verbatim'))
        $replyNeedle = ''
        $marker = '<|assistant|>'
        $at = $prevHistory.IndexOf($marker)
        if ($at -ge 0) {
            $tail = $prevHistory.Substring($at + $marker.Length)
            $end = $tail.IndexOf('<|end|>')
            if ($end -ge 0) { $tail = $tail.Substring(0, $end) }
            # A long reply is truncated by the screen buffer; a prefix is
            # enough to tell "carried forward" from "dropped", and asking for
            # the whole thing would fail on the rendering rather than on the
            # history.
            if ($tail.Length -gt 40) { $tail = $tail.Substring(0, 40) }
            $replyNeedle = $tail
        }
        $replyCheckable = ($replyNeedle.Length -gt 0)
        $hasReply = ($replyCheckable -and $thisHistory.Contains($replyNeedle))

        $rows += [ordered]@{
            turn = (Get-Field $turnList[$k] 'index')
            previous_turn_prompt = $prevPrompt
            history_contains_previous_prompt = $hasPrompt
            history_contains_previous_reply = $hasReply
            previous_reply_checkable = $replyCheckable
            # Both halves when both can be read, because "complete rendered
            # history" is the claim. When the reply cannot be read at all the
            # prompt alone decides -- and the row says so, so the caller can
            # report the weaker check instead of publishing it as the strong
            # one.
            history_contains_previous_turn = $(if ($replyCheckable) { $hasPrompt -and $hasReply } else { $hasPrompt })
            previous_reply_probe = $replyNeedle
        }
    }
    return , @($rows)
}

# Which steps carry per-turn evidence the document is expected to render?
#
# Derived from the record, never handed in. The bug this file exists because of
# lived precisely in this derivation, and a verifier that receives the answer
# cannot check it.
function Get-StepsWithTurns {
    param($Record)
    $out = @()
    foreach ($step in @(Get-Field $Record 'steps')) {
        $count = Get-FieldCount (Get-Field $step 'evidence') 'per_turn'
        if ($count -gt 0) {
            $out += [pscustomobject]@{ id = [string](Get-Field $step 'id'); count = $count }
        }
    }
    return , @($out)
}

# Is the rendered document actually populated?
#
# The run stamp proves the document is FRESH; it says nothing about whether the
# verbatim blocks contain anything. Since the renderer reads evidence through
# Get-Field, which yields $null rather than throwing, a row of unexpected shape
# renders blank headings and would otherwise pass. The verbatim completions are
# what a human judges this acceptance on, so a blank one is not cosmetic.
#
# Returns $null when the document is good, or a message describing the first
# problem found.
function Test-RenderedDocument {
    param($Record, $RenderedTurnPrompts, [string]$Written)
    $rendered = @($RenderedTurnPrompts)

    $blank = @($rendered | Where-Object {
        $p = [string](Get-Field $_ 'prompt')
        (-not $p) -or ($p.Trim().Length -eq 0) })
    if ($blank.Count -gt 0) {
        $steps = @($blank | ForEach-Object { [string](Get-Field $_ 'step') } | Sort-Object -Unique)
        return ("the document rendered $($blank.Count) turn(s) with an empty prompt for step(s) " +
                ($steps -join ', ') + ': the verbatim evidence did not reach it')
    }

    foreach ($rt in $rendered) {
        $p = [string](Get-Field $rt 'prompt')
        if ($Written -notmatch [regex]::Escape($p)) {
            return "step $([string](Get-Field $rt 'step')): a turn prompt was rendered but is not present in the written document"
        }
    }

    foreach ($swt in (Get-StepsWithTurns $Record)) {
        $got = @($rendered | Where-Object { [string](Get-Field $_ 'step') -eq $swt.id }).Count
        if ($got -ne $swt.count) {
            return "step $($swt.id): $($swt.count) turn(s) of evidence but $got rendered into the document"
        }
    }
    return $null
}

# Does a reply read like English a person wrote, rather than fluent-looking
# garbage?
#
# Deliberately weak, and says so. The first thing a wrong weight map or a stale
# KV row produces is text that passes every mechanical check, which is why the
# brief requires the verbatim completion in the document for a HUMAN to judge.
# What this can do is catch the failures that are not subtle -- empty output,
# no letters, one phrase repeated forever, and language that stops being
# language partway through.
#
# It lives here rather than in the harness because the recorded run contains
# real examples of two of those failures, so the verifier can drive this
# function against them offline instead of restating what it believes the rule
# to be.
#
# WHY THE SEGMENT CHECK EXISTS. The whole-text letter ratio below could not see
# the most alarming thing in the recorded run. One turn produced hundreds of
# clean tokens and then, about 85% of the way in, broke mid-word and emitted
# high-entropy punctuation and digits to the cap. Averaged over the whole
# reply the letter ratio stayed well above the floor, so the only reason
# recorded for that turn was "dominated by a repeated token" -- true of its
# earlier repetition loop, and wrong about the collapse. A model losing the
# thread repeats, drifts or confabulates; it does not emit uniform random
# punctuation. Those are different failures and the record has to say so.
function Test-LooksLikeEnglish {
    param([string]$Text, [string]$MustContain = '')
    $t = ($Text -replace '\[FLM\][^\n]*', '') -replace '\s+', ' '
    $t = $t.Trim()
    $reasons = @()
    if ($t.Length -lt 8) { $reasons += 'reply is shorter than 8 characters' }
    $words = @($t -split '\s+' | Where-Object { $_ -match '[A-Za-z]' })
    if ($words.Count -lt 3) { $reasons += 'fewer than three alphabetic words' }
    if ($words.Count -ge 6) {
        $distinct = @($words | ForEach-Object { $_.ToLowerInvariant() } | Sort-Object -Unique)
        if ($distinct.Count -lt [math]::Ceiling($words.Count / 4)) {
            $reasons += 'the reply is dominated by a repeated token or phrase'
        }
    }
    $letters = ([regex]::Matches($t, '[A-Za-z]')).Count
    if ($t.Length -gt 0 -and ($letters / [double]$t.Length) -lt 0.5) {
        $reasons += 'fewer than half the characters are letters'
    }

    # Segment scan: a stretch that stops being language, after a stretch that
    # was. Reported with its offset, because that offset is the only precise
    # entry point a later investigation has.
    $window = 200
    if ($t.Length -ge ($window * 4)) {
        $ratios = @()
        for ($start = 0; $start + $window -le $t.Length; $start += $window) {
            $slice = $t.Substring($start, $window)
            $ratios += [pscustomobject]@{
                Start = $start
                Ratio = ([regex]::Matches($slice, '[A-Za-z]')).Count / [double]$window
            }
        }
        $healthyBefore = $false
        foreach ($r in $ratios) {
            if ($r.Ratio -ge 0.7) { $healthyBefore = $true; continue }
            if ($healthyBefore -and $r.Ratio -lt 0.35) {
                $reasons += (
                    'the reply stops producing language at character {0} of {1}: ' -f $r.Start, $t.Length) +
                    ('that stretch is {0:N0}% letters, after an earlier stretch that was language' -f ($r.Ratio * 100))
                break
            }
        }
    }

    if ($MustContain -and ($t -notmatch [regex]::Escape($MustContain))) {
        $reasons += "the reply does not contain the expected substring '$MustContain'"
    }
    return @{ Ok = ($reasons.Count -eq 0); Reasons = $reasons; Normalized = $t }
}

# ---------------------------------------------------------------------------
# Publishability: a UTF-8 BOM stops a Jekyll page being a page
# ---------------------------------------------------------------------------
#
# The acceptance record -- this branch's headline evidence -- shipped with a
# UTF-8 BOM and was therefore not published at all. Jekyll's
# `Utils.has_yaml_header?` matches `\A---\s*\r?\n` against the RAW first line
# and does not strip a BOM, so a file whose first byte is not `-` has no front
# matter: no `layout: docs`, no site chrome, no nav entry, no conversion, and
# no entry in `docs/search.json` (which selects `p.layout == 'docs'`). GitHub
# Pages copies it through as a static file and the pretty URL does not exist.
#
# The cause was `Set-Content -Encoding utf8`, which in Windows PowerShell 5.1
# means "UTF-8 WITH BOM". The same harness already wrote its JSON correctly
# with a UTF8Encoding($false); the markdown write did not, and the render guard
# that reads the file back for its run stamp cannot see a BOM -- it proves the
# document is fresh and populated and says nothing about whether it is a page.
#
# So the write goes through one function, and the check is a byte test the
# verifier runs over EVERY `.md` under docs/, not just the one that was wrong.

# Write UTF-8 with no byte-order mark, and no trailing-newline surprises.
# `Set-Content` must not be used for anything that Jekyll parses.
function Write-Utf8NoBom {
    param([Parameter(Mandatory = $true)][string]$Path,
          [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text)
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

# True when the file's first three bytes are EF BB BF. Reads bytes, not text:
# every text reader in PowerShell strips the BOM, which is exactly why this
# defect survived a read-back check.
function Test-Utf8Bom {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $head = New-Object byte[] 3
        $read = $stream.Read($head, 0, 3)
    } finally { $stream.Dispose() }
    return ($read -eq 3 -and $head[0] -eq 0xEF -and $head[1] -eq 0xBB -and $head[2] -eq 0xBF)
}

# Every `.md` under $Root whose first bytes are a BOM. Comma-wrapped so an
# empty result survives assignment; assign it, never `@(Get-BomMarkdownFile ...)`
# -- the lint in verify_acceptance_guards.ps1 enforces that.
function Get-BomMarkdownFile {
    param([Parameter(Mandatory = $true)][string]$Root)
    $hits = @()
    foreach ($f in @(Get-ChildItem -LiteralPath $Root -Recurse -File -Filter '*.md' | Sort-Object FullName)) {
        if (Test-Utf8Bom $f.FullName) { $hits += $f.FullName }
    }
    return , @($hits)
}

# ---------------------------------------------------------------------------
# Derived revisions
# ---------------------------------------------------------------------------
#
# Lifted verbatim from run_hardware_suite.ps1, which derived its revisions
# while the acceptance harness took -CorelibSourceRevision as a plain optional
# string and wrote it straight to the record. The two artifacts then disagreed
# about the same checkout: every baseline says `e5258d29...-untracked-only`
# and the acceptance record says the bare SHA, i.e. a pristine tree. The one
# identifier answering "which corelib source built the DLL these numbers came
# from" must not be free text.
#
# Both scripts now use this one definition.
function Get-GitRevision {
    param([Parameter(Mandatory = $true)][string]$Directory)
    if (-not $Directory -or -not (Test-Path $Directory)) { return '' }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $revision = (& git -C $Directory rev-parse HEAD 2>$null)
        if ($LASTEXITCODE -ne 0 -or -not $revision) { return '' }
        $revision = ([string]$revision).Trim()
        # A tree that is not pristine is recorded as such. A SHA on its own
        # would claim the baseline describes a committed revision when it does
        # not, and that is a claim nobody can check later.
        #
        # Tracked modifications and untracked files are labelled DIFFERENTLY,
        # because they mean different things and a single `-dirty` makes the
        # harmless case look like the serious one. The corelib checkout on the
        # AIE4 target carries four untracked build output directories and no
        # tracked change at all; calling that "dirty" would tell a later reader
        # the source had been edited.
        $status = @(
            (& git -C $Directory status --porcelain 2>$null) |
                Where-Object { $_ })
        if ($LASTEXITCODE -eq 0 -and $status.Count -gt 0) {
            $tracked = @(
                $status | Where-Object { -not $_.StartsWith('??') })
            if ($tracked.Count -gt 0) {
                $revision = "$revision-dirty"
            } else {
                $revision = "$revision-untracked-only"
            }
        }
        return $revision
    } finally {
        $ErrorActionPreference = $prev
    }
}

# ---------------------------------------------------------------------------
# The rollup must fail CLOSED on a status nobody recognises
# ---------------------------------------------------------------------------
#
# The rank table and the check that guards it live here, once, because the
# verifier has to be able to drive the real thing. A copy in the verifier would
# agree with the harness by construction -- which is how the last two guard
# bugs on this branch shipped.
#
# The defect: `$rank['<anything else>']` is $null, `$null -gt 0` is $false, so
# an unrecognised step status rolled up to `met`, and the step counters that
# feed the exit ladder match only the three exact literals, so it reached none
# of them either. The run printed `RESULT: ACCEPTED` and exited 0. Reproduced
# in Windows PowerShell 5.1 and pwsh 7.
#
# `run_hardware_suite.ps1` fixed exactly this shape and called it "the tenth
# instance of this project's recurring pattern"; the fix never reached the
# harness that writes the record.

# The one rank table. Returned fresh each call so no caller can mutate the
# shared copy out from under another.
function Get-StepStatusRank {
    return @{ 'met' = 0; 'partial' = 1; 'not_exercised' = 2; 'not_met' = 3 }
}

# One row per step whose status is not in the rank table. Comma-wrapped so an
# empty result survives assignment -- assign it, never `@(Get-UnrecognisedStepStatus ...)`;
# the AST lint in verify_acceptance_guards.ps1 enforces that and reads this
# function's source to learn the convention.
#
# Note that PowerShell hashtable lookup and `-eq` are both case-insensitive, so
# 'MET' is a RECOGNISED status and correctly ranks as `met`. The check is about
# statuses that are not in the table at all: '', 'failed', 'bogus', and
# anything an older schema or a truncated record carries.
function Get-UnrecognisedStepStatus {
    param($Steps)
    $rank = Get-StepStatusRank
    $rows = @()
    foreach ($s in @($Steps)) {
        $status = [string](Get-Field $s 'status')
        if (-not $rank.ContainsKey($status)) {
            $rows += ("step {0}: '{1}'" -f (Get-Field $s 'id'), $status)
        }
    }
    return , @($rows)
}
