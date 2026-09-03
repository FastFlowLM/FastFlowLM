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
        $hasReply = ($replyNeedle.Length -gt 0 -and $thisHistory.Contains($replyNeedle))

        $rows += [ordered]@{
            turn = (Get-Field $turnList[$k] 'index')
            previous_turn_prompt = $prevPrompt
            history_contains_previous_prompt = $hasPrompt
            history_contains_previous_reply = $hasReply
            # Both halves, because "complete rendered history" is the claim.
            history_contains_previous_turn = ($hasPrompt -and $hasReply)
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
