# Task 16: real-model acceptance for Phi-4 AIE4 on xcomedusad-43.
#
# Tasks 12 and 13 proved the ENGINE against explicit token IDs and a reference
# driver. This script proves the SHIPPED PRODUCT against the real published
# model, on the real target, the way a user runs it. Nothing here is allowed to
# stand in for something it did not run.
#
# THREE RULES THIS FILE EXISTS TO ENFORCE
#
# 1. Skipped work reaches the exit code. A step that could not run is
#    `not_exercised`, never `met`, and `not_exercised` alone makes the script
#    exit 77 (CTest's SKIP_RETURN_CODE) even when nothing failed. Eleven times
#    in this effort a check reported success for work it did not do; an
#    acceptance record that cannot say "not tested" is not an acceptance record.
#
# 2. Every claim cites its artifact. Each step records `evidence`: file paths,
#    exit codes and verbatim output that a reader can go and check. A step with
#    a verdict and no evidence is a defect in this script.
#
# 3. The Section 17 roll-up is derived, never asserted. Step 13 maps each
#    acceptance criterion to the step that exercised it and inherits that step's
#    status. A criterion with no step mapped to it renders `not_exercised`. It
#    is not possible to hand-write `met` next to a criterion here.
#
# ORDERING IS LOAD-BEARING. Step 12b deliberately kills a process to prove the
# terminal-failure path carries corelib's own diagnostic, so it runs last. Two
# processes holding AIE4 device contexts at once fail in ways that look like
# defects, so nothing here runs in parallel.

[CmdletBinding()]
param(
    # The product binary under test, built on THIS box from a committed
    # revision. Never a binary copied from the development box.
    [Parameter(Mandatory = $true)][string]$FlmExe,

    # The FastFlow commit the binary was built from. Recorded in the JSON and
    # asserted against the checkout, because "which code produced this number"
    # is the first question anyone will ask of these results.
    [Parameter(Mandatory = $true)][string]$FastFlowRevision,

    # The assembled model directory flm.exe loads: upstream files plus the four
    # FastFlow overlays (design 8.1, PACKAGE-1 / MODEL-2).
    [Parameter(Mandatory = $true)][string]$ModelDir,

    [string]$ModelTag = 'phi4-mini-it-aie4:4b',

    # Committed overlay source, catalog and model info.
    [string]$OverlayDir,
    [string]$ModelList,
    [string]$ModelInfo,

    # Staged AIE4 runtime closure (CLOSURE-1), and the corelib checkout the DLL
    # was built from. get_version reports a hard-coded 0.1.0 across all of 0.x
    # and cannot identify a revision, so the revision has to be supplied.
    [string]$CorelibRuntimeDir,
    [string]$CorelibSource,
    [string]$CorelibSourceRevision,

    [string]$RepoRoot,
    [string]$BuildDir,
    [string]$Cmake,
    [string]$Python = 'python',

    [int]$Port = 11434,

    # Design 15.3 / Task 13 measure 128 tokens on explicit IDs. Step 11 is the
    # product path with a real tokenizer and sampler in the loop, and the brief
    # asks for at least 512.
    [int]$SustainedTokens = 512,

    # 'all', or a comma/space list like '1,2,3' or '7 8 9'. Steps are recorded
    # as not_exercised when they are not selected, so a partial run still
    # produces an honest record -- it just cannot exit 0.
    [string[]]$Steps = @('all'),

    # Merge into an existing record rather than starting empty, so a run split
    # across sessions (SSH here is not reliable enough to assume otherwise)
    # accumulates instead of overwriting.
    [switch]$Append,

    [Parameter(Mandatory = $true)][string]$OutJson,

    # When set, Step 13 also renders the human-readable acceptance document.
    [string]$Markdown
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$script:suiteDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RepoRoot) {
    $script:sourceDir = (Resolve-Path (Join-Path (Join-Path $suiteDir '..') '..')).Path
    $RepoRoot = (Resolve-Path (Join-Path $script:sourceDir '..')).Path
} else {
    $script:sourceDir = (Join-Path $RepoRoot 'src')
}
if (-not $OverlayDir) { $OverlayDir = Join-Path $script:sourceDir 'model_overlays/phi4-mini-it-aie4' }
if (-not $ModelList) { $ModelList = Join-Path $script:sourceDir 'model_list.json' }
if (-not $ModelInfo) { $ModelInfo = Join-Path $script:sourceDir 'model_info.json' }
if (-not $BuildDir) { $BuildDir = Join-Path $script:sourceDir 'build/phi4-hardware' }

$script:runStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$script:artifactDir = Join-Path $BuildDir "acceptance/$script:runStamp-$PID"
New-Item -ItemType Directory -Force -Path $script:artifactDir | Out-Null

# ---------------------------------------------------------------------------
# Record keeping
# ---------------------------------------------------------------------------

# Computed once, up front: every step result is stamped with it, and the
# carry-forward rule below refuses to reuse a result recorded against a
# different binary.
$script:identityExeSha = $(if (Test-Path $FlmExe) { (Get-FileHash -Path $FlmExe -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null })

# The harness hashes itself and its console driver.
#
# `checkout_revision` alone is not enough to say what produced a result. This
# script is edited and copied to the target dozens of times while it is being
# brought up, so a record can truthfully name a commit while having been
# produced by a script that differs from that commit's content. Recording the
# script's own SHA-256 makes the claim checkable: `git show <rev>:<path>`
# hashed against this field either matches or it does not.
#
# It also gates the carry-forward rule below, so a result produced by an
# earlier version of this harness is never carried into a run of a later one.
$script:harnessSha = (Get-FileHash -Path $MyInvocation.MyCommand.Path -Algorithm SHA256).Hash.ToLowerInvariant()
$script:driverPath = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'drive_flm_console.ps1'
$script:driverSha = $(if (Test-Path $script:driverPath) { (Get-FileHash -Path $script:driverPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null })

$script:record = [ordered]@{}
if ($Append -and (Test-Path $OutJson)) {
    $existing = Get-Content $OutJson -Raw | ConvertFrom-Json
    foreach ($p in $existing.PSObject.Properties) { $script:record[$p.Name] = $p.Value }
}

function Set-Field { param([string]$Name, $Value) $script:record[$Name] = $Value }

function Trace-Progress {
    param([string]$Text)
    try {
        Add-Content -Path (Join-Path $script:artifactDir 'progress.log') `
            -Value ("[{0}] {1}" -f (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $Text) -Encoding utf8
    } catch { }
}

# Reduce a value to strings, numbers, booleans, arrays and dictionaries.
#
# ConvertTo-Json in Windows PowerShell 5.1 walks whatever object graph it is
# handed, and a live .NET object reached by accident -- a FileInfo, a Process,
# a JSON node still attached to its parent document -- drags in a graph that is
# not obviously large from the call site. Measured on this box: a record that
# serialised in 9 ms grew one evidence field and then span for four minutes
# past two gigabytes of working set without producing output.
#
# The record is the deliverable, so it cannot be at the mercy of that. Every
# value is copied into a plain shape first, with an explicit depth limit, and
# anything past the limit becomes a string rather than a recursion. The
# conversion is lossy by design and says so: a truncated value is visible in
# the artifact as text, which is a far better failure than a run that never
# writes its result.
function ConvertTo-PlainData {
    param($Value, [int]$Depth = 0)
    if ($null -eq $Value) { return $null }
    if ($Depth -ge 12) { return ('<depth limit: ' + $Value.GetType().Name + '>') }
    if ($Value -is [string]) { return $Value }
    if ($Value -is [bool] -or $Value -is [int] -or $Value -is [long] -or
        $Value -is [double] -or $Value -is [decimal] -or $Value -is [int64] -or
        $Value -is [uint32] -or $Value -is [uint64]) { return $Value }
    if ($Value -is [System.Collections.IDictionary]) {
        $out = [ordered]@{}
        foreach ($k in @($Value.Keys)) { $out["$k"] = ConvertTo-PlainData -Value $Value[$k] -Depth ($Depth + 1) }
        return $out
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $out = [ordered]@{}
        foreach ($p in $Value.PSObject.Properties) { $out[$p.Name] = ConvertTo-PlainData -Value $p.Value -Depth ($Depth + 1) }
        return $out
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $out = @()
        foreach ($item in $Value) { $out += , (ConvertTo-PlainData -Value $item -Depth ($Depth + 1)) }
        return , $out
    }
    return [string]$Value
}

# Write the JSON directly, rather than through ConvertTo-Json.
#
# Windows PowerShell 5.1's ConvertTo-Json was measured on this box to hang --
# not fail, hang, at 100% CPU past two gigabytes of working set -- on a record
# that had serialised in nine milliseconds one step earlier, after a handful of
# scalar fields were added. Normalising the graph to plain dictionaries,
# arrays, strings and numbers first did not help, which rules out the object
# graph and leaves the cmdlet.
#
# The record is the deliverable of this whole task. It is not acceptable for it
# to depend on a cmdlet that can silently stop producing output, so the writer
# below is explicit: recursive descent over the normalised data, JSON string
# escaping, and the depth limit the normaliser has already applied. It is
# short, and it always terminates.
function ConvertTo-JsonString {
    param([string]$Text)
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('"')
    foreach ($ch in $Text.ToCharArray()) {
        $code = [int]$ch
        if ($code -eq 34) { [void]$sb.Append('\"') }
        elseif ($code -eq 92) { [void]$sb.Append('\\') }
        elseif ($code -eq 8) { [void]$sb.Append('\b') }
        elseif ($code -eq 12) { [void]$sb.Append('\f') }
        elseif ($code -eq 10) { [void]$sb.Append('\n') }
        elseif ($code -eq 13) { [void]$sb.Append('\r') }
        elseif ($code -eq 9) { [void]$sb.Append('\t') }
        elseif ($code -lt 32 -or $code -gt 126) { [void]$sb.Append('\u' + $code.ToString('x4')) }
        else { [void]$sb.Append($ch) }
    }
    [void]$sb.Append('"')
    return $sb.ToString()
}

function ConvertTo-JsonText {
    param($Value, [int]$Indent = 0)
    $pad = ' ' * ($Indent * 2)
    $padIn = ' ' * (($Indent + 1) * 2)
    $nl = [string][char]10
    if ($null -eq $Value) { return 'null' }
    if ($Value -is [bool]) { return $(if ($Value) { 'true' } else { 'false' }) }
    if ($Value -is [int] -or $Value -is [long] -or $Value -is [int64] -or
        $Value -is [uint32] -or $Value -is [uint64]) {
        return $Value.ToString([Globalization.CultureInfo]::InvariantCulture)
    }
    if ($Value -is [double] -or $Value -is [decimal] -or $Value -is [single]) {
        $d = [double]$Value
        if ([double]::IsNaN($d) -or [double]::IsInfinity($d)) { return (ConvertTo-JsonString $d.ToString()) }
        return $d.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $keys = @($Value.Keys)
        if ($keys.Count -eq 0) { return '{}' }
        $parts = @()
        foreach ($k in $keys) {
            $parts += ($padIn + (ConvertTo-JsonString ([string]$k)) + ': ' +
                       (ConvertTo-JsonText -Value $Value[$k] -Indent ($Indent + 1)))
        }
        return ('{' + $nl + ($parts -join (',' + $nl)) + $nl + $pad + '}')
    }
    if ($Value -isnot [string] -and $Value -is [System.Collections.IEnumerable]) {
        $items = @($Value)
        if ($items.Count -eq 0) { return '[]' }
        $parts = @()
        foreach ($item in $items) {
            $parts += ($padIn + (ConvertTo-JsonText -Value $item -Indent ($Indent + 1)))
        }
        return ('[' + $nl + ($parts -join (',' + $nl)) + $nl + $pad + ']')
    }
    return (ConvertTo-JsonString ([string]$Value))
}

function Save-Record {
    $dir = Split-Path -Parent $OutJson
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $plain = ConvertTo-PlainData -Value $script:record
    $json = ConvertTo-JsonText -Value $plain
    [System.IO.File]::WriteAllText($OutJson, $json, (New-Object System.Text.UTF8Encoding($false)))
}

# The only way a step result enters the record.
#
# `Status` is constrained to three values on purpose. A fourth ("partial",
# "mostly") is how "not exercised" becomes "met" one report at a time.
function Add-StepResult {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)]
        [ValidateSet('met', 'not_met', 'not_exercised')]
        [string]$Status,
        [Parameter(Mandatory = $true)][string]$Detail,
        $Evidence = @{},
        [string[]]$Criteria = @()
    )
    if (-not $script:record.Contains('steps')) { $script:record['steps'] = @() }
    $steps = @($script:record['steps'] | Where-Object { $_.id -ne $Id })
    $steps += [ordered]@{
        id        = $Id
        title     = $Title
        status    = $Status
        detail    = $Detail
        criteria  = $Criteria
        evidence  = $Evidence
        recorded  = (Get-Date).ToUniversalTime().ToString('o')
        run_stamp = $script:runStamp
        # Stamped on every result so a later run can tell whether an earlier
        # one is still about the same artifact. Without these two fields,
        # carrying a result forward across runs would be an act of faith.
        flm_exe_sha256   = $script:identityExeSha
        fastflow_revision = $FastFlowRevision
        harness_sha256   = $script:harnessSha
        driver_sha256    = $script:driverSha
        carried_over = $false
    }
    $script:record['steps'] = @($steps | Sort-Object { [double]($_.id -replace '[^0-9.]', '') })
    Trace-Progress ("add: {0} sorted, normalising" -f $Id)
    Save-Record
    Trace-Progress ("add: {0} saved" -f $Id)
    $marker = switch ($Status) { 'met' { 'MET' } 'not_met' { 'NOT MET' } default { 'NOT EXERCISED' } }
    $line = "  [{0}] step {1}: {2}" -f $marker, $Id, $Detail
    Write-Output $line
    try { Add-Content -Path (Join-Path $script:artifactDir 'progress.log') -Value $line -Encoding utf8 } catch { }
}

# Timestamped, and flushed to a side file as well as stdout.
#
# Over SSH the pipeline buffers, so a run that is stuck looks identical to a
# run that is working. The progress file is the only way to tell, and telling
# them apart cost most of an afternoon once.
# Record a step that this run did not select.
#
# Not simply `not_exercised`. Steps 6 through 12 each take tens of minutes on
# hardware and SSH to this box drops sessions, so the acceptance has to be
# runnable in groups and accumulate. Overwriting a `met` result with "not
# selected this time" would make the final record depend on which group ran
# last -- which is how a green run turns red for no reason, and, worse, how a
# red one could turn green.
#
# But carrying a result forward is exactly the move that lets a stale claim
# survive a code change, so it is gated: a previous result is only carried if
# it was recorded against the SAME flm.exe SHA-256 and the SAME FastFlow
# revision. Anything else is discarded and reported as not exercised. A
# carried result is marked `carried_over` with the run stamp that produced it,
# so nothing in the document can quietly present it as fresh.
# Read one field from either shape a step result can take: an ordered
# dictionary built in this session, or a PSCustomObject loaded from the JSON.
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

function Add-UnselectedStep {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [Parameter(Mandatory = $true)][string]$Title
    )
    $prior = @($script:record['steps'] | Where-Object { $_.id -eq $Id })
    if ($prior.Count -gt 0) {
        $p = $prior[0]
        $priorStatus = [string](Get-Field $p 'status')
        $priorExe = [string](Get-Field $p 'flm_exe_sha256')
        $priorRev = [string](Get-Field $p 'fastflow_revision')
        $priorHarness = [string](Get-Field $p 'harness_sha256')
        $priorDriver = [string](Get-Field $p 'driver_sha256')
        # Same binary, same revision, AND same harness. Editing this script
        # invalidates every earlier result, which is the point: a verdict
        # reached by different code is a different verdict.
        if ($priorStatus -ne 'not_exercised' -and
            $priorExe -eq $script:identityExeSha -and
            $priorRev -eq $FastFlowRevision -and
            $priorHarness -eq $script:harnessSha -and
            $priorDriver -eq $script:driverSha) {
            # Rebuilt as a fresh ordered dictionary rather than mutated in
            # place: a prior entry loaded from the JSON file is a
            # PSCustomObject and a prior entry from this session is an ordered
            # dictionary, and adding a property to each needs different code.
            # Copying sidesteps the difference entirely.
            $copy = [ordered]@{}
            if ($p -is [System.Collections.IDictionary]) {
                foreach ($k in @($p.Keys)) { $copy["$k"] = $p[$k] }
            } else {
                foreach ($prop in @($p.PSObject.Properties)) { $copy[$prop.Name] = $prop.Value }
            }
            $copy['carried_over'] = $true
            $carriedInto = @()
            if ($copy.Contains('carried_into')) { $carriedInto = @($copy['carried_into']) }
            $copy['carried_into'] = ($carriedInto + $script:runStamp)
            $others = @($script:record['steps'] | Where-Object { $_.id -ne $Id })
            $script:record['steps'] = @(($others + $copy) | Sort-Object { [double]($_.id -replace '[^0-9.]', '') })
            Save-Record
            $line = "  [{0}] step {1}: carried over from run {2} (same binary, same revision)" -f `
                $(if ($priorStatus -eq 'met') { 'MET' } else { 'NOT MET' }), $Id, (Get-Field $p 'recorded')
            Write-Output $line
            try { Add-Content -Path (Join-Path $script:artifactDir 'progress.log') -Value $line -Encoding utf8 } catch { }
            return
        }
        if ($priorStatus -ne 'not_exercised') {
            Write-Output ("  [DISCARDED] step {0}: a prior '{1}' result was recorded against a different binary, revision or harness" -f $Id, $priorStatus)
        }
    }
    Add-StepResult -Id $Id -Title $Title -Status 'not_exercised' -Detail 'step not selected for this run'
}

function Write-Section {
    param([string]$Name)
    $line = "[{0}] === {1} ===" -f (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $Name
    Write-Output ''
    Write-Output $line
    try { Add-Content -Path (Join-Path $script:artifactDir 'progress.log') -Value $line -Encoding utf8 } catch { }
}

function Test-Selected {
    param([string]$Id)
    if ($Steps -contains 'all') { return $true }
    $wanted = @()
    foreach ($s in $Steps) { $wanted += ($s -split '[,\s]+' | Where-Object { $_ }) }
    $base = ($Id -split '\.')[0] -replace '[^0-9]', ''
    return ($wanted -contains $Id) -or ($wanted -contains $base)
}

function Get-Sha256 {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

# Run a native command, capture everything, never let a stderr line become a
# terminating error.
#
# `$ErrorActionPreference = 'Stop'` turns the first stderr line of a native
# command into a NativeCommandError. That is how test_packaged_runtime's dumpbin
# diagnostic became dead code, and it is worth not repeating.
function Invoke-Capture {
    param([string]$Exe, [string[]]$Arguments, [string]$LogName,
          [string]$WorkingDirectory = '')
    $log = Join-Path $script:artifactDir $LogName
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $pushed = $false
    try {
        if ($WorkingDirectory) { Push-Location $WorkingDirectory; $pushed = $true }
        & $Exe @Arguments *>&1 | Out-File -FilePath $log -Encoding utf8
        $code = $LASTEXITCODE
    } finally {
        if ($pushed) { Pop-Location }
        $ErrorActionPreference = $prev
    }
    $text = if (Test-Path $log) { Get-Content $log -Raw } else { '' }
    if ($null -eq $text) { $text = '' }
    return @{ ExitCode = $code; Log = $log; Text = $text }
}

Set-Field 'schema' 'phi4-aie4-acceptance/1'
Set-Field 'generated_utc' ((Get-Date).ToUniversalTime().ToString('o'))
Set-Field 'run_stamp' $script:runStamp
Set-Field 'artifact_dir' $script:artifactDir
Set-Field 'selected_steps' $Steps
Save-Record

# ---------------------------------------------------------------------------
# Step 0: identity. Not an acceptance criterion; every other step's evidence
# is meaningless without it.
# ---------------------------------------------------------------------------

Write-Section 'Identity'
$identity = [ordered]@{
    machine            = $env:COMPUTERNAME
    user               = "$env:USERDOMAIN\$env:USERNAME"
    utc                = (Get-Date).ToUniversalTime().ToString('o')
    # The commit flm.exe was BUILT from. `checkout_revision` below is the
    # commit this harness ran from; they are allowed to differ only when no
    # product source changed between them, which is checked immediately after.
    binary_revision    = $FastFlowRevision
    fastflow_revision  = $FastFlowRevision
    flm_exe            = $FlmExe
    flm_exe_sha256     = (Get-Sha256 $FlmExe)
    flm_exe_bytes      = $(if (Test-Path $FlmExe) { (Get-Item $FlmExe).Length } else { $null })
    model_dir          = $ModelDir
    model_tag          = $ModelTag
    corelib_runtime_dir = $CorelibRuntimeDir
    corelib_source_revision = $CorelibSourceRevision
    hf_revision        = 'e751fb68c2cfffe6b0d32942118f75ac0a0365bb'
    harness_sha256     = $script:harnessSha
    driver_sha256      = $script:driverSha
    harness_path       = $MyInvocation.MyCommand.Path
}
try {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $identity['cpu'] = $cpu.Name
} catch { $identity['cpu'] = 'unavailable' }
try {
    $npu = Get-PnpDevice -FriendlyName '*NPU*' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($npu) {
        $identity['npu'] = $npu.FriendlyName
        $identity['npu_status'] = $npu.Status
        $drv = Get-PnpDeviceProperty -InstanceId $npu.InstanceId -KeyName 'DEVPKEY_Device_DriverVersion' -ErrorAction SilentlyContinue
        if ($drv) { $identity['npu_driver'] = $drv.Data }
    }
} catch { }
# Every recorded numeric result must name the SHA-256 of the corelib DLL that
# produced it.
$identity['corelib_dlls'] = @()
if ($CorelibRuntimeDir -and (Test-Path $CorelibRuntimeDir)) {
    foreach ($f in (Get-ChildItem -Path $CorelibRuntimeDir -Filter '*.dll' | Sort-Object Name)) {
        $identity['corelib_dlls'] += [ordered]@{
            name = $f.Name; bytes = $f.Length; sha256 = (Get-Sha256 $f.FullName)
        }
    }
}
# The checkout must actually be at the revision we are claiming.
$identity['checkout_revision'] = $null
$identity['checkout_clean'] = $null
try {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $identity['checkout_revision'] = (& git -C $RepoRoot rev-parse HEAD 2>$null).Trim()
    $dirty = @(& git -C $RepoRoot status --porcelain 2>$null | Where-Object { $_ -notmatch '^\?\?' })
    $identity['checkout_clean'] = ($dirty.Count -eq 0)
    $identity['checkout_dirty_tracked'] = $dirty
    $ErrorActionPreference = $prev
} catch { }
Set-Field 'identity' $identity
Save-Record
$identity.GetEnumerator() | ForEach-Object { Write-Output ("  {0,-24} {1}" -f $_.Key, $_.Value) }

# The binary and the harness may sit at different commits, and that is fine
# exactly as long as nothing the BINARY is built from changed between them.
#
# -FastFlowRevision names the commit flm.exe was built from. The checkout will
# usually be ahead of it, because committing this harness moves HEAD without
# touching a line of product code. Demanding equality would force a
# two-and-a-half-hour rebuild and re-run for a documentation commit, and
# people who are forced to do that start passing the check a stale value
# instead. Asking the sharper question -- did any product source change
# between the two commits -- costs one git command and cannot be satisfied by
# a stale argument.
if ($identity['checkout_revision'] -and $identity['checkout_revision'] -ne $FastFlowRevision) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $changed = @(& git -C $RepoRoot diff --name-only "$FastFlowRevision..$($identity['checkout_revision'])" 2>$null)
    $gitOk = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prev
    $identity['files_changed_since_binary_revision'] = $changed
    # src/test/** and the docs do not enter flm.exe. Anything else under src/
    # does.
    $productChanged = @($changed | Where-Object {
        $_ -like 'src/*' -and $_ -notlike 'src/test/*' })
    $identity['product_sources_changed_since_binary_revision'] = $productChanged
    if (-not $gitOk) {
        Write-Output ''
        Write-Output ("  Cannot compare the binary's revision {0} with the checkout {1}: git failed." -f `
            $FastFlowRevision, $identity['checkout_revision'])
        Write-Output '  Without that comparison every result below is attributed to code that may not have produced it.'
        Set-Field 'aborted' 'could not determine whether product sources changed between the binary revision and the checkout'
        Save-Record
        exit 1
    }
    if ($productChanged.Count -gt 0) {
        Write-Output ''
        Write-Output ("  STALE BINARY: {0} product source file(s) changed between {1} (which built flm.exe) and the checkout {2}:" -f `
            $productChanged.Count, $FastFlowRevision, $identity['checkout_revision'])
        $productChanged | ForEach-Object { Write-Output "    $_" }
        Write-Output '  Rebuild before running. Refusing to attribute these results to code the binary does not contain.'
        Set-Field 'aborted' 'product sources changed after the binary under test was built'
        Save-Record
        exit 1
    }
    Write-Output ''
    Write-Output ("  flm.exe was built from {0}; the checkout is at {1}. No product source differs between them" -f `
        $FastFlowRevision, $identity['checkout_revision'])
    Write-Output ("  ({0} file(s) changed, all under src/test or outside src/), so the binary still corresponds to this tree." -f $changed.Count)
}

# ---------------------------------------------------------------------------
# Step 1: the assembled package matches the ratified contract
# ---------------------------------------------------------------------------

if (Test-Selected '1') {
    Write-Section 'Step 1: assembled package vs the ratified contract (PACKAGE-1, MODEL-2)'
    $ev = [ordered]@{}
    $problems = @()

    # 1a/1b/1d: the offline catalog validator checks the two provenances --
    # every upstream file has an HF metadata record with a matching Git blob
    # oid or LFS sha256/size, every overlay file has none, and provenance.json
    # names the pinned commit. It is the committed contract checker; running it
    # here proves the tree ON THIS BOX satisfies it.
    $validate = Invoke-Capture -Exe $Python -Arguments @(
        '-m', 'tools.package_phi4_corelib_aie4', 'validate-catalog',
        '--overlay-dir', $OverlayDir,
        '--model-list', $ModelList,
        '--model-info', $ModelInfo) -LogName 'step1-validate-catalog.log' -WorkingDirectory $RepoRoot
    $ev['validate_catalog_exit'] = $validate.ExitCode
    $ev['validate_catalog_log'] = $validate.Log
    if ($validate.ExitCode -ne 0) { $problems += "validate-catalog exited $($validate.ExitCode)" }

    # The assembled directory, not just the catalog. A catalog can be perfect
    # while the directory on disk is wrong, and the directory is what loads.
    # model_list.json is { model_path, models: { <family> : { <size> : entry } } }
    # and the tag is "<family>:<size>".
    $catalog = Get-Content $ModelList -Raw | ConvertFrom-Json
    $entry = $null
    $tagParts = $ModelTag -split ':'
    try {
        $family = $catalog.models.($tagParts[0])
        if ($family) { $entry = $family.($tagParts[1]) }
    } catch { }
    $ev['catalog_entry_found'] = [bool]$entry
    if ($entry) {
        foreach ($k in @('size', 'footprint', 'flm_min_version', 'max_prefill_len',
                         'default_context_length', 'modelscope_supported')) {
            if ($entry.PSObject.Properties.Name -contains $k) { $ev["catalog_$k"] = $entry.$k }
        }
        # "A new phi4-mini-it-aie4:4b tag selects only corelib_aie4."
        $backend = $null
        try { $backend = $entry.details.execution_backend } catch { }
        $ev['catalog_execution_backend'] = $backend
        if ($backend -ne 'corelib_aie4') { $problems += "catalog entry selects backend '$backend', not corelib_aie4" }
        # The catalog must NOT invent a maximum-context field; it reuses the
        # existing context fields.
        $ev['catalog_fields'] = @($entry.PSObject.Properties.Name)
        $invented = @($entry.PSObject.Properties.Name | Where-Object { $_ -match '(?i)max_context' })
        $ev['invented_max_context_fields'] = $invented
        if ($invented.Count -gt 0) { $problems += "catalog entry uses a nonexistent maximum-context field: $($invented -join ', ')" }
        # Existing Q4NX/NPU2 selection must be unchanged. The legacy entry
        # selects that path by having NO execution_backend at all, so the
        # check is that the entry still exists, still ships model.q4nx, and
        # still does not name a backend -- not that it names a particular one.
        $legacyEntry = $null
        try { $legacyEntry = $catalog.models.'phi4-mini-it'.'4b' } catch { }
        if (-not $legacyEntry) {
            $problems += 'the legacy phi4-mini-it:4b catalog entry is missing'
        } else {
            $legacyBackend = $null
            try {
                if ($legacyEntry.details.PSObject.Properties.Name -contains 'execution_backend') {
                    $legacyBackend = $legacyEntry.details.execution_backend
                }
            } catch { }
            $ev['legacy_phi4_backend'] = $legacyBackend
            $ev['legacy_phi4_files'] = @($legacyEntry.files)
            $ev['legacy_phi4_default_context_length'] = $legacyEntry.default_context_length
            if ($legacyBackend) { $problems += "the legacy phi4-mini-it:4b entry now names execution_backend '$legacyBackend'; it selected the Q4NX/NPU2 path by naming none" }
            if (@($legacyEntry.files) -notcontains 'model.q4nx') { $problems += 'the legacy phi4-mini-it:4b entry no longer ships model.q4nx' }
        }
    } else {
        $problems += "no catalog entry for $ModelTag in $ModelList"
    }

    # Upstream vs overlay membership, on disk.
    $overlayNames = @(Get-ChildItem -Path $OverlayDir -File | ForEach-Object { $_.Name })
    $present = @(Get-ChildItem -Path $ModelDir -File | ForEach-Object { $_.Name })
    $ev['overlay_files'] = $overlayNames
    $ev['model_dir_files'] = $present
    $missingOverlay = @($overlayNames | Where-Object { $present -notcontains $_ })
    if ($missingOverlay.Count -gt 0) { $problems += "overlay files absent from the model dir: $($missingOverlay -join ', ')" }

    # Overlay bytes on disk must equal the committed overlay bytes. If they
    # differ, the thing under test is not the thing that was reviewed.
    $overlayHashes = @()
    foreach ($n in $overlayNames) {
        $src = Join-Path $OverlayDir $n
        $dst = Join-Path $ModelDir $n
        $srcHash = Get-Sha256 $src
        $dstHash = if (Test-Path $dst) { Get-Sha256 $dst } else { $null }
        $overlayHashes += [ordered]@{ name = $n; committed_sha256 = $srcHash; installed_sha256 = $dstHash; identical = ($srcHash -eq $dstHash) }
        if ($srcHash -ne $dstHash) { $problems += "overlay $n on disk differs from the committed overlay" }
    }
    $ev['overlay_hashes'] = $overlayHashes

    # provenance.json: pinned commit and per-file oid/size actually recorded.
    $provPath = Join-Path $ModelDir 'provenance.json'
    if (Test-Path $provPath) {
        $prov = Get-Content $provPath -Raw | ConvertFrom-Json
        $ev['provenance_upstream_commit'] = $prov.upstream.commit
        $gitFiles = @($prov.upstream.git_files)
        $upstreamFiles = @($gitFiles | ForEach-Object { $_.path })
        $ev['provenance_upstream_file_count'] = $upstreamFiles.Count
        $ev['provenance_upstream_files'] = $upstreamFiles
        $withOid = @($gitFiles | Where-Object {
            ($_.PSObject.Properties.Name -contains 'oid') -and ($_.oid) -and
            ($_.PSObject.Properties.Name -contains 'size') -and ($null -ne $_.size) })
        $ev['provenance_files_with_oid_and_size'] = $withOid.Count
        if ($prov.upstream.commit -ne 'e751fb68c2cfffe6b0d32942118f75ac0a0365bb') {
            $problems += "provenance.json names commit $($prov.upstream.commit), not the pinned e751fb68..."
        }
        if ($withOid.Count -ne $upstreamFiles.Count) {
            $problems += 'provenance.json has upstream files without both oid and size'
        }

        # Every upstream file present on disk must have a metadata record, and
        # its bytes must match the recorded size. Size is not a hash, and this
        # does not claim to be one: the LFS oid IS a sha256 and is checked
        # below where one is published.
        $upstreamOnDisk = @($present | Where-Object { $overlayNames -notcontains $_ })
        $ev['upstream_files_on_disk'] = $upstreamOnDisk
        $noRecord = @($upstreamOnDisk | Where-Object { $upstreamFiles -notcontains $_ })
        $ev['upstream_files_without_a_metadata_record'] = $noRecord
        if ($noRecord.Count -gt 0) { $problems += "upstream file(s) on disk with no HF metadata record: $($noRecord -join ', ')" }
        $sizeMismatch = @()
        $lfsVerified = @()
        foreach ($rec in $gitFiles) {
            # tokenizer_config.json legitimately SHADOWS its upstream file:
            # the published one carries neither a chat template nor the EOS
            # token IDs this backend needs. Comparing the shipped overlay's
            # bytes against the upstream record would report the overlay
            # working as designed as a corruption. Its identity is checked
            # against the COMMITTED overlay above, which is the contract that
            # actually applies to it.
            if ($overlayNames -contains $rec.path) { continue }
            $disk = Join-Path $ModelDir $rec.path
            if (-not (Test-Path $disk)) { continue }
            $len = (Get-Item $disk).Length
            if ($len -ne $rec.size) { $sizeMismatch += "$($rec.path): on disk $len, recorded $($rec.size)" }
            if ($rec.PSObject.Properties.Name -contains 'lfs' -and $rec.lfs) {
                # The LFS oid is the sha256 of the file's real content, so this
                # is a genuine byte-identity check against the pinned revision
                # -- including the 3.2 GB model.onnx.data.
                $h = Get-Sha256 $disk
                $lfsVerified += [ordered]@{ path = $rec.path; expected = $rec.lfs.oid; actual = $h; identical = ($h -eq $rec.lfs.oid) }
                if ($h -ne $rec.lfs.oid) { $problems += "$($rec.path) does not match the pinned revision's LFS sha256" }
            }
        }
        $ev['upstream_size_mismatches'] = $sizeMismatch
        $ev['upstream_lfs_sha256_verified'] = $lfsVerified
        if ($sizeMismatch.Count -gt 0) { $problems += "upstream file size mismatch: $($sizeMismatch -join '; ')" }

        # The two provenances must not blur: an authored overlay carrying an
        # upstream metadata record would mean FastFlow's own package contract
        # had been published to the model repository. tokenizer_config.json is
        # the deliberate exception -- it SHADOWS an upstream file.
        $overlayWithoutUpstream = @('config.json', 'corelib_phi4_manifest.json', 'provenance.json')
        $leaked = @($overlayWithoutUpstream | Where-Object { $upstreamFiles -contains $_ })
        $ev['authored_overlays_with_upstream_record'] = $leaked
        if ($leaked.Count -gt 0) { $problems += "authored overlay(s) carry an upstream metadata record: $($leaked -join ', ')" }
        $ev['shadowing_overlay_has_upstream_record'] = ($upstreamFiles -contains 'tokenizer_config.json')
        if (-not ($upstreamFiles -contains 'tokenizer_config.json')) {
            $problems += 'tokenizer_config.json shadows an upstream file but carries no upstream metadata record'
        }
    } else {
        $problems += 'provenance.json missing from the assembled model directory'
    }

    # The manifest generator must take model identity from the ONNX
    # initializers, never from the overlay config.json. Checked by reading the
    # generator: config.json must not be an input to it at all.
    $genPath = Join-Path $RepoRoot 'tools/generate_phi4_corelib_manifest.py'
    if (Test-Path $genPath) {
        $genText = Get-Content $genPath -Raw
        $readsOverlay = ($genText -match "config\.json")
        $ev['manifest_generator_mentions_config_json'] = $readsOverlay
        if ($readsOverlay) { $problems += 'tools/generate_phi4_corelib_manifest.py mentions config.json; the overlay must never be the source of model identity' }
    } else {
        $problems += "manifest generator not found at $genPath"
    }

    if ($problems.Count -eq 0) {
        Add-StepResult -Id '1' -Title 'Assembled package matches the ratified contract' -Status 'met' `
            -Detail 'validate-catalog passed; overlays present and byte-identical to the committed overlays; provenance.json pins e751fb68 with oid+size per upstream file; the manifest generator does not read config.json' `
            -Evidence $ev -Criteria @('C01', 'C02', 'C03', 'C08')
    } else {
        $ev['problems'] = $problems
        Add-StepResult -Id '1' -Title 'Assembled package matches the ratified contract' -Status 'not_met' `
            -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C01', 'C02', 'C03', 'C08')
    }

    # 1c (MODEL-2) is a separate result, because it is a separate claim and
    # collapsing it into the one above is how a negative test disappears.
    Write-Section 'Step 1c: a corrupted overlay config.json fails model load (MODEL-2)'
    $ev2 = [ordered]@{}
    $cfgPath = Join-Path $ModelDir 'config.json'
    $backup = Join-Path $script:artifactDir 'config.json.orig'
    if (-not (Test-Path $cfgPath)) {
        Add-StepResult -Id '1c' -Title 'Corrupted overlay config.json fails model load (MODEL-2)' `
            -Status 'not_exercised' -Detail "no config.json at $cfgPath to corrupt" -Evidence $ev2 -Criteria @('C08')
    } else {
        Trace-Progress '1c: backing up'
        Copy-Item $cfgPath $backup -Force
        try {
            # Corrupt the TEXT, not a parsed object.
            #
            # Round-tripping through ConvertFrom-Json/ConvertTo-Json rewrites
            # the whole file -- reindenting it and turning 1e-05 into 1E-05 --
            # so the restore afterwards puts back a file that is no longer
            # byte-identical to the committed overlay. Measured: the overlay
            # went from 257 bytes to 302, and the NEXT run correctly reported
            # the model directory as corrupt, for a corruption this check had
            # introduced.
            $cfgText = Get-Content $cfgPath -Raw
            # num_hidden_layers is a Section 5.1 constant AND is derivable from
            # the ONNX initializers, so a disagreement is exactly the case
            # MODEL-2 is about: the overlay restates the contract, it must
            # never override it.
            $field = $null
            foreach ($cand in @('num_hidden_layers', 'hidden_size', 'num_attention_heads')) {
                if ($cfgText -match ('"' + $cand + '"\s*:\s*(\d+)')) { $field = $cand; break }
            }
            $ev2['corrupted_field'] = $field
            if (-not $field) {
                Add-StepResult -Id '1c' -Title 'Corrupted overlay config.json fails model load (MODEL-2)' `
                    -Status 'not_exercised' -Detail 'overlay config.json carries none of the Section 5.1 constants this check corrupts' `
                    -Evidence $ev2 -Criteria @('C08')
            } else {
                [void]($cfgText -match ('"' + $field + '"\s*:\s*(\d+)'))
                $original = [int]$Matches[1]
                $ev2['original_value'] = $original
                $ev2['corrupted_value'] = $original + 1
                $corruptText = $cfgText -replace ('"' + $field + '"(\s*:\s*)\d+'), ('"' + $field + '"${1}' + ($original + 1))
                Trace-Progress '1c: writing corrupted overlay'
                [System.IO.File]::WriteAllText($cfgPath, $corruptText)
                Trace-Progress '1c: launching flm run'
                $run = Invoke-Capture -Exe $FlmExe -Arguments @('run', $ModelTag) -LogName 'step1c-corrupt-load.log'
                $ev2['flm_exit'] = $run.ExitCode
                $ev2['flm_output'] = $run.Text
                $ev2['log'] = $run.Log
                # A load that SUCCEEDS against a corrupted contract is the
                # failure: it means the ONNX-validated constants were silently
                # overridden, or ignored.
                #
                # The exit code alone will not do. Under a redirected stdin the
                # REPL prints its banner and returns 0 on the first
                # ReadConsoleInput, so exit 0 is what a SUCCESSFUL load looks
                # like here too. "Did it reach the REPL banner" is the signal
                # that separates a completed load from a rejected one.
                Trace-Progress ('1c: flm exited ' + $run.ExitCode)
                $reachedRepl = ($run.Text -match 'Type /\? for help')
                $ev2['reached_repl_banner'] = $reachedRepl
                if ($run.ExitCode -eq 0 -and $reachedRepl) {
                    Add-StepResult -Id '1c' -Title 'Corrupted overlay config.json fails model load (MODEL-2)' `
                        -Status 'not_met' -Detail 'model load reached the REPL banner with a corrupted overlay config.json value; the ONNX-validated constant was not enforced' `
                        -Evidence $ev2 -Criteria @('C08')
                } else {
                    Add-StepResult -Id '1c' -Title 'Corrupted overlay config.json fails model load (MODEL-2)' `
                        -Status 'met' -Detail "model load rejected the corrupted overlay value (exit $($run.ExitCode), REPL banner reached = $reachedRepl)" `
                        -Evidence $ev2 -Criteria @('C08')
                }
            }
        } finally {
            # Restore from the COMMITTED overlay, not from the backup taken a
            # moment ago. If a previous run of this step was killed before its
            # own restore, that backup is itself corrupt, and restoring it
            # would launder the corruption forward one run at a time. The
            # committed overlay is the only authoritative copy.
            Copy-Item (Join-Path $OverlayDir 'config.json') $cfgPath -Force
            $restored = Get-Sha256 $cfgPath
            $committed = Get-Sha256 (Join-Path $OverlayDir 'config.json')
            $ev2['config_restored_to_committed_bytes'] = ($restored -eq $committed)
            Write-Output ("  config.json restored: sha256 matches committed overlay = {0}" -f ($restored -eq $committed))
        }
    }
} else {
    Add-UnselectedStep -Id '1' -Title 'Assembled package matches the ratified contract'
    Add-UnselectedStep -Id '1c' -Title 'Corrupted overlay config.json fails model load (MODEL-2)'
}

# ---------------------------------------------------------------------------
# Step 2: the binary built here does not import the corelib
# ---------------------------------------------------------------------------

if (Test-Selected '2') {
    Write-Section 'Step 2: dumpbin /DEPENDENTS no-import check on the binary built on this box'
    $ev = [ordered]@{}
    # Bounded discovery. <MSVC>\<version>\bin\Hostx64\x64\dumpbin.exe is a
    # fixed shape, so enumerating one directory level is enough. An earlier
    # version walked the whole MSVC tree with -Recurse; that is tens of
    # thousands of files, and it is a poor trade for a path that is known.
    $dumpbin = $null
    foreach ($root in @(
        'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC',
        'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC',
        'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC')) {
        if (-not (Test-Path $root)) { continue }
        foreach ($ver in (Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
                          Sort-Object Name -Descending)) {
            $candidate = Join-Path $ver.FullName 'bin\Hostx64\x64\dumpbin.exe'
            if (Test-Path $candidate) { $dumpbin = $candidate; break }
        }
        if ($dumpbin) { break }
    }
    if (-not $dumpbin) { $dumpbin = (Get-Command dumpbin.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Source }
    $ev['dumpbin'] = $dumpbin
    if (-not $dumpbin) {
        Add-StepResult -Id '2' -Title 'flm.exe does not import the corelib import library' -Status 'not_exercised' `
            -Detail 'dumpbin.exe not found on this box; the no-import check could not be run' -Evidence $ev -Criteria @('C05')
    } else {
        $dump = Invoke-Capture -Exe $dumpbin -Arguments @('/nologo', '/dependents', $FlmExe) -LogName 'step2-dumpbin.log'
        $ev['exit'] = $dump.ExitCode
        $ev['log'] = $dump.Log
        $imports = @()
        foreach ($line in ($dump.Text -split "`r?`n")) {
            $t = $line.Trim()
            if ($t -match '^[A-Za-z0-9_.\-+]+\.dll$') { $imports += $t }
        }
        $ev['imports'] = $imports
        $ev['saw_header'] = ($dump.Text -match 'Image has the following dependencies')
        $bad = @($imports | Where-Object { $_ -match '(?i)^ryzenai_corelib\.dll$' })
        $ev['corelib_imports'] = $bad
        if ($dump.ExitCode -ne 0) {
            Add-StepResult -Id '2' -Title 'flm.exe does not import the corelib import library' -Status 'not_met' `
                -Detail "dumpbin exited $($dump.ExitCode)" -Evidence $ev -Criteria @('C05')
        } elseif (-not $ev['saw_header']) {
            Add-StepResult -Id '2' -Title 'flm.exe does not import the corelib import library' -Status 'not_met' `
                -Detail 'dumpbin produced no dependency listing, so the absence of an import proves nothing' -Evidence $ev -Criteria @('C05')
        } elseif ($bad.Count -gt 0) {
            Add-StepResult -Id '2' -Title 'flm.exe does not import the corelib import library' -Status 'not_met' `
                -Detail "flm.exe imports $($bad -join ', ')" -Evidence $ev -Criteria @('C05')
        } else {
            Add-StepResult -Id '2' -Title 'flm.exe does not import the corelib import library' -Status 'met' `
                -Detail "dumpbin listed $($imports.Count) imports on the binary built here; ryzenai_corelib.dll is not among them" `
                -Evidence $ev -Criteria @('C05')
        }
    }
} else {
    Add-UnselectedStep -Id '2' -Title 'flm.exe does not import the corelib import library'
}

# ---------------------------------------------------------------------------
# Step 3: runtime closure with THIS box's DD linkage
# ---------------------------------------------------------------------------

if (Test-Selected '3') {
    Write-Section 'Step 3: runtime closure, flm validate readiness, and non-AIE4 startup without it'
    $ev = [ordered]@{}
    $problems = @()

    $ev['staged_files'] = @()
    if ($CorelibRuntimeDir -and (Test-Path $CorelibRuntimeDir)) {
        foreach ($f in (Get-ChildItem -Path $CorelibRuntimeDir -File | Sort-Object Name)) {
            $ev['staged_files'] += [ordered]@{ name = $f.Name; bytes = $f.Length; sha256 = (Get-Sha256 $f.FullName) }
        }
    } else {
        $problems += "staged runtime directory $CorelibRuntimeDir not found"
    }

    # flm validate, with the JSON PARSED. Asserting only exit 0 is how a "ready"
    # claim survives a runtime that is not ready: `validate` returns 0 for a
    # clean report of a broken backend too.
    $val = Invoke-Capture -Exe $FlmExe -Arguments @('validate', '--json') -LogName 'step3-validate.log'
    $ev['validate_exit'] = $val.ExitCode
    $ev['validate_log'] = $val.Log
    $ev['validate_raw'] = $val.Text
    $ready = $null
    try {
        $jsonStart = $val.Text.IndexOf('{')
        if ($jsonStart -ge 0) {
            $parsed = $val.Text.Substring($jsonStart) | ConvertFrom-Json
            $ev['validate_json'] = $parsed
            if ($parsed.PSObject.Properties.Name -contains 'corelib_aie4') {
                $ready = [bool]$parsed.corelib_aie4.ready
                $ev['corelib_aie4_ready'] = $ready
            }
        }
    } catch { $ev['validate_json_parse_error'] = $_.Exception.Message }
    if ($null -eq $ready) {
        # Fall back to the human line, and SAY that is what happened.
        $humanReady = ($val.Text -match 'Corelib AIE4: ready')
        $ev['corelib_aie4_ready_from_human_output'] = $humanReady
        $ev['note'] = 'flm validate --json did not yield a parsable corelib_aie4.ready field; the human-readable line was used instead'
        if (-not $humanReady) { $problems += 'flm validate did not report corelib AIE4 readiness' }
    } elseif (-not $ready) {
        $problems += 'flm validate reported corelib_aie4.ready = false'
    }

    # Removing the aie4 directory must leave non-AIE4 FastFlow working. Done on
    # a COPY: the original staged closure is what everything else in this run
    # depends on.
    # Copy the executable and its sibling FILES only.
    #
    # Not -Recurse over the whole directory: the aie4 subdirectory alone is
    # about a gigabyte (dyn_bins.dll and dyn_dispatch_core.dll), and the build
    # tree beside it holds this suite's own artifacts. An earlier version of
    # this step copied all of it and spent sixteen minutes of CPU doing so.
    # The aie4 directory is the one thing this check exists to leave out, so
    # not copying it is the check.
    $exeDir = Split-Path -Parent $FlmExe
    $copyDir = Join-Path $script:artifactDir 'no-aie4'
    New-Item -ItemType Directory -Force -Path $copyDir | Out-Null
    Get-ChildItem -Path $exeDir -File | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $copyDir -Force -ErrorAction SilentlyContinue
    }
    $ev['source_had_aie4_dir'] = (Test-Path (Join-Path $exeDir 'aie4'))
    $ev['copy_has_aie4_dir'] = (Test-Path (Join-Path $copyDir 'aie4'))
    $ev['copied_files'] = @(Get-ChildItem -Path $copyDir -File | ForEach-Object { $_.Name })
    if (-not $ev['source_had_aie4_dir']) {
        $problems += "the product directory $exeDir has no aie4 subdirectory, so removing it proves nothing"
    }
    $copiedExe = Join-Path $copyDir (Split-Path -Leaf $FlmExe)
    if (Test-Path $copiedExe) {
        # RYZENAI_CORELIB_PATH would defeat the point: it pins the DLL by
        # absolute path, so deleting the aie4 directory changes nothing.
        $savedPath = $env:RYZENAI_CORELIB_PATH
        try {
            Remove-Item Env:\RYZENAI_CORELIB_PATH -ErrorAction SilentlyContinue
            $ver = Invoke-Capture -Exe $copiedExe -Arguments @('version') -LogName 'step3-version-no-aie4.log'
            $ev['version_without_aie4_exit'] = $ver.ExitCode
            $ev['version_without_aie4_output'] = $ver.Text
            if ($ver.ExitCode -ne 0) { $problems += "flm version failed (exit $($ver.ExitCode)) after removing the aie4 directory" }

            # The reported version must have all three components.
            #
            # The binary this acceptance first ran against printed "FLM v1",
            # because an unquoted -DFLM_VERSION=1.0.4 in a PowerShell build
            # script reaches CMake as -DFLM_VERSION=1. Nothing failed: the
            # build succeeded and the binary ran. src/CMakeLists.txt now
            # refuses such a value at configure time, and this asserts the
            # same property from the other end, on the shipped artifact, so
            # a binary built by some path that bypasses the CMake guard still
            # cannot pass this acceptance.
            if ($ver.Text -match 'FLM v(\S+)') {
                $ev['reported_version'] = $Matches[1]
                if ($Matches[1] -notmatch '^[0-9]+\.[0-9]+\.[0-9]+') {
                    $problems += "flm version reports '$($Matches[1])', which is not a major.minor.patch version; the build's FLM_VERSION was truncated"
                }
            } else {
                $problems += 'flm version printed no "FLM v<version>" line, so the compiled-in version could not be checked'
            }
        } finally {
            if ($savedPath) { $env:RYZENAI_CORELIB_PATH = $savedPath }
        }
    } else {
        $problems += 'could not stage a copy of the product directory to test aie4 removal'
    }

    # API-5 is NOT claimed by this step. It is measured by Step 3b, which runs
    # test_phi4_hardware here rather than citing a run that did not happen.
    $ev['api5_version_gate'] = 'measured by step 3b in this same run; this step does not claim it'

    if ($problems.Count -eq 0) {
        Add-StepResult -Id '3' -Title 'Runtime closure on this box''s DD linkage' -Status 'met' `
            -Detail "closure staged as $($ev['staged_files'].Count) files; flm validate reports corelib AIE4 ready; flm version still works with the aie4 directory removed" `
            -Evidence $ev -Criteria @('C06', 'C24', 'C25')
    } else {
        $ev['problems'] = $problems
        Add-StepResult -Id '3' -Title 'Runtime closure on this box''s DD linkage' -Status 'not_met' `
            -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C06', 'C24', 'C25')
    }
} else {
    Add-UnselectedStep -Id '3' -Title 'Runtime closure on this box''s DD linkage'
}

# ---------------------------------------------------------------------------
# Step 4: the fatal-log precondition, positive and negative
# ---------------------------------------------------------------------------

if (Test-Selected '4') {
    Write-Section 'Step 4: fatal-log directory precondition (design 12.1)'
    $ev = [ordered]@{}
    $problems = @()
    $logDir = Join-Path $env:LOCALAPPDATA 'FastFlowLM\logs'
    $ev['log_dir'] = $logDir
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null

    # Positive: writable for this account.
    $probe = Join-Path $logDir ("acceptance-probe-$PID.tmp")
    try {
        Set-Content -Path $probe -Value 'probe' -Encoding ascii
        $ev['writable'] = $true
        Remove-Item $probe -Force -ErrorAction SilentlyContinue
    } catch {
        $ev['writable'] = $false
        $problems += "fatal-log directory is not writable: $($_.Exception.Message)"
    }

    # Negative. FatalRecordStore resolves the directory through
    # SHGetKnownFolderPath, deliberately NOT the LOCALAPPDATA environment
    # variable, so it cannot be redirected. The only way to make it unwritable
    # is to deny write on the real directory with an ACL.
    $denied = $false
    try {
        $acl = Get-Acl $logDir
        $ev['acl_before'] = ($acl.Access | ForEach-Object { "$($_.IdentityReference):$($_.FileSystemRights):$($_.AccessControlType)" })
        # Built into a variable first, and combined as integers.
        #
        # Spelling the -bor chain inline inside New-Object's argument list
        # makes PowerShell parse the continuation lines as ARRAY ELEMENTS, and
        # the constructor then fails with "System.Object[] does not contain a
        # method named op_BitwiseOr" -- an error that says nothing about the
        # real cause. Measured here, once.
        $rights = [System.Security.AccessControl.FileSystemRights](
            [int][System.Security.AccessControl.FileSystemRights]::CreateFiles -bor
            [int][System.Security.AccessControl.FileSystemRights]::WriteData -bor
            [int][System.Security.AccessControl.FileSystemRights]::AppendData)
        $ev['denied_rights'] = [string]$rights
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            "$env:USERDOMAIN\$env:USERNAME", $rights,
            'ContainerInherit,ObjectInherit', 'None', 'Deny')
        $acl.AddAccessRule($rule)
        Set-Acl -Path $logDir -AclObject $acl
        $denied = $true

        # Confirm the deny actually bit. Without this the whole negative case
        # can pass vacuously against a still-writable directory.
        $stillWritable = $true
        try {
            Set-Content -Path $probe -Value 'probe' -Encoding ascii
            Remove-Item $probe -Force -ErrorAction SilentlyContinue
        } catch { $stillWritable = $false }
        $ev['deny_took_effect'] = (-not $stillWritable)
        if ($stillWritable) { $problems += 'the Deny ACL did not make the directory unwritable, so the negative path was never exercised' }

        # The AIE4 tag must be rejected with an actionable error...
        $aie4 = Invoke-Capture -Exe $FlmExe -Arguments @('run', $ModelTag) -LogName 'step4-aie4-unwritable.log'
        $ev['aie4_exit'] = $aie4.ExitCode
        $ev['aie4_output'] = $aie4.Text
        # "Actionable" means a reader can find the thing to fix from the
        # message alone: it has to name the fatal-log location and say that
        # writing there failed. It deliberately does NOT require particular
        # words like "permission" -- the product reports the Win32 status, and
        # demanding prose would be this check asserting a style rather than a
        # property.
        $namesPath = ($aie4.Text -match [regex]::Escape($logDir))
        $namesFailure = ($aie4.Text -match '(?i)fatal record') -and
                        ($aie4.Text -match '(?i)(fail|error)')
        $actionable = $namesPath -and $namesFailure
        $ev['aie4_error_names_the_log_path'] = $namesPath
        $ev['aie4_error_names_the_failure'] = $namesFailure
        $ev['aie4_error_is_actionable'] = $actionable
        # Recorded, not gated: the message carries the raw Win32 status ("error
        # 5") rather than its meaning ("access is denied"). A reader who does
        # not know the code has to look it up. That is a legibility nit on a
        # message that is otherwise precise, and it is written down here rather
        # than either ignored or inflated into a failure.
        if ($aie4.Text -match '\(error (\d+)\)') {
            $ev['aie4_error_reports_bare_win32_code'] = [int]$Matches[1]
            $ev['aie4_error_legibility_note'] = "the rejection names the file and the Win32 status $($Matches[1]) but not its meaning; error 5 is ERROR_ACCESS_DENIED"
        }
        if ($aie4.ExitCode -eq 0) { $problems += 'the AIE4 tag was NOT rejected while the fatal-log directory was unwritable' }
        elseif (-not $actionable) { $problems += 'the AIE4 rejection message does not name the log path and the write failure, so it is not actionable' }

        # ...while other tags still load. `flm list` and `flm version` are the
        # non-AIE4 startup paths that must be unaffected.
        $other = Invoke-Capture -Exe $FlmExe -Arguments @('list') -LogName 'step4-list-unwritable.log'
        $ev['non_aie4_exit'] = $other.ExitCode
        $ev['non_aie4_output'] = $other.Text
        if ($other.ExitCode -ne 0) { $problems += "non-AIE4 startup (flm list) also failed (exit $($other.ExitCode)) while the log directory was unwritable" }
    } catch {
        $problems += "could not exercise the unwritable-log path: $($_.Exception.Message)"
        $ev['negative_path_error'] = $_.Exception.Message
    } finally {
        if ($denied) {
            $acl2 = Get-Acl $logDir
            $toRemove = @($acl2.Access | Where-Object {
                $_.AccessControlType -eq 'Deny' -and
                $_.IdentityReference.Value -eq "$env:USERDOMAIN\$env:USERNAME" })
            foreach ($r in $toRemove) { [void]$acl2.RemoveAccessRule($r) }
            Set-Acl -Path $logDir -AclObject $acl2
            $restoredOk = $true
            try {
                Set-Content -Path $probe -Value 'probe' -Encoding ascii
                Remove-Item $probe -Force -ErrorAction SilentlyContinue
            } catch { $restoredOk = $false }
            $ev['acl_restored_and_writable'] = $restoredOk
            Write-Output "  fatal-log ACL restored; writable again = $restoredOk"
            if (-not $restoredOk) { $problems += 'FAILED TO RESTORE the fatal-log ACL; every later step is compromised' }
        }
    }

    if ($problems.Count -eq 0) {
        Add-StepResult -Id '4' -Title 'Unwritable fatal-log directory rejects the AIE4 tag only' -Status 'met' `
            -Detail 'the directory is writable normally; with write denied the AIE4 tag is rejected with a message naming the log write failure, and non-AIE4 startup is unaffected' `
            -Evidence $ev -Criteria @('C07')
    } else {
        $ev['problems'] = $problems
        Add-StepResult -Id '4' -Title 'Unwritable fatal-log directory rejects the AIE4 tag only' -Status 'not_met' `
            -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C07')
    }
} else {
    Add-UnselectedStep -Id '4' -Title 'Unwritable fatal-log directory rejects the AIE4 tag only'
}

# ---------------------------------------------------------------------------
# Step 5: build the AIE4 package from the published model
# ---------------------------------------------------------------------------

if (Test-Selected '5') {
    Write-Section 'Step 5: manifest generation against the real published model'
    $ev = [ordered]@{}
    $problems = @()
    $out = Join-Path $script:artifactDir 'corelib_phi4_manifest.regenerated.json'
    # --full-hash, deliberately.
    #
    # Without it the generator records only each file's size, and the resulting
    # manifest is 168 bytes shorter than the committed overlay -- which reads
    # as manifest drift when it is nothing of the kind. With it, the generator
    # hashes model.onnx and model.onnx.data, so the byte-identity comparison
    # below is a real one AND the recorded hashes are checked against the
    # published LFS oids for free. It costs a full read of 3.2 GB; that is the
    # right price for the difference between "the same" and "the same size".
    $gen = Invoke-Capture -Exe $Python -Arguments @(
        '-m', 'tools.generate_phi4_corelib_manifest',
        '--model-dir', $ModelDir,
        '--output', $out, '--full-hash') -LogName 'step5-generate-manifest.log' -WorkingDirectory $RepoRoot
    $ev['generator_exit'] = $gen.ExitCode
    $ev['generator_log'] = $gen.Log
    $ev['manifest'] = $out
    if ($gen.ExitCode -ne 0) {
        $problems += "manifest generator exited $($gen.ExitCode)"
    } else {
        $m = Get-Content $out -Raw | ConvertFrom-Json
        $ev['manifest_sha256'] = (Get-Sha256 $out)
        $ev['manifest_schema_version'] = $m.schema_version
        $ev['execution_backend'] = $m.execution_backend

        # The generator asserts 743 roles / 161 weight objects internally, but
        # an assertion inside the thing under test is not evidence about the
        # real model. Count what actually landed in the document.
        $initializers = @($m.initializers.PSObject.Properties)
        $ev['initializer_count'] = $initializers.Count
        if ($initializers.Count -ne 743) { $problems += "expected 743 resolved initializers, found $($initializers.Count)" }

        $weights = @($m.weight_objects)
        $ev['weight_object_count'] = $weights.Count
        if ($weights.Count -ne 161) { $problems += "expected 161 weight objects, found $($weights.Count)" }

        $kinds = @{}
        foreach ($w in $weights) {
            $k = [string]$w.kind
            if (-not $kinds.ContainsKey($k)) { $kinds[$k] = 0 }
            $kinds[$k] = $kinds[$k] + 1
        }
        $kindSummary = [ordered]@{}
        foreach ($k in @($kinds.Keys | Sort-Object)) { $kindSummary[$k] = $kinds[$k] }
        $ev['weight_object_kinds'] = $kindSummary

        # Design Section 17: every MatMul descriptor has has_bias = false.
        # A missing field is not a pass -- a descriptor with no has_bias at all
        # would satisfy "none of them is true" while satisfying nothing else.
        $biasTrue = 0
        $biasMissing = 0
        foreach ($w in $weights) {
            if ([string]$w.kind -ne 'matmul') { continue }
            $names = @($w.descriptor.PSObject.Properties.Name)
            if ($names -notcontains 'has_bias') { $biasMissing++; continue }
            if ([bool]$w.descriptor.has_bias) { $biasTrue++ }
        }
        $ev['matmul_descriptors_with_has_bias_true'] = $biasTrue
        $ev['matmul_descriptors_missing_has_bias'] = $biasMissing
        if ($biasTrue -gt 0) { $problems += "$biasTrue MatMul descriptor(s) carry has_bias = true" }
        if ($biasMissing -gt 0) { $problems += "$biasMissing MatMul descriptor(s) have no has_bias field at all" }

        # Every role a weight object names must resolve to a real initializer.
        # This is what "all 161 weights load from ONNX-layout components" means
        # at the manifest level.
        $unresolved = @()
        $referenced = @{}
        foreach ($w in $weights) {
            foreach ($r in @($w.roles.PSObject.Properties)) {
                $target = [string]$r.Value
                $referenced[$target] = $true
                if (-not ($m.initializers.PSObject.Properties.Name -contains $target)) {
                    $unresolved += "$($w.name).$($r.Name) -> $target"
                }
            }
        }
        $ev['distinct_initializers_referenced'] = $referenced.Count
        $ev['unresolved_roles'] = $unresolved
        if ($unresolved.Count -gt 0) { $problems += "$($unresolved.Count) weight role(s) name an initializer the manifest does not resolve" }

        # FP16 scales, on the REAL model. A rejection here is a finding about
        # the published package, not a bug to work around.
        $scaleDtypes = @{}
        $scaleCount = 0
        foreach ($p in $initializers) {
            if ([string]$p.Value.role -ne 'matmul.scales') { continue }
            $scaleCount++
            $scaleDtypes[[string]$p.Value.dtype] = $true
        }
        $ev['scale_initializer_count'] = $scaleCount
        $ev['scale_dtypes'] = @($scaleDtypes.Keys | Sort-Object)
        if ($scaleCount -eq 0) { $problems += 'the manifest resolves no matmul.scales initializers at all' }
        $nonFp16 = @($scaleDtypes.Keys | Where-Object { $_ -ne 'float16' })
        if ($nonFp16.Count -gt 0) { $problems += "scales are not uniformly float16: $($nonFp16 -join ', ')" }

        # RoPE. Design 9.3's host strided-gather path exists because the source
        # row width is EXPECTED to exceed 48. Whether it actually does is a
        # measurement on the published model, so it is recorded either way
        # rather than assumed -- and if it does not, the gather degenerating to
        # a contiguous copy is a finding about this model, not a defect.
        foreach ($role in @('cos_cache', 'sin_cache')) {
            $h = $null
            if ($m.initializers.PSObject.Properties.Name -contains $role) { $h = $m.initializers.$role }
            if ($h) {
                $shape = @($h.shape)
                $ev["${role}_dtype"] = $h.dtype
                $ev["${role}_shape"] = $shape
                $ev["${role}_bytes"] = $h.length
                if ($shape.Count -ge 2) {
                    $rows = [int]$shape[0]
                    $cols = [int]$shape[1]
                    $ev["${role}_source_rows"] = $rows
                    $ev["${role}_source_row_width"] = $cols
                    $ev["${role}_strided_gather_needed"] = ($cols -gt 48)
                    if ($rows -lt 4096) { $problems += "$role has $rows rows, fewer than the required 4096" }
                    if ($cols -lt 48) { $problems += "$role has $cols columns, fewer than the required 48" }
                } else {
                    $problems += "$role has rank $($shape.Count); a [rows, width] source is required"
                }
            } else {
                $problems += "$role absent from the generated manifest"
            }
        }
        if ($ev.Contains('cos_cache_source_row_width') -and -not $ev['cos_cache_strided_gather_needed']) {
            $ev['rope_finding'] = "The published model's RoPE source row width is exactly 48, not wider. Design 9.3's host strided gather therefore degenerates to a contiguous copy for THIS model. The gather is still the right thing to ship -- the manifest records the actual width and a future package may be wider -- but no run in this acceptance exercised a stride greater than one."
        }

        # The Section 5.1 constants, read back from the manifest the generator
        # derived from the ONNX initializers.
        $expectedModel = [ordered]@{
            family = 'phi4'; group_size = 128; head_size = 128; hidden_size = 3072
            intermediate_size = 8192; kv_heads = 8; layers = 32; num_heads = 24
            rope_dim = 96; vocab_size = 200064
        }
        $modelBlock = [ordered]@{}
        foreach ($p in @($m.model.PSObject.Properties)) { $modelBlock[$p.Name] = $p.Value }
        $ev['model_block'] = $modelBlock
        foreach ($k in @($expectedModel.Keys)) {
            if ($modelBlock.Contains($k)) {
                if ([string]$modelBlock[$k] -ne [string]$expectedModel[$k]) {
                    $problems += "model.$k is $($modelBlock[$k]), expected $($expectedModel[$k])"
                }
            } else {
                $problems += "the manifest's model block has no $k"
            }
        }

        $ev['backend_max_seq'] = $(if ($m.PSObject.Properties.Name -contains 'backend') { $m.backend.max_seq } else { $null })
        if ($ev['backend_max_seq'] -ne 4096) { $problems += "backend.max_seq is $($ev['backend_max_seq']), not 4096" }

        # The hashes --full-hash just computed over the real files, against the
        # LFS oids the pinned revision publishes. This is the strongest form of
        # "the packaged model is the published model" available: an LFS oid IS
        # the sha256 of the file's content.
        $provPath5 = Join-Path $ModelDir 'provenance.json'
        $fileHashes = [ordered]@{}
        foreach ($p in @($m.files.PSObject.Properties)) {
            $fileHashes[$p.Name] = [ordered]@{ size = $p.Value.size; sha256 = $p.Value.sha256 }
        }
        $ev['manifest_file_hashes'] = $fileHashes
        if (Test-Path $provPath5) {
            $prov5 = Get-Content $provPath5 -Raw | ConvertFrom-Json
            foreach ($rec in @($prov5.upstream.git_files)) {
                if (-not ($fileHashes.Contains($rec.path))) { continue }
                if (-not ($rec.PSObject.Properties.Name -contains 'lfs') -or -not $rec.lfs) { continue }
                $got = [string]$fileHashes[$rec.path].sha256
                if ($got -and $got -ne [string]$rec.lfs.oid) {
                    $problems += "$($rec.path): the manifest's measured sha256 does not match the pinned revision's LFS oid"
                }
            }
        }

        # The regenerated manifest must match the committed overlay. If it does
        # not, the manifest that ships does not describe the model that ships.
        $committedManifest = Join-Path $OverlayDir 'corelib_phi4_manifest.json'
        if (Test-Path $committedManifest) {
            $ev['committed_manifest_sha256'] = (Get-Sha256 $committedManifest)
            $ev['regenerated_matches_committed'] = ($ev['manifest_sha256'] -eq $ev['committed_manifest_sha256'])
            if (-not $ev['regenerated_matches_committed']) {
                $problems += 'the manifest regenerated from the real model differs byte-for-byte from the committed overlay manifest'
            }
        } else {
            $problems += "no committed overlay manifest at $committedManifest to compare against"
        }
    }


    # Measured size and on-disk footprint in the catalog entry.
    $onDisk = 0
    Get-ChildItem -Path $ModelDir -File | ForEach-Object { $onDisk += $_.Length }
    $ev['model_dir_bytes_on_disk'] = $onDisk
    $ev['model_dir_gib_on_disk'] = [math]::Round($onDisk / 1GB, 2)
    $catalogStep = @($script:record['steps'] | Where-Object { (Get-Field $_ 'id') -eq '1' })
    if ($catalogStep.Count -gt 0) {
        $catEv = Get-Field $catalogStep[0] 'evidence'
        $ev['catalog_size'] = Get-Field $catEv 'catalog_size'
        $ev['catalog_footprint'] = Get-Field $catEv 'catalog_footprint'
        if ($null -ne $ev['catalog_size'] -and [int64]$ev['catalog_size'] -ne [int64]$ev['model_dir_bytes_on_disk']) {
            # Not a failure: the catalog `size` is the remote logical download
            # size plus the overlay bytes, which is what a user needs before
            # downloading. The on-disk footprint of the assembled directory is
            # a different number by construction. Both are recorded so nobody
            # has to guess which is which.
            $ev['catalog_size_vs_on_disk_note'] = "catalog size $($ev['catalog_size']) is the remote logical size plus overlay bytes; the assembled directory measures $($ev['model_dir_bytes_on_disk']) bytes on disk. These are different quantities and are expected to differ."
        }
    }

    if ($problems.Count -eq 0) {
        Add-StepResult -Id '5' -Title 'AIE4 package built from the published model' -Status 'met' `
            -Detail "manifest regenerated from the real model at HF e751fb68: $($ev['initializer_count']) initializers, $($ev['weight_object_count']) weight objects, every MatMul has_bias false, scales uniformly $($ev['scale_dtypes'] -join '/'), RoPE source $($ev['cos_cache_dtype']) $($ev['cos_cache_shape'] -join 'x'); byte-identical to the committed overlay manifest" `
            -Evidence $ev -Criteria @('C08', 'C10', 'C11', 'C02')
    } else {
        $ev['problems'] = $problems
        Add-StepResult -Id '5' -Title 'AIE4 package built from the published model' -Status 'not_met' `
            -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C08', 'C10', 'C11', 'C02')
    }
} else {
    Add-UnselectedStep -Id '5' -Title 'AIE4 package built from the published model'
}

# ---------------------------------------------------------------------------
# Console-driven CLI helper
#
# `flm run` reads keystrokes with ReadConsoleInput, which cannot see a
# redirected stdin, so it is driven through drive_flm_console.ps1 -- a separate
# PROCESS, because that driver has to AllocConsole() and would take this
# script's stdout with it. See that file for why this is the product's own
# interactive path and not a substitute for it.
# ---------------------------------------------------------------------------

function Invoke-FlmConsoleSession {
    param(
        [string[]]$Turns,
        [string]$Name,
        [int]$LoadTimeoutSec = 1800,
        [int]$TurnTimeoutSec = 1800,
        [int]$IdleSec = 6
    )
    $driver = Join-Path $script:suiteDir 'drive_flm_console.ps1'
    $outJson = Join-Path $script:artifactDir "$Name.json"
    $screen = Join-Path $script:artifactDir "$Name-screen.txt"
    if (-not (Test-Path $driver)) {
        return @{ Ok = $false; Reason = "console driver not found at $driver"; Json = $null; Path = $outJson; ExitCode = $null }
    }
    $psExe = (Get-Process -Id $PID).Path
    # Turns go through a FILE, not as repeated -Turns values.
    #
    # `powershell.exe -File driver.ps1 -Turns "a b" "c d"` binds only the
    # first value and then fails to place the second, exiting 1 with no
    # output. One turn works, two do not, so Step 7 passed while Step 8 was
    # reported as "the driver wrote no result" -- a failure mode that says
    # nothing about the product under test.
    $turnsFile = Join-Path $script:artifactDir "$Name-turns.txt"
    Set-Content -Path $turnsFile -Value ($Turns -join "`r`n") -Encoding utf8
    $callArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $driver,
              '-FlmExe', $FlmExe, '-ModelTag', $ModelTag,
              '-OutJson', $outJson, '-ScreenLog', $screen,
              '-LoadTimeoutSec', "$LoadTimeoutSec", '-TurnTimeoutSec', "$TurnTimeoutSec",
              '-IdleSec', "$IdleSec", '-TurnsFile', $turnsFile)
    # Invoked as a plain child of this script, with its output left alone.
    #
    # This is the one launch shape measured to work, and each rejected
    # alternative failed in the same way -- flm.exe spinning at 100% CPU on a
    # ReadConsoleInput it could never satisfy, against a screen buffer that
    # stayed empty:
    #
    #   & $psExe ... | Out-Null   -> the pipeline redirects the driver's
    #                               stdout, AllocConsole then leaves the
    #                               redirected handles alone, and flm inherits
    #                               a pipe instead of the console;
    #   Start-Process (plain)     -> the driver gets its own console FIRST,
    #                               and the later FreeConsole/AllocConsole
    #                               leaves its standard handles pointing at
    #                               the console it just abandoned.
    #
    # Leaving the driver's streams untouched lets its own FreeConsole /
    # AllocConsole / SetStdHandle sequence be the only thing that decides what
    # flm.exe inherits. The driver reports through -OutJson and -ScreenLog, so
    # nothing is lost by not capturing its output -- there is none to capture
    # once it has taken a console.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $code = $null
    try {
        & $psExe @callArgs
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prev }
    if (-not (Test-Path $outJson)) {
        return @{ Ok = $false; Reason = "console driver exited $code and wrote no result"; Json = $null; Path = $outJson; ExitCode = $code }
    }
    $doc = Get-Content $outJson -Raw | ConvertFrom-Json
    return @{ Ok = ($code -eq 0); Reason = $doc.error; Json = $doc; Path = $outJson; Screen = $screen; ExitCode = $code }
}

# Does a reply read like English a person wrote, rather than fluent-looking
# garbage?
#
# This is deliberately weak, and says so. The first thing a wrong weight map or
# a stale KV row produces is text that passes every mechanical check, which is
# why the brief requires the verbatim completion in the document for a HUMAN to
# judge. What this function can do is catch the failures that are not subtle --
# empty output, no letters, one token repeated forever -- so they are caught
# without waiting for a reader.
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
            $reasons += 'the reply is dominated by a repeated token'
        }
    }
    $letters = ([regex]::Matches($t, '[A-Za-z]')).Count
    if ($t.Length -gt 0 -and ($letters / [double]$t.Length) -lt 0.5) {
        $reasons += 'fewer than half the characters are letters'
    }
    if ($MustContain -and ($t -notmatch [regex]::Escape($MustContain))) {
        $reasons += "the reply does not contain the expected substring '$MustContain'"
    }
    return @{ Ok = ($reasons.Count -eq 0); Reasons = $reasons; Normalized = $t }
}

# ---------------------------------------------------------------------------
# Step 6: first real load
# ---------------------------------------------------------------------------

if (Test-Selected '6') {
    Write-Section 'Step 6: first real load of the AIE4 tag'
    $ev = [ordered]@{}
    $problems = @()
    # `/status`, not `/show`.
    #
    # Both exist in the REPL and they are not the same thing: `/show` prints
    # the model card (head_dim, layers, context length) and `/status` is the
    # one wired to AutoModel::show_profile, which is where the Phi-4 AIE4
    # block lives -- load breakdown, helper transition counts, dispatch and
    # synchronize counters, byte totals and the resolved corelib DLL. Reading
    # those from the product rather than from a test harness is the point of
    # this step, and `/show` would have returned a plausible-looking screen
    # with none of them on it.
    $session = Invoke-FlmConsoleSession -Turns @('/status') -Name 'step6-load'
    $ev['driver_result'] = $session.Path
    $ev['driver_exit'] = $session.ExitCode
    if (-not $session.Json) {
        Add-StepResult -Id '6' -Title 'First real load' -Status 'not_exercised' `
            -Detail $session.Reason -Evidence $ev -Criteria @('C13')
    } else {
        $ev['model_load_wall_seconds'] = $session.Json.load_seconds
        $ev['reached_repl'] = $session.Json.reached_prompt
        $profileText = ''
        if ($session.Json.turns -and @($session.Json.turns).Count -gt 0) {
            $profileText = @($session.Json.turns)[0].reply_raw
        }
        $ev['status_profile_verbatim'] = $profileText
        foreach ($pair in @(
            @('cold_model_load_ns', 'Cold model load:\s+(\d+) ns'),
            @('cold_weight_pack_ns', 'Cold weight pack:\s+(\d+) ns'),
            @('dispatches', 'Dispatches:\s+(\d+)'),
            @('synchronizations', 'Synchronizations:\s+(\d+)'),
            @('packed_weight_bytes', 'Packed weights:\s+(\d+) bytes'),
            @('mapped_source_bytes', 'Mapped source:\s+(\d+) bytes'),
            @('kv_bytes', 'KV storage:\s+(\d+) bytes'),
            @('scratch_bytes', 'Scratch storage:\s+(\d+) bytes'),
            @('append_threshold', 'Append threshold:\s+(\d+)'))) {
            if ($profileText -match $pair[1]) { $ev[$pair[0]] = [int64]$Matches[1] }
        }
        if ($profileText -match 'Helper transitions:\s+([0-9/]+)') { $ev['helper_transitions'] = $Matches[1] }
        if ($profileText -match 'Corelib DLL:\s+(.+)') { $ev['corelib_dll'] = $Matches[1].Trim() }
        if ($profileText -match 'Engine:\s+(\S+)') { $ev['engine'] = $Matches[1] }

        if (-not $session.Json.reached_prompt) { $problems += 'the REPL was never reached' }
        if ($ev.Contains('engine') -and $ev['engine'] -ne 'corelib_aie4') { $problems += "the tag selected engine '$($ev['engine'])', not corelib_aie4" }
        # Design 10.1: K/V are separate BF16 [8,4096,128] tensors per layer.
        # 32 layers x 2 x 8 x 4096 x 128 x 2 bytes = 536,870,912.
        if ($ev.Contains('kv_bytes') -and $ev['kv_bytes'] -ne 536870912) {
            $problems += "KV storage is $($ev['kv_bytes']) bytes, not the 536870912 that separate BF16 [8,4096,128] caches require"
        }
        if (-not $ev.Contains('cold_model_load_ns')) { $problems += '/status did not report a cold model load time' }

        # Padded K/N equality and the discovered row transitions are asserted
        # against the REAL library by test_phi4_hardware / test_real_corelib,
        # which compare the running kernel set to the fake's table. Named here,
        # not re-derived, and not claimed by this step.
        # NOT claimed by this step. Measured by Step 6b, in this same run.
        $ev['padding_and_transition_assertions'] = 'measured by step 6b in this same run; this step does not claim it'

        if ($problems.Count -eq 0) {
            Add-StepResult -Id '6' -Title 'First real load' -Status 'met' `
                -Detail "loaded in $($ev['model_load_wall_seconds']) s wall; engine corelib_aie4; helper transitions $($ev['helper_transitions']); KV $($ev['kv_bytes']) bytes" `
                -Evidence $ev -Criteria @('C01', 'C13', 'C29')
        } else {
            $ev['problems'] = $problems
            Add-StepResult -Id '6' -Title 'First real load' -Status 'not_met' `
                -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C01', 'C13', 'C29')
        }
    }
} else {
    Add-UnselectedStep -Id '6' -Title 'First real load'
}

# ---------------------------------------------------------------------------
# test_phi4_hardware, run here rather than pointed at
#
# Steps 3 and 6 used to note that the API-5 version gate and the padded-K/N and
# row-transition assertions "are asserted by test_phi4_hardware" and carry on
# to a `met` verdict. That is a citation, not a result: the binary was never
# invoked by this harness, so those two criteria rested on a run that had not
# happened. It runs once here and both steps read its output.
#
# It takes no arguments, requires FLM_AIE4_HARDWARE=1, holds an AIE4 device
# context for its whole life and calls cleanup() at the end -- so it must not
# overlap any other step that touches the device.
# ---------------------------------------------------------------------------

$script:hwRun = $null
function Invoke-HardwareChecks {
    if ($null -ne $script:hwRun) { return $script:hwRun }
    $exe = Join-Path $BuildDir 'Release/test_phi4_hardware.exe'
    if (-not (Test-Path $exe)) {
        $script:hwRun = @{ Available = $false; Reason = "test_phi4_hardware.exe not found at $exe"; Text = ''; ExitCode = $null; Log = $null }
        return $script:hwRun
    }
    $saved = $env:FLM_AIE4_HARDWARE
    $env:FLM_AIE4_HARDWARE = '1'
    try {
        $r = Invoke-Capture -Exe $exe -Arguments @() -LogName 'hardware-checks.log'
    } finally {
        if ($saved) { $env:FLM_AIE4_HARDWARE = $saved } else { Remove-Item Env:\FLM_AIE4_HARDWARE -ErrorAction SilentlyContinue }
    }
    # 77 is CTest's SKIP_RETURN_CODE. A skip is not a pass and must not be
    # read as one.
    $script:hwRun = @{
        Available = $true
        Skipped = ($r.ExitCode -eq 77)
        Reason = ''
        Text = $r.Text
        ExitCode = $r.ExitCode
        Log = $r.Log
    }
    return $script:hwRun
}

if (Test-Selected '3') {
    Write-Section 'Step 3b: the corelib version gate (API-5)'
    $ev = [ordered]@{}
    $hw = Invoke-HardwareChecks
    $ev['exit_code'] = $hw.ExitCode
    $ev['log'] = $hw.Log
    if (-not $hw.Available) {
        Add-StepResult -Id '3b' -Title 'Corelib version gate (API-5)' -Status 'not_exercised' `
            -Detail $hw.Reason -Evidence $ev -Criteria @()
    } elseif ($hw.Skipped) {
        Add-StepResult -Id '3b' -Title 'Corelib version gate (API-5)' -Status 'not_exercised' `
            -Detail "test_phi4_hardware skipped (exit 77); its own output says why" -Evidence $ev -Criteria @()
    } else {
        # CorelibApi::Load enforces the exact major.minor.patch match BEFORE
        # resolving any other symbol, so loading at all means the gate passed;
        # CheckVersionIdentity then restates it and prints both sides.
        $line = $null
        foreach ($l in ($hw.Text -split "`r?`n")) {
            if ($l -match 'runtime corelib version (\S+), compiled against (\S+)') {
                $line = $l.Trim()
                $ev['runtime_version'] = $Matches[1]
                $ev['compiled_against_version'] = $Matches[2]
            }
        }
        $ev['version_line_verbatim'] = $line
        $problems = @()
        if (-not $line) { $problems += 'test_phi4_hardware printed no version identity line' }
        elseif ($ev['runtime_version'] -ne $ev['compiled_against_version']) {
            $problems += "runtime corelib $($ev['runtime_version']) does not equal the compiled-against $($ev['compiled_against_version'])"
        }
        if ($hw.ExitCode -ne 0) { $problems += "test_phi4_hardware exited $($hw.ExitCode)" }
        if ($problems.Count -eq 0) {
            Add-StepResult -Id '3b' -Title 'Corelib version gate (API-5)' -Status 'met' `
                -Detail "the runtime corelib reports $($ev['runtime_version']), exactly matching the compiled-against version, and the gate runs inside CorelibApi::Load before any other symbol is resolved" `
                -Evidence $ev -Criteria @()
        } else {
            $ev['problems'] = $problems
            Add-StepResult -Id '3b' -Title 'Corelib version gate (API-5)' -Status 'not_met' `
                -Detail ($problems -join '; ') -Evidence $ev -Criteria @()
        }
    }
} else {
    Add-UnselectedStep -Id '3b' -Title 'Corelib version gate (API-5)'
}

if (Test-Selected '6') {
    Write-Section 'Step 6b: padded K/N and the discovered row transitions'
    $ev = [ordered]@{}
    $hw = Invoke-HardwareChecks
    $ev['exit_code'] = $hw.ExitCode
    $ev['log'] = $hw.Log
    if (-not $hw.Available) {
        Add-StepResult -Id '6b' -Title 'Padded K/N and row transitions against the real library' -Status 'not_exercised' `
            -Detail $hw.Reason -Evidence $ev -Criteria @()
    } elseif ($hw.Skipped) {
        Add-StepResult -Id '6b' -Title 'Padded K/N and row transitions against the real library' -Status 'not_exercised' `
            -Detail "test_phi4_hardware skipped (exit 77); its own output says why" -Evidence $ev -Criteria @()
    } else {
        $ev['transition_lines'] = @(($hw.Text -split "`r?`n") |
            Where-Object { $_ -match 'transitions,' } | ForEach-Object { $_.Trim() })
        $ev['boundaries_line'] = @(($hw.Text -split "`r?`n") |
            Where-Object { $_ -match 'helper boundaries agree with the running kernel set' })
        $problems = @()
        if ($hw.ExitCode -ne 0) { $problems += "test_phi4_hardware exited $($hw.ExitCode)" }
        if (@($ev['boundaries_line']).Count -eq 0) {
            $problems += 'test_phi4_hardware did not report that the helper boundaries agree with the running kernel set'
        }
        if (@($ev['transition_lines']).Count -eq 0) {
            $problems += 'test_phi4_hardware printed no row-transition summary'
        }
        if ($problems.Count -eq 0) {
            Add-StepResult -Id '6b' -Title 'Padded K/N and row transitions against the real library' -Status 'met' `
                -Detail "against the real library: every required MatMul padded K/N equals its logical K/N, rows = 1 stays unpadded, and the discovered row transitions agree with the running kernel set ($(@($ev['transition_lines']).Count) row uses reported)" `
                -Evidence $ev -Criteria @()
        } else {
            $ev['problems'] = $problems
            Add-StepResult -Id '6b' -Title 'Padded K/N and row transitions against the real library' -Status 'not_met' `
                -Detail ($problems -join '; ') -Evidence $ev -Criteria @()
        }
    }
} else {
    Add-UnselectedStep -Id '6b' -Title 'Padded K/N and row transitions against the real library'
}

# ---------------------------------------------------------------------------
# Step 7: first real generation
# ---------------------------------------------------------------------------

if (Test-Selected '7') {
    Write-Section 'Step 7: real generation through flm run'
    $ev = [ordered]@{}

    # Several prompts, not one, and each with an anchor a wrong answer cannot
    # accidentally satisfy.
    #
    # The acceptance question for this release is whether the shipped product,
    # run the way a user runs it, against the real published model, produces
    # COHERENT OUTPUT. One prompt cannot answer that: a single lucky
    # completion is exactly what a subtly wrong weight map produces. These
    # exercise recall, arithmetic, instruction-following, short-form
    # generation and a refusal-shaped question, so that a reader has enough
    # material to judge, which is what the acceptance actually rests on.
    #
    # `anchor` is a substring the correct answer must contain. It is a floor,
    # never a ceiling: a reply can contain it and still be nonsense, which is
    # why every completion is recorded verbatim in the document for a human.
    $probes = @(
        @{ name = 'capital';    prompt = 'What is the capital of France? Answer in one sentence.'; anchor = 'Paris' },
        @{ name = 'arithmetic'; prompt = 'What is 17 multiplied by 4? Reply with just the number.'; anchor = '68' },
        @{ name = 'instruction'; prompt = 'List exactly three primary colours, separated by commas, and nothing else.'; anchor = 'red' },
        @{ name = 'shortform';  prompt = 'In two sentences, explain what a compiler does.'; anchor = '' },
        @{ name = 'recall';     prompt = 'Who wrote the play Hamlet? Answer with the name only.'; anchor = 'Shakespeare' }
    )
    $results = @()
    $problems = @()
    $driverFailed = $false
    foreach ($probe in $probes) {
        $session = Invoke-FlmConsoleSession -Turns @($probe.prompt) -Name ("step7-" + $probe.name)
        if (-not $session.Json) {
            $problems += "$($probe.name): $($session.Reason)"
            $driverFailed = $true
            $results += [ordered]@{ name = $probe.name; prompt_verbatim = $probe.prompt; error = $session.Reason }
            continue
        }
        $reply = ''
        $seconds = $null
        if ($session.Json.turns -and @($session.Json.turns).Count -gt 0) {
            $reply = @($session.Json.turns)[0].reply_raw
            $seconds = @($session.Json.turns)[0].seconds
        }
        $judged = Test-LooksLikeEnglish -Text $reply -MustContain $probe.anchor
        $results += [ordered]@{
            name = $probe.name
            prompt_verbatim = $probe.prompt
            completion_verbatim = $reply
            expected_substring = $probe.anchor
            seconds = $seconds
            model_load_seconds = $session.Json.load_seconds
            coherent = $judged.Ok
            coherence_reasons = $judged.Reasons
            driver_result = $session.Path
        }
        if (-not $judged.Ok) { $problems += "$($probe.name): $($judged.Reasons -join ', ')" }
    }
    $ev['probes'] = $results
    $ev['probe_count'] = $results.Count
    $ev['coherent_count'] = @($results | Where-Object { $_.coherent }).Count
    # The first probe is also surfaced under the names the document's verbatim
    # section looks for, so the headline prompt and completion appear in full.
    if ($results.Count -gt 0 -and -not $driverFailed) {
        $ev['prompt_verbatim'] = $results[0].prompt_verbatim
        $ev['completion_verbatim'] = $results[0].completion_verbatim
    }
    $ev['judgement_scope'] = 'the mechanical check below only rules out the gross failures -- empty output, a repeated token, a missing factual anchor. Whether these completions read as a working model is a human judgement, and the verbatim text is recorded so it can be made.'

    if ($driverFailed) {
        $ev['problems'] = $problems
        Add-StepResult -Id '7' -Title 'Real generation through flm run' -Status 'not_exercised' `
            -Detail ("the console driver did not complete: " + ($problems -join '; ')) -Evidence $ev -Criteria @('C24')
    } elseif ($problems.Count -eq 0) {
        Add-StepResult -Id '7' -Title 'Real generation through flm run' -Status 'met' `
            -Detail "$($ev['coherent_count']) of $($ev['probe_count']) prompts produced coherent English carrying the expected answer; every prompt and completion is recorded verbatim for a human to judge" `
            -Evidence $ev -Criteria @('C24')
    } else {
        $ev['problems'] = $problems
        Add-StepResult -Id '7' -Title 'Real generation through flm run' -Status 'not_met' `
            -Detail ("$($ev['coherent_count']) of $($ev['probe_count']) prompts produced a coherent answer; failures: " + ($problems -join '; ')) `
            -Evidence $ev -Criteria @('C24')
    }

    # The golden token-ID half of Step 7 is a separate result, because it is a
    # separate claim.
    Write-Section 'Step 7b: golden token IDs through the product path'
    $ev2 = [ordered]@{}
    $ev2['why'] = @(
        'The Task 12 goldens are top-1 token IDs for an EXPLICIT input ID sequence,',
        'chosen so that a mismatch cannot be blamed on a tokenizer. The product CLI',
        'accepts text, renders a chat template around it and emits text; it exposes',
        'neither the input IDs nor the output IDs, and it has no control surface for',
        'the deterministic sampler settings the goldens were measured under.',
        'Comparing the two would be comparing different inputs, and a match or a',
        'mismatch would mean nothing about either.',
        'The golden comparison IS run, at the engine level, by',
        'run_hardware_suite.ps1 via test_phi4_e2e --continuation-route with',
        'src/tools/compare_phi4_corelib_e2e.py against phi4_expected_tokens.json.',
        'That result belongs to the hardware suite and is cited, not restated.')
    Add-StepResult -Id '7b' -Title 'Golden token IDs through the product CLI' -Status 'not_exercised' `
        -Detail 'the product CLI exposes neither token IDs nor sampler settings, so the Task 12 golden cannot be compared against it; the golden runs at the engine level in the hardware suite' `
        -Evidence $ev2 -Criteria @('C33')
} else {
    Add-UnselectedStep -Id '7' -Title 'First real generation'
    Add-UnselectedStep -Id '7b' -Title 'Golden token IDs through the product CLI'
}

# ---------------------------------------------------------------------------
# Step 8: real multi-turn, and what can and cannot be forced from the product
# ---------------------------------------------------------------------------

if (Test-Selected '8') {
    Write-Section 'Step 8: multi-turn conversation with a growing history'
    $ev = [ordered]@{}
    $problems = @()
    # Turns 2..4 each depend on an earlier turn, so a reply that ignores the
    # history is visible in the text rather than only in a counter.
    $turns = @(
        'My favourite colour is teal. Please remember it.',
        'I also have a cat named Mabel. What is my favourite colour?',
        'What is my cat called?',
        'In one line, name my favourite colour and my cat.')
    # /status AND /history after every turn.
    #
    # /status carries the continuation route and the logical KV position;
    # /history prints the conversation the frontend believes it has, token
    # count and all. The history probe is what turns "the model did not recall
    # turn 1" from an observation about text into an observation about state,
    # and an earlier version of this report cited that evidence from an ad-hoc
    # session that was not in the record. Evidence a claim depends on belongs
    # in the artifact.
    $interleaved = @()
    foreach ($t in $turns) { $interleaved += $t; $interleaved += '/status'; $interleaved += '/history' }
    $ev['turns_verbatim'] = $turns
    $session = Invoke-FlmConsoleSession -Turns $interleaved -Name 'step8-multiturn'
    $ev['driver_result'] = $session.Path
    $ev['driver_exit'] = $session.ExitCode
    if (-not $session.Json) {
        Add-StepResult -Id '8' -Title 'Real multi-turn conversation' -Status 'not_exercised' `
            -Detail $session.Reason -Evidence $ev -Criteria @('C20')
    } else {
        $all = @($session.Json.turns)
        $perTurn = @()
        # Three driver turns per conversational turn: the prompt, /status,
        # /history.
        for ($i = 0; $i -lt $all.Count; $i += 3) {
            $reply = $all[$i].reply_raw
            $profile = if (($i + 1) -lt $all.Count) { $all[$i + 1].reply_raw } else { '' }
            $historyText = if (($i + 2) -lt $all.Count) { $all[$i + 2].reply_raw } else { '' }
            $route = if ($profile -match 'Continuation route:\s+(\S+)') { $Matches[1] } else { $null }
            $contNs = if ($profile -match 'Continuation time:\s+(\d+) ns') { [int64]$Matches[1] } else { $null }
            $appendNs = if ($profile -match 'Warm append total:\s+(\d+) ns') { [int64]$Matches[1] } else { $null }
            $reNs = if ($profile -match 'Warm reprefill total:\s+(\d+) ns') { [int64]$Matches[1] } else { $null }
            # The logical KV position, printed by the product.
            #
            # `/status` renders "Total tokens: N (M)" where N is the
            # frontend's count and M is lm_engine->get_current_context_length(),
            # the engine's own logical position. An earlier version of this
            # harness recorded Step 8c as not exercised on the grounds that the
            # product never prints the position -- while this very field sat in
            # its own evidence for all four turns. It is parsed now.
            $frontendPos = $null
            $enginePos = $null
            if ($profile -match 'Total tokens:\s*(\d+)\s*\((\d+)\)') {
                $frontendPos = [int]$Matches[1]
                $enginePos = [int]$Matches[2]
            }
            $historyTokens = $null
            if ($historyText -match '(?m)^Tokens:\s*(\d+)') { $historyTokens = [int]$Matches[1] }
            $judged = Test-LooksLikeEnglish -Text $reply
            $idx = [int]($i / 3) + 1
            $perTurn += [ordered]@{
                index = $idx
                prompt = $all[$i].prompt
                reply_verbatim = $reply
                seconds = $all[$i].seconds
                status_profile_verbatim = $profile
                history_verbatim = $historyText
                history_tokens = $historyTokens
                frontend_position = $frontendPos
                engine_logical_position = $enginePos
                continuation_route = $route
                continuation_ns = $contNs
                warm_append_total_ns = $appendNs
                warm_reprefill_total_ns = $reNs
                coherent = $judged.Ok
                coherence_reasons = $judged.Reasons
            }
            if (-not $judged.Ok) { $problems += "turn $idx reply did not read as coherent: $($judged.Reasons -join ', ')" }
            if (-not $route) { $problems += "turn $idx did not report a continuation route" }
        }
        $ev['per_turn'] = $perTurn
        # Each turn must reference the earlier conversation.
        if ($perTurn.Count -ge 2 -and $perTurn[1].reply_verbatim -notmatch '(?i)teal') {
            $problems += 'turn 2 did not recall the colour established in turn 1'
        }
        if ($perTurn.Count -ge 3 -and $perTurn[2].reply_verbatim -notmatch '(?i)mabel') {
            $problems += 'turn 3 did not recall the name established in turn 2'
        }
        if ($perTurn.Count -ge 4) {
            if ($perTurn[3].reply_verbatim -notmatch '(?i)teal') { $problems += 'turn 4 did not recall the colour' }
            if ($perTurn[3].reply_verbatim -notmatch '(?i)mabel') { $problems += 'turn 4 did not recall the name' }
        }
        $ev['routes_selected'] = @($perTurn | ForEach-Object { $_.continuation_route })

        # The mechanism, not just the symptom.
        #
        # "The model did not recall turn 1" is an observation about text and a
        # reader is entitled to ask whether it is just a weak model. /history
        # answers that: it prints the conversation the frontend believes it
        # has. If the history were accumulating, its token count would grow
        # monotonically across turns. If each turn replaces it, the count
        # tracks the length of the current turn instead.
        $historyCounts = @($perTurn | ForEach-Object { $_.history_tokens })
        $ev['history_tokens_per_turn'] = $historyCounts
        $ev['history_verbatim_per_turn'] = @($perTurn | ForEach-Object { $_.history_verbatim })
        $growing = $true
        for ($k = 1; $k -lt $historyCounts.Count; $k++) {
            if ($null -ne $historyCounts[$k] -and $null -ne $historyCounts[$k-1] -and
                $historyCounts[$k] -le $historyCounts[$k-1]) { $growing = $false }
        }
        $ev['history_grows_monotonically'] = $growing
        if (@($historyCounts | Where-Object { $null -ne $_ }).Count -lt 2) {
            $problems += '/history reported fewer than two token counts, so history accumulation could not be measured'
        } elseif (-not $growing) {
            $problems += ("the conversation history is not accumulating: /history reported " +
                          ($historyCounts -join ' then ') +
                          ' tokens across the turns, so each turn replaces the previous history rather than appending to it')
        }

        if ($problems.Count -eq 0) {
            Add-StepResult -Id '8' -Title 'Real multi-turn conversation' -Status 'met' `
                -Detail "$($perTurn.Count) turns, each coherent and referencing earlier turns; per-turn continuation route recorded: $(@($perTurn | ForEach-Object { $_.continuation_route }) -join ', ')" `
                -Evidence $ev -Criteria @('C20', 'C21')
        } else {
            $ev['problems'] = $problems
            Add-StepResult -Id '8' -Title 'Real multi-turn conversation' -Status 'not_met' `
                -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C20', 'C21')
        }
    }

    Write-Section 'Step 8b: forcing the other continuation route on the same conversation'
    $ev2 = [ordered]@{}
    $ev2['why'] = @(
        'forced_continuation_route_ on the Phi4 frontend defaults to Automatic and',
        'has no setter reachable from any product surface: not a CLI flag, not a',
        '/set command, not a REST field, not an environment variable. Forcing a',
        'route is available only to the test harness.',
        'Both routes ARE forced and gated, at the engine level, by',
        'test_phi4_e2e --continuation-route force_append|force_reprefill under',
        'run_hardware_suite.ps1, and both are required to meet the same',
        'Section 15.3 gates there. What cannot be done is forcing the other route',
        'on THIS conversation through the product, which is what this step asked',
        'for, so it is recorded as not exercised rather than as met by proxy.')
    Add-StepResult -Id '8b' -Title 'Forcing the other continuation route from the product' -Status 'not_exercised' `
        -Detail 'the product exposes no control surface for the continuation route; both routes are forced and gated at the engine level in the hardware suite' `
        -Evidence $ev2 -Criteria @('C21')

    Write-Section 'Step 8c: the logical KV position the product reports'
    $ev3 = [ordered]@{}
    $problems3 = @()
    $step8 = @($script:record['steps'] | Where-Object { (Get-Field $_ 'id') -eq '8' })
    $turnsSeen = @()
    if ($step8.Count -gt 0) { $turnsSeen = @((Get-Field (Get-Field $step8[0] 'evidence') 'per_turn')) }
    if ($turnsSeen.Count -eq 0) {
        Add-StepResult -Id '8c' -Title 'Logical KV position per turn' -Status 'not_exercised' `
            -Detail 'Step 8 recorded no turns, so there is no position data to read' -Evidence $ev3 -Criteria @()
    } else {
        # `/status` prints "Total tokens: N (M)". N is the frontend's count and
        # M is lm_engine->get_current_context_length() -- the engine's own
        # logical KV position, which for this backend is impl_->position.
        # Both are the product's own output; nothing here is inferred.
        $rows = @()
        foreach ($t in $turnsSeen) {
            $rows += [ordered]@{
                turn = (Get-Field $t 'index')
                frontend_position = (Get-Field $t 'frontend_position')
                engine_logical_position = (Get-Field $t 'engine_logical_position')
                history_tokens = (Get-Field $t 'history_tokens')
                continuation_route = (Get-Field $t 'continuation_route')
            }
        }
        # NOT `per_turn`: Step 8 already uses that name for conversational
        # turns, and the document's verbatim renderer keys off it. Two
        # different row shapes under one field name crashed the renderer on
        # `.index`, and because the renderer ran last the run still exited
        # with its real verdict while quietly leaving the previous document
        # in place -- a stale artifact contradicting the fresh record.
        $ev3['positions_per_turn'] = $rows
        $missing = @($rows | Where-Object { $null -eq $_.engine_logical_position })
        $ev3['turns_without_a_position'] = $missing.Count
        if ($missing.Count -gt 0) {
            $problems3 += "$($missing.Count) turn(s) reported no logical position"
        }
        # 1. Frontend and engine must agree at every turn.
        $disagree = @($rows | Where-Object {
            $null -ne $_.engine_logical_position -and $_.frontend_position -ne $_.engine_logical_position })
        $ev3['turns_where_frontend_and_engine_disagree'] = @($disagree | ForEach-Object { $_.turn })
        if ($disagree.Count -gt 0) {
            $problems3 += "the frontend and engine positions disagree on turn(s) $(@($disagree | ForEach-Object { $_.turn }) -join ', ')"
        }
        # 2. The position must equal what the frontend actually rendered, and
        #    /history reports exactly that. If they match, the position tracks
        #    the rendered prompt faithfully -- which is the invariant. Whether
        #    the rendered prompt is the whole CONVERSATION is Step 8's
        #    question, and a different one.
        $posVsHistory = @()
        foreach ($r in $rows) {
            if ($null -eq $r.history_tokens -or $null -eq $r.engine_logical_position) { continue }
            $posVsHistory += [ordered]@{
                turn = $r.turn
                engine_position = $r.engine_logical_position
                history_tokens = $r.history_tokens
                equal = ($r.engine_logical_position -eq $r.history_tokens)
            }
        }
        $ev3['position_vs_rendered_history'] = $posVsHistory
        $mismatch = @($posVsHistory | Where-Object { -not $_.equal })
        if ($posVsHistory.Count -eq 0) {
            $problems3 += '/history reported no token counts, so the position could not be compared against the rendered history'
        } elseif ($mismatch.Count -gt 0) {
            $problems3 += "the logical position does not equal the rendered history length on turn(s) $(@($mismatch | ForEach-Object { $_.turn }) -join ', ')"
        }
        # 3. Recorded, not gated: is the position monotonic across turns? It
        #    should be for a growing conversation, and it is not here -- but
        #    that is Step 8's defect showing through, not a separate failure of
        #    this invariant. Failing it twice would double-count one bug.
        $sequence = @($rows | ForEach-Object { $_.engine_logical_position })
        $ev3['position_sequence'] = $sequence
        $monotonic = $true
        for ($k = 1; $k -lt $sequence.Count; $k++) {
            if ($null -ne $sequence[$k] -and $null -ne $sequence[$k-1] -and $sequence[$k] -lt $sequence[$k-1]) { $monotonic = $false }
        }
        $ev3['position_monotonic_across_turns'] = $monotonic
        if (-not $monotonic) {
            $ev3['non_monotonic_note'] = "the position falls between turns ($($sequence -join ' -> ')) instead of growing. That is the Step 8 history defect visible in a second, independent measurement: the frontend renders only the current turn, so the position it reaches is the length of that turn alone. It is recorded here and gated in Step 8, not gated twice."
        }

        if ($problems3.Count -eq 0) {
            Add-StepResult -Id '8c' -Title 'Logical KV position per turn' -Status 'met' `
                -Detail ("the product reports the engine's logical position every turn and it equals both the frontend count and the rendered history length at each of $($rows.Count) turns (sequence $($sequence -join ', ')); it is NOT monotonic, which is the Step 8 defect seen a second way") `
                -Evidence $ev3 -Criteria @()
        } else {
            $ev3['problems'] = $problems3
            Add-StepResult -Id '8c' -Title 'Logical KV position per turn' -Status 'not_met' `
                -Detail ($problems3 -join '; ') -Evidence $ev3 -Criteria @()
        }
    }
} else {
    Add-UnselectedStep -Id '8' -Title 'Real multi-turn conversation'
    Add-UnselectedStep -Id '8b' -Title 'Forcing the other continuation route from the product'
    Add-UnselectedStep -Id '8c' -Title 'Logical KV position per turn'
}

# ---------------------------------------------------------------------------
# HTTP helpers for Steps 9-11
# ---------------------------------------------------------------------------

$script:server = $null

function Stop-FlmServer {
    if ($script:server -and -not $script:server.HasExited) {
        Stop-Process -Id $script:server.Id -Force -ErrorAction SilentlyContinue
    }
    Get-Process flm -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 4
    $script:server = $null
}

function Start-FlmServer {
    Stop-FlmServer
    $out = Join-Path $script:artifactDir 'flm-serve.log'
    $err = Join-Path $script:artifactDir 'flm-serve.err'
    $script:server = Start-Process -FilePath $FlmExe `
        -ArgumentList @('serve', $ModelTag, '-p', "$Port") `
        -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    for ($i = 0; $i -lt 200; $i++) {
        Start-Sleep -Seconds 3
        if ($script:server.HasExited) { throw "flm serve exited during startup with $($script:server.ExitCode)" }
        $probe = & curl.exe -s -o NUL -w '%{http_code}' --max-time 10 "http://127.0.0.1:$Port/api/tags"
        if ($probe -eq '200') { return }
    }
    throw 'flm serve never became ready'
}

# curl.exe rather than Invoke-WebRequest: Windows PowerShell 5.1 throws on any
# non-2xx, and the 400s are half of what these steps are checking.
function Invoke-Rest {
    param([string]$Path, [string]$Body, [string]$Name, [int]$TimeoutSec = 1800)
    $bodyFile = Join-Path $script:artifactDir "$Name.req.json"
    $respFile = Join-Path $script:artifactDir "$Name.resp.json"
    Remove-Item $respFile -Force -ErrorAction SilentlyContinue
    Set-Content -Path $bodyFile -Value $Body -Encoding utf8
    $t0 = Get-Date
    $code = & curl.exe -s -o $respFile -w '%{http_code}' `
        -X POST "http://127.0.0.1:$Port$Path" `
        -H 'Content-Type: application/json' `
        --data-binary "@$bodyFile" --max-time $TimeoutSec
    $elapsed = ((Get-Date) - $t0).TotalSeconds
    $text = if (Test-Path $respFile) { Get-Content $respFile -Raw } else { '' }
    if ($null -eq $text) { $text = '' }
    $died = ($script:server -and $script:server.HasExited)
    return @{
        Code = $code; Body = $text; Seconds = [math]::Round($elapsed, 3)
        RequestFile = $bodyFile; ResponseFile = $respFile
        ServerDied = $died
        ServerExit = $(if ($died) { '0x{0:X}' -f $script:server.ExitCode } else { '' })
    }
}

# ---------------------------------------------------------------------------
# Step 9: real server, all four generation endpoints, plus concurrency
# ---------------------------------------------------------------------------

if (Test-Selected '9') {
    Write-Section 'Step 9: flm serve, all four generation endpoints'
    $ev = [ordered]@{}
    $problems = @()
    $endpointScript = Join-Path $script:suiteDir 'run_server_endpoints.ps1'
    $ev['endpoint_script'] = $endpointScript
    if (-not (Test-Path $endpointScript)) {
        Add-StepResult -Id '9' -Title 'Server: four generation endpoints' -Status 'not_exercised' `
            -Detail "run_server_endpoints.ps1 not found at $endpointScript" -Evidence $ev -Criteria @('C24', 'C28')
    } else {
        Stop-FlmServer
        $psExe = (Get-Process -Id $PID).Path
        $run = Invoke-Capture -Exe $psExe -Arguments @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $endpointScript,
            '-FlmExe', $FlmExe, '-ModelTag', $ModelTag, '-Port', "$Port",
            '-WorkDir', $script:artifactDir) -LogName 'step9-endpoints.log'
        $ev['exit'] = $run.ExitCode
        $ev['log'] = $run.Log
        $ev['transcript'] = $run.Text
        foreach ($k in @('STEP7_TOTAL', 'STEP7_FAILED', 'STEP7_SERVER_DEATHS')) {
            if ($run.Text -match "$k=(\d+)") { $ev[$k.ToLowerInvariant()] = [int]$Matches[1] }
        }
        if (-not $ev.Contains('step7_total')) { $problems += 'the endpoint script produced no summary line, so its result cannot be read' }
        else {
            if ($ev['step7_total'] -lt 10) { $problems += "only $($ev['step7_total']) endpoint cases ran; the four endpoints need at least 10" }
            if ($ev['step7_failed'] -gt 0) { $problems += "$($ev['step7_failed']) endpoint case(s) failed" }
            if ($ev['step7_server_deaths'] -gt 0) { $problems += "the server terminated during $($ev['step7_server_deaths']) case(s)" }
        }
        if ($problems.Count -eq 0) {
            Add-StepResult -Id '9' -Title 'Server: four generation endpoints' -Status 'met' `
                -Detail "$($ev['step7_total']) cases, 0 failures, 0 server deaths: each endpoint admits a default request with no generation-limit field and refuses an explicit over-limit request with HTTP 400 naming the active context cap" `
                -Evidence $ev -Criteria @('C24', 'C28', 'C27')
        } else {
            $ev['problems'] = $problems
            Add-StepResult -Id '9' -Title 'Server: four generation endpoints' -Status 'not_met' `
                -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C24', 'C28', 'C27')
        }
    }

    Write-Section 'Step 9b: two concurrent clients, two conversations'
    $ev2 = [ordered]@{}
    $problems2 = @()
    try {
        Start-FlmServer
        # Two distinct conversations, each with an answer only its own history
        # can produce. If the single mutable KV session leaked between them,
        # the wrong fact comes back -- which a status-code check would miss
        # entirely.
        $bodyA = '{"model":"' + $ModelTag + '","messages":[' +
            '{"role":"user","content":"Remember this codeword: ELEPHANT. Reply OK."},' +
            '{"role":"assistant","content":"OK"},' +
            '{"role":"user","content":"What was the codeword? Answer with the single word."}],"stream":false}'
        $bodyB = '{"model":"' + $ModelTag + '","messages":[' +
            '{"role":"user","content":"Remember this codeword: TROMBONE. Reply OK."},' +
            '{"role":"assistant","content":"OK"},' +
            '{"role":"user","content":"What was the codeword? Answer with the single word."}],"stream":false}'
        $fa = Join-Path $script:artifactDir 'step9b-a.req.json'
        $fb = Join-Path $script:artifactDir 'step9b-b.req.json'
        $ra = Join-Path $script:artifactDir 'step9b-a.resp.json'
        $rb = Join-Path $script:artifactDir 'step9b-b.resp.json'
        Set-Content -Path $fa -Value $bodyA -Encoding utf8
        Set-Content -Path $fb -Value $bodyB -Encoding utf8
        $t0 = Get-Date
        $jobA = Start-Process -FilePath 'curl.exe' -PassThru -NoNewWindow -ArgumentList @(
            '-s', '-o', $ra, '-w', '%{http_code}\n%{time_total}',
            '-X', 'POST', "http://127.0.0.1:$Port/api/chat",
            '-H', 'Content-Type: application/json', '--data-binary', "@$fa",
            '--max-time', '900') -RedirectStandardOutput (Join-Path $script:artifactDir 'step9b-a.meta')
        $jobB = Start-Process -FilePath 'curl.exe' -PassThru -NoNewWindow -ArgumentList @(
            '-s', '-o', $rb, '-w', '%{http_code}\n%{time_total}',
            '-X', 'POST', "http://127.0.0.1:$Port/api/chat",
            '-H', 'Content-Type: application/json', '--data-binary', "@$fb",
            '--max-time', '900') -RedirectStandardOutput (Join-Path $script:artifactDir 'step9b-b.meta')
        $jobA.WaitForExit(); $jobB.WaitForExit()
        $wall = ((Get-Date) - $t0).TotalSeconds
        $metaA = @(Get-Content (Join-Path $script:artifactDir 'step9b-a.meta') -ErrorAction SilentlyContinue)
        $metaB = @(Get-Content (Join-Path $script:artifactDir 'step9b-b.meta') -ErrorAction SilentlyContinue)
        $ev2['a_http'] = $(if ($metaA.Count -gt 0) { $metaA[0] } else { '' })
        $ev2['b_http'] = $(if ($metaB.Count -gt 0) { $metaB[0] } else { '' })
        $ev2['a_seconds'] = $(if ($metaA.Count -gt 1) { [double]$metaA[1] } else { $null })
        $ev2['b_seconds'] = $(if ($metaB.Count -gt 1) { [double]$metaB[1] } else { $null })
        $ev2['wall_seconds'] = [math]::Round($wall, 3)
        $textA = if (Test-Path $ra) { Get-Content $ra -Raw } else { '' }
        $textB = if (Test-Path $rb) { Get-Content $rb -Raw } else { '' }
        $ev2['a_response_verbatim'] = $textA
        $ev2['b_response_verbatim'] = $textB
        $ev2['server_died'] = ($script:server.HasExited)
        if ($ev2['a_http'] -ne '200') { $problems2 += "concurrent client A returned HTTP $($ev2['a_http'])" }
        if ($ev2['b_http'] -ne '200') { $problems2 += "concurrent client B returned HTTP $($ev2['b_http'])" }
        if ($script:server.HasExited) { $problems2 += "the server terminated during the concurrent run (exit 0x{0:X})" -f $script:server.ExitCode }

        # ISOLATION is the gate. RECALL is reported and never assumed.
        #
        # These are two different claims and an earlier version of this step
        # collapsed them, hard-coding "each recalled only its own secret" into
        # the success message while the run had already recorded
        # b_recalled_own_secret = false. The reply was a safety refusal -- the
        # question was phrased as "what is my password word", which the model
        # declines -- so the recall half simply did not happen, and the record
        # asserted that it had. The prompt is now a neutral codeword so that
        # recall is answerable, but the important change is structural: the
        # gate is leakage, the recall result is computed and reported whatever
        # it is, and the success message is BUILT from the measurements rather
        # than written next to them.
        $leakAintoB = ($textB -match '(?i)ELEPHANT')
        $leakBintoA = ($textA -match '(?i)TROMBONE')
        $ev2['a_recalled_own_secret'] = ($textA -match '(?i)ELEPHANT')
        $ev2['b_recalled_own_secret'] = ($textB -match '(?i)TROMBONE')
        $ev2['a_contains_b_secret'] = $leakBintoA
        $ev2['b_contains_a_secret'] = $leakAintoB
        if ($leakBintoA) { $problems2 += 'conversation A''s reply contains conversation B''s codeword: KV state leaked between requests' }
        if ($leakAintoB) { $problems2 += 'conversation B''s reply contains conversation A''s codeword: KV state leaked between requests' }

        # A conversation that answered nothing cannot demonstrate isolation.
        #
        # If NEITHER reply recalls its own codeword, "no leakage" is vacuous --
        # two refusals contain no secrets and would pass a leak check while
        # showing nothing about whether the histories were kept apart. At least
        # one side has to have actually used its own history.
        $recallCount = @(@($ev2['a_recalled_own_secret'], $ev2['b_recalled_own_secret']) | Where-Object { $_ }).Count
        $ev2['own_recall_count'] = $recallCount
        if ($recallCount -eq 0) {
            $problems2 += 'neither reply recalled its own codeword, so the absence of the other conversation''s codeword shows nothing about isolation'
        }
        $ev2['isolation_basis'] = "client A recalled its own codeword: $($ev2['a_recalled_own_secret']); client B: $($ev2['b_recalled_own_secret']); neither reply contained the other's. Isolation is demonstrated by at least one side using its own history while showing no trace of the other's."

        # Serialization is NOT established by these numbers, and this step no
        # longer pretends otherwise.
        #
        # The previous version reasoned that serialized execution predicts the
        # wall clock tracking the later completion. Its own measurements matched
        # neither that prediction nor its opposite -- wall 10.56 s against
        # max(a,b) 8.19 s and a+b 12.95 s -- because the two requests do
        # different amounts of work and each client's timer includes the time it
        # spent queued. Two unequal requests cannot separate "serialized" from
        # "interleaved" by wall clock at all.
        #
        # Timings are still recorded, because they are real and someone may want
        # them. What is removed is the inference. Measuring serialization
        # properly needs equal-work requests compared against a single-request
        # baseline, or instrumentation inside the server; Step 9c below does the
        # former.
        if ($null -ne $ev2['a_seconds'] -and $null -ne $ev2['b_seconds']) {
            $sum = $ev2['a_seconds'] + $ev2['b_seconds']
            $max = [math]::Max($ev2['a_seconds'], $ev2['b_seconds'])
            $ev2['timings_note'] = "wall $($ev2['wall_seconds'])s, max(a,b) $([math]::Round($max,3))s, a+b $([math]::Round($sum,3))s. These do not distinguish serialized from interleaved execution: the two requests are not equal work and each client's timer includes its queue wait. Recorded, not interpreted."
        }
    } catch {
        $problems2 += "concurrent client run failed: $($_.Exception.Message)"
    } finally {
        Stop-FlmServer
    }
    if ($problems2.Count -eq 0) {
        Add-StepResult -Id '9b' -Title 'Two concurrent conversations stay isolated' -Status 'met' `
            -Detail ("both conversations returned HTTP 200 with the server alive; own-codeword recall was A=$($ev2['a_recalled_own_secret']) B=$($ev2['b_recalled_own_secret']) ($recallCount of 2), and neither reply contained the other conversation's codeword") `
            -Evidence $ev2 -Criteria @()
    } else {
        $ev2['problems'] = $problems2
        Add-StepResult -Id '9b' -Title 'Two concurrent conversations stay isolated' -Status 'not_met' `
            -Detail ($problems2 -join '; ') -Evidence $ev2 -Criteria @()
    }

    # -----------------------------------------------------------------------
    # Step 9c: serialization, measured against per-request baselines
    # -----------------------------------------------------------------------
    Write-Section 'Step 9c: are complete requests serialized?'
    $ev3 = [ordered]@{}
    $problems3 = @()
    $verdict = 'inconclusive'
    try {
        Start-FlmServer
        # A WARM-UP FIRST, and two DIFFERENT prompts.
        #
        # The first attempt compared two identical concurrent requests against
        # a single-request baseline and reported 1.077x -- "not consistent with
        # serialization". Both of its ingredients were wrong.
        #
        # The baseline was the FIRST request after the server started, so it
        # carried warm-up the concurrent pair did not: 6.241 s against the
        # 2.03 s the same shape of request takes warm. That alone accounts for
        # the ratio, and the step was reporting a cold-start artifact as a
        # concurrency finding. Hence the discarded warm-up below.
        #
        # I first assumed the cause was prefix reuse on the identical second
        # prompt, and that guess was also wrong -- worth recording, because it
        # is the kind of plausible story that survives review. Measured here:
        # an immediate repeat of the same prompt takes 0.996x the first, no
        # saving at all. A prefix hit needs matched == history_size, and a
        # repeated single-turn prompt is SHORTER than the history it is
        # compared against (which holds the previous prompt AND its reply), so
        # it never qualifies. Identical prompts were still the wrong choice,
        # but for the ordinary reason that they make sum and max harder to
        # separate, not for that one.
        #
        # Two prompts sharing no prefix beyond the chat template both pay a
        # full prefill, so:
        #     serialized  -> wall is close to t1 + t2
        #     parallel    -> wall is close to max(t1, t2)
        # Those are far apart, and each is measured on this box in this run
        # rather than assumed.
        $p1 = 'Name three rivers in Europe and say which is longest.'
        $p2 = 'Describe the water cycle briefly, in your own words.'
        $mk = { param($p) '{"model":"' + $ModelTag + '","prompt":"' + $p + '","stream":false,"max_tokens":48}' }

        # Warm first, and discard it: the first request after a load pays costs
        # that have nothing to do with concurrency.
        [void](Invoke-Rest -Path '/api/generate' -Body (& $mk 'Say hello.') -Name 'step9c-warm')

        $r1 = Invoke-Rest -Path '/api/generate' -Body (& $mk $p1) -Name 'step9c-t1'
        $r2 = Invoke-Rest -Path '/api/generate' -Body (& $mk $p2) -Name 'step9c-t2'
        $ev3['t1_http'] = $r1.Code; $ev3['t1_seconds'] = $r1.Seconds
        $ev3['t2_http'] = $r2.Code; $ev3['t2_seconds'] = $r2.Seconds
        if ($r1.Code -ne '200' -or $r2.Code -ne '200') {
            $problems3 += "the sequential baselines returned HTTP $($r1.Code) and $($r2.Code)"
        }

        # Recorded, not gated: does an immediate repeat of p1 come back much
        # faster? If it does, token_history survived the request and the repeat
        # took the append route with a zero-token suffix. That is the
        # observation that explains why the identical-prompt design failed, and
        # it is the only evidence in this run bearing on whether the append
        # route is reachable through the server at all.
        $r1again = Invoke-Rest -Path '/api/generate' -Body (& $mk $p1) -Name 'step9c-t1-repeat'
        $ev3['t1_repeat_seconds'] = $r1again.Seconds
        if ($r1.Seconds -gt 0) {
            $ev3['repeat_over_first_ratio'] = [math]::Round($r1again.Seconds / $r1.Seconds, 3)
            $ev3['prefix_reuse_note'] = "an immediate repeat of the same prompt took $($ev3['repeat_over_first_ratio'])x the original. rest_handler does not clear_context() on success, so token_history survives and a prefix hit is architecturally possible on the server -- but a ratio at or near 1 means no prefill was skipped here, which is what the prefix rule predicts: a hit needs matched == history_size, and a repeated single-turn prompt is shorter than the stored history (previous prompt AND reply), so it cannot match. A genuinely GROWING chat conversation is the case that could hit, and this run does not exercise it. RECORDED, NOT ASSERTED: the REST surface does not report the continuation route."
        }

        $f1 = Join-Path $script:artifactDir 'step9c-p1.req.json'
        $f2 = Join-Path $script:artifactDir 'step9c-p2.req.json'
        Set-Content -Path $f1 -Value (& $mk $p1) -Encoding utf8
        Set-Content -Path $f2 -Value (& $mk $p2) -Encoding utf8
        $m1 = Join-Path $script:artifactDir 'step9c-c1.meta'
        $m2 = Join-Path $script:artifactDir 'step9c-c2.meta'
        $o1 = Join-Path $script:artifactDir 'step9c-c1.resp.json'
        $o2 = Join-Path $script:artifactDir 'step9c-c2.resp.json'
        $t0 = Get-Date
        $c1 = Start-Process -FilePath 'curl.exe' -PassThru -NoNewWindow -ArgumentList @(
            '-s', '-o', $o1, '-w', '%{http_code}', '-X', 'POST',
            "http://127.0.0.1:$Port/api/generate", '-H', 'Content-Type: application/json',
            '--data-binary', "@$f1", '--max-time', '900') -RedirectStandardOutput $m1
        $c2 = Start-Process -FilePath 'curl.exe' -PassThru -NoNewWindow -ArgumentList @(
            '-s', '-o', $o2, '-w', '%{http_code}', '-X', 'POST',
            "http://127.0.0.1:$Port/api/generate", '-H', 'Content-Type: application/json',
            '--data-binary', "@$f2", '--max-time', '900') -RedirectStandardOutput $m2
        $c1.WaitForExit(); $c2.WaitForExit()
        $ev3['concurrent_wall_seconds'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 3)
        # Success is judged from the RESPONSE BODY, not from curl's -w status.
        #
        # `-w '%{http_code}'` under Start-Process writes "200000" here for both
        # clients -- reproducibly, and not against an ordinary server, where
        # the same call shape writes a clean "200". Whatever produces the
        # trailing "000" (curl reports 000 when it has no HTTP response), the
        # status text is not trustworthy on this path, and an earlier check of
        # `-notmatch '200'` accepted it silently. The body is the better
        # evidence regardless: a completed generation contains "response",
        # which no failure mode produces. run_server_endpoints.ps1 matches on
        # the body for the same reason.
        $ev3['c1_status_text_raw'] = ((Get-Content $m1 -Raw -ErrorAction SilentlyContinue) -replace '\s', '')
        $ev3['c2_status_text_raw'] = ((Get-Content $m2 -Raw -ErrorAction SilentlyContinue) -replace '\s', '')
        $ev3['status_text_note'] = 'curl -w %{http_code} under Start-Process is unreliable on this path (it writes 200000); the response bodies below are what this step judges'
        $b1 = if (Test-Path $o1) { Get-Content $o1 -Raw } else { '' }
        $b2 = if (Test-Path $o2) { Get-Content $o2 -Raw } else { '' }
        $ev3['c1_body_head'] = $b1.Substring(0, [math]::Min(300, $b1.Length))
        $ev3['c2_body_head'] = $b2.Substring(0, [math]::Min(300, $b2.Length))
        $ev3['c1_completed'] = ($b1 -match '"response"')
        $ev3['c2_completed'] = ($b2 -match '"response"')
        if (-not $ev3['c1_completed'] -or -not $ev3['c2_completed']) {
            $problems3 += "a concurrent request did not return a completed generation (c1 ok = $($ev3['c1_completed']), c2 ok = $($ev3['c2_completed']))"
        }
        if ($script:server.HasExited) { $problems3 += 'the server terminated during the concurrent run' }

        $sum = $ev3['t1_seconds'] + $ev3['t2_seconds']
        $max = [math]::Max($ev3['t1_seconds'], $ev3['t2_seconds'])
        $wall = $ev3['concurrent_wall_seconds']
        $ev3['sum_of_baselines'] = [math]::Round($sum, 3)
        $ev3['max_of_baselines'] = [math]::Round($max, 3)
        # The two predictions have to be far enough apart to choose between.
        # If t1 and t2 are too close in size, sum and max are not separable and
        # the whole comparison is meaningless -- say so rather than pick one.
        $separation = $(if ($max -gt 0) { $sum / $max } else { 0 })
        $ev3['prediction_separation'] = [math]::Round($separation, 3)
        if ($separation -lt 1.4) {
            $verdict = 'inconclusive'
            $ev3['why_inconclusive'] = "the two baselines ($($ev3['t1_seconds'])s and $($ev3['t2_seconds'])s) are too unequal for sum and max to be distinguishable predictions"
        } else {
            $dSum = [math]::Abs($wall - $sum) / $sum
            $dMax = [math]::Abs($wall - $max) / $max
            $ev3['distance_to_serialized_prediction'] = [math]::Round($dSum, 3)
            $ev3['distance_to_parallel_prediction'] = [math]::Round($dMax, 3)
            if ($dSum -le 0.25 -and $dSum -lt $dMax) { $verdict = 'serialized' }
            elseif ($dMax -le 0.25 -and $dMax -lt $dSum) { $verdict = 'parallel' }
            else { $verdict = 'inconclusive'; $ev3['why_inconclusive'] = 'the wall clock is not close to either prediction' }
        }
        $ev3['verdict'] = $verdict
        $ev3['prediction'] = "serialized predicts a wall clock near t1+t2 = $($ev3['sum_of_baselines'])s; parallel predicts near max(t1,t2) = $($ev3['max_of_baselines'])s; measured $wall s"
    } catch {
        $problems3 += "serialization probe failed: $($_.Exception.Message)"
        $ev3['error'] = $_.Exception.Message
    } finally {
        Stop-FlmServer
    }
    if ($problems3.Count -gt 0) {
        $ev3['problems'] = $problems3
        Add-StepResult -Id '9c' -Title 'Complete requests are serialized' -Status 'not_met' `
            -Detail ($problems3 -join '; ') -Evidence $ev3 -Criteria @()
    } elseif ($verdict -eq 'serialized') {
        Add-StepResult -Id '9c' -Title 'Complete requests are serialized' -Status 'met' `
            -Detail "two distinct requests run concurrently took $($ev3['concurrent_wall_seconds'])s against baselines summing to $($ev3['sum_of_baselines'])s (parallel would predict $($ev3['max_of_baselines'])s), which is the serialized prediction" `
            -Evidence $ev3 -Criteria @()
    } elseif ($verdict -eq 'parallel') {
        Add-StepResult -Id '9c' -Title 'Complete requests are serialized' -Status 'not_met' `
            -Detail "two distinct requests run concurrently took $($ev3['concurrent_wall_seconds'])s, close to max(t1,t2) = $($ev3['max_of_baselines'])s rather than the serialized t1+t2 = $($ev3['sum_of_baselines'])s: they appear to have overlapped" `
            -Evidence $ev3 -Criteria @()
    } else {
        Add-StepResult -Id '9c' -Title 'Complete requests are serialized' -Status 'not_exercised' `
            -Detail "the timing comparison was inconclusive: $($ev3['why_inconclusive']). Wall $($ev3['concurrent_wall_seconds'])s against t1+t2 $($ev3['sum_of_baselines'])s and max(t1,t2) $($ev3['max_of_baselines'])s" `
            -Evidence $ev3 -Criteria @()
    }
} else {
    Add-UnselectedStep -Id '9' -Title 'Server: four generation endpoints'
    Add-UnselectedStep -Id '9b' -Title 'Two concurrent conversations stay isolated'
    Add-UnselectedStep -Id '9c' -Title 'Complete requests are serialized'
}

# ---------------------------------------------------------------------------
# Step 10: the 4096 boundary, with real tokenizer output
# ---------------------------------------------------------------------------

if (Test-Selected '10') {
    Write-Section 'Step 10: 4096 boundary with real tokenizer output'
    $ev = [ordered]@{}
    $problems = @()
    try {
        Start-FlmServer

        # The 400 body reports the RENDERED prompt length in real tokens, which
        # is the only place the product exposes the tokenizer's own count. One
        # deliberately over-long request therefore calibrates tokens-per-word
        # for this tokenizer instead of guessing at it.
        $unit = 'The quick brown fox jumps over the lazy dog. '
        $calBody = '{"model":"' + $ModelTag + '","prompt":"' + ($unit * 1200).Trim() + '","stream":false,"max_tokens":100000}'
        $cal = Invoke-Rest -Path '/api/generate' -Body $calBody -Name 'step10-calibrate'
        $ev['calibration_http'] = $cal.Code
        $ev['calibration_body'] = $cal.Body
        $rendered = $null
        if ($cal.Body -match 'rendered prompt has (\d+) tokens') { $rendered = [int]$Matches[1] }
        $ev['calibration_rendered_tokens'] = $rendered

        # The ACTIVE CAP is read from the product, not assumed to be 4096.
        #
        # It is 4095, and that is deliberate rather than an off-by-one: the KV
        # window is 4096 rows, but no token-attention kernel ships for a
        # 4096-token window, so aie4_active_cap() bounds prompt plus generated
        # tokens at 4095 and the engine refuses a 4096-wide decode before
        # submitting. An earlier version of this step hard-coded 4096, asked
        # for exactly the capacity that implied, and reported the product's
        # correct refusal as a failure. Reading the cap out of the refusal
        # makes this step check the RULE -- exactly the remaining capacity is
        # admitted, one more is not -- instead of checking a constant it
        # guessed.
        $activeCap = $null
        if ($cal.Body -match 'active context cap (\d+)') { $activeCap = [int]$Matches[1] }
        $ev['active_context_cap'] = $activeCap
        $ev['active_cap_note'] = 'the frontend caps prompt + generated tokens at the value above; the KV window is 4096 rows and the cap is one less because no token-attention kernel ships for a 4096-token window'
        if ($activeCap -ne 4095) {
            $problems += "the active context cap is $activeCap; 4095 is the shipped aie4_active_cap value and a change to it is a release decision, not a measurement"
        }
        if (-not $activeCap) { $activeCap = 4095 }

        if (-not $rendered) {
            $problems += 'could not read a rendered token count from the over-limit rejection, so the boundary cases cannot be aimed'
        } else {
            $tokensPerUnit = $rendered / 1200.0
            $ev['tokens_per_repeat_unit'] = [math]::Round($tokensPerUnit, 4)

            # (a) A prompt just under the cap, generating up to the cap.
            $target = $activeCap - 200
            $units = [int][math]::Floor($target / $tokensPerUnit)
            $underPrompt = ($unit * $units).Trim()
            $probeBody = '{"model":"' + $ModelTag + '","prompt":"' + $underPrompt + '","stream":false,"max_tokens":100000}'
            $probe = Invoke-Rest -Path '/api/generate' -Body $probeBody -Name 'step10-probe'
            $under = $null
            if ($probe.Body -match 'rendered prompt has (\d+) tokens') { $under = [int]$Matches[1] }
            $ev['under_cap_rendered_tokens'] = $under
            if ($under -and $under -lt $activeCap) {
                $room = $activeCap - $under
                $ev['remaining_capacity'] = $room
                # Exactly the remaining capacity must be ADMITTED: this is the
                # boundary, not a value near it.
                $atBody = '{"model":"' + $ModelTag + '","prompt":"' + $underPrompt + '","stream":false,"max_tokens":' + $room + '}'
                $at = Invoke-Rest -Path '/api/generate' -Body $atBody -Name 'step10-at-cap'
                $ev['exactly_remaining_http'] = $at.Code
                $ev['exactly_remaining_body_head'] = $at.Body.Substring(0, [math]::Min(600, $at.Body.Length))
                $ev['exactly_remaining_seconds'] = $at.Seconds
                if ($at.Body -match '"eval_count"\s*:\s*(\d+)') { $ev['exactly_remaining_eval_count'] = [int]$Matches[1] }
                if ($at.Body -match '"prompt_eval_count"\s*:\s*(\d+)') { $ev['exactly_remaining_prompt_eval_count'] = [int]$Matches[1] }
                if ($at.Code -ne '200') { $problems += "a request for exactly the remaining capacity ($room tokens) was refused with HTTP $($at.Code)" }
                if ($at.ServerDied) { $problems += "the server terminated generating to the cap (exit $($at.ServerExit))" }
                # One more than the remaining capacity must be REFUSED.
                $overBody = '{"model":"' + $ModelTag + '","prompt":"' + $underPrompt + '","stream":false,"max_tokens":' + ($room + 1) + '}'
                $over = Invoke-Rest -Path '/api/generate' -Body $overBody -Name 'step10-one-over'
                $ev['one_over_http'] = $over.Code
                $ev['one_over_body'] = $over.Body
                if ($over.Code -ne '400') { $problems += "a request for one token more than the remaining capacity returned HTTP $($over.Code), not 400" }
                elseif ($over.Body -notmatch 'exceeds the active context cap') { $problems += 'the one-over refusal does not name the active context cap' }
            } else {
                $problems += "could not construct a prompt under the cap of $activeCap (measured $under tokens)"
            }

            # (b) A prompt AT the cap requesting output: refused before submit,
            # with no generation-limit field at all.
            $atCapUnits = [int][math]::Ceiling(($activeCap + 100) / $tokensPerUnit)
            $atCapPrompt = ($unit * $atCapUnits).Trim()
            $atCapBody = '{"model":"' + $ModelTag + '","prompt":"' + $atCapPrompt + '","stream":false}'
            $atCap = Invoke-Rest -Path '/api/generate' -Body $atCapBody -Name 'step10-prompt-at-cap'
            $ev['prompt_at_cap_http'] = $atCap.Code
            $ev['prompt_at_cap_body'] = $atCap.Body
            if ($atCap.Code -ne '400') { $problems += "a prompt at or over the cap returned HTTP $($atCap.Code), not 400" }
            elseif ($atCap.Body -notmatch 'exceeds the active context cap') { $problems += 'the at-cap refusal does not name the active context cap' }

            # (c) A conversation grown across turns until admission fails.
            # No eviction, no partial append, no silent truncation: the
            # rendered-token count reported at the refusal must EXCEED the cap,
            # which is only true if the whole history was counted.
            # Each turn carries an explicit small generation limit, and the
            # blocks are large.
            #
            # This step is about ADMISSION at the cap, not about generation
            # quality, and generation here is expensive: a default request
            # was measured taking 182 seconds when the model ran on to the
            # context cap. Forty default turns is hours of device time to
            # test a rule that fires before a single token is produced.
            # Large blocks reach the boundary in a handful of turns and
            # `max_tokens` keeps each turn short; neither weakens the
            # admission check, because admission is computed from the
            # rendered prompt plus the requested output.
            $messages = @()
            $growth = @()
            $refused = $false
            $block = ($unit * 200).Trim()
            for ($turn = 1; $turn -le 12; $turn++) {
                $messages += ('{"role":"user","content":"' + $block + ' Turn ' + $turn + '. Reply with one short sentence."}')
                $body = '{"model":"' + $ModelTag + '","messages":[' + ($messages -join ',') + '],"stream":false,"max_tokens":8}'
                $r = Invoke-Rest -Path '/api/chat' -Body $body -Name "step10-grow-$turn"
                $growthRendered = $null
                if ($r.Body -match 'rendered prompt has (\d+) tokens') { $growthRendered = [int]$Matches[1] }
                $growth += [ordered]@{ turn = $turn; http = $r.Code; rendered_tokens = $growthRendered; seconds = $r.Seconds }
                if ($r.Code -eq '400') {
                    $refused = $true
                    $ev['growth_refusal_body'] = $r.Body
                    $ev['growth_refusal_rendered_tokens'] = $growthRendered
                    if ($r.Body -notmatch 'exceeds the active context cap') { $problems += 'the grown-conversation refusal does not name the active context cap' }
                    if ($growthRendered -and $growthRendered -lt $activeCap) {
                        $problems += "the grown conversation was refused at $growthRendered rendered tokens, below the cap of ${activeCap}: the whole history was not counted"
                    }
                    break
                }
                if ($r.Code -ne '200') { $problems += "turn $turn returned HTTP $($r.Code)"; break }
                # Keep the assistant reply in the history so the next turn's
                # rendering is the complete conversation.
                $reply = ''
                if ($r.Body -match '"content"\s*:\s*"((\\.|[^"\\])*)"') { $reply = $Matches[1] }
                $messages += ('{"role":"assistant","content":"' + $reply + '"}')
                if ($r.ServerDied) { $problems += "the server terminated growing the conversation at turn $turn (exit $($r.ServerExit))"; break }
            }
            $ev['growth'] = $growth
            $ev['growth_refused'] = $refused
            if (-not $refused) { $problems += 'the conversation never reached a refusal in 12 turns, so the growth boundary was not exercised' }
        }
    } catch {
        $problems += "boundary run failed: $($_.Exception.Message)"
        $ev['error'] = $_.Exception.Message
    } finally {
        Stop-FlmServer
    }
    if ($problems.Count -eq 0) {
        Add-StepResult -Id '10' -Title '4096 boundary on real tokenizer output' -Status 'met' `
            -Detail "against a cap of $($ev['active_context_cap']) read from the product: exactly the remaining capacity is admitted and one more is refused with HTTP 400; a prompt at the cap with no generation-limit field is refused; a grown conversation is refused only once the complete rendered history exceeds the cap" `
            -Evidence $ev -Criteria @('C19', 'C20', 'C23', 'C27')
    } else {
        $ev['problems'] = $problems
        Add-StepResult -Id '10' -Title '4096 boundary on real tokenizer output' -Status 'not_met' `
            -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C19', 'C20', 'C23', 'C27')
    }
} else {
    Add-UnselectedStep -Id '10' -Title '4096 boundary on real tokenizer output'
}

# ---------------------------------------------------------------------------
# Step 11: sustained run and stability, on the product path
# ---------------------------------------------------------------------------

if (Test-Selected '11') {
    Write-Section "Step 11: sustained generation of at least $SustainedTokens tokens"
    $ev = [ordered]@{}
    $problems = @()
    try {
        Start-FlmServer
        $samples = @()
        $proc = $script:server
        # Warm the session first. Design 15.3's memory gates are about growth
        # AFTER warmup, and sampling from a cold process measures the load, not
        # the decode loop.
        $warm = Invoke-Rest -Path '/api/generate' -Body ('{"model":"' + $ModelTag + '","prompt":"Say hello.","stream":false,"max_tokens":16}') -Name 'step11-warm'
        $ev['warm_http'] = $warm.Code
        $proc.Refresh()
        $ev['private_bytes_after_warmup'] = $proc.PrivateMemorySize64
        $ev['working_set_after_warmup'] = $proc.WorkingSet64

        $body = '{"model":"' + $ModelTag + '","prompt":"Write a long, detailed essay about the history of bridge building, covering ancient, medieval and modern eras.","stream":false,"max_tokens":' + $SustainedTokens + '}'
        $reqFile = Join-Path $script:artifactDir 'step11.req.json'
        $respFile = Join-Path $script:artifactDir 'step11.resp.json'
        Set-Content -Path $reqFile -Value $body -Encoding utf8
        $t0 = Get-Date
        $curl = Start-Process -FilePath 'curl.exe' -PassThru -NoNewWindow -ArgumentList @(
            '-s', '-o', $respFile, '-w', '%{http_code}',
            '-X', 'POST', "http://127.0.0.1:$Port/api/generate",
            '-H', 'Content-Type: application/json', '--data-binary', "@$reqFile",
            '--max-time', '3600') -RedirectStandardOutput (Join-Path $script:artifactDir 'step11.meta')
        while (-not $curl.HasExited) {
            Start-Sleep -Milliseconds 750
            if ($proc.HasExited) { break }
            $proc.Refresh()
            $samples += [ordered]@{
                t = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
                private_bytes = $proc.PrivateMemorySize64
                working_set = $proc.WorkingSet64
                handles = $proc.HandleCount
            }
        }
        $curl.WaitForExit()
        $ev['http'] = (Get-Content (Join-Path $script:artifactDir 'step11.meta') -ErrorAction SilentlyContinue)
        $respText = if (Test-Path $respFile) { Get-Content $respFile -Raw } else { '' }
        $ev['response_file'] = $respFile
        if ($respText -match '"eval_count"\s*:\s*(\d+)') { $ev['eval_count'] = [int]$Matches[1] }
        if ($respText -match '"prompt_eval_count"\s*:\s*(\d+)') { $ev['prompt_eval_count'] = [int]$Matches[1] }
        if ($respText -match '"response"\s*:\s*"((\\.|[^"\\])*)"') {
            $ev['completion_head'] = $Matches[1].Substring(0, [math]::Min(400, $Matches[1].Length))
        }
        $ev['seconds'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 3)
        $ev['samples'] = $samples
        $ev['sample_count'] = $samples.Count

        if (-not $ev.Contains('eval_count')) { $problems += 'the response carried no eval_count, so the number of generated tokens is unknown' }
        elseif ($ev['eval_count'] -lt $SustainedTokens) {
            # Stopping early on EOS is legitimate model behaviour, not a
            # defect, but it means this step did not measure what it claims to.
            $problems += "generation stopped at $($ev['eval_count']) tokens, short of the $SustainedTokens this step measures"
        }

        # Private-byte slope across the decode loop, on the product path.
        # Ordinary least squares over the samples, in bytes per second and then
        # normalised per token using the measured rate.
        if ($samples.Count -ge 8) {
            $n = $samples.Count
            $sx = 0.0; $sy = 0.0; $sxx = 0.0; $sxy = 0.0
            foreach ($s in $samples) {
                $sx += $s.t; $sy += [double]$s.private_bytes
                $sxx += $s.t * $s.t; $sxy += $s.t * [double]$s.private_bytes
            }
            $den = ($n * $sxx) - ($sx * $sx)
            if ($den -ne 0) {
                $slopePerSecond = (($n * $sxy) - ($sx * $sy)) / $den
                $ev['private_bytes_slope_per_second'] = [math]::Round($slopePerSecond, 1)
                if ($ev.Contains('eval_count') -and $ev['seconds'] -gt 0) {
                    $tokPerSec = $ev['eval_count'] / $ev['seconds']
                    $ev['tokens_per_second'] = [math]::Round($tokPerSec, 3)
                    if ($tokPerSec -gt 0) {
                        $ev['private_bytes_slope_per_token'] = [math]::Round($slopePerSecond / $tokPerSec, 1)
                    }
                }
            }
            $ev['private_bytes_first'] = $samples[0].private_bytes
            $ev['private_bytes_last'] = $samples[$n - 1].private_bytes
            $ev['private_bytes_delta'] = $samples[$n - 1].private_bytes - $samples[0].private_bytes
        } else {
            $ev['slope_note'] = "only $($samples.Count) memory samples were taken; too few for a slope"
        }

        # What this step CANNOT see, said plainly. Device-tensor and weight
        # creation counts live behind debug_snapshot() in a DEV_BUILD, which
        # the shipped flm.exe is not. The gates on those counters are asserted
        # by benchmark_phi4_aie4 and reported by the Task 13 baseline; this
        # step measures the process-level consequence, not the counter.
        $ev['device_object_counters'] = 'device_tensors_created_after_load and weight_objects_created_after_load are asserted == 0 by benchmark_phi4_aie4, which requires DEV_BUILD; the shipped flm.exe does not expose them, so this step measures process private bytes instead and does not claim the counter gates'

        # The slope is REPORTED and loosely bounded, and it is deliberately
        # not offered as evidence for "decode shows no per-token memory growth
        # after warmup".
        #
        # This measurement is of the whole flm.exe process while it serves an
        # HTTP request: the decode loop, the tokenizer, the response buffer,
        # the JSON encoder and the HTTP stack all allocate inside the window.
        # A positive slope here therefore does not establish growth in the
        # decode loop, and a flat one would not establish its absence. The
        # criterion that makes that claim is measured where it can be --
        # benchmark_phi4_aie4's memory window over explicit token IDs, with no
        # server in the process -- and this step does not stand in for it.
        #
        # What the bound below is for is the failure this step CAN see: an
        # unbounded leak. It is proportional rather than absolute so that it
        # means the same thing on a machine with a different footprint.
        if ($ev.Contains('private_bytes_delta') -and $ev.Contains('private_bytes_after_warmup')) {
            $base = [double]$ev['private_bytes_after_warmup']
            $grew = [double]$ev['private_bytes_delta']
            $fraction = $(if ($base -gt 0) { $grew / $base } else { 0 })
            $ev['private_bytes_growth_fraction'] = [math]::Round($fraction, 6)
            $ev['memory_claim_scope'] = 'process-level, on the product HTTP path; this does NOT measure decode-loop growth in isolation and is not offered as evidence for the design criterion about it'
            if ($fraction -gt 0.10) {
                $problems += ("private bytes grew {0:P2} of the post-warmup footprint over {1} tokens, which is a leak rather than allocator noise" -f $fraction, $ev['eval_count'])
            }
        }
    } catch {
        $problems += "sustained run failed: $($_.Exception.Message)"
        $ev['error'] = $_.Exception.Message
    } finally {
        Stop-FlmServer
    }
    if ($problems.Count -eq 0) {
        Add-StepResult -Id '11' -Title 'Sustained generation and memory stability' -Status 'met' `
            -Detail "generated $($ev['eval_count']) tokens in one session at $($ev['tokens_per_second']) tok/s; process private bytes moved $($ev['private_bytes_delta']) over the run ($($ev['private_bytes_slope_per_token']) bytes/token, $($ev['private_bytes_growth_fraction']) of the post-warmup footprint) -- recorded, and NOT offered as evidence about decode-loop growth" `
            -Evidence $ev -Criteria @('C40')
    } else {
        $ev['problems'] = $problems
        Add-StepResult -Id '11' -Title 'Sustained generation and memory stability' -Status 'not_met' `
            -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C40')
    }
} else {
    Add-UnselectedStep -Id '11' -Title 'Sustained generation and memory stability'
}

# ---------------------------------------------------------------------------
# Step 12: terminal-failure behaviour on real hardware
#
# Two distinct claims, and the fake cannot make the second one:
#   (a) the PLUMBING is FastFlow's -- admission closes, a record is persisted,
#       the process hard-terminates with 0xE0040001, and the next flm start
#       prints the record and removes it;
#   (b) the DIAGNOSTIC is genuinely corelib's -- the persisted `detail` is the
#       library's own text, not one FastFlow composed.
#
# Last, deliberately: (b) leaves a process dead by design.
# ---------------------------------------------------------------------------

if (Test-Selected '12') {
    Write-Section 'Step 12: terminal failure, plumbing and diagnostic'
    $ev = [ordered]@{}
    $problems = @()
    $child = Join-Path $BuildDir 'Release/test_fatal_child.exe'
    $ev['fault_child'] = $child
    $logDir = Join-Path $env:LOCALAPPDATA 'FastFlowLM\logs'
    $ev['log_dir'] = $logDir

    if (-not (Test-Path $child)) {
        Add-StepResult -Id '12' -Title 'Terminal failure: plumbing and corelib diagnostic' -Status 'not_exercised' `
            -Detail "test_fatal_child.exe not found at $child; build the hardware suite first" -Evidence $ev -Criteria @('C37', 'C38', 'C39')
    } else {
        # Start from a clean slate so a record found afterwards is this run's.
        $before = @(Get-ChildItem (Join-Path $logDir 'corelib-fatal-*.json') -ErrorAction SilentlyContinue)
        foreach ($f in $before) { Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue }
        $ev['pre_existing_records_removed'] = @($before | ForEach-Object { $_.Name })

        $markerDir = Join-Path $script:artifactDir 'fatal-markers'
        New-Item -ItemType Directory -Force -Path $markerDir | Out-Null
        $savedHw = $env:FLM_AIE4_HARDWARE
        $env:FLM_AIE4_HARDWARE = '1'
        try {
            $run = Invoke-Capture -Exe $child -Arguments @('--child', 'after_submit', $markerDir) -LogName 'step12-child.log'
            $ev['child_exit_decimal'] = $run.ExitCode
            # Formatted through an unchecked 32-bit reinterpretation.
            #
            # 0xE0040001 does not fit in a signed Int32, so the process exit
            # code arrives as -536608767, and casting the -band result
            # straight to [uint32] throws InvalidCastIConvertible. Going via
            # BitConverter reinterprets the bits instead of converting the
            # value, which is what "show me the exit code in hex" means.
            $ev['child_exit_hex'] = $(
                if ($null -eq $run.ExitCode) { '(none)' }
                else {
                    '0x{0:X8}' -f [System.BitConverter]::ToUInt32(
                        [System.BitConverter]::GetBytes([int]$run.ExitCode), 0)
                })
            $ev['child_output'] = $run.Text
            # 0xE0040001 as a signed 32-bit value is -536608767.
            $expected = -536608767
            if ($run.ExitCode -ne $expected) {
                $problems += "the post-submit failure exited $($ev['child_exit_hex']), not 0xE0040001"
            }
            if ($run.Text -notmatch 'AIE4 terminal failure') { $problems += 'the terminating process did not print "AIE4 terminal failure"' }
        } finally {
            if ($savedHw) { $env:FLM_AIE4_HARDWARE = $savedHw } else { Remove-Item Env:\FLM_AIE4_HARDWARE -ErrorAction SilentlyContinue }
        }

        $records = @(Get-ChildItem (Join-Path $logDir 'corelib-fatal-*.json') -ErrorAction SilentlyContinue)
        $ev['records_written'] = @($records | ForEach-Object { $_.Name })
        if ($records.Count -ne 1) {
            $problems += "expected exactly one persisted fatal record, found $($records.Count)"
        } else {
            $recordText = Get-Content $records[0].FullName -Raw
            $ev['record_verbatim'] = $recordText
            $rec = $null
            try { $rec = $recordText | ConvertFrom-Json } catch { $problems += 'the persisted fatal record is not valid JSON' }
            if ($rec) {
                foreach ($f in @('status', 'call', 'detail', 'phase', 'layer', 'rows', 'position', 'pid')) {
                    if ($rec.PSObject.Properties.Name -contains $f) { $ev["record_$f"] = $rec.$f }
                    else { $problems += "the persisted record has no '$f' field" }
                }
                # (b) The detail must be CORELIB's text. The child writes what
                # the library actually returned to expected-detail.marker
                # before the policy runs, so the comparison is against the
                # library, not against FastFlow's idea of it.
                $detailMarker = Join-Path $markerDir 'expected-detail.marker'
                if (Test-Path $detailMarker) {
                    $expectedDetail = (Get-Content $detailMarker -Raw).Trim()
                    $ev['corelib_detail_marker_verbatim'] = $expectedDetail
                    $ev['record_detail_equals_corelib_text'] = ($rec.detail -eq $expectedDetail)
                    if ($rec.detail -ne $expectedDetail) {
                        $problems += 'the persisted detail is not the corelib text the library returned'
                    }
                    # A FastFlow-composed message would read like FastFlow.
                    # This is a weak check and is recorded as one; the equality
                    # above is the real evidence.
                    $ev['detail_looks_composed_by_fastflow'] = ($expectedDetail -match '(?i)fastflow|AIE4 terminal')
                } else {
                    $problems += 'the fault child wrote no expected-detail.marker, so the diagnostic cannot be attributed to corelib'
                }
                if ([string]$rec.detail -match '(?i)akholodn') {
                    $ev['corelib_detail_embeds_foreign_build_path'] = 'the library''s message carries an absolute source path under another user''s tree, because the DynamicDispatch binaries in this closure were built there and copied. It is cosmetic for this run -- the message is still corelib''s own and still names the offending shape -- but it means a diagnostic shipped to a user would point at a path that does not exist on their machine.'
                }
                $ev['real_failure_mechanism'] = 'ryzenai_corelib matmul_pad_shape(m=4096, k=3072, n=200064, group=128), which the header documents as a clean rejection naming the offending argument. The brief suggested dispatching against a tensor allocated below the padded extent; that alternative was NOT used, and this deviation is recorded rather than glossed. Both are real corelib refusals carrying the library''s own message; neither is fake-injected.'
            }

            # (a) continued: the NEXT flm start prints the record and removes
            # it. Nothing before this ran flm.exe twice to check that.
            $first = Invoke-Capture -Exe $FlmExe -Arguments @('version') -LogName 'step12-drain-first.log'
            $ev['first_start_exit'] = $first.ExitCode
            $ev['first_start_output'] = $first.Text
            # Matched on fields that survive JSON escaping.
            #
            # The record is printed verbatim as JSON, so its `detail` appears
            # with backslashes doubled and cannot be found by searching for
            # the unescaped string -- and the "AIE4 fatal record:" header is
            # emitted for an INCOMPLETE pending file, not for a complete one.
            # Matching on the pid and the failure timestamp identifies THIS
            # record specifically and contains nothing that gets escaped.
            $marks = @()
            if ($rec) {
                if ($rec.PSObject.Properties.Name -contains 'failure_utc') { $marks += [string]$rec.failure_utc }
                if ($rec.PSObject.Properties.Name -contains 'pid') { $marks += ('"pid":' + $rec.pid) }
                if ($rec.PSObject.Properties.Name -contains 'call') { $marks += [string]$rec.call }
            }
            $ev['first_start_match_markers'] = $marks
            $printed = $false
            if ($marks.Count -gt 0) {
                $printed = $true
                foreach ($mark in $marks) {
                    if ($first.Text -notmatch [regex]::Escape($mark)) { $printed = $false }
                }
            }
            $ev['first_start_printed_the_record'] = $printed
            if (-not $printed) { $problems += 'the next flm start did not print the persisted fatal record' }
            $after = @(Get-ChildItem (Join-Path $logDir 'corelib-fatal-*.json') -ErrorAction SilentlyContinue)
            $ev['records_after_first_start'] = @($after | ForEach-Object { $_.Name })
            if ($after.Count -ne 0) { $problems += 'the next flm start did not remove the record it reported' }
            $second = Invoke-Capture -Exe $FlmExe -Arguments @('version') -LogName 'step12-drain-second.log'
            $ev['second_start_output'] = $second.Text
            $ev['second_start_silent'] = ($second.Text -notmatch 'AIE4 fatal record')
            if ($second.Text -match 'AIE4 fatal record') { $problems += 'a second flm start reported the record again; it was not consumed' }
        }

        if ($problems.Count -eq 0) {
            Add-StepResult -Id '12' -Title 'Terminal failure: plumbing and corelib diagnostic' -Status 'met' `
                -Detail 'a real corelib refusal after first submit hard-terminated with 0xE0040001, persisted exactly one record whose detail is the library''s own verbatim text, and the next flm start printed and removed it while the one after was silent' `
                -Evidence $ev -Criteria @('C37', 'C38', 'C39')
        } else {
            $ev['problems'] = $problems
            Add-StepResult -Id '12' -Title 'Terminal failure: plumbing and corelib diagnostic' -Status 'not_met' `
                -Detail ($problems -join '; ') -Evidence $ev -Criteria @('C37', 'C38', 'C39')
        }
    }
} else {
    Add-UnselectedStep -Id '12' -Title 'Terminal failure: plumbing and corelib diagnostic'
}

# ---------------------------------------------------------------------------
# Step 13: the acceptance result
#
# The Section 17 table is DERIVED. Each criterion names the step ids that
# exercise it, and inherits the worst status among them; a criterion with no
# step mapped to it is not_exercised by construction. There is no way to write
# `met` next to a criterion here, which is the point -- four times on this
# branch a retraction reached the code and the report but not the rendered
# artifact.
# ---------------------------------------------------------------------------

# id, text, the step(s) that exercise it, and -- where the steps establish only
# part of what the criterion says -- an explicit `gap`.
#
# `gap` exists because a two-valued roll-up lies by rounding. A criterion whose
# mapped steps cover half of it renders MET, and a reader has no way to tell
# that from one covered end to end. Any criterion with a non-empty `gap`
# renders PARTIAL and can never render MET, however green its steps are.
#
# An empty step list is not an oversight to be filled in with a verdict: it
# means this acceptance run does not exercise the criterion, and the table must
# say so.
$criteria = @(
  @{ id='C01'; text='A new phi4-mini-it-aie4:4b tag selects only corelib_aie4.'; steps=@('1','6'); gap='' }
  @{ id='C02'; text='Its catalog entry uses existing context fields and measured size and on-disk footprint; no nonexistent maximum-context field is assumed.'; steps=@('1','5'); gap='' }
  @{ id='C03'; text='Existing phi4-mini-it:4b Q4NX/NPU2 selection is unchanged.'; steps=@('1'); gap='the catalog entry is checked; no Q4NX/NPU2 model was loaded, because this box has no XDNA2 hardware' }
  @{ id='C04'; text='FastFlow invokes corelib from native C++ with no Python runtime.'; steps=@(); gap='' }
  @{ id='C05'; text='The executable does not link the corelib import library.'; steps=@('2'); gap='' }
  @{ id='C06'; text='Missing corelib DLLs do not prevent non-AIE4 FastFlow startup.'; steps=@('3'); gap='' }
  @{ id='C07'; text='An unwritable fatal-log directory rejects the AIE4 tag with an actionable error but does not block non-AIE4 startup.'; steps=@('4'); gap='' }
  @{ id='C08'; text='The exact Phi-4 model contract is validated before packing.'; steps=@('1','1c','5'); gap='' }
  @{ id='C09'; text='Every MatMul descriptor has has_bias = false.'; steps=@('5'); gap='' }
  @{ id='C10'; text='All 161 weights load from ONNX-layout components.'; steps=@('5'); gap='the manifest resolves all 161 objects and every role they name; that the engine LOADS all 161 is not separately observed here' }
  @{ id='C11'; text='Exact component shapes are validated; scales passed to corelib are contiguous FP16.'; steps=@('5'); gap='shapes and FP16 dtype are checked in the manifest; contiguity as passed to corelib is not observable from the product' }
  @{ id='C12'; text='Source ONNX mappings and derived scale/norm buffers remain alive and unmodified for every weight object.'; steps=@(); gap='' }
  @{ id='C13'; text='One persistent Stream is reused across tokens and requests.'; steps=@('6'); gap='Step 6 records dispatch and synchronize counters from a loaded model; it does not observe Stream identity, and nothing here shows the same Stream is reused ACROSS REQUESTS' }
  @{ id='C14'; text='Every layer executes q, k, v, MHA, o, and SSMLP on AIE4.'; steps=@(); gap='' }
  @{ id='C15'; text='Embedding, layer-0 input RMSNorm, V scatter, last-row extraction, and sampling are the only planned host islands.'; steps=@(); gap='' }
  @{ id='C16'; text='Fresh multi-row prefill runs only at position zero.'; steps=@(); gap='' }
  @{ id='C17'; text='Nonzero-position prompt continuation uses only one-row calls.'; steps=@(); gap='' }
  @{ id='C18'; text='Prefix hits choose between one-row suffix append and full fresh re-prefill using the measured release-fixed suffix threshold, and both routes meet the same Section 15.3 gates.'; steps=@('8b'); gap='' }
  @{ id='C19'; text='Append-winning sampled suffix lengths are prefix-monotonic; otherwise the published threshold is zero.'; steps=@(); gap='' }
  @{ id='C20'; text='User-visible prompt plus generated tokens and logical KV position are each bounded at 4096, with over-limit requests rejected before submit.'; steps=@('10','8c'); gap='the enforced bound is 4095, not the 4096 this criterion states -- no token-attention kernel ships for a 4096-token window. The RULE is verified exactly (the remaining capacity is admitted, one more is refused) but against a different constant than the design text, and that discrepancy is a design question, not a measurement' }
  @{ id='C21'; text='Limits count complete rendered multi-turn history; no partial append, eviction, or sliding window occurs.'; steps=@('8','10'); gap='' }
  @{ id='C22'; text='set_context_length and update_max_length obey the fixed-pitch behavior in Section 7.3.'; steps=@(); gap='' }
  @{ id='C23'; text='Lowering the soft cap below live logical context is rejected atomically; clearing first permits the lower cap.'; steps=@(); gap='' }
  @{ id='C24'; text='Invalid CLI/REST context requests return without mutating either cap; REST over-limit admission is HTTP 400.'; steps=@('9','10'); gap='the REST over-limit HTTP 400 half is verified on all four endpoints; that a refused request leaves BOTH CAPS UNMUTATED is not observed, and no invalid CLI context request was issued' }
  @{ id='C25'; text='An omitted REST generation limit is not interpreted as an explicit 4096; all three handlers admit a nonempty default request and stop at the active soft cap, including after that cap is lowered.'; steps=@('9'); gap='default requests are admitted and answered on every handler; the "including after that cap is lowered" clause is not exercised, because nothing in this run lowers the cap' }
  @{ id='C26'; text='K/V caches are separate BF16 [8,4096,128] tensors per layer.'; steps=@('6'); gap='the reported KV byte total is exactly what separate BF16 [8,4096,128] caches require for 32 layers; separateness and layout are inferred from that total, not observed directly' }
  @{ id='C27'; text='Only live V rows are scattered.'; steps=@(); gap='' }
  @{ id='C28'; text='Every allocation row extent is derived from the three pad helpers, and every required MatMul padded K/N equals its logical K/N.'; steps=@('6b'); gap='' }
  @{ id='C29'; text='SSMLP uses four distinct buffers; input and normalized output never alias.'; steps=@(); gap='' }
  @{ id='C30'; text='RoPE is gathered on the host from its actual source shape/dtype and written once into the contiguous FP32 [4096,48] device tensor.'; steps=@('5'); gap='the SOURCE shape and dtype are read from the real model (float16 [135168,48]); the host gather and the single write into the FP32 [4096,48] device tensor are not observed from the product' }
  @{ id='C31'; text='The runtime corelib version is checked against the compiled-against version before any other symbol is used.'; steps=@('3b'); gap='' }
  @{ id='C32'; text='All conversion crosses tensor_write/tensor_read; the only FastFlow host converter is the FP32-to-BF16 helper for SSMLP norm/epsilon blobs.'; steps=@(); gap='' }
  @{ id='C33'; text='No tensor_write/tensor_read call site passes a byte count or byte offset.'; steps=@(); gap='' }
  @{ id='C34'; text='QKV, MHA, O, SSMLP, and LM-head dependency boundaries synchronize as specified.'; steps=@(); gap='' }
  @{ id='C35'; text='LM head synchronizes before logits are read.'; steps=@(); gap='' }
  @{ id='C36'; text='Golden logits meet correlation, top-1, top-5, top-32 max-difference, and lowest-ID tie-breaking gates.'; steps=@('7b'); gap='' }
  @{ id='C37'; text='FastFlow resolves EOS IDs 200020 and 199999 from the accepted tokenizer configuration.'; steps=@(); gap='' }
  @{ id='C38'; text='CLI and streaming/non-streaming server E2E pass on xcomedusad-43.'; steps=@('7','8','9'); gap='' }
  @{ id='C39'; text='flm validate can report corelib AIE4 readiness independently of legacy XDNA2 readiness.'; steps=@('3'); gap='' }
  @{ id='C40'; text='Complete requests are serialized for the single mutable KV session.'; steps=@('9c'); gap='' }
  @{ id='C41'; text='Cancellation never abandons submitted work without synchronize.'; steps=@(); gap='' }
  @{ id='C42'; text='Detailed corelib errors are copied immediately and remain durable.'; steps=@('12'); gap='' }
  @{ id='C43'; text='A pre-first-submit operation failure remains nonterminal; any post-first-submit operation failure or synchronize failure enters the process-wide hard-termination path with no fallback.'; steps=@('12'); gap='only the POST-submit half is exercised: a real corelib refusal after first submit hard-terminates with 0xE0040001. No pre-first-submit failure was injected, so the nonterminal half of the criterion is untested here' }
  @{ id='C44'; text='A pre-first-submit failure clears all session state, and interactive CLI output explicitly reports that the conversation was cleared.'; steps=@(); gap='' }
  @{ id='C45'; text='Healthy shutdown destroys all handles before corelib cleanup.'; steps=@('3'); gap='flm validate completes a load-and-shutdown cycle and reports shutdown_ok; the ORDER in which handles are destroyed relative to corelib cleanup is not observed' }
  @{ id='C46'; text='A terminal runtime failure flushes the real corelib diagnostic, skips normal unload, and hard-terminates the whole FastFlow process.'; steps=@('12'); gap='' }
  @{ id='C47'; text='Concurrent FastFlow processes use PID/start-time-qualified fatal files and the next startup reports every completed record deterministically without disturbing a live process pending file.'; steps=@('12'); gap='the record is PID- and start-time-qualified and the next startup reports and removes it; no two FastFlow processes were run concurrently, so nothing here shows a live process pending file left undisturbed' }
  @{ id='C48'; text='Failure to query another process start time leaves its pending file untouched.'; steps=@(); gap='' }
  @{ id='C49'; text='Tests distinguish pre-submit dependency isolation from the process-wide impact of a terminal runtime failure.'; steps=@(); gap='' }
  # Deliberately mapped to NO step. Step 11 sustains 512 tokens on the product
  # path and records the process's private-byte slope, but that window contains
  # the HTTP stack, the tokenizer and the response buffer as well as the decode
  # loop, so it can neither establish nor refute per-token growth IN DECODE.
  # Mapping it here would turn a measurement of something else into a met
  # criterion, which is the exact move this file exists to prevent.
  @{ id='C50'; text='Decode shows no per-token memory growth after warmup.'; steps=@(); gap='' }
  @{ id='C51'; text='Cold/warm, fresh-prefill, continuation-ingest, decode, memory, and V-scatter baselines are recorded.'; steps=@(); gap='' }
  @{ id='C52'; text='No corelib, HIP-EP, ORT graph, GPU hybrid, or AIE4 kernel change is required.'; steps=@(); gap='' }
)

# Re-read the record from the file before rolling anything up.
#
# Two reasons, and the second is the important one. First, it makes every step
# result the same shape (PSCustomObject) whether it was produced in this
# session or carried over from the JSON. Second, and this is the point: the
# Section 17 table and the document are then rendered from THE ARTIFACT THAT
# WAS WRITTEN, not from the in-memory state that produced it. Four times on
# this branch a correction reached the code and the report but not the
# rendered document. Rendering from the file makes that particular divergence
# impossible: if the document says it, the JSON says it.
$onDisk = Get-Content $OutJson -Raw | ConvertFrom-Json
$reloaded = [ordered]@{}
foreach ($prop in $onDisk.PSObject.Properties) { $reloaded[$prop.Name] = $prop.Value }
$script:record = $reloaded

# Say what this run did NOT measure about determinism, always, in the record
# and in the document.
#
# Design DETERM-5 (human decision) makes run-to-run divergence a known,
# measured, bounded defect carried alongside this release rather than a
# release blocker, so a DETERM breach does not fail this acceptance verdict.
# The gates and the DETERM-2 seal are untouched -- they are how a CHANGE in
# the behaviour gets noticed, and nothing here widens a bound or makes a gate
# advisory.
#
# What this acceptance run must not do is let the defect go silently absent.
# It runs no determinism campaign at all: no repeated golden, no bit-identity
# comparison, no rate. The observed divergence rate is a few percent, so even
# a run that DID measure and came out clean would not be evidence the defect
# is gone. Stating that here means the reader cannot infer from a green
# acceptance that the question is settled.
Set-Field 'determinism_scope' ([ordered]@{
    exercised = $false
    why = 'this acceptance run measures the shipped product against the real model; it runs no repeated-golden or bit-identity campaign, so it produces no determinism evidence in either direction'
    policy = 'DETERM-5: run-to-run divergence is a known, measured, bounded defect carried with this release and is not a blocker for this acceptance verdict. The DETERM-1/2 gates and the sealed constants are unchanged.'
    clean_run_is_not_absence = 'the observed divergence rate is a few percent, so a run that did not breach a gate would not be evidence the defect is gone'
    measured_by = 'run_hardware_suite.ps1 -DeterminismRuns <n> with src/tools/compare_phi4_corelib_e2e.py; baseline in docs/docs/benchmarks/phi4_results.md'
    sharpened_hypothesis = 'precision alone cannot produce this: deterministic rounding of deterministic inputs is reproducible. Identical inputs giving different results additionally requires the ORDER of operations to vary -- a parallel reduction combining partials in completion order, an unpinned tiling or work split, or a dispatch-time scheduling choice. Floating-point addition is not associative, so varying order varies the last bits. The search is therefore for an operator whose accumulation or work-partitioning order is not pinned, not for the least accurate one, and it is confined to the 32-layer model body because the LM head is exonerated against an FP64 host reference.'
})
Save-Record

$stepById = @{}
foreach ($s in @($script:record['steps'])) { $stepById[$s.id] = $s }

$rank = @{ 'met' = 0; 'partial' = 1; 'not_exercised' = 2; 'not_met' = 3 }
$rollup = @()
foreach ($c in $criteria) {
    $status = 'not_exercised'
    $why = 'no step in this acceptance run exercises this criterion'
    $cited = @()
    if ($c.steps.Count -gt 0) {
        $worst = 'met'
        $details = @()
        foreach ($sid in $c.steps) {
            if ($stepById.ContainsKey($sid)) {
                $st = [string](Get-Field $stepById[$sid] 'status')
                $cited += $sid
                $details += "step ${sid}: $st"
                if ($rank[$st] -gt $rank[$worst]) { $worst = $st }
            } else {
                $details += "step ${sid}: absent from this record"
                if ($rank['not_exercised'] -gt $rank[$worst]) { $worst = 'not_exercised' }
            }
        }
        $status = $worst
        $why = ($details -join '; ')
    }
    # A declared gap caps the result at PARTIAL. It can never be argued back up
    # to MET by the steps being green, because the steps being green is exactly
    # what the gap is about: they pass, and they cover less than the criterion
    # says. A gap on a criterion that already failed or was never exercised
    # changes nothing -- the worse verdict stands.
    if ($c.gap -and $status -eq 'met') {
        $status = 'partial'
        $why = "$why -- PARTIAL: $($c.gap)"
    } elseif ($c.gap) {
        $why = "$why (declared coverage gap: $($c.gap))"
    }
    $rollup += [ordered]@{
        id = $c.id; criterion = $c.text; status = $status
        from_steps = $cited; coverage_gap = $c.gap; basis = $why
    }
}
Set-Field 'section_17' $rollup

# One mapping, not two.
#
# Each step carries a `criteria` list, and until this point that list came
# from the -Criteria argument at the call site -- a SECOND mapping, which can
# and did disagree with the table above. Two mappings means one of them is
# wrong and nothing notices. The table is the source of truth, so the per-step
# lists are rewritten from it here and the call-site values are discarded.
$byStep = @{}
foreach ($c in $criteria) {
    foreach ($sid in $c.steps) {
        if (-not $byStep.ContainsKey($sid)) { $byStep[$sid] = @() }
        $byStep[$sid] += $c.id
    }
}
foreach ($st in @($script:record['steps'])) {
    $st.criteria = $(if ($byStep.ContainsKey($st.id)) { $byStep[$st.id] } else { @() })
}
Save-Record

$met = @($rollup | Where-Object { $_.status -eq 'met' }).Count
$partial = @($rollup | Where-Object { $_.status -eq 'partial' }).Count
$notMet = @($rollup | Where-Object { $_.status -eq 'not_met' }).Count
$notEx = @($rollup | Where-Object { $_.status -eq 'not_exercised' }).Count
$stepsMet = @($script:record['steps'] | Where-Object { $_.status -eq 'met' }).Count
$stepsNotMet = @($script:record['steps'] | Where-Object { $_.status -eq 'not_met' }).Count
$stepsNotEx = @($script:record['steps'] | Where-Object { $_.status -eq 'not_exercised' }).Count
Set-Field 'summary' ([ordered]@{
    criteria_total = $rollup.Count
    criteria_met = $met
    criteria_partial = $partial
    criteria_not_met = $notMet
    criteria_not_exercised = $notEx
    steps_met = $stepsMet
    steps_not_met = $stepsNotMet
    steps_not_exercised = $stepsNotEx
})
Save-Record

Write-Section 'Design Section 17, line by line'
foreach ($r in $rollup) {
    $marker = switch ($r.status) {
        'met'     { 'MET          ' }
        'partial' { 'PARTIAL      ' }
        'not_met' { 'NOT MET      ' }
        default   { 'NOT EXERCISED' }
    }
    Write-Output ("  {0}  {1}  {2}" -f $r.id, $marker, $r.criterion)
}

Write-Section 'Summary'
Write-Output "  steps:    $stepsMet met, $stepsNotMet not met, $stepsNotEx not exercised"
Write-Output "  criteria: $met met, $partial partial, $notMet not met, $notEx not exercised"
Write-Output "  record:   $OutJson"

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

$script:renderError = $null
if ($Markdown) {
  try {
    $md = New-Object System.Collections.Generic.List[string]
    $md.Add('---')
    $md.Add('layout: docs')
    $md.Add('title: Phi-4 AIE4 acceptance')
    $md.Add('nav_order: 20')
    $md.Add('parent: Benchmarks')
    $md.Add('---')
    $md.Add('')
    $md.Add('# Phi-4 AIE4 real-model acceptance')
    $md.Add('')
    $md.Add('<!-- GENERATED by src/test/phi4_corelib_aie4/run_real_model_acceptance.ps1.')
    $md.Add('     Every verdict below is derived from phi4_aie4_acceptance.json.')
    $md.Add('     Do not hand-edit: the next run overwrites this file. -->')
    $md.Add('')
    $id = $script:record['identity']
    $md.Add('## What produced these results')
    $md.Add('')
    $md.Add('| | |')
    $md.Add('| --- | --- |')
    $md.Add("| Machine | ``$($id.machine)`` |")
    $md.Add("| CPU | $($id.cpu) |")
    $md.Add("| NPU | $($id.npu), driver $($id.npu_driver), status $($id.npu_status) |")
    $md.Add("| FastFlow commit ``flm.exe`` was built from | ``$($id.binary_revision)`` |")
    $md.Add("| Commit this acceptance ran at | ``$($id.checkout_revision)`` |")
    $md.Add("| Harness SHA-256 | ``$($id.harness_sha256)`` |")
    $md.Add("| Console driver SHA-256 | ``$($id.driver_sha256)`` |")
    $changedSince = Get-Field $id 'product_sources_changed_since_binary_revision'
    if ($null -ne $changedSince) {
        $md.Add("| Product sources differing between them | $(@($changedSince).Count) |")
    }
    $md.Add("| ``flm.exe`` SHA-256 | ``$($id.flm_exe_sha256)`` |")
    $md.Add("| corelib source revision | ``$($id.corelib_source_revision)`` |")
    $md.Add("| Model | ``amd/phi-4-mini-instruct-oga-dml`` at HF ``$($id.hf_revision)`` |")
    $md.Add("| Model tag | ``$($id.model_tag)`` |")
    $md.Add("| Run (UTC) | $($id.utc) |")
    $md.Add('')
    $md.Add('AIE4 runtime closure under test:')
    $md.Add('')
    $md.Add('| DLL | Bytes | SHA-256 |')
    $md.Add('| --- | ---: | --- |')
    foreach ($d in @($id.corelib_dlls)) { $md.Add("| ``$($d.name)`` | $($d.bytes) | ``$($d.sha256)`` |") }
    $md.Add('')
    $s = $script:record['summary']
    $md.Add('## Result')
    $md.Add('')
    $md.Add("**Steps:** $($s.steps_met) met, $($s.steps_not_met) not met, $($s.steps_not_exercised) not exercised.")
    $md.Add('')
    $md.Add("**Design Section 17 criteria:** $($s.criteria_met) met, $($s.criteria_partial) partial, $($s.criteria_not_met) not met, $($s.criteria_not_exercised) not exercised.")
    $md.Add('')
    $md.Add('"Not exercised" is not a soft pass. It means this acceptance run did')
    $md.Add('not test the criterion, and nothing here should be read as evidence')
    $md.Add('that it holds.')
    $md.Add('')
    $md.Add('"Partial" means the steps mapped to a criterion passed but cover less')
    $md.Add('than the criterion says. The uncovered part is named in the Basis column')
    $md.Add('of every partial row. A partial criterion is not met.')
    $md.Add('')
    $det = Get-Field $script:record 'determinism_scope'
    if ($det) {
        $md.Add('### Run-to-run determinism is not measured here')
        $md.Add('')
        $md.Add((Get-Field $det 'why') + '.')
        $md.Add('')
        $md.Add((Get-Field $det 'policy'))
        $md.Add('')
        $md.Add('**A clean acceptance run is not evidence that the divergence is gone.** ' +
                (Get-Field $det 'clean_run_is_not_absence') + '.')
        $md.Add('')
        $md.Add('It is measured by `' + (Get-Field $det 'measured_by') + '`.')
        $md.Add('')
        $md.Add('On what a future investigation should look for: ' + (Get-Field $det 'sharpened_hypothesis'))
        $md.Add('')
    }
    $md.Add('## Steps')
    $md.Add('')
    $md.Add('| Step | Result | What was checked |')
    $md.Add('| --- | --- | --- |')
    foreach ($st in @($script:record['steps'])) {
        $mk = switch ($st.status) { 'met' { 'MET' } 'not_met' { '**NOT MET**' } default { '*not exercised*' } }
        $d = ($st.detail -replace '\|', '\|')
        $md.Add("| $($st.id). $($st.title) | $mk | $d |")
    }
    $md.Add('')
    $md.Add('## Design Section 17, line by line')
    $md.Add('')
    $md.Add('| | Criterion | Result | Basis |')
    $md.Add('| --- | --- | --- | --- |')
    foreach ($r in @($script:record['section_17'])) {
        $mk = switch ($r.status) {
            'met'     { 'MET' }
            'partial' { '**PARTIAL**' }
            'not_met' { '**NOT MET**' }
            default   { '*not exercised*' }
        }
        $t = ($r.criterion -replace '\|', '\|')
        $b = ($r.basis -replace '\|', '\|')
        $md.Add("| $($r.id) | $t | $mk | $b |")
    }
    $md.Add('')

    # Verbatim prompts and completions. The brief asks for these so a human can
    # judge coherence, which no assertion in this script can do for them.
    $md.Add('## Verbatim prompts and completions')
    $md.Add('')
    $any = $false
    foreach ($st in @($script:record['steps'])) {
        $e = $st.evidence
        if (-not $e) { continue }
        # @($e.PSObject.Properties | ForEach-Object { $_.Name }), not
        # @($e.PSObject.Properties.Name): member enumeration over an EMPTY
        # property collection throws under Set-StrictMode -Version Latest,
        # which is how the document generator died after the summary had
        # already printed a clean result.
        $names = @($e.PSObject.Properties | ForEach-Object { $_.Name })
        if ($names -contains 'probes') {
            $any = $true
            $md.Add("### Step $($st.id): $($st.title)")
            $md.Add('')
            $md.Add('The mechanical check below each completion only rules out gross')
            $md.Add('failure. Whether these read as a working model is a human')
            $md.Add('judgement, and making it is what this section is for.')
            $md.Add('')
            foreach ($pr in @($e.probes)) {
                $verdict = $(if (Get-Field $pr 'coherent') { 'passed the mechanical check' } else { 'FAILED the mechanical check: ' + ((Get-Field $pr 'coherence_reasons') -join ', ') })
                $secs = Get-Field $pr 'seconds'
                $md.Add("**$(Get-Field $pr 'name')** ($secs s, $verdict)")
                $md.Add('')
                $md.Add('```')
                $md.Add(">>> $(Get-Field $pr 'prompt_verbatim')")
                foreach ($line in ([string](Get-Field $pr 'completion_verbatim') -split "`r?`n")) { $md.Add($line) }
                $md.Add('```')
                $md.Add('')
            }
            continue
        }
        if ($names -contains 'prompt_verbatim' -and $names -contains 'completion_verbatim') {
            $any = $true
            $md.Add("### Step $($st.id): $($st.title)")
            $md.Add('')
            $md.Add('Prompt:')
            $md.Add('')
            $md.Add('```')
            $md.Add([string]$e.prompt_verbatim)
            $md.Add('```')
            $md.Add('')
            $md.Add('Completion, exactly as the console rendered it:')
            $md.Add('')
            $md.Add('```')
            foreach ($line in ([string]$e.completion_verbatim -split "`r?`n")) { $md.Add($line) }
            $md.Add('```')
            $md.Add('')
        }
        if ($names -contains 'per_turn') {
            $any = $true
            $md.Add("### Step $($st.id): $($st.title)")
            $md.Add('')
            foreach ($t in @($e.per_turn)) {
                $md.Add("**Turn $(Get-Field $t 'index')** ($(Get-Field $t 'seconds') s, continuation route ``$(Get-Field $t 'continuation_route')``)")
                $md.Add('')
                $md.Add('```')
                $md.Add("> $(Get-Field $t 'prompt')")
                foreach ($line in ([string](Get-Field $t 'reply_verbatim') -split "`r?`n")) { $md.Add($line) }
                $md.Add('```')
                $md.Add('')
            }
        }
    }
    if (-not $any) {
        $md.Add('No generation step in this run recorded a prompt and completion.')
        $md.Add('')
    }

    $md.Add('## Measured timings')
    $md.Add('')
    $md.Add('| Measurement | Value | From |')
    $md.Add('| --- | ---: | --- |')
    foreach ($st in @($script:record['steps'])) {
        $e = $st.evidence
        if (-not $e) { continue }
        foreach ($n in @($e.PSObject.Properties | ForEach-Object { $_.Name })) {
            if ($n -match '(_ns|_seconds|_per_second|_per_token|_bytes|_count|_tokens)$') {
                $v = $e.$n
                if ($null -ne $v -and $v -isnot [array] -and $v -isnot [System.Management.Automation.PSCustomObject]) {
                    $md.Add("| ``$n`` | $v | step $($st.id) |")
                }
            }
        }
    }
    $md.Add('')
    $md.Add('Every number above was produced by the ``flm.exe`` and corelib DLL')
    $md.Add('identified at the top of this document. Single-run figures on this')
    $md.Add('box have been measured to vary by up to 1.8x between runs; treat any')
    $md.Add('one of them as an observation, not a specification.')
    $md.Add('')
    $md.Add("Generated $((Get-Date).ToUniversalTime().ToString('u')) from ``$OutJson``.")

    $md.Add('')
    $md.Add("<!-- run-stamp: $script:runStamp -->")

    $mdDir = Split-Path -Parent $Markdown
    if ($mdDir -and -not (Test-Path $mdDir)) { New-Item -ItemType Directory -Force -Path $mdDir | Out-Null }
    # Deleted first, then written, then READ BACK and checked for this run's
    # stamp.
    #
    # The renderer runs last, so an exception in it leaves the previous
    # document on disk while the run still exits with its real verdict. That
    # happened: a field-name collision crashed rendering, and the committed
    # document went on asserting a Step 9b claim the record had already
    # retracted -- the exact "the fix reached the code but not the rendered
    # artifact" failure this project has hit four times. A stale document must
    # be impossible to mistake for a fresh one, so absence is the failure mode
    # rather than staleness, and the stamp proves which run wrote it.
    Remove-Item -LiteralPath $Markdown -Force -ErrorAction SilentlyContinue
    ($md -join "`n") | Set-Content -Path $Markdown -Encoding utf8
    $written = Get-Content -LiteralPath $Markdown -Raw -ErrorAction SilentlyContinue
    if (-not $written -or $written -notmatch [regex]::Escape("run-stamp: $script:runStamp")) {
        throw "the acceptance document at $Markdown was not written by this run"
    }
    Write-Output "  document: $Markdown (run-stamp $script:runStamp)"
  } catch {
    # Recorded in the artifact and pushed into the exit code. A document that
    # could not be rendered is a failed run, not a footnote.
    $script:renderError = $_.Exception.Message
    Set-Field 'document_render_error' $script:renderError
    Save-Record
    Write-Output ''
    Write-Output "  DOCUMENT NOT RENDERED: $script:renderError"
    Write-Output '  The JSON record is complete; the human-readable document is not.'
  }
}

# ---------------------------------------------------------------------------
# Exit code. Skipped work reaches it.
# ---------------------------------------------------------------------------

if ($script:renderError) {
    Write-Output ''
    Write-Output 'RESULT: FAILED -- the acceptance document could not be rendered, so the'
    Write-Output 'human-readable artifact does not reflect this run.'
    exit 1
}
if ($stepsNotMet -gt 0 -or $notMet -gt 0) {
    Write-Output ''
    Write-Output 'RESULT: FAILED -- at least one acceptance step or criterion is NOT MET.'
    exit 1
}
if ($stepsNotEx -gt 0 -or $notEx -gt 0 -or $partial -gt 0) {
    Write-Output ''
    Write-Output 'RESULT: INCOMPLETE -- nothing failed, but work was not exercised or was'
    Write-Output 'only partly covered, and an acceptance run that did not test something'
    Write-Output 'must not exit 0.'
    exit 77
}
Write-Output ''
Write-Output 'RESULT: ACCEPTED -- every step ran and every criterion is met.'
exit 0
