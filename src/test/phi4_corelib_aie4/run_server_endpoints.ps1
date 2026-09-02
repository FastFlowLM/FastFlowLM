# Task 12 Step 7: the four generation endpoints, on real AIE4 hardware.
#
# Each endpoint is exercised twice: a DEFAULT request carrying no supported
# limit field, and an explicit OVER-LIMIT request that must be refused with
# HTTP 400. The default case is the interesting one. On every other backend an
# absent limit field falls back to kLegacyDefaultGenerationLimit (4096), but
# GenerationLoopLimit returns kNoExplicitGenerationLimit for AIE4, so the
# decode loop is bounded only by EOS.
#
# MEASURED CONSEQUENCE, 2026-09-02: a default /api/chat request once ran the
# decode loop to the context cap and TERMINATED THE SERVER, leaving
#   {"status":3,"call":"flat_mha",
#    "detail":"no token attention kernel ships for a 4096-token window",
#    "phase":"flat_mha","layer":0,"rows":1,"position":4095}
# It did not reproduce on the next run, because whether the model emits EOS
# before position 4095 is data-dependent. That is why this script restarts the
# server when it dies and records the death per case rather than assuming one
# server lifetime -- an earlier version did assume it, and reported the
# remaining eight cases as connection failures against a stale response body.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$FlmExe,
    [string]$ModelTag = 'phi4-mini-it-aie4:4b',
    [int]$Port = 11434,
    [string]$WorkDir = $env:TEMP
)

$ErrorActionPreference = 'Stop'


$port = $Port
$tag = $ModelTag
$work = $WorkDir
$serverOut = Join-Path $work 'flm-serve.log'
$serverErr = Join-Path $work 'flm-serve.err'
$script:results = @()
$script:server = $null

# Logging goes to the console stream, never the output stream: a PowerShell
# function returns everything written to the output stream, so a Write-Output
# used for progress becomes part of the return value.
function Say { param([string]$Text) [Console]::Out.WriteLine($Text) }

function Stop-Flm {
    if ($script:server -and -not $script:server.HasExited) {
        Stop-Process -Id $script:server.Id -Force -ErrorAction SilentlyContinue
    }
    Get-Process flm -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 4
}

function Start-Flm {
    Stop-Flm
    $script:server = Start-Process -FilePath $FlmExe `
        -ArgumentList @('serve', $tag, '-p', "$port") `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr
    for ($i = 0; $i -lt 150; $i++) {
        Start-Sleep -Seconds 4
        if ($script:server.HasExited) {
            throw "flm serve exited during startup with $($script:server.ExitCode)"
        }
        $probe = & curl.exe -s -o NUL -w '%{http_code}' --max-time 10 "http://127.0.0.1:$port/api/tags"
        if ($probe -eq '200') { return }
    }
    throw 'server never became ready'
}

# `$MustContain` is not optional decoration on the 400 cases.
#
# Asserting only the status code means ANY 400 passes: malformed JSON, an
# unknown model, a field name typo in this very script. The report claimed
# the refusals carry the capacity message while nothing checked that it
# did -- the same shape as the C2 over-claim, reappearing one layer down.
# Matching the body is what makes these cases evidence that the AIE4
# capacity rule fired, rather than evidence that something went wrong.
function Invoke-Endpoint {
    param(
        [string]$Name,
        [string]$Path,
        [string]$Body,
        [int]$Expect,
        [string]$MustContain = ''
    )
    if (-not $script:server -or $script:server.HasExited) {
        Say '  (server not running -- restarting)'
        Start-Flm
    }
    $bodyFile = Join-Path $work 'req.json'
    $respFile = Join-Path $work 'resp.json'
    # Deleted every time. Leaving it meant a failed request silently reported
    # the PREVIOUS request's body, which is how eight dead-server cases looked
    # like they had returned a 400.
    Remove-Item $respFile -Force -ErrorAction SilentlyContinue
    Set-Content -Path $bodyFile -Value $Body -Encoding utf8
    # curl.exe, not Invoke-WebRequest: Windows PowerShell 5.1 throws on non-2xx
    # and makes the status code awkward to read, and the 400s are the point.
    $code = & curl.exe -s -o $respFile -w '%{http_code}' `
        -X POST "http://127.0.0.1:$port$Path" `
        -H 'Content-Type: application/json' `
        --data-binary "@$bodyFile" --max-time 900
    $body = if (Test-Path $respFile) { (Get-Content $respFile -Raw) } else { '(no response body)' }
    if ($null -eq $body) { $body = '(empty)' }
    $body = ($body -replace '\s+', ' ').Trim()
    $died = $script:server.HasExited
    $exitCode = if ($died) { '0x{0:X}' -f $script:server.ExitCode } else { '' }
    $bodyOk = ($MustContain -eq '') -or ($body -like "*$MustContain*")
    $ok = ($code -eq "$Expect") -and (-not $died) -and $bodyOk
    Say ("{0,-56} expect {1} got {2}  {3}{4}" -f `
        $Name, $Expect, $code, $(if ($ok) { 'PASS' } else { 'FAIL' }), `
        $(if ($died) { "  SERVER DIED exit=$exitCode" } else { '' }))
    Say ("      " + $body.Substring(0, [Math]::Min(280, $body.Length)))
    if (-not $bodyOk) {
        Say ("      BODY MISMATCH: expected to contain '" + $MustContain + "'")
    }
    $script:results += [pscustomobject]@{
        Name = $Name; Ok = $ok; Code = $code; Died = $died;
        ExitCode = $exitCode; BodyOk = $bodyOk }
}

Say "=== Step 7: flm serve $tag on port $port ==="
try {
    Start-Flm
    Say 'server ready'
    Say ''

    Invoke-Endpoint '/api/generate (default, no limit field)' '/api/generate' `
        "{`"model`":`"$tag`",`"prompt`":`"What is the capital of France?`",`"stream`":false}" 200 -MustContain '"response"'
    Invoke-Endpoint '/api/generate (over-limit max_tokens)' '/api/generate' `
        "{`"model`":`"$tag`",`"prompt`":`"What is the capital of France?`",`"stream`":false,`"max_tokens`":100000}" 400 -MustContain 'exceeds the active context cap'

    Invoke-Endpoint '/api/chat (default, no limit field)' '/api/chat' `
        "{`"model`":`"$tag`",`"messages`":[{`"role`":`"user`",`"content`":`"What is the capital of France?`"}],`"stream`":false}" 200 -MustContain '"message"'
    Invoke-Endpoint '/api/chat (over-limit options.num_predict)' '/api/chat' `
        "{`"model`":`"$tag`",`"messages`":[{`"role`":`"user`",`"content`":`"What is the capital of France?`"}],`"stream`":false,`"options`":{`"num_predict`":100000}}" 400 -MustContain 'exceeds the active context cap'

    Invoke-Endpoint '/v1/chat/completions (default, non-streaming)' '/v1/chat/completions' `
        "{`"model`":`"$tag`",`"messages`":[{`"role`":`"user`",`"content`":`"What is the capital of France?`"}],`"stream`":false}" 200 -MustContain '"choices"'
    Invoke-Endpoint '/v1/chat/completions (default, STREAMING)' '/v1/chat/completions' `
        "{`"model`":`"$tag`",`"messages`":[{`"role`":`"user`",`"content`":`"What is the capital of France?`"}],`"stream`":true}" 200 -MustContain 'data:'
    Invoke-Endpoint '/v1/chat/completions (over-limit max_tokens)' '/v1/chat/completions' `
        "{`"model`":`"$tag`",`"messages`":[{`"role`":`"user`",`"content`":`"Hi`"}],`"stream`":false,`"max_tokens`":100000}" 400 -MustContain 'exceeds the active context cap'
    # The OpenAI chat endpoint accepts a SECOND spelling, parsed only when
    # max_tokens is absent, so it needs its own case.
    Invoke-Endpoint '/v1/chat/completions (over-limit max_completion_tokens)' '/v1/chat/completions' `
        "{`"model`":`"$tag`",`"messages`":[{`"role`":`"user`",`"content`":`"Hi`"}],`"stream`":false,`"max_completion_tokens`":100000}" 400 -MustContain 'exceeds the active context cap'

    Invoke-Endpoint '/v1/completions (default, no limit field)' '/v1/completions' `
        "{`"model`":`"$tag`",`"prompt`":`"What is the capital of France?`",`"stream`":false}" 200 -MustContain '"choices"'
    Invoke-Endpoint '/v1/completions (over-limit max_tokens)' '/v1/completions' `
        "{`"model`":`"$tag`",`"prompt`":`"Hi`",`"stream`":false,`"max_tokens`":100000}" 400 -MustContain 'exceeds the active context cap'
}
finally { Stop-Flm }

Say ''
Say '=== Step 7 summary ==='
foreach ($r in $script:results) {
    Say ("{0,-56} {1}{2}" -f $r.Name, `
        $(if ($r.Ok) { 'PASS' } else { "FAIL (http $($r.Code))" }), `
        $(if ($r.Died) { "  [server terminated $($r.ExitCode)]" } else { '' }))
}
$failed = @($script:results | Where-Object { -not $_.Ok })
$died = @($script:results | Where-Object { $_.Died })
Say "STEP7_TOTAL=$($script:results.Count) STEP7_FAILED=$($failed.Count) STEP7_SERVER_DEATHS=$($died.Count)"
Say '--- fatal records written during this run ---'
Get-ChildItem (Join-Path $env:LOCALAPPDATA 'FastFlowLM\logs\corelib-fatal-*.json') -ErrorAction SilentlyContinue |
    ForEach-Object { Say (Get-Content $_.FullName -Raw) }
if ($failed.Count -gt 0) { exit 1 } else { exit 0 }
