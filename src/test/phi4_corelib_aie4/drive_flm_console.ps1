# Task 16: drive the interactive `flm run` REPL on Windows, for real.
#
# WHY THIS FILE EXISTS
#
# Tasks 11 and 12 recorded interactive `flm run` generation as unverifiable on
# Windows, and the reason was correct as far as it went: runner/cli_wide.cpp
# reads keystrokes with ReadConsoleInput() against a console input handle, and
# ReadConsoleInput cannot see a redirected stdin pipe. Piping a prompt into
# `flm run` therefore produces a REPL that has entered its input loop and will
# sit there until it is killed. Every attempt to smuggle a prompt in through
# stdin failed for that reason, and none of them was a defect in flm.
#
# But "stdin cannot reach it" is not the same claim as "it cannot be driven".
# ReadConsoleInput reads the console INPUT BUFFER, and that buffer is writable
# by any process attached to the same console, through WriteConsoleInput. So a
# driver can:
#
#   1. FreeConsole() then AllocConsole() to get a console of its own -- an SSH
#      exec session has no usable console, so one has to be created;
#   2. start flm.exe as a child with -NoNewWindow so it INHERITS that console
#      (Start-Process without -NoNewWindow allocates the child a fresh console
#      and the injected keystrokes go to the wrong buffer -- measured, and it
#      looks exactly like the input being ignored);
#   3. inject KEY_EVENT records into CONIN$, which is the same buffer
#      ReadConsoleInput drains; and
#   4. read the completion back out of the console SCREEN buffer with
#      ReadConsoleOutputCharacter, because the child's stdout is the console,
#      not a pipe we could have captured.
#
# This is the product's own interactive path, unmodified: the same binary, the
# same ReadConsoleInput loop, the same rendering. It is not a stub, and it is
# not `flm serve` wearing a CLI costume.
#
# WHAT IT COSTS, STATED PLAINLY
#
# Reading the screen buffer returns the RENDERED screen, not the byte stream.
# Trailing spaces on each row are not recoverable, a line longer than the
# buffer width is wrapped with no marker distinguishing that from a real
# newline, and anything scrolled off the top is gone. The buffer is therefore
# set to 200x9000 up front so a 512-token completion cannot scroll away, and
# every recorded completion in the acceptance document should be read as
# "the text a user would have seen", which is what Step 7 actually asks for.
#
# HOW OUTPUT GETS BACK
#
# After AllocConsole() this process's stdout IS the new console, so anything
# written to the success stream disappears from the SSH session. Every result
# goes to -OutJson instead. That is also why this is a separate script rather
# than a function inside run_real_model_acceptance.ps1: the orchestrator has to
# keep its own stdout.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$FlmExe,

    # Extra arguments after `run <tag>`; e.g. @('-c','4096').
    [string[]]$FlmArgs = @(),

    [Parameter(Mandatory = $true)][string]$ModelTag,

    # One entry per turn. Each is typed at a `>>> ` prompt and followed by
    # Enter. `/bye` is appended by the driver; do not include it.
    [string[]]$Turns = @(),

    # A file with one turn per line, used INSTEAD of -Turns.
    #
    # `powershell.exe -File script.ps1 -Turns "a b" "c d"` does not bind an
    # array: the first quoted value binds to -Turns and the second is offered
    # positionally, the script has no positional parameter, and the process
    # exits 1 having produced nothing. A single turn works and several do not,
    # which is a trap worth closing rather than remembering. Callers that pass
    # more than one turn should use this.
    [string]$TurnsFile = '',

    [Parameter(Mandatory = $true)][string]$OutJson,

    # Seconds to wait for the first `>>> ` prompt. Model load is ~1-2 minutes
    # on this box and a cold file cache makes it worse.
    [int]$LoadTimeoutSec = 900,

    # Seconds to wait for one turn's reply to finish.
    [int]$TurnTimeoutSec = 900,

    # A turn is finished when the screen has been unchanged for this long AND
    # a fresh `>>> ` prompt is present. The idle requirement is what stops a
    # pause between two streamed tokens being read as the end of the reply.
    [int]$IdleSec = 6,

    [int]$BufferWidth = 200,
    [int]$BufferHeight = 9000,

    # Every poll overwrites this file with the current console screen. The only
    # window into a session that is hung: the child's stdout is the console, so
    # there is no pipe to tail and no log to read. Without it a stall is
    # indistinguishable from a slow load -- which is exactly how the first run
    # of this driver was misread.
    [string]$ScreenLog = ''
)

$ErrorActionPreference = 'Stop'

if ($TurnsFile) {
    if (-not (Test-Path $TurnsFile)) { throw "TurnsFile not found: $TurnsFile" }
    # Not Get-Content's array form: a turn may legitimately be an empty line
    # only if the caller wrote one, and -Raw plus an explicit split keeps that
    # decision with the caller instead of with the cmdlet.
    $Turns = @((Get-Content $TurnsFile -Raw) -split "`r?`n" | Where-Object { $_.Length -gt 0 })
}
if ($Turns.Count -eq 0) { throw 'no turns supplied: pass -Turns or -TurnsFile' }

$result = [ordered]@{
    flm_exe          = $FlmExe
    model_tag        = $ModelTag
    turns            = @()
    load_seconds     = $null
    reached_prompt   = $false
    exit_code        = $null
    error            = $null
    screen_final     = ''
    driver           = 'WriteConsoleInput/ReadConsoleOutputCharacter'
    console_screen_info_ok = $null
    console_conin    = $null
    console_conout   = $null
}

function Save-Result {
    $dir = Split-Path -Parent $OutJson
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    ($result | ConvertTo-Json -Depth 8) | Set-Content -Path $OutJson -Encoding utf8
}

Add-Type -Namespace FlmCon -Name Api -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError=true)] public static extern bool AllocConsole();
[DllImport("kernel32.dll", SetLastError=true)] public static extern bool FreeConsole();
[DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetConsoleScreenBufferSize(IntPtr h, COORD size);
[DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetStdHandle(int which, IntPtr handle);

[StructLayout(LayoutKind.Sequential)] public struct COORD { public short X; public short Y; }
[StructLayout(LayoutKind.Sequential)] public struct SMALL_RECT { public short Left, Top, Right, Bottom; }
[StructLayout(LayoutKind.Sequential)] public struct CONSOLE_SCREEN_BUFFER_INFO {
  public COORD dwSize; public COORD dwCursorPosition; public short wAttributes;
  public SMALL_RECT srWindow; public COORD dwMaximumWindowSize; }

[StructLayout(LayoutKind.Explicit)] public struct INPUT_RECORD {
  [FieldOffset(0)] public ushort EventType;
  [FieldOffset(4)] public int bKeyDown;
  [FieldOffset(8)] public ushort wRepeatCount;
  [FieldOffset(10)] public ushort wVirtualKeyCode;
  [FieldOffset(12)] public ushort wVirtualScanCode;
  [FieldOffset(14)] public ushort UnicodeChar;
  [FieldOffset(16)] public uint dwControlKeyState; }

[StructLayout(LayoutKind.Sequential)] public struct SECURITY_ATTRIBUTES {
  public int nLength; public IntPtr lpSecurityDescriptor; public int bInheritHandle; }

[DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
public static extern IntPtr CreateFileW(string name, uint access, uint share, ref SECURITY_ATTRIBUTES sa, uint disp, uint flags, IntPtr tmpl);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool WriteConsoleInputW(IntPtr h, INPUT_RECORD[] recs, uint n, out uint written);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool GetNumberOfConsoleInputEvents(IntPtr h, out uint n);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool GetConsoleScreenBufferInfo(IntPtr h, out CONSOLE_SCREEN_BUFFER_INFO info);
[DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
public static extern bool ReadConsoleOutputCharacterW(IntPtr h, [Out] char[] buf, uint len, COORD coord, out uint read);
'@

$script:conin = [IntPtr]::Zero
$script:conout = [IntPtr]::Zero
$script:child = $null

function New-KeyRecord {
    param([char]$Ch, [uint16]$Vk = 0)
    $r = New-Object FlmCon.Api+INPUT_RECORD
    $r.EventType = 1              # KEY_EVENT
    $r.bKeyDown = 1
    $r.wRepeatCount = 1
    $r.wVirtualKeyCode = $Vk
    $r.wVirtualScanCode = 0
    $r.UnicodeChar = [uint16][char]$Ch
    $r.dwControlKeyState = 0
    return $r
}

# One record per call, and the buffer is drained to empty before the next.
#
# cli_wide.cpp treats "more than one event pending" as a PASTE and takes a
# different code path. Feeding a whole line at once therefore exercises the
# paste handler rather than the keystroke handler, which is not the path a
# user takes. Typing one character at a time keeps GetNumberOfConsoleInputEvents
# at 1, so the ordinary path runs.
function Send-Key {
    param([char]$Ch, [uint16]$Vk = 0)
    $recs = @((New-KeyRecord -Ch $Ch -Vk $Vk))
    $written = 0
    if (-not [FlmCon.Api]::WriteConsoleInputW($script:conin, $recs, [uint32]1, [ref]$written)) {
        throw "WriteConsoleInput failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
    }
    # Wait for the child to consume it, so the next key does not look like a
    # paste. Bounded: if the child is not reading, fall through rather than
    # hanging here forever -- the turn timeout is the real guard.
    for ($i = 0; $i -lt 200; $i++) {
        $n = 0
        [void][FlmCon.Api]::GetNumberOfConsoleInputEvents($script:conin, [ref]$n)
        if ($n -eq 0) { return }
        Start-Sleep -Milliseconds 10
    }
}

function Send-Line {
    param([string]$Text)
    foreach ($c in $Text.ToCharArray()) { Send-Key -Ch $c }
    Send-Key -Ch "`r" -Vk 13      # VK_RETURN
}

function Read-Screen {
    $info = New-Object FlmCon.Api+CONSOLE_SCREEN_BUFFER_INFO
    if (-not [FlmCon.Api]::GetConsoleScreenBufferInfo($script:conout, [ref]$info)) { return '' }
    $cols = [int]$info.dwSize.X
    $rows = [int]$info.dwCursorPosition.Y + 1
    if ($rows -le 0 -or $cols -le 0) { return '' }
    $total = $rows * $cols
    $buf = New-Object char[] $total
    $origin = New-Object FlmCon.Api+COORD
    $origin.X = 0; $origin.Y = 0
    $read = 0
    if (-not [FlmCon.Api]::ReadConsoleOutputCharacterW($script:conout, $buf, [uint32]$total, $origin, [ref]$read)) { return '' }
    $flat = -join $buf
    $lines = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $rows; $i++) {
        $lines.Add(($flat.Substring($i * $cols, $cols)).TrimEnd())
    }
    return ($lines -join "`n")
}

# Wait until the screen stops changing for -IdleSec AND satisfies $Predicate.
#
# Both conditions, deliberately. Idle alone fires during a slow prefill;
# the predicate alone fires the instant the string appears, which for a
# `>>> ` prompt is BEFORE the reply to the previous turn has been printed.
function Wait-ForScreen {
    param(
        [scriptblock]$Predicate,
        [int]$TimeoutSec,
        [string]$Label
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $lastText = ''
    $lastChange = Get-Date
    while ((Get-Date) -lt $deadline) {
        if ($script:child.HasExited) {
            return @{ Ok = $false; Screen = (Read-Screen); Reason = "child exited with $($script:child.ExitCode)" }
        }
        $text = Read-Screen
        if ($ScreenLog) {
            try { Set-Content -Path $ScreenLog -Value "[$Label]`n$text" -Encoding utf8 } catch { }
        }
        if ($text -ne $lastText) { $lastText = $text; $lastChange = Get-Date }
        $idle = ((Get-Date) - $lastChange).TotalSeconds
        if ($idle -ge $IdleSec -and (& $Predicate $text)) {
            return @{ Ok = $true; Screen = $text; Reason = '' }
        }
        Start-Sleep -Milliseconds 500
    }
    return @{ Ok = $false; Screen = (Read-Screen); Reason = "timed out after ${TimeoutSec}s waiting for $Label" }
}

# Count of `>>> ` prompts on screen. Turn N is complete when prompt N+1 shows.
#
# The pattern has NO trailing space, and that is not sloppiness. Read-Screen
# TrimEnd()s every row, so the prompt the user sees as ">>> " arrives here as
# ">>>". Matching '^>>> ' therefore never fires, the driver waits out its whole
# load timeout against a REPL that is sitting at the prompt, and the run reads
# as "the model never loaded". Measured, on the first run of this driver.
function Get-PromptCount {
    param([string]$Text)
    if (-not $Text) { return 0 }
    return ([regex]::Matches($Text, '(?m)^>>>')).Count
}

try {
    [void][FlmCon.Api]::FreeConsole()
    if (-not [FlmCon.Api]::AllocConsole()) {
        $result.error = "AllocConsole failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
        Save-Result
        exit 2
    }

    $access = [uint32]3221225472   # GENERIC_READ | GENERIC_WRITE, spelled in
                                   # decimal because Windows PowerShell 5.1
                                   # parses 0xC0000000 as a negative Int32.

    # bInheritHandle = TRUE, and this is the whole ballgame.
    #
    # .NET's Process.Start with UseShellExecute = false always passes
    # STARTF_USESTDHANDLES and hands the child THIS process's standard
    # handles. If the handles installed by SetStdHandle below were opened
    # without inheritance, the child receives handles it cannot use:
    # ReadConsoleInput fails immediately, the REPL spins on the failure at
    # 100% CPU, and nothing is ever written to the screen buffer this driver
    # reads back. That is precisely the symptom this cost four runs to find,
    # and it is invisible from the outside -- the console is real, the console
    # is readable, and it is simply not the child's.
    $sa = New-Object FlmCon.Api+SECURITY_ATTRIBUTES
    $sa.nLength = [System.Runtime.InteropServices.Marshal]::SizeOf($sa)
    $sa.lpSecurityDescriptor = [IntPtr]::Zero
    $sa.bInheritHandle = 1
    $script:conin = [FlmCon.Api]::CreateFileW('CONIN$', $access, [uint32]3, [ref]$sa, [uint32]3, [uint32]0, [IntPtr]::Zero)
    $script:conout = [FlmCon.Api]::CreateFileW('CONOUT$', $access, [uint32]3, [ref]$sa, [uint32]3, [uint32]0, [IntPtr]::Zero)
    if ($script:conin -eq [IntPtr]::new(-1) -or $script:conout -eq [IntPtr]::new(-1)) {
        $result.error = 'could not open CONIN$/CONOUT$ on the allocated console'
        Save-Result
        exit 2
    }

    # Point this process's STANDARD HANDLES at the new console.
    #
    # AllocConsole only rebinds the standard handles that were not already
    # redirected. When this driver is launched by another script -- which is
    # how the acceptance harness runs it -- its stdin/stdout are pipes, and
    # they stay pipes. Start-Process -NoNewWindow then hands those pipes to
    # flm.exe, so ReadConsoleInput has nothing to read (the REPL spins at 100%
    # CPU on a failing read) and everything flm prints goes into a pipe rather
    # than the console screen buffer this driver reads back. Measured: an
    # empty screen, an flm.exe at 944 seconds of CPU, and a driver that looked
    # like the model had never loaded.
    #
    # Rebinding them explicitly makes the driver behave the same whether it is
    # run from a terminal, from another script, or over SSH.
    [void][FlmCon.Api]::SetStdHandle(-10, $script:conin)   # STD_INPUT_HANDLE
    [void][FlmCon.Api]::SetStdHandle(-11, $script:conout)  # STD_OUTPUT_HANDLE
    [void][FlmCon.Api]::SetStdHandle(-12, $script:conout)  # STD_ERROR_HANDLE

    # Recorded so that a driver that never got a usable console is
    # diagnosable from the artifact rather than from a process listing.
    $probe = New-Object FlmCon.Api+CONSOLE_SCREEN_BUFFER_INFO
    $result.console_screen_info_ok = [bool][FlmCon.Api]::GetConsoleScreenBufferInfo($script:conout, [ref]$probe)
    $result.console_conin = [string]$script:conin
    $result.console_conout = [string]$script:conout
    if (-not $result.console_screen_info_ok) {
        $result.error = 'the allocated console has no readable screen buffer; flm output could not have been captured'
        Save-Result
        exit 2
    }

    $size = New-Object FlmCon.Api+COORD
    $size.X = [int16]$BufferWidth
    $size.Y = [int16]$BufferHeight
    [void][FlmCon.Api]::SetConsoleScreenBufferSize($script:conout, $size)

    $argList = @('run', $ModelTag) + $FlmArgs
    $result.flm_argv = (@($FlmExe) + $argList) -join ' '
    $loadStart = Get-Date

    # ProcessStartInfo directly, NOT Start-Process.
    #
    # flm.exe has to inherit THIS process's console, and the two obvious
    # spellings both fail to deliver that:
    #
    #   Start-Process (plain)        -> the child gets a brand new console, so
    #                                   injected keystrokes go to a buffer
    #                                   nobody is reading;
    #   Start-Process -NoNewWindow   -> Windows PowerShell 5.1 sets
    #                                   CreateNoWindow, and a console
    #                                   application started that way also gets
    #                                   a console of its own rather than the
    #                                   caller's.
    #
    # Measured with -NoNewWindow: an empty screen buffer and an flm.exe at
    # 100% CPU for 150 seconds, spinning on a ReadConsoleInput that can never
    # succeed. UseShellExecute = false with CreateNoWindow = false and no
    # redirection is the combination that actually inherits the console.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FlmExe
    $psi.Arguments = ($argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $false
    $psi.RedirectStandardInput = $false
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError = $false
    $script:child = [System.Diagnostics.Process]::Start($psi)

    $ready = Wait-ForScreen -TimeoutSec $LoadTimeoutSec -Label 'the first >>> prompt' -Predicate {
        param($t) (Get-PromptCount $t) -ge 1
    }
    $result.load_seconds = [math]::Round(((Get-Date) - $loadStart).TotalSeconds, 3)
    $result.reached_prompt = $ready.Ok
    if (-not $ready.Ok) {
        $result.error = "model load / REPL entry: $($ready.Reason)"
        $result.screen_final = $ready.Screen
        Save-Result
        exit 3
    }

    $prevScreen = $ready.Screen
    $turnIndex = 0
    foreach ($turn in $Turns) {
        $turnIndex++
        $expectPrompts = (Get-PromptCount $prevScreen) + 1
        $t0 = Get-Date
        Send-Line -Text $turn
        $done = Wait-ForScreen -TimeoutSec $TurnTimeoutSec -Label "turn $turnIndex to finish" -Predicate {
            param($t) (Get-PromptCount $t) -ge $expectPrompts
        }
        $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 3)
        # The reply is what this turn ADDED to the screen. Taking the delta
        # rather than re-parsing the whole screen keeps a multi-turn session
        # from attributing turn 1's reply to turn 4.
        $newText = $done.Screen
        $reply = if ($newText.StartsWith($prevScreen)) { $newText.Substring($prevScreen.Length) } else { $newText }
        $result.turns += [ordered]@{
            index    = $turnIndex
            prompt   = $turn
            ok       = $done.Ok
            reason   = $done.Reason
            seconds  = $elapsed
            reply_raw = $reply
        }
        $prevScreen = $newText
        if (-not $done.Ok) {
            $result.error = "turn ${turnIndex}: $($done.Reason)"
            $result.screen_final = $newText
            Save-Result
            exit 4
        }
    }

    Send-Line -Text '/bye'
    for ($i = 0; $i -lt 120; $i++) {
        if ($script:child.HasExited) { break }
        Start-Sleep -Milliseconds 500
    }
    $result.screen_final = Read-Screen
    if ($script:child.HasExited) {
        $result.exit_code = $script:child.ExitCode
    } else {
        $result.error = '/bye did not terminate the REPL'
    }
    Save-Result
    if ($result.error) { exit 5 } else { exit 0 }
}
catch {
    $result.error = "$($_.Exception.Message)"
    try { $result.screen_final = Read-Screen } catch { }
    Save-Result
    exit 6
}
finally {
    if ($script:child -and -not $script:child.HasExited) {
        Stop-Process -Id $script:child.Id -Force -ErrorAction SilentlyContinue
    }
}
