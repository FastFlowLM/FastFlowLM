# Reads the version out of a ryzenai_corelib.dll by calling into it.
#
# Prints exactly one line, `MAJOR.MINOR.PATCH`, on success. Anything else is a
# failure and exits non-zero with the reason on stderr.
#
# The version is read from the LOADED library rather than from the headers the
# product compiled against, because the question this answers in the field is
# "which runtime produced this result", and the headers cannot answer it. The
# same reasoning gave every acceptance record the resolved DLL's SHA-256: the
# run-to-run divergence investigation was crippled for days because nothing
# recorded which library had run.
#
# Known and deliberately not hidden: through the whole 0.x series
# `ryzenai_corelib_get_version` returns a hard-coded 0.1.0, so the version
# alone does not identify a build. The SHA-256 recorded beside it is the field
# that does. The version is still worth carrying because it is the only
# self-describing identity the ABI offers, and a future 1.x that actually moves
# it will be readable by a consumer that already knows where to look.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Dll
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Dll -PathType Leaf)) {
    [Console]::Error.WriteLine("ReadCorelibVersion: not a file: $Dll")
    exit 1
}
$resolved = (Resolve-Path -LiteralPath $Dll).Path

# Add-Type compiles C#, the C# compiler reads LIB and INCLUDE, and PowerShell
# runs it with warnings-as-errors. Under MSBuild those variables are set to the
# MSVC toolchain's paths, one of which is relative -- so this probe failed with
#
#   Warning as Error: Invalid search path 'lib\um\x64' specified in
#   'LIB environment variable'
#
# and took the whole flm build down with it, because the staging step it runs
# from is a POST_BUILD command. It passed every earlier test because those ran
# it from a plain shell, where LIB is unset. Nothing here needs either
# variable, so they are cleared for the compile and restored afterwards.
$savedLib = $env:LIB
$savedInclude = $env:INCLUDE
try {
    Remove-Item Env:LIB -ErrorAction SilentlyContinue
    Remove-Item Env:INCLUDE -ErrorAction SilentlyContinue

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class FlmCorelibVersionProbe {
    // LOAD_WITH_ALTERED_SEARCH_PATH (0x8) makes the loader resolve this
    // module's own imports out of the directory it was loaded from, which is
    // the staged closure. Without it the imports are resolved against the
    // PowerShell host's directory and the load fails with Win32 126 even
    // though the closure beside the DLL is complete.
    public const uint LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008;

    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr LoadLibraryEx(string path, IntPtr file, uint flags);

    [DllImport("kernel32", CharSet = CharSet.Ansi, SetLastError = true,
               BestFitMapping = false, ThrowOnUnmappableChar = true)]
    public static extern IntPtr GetProcAddress(IntPtr module, string name);

    [DllImport("kernel32", SetLastError = true)]
    public static extern bool FreeLibrary(IntPtr module);

    // extern "C" void ryzenai_corelib_get_version(
    //     uint32_t* major, uint32_t* minor, uint32_t* patch);
    // x64 Windows has a single native calling convention, so Cdecl here is
    // exact rather than an assumption that happens to work.
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void GetVersion(out uint major, out uint minor, out uint patch);
}
"@
} finally {
    if ($null -ne $savedLib) { $env:LIB = $savedLib }
    if ($null -ne $savedInclude) { $env:INCLUDE = $savedInclude }
}

$module = [FlmCorelibVersionProbe]::LoadLibraryEx(
    $resolved, [IntPtr]::Zero,
    [FlmCorelibVersionProbe]::LOAD_WITH_ALTERED_SEARCH_PATH)
if ($module -eq [IntPtr]::Zero) {
    $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    [Console]::Error.WriteLine(
        "ReadCorelibVersion: LoadLibraryEx failed with Win32 error $code for " +
        "$resolved. The staged closure beside it is incomplete.")
    exit 1
}
try {
    $entry = [FlmCorelibVersionProbe]::GetProcAddress(
        $module, 'ryzenai_corelib_get_version')
    if ($entry -eq [IntPtr]::Zero) {
        $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        [Console]::Error.WriteLine(
            "ReadCorelibVersion: $resolved exports no " +
            "ryzenai_corelib_get_version (Win32 error $code)")
        exit 1
    }
    $call = [Runtime.InteropServices.Marshal]::GetDelegateForFunctionPointer(
        $entry, [Type][FlmCorelibVersionProbe+GetVersion])
    $major = 0
    $minor = 0
    $patch = 0
    $call.Invoke([ref]$major, [ref]$minor, [ref]$patch)
} finally {
    [void][FlmCorelibVersionProbe]::FreeLibrary($module)
}

Write-Output ("{0}.{1}.{2}" -f $major, $minor, $patch)
