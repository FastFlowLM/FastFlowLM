@echo off
echo Copying files to inno folder...

REM Copy flm.exe from build folder
echo Copying flm.exe...
copy "..\build\flm.exe" "flm.exe"

REM Copy all DLL files from lib folder
echo Copying DLL files...
copy "..\lib\*.dll" "."
copy "..\lib\xrt\*.dll" "."

REM Copy model_list.json from root
echo Copying model_list.json...
copy "..\model_list.json" "model_list.json"
copy "..\model_info.json" "model_info.json"

REM Copy the optional, derived AIE4 runtime closure when one has been staged.
REM The AIE4 feature is optional, so a missing closure is a skip and not an
REM error: requiring it here would make the AIE4 build a precondition of
REM shipping the ordinary NPU2 product.
if exist "..\build\aie4\ryzenai_corelib.dll" (
    echo Copying optional AIE4 runtime closure...
    if not exist "aie4" mkdir "aie4"
    xcopy "..\build\aie4\*" "aie4\" /E /I /Y
    REM The closure report ships beside the runtime. It is the only thing in
    REM an installed tree that says which corelib produced a given result, so
    REM it is copied deliberately and its arrival is checked -- the wholesale
    REM xcopy above already carries it, and that is exactly why: an incidental
    REM dependency breaks silently the day somebody narrows the copy.
    copy /Y "..\build\aie4\aie4-closure.txt" "aie4\aie4-closure.txt" >nul
    if not exist "aie4\aie4-closure.txt" goto :aie4_missing_closure_report
) else (
    echo No AIE4 runtime closure found; building without the AIE4 feature.
)

echo Done!
exit /b 0

REM Reported by jumping out of the block rather than by exiting inside it.
REM cmd.exe discards the exit code of an `exit /b` that runs inside a
REM parenthesised block containing redirections -- the diagnostic prints,
REM the batch stops, and the caller still sees 0. Verified on the target:
REM the same nested `exit /b 1` returns 1 without the redirected lines and
REM 0 with them.
:aie4_missing_closure_report
echo ERROR: the staged AIE4 closure has no aie4-closure.txt, so the
echo        installed runtime would carry no provenance. Re-run the AIE4
echo        staging step: cmake -P src/cmake/StageAie4Runtime.cmake
exit /b 1
