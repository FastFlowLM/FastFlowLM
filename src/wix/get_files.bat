@echo off
echo Copying files to wix\package folder...

if not exist package mkdir package

REM Copy flm.exe from build folder
echo Copying flm.exe...
copy "..\build\flm.exe" "package\flm.exe"

REM Copy all DLL files from lib folder
echo Copying DLL files...
copy "..\lib\*.dll" "package\"
copy "..\lib\xrt\*.dll" "package\"

REM Copy model_list.json and model_info.json from root
echo Copying model_list.json...
copy "..\model_list.json" "package\model_list.json"
copy "..\model_info.json" "package\model_info.json"

REM Copy static installer assets
echo Copying static assets...
copy "..\inno\logo.ico" "package\logo.ico"
copy "..\inno\terms.rtf" "package\terms.rtf"

REM Copy the optional, derived AIE4 runtime closure when one has been staged.
REM The AIE4 feature is optional, so a missing closure is a skip and not an
REM error: requiring it here would make the AIE4 build a precondition of
REM shipping the ordinary NPU2 product.
if exist "..\build\aie4\ryzenai_corelib.dll" (
    echo Copying optional AIE4 runtime closure...
    if not exist "package\aie4" mkdir "package\aie4"
    xcopy "..\build\aie4\*" "package\aie4\" /E /I /Y
    REM The closure report ships beside the runtime. It is the only thing in
    REM an installed tree that says which corelib produced a given result, so
    REM it is copied deliberately and its arrival is checked -- the wholesale
    REM xcopy above already carries it, and that is exactly why: an incidental
    REM dependency breaks silently the day somebody narrows the copy.
    copy /Y "..\build\aie4\aie4-closure.txt" "package\aie4\aie4-closure.txt" >nul
    if not exist "package\aie4\aie4-closure.txt" goto :aie4_missing_closure_report
    > "package\aie4.wxi" echo ^<Include xmlns="http://wixtoolset.org/schemas/v4/wxs"^>
    >> "package\aie4.wxi" echo   ^<ComponentGroup Id="Aie4RuntimeComponents" Directory="AIE4FOLDER"^>
    >> "package\aie4.wxi" echo     ^<Files Include="aie4\**" /^>
    >> "package\aie4.wxi" echo   ^</ComponentGroup^>
    >> "package\aie4.wxi" echo ^</Include^>
) else (
    echo No AIE4 runtime closure found; building without the AIE4 feature.
    if not exist "package\aie4" mkdir "package\aie4"
    > "package\aie4.wxi" echo ^<Include xmlns="http://wixtoolset.org/schemas/v4/wxs"^>
    >> "package\aie4.wxi" echo   ^<ComponentGroup Id="Aie4RuntimeComponents" Directory="AIE4FOLDER" /^>
    >> "package\aie4.wxi" echo ^</Include^>
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
