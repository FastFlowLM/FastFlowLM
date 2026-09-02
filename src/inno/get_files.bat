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
) else (
    echo No AIE4 runtime closure found; building without the AIE4 feature.
)

echo Done!
