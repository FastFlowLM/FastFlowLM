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

REM Copy the validated optional AIE4 runtime closure
if not exist "..\build\aie4\ryzenai_corelib.dll" (
    echo ERROR: Build src with FLM_ENABLE_CORELIB_AIE4=ON before packaging.
    exit /b 1
)
if not exist "aie4" mkdir "aie4"
xcopy "..\build\aie4\*" "aie4\" /E /I /Y

echo Done!
