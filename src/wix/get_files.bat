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
    > "package\aie4.wxi" echo ^<Include^>
    >> "package\aie4.wxi" echo   ^<ComponentGroup Id="Aie4RuntimeComponents" Directory="AIE4FOLDER"^>
    >> "package\aie4.wxi" echo     ^<Files Include="package\aie4\**" /^>
    >> "package\aie4.wxi" echo   ^</ComponentGroup^>
    >> "package\aie4.wxi" echo ^</Include^>
) else (
    echo No AIE4 runtime closure found; building without the AIE4 feature.
    if not exist "package\aie4" mkdir "package\aie4"
    > "package\aie4.wxi" echo ^<Include^>
    >> "package\aie4.wxi" echo   ^<ComponentGroup Id="Aie4RuntimeComponents" Directory="AIE4FOLDER" /^>
    >> "package\aie4.wxi" echo ^</Include^>
)

echo Done!
