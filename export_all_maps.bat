@echo off
rem ===========================================================================
rem  export_all_maps.bat - export every World War Z map for the viewer, then
rem  open the viewer. Double-click it; everything else is automatic:
rem
rem    1. creates the Python environment (.venv) and installs the extractor,
rem       the first time only;
rem    2. finds the game's Content folder (config\paths.toml), and asks for it
rem       only when it is not in a usual place;
rem    3. starts the local viewer server in its own minimized window, so maps
rem       can be looked at as soon as they are exported;
rem    4. exports every map with its real geometry, materials and baked
rem       lighting into output\maps\ (about 1.5 hours and 11 GB the first time).
rem       Maps already exported by the same extractor code are skipped, so
rem       running it again after an interruption carries on where it stopped
rem       (after an extractor update every map is redone);
rem    5. opens the viewer in the browser.
rem
rem  The game folder is only ever read. Progress and full error details also
rem  go to output\export.log. A map that fails does not stop the others.
rem
rem  Options:
rem    export_all_maps.bat --fresh     re-export every map, even finished ones
rem    export_all_maps.bat --no-view   do not start the viewer afterwards
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title WWZ map export

set "RESUME=--resume"
set "VIEW=1"
:parse_options
if "%~1"=="" goto :options_done
if /i "%~1"=="--fresh" (
    set "RESUME="
) else if /i "%~1"=="--no-view" (
    set "VIEW=0"
) else (
    echo Unknown option %1. Use --fresh and/or --no-view.
    goto :fail
)
shift
goto :parse_options
:options_done

set "VENV_PY=.venv\Scripts\python.exe"

rem --- 1. Python environment ------------------------------------------------
if exist "%VENV_PY%" goto :venv_ready
echo Setting up the Python environment (first run only)...
call :find_python
if not defined PYTHON (
    echo.
    echo Python 3.11 or newer is needed. Install it from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH"^), then run this file again.
    goto :fail
)
%PYTHON% -m venv .venv
if errorlevel 1 (
    echo Could not create the Python environment in .venv
    goto :fail
)
:venv_ready
"%VENV_PY%" -c "import numpy, PIL, scipy, extractor" >nul 2>&1
if not errorlevel 1 goto :packages_ready
echo Installing the extractor and its packages (numpy, Pillow, SciPy)...
"%VENV_PY%" -m pip install --disable-pip-version-check --quiet -e .
if errorlevel 1 (
    echo Installing the packages failed; see the messages above. An internet connection is needed once.
    goto :fail
)
:packages_ready

rem --- 2. Game folder ---------------------------------------------------------
if not exist "config\paths.toml" copy /y "config\paths.example.toml" "config\paths.toml" >nul
"%VENV_PY%" -m extractor maps >nul 2>&1
if not errorlevel 1 goto :game_ready
call :detect_game
if defined CONTENT goto :write_config
echo.
echo The game's Content folder was not found.
:ask_game
set "CONTENT="
set /p "CONTENT=Paste the path of World War Z's Content folder (it contains client and server): "
if not defined CONTENT goto :ask_game
set "CONTENT=%CONTENT:"=%"
if exist "%CONTENT%\client\" goto :write_config
if exist "%CONTENT%\Content\client\" (
    set "CONTENT=%CONTENT%\Content"
    goto :write_config
)
echo That folder has no client subfolder. Try again.
goto :ask_game
:write_config
echo Using the game at %CONTENT%
rem TOML literal strings ('...') keep Windows backslashes as they are.
> "config\paths.toml" echo content_dir = '%CONTENT%'
>> "config\paths.toml" echo output_dir = "output"
"%VENV_PY%" -m extractor maps >nul
if errorlevel 1 (
    echo The extractor still cannot read the game; see the message above.
    goto :fail
)
:game_ready

rem --- 3. Viewer server --------------------------------------------------------
rem Started first, in its own window, so finished maps can be viewed while the
rem rest export. If a viewer server is already running, this one just exits.
if "%VIEW%"=="1" start "WWZ viewer server" /min "%VENV_PY%" serve.py

rem --- 4. Export every map ----------------------------------------------------
"%VENV_PY%" -c "import shutil, sys; sys.exit(shutil.disk_usage('.').free < 12 * 2**30)"
if errorlevel 1 (
    echo.
    echo Warning: less than 12 GB of free disk space. A full export needs about 11 GB.
    echo Press a key to continue anyway, or close this window to stop.
    pause >nul
)
echo.
if defined RESUME (
    echo Exporting every map. Maps finished by an earlier run are skipped.
) else (
    echo Exporting every map from scratch.
)
echo Each map takes 1 to 3 minutes; the PC is kept awake meanwhile.
if "%VIEW%"=="1" echo Finished maps can already be viewed at http://127.0.0.1:8765/viewer/
echo.
"%VENV_PY%" -m extractor export --all --log %RESUME%
set "EXPORT_RESULT=%errorlevel%"
if "%EXPORT_RESULT%"=="1" (
    echo.
    echo Some maps failed; the list is above and the details are in output\export.log.
    echo The other maps were exported and can be viewed.
) else if not "%EXPORT_RESULT%"=="0" (
    echo.
    echo The export stopped with an error ^(code %EXPORT_RESULT%^); see the message above.
    echo Maps exported before it stopped can still be viewed.
)

rem --- 5. Open the viewer -----------------------------------------------------
if "%VIEW%"=="0" goto :done
"%VENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/viewer/', timeout=5)" >nul 2>&1
if errorlevel 1 goto :serve_here
start "" "http://127.0.0.1:8765/viewer/"
echo.
echo The viewer is open in your browser. Its server runs in the minimized
echo "WWZ viewer server" window; close that window when you are done.
echo.
echo Press a key to close this window.
pause >nul
goto :done

:serve_here
rem The background server is not answering (closed, or blocked): serve from this window instead.
echo.
echo Starting the viewer at http://127.0.0.1:8765/viewer/
echo Keep this window open while you use it; close it to stop the viewer.
"%VENV_PY%" serve.py --open
goto :done

rem --- helpers ----------------------------------------------------------------
:find_python
rem The "py" launcher first, then python on PATH (skipping the Microsoft Store stub).
set "PYTHON="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py -3"
    goto :eof
)
python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 set "PYTHON=python"
goto :eof

:detect_game
rem Usual install places (Xbox app / Game Pass, Steam, Epic) on every drive.
set "CONTENT="
for %%D in (C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
    if not defined CONTENT if exist "%%D:\" call :try_root "%%D:"
)
goto :eof

:try_root
for %%P in (
    "%~1\Games\World War Z\Content"
    "%~1\XboxGames\World War Z\Content"
    "%~1\XboxGames\World War Z\Content\Content"
    "%~1\SteamLibrary\steamapps\common\World War Z\Content"
    "%~1\Program Files (x86)\Steam\steamapps\common\World War Z\Content"
    "%~1\Program Files\Steam\steamapps\common\World War Z\Content"
    "%~1\Program Files\Epic Games\WorldWarZ\Content"
    "%~1\Epic Games\WorldWarZ\Content"
) do (
    if not defined CONTENT if exist "%%~P\client\" set "CONTENT=%%~P"
)
goto :eof

:fail
echo.
pause
exit /b 1

:done
endlocal
exit /b 0
