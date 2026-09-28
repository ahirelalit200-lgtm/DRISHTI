@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title AEGIS AI-HAR - Installer

echo ============================================================
echo   AEGIS  //  AI-HAR for On-board BAS Experiments
echo   Installer
echo ============================================================
echo.

REM ---------- locate a suitable Python ----------------------------------
set "PY_CMD="
for %%V in (3.11 3.12 3.10) do (
    if not defined PY_CMD (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 set "PY_CMD=py -%%V"
    )
)
if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<(3,13) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY_CMD=python"
    )
)
if not defined PY_CMD (
    echo [ERROR] No suitable Python found.
    echo.
    echo   Install 64-bit Python 3.11 from https://www.python.org/downloads/
    echo   During setup you MUST tick:
    echo      [x] Add python.exe to PATH
    echo      [x] tcl/tk and IDLE
    echo.
    pause
    exit /b 1
)
echo [1/5] Using: %PY_CMD%
%PY_CMD% -c "import sys;print('      '+sys.version.split()[0]+' at '+sys.executable)"

REM ---------- virtual environment ---------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [2/5] Creating isolated environment in .venv ...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/5] Reusing existing .venv
)
set "VPY=.venv\Scripts\python.exe"

REM ---------- dependencies ----------------------------------------------
echo [3/5] Installing dependencies ^(this can take 3-6 minutes^)...
"%VPY%" -m pip install --disable-pip-version-check --quiet --upgrade pip setuptools wheel
if errorlevel 1 goto :piperror
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :piperror

REM ---------- MediaPipe model bundles (only if this wheel needs them) ----
echo [4/5] Checking the landmark backend...
set "PYTHONPATH=%CD%\src"
"%VPY%" -c "import mediapipe as mp; raise SystemExit(0 if hasattr(mp,'solutions') else 3)" >nul 2>nul
if errorlevel 3 (
    echo       This MediaPipe build needs the .task model bundles. Downloading...
    "%VPY%" -m aegis.tools.fetch_models
) else (
    echo       Classic MediaPipe solutions available - no extra downloads needed.
)

REM ---------- verify ------------------------------------------------------
echo [5/5] Running diagnostics...
echo.
"%VPY%" -m aegis.tools.diagnostics --json artifacts\install_report.json
set "DIAG=%errorlevel%"

echo.
if "%DIAG%"=="2" (
    echo [FAILED] The environment is not usable. See the FAIL items above.
    pause
    exit /b 2
)
echo ============================================================
echo   INSTALL COMPLETE
echo ============================================================
echo.
echo   Next steps:
echo     1. CALIBRATE_ZONES.bat   define the rack and the work zones
echo     2. START.bat             run the mission console
echo.
echo   To train your own model ^(much better accuracy^):
echo     3. RECORD_DATASET.bat    record ~30 min of labelled clips
echo     4. TRAIN_MODEL.bat       train and export it
echo.
pause
exit /b 0

:piperror
echo.
echo [ERROR] Dependency installation failed.
echo   Common causes:
echo     - no internet connection
echo     - corporate proxy blocking PyPI
echo     - Python 3.13+ ^(MediaPipe has no wheel yet^); install Python 3.11
echo.
pause
exit /b 1
