@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title VIKRAM 1 - Mission Console

if not exist ".venv\Scripts\python.exe" (
    echo [VIKRAM 1] First launch - running the installer...
    call INSTALL.bat
    if errorlevel 1 exit /b 1
)

set "PYTHONPATH=%CD%\src"
echo [VIKRAM 1] Launching mission console...
".venv\Scripts\python.exe" -m aegis.gui.app %*
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo [VIKRAM 1] The console exited with an error.
echo         Run DIAGNOSTICS.bat for a full environment report.
echo.
pause
exit /b 1

