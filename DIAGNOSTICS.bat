@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title VIKRAM 1 - Diagnostics
if not exist ".venv\Scripts\python.exe" (
    echo [VIKRAM 1] No environment found. Run INSTALL.bat first.
    pause
    exit /b 1
)
set "PYTHONPATH=%CD%\src"
".venv\Scripts\python.exe" -m aegis.tools.diagnostics --json artifacts\diagnostics.json
echo.
echo A JSON copy was written to artifacts\diagnostics.json
echo.
pause
exit /b 0
