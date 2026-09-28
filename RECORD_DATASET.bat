@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title AEGIS - Dataset Recorder
if not exist ".venv\Scripts\python.exe" ( call INSTALL.bat || exit /b 1 )
set "PYTHONPATH=%CD%\src"

echo ============================================================
echo   DATASET RECORDER
echo ============================================================
echo.
echo   Record 12-20 clips per action, with at least 3 different
echo   people. Keep one person entirely out of training so you
echo   have an honest accuracy number.
echo.
set "OPERATOR="
set /p OPERATOR=Operator ID (e.g. op1, op2, op3) [op1]: 
if "!OPERATOR!"=="" set "OPERATOR=op1"
echo.
echo   Keys: 1-9 pick label   TAB cycle   i idle
echo         SPACE start/stop clip   d delete last   q finish
echo.
".venv\Scripts\python.exe" -m aegis.tools.record_dataset --operator "!OPERATOR!" %*
echo.
pause
exit /b 0
