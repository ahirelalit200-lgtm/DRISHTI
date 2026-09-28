@echo off
REM Launch straight into a running session with streaming enabled.
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title DRISHTI - Live Session
if not exist ".venv\Scripts\python.exe" ( call INSTALL.bat || exit /b 1 )
set "PYTHONPATH=%CD%\src"
".venv\Scripts\python.exe" -m aegis.gui.app --autostart --stream
if errorlevel 1 pause
exit /b 0
