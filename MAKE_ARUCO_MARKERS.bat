@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title AEGIS - Generate Rack Markers
if not exist ".venv\Scripts\python.exe" ( call INSTALL.bat || exit /b 1 )
set "PYTHONPATH=%CD%\src"
".venv\Scripts\python.exe" -m aegis.tools.make_markers %*
echo.
pause
exit /b 0
