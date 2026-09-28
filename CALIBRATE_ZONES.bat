@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title AEGIS - Calibration
if not exist ".venv\Scripts\python.exe" ( call INSTALL.bat || exit /b 1 )
set "PYTHONPATH=%CD%\src"
echo ============================================================
echo   ZONE CALIBRATION
echo ============================================================
echo.
echo   Stage 1: click the 4 corners of the payload rack
echo   Stage 2: draw a polygon around each work zone
echo.
echo   Keys:  LEFT CLICK add   RIGHT CLICK undo   ENTER accept
echo          R restart   S skip   F freeze video   Q save+quit
echo.
".venv\Scripts\python.exe" -m aegis.tools.calibrate %*
echo.
pause
exit /b 0
