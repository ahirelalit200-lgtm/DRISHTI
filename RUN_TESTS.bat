@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title AEGIS - Test Suite
if not exist ".venv\Scripts\python.exe" ( call INSTALL.bat || exit /b 1 )
set "PYTHONPATH=%CD%\src"
echo [AEGIS] Installing test dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --quiet pytest onnx
echo.
echo ============================================================
echo   ACCEPTANCE TESTS - no camera or GPU required
echo ============================================================
echo.
".venv\Scripts\python.exe" -m pytest tests -v
echo.
pause
exit /b 0
