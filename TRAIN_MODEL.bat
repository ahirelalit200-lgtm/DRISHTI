@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title AEGIS - Train Action Model
if not exist ".venv\Scripts\python.exe" ( call INSTALL.bat || exit /b 1 )
set "PYTHONPATH=%CD%\src"

echo [AEGIS] Ensuring training dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --quiet onnx
echo.
echo ============================================================
echo   TRAINING THE TEMPORAL ACTION MODEL
echo   Validation holds out an entire operator - the honest test.
echo ============================================================
echo.
".venv\Scripts\python.exe" -m aegis.tools.train_action_model %*
echo.
pause
exit /b 0
