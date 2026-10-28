@echo off
title RSNA Knee MRI CADx - Presentation Demo
cd /d "%~dp0"

echo ======================================================================
echo    RSNA Knee MRI CADx - Interactive Presentation Workstation
echo ======================================================================
echo.

if exist "D:\Miniconda3\envs\rsna_env\python.exe" (
    "D:\Miniconda3\envs\rsna_env\python.exe" server.py
) else (
    python server.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server exited with error. Press any key to close.
    pause >nul
)
