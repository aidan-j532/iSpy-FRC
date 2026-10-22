@echo off
title iSpy Installer
echo Pre-checking platform requirements...
python install.py
if %errorlevel% neq 0 (
    echo.
    echo Installation failed with exit code %errorlevel%.
    pause
    exit /b %errorlevel%
)
echo.
echo Installation completed successfully!
pause