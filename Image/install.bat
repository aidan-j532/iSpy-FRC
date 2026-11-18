@echo off
setlocal
title iSpy Installer
cd /d "%~dp0"

echo Installing iSpy (runtime dependencies only)...
python -m pip install -e .
if %errorlevel% neq 0 (
    echo.
    echo Failed to install the iSpy package. Is Python 3.10+ on PATH?
    pause
    exit /b %errorlevel%
)

echo.
echo Running hardware-aware first-time setup...
python -m iSpy.boot.boot -f
if %errorlevel% neq 0 (
    echo.
    echo Setup failed. Fix the error above and rerun install.bat.
    pause
    exit /b %errorlevel%
)

echo.
echo Installation completed successfully!
echo Run: ispy start
pause
