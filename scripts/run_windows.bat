@echo off
setlocal enabledelayedexpansion

title PhantomTouch - Gesture-Controlled Invisibility
color 0A

echo =====================================================================
echo                     PHANTOM TOUCH LAUNCHER
echo      Gesture-Controlled Real-Time Webcam Invisibility System
echo =====================================================================
echo.

:: Navigate to project root (one level up from scripts)
cd /d "%~dp0\.."

:: 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your system PATH!
    echo Please install Python 3.11 or later from https://www.python.org/
    echo and ensure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

:: 2. Check virtual environment or dependencies
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call ".venv\Scripts\activate.bat"
)

:: 3. Verify MediaPipe models
if not exist "models\hand_landmarker.task" (
    echo [INFO] Required vision models not found. Downloading models now...
    python scripts\download_models.py
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to download required models.
        pause
        exit /b 1
    )
)

:: 4. Launch Desktop Application
echo.
echo [INFO] Launching PhantomTouch Desktop Application...
echo.
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)

endlocal
