@echo off
REM Chatterbox TTS Demo Launcher (Windows)

echo 🎭 Starting Chatterbox TTS Demo Menu...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python.
    pause
    exit /b 1
)

REM Check if demo_menu.py exists
if not exist "demo_menu.py" (
    echo ❌ demo_menu.py not found. Please ensure the file is in the current directory.
    pause
    exit /b 1
)

REM Run the demo menu
python demo_menu.py

pause