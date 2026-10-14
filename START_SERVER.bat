@echo off
title Deepfake Audio Detector Launcher
echo ============================================================
echo   Restarting Deepfake Audio Detector Server...
echo ============================================================
echo.
echo Stopping any old server running on port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING') do (
    echo Terminating PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 >nul
echo.
echo Starting Deepfake Audio Detector...
echo Open http://localhost:5000 in your browser.
echo.
python run.py
pause
