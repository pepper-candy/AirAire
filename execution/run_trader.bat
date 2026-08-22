@echo off
REM Copy this file to the GPU VM desktop (C:\Users\klmong\Desktop).
REM It always cds into the project; do not rely on the .bat living inside airaire.
title AirAire - Paper Trader (Continuous)
cd /d C:\Users\klmong\Desktop\airaire

echo [1/3] Activating virtual environment...
call venv_gpu\Scripts\activate.bat

echo [2/3] Starting Futu OpenD in the background...
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en

echo Waiting 5 seconds for OpenD to initialize...
timeout /t 5 /nobreak > nul

echo [3/3] Starting inference bot (Continuous Mode)...
echo Press CTRL+C to stop.
python -m src.inference --poll-seconds 60

pause
