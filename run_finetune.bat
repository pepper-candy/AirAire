@echo off
title AirAire - Daily Fine-Tune
cd /d C:\Users\klmong\Desktop\airaire

echo [1/3] Activating virtual environment...
call venv_gpu\Scripts\activate.bat

echo [2/3] Starting Futu OpenD (for data fetch)...
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en
timeout /t 5 /nobreak > nul

echo [3/3] Running Fine-Tune (Latest 1 Window)...
python -m src.finetune_latest --windows 1 --device cuda

echo.
echo Fine-tune complete! Check logs/ for details.
pause