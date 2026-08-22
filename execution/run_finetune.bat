@echo off
REM Copy this file to the GPU VM desktop (C:\Users\klmong\Desktop).
REM It always cds into the project; do not rely on the .bat living inside airaire.
title AirAire - Daily Fine-Tune
cd /d C:\Users\klmong\Desktop\airaire

echo [1/3] Activating virtual environment...
call venv_gpu\Scripts\activate.bat

echo [2/3] Starting Futu OpenD (for data fetch)...
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en
echo Log in in the OpenD window, then come back here.
pause

echo [3/3] Futu bars + Alpha Vantage news (last 30 days), then 1-window PPO...
echo Telegram Promote/Keep wait is 10 minutes after a better Calmar.
python -m src.finetune_latest --windows 1 --device cuda

echo.
echo Fine-tune complete. Check the PROMOTION CHECK in the terminal.
echo If you missed the Telegram buttons: python -m src.finetune_latest --promote-zip models\news_gpu_v2\finetuned_YYYY-MM-DD.zip
pause
