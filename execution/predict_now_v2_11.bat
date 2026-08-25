@echo off
REM Safe while run_trader.bat (live V2) is running. No orders. No state write.
title AirAire - Predict Now V2.11
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat

echo Starting OpenD if it is not already up...
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en
timeout /t 3 /nobreak > nul

echo Predict only — no trade. HK actions must log >= 0; US may be negative.
python -m src.inference_v2_11 --predict-now
pause
