@echo off
REM Copy this file to the GPU VM desktop (C:\Users\klmong\Desktop).
REM Double-click anytime, including while run_trader.bat is running.
REM Predict only: live quotes + news. No Futu orders. Does not write state.pkl.
title AirAire - Predict Now
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat

echo Starting OpenD if it is not already up...
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en
timeout /t 3 /nobreak > nul

echo Predict only — no trade.
python -m src.inference --predict-now
pause
