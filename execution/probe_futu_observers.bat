@echo off
REM Quote-only: try HSI / SPX / SPY on OpenD. No orders. No pickle write.
REM Safe while run_trader_v3.bat is running.
title AirAire - Probe Futu HSI/SPX
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.probe_futu_observers
pause
