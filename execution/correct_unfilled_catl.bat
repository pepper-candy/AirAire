@echo off
REM Run on the GPU box AFTER stopping the V3 trader.
REM Restores unfilled CATL sells in state_v3.pkl + dashboard.
title AirAire - Correct unfilled CATL sells
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.dashboard_push --correct-unfilled
pause
