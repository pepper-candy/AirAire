@echo off
REM Instant-fill HK session: V2 08-12, V2 best, V3, V3.1, V3.2. OpenD logged in.
REM Does not place SIMULATE orders. Does not write state_v3.pkl.
title AirAire - Replay today V2 vs V3 vs V3.1
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.replay_day
pause
