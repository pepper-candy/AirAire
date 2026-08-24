@echo off
REM V2-only interactive replay. OpenD logged in.
REM Choose date, pick a zip, then N / Enter or paste another zip to compare.
REM Does not place SIMULATE orders. Does not write state.pkl.
title AirAire - Replay V2 pick date + models
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.replay_day --family v2 --interactive --tag v2
pause
