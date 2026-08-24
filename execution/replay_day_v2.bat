@echo off
REM Copy to the GPU box (C:\Users\klmong\Desktop\airaire) or run from replay\.
REM V2-only interactive: date, then model zip, then optional extra zips.
title AirAire - Replay V2 pick date + models
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.replay_day --family v2 --interactive --tag v2
pause
