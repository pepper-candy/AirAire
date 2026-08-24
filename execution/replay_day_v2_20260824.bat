@echo off
REM Copy to the GPU box. Date locked to 2026-08-24. V2 env only.
title AirAire - Replay 24 Aug 2026 V2
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.replay_day --family v2 --date 2026-08-24 --interactive --tag v2
pause
