@echo off
REM Same as replay_day_v2.bat but date locked to 24 Aug 2026.
REM Pick V2 zips, then N / Enter or paste another path to compare.
REM Does not place SIMULATE orders. Does not write state.pkl.
title AirAire - Replay 24 Aug 2026 V2
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.replay_day --family v2 --date 2026-08-24 --interactive --tag v2
pause
