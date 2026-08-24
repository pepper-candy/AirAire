@echo off
REM Build data\enhanced\enhanced_v4.parquet from Bloomberg (volume=0) + V2 news.
REM Optional OpenD overlay for latest OHLC. Does not touch V2/V3 parquets.
title AirAire - Build V4 Bloomberg panel
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.data_loader_v4 --force-rebuild
pause
