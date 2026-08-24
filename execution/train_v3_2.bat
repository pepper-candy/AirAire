@echo off
REM V3.2: long-only on the same V3 parquet / dates as V3.1. Does not overwrite V2/V3/V3.1.
title AirAire - GPU train V3.2 long-only (Jun 15 - Aug 21)
cd /d C:\Users\klmong\Desktop\airaire

echo [1/2] Activating virtual environment...
call venv_gpu\Scripts\activate.bat

echo [2/2] V3.2 long-only slice train -^> models\news_gpu_v3_2
echo Same tape as V3.1. Actions [0,1]. Cached enhanced_v3.parquet. --skip-futu.
echo Press CTRL+C to stop.
python -m src.train_gpu_v3_2 --device cuda --skip-futu

pause
