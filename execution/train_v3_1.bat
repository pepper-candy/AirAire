@echo off
REM V3.1: same V3 env, recent slice only. Does not overwrite models/news_gpu_v3.
title AirAire - GPU train V3.1 (Jun 15 - Aug 21)
cd /d C:\Users\klmong\Desktop\airaire

echo [1/2] Activating virtual environment...
call venv_gpu\Scripts\activate.bat

echo [2/2] V3.1 slice train -^> models\news_gpu_v3_1
echo Uses cached enhanced_v3.parquet. --skip-futu so OpenD stays with the V2 trader.
echo Press CTRL+C to stop.
python -m src.train_gpu_v3 --device cuda --start 2026-06-15 --end 2026-08-21 --output models/news_gpu_v3_1 --skip-futu

pause
