@echo off
REM V4 full walk. Isolated models\news_gpu_v4. Does not touch V2/V3 zips.
REM Run AFTER probe_futu_observers.bat and build_v4_panel.bat.
REM --skip-futu keeps OpenD on the live V3.2 trader.
title AirAire - GPU train V4 (Bloomberg 7-name, volume=0)
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
echo V4 train -^> models\news_gpu_v4
echo Bloomberg OHLC + HSI/SPX observers, volume forced 0, V2 news.
echo Press CTRL+C to stop.
python -m src.train_gpu_v4 --device cuda --skip-futu --output models/news_gpu_v4
pause
