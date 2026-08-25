@echo off
REM V2.10 full walk through 2026-08-21. Isolated models\news_gpu_v2_10.
REM Run AFTER build_v2_10_panel.bat. --skip-futu: OpenD stays with live V2. 24 Aug not in the tape.
title AirAire - GPU train V2.10 (through 21 Aug, fair test 24 Aug)
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
echo V2.10 train -^> models\news_gpu_v2_10  end=2026-08-21
echo Press CTRL+C to stop.
python -m src.train_gpu_v2_10 --device cuda --skip-futu --end 2026-08-21
pause
