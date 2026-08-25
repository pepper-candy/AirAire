@echo off
REM Optional full walk into models\news_gpu_v2_11. Prefer finetune_v2_11.bat (1 window).
title AirAire - GPU train V2.11
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
echo V2.11 train -^> models\news_gpu_v2_11
echo Warm-start from GPU paper best_model.zip (banner). Not V2.10, not 2026-08-20.
echo Press CTRL+C to stop.
python -m src.train_gpu_v2_11 --device cuda
pause
