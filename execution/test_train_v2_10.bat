@echo off
REM One-window smoke test. No checkpoints. Isolated from live V2.
title AirAire - GPU V2.10 --test
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python -m src.train_gpu_v2_10 --device cuda --skip-futu --end 2026-08-21 --test
pause
