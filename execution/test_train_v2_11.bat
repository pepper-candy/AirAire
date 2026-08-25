@echo off
REM One-window smoke test. No checkpoints into live V2.
title AirAire - GPU V2.11 --test
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
echo V2.11 --test (one window, no save). Seed = GPU paper zip.
python -m src.train_gpu_v2_11 --device cuda --test
pause
