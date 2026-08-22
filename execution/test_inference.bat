@echo off
REM Copy this file to the GPU VM desktop (C:\Users\klmong\Desktop).
REM It always cds into the project; do not rely on the .bat living inside airaire.
title AirAire - Test (Dry Run)
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
echo Running inference in dry-run mode. No Futu SIMULATE orders.
echo Loads models\news_gpu_v2\best_model.zip and seeks the env to now.
python -m src.inference --dry-run --once
pause
