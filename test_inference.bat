@echo off
title AirAire - Test (Dry Run)
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
echo Running inference in dry-run mode. Will not place real simulated orders.
python -m src.inference --dry-run --once
pause