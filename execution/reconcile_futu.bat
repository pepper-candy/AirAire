@echo off
REM Run on the GPU box. Stop V3 first if you will --apply.
title AirAire - Reconcile Futu vs pickle
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
echo Print-only compare (pickle vs Futu vs dashboard):
python -m src.reconcile_futu
echo.
echo If DRIFT and V3 is STOPPED, run:
echo   python -m src.reconcile_futu --apply
pause
