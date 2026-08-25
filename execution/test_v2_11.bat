@echo off
REM No GPU / OpenD. Safe anytime.
title AirAire - V2.11 hybrid unit tests
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
python test\v2_11-hybrid-test.py
pause
