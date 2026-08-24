@echo off
REM V3 predict-now. Safe while run_trader.bat (V2) is running.
REM No orders. Does not write state.pkl or state_v3.pkl.
title AirAire - Predict Now V3
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat

echo V3 predict only — no trade, does not write state.pkl / state_v3.pkl. Reads V2 leftover if state_v3.pkl is missing.
python -m src.inference_v3 --predict-now --model models/news_gpu_v3_1/best_model.zip
pause
