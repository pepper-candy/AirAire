@echo off
REM Copy this file to the GPU VM desktop (C:\Users\klmong\Desktop).
REM It always cds into the project; do not rely on the .bat living inside airaire.
title AirAire - Paper Trader V3 (Continuous)
cd /d C:\Users\klmong\Desktop\airaire

echo ============================================================
echo  V3 paper trader — models\news_gpu_v3_2\best_model.zip
echo  Writes state_v3.pkl only. Does not write V2 state.pkl.
echo.
echo  STOP run_trader.bat (V2) first.
echo  V2 and V3 share the same Futu SIMULATE account.
echo  Running both will double-order and fight over positions.
echo ============================================================
echo.
pause

echo [1/3] Activating virtual environment...
call venv_gpu\Scripts\activate.bat

echo [2/3] Starting Futu OpenD in the background...
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en
echo Log in in the OpenD window, then come back here.
pause

echo [3/3] Starting V3 inference bot (Continuous Mode)...
echo If state_v3.pkl is missing, the book continues from V2 state.pkl leftover.
echo Dashboard push is ON — this overwrites the V2 blotter row.
echo Press CTRL+C to stop.
python -m src.inference_v3 --poll-seconds 60 --model models/news_gpu_v3_2/best_model.zip --push-dashboard

pause
