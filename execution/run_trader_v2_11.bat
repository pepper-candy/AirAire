@echo off
REM Isolated V2.11 paper trader. STOP run_trader.bat (live V2) first.
REM Same Futu SIMULATE account — running both will double-order.
title AirAire - Paper Trader V2.11 (HK-long / US-short)
cd /d C:\Users\klmong\Desktop\airaire

echo ============================================================
echo  V2.11 paper trader — models\news_gpu_v2_11\best_model.zip
echo  Writes state_v2_11.pkl only. Does not write V2 state.pkl
echo  or models\news_gpu_v2.
echo.
echo  STOP run_trader.bat (V2) first.
echo  V2 and V2.11 share the same Futu SIMULATE account.
echo ============================================================
echo.
pause

echo [1/3] Activating virtual environment...
call venv_gpu\Scripts\activate.bat

echo [2/3] Starting Futu OpenD in the background...
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en
echo Log in in the OpenD window, then come back here.
pause

echo [3/3] Starting V2.11 inference bot...
echo Confirm the startup banner zip matches the GPU trader banner.
echo Terminal copy is also written to logs\trader_v2_11_YYYYMMDD_HHMMSS.txt
echo Press CTRL+C to stop.
python -m src.inference_v2_11 --poll-seconds 60

pause
