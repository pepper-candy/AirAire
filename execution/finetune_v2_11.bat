@echo off
REM Daily 1-window V2.11 fine-tune after US close. Isolated models\news_gpu_v2_11.
REM Does not write models\news_gpu_v2. Seed = GPU trader banner zip.
title AirAire - Fine-tune V2.11
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat

echo [1/3] Activating venv...
echo [2/3] Starting Futu OpenD (for data fetch)...
start "" "C:\Users\klmong\Desktop\Futu_OpenD_10.10.7008_Windows\FutuOpenD.exe" -lang=en
echo Log in in the OpenD window, then come back here.
pause

echo [3/3] Futu overlay + news, then 1-window PPO into models\news_gpu_v2_11
echo Confirm the seed banner vs GPU run_trader.bat (not checkpoint_2026-08-20).
python -m src.finetune_v2_11 --windows 1 --device cuda

echo.
echo Fine-tune complete. Live V2 best_model.zip was not touched.
echo Next: execution\predict_now_v2_11.bat — HK actions must be >= 0.
pause
