@echo off
REM Bloomberg through ~18 Aug 10:00 HKT, then Futu 17-21 Aug, clip calendar 2026-08-21.
REM 2026-08-24 is held out as the fair test. Needs OpenD (pause live V2 or wait for a quiet moment).
REM Copy data\raw\news\ onto the VM first so news uses cache (no 2-year AV refill).
title AirAire - Build V2.10 panel (Futu gap, hold out 24 Aug)
cd /d C:\Users\klmong\Desktop\airaire
call venv_gpu\Scripts\activate.bat
echo Building enhanced_v2_10.parquet
echo Futu 2026-08-17 through US Fri close 2026-08-21 (04:00 HKT 22 Aug). Clip to 2026-08-21.
python -m src.data_loader_v2_10 --force-rebuild --futu-start 2026-08-17 --futu-end 2026-08-22 --panel-end 2026-08-21
pause
