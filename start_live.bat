@echo off
REM Smart Election - start the server on its fixed public address (Tailscale Funnel).
REM Double-click this file and keep the server window open while people use the site.
cd /d "%~dp0"

if not exist models\face_recognition_sface_2021dec.onnx (
    echo Downloading AI face models...
    .venv\Scripts\python.exe download_models.py
)

echo Making sure the public link is switched on...
"C:\Program Files\Tailscale\tailscale.exe" funnel --bg 5000

echo.
echo Starting the Smart Election server...
echo Your public link:
"C:\Program Files\Tailscale\tailscale.exe" funnel status
echo Officers sign in at the same link via "Officer login".
echo Keep this window open. Close it to take the site offline.
echo.
.venv\Scripts\python.exe serve.py
