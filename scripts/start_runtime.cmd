@echo off
cd /d C:\slm
C:\slm\venv\Scripts\python.exe -m uvicorn runtime.app:app --port 8200 --host 127.0.0.1 >> C:\slm\logs\runtime.log 2>&1
