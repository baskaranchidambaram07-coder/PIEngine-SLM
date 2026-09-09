@echo off
cd /d C:\slm
C:\slm\venv\Scripts\python.exe -m uvicorn studio.app:app --port 8100 --host 127.0.0.1 >> C:\slm\logs\studio.log 2>&1
