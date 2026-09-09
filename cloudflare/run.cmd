@echo off
REM Starts the Cloudflare named tunnel for this project (stable hostnames).
REM Used by the scheduled task and safe to double-click.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
