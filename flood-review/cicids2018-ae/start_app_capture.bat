@echo off
rem Start the dashboard WITH live capture enabled (needs Npcap). Capture only on a network you own or may monitor.
rem If the dashboard is already running without capture, close that window first: the setting is read when the server starts.
cd /d "%~dp0"
python run_app.py --host 127.0.0.1 --port 8000 --open --enable-capture
pause
