@echo off
rem Start the dashboard and backend on this computer and open it in the browser once it is ready.
rem If it is already running, this just says so and opens it.
cd /d "%~dp0"
python run_app.py --host 127.0.0.1 --port 8000 --open
pause
