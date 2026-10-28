@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The local Python environment is missing. See LOCAL_SETUP.md.
    pause
    exit /b 1
)
echo Opening FinGuard AI at http://127.0.0.1:8501
echo Keep this window open while using the app. Press Ctrl+C to stop it.
".venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.address=127.0.0.1 --server.port=8501 --server.headless=false --browser.serverAddress=127.0.0.1 --browser.gatherUsageStats=false
if errorlevel 1 (
    echo If the app is already running, open http://127.0.0.1:8501 in your browser.
    pause
)
