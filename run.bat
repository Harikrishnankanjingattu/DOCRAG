@echo off
echo ===================================================
echo   Starting Document RAG Interactive Studio
echo ===================================================
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
