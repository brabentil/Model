@echo off
setlocal

:: Set environment variables
set PYTHONPATH=%CD%
set API_PORT=8000

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Start the API server
python main.py

endlocal
