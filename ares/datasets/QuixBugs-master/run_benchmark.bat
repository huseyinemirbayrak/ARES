@echo off
setlocal

set PYTHON_EXE=C:\Users\LENOVO\OneDrive\Desktop\CS 401\.venv\Scripts\python.exe

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" "%~dp0run_agents_benchmark.py" %*
) else (
    python "%~dp0run_agents_benchmark.py" %*
)
