@echo off
setlocal

set PYTHON_EXE=%~dp0..\..\..\.venv\Scripts\python.exe

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" "%~dp0run_agents_benchmark.py" %*
) else (
    python "%~dp0run_agents_benchmark.py" %*
)
