@echo off
chcp 65001 >nul
setlocal

echo ============================================================================
echo   4'LU AJAN HAFIZA TESTI (Mem0 vs Cognee vs Zep vs Qdrant)
echo   Kapsam: CleanVul + Agent-Green + QuixBugs
echo ============================================================================

set PYTHON_EXE=C:\Users\LENOVO\OneDrive\Desktop\CS 401\.venv\Scripts\python.exe

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" "%~dp0run_cleanvul_agentgreen_benchmark.py" %*
) else (
    python "%~dp0run_cleanvul_agentgreen_benchmark.py" %*
)
pause

