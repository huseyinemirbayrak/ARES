# PowerShell Launcher for QuixBugs 4-Agent Memory Benchmark
param(
    [string]$Dataset = "quixbugs",
    [string]$Agent = "all",
    [string]$Provider = "ollama",
    [string]$Model = "qwen2.5-coder:7b",
    [int]$Limit = 80,
    [int]$Rounds = 2,
    [switch]$ClearMemory
)

$venvPython = "C:\Users\LENOVO\OneDrive\Desktop\CS 401\.venv\Scripts\python.exe"
$pyCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

$argsList = @(
    "$PSScriptRoot\run_agents_benchmark.py",
    "--dataset", $Dataset,
    "--agent", $Agent,
    "--provider", $Provider,
    "--model", $Model,
    "--limit", $Limit,
    "--rounds", $Rounds
)

if ($ClearMemory) {
    $argsList += "--clear-memory"
}

& $pyCmd $argsList

