"""Configuration settings for QuixBugs Multi-Agent Memory Benchmark."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Root of QuixBugs-master workspace
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Load .env if present
ENV_PATH = WORKSPACE_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# QuixBugs Dataset Paths
PYTHON_PROGRAMS_DIR = WORKSPACE_ROOT / "python_programs"
CORRECT_PYTHON_PROGRAMS_DIR = WORKSPACE_ROOT / "correct_python_programs"
JSON_TESTCASES_DIR = WORKSPACE_ROOT / "json_testcases"

# CleanVul & Agent-Green Dataset Paths
CLEANVUL_DIR = WORKSPACE_ROOT / "CleanVul-main"
AGENT_GREEN_DIR = WORKSPACE_ROOT / "agent-green-main"
AGENT_GREEN_VULN_FILE = AGENT_GREEN_DIR / "vuln_database" / "VulTrial_386_samples_balanced.jsonl"
AGENT_GREEN_TD_FILE = AGENT_GREEN_DIR / "data" / "mlcq_cleaned_and_pruned_dataset_385.csv"

# Output and storage directories
RESULTS_DIR = WORKSPACE_ROOT / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_STORAGE_DIR = WORKSPACE_ROOT / ".memory_data"
MEMORY_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration Defaults
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_FEEDBACK_ROUNDS = int(os.getenv("MAX_FEEDBACK_ROUNDS", "2"))

# Native VirtualEnv Path if available
CS401_VENV_PYTHON = Path(r"C:\Users\LENOVO\OneDrive\Desktop\CS 401\.venv\Scripts\python.exe")
