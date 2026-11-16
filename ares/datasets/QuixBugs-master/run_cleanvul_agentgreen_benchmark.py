"""
Unified 4-Agent Memory Benchmark Runner for CleanVul, Agent-Green, and QuixBugs.

Runs all 4 LangGraph Memory Architectures:
1. LangGraph + Mem0 (Episodic Memory)
2. LangGraph + Cognee (AST & Security Ontology Knowledge Graph)
3. LangGraph + Zep (Temporal Memory & Chronological Fact Synthesis)
4. LangGraph + Qdrant (Dense Vector Search & Payload Filtering)

Across:
- CleanVul-main (Function-level Vulnerability Pairs: func_before vs func_after)
- agent-green-main (VulTrial Security Vulnerabilities & MLCQ Technical Debt)
- QuixBugs (Algorithmic Defect Benchmarks)
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Ensure utf-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from quixbugs_agents.config import WORKSPACE_ROOT, CS401_VENV_PYTHON

PYTHON_EXE = str(CS401_VENV_PYTHON) if CS401_VENV_PYTHON.exists() else sys.executable


def main():
    parser = argparse.ArgumentParser(description="4-Agent Benchmark across CleanVul & Agent-Green")
    parser.add_argument(
        "--dataset", type=str, default="all_suites",
        choices=["cleanvul", "agent_green", "agent_green_td", "quixbugs", "all", "all_suites"],
        help="Which dataset suite(s) to run (default: all_suites -> runs CleanVul and Agent-Green sequentially)"
    )
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "mock", "gemini", "openai"])
    parser.add_argument("--model", type=str, default="qwen2.5-coder:7b")
    parser.add_argument("--limit", type=int, default=20, help="Number of balanced samples per dataset (default: 20)")
    parser.add_argument("--rounds", type=int, default=2, help="Feedback rounds per sample (default: 2)")
    args = parser.parse_args()

    runner_script = str(WORKSPACE_ROOT / "run_agents_benchmark.py")

    if args.dataset == "all_suites":
        suites = [
            ("cleanvul", "benchmark_cleanvul_summary.json", "BENCHMARK_RAPORU_CLEANVUL_4_AJAN.md"),
            ("agent_green", "benchmark_agent_green_summary.json", "BENCHMARK_RAPORU_AGENT_GREEN_4_AJAN.md"),
        ]
    else:
        ds = args.dataset
        suites = [
            (ds, f"benchmark_{ds}_summary.json", f"BENCHMARK_RAPORU_{ds.upper()}_4_AJAN.md")
        ]

    for ds_name, out_json, report_md in suites:
        print("\n" + "=" * 88)
        print(f"🚀 Running 4-Agent Benchmark (Mem0 | Cognee | Zep | Qdrant) on Dataset: {ds_name.upper()}")
        print("=" * 88)
        cmd = [
            PYTHON_EXE,
            runner_script,
            "--dataset", ds_name,
            "--provider", args.provider,
            "--model", args.model,
            "--limit", str(args.limit),
            "--rounds", str(args.rounds),
            "--clear-memory",
            "--out", out_json,
            "--report", report_md
        ]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()

