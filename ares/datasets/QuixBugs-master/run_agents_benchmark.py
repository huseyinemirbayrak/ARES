"""
Comprehensive Multi-Agent Memory Benchmark Runner on QuixBugs Dataset.

Evaluates 4 LangGraph agents across QuixBugs algorithms:
1. LangGraph + Mem0 (Episodic Memory)
2. LangGraph + Cognee (AST Knowledge Graph & Ontologies)
3. LangGraph + Zep (Temporal Memory & Fact Synthesis)
4. LangGraph + Qdrant (Dense Vector Similarity & Filtering)

Computes:
- Bug Detection Rate (Recall / TP)
- False Positive Resistance (Precision / FP)
- F1-Score & Accuracy
- Round 1 vs Final Round Revision Improvement
- Consensus Approval Rate
- Execution Latency
"""

import sys
import os
import time
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Ensure utf-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add workspace to sys.path
workspace_dir = Path(__file__).resolve().parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from quixbugs_agents.dataset_loader import QuixBugsDatasetLoader
from quixbugs_agents.config import RESULTS_DIR, WORKSPACE_ROOT
from quixbugs_agents.memory import clear_all_memories
from quixbugs_agents.report_generator import generate_benchmark_report
from quixbugs_agents.agents.langgraph_mem0 import LangGraphMem0Agent
from quixbugs_agents.agents.langgraph_cognee import LangGraphCogneeAgent
from quixbugs_agents.agents.langgraph_zep import LangGraphZepAgent
from quixbugs_agents.agents.langgraph_qdrant import LangGraphQdrantAgent


console = Console(width=120)

AGENT_REGISTRY = {
    "mem0": ("LangGraph + Mem0", LangGraphMem0Agent),
    "cognee": ("LangGraph + Cognee", LangGraphCogneeAgent),
    "zep": ("LangGraph + Zep", LangGraphZepAgent),
    "qdrant": ("LangGraph + Qdrant", LangGraphQdrantAgent)
}


def compute_metrics(results: List[Dict[str, Any]], elapsed_total: float, framework_name: str) -> Dict[str, Any]:
    """Computes comprehensive scientific evaluation metrics for an agent."""
    n = len(results)
    if n == 0:
        return {"framework": framework_name, "total_samples": 0}

    r1_correct = sum(1 for r in results if r.get("round_1_matches_gt") is True)
    final_correct = sum(1 for r in results if r.get("final_matches_gt") is True)
    improved = sum(1 for r in results if r.get("feedback_improved") is True)
    consensus_count = sum(1 for r in results if r.get("consensus_reached") is True)

    # Confusion Matrix
    tp = sum(1 for r in results if r.get("ground_truth_is_buggy") is True and r.get("final_is_buggy") is True)
    fp = sum(1 for r in results if r.get("ground_truth_is_buggy") is False and r.get("final_is_buggy") is True)
    tn = sum(1 for r in results if r.get("ground_truth_is_buggy") is False and r.get("final_is_buggy") is False)
    fn = sum(1 for r in results if r.get("ground_truth_is_buggy") is True and r.get("final_is_buggy") is False)

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    r1_acc = round((r1_correct / n) * 100, 1)
    final_acc = round((final_correct / n) * 100, 1)
    delta = round(final_acc - r1_acc, 1)

    return {
        "framework": framework_name,
        "total_samples": n,
        "round_1_accuracy": r1_acc,
        "final_accuracy": final_acc,
        "accuracy_improvement_pct": delta,
        "feedback_improved_count": improved,
        "consensus_rate_pct": round((consensus_count / n) * 100, 1),
        "elapsed_total_sec": round(elapsed_total, 2),
        "avg_latency_per_sample_sec": round(elapsed_total / n, 2),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    }


def run_benchmark():
    parser = argparse.ArgumentParser(description="Multi-Agent 4-Memory Benchmark (QuixBugs / CleanVul / Agent-Green)")
    parser.add_argument(
        "--dataset", type=str, default="quixbugs",
        choices=["quixbugs", "cleanvul", "agent_green", "agent_green_td", "all"],
        help="Dataset to evaluate: 'quixbugs', 'cleanvul', 'agent_green', 'agent_green_td', or 'all' (default: quixbugs)"
    )
    parser.add_argument(
        "--agent", type=str, default="all",
        help="Target agent: 'all', or comma-separated 'mem0,cognee,zep,qdrant' (default: all)"
    )
    parser.add_argument(
        "--provider", type=str, default="ollama", choices=["ollama", "mock", "gemini", "openai"],
        help="LLM provider to use (default: ollama)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model name (e.g. qwen2.5-coder:7b)"
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Number of QuixBugs samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="Maximum feedback iterations per sample (default: 2)"
    )
    parser.add_argument(
        "--clear-memory", action="store_true",
        help="Clear all stored memory caches and re-seed before benchmarking"
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Output Markdown report filename (default: auto derived from model/out)"
    )
    parser.add_argument(
        "--out", type=str, default="quixbugs_benchmark_summary.json",
        help="Output summary JSON filename"
    )

    args = parser.parse_args()

    # Clear memory if requested
    if args.clear_memory:
        console.print("[bold yellow]🧹 Clearing all agent memory databases and cache files...[/bold yellow]")
        clear_all_memories()
        console.print("[green]✅ Memory cleared and re-seeded to initial pristine state.[/green]")

    # Determine agents to run
    if args.agent.lower() == "all":
        selected_agents = list(AGENT_REGISTRY.keys())
    else:
        selected_agents = [
            a.strip().lower() for a in args.agent.split(",") if a.strip().lower() in AGENT_REGISTRY
        ]
        if not selected_agents:
            selected_agents = list(AGENT_REGISTRY.keys())

    # Load benchmark dataset (QuixBugs / CleanVul / Agent-Green / All)
    samples = QuixBugsDatasetLoader.load_dataset(limit=args.limit, balance=True, dataset=args.dataset)
    if not samples:
        console.print(f"[red][ERROR] No samples loaded from dataset '{args.dataset}'![/red]")
        return

    buggy_count = sum(1 for s in samples if s.get("ground_truth_is_buggy"))
    clean_count = len(samples) - buggy_count

    console.print()
    header_text = (
        f"[bold cyan]4-Agent Multi-Memory Benchmark ({args.dataset.upper()})[/bold cyan]\n"
        f"LLM Provider: [green]{args.provider.upper()}[/green] | Model: [green]{args.model or 'Default'}[/green]\n"
        f"Dataset: [bold yellow]{args.dataset.upper()}[/bold yellow] | Samples: [yellow]{len(samples)}[/yellow] (Buggy/Vuln: [red]{buggy_count}[/red], Clean: [blue]{clean_count}[/blue])\n"
        f"Agents: [magenta]{', '.join([AGENT_REGISTRY[a][0] for a in selected_agents])}[/magenta]"
    )
    console.print(Panel(header_text, border_style="cyan"))

    all_summaries = {}
    all_raw_results = {}

    ds_prefix = "" if args.dataset == "quixbugs" else f"{args.dataset.upper()}_"
    if args.report:
        report_file_name = args.report if args.report.endswith(".md") else f"{args.report}.md"
    elif args.model:
        clean_model = args.model.replace(":", "_").replace(".", "_")
        report_file_name = f"BENCHMARK_RAPORU_{ds_prefix}{clean_model.upper()}.md"
    else:
        report_file_name = f"BENCHMARK_RAPORU_{ds_prefix}4_AJAN_{len(samples)}_TEST.md"

    output_path = RESULTS_DIR / args.out
    report_root = WORKSPACE_ROOT / report_file_name
    report_results = RESULTS_DIR / report_file_name


    for agent_key in selected_agents:
        display_name, agent_cls = AGENT_REGISTRY[agent_key]
        console.print(f"\n[bold yellow]>>> Running {display_name} on {len(samples)} samples...[/bold yellow]")

        agent_instance = agent_cls(
            provider=args.provider,
            model=args.model,
            max_iterations=args.rounds
        )

        agent_results = []
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Evaluating {display_name}", total=len(samples))

            for s in samples:
                res = agent_instance.evaluate_sample(s)
                agent_results.append(res)

                # Live progress icon
                status_icon = "✅" if res.get("final_matches_gt") else "❌"
                improved_tag = " [bold green](Revised)[/bold green]" if res.get("feedback_improved") else ""
                progress.console.print(
                    f"  {status_icon} \\[{escape(s.get('algorithm', ''))}] Verdict: {'BUGGY' if res.get('final_is_buggy') else 'CLEAN'} "
                    f"(GT: {'BUGGY' if s.get('ground_truth_is_buggy') else 'CLEAN'}) in {res.get('elapsed_sec')}s{improved_tag}"
                )
                progress.advance(task)

        elapsed = time.time() - start_time
        summary = compute_metrics(agent_results, elapsed, display_name)
        all_summaries[agent_key] = summary
        all_raw_results[agent_key] = agent_results

        # Incremental Save after each completed agent
        save_payload = {
            "timestamp": datetime.now().isoformat(),
            "dataset": args.dataset,
            "provider": args.provider,
            "model": args.model or "default",
            "total_samples": len(samples),
            "buggy_samples": buggy_count,
            "clean_samples": clean_count,
            "summaries": all_summaries,
            "raw_results": all_raw_results
        }
        output_path.write_text(json.dumps(save_payload, indent=2), encoding="utf-8")

        # Live generate / update the Markdown report
        try:
            generate_benchmark_report(save_payload, report_root)
            generate_benchmark_report(save_payload, report_results)
            console.print(f"[dim green]📄 Live report updated: {report_root}[/dim green]")
        except Exception as e:
            console.print(f"[dim red]Warning generating live report: {e}[/dim red]")

    # Display Comparative Results Table
    console.print("\n" + "=" * 85)
    console.print("[bold cyan]🏆 QUIXBUGS MULTI-AGENT PERFORMANCE COMPARISON MATRIX[/bold cyan]")
    console.print("=" * 85)

    table = Table(title="Agentic Memory Framework Comparison on QuixBugs", header_style="bold magenta", expand=True)
    table.add_column("Agent Architecture", style="bold white", min_width=20)
    table.add_column("Memory Type", style="dim", min_width=18)
    table.add_column("R1 Acc", justify="center")
    table.add_column("Final Acc", justify="center")
    table.add_column("Δ Imp", justify="center", style="bold green")
    table.add_column("Prec", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("F1", justify="center", style="bold yellow")
    table.add_column("Approv", justify="center")
    table.add_column("Latency", justify="center")

    memory_tags = {
        "mem0": "Episodic Experience",
        "cognee": "AST Knowledge Graph",
        "zep": "Temporal Fact Graph",
        "qdrant": "Dense Vector (k-NN)"
    }

    for agent_key in selected_agents:
        s = all_summaries[agent_key]
        cm = s["confusion_matrix"]
        delta_str = f"+{s['accuracy_improvement_pct']}%" if s['accuracy_improvement_pct'] > 0 else f"{s['accuracy_improvement_pct']}%"

        table.add_row(
            s["framework"],
            memory_tags.get(agent_key, "Memory"),
            f"{s['round_1_accuracy']}%",
            f"{s['final_accuracy']}%",
            delta_str,
            f"{cm['precision']:.3f}",
            f"{cm['recall']:.3f}",
            f"{cm['f1_score']:.3f}",
            f"{s['consensus_rate_pct']}%",
            f"{s['avg_latency_per_sample_sec']}s"
        )

    console.print(table)

    # Save final JSON
    save_payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "provider": args.provider,
        "model": args.model or "default",
        "total_samples": len(samples),
        "buggy_samples": buggy_count,
        "clean_samples": clean_count,
        "summaries": all_summaries,
        "raw_results": all_raw_results
    }
    output_path.write_text(json.dumps(save_payload, indent=2), encoding="utf-8")
    console.print(f"\n[green]✅ Detailed benchmark results saved to: {output_path}[/green]")

    # Save CSV Summary
    csv_path = RESULTS_DIR / args.out.replace(".json", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Framework", "Memory_Type", "Round_1_Acc", "Final_Acc", "Delta_Pct",
            "Precision", "Recall", "F1_Score", "Consensus_Rate", "Avg_Latency_Sec"
        ])
        for agent_key in selected_agents:
            s = all_summaries[agent_key]
            cm = s["confusion_matrix"]
            writer.writerow([
                s["framework"], memory_tags.get(agent_key, "Memory"),
                s["round_1_accuracy"], s["final_accuracy"], s["accuracy_improvement_pct"],
                cm["precision"], cm["recall"], cm["f1_score"],
                s["consensus_rate_pct"], s["avg_latency_per_sample_sec"]
            ])
    console.print(f"[green]✅ Summary CSV saved to: {csv_path}[/green]")

    # Final report generation
    generate_benchmark_report(save_payload, report_root)
    generate_benchmark_report(save_payload, report_results)
    console.print(f"[bold green]📊 Final comprehensive report generated at:\n  - {report_root}\n  - {report_results}[/bold green]\n")



if __name__ == "__main__":
    run_benchmark()
