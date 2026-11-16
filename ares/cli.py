"""Command-Line Interface and Rich Dashboard for ARES Adversarial Self-Play.

Supports Plug-and-Play ('Tak-Çıkar') datasets (`.json`, `.jsonl`, `.yaml`, `.csv`)
and on-demand Cognee Portable Memory export/import/merge/inspection (`.json` and `.zip`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ares.agents.attacker import AttackerAgent
from ares.agents.defender import DefenderAgent
from ares.agents.llm_client import LocalLLMClient
from ares.datasets.adapter import PlugAndPlayDatasetAdapter
from ares.datasets.loader import load_problems
from ares.memory.cognee_memory import CogneeMemoryManager
from ares.models import ExamResult, TurnRecord
from ares.orchestrator.langgraph_engine import ARESLangGraphOrchestrator
from ares.referee.deterministic_referee import DeterministicReferee


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ares",
        description="ARES: Plug-and-Play Adversarial Self-Play for Software Analysis (LangGraph + Cognee + Deterministic Referee)",
    )
    # Self-Play & Exam Controls
    parser.add_argument(
        "--turns",
        type=int,
        default=25,
        help="Number of adversarial self-play turns to execute (default: 25)",
    )
    parser.add_argument(
        "--exam-interval",
        type=int,
        default=25,
        help="Run external benchmark exam ('Dışarıdan Sınav') every N turns (default: 25)",
    )

    # Plug-and-Play Dataset Controls
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to ANY custom self-play dataset (.json, .jsonl, .yaml, .csv) to plug into the arena",
    )
    parser.add_argument(
        "--exam-dataset",
        type=str,
        default=None,
        help="Path to ANY custom external exam dataset (.json, .jsonl, .yaml, .csv) for periodic evaluation",
    )
    parser.add_argument(
        "--validate-dataset",
        type=str,
        default=None,
        help="Validate and inspect a custom dataset file (.json, .jsonl, .yaml, .csv) on demand and exit",
    )

    # On-Demand Portable Memory Export / Import / Merge / Inspect Controls
    parser.add_argument(
        "--memory-dir",
        type=str,
        default="memory_store",
        help="Directory for active portable Cognee Knowledge Graph & SQLite store",
    )
    parser.add_argument(
        "--import-memory",
        type=str,
        default=None,
        help="Path to a portable Cognee bundle (.json or .zip) to import before running self-play",
    )
    parser.add_argument(
        "--import-mode",
        choices=["replace", "merge"],
        default="replace",
        help="How to import memory: 'replace' overwrites active memory, 'merge' combines with existing memory (default: replace)",
    )
    parser.add_argument(
        "--export-memory",
        type=str,
        default=None,
        help="Path (.json or .zip) to export the portable Cognee memory bundle after self-play",
    )
    parser.add_argument(
        "--export-role",
        choices=["both", "attacker", "defender"],
        default="both",
        help="Which agent's memory to include when exporting (default: both)",
    )
    parser.add_argument(
        "--export-only",
        type=str,
        default=None,
        help="On-demand memory export: immediately export current memory store to the given .json or .zip path and exit",
    )
    parser.add_argument(
        "--import-only",
        type=str,
        default=None,
        help="On-demand memory import: immediately import/merge the given .json or .zip bundle into active memory store and exit",
    )
    parser.add_argument(
        "--inspect-memory",
        type=str,
        default=None,
        help="Inspect any exported .json or .zip Cognee memory bundle and display its contents",
    )
    parser.add_argument(
        "--fresh-memory",
        action="store_true",
        help="Start with a clean Cognee memory store to observe full cold-start to Turn-N learning curve",
    )

    # Local LLM Controls
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "ollama", "openai_compatible", "adaptive_local"],
        default="auto",
        help="Local LLM execution mode (default: auto)",
    )
    parser.add_argument(
        "--attacker-model",
        type=str,
        default="qwen2.5-coder:7b",
        help="Local model name for Attacker Agent",
    )
    parser.add_argument(
        "--defender-model",
        type=str,
        default="qwen2.5-coder:7b",
        help="Local model name for Defender Agent",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for Ollama or OpenAI-compatible local LLM server",
    )
    parser.add_argument(
        "--blind-attacker",
        action="store_true",
        help="Hide unit test cases from the Attacker Agent prompt (blind black-box attack mode)",
    )
    parser.add_argument(
        "--blind-defender",
        action="store_true",
        help="Hide failing unit test details from the Defender Agent prompt (blind specification-driven repair mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for turn logs, exam results, and scoreboard summary",
    )
    return parser


def _handle_validate_dataset(dataset_path: str) -> int:
    report = PlugAndPlayDatasetAdapter.validate_dataset_report(dataset_path)
    table = Table(
        title=f"Plug-and-Play Dataset Validation Report: {report['dataset_path']}",
        header_style="bold cyan",
    )
    table.add_column("Problem ID", style="bold")
    table.add_column("Function Name", style="yellow")
    table.add_column("Test Cases", justify="right")
    table.add_column("Clean Code Passes?", justify="center")
    table.add_column("Total Candidate Mutations", justify="right")
    table.add_column("Valid 1-2 Broken Mutations", justify="right")

    for row in report["problems"]:
        badge = "[bold green]PASS[/bold green]" if row["clean_code_passes_all"] else "[bold red]FAIL[/bold red]"
        table.add_row(
            row["problem_id"],
            row["function_name"],
            str(row["num_tests"]),
            badge,
            str(row["synthesized_or_loaded_mutations"]),
            str(row["valid_1_to_2_broken_mutations"]),
        )
    console.print(table)
    return 0 if report["all_clean_code_valid"] else 1


def _handle_inspect_memory(bundle_path: str) -> int:
    info = CogneeMemoryManager.inspect_bundle(bundle_path)
    summ = info.get("summary", {})
    console.print(
        Panel.fit(
            f"[bold magenta]Cognee Portable Memory Bundle Inspection[/bold magenta]\n"
            f"• File: `{info['file']}`\n"
            f"• Schema Version: `{info['schema_version']}` | Role: `{info['exported_role']}` | Exported At: `{info['exported_at']}`\n"
            f"• Knowledge Graph Nodes: [bold]{summ.get('total_nodes', 0)}[/bold] | "
            f"Edges: [bold]{summ.get('total_edges', 0)}[/bold] | "
            f"Episodes: [bold]{summ.get('total_episodes', 0)}[/bold]\n"
            f"• Tracked Attack Strategies: [bold]{summ.get('tracked_attack_strategies', 0)}[/bold]",
            border_style="magenta",
        )
    )
    def_table = Table(title="Learned Defender Defect Categories & Verified Functions", header_style="bold green")
    def_table.add_column("Defect Category", style="bold")
    def_table.add_column("Encounters", justify="right")
    def_table.add_column("Repairs Succeeded", justify="right")
    def_table.add_column("Verified Function Repairs")

    for dtype, stats in info.get("defender_defect_breakdown", {}).items():
        def_table.add_row(
            dtype,
            str(stats["encounters"]),
            str(stats["repairs_succeeded"]),
            ", ".join(stats["verified_functions"]) or "-",
        )
    console.print(def_table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. On-demand Dataset Validation Mode
    if args.validate_dataset:
        return _handle_validate_dataset(args.validate_dataset)

    # 2. On-demand Memory Inspection Mode
    if args.inspect_memory:
        return _handle_inspect_memory(args.inspect_memory)

    memory_dir = Path(args.memory_dir)
    if args.fresh_memory and memory_dir.exists():
        for f in memory_dir.glob("*"):
            if f.is_file():
                f.unlink()

    memory = CogneeMemoryManager(store_dir=memory_dir)

    # 3. On-demand Standalone Import Mode
    if args.import_only:
        res = memory.import_portable_bundle(args.import_only, mode=args.import_mode)
        memory.export_portable_bundle()  # Persist into active memory_store
        console.print(
            Panel.fit(
                f"[bold green]On-Demand Cognee Memory Import Complete ({args.import_mode.upper()} mode)[/bold green]\n"
                f"• Source: `{args.import_only}` -> Active Store: `{memory.bundle_path}`\n"
                f"• Total KG Nodes: [bold]{res['total_nodes']}[/bold] | "
                f"Edges: [bold]{res['total_edges']}[/bold] | "
                f"Episodes: [bold]{res['total_episodes']}[/bold]\n"
                f"• Learned Defect Types: `{res['learned_defect_types']}`",
                border_style="green",
            )
        )
        return 0

    # 4. On-demand Standalone Export Mode
    if args.export_only:
        out_file = memory.export_portable_bundle(
            args.export_only, role=args.export_role
        )
        console.print(
            Panel.fit(
                f"[bold cyan]On-Demand Cognee Memory Export Complete[/bold cyan]\n"
                f"• Exported To: `{out_file}` (Role: `{args.export_role}`)\n"
                f"• Total KG Nodes: [bold]{len(memory.nodes)}[/bold] | "
                f"Edges: [bold]{len(memory.edges)}[/bold] | "
                f"Episodes: [bold]{len(memory.episodes)}[/bold]",
                border_style="cyan",
            )
        )
        return 0

    # 5. Pre-run Memory Import (if requested alongside a self-play run)
    if args.import_memory:
        summary_loaded = memory.import_portable_bundle(
            args.import_memory, mode=args.import_mode
        )
        console.print(
            f"[bold cyan]Imported portable Cognee memory bundle ({args.import_mode}):[/bold cyan] {summary_loaded}"
        )

    provider = "openai_compatible" if args.llm_mode == "openai_compatible" else "ollama"
    attacker_llm = LocalLLMClient(
        provider=provider,
        model_name=args.attacker_model,
        base_url=args.base_url,
        temperature=0.7,
        mode=args.llm_mode,
    )
    defender_llm = LocalLLMClient(
        provider=provider,
        model_name=args.defender_model,
        base_url=args.base_url,
        temperature=0.2,
        mode=args.llm_mode,
    )

    attacker = AttackerAgent(llm_client=attacker_llm, blind_mode=args.blind_attacker)
    defender = DefenderAgent(llm_client=defender_llm, blind_mode=args.blind_defender)
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)

    problems = load_problems(path=args.dataset, exam=False)
    exam_problems = load_problems(path=args.exam_dataset, exam=True)

    console.print(
        Panel.fit(
            f"[bold gold1]ARES: Plug-and-Play Adversarial Self-Play for Software Analysis[/bold gold1]\n"
            f"Orchestrator: [bold green]LangGraph StateGraph[/bold green] | "
            f"Memory: [bold magenta]Cognee Portable Knowledge Graph[/bold magenta] | "
            f"Referee: [bold cyan]100% Deterministic (1-2/5 Broken Test Rule)[/bold cyan]\n"
            f"Turns: [bold]{args.turns}[/bold] | External Exam Interval: [bold]{args.exam_interval}[/bold] | "
            f"Blind Mode: Attacker=[bold]{'ON' if args.blind_attacker else 'OFF'}[/bold], "
            f"Defender=[bold]{'ON' if args.blind_defender else 'OFF'}[/bold]\n"
            f"Arena Dataset: [bold]{len(problems)} problems[/bold] (`{args.dataset or 'default MLCQ'}`) | "
            f"External Exam Suite: [bold]{len(exam_problems)} problems[/bold] (`{args.exam_dataset or 'default Exam'}`)",
            border_style="gold1",
        )
    )

    turn_table = Table(
        title="Adversarial Self-Play Turn Log",
        show_lines=False,
        header_style="bold cyan",
    )
    turn_table.add_column("Turn", justify="right")
    turn_table.add_column("Problem", style="bold")
    turn_table.add_column("Injected Defect", style="yellow")
    turn_table.add_column("Attacker Tests Broken", justify="center")
    turn_table.add_column("Defender Tests Passed", justify="center")
    turn_table.add_column("Classified?", justify="center")
    turn_table.add_column("Winner", justify="center")
    turn_table.add_column("ELO (Att / Def)", justify="right")

    exam_records: list[ExamResult] = []

    def handle_turn(rec: TurnRecord) -> None:
        v = rec.verdict
        att_broken = f"{v.attacker_suite_result.failed_count}/{v.attacker_suite_result.total_tests}"
        def_passed = f"{v.defender_suite_result.passed_count}/{v.defender_suite_result.total_tests}"
        winner_badge = (
            "[bold red]ATTACKER[/bold red]"
            if v.winner == "attacker"
            else "[bold green]DEFENDER[/bold green]"
            if v.winner == "defender"
            else "[dim]DRAW[/dim]"
        )
        cls_badge = "[green]YES[/green]" if v.classification_correct else "[red]NO[/red]"
        llm_tag = f"LLM(att={rec.attacker_output.used_llm},def={rec.defender_output.used_llm})"
        console.print(
            f"[dim]Turn {rec.turn_id:02d}[/dim] {rec.problem_id} | "
            f"broken={att_broken} repaired={def_passed} | "
            f"winner={winner_badge} | {llm_tag}"
        )
        turn_table.add_row(
            str(rec.turn_id),
            rec.problem_id,
            rec.attacker_output.injected_defect_type,
            att_broken,
            def_passed,
            cls_badge,
            winner_badge,
            f"{rec.attacker_elo_after:.0f} / {rec.defender_elo_after:.0f}",
        )

    def handle_exam(ex: ExamResult) -> None:
        exam_records.append(ex)

    orchestrator = ARESLangGraphOrchestrator(
        problems=problems,
        exam_problems=exam_problems,
        attacker=attacker,
        defender=defender,
        referee=referee,
        memory=memory,
        output_dir=args.output_dir,
        on_turn_complete=handle_turn,
        on_exam_complete=handle_exam,
    )

    scoreboard = orchestrator.run_self_play(
        num_turns=args.turns,
        exam_interval=args.exam_interval,
        run_baseline_exam=True,
    )

    if args.export_memory:
        exported_path = memory.export_portable_bundle(
            args.export_memory, role=args.export_role
        )
        console.print(
            f"[bold green]Exported portable Cognee memory bundle ({args.export_role}) to:[/bold green] {exported_path}"
        )

    console.print(turn_table)

    exam_table = Table(
        title="Periodic External Benchmark Exams ('Dışarıdan Sınavlar' - Unseen Code Suite)",
        header_style="bold magenta",
    )
    exam_table.add_column("Exam ID", style="bold")
    exam_table.add_column("After Turn", justify="right")
    exam_table.add_column("Cognee KG Nodes", justify="right")
    exam_table.add_column("Attacker Valid 1-2 Broken (%)", justify="right")
    exam_table.add_column("Defender Full Repair (%)", justify="right")
    exam_table.add_column("Defender Classification (%)", justify="right")
    exam_table.add_column("Defender Combined Score (%)", justify="right")

    for ex in exam_records:
        exam_table.add_row(
            ex.exam_id,
            str(ex.after_turn),
            str(ex.cognee_knowledge_nodes),
            f"{ex.attacker_valid_subtle_rate:.1f}%",
            f"{ex.defender_full_repair_rate:.1f}%",
            f"{ex.defender_classification_accuracy:.1f}%",
            f"{ex.defender_combined_score:.1f}%",
        )
    console.print(exam_table)

    imp = scoreboard.get("external_exam_improvement", {})
    console.print(
        Panel.fit(
            f"[bold green]Self-Play & External Exam Summary[/bold green]\n"
            f"• Winner Scoreboard: Attacker Wins = [bold red]{scoreboard['attacker_wins']}[/bold red] | "
            f"Defender Wins = [bold green]{scoreboard['defender_wins']}[/bold green] | "
            f"Draws = {scoreboard['draws']}\n"
            f"• Final ELO Ratings: Attacker = [bold]{scoreboard['attacker_final_elo']}[/bold] | "
            f"Defender = [bold]{scoreboard['defender_final_elo']}[/bold]\n"
            f"• Portable Cognee Memory: [bold]{scoreboard['cognee_portable_memory']['total_nodes']} KG nodes[/bold], "
            f"[bold]{scoreboard['cognee_portable_memory']['total_edges']} edges[/bold], "
            f"[bold]{scoreboard['cognee_portable_memory']['total_episodes']} episodes[/bold]\n"
            f"  Bundle saved at: `{scoreboard['cognee_portable_memory']['bundle_path']}`\n"
            f"• External Exam Delta (Turn {imp.get('baseline_turn', 0)} -> Turn {imp.get('final_turn', args.turns)}):\n"
            f"  - Attacker Valid Subtle Attack Rate Delta: [bold cyan]{imp.get('attacker_valid_subtle_rate_delta', 0.0):+.2f}%[/bold cyan]\n"
            f"  - Defender Full Repair Rate Delta:         [bold green]{imp.get('defender_full_repair_rate_delta', 0.0):+.2f}%[/bold green]\n"
            f"  - Defender Classification Accuracy Delta:  [bold green]{imp.get('defender_classification_accuracy_delta', 0.0):+.2f}%[/bold green]\n"
            f"  - Defender Combined Score Delta:           [bold gold1]{imp.get('defender_combined_score_delta', 0.0):+.2f}%[/bold gold1]",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
