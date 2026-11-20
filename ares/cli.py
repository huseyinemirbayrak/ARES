"""Command-Line Interface and Rich Dashboard for ARES Adversarial Self-Play.

Supports Plug-and-Play ('Tak-Çıkar') datasets (`.json`, `.jsonl`, `.yaml`, `.csv`)
and on-demand Cognee Portable Memory export/import/merge/inspection (`.json` and `.zip`).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys
from typing import Any
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from ares.agents.attacker import AttackerAgent
from ares.agents.defender import DefenderAgent
from ares.agents.llm_client import LocalLLMClient, infer_llm_provider
from ares.datasets.adapter import PlugAndPlayDatasetAdapter
from ares.datasets.loader import load_problems
from ares.memory.cognee_memory import CogneeMemoryManager
from ares.models import ExamResult, TurnRecord
from ares.orchestrator.langgraph_engine import ARESLangGraphOrchestrator
from ares.referee.deterministic_referee import DeterministicReferee


console = Console()
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "game_config.yaml"


def load_game_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML game configuration (`configs/game_config.yaml`) with graceful empty fallback."""
    target = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target.exists():
        return {}
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _configure_logging(verbose: bool = False) -> None:
    """Attach a RichHandler so warnings from LocalLLMClient and RealCogneeEngine are always visible."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=False)],
        force=True,
    )


_KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-large": 3072,
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


def _resolve_embedding_dimensions(
    explicit_dims: int | None,
    embedding_model: str,
    yaml_default: int = 768,
) -> int:
    """Resolve embedding dimensions: CLI flag > known model dimension > YAML default."""
    if explicit_dims is not None:
        return int(explicit_dims)
    base = embedding_model.split(":")[0].strip().lower()
    for prefix, dim in _KNOWN_EMBEDDING_DIMS.items():
        if base == prefix or base.endswith("/" + prefix):
            return dim
    return int(yaml_default)


def build_parser(cfg: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    cfg = cfg or {}
    sp_cfg = cfg.get("self_play", {})
    mem_cfg = cfg.get("memory", {})
    llm_cfg = cfg.get("llm", {})
    att_llm_cfg = llm_cfg.get("attacker_model", {})
    def_llm_cfg = llm_cfg.get("defender_model", {})
    log_cfg = cfg.get("logging", {})

    parser = argparse.ArgumentParser(
        prog="ares",
        description="ARES: Plug-and-Play Adversarial Self-Play for Software Analysis (LangGraph + Cognee + Deterministic Referee)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to game_config.yaml (CLI flags override YAML defaults)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level diagnostic logging",
    )
    # Self-Play & Exam Controls
    parser.add_argument(
        "--turns",
        type=int,
        default=int(sp_cfg.get("default_turns", 25)),
        help="Number of adversarial self-play turns to execute (default from game_config.yaml)",
    )
    parser.add_argument(
        "--exam-interval",
        type=int,
        default=int(sp_cfg.get("exam_interval", 25)),
        help="Run external benchmark exam ('Dışarıdan Sınav') every N turns",
    )

    # Plug-and-Play Dataset Controls
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Custom dataset path (.json, .jsonl, .yaml, .csv) or suite alias (quixbugs, agent_green, cleanvul, all_new)",
    )
    parser.add_argument(
        "--exam-dataset",
        type=str,
        default=None,
        help="Custom external exam dataset path or suite alias (quixbugs, agent_green, cleanvul, all_new)",
    )
    parser.add_argument(
        "--validate-dataset",
        type=str,
        default=None,
        help="Validate and inspect a custom dataset file or suite alias on demand and exit",
    )

    # On-Demand Portable Memory Export / Import / Merge / Inspect Controls
    parser.add_argument(
        "--memory-dir",
        type=str,
        default=str(mem_cfg.get("portable_store_dir", "memory_store")),
        help="Directory for active memory store",
    )
    parser.add_argument(
        "--import-memory",
        type=str,
        default=None,
        help="Path to a portable memory bundle (.json or .zip) to import before running self-play",
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
        help="Path (.json or .zip) to export the portable memory bundle after self-play",
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
        help="Inspect any exported .json or .zip memory bundle and display its contents",
    )
    parser.add_argument(
        "--fresh-memory",
        action="store_true",
        help="Start with a clean memory store to observe full cold-start to Turn-N learning curve",
    )

    # Local LLM Controls
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "ollama", "openai_compatible", "adaptive_local"],
        default=str(llm_cfg.get("mode", "auto")),
        help="Local LLM execution mode (default from game_config.yaml)",
    )
    parser.add_argument(
        "--attacker-model",
        type=str,
        default=str(att_llm_cfg.get("model_name", "qwen2.5-coder:7b")),
        help="Local model name for Attacker Agent",
    )
    parser.add_argument(
        "--defender-model",
        type=str,
        default=str(def_llm_cfg.get("model_name", "qwen2.5-coder:7b")),
        help="Local model name for Defender Agent",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=str(att_llm_cfg.get("base_url", "http://localhost:11434")),
        help="Base URL for Ollama or ANY OpenAI-compatible server (LM Studio, vLLM, llama.cpp, OpenAI, OpenRouter, Groq, Gemini)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional Bearer API key for hosted/authenticated OpenAI-compatible endpoints (or set LLM_API_KEY / OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=float(llm_cfg.get("timeout_seconds", 60.0)),
        help="HTTP timeout in seconds for each LLM call (default from game_config.yaml: 60.0)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=int(llm_cfg.get("num_ctx", 8192)),
        help="Ollama context window size `num_ctx` to prevent system-prompt truncation (default: 8192)",
    )
    parser.add_argument(
        "--blind-attacker",
        action=argparse.BooleanOptionalAction,
        default=bool(sp_cfg.get("blind_attacker", False)),
        help="Hide unit test cases from the Attacker Agent prompt (blind black-box attack mode)",
    )
    parser.add_argument(
        "--blind-defender",
        action=argparse.BooleanOptionalAction,
        default=bool(sp_cfg.get("blind_defender", False)),
        help="Hide unit test cases from the Defender Agent prompt (blind specification-driven repair mode)",
    )
    parser.add_argument(
        "--hide-test-results",
        action=argparse.BooleanOptionalAction,
        default=bool(sp_cfg.get("hide_test_results", True)),
        help="Show unit test definitions to both agents while hiding test execution pass/fail results (default: enabled)",
    )
    parser.add_argument(
        "--strict-llm",
        action=argparse.BooleanOptionalAction,
        default=bool(sp_cfg.get("strict_llm", True)),
        help="Disable heuristic AST fallbacks so every turn measures pure Local LLM performance (default: enabled; pass --no-strict-llm for heuristic baseline)",
    )
    parser.add_argument(
        "--memory-backend",
        choices=["auto", "cognee", "local_graph"],
        default=str(mem_cfg.get("backend", "auto")),
        help="Memory engine: 'cognee' (strict real Cognee SDK), 'local_graph' (honest local SQLite+AST/Diff), or 'auto'",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=mem_cfg.get("cognee_embedding_provider"),
        help="Optional embedding provider override for Cognee (e.g. 'ollama' or 'openai')",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=str(mem_cfg.get("cognee_embedding_model", "nomic-embed-text")),
        help="Embedding model for real Cognee SDK (e.g. 'nomic-embed-text', 'text-embedding-3-small')",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
        help="Optional embedding vector dimensions override (auto-inferred for common models, else YAML default 768)",
    )
    parser.add_argument(
        "--cognee-search-type",
        choices=["CHUNKS", "GRAPH_COMPLETION", "SUMMARIES"],
        default=str(mem_cfg.get("cognee_search_type", "CHUNKS")),
        help="SearchType used when querying real Cognee memory",
    )
    parser.add_argument(
        "--cognify-interval",
        type=int,
        default=int(mem_cfg.get("cognify_interval", 1)),
        help="Run batched cognee.cognify() every N turns (default: 1; increase to 3-5 for faster local runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(log_cfg.get("output_dir", "outputs")),
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
            f"[bold magenta]ARES Portable Memory Bundle Inspection[/bold magenta]\n"
            f"• File: `{info['file']}`\n"
            f"• Schema Version: `{info['schema_version']}` | Backend: `{info.get('active_backend', 'local_graph')}` | "
            f"Role: `{info['exported_role']}` | Exported At: `{info['exported_at']}`\n"
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


_REPO_ROOT = Path(__file__).resolve().parent.parent
_KNOWN_SUITE_ALIASES: set[str] = {
    "quixbugs",
    "quixbugs-master",
    "agent_green",
    "agent-green",
    "agent-green-main",
    "cleanvul",
    "cleanvul-main",
    "all_new",
    "all-new",
    "external_all",
}


def _resolve_dataset_arg(cli_val: str | None, yaml_val: Any) -> str | None:
    """Resolve dataset path or suite alias with CLI taking precedence over game_config.yaml."""
    if cli_val:
        return cli_val
    if not yaml_val:
        return None
    raw = str(yaml_val).strip()
    if raw.rstrip("/\\").lower() in _KNOWN_SUITE_ALIASES:
        return raw
    candidate = Path(raw)
    if candidate.exists():
        return raw
    repo_candidate = _REPO_ROOT / raw
    if repo_candidate.exists():
        return str(repo_candidate)
    return raw


def _resolve_strict_llm_mode(
    args: argparse.Namespace,
    raw_argv: list[str],
    parser: argparse.ArgumentParser,
) -> bool:
    """Resolve interaction between `--llm-mode adaptive_local` and `--strict-llm`."""
    explicit_strict = "--strict-llm" in raw_argv
    if args.llm_mode == "adaptive_local":
        if explicit_strict:
            parser.error(
                "`--llm-mode adaptive_local` disables live LLM calls and cannot be combined with `--strict-llm`."
            )
        if args.strict_llm:
            console.print(
                "[bold yellow]NOTE:[/bold yellow] `--llm-mode adaptive_local` explicitly selected; "
                "disabling `strict_llm` to run the offline AST heuristic baseline."
            )
            return False
    return bool(args.strict_llm)


def main(argv: list[str] | None = None) -> int:
    effective_argv = sys.argv[1:] if argv is None else list(argv)

    # Pre-parse `--config` so YAML defaults take effect before full CLI parsing
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(effective_argv)
    game_cfg = load_game_config(pre_args.config)

    parser = build_parser(game_cfg)
    args = parser.parse_args(effective_argv)
    _configure_logging(verbose=args.verbose)

    sp_cfg = game_cfg.get("self_play", {})
    ref_cfg = game_cfg.get("referee", {})
    mem_cfg = game_cfg.get("memory", {})
    llm_cfg = game_cfg.get("llm", {})
    ds_cfg = game_cfg.get("datasets", {})

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
            elif f.is_dir() and f.name in ("cognee_system", "cognee_data"):
                shutil.rmtree(f, ignore_errors=True)

    explicit_base_url = any(
        tok == "--base-url" or tok.startswith("--base-url=") for tok in effective_argv
    )
    attacker_base_url = args.base_url
    defender_base_url = (
        args.base_url
        if explicit_base_url
        else str(llm_cfg.get("defender_model", {}).get("base_url", args.base_url))
    )
    attacker_provider = infer_llm_provider(args.llm_mode, attacker_base_url, args.api_key)
    defender_provider = infer_llm_provider(args.llm_mode, defender_base_url, args.api_key)
    resolved_embed_dims = _resolve_embedding_dimensions(
        args.embedding_dimensions,
        args.embedding_model,
        yaml_default=int(mem_cfg.get("cognee_embedding_dimensions", 768)),
    )
    bundle_filename = (
        Path(str(mem_cfg["portable_bundle_file"])).name
        if mem_cfg.get("portable_bundle_file")
        else "cognee_portable_graph.json"
    )
    memory = CogneeMemoryManager(
        store_dir=memory_dir,
        bundle_filename=bundle_filename,
        backend=args.memory_backend,
        in_context_window=int(sp_cfg.get("in_context_history_window", 6)),
        llm_provider=defender_provider,
        llm_model=args.defender_model,
        llm_endpoint=defender_base_url,
        llm_api_key=args.api_key,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimensions=resolved_embed_dims,
        cognee_search_type=args.cognee_search_type,
        cognify_interval=args.cognify_interval,
        top_k_memories=int(mem_cfg.get("top_k_memories", 4)),
        auto_export=bool(mem_cfg.get("auto_export_portable_snapshot", True)),
    )

    try:
        # 3. On-demand Standalone Import Mode
        if args.import_only:
            res = memory.import_portable_bundle(args.import_only, mode=args.import_mode)
            memory.export_portable_bundle()  # Persist into active memory_store
            console.print(
                Panel.fit(
                    f"[bold green]On-Demand ARES Memory Import Complete ({args.import_mode.upper()} mode)[/bold green]\n"
                    f"• Source: `{args.import_only}` -> Active Store: `{memory.bundle_path}`\n"
                    f"• Active Backend: `{memory.active_backend}` | Total KG Nodes: [bold]{res['total_nodes']}[/bold] | "
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
                    f"[bold cyan]On-Demand ARES Memory Export Complete[/bold cyan]\n"
                    f"• Exported To: `{out_file}` (Role: `{args.export_role}` | Backend: `{memory.active_backend}`)\n"
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
                f"[bold cyan]Imported portable ARES memory bundle ({args.import_mode}):[/bold cyan] {summary_loaded}"
            )
        elif not args.fresh_memory and memory.loaded_preexisting_episodes > 0:
            console.print(
                f"[bold yellow]WARNING:[/bold yellow] Loaded {memory.loaded_preexisting_episodes} pre-existing "
                f"episode(s) from `{memory.bundle_path}` (schema={memory.migrated_legacy_schema or 'current'}). "
                f"Turn-0 baseline is NOT a cold start — pass `--fresh-memory` for a clean baseline experiment."
            )

        args.strict_llm = _resolve_strict_llm_mode(args, effective_argv, parser)

        attacker_llm = LocalLLMClient(
            provider=attacker_provider,
            model_name=args.attacker_model,
            base_url=attacker_base_url,
            temperature=float(llm_cfg.get("attacker_model", {}).get("temperature", 0.7)),
            mode=args.llm_mode,
            timeout_seconds=args.llm_timeout,
            api_key=args.api_key,
            num_ctx=args.num_ctx,
        )
        defender_llm = LocalLLMClient(
            provider=defender_provider,
            model_name=args.defender_model,
            base_url=defender_base_url,
            temperature=float(llm_cfg.get("defender_model", {}).get("temperature", 0.2)),
            mode=args.llm_mode,
            timeout_seconds=args.llm_timeout,
            api_key=args.api_key,
            num_ctx=args.num_ctx,
        )

        attacker = AttackerAgent(
            llm_client=attacker_llm,
            blind_mode=args.blind_attacker,
            strict_llm=args.strict_llm,
            hide_test_results=args.hide_test_results,
        )
        defender = DefenderAgent(
            llm_client=defender_llm,
            blind_mode=args.blind_defender,
            strict_llm=args.strict_llm,
            hide_test_results=args.hide_test_results,
        )
        referee = DeterministicReferee(
            min_broken_tests=int(ref_cfg.get("min_broken_tests_for_valid_attack", 1)),
            max_broken_tests=int(ref_cfg.get("max_broken_tests_for_valid_attack", 2)),
            timeout_seconds=float(ref_cfg.get("execution_timeout_seconds", 3.0)),
            elo_k_factor=float(sp_cfg.get("elo_k_factor", 32.0)),
            rewards=ref_cfg.get("rewards") if isinstance(ref_cfg.get("rewards"), dict) else None,
        )

        resolved_dataset = _resolve_dataset_arg(
            args.dataset, ds_cfg.get("self_play_dataset_path")
        )
        resolved_exam_dataset = _resolve_dataset_arg(
            args.exam_dataset, ds_cfg.get("external_exam_dataset_path")
        )
        problems = load_problems(path=resolved_dataset, exam=False)
        exam_problems = load_problems(path=resolved_exam_dataset, exam=True)

        arena_ids = {p.problem_id for p in problems}
        arena_fns = {p.function_name for p in problems}
        overlap_ids = sorted(arena_ids & {p.problem_id for p in exam_problems})
        overlap_fns = sorted(arena_fns & {p.function_name for p in exam_problems})
        if overlap_ids or overlap_fns:
            console.print(
                f"[bold yellow]WARNING:[/bold yellow] Arena dataset and External Exam suite overlap "
                f"(problem_ids={overlap_ids[:5]}, functions={overlap_fns[:5]}). "
                f"Pass a disjoint `--exam-dataset` for strict unseen generalization measurement."
            )

        backend_badge = (
            "[bold green]REAL COGNEE SDK (add/cognify/search)[/bold green]"
            if memory.active_backend == "cognee"
            else "[bold yellow]LOCAL EPISODIC GRAPH (SQLite + AST/Diff)[/bold yellow]"
        )

        console.print(
            Panel.fit(
                f"[bold gold1]ARES: Plug-and-Play Adversarial Self-Play for Software Analysis[/bold gold1]\n"
                f"Orchestrator: [bold green]Real LangGraph StateGraph + MemorySaver[/bold green] | "
                f"Memory Backend: {backend_badge}\n"
                f"Memory Status: [dim]{memory.backend_status_reason}[/dim]\n"
                f"Referee: [bold cyan]100% Deterministic Subprocess-Isolated ({referee.min_broken_tests}-{referee.max_broken_tests} Broken Rule)[/bold cyan] | "
                f"Strict LLM: [bold]{'ON' if args.strict_llm else 'OFF'}[/bold]\n"
                f"Turns: [bold]{args.turns}[/bold] | External Exam Interval: [bold]{args.exam_interval}[/bold] | "
                f"Blind Mode: Attacker=[bold]{'ON' if args.blind_attacker else 'OFF'}[/bold], "
                f"Defender=[bold]{'ON' if args.blind_defender else 'OFF'}[/bold] | "
                f"Test Results Hidden: [bold]{'ON' if args.hide_test_results else 'OFF'}[/bold]\n"
                f"Arena Dataset: [bold]{len(problems)} problems[/bold] (`{args.dataset or ds_cfg.get('self_play_dataset_path') or 'default MLCQ'}`) | "
                f"External Exam Suite: [bold]{len(exam_problems)} problems[/bold] (`{args.exam_dataset or ds_cfg.get('external_exam_dataset_path') or 'default Exam'}`)",
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
            initial_elo=float(sp_cfg.get("elo_initial_rating", 1200.0)),
        )

        scoreboard = orchestrator.run_self_play(
            num_turns=args.turns,
            exam_interval=args.exam_interval,
            run_baseline_exam=bool(sp_cfg.get("run_baseline_exam", True)),
        )

        if args.export_memory:
            exported_path = memory.export_portable_bundle(
                args.export_memory, role=args.export_role
            )
            console.print(
                f"[bold green]Exported portable ARES memory bundle ({args.export_role}) to:[/bold green] {exported_path}"
            )

        console.print(turn_table)

        exam_table = Table(
            title="Periodic External Benchmark Exams ('Dışarıdan Sınavlar' - Unseen Code Suite)",
            header_style="bold magenta",
        )
        exam_table.add_column("Exam ID", style="bold")
        exam_table.add_column("After Turn", justify="right")
        exam_table.add_column("Memory KG Nodes", justify="right")
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
        mem_sys = scoreboard.get("memory_system", scoreboard["cognee_portable_memory"])
        llm_m = scoreboard.get("llm_usage_metrics", {})
        if imp:
            delta_block = (
                f"• External Exam Delta (Turn {imp['baseline_turn']} -> Turn {imp['final_turn']}):\n"
                f"  - Attacker Valid Subtle Attack Rate Delta: [bold cyan]{imp.get('attacker_valid_subtle_rate_delta', 0.0):+.2f}%[/bold cyan]\n"
                f"  - Defender Full Repair Rate Delta:         [bold green]{imp.get('defender_full_repair_rate_delta', 0.0):+.2f}%[/bold green]\n"
                f"  - Defender Classification Accuracy Delta:  [bold green]{imp.get('defender_classification_accuracy_delta', 0.0):+.2f}%[/bold green]\n"
                f"  - Defender Combined Score Delta:           [bold gold1]{imp.get('defender_combined_score_delta', 0.0):+.2f}%[/bold gold1]"
            )
        else:
            delta_block = (
                "• External Exam Delta: [dim]N/A (fewer than 2 exams completed in this run)[/dim]"
            )
        console.print(
            Panel.fit(
                f"[bold green]Self-Play & External Exam Summary[/bold green]\n"
                f"• Winner Scoreboard: Attacker Wins = [bold red]{scoreboard['attacker_wins']}[/bold red] | "
                f"Defender Wins = [bold green]{scoreboard['defender_wins']}[/bold green] | "
                f"Draws = {scoreboard['draws']}\n"
                f"• Final ELO Ratings: Attacker = [bold]{scoreboard['attacker_final_elo']}[/bold] | "
                f"Defender = [bold]{scoreboard['defender_final_elo']}[/bold]\n"
                f"• Real LLM Usage: Attacker = [bold]{llm_m.get('attacker_llm_rate_pct', 0.0)}%[/bold] | "
                f"Defender = [bold]{llm_m.get('defender_llm_rate_pct', 0.0)}%[/bold]\n"
                f"• Active Memory Backend: [bold magenta]{mem_sys.get('active_backend', 'local_graph').upper()}[/bold magenta] "
                f"(Real Cognee Ingests={mem_sys.get('cognee_ingested_turns', 0)}, Searches={mem_sys.get('cognee_search_calls', 0)})\n"
                f"• Graph & Episodic Store: [bold]{mem_sys['total_nodes']} KG nodes[/bold], "
                f"[bold]{mem_sys['total_edges']} edges[/bold], "
                f"[bold]{mem_sys['total_episodes']} episodes[/bold]\n"
                f"  Bundle saved at: `{mem_sys['bundle_path']}`\n"
                f"{delta_block}",
                border_style="green",
            )
        )
        return 0
    finally:
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
