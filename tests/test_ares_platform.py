"""Comprehensive unit and end-to-end integration tests for the ARES platform."""

from __future__ import annotations

from pathlib import Path
from ares.agents.attacker import AttackerAgent
from ares.agents.defender import DefenderAgent
from ares.agents.llm_client import LocalLLMClient
from ares.agents.prompt_loader import (
    DEFAULT_ATTACKER_PROMPT_PATH,
    DEFAULT_DEFENDER_PROMPT_PATH,
    load_prompt_config,
)
from ares.datasets.adapter import PlugAndPlayDatasetAdapter
from ares.datasets.loader import load_problems
from ares.memory.cognee_memory import CogneeMemoryManager
from ares.models import AttackerOutput, DefenderOutput
from ares.orchestrator.langgraph_engine import ARESLangGraphOrchestrator
from ares.referee.deterministic_referee import DeterministicReferee


def test_structured_prompt_configs_have_all_five_sections() -> None:
    """Task (i): Verify Attacker and Defender 5-part structured prompts (Task, Recipe, Output, Validation, Examples)."""
    att_cfg = load_prompt_config(DEFAULT_ATTACKER_PROMPT_PATH)
    def_cfg = load_prompt_config(DEFAULT_DEFENDER_PROMPT_PATH)

    assert att_cfg.agent_role == "attacker"
    assert def_cfg.agent_role == "defender"

    for cfg in (att_cfg, def_cfg):
        assert len(cfg.task.strip()) > 20
        assert len(cfg.recipe.strip()) > 20
        assert len(cfg.output.strip()) > 20
        assert len(cfg.validation.strip()) > 20
        assert len(cfg.examples) >= 1
        rendered = cfg.render_system_prompt(memories=["Sample Cognee Graph Memory"])
        assert "[1. TASK]" in rendered
        assert "[2. RECIPE]" in rendered
        assert "[3. OUTPUT SCHEMA]" in rendered
        assert "[4. VALIDATION CHECKLIST]" in rendered
        assert "[5. EXAMPLES]" in rendered
        assert "Sample Cognee Graph Memory" in rendered


def test_datasets_clean_code_and_subtle_mutations() -> None:
    """Verify all MLCQ Self-Play and External Exam problems have 5 deterministic tests and valid subtle mutations."""
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    arena_problems = load_problems(exam=False)
    exam_problems = load_problems(exam=True)

    assert len(arena_problems) >= 8
    assert len(exam_problems) >= 6

    for prob in arena_problems + exam_problems:
        assert 4 <= len(prob.test_cases) <= 5
        clean_res = referee.execute_test_suite(prob.clean_code, prob.test_cases)
        assert clean_res.syntax_valid, f"{prob.problem_id} clean_code has syntax error"
        assert clean_res.failed_count == 0, (
            f"{prob.problem_id} clean_code failed tests: {clean_res.failed_test_ids}"
        )

        subtle_found = False
        for mut in prob.candidate_mutations:
            mut_res = referee.execute_test_suite(mut.mutated_code, prob.test_cases)
            assert mut_res.syntax_valid
            assert mut_res.failed_count == mut.expected_broken_count, (
                f"{prob.problem_id}::{mut.mutation_id} expected {mut.expected_broken_count} "
                f"broken tests, got {mut_res.failed_count} ({mut_res.failed_test_ids})"
            )
            if 1 <= mut_res.failed_count <= 2:
                subtle_found = True
        assert subtle_found, f"{prob.problem_id} lacks a valid 1-2 test subtle mutation"


def test_deterministic_referee_scoring_rules() -> None:
    """Task (ii): Verify Deterministic Referee enforces 1-2 broken test rule, rewards, and feedback."""
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    problem = load_problems(exam=False)[0]  # MLCQ-TD-001

    subtle_mut = problem.candidate_mutations[0]  # breaks 1 test (T4)
    over_mut = problem.candidate_mutations[2]    # breaks 4 tests

    att_valid = AttackerOutput(
        mutated_code=subtle_mut.mutated_code,
        injected_defect_type=subtle_mut.defect_type,
        targeted_edge_case=subtle_mut.targeted_edge_case,
        attack_strategy=subtle_mut.strategy_name,
        expected_failing_tests=1,
        explanation=subtle_mut.explanation,
    )
    def_repaired = DefenderOutput(
        is_defective=True,
        predicted_defect_type=subtle_mut.defect_type,
        defect_location_line=4,
        diagnosis="Zero initialization bug",
        defense_strategy="Restore nums[0]",
        repaired_code=problem.clean_code,
    )
    verdict_a = referee.evaluate_turn(1, problem, att_valid, def_repaired)
    assert verdict_a.attack_validity == "valid_subtle_attack"
    assert verdict_a.attacker_suite_result.failed_count == 1
    assert verdict_a.defense_repaired_all_tests is True
    assert verdict_a.classification_correct is True
    assert verdict_a.winner == "defender"
    assert verdict_a.defender_reward == 1.0
    assert verdict_a.attacker_reward == 0.25

    def_failed = DefenderOutput(
        is_defective=True,
        predicted_defect_type="off_by_one_boundary",
        defect_location_line=4,
        diagnosis="Missed root cause",
        defense_strategy="Unchanged",
        repaired_code=subtle_mut.mutated_code,
    )
    verdict_b = referee.evaluate_turn(2, problem, att_valid, def_failed)
    assert verdict_b.attack_validity == "valid_subtle_attack"
    assert verdict_b.defense_repaired_all_tests is False
    assert verdict_b.winner == "attacker"
    assert verdict_b.attacker_reward == 1.0
    assert verdict_b.defender_reward < 0

    att_over = AttackerOutput(
        mutated_code=over_mut.mutated_code,
        injected_defect_type=over_mut.defect_type,
        targeted_edge_case=over_mut.targeted_edge_case,
        attack_strategy=over_mut.strategy_name,
        expected_failing_tests=4,
        explanation=over_mut.explanation,
    )
    verdict_c = referee.evaluate_turn(3, problem, att_over, def_repaired)
    assert verdict_c.attack_validity == "invalid_over_broken_attack"
    assert verdict_c.attacker_reward == -0.9
    assert verdict_c.winner == "defender"


def test_on_demand_memory_zip_export_import_and_merge(tmp_path: Path) -> None:
    """Task (iii): Verify Cognee Portable Memory export/import (.json and .zip) and merge mode."""
    problems = load_problems(exam=False)
    referee = DeterministicReferee()

    # Memory Store A: learns problem 0
    mem_a = CogneeMemoryManager(store_dir=tmp_path / "mem_a", backend="local_graph")
    p0 = problems[0]
    m0 = p0.candidate_mutations[0]
    att0 = AttackerOutput(
        mutated_code=m0.mutated_code,
        injected_defect_type=m0.defect_type,
        targeted_edge_case=m0.targeted_edge_case,
        attack_strategy=m0.strategy_name,
        expected_failing_tests=1,
        explanation=m0.explanation,
    )
    def0 = DefenderOutput(
        is_defective=True,
        predicted_defect_type=m0.defect_type,
        defect_location_line=4,
        diagnosis="Fixed",
        defense_strategy="Restored nums[0]",
        repaired_code=p0.clean_code,
    )
    v0 = referee.evaluate_turn(1, p0, att0, def0)
    mem_a.record_turn_memory(1, p0, att0, def0, v0)

    zip_path = tmp_path / "portable_brain.zip"
    mem_a.export_portable_bundle(zip_path, role="both")
    assert zip_path.exists()

    # Inspect the exported zip bundle
    inspected = CogneeMemoryManager.inspect_bundle(zip_path)
    assert inspected["summary"]["total_episodes"] == 1
    assert m0.defect_type in inspected["defender_defect_breakdown"]

    # Memory Store B: learns problem 5 (tie_breaker_or_duplicate_debt) and then MERGES zip_path!
    mem_b = CogneeMemoryManager(store_dir=tmp_path / "mem_b", backend="local_graph")
    p5 = problems[5]
    m5 = p5.candidate_mutations[0]
    att5 = AttackerOutput(
        mutated_code=m5.mutated_code,
        injected_defect_type=m5.defect_type,
        targeted_edge_case=m5.targeted_edge_case,
        attack_strategy=m5.strategy_name,
        expected_failing_tests=1,
        explanation=m5.explanation,
    )
    def5 = DefenderOutput(
        is_defective=True,
        predicted_defect_type=m5.defect_type,
        defect_location_line=4,
        diagnosis="Fixed tie break",
        defense_strategy="Restored (-item[1], item[0])",
        repaired_code=p5.clean_code,
    )
    v5 = referee.evaluate_turn(2, p5, att5, def5)
    mem_b.record_turn_memory(2, p5, att5, def5, v5)

    merge_summary = mem_b.import_portable_bundle(zip_path, mode="merge")
    assert merge_summary["total_episodes"] == 2
    assert mem_b.has_defender_learned_defect(m0.defect_type)
    assert mem_b.has_defender_learned_defect(m5.defect_type)


def test_plug_and_play_custom_dataset_without_candidate_mutations(tmp_path: Path) -> None:
    """Verify any raw external dataset (.jsonl with assert strings and no candidate_mutations) runs end-to-end."""
    sample_jsonl = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sample_plug_and_play_dataset.jsonl"
    )
    report = PlugAndPlayDatasetAdapter.validate_dataset_report(sample_jsonl)
    assert report["total_problems"] == 3
    assert report["all_clean_code_valid"] is True
    for row in report["problems"]:
        assert row["valid_1_to_2_broken_mutations"] >= 1

    custom_problems = load_problems(path=sample_jsonl)
    llm = LocalLLMClient(mode="adaptive_local")
    orchestrator = ARESLangGraphOrchestrator(
        problems=custom_problems,
        exam_problems=custom_problems,
        attacker=AttackerAgent(llm_client=llm),
        defender=DefenderAgent(llm_client=llm),
        referee=DeterministicReferee(min_broken_tests=1, max_broken_tests=2),
        memory=CogneeMemoryManager(store_dir=tmp_path / "plug_mem", backend="local_graph"),
        output_dir=tmp_path / "plug_out",
    )

    scoreboard = orchestrator.run_self_play(
        num_turns=9,
        exam_interval=9,
        run_baseline_exam=True,
    )
    assert scoreboard["total_turns"] == 9
    assert scoreboard["external_exams"][-1]["defender_full_repair_rate"] == 100.0


def test_end_to_end_langgraph_25_turns_and_external_exams(tmp_path: Path) -> None:
    """Task (iii) & (iv): Validate 25-turn real LangGraph StateGraph + MemorySaver self-play and exams."""
    llm = LocalLLMClient(mode="adaptive_local")
    attacker = AttackerAgent(llm_client=llm)
    defender = DefenderAgent(llm_client=llm)
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    memory = CogneeMemoryManager(store_dir=tmp_path / "cognee_store", backend="local_graph")

    orchestrator = ARESLangGraphOrchestrator(
        attacker=attacker,
        defender=defender,
        referee=referee,
        memory=memory,
        output_dir=tmp_path / "outputs",
    )

    scoreboard = orchestrator.run_self_play(
        num_turns=25,
        exam_interval=25,
        run_baseline_exam=True,
    )

    assert scoreboard["total_turns"] == 25
    assert scoreboard["attacker_wins"] + scoreboard["defender_wins"] + scoreboard["draws"] == 25
    assert scoreboard["langgraph_orchestrator"]["checkpointer"] == "MemorySaver"
    assert "generate_reflexion" in scoreboard["langgraph_orchestrator"]["nodes"]
    assert scoreboard["memory_system"]["active_backend"] == "local_graph"
    assert scoreboard["external_exams"][-1]["defender_full_repair_rate"] == 100.0
    assert scoreboard["external_exams"][-1]["attacker_valid_subtle_rate"] == 100.0

    # Verify real LangGraph MemorySaver checkpoint state is persisted
    snapshot = orchestrator.get_graph_checkpoint_state()
    assert snapshot.values["turn_id"] == 25
    assert "reflexion" in snapshot.values
    assert snapshot.values["reflexion"]["attack_diff"]


def test_blind_mode_hides_test_cases_from_prompts() -> None:
    """Verify blind_mode toggles hide test case expressions and failing details from agent prompts."""
    problem = load_problems(exam=False)[0]
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    mutated_code = problem.candidate_mutations[0].mutated_code
    suite = referee.execute_test_suite(mutated_code, problem.test_cases)
    failed_tc = next(tc for tc in problem.test_cases if tc.test_id in suite.failed_test_ids)

    open_attacker = AttackerAgent(blind_mode=False)
    blind_attacker = AttackerAgent(blind_mode=True)
    _, open_att_prompt = open_attacker.build_prompts(problem, memories=[])
    _, blind_att_prompt = blind_attacker.build_prompts(problem, memories=[])

    assert problem.test_cases[0].call_expr in open_att_prompt
    assert "[HIDDEN - BLIND MODE]" in blind_att_prompt
    for tc in problem.test_cases:
        assert tc.call_expr not in blind_att_prompt

    open_defender = DefenderAgent(blind_mode=False)
    blind_defender = DefenderAgent(blind_mode=True)
    _, open_def_prompt = open_defender.build_prompts(problem, mutated_code, suite, memories=[])
    _, blind_def_prompt = blind_defender.build_prompts(problem, mutated_code, suite, memories=[])

    assert failed_tc.call_expr in open_def_prompt
    assert "[HIDDEN - BLIND MODE]" in blind_def_prompt
    assert failed_tc.call_expr not in blind_def_prompt
    assert f"{suite.failed_count}/{suite.total_tests} hidden tests failing" in blind_def_prompt


def test_referee_kills_infinite_loop_without_hanging() -> None:
    """Verify DeterministicReferee terminates infinite loops cleanly via subprocess isolation."""
    from ares.models import UnitTestCase

    referee = DeterministicReferee(timeout_seconds=0.5)
    infinite_code = "def spin(n):\n    while True:\n        pass\n"
    tc = UnitTestCase(
        test_id="T1",
        description="Infinite loop timeout test",
        call_expr="spin(1)",
        expected_output=1,
        is_edge_case=True,
    )
    res = referee.execute_test_suite(infinite_code, [tc])
    assert res.syntax_valid is True
    assert res.timed_out is True
    assert res.failed_count == 1
    assert "TimeoutExceeded" in (res.test_details[0].error_message or "")


def test_extract_json_object_handles_nested_braces() -> None:
    """Verify extract_json_object does not truncate nested braces inside fenced JSON."""
    from ares.agents.llm_client import extract_json_object

    raw = '```json\n{"mutated_code": "def f(): return {1: 2}", "expected_failing_tests": 1}\n```'
    parsed = extract_json_object(raw)
    assert parsed["mutated_code"] == "def f(): return {1: 2}"
    assert parsed["expected_failing_tests"] == 1


def test_llm_output_normalization_prevents_validation_crash() -> None:
    """Verify non-standard LLM defect labels and integer strings normalize without ValidationError."""
    from ares.models import normalize_defect_category, safe_int

    assert normalize_defect_category("off-by-one boundary") == "off_by_one_boundary"
    assert normalize_defect_category("boundary_error") == "off_by_one_boundary"
    assert normalize_defect_category("empty list edge case") == "empty_or_singleton_debt"
    assert normalize_defect_category("negative value bug") == "negative_or_zero_edge_debt"
    assert normalize_defect_category("float rounding issue") == "precision_or_rounding_debt"
    assert normalize_defect_category("duplicate tie break") == "tie_breaker_or_duplicate_debt"
    assert normalize_defect_category("accumulator state") == "state_accumulation_debt"
    assert normalize_defect_category(None) == "off_by_one_boundary"

    assert safe_int("line 4", default=1) == 4
    assert safe_int("1-2 tests", default=1) == 1
    assert safe_int(None, default=3) == 3


def test_real_cognee_backend_is_honest_and_never_faked(tmp_path: Path) -> None:
    """Verify explicit backend='cognee' raises when unavailable, and 'auto' honestly reports 'local_graph'."""
    import pytest

    mem_auto = CogneeMemoryManager(
        store_dir=tmp_path / "auto_mem",
        backend="auto",
        llm_endpoint="http://127.0.0.1:1",  # Guaranteed unreachable port
    )
    assert mem_auto.active_backend == "local_graph"

    with pytest.raises(RuntimeError, match="Real Cognee backend"):
        CogneeMemoryManager(
            store_dir=tmp_path / "strict_cognee_mem",
            backend="cognee",
            llm_endpoint="http://127.0.0.1:1",
        )


def test_reflexion_produces_diffs_and_never_leaks_clean_code_on_failure(tmp_path: Path) -> None:
    """Verify failed Defender repairs NEVER leak ground-truth clean_code into verified_repairs."""
    from ares.memory.reflexion import build_turn_reflexion, extract_ast_features

    problem = load_problems(exam=False)[0]
    mut = problem.candidate_mutations[0]
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    memory = CogneeMemoryManager(store_dir=tmp_path / "no_leak_mem", backend="local_graph")

    att_out = AttackerOutput(
        mutated_code=mut.mutated_code,
        injected_defect_type=mut.defect_type,
        targeted_edge_case=mut.targeted_edge_case,
        attack_strategy=mut.strategy_name,
        expected_failing_tests=1,
        explanation=mut.explanation,
    )
    def_fail = DefenderOutput(
        is_defective=True,
        predicted_defect_type=mut.defect_type,
        defect_location_line=4,
        diagnosis="Incomplete fix",
        defense_strategy="Left code unchanged",
        repaired_code=mut.mutated_code,
    )
    verdict = referee.evaluate_turn(1, problem, att_out, def_fail)
    assert verdict.defense_repaired_all_tests is False

    reflexion = build_turn_reflexion(
        1, problem, att_out, def_fail, verdict, blind_attacker=True, blind_defender=True
    )
    assert len(extract_ast_features(problem.clean_code)) >= 1
    assert "-" in reflexion.attack_diff and "+" in reflexion.attack_diff
    assert "[WINNING SUBTLE ATTACK" in reflexion.attacker_lesson
    assert "[FAILED REPAIR REFLEXION" in reflexion.defender_lesson
    assert problem.clean_code.strip() not in reflexion.defender_lesson

    memory.record_turn_memory(
        1, problem, att_out, def_fail, verdict, reflexion=reflexion, blind_defender=True
    )
    d_know = memory.defender_repair_knowledge[verdict.ground_truth_defect]
    assert problem.function_name not in d_know["verified_repairs"]


def test_strict_llm_static_retry_recovers_without_heuristic_fallback(tmp_path: Path) -> None:
    """Verify AttackerAgent retries once on invalid/no-op LLM output and succeeds with used_llm=True."""

    class FakeRetryLLM(LocalLLMClient):
        def __init__(self, clean_code: str, good_mutated: str) -> None:
            super().__init__(mode="ollama")
            self.calls = 0
            self.clean_code = clean_code
            self.good_mutated = good_mutated

        def is_live_llm_available(self) -> bool:
            return True

        def generate_json(self, system_prompt: str, user_prompt: str) -> dict | None:
            self.calls += 1
            if self.calls == 1:
                # First attempt: returns unchanged code (no-op) -> triggers static retry
                return {
                    "mutated_code": self.clean_code,
                    "injected_defect_type": "negative_or_zero_edge_debt",
                }
            # Second attempt: valid mutation
            return {
                "mutated_code": self.good_mutated,
                "injected_defect_type": "negative_or_zero_edge_debt",
                "targeted_edge_case": "All negative values",
                "attack_strategy": "Zero init",
                "expected_failing_tests": 1,
                "explanation": "Zero floor breaks negative-only input",
            }

    problem = load_problems(exam=False)[0]
    mut = problem.candidate_mutations[0]
    fake_llm = FakeRetryLLM(problem.clean_code, mut.mutated_code)
    memory = CogneeMemoryManager(store_dir=tmp_path / "retry_mem", backend="local_graph")
    attacker = AttackerAgent(llm_client=fake_llm, blind_mode=True, strict_llm=True)

    out = attacker.attack(problem, memory, turn_id=1)
    assert fake_llm.calls == 2
    assert out.used_llm is True
    assert out.mutated_code == mut.mutated_code


def test_referee_blocks_dangerous_code_and_draws_on_unrepaired_invalid_attack() -> None:
    """Verify AST safety filter blocks dangerous imports/calls and lazy Defender gets draw on invalid attack."""
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    problem = load_problems(exam=False)[0]

    malicious_code = "import os\ndef max_subarray_sum(nums):\n    return os.getcwd()\n"
    res = referee.execute_test_suite(malicious_code, problem.test_cases)
    assert res.syntax_valid is False
    assert "SecurityPolicyViolation" in (res.syntax_or_runtime_error or "")

    over_mut = problem.candidate_mutations[2]
    att_over = AttackerOutput(
        mutated_code=over_mut.mutated_code,
        injected_defect_type=over_mut.defect_type,
        targeted_edge_case=over_mut.targeted_edge_case,
        attack_strategy=over_mut.strategy_name,
        expected_failing_tests=4,
        explanation=over_mut.explanation,
    )
    lazy_def = DefenderOutput(
        is_defective=True,
        predicted_defect_type="off_by_one_boundary",
        defect_location_line=1,
        diagnosis="Did not fix",
        defense_strategy="No change",
        repaired_code=over_mut.mutated_code,
    )
    verdict = referee.evaluate_turn(1, problem, att_over, lazy_def)
    assert verdict.attack_validity == "invalid_over_broken_attack"
    assert verdict.defense_repaired_all_tests is False
    assert verdict.winner == "draw"


def test_extract_json_multi_fence_and_embedding_model_filter() -> None:
    """Verify multi-fence JSON extraction and embedding model filtering in LocalLLMClient."""
    from ares.agents.llm_client import _is_embedding_model, extract_json_object

    multi_fence = (
        "Here is my analysis:\n```python\ndef foo(x): return x + 1\n```\n"
        'Final output:\n```json\n{"mutated_code": "def foo(x): return x", "expected_failing_tests": 1}\n```'
    )
    parsed = extract_json_object(multi_fence)
    assert parsed["mutated_code"] == "def foo(x): return x"

    assert _is_embedding_model("nomic-embed-text:latest") is True
    assert _is_embedding_model("bge-m3:latest") is True
    assert _is_embedding_model("qwen2.5-coder:7b") is False
    assert _is_embedding_model("llama3.1:8b") is False


def test_import_portable_bundle_syncs_sqlite_database(tmp_path: Path) -> None:
    """Verify SQLite episodes/nodes stay synchronized after import_portable_bundle."""
    import sqlite3

    problems = load_problems(exam=False)
    referee = DeterministicReferee()
    mem_src = CogneeMemoryManager(store_dir=tmp_path / "src_mem", backend="local_graph")
    p0 = problems[0]
    m0 = p0.candidate_mutations[0]
    att0 = AttackerOutput(
        mutated_code=m0.mutated_code,
        injected_defect_type=m0.defect_type,
        targeted_edge_case=m0.targeted_edge_case,
        attack_strategy=m0.strategy_name,
        expected_failing_tests=1,
        explanation=m0.explanation,
    )
    def0 = DefenderOutput(
        is_defective=True,
        predicted_defect_type=m0.defect_type,
        defect_location_line=4,
        diagnosis="Fixed",
        defense_strategy="Restored init",
        repaired_code=p0.clean_code,
    )
    v0 = referee.evaluate_turn(1, p0, att0, def0)
    mem_src.record_turn_memory(1, p0, att0, def0, v0)
    bundle = mem_src.export_portable_bundle(tmp_path / "brain.json")

    mem_dst = CogneeMemoryManager(store_dir=tmp_path / "dst_mem", backend="local_graph")
    mem_dst.import_portable_bundle(bundle, mode="merge")

    with sqlite3.connect(mem_dst.sqlite_path) as conn:
        ep_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        node_count = conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
    assert ep_count == 1
    assert node_count == len(mem_dst.nodes)


def test_game_config_yaml_loads_into_cli_parser() -> None:
    """Verify configs/game_config.yaml is actually loaded into CLI defaults."""
    from ares.cli import build_parser, load_game_config

    cfg = load_game_config()
    assert cfg.get("self_play", {}).get("default_turns") == 25
    assert cfg.get("self_play", {}).get("strict_llm") is True
    parser = build_parser(cfg)
    args = parser.parse_args([])
    assert args.turns == 25
    assert args.embedding_model == "nomic-embed-text"
    assert args.cognify_interval == 1
    assert args.num_ctx == 8192
    assert args.llm_timeout == 60.0
    assert args.strict_llm is True
    assert args.hide_test_results is True
    assert args.blind_attacker is False
    assert args.blind_defender is False

    # Verify BooleanOptionalAction toggles override YAML defaults cleanly
    toggled = parser.parse_args(
        ["--no-strict-llm", "--no-hide-test-results", "--blind-attacker", "--blind-defender"]
    )
    assert toggled.strict_llm is False
    assert toggled.hide_test_results is False
    assert toggled.blind_attacker is True
    assert toggled.blind_defender is True

    untoggled = parser.parse_args(
        ["--strict-llm", "--hide-test-results", "--no-blind-attacker", "--no-blind-defender"]
    )
    assert untoggled.strict_llm is True
    assert untoggled.hide_test_results is True
    assert untoggled.blind_attacker is False
    assert untoggled.blind_defender is False


def test_strict_llm_and_adaptive_local_cli_interaction(tmp_path: Path) -> None:
    """Verify adaptive_local auto-disables default strict_llm, rejects explicit --strict-llm, and gives clear errors."""
    import pytest
    from ares.cli import _resolve_strict_llm_mode, build_parser, load_game_config

    cfg = load_game_config()
    parser = build_parser(cfg)

    # 1. Explicit `--llm-mode adaptive_local` without `--strict-llm` disables strict_llm cleanly
    argv_auto_off = ["--llm-mode", "adaptive_local"]
    args_auto_off = parser.parse_args(argv_auto_off)
    assert args_auto_off.strict_llm is True
    assert _resolve_strict_llm_mode(args_auto_off, argv_auto_off, parser) is False

    # 2. Contradictory `--llm-mode adaptive_local --strict-llm` fails fast at CLI parsing
    argv_conflict = ["--llm-mode", "adaptive_local", "--strict-llm"]
    args_conflict = parser.parse_args(argv_conflict)
    with pytest.raises(SystemExit) as exc_info:
        _resolve_strict_llm_mode(args_conflict, argv_conflict, parser)
    assert exc_info.value.code == 2

    # 3. Direct Python API with adaptive_local + strict_llm=True raises an accurate, actionable RuntimeError
    problem = load_problems(exam=False)[0]
    memory = CogneeMemoryManager(store_dir=tmp_path / "strict_err_mem", backend="local_graph")
    offline_llm = LocalLLMClient(mode="adaptive_local")

    attacker = AttackerAgent(llm_client=offline_llm, strict_llm=True)
    with pytest.raises(RuntimeError, match="adaptive_local") as att_err:
        attacker.attack(problem, memory, turn_id=1)
    assert "--no-strict-llm" in str(att_err.value)

    referee = DeterministicReferee()
    suite = referee.execute_test_suite(problem.candidate_mutations[0].mutated_code, problem.test_cases)
    defender = DefenderAgent(llm_client=offline_llm, strict_llm=True)
    with pytest.raises(RuntimeError, match="adaptive_local") as def_err:
        defender.defend(problem, problem.candidate_mutations[0].mutated_code, suite, memory, turn_id=1)
    assert "--no-strict-llm" in str(def_err.value)


def test_legacy_v1_bundle_migrates_without_keyerror_and_strips_leaked_repairs(
    tmp_path: Path,
) -> None:
    """Verify loading a pre-v3 bundle (missing verified_repair_diffs) never crashes and strips legacy leaks."""
    import json

    problems = load_problems(exam=False)
    p0 = problems[0]
    m0 = p0.candidate_mutations[0]
    referee = DeterministicReferee()

    legacy_bundle = {
        "schema_version": "ares-cognee-portable-1.0",
        "nodes": {},
        "edges": [],
        "attacker_strategy_stats": {},
        "defender_repair_knowledge": {
            m0.defect_type: {
                "defect_type": m0.defect_type,
                "encounters": 3,
                "repairs_succeeded": 1,
                "classifications_succeeded": 2,
                "known_code_clues": ["old clue"],
                "verified_repairs": {p0.function_name: p0.clean_code},
                # Intentionally missing `verified_repair_diffs`
            }
        },
        "attacker_in_context": [],
        "defender_in_context": [],
        "episodes": [],
    }
    store_dir = tmp_path / "legacy_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "cognee_portable_graph.json").write_text(
        json.dumps(legacy_bundle), encoding="utf-8"
    )

    memory = CogneeMemoryManager(store_dir=store_dir, backend="local_graph")
    assert memory.migrated_legacy_schema == "ares-cognee-portable-1.0"
    assert memory.defender_repair_knowledge[m0.defect_type]["verified_repairs"] == {}
    assert memory.defender_repair_knowledge[m0.defect_type]["verified_repair_diffs"] == []

    att0 = AttackerOutput(
        mutated_code=m0.mutated_code,
        injected_defect_type=m0.defect_type,
        targeted_edge_case=m0.targeted_edge_case,
        attack_strategy=m0.strategy_name,
        expected_failing_tests=1,
        explanation=m0.explanation,
    )
    def0 = DefenderOutput(
        is_defective=True,
        predicted_defect_type=m0.defect_type,
        defect_location_line=4,
        diagnosis="Fixed",
        defense_strategy="Restored init",
        repaired_code=p0.clean_code,
    )
    v0 = referee.evaluate_turn(1, p0, att0, def0)
    memory.record_turn_memory(1, p0, att0, def0, v0)
    assert len(memory.defender_repair_knowledge[m0.defect_type]["verified_repair_diffs"]) == 1


def test_referee_setup_code_isolation_scrubbed_env_and_getattr_block(
    monkeypatch,
) -> None:
    """Verify trusted setup_code executes cleanly, secrets are scrubbed from env, and getattr() is blocked."""
    from ares.models import UnitTestCase
    from ares.referee.deterministic_referee import _safe_subprocess_env

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-must-not-leak")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret-must-not-leak")
    safe_env = _safe_subprocess_env()
    assert "OPENAI_API_KEY" not in safe_env
    assert "LLM_API_KEY" not in safe_env

    referee = DeterministicReferee()
    tc = UnitTestCase(
        test_id="T1",
        description="Trusted helper class in setup_code",
        call_expr="unwrap(Box(42))",
        expected_output=42,
        setup_code="class Box:\n    def __init__(self, v):\n        self.v = v\n",
    )
    res_ok = referee.execute_test_suite("def unwrap(b):\n    return b.v\n", [tc])
    assert res_ok.syntax_valid is True
    assert res_ok.failed_count == 0

    bypass_code = "def unwrap(b):\n    return getattr(b, 'v')\n"
    res_blocked = referee.execute_test_suite(bypass_code, [tc])
    assert res_blocked.syntax_valid is False
    assert "SecurityPolicyViolation" in (res_blocked.syntax_or_runtime_error or "")


def test_referee_tracks_ground_truth_source() -> None:
    """Verify RefereeVerdict distinguishes curated, ast_diff (clean), and attacker_claim labels."""
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    problem = load_problems(exam=False)[0]
    curated = problem.candidate_mutations[0]

    def_clean = DefenderOutput(
        is_defective=True,
        predicted_defect_type=curated.defect_type,
        defect_location_line=1,
        diagnosis="Fix",
        defense_strategy="Fix",
        repaired_code=problem.clean_code,
    )

    att_curated = AttackerOutput(
        mutated_code=curated.mutated_code,
        injected_defect_type="off_by_one_boundary",  # Intentionally mismatched claim
        targeted_edge_case="edge",
        attack_strategy="curated",
        explanation="curated",
    )
    v_curated = referee.evaluate_turn(1, problem, att_curated, def_clean)
    assert v_curated.ground_truth_source == "curated"
    assert v_curated.ground_truth_defect == curated.defect_type

    att_noop = AttackerOutput(
        mutated_code=problem.clean_code,
        injected_defect_type="off_by_one_boundary",
        targeted_edge_case="none",
        attack_strategy="noop",
        explanation="noop",
    )
    v_noop = referee.evaluate_turn(2, problem, att_noop, def_clean)
    assert v_noop.ground_truth_source == "ast_diff"
    assert v_noop.ground_truth_defect == "clean"

    novel_code = problem.clean_code.replace("return best", "return best + 1")
    assert novel_code != problem.clean_code
    att_novel = AttackerOutput(
        mutated_code=novel_code,
        injected_defect_type="off_by_one_boundary",
        targeted_edge_case="all",
        attack_strategy="novel",
        explanation="novel",
    )
    v_novel = referee.evaluate_turn(3, problem, att_novel, def_clean)
    assert v_novel.ground_truth_source == "attacker_claim"
    assert v_novel.ground_truth_defect == "off_by_one_boundary"


def test_provider_inference_embedding_dims_and_persistent_cognee_loop(
    tmp_path: Path,
) -> None:
    """Verify auto provider detection, embedding dim resolution, URL joining, and persistent async loop."""
    import asyncio
    from ares.agents.llm_client import infer_llm_provider
    from ares.cli import _resolve_embedding_dimensions
    from ares.memory.cognee_engine import RealCogneeEngine

    assert infer_llm_provider("auto", "http://localhost:11434", None) == "ollama"
    assert infer_llm_provider("auto", "http://localhost:1234/v1", None) == "openai_compatible"
    assert infer_llm_provider("auto", "https://api.openai.com/v1", "sk-test") == "openai_compatible"

    assert _resolve_embedding_dimensions(None, "text-embedding-3-small", 768) == 1536
    assert _resolve_embedding_dimensions(None, "text-embedding-3-large", 768) == 3072
    assert _resolve_embedding_dimensions(None, "nomic-embed-text:latest", 768) == 768
    assert _resolve_embedding_dimensions(512, "text-embedding-3-small", 768) == 512

    engine = RealCogneeEngine(
        store_dir=tmp_path / "cognee_loop",
        llm_provider="openai_compatible",
        llm_endpoint="http://localhost:1234/v1",
    )
    assert engine._openai_models_url() == "http://localhost:1234/v1/models"

    async def _get_loop_id(val: int) -> tuple[int, int]:
        return val, id(asyncio.get_running_loop())

    r1 = engine._run_async(_get_loop_id(1))
    r2 = engine._run_async(_get_loop_id(2))
    engine.close()
    assert r1[0] == 1 and r2[0] == 2
    assert r1[1] == r2[1]


def test_referee_strict_bool_int_min_broken_and_stdout_noise() -> None:
    """Verify bool != int, min_broken_tests > 1 enforcement, and noisy stdout safety in single runner."""
    from ares.models import UnitTestCase
    from ares.referee.deterministic_referee import _run_single_test_isolated

    referee = DeterministicReferee(min_broken_tests=2, max_broken_tests=3)
    tc_bool = UnitTestCase(
        test_id="T1",
        description="Must return bool True, not int 1",
        call_expr="is_pos(5)",
        expected_output=True,
    )
    res_int = referee.execute_test_suite("def is_pos(x):\n    return 1\n", [tc_bool])
    assert res_int.failed_count == 1
    res_bool = referee.execute_test_suite("def is_pos(x):\n    return True\n", [tc_bool])
    assert res_bool.failed_count == 0

    # Single-test subprocess must ignore print(..., end='') noise before result marker
    ok, actual, err, timed_out = _run_single_test_isolated(
        "def noisy(x):\n    print('debug', end='')\n    return x * 2\n",
        "noisy(21)",
        timeout_seconds=3.0,
    )
    assert ok is True and actual == 42 and err is None and timed_out is False

    # Enforce min_broken_tests=2: a 1-broken mutation must be rejected as invalid_no_op_attack
    problem = load_problems(exam=False)[0]
    one_broken_mut = problem.candidate_mutations[0]
    att_out = AttackerOutput(
        mutated_code=one_broken_mut.mutated_code,
        injected_defect_type=one_broken_mut.defect_type,
        targeted_edge_case=one_broken_mut.targeted_edge_case,
        attack_strategy=one_broken_mut.strategy_name,
        explanation=one_broken_mut.explanation,
    )
    def_out = DefenderOutput(
        is_defective=True,
        predicted_defect_type=one_broken_mut.defect_type,
        defect_location_line=1,
        diagnosis="Fix",
        defense_strategy="Fix",
        repaired_code=problem.clean_code,
    )
    verdict = referee.evaluate_turn(1, problem, att_out, def_out)
    assert verdict.attacker_suite_result.failed_count == 1
    assert verdict.attack_validity == "invalid_no_op_attack"


def test_memory_zip_slip_blocked_and_resumed_episode_ids_unique(tmp_path: Path) -> None:
    """Verify zip-slip traversal entries are rejected and resumed runs produce unique episode IDs."""
    import json
    import sqlite3
    import zipfile
    import pytest

    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("cognee_portable_graph.json", json.dumps({"schema_version": "ares-memory-portable-3.0"}))
        zf.writestr("cognee_system/../../escaped.txt", "pwned")

    mem = CogneeMemoryManager(store_dir=tmp_path / "safe_store", backend="local_graph")
    with pytest.raises(ValueError, match="Unsafe path traversal"):
        mem.import_portable_bundle(bad_zip, mode="replace")
    assert not (tmp_path / "escaped.txt").exists()

    # Recording turn_id=1 twice across resumed runs must keep unique episode IDs and SQLite in sync
    problem = load_problems(exam=False)[0]
    mut = problem.candidate_mutations[0]
    referee = DeterministicReferee()
    att = AttackerOutput(
        mutated_code=mut.mutated_code,
        injected_defect_type=mut.defect_type,
        targeted_edge_case=mut.targeted_edge_case,
        attack_strategy=mut.strategy_name,
        explanation=mut.explanation,
    )
    dfn = DefenderOutput(
        is_defective=True,
        predicted_defect_type=mut.defect_type,
        defect_location_line=4,
        diagnosis="Fix",
        defense_strategy="Fix",
        repaired_code=problem.clean_code,
    )
    verdict = referee.evaluate_turn(1, problem, att, dfn)
    ids1 = mem.record_turn_memory(1, problem, att, dfn, verdict)
    ids2 = mem.record_turn_memory(1, problem, att, dfn, verdict)
    assert ids1[0] != ids2[0]
    assert len(mem.episodes) == 2
    with sqlite3.connect(mem.sqlite_path) as conn:
        ep_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert ep_count == 2


def test_defender_preserves_clean_code_and_ast_fn_validation(tmp_path: Path) -> None:
    """Verify blind Defender does not corrupt clean code and LLM validation checks true AST FunctionDef."""
    problem = load_problems(exam=False)[0]
    referee = DeterministicReferee()
    clean_suite = referee.execute_test_suite(problem.clean_code, problem.test_cases)
    assert clean_suite.failed_count == 0

    mem = CogneeMemoryManager(store_dir=tmp_path / "clean_def_mem", backend="local_graph")
    blind_def = DefenderAgent(llm_client=LocalLLMClient(mode="adaptive_local"), blind_mode=True)
    out = blind_def.defend(problem, problem.clean_code, clean_suite, mem, turn_id=1)
    assert out.predicted_defect_type == "clean"
    assert out.repaired_code == problem.clean_code

    # Comment or prefix-spoofed function names must be rejected by AST validation
    attacker = AttackerAgent(llm_client=LocalLLMClient(mode="adaptive_local"))
    ok_comment, _ = attacker._validate_llm_attack_reason(
        {"mutated_code": f"# def {problem.function_name}\ndef other(x): return x\n"},
        problem,
    )
    ok_prefix, _ = attacker._validate_llm_attack_reason(
        {"mutated_code": f"def {problem.function_name}_extra(x): return x\n"},
        problem,
    )
    assert ok_comment is False
    assert ok_prefix is False


def test_extract_json_strips_think_tags_and_adapter_preserves_setup_code(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify <think> tag stripping, Ollama key non-leakage, adapter setup_code, `is None`, and empty YAML."""
    import json
    from ares.agents.llm_client import extract_json_object
    from ares.memory.cognee_engine import RealCogneeEngine

    raw = (
        '<think>Maybe {"mutated_code": "def draft(): pass"}</think>\n'
        '{"mutated_code": "def final(): return 1", "expected_failing_tests": 1}'
    )
    parsed = extract_json_object(raw)
    assert parsed["mutated_code"] == "def final(): return 1"

    # Host OPENAI_API_KEY must never leak into local Ollama clients/engines
    monkeypatch.setenv("OPENAI_API_KEY", "sk-host-secret")
    ollama_client = LocalLLMClient(provider="ollama", mode="ollama")
    assert ollama_client.api_key is None
    ollama_cognee = RealCogneeEngine(store_dir=tmp_path / "cg", llm_provider="ollama")
    assert ollama_cognee.llm_api_key == "ollama"

    # Adapter preserves setup_code, explicit None expected output, and `assert ... is None`
    ds_json = tmp_path / "with_setup.json"
    ds_json.write_text(
        json.dumps(
            [
                {
                    "id": "SETUP-1",
                    "code": "def unwrap(b):\n    return b.v if b is not None else None\n",
                    "test_cases": [
                        {
                            "test_id": "T1",
                            "call_expr": "unwrap(Box(7))",
                            "expected_output": 7,
                            "setup_code": "class Box:\n    def __init__(self, v): self.v = v\n",
                        },
                        {
                            "test_id": "T2",
                            "call_expr": "unwrap(None)",
                            "expected": None,
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded = PlugAndPlayDatasetAdapter.load_any_dataset(ds_json)
    assert loaded[0].test_cases[0].setup_code.startswith("class Box:")
    assert loaded[0].test_cases[1].expected_output is None

    ds_is = tmp_path / "is_none.jsonl"
    ds_is.write_text(
        json.dumps(
            {
                "id": "IS-1",
                "code": "def maybe_none(x):\n    return None if x == 0 else x\n",
                "tests": ["assert maybe_none(0) is None", "assert maybe_none(3) == 3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded_is = PlugAndPlayDatasetAdapter.load_any_dataset(ds_is)
    assert loaded_is[0].test_cases[0].expected_output is None

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    assert PlugAndPlayDatasetAdapter.load_any_dataset(empty_yaml) == []


def test_orchestrator_zero_turns_and_custom_initial_elo(tmp_path: Path) -> None:
    """Verify num_turns=0 runs only baseline exam without executing Turn 1, and honors initial_elo."""
    llm = LocalLLMClient(mode="adaptive_local")
    orchestrator = ARESLangGraphOrchestrator(
        attacker=AttackerAgent(llm_client=llm),
        defender=DefenderAgent(llm_client=llm),
        referee=DeterministicReferee(),
        memory=CogneeMemoryManager(store_dir=tmp_path / "zero_mem", backend="local_graph"),
        output_dir=tmp_path / "zero_out",
        initial_elo=1500.0,
    )
    summary = orchestrator.run_self_play(num_turns=0, exam_interval=5, run_baseline_exam=True)
    assert summary["total_turns"] == 0
    assert summary["attacker_final_elo"] == 1500.0
    assert summary["defender_final_elo"] == 1500.0
    assert len(summary["external_exams"]) == 1
    assert summary["external_exams"][0]["after_turn"] == 0


def test_hide_test_results_shows_tests_but_hides_execution_outcomes(tmp_path: Path) -> None:
    """Phase 1: Both agents see test definitions, NEVER test execution results (pass/fail/errors)."""
    problem = load_problems(exam=False)[0]
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    mutated_code = problem.candidate_mutations[0].mutated_code
    suite = referee.execute_test_suite(mutated_code, problem.test_cases)
    assert suite.failed_count >= 1
    failed_detail = next(td for td in suite.test_details if not td.passed)

    # 1. Attacker prompt: sees test definitions, no execution outcomes
    attacker = AttackerAgent(blind_mode=False)
    _, att_user = attacker.build_prompts(problem, memories=[])
    for tc in problem.test_cases:
        assert tc.call_expr in att_user
    assert "Expected -2, got" not in att_user

    # 2. Defender prompt with hide_test_results=True: sees ALL test definitions, ZERO execution results
    defender = DefenderAgent(blind_mode=False, hide_test_results=True)
    def_sys, def_user = defender.build_prompts(problem, mutated_code, suite, memories=[])
    assert "[NO TEST RESULTS MODE" in def_sys
    for tc in problem.test_cases:
        assert tc.call_expr in def_user
    assert "tests failing" not in def_user
    assert "Failing Deterministic Unit Tests" not in def_user
    assert (failed_detail.error_message or "Expected ") not in def_user

    # 3. Blind + hide_test_results=True also hides the failure count header
    blind_defender = DefenderAgent(blind_mode=True, hide_test_results=True)
    _, blind_user = blind_defender.build_prompts(problem, mutated_code, suite, memories=[])
    assert "tests failing" not in blind_user
    for tc in problem.test_cases:
        assert tc.call_expr not in blind_user

    # 4. Static validation in hide_test_results=True must NOT leak failure via unchanged-code rejection when claiming clean
    ok_clean, _ = defender._validate_llm_defense_reason(
        {"is_defective": False, "predicted_defect_type": "clean", "repaired_code": mutated_code},
        problem,
        mutated_code,
        suite,
    )
    assert ok_clean is True

    # Missing predicted_defect_type with is_defective=False also defaults cleanly to `clean`
    ok_clean_implicit, _ = defender._validate_llm_defense_reason(
        {"is_defective": False, "repaired_code": mutated_code},
        problem,
        mutated_code,
        suite,
    )
    assert ok_clean_implicit is True

    # Offline Defender with hide_test_results=True must preserve clean code without peeking at test results
    clean_mem = CogneeMemoryManager(store_dir=tmp_path / "p1_clean_mem", backend="local_graph")
    offline_def = DefenderAgent(
        llm_client=LocalLLMClient(mode="adaptive_local"), hide_test_results=True
    )
    clean_out = offline_def.defend(problem, problem.clean_code, None, clean_mem, turn_id=1)
    assert clean_out.predicted_defect_type == "clean"
    assert clean_out.repaired_code == problem.clean_code

    # Attacker with hide_test_results=True must not run sandbox test checks in _ensure_candidates
    no_cand_prob = problem.model_copy(update={"candidate_mutations": []})
    sym_attacker = AttackerAgent(
        llm_client=LocalLLMClient(mode="adaptive_local"), hide_test_results=True
    )
    def _forbid_exec(*args, **kwargs):
        raise AssertionError("Attacker must not execute unit tests when hide_test_results=True")
    sym_attacker._sandbox_checker.execute_test_suite = _forbid_exec  # type: ignore[method-assign]
    assert len(sym_attacker._ensure_candidates(no_cand_prob)) >= 1

    # Self-contradictory "is_defective=True with unchanged code" IS rejected without peeking at test results
    ok_contradict, reason = defender._validate_llm_defense_reason(
        {
            "is_defective": True,
            "predicted_defect_type": "negative_or_zero_edge_debt",
            "repaired_code": mutated_code,
        },
        problem,
        mutated_code,
        suite,
    )
    assert ok_contradict is False
    assert "identical" in reason

    # 5. End-to-end with Fake LLM verifies orchestrator passes ZERO test results to Defender prompt
    captured_prompts: list[str] = []

    class CaptureLLM(LocalLLMClient):
        def __init__(self, role: str) -> None:
            super().__init__(mode="ollama")
            self.role = role

        def is_live_llm_available(self) -> bool:
            return True

        def generate_json(self, system_prompt: str, user_prompt: str) -> dict | None:
            captured_prompts.append(user_prompt)
            if self.role == "attacker":
                return {
                    "mutated_code": mutated_code,
                    "injected_defect_type": "negative_or_zero_edge_debt",
                    "targeted_edge_case": "all negative",
                    "attack_strategy": "zero floor",
                    "expected_failing_tests": 1,
                    "explanation": "zero floor",
                }
            return {
                "is_defective": True,
                "predicted_defect_type": "negative_or_zero_edge_debt",
                "defect_location_line": 4,
                "diagnosis": "zero floor bug",
                "defense_strategy": "restore nums[0]",
                "repaired_code": problem.clean_code,
            }

    orch = ARESLangGraphOrchestrator(
        problems=[problem],
        exam_problems=[problem],
        attacker=AttackerAgent(llm_client=CaptureLLM("attacker"), strict_llm=True),
        defender=DefenderAgent(
            llm_client=CaptureLLM("defender"), strict_llm=True, hide_test_results=True
        ),
        referee=referee,
        memory=CogneeMemoryManager(store_dir=tmp_path / "p1_mem", backend="local_graph"),
        output_dir=tmp_path / "p1_out",
    )
    orch.run_self_play(num_turns=1, exam_interval=0, run_baseline_exam=False)
    assert len(captured_prompts) == 2
    defender_prompt_seen = captured_prompts[1]
    for tc in problem.test_cases:
        assert tc.call_expr in defender_prompt_seen
    assert "tests failing" not in defender_prompt_seen
    assert (failed_detail.error_message or "Expected ") not in defender_prompt_seen
