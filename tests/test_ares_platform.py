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
    mem_a = CogneeMemoryManager(store_dir=tmp_path / "mem_a")
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
    mem_b = CogneeMemoryManager(store_dir=tmp_path / "mem_b")
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
        memory=CogneeMemoryManager(store_dir=tmp_path / "plug_mem"),
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
    """Task (iii) & (iv): Validate 25-turn LangGraph self-play and measurable improvement on External Exams."""
    llm = LocalLLMClient(mode="adaptive_local")
    attacker = AttackerAgent(llm_client=llm)
    defender = DefenderAgent(llm_client=llm)
    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    memory = CogneeMemoryManager(store_dir=tmp_path / "cognee_store")

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
    assert scoreboard["attacker_wins"] >= 1
    assert scoreboard["defender_wins"] > scoreboard["attacker_wins"]

    imp = scoreboard["external_exam_improvement"]
    assert imp["baseline_turn"] == 0
    assert imp["final_turn"] == 25
    assert imp["attacker_valid_subtle_rate_delta"] > 0.0
    assert imp["defender_full_repair_rate_delta"] > 0.0
    assert imp["defender_classification_accuracy_delta"] > 0.0
    assert scoreboard["external_exams"][-1]["defender_full_repair_rate"] == 100.0
    assert scoreboard["external_exams"][-1]["attacker_valid_subtle_rate"] == 100.0


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
