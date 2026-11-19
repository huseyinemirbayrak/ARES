"""Comprehensive 5-test-case per-method test suite for QuixBugs, CleanVul, and Agent-Green.

Verifies that:
1. Every single clean method across all 3 datasets (40 QuixBugs + 12 Agent-Green + 5 CleanVul
   + 5 Agent-Green CSV methods = 62 clean methods total, 310 test cases) has 5 deterministic
   test cases and passes 100% of them.
2. All 3 datasets are wired directly into `load_problems` / `PlugAndPlayDatasetAdapter`
   and `ARESLangGraphOrchestrator` for seamless local LLM execution.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from ares.agents.attacker import AttackerAgent
from ares.agents.defender import DefenderAgent
from ares.agents.llm_client import LocalLLMClient
from ares.datasets import (
    load_agent_green_suite,
    load_all_external_suites,
    load_cleanvul_suite,
    load_problems,
    load_quixbugs_suite,
)
from ares.memory.cognee_memory import CogneeMemoryManager
from ares.models import CodeProblem
from ares.orchestrator.langgraph_engine import ARESLangGraphOrchestrator
from ares.referee.deterministic_referee import DeterministicReferee


DATASETS_ROOT = Path(__file__).resolve().parent.parent / "ares" / "datasets"
AGENT_GREEN_DIR = DATASETS_ROOT / "agent-green-main"

QUIXBUGS_PROBLEMS = load_quixbugs_suite()
AGENT_GREEN_PROBLEMS = load_agent_green_suite()
CLEANVUL_PROBLEMS = load_cleanvul_suite()
REF = DeterministicReferee(timeout_seconds=3.0)


def _load_module_from_path(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. QuixBugs-master: All 40 Clean Methods x 5 Test Cases (200 Test Cases)
# ---------------------------------------------------------------------------


def test_quixbugs_suite_has_all_40_clean_methods() -> None:
    assert len(QUIXBUGS_PROBLEMS) == 40
    for prob in QUIXBUGS_PROBLEMS:
        assert len(prob.test_cases) == 5, f"{prob.function_name} must have 5 test cases"


@pytest.mark.parametrize(
    "problem",
    QUIXBUGS_PROBLEMS,
    ids=[p.function_name for p in QUIXBUGS_PROBLEMS],
)
def test_quixbugs_each_clean_method_passes_5_tests(problem: CodeProblem) -> None:
    """Verify each of the 40 clean QuixBugs methods passes all 5 deterministic test cases."""
    assert len(problem.test_cases) == 5
    res = REF.execute_test_suite(problem.clean_code, problem.test_cases)
    assert res.syntax_valid, f"{problem.function_name} syntax error: {res.syntax_or_runtime_error}"
    assert res.failed_count == 0, (
        f"{problem.function_name} failed {res.failed_count}/5 tests: "
        f"{[td.error_message for td in res.test_details if not td.passed]}"
    )


# ---------------------------------------------------------------------------
# 2. Agent-Green-main: 12 Standalone Clean Methods x 5 Test Cases + 5 CSV Methods x 5 Test Cases
# ---------------------------------------------------------------------------


def test_agent_green_suite_has_12_standalone_clean_methods() -> None:
    assert len(AGENT_GREEN_PROBLEMS) == 12
    for prob in AGENT_GREEN_PROBLEMS:
        assert len(prob.test_cases) == 5, f"{prob.function_name} must have 5 test cases"


@pytest.mark.parametrize(
    "problem",
    AGENT_GREEN_PROBLEMS,
    ids=[p.function_name for p in AGENT_GREEN_PROBLEMS],
)
def test_agent_green_each_clean_method_passes_5_tests(problem: CodeProblem) -> None:
    """Verify each of the 12 standalone Agent-Green methods passes all 5 deterministic test cases."""
    assert len(problem.test_cases) == 5
    res = REF.execute_test_suite(problem.clean_code, problem.test_cases)
    assert res.syntax_valid, f"{problem.function_name} syntax error: {res.syntax_or_runtime_error}"
    assert res.failed_count == 0, (
        f"{problem.function_name} failed {res.failed_count}/5 tests: "
        f"{[td.error_message for td in res.test_details if not td.passed]}"
    )


def test_agent_green_debt_utils_5_csv_methods_5_cases_each(tmp_path: Path) -> None:
    """Verify the 5 CSV methods in debt_utils.py with 5 test cases each (25 assertions)."""
    debt_utils_path = AGENT_GREEN_DIR / "src" / "debt_utils.py"
    mlcq_csv_path = AGENT_GREEN_DIR / "data" / "mlcq_cleaned_and_pruned_dataset_385.csv"
    debt_utils = _load_module_from_path("agent_green_debt_utils", debt_utils_path)

    c1 = tmp_path / "c1.csv"
    c1.write_text("smell;severity;code_snippet\nblob;major;def a(): pass\n", encoding="utf-8")
    c2 = tmp_path / "c2.csv"
    c2.write_text("smell;severity;code_snippet\nblob;major;\ndata_class;minor;class B: pass\n", encoding="utf-8")
    c3 = tmp_path / "c3.csv"
    c3.write_text("smell;severity;code_snippet\n", encoding="utf-8")
    c4 = tmp_path / "c4.csv"
    c4.write_text(
        "smell;severity;code_snippet\nfeature envy;critical;x = 1\nlong method;major;y = 2\nblob;minor;z = 3\n",
        encoding="utf-8",
    )

    # 1. get_code_snippets (5 test cases)
    assert debt_utils.get_code_snippets(str(c1)) == ["def a(): pass"]
    assert debt_utils.get_code_snippets(str(c2)) == ["class B: pass"]
    assert debt_utils.get_code_snippets(str(c3)) == []
    assert debt_utils.get_code_snippets(str(c4)) == ["x = 1", "y = 2", "z = 3"]
    assert len(debt_utils.get_code_snippets(str(mlcq_csv_path))) > 100

    # 2. get_td_ground_truth (5 test cases)
    assert len(debt_utils.get_td_ground_truth(str(c1))) == 1
    assert debt_utils.get_td_ground_truth(str(c2))[0]["smell"] == "data_class"
    assert debt_utils.get_td_ground_truth(str(c3)) == []
    assert len(debt_utils.get_td_ground_truth(str(c4))) == 3
    assert len(debt_utils.get_td_ground_truth(str(mlcq_csv_path))) > 100

    # 3. get_td_all_data (5 test cases)
    assert len(debt_utils.get_td_all_data(str(c1))) == 1
    assert len(debt_utils.get_td_all_data(str(c2))) == 2
    assert debt_utils.get_td_all_data(str(c3)) == []
    assert len(debt_utils.get_td_all_data(str(c4))) == 3
    assert len(debt_utils.get_td_all_data(str(mlcq_csv_path))) == 385

    # 4. filter_by_smell (5 test cases)
    assert len(debt_utils.filter_by_smell(str(c1), "blob")) == 1
    assert len(debt_utils.filter_by_smell(str(c1), "data_class")) == 0
    assert len(debt_utils.filter_by_smell(str(c4), "FEATURE ENVY")) == 1
    assert len(debt_utils.filter_by_smell(str(c4), "long method")) == 1
    assert len(debt_utils.filter_by_smell(str(mlcq_csv_path), "blob")) > 0

    # 5. get_unique_smells (5 test cases)
    assert debt_utils.get_unique_smells(str(c1)) == {"blob"}
    assert debt_utils.get_unique_smells(str(c2)) == {"blob", "data_class"}
    assert debt_utils.get_unique_smells(str(c3)) == set()
    assert debt_utils.get_unique_smells(str(c4)) == {"feature envy", "long method", "blob"}
    assert "blob" in debt_utils.get_unique_smells(str(mlcq_csv_path))


# ---------------------------------------------------------------------------
# 3. CleanVul-main: 5 Clean Standalone Python Methods x 5 Test Cases (25 Test Cases)
# ---------------------------------------------------------------------------


def test_cleanvul_suite_has_5_clean_methods() -> None:
    assert len(CLEANVUL_PROBLEMS) == 5
    for prob in CLEANVUL_PROBLEMS:
        assert len(prob.test_cases) == 5, f"{prob.function_name} must have 5 test cases"


@pytest.mark.parametrize(
    "problem",
    CLEANVUL_PROBLEMS,
    ids=[p.function_name for p in CLEANVUL_PROBLEMS],
)
def test_cleanvul_each_clean_method_passes_5_tests(problem: CodeProblem) -> None:
    """Verify each of the 5 clean CleanVul Python methods passes all 5 deterministic test cases."""
    assert len(problem.test_cases) == 5
    res = REF.execute_test_suite(problem.clean_code, problem.test_cases)
    assert res.syntax_valid, f"{problem.function_name} syntax error: {res.syntax_or_runtime_error}"
    assert res.failed_count == 0, (
        f"{problem.function_name} failed {res.failed_count}/5 tests: "
        f"{[td.error_message for td in res.test_details if not td.passed]}"
    )


# ---------------------------------------------------------------------------
# 4. System Integration: `load_problems` & `ARESLangGraphOrchestrator` Connection
# ---------------------------------------------------------------------------


def test_new_datasets_connected_to_load_problems_and_orchestrator(tmp_path: Path) -> None:
    """Verify `quixbugs`, `agent_green`, `cleanvul`, and `all_new` plug directly into ARES."""
    qb = load_problems("quixbugs")
    ag = load_problems("agent_green")
    cv = load_problems("cleanvul")
    all_new = load_problems("all_new")

    assert len(qb) == 40
    assert len(ag) == 12
    assert len(cv) == 5
    assert len(all_new) == 57
    assert all(len(p.test_cases) == 5 for p in all_new)

    llm = LocalLLMClient(mode="adaptive_local")
    orchestrator = ARESLangGraphOrchestrator(
        problems=[qb[0], ag[0], cv[0]],
        exam_problems=[qb[1], ag[1], cv[1]],
        attacker=AttackerAgent(llm_client=llm, blind_mode=True),
        defender=DefenderAgent(llm_client=llm, blind_mode=True),
        referee=DeterministicReferee(min_broken_tests=1, max_broken_tests=2),
        memory=CogneeMemoryManager(store_dir=tmp_path / "mem", backend="local_graph"),
        output_dir=tmp_path / "out",
    )
    scoreboard = orchestrator.run_self_play(num_turns=3, exam_interval=3, run_baseline_exam=False)
    assert scoreboard["total_turns"] == 3
