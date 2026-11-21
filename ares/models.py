"""Core Pydantic data models and state definitions for the ARES framework."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict, cast
from pydantic import BaseModel, Field


DefectCategory = Literal[
    "off_by_one_boundary",
    "empty_or_singleton_debt",
    "negative_or_zero_edge_debt",
    "precision_or_rounding_debt",
    "tie_breaker_or_duplicate_debt",
    "state_accumulation_debt",
    "clean",
]

VALID_DEFECT_CATEGORIES: set[str] = {
    "off_by_one_boundary",
    "empty_or_singleton_debt",
    "negative_or_zero_edge_debt",
    "precision_or_rounding_debt",
    "tie_breaker_or_duplicate_debt",
    "state_accumulation_debt",
    "clean",
}


def normalize_defect_category(
    raw: Any, default: DefectCategory = "off_by_one_boundary"
) -> DefectCategory:
    """Normalize an LLM-provided defect category string into a valid DefectCategory literal."""
    if not isinstance(raw, str):
        return default
    cleaned = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned in VALID_DEFECT_CATEGORIES:
        return cast(DefectCategory, cleaned)
    if "empty" in cleaned or "singleton" in cleaned:
        return "empty_or_singleton_debt"
    if "neg" in cleaned or "zero" in cleaned:
        return "negative_or_zero_edge_debt"
    if "precis" in cleaned or "round" in cleaned or "float" in cleaned:
        return "precision_or_rounding_debt"
    if "tie" in cleaned or "dup" in cleaned or "sort" in cleaned:
        return "tie_breaker_or_duplicate_debt"
    if "state" in cleaned or "accum" in cleaned:
        return "state_accumulation_debt"
    if "clean" in cleaned or "none" in cleaned:
        return "clean"
    if "off" in cleaned or "bound" in cleaned or "index" in cleaned:
        return "off_by_one_boundary"
    return default


def safe_int(val: Any, default: int = 1) -> int:
    """Safely parse an integer from LLM JSON output (e.g., 'line 4' -> 4)."""
    if isinstance(val, bool):
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        match = re.search(r"-?\d+", val)
        if match:
            return int(match.group(0))
    return default


def safe_bool(val: Any, default: bool = True) -> bool:
    """Safely parse a boolean from LLM JSON output (handling string 'false'/'no'/'0')."""
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        lowered = val.strip().lower()
        if lowered in ("true", "yes", "1", "y", "on"):
            return True
        if lowered in ("false", "no", "0", "n", "off", "none", "clean", ""):
            return False
    return default


class UnitTestCase(BaseModel):
    """A single deterministic test case for a code analysis problem."""

    test_id: str
    description: str
    call_expr: str = Field(
        description="Python expression calling the target function, e.g. 'max_subarray_sum([-2, 1, -3, 4])'"
    )
    expected_output: Any
    is_edge_case: bool = False
    setup_code: str = Field(
        default="",
        description=(
            "Optional trusted test-harness code (e.g. helper classes) executed after the artifact "
            "and before `call_expr`. It is NOT part of `clean_code`, is never shown to or mutated "
            "by the agents."
        ),
    )


class CandidateMutation(BaseModel):
    """Pre-analyzed or template mutation pattern associated with a problem."""

    mutation_id: str
    defect_type: DefectCategory
    strategy_name: str
    targeted_edge_case: str
    mutated_code: str
    explanation: str
    expected_broken_count: int = 1


class CodeProblem(BaseModel):
    """A software analysis & technical debt scenario with 4-5 deterministic test cases."""

    problem_id: str
    title: str
    mlcq_category: str = Field(
        description="MLCQ technical debt / code smell category (e.g., Boundary Condition Debt, Complex Conditional)"
    )
    severity: str = "major"
    function_name: str
    description: str
    clean_code: str
    test_cases: list[UnitTestCase]
    candidate_mutations: list[CandidateMutation] = Field(default_factory=list)


class StructuredPromptConfig(BaseModel):
    """Configurable 5-part structured prompt (Task, Recipe, Output, Validation, Examples)."""

    agent_role: Literal["attacker", "defender"]
    version: str = "1.0.0"
    description: str = ""
    task: str
    recipe: str
    output: str
    validation: str
    examples: list[dict[str, Any]] = Field(default_factory=list)

    def render_system_prompt(
        self,
        memories: list[str] | None = None,
        blind_mode: bool = False,
        hide_test_results: bool = False,
    ) -> str:
        """Render the full 5-part structured prompt with retrieved memories and visibility rules."""
        examples_str = ""
        for idx, ex in enumerate(self.examples, start=1):
            examples_str += f"\n--- Example {idx}: {ex.get('title', 'Sample')} ---\n"
            for k, v in ex.items():
                if k != "title":
                    examples_str += f"{k.upper()}:\n{v}\n"

        memory_block = (
            "\n".join(f"- {m}" for m in memories)
            if memories
            else "- No prior episodic memories recorded yet."
        )

        visibility_directive = ""
        if blind_mode:
            if self.agent_role == "attacker":
                visibility_directive = (
                    "\n=== [BLIND MODE OVERRIDE] ===\n"
                    "Unit test cases are intentionally HIDDEN. Do NOT expect test inputs in the user prompt. "
                    "Analyze the function specification, control flow, edge conditions, and your past CODE DIFF "
                    "memories below to craft a minimal 1-2 line mutation that breaks only subtle edge cases.\n"
                )
            elif hide_test_results:
                visibility_directive = (
                    "\n=== [BLIND + NO TEST RESULTS OVERRIDE] ===\n"
                    "Both unit test cases AND test execution results (pass/fail counts, assertion errors) "
                    "are intentionally HIDDEN. Perform careful static code review against the Function "
                    "Specification, check boundary/edge conditions, and leverage your past VERIFIED REPAIR "
                    "DIFF memories below. Classify as `clean` if no defect exists.\n"
                )
            else:
                visibility_directive = (
                    "\n=== [BLIND MODE OVERRIDE] ===\n"
                    "Failing unit test inputs and assertion errors are intentionally HIDDEN. "
                    "Perform careful static code review against the Function Specification, check boundary/edge "
                    "conditions, and leverage your past VERIFIED REPAIR DIFF memories below.\n"
                )
        elif hide_test_results and self.agent_role == "defender":
            visibility_directive = (
                "\n=== [NO TEST RESULTS MODE (TESTS-ONLY SPECIFICATION)] ===\n"
                "You are given the full deterministic unit test definitions (inputs and expected outputs) "
                "as the specification contract, but test execution results (pass/fail counts, failing test "
                "IDs, actual outputs, and assertion errors) are intentionally HIDDEN.\n"
                "Statically trace the code against every unit test case and the Function Specification to "
                "determine whether a defect exists. If a defect exists, classify it accurately and repair "
                "the code so ALL unit tests pass. If the code already satisfies every unit test, classify "
                "it as `clean` and preserve it unchanged.\n"
            )

        return (
            f"=== [1. TASK] ===\n{self.task.strip()}\n"
            f"{visibility_directive}\n"
            f"=== [2. RECIPE] ===\n{self.recipe.strip()}\n\n"
            f"=== [3. OUTPUT SCHEMA] ===\n{self.output.strip()}\n\n"
            f"=== [4. VALIDATION CHECKLIST] ===\n{self.validation.strip()}\n\n"
            f"=== [5. EXAMPLES] ===\n{examples_str.strip()}\n\n"
            f"=== [RETRIEVED EPISODIC & DIFF MEMORY — APPLY THESE LESSONS] ===\n{memory_block}\n"
        )


class AttackerOutput(BaseModel):
    """Structured artifact emitted by the Attacker Agent."""

    mutated_code: str
    injected_defect_type: DefectCategory
    targeted_edge_case: str
    attack_strategy: str
    expected_failing_tests: int = 1
    explanation: str
    used_llm: bool = False


class DefenderOutput(BaseModel):
    """Structured artifact emitted by the Defender Agent."""

    is_defective: bool
    predicted_defect_type: DefectCategory
    defect_location_line: int = 1
    diagnosis: str
    defense_strategy: str
    repaired_code: str
    used_llm: bool = False


class SingleTestExecution(BaseModel):
    """Result of executing one deterministic unit test case."""

    test_id: str
    description: str
    call_expr: str
    expected_output: Any
    actual_output: Any = None
    passed: bool
    is_edge_case: bool = False
    error_message: str | None = None


class SuiteExecutionResult(BaseModel):
    """Aggregate result of running all 4-5 unit tests on a code snippet."""

    syntax_valid: bool
    timed_out: bool = False
    total_tests: int
    passed_count: int
    failed_count: int
    failed_test_ids: list[str] = Field(default_factory=list)
    passed_test_ids: list[str] = Field(default_factory=list)
    test_details: list[SingleTestExecution] = Field(default_factory=list)
    syntax_or_runtime_error: str | None = None


class RefereeVerdict(BaseModel):
    """Deterministic evaluation result produced by the ARES Referee."""

    turn_id: int
    problem_id: str
    attack_validity: Literal[
        "valid_subtle_attack",
        "invalid_no_op_attack",
        "invalid_over_broken_attack",
        "invalid_syntax_or_timeout",
    ]
    attacker_suite_result: SuiteExecutionResult
    defender_suite_result: SuiteExecutionResult
    ground_truth_defect: DefectCategory
    ground_truth_source: Literal["curated", "ast_diff", "attacker_claim"] = "attacker_claim"
    defender_predicted_defect: DefectCategory
    classification_correct: bool
    defense_repaired_all_tests: bool
    winner: Literal["attacker", "defender", "draw"]
    attacker_reward: float
    defender_reward: float
    attacker_feedback: str
    defender_feedback: str


class TurnRecord(BaseModel):
    """Complete record of one adversarial self-play turn."""

    turn_id: int
    problem_id: str
    problem_title: str
    mlcq_category: str
    attacker_output: AttackerOutput
    defender_output: DefenderOutput
    verdict: RefereeVerdict
    attacker_elo_after: float
    defender_elo_after: float
    cognee_memory_ids: list[str] = Field(default_factory=list)


class ExamResult(BaseModel):
    """Evaluation snapshot from a periodic External Benchmark Exam ('Dışarıdan Sınav')."""

    exam_id: str
    after_turn: int
    num_exam_problems: int
    attacker_valid_subtle_rate: float
    attacker_avg_broken_tests: float
    defender_full_repair_rate: float
    defender_classification_accuracy: float
    defender_combined_score: float
    defender_bug_detection_rate: float = 0.0
    defender_avg_test_pass_rate: float = 0.0
    attacker_elo: float
    defender_elo: float
    cognee_knowledge_nodes: int
    per_problem_details: list[dict[str, Any]] = Field(default_factory=list)


class TurnReflexion(BaseModel):
    """Structured code-diff and AST-feature reflexion produced after each turn."""

    turn_id: int
    problem_id: str
    function_name: str
    ast_features: list[str] = Field(default_factory=list)
    attack_diff: str = ""
    repair_diff: str = ""
    attacker_outcome: Literal["win", "caught", "invalid"] = "invalid"
    defender_outcome: Literal["repaired", "failed"] = "failed"
    attacker_lesson: str
    defender_lesson: str


class ARESGraphState(TypedDict, total=False):
    """Typed state passed across nodes in the LangGraph StateGraph orchestrator."""

    turn_id: int
    max_turns: int
    exam_interval: int
    current_problem: dict[str, Any]
    attacker_memories: list[str]
    defender_memories: list[str]
    attacker_output: dict[str, Any]
    attacker_suite_result: dict[str, Any]
    defender_output: dict[str, Any]
    verdict: dict[str, Any]
    reflexion: dict[str, Any]
    attacker_elo: float
    defender_elo: float
    attacker_wins: int
    defender_wins: int
    draws: int
    turn_history: list[dict[str, Any]]
    exam_history: list[dict[str, Any]]

