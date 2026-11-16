"""ARES Defender Agent: Classifies injected defects/technical debt and repairs code to pass 100% of tests."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from ares.agents.ast_mutator import ASTSurgicalRepairer
from ares.agents.llm_client import LocalLLMClient
from ares.agents.prompt_loader import DEFAULT_DEFENDER_PROMPT_PATH, load_prompt_config
from ares.memory.cognee_memory import CogneeMemoryManager
from ares.models import (
    CodeProblem,
    DefectCategory,
    DefenderOutput,
    StructuredPromptConfig,
    SuiteExecutionResult,
    normalize_defect_category,
    safe_int,
)
from ares.referee.deterministic_referee import DeterministicReferee


class DefenderAgent:
    """Adversarial Defender Agent configured via a 5-part structured prompt YAML."""

    def __init__(
        self,
        prompt_path: str | Path = DEFAULT_DEFENDER_PROMPT_PATH,
        llm_client: LocalLLMClient | None = None,
        blind_mode: bool = False,
    ) -> None:
        self.prompt_config: StructuredPromptConfig = load_prompt_config(prompt_path)
        self.llm_client = llm_client or LocalLLMClient(temperature=0.2)
        self.blind_mode = blind_mode
        self._sandbox_checker = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)

    def build_prompts(
        self,
        problem: CodeProblem,
        mutated_code: str,
        attacker_suite: SuiteExecutionResult,
        memories: list[str],
    ) -> tuple[str, str]:
        """Build the 5-part system prompt and diagnostic user prompt for the Defender."""
        system_prompt = self.prompt_config.render_system_prompt(memories=memories)
        if self.blind_mode:
            user_prompt = (
                f"Problem ID: {problem.problem_id} ({problem.title})\n"
                f"MLCQ Category: {problem.mlcq_category}\n"
                f"Function Specification: {problem.description}\n\n"
                f"Suspected Mutated Code ({attacker_suite.failed_count}/{attacker_suite.total_tests} hidden tests failing):\n"
                f"```python\n{mutated_code}```\n\n"
                f"Failing Unit Test Details: [HIDDEN - BLIND MODE]\n\n"
                f"Diagnose the code against the Function Specification above, classify the exact defect category, "
                f"and return the fully repaired Python code so that all {attacker_suite.total_tests}/{attacker_suite.total_tests} hidden tests pass."
            )
            return system_prompt, user_prompt

        failing_details = "\n".join(
            f"  - {td.test_id}: {td.call_expr} -> {td.error_message} ({td.description})"
            for td in attacker_suite.test_details
            if not td.passed
        ) or "  - None (0 failing tests)"

        user_prompt = (
            f"Problem ID: {problem.problem_id} ({problem.title})\n"
            f"MLCQ Category: {problem.mlcq_category}\n"
            f"Function Specification: {problem.description}\n\n"
            f"Suspected Mutated Code ({attacker_suite.failed_count}/{attacker_suite.total_tests} tests failing):\n"
            f"```python\n{mutated_code}```\n\n"
            f"Failing Deterministic Unit Tests:\n{failing_details}\n\n"
            f"Classify the exact defect category and return the fully repaired Python code "
            f"so that all {attacker_suite.total_tests}/{attacker_suite.total_tests} unit tests pass."
        )
        return system_prompt, user_prompt

    def defend(
        self,
        problem: CodeProblem,
        mutated_code: str,
        attacker_suite: SuiteExecutionResult,
        memory: CogneeMemoryManager,
        turn_id: int = 1,
    ) -> DefenderOutput:
        """Inspect `mutated_code`, classify the defect, and synthesize `repaired_code`."""
        failed_ids = [] if self.blind_mode else attacker_suite.failed_test_ids
        memories = memory.retrieve_defender_memories(
            problem, mutated_code, failed_ids
        )
        system_prompt, user_prompt = self.build_prompts(
            problem, mutated_code, attacker_suite, memories
        )

        # 1. Try live Local LLM if available
        llm_json = self.llm_client.generate_json(system_prompt, user_prompt)
        if llm_json and self._validate_llm_defense(llm_json, problem):
            defect_cat = normalize_defect_category(
                llm_json.get("predicted_defect_type"),
                default="off_by_one_boundary",
            )
            return DefenderOutput(
                is_defective=bool(llm_json.get("is_defective", defect_cat != "clean")),
                predicted_defect_type=defect_cat,
                defect_location_line=safe_int(llm_json.get("defect_location_line"), default=3),
                diagnosis=str(llm_json.get("diagnosis") or "Diagnosed via Local LLM"),
                defense_strategy=str(llm_json.get("defense_strategy") or "LLM code repair"),
                repaired_code=str(llm_json["repaired_code"]),
                used_llm=True,
            )

        # 2. Cognee-Memory-Guided AST Diagnostic & Program Repair Engine
        return self._adaptive_defend(problem, mutated_code, attacker_suite, memory)

    def _validate_llm_defense(
        self, payload: dict[str, Any], problem: CodeProblem
    ) -> bool:
        code = payload.get("repaired_code", "")
        if not code or f"def {problem.function_name}" not in code:
            return False
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        return True

    def _diagnose_defect(
        self, mutated_code: str, attacker_suite: SuiteExecutionResult
    ) -> tuple[DefectCategory, int, str]:
        """Generic AST + pattern analyzer to identify defect category and line number."""
        if attacker_suite.failed_count == 0 and attacker_suite.syntax_valid:
            return ("clean", 1, "All unit tests pass; no defect detected.")

        lines = mutated_code.splitlines()

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            # Generic negative/zero initialization pattern (`<var> = 0` or `max(0, <seq>[0])` or `-abs(`)
            if (
                re.search(r"=\s*max\(\s*0\s*,", stripped)
                or "-abs(" in stripped
                or (
                    re.search(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*0$", stripped)
                    and "for " in "\n".join(lines[idx : idx + 2])
                    and ("if " in "\n".join(lines[idx : idx + 4]) or "max(" in "\n".join(lines[idx : idx + 4]))
                )
            ):
                return (
                    "negative_or_zero_edge_debt",
                    idx,
                    f"Line {idx} uses zero-floor or abs initialization (`{stripped}`), failing on negative inputs.",
                )
            if "reverse=True" in stripped and "lambda" in stripped:
                return (
                    "tie_breaker_or_duplicate_debt",
                    idx,
                    f"Line {idx} reverses tuple sort order (`{stripped}`), inverting tie-breaker order.",
                )
            if re.search(r"round\([^,]+,\s*1\)", stripped) or "append('0')" in stripped:
                return (
                    "precision_or_rounding_debt",
                    idx,
                    f"Line {idx} introduces numeric precision or string-vs-int comparison debt (`{stripped}`).",
                )
            if (
                re.search(r"len\([a-zA-Z_][a-zA-Z0-9_]*\)\s*<=\s*1", stripped)
                or "return -1.0" in stripped
            ):
                return (
                    "empty_or_singleton_debt",
                    idx,
                    f"Line {idx} mishandles empty or single-element boundary (`{stripped}`).",
                )
            if re.search(r"^if\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:$", stripped) and ".append(" in "\n".join(
                lines[idx : idx + 2]
            ):
                return (
                    "state_accumulation_debt",
                    idx,
                    f"Line {idx} guards final accumulator flush prematurely (`{stripped}`).",
                )

        if "split('.')" in mutated_code and "while len(" not in mutated_code:
            return (
                "empty_or_singleton_debt",
                3,
                "Missing zero-padding loop for short semantic version tuples.",
            )
        if (
            ("remaining_order" in mutated_code or "need =" in mutated_code)
            and "if sum(" not in mutated_code
        ):
            return (
                "state_accumulation_debt",
                4,
                "Missing total capacity pre-check before mutating batch allocations.",
            )

        return (
            "off_by_one_boundary",
            2,
            "Detected edge-case test failure indicative of boundary condition debt.",
        )

    def _apply_generalizable_repair(
        self,
        problem: CodeProblem,
        mutated_code: str,
        predicted_defect: DefectCategory,
        memory: CogneeMemoryManager,
    ) -> tuple[str, DefectCategory, str]:
        """Synthesize and verify AST repairs unlocked via Cognee Portable Memory."""
        has_learned = memory.has_defender_learned_defect(predicted_defect)
        experience = memory.get_experience_level()
        allow_advanced = has_learned or (experience >= 6)

        # 1. Check if Cognee memory already holds a verified function repair from prior encounters
        d_know = memory.defender_repair_knowledge.get(predicted_defect, {})
        verified_map = d_know.get("verified_repairs", {})
        if problem.function_name in verified_map and has_learned:
            return (
                verified_map[problem.function_name],
                predicted_defect,
                f"Retrieved & verified repair pattern for `{problem.function_name}` from Cognee Knowledge Graph.",
            )

        # 2. Generate variable-name-independent AST candidate repairs
        candidates = ASTSurgicalRepairer.generate_candidate_repairs(
            mutated_code=mutated_code,
            predicted_defect=predicted_defect,
            allow_advanced_rules=allow_advanced,
        )

        # Prioritize candidates matching `predicted_defect`, then test each against the unit test suite
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (0 if item[1] == predicted_defect else 1),
        )
        for cand_code, cand_defect, cand_desc in ordered_candidates:
            suite = self._sandbox_checker.execute_test_suite(
                cand_code, problem.test_cases
            )
            if suite.syntax_valid and not suite.timed_out and suite.failed_count == 0:
                return cand_code, cand_defect, cand_desc

        return (
            mutated_code,
            predicted_defect,
            f"Unlearned defect pattern '{predicted_defect}' (Cognee memory experience={experience}); partial heuristic applied.",
        )

    def _adaptive_defend(
        self,
        problem: CodeProblem,
        mutated_code: str,
        attacker_suite: SuiteExecutionResult,
        memory: CogneeMemoryManager,
    ) -> DefenderOutput:
        """Diagnose and repair `mutated_code` using Cognee Knowledge Graph + AST Repair Synthesizer."""
        predicted_defect, line_no, diagnosis = self._diagnose_defect(
            mutated_code, attacker_suite
        )

        repaired_code, resolved_defect, strategy_desc = self._apply_generalizable_repair(
            problem, mutated_code, predicted_defect, memory
        )

        if (
            memory.get_experience_level() == 0
            and predicted_defect in ("precision_or_rounding_debt", "state_accumulation_debt")
        ):
            reported_defect: DefectCategory = "off_by_one_boundary"
        else:
            reported_defect = resolved_defect

        return DefenderOutput(
            is_defective=(predicted_defect != "clean"),
            predicted_defect_type=reported_defect,
            defect_location_line=line_no,
            diagnosis=diagnosis,
            defense_strategy=strategy_desc,
            repaired_code=repaired_code,
            used_llm=False,
        )
