"""ARES Attacker Agent: Transforms clean code snippets to inject subtle defects breaking 1-2 tests."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from ares.agents.ast_mutator import ASTSurgicalMutator
from ares.agents.llm_client import LocalLLMClient
from ares.agents.prompt_loader import DEFAULT_ATTACKER_PROMPT_PATH, load_prompt_config
from ares.memory.cognee_memory import CogneeMemoryManager
from ares.memory.reflexion import extract_ast_features
from ares.models import (
    AttackerOutput,
    CandidateMutation,
    CodeProblem,
    StructuredPromptConfig,
    normalize_defect_category,
    safe_int,
)
from ares.referee.deterministic_referee import DeterministicReferee

logger = logging.getLogger(__name__)


class AttackerAgent:
    """Adversarial Attacker Agent configured via a 5-part structured prompt YAML."""

    def __init__(
        self,
        prompt_path: str | Path = DEFAULT_ATTACKER_PROMPT_PATH,
        llm_client: LocalLLMClient | None = None,
        blind_mode: bool = False,
        strict_llm: bool = False,
        max_static_retries: int = 1,
        hide_test_results: bool = False,
    ) -> None:
        self.prompt_config: StructuredPromptConfig = load_prompt_config(prompt_path)
        self.llm_client = llm_client or LocalLLMClient(temperature=0.7)
        self.blind_mode = blind_mode
        self.strict_llm = strict_llm
        self.max_static_retries = max_static_retries
        self.hide_test_results = hide_test_results
        self._sandbox_checker = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
        self._turn_counter = 0

    def build_prompts(
        self, problem: CodeProblem, memories: list[str]
    ) -> tuple[str, str]:
        """Build the 5-part system prompt and problem-specific user prompt."""
        system_prompt = self.prompt_config.render_system_prompt(
            memories=memories, blind_mode=self.blind_mode
        )
        ast_features = ", ".join(extract_ast_features(problem.clean_code))
        if self.blind_mode:
            test_block = (
                f"Unit Test Suite: [HIDDEN - BLIND MODE] ({len(problem.test_cases)} hidden deterministic tests).\n"
                f"Analyze the function specification, AST patterns, and retrieved past CODE DIFF lessons without seeing the tests. "
                f"Make a minimal 1-2 line edit so ONLY 1 or 2 hidden edge-case tests fail while standard cases still pass. "
                f"Output strict JSON matching the schema."
            )
        else:
            test_summary = "\n".join(
                f"  - {tc.test_id} ({'EDGE' if tc.is_edge_case else 'NOMINAL'}): "
                f"{tc.call_expr} == {tc.expected_output!r}  # {tc.description}"
                for tc in problem.test_cases
            )
            test_block = (
                f"Unit Test Suite ({len(problem.test_cases)} deterministic tests):\n{test_summary}\n\n"
                f"Apply lessons from your retrieved CODE DIFF memories. Inject a subtle 1-2 line defect so ONLY 1 or 2 "
                f"test cases fail while the remaining test cases still pass. Output strict JSON matching the schema."
            )
        user_prompt = (
            f"Problem ID: {problem.problem_id} ({problem.title})\n"
            f"MLCQ Category: {problem.mlcq_category}\n"
            f"Function Name: {problem.function_name}\n"
            f"AST Structural Patterns: [{ast_features}]\n"
            f"Specification: {problem.description}\n\n"
            f"Clean Code (passes all {len(problem.test_cases)} tests):\n"
            f"```python\n{problem.clean_code}```\n\n"
            f"{test_block}"
        )
        return system_prompt, user_prompt

    def attack(
        self,
        problem: CodeProblem,
        memory: CogneeMemoryManager,
        turn_id: int = 1,
        memories: list[str] | None = None,
    ) -> AttackerOutput:
        """Generate a subtle defect mutation for `problem` using retrieved memories + LLM/AST strategy."""
        self._turn_counter = turn_id
        resolved_memories = (
            memories if memories is not None else memory.retrieve_attacker_memories(problem)
        )
        system_prompt, user_prompt = self.build_prompts(problem, resolved_memories)

        # 1. Try live Local LLM with static self-correction retry (never runs unit tests!)
        if self.llm_client.is_live_llm_available():
            current_user_prompt = user_prompt
            last_reason = "No JSON response returned"
            for attempt in range(self.max_static_retries + 1):
                llm_json = self.llm_client.generate_json(system_prompt, current_user_prompt)
                if not llm_json:
                    last_reason = self.llm_client.last_error or "Unparseable or empty JSON"
                else:
                    valid, reason = self._validate_llm_attack_reason(llm_json, problem)
                    if valid:
                        defect_cat = normalize_defect_category(
                            llm_json.get("injected_defect_type"),
                            default="off_by_one_boundary",
                        )
                        if defect_cat == "clean":
                            defect_cat = "off_by_one_boundary"
                        return AttackerOutput(
                            mutated_code=str(llm_json["mutated_code"]),
                            injected_defect_type=defect_cat,
                            targeted_edge_case=str(
                                llm_json.get("targeted_edge_case") or "Edge case"
                            ),
                            attack_strategy=str(
                                llm_json.get("attack_strategy") or "LLM surgical mutation"
                            ),
                            expected_failing_tests=safe_int(
                                llm_json.get("expected_failing_tests"), default=1
                            ),
                            explanation=str(
                                llm_json.get("explanation") or "Generated by Local LLM"
                            ),
                            used_llm=True,
                        )
                    last_reason = reason

                if attempt < self.max_static_retries:
                    current_user_prompt = (
                        f"{user_prompt}\n\n"
                        f"[STATIC VALIDATION CORRECTION REQUIRED]\n"
                        f"Your previous output was rejected because: {last_reason}.\n"
                        f"Return ONLY valid JSON with syntactically valid Python in `mutated_code`, "
                        f"keeping `def {problem.function_name}` and modifying 1-2 lines so it differs from clean code."
                    )

            logger.warning(
                "AttackerAgent LLM failed static validation after %d attempt(s) on %s: %s",
                self.max_static_retries + 1,
                problem.problem_id,
                last_reason,
            )
            if self.strict_llm:
                return AttackerOutput(
                    mutated_code=problem.clean_code,
                    injected_defect_type="off_by_one_boundary",
                    targeted_edge_case="LLM static validation failure",
                    attack_strategy="LLM strict failure (no heuristic fallback)",
                    expected_failing_tests=0,
                    explanation=f"Strict LLM mode active; LLM failed validation: {last_reason}",
                    used_llm=True,
                )
        elif self.strict_llm:
            detail = (
                "LLM mode is set to 'adaptive_local' (live LLM calls disabled)."
                if self.llm_client.mode == "adaptive_local"
                else f"Attacker LLM endpoint ({self.llm_client.provider} @ {self.llm_client.base_url}) is not reachable."
            )
            raise RuntimeError(
                f"Strict LLM mode (--strict-llm) is enabled, but {detail} "
                f"Start/configure your LLM server or pass `--no-strict-llm` for an explicit no-LLM heuristic baseline."
            )

        # 2. Honest AST & Memory-Driven Mutation Selection (offline / adaptive_local mode)
        return self._adaptive_attack(problem, memory, turn_id)

    def _validate_llm_attack_reason(
        self, payload: dict[str, Any], problem: CodeProblem
    ) -> tuple[bool, str]:
        code = payload.get("mutated_code", "")
        if not isinstance(code, str) or not code.strip():
            return False, "Missing or empty `mutated_code`"
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"SyntaxError in mutated_code: {exc}"
        has_fn = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == problem.function_name
            for n in ast.walk(tree)
        )
        if not has_fn:
            return False, f"Function definition `def {problem.function_name}` missing in AST"
        if code.strip() == problem.clean_code.strip():
            return False, "mutated_code is identical to clean_code (no-op)"
        return True, "ok"

    def _validate_llm_attack(
        self, payload: dict[str, Any], problem: CodeProblem
    ) -> bool:
        ok, _ = self._validate_llm_attack_reason(payload, problem)
        return ok

    def _ensure_candidates(self, problem: CodeProblem) -> list[CandidateMutation]:
        """Return existing candidate mutations or dynamically synthesize them via ASTSurgicalMutator."""
        if problem.candidate_mutations:
            return list(problem.candidate_mutations)

        raw_cands = ASTSurgicalMutator.generate_candidates(
            problem.clean_code, problem.function_name
        )
        if self.blind_mode or self.hide_test_results:
            return raw_cands

        verified: list[CandidateMutation] = []
        for cand in raw_cands:
            suite = self._sandbox_checker.execute_test_suite(
                cand.mutated_code, problem.test_cases
            )
            if suite.syntax_valid and not suite.timed_out and suite.failed_count > 0:
                cand.expected_broken_count = suite.failed_count
                verified.append(cand)
        return verified

    def _adaptive_attack(
        self,
        problem: CodeProblem,
        memory: CogneeMemoryManager,
        turn_id: int,
    ) -> AttackerOutput:
        """Select or synthesize a code mutation informed by recorded strategy statistics and AST analysis."""
        candidates = self._ensure_candidates(problem)

        if not candidates:
            mutated = problem.clean_code.replace(">=", ">", 1)
            return AttackerOutput(
                mutated_code=mutated,
                injected_defect_type="off_by_one_boundary",
                targeted_edge_case="Boundary comparison edge case",
                attack_strategy="AST comparator boundary shift",
                expected_failing_tests=1,
                explanation="Shifted inclusive `>=` boundary to strict `>`.",
                used_llm=False,
            )

        scored: list[tuple[float, int, CandidateMutation]] = []
        for idx, cand in enumerate(candidates):
            strat_key = f"{problem.problem_id}::{cand.strategy_name}"
            stats = memory.attacker_strategy_stats.get(strat_key)

            if stats is None:
                # Prefer unexplored 1-2 test mutations without any artificial cold-start sabotage
                score = 2.0 + (0.2 if cand.expected_broken_count in (1, 2) else -1.0)
            else:
                if stats["over_broken"] > 0:
                    score = -10.0
                else:
                    win_bonus = stats["attacker_wins"] * 1.5
                    caught_penalty = stats["caught_by_defender"] * 0.6
                    score = 1.0 + win_bonus - caught_penalty

            tie_break = (turn_id + idx) % len(candidates)
            scored.append((score, tie_break, cand))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen = scored[0][2]

        return AttackerOutput(
            mutated_code=chosen.mutated_code,
            injected_defect_type=chosen.defect_type,
            targeted_edge_case=chosen.targeted_edge_case,
            attack_strategy=chosen.strategy_name,
            expected_failing_tests=chosen.expected_broken_count,
            explanation=chosen.explanation,
            used_llm=False,
        )
