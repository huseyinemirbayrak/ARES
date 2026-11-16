"""Deterministic Referee for ARES Adversarial Self-Play.

Evaluates Attacker mutations and Defender repairs by executing the 4-5 unit test cases
deterministically, enforcing the 1-2 broken test case rule for subtle attacks, computing
reward signals, updating ELO ratings, and generating structured feedback.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import subprocess
import sys
from typing import Any

from ares.models import (
    AttackerOutput,
    CodeProblem,
    DefenderOutput,
    RefereeVerdict,
    SingleTestExecution,
    SuiteExecutionResult,
    UnitTestCase,
)


def _normalize_value(val: Any) -> Any:
    """Recursively normalize tuples/generators/sets to lists and round floats for deterministic comparison."""
    if inspect.isgenerator(val):
        return [_normalize_value(v) for v in val]
    if isinstance(val, set):
        return sorted((_normalize_value(v) for v in val), key=repr)
    if isinstance(val, (tuple, list)):
        return [_normalize_value(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _normalize_value(v) for k, v in val.items()}
    if isinstance(val, float):
        if math.isinf(val):
            return "INF" if val > 0 else "-INF"
        return round(val, 6)
    return val


_SUBPROCESS_RUNNER = """
import inspect, json, math, sys

def _normalize(val):
    if inspect.isgenerator(val):
        return [_normalize(v) for v in val]
    if isinstance(val, set):
        return sorted((_normalize(v) for v in val), key=repr)
    if isinstance(val, (tuple, list)):
        return [_normalize(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _normalize(v) for k, v in val.items()}
    if isinstance(val, float):
        if math.isinf(val):
            return "INF" if val > 0 else "-INF"
        return round(val, 6)
    return val

payload = json.loads(sys.stdin.read())
env = {"math": math}
try:
    exec(compile(payload["code"], "<ares_artifact>", "exec"), env, env)
    actual = eval(compile(payload["call_expr"], "<ares_test_call>", "eval"), env, env)
    print(json.dumps({"ok": True, "actual": _normalize(actual)}, default=repr))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
"""


def _run_single_test_in_ns(
    code: str, call_expr: str, timeout_seconds: float = 3.0
) -> Any:
    """Execute code definition and evaluate call_expr in an isolated subprocess."""
    ok, actual, err_msg, _ = _run_single_test_isolated(code, call_expr, timeout_seconds)
    if not ok:
        raise RuntimeError(err_msg or "Test execution failed")
    return actual


def _run_single_test_isolated(
    code: str, call_expr: str, timeout_seconds: float
) -> tuple[bool, Any, str | None, bool]:
    """Execute code + call_expr in an isolated subprocess that is killed on timeout."""
    payload = json.dumps({"code": code, "call_expr": call_expr})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_RUNNER],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, None, f"TimeoutExceeded (> {timeout_seconds}s)", True

    stdout = proc.stdout.strip()
    if proc.returncode != 0 or not stdout:
        err = proc.stderr.strip() or f"ProcessExitedWithCode({proc.returncode})"
        return False, None, err, False

    try:
        res = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return False, None, f"InvalidRunnerOutput: {stdout[:200]}", False

    if not res.get("ok"):
        return False, None, res.get("error", "UnknownExecutionError"), False
    return True, res.get("actual"), None, False


class DeterministicReferee:
    """100% Deterministic Referee that scores Attacker and Defender via unit test execution."""

    def __init__(
        self,
        min_broken_tests: int = 1,
        max_broken_tests: int = 2,
        timeout_seconds: float = 3.0,
        elo_k_factor: float = 32.0,
    ) -> None:
        self.min_broken_tests = min_broken_tests
        self.max_broken_tests = max_broken_tests
        self.timeout_seconds = timeout_seconds
        self.elo_k_factor = elo_k_factor

    def execute_test_suite(
        self, code: str, test_cases: list[UnitTestCase]
    ) -> SuiteExecutionResult:
        """Deterministically parse and execute all 4-5 unit test cases against `code`."""
        total = len(test_cases)
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return SuiteExecutionResult(
                syntax_valid=False,
                timed_out=False,
                total_tests=total,
                passed_count=0,
                failed_count=total,
                failed_test_ids=[tc.test_id for tc in test_cases],
                passed_test_ids=[],
                test_details=[],
                syntax_or_runtime_error=f"SyntaxError: {exc}",
            )

        passed_ids: list[str] = []
        failed_ids: list[str] = []
        details: list[SingleTestExecution] = []
        any_timeout = False

        for tc in test_cases:
            ok, actual, exec_err, timed_out = _run_single_test_isolated(
                code, tc.call_expr, self.timeout_seconds
            )
            if timed_out:
                any_timeout = True
                passed = False
                err_msg = exec_err
            elif not ok:
                passed = False
                err_msg = exec_err
            else:
                norm_actual = _normalize_value(actual)
                norm_expected = _normalize_value(tc.expected_output)
                passed = norm_actual == norm_expected
                err_msg = (
                    None
                    if passed
                    else f"Expected {tc.expected_output!r}, got {actual!r}"
                )

            if passed:
                passed_ids.append(tc.test_id)
            else:
                failed_ids.append(tc.test_id)

            details.append(
                SingleTestExecution(
                    test_id=tc.test_id,
                    description=tc.description,
                    call_expr=tc.call_expr,
                    expected_output=tc.expected_output,
                    actual_output=actual,
                    passed=passed,
                    is_edge_case=tc.is_edge_case,
                    error_message=err_msg,
                )
            )

        return SuiteExecutionResult(
            syntax_valid=True,
            timed_out=any_timeout,
            total_tests=total,
            passed_count=len(passed_ids),
            failed_count=len(failed_ids),
            failed_test_ids=failed_ids,
            passed_test_ids=passed_ids,
            test_details=details,
            syntax_or_runtime_error="TimeoutExceeded" if any_timeout else None,
        )

    def evaluate_turn(
        self,
        turn_id: int,
        problem: CodeProblem,
        attacker_output: AttackerOutput,
        defender_output: DefenderOutput,
        attacker_suite: SuiteExecutionResult | None = None,
    ) -> RefereeVerdict:
        """Score a single Attacker vs. Defender interaction deterministically."""
        if attacker_suite is None:
            attacker_suite = self.execute_test_suite(
                attacker_output.mutated_code, problem.test_cases
            )
        defender_suite = self.execute_test_suite(
            defender_output.repaired_code, problem.test_cases
        )

        # 1. Classify Attack Validity based on the 1-2 broken test rule
        broken = attacker_suite.failed_count
        total = attacker_suite.total_tests

        if not attacker_suite.syntax_valid or attacker_suite.timed_out:
            attack_validity = "invalid_syntax_or_timeout"
        elif broken == 0:
            attack_validity = "invalid_no_op_attack"
        elif broken > self.max_broken_tests:
            attack_validity = "invalid_over_broken_attack"
        else:
            attack_validity = "valid_subtle_attack"

        # 2. Check Defender classification and repair
        ground_truth_defect = (
            attacker_output.injected_defect_type
            if broken > 0
            else "clean"
        )
        classification_correct = (
            defender_output.predicted_defect_type == ground_truth_defect
        )
        defense_repaired_all = (
            defender_suite.syntax_valid
            and not defender_suite.timed_out
            and defender_suite.failed_count == 0
        )

        # 3. Assign Deterministic Rewards & Winner
        if attack_validity == "valid_subtle_attack":
            if defense_repaired_all:
                winner = "defender"
                attacker_reward = 0.25
                defender_reward = 1.0 if classification_correct else 0.75
                attacker_feedback = (
                    f"[VALID ATTACK CAUGHT] Your mutation '{attacker_output.attack_strategy}' "
                    f"({attacker_output.injected_defect_type}) broke {broken}/{total} tests "
                    f"({attacker_suite.failed_test_ids}), satisfying the 1-2 test rule, "
                    f"but Defender repaired all {total}/{total} tests. "
                    f"Try a less predictable defect category on {problem.function_name}."
                )
                defender_feedback = (
                    f"[DEFENSE SUCCESS] You repaired all {total}/{total} test cases "
                    f"(resolved failing tests {attacker_suite.failed_test_ids}). "
                    f"Defect classification: predicted='{defender_output.predicted_defect_type}', "
                    f"actual='{ground_truth_defect}' (match={classification_correct})."
                )
            else:
                winner = "attacker"
                attacker_reward = 1.0
                defender_reward = -0.3 if classification_correct else -0.8
                attacker_feedback = (
                    f"[ATTACK WIN] Surgical defect '{attacker_output.attack_strategy}' "
                    f"({attacker_output.injected_defect_type}) broke {broken}/{total} tests "
                    f"({attacker_suite.failed_test_ids}) and evaded Defender repair "
                    f"(Defender still failed {defender_suite.failed_test_ids})."
                )
                defender_feedback = (
                    f"[DEFENSE FAILED] Attacker injected '{ground_truth_defect}' breaking "
                    f"{attacker_suite.failed_test_ids}. Your repaired code still failed "
                    f"{defender_suite.failed_count}/{total} tests ({defender_suite.failed_test_ids}). "
                    f"Root cause hint: {attacker_output.explanation}"
                )

        elif attack_validity == "invalid_no_op_attack":
            attacker_reward = -0.8
            if defense_repaired_all:
                winner = "defender"
                defender_reward = 0.5
            else:
                winner = "draw"
                defender_reward = -0.5
            attacker_feedback = (
                f"[INVALID NO-OP ATTACK] Your mutation broke 0/{total} tests. "
                f"You MUST break 1 or 2 edge-case tests to score a valid attack."
            )
            defender_feedback = (
                f"[NO-OP ATTACK HANDLED] Attacker broke 0/{total} tests; "
                f"your output passed {defender_suite.passed_count}/{total} tests."
            )

        elif attack_validity == "invalid_over_broken_attack":
            attacker_reward = -0.9
            winner = "defender"
            defender_reward = 0.85 if defense_repaired_all else 0.2
            attacker_feedback = (
                f"[INVALID OVER-BROKEN ATTACK] Your mutation broke {broken}/{total} tests "
                f"({attacker_suite.failed_test_ids}), exceeding the maximum of "
                f"{self.max_broken_tests} broken tests. Keep nominal tests passing!"
            )
            defender_feedback = (
                f"[OVER-BROKEN ATTACK PENALIZED] Attacker broke {broken}/{total} tests "
                f"(disqualified for excessive breakage). Your repair passed "
                f"{defender_suite.passed_count}/{total} tests."
            )

        else:  # invalid_syntax_or_timeout
            attacker_reward = -1.0
            winner = "defender"
            defender_reward = 0.8 if defense_repaired_all else 0.2
            attacker_feedback = (
                f"[INVALID SYNTAX/TIMEOUT] Your mutated code failed execution: "
                f"{attacker_suite.syntax_or_runtime_error}"
            )
            defender_feedback = (
                f"[SYNTAX/TIMEOUT DISQUALIFICATION] Attacker produced invalid code; "
                f"your repair passed {defender_suite.passed_count}/{total} tests."
            )

        return RefereeVerdict(
            turn_id=turn_id,
            problem_id=problem.problem_id,
            attack_validity=attack_validity,
            attacker_suite_result=attacker_suite,
            defender_suite_result=defender_suite,
            ground_truth_defect=ground_truth_defect,
            defender_predicted_defect=defender_output.predicted_defect_type,
            classification_correct=classification_correct,
            defense_repaired_all_tests=defense_repaired_all,
            winner=winner,
            attacker_reward=round(attacker_reward, 3),
            defender_reward=round(defender_reward, 3),
            attacker_feedback=attacker_feedback,
            defender_feedback=defender_feedback,
        )

    def update_elo(
        self,
        attacker_elo: float,
        defender_elo: float,
        winner: str,
    ) -> tuple[float, float]:
        """Compute standard ELO rating updates for Attacker and Defender."""
        expected_a = 1.0 / (1.0 + math.pow(10.0, (defender_elo - attacker_elo) / 400.0))
        expected_d = 1.0 - expected_a

        if winner == "attacker":
            score_a, score_d = 1.0, 0.0
        elif winner == "defender":
            score_a, score_d = 0.0, 1.0
        else:
            score_a, score_d = 0.5, 0.5

        new_a = round(attacker_elo + self.elo_k_factor * (score_a - expected_a), 2)
        new_d = round(defender_elo + self.elo_k_factor * (score_d - expected_d), 2)
        return new_a, new_d

