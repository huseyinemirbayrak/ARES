"""AST Structural Feature Extraction, Compact Code Diffs, and Turn Reflexion Engine.

Produces concrete, transferable code-diff lessons (what worked vs. what failed) so local
LLMs genuinely improve across turns via in-context learning and Cognee semantic retrieval,
without leaking ground-truth `clean_code` or hidden unit test inputs.
"""

from __future__ import annotations

import ast
import difflib

from ares.models import (
    AttackerOutput,
    CodeProblem,
    DefenderOutput,
    RefereeVerdict,
    TurnReflexion,
)


def extract_ast_features(code: str) -> list[str]:
    """Extract structural Python AST capability tags for cross-problem similarity retrieval."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["syntax_invalid"]

    features: set[str] = set()
    func_names: set[str] = {
        n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            features.add("while_loop")
        elif isinstance(node, ast.For):
            features.add("for_loop")
        elif isinstance(node, ast.ClassDef):
            features.add("class_or_node_graph")
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            features.add("generator_yield")
        elif isinstance(node, ast.Try):
            features.add("exception_handling")
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    features.add("boundary_comparison")
                elif isinstance(op, (ast.Eq, ast.NotEq)):
                    features.add("equality_comparison")
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)) and isinstance(node.right, ast.Constant):
                if node.right.value in (1, 0):
                    features.add("arithmetic_offset")
            elif isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                features.add("float_or_rounding")
        elif isinstance(node, ast.Subscript):
            features.add("list_indexing_or_slicing")
        elif isinstance(node, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp)):
            features.add("dict_or_set_state")
        elif isinstance(node, ast.If):
            if isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
                features.add("empty_or_len_guard")
            elif any(
                isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "len"
                for c in ast.walk(node.test)
            ):
                features.add("empty_or_len_guard")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in func_names:
                    features.add("recursion")
                if fname in ("sorted", "min", "max", "heapify", "heappush", "heappop"):
                    features.add("sorting_or_heap")
                if fname in ("round", "float", "abs"):
                    features.add("float_or_rounding")
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in ("sort", "heappush", "heappop", "heappushpop", "heapify"):
                    features.add("sorting_or_heap")
                if attr in ("split", "strip", "replace", "lower", "upper", "sub", "search", "match", "findall"):
                    features.add("string_or_regex")

    if not features:
        features.add("straight_line_logic")
    return sorted(features)


def compute_compact_diff(before_code: str, after_code: str, max_lines: int = 14) -> str:
    """Return a concise unified diff showing only changed hunks suitable for 7B/14B prompts."""
    before_lines = before_code.strip().splitlines()
    after_lines = after_code.strip().splitlines()
    if before_lines == after_lines:
        return "(no code change)"

    raw_diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=1,
        )
    )
    # Strip --- / +++ file headers to save tokens
    body = [line for line in raw_diff if not line.startswith(("---", "+++"))]
    if len(body) > max_lines:
        body = body[:max_lines] + ["... (diff truncated)"]
    return "\n".join(body)


def build_turn_reflexion(
    turn_id: int,
    problem: CodeProblem,
    attacker_output: AttackerOutput,
    defender_output: DefenderOutput,
    verdict: RefereeVerdict,
    blind_attacker: bool = False,
    blind_defender: bool = False,
) -> TurnReflexion:
    """Synthesize concrete, generalizable, leak-free lessons for both Attacker and Defender."""
    ast_features = extract_ast_features(problem.clean_code)
    feat_str = ", ".join(ast_features)

    attack_diff = compute_compact_diff(problem.clean_code, attacker_output.mutated_code)
    repair_diff = compute_compact_diff(attacker_output.mutated_code, defender_output.repaired_code)

    att_failed = verdict.attacker_suite_result.failed_count
    total = verdict.attacker_suite_result.total_tests
    def_failed = verdict.defender_suite_result.failed_count

    # 1. Attacker Reflexion Lesson
    if verdict.attack_validity == "valid_subtle_attack" and verdict.winner == "attacker":
        att_outcome = "win"
        att_lesson = (
            f"[WINNING SUBTLE ATTACK | AST: {feat_str} | Type: {attacker_output.injected_defect_type}] "
            f"Surgical edit broke {att_failed}/{total} tests and EVADED Defender.\n"
            f"Effective mutation diff:\n{attack_diff}\n"
            f"Takeaway: Targeting a single edge/boundary condition with a 1-2 line edit preserves nominal paths "
            f"and defeats Defender on functions with [{feat_str}]."
        )
    elif verdict.attack_validity == "valid_subtle_attack":
        att_outcome = "caught"
        att_lesson = (
            f"[CAUGHT SUBTLE ATTACK | AST: {feat_str} | Type: {attacker_output.injected_defect_type}] "
            f"Edit broke {att_failed}/{total} tests (valid subtlety), but Defender repaired it via: "
            f"'{defender_output.defense_strategy}'.\n"
            f"Predictable diff:\n{attack_diff}\n"
            f"Takeaway: Avoid repeating this exact mutation pattern on [{feat_str}]; target a subtler "
            f"invariant (e.g., tie-break, state reset, or negative/empty corner case)."
        )
    elif verdict.attack_validity == "invalid_over_broken_attack":
        att_outcome = "invalid"
        att_lesson = (
            f"[FAILED ATTACK — OVER-BROKEN ({att_failed}/{total} tests broken) | AST: {feat_str}] "
            f"DO NOT make destructive changes like:\n{attack_diff}\n"
            f"Takeaway: You broke core nominal behavior (>2 tests failed). Next time modify ONLY one "
            f"boundary operator (`<` vs `<=`), initializer, or edge guard so standard inputs still pass."
        )
    elif verdict.attack_validity == "invalid_no_op_attack":
        att_outcome = "invalid"
        att_lesson = (
            f"[FAILED ATTACK — NO-OP (0/{total} tests broken) | AST: {feat_str}] "
            f"This edit had zero effect on outputs:\n{attack_diff}\n"
            f"Takeaway: Do not return unchanged code or dead-branch edits. Alter an active condition, "
            f"loop bound, or return edge branch."
        )
    else:
        att_outcome = "invalid"
        err = verdict.attacker_suite_result.syntax_or_runtime_error or "SyntaxError/Timeout"
        att_lesson = (
            f"[FAILED ATTACK — SYNTAX/TIMEOUT | AST: {feat_str}] Error: {err}.\n"
            f"Takeaway: Always keep valid Python syntax, preserve `def {problem.function_name}`, "
            f"and avoid infinite loops."
        )

    # 2. Defender Reflexion Lesson (NEVER leaks ground-truth `problem.clean_code`!)
    if verdict.defense_repaired_all_tests:
        def_outcome = "repaired"
        cls_note = (
            "Classification matched."
            if verdict.classification_correct
            else (
                f"Note: defect category was '{verdict.ground_truth_defect}', "
                f"not '{defender_output.predicted_defect_type}'."
            )
        )
        def_lesson = (
            f"[VERIFIED REPAIR PATTERN | AST: {feat_str} | Defect: {verdict.ground_truth_defect}] "
            f"Successfully restored {total}/{total} tests. {cls_note}\n"
            f"Verified patch diff:\n{repair_diff}\n"
            f"Rule: {defender_output.defense_strategy}"
        )
    else:
        def_outcome = "failed"
        test_scope_note = (
            f"{def_failed}/{total} hidden tests still failed"
            if blind_defender
            else f"{def_failed}/{total} tests still failed ({verdict.defender_suite_result.failed_test_ids})"
        )
        def_lesson = (
            f"[FAILED REPAIR REFLEXION | AST: {feat_str} | Actual Defect: {verdict.ground_truth_defect}] "
            f"Your patch did NOT resolve the bug ({test_scope_note}).\n"
            f"Your insufficient diff:\n{repair_diff}\n"
            f"Diagnosis lesson: On functions with [{feat_str}], inspect for '{verdict.ground_truth_defect}' "
            f"— check edge initializers, loop bounds, boundary operators (`<` vs `<=`), and empty/singleton guards "
            f"against the function specification rather than leaving the code unchanged."
        )

    return TurnReflexion(
        turn_id=turn_id,
        problem_id=problem.problem_id,
        function_name=problem.function_name,
        ast_features=ast_features,
        attack_diff=attack_diff,
        repair_diff=repair_diff if verdict.defense_repaired_all_tests else "",
        attacker_outcome=att_outcome,
        defender_outcome=def_outcome,
        attacker_lesson=att_lesson,
        defender_lesson=def_lesson,
    )
