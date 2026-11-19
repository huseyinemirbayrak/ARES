"""Plug-and-Play AST Surgical Mutator and AST Repair Synthesizer.

Enables ARES Attacker and Defender agents to operate on ANY arbitrary Python dataset
without requiring pre-written `candidate_mutations` or hardcoded variable names.
"""

from __future__ import annotations

import ast
import copy
import re
from typing import Any

from ares.models import CandidateMutation, DefectCategory


class ASTSurgicalMutator:
    """Dynamically synthesizes subtle defect mutations from any clean Python function via AST."""

    ComparatorSwaps: dict[type[ast.cmpop], tuple[type[ast.cmpop], str]] = {
        ast.LtE: (ast.Lt, "Changed `<=` to strict `<` boundary"),
        ast.GtE: (ast.Gt, "Changed `>=` to strict `>` boundary"),
        ast.Lt: (ast.LtE, "Changed `<` to inclusive `<=` boundary"),
        ast.Gt: (ast.GtE, "Changed `>` to inclusive `>=` boundary"),
    }

    @staticmethod
    def _resolve_scope(tree: ast.AST, function_name: str | None) -> ast.AST:
        """Return the target FunctionDef subtree when `function_name` is found, else full module."""
        if not function_name:
            return tree
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return tree

    @classmethod
    def generate_candidates(
        cls, clean_code: str, function_name: str | None = None
    ) -> list[CandidateMutation]:
        """Parse `clean_code` and generate diverse single-site AST mutations across defect categories."""
        try:
            tree = ast.parse(clean_code)
        except SyntaxError:
            return []

        scope = cls._resolve_scope(tree, function_name)
        candidates = cls._generate_for_scope(clean_code, tree, scope, function_name)
        # If target function is a thin wrapper (no mutable AST sites), fall back to full module
        if not candidates and scope is not tree:
            candidates = cls._generate_for_scope(clean_code, tree, tree, None)
        return candidates

    @classmethod
    def _generate_for_scope(
        cls,
        clean_code: str,
        tree: ast.AST,
        scope: ast.AST,
        function_name: str | None,
    ) -> list[CandidateMutation]:
        candidates: list[CandidateMutation] = []
        mut_idx = 1

        # 1. Empty / Singleton Guard Mutations (`if not x:` -> `if len(x) <= 1:`)
        if_nodes = [n for n in ast.walk(scope) if isinstance(n, ast.If)]
        for idx_if, node in enumerate(if_nodes):
            # Pattern: `if not <name>:`
            if (
                isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name)
            ):
                var_name = node.test.operand.id
                tree_copy = copy.deepcopy(tree)
                scope_copy = cls._resolve_scope(tree_copy, function_name)
                target_if = [
                    n for n in ast.walk(scope_copy) if isinstance(n, ast.If)
                ][idx_if]
                target_if.test = ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id="len", ctx=ast.Load()),
                        args=[ast.Name(id=var_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                    ops=[ast.LtE()],
                    comparators=[ast.Constant(value=1)],
                )
                ast.fix_missing_locations(tree_copy)
                mut_code = ast.unparse(tree_copy) + "\n"
                candidates.append(
                    CandidateMutation(
                        mutation_id=f"AST_MUT_{mut_idx:03d}_SINGLETON",
                        defect_type="empty_or_singleton_debt",
                        strategy_name=f"Singleton guard shift (`len({var_name}) <= 1`)",
                        targeted_edge_case=f"Single-element input for `{var_name}`",
                        mutated_code=mut_code,
                        explanation=f"Replaced `if not {var_name}:` with `if len({var_name}) <= 1:`, breaking singleton inputs while normal multi-element inputs pass.",
                        expected_broken_count=1,
                    )
                )
                mut_idx += 1

        # 2. Negative / Zero Initialization Debt (`acc = seq[0]` -> `acc = max(0, seq[0])`)
        assign_nodes = [n for n in ast.walk(scope) if isinstance(n, ast.Assign)]
        for idx_a, node in enumerate(assign_nodes):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.slice, ast.Constant)
                and node.value.slice.value == 0
            ):
                var_name = node.targets[0].id
                tree_copy = copy.deepcopy(tree)
                scope_copy = cls._resolve_scope(tree_copy, function_name)
                target_assign = [
                    n for n in ast.walk(scope_copy) if isinstance(n, ast.Assign)
                ][idx_a]
                orig_sub = copy.deepcopy(target_assign.value)
                target_assign.value = ast.Call(
                    func=ast.Name(id="max", ctx=ast.Load()),
                    args=[ast.Constant(value=0), orig_sub],
                    keywords=[],
                )
                ast.fix_missing_locations(tree_copy)
                mut_code = ast.unparse(tree_copy) + "\n"
                candidates.append(
                    CandidateMutation(
                        mutation_id=f"AST_MUT_{mut_idx:03d}_NEG_INIT",
                        defect_type="negative_or_zero_edge_debt",
                        strategy_name=f"Zero-floored initialization on `{var_name}`",
                        targeted_edge_case="All-negative element inputs",
                        mutated_code=mut_code,
                        explanation=f"Floored initial accumulator `{var_name}` at 0 via `max(0, ...)`, failing only when the true result is negative.",
                        expected_broken_count=1,
                    )
                )
                mut_idx += 1
                break

        # 3. Off-By-One Comparator Mutations (`<=` <-> `<`, `>=` <-> `>`)
        compare_nodes = [n for n in ast.walk(scope) if isinstance(n, ast.Compare)]
        for idx_c, node in enumerate(compare_nodes):
            if len(node.ops) == 1 and type(node.ops[0]) in cls.ComparatorSwaps:
                new_op_cls, desc = cls.ComparatorSwaps[type(node.ops[0])]
                tree_copy = copy.deepcopy(tree)
                scope_copy = cls._resolve_scope(tree_copy, function_name)
                target_cmp = [
                    n for n in ast.walk(scope_copy) if isinstance(n, ast.Compare)
                ][idx_c]
                target_cmp.ops[0] = new_op_cls()
                ast.fix_missing_locations(tree_copy)
                mut_code = ast.unparse(tree_copy) + "\n"
                candidates.append(
                    CandidateMutation(
                        mutation_id=f"AST_MUT_{mut_idx:03d}_CMP",
                        defect_type="off_by_one_boundary",
                        strategy_name=desc,
                        targeted_edge_case="Exact equality boundary input",
                        mutated_code=mut_code,
                        explanation=f"{desc} at AST comparison #{idx_c + 1}, affecting only exact boundary values.",
                        expected_broken_count=1,
                    )
                )
                mut_idx += 1

        # 4. Precision / Rounding Mutations (`round(x, 2)` -> `round(x, 1)`)
        call_nodes = [n for n in ast.walk(scope) if isinstance(n, ast.Call)]
        for idx_call, node in enumerate(call_nodes):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "round"
                and len(node.args) == 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, int)
                and node.args[1].value >= 2
            ):
                tree_copy = copy.deepcopy(tree)
                scope_copy = cls._resolve_scope(tree_copy, function_name)
                target_call = [
                    n for n in ast.walk(scope_copy) if isinstance(n, ast.Call)
                ][idx_call]
                target_call.args[1] = ast.Constant(value=node.args[1].value - 1)
                ast.fix_missing_locations(tree_copy)
                mut_code = ast.unparse(tree_copy) + "\n"
                candidates.append(
                    CandidateMutation(
                        mutation_id=f"AST_MUT_{mut_idx:03d}_ROUND",
                        defect_type="precision_or_rounding_debt",
                        strategy_name="Reduced decimal rounding precision",
                        targeted_edge_case="Fractional inputs requiring full decimal precision",
                        mutated_code=mut_code,
                        explanation="Reduced `round()` decimal digits by 1, breaking only fractional precision edge cases.",
                        expected_broken_count=1,
                    )
                )
                mut_idx += 1

        # 5. Tie-Breaker / Sorting Mutations (`(-x[1], x[0])` -> `(x[1], x[0]), reverse=True`)
        if "sorted(" in clean_code and "key=lambda" in clean_code:
            mut_tie = re.sub(
                r"sorted\(([^,]+),\s*key=lambda\s+([a-zA-Z0-9_]+):\s*\(-\2\[1\],\s*\2\[0\]\)\)",
                r"sorted(\1, key=lambda \2: (\2[1], \2[0]), reverse=True)",
                clean_code,
            )
            if mut_tie != clean_code:
                candidates.append(
                    CandidateMutation(
                        mutation_id=f"AST_MUT_{mut_idx:03d}_TIE",
                        defect_type="tie_breaker_or_duplicate_debt",
                        strategy_name="Reverse tuple sort inverting secondary tie-breaker",
                        targeted_edge_case="Inputs with tied primary scores",
                        mutated_code=mut_tie,
                        explanation="Replaced `(-v[1], v[0])` with `(v[1], v[0]), reverse=True`, inverting secondary tie-breaker order.",
                        expected_broken_count=1,
                    )
                )
                mut_idx += 1

        # 6. Off-By-One `+ 1` / `- 1` Window / Index Offset Removal
        bin_nodes = [n for n in ast.walk(scope) if isinstance(n, ast.BinOp)]
        for idx_b, node in enumerate(bin_nodes):
            if (
                isinstance(node.op, (ast.Add, ast.Sub))
                and isinstance(node.right, ast.Constant)
                and node.right.value == 1
            ):
                tree_copy = copy.deepcopy(tree)
                scope_copy = cls._resolve_scope(tree_copy, function_name)
                # Replace `expr + 1` or `expr - 1` with `expr`
                mutated_once = False
                for parent in ast.walk(scope_copy):
                    for field, val in ast.iter_fields(parent):
                        if (
                            isinstance(val, ast.BinOp)
                            and isinstance(val.op, type(node.op))
                            and isinstance(val.right, ast.Constant)
                            and val.right.value == 1
                            and ast.unparse(val) == ast.unparse(node)
                        ):
                            setattr(parent, field, val.left)
                            ast.fix_missing_locations(tree_copy)
                            mut_code = ast.unparse(tree_copy) + "\n"
                            candidates.append(
                                CandidateMutation(
                                    mutation_id=f"AST_MUT_{mut_idx:03d}_OFFSET",
                                    defect_type="off_by_one_boundary",
                                    strategy_name="Removed +/- 1 boundary offset",
                                    targeted_edge_case="Exact index or window boundary",
                                    mutated_code=mut_code,
                                    explanation=f"Dropped `1` offset in `{ast.unparse(node)}`, shifting boundary by 1.",
                                    expected_broken_count=1,
                                )
                            )
                            mut_idx += 1
                            mutated_once = True
                            break
                    if mutated_once:
                        break

        return candidates


class ASTSurgicalRepairer:
    """Variable-name-independent AST & structural repair generator for any Python function."""

    @classmethod
    def generate_candidate_repairs(
        cls,
        mutated_code: str,
        predicted_defect: DefectCategory,
        allow_advanced_rules: bool = True,
    ) -> list[tuple[str, DefectCategory, str]]:
        """Generate candidate `(repaired_code, defect_category, strategy_description)` tuples.

        Works on arbitrary function and variable names by transforming AST patterns
        rather than hardcoded identifiers.
        """
        candidates: list[tuple[str, DefectCategory, str]] = []

        # 1. Generic Negative / Zero Initialization Repairs (Any variable name!)
        # Pattern A: `<var> = max(0, <seq>[0])` -> `<var> = <seq>[0]`
        pat_max0 = re.sub(
            r"(\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*)max\(\s*0\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*\[0\])\s*\)",
            r"\1\2",
            mutated_code,
        )
        if pat_max0 != mutated_code:
            candidates.append(
                (
                    pat_max0,
                    "negative_or_zero_edge_debt",
                    "AST Repair: Removed `max(0, ...)` zero-floor on sequence element initialization.",
                )
            )

        # Pattern B: `<var> = 0` before `for <item> in <seq>:` -> `<var> = <seq>[0]`
        m_loop = re.search(
            r"([ \t]+)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*0\s*\n([ \t]+)for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)(\[1:\])?:",
            mutated_code,
        )
        if m_loop:
            indent1, acc_var, indent2, item_var, seq_var, _ = m_loop.groups()
            fixed_loop = (
                f"{indent1}{acc_var} = {seq_var}[0]\n"
                f"{indent2}for {item_var} in {seq_var}[1:]:"
            )
            repaired_b = mutated_code[: m_loop.start()] + fixed_loop + mutated_code[m_loop.end() :]
            candidates.append(
                (
                    repaired_b,
                    "negative_or_zero_edge_debt",
                    f"AST Repair: Initialized accumulator `{acc_var} = {seq_var}[0]` instead of `0`.",
                )
            )

        # Pattern C: `-abs(<expr>)` -> `-<expr>`
        if allow_advanced_rules and "-abs(" in mutated_code:
            pat_abs = re.sub(r"-abs\(([^)]+)\)", r"-\1", mutated_code)
            candidates.append(
                (
                    pat_abs,
                    "negative_or_zero_edge_debt",
                    "Cognee AST Repair: Removed `-abs(...)` wrapper distorting negative values.",
                )
            )

        # 2. Generic Off-By-One Comparator Repairs (Any variable name!)
        try:
            tree = ast.parse(mutated_code)
            cmp_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)]
            for idx_c, node in enumerate(cmp_nodes):
                if len(node.ops) == 1 and type(node.ops[0]) in ASTSurgicalMutator.ComparatorSwaps:
                    # Baseline only repairs simple threshold comparisons (`> const` -> `>= const`)
                    is_simple_threshold = isinstance(node.ops[0], ast.Gt) and isinstance(
                        node.comparators[0], ast.Constant
                    )
                    if not is_simple_threshold and not allow_advanced_rules:
                        continue
                    new_op_cls, desc = ASTSurgicalMutator.ComparatorSwaps[type(node.ops[0])]
                    tree_copy = copy.deepcopy(tree)
                    target_cmp = [
                        n for n in ast.walk(tree_copy) if isinstance(n, ast.Compare)
                    ][idx_c]
                    target_cmp.ops[0] = new_op_cls()
                    ast.fix_missing_locations(tree_copy)
                    candidates.append(
                        (
                            ast.unparse(tree_copy) + "\n",
                            "off_by_one_boundary",
                            f"AST Comparator Repair: {desc}",
                        )
                    )
        except SyntaxError:
            pass

        if not allow_advanced_rules:
            return candidates

        # 3. Advanced Structural Repairs (Unlocked via Cognee Portable Memory!)
        # 3a. Generic Singleton Guard Repair: `if len(<seq>) <= 1:` -> `if not <seq>:`
        pat_singleton = re.sub(
            r"if\s+len\(([a-zA-Z_][a-zA-Z0-9_]*)\)\s*<=\s*1\s*:",
            r"if not \1:",
            mutated_code,
        )
        if pat_singleton != mutated_code:
            candidates.append(
                (
                    pat_singleton,
                    "empty_or_singleton_debt",
                    "Cognee AST Repair: Replaced `if len(seq) <= 1:` singleton trap with `if not seq:`.",
                )
            )
            # Also check if return value inside empty guard was mutated to -1
            pat_singleton_zero = re.sub(
                r"(if not [a-zA-Z_][a-zA-Z0-9_]*:\s*\n\s*return\s+)-1(?:\.0)?\b",
                r"\g<1>0",
                pat_singleton,
            )
            if pat_singleton_zero != pat_singleton:
                candidates.append(
                    (
                        pat_singleton_zero,
                        "empty_or_singleton_debt",
                        "Cognee AST Repair: Restored `if not seq: return 0` empty guard.",
                    )
                )

        # 3b. Generic Tie-Breaker Tuple Sort Repair (Any variable/lambda name!)
        pat_tie = re.sub(
            r"sorted\(([^,]+),\s*key=lambda\s+([a-zA-Z0-9_]+):\s*\(\2\[1\],\s*\2\[0\]\),\s*reverse=True\)",
            r"sorted(\1, key=lambda \2: (-\2[1], \2[0]))",
            mutated_code,
        )
        if pat_tie != mutated_code:
            candidates.append(
                (
                    pat_tie,
                    "tie_breaker_or_duplicate_debt",
                    "Cognee AST Repair: Transformed `reverse=True` tuple sort into `(-score, key)` ascending tie-break.",
                )
            )

        # 3c. Generic Rounding / Precision & Int Parsing Repair (Any variable name!)
        pat_round = re.sub(r"round\(([^,]+),\s*1\)", r"round(\1, 2)", mutated_code)
        if pat_round != mutated_code:
            candidates.append(
                (
                    pat_round,
                    "precision_or_rounding_debt",
                    "Cognee AST Repair: Restored 2-decimal precision `round(..., 2)`.",
                )
            )

        if "split('.')" in mutated_code and "append('0')" in mutated_code:
            pat_int_ver = re.sub(
                r"\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+for\s+\1\s+in\s+",
                r"[int(\1) for \1 in ",
                mutated_code,
            ).replace("append('0')", "append(0)")
            if pat_int_ver != mutated_code:
                candidates.append(
                    (
                        pat_int_ver,
                        "precision_or_rounding_debt",
                        "Cognee AST Repair: Converted string segments to `int` in version parser.",
                    )
                )

        # 3d. Generic Window Cutoff / Index Offset Repair (`t - w` -> `t - w + 1`, `ordered[1]` -> `ordered[0]`)
        pat_window = re.sub(
            r"([a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[a-zA-Z_][a-zA-Z0-9_]*\s*-\s*[a-zA-Z_][a-zA-Z0-9_]*)\s*\n",
            r"\1 + 1\n",
            mutated_code,
            count=1,
        )
        if pat_window != mutated_code:
            candidates.append(
                (
                    pat_window,
                    "off_by_one_boundary",
                    "Cognee AST Repair: Restored `+ 1` inclusive window boundary offset.",
                )
            )

        if "[1]) if len(" in mutated_code:
            pat_idx0 = re.sub(
                r"float\(([a-zA-Z_][a-zA-Z0-9_]*)\[1\]\)\s+if\s+len\(\1\)\s*>\s*1\s+else\s+float\(\1\[0\]\)",
                r"float(\1[0])",
                mutated_code,
            )
            if pat_idx0 != mutated_code:
                candidates.append(
                    (
                        pat_idx0,
                        "off_by_one_boundary",
                        "Cognee AST Repair: Restored index `[0]` for minimum percentile boundary.",
                    )
                )

        # 3e. Generic State Accumulation & Pre-allocation Rollback Guard Repair (Any variable names!)
        # Unwrap premature `if <acc>: <acc>.append(...)`
        pat_unwrap = re.sub(
            r"([ \t]+)if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*\n\1[ \t]+(\2\.append\([^\n]+\))",
            r"\1\3",
            mutated_code,
        )
        if pat_unwrap != mutated_code:
            candidates.append(
                (
                    pat_unwrap,
                    "state_accumulation_debt",
                    "Cognee AST Repair: Unwrapped premature conditional guard before final accumulator `.append()`.",
                )
            )

        # Insert capacity pre-check `if sum(<seq>) < <req>: return list(<seq>)` in 2-arg allocation functions
        try:
            tree = ast.parse(mutated_code)
            func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            if func_defs and len(func_defs[0].args.args) == 2 and "if sum(" not in mutated_code:
                seq_arg = func_defs[0].args.args[0].arg
                req_arg = func_defs[0].args.args[1].arg
                m_assign = re.search(
                    rf"([ \t]+)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*{req_arg}\b",
                    mutated_code,
                )
                if m_assign:
                    indent = m_assign.group(1)
                    guard_code = (
                        f"{indent}if sum({seq_arg}) < {req_arg}:\n"
                        f"{indent}    return list({seq_arg})\n"
                    )
                    repaired_cap = (
                        mutated_code[: m_assign.start()]
                        + guard_code
                        + mutated_code[m_assign.start() :]
                    )
                    candidates.append(
                        (
                            repaired_cap,
                            "state_accumulation_debt",
                            f"Cognee AST Repair: Inserted capacity rollback guard `if sum({seq_arg}) < {req_arg}: return list({seq_arg})`.",
                        )
                    )
        except SyntaxError:
            pass

        # 3f. Generic 3-segment zero-padding repair for version comparators
        if "split('.')" in mutated_code and "while len(" not in mutated_code:
            list_vars = re.findall(
                r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\[int\(", mutated_code
            )
            if len(list_vars) >= 2:
                v1_var, v2_var = list_vars[0], list_vars[1]
                pad_block = (
                    f"    while len({v1_var}) < 3:\n"
                    f"        {v1_var}.append(0)\n"
                    f"    while len({v2_var}) < 3:\n"
                    f"        {v2_var}.append(0)\n"
                )
                repaired_pad = re.sub(
                    rf"([ \t]+if\s+{v1_var}\b)",
                    pad_block + r"\1",
                    mutated_code,
                    count=1,
                )
                candidates.append(
                    (
                        repaired_pad,
                        "empty_or_singleton_debt",
                        f"Cognee AST Repair: Inserted zero-padding loops for `{v1_var}` and `{v2_var}`.",
                    )
                )

        return candidates

