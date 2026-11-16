"""Universal Plug-and-Play ('Tak-Çıkar') Dataset Adapter for ARES.

Allows users to plug in ANY external dataset in `.json`, `.jsonl`, `.yaml`, or `.csv`
format—even with minimal fields (e.g., just `code` and `assert` statements or `input`/`output` pairs,
without `candidate_mutations` or `function_name`)—and automatically normalizes, validates,
and synthesizes AST candidate mutations for immediate self-play or external exams.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any
import yaml

from ares.agents.ast_mutator import ASTSurgicalMutator
from ares.models import CandidateMutation, CodeProblem, UnitTestCase
from ares.referee.deterministic_referee import DeterministicReferee, _run_single_test_in_ns


def _extract_function_name(clean_code: str) -> str:
    """Extract the primary function name from Python source code using AST."""
    try:
        tree = ast.parse(clean_code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
    except SyntaxError:
        pass
    raise ValueError(f"Could not find a valid `def <function_name>` in code:\n{clean_code}")


def _parse_assert_line(line: str, fn_name: str, idx: int) -> UnitTestCase | None:
    """Parse an `assert fn(...) == expected` string into a UnitTestCase."""
    cleaned = line.strip()
    if not cleaned:
        return None
    if not cleaned.startswith("assert "):
        cleaned_try = f"assert {cleaned}"
    else:
        cleaned_try = cleaned

    try:
        tree = ast.parse(cleaned_try)
        stmt = tree.body[0]
        if isinstance(stmt, ast.Assert) and isinstance(stmt.test, ast.Compare):
            cmp = stmt.test
            if len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.Eq):
                call_expr = ast.unparse(cmp.left)
                expected_val = ast.literal_eval(cmp.comparators[0])
                return UnitTestCase(
                    test_id=f"T{idx}",
                    description=f"Auto-parsed assertion #{idx}",
                    call_expr=call_expr,
                    expected_output=expected_val,
                    is_edge_case=(idx >= 4),
                )
    except Exception:
        pass
    return None


def _normalize_test_cases(
    raw_tests: Any, clean_code: str, fn_name: str
) -> list[UnitTestCase]:
    """Convert any common test representation into normalized `UnitTestCase` objects."""
    if isinstance(raw_tests, str):
        # Could be JSON string or newline-separated assert statements
        stripped = raw_tests.strip()
        if stripped.startswith("["):
            raw_tests = json.loads(stripped)
        else:
            raw_tests = [line for line in stripped.splitlines() if line.strip()]

    if not isinstance(raw_tests, list):
        raise ValueError(f"Unsupported test_cases format: {type(raw_tests)}")

    normalized: list[UnitTestCase] = []
    for idx, item in enumerate(raw_tests, start=1):
        if isinstance(item, UnitTestCase):
            normalized.append(item)
            continue

        if isinstance(item, str):
            # Case A: `"assert fn(1, 2) == 3"` or `"fn(1, 2) == 3"`
            parsed = _parse_assert_line(item, fn_name, idx)
            if parsed is not None:
                normalized.append(parsed)
                continue
            # Case B: Bare call expression `"fn(1, 2)"` -> compute expected output from clean_code!
            expected = _run_single_test_in_ns(clean_code, item.strip())
            normalized.append(
                UnitTestCase(
                    test_id=f"T{idx}",
                    description=f"Auto-evaluated expression #{idx}",
                    call_expr=item.strip(),
                    expected_output=expected,
                    is_edge_case=(idx >= 4),
                )
            )
            continue

        if isinstance(item, dict):
            # Case C: Native ARES format (`call_expr` + `expected_output`)
            if "call_expr" in item:
                expected = item.get(
                    "expected_output",
                    item.get("expected", item.get("output")),
                )
                if expected is None and "expected_output" not in item:
                    expected = _run_single_test_in_ns(clean_code, item["call_expr"])
                normalized.append(
                    UnitTestCase(
                        test_id=str(item.get("test_id", f"T{idx}")),
                        description=str(item.get("description", f"Test case #{idx}")),
                        call_expr=str(item["call_expr"]),
                        expected_output=expected,
                        is_edge_case=bool(item.get("is_edge_case", idx >= 4)),
                    )
                )
                continue

            # Case D: Input / Output dict (`{"input": [1, 2], "output": 3}` or `{"args": ...}`)
            raw_args = item.get("args", item.get("input", item.get("inputs")))
            raw_kwargs = item.get("kwargs", {})
            if isinstance(raw_args, list) and not item.get("single_list_arg", False):
                args_repr = ", ".join(repr(a) for a in raw_args)
            else:
                args_repr = repr(raw_args)
            if raw_kwargs and isinstance(raw_kwargs, dict):
                kw_repr = ", ".join(f"{k}={v!r}" for k, v in raw_kwargs.items())
                call_expr = f"{fn_name}({args_repr}, {kw_repr})" if args_repr else f"{fn_name}({kw_repr})"
            else:
                call_expr = f"{fn_name}({args_repr})"

            expected = item.get(
                "expected_output",
                item.get("expected", item.get("output")),
            )
            if expected is None and "output" not in item and "expected" not in item:
                expected = _run_single_test_in_ns(clean_code, call_expr)

            normalized.append(
                UnitTestCase(
                    test_id=str(item.get("test_id", f"T{idx}")),
                    description=str(item.get("description", f"Input/Output test #{idx}")),
                    call_expr=call_expr,
                    expected_output=expected,
                    is_edge_case=bool(item.get("is_edge_case", idx >= 4)),
                )
            )

    if not normalized:
        raise ValueError(f"No valid test cases could be extracted for function `{fn_name}`.")
    return normalized


def _auto_populate_mutations(
    clean_code: str,
    fn_name: str,
    test_cases: list[UnitTestCase],
    existing_mutations: list[dict[str, Any]] | None,
) -> list[CandidateMutation]:
    """Validate existing mutations or auto-synthesize subtle 1-2 test-breaking AST mutations."""
    if existing_mutations:
        return [CandidateMutation.model_validate(m) for m in existing_mutations]

    referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
    raw_ast_candidates = ASTSurgicalMutator.generate_candidates(clean_code, fn_name)
    verified_subtle: list[CandidateMutation] = []
    fallback_candidates: list[CandidateMutation] = []

    for cand in raw_ast_candidates:
        suite = referee.execute_test_suite(cand.mutated_code, test_cases)
        if not suite.syntax_valid or suite.timed_out:
            continue
        cand.expected_broken_count = suite.failed_count
        if 1 <= suite.failed_count <= 2:
            cand.targeted_edge_case = f"Breaks test(s) {suite.failed_test_ids}"
            verified_subtle.append(cand)
        elif suite.failed_count > 2:
            fallback_candidates.append(cand)

    return verified_subtle if verified_subtle else fallback_candidates


class PlugAndPlayDatasetAdapter:
    """Loads, adapts, and validates any external software analysis dataset."""

    @classmethod
    def load_any_dataset(cls, path: str | Path) -> list[CodeProblem]:
        raw_str = str(path).strip().lower()
        from ares.datasets.external_suites import (
            load_agent_green_suite,
            load_all_external_suites,
            load_cleanvul_suite,
            load_quixbugs_suite,
        )

        if raw_str in ("quixbugs", "quixbugs-master") or raw_str.endswith("quixbugs-master"):
            return load_quixbugs_suite()
        if raw_str in ("agent_green", "agent-green", "agent-green-main") or raw_str.endswith("agent-green-main"):
            return load_agent_green_suite()
        if raw_str in ("cleanvul", "cleanvul-main") or raw_str.endswith("cleanvul-main"):
            return load_cleanvul_suite()
        if raw_str in ("all_new", "all-new", "external_all"):
            return load_all_external_suites()

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        suffix = file_path.suffix.lower()
        raw_items: list[dict[str, Any]] = []

        if suffix == ".jsonl":
            for line in file_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw_items.append(json.loads(line))
        elif suffix in (".yaml", ".yml"):
            loaded = yaml.safe_load(file_path.read_text(encoding="utf-8"))
            raw_items = (
                loaded
                if isinstance(loaded, list)
                else loaded.get("problems", loaded.get("data", loaded.get("items", [])))
            )
        elif suffix == ".csv":
            with file_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                raw_items = list(reader)
        else:  # Default JSON
            loaded = json.loads(file_path.read_text(encoding="utf-8"))
            raw_items = (
                loaded
                if isinstance(loaded, list)
                else loaded.get("problems", loaded.get("data", loaded.get("items", [])))
            )

        problems: list[CodeProblem] = []
        for idx, item in enumerate(raw_items, start=1):
            clean_code = (
                item.get("clean_code")
                or item.get("code")
                or item.get("solution")
                or item.get("canonical_solution")
                or item.get("source_code")
                or item.get("snippet")
            )
            if not clean_code:
                raise ValueError(f"Dataset item #{idx} is missing `clean_code` / `code`.")

            fn_name = item.get("function_name") or _extract_function_name(clean_code)
            problem_id = str(
                item.get("problem_id")
                or item.get("id")
                or item.get("task_id")
                or f"PLUG-TD-{idx:03d}"
            )
            title = str(item.get("title") or item.get("name") or f"Function `{fn_name}` Analysis")
            mlcq_category = str(
                item.get("mlcq_category")
                or item.get("category")
                or item.get("smell")
                or "Plug-and-Play Technical Debt Scenario"
            )
            description = str(
                item.get("description")
                or item.get("prompt")
                or item.get("docstring")
                or f"Analyze, attack, and repair `{fn_name}`."
            )

            raw_tests = (
                item.get("test_cases")
                or item.get("tests")
                or item.get("assertions")
                or item.get("test_list")
            )
            test_cases = _normalize_test_cases(raw_tests, clean_code, fn_name)
            candidate_mutations = _auto_populate_mutations(
                clean_code,
                fn_name,
                test_cases,
                item.get("candidate_mutations"),
            )

            problems.append(
                CodeProblem(
                    problem_id=problem_id,
                    title=title,
                    mlcq_category=mlcq_category,
                    severity=str(item.get("severity", "major")),
                    function_name=fn_name,
                    description=description,
                    clean_code=clean_code,
                    test_cases=test_cases,
                    candidate_mutations=candidate_mutations,
                )
            )

        return problems

    @classmethod
    def validate_dataset_report(cls, path: str | Path) -> dict[str, Any]:
        """Load any dataset, run clean-code tests & AST mutation synthesis, and return a readiness report."""
        problems = cls.load_any_dataset(path)
        referee = DeterministicReferee(min_broken_tests=1, max_broken_tests=2)
        report_rows: list[dict[str, Any]] = []
        all_clean_pass = True

        for prob in problems:
            clean_suite = referee.execute_test_suite(prob.clean_code, prob.test_cases)
            if not clean_suite.syntax_valid or clean_suite.failed_count > 0:
                all_clean_pass = False

            subtle_count = sum(
                1
                for m in prob.candidate_mutations
                if 1
                <= referee.execute_test_suite(m.mutated_code, prob.test_cases).failed_count
                <= 2
            )
            report_rows.append(
                {
                    "problem_id": prob.problem_id,
                    "function_name": prob.function_name,
                    "num_tests": len(prob.test_cases),
                    "clean_code_passes_all": (
                        clean_suite.syntax_valid and clean_suite.failed_count == 0
                    ),
                    "synthesized_or_loaded_mutations": len(prob.candidate_mutations),
                    "valid_1_to_2_broken_mutations": subtle_count,
                }
            )

        return {
            "dataset_path": str(path),
            "total_problems": len(problems),
            "all_clean_code_valid": all_clean_pass,
            "problems": report_rows,
        }

