"""Structured 5-Test-Case Suites for QuixBugs, CleanVul, and Agent-Green Datasets.

Extracts every clean, standalone Python method from the 3 external datasets and pairs
each method with 5 deterministic unit test cases (`T1`..`T5`) so they plug directly
into ARES (`load_problems`, `PlugAndPlayDatasetAdapter`, `DeterministicReferee`, and `run_ares.py`).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ares.models import CodeProblem, UnitTestCase


DATASETS_DIR = Path(__file__).resolve().parent
QUIXBUGS_DIR = DATASETS_DIR / "QuixBugs-master"
CLEANVUL_DIR = DATASETS_DIR / "CleanVul-main"
AGENT_GREEN_DIR = DATASETS_DIR / "agent-green-main"


_NODE_HELPER_HEADER = """class Node:
    def __init__(self, value=None, successor=None, successors=None, predecessors=None, incoming_nodes=None, outgoing_nodes=None):
        self.value = value
        self.successor = successor
        self.successors = list(successors) if successors else []
        self.predecessors = list(predecessors) if predecessors else []
        self.incoming_nodes = list(incoming_nodes) if incoming_nodes else []
        self.outgoing_nodes = list(outgoing_nodes) if outgoing_nodes else []

    def __lt__(self, other):
        return str(self.value) < str(getattr(other, "value", ""))

    @staticmethod
    def chain(values):
        head = None
        for v in reversed(values):
            head = Node(value=v, successor=head)
        return head

    @staticmethod
    def cycle(values, target_idx=0):
        nodes = [Node(value=v) for v in values]
        for i in range(len(nodes) - 1):
            nodes[i].successor = nodes[i + 1]
        if nodes:
            nodes[-1].successor = nodes[target_idx]
        return nodes[0] if nodes else None

    @staticmethod
    def to_values(node):
        out = []
        seen = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            out.append(node.value)
            node = node.successor
        return out

    @staticmethod
    def graph_reach(edges, start_val, goal_val, fn):
        nodes = {}
        for u, v in edges:
            nodes.setdefault(u, Node(value=u))
            nodes.setdefault(v, Node(value=v))
            nodes[u].successors.append(nodes[v])
        nodes.setdefault(start_val, Node(value=start_val))
        nodes.setdefault(goal_val, Node(value=goal_val))
        return fn(nodes[start_val], nodes[goal_val])

    @staticmethod
    def dag_order(edges, isolated, fn):
        nodes = {v: Node(value=v) for v in isolated}
        for u, v in edges:
            nodes.setdefault(u, Node(value=u))
            nodes.setdefault(v, Node(value=v))
            nodes[u].outgoing_nodes.append(nodes[v])
            nodes[v].incoming_nodes.append(nodes[u])
        ordered = fn([nodes[k] for k in sorted(nodes)])
        return [n.value for n in ordered]

    @staticmethod
    def weighted_sp(weighted_edges, start_val, goal_val, fn):
        nodes = {}
        edge_map = {}
        for u, v, w in weighted_edges:
            nodes.setdefault(u, Node(value=u))
            nodes.setdefault(v, Node(value=v))
            nodes[u].successors.append(nodes[v])
            edge_map[(nodes[u], nodes[v])] = w
        nodes.setdefault(start_val, Node(value=start_val))
        nodes.setdefault(goal_val, Node(value=goal_val))
        return fn(edge_map, nodes[start_val], nodes[goal_val])
"""


def _make_tests(fn_name: str, pairs: list[tuple[str, Any]]) -> list[UnitTestCase]:
    """Build 5 normalized UnitTestCase objects from (call_expr, expected_output) pairs."""
    return [
        UnitTestCase(
            test_id=f"T{idx}",
            description=f"Deterministic test #{idx} for `{fn_name}`",
            call_expr=call_expr,
            expected_output=expected,
            is_edge_case=(idx >= 4),
        )
        for idx, (call_expr, expected) in enumerate(pairs[:5], start=1)
    ]


def _extract_functions_from_file(py_path: Path, func_names: set[str], preamble: str = "") -> dict[str, str]:
    """Extract clean standalone function definitions from a Python file using AST."""
    source = py_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    extracted: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            seg = ast.get_source_segment(source, node)
            if seg:
                extracted[node.name] = (preamble + "\n\n" + seg).strip() + "\n"
    return extracted


def load_quixbugs_suite() -> list[CodeProblem]:
    """Load all 40 clean methods from QuixBugs-master, each with 5 deterministic test cases."""
    correct_dir = QUIXBUGS_DIR / "correct_python_programs"
    json_dir = QUIXBUGS_DIR / "json_testcases"
    problems: list[CodeProblem] = []

    extra_pads: dict[str, list[tuple[list[Any], Any]]] = {
        "is_valid_parenthesization": [
            ([""], True),
            (["()()()"], True),
        ],
        "kheapsort": [
            ([[2, 1], 1], [1, 2]),
        ],
    }

    # Part 1: All 31 JSON-backed QuixBugs algorithms
    for idx, json_file in enumerate(sorted(json_dir.glob("*.json")), start=1):
        prog_name = json_file.stem
        py_file = correct_dir / f"{prog_name}.py"
        if not py_file.exists():
            continue
        code = py_file.read_text(encoding="utf-8").strip() + "\n"
        raw_pairs: list[tuple[list[Any], Any]] = []
        for line in json_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                arr = json.loads(line)
                raw_pairs.append((arr[0], arr[1]))
        if prog_name in extra_pads:
            raw_pairs.extend(extra_pads[prog_name])

        call_pairs = [
            (f"{prog_name}({', '.join(repr(a) for a in args)})", expected)
            for args, expected in raw_pairs[:5]
        ]
        problems.append(
            CodeProblem(
                problem_id=f"QUIX-{idx:03d}",
                title=f"QuixBugs: {prog_name}",
                mlcq_category="QuixBugs Clean Algorithm Suite",
                function_name=prog_name,
                description=f"Clean reference implementation of `{prog_name}` from QuixBugs.",
                clean_code=code,
                test_cases=_make_tests(prog_name, call_pairs),
            )
        )

    # Part 2: Remaining 9 Graph & Node-based QuixBugs algorithms (5 tests each)
    graph_specs: list[tuple[str, bool, list[tuple[str, Any]]]] = [
        (
            "shortest_paths",
            False,
            [
                ("shortest_paths('A', {('A', 'B'): 3, ('B', 'C'): 2, ('A', 'C'): 6})", {"A": 0, "B": 3, "C": 5}),
                ("shortest_paths('A', {('A', 'B'): 1, ('B', 'C'): 1, ('C', 'D'): 1})", {"A": 0, "B": 1, "C": 2, "D": 3}),
                ("shortest_paths('X', {('X', 'Y'): 5, ('X', 'Z'): 2, ('Z', 'Y'): 1})", {"X": 0, "Y": 3, "Z": 2}),
                ("shortest_paths('S', {('S', 'T'): 10})", {"S": 0, "T": 10}),
                ("shortest_paths('A', {('A', 'B'): 4, ('A', 'C'): 2, ('C', 'B'): -1})", {"A": 0, "B": 1, "C": 2}),
            ],
        ),
        (
            "shortest_path_lengths",
            False,
            [
                ("shortest_path_lengths(1, {})", {"(0, 0)": 0}),
                ("shortest_path_lengths(2, {(0, 1): 3, (1, 0): 4})", {"(0, 0)": 0, "(0, 1)": 3, "(1, 0)": 4, "(1, 1)": 0}),
                ("shortest_path_lengths(2, {(0, 1): 5})", {"(0, 0)": 0, "(0, 1)": 5, "(1, 0)": "INF", "(1, 1)": 0}),
                ("shortest_path_lengths(2, {(0, 1): 0, (1, 0): 0})", {"(0, 0)": 0, "(0, 1)": 0, "(1, 0)": 0, "(1, 1)": 0}),
                (
                    "shortest_path_lengths(3, {(0, 1): 2, (1, 2): 3, (0, 2): 10, (2, 0): 1})",
                    {
                        "(0, 0)": 0, "(0, 1)": 2, "(0, 2)": 5,
                        "(1, 0)": 4, "(1, 1)": 0, "(1, 2)": 3,
                        "(2, 0)": 1, "(2, 1)": 3, "(2, 2)": 0,
                    },
                ),
            ],
        ),
        (
            "minimum_spanning_tree",
            False,
            [
                ("minimum_spanning_tree({(1, 2): 10, (2, 3): 15, (1, 3): 5, (1, 4): 2})", [[1, 2], [1, 3], [1, 4]]),
                ("minimum_spanning_tree({(1, 2): 1, (2, 3): 2})", [[1, 2], [2, 3]]),
                ("minimum_spanning_tree({(1, 2): 5})", [[1, 2]]),
                ("minimum_spanning_tree({})", []),
                ("minimum_spanning_tree({(1, 2): 4, (1, 3): 1, (3, 2): 2})", [[1, 3], [3, 2]]),
            ],
        ),
        (
            "detect_cycle",
            True,
            [
                ("detect_cycle(Node.chain([1, 2, 3, 4]))", False),
                ("detect_cycle(Node.cycle([1, 2, 3], 0))", True),
                ("detect_cycle(Node.cycle([1, 2, 3, 4], 2))", True),
                ("detect_cycle(Node.chain([42]))", False),
                ("detect_cycle(None)", False),
            ],
        ),
        (
            "reverse_linked_list",
            True,
            [
                ("Node.to_values(reverse_linked_list(Node.chain([1, 2, 3, 4, 5])))", [5, 4, 3, 2, 1]),
                ("Node.to_values(reverse_linked_list(Node.chain([10, 20])))", [20, 10]),
                ("Node.to_values(reverse_linked_list(Node.chain([99])))", [99]),
                ("Node.to_values(reverse_linked_list(None))", []),
                ("Node.to_values(reverse_linked_list(Node.chain([3, 1, 2])))", [2, 1, 3]),
            ],
        ),
        (
            "breadth_first_search",
            True,
            [
                ("Node.graph_reach([(1, 2), (2, 3)], 1, 3, breadth_first_search)", True),
                ("Node.graph_reach([(1, 2), (3, 4)], 1, 4, breadth_first_search)", False),
                ("Node.graph_reach([(1, 2), (2, 1)], 1, 1, breadth_first_search)", True),
                ("Node.graph_reach([(1, 2), (2, 1)], 1, 3, breadth_first_search)", False),
                ("Node.graph_reach([(1, 2), (1, 3), (3, 4)], 1, 4, breadth_first_search)", True),
            ],
        ),
        (
            "depth_first_search",
            True,
            [
                ("Node.graph_reach([(1, 2), (2, 3)], 1, 3, depth_first_search)", True),
                ("Node.graph_reach([(1, 2), (3, 4)], 1, 4, depth_first_search)", False),
                ("Node.graph_reach([(1, 2), (2, 1)], 1, 1, depth_first_search)", True),
                ("Node.graph_reach([(1, 2), (2, 1)], 1, 3, depth_first_search)", False),
                ("Node.graph_reach([(1, 2), (1, 3), (3, 4)], 1, 4, depth_first_search)", True),
            ],
        ),
        (
            "topological_ordering",
            True,
            [
                ("Node.dag_order([(1, 2), (2, 3)], [1, 2, 3], topological_ordering)", [1, 2, 3]),
                ("Node.dag_order([(1, 3), (2, 3)], [1, 2, 3], topological_ordering)", [1, 2, 3]),
                ("Node.dag_order([], [1, 2], topological_ordering)", [1, 2]),
                ("Node.dag_order([(1, 2)], [1], topological_ordering)", [1, 2]),
                ("Node.dag_order([], [5], topological_ordering)", [5]),
            ],
        ),
        (
            "shortest_path_length",
            True,
            [
                ("Node.weighted_sp([(1, 2, 3), (2, 3, 4)], 1, 3, shortest_path_length)", 7),
                ("Node.weighted_sp([(1, 2, 10), (1, 3, 2), (3, 2, 3)], 1, 2, shortest_path_length)", 5),
                ("Node.weighted_sp([(1, 2, 5)], 1, 1, shortest_path_length)", 0),
                ("Node.weighted_sp([(1, 2, 5)], 2, 1, shortest_path_length)", "INF"),
                ("Node.weighted_sp([(1, 2, 1), (2, 3, 1), (3, 4, 1)], 1, 4, shortest_path_length)", 3),
            ],
        ),
    ]

    base_idx = len(problems)
    for offset, (prog_name, needs_node, call_pairs) in enumerate(graph_specs, start=1):
        py_file = correct_dir / f"{prog_name}.py"
        if not py_file.exists():
            continue
        raw_code = py_file.read_text(encoding="utf-8").strip()
        full_code = (_NODE_HELPER_HEADER + "\n\n" + raw_code + "\n") if needs_node else (raw_code + "\n")
        problems.append(
            CodeProblem(
                problem_id=f"QUIX-{base_idx + offset:03d}",
                title=f"QuixBugs: {prog_name}",
                mlcq_category="QuixBugs Graph & Linked Structure Suite",
                function_name=prog_name,
                description=f"Clean reference implementation of `{prog_name}` from QuixBugs.",
                clean_code=full_code,
                test_cases=_make_tests(prog_name, call_pairs),
            )
        )

    return problems


def load_agent_green_suite() -> list[CodeProblem]:
    """Load 12 clean standalone methods from agent-green-main/src, each with 5 test cases."""
    src_dir = AGENT_GREEN_DIR / "src"
    fmt_funcs = _extract_functions_from_file(
        src_dir / "format_output.py",
        {"format_response", "format_string"},
        preamble="import logging\nimport re\nlogger = logging.getLogger('agent_green')",
    )
    debt_funcs = _extract_functions_from_file(
        src_dir / "debt_utils.py",
        {"normalize_td_label", "map_ground_truth_label"},
        preamble=(
            "import re\nfrom typing import Dict, Any\n"
            'TD_LABELS = {"0": "No smell", "1": "Blob", "2": "Data Class", "3": "Feature Envy", "4": "Long Method"}'
        ),
    )
    vuln_eval_funcs = _extract_functions_from_file(
        src_dir / "vuln_evaluation.py",
        {
            "normalize_vulnerability_basic",
            "normalize_vulnerability_conservative",
            "normalize_vulnerability_strict",
        },
    )
    lang_funcs = _extract_functions_from_file(
        src_dir / "single_agent_vuln_detection.py",
        {"detect_programming_language", "analyze_dataset_languages"},
    )
    if "analyze_dataset_languages" in lang_funcs and "detect_programming_language" in lang_funcs:
        lang_funcs["analyze_dataset_languages"] = (
            lang_funcs["detect_programming_language"] + "\n" + lang_funcs["analyze_dataset_languages"]
        )

    codegen_funcs = _extract_functions_from_file(
        src_dir / "dual_agent_code_generation.py",
        {"extract_code_from_response"},
        preamble="import re",
    )
    dual_vuln_funcs = _extract_functions_from_file(
        src_dir / "dual_agent_vuln.py",
        {"extract_vulnerability_decision", "calculate_pair_wise_metrics"},
        preamble="import json",
    )

    all_extracted: dict[str, str] = {}
    for d in (fmt_funcs, debt_funcs, vuln_eval_funcs, lang_funcs, codegen_funcs, dual_vuln_funcs):
        all_extracted.update(d)

    specs: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "format_response",
            [
                ("format_response('Template: `Receiving block <*> src: <*>`')", "Receiving block <*> src: <*>"),
                ("format_response('First `ignore` then `final <*> template`')", "final <*> template"),
                ("format_response('   plain response without backticks   ')", "plain response without backticks"),
                ("format_response('`PacketResponder <*> terminating`')", "PacketResponder <*> terminating"),
                ("format_response('multiline\\nresponse')", "multiline\nresponse"),
            ],
        ),
        (
            "format_string",
            [
                ("format_string('User {alice} logged in')", "User <*> logged in"),
                ("format_string('Block {blk_1} from {10.0.0.1}')", "Block <*> from <*>"),
                ("format_string('No placeholders here')", "No placeholders here"),
                ("format_string('{start} middle {end}')", "<*> middle <*>"),
                ("format_string('')", ""),
            ],
        ),
        (
            "normalize_td_label",
            [
                ("normalize_td_label('1')", "1"),
                ("normalize_td_label('APPROVED|3|Feature Envy')", "3"),
                ("normalize_td_label('REJECTED|4|Long Method')", "4"),
                ("normalize_td_label(None)", "0"),
                ("normalize_td_label('Invalid label 9')", "0"),
            ],
        ),
        (
            "map_ground_truth_label",
            [
                ("map_ground_truth_label({'severity': 'none', 'smell': 'blob'})", "0"),
                ("map_ground_truth_label({'severity': 'major', 'smell': 'blob'})", "1"),
                ("map_ground_truth_label({'severity': 'minor', 'smell': 'data class'})", "2"),
                ("map_ground_truth_label({'severity': 'critical', 'smell': 'feature envy'})", "3"),
                ("map_ground_truth_label({'severity': 'major', 'smell': 'long method'})", "4"),
            ],
        ),
        (
            "normalize_vulnerability_basic",
            [
                ("normalize_vulnerability_basic(None)", 1),
                ("normalize_vulnerability_basic(0)", 0),
                ("normalize_vulnerability_basic(1)", 1),
                ("normalize_vulnerability_basic('0')", 0),
                ("normalize_vulnerability_basic('1')", 1),
            ],
        ),
        (
            "normalize_vulnerability_conservative",
            [
                ("normalize_vulnerability_conservative(None)", 1),
                ("normalize_vulnerability_conservative('')", 1),
                ("normalize_vulnerability_conservative('invalid')", 1),
                ("normalize_vulnerability_conservative('0')", 0),
                ("normalize_vulnerability_conservative(1)", 1),
            ],
        ),
        (
            "normalize_vulnerability_strict",
            [
                ("normalize_vulnerability_strict(None)", 0),
                ("normalize_vulnerability_strict('')", 0),
                ("normalize_vulnerability_strict('invalid')", 0),
                ("normalize_vulnerability_strict('1')", 1),
                ("normalize_vulnerability_strict(0)", 0),
            ],
        ),
        (
            "detect_programming_language",
            [
                ("detect_programming_language('#include <stdio.h>\\nprintf(\"hi\");')", "c"),
                ("detect_programming_language('#include <vector>\\nstd::vector<int> v;')", "cpp"),
                ("detect_programming_language('public class Main { System.out.println(); }')", "java"),
                ("detect_programming_language('def foo(x):\\n    return x + 1')", "python"),
                ("detect_programming_language('SELECT * FROM users;')", "unknown"),
            ],
        ),
        (
            "analyze_dataset_languages",
            [
                (
                    "analyze_dataset_languages([{'func': 'def a(): pass'}, {'func': 'def b(): pass'}])",
                    {"python": 2},
                ),
                (
                    "analyze_dataset_languages([{'func': 'public class A {}'}, {'func': 'def b(): pass'}])",
                    {"java": 1, "python": 1},
                ),
                (
                    "analyze_dataset_languages([{'func': '#include <stdio.h>\\nmalloc(4);'}])",
                    {"c": 1},
                ),
                (
                    "analyze_dataset_languages([{'func': 'function f() { console.log(1); }'}])",
                    {"javascript": 1},
                ),
                (
                    "analyze_dataset_languages([{'func': 'plain text'}])",
                    {"unknown": 1},
                ),
            ],
        ),
        (
            "extract_code_from_response",
            [
                ("extract_code_from_response('```python\\ndef add(a, b):\\n    return a + b\\n```')", "def add(a, b):\n    return a + b"),
                ("extract_code_from_response('<think>reasoning</think>```python\\ndef f(): pass\\n```')", "def f(): pass"),
                ("extract_code_from_response('```\\npython\\ndef g(): return 1\\n```')", "def g(): return 1"),
                ("extract_code_from_response('Here is the solution:\\ndef h(x):\\n    return x')", "def h(x):\n    return x"),
                ("extract_code_from_response('')", ""),
            ],
        ),
        (
            "extract_vulnerability_decision",
            [
                ("extract_vulnerability_decision('{\"vulnerability_detected\": true, \"analysis\": \"buffer overflow\"}')[0]", 1),
                ("extract_vulnerability_decision('{\"vulnerability_detected\": false, \"analysis\": \"safe\"}')[0]", 0),
                ("extract_vulnerability_decision('This code is vulnerable to SQL injection')[0]", 1),
                ("extract_vulnerability_decision('The implementation looks clean and safe')[0]", 0),
                ("extract_vulnerability_decision('[{\"vulnerability_detected\": true, \"reasoning\": \"xss\"}]')[0]", 1),
            ],
        ),
        (
            "calculate_pair_wise_metrics",
            [
                ("calculate_pair_wise_metrics([1, 0], [1, 0])", {"P-C": 100.0, "P-V": 100.0, "P-B": 100.0, "P-R": 0.0, "FPR": 0.0}),
                ("calculate_pair_wise_metrics([0, 1], [1, 0])", {"P-C": 0.0, "P-V": 0.0, "P-B": 0.0, "P-R": 100.0, "FPR": 100.0}),
                ("calculate_pair_wise_metrics([1, 1], [1, 0])", {"P-C": 50.0, "P-V": 100.0, "P-B": 0.0, "P-R": 50.0, "FPR": 100.0}),
                ("calculate_pair_wise_metrics([0, 0], [1, 0])", {"P-C": 50.0, "P-V": 0.0, "P-B": 100.0, "P-R": 50.0, "FPR": 0.0}),
                ("calculate_pair_wise_metrics([1, 0, 1, 0], [1, 0, 0, 0])", {"P-C": 75.0, "P-V": 100.0, "P-B": 66.666667, "P-R": 25.0, "FPR": 33.333333}),
            ],
        ),
    ]

    problems: list[CodeProblem] = []
    for idx, (fn_name, call_pairs) in enumerate(specs, start=1):
        code = all_extracted[fn_name]
        problems.append(
            CodeProblem(
                problem_id=f"AGREEN-{idx:03d}",
                title=f"Agent-Green: {fn_name}",
                mlcq_category="Agent-Green Clean Utility Suite",
                function_name=fn_name,
                description=f"Clean utility function `{fn_name}` from agent-green-main.",
                clean_code=code,
                test_cases=_make_tests(fn_name, call_pairs),
            )
        )
    return problems


def load_cleanvul_suite() -> list[CodeProblem]:
    """Load 5 clean standalone Python methods from CleanVul-main, each with 5 test cases."""
    analyzer_path = CLEANVUL_DIR / "src" / "test_related_identifier.py"
    analyzer_src = analyzer_path.read_text(encoding="utf-8")

    is_test_fn_code = (
        analyzer_src.split("if __name__")[0].strip()
        + "\n\ndef is_test_function_check(code: str) -> bool:\n"
        + "    return FunctionTestAnalyzer().is_test_function(code)\n"
    )
    is_test_file_code = (
        analyzer_src.split("if __name__")[0].strip()
        + "\n\ndef is_test_file_check(filename: str) -> bool:\n"
        + "    return FunctionTestAnalyzer().is_test_file(filename)\n"
    )

    prepare_linkifier_code = (
        "BEFORE_CAPTURE_GROUP = 'before'\n"
        "OUTER_CAPTURE_GROUP = 'outer'\n"
        "AFTER_CAPTURE_GROUP = 'after'\n\n"
        "def prepare_linkifier_pattern(source: str) -> str:\n"
        '    return fr"""(?P<{BEFORE_CAPTURE_GROUP}>^|\\s|[\'\\"\\(,:<])(?P<{OUTER_CAPTURE_GROUP}>{source})(?P<{AFTER_CAPTURE_GROUP}>$|[^\\pL\\pN])"""\n'
    )

    escape_link_code = (
        "import html\nimport re\n\n"
        "_scheme_blacklist = ('javascript:', 'vbscript:', 'data:')\n\n"
        "def escape_link(url: str) -> str:\n"
        "    lower_url = url.lower().strip('\\x00\\x1a \\n\\r\\t')\n"
        "    for scheme in _scheme_blacklist:\n"
        "        if re.sub(r'[^A-Za-z0-9\\/:]+', '', lower_url).startswith(scheme):\n"
        "            return ''\n"
        "    return html.escape(url, quote=True)\n"
    )

    safe_join_path_code = (
        "from pathlib import PurePosixPath\n\n"
        "def resolve_safe_export_path(destination: str, rel_path: str) -> str:\n"
        "    if rel_path.endswith('/'):\n"
        "        rel_path = rel_path + 'index.html'\n"
        "    clean_parts = [p for p in rel_path.lstrip('/').split('/') if p and p != '.']\n"
        "    if '..' in clean_parts:\n"
        "        raise ValueError('Path traversal detected')\n"
        "    return str(PurePosixPath(destination).joinpath(*clean_parts))\n"
    )

    specs: list[tuple[str, str, list[tuple[str, Any]]]] = [
        (
            "is_test_function_check",
            is_test_fn_code,
            [
                ("is_test_function_check('def test_login():\\n    assert True')", True),
                ("is_test_function_check('@Test\\npublic void testAdd() {}')", True),
                ("is_test_function_check('TEST(Suite, Case) {}')", True),
                ("is_test_function_check('def calculate_total(items):\\n    return sum(items)')", False),
                ("is_test_function_check('public int add(int a, int b) { return a + b; }')", False),
            ],
        ),
        (
            "is_test_file_check",
            is_test_file_code,
            [
                ("is_test_file_check('test_calculator.py')", True),
                ("is_test_file_check('CalculatorTest.java')", True),
                ("is_test_file_check('calculator_test.cpp')", True),
                ("is_test_file_check('calculator.py')", False),
                ("is_test_file_check('main_service.py')", False),
            ],
        ),
        (
            "prepare_linkifier_pattern",
            prepare_linkifier_code,
            [
                (
                    "prepare_linkifier_pattern('#(?P<id>[0-9]+)')",
                    r"""(?P<before>^|\s|['"\(,:<])(?P<outer>#(?P<id>[0-9]+))(?P<after>$|[^\pL\pN])""",
                ),
                (
                    "prepare_linkifier_pattern('PR-(?P<pr>[0-9]+)')",
                    r"""(?P<before>^|\s|['"\(,:<])(?P<outer>PR-(?P<pr>[0-9]+))(?P<after>$|[^\pL\pN])""",
                ),
                (
                    "prepare_linkifier_pattern('CVE-[0-9]+')",
                    r"""(?P<before>^|\s|['"\(,:<])(?P<outer>CVE-[0-9]+)(?P<after>$|[^\pL\pN])""",
                ),
                (
                    "prepare_linkifier_pattern('bug')",
                    r"""(?P<before>^|\s|['"\(,:<])(?P<outer>bug)(?P<after>$|[^\pL\pN])""",
                ),
                (
                    "prepare_linkifier_pattern('')",
                    r"""(?P<before>^|\s|['"\(,:<])(?P<outer>)(?P<after>$|[^\pL\pN])""",
                ),
            ],
        ),
        (
            "escape_link",
            escape_link_code,
            [
                ("escape_link('https://example.com/a?x=1')", "https://example.com/a?x=1"),
                ("escape_link('javascript:alert(1)')", ""),
                ("escape_link('  JAVASCRIPT:alert(1)  ')", ""),
                ("escape_link('jav\\tascript:alert(1)')", ""),
                ("escape_link('https://example.com/\"quote\"')", "https://example.com/&quot;quote&quot;"),
            ],
        ),
        (
            "resolve_safe_export_path",
            safe_join_path_code,
            [
                ("resolve_safe_export_path('/var/www', 'schedule/')", "/var/www/schedule/index.html"),
                ("resolve_safe_export_path('/var/www', '/talk/101.html')", "/var/www/talk/101.html"),
                ("resolve_safe_export_path('/var/www', '/')", "/var/www/index.html"),
                ("resolve_safe_export_path('/export', 'assets/app.css')", "/export/assets/app.css"),
                ("resolve_safe_export_path('/export', './index.html')", "/export/index.html"),
            ],
        ),
    ]

    problems: list[CodeProblem] = []
    for idx, (fn_name, code, call_pairs) in enumerate(specs, start=1):
        problems.append(
            CodeProblem(
                problem_id=f"CLNVUL-{idx:03d}",
                title=f"CleanVul: {fn_name}",
                mlcq_category="CleanVul Security & Sanitization Suite",
                function_name=fn_name,
                description=f"Clean fixed Python function `{fn_name}` from CleanVul.",
                clean_code=code,
                test_cases=_make_tests(fn_name, call_pairs),
            )
        )
    return problems


def load_all_external_suites() -> list[CodeProblem]:
    """Return all 57 clean methods (40 QuixBugs + 12 Agent-Green + 5 CleanVul), each with 5 tests."""
    return load_quixbugs_suite() + load_agent_green_suite() + load_cleanvul_suite()
