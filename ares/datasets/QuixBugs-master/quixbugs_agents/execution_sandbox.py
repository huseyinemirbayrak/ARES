"""Execution Sandbox for safely evaluating Python algorithms and testcases."""

import sys
import json
import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import WORKSPACE_ROOT, PYTHON_PROGRAMS_DIR, JSON_TESTCASES_DIR


class PythonSandbox:
    """Runs Python algorithm snippets against QuixBugs test cases in isolated subprocesses."""

    @classmethod
    def run_quixbugs_tests(
        cls,
        code: str,
        algorithm: str,
        test_cases: Optional[List[Any]] = None,
        timeout_sec: int = 3
    ) -> Dict[str, Any]:
        """
        Executes Python code against QuixBugs test cases with strict timeout protection.
        """
        # If no explicit test_cases passed, try loading from json_testcases/<algorithm>.json
        if not test_cases:
            json_file = JSON_TESTCASES_DIR / f"{algorithm}.json"
            if json_file.exists():
                test_cases = []
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                test_cases.append(json.loads(line))
                except Exception as e:
                    return {"run": False, "passed": False, "details": f"Failed reading test cases: {e}"}

        # If it's a graph algorithm without JSON test cases, check for python_programs/<algorithm>_test.py
        if not test_cases:
            return cls._run_script_test(code, algorithm, timeout_sec=timeout_sec)

        # Prepare isolated runner script
        return cls._run_in_subprocess(code, test_cases, timeout_sec=timeout_sec)

    @classmethod
    def _run_in_subprocess(cls, code: str, test_cases: List[Any], timeout_sec: int = 3) -> Dict[str, Any]:
        """Constructs an executable Python script and evaluates inputs in a safe subprocess."""
        # Find function name in code
        try:
            parsed = ast.parse(code)
            func_name = None
            for node in parsed.body:
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    break
        except Exception as e:
            return {
                "run": True,
                "passed": False,
                "syntax_valid": False,
                "details": f"SyntaxError parsing code: {e}"
            }

        if not func_name:
            return {
                "run": True,
                "passed": False,
                "syntax_valid": False,
                "details": "No function definition found in code snippet."
            }

        runner_template = f"""import sys
import json
import copy

{code}

func = globals().get({repr(func_name)})
if not func:
    print(json.dumps({{"run": True, "passed": False, "details": "Function not defined in namespace"}}))
    sys.exit(0)

tests = {json.dumps(test_cases)}
passed_count = 0
failed_cases = []

for idx, test in enumerate(tests):
    try:
        if isinstance(test, list) and len(test) == 2:
            test_in, expected = test[0], test[1]
            if not isinstance(test_in, list):
                test_in = [test_in]
            
            actual = func(*copy.deepcopy(test_in))
            if actual == expected:
                passed_count += 1
            else:
                failed_cases.append({{"index": idx, "input": str(test_in), "expected": str(expected), "actual": str(actual)}})
        else:
            # Single input without expected output
            func(test)
            passed_count += 1
    except Exception as exc:
        failed_cases.append({{"index": idx, "input": str(test), "error": str(exc)}})

all_passed = (len(failed_cases) == 0 and passed_count > 0)
summary = f"Passed {{passed_count}}/{{len(tests)}} test cases."
if failed_cases:
    first_fail = failed_cases[0]
    if "error" in first_fail:
        summary += f" Failure on input {{first_fail['input']}}: {{first_fail['error']}}"
    else:
        summary += f" Mismatch on input {{first_fail['input']}}: Expected {{first_fail['expected']}}, Got {{first_fail['actual']}}"

print(json.dumps({{
    "run": True,
    "passed": all_passed,
    "passed_count": passed_count,
    "total_tests": len(tests),
    "details": summary,
    "failures": failed_cases[:3]
}}))
"""

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(runner_template)
            temp_path = f.name

        try:
            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            stdout = res.stdout.strip()
            if stdout:
                try:
                    return json.loads(stdout)
                except Exception:
                    return {"run": True, "passed": False, "details": stdout}
            else:
                stderr = res.stderr.strip()
                return {"run": True, "passed": False, "details": f"Execution error: {stderr}"}
        except subprocess.TimeoutExpired:
            return {
                "run": True,
                "passed": False,
                "timeout": True,
                "details": f"Execution Timed Out (> {timeout_sec}s). Likely Infinite Loop in algorithm!"
            }
        except Exception as ex:
            return {"run": False, "passed": False, "details": f"Sandbox error: {ex}"}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @classmethod
    def _run_script_test(cls, code: str, algorithm: str, timeout_sec: int = 3) -> Dict[str, Any]:
        """Runs QuixBugs graph algorithm test script if available."""
        test_script = PYTHON_PROGRAMS_DIR / f"{algorithm}_test.py"
        if not test_script.exists():
            # Perform Static Security & Bounds Sanity Check for CleanVul / Agent-Green samples
            import re
            warnings = []
            if re.search(r'\b(strcpy|sprintf|gets|eval|exec)\s*\(', code):
                warnings.append("Static Security Warning: Unbounded copy or dynamic execution primitive detected.")
            if re.search(r'subprocess\.(?:Popen|call|run)\([^)]*shell\s*=\s*True', code):
                warnings.append("Static Security Warning: subprocess with shell=True risks command injection (CWE-78).")
            if "SELECT " in code.upper() and ("+" in code or "%s" in code or "f\"" in code or "f'" in code):
                warnings.append("Static Security Warning: Unparameterized SQL query construction detected (CWE-89).")
            if warnings:
                return {
                    "run": True,
                    "passed": False,
                    "details": "Static Sandbox Audit: " + " | ".join(warnings)
                }
            return {
                "run": False,
                "passed": True,
                "details": "Static Sandbox Audit: No runtime unit test script attached; rely on semantic & security invariant analysis."
            }

        # Write code to a temp file and test
        with tempfile.TemporaryDirectory() as tmp_dir:
            algo_file = Path(tmp_dir) / f"{algorithm}.py"
            algo_file.write_text(code, encoding="utf-8")

            runner_code = f"""import sys
sys.path.insert(0, {repr(tmp_dir)})
sys.path.insert(0, {repr(str(PYTHON_PROGRAMS_DIR))})
import {algorithm}_test
{algorithm}_test.main()
print("TESTS_PASSED")
"""
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as rf:
                rf.write(runner_code)
                r_path = rf.name

            try:
                res = subprocess.run(
                    [sys.executable, r_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec
                )
                if "TESTS_PASSED" in res.stdout:
                    return {"run": True, "passed": True, "details": "Graph test cases passed successfully."}
                else:
                    err = res.stderr.strip() or res.stdout.strip()
                    return {"run": True, "passed": False, "details": f"Graph test failed: {err[:200]}"}
            except subprocess.TimeoutExpired:
                return {
                    "run": True,
                    "passed": False,
                    "timeout": True,
                    "details": f"Graph test timed out (> {timeout_sec}s). Infinite loop or cycle traversal."
                }
            except Exception as e:
                return {"run": False, "passed": False, "details": str(e)}
            finally:
                Path(r_path).unlink(missing_ok=True)
