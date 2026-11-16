"""Unified Dataset Loader for QuixBugs, CleanVul, and Agent-Green Benchmarks."""

import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Increase CSV field size limit for large source code functions in CleanVul / MLCQ
try:
    csv.field_size_limit(min(sys.maxsize, 2147483647))
except Exception:
    pass

from .config import (
    PYTHON_PROGRAMS_DIR,
    CORRECT_PYTHON_PROGRAMS_DIR,
    JSON_TESTCASES_DIR,
    CLEANVUL_DIR,
    AGENT_GREEN_VULN_FILE,
    AGENT_GREEN_TD_FILE
)


class CleanVulDatasetLoader:
    """Loader for CleanVul Function-Level Vulnerability Dataset (paired func_before / func_after)."""

    @classmethod
    def load_dataset(
        cls,
        limit: Optional[int] = 40,
        balance: bool = True,
        score_file: str = "vulnerability_score_4.csv",
        prefer_python: bool = True
    ) -> List[Dict[str, Any]]:
        csv_path = CLEANVUL_DIR / score_file
        if not csv_path.exists():
            # Fallback to any available score CSV
            candidates = sorted(CLEANVUL_DIR.glob("vulnerability_score_*.csv"), reverse=True)
            if not candidates:
                return []
            csv_path = candidates[0]

        raw_rows = []
        py_rows = []
        max_scan = max((limit or 80) * 15, 500)

        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    func_before = (row.get("func_before") or "").strip()
                    func_after = (row.get("func_after") or "").strip()
                    if not func_before or not func_after or func_before == func_after:
                        continue
                    ext = (row.get("extension") or "").strip().lower()
                    item = (idx, row)
                    if ext in ("py", ".py", "python"):
                        py_rows.append(item)
                    else:
                        raw_rows.append(item)

                    if prefer_python and len(py_rows) >= (limit or 80):
                        break
                    if not prefer_python and len(raw_rows) >= (limit or 80):
                        break
                    if idx >= max_scan and (len(py_rows) + len(raw_rows)) >= (limit or 40):
                        break
        except Exception:
            return []

        selected_rows = py_rows + raw_rows if prefer_python else raw_rows + py_rows
        samples: List[Dict[str, Any]] = []

        for idx, row in selected_rows:
            file_name = (row.get("file_name") or f"sample_{idx}").strip()
            cve_id = (row.get("cve_id") or "N/A").strip()
            cwe_id = (row.get("cwe_id") or "CWE-Unknown").strip()
            ext = (row.get("extension") or "code").strip()
            commit_msg = (row.get("commit_msg") or "").strip().splitlines()[0][:120]
            func_before = row["func_before"].strip()
            func_after = row["func_after"].strip()

            algo_label = f"cleanvul_{Path(file_name).stem}_{idx}"

            # 1. Vulnerable variant (func_before)
            samples.append({
                "task_id": f"{algo_label}_buggy",
                "task_name": f"CleanVul [{cwe_id}/{cve_id}]: {file_name} (vulnerable)",
                "dataset": "cleanvul",
                "algorithm": algo_label,
                "code": func_before,
                "language": "python" if ext in ("py", ".py", "python") else ext,
                "task_type": "vulnerability_detection",
                "ground_truth_is_buggy": True,
                "ground_truth_fix": func_after,
                "metadata": {
                    "cve_id": cve_id,
                    "cwe_id": cwe_id,
                    "commit_msg": commit_msg,
                    "score": row.get("vulnerability_score", "4")
                },
                "test_cases": []
            })

            if balance:
                # 2. Fixed/Clean variant (func_after)
                samples.append({
                    "task_id": f"{algo_label}_clean",
                    "task_name": f"CleanVul [{cwe_id}/{cve_id}]: {file_name} (fixed/clean)",
                    "dataset": "cleanvul",
                    "algorithm": algo_label,
                    "code": func_after,
                    "language": "python" if ext in ("py", ".py", "python") else ext,
                    "task_type": "vulnerability_detection",
                    "ground_truth_is_buggy": False,
                    "ground_truth_fix": func_after,
                    "metadata": {
                        "cve_id": cve_id,
                        "cwe_id": cwe_id,
                        "commit_msg": commit_msg,
                        "score": row.get("vulnerability_score", "4")
                    },
                    "test_cases": []
                })

            if limit and len(samples) >= limit:
                break

        return samples[:limit] if limit else samples


class AgentGreenDatasetLoader:
    """Loader for Agent-Green Benchmark Datasets (VulTrial Vulnerability & MLCQ Technical Debt)."""

    @classmethod
    def load_vuln_dataset(
        cls,
        limit: Optional[int] = 40,
        balance: bool = True
    ) -> List[Dict[str, Any]]:
        if not AGENT_GREEN_VULN_FILE.exists():
            return []

        buggy_pool: List[Dict[str, Any]] = []
        clean_pool: List[Dict[str, Any]] = []

        try:
            with open(AGENT_GREEN_VULN_FILE, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    target = int(row.get("target", 0))
                    is_buggy = (target == 1)
                    project = row.get("project", "project")
                    file_name = row.get("file_name", f"file_{line_idx}")
                    cve = row.get("cve") or "N/A"
                    cwe_list = row.get("cwe") or []
                    cwe_str = cwe_list[0] if isinstance(cwe_list, list) and cwe_list else str(cwe_list)
                    code = (row.get("func") or "").strip()
                    if not code:
                        continue

                    algo_label = f"ag_{project}_{Path(str(file_name)).stem}_{row.get('idx', line_idx)}"
                    variant_str = "vulnerable" if is_buggy else "clean"

                    sample = {
                        "task_id": f"{algo_label}_{variant_str}",
                        "task_name": f"AgentGreen [{project}/{cve}]: {file_name} ({variant_str})",
                        "dataset": "agent_green",
                        "algorithm": algo_label,
                        "code": code,
                        "language": "c",
                        "task_type": "vulnerability_detection",
                        "ground_truth_is_buggy": is_buggy,
                        "ground_truth_fix": None,
                        "metadata": {
                            "project": project,
                            "cve": cve,
                            "cwe": cwe_str,
                            "cve_desc": (row.get("cve_desc") or "")[:200]
                        },
                        "test_cases": []
                    }

                    if is_buggy:
                        buggy_pool.append(sample)
                    else:
                        clean_pool.append(sample)
        except Exception:
            return []

        if not balance:
            combined = buggy_pool + clean_pool
            return combined[:limit] if limit else combined

        # Interleave buggy and clean for balanced precision/recall evaluation
        samples: List[Dict[str, Any]] = []
        max_pairs = max(len(buggy_pool), len(clean_pool))
        for i in range(max_pairs):
            if i < len(buggy_pool):
                samples.append(buggy_pool[i])
            if i < len(clean_pool):
                samples.append(clean_pool[i])
            if limit and len(samples) >= limit:
                break

        return samples[:limit] if limit else samples

    @classmethod
    def load_td_dataset(
        cls,
        limit: Optional[int] = 40,
        balance: bool = True
    ) -> List[Dict[str, Any]]:
        """Loads Agent-Green MLCQ Technical Debt / Code Smell dataset."""
        if not AGENT_GREEN_TD_FILE.exists():
            return []

        buggy_pool: List[Dict[str, Any]] = []
        clean_pool: List[Dict[str, Any]] = []

        try:
            with open(AGENT_GREEN_TD_FILE, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=";")
                for idx, row in enumerate(reader):
                    code = (row.get("code_snippet") or "").strip()
                    if not code:
                        continue
                    smell = (row.get("smell") or "none").strip().lower()
                    severity = (row.get("severity") or "none").strip().lower()
                    is_buggy = (severity != "none" and smell != "none")
                    code_name = (row.get("code_name") or f"td_{idx}").split(".")[-1]
                    algo_label = f"ag_td_{code_name}_{row.get('id', idx)}"
                    variant_str = f"smell_{smell}" if is_buggy else "clean"

                    sample = {
                        "task_id": f"{algo_label}_{variant_str}",
                        "task_name": f"AgentGreen-TD [{smell}/{severity}]: {code_name} ({'buggy' if is_buggy else 'correct'})",
                        "dataset": "agent_green_td",
                        "algorithm": algo_label,
                        "code": code,
                        "language": "java",
                        "task_type": "technical_debt_detection",
                        "ground_truth_is_buggy": is_buggy,
                        "ground_truth_fix": None,
                        "metadata": {
                            "smell": smell,
                            "severity": severity,
                            "repository": row.get("repository", "")
                        },
                        "test_cases": []
                    }
                    if is_buggy:
                        buggy_pool.append(sample)
                    else:
                        clean_pool.append(sample)

                    if len(buggy_pool) >= (limit or 40) and len(clean_pool) >= (limit or 40):
                        break
        except Exception:
            return []

        samples: List[Dict[str, Any]] = []
        for i in range(max(len(buggy_pool), len(clean_pool))):
            if i < len(buggy_pool):
                samples.append(buggy_pool[i])
            if i < len(clean_pool):
                samples.append(clean_pool[i])
            if limit and len(samples) >= limit:
                break

        return samples[:limit] if limit else samples


class QuixBugsDatasetLoader:
    """Unified loader for QuixBugs, CleanVul, and Agent-Green benchmark programs."""

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Returns sorted list of algorithm names available in QuixBugs."""
        if not PYTHON_PROGRAMS_DIR.exists():
            return []
        algos = []
        for f in PYTHON_PROGRAMS_DIR.glob("*.py"):
            name = f.stem
            if not name.endswith("_test") and name != "node":
                algos.append(name)
        return sorted(algos)

    @classmethod
    def load_sample(cls, algo: str, variant: str = "buggy") -> Optional[Dict[str, Any]]:
        """
        Loads a single QuixBugs sample.
        variant: 'buggy' or 'correct'
        """
        source_dir = PYTHON_PROGRAMS_DIR if variant == "buggy" else CORRECT_PYTHON_PROGRAMS_DIR
        code_file = source_dir / f"{algo}.py"
        correct_file = CORRECT_PYTHON_PROGRAMS_DIR / f"{algo}.py"
        json_file = JSON_TESTCASES_DIR / f"{algo}.json"

        if not code_file.exists():
            return None

        code = code_file.read_text(encoding="utf-8")
        ground_truth_fix = correct_file.read_text(encoding="utf-8") if correct_file.exists() else None

        test_cases = []
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            test_cases.append(json.loads(line))
            except Exception:
                test_cases = []

        return {
            "task_id": f"quixbugs_{algo}_{variant}",
            "task_name": f"QuixBugs: {algo} ({variant})",
            "dataset": "quixbugs",
            "algorithm": algo,
            "code": code,
            "language": "python",
            "task_type": "bug_detection",
            "ground_truth_is_buggy": (variant == "buggy"),
            "ground_truth_fix": ground_truth_fix,
            "test_cases": test_cases
        }

    @classmethod
    def load_dataset(
        cls,
        limit: Optional[int] = None,
        balance: bool = True,
        shuffle: bool = False,
        seed: int = 42,
        dataset: str = "quixbugs"
    ) -> List[Dict[str, Any]]:
        """
        Loads a benchmark suite of samples from the specified dataset:
        - 'quixbugs': QuixBugs algorithmic defects (Python)
        - 'cleanvul': CleanVul function-level security vulnerabilities
        - 'agent_green': Agent-Green VulTrial security vulnerabilities
        - 'agent_green_td': Agent-Green MLCQ technical debt / code smells
        - 'all': Combined balanced suite across QuixBugs, CleanVul, and Agent-Green
        """
        ds = (dataset or "quixbugs").strip().lower()

        if ds == "cleanvul":
            return CleanVulDatasetLoader.load_dataset(limit=limit, balance=balance)
        if ds in ("agent_green", "agent-green", "vultrial"):
            return AgentGreenDatasetLoader.load_vuln_dataset(limit=limit, balance=balance)
        if ds in ("agent_green_td", "mlcq", "td"):
            return AgentGreenDatasetLoader.load_td_dataset(limit=limit, balance=balance)
        if ds == "all":
            per_ds = max(2, (limit // 3) if limit else 20)
            # Ensure even number per dataset when balanced
            if balance and per_ds % 2 != 0:
                per_ds += 1
            s_qb = cls.load_dataset(limit=per_ds, balance=balance, shuffle=shuffle, seed=seed, dataset="quixbugs")
            s_cv = CleanVulDatasetLoader.load_dataset(limit=per_ds, balance=balance)
            s_ag = AgentGreenDatasetLoader.load_vuln_dataset(limit=per_ds, balance=balance)
            combined = s_qb + s_cv + s_ag
            return combined[:limit] if limit else combined

        algos = cls.list_algorithms()
        if shuffle:
            r = random.Random(seed)
            r.shuffle(algos)

        samples = []
        for algo in algos:
            buggy_sample = cls.load_sample(algo, variant="buggy")
            if buggy_sample:
                samples.append(buggy_sample)

            if balance:
                correct_sample = cls.load_sample(algo, variant="correct")
                if correct_sample:
                    samples.append(correct_sample)

            if limit and len(samples) >= limit:
                break

        return samples[:limit] if limit else samples
