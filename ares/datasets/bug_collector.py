"""Verified Bug Dataset Collector for ARES Adversarial Self-Play.

Captures every genuine, Referee-verified subtle bug produced by the Attacker during
self-play (both Defender-caught and Defender-evaded), deduplicates by AST fingerprint,
and persists them into a dedicated, reusable directory separate from `outputs/` and
`memory_store/`:

- `generated_bugs.jsonl`: Append-only structured records with full provenance & diffs.
- `generated_exam_benchmark.json`: Exact `external_exam_benchmark.json` (`CodeProblem`)
  schema directly compatible with `--validate-dataset` and `--exam-dataset`.
- `dataset_manifest.json`: Aggregate counts, rejection statistics, and difficulty breakdown.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ares.memory.reflexion import compute_compact_diff
from ares.models import (
    AttackerOutput,
    CandidateMutation,
    CodeProblem,
    DefenderOutput,
    RefereeVerdict,
)
from ares.referee.deterministic_referee import DeterministicReferee


def _code_ast_fingerprint(problem_id: str, code: str) -> str:
    """Compute a comment/whitespace-invariant SHA-256 fingerprint for deduplication."""
    try:
        canonical = ast.dump(ast.parse(code))
    except SyntaxError:
        canonical = code.strip()
    raw = f"{problem_id}::{canonical}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _mutation_record_fingerprint(
    problem_id: str, function_name: str, clean_code: str, buggy_code: str
) -> str:
    """Compute a collision-free AST fingerprint scoped by problem ID, function name, and clean AST."""
    clean_fp = _code_ast_fingerprint(problem_id, clean_code)
    return _code_ast_fingerprint(f"{problem_id}::{function_name}::{clean_fp}", buggy_code)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write file atomically via temporary sibling + os.replace() to prevent partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


class GeneratedBugDatasetStore:
    """Persistent, deduplicated store for Referee-verified 1-2 test-breaking Attacker bugs."""

    def __init__(self, output_dir: str | Path = "generated_datasets") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "generated_bugs.jsonl"
        self.benchmark_json_path = self.output_dir / "generated_exam_benchmark.json"
        self.manifest_path = self.output_dir / "dataset_manifest.json"

        self._records: list[dict[str, Any]] = []
        self._fingerprints: set[str] = set()
        self._used_sample_ids: set[str] = set()
        self._next_sample_num: int = 1
        self._clean_verified_cache: dict[tuple[str, str, int], bool] = {}

        # Per-run counters
        self.saved_this_run: int = 0
        self.duplicates_skipped: int = 0
        self.rejected_this_run: int = 0
        self.rejection_reasons: dict[str, int] = {}

        self._load_existing()

    def _load_existing(self) -> None:
        """Load previously saved records so multiple self-play runs accumulate without duplicates."""
        if not self.jsonl_path.exists():
            return
        required_keys = (
            "source_problem_id",
            "function_name",
            "clean_code",
            "buggy_code",
            "test_cases",
            "defect_type",
            "broken_test_count",
        )
        max_num = 0
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or not all(k in rec for k in required_keys):
                continue
            sid = str(rec.get("sample_id", "")).strip()
            if sid:
                self._used_sample_ids.add(sid)
                if sid.startswith("GEN-") and sid[4:].isdigit():
                    max_num = max(max_num, int(sid[4:]))
            canonical_fp = _mutation_record_fingerprint(
                str(rec.get("source_problem_id", "")),
                str(rec.get("function_name", "")),
                str(rec.get("clean_code", "")),
                str(rec.get("buggy_code", "")),
            )
            stored_fp = str(rec.get("fingerprint") or "").strip()
            rec["fingerprint"] = stored_fp or canonical_fp
            if canonical_fp not in self._fingerprints and (
                not stored_fp or stored_fp not in self._fingerprints
            ):
                self._fingerprints.add(canonical_fp)
                if stored_fp:
                    self._fingerprints.add(stored_fp)
                self._records.append(rec)
        self._next_sample_num = max(max_num, len(self._records)) + 1

    def _allocate_sample_id(self) -> str:
        """Allocate a monotonically increasing, collision-free GEN-NNNN identifier."""
        while True:
            candidate = f"GEN-{self._next_sample_num:04d}"
            self._next_sample_num += 1
            if candidate not in self._used_sample_ids:
                self._used_sample_ids.add(candidate)
                return candidate

    def _bump_rejection(self, reason: str) -> None:
        self.rejected_this_run += 1
        bucket = reason.split(":")[0].split(" ")[0].strip() or "invalid"
        self.rejection_reasons[bucket] = self.rejection_reasons.get(bucket, 0) + 1

    def record_turn_bug(
        self,
        *,
        run_id: str,
        turn_id: int,
        problem: CodeProblem,
        attacker_output: AttackerOutput,
        defender_output: DefenderOutput,
        verdict: RefereeVerdict,
        referee: DeterministicReferee,
        attacker_model: str = "unknown",
        defender_model: str = "unknown",
    ) -> dict[str, Any]:
        """Validate and persist a turn's mutated code iff it is a genuine 1-2 test-breaking bug.

        Saves ALL verified subtle bugs (both Defender-caught and Defender-evaded),
        tagged with `evaded_defender`, while strictly rejecting no-ops, comment-only
        changes, syntax/security errors, timeouts, over-broken (>2) mutations, and duplicates.
        """
        # 0. Verify baseline clean_code genuinely passes all tests (cached per problem + clean AST + test count)
        clean_cache_key = (
            problem.problem_id,
            _code_ast_fingerprint(problem.problem_id, problem.clean_code),
            len(problem.test_cases),
        )
        clean_ok = self._clean_verified_cache.get(clean_cache_key)
        if clean_ok is None:
            clean_suite = referee.execute_test_suite(problem.clean_code, problem.test_cases)
            clean_ok = (
                clean_suite.syntax_valid
                and not clean_suite.timed_out
                and clean_suite.failed_count == 0
            )
            self._clean_verified_cache[clean_cache_key] = clean_ok
        if not clean_ok:
            self._bump_rejection("invalid_clean_baseline")
            return {"status": "rejected", "reason": "invalid_clean_baseline"}

        # 1. Strict Referee verification gate (1-2 broken tests + valid syntax + real AST change)
        reusable_suite = (
            verdict.attacker_suite_result
            if verdict.attacker_suite_result.total_tests == len(problem.test_cases)
            else None
        )
        is_real_bug, reason, verified_suite = referee.verify_bug_mutation(
            problem.clean_code,
            attacker_output.mutated_code,
            problem.test_cases,
            suite_result=reusable_suite,
        )
        if not is_real_bug:
            self._bump_rejection(reason)
            return {"status": "rejected", "reason": reason}

        # 2. AST-normalized deduplication scoped by problem + function + clean AST
        fp = _mutation_record_fingerprint(
            problem.problem_id,
            problem.function_name,
            problem.clean_code,
            attacker_output.mutated_code,
        )
        if fp in self._fingerprints:
            self.duplicates_skipped += 1
            return {"status": "duplicate", "fingerprint": fp}

        sample_id = self._allocate_sample_id()
        now = datetime.now(timezone.utc).isoformat()
        diff_text = compute_compact_diff(problem.clean_code, attacker_output.mutated_code)
        evaded_defender = not verdict.defense_repaired_all_tests

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "fingerprint": fp,
            "created_at": now,
            "run_id": run_id,
            "turn_id": turn_id,
            "attacker_model": attacker_model,
            "defender_model": defender_model,
            "source_problem_id": problem.problem_id,
            "title": problem.title,
            "mlcq_category": problem.mlcq_category,
            "severity": problem.severity,
            "function_name": problem.function_name,
            "description": problem.description,
            "clean_code": problem.clean_code,
            "buggy_code": attacker_output.mutated_code,
            "unified_diff": diff_text,
            "test_cases": [tc.model_dump() for tc in problem.test_cases],
            "defect_type": verdict.ground_truth_defect,
            "ground_truth_source": verdict.ground_truth_source,
            "attack_strategy": attacker_output.attack_strategy,
            "targeted_edge_case": attacker_output.targeted_edge_case,
            "explanation": attacker_output.explanation,
            "broken_test_count": verified_suite.failed_count,
            "total_tests": verified_suite.total_tests,
            "failed_test_ids": list(verified_suite.failed_test_ids),
            "attack_validity": verdict.attack_validity,
            "evaded_defender": evaded_defender,
            "defender_repaired_all_tests": verdict.defense_repaired_all_tests,
            "defender_predicted_defect": defender_output.predicted_defect_type,
            "defender_repaired_code": defender_output.repaired_code,
            "attacker_used_llm": attacker_output.used_llm,
            "defender_used_llm": defender_output.used_llm,
        }

        self._fingerprints.add(fp)
        self._records.append(record)
        self.saved_this_run += 1

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._write_benchmark_and_manifest()
        return {"status": "saved", "sample_id": sample_id, "fingerprint": fp}

    def build_exam_benchmark_entries(self) -> list[dict[str, Any]]:
        """Group all saved verified bugs into exact `external_exam_benchmark.json` (`CodeProblem`) schema."""
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        used_problem_ids: set[str] = set()
        for rec in self._records:
            src_id = str(rec["source_problem_id"])
            prob_sig = self._problem_signature(rec)
            group_key = (src_id, prob_sig)
            if group_key not in grouped:
                base_pid = f"GEN-{src_id}"
                pid = base_pid
                suffix_idx = 2
                while pid in used_problem_ids:
                    pid = f"{base_pid}-{suffix_idx}"
                    suffix_idx += 1
                used_problem_ids.add(pid)
                grouped[group_key] = {
                    "problem_id": pid,
                    "title": rec.get("title") or f"Generated Benchmark: {rec['function_name']}",
                    "mlcq_category": rec.get("mlcq_category") or "ARES Attacker-Generated Subtle Bugs",
                    "severity": rec.get("severity") or "major",
                    "function_name": rec["function_name"],
                    "description": rec.get("description") or f"Analyze and repair `{rec['function_name']}`.",
                    "clean_code": rec["clean_code"],
                    "test_cases": rec["test_cases"],
                    "candidate_mutations": [],
                }
            sid = str(rec.get("sample_id") or f"GEN-{rec['fingerprint'][:8]}")
            mut = CandidateMutation(
                mutation_id=f"{sid}_{rec['fingerprint'][:8]}",
                defect_type=rec["defect_type"],
                strategy_name=rec.get("attack_strategy") or "Attacker subtle bug",
                targeted_edge_case=rec.get("targeted_edge_case")
                or f"Breaks {rec.get('failed_test_ids', [])}",
                mutated_code=rec["buggy_code"],
                explanation=rec.get("explanation") or "Verified 1-2 test-breaking mutation.",
                expected_broken_count=int(rec["broken_test_count"]),
            )
            grouped[group_key]["candidate_mutations"].append(mut.model_dump())
        return list(grouped.values())

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate dataset statistics for scoreboard & manifest persistence."""
        total = len(self._records)
        evaded = sum(1 for r in self._records if r.get("evaded_defender"))
        caught = total - evaded
        defect_dist: dict[str, int] = {}
        unique_sources: set[str] = set()
        for r in self._records:
            dtype = str(r.get("defect_type", "unknown"))
            defect_dist[dtype] = defect_dist.get(dtype, 0) + 1
            unique_sources.add(str(r.get("source_problem_id", "")))

        return {
            "output_dir": str(self.output_dir),
            "total_saved_bugs": total,
            "saved_this_run": self.saved_this_run,
            "duplicates_skipped_this_run": self.duplicates_skipped,
            "rejected_this_run": self.rejected_this_run,
            "rejection_reasons_this_run": dict(self.rejection_reasons),
            "evaded_defender_count": evaded,
            "caught_by_defender_count": caught,
            "evaded_defender_rate_pct": round((evaded / max(1, total)) * 100.0, 2) if total else 0.0,
            "unique_source_problems": len(unique_sources),
            "defect_type_distribution": defect_dist,
            "files": {
                "jsonl": str(self.jsonl_path),
                "exam_benchmark_json": str(self.benchmark_json_path),
                "manifest_json": str(self.manifest_path),
            },
        }

    def _write_benchmark_and_manifest(self) -> None:
        """Persist `generated_exam_benchmark.json` and `dataset_manifest.json` atomically."""
        entries = self.build_exam_benchmark_entries()
        _atomic_write_text(
            self.benchmark_json_path,
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        )
        _atomic_write_text(
            self.manifest_path,
            json.dumps(self.get_summary(), indent=2, ensure_ascii=False) + "\n",
        )

    def flush(self) -> dict[str, Any]:
        """Ensure jsonl, manifest, and benchmark files exist on disk even if 0 bugs were collected."""
        if not self.jsonl_path.exists():
            self.jsonl_path.touch()
        self._write_benchmark_and_manifest()
        return self.get_summary()

    @staticmethod
    def _problem_signature(item: dict[str, Any]) -> str:
        """Canonical signature matching function name, AST-normalized clean code, and test suite."""
        fn = str(item.get("function_name", ""))
        clean_fp = _code_ast_fingerprint(fn, str(item.get("clean_code", "")))
        tests_norm = [
            (
                str(tc.get("call_expr", "")),
                json.dumps(tc.get("expected_output"), sort_keys=True, default=str),
                str(tc.get("setup_code", "")),
            )
            for tc in item.get("test_cases", [])
            if isinstance(tc, dict)
        ]
        raw = json.dumps([fn, clean_fp, tests_norm], ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def append_to_exam_benchmark(self, target_exam_path: str | Path) -> dict[str, int]:
        """Post-run optional sync: merge verified generated bugs into a target exam JSON benchmark.

        Must be called ONLY after Turn N final exam completes so Pre/Post metrics stay uncontaminated.
        Only merges into an existing problem when function name, clean_code AST, and test cases all
        match; otherwise inserts a distinct problem with a guaranteed unique `problem_id`.
        Deduplicates candidate mutations by AST fingerprint and writes atomically.
        """
        target = Path(target_exam_path)
        if not target.exists():
            raise FileNotFoundError(f"Target exam benchmark JSON not found: {target}")
        if target.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json exam benchmark file, got: {target}")

        loaded = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Target exam benchmark JSON must be a list of CodeProblem objects: {target}")

        by_sig: dict[str, dict[str, Any]] = {}
        existing_problem_ids: set[str] = set()
        existing_mut_fps: set[str] = set()
        for item in loaded:
            if not isinstance(item, dict):
                continue
            existing_problem_ids.add(str(item.get("problem_id", "")))
            sig = self._problem_signature(item)
            by_sig[sig] = item
            for mut in item.get("candidate_mutations", []):
                existing_mut_fps.add(_code_ast_fingerprint(sig, str(mut.get("mutated_code", ""))))

        new_problems_added = 0
        new_mutations_added = 0
        skipped_existing = 0

        for entry in self.build_exam_benchmark_entries():
            sig = self._problem_signature(entry)
            if sig in by_sig:
                target_item = by_sig[sig]
                target_muts = target_item.setdefault("candidate_mutations", [])
                used_mut_ids = {str(m.get("mutation_id", "")) for m in target_muts}
                for mut in entry["candidate_mutations"]:
                    fp = _code_ast_fingerprint(sig, str(mut["mutated_code"]))
                    if fp in existing_mut_fps:
                        skipped_existing += 1
                        continue
                    existing_mut_fps.add(fp)
                    mut_copy = dict(mut)
                    base_mid = str(mut_copy.get("mutation_id", f"MUT_{fp[:8]}"))
                    mid = base_mid
                    suffix_idx = 2
                    while mid in used_mut_ids:
                        mid = f"{base_mid}_{suffix_idx}"
                        suffix_idx += 1
                    mut_copy["mutation_id"] = mid
                    used_mut_ids.add(mid)
                    target_muts.append(mut_copy)
                    new_mutations_added += 1
            else:
                fresh_muts: list[dict[str, Any]] = []
                for mut in entry["candidate_mutations"]:
                    fp = _code_ast_fingerprint(sig, str(mut["mutated_code"]))
                    if fp in existing_mut_fps:
                        skipped_existing += 1
                        continue
                    existing_mut_fps.add(fp)
                    fresh_muts.append(dict(mut))
                if fresh_muts:
                    new_entry = dict(entry)
                    base_pid = str(new_entry["problem_id"])
                    pid = base_pid
                    suffix_idx = 2
                    while pid in existing_problem_ids:
                        pid = f"{base_pid}-{suffix_idx}"
                        suffix_idx += 1
                    new_entry["problem_id"] = pid
                    existing_problem_ids.add(pid)
                    new_entry["candidate_mutations"] = fresh_muts
                    loaded.append(new_entry)
                    by_sig[sig] = new_entry
                    new_problems_added += 1
                    new_mutations_added += len(fresh_muts)

        if new_problems_added > 0 or new_mutations_added > 0:
            _atomic_write_text(
                target,
                json.dumps(loaded, indent=2, ensure_ascii=False) + "\n",
            )

        return {
            "new_problems_added": new_problems_added,
            "new_mutations_added": new_mutations_added,
            "skipped_existing": skipped_existing,
            "total_problems_in_target": len(loaded),
        }
