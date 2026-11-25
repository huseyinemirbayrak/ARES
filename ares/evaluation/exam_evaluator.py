"""External Benchmark Exam Evaluator ('Dışarıdan Sınavlar') for ARES.

Periodically evaluates Attacker and Defender on a held-out benchmark suite of unseen
software analysis & technical debt problems (without modifying memory during the exam)
to measure whether accumulated Cognee Portable Memory improves agent capability over turns.
"""

from __future__ import annotations

from typing import Any

from ares.agents.attacker import AttackerAgent
from ares.agents.defender import DefenderAgent
from ares.memory.cognee_memory import CogneeMemoryManager
from ares.models import CodeProblem, ExamResult
from ares.referee.deterministic_referee import DeterministicReferee


class ExternalExamEvaluator:
    """Runs held-out External Benchmark Exams at baseline (Turn 0) and every N turns (e.g., Turn 25)."""

    def __init__(
        self,
        exam_problems: list[CodeProblem],
        referee: DeterministicReferee,
    ) -> None:
        self.exam_problems = exam_problems
        self.referee = referee

    def run_exam(
        self,
        after_turn: int,
        attacker: AttackerAgent,
        defender: DefenderAgent,
        memory: CogneeMemoryManager,
        attacker_elo: float,
        defender_elo: float,
    ) -> ExamResult:
        """Evaluate Attacker and Defender on all unseen exam problems using current Cognee memory."""
        exam_id = f"EXAM_TURN_{after_turn:03d}"
        valid_attacks = 0
        total_broken_by_attacker = 0
        full_repairs = 0
        correct_classifications = 0
        correct_detections = 0
        total_pass_rate_pct = 0.0
        details: list[dict[str, Any]] = []

        for idx, problem in enumerate(self.exam_problems, start=1):
            # Part A: Evaluate Attacker's ability to craft a 1-2 test subtle defect on unseen code
            att_out = attacker.attack(problem, memory, turn_id=after_turn + idx)
            is_valid_subtle, _, att_suite = self.referee.verify_bug_mutation(
                problem.clean_code, att_out.mutated_code, problem.test_cases
            )
            if is_valid_subtle:
                valid_attacks += 1
            total_broken_by_attacker += att_suite.failed_count

            # Part B: Evaluate Defender's ability to detect, classify & repair the canonical subtle defect
            canonical_mut = None
            canonical_suite = None
            if problem.candidate_mutations:
                ordered_cands = sorted(
                    problem.candidate_mutations,
                    key=lambda m: (
                        0
                        if self.referee.min_broken_tests
                        <= m.expected_broken_count
                        <= self.referee.max_broken_tests
                        else 1
                    ),
                )
                for cand in ordered_cands:
                    ok_cand, _, cand_suite = self.referee.verify_bug_mutation(
                        problem.clean_code, cand.mutated_code, problem.test_cases
                    )
                    if ok_cand:
                        canonical_mut = cand
                        canonical_suite = cand_suite
                        break
                if canonical_mut is None:
                    canonical_mut = ordered_cands[0]
            else:
                from ares.agents.ast_mutator import ASTSurgicalMutator
                from ares.models import CandidateMutation

                cands = ASTSurgicalMutator.generate_candidates(
                    problem.clean_code, problem.function_name
                )
                for cand in cands:
                    ok_cand, _, cand_suite = self.referee.verify_bug_mutation(
                        problem.clean_code, cand.mutated_code, problem.test_cases
                    )
                    if ok_cand:
                        canonical_mut = cand
                        canonical_suite = cand_suite
                        break
                if canonical_mut is None and cands:
                    canonical_mut = cands[0]
                if canonical_mut is None:
                    canonical_mut = CandidateMutation(
                        mutation_id="EXAM_FALLBACK_MUT",
                        defect_type=att_out.injected_defect_type,
                        strategy_name=att_out.attack_strategy,
                        targeted_edge_case=att_out.targeted_edge_case,
                        mutated_code=att_out.mutated_code,
                        explanation=att_out.explanation,
                        expected_broken_count=max(1, att_suite.failed_count),
                    )
            if canonical_suite is None:
                canonical_suite = self.referee.execute_test_suite(
                    canonical_mut.mutated_code, problem.test_cases
                )
            def_out = defender.defend(
                problem=problem,
                mutated_code=canonical_mut.mutated_code,
                attacker_suite=None if defender.hide_test_results else canonical_suite,
                memory=memory,
                turn_id=after_turn + idx,
            )
            def_suite = self.referee.execute_test_suite(
                def_out.repaired_code, problem.test_cases
            )
            repaired_all = (
                def_suite.syntax_valid
                and not def_suite.timed_out
                and def_suite.failed_count == 0
            )
            expected_defect = (
                "clean"
                if canonical_suite.syntax_valid
                and not canonical_suite.timed_out
                and canonical_suite.failed_count == 0
                else canonical_mut.defect_type
            )
            ground_truth_is_defective = expected_defect != "clean"
            detected_ok = def_out.is_defective == ground_truth_is_defective
            classified_ok = def_out.predicted_defect_type == expected_defect
            test_pass_rate = (
                round((def_suite.passed_count / max(1, def_suite.total_tests)) * 100.0, 2)
                if def_suite.syntax_valid and not def_suite.timed_out
                else 0.0
            )

            if repaired_all:
                full_repairs += 1
            if classified_ok:
                correct_classifications += 1
            if detected_ok:
                correct_detections += 1
            total_pass_rate_pct += test_pass_rate

            details.append(
                {
                    "problem_id": problem.problem_id,
                    "title": problem.title,
                    "function_name": problem.function_name,
                    "canonical_mutation_id": canonical_mut.mutation_id,
                    "attacker_broken_tests": att_suite.failed_count,
                    "attacker_valid_subtle": is_valid_subtle,
                    "attacker_used_llm": att_out.used_llm,
                    "ground_truth_is_defective": ground_truth_is_defective,
                    "ground_truth_defect": expected_defect,
                    "defender_is_defective": def_out.is_defective,
                    "defender_bug_detected_correctly": detected_ok,
                    "defender_predicted_defect": def_out.predicted_defect_type,
                    "defender_classification_correct": classified_ok,
                    "defender_repaired_all_5_tests": repaired_all,
                    "defender_passed_count": def_suite.passed_count,
                    "defender_total_tests": def_suite.total_tests,
                    "defender_test_pass_rate_pct": test_pass_rate,
                    "defender_passed_tests": f"{def_suite.passed_count}/{def_suite.total_tests}",
                    "defender_used_llm": def_out.used_llm,
                    "hide_test_results": defender.hide_test_results,
                    "defense_strategy": def_out.defense_strategy,
                }
            )

        n = max(1, len(self.exam_problems))
        att_rate = round((valid_attacks / n) * 100.0, 2)
        avg_broken = round(total_broken_by_attacker / n, 2)
        def_repair_rate = round((full_repairs / n) * 100.0, 2)
        def_class_acc = round((correct_classifications / n) * 100.0, 2)
        def_detect_rate = round((correct_detections / n) * 100.0, 2)
        def_avg_pass_rate = round(total_pass_rate_pct / n, 2)
        def_combined = round((def_repair_rate + def_class_acc) / 2.0, 2)

        return ExamResult(
            exam_id=exam_id,
            after_turn=after_turn,
            num_exam_problems=len(self.exam_problems),
            attacker_valid_subtle_rate=att_rate,
            attacker_avg_broken_tests=avg_broken,
            defender_full_repair_rate=def_repair_rate,
            defender_classification_accuracy=def_class_acc,
            defender_combined_score=def_combined,
            defender_bug_detection_rate=def_detect_rate,
            defender_avg_test_pass_rate=def_avg_pass_rate,
            attacker_elo=attacker_elo,
            defender_elo=defender_elo,
            cognee_knowledge_nodes=len(memory.nodes),
            per_problem_details=details,
        )


def build_pre_post_comparison(
    baseline_exam: dict[str, Any],
    final_exam: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured Pre-Game (Turn 0) vs. Post-Game (Turn N) comparison report."""
    metric_keys = [
        ("defender_full_repair_rate", "Defender Full Repair Rate (%)"),
        ("defender_bug_detection_rate", "Defender Bug Detection Rate (%)"),
        ("defender_classification_accuracy", "Defender Classification Accuracy (%)"),
        ("defender_avg_test_pass_rate", "Defender Avg Test Pass Rate (%)"),
        ("defender_combined_score", "Defender Combined Score (%)"),
        ("attacker_valid_subtle_rate", "Attacker Valid Subtle Attack Rate (%)"),
    ]
    metrics: dict[str, dict[str, Any]] = {}
    for key, label in metric_keys:
        before_val = float(baseline_exam.get(key, 0.0))
        after_val = float(final_exam.get(key, 0.0))
        metrics[key] = {
            "label": label,
            "before": round(before_val, 2),
            "after": round(after_val, 2),
            "delta": round(after_val - before_val, 2),
        }
    nodes_before = int(baseline_exam.get("cognee_knowledge_nodes", 0))
    nodes_after = int(final_exam.get("cognee_knowledge_nodes", 0))
    metrics["memory_knowledge_nodes"] = {
        "label": "Memory Knowledge Graph Nodes",
        "before": nodes_before,
        "after": nodes_after,
        "delta": nodes_after - nodes_before,
    }

    before_by_id = {
        row["problem_id"]: row for row in baseline_exam.get("per_problem_details", [])
    }
    after_by_id = {
        row["problem_id"]: row for row in final_exam.get("per_problem_details", [])
    }
    ordered_ids = list(before_by_id.keys()) + [
        pid for pid in after_by_id.keys() if pid not in before_by_id
    ]

    repair_transitions: dict[str, list[str]] = {
        "gained": [],
        "regressed": [],
        "always_solved": [],
        "still_unsolved": [],
    }
    detection_transitions: dict[str, list[str]] = {
        "gained": [],
        "regressed": [],
        "always_correct": [],
        "still_missed": [],
    }
    per_problem_rows: list[dict[str, Any]] = []

    for pid in ordered_ids:
        b = before_by_id.get(pid, {})
        a = after_by_id.get(pid, {})
        b_rep = bool(b.get("defender_repaired_all_5_tests", False))
        a_rep = bool(a.get("defender_repaired_all_5_tests", False))
        b_det = bool(b.get("defender_bug_detected_correctly", b.get("defender_classification_correct", False)))
        a_det = bool(a.get("defender_bug_detected_correctly", a.get("defender_classification_correct", False)))
        b_cls = bool(b.get("defender_classification_correct", False))
        a_cls = bool(a.get("defender_classification_correct", False))

        if not b_rep and a_rep:
            rep_status = "GAINED"
            repair_transitions["gained"].append(pid)
        elif b_rep and not a_rep:
            rep_status = "REGRESSED"
            repair_transitions["regressed"].append(pid)
        elif b_rep and a_rep:
            rep_status = "ALWAYS_SOLVED"
            repair_transitions["always_solved"].append(pid)
        else:
            rep_status = "STILL_UNSOLVED"
            repair_transitions["still_unsolved"].append(pid)

        if not b_det and a_det:
            detection_transitions["gained"].append(pid)
        elif b_det and not a_det:
            detection_transitions["regressed"].append(pid)
        elif b_det and a_det:
            detection_transitions["always_correct"].append(pid)
        else:
            detection_transitions["still_missed"].append(pid)

        per_problem_rows.append(
            {
                "problem_id": pid,
                "title": a.get("title") or b.get("title") or pid,
                "function_name": a.get("function_name") or b.get("function_name") or "",
                "ground_truth_defect": a.get("ground_truth_defect") or b.get("ground_truth_defect") or "unknown",
                "before_repaired": b_rep,
                "after_repaired": a_rep,
                "repair_status": rep_status,
                "before_passed_tests": b.get("defender_passed_tests", "0/0"),
                "after_passed_tests": a.get("defender_passed_tests", "0/0"),
                "before_detected": b_det,
                "after_detected": a_det,
                "before_classified": b_cls,
                "after_classified": a_cls,
                "before_predicted_defect": b.get("defender_predicted_defect", "-"),
                "after_predicted_defect": a.get("defender_predicted_defect", "-"),
            }
        )

    return {
        "baseline_exam_id": baseline_exam.get("exam_id", "EXAM_TURN_000"),
        "final_exam_id": final_exam.get("exam_id", "EXAM_FINAL"),
        "baseline_turn": int(baseline_exam.get("after_turn", 0)),
        "final_turn": int(final_exam.get("after_turn", 0)),
        "num_exam_problems": int(final_exam.get("num_exam_problems", len(per_problem_rows))),
        "metrics": metrics,
        "repair_transitions": repair_transitions,
        "detection_transitions": detection_transitions,
        "per_problem_comparison": per_problem_rows,
    }


def render_pre_post_markdown(comparison: dict[str, Any]) -> str:
    """Render a publication-ready Markdown Pre vs. Post comparison report."""
    b_turn = comparison["baseline_turn"]
    f_turn = comparison["final_turn"]
    n_probs = comparison["num_exam_problems"]
    rt = comparison["repair_transitions"]
    dt = comparison["detection_transitions"]

    lines = [
        f"# ARES Pre-Game vs. Post-Game External Benchmark Report (Turn {b_turn} -> Turn {f_turn})",
        "",
        f"- **Baseline Exam:** `{comparison['baseline_exam_id']}` (Turn {b_turn})",
        f"- **Final Exam:** `{comparison['final_exam_id']}` (Turn {f_turn})",
        f"- **Held-Out Problems Evaluated:** {n_probs}",
        "",
        "## Aggregate Metrics Summary",
        "",
        f"| Metric | Turn {b_turn} (Pre) | Turn {f_turn} (Post) | Delta |",
        "| :--- | ---: | ---: | ---: |",
    ]

    for key, m in comparison["metrics"].items():
        if key == "memory_knowledge_nodes":
            lines.append(f"| {m['label']} | {m['before']} | {m['after']} | {m['delta']:+d} |")
        else:
            lines.append(
                f"| {m['label']} | {m['before']:.2f}% | {m['after']:.2f}% | {m['delta']:+.2f}% |"
            )

    lines.extend(
        [
            "",
            "## Problem Transitions",
            "",
            f"- **Repair Gained ({len(rt['gained'])}):** {', '.join(rt['gained']) or 'None'}",
            f"- **Repair Regressed ({len(rt['regressed'])}):** {', '.join(rt['regressed']) or 'None'}",
            f"- **Always Solved ({len(rt['always_solved'])}):** {', '.join(rt['always_solved']) or 'None'}",
            f"- **Still Unsolved ({len(rt['still_unsolved'])}):** {', '.join(rt['still_unsolved']) or 'None'}",
            f"- **Bug Detection Gained ({len(dt['gained'])}):** {', '.join(dt['gained']) or 'None'}",
            f"- **Bug Detection Regressed ({len(dt['regressed'])}):** {', '.join(dt['regressed']) or 'None'}",
            "",
            "## Per-Problem Breakdown",
            "",
            f"| Problem | Defect | Tests (T{b_turn} -> T{f_turn}) | Repair (T{b_turn} -> T{f_turn}) | Detection | Classification | Status |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    def _esc(val: Any) -> str:
        return str(val).replace("|", "\\|").replace("\n", " ")

    for row in comparison["per_problem_comparison"]:
        b_rep = "PASS" if row["before_repaired"] else "FAIL"
        a_rep = "PASS" if row["after_repaired"] else "FAIL"
        b_det = "YES" if row["before_detected"] else "NO"
        a_det = "YES" if row["after_detected"] else "NO"
        b_cls = "YES" if row["before_classified"] else "NO"
        a_cls = "YES" if row["after_classified"] else "NO"
        label = _esc(row["function_name"] or row["title"])
        lines.append(
            f"| `{_esc(row['problem_id'])}` ({label}) "
            f"| `{_esc(row['ground_truth_defect'])}` "
            f"| {row['before_passed_tests']} -> {row['after_passed_tests']} "
            f"| {b_rep} -> {a_rep} "
            f"| {b_det} -> {a_det} "
            f"| {b_cls} -> {a_cls} "
            f"| **{row['repair_status']}** |"
        )

    return "\n".join(lines) + "\n"

