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
        details: list[dict[str, Any]] = []

        for idx, problem in enumerate(self.exam_problems, start=1):
            # Part A: Evaluate Attacker's ability to craft a 1-2 test subtle defect on unseen code
            att_out = attacker.attack(problem, memory, turn_id=after_turn + idx)
            att_suite = self.referee.execute_test_suite(
                att_out.mutated_code, problem.test_cases
            )
            is_valid_subtle = (
                att_suite.syntax_valid
                and not att_suite.timed_out
                and self.referee.min_broken_tests
                <= att_suite.failed_count
                <= self.referee.max_broken_tests
            )
            if is_valid_subtle:
                valid_attacks += 1
            total_broken_by_attacker += att_suite.failed_count

            # Part B: Evaluate Defender's ability to classify & repair the canonical subtle defect
            if problem.candidate_mutations:
                canonical_mut = problem.candidate_mutations[0]
            else:
                from ares.agents.ast_mutator import ASTSurgicalMutator
                canonical_mut = ASTSurgicalMutator.generate_candidates(
                    problem.clean_code, problem.function_name
                )[0]
            canonical_suite = self.referee.execute_test_suite(
                canonical_mut.mutated_code, problem.test_cases
            )
            def_out = defender.defend(
                problem=problem,
                mutated_code=canonical_mut.mutated_code,
                attacker_suite=canonical_suite,
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
            classified_ok = def_out.predicted_defect_type == canonical_mut.defect_type

            if repaired_all:
                full_repairs += 1
            if classified_ok:
                correct_classifications += 1

            details.append(
                {
                    "problem_id": problem.problem_id,
                    "title": problem.title,
                    "attacker_broken_tests": att_suite.failed_count,
                    "attacker_valid_subtle": is_valid_subtle,
                    "ground_truth_defect": canonical_mut.defect_type,
                    "defender_predicted_defect": def_out.predicted_defect_type,
                    "defender_classification_correct": classified_ok,
                    "defender_repaired_all_5_tests": repaired_all,
                    "defender_passed_tests": f"{def_suite.passed_count}/{def_suite.total_tests}",
                    "defense_strategy": def_out.defense_strategy,
                }
            )

        n = max(1, len(self.exam_problems))
        att_rate = round((valid_attacks / n) * 100.0, 2)
        avg_broken = round(total_broken_by_attacker / n, 2)
        def_repair_rate = round((full_repairs / n) * 100.0, 2)
        def_class_acc = round((correct_classifications / n) * 100.0, 2)
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
            attacker_elo=attacker_elo,
            defender_elo=defender_elo,
            cognee_knowledge_nodes=len(memory.nodes),
            per_problem_details=details,
        )

