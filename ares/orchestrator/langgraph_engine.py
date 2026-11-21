"""Real LangGraph StateGraph Orchestrator for ARES Adversarial Self-Play.

Uses genuine `langgraph.graph.StateGraph` + `MemorySaver` checkpointing to coordinate:
  select_problem_and_memory -> attacker_turn -> defender_turn
  -> referee_evaluation -> generate_reflexion -> update_memory
  -> (periodic_external_exam | next_turn | END)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ares.agents.attacker import AttackerAgent
from ares.agents.defender import DefenderAgent
from ares.datasets.loader import load_problems
from ares.evaluation.exam_evaluator import (
    ExternalExamEvaluator,
    build_pre_post_comparison,
    render_pre_post_markdown,
)
from ares.memory.cognee_memory import CogneeMemoryManager
from ares.memory.reflexion import build_turn_reflexion
from ares.models import (
    ARESGraphState,
    AttackerOutput,
    CodeProblem,
    DefenderOutput,
    ExamResult,
    RefereeVerdict,
    SuiteExecutionResult,
    TurnRecord,
    TurnReflexion,
)
from ares.referee.deterministic_referee import DeterministicReferee


class ARESLangGraphOrchestrator:
    """Real LangGraph StateGraph + MemorySaver coordinator for ARES Self-Play."""

    def __init__(
        self,
        problems: list[CodeProblem] | None = None,
        exam_problems: list[CodeProblem] | None = None,
        attacker: AttackerAgent | None = None,
        defender: DefenderAgent | None = None,
        referee: DeterministicReferee | None = None,
        memory: CogneeMemoryManager | None = None,
        output_dir: str | Path = "outputs",
        on_turn_complete: Callable[[TurnRecord], None] | None = None,
        on_exam_complete: Callable[[ExamResult], None] | None = None,
        thread_id: str | None = None,
        initial_elo: float = 1200.0,
    ) -> None:
        self.problems = problems if problems is not None else load_problems(exam=False)
        self.exam_problems = exam_problems if exam_problems is not None else load_problems(exam=True)
        self.attacker = attacker or AttackerAgent()
        self.defender = defender or DefenderAgent()
        self.referee = referee or DeterministicReferee()
        self.memory = memory or CogneeMemoryManager()
        self.initial_elo = float(initial_elo)
        self.exam_evaluator = ExternalExamEvaluator(
            exam_problems=self.exam_problems, referee=self.referee
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_turn_complete = on_turn_complete
        self.on_exam_complete = on_exam_complete
        self.thread_id = thread_id or f"ares-run-{uuid.uuid4().hex[:8]}"
        self.checkpointer = MemorySaver()
        self._turn_records: list[dict[str, Any]] = []
        self._exam_records: list[dict[str, Any]] = []

        self.graph = self._build_langgraph()

    def _build_langgraph(self):
        """Construct and compile the real LangGraph StateGraph workflow with MemorySaver."""
        workflow = StateGraph(ARESGraphState)

        workflow.add_node("select_problem_and_memory", self._node_select_problem_and_memory)
        workflow.add_node("attacker_turn", self._node_attacker_turn)
        workflow.add_node("defender_turn", self._node_defender_turn)
        workflow.add_node("referee_evaluation", self._node_referee_evaluation)
        workflow.add_node("generate_reflexion", self._node_generate_reflexion)
        workflow.add_node("update_memory", self._node_update_memory)
        workflow.add_node("periodic_external_exam", self._node_periodic_external_exam)

        workflow.add_edge(START, "select_problem_and_memory")
        workflow.add_edge("select_problem_and_memory", "attacker_turn")
        workflow.add_edge("attacker_turn", "defender_turn")
        workflow.add_edge("defender_turn", "referee_evaluation")
        workflow.add_edge("referee_evaluation", "generate_reflexion")
        workflow.add_edge("generate_reflexion", "update_memory")

        workflow.add_conditional_edges(
            "update_memory",
            self._route_after_memory_update,
            {
                "run_exam": "periodic_external_exam",
                "next_turn": "select_problem_and_memory",
                "finish": END,
            },
        )

        workflow.add_conditional_edges(
            "periodic_external_exam",
            self._route_after_exam,
            {
                "next_turn": "select_problem_and_memory",
                "finish": END,
            },
        )

        return workflow.compile(checkpointer=self.checkpointer)

    def _node_select_problem_and_memory(
        self, state: ARESGraphState
    ) -> dict[str, Any]:
        next_turn = state.get("turn_id", 0) + 1
        problem = self.problems[(next_turn - 1) % len(self.problems)]
        att_mem = self.memory.retrieve_attacker_memories(problem)
        return {
            "turn_id": next_turn,
            "current_problem": problem.model_dump(),
            "attacker_memories": att_mem,
        }

    def _node_attacker_turn(self, state: ARESGraphState) -> dict[str, Any]:
        turn_id = state["turn_id"]
        problem = CodeProblem.model_validate(state["current_problem"])
        att_memories = state.get("attacker_memories", [])
        att_output = self.attacker.attack(
            problem, self.memory, turn_id=turn_id, memories=att_memories
        )
        att_suite = self.referee.execute_test_suite(
            att_output.mutated_code, problem.test_cases
        )
        return {
            "attacker_output": att_output.model_dump(),
            "attacker_suite_result": att_suite.model_dump(),
        }

    def _node_defender_turn(self, state: ARESGraphState) -> dict[str, Any]:
        turn_id = state["turn_id"]
        problem = CodeProblem.model_validate(state["current_problem"])
        att_output = AttackerOutput.model_validate(state["attacker_output"])
        att_suite = SuiteExecutionResult.model_validate(state["attacker_suite_result"])
        hide_results = self.defender.blind_mode or self.defender.hide_test_results
        failed_ids = [] if hide_results else att_suite.failed_test_ids
        def_mem = self.memory.retrieve_defender_memories(
            problem, att_output.mutated_code, failed_ids
        )
        def_output = self.defender.defend(
            problem=problem,
            mutated_code=att_output.mutated_code,
            attacker_suite=None if self.defender.hide_test_results else att_suite,
            memory=self.memory,
            turn_id=turn_id,
            memories=def_mem,
        )
        return {
            "defender_memories": def_mem,
            "defender_output": def_output.model_dump(),
        }

    def _node_referee_evaluation(self, state: ARESGraphState) -> dict[str, Any]:
        turn_id = state["turn_id"]
        problem = CodeProblem.model_validate(state["current_problem"])
        att_output = AttackerOutput.model_validate(state["attacker_output"])
        att_suite = SuiteExecutionResult.model_validate(state["attacker_suite_result"])
        def_output = DefenderOutput.model_validate(state["defender_output"])

        verdict = self.referee.evaluate_turn(
            turn_id=turn_id,
            problem=problem,
            attacker_output=att_output,
            defender_output=def_output,
            attacker_suite=att_suite,
        )
        new_att_elo, new_def_elo = self.referee.update_elo(
            state.get("attacker_elo", self.initial_elo),
            state.get("defender_elo", self.initial_elo),
            verdict.winner,
        )

        att_wins = state.get("attacker_wins", 0) + (1 if verdict.winner == "attacker" else 0)
        def_wins = state.get("defender_wins", 0) + (1 if verdict.winner == "defender" else 0)
        draws = state.get("draws", 0) + (1 if verdict.winner == "draw" else 0)

        return {
            "verdict": verdict.model_dump(),
            "attacker_elo": new_att_elo,
            "defender_elo": new_def_elo,
            "attacker_wins": att_wins,
            "defender_wins": def_wins,
            "draws": draws,
        }

    def _node_generate_reflexion(self, state: ARESGraphState) -> dict[str, Any]:
        """Dedicated LangGraph node synthesizing diff + AST Reflexion lessons for both agents."""
        turn_id = state["turn_id"]
        problem = CodeProblem.model_validate(state["current_problem"])
        att_output = AttackerOutput.model_validate(state["attacker_output"])
        def_output = DefenderOutput.model_validate(state["defender_output"])
        verdict = RefereeVerdict.model_validate(state["verdict"])

        hide_results = self.defender.hide_test_results or self.attacker.hide_test_results
        reflexion = build_turn_reflexion(
            turn_id=turn_id,
            problem=problem,
            attacker_output=att_output,
            defender_output=def_output,
            verdict=verdict,
            blind_attacker=self.attacker.blind_mode,
            blind_defender=self.defender.blind_mode,
            hide_test_results=hide_results,
        )
        return {"reflexion": reflexion.model_dump()}

    def _node_update_memory(self, state: ARESGraphState) -> dict[str, Any]:
        turn_id = state["turn_id"]
        problem = CodeProblem.model_validate(state["current_problem"])
        att_output = AttackerOutput.model_validate(state["attacker_output"])
        def_output = DefenderOutput.model_validate(state["defender_output"])
        verdict = RefereeVerdict.model_validate(state["verdict"])
        reflexion = (
            TurnReflexion.model_validate(state["reflexion"])
            if state.get("reflexion")
            else None
        )

        hide_results = self.defender.hide_test_results or self.attacker.hide_test_results
        mem_ids = self.memory.record_turn_memory(
            turn_id=turn_id,
            problem=problem,
            attacker_output=att_output,
            defender_output=def_output,
            verdict=verdict,
            reflexion=reflexion,
            blind_attacker=self.attacker.blind_mode,
            blind_defender=self.defender.blind_mode,
            hide_test_results=hide_results,
        )

        record = TurnRecord(
            turn_id=turn_id,
            problem_id=problem.problem_id,
            problem_title=problem.title,
            mlcq_category=problem.mlcq_category,
            attacker_output=att_output,
            defender_output=def_output,
            verdict=verdict,
            attacker_elo_after=state["attacker_elo"],
            defender_elo_after=state["defender_elo"],
            cognee_memory_ids=mem_ids,
        )

        if self.on_turn_complete:
            self.on_turn_complete(record)

        # Store full turn logs outside LangGraph state to avoid O(N^2) MemorySaver snapshot bloat
        self._turn_records.append(record.model_dump())
        return {}

    def _node_periodic_external_exam(self, state: ARESGraphState) -> dict[str, Any]:
        # Flush any staged real Cognee lessons before running the external benchmark exam
        self.memory.flush_cognee()
        turn_id = state["turn_id"]
        exam_res = self.exam_evaluator.run_exam(
            after_turn=turn_id,
            attacker=self.attacker,
            defender=self.defender,
            memory=self.memory,
            attacker_elo=state.get("attacker_elo", self.initial_elo),
            defender_elo=state.get("defender_elo", self.initial_elo),
        )
        if self.on_exam_complete:
            self.on_exam_complete(exam_res)

        self._exam_records.append(exam_res.model_dump())
        return {}

    def _route_after_memory_update(self, state: ARESGraphState) -> str:
        turn_id = state.get("turn_id", 0)
        max_turns = state.get("max_turns", 25)
        exam_interval = state.get("exam_interval", 25)

        if exam_interval > 0 and (turn_id % exam_interval == 0 or turn_id == max_turns):
            return "run_exam"
        if turn_id < max_turns:
            return "next_turn"
        return "finish"

    def _route_after_exam(self, state: ARESGraphState) -> str:
        turn_id = state.get("turn_id", 0)
        max_turns = state.get("max_turns", 25)
        if turn_id < max_turns:
            return "next_turn"
        return "finish"

    def get_graph_checkpoint_state(self) -> Any:
        """Inspect the latest LangGraph MemorySaver checkpoint for this thread."""
        return self.graph.get_state({"configurable": {"thread_id": self.thread_id}})

    def run_self_play(
        self,
        num_turns: int = 25,
        exam_interval: int = 25,
        run_baseline_exam: bool = True,
    ) -> dict[str, Any]:
        """Execute the full LangGraph self-play loop and persist logs, exams, and scoreboard."""
        if num_turns < 0:
            raise ValueError(f"num_turns must be >= 0, got {num_turns}")
        if num_turns > 0 and not self.problems:
            raise ValueError("Cannot run self-play with an empty arena problem list.")

        self._turn_records = []
        self._exam_records = []

        if run_baseline_exam:
            baseline_exam = self.exam_evaluator.run_exam(
                after_turn=0,
                attacker=self.attacker,
                defender=self.defender,
                memory=self.memory,
                attacker_elo=self.initial_elo,
                defender_elo=self.initial_elo,
            )
            if self.on_exam_complete:
                self.on_exam_complete(baseline_exam)
            self._exam_records.append(baseline_exam.model_dump())

        initial_state: ARESGraphState = {
            "turn_id": 0,
            "max_turns": num_turns,
            "exam_interval": exam_interval,
            "attacker_elo": self.initial_elo,
            "defender_elo": self.initial_elo,
            "attacker_wins": 0,
            "defender_wins": 0,
            "draws": 0,
        }

        if num_turns > 0:
            recursion_limit = max(500, num_turns * 14)
            final_state = self.graph.invoke(
                initial_state,
                config={
                    "configurable": {"thread_id": self.thread_id},
                    "recursion_limit": recursion_limit,
                },
            )
        else:
            final_state = initial_state
        self.memory.flush_cognee()

        summary = self._persist_outputs(final_state)
        return summary

    def _persist_outputs(self, final_state: ARESGraphState) -> dict[str, Any]:
        """Write JSONL turn logs, external exam comparisons, and honest scoreboard summary."""
        turns = list(self._turn_records)
        exams = list(self._exam_records)

        turns_path = self.output_dir / "self_play_turns.jsonl"
        with turns_path.open("w", encoding="utf-8") as f:
            for t in turns:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        exams_path = self.output_dir / "external_exams.json"
        exams_path.write_text(
            json.dumps(exams, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        total_turns = len(turns)
        valid_attacks = sum(
            1
            for t in turns
            if t["verdict"]["attack_validity"] == "valid_subtle_attack"
        )
        full_repairs = sum(
            1 for t in turns if t["verdict"]["defense_repaired_all_tests"]
        )
        correct_classifications = sum(
            1 for t in turns if t["verdict"]["classification_correct"]
        )
        attacker_llm_turns = sum(1 for t in turns if t["attacker_output"].get("used_llm"))
        defender_llm_turns = sum(1 for t in turns if t["defender_output"].get("used_llm"))

        exam_improvement: dict[str, Any] = {}
        pre_post_comparison: dict[str, Any] = {}
        pre_post_report_paths: dict[str, str] = {}
        if len(exams) >= 2:
            first_ex = exams[0]
            last_ex = exams[-1]
            exam_improvement = {
                "baseline_turn": first_ex["after_turn"],
                "final_turn": last_ex["after_turn"],
                "attacker_valid_subtle_rate_delta": round(
                    last_ex["attacker_valid_subtle_rate"]
                    - first_ex["attacker_valid_subtle_rate"],
                    2,
                ),
                "defender_full_repair_rate_delta": round(
                    last_ex["defender_full_repair_rate"]
                    - first_ex["defender_full_repair_rate"],
                    2,
                ),
                "defender_bug_detection_rate_delta": round(
                    last_ex.get("defender_bug_detection_rate", 0.0)
                    - first_ex.get("defender_bug_detection_rate", 0.0),
                    2,
                ),
                "defender_avg_test_pass_rate_delta": round(
                    last_ex.get("defender_avg_test_pass_rate", 0.0)
                    - first_ex.get("defender_avg_test_pass_rate", 0.0),
                    2,
                ),
                "defender_classification_accuracy_delta": round(
                    last_ex["defender_classification_accuracy"]
                    - first_ex["defender_classification_accuracy"],
                    2,
                ),
                "defender_combined_score_delta": round(
                    last_ex["defender_combined_score"]
                    - first_ex["defender_combined_score"],
                    2,
                ),
                "cognee_knowledge_nodes_added": (
                    last_ex["cognee_knowledge_nodes"]
                    - first_ex["cognee_knowledge_nodes"]
                ),
            }
            pre_post_comparison = build_pre_post_comparison(first_ex, last_ex)
            pre_post_json_path = self.output_dir / "pre_post_comparison.json"
            pre_post_md_path = self.output_dir / "pre_post_comparison.md"
            pre_post_json_path.write_text(
                json.dumps(pre_post_comparison, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            pre_post_md_path.write_text(
                render_pre_post_markdown(pre_post_comparison),
                encoding="utf-8",
            )
            pre_post_report_paths = {
                "json": str(pre_post_json_path),
                "markdown": str(pre_post_md_path),
            }

        memory_meta = {
            "active_backend": self.memory.active_backend,
            "backend_status_reason": self.memory.backend_status_reason,
            "cognee_ingested_turns": self.memory.cognee_ingested_count,
            "cognee_search_calls": self.memory.cognee_search_count,
            "cognee_errors": list(self.memory.cognee_engine.errors[-10:]),
            "bundle_path": str(self.memory.bundle_path),
            "sqlite_path": str(self.memory.sqlite_path),
            "total_nodes": len(self.memory.nodes),
            "total_edges": len(self.memory.edges),
            "total_episodes": len(self.memory.episodes),
        }

        scoreboard = {
            "total_turns": total_turns,
            "attacker_wins": final_state.get("attacker_wins", 0),
            "defender_wins": final_state.get("defender_wins", 0),
            "draws": final_state.get("draws", 0),
            "attacker_final_elo": final_state.get("attacker_elo", self.initial_elo),
            "defender_final_elo": final_state.get("defender_elo", self.initial_elo),
            "langgraph_orchestrator": {
                "framework": "langgraph.StateGraph",
                "checkpointer": "MemorySaver",
                "thread_id": self.thread_id,
                "nodes": [
                    "select_problem_and_memory",
                    "attacker_turn",
                    "defender_turn",
                    "referee_evaluation",
                    "generate_reflexion",
                    "update_memory",
                    "periodic_external_exam",
                ],
            },
            "llm_usage_metrics": {
                "attacker_llm_turns": attacker_llm_turns,
                "defender_llm_turns": defender_llm_turns,
                "attacker_llm_rate_pct": round(
                    (attacker_llm_turns / max(1, total_turns)) * 100.0, 2
                ),
                "defender_llm_rate_pct": round(
                    (defender_llm_turns / max(1, total_turns)) * 100.0, 2
                ),
            },
            "self_play_metrics": {
                "valid_subtle_attack_rate_pct": round(
                    (valid_attacks / max(1, total_turns)) * 100.0, 2
                ),
                "defender_full_repair_rate_pct": round(
                    (full_repairs / max(1, total_turns)) * 100.0, 2
                ),
                "defender_classification_accuracy_pct": round(
                    (correct_classifications / max(1, total_turns)) * 100.0, 2
                ),
            },
            "memory_system": memory_meta,
            "cognee_portable_memory": memory_meta,
            "external_exam_improvement": exam_improvement,
            "pre_post_comparison": pre_post_comparison,
            "pre_post_report_paths": pre_post_report_paths,
            "external_exams": exams,
        }

        scoreboard_path = self.output_dir / "scoreboard_summary.json"
        scoreboard_path.write_text(
            json.dumps(scoreboard, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return scoreboard
