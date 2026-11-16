"""LangGraph + Mem0 Episodic Memory Agent for QuixBugs Defect Detection."""

import time
from typing import Dict, Any, List, Optional, Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from ..state import CodeAnalysisState
from ..prompts import (
    ANALYZER_SYSTEM_PROMPT, ANALYZER_INITIAL_USER_PROMPT, ANALYZER_REVISION_USER_PROMPT,
    VERIFIER_SYSTEM_PROMPT, VERIFIER_USER_PROMPT
)
from ..llm_factory import get_llm, extract_json_from_llm_response
from ..execution_sandbox import PythonSandbox
from ..memory.mem0_engine import Mem0Manager


class LangGraphMem0Agent:
    """
    Autonomous Bug Detection Agent powered by LangGraph cyclic orchestration
    and Mem0 episodic memory retrieval & continuous lesson learning.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_iterations: int = 2,
        llm=None
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.llm = llm if llm is not None else get_llm(provider=provider, model=model, api_key=api_key)
        self.mem0 = Mem0Manager.get_instance()
        self.graph = self._build_graph()

    def _build_graph(self):
        llm = self.llm
        mem0 = self.mem0

        # 1. MEM0 RECALL NODE
        def mem0_recall_node(state: CodeAnalysisState) -> Dict[str, Any]:
            task_name = state.get("task_name", "")
            algorithm = state.get("algorithm", "")
            code = state.get("code", "")
            query = f"{task_name} {algorithm} bug flaw infinite loop {code[:80]}"
            hits = mem0.search(query=query, limit=3)

            mem0_context = ""
            if hits:
                mem0_context = "\n### [Mem0 Long-Term Episodic Memory & Past Lessons]:\n"
                for h in hits:
                    mem0_context += f"- {h.get('text')}\n"

            telemetry = dict(state.get("telemetry") or {})
            telemetry["mem0_hits_count"] = len(hits)
            return {"mem0_context": mem0_context, "telemetry": telemetry}

        # 2. ANALYZER NODE
        def analyzer_node(state: CodeAnalysisState) -> Dict[str, Any]:
            iteration = state.get("iteration", 0)
            code = state["code"]
            algorithm = state.get("algorithm", "algorithm")
            task_name = state.get("task_name", algorithm)
            history = list(state.get("history", []))
            mem0_ctx = state.get("mem0_context", "")

            if iteration == 0:
                prompt_content = ANALYZER_INITIAL_USER_PROMPT.format(
                    algorithm=algorithm,
                    task_name=task_name,
                    code=code,
                    memory_context=mem0_ctx
                )
            else:
                prev_analysis = state.get("analyzer_result", {})
                verifier_result = state.get("verifier_result", {})
                test_result = state.get("test_result") or {}

                test_feedback = ""
                if test_result.get("run"):
                    test_feedback = f"Sandbox Test Output: {test_result.get('details', '')}"

                prompt_content = ANALYZER_REVISION_USER_PROMPT.format(
                    algorithm=algorithm,
                    code=code,
                    prev_has_error=prev_analysis.get("has_error", True),
                    prev_error_type=prev_analysis.get("error_type", "Unknown"),
                    prev_explanation=prev_analysis.get("explanation", ""),
                    verifier_feedback=verifier_result.get("feedback", ""),
                    test_feedback=test_feedback
                )

            messages = [
                SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content)
            ]

            response = llm.invoke(messages)
            parsed = extract_json_from_llm_response(response.content)

            has_error = parsed.get("has_error", True) if isinstance(parsed.get("has_error"), bool) else True
            error_type = parsed.get("error_type") or "Detected Defect"
            location = parsed.get("location") or "N/A"
            explanation = parsed.get("explanation") or parsed.get("raw_text") or "No detailed explanation."
            proposed_fix = parsed.get("proposed_fix") or code
            confidence = parsed.get("confidence") or 0.85

            analyzer_output = {
                "has_error": has_error,
                "error_type": error_type,
                "location": location,
                "explanation": explanation,
                "proposed_fix": proposed_fix,
                "confidence": confidence
            }

            round_1_match = state.get("round_1_matches_ground_truth")
            if iteration == 0 and state.get("ground_truth_is_buggy") is not None:
                round_1_match = (analyzer_output["has_error"] == state["ground_truth_is_buggy"])

            history.append({
                "round": iteration + 1,
                "agent": "LangGraph+Mem0-Analyzer",
                "action": "Initial Diagnosis (Mem0 Augmented)" if iteration == 0 else "Revision via Critic Feedback",
                "data": analyzer_output
            })

            return {
                "analyzer_result": analyzer_output,
                "history": history,
                "round_1_matches_ground_truth": round_1_match
            }

        # 3. TESTER NODE
        def tester_node(state: CodeAnalysisState) -> Dict[str, Any]:
            algorithm = state.get("algorithm", "")
            code = state.get("code", "")
            test_cases = state.get("test_cases")

            test_run = PythonSandbox.run_quixbugs_tests(code=code, algorithm=algorithm, test_cases=test_cases)
            return {"test_result": test_run}

        # 4. VERIFIER NODE
        def verifier_node(state: CodeAnalysisState) -> Dict[str, Any]:
            iteration = state.get("iteration", 0)
            code = state["code"]
            algorithm = state.get("algorithm", "algorithm")
            analyzer_result = state.get("analyzer_result", {})
            test_result = state.get("test_result") or {}
            history = list(state.get("history", []))

            sandbox_output = test_result.get("details", "No unit test execution output.")

            prompt_content = VERIFIER_USER_PROMPT.format(
                algorithm=algorithm,
                code=code,
                analyzer_has_error=analyzer_result.get("has_error", True),
                analyzer_error_type=analyzer_result.get("error_type", "Unknown"),
                analyzer_location=analyzer_result.get("location", "N/A"),
                analyzer_explanation=analyzer_result.get("explanation", ""),
                proposed_fix=analyzer_result.get("proposed_fix", ""),
                sandbox_output=sandbox_output
            )

            messages = [
                SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content)
            ]

            response = llm.invoke(messages)
            parsed = extract_json_from_llm_response(response.content)

            is_approved = parsed.get("is_approved", True) if isinstance(parsed.get("is_approved"), bool) else True
            verifier_output = {
                "is_approved": is_approved,
                "critique": parsed.get("critique", "Audit finished"),
                "feedback": parsed.get("feedback", "No revisions requested"),
                "confidence": parsed.get("confidence", 0.9)
            }

            history.append({
                "round": iteration + 1,
                "agent": "LangGraph+Mem0-Verifier",
                "action": "Audit Verdict",
                "data": verifier_output
            })

            return {
                "verifier_result": verifier_output,
                "history": history,
                "iteration": iteration + 1
            }

        # 5. CONDITIONAL ROUTER
        def should_continue(state: CodeAnalysisState) -> Literal["analyzer_node", "mem0_commit_node"]:
            verifier_res = state.get("verifier_result", {})
            iteration = state.get("iteration", 0)
            max_iter = state.get("max_iterations", self.max_iterations)

            if verifier_res.get("is_approved", False) or iteration >= max_iter:
                return "mem0_commit_node"
            return "analyzer_node"

        # 6. MEM0 COMMIT NODE
        def mem0_commit_node(state: CodeAnalysisState) -> Dict[str, Any]:
            analyzer_res = state.get("analyzer_result", {})
            verifier_res = state.get("verifier_result", {})
            algorithm = state.get("algorithm", "")
            has_error = analyzer_res.get("has_error", True)
            is_approved = verifier_res.get("is_approved", False)

            expl = analyzer_res.get("explanation") or ""
            err_type = analyzer_res.get("error_type") or "Detected Defect"
            loc = analyzer_res.get("location") or "N/A"
            if has_error:
                lesson_text = f"Experience on {algorithm}: Identified defect '{err_type}' at {loc}. Fix logic: {expl[:150]}"
            else:
                lesson_text = f"Experience on {algorithm}: Verified clean code. Boundary conditions and logic confirmed sound."

            saved = mem0.add(
                text=lesson_text,
                user_id="evaluator",
                agent_id="langgraph_mem0",
                metadata={"algorithm": algorithm, "is_buggy": has_error, "approved": is_approved}
            )

            telemetry = dict(state.get("telemetry") or {})
            telemetry["mem0_saved"] = saved.get("id")
            return {"telemetry": telemetry}

        # 7. CONSENSUS NODE
        def consensus_node(state: CodeAnalysisState) -> Dict[str, Any]:
            verifier_res = state.get("verifier_result", {})
            analyzer_res = state.get("analyzer_result", {})
            iteration = state.get("iteration", 0)
            max_iter = state.get("max_iterations", self.max_iterations)
            is_approved = verifier_res.get("is_approved", False)
            final_is_buggy = analyzer_res.get("has_error", True)
            gt_buggy = state.get("ground_truth_is_buggy")
            r1_match = state.get("round_1_matches_ground_truth")

            status = "APPROVED" if is_approved else ("MAX_ROUNDS_REACHED" if iteration >= max_iter else "COMPLETED")

            final_match = None
            feedback_improved = False
            if gt_buggy is not None:
                final_match = (final_is_buggy == gt_buggy)
                if r1_match is not None and (not r1_match) and final_match:
                    feedback_improved = True

            return {
                "status": status,
                "consensus_reached": is_approved,
                "final_is_buggy": final_is_buggy,
                "final_verdict_matches_ground_truth": final_match,
                "feedback_improved_outcome": feedback_improved
            }

        # Build Graph
        workflow = StateGraph(CodeAnalysisState)
        workflow.add_node("mem0_recall_node", mem0_recall_node)
        workflow.add_node("analyzer_node", analyzer_node)
        workflow.add_node("tester_node", tester_node)
        workflow.add_node("verifier_node", verifier_node)
        workflow.add_node("mem0_commit_node", mem0_commit_node)
        workflow.add_node("consensus_node", consensus_node)

        workflow.set_entry_point("mem0_recall_node")
        workflow.add_edge("mem0_recall_node", "analyzer_node")
        workflow.add_edge("analyzer_node", "tester_node")
        workflow.add_edge("tester_node", "verifier_node")
        workflow.add_conditional_edges("verifier_node", should_continue)
        workflow.add_edge("mem0_commit_node", "consensus_node")
        workflow.add_edge("consensus_node", END)

        return workflow.compile()

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the LangGraph+Mem0 multi-agent workflow on a single QuixBugs sample."""
        start_time = time.time()
        initial_state: CodeAnalysisState = {
            "task_id": sample.get("task_id", "sample"),
            "task_name": sample.get("task_name", "QuixBugs Task"),
            "algorithm": sample.get("algorithm", "algorithm"),
            "code": sample.get("code", ""),
            "language": sample.get("language", "python"),
            "task_type": sample.get("task_type", "bug_detection"),
            "ground_truth_is_buggy": sample.get("ground_truth_is_buggy"),
            "ground_truth_fix": sample.get("ground_truth_fix"),
            "test_cases": sample.get("test_cases"),
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "analyzer_result": None,
            "test_result": None,
            "verifier_result": None,
            "history": [],
            "status": "RUNNING",
            "consensus_reached": False,
            "final_is_buggy": None,
            "final_verdict_matches_ground_truth": None,
            "round_1_matches_ground_truth": None,
            "feedback_improved_outcome": False,
            "mem0_context": None,
            "telemetry": {}
        }

        final_state = self.graph.invoke(initial_state)
        elapsed = time.time() - start_time

        return {
            "framework": "langgraph_mem0",
            "task_id": final_state["task_id"],
            "task_name": final_state["task_name"],
            "algorithm": final_state["algorithm"],
            "dataset": sample.get("dataset", "quixbugs"),
            "ground_truth_is_buggy": final_state["ground_truth_is_buggy"],
            "round_1_matches_gt": final_state["round_1_matches_ground_truth"],
            "final_matches_gt": final_state["final_verdict_matches_ground_truth"],
            "feedback_improved": final_state["feedback_improved_outcome"],
            "final_is_buggy": final_state["final_is_buggy"],
            "consensus_reached": final_state["consensus_reached"],
            "total_rounds": final_state["iteration"],
            "elapsed_sec": round(elapsed, 2),
            "status": final_state["status"],
            "history": final_state["history"],
            "analyzer_result": final_state["analyzer_result"],
            "verifier_result": final_state["verifier_result"],
            "telemetry": final_state.get("telemetry", {})
        }
