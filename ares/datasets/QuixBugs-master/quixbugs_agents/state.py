"""Unified State Definition for LangGraph Code Analysis Workflows."""

from typing import TypedDict, Optional, List, Dict, Any


class CodeAnalysisState(TypedDict, total=False):
    """
    TypedDict representing the central execution state across LangGraph nodes.
    Maintains code under review, actor findings, sandbox test outputs,
    critic verification feedback, memory retrieval contexts, and benchmark telemetry.
    """
    # Core Task Attributes
    task_id: str
    task_name: str
    code: str
    language: str
    task_type: str
    algorithm: str

    # Ground Truth Validation Metadata
    ground_truth_is_buggy: Optional[bool]
    ground_truth_fix: Optional[str]
    test_cases: Optional[List[Any]]

    # Cyclic Feedback Loop Management
    iteration: int
    max_iterations: int

    # Multi-Agent Node Outputs
    analyzer_result: Optional[Dict[str, Any]]
    test_result: Optional[Dict[str, Any]]
    verifier_result: Optional[Dict[str, Any]]
    history: List[Dict[str, Any]]

    # Outcome & Evaluation Flags
    status: str
    consensus_reached: bool
    final_is_buggy: Optional[bool]
    final_verdict_matches_ground_truth: Optional[bool]
    round_1_matches_ground_truth: Optional[bool]
    feedback_improved_outcome: bool

    # Framework-Specific Memory Contexts
    mem0_context: Optional[str]
    cognee_context: Optional[str]
    zep_context: Optional[str]
    qdrant_context: Optional[str]

    # Telemetry and Execution Stats
    telemetry: Dict[str, Any]
