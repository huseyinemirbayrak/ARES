"""Memory package for ARES: Real Cognee SDK Engine, AST/Diff Reflexion, and Dual-Backend Manager."""

from ares.memory.cognee_engine import RealCogneeEngine
from ares.memory.cognee_memory import ARESMemoryManager, CogneeMemoryManager
from ares.memory.reflexion import (
    build_turn_reflexion,
    compute_compact_diff,
    extract_ast_features,
)

__all__ = [
    "ARESMemoryManager",
    "CogneeMemoryManager",
    "RealCogneeEngine",
    "build_turn_reflexion",
    "compute_compact_diff",
    "extract_ast_features",
]
