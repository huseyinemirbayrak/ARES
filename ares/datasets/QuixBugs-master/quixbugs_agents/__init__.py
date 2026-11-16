"""
QuixBugs Multi-Agent Memory Benchmark Suite.

Provides 4 LangGraph-based agents paired with distinct memory architectures:
1. LangGraph + Mem0: Episodic and continuous lesson memory.
2. LangGraph + Cognee: AST knowledge graph & code ontologies.
3. LangGraph + Zep: Temporal session graphs and fact synthesis.
4. LangGraph + Qdrant: Dense vector similarity & payload filtering.
"""

from .agents import (
    LangGraphMem0Agent,
    LangGraphCogneeAgent,
    LangGraphZepAgent,
    LangGraphQdrantAgent
)
from .dataset_loader import QuixBugsDatasetLoader
from .execution_sandbox import PythonSandbox
from .llm_factory import get_llm, extract_json_from_llm_response

__version__ = "1.0.0"

__all__ = [
    "LangGraphMem0Agent",
    "LangGraphCogneeAgent",
    "LangGraphZepAgent",
    "LangGraphQdrantAgent",
    "QuixBugsDatasetLoader",
    "PythonSandbox",
    "get_llm",
    "extract_json_from_llm_response"
]
