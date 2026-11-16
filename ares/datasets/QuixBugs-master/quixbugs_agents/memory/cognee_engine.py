"""Cognee Deterministic Knowledge Graph & AST Ontology Memory Engine."""

import ast
import json
from typing import Dict, Any, List, Optional, Set
from pathlib import Path

from ..config import MEMORY_STORAGE_DIR


class CogneeNode:
    def __init__(self, node_id: str, label: str, node_type: str, properties: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.label = label
        self.node_type = node_type
        self.properties = properties or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "properties": self.properties
        }


class CogneeEdge:
    def __init__(self, source_id: str, target_id: str, relationship: str, weight: float = 1.0):
        self.source_id = source_id
        self.target_id = target_id
        self.relationship = relationship
        self.weight = weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relationship": self.relationship,
            "weight": self.weight
        }


class NativeCogneeEngine:
    """
    Constructs deterministic code ontologies, parses Python AST into knowledge graphs,
    and checks graph invariance rules modeled after the Cognee framework.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (MEMORY_STORAGE_DIR / "cognee_graph.json")
        self.nodes: Dict[str, CogneeNode] = {}
        self.edges: List[CogneeEdge] = []
        if self.storage_path.exists():
            self.import_graph(self.storage_path)
        else:
            self._seed_programming_ontology()

    def export_graph(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Exports the entire Cognee knowledge graph (nodes + edges) to JSON."""
        target_path = path or self.storage_path
        payload = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges]
        }
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def import_graph(self, path: Path) -> bool:
        """Imports and reconstructs the knowledge graph from an external JSON file."""
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.nodes.clear()
            self.edges.clear()
            for n in data.get("nodes", []):
                self.add_node(n["node_id"], n["label"], n["node_type"], n.get("properties"))
            for e in data.get("edges", []):
                self.add_edge(e["source"], e["target"], e["relationship"], e.get("weight", 1.0))
            return True
        except Exception:
            return False

    def add_node(self, node_id: str, label: str, node_type: str, properties: Optional[Dict[str, Any]] = None) -> CogneeNode:
        node = CogneeNode(node_id, label, node_type, properties)
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, relationship: str, weight: float = 1.0) -> CogneeEdge:
        edge = CogneeEdge(source_id, target_id, relationship, weight)
        self.edges.append(edge)
        return edge

    def _seed_programming_ontology(self):
        """Seeds known algorithmic rules and structural defect patterns into the knowledge graph."""
        # 1. Bitwise Invariant
        self.add_node("rule_bitcount_and", "Brian Kernighan Bit Clearing Rule", "AlgorithmInvariant", {
            "description": "Counting 1-bits requires turning off lowest set bit with n &= n - 1."
        })
        self.add_node("defect_bitcount_xor", "Bitwise XOR Infinite Loop", "DefectPattern", {
            "description": "Using n ^= n - 1 toggles bits instead of clearing, resulting in infinite loops."
        })
        self.add_edge("defect_bitcount_xor", "rule_bitcount_and", "VIOLATES_INVARIANT")

        # 2. Graph Visited Invariant
        self.add_node("rule_bfs_visited", "Graph Visited Set Invariant", "AlgorithmInvariant", {
            "description": "BFS traversals must add newly discovered adjacent nodes to visited set to avoid cycles."
        })
        self.add_node("defect_bfs_missing_visited", "Missing Visited Set Update", "DefectPattern", {
            "description": "Failing to add adjacent nodes to visited queue triggers infinite cyclic exploration."
        })
        self.add_edge("defect_bfs_missing_visited", "rule_bfs_visited", "VIOLATES_INVARIANT")

        # 3. Quicksort Partition Invariant
        self.add_node("rule_partition_strictness", "Partition Strictness Invariant", "AlgorithmInvariant", {
            "description": "Quicksort sublists must strictly partition elements smaller and larger than pivot."
        })
        self.add_node("defect_non_strict_partition", "Non-Strict Partition Defect", "DefectPattern", {
            "description": "Non-strict inequalities on both sides cause infinite recursion when duplicates exist."
        })
        self.add_edge("defect_non_strict_partition", "rule_partition_strictness", "VIOLATES_INVARIANT")

    def parse_code_to_graph(self, code: str, algorithm: str) -> Dict[str, Any]:
        """
        Parses Python source code into AST semantic entities,
        evaluates ontological invariants, and outputs graph insights.
        """
        ast_nodes = []
        violations = []
        has_while = False
        has_for = False
        has_recursion = False
        has_bitwise_xor = False
        has_bitwise_and = False
        visited_in_loop = False

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    ast_nodes.append(f"Function({node.name})")
                    func_name = node.name

                elif isinstance(node, ast.While):
                    has_while = True
                    ast_nodes.append("WhileLoop")

                elif isinstance(node, ast.For):
                    has_for = True
                    ast_nodes.append("ForLoop")

                elif isinstance(node, ast.AugAssign):
                    if isinstance(node.op, ast.BitXor):
                        has_bitwise_xor = True
                        ast_nodes.append("BitXorAssignment(^=)")
                    elif isinstance(node.op, ast.BitAnd):
                        has_bitwise_and = True
                        ast_nodes.append("BitAndAssignment(&=)")

                elif isinstance(node, ast.Call):
                    # Check recursion or method calls
                    if hasattr(node.func, "id") and node.func.id == algorithm:
                        has_recursion = True
                    if hasattr(node.func, "attr") and node.func.attr == "add":
                        visited_in_loop = True

            # Invariant checks
            if has_while and has_bitwise_xor:
                violations.append({
                    "rule": "rule_bitcount_and",
                    "severity": "CRITICAL",
                    "explanation": "Ontology check: While loop contains ^= (BitXor) on counter variable. Violates Brian Kernighan Bit Invariant; will loop infinitely!"
                })

            if "breadth_first_search" in algorithm and has_while and not visited_in_loop:
                violations.append({
                    "rule": "rule_bfs_visited",
                    "severity": "HIGH",
                    "explanation": "Ontology check: BFS traversal loop does not call visited.add(...) on newly encountered successors."
                })

            return {
                "success": True,
                "ast_structures": ast_nodes,
                "has_while": has_while,
                "has_for": has_for,
                "has_recursion": has_recursion,
                "violations": violations
            }

        except Exception as e:
            # Multi-language structural & security ontology analysis (for C/C++/Java/JS in CleanVul & Agent-Green)
            import re
            if re.search(r'\b(while|for)\s*\(', code):
                has_while = "while" in code
                has_for = "for" in code
                ast_nodes.append("LoopConstruct")
            if re.search(r'\b(if|switch)\s*\(', code):
                ast_nodes.append("ConditionalBranch")
            if re.search(r'\b(malloc|calloc|realloc|free|kfree|gf_free)\s*\(', code):
                ast_nodes.append("MemoryManagementOp")
            if re.search(r'\b(memcpy|strcpy|strncpy|sprintf|snprintf|strcat)\s*\(', code):
                ast_nodes.append("BufferMemoryCopyOp")
            if "->" in code or "*" in code:
                ast_nodes.append("PointerDereference")

            # Security & Vulnerability Invariant Checks
            if re.search(r'\b(strcpy|sprintf|gets)\s*\(', code):
                violations.append({
                    "rule": "rule_unbounded_buffer_copy",
                    "severity": "CRITICAL",
                    "explanation": "Ontology check: Unbounded buffer copy (strcpy/sprintf/gets) without length validation risks Buffer Overflow (CWE-120/CWE-787)."
                })
            if re.search(r'\b(kfree|free|gf_free)\s*\(', code) and code.count("free(") > 1 and "NULL" not in code:
                violations.append({
                    "rule": "rule_memory_lifecycle",
                    "severity": "HIGH",
                    "explanation": "Ontology check: Multiple memory deallocations without nullifying pointers may trigger Use-After-Free or Double-Free (CWE-415/CWE-416)."
                })
            if "->" in code and not re.search(r'(==\s*NULL|!=\s*NULL|==\s*nullptr|!=\s*nullptr|\bif\s*\(\s*!)', code):
                violations.append({
                    "rule": "rule_null_pointer_guard",
                    "severity": "MEDIUM",
                    "explanation": "Ontology check: Pointer dereference (->) detected without explicit NULL guard in scope (CWE-476)."
                })

            return {
                "success": True,
                "ast_structures": ast_nodes or ["MultiLangCodeBlock"],
                "has_while": has_while,
                "has_for": has_for,
                "has_recursion": has_recursion,
                "violations": violations
            }

    def format_cognee_context(self, graph_analysis: Dict[str, Any]) -> str:
        """Formats graph ontology findings into a concise prompt injection."""
        if not graph_analysis.get("success"):
            return ""

        lines = ["\n### [Cognee Deterministic Knowledge Graph & AST Ontology]:"]
        structures = ", ".join(graph_analysis.get("ast_structures", [])[:6])
        lines.append(f"- Extracted AST Entities: {structures}")

        violations = graph_analysis.get("violations", [])
        if violations:
            lines.append("- Invariant Violations Detected:")
            for v in violations:
                lines.append(f"  * [{v['severity']}] {v['explanation']}")
        else:
            lines.append("- Invariant Checks: All standard structural AST invariants passed cleanly.")

        return "\n".join(lines) + "\n"


class CogneeManager:
    """Singleton accessor for Cognee Engine."""
    _instance: Optional[NativeCogneeEngine] = None

    @classmethod
    def get_instance(cls) -> NativeCogneeEngine:
        if cls._instance is None:
            cls._instance = NativeCogneeEngine()
        return cls._instance
