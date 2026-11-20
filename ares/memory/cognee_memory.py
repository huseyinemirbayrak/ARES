"""Dual-Backend Memory Manager for ARES: Real Cognee SDK + Honest Local Episodic Graph Store.

Architecture:
- `backend="cognee"`: Uses the real `cognee` Python SDK (`cognee.add`, `cognee.cognify`,
  `cognee.search`, `cognee.prune`) wired to Ollama / OpenAI-compatible LLM & embeddings.
  Raises RuntimeError if `cognee` or the embedding/LLM service is unavailable (never fakes Cognee).
- `backend="local_graph"`: Honest local SQLite + JSON Knowledge Graph using AST structural
  feature similarity and unified code-diff reflexions (zero external API calls).
- `backend="auto"`: Uses real Cognee when installed and reachable; otherwise falls back to
  `local_graph` and explicitly reports `active_backend = "local_graph"`.

Key Scientific Integrity Guarantees:
- NEVER leaks ground-truth `problem.clean_code` into Defender memory on failed repairs.
- Stores concrete unified code diffs (`before -> after`), AST structural features, and
  actionable Reflexion lessons so real LLMs genuinely improve across turns.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Literal

from ares.memory.cognee_engine import RealCogneeEngine
from ares.memory.reflexion import build_turn_reflexion, extract_ast_features
from ares.models import (
    AttackerOutput,
    CodeProblem,
    DefenderOutput,
    RefereeVerdict,
    TurnReflexion,
)

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    """Extract normalized alphanumeric and operator tokens from text/code."""
    words = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|<=|>=|==|!=", text.lower()))
    return {w for w in words if len(w) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class CogneeMemoryManager:
    """Memory Manager supporting genuine Cognee SDK (`cognee`) and honest `local_graph` fallback."""

    def __init__(
        self,
        store_dir: str | Path = "memory_store",
        bundle_filename: str = "cognee_portable_graph.json",
        backend: Literal["auto", "cognee", "local_graph"] = "auto",
        enable_native_cognee: bool = True,
        in_context_window: int = 6,
        llm_provider: str = "ollama",
        llm_model: str = "qwen2.5-coder:7b",
        llm_endpoint: str = "http://localhost:11434",
        llm_api_key: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str = "nomic-embed-text",
        embedding_endpoint: str | None = None,
        embedding_dimensions: int = 768,
        cognee_search_type: str = "CHUNKS",
        cognify_interval: int = 1,
        top_k_memories: int = 4,
        auto_export: bool = True,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_path = self.store_dir / bundle_filename
        self.sqlite_path = self.store_dir / "ares_episodic_graph.sqlite3"
        self.in_context_window = in_context_window
        self.top_k_memories = max(1, int(top_k_memories))
        self.auto_export = bool(auto_export)
        self.cognify_interval = max(1, cognify_interval)
        self._pending_cognee_turns = 0
        self.requested_backend = backend if enable_native_cognee else "local_graph"

        resolved_embed_provider = embedding_provider or llm_provider
        self.cognee_engine = RealCogneeEngine(
            store_dir=self.store_dir,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key or "ollama",
            embedding_provider=resolved_embed_provider,
            embedding_model=embedding_model,
            embedding_endpoint=embedding_endpoint,
            embedding_dimensions=embedding_dimensions,
            search_type=cognee_search_type,
        )

        # Honest backend resolution — NEVER silently claims Cognee when unavailable
        self.active_backend: Literal["cognee", "local_graph"] = "local_graph"
        self.backend_status_reason: str = ""
        self.cognee_available: bool = False
        self.cognee_ingested_count: int = 0
        self.cognee_search_count: int = 0
        self._has_persisted_cognee_graph: bool = (
            self.cognee_engine.system_root.exists()
            and any(self.cognee_engine.system_root.rglob("*"))
        )
        self.loaded_preexisting_episodes: int = 0
        self.migrated_legacy_schema: str | None = None

        if self.requested_backend == "local_graph":
            self.active_backend = "local_graph"
            self.backend_status_reason = "Explicitly configured local SQLite + AST/Diff episodic graph store."
        else:
            ready, reason = self.cognee_engine.check_readiness()
            if ready:
                self.active_backend = "cognee"
                self.cognee_available = True
                self.backend_status_reason = reason
            elif self.requested_backend == "cognee":
                raise RuntimeError(
                    f"Real Cognee backend was explicitly requested (--memory-backend cognee), "
                    f"but could not be initialized: {reason}"
                )
            else:
                self.active_backend = "local_graph"
                self.backend_status_reason = f"Real Cognee unavailable ({reason}); using honest local_graph backend."
                logger.info("%s", self.backend_status_reason)

        # Structured Graph & Episodic Memory (also synced to SQLite + JSON for portability)
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []

        self.attacker_strategy_stats: dict[str, dict[str, Any]] = {}
        self.defender_repair_knowledge: dict[str, dict[str, Any]] = {}

        self.attacker_in_context: list[str] = []
        self.defender_in_context: list[str] = []

        self._init_sqlite()
        if self.bundle_path.exists():
            self.import_portable_bundle(self.bundle_path, mode="replace")
            self.loaded_preexisting_episodes = len(self.episodes)

    def _init_sqlite(self) -> None:
        """Initialize SQLite backing store for graph nodes, edges, and diff-rich episodes."""
        with contextlib.closing(sqlite3.connect(self.sqlite_path)) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kg_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    turn_id INTEGER NOT NULL,
                    problem_id TEXT NOT NULL,
                    defect_type TEXT NOT NULL,
                    attack_validity TEXT NOT NULL,
                    winner TEXT NOT NULL,
                    attacker_reward REAL NOT NULL,
                    defender_reward REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _stage_node(
        self,
        conn: sqlite3.Connection,
        node_id: str,
        node_type: str,
        label: str,
        properties: dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
            "properties": properties,
            "updated_at": now,
        }
        conn.execute(
            """
            INSERT OR REPLACE INTO kg_nodes (node_id, node_type, label, properties_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (node_id, node_type, label, json.dumps(properties), now),
        )

    def _stage_edge(
        self,
        conn: sqlite3.Connection,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = metadata or {}
        raw_key = f"{source_id}--{relation}-->{target_id}"
        edge_id = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]
        edge_obj = {
            "edge_id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation,
            "weight": weight,
            "metadata": meta,
        }
        self.edges = [e for e in self.edges if e["edge_id"] != edge_id]
        self.edges.append(edge_obj)
        conn.execute(
            """
            INSERT OR REPLACE INTO kg_edges (edge_id, source_id, target_id, relation, weight, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (edge_id, source_id, target_id, relation, weight, json.dumps(meta)),
        )

    def record_turn_memory(
        self,
        turn_id: int,
        problem: CodeProblem,
        attacker_output: AttackerOutput,
        defender_output: DefenderOutput,
        verdict: RefereeVerdict,
        reflexion: TurnReflexion | None = None,
        blind_attacker: bool = False,
        blind_defender: bool = False,
    ) -> list[str]:
        """Ingest a turn's code-diff reflexions into Real Cognee (when active) and local graph store."""
        now = datetime.now(timezone.utc).isoformat()
        if reflexion is None:
            reflexion = build_turn_reflexion(
                turn_id=turn_id,
                problem=problem,
                attacker_output=attacker_output,
                defender_output=defender_output,
                verdict=verdict,
                blind_attacker=blind_attacker,
                blind_defender=blind_defender,
            )

        base_episode_id = f"ep_turn_{turn_id:04d}_{problem.problem_id}"
        existing_ids = {ep.get("episode_id") for ep in self.episodes}
        episode_id = base_episode_id
        if episode_id in existing_ids:
            suffix = 2
            while f"{base_episode_id}_r{suffix}" in existing_ids:
                suffix += 1
            episode_id = f"{base_episode_id}_r{suffix}"
        prob_node_id = f"problem:{problem.problem_id}"
        defect_node_id = f"defect:{verdict.ground_truth_defect}"
        strat_key = f"{problem.problem_id}::{attacker_output.attack_strategy}"
        strat_node_id = f"strategy:{hashlib.md5(strat_key.encode()).hexdigest()[:10]}"
        repair_node_id = f"repair:{verdict.ground_truth_defect}"

        # 1. Update Attacker strategy statistics
        s_stats = self.attacker_strategy_stats.get(
            strat_key,
            {
                "problem_id": problem.problem_id,
                "strategy_name": attacker_output.attack_strategy,
                "defect_type": attacker_output.injected_defect_type,
                "uses": 0,
                "valid_attacks": 0,
                "attacker_wins": 0,
                "caught_by_defender": 0,
                "over_broken": 0,
                "broken_test_count": verdict.attacker_suite_result.failed_count,
                "last_reward": 0.0,
                "last_attack_diff": "",
            },
        )
        s_stats.setdefault("uses", 0)
        s_stats.setdefault("valid_attacks", 0)
        s_stats.setdefault("attacker_wins", 0)
        s_stats.setdefault("caught_by_defender", 0)
        s_stats.setdefault("over_broken", 0)
        s_stats["uses"] += 1
        s_stats["broken_test_count"] = verdict.attacker_suite_result.failed_count
        s_stats["last_reward"] = verdict.attacker_reward
        s_stats["last_attack_diff"] = reflexion.attack_diff
        if verdict.attack_validity == "valid_subtle_attack":
            s_stats["valid_attacks"] += 1
            if verdict.winner == "attacker":
                s_stats["attacker_wins"] += 1
            else:
                s_stats["caught_by_defender"] += 1
        elif verdict.attack_validity == "invalid_over_broken_attack":
            s_stats["over_broken"] += 1
        self.attacker_strategy_stats[strat_key] = s_stats

        # 2. Update Defender repair knowledge — STRICTLY NO ground-truth `clean_code` leakage!
        d_know = self.defender_repair_knowledge.get(
            verdict.ground_truth_defect,
            {
                "defect_type": verdict.ground_truth_defect,
                "encounters": 0,
                "repairs_succeeded": 0,
                "classifications_succeeded": 0,
                "known_code_clues": [],
                "verified_repair_diffs": [],
                "verified_repairs": {},
            },
        )
        d_know.setdefault("defect_type", verdict.ground_truth_defect)
        d_know.setdefault("encounters", 0)
        d_know.setdefault("repairs_succeeded", 0)
        d_know.setdefault("classifications_succeeded", 0)
        clues = d_know.setdefault("known_code_clues", [])
        repair_diffs = d_know.setdefault("verified_repair_diffs", [])
        verified_repairs = d_know.setdefault("verified_repairs", {})

        d_know["encounters"] += 1
        if verdict.defense_repaired_all_tests:
            d_know["repairs_succeeded"] += 1
            verified_repairs[problem.function_name] = defender_output.repaired_code
            if reflexion.repair_diff and reflexion.repair_diff not in repair_diffs:
                repair_diffs.append(reflexion.repair_diff)
                d_know["verified_repair_diffs"] = repair_diffs[-8:]

        if verdict.classification_correct:
            d_know["classifications_succeeded"] += 1

        if reflexion.defender_lesson not in clues:
            clues.append(reflexion.defender_lesson)
            d_know["known_code_clues"] = clues[-8:]

        self.defender_repair_knowledge[verdict.ground_truth_defect] = d_know

        episode_payload = {
            "episode_id": episode_id,
            "turn_id": turn_id,
            "problem_id": problem.problem_id,
            "function_name": problem.function_name,
            "mlcq_category": problem.mlcq_category,
            "ast_features": reflexion.ast_features,
            "ground_truth_defect": verdict.ground_truth_defect,
            "injected_defect_type": attacker_output.injected_defect_type,
            "attack_strategy": attacker_output.attack_strategy,
            "targeted_edge_case": attacker_output.targeted_edge_case,
            "attack_validity": verdict.attack_validity,
            "broken_tests": verdict.attacker_suite_result.failed_test_ids,
            "predicted_defect_type": defender_output.predicted_defect_type,
            "classification_correct": verdict.classification_correct,
            "defense_repaired_all_tests": verdict.defense_repaired_all_tests,
            "winner": verdict.winner,
            "attacker_reward": verdict.attacker_reward,
            "defender_reward": verdict.defender_reward,
            "attack_diff": reflexion.attack_diff,
            "repair_diff": reflexion.repair_diff,
            "attacker_lesson": reflexion.attacker_lesson,
            "defender_lesson": reflexion.defender_lesson,
            "attacker_feedback": verdict.attacker_feedback,
            "defender_feedback": verdict.defender_feedback,
            "memory_backend": self.active_backend,
            "created_at": now,
        }
        self.episodes.append(episode_payload)

        # 3. Single SQLite transaction for all graph updates + episode persistence
        with contextlib.closing(sqlite3.connect(self.sqlite_path)) as conn, conn:
            self._stage_node(
                conn,
                prob_node_id,
                "CodeProblem",
                problem.title,
                {
                    "problem_id": problem.problem_id,
                    "function_name": problem.function_name,
                    "mlcq_category": problem.mlcq_category,
                    "ast_features": reflexion.ast_features,
                },
            )
            self._stage_node(
                conn,
                defect_node_id,
                "DefectCategory",
                verdict.ground_truth_defect,
                {
                    "defect_type": verdict.ground_truth_defect,
                    "targeted_edge_case": attacker_output.targeted_edge_case,
                },
            )
            self._stage_node(
                conn,
                strat_node_id,
                "AttackStrategy",
                attacker_output.attack_strategy,
                s_stats,
            )
            self._stage_node(
                conn,
                repair_node_id,
                "DefenderRepairKnowledge",
                f"Repair rule for {verdict.ground_truth_defect}",
                {
                    "defect_type": d_know["defect_type"],
                    "encounters": d_know["encounters"],
                    "repairs_succeeded": d_know["repairs_succeeded"],
                    "verified_repair_diffs": d_know.get("verified_repair_diffs", []),
                },
            )
            for feat in reflexion.ast_features:
                feat_node_id = f"ast:{feat}"
                self._stage_node(conn, feat_node_id, "ASTPattern", feat, {"feature": feat})
                self._stage_edge(conn, prob_node_id, feat_node_id, "HAS_AST_PATTERN", weight=1.0)

            self._stage_edge(
                conn,
                strat_node_id,
                prob_node_id,
                "APPLIED_TO_PROBLEM",
                weight=verdict.attacker_reward,
                metadata={"turn_id": turn_id, "validity": verdict.attack_validity},
            )
            self._stage_edge(
                conn,
                strat_node_id,
                defect_node_id,
                "INJECTS_DEFECT",
                weight=1.0,
                metadata={"attack_diff": reflexion.attack_diff},
            )
            self._stage_edge(
                conn,
                repair_node_id,
                defect_node_id,
                "MITIGATES_DEFECT",
                weight=verdict.defender_reward,
                metadata={"repaired_all": verdict.defense_repaired_all_tests},
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, turn_id, problem_id, defect_type, attack_validity,
                    winner, attacker_reward, defender_reward, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    turn_id,
                    problem.problem_id,
                    verdict.ground_truth_defect,
                    verdict.attack_validity,
                    verdict.winner,
                    verdict.attacker_reward,
                    verdict.defender_reward,
                    json.dumps(episode_payload),
                    now,
                ),
            )
            conn.commit()

        # 4. Update rolling in-context buffers with concrete diff-based Reflexions
        self.attacker_in_context.append(reflexion.attacker_lesson)
        self.attacker_in_context = self.attacker_in_context[-self.in_context_window :]

        self.defender_in_context.append(reflexion.defender_lesson)
        self.defender_in_context = self.defender_in_context[-self.in_context_window :]

        # 5. Batched ingestion into REAL Cognee SDK (`cognee.add` + single `cognee.cognify`)
        if self.active_backend == "cognee":
            items = [
                (
                    RealCogneeEngine.ATTACKER_DATASET,
                    f"Turn {turn_id} | Function `{problem.function_name}` | "
                    f"AST: {', '.join(reflexion.ast_features)}\n{reflexion.attacker_lesson}",
                ),
                (
                    RealCogneeEngine.DEFENDER_DATASET,
                    f"Turn {turn_id} | Function `{problem.function_name}` | "
                    f"AST: {', '.join(reflexion.ast_features)}\n{reflexion.defender_lesson}",
                ),
            ]
            if self.cognee_engine.add_lessons(items):
                self._pending_cognee_turns += 1
                if self._pending_cognee_turns >= self.cognify_interval:
                    self.flush_cognee()

        if self.auto_export:
            self.export_portable_bundle(self.bundle_path)
        return [episode_id, prob_node_id, defect_node_id, strat_node_id, repair_node_id]

    def close(self) -> None:
        """Cleanly release background Cognee resources."""
        self.cognee_engine.close()

    def flush_cognee(self) -> bool:
        """Cognify any staged Cognee lessons in a single batched pass (e.g. before exams or end of run)."""
        if self.active_backend != "cognee" or self._pending_cognee_turns <= 0:
            return True
        ok = self.cognee_engine.cognify_datasets(
            [RealCogneeEngine.ATTACKER_DATASET, RealCogneeEngine.DEFENDER_DATASET]
        )
        if ok:
            self.cognee_ingested_count += self._pending_cognee_turns
            self._pending_cognee_turns = 0
        return ok

    def _sync_sqlite_from_memory(self) -> None:
        """Persist in-memory nodes, edges, and episodes to SQLite cleanly in one transaction."""
        with contextlib.closing(sqlite3.connect(self.sqlite_path)) as conn, conn:
            conn.execute("DELETE FROM kg_nodes")
            conn.execute("DELETE FROM kg_edges")
            conn.execute("DELETE FROM episodes")
            for n in self.nodes.values():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO kg_nodes (node_id, node_type, label, properties_json, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        n["node_id"],
                        n["node_type"],
                        n["label"],
                        json.dumps(n.get("properties", {})),
                        n.get("updated_at", datetime.now(timezone.utc).isoformat()),
                    ),
                )
            for e in self.edges:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO kg_edges (edge_id, source_id, target_id, relation, weight, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        e["edge_id"],
                        e["source_id"],
                        e["target_id"],
                        e["relation"],
                        e.get("weight", 1.0),
                        json.dumps(e.get("metadata", {})),
                    ),
                )
            for ep in self.episodes:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO episodes (
                        episode_id, turn_id, problem_id, defect_type, attack_validity,
                        winner, attacker_reward, defender_reward, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ep["episode_id"],
                        ep.get("turn_id", 0),
                        ep.get("problem_id", ""),
                        ep.get("ground_truth_defect") or ep.get("injected_defect_type", "off_by_one_boundary"),
                        ep.get("attack_validity", "valid_subtle_attack"),
                        ep.get("winner", "draw"),
                        float(ep.get("attacker_reward", 0.0)),
                        float(ep.get("defender_reward", 0.0)),
                        json.dumps(ep),
                        ep.get("created_at", datetime.now(timezone.utc).isoformat()),
                    ),
                )
            conn.commit()

    def retrieve_attacker_memories(
        self, problem: CodeProblem, top_k: int | None = None
    ) -> list[str]:
        """Retrieve actionable attack diffs & reflexions via Real Cognee + AST structural similarity."""
        k = self.top_k_memories if top_k is None else max(1, int(top_k))
        memories: list[str] = []
        query_features = set(extract_ast_features(problem.clean_code))
        query_tokens = _tokenize(problem.clean_code + " " + problem.description)

        # 1. Real Cognee semantic/graph retrieval when active AND at least one cognify pass exists
        if self.active_backend == "cognee" and (
            self.cognee_ingested_count > 0 or self._has_persisted_cognee_graph
        ):
            cognee_query = (
                f"Subtle 1-2 test-breaking Python mutation diffs and mistakes to avoid for "
                f"functions with AST patterns [{', '.join(sorted(query_features))}]: {problem.description}"
            )
            cognee_hits = self.cognee_engine.search_lessons(
                RealCogneeEngine.ATTACKER_DATASET, cognee_query, top_k=max(2, k // 2)
            )
            if cognee_hits:
                self.cognee_search_count += 1
                memories.extend(cognee_hits)

        # 2. Rank episodic lessons by AST structural overlap (65%) + code tokens (35%)
        scored_eps: list[tuple[float, dict[str, Any]]] = []
        for ep in self.episodes:
            ep_feats = set(ep.get("ast_features", []))
            ep_tokens = _tokenize(
                f"{ep.get('function_name', '')} {ep.get('attack_diff', '')} {ep.get('injected_defect_type', '')}"
            )
            score = 0.65 * _jaccard(query_features, ep_feats) + 0.35 * _jaccard(query_tokens, ep_tokens)
            # Prioritize high-signal lessons: winning subtle attacks and severe mistakes to avoid
            if ep.get("winner") == "attacker" and ep.get("attack_validity") == "valid_subtle_attack":
                score += 0.25
            elif ep.get("attack_validity") in ("invalid_over_broken_attack", "invalid_no_op_attack"):
                score += 0.15
            if ep.get("problem_id") == problem.problem_id:
                score += 0.20
            scored_eps.append((score, ep))

        scored_eps.sort(key=lambda x: x[0], reverse=True)
        for _, ep in scored_eps[:k]:
            lesson = ep.get("attacker_lesson") or (
                f"[Episode {ep['episode_id']}] `{ep['function_name']}` ({ep['injected_defect_type']}): "
                f"validity={ep['attack_validity']}, winner={ep['winner']}."
            )
            if lesson not in memories:
                memories.append(lesson)

        # 3. Most recent turn reflexion for fast short-term adaptation
        for ic in self.attacker_in_context[-2:]:
            if ic not in memories:
                memories.append(ic)

        return memories[: k + 2]

    def retrieve_defender_memories(
        self,
        problem: CodeProblem,
        mutated_code: str,
        failed_test_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        """Retrieve verified repair diffs & failure reflexions via Real Cognee + AST similarity."""
        k = self.top_k_memories if top_k is None else max(1, int(top_k))
        memories: list[str] = []
        code_features = set(extract_ast_features(mutated_code))
        code_tokens = _tokenize(mutated_code + " " + problem.description)

        # 1. Real Cognee semantic/graph retrieval when active AND at least one cognify pass exists
        if self.active_backend == "cognee" and (
            self.cognee_ingested_count > 0 or self._has_persisted_cognee_graph
        ):
            cognee_query = (
                f"Verified Python bug repair diffs and defect patterns for code with AST patterns "
                f"[{', '.join(sorted(code_features))}]: {problem.description}"
            )
            cognee_hits = self.cognee_engine.search_lessons(
                RealCogneeEngine.DEFENDER_DATASET, cognee_query, top_k=max(2, k // 2)
            )
            if cognee_hits:
                self.cognee_search_count += 1
                memories.extend(cognee_hits)

        # 2. Rank past Defender lessons by AST structural similarity + prioritize verified repair diffs
        scored_eps: list[tuple[float, dict[str, Any]]] = []
        for ep in self.episodes:
            ep_feats = set(ep.get("ast_features", []))
            ep_tokens = _tokenize(
                f"{ep.get('function_name', '')} {ep.get('repair_diff', '')} {ep.get('injected_defect_type', '')}"
            )
            score = 0.65 * _jaccard(code_features, ep_feats) + 0.35 * _jaccard(code_tokens, ep_tokens)
            if ep.get("defense_repaired_all_tests"):
                score += 0.25  # Prefer verified working repair diffs as few-shot examples
            if ep.get("problem_id") == problem.problem_id:
                score += 0.20
            scored_eps.append((score, ep))

        scored_eps.sort(key=lambda x: x[0], reverse=True)
        for _, ep in scored_eps[:k]:
            lesson = ep.get("defender_lesson") or (
                f"[Episode {ep['episode_id']}] `{ep['function_name']}` defect='{ep['injected_defect_type']}'."
            )
            if lesson not in memories:
                memories.append(lesson)

        # 3. Most recent reflexion lessons
        for ic in self.defender_in_context[-2:]:
            if ic not in memories:
                memories.append(ic)

        return memories[: k + 2]

    def get_experience_level(self) -> int:
        """Return total number of recorded episodes or learned knowledge encounters."""
        return max(
            len(self.episodes),
            sum(d.get("encounters", 0) for d in self.defender_repair_knowledge.values()),
        )

    def has_defender_learned_defect(self, defect_type: str) -> bool:
        """Check whether the Defender has accumulated encounters for `defect_type`."""
        info = self.defender_repair_knowledge.get(defect_type)
        return bool(info and info.get("encounters", 0) >= 1)

    def _build_bundle_dict(
        self, role: Literal["both", "attacker", "defender"] = "both"
    ) -> dict[str, Any]:
        """Construct the serializable dictionary for portable export."""
        att_stats = self.attacker_strategy_stats if role in ("both", "attacker") else {}
        def_know = self.defender_repair_knowledge if role in ("both", "defender") else {}
        att_ic = self.attacker_in_context if role in ("both", "attacker") else []
        def_ic = self.defender_in_context if role in ("both", "defender") else []

        return {
            "schema_version": "ares-memory-portable-3.0",
            "active_backend": self.active_backend,
            "backend_status_reason": self.backend_status_reason,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "exported_role": role,
            "summary": {
                "active_backend": self.active_backend,
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "total_episodes": len(self.episodes),
                "cognee_ingested_turns": self.cognee_ingested_count,
                "cognee_search_calls": self.cognee_search_count,
                "learned_defect_types": list(def_know.keys()),
                "tracked_attack_strategies": len(att_stats),
            },
            "nodes": self.nodes,
            "edges": self.edges,
            "attacker_strategy_stats": att_stats,
            "defender_repair_knowledge": def_know,
            "attacker_in_context": att_ic,
            "defender_in_context": def_ic,
            "episodes": self.episodes,
        }

    def export_portable_bundle(
        self,
        target_path: str | Path | None = None,
        role: Literal["both", "attacker", "defender"] = "both",
    ) -> Path:
        """Export Graph, Episodic Diffs, and (if present) Real Cognee directories to `.json` or `.zip`."""
        out_path = Path(target_path) if target_path else self.bundle_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = self._build_bundle_dict(role=role)

        if out_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_root = Path(tmpdir)
                json_file = tmp_root / "cognee_portable_graph.json"
                json_file.write_text(
                    json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                jsonl_file = tmp_root / "episodes.jsonl"
                with jsonl_file.open("w", encoding="utf-8") as f:
                    for ep in self.episodes:
                        f.write(json.dumps(ep, ensure_ascii=False) + "\n")

                if self.sqlite_path.exists():
                    shutil.copy2(self.sqlite_path, tmp_root / self.sqlite_path.name)

                with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for item in tmp_root.iterdir():
                        zf.write(item, arcname=item.name)
                    # Also package real Cognee system/data directories if they exist
                    for sub_name in ("cognee_system", "cognee_data"):
                        sub_dir = self.store_dir / sub_name
                        if sub_dir.exists():
                            for fp in sub_dir.rglob("*"):
                                if fp.is_file():
                                    zf.write(fp, arcname=fp.relative_to(self.store_dir).as_posix())
            return out_path

        # Atomic write to avoid corrupting portable JSON on mid-run interruption
        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_out.write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_out, out_path)
        return out_path

    def import_portable_bundle(
        self,
        source_path: str | Path,
        mode: Literal["replace", "merge"] = "replace",
    ) -> dict[str, Any]:
        """Import a portable memory bundle (`.json` or `.zip`) in `replace` or `merge` mode."""
        in_path = Path(source_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Portable memory bundle not found: {in_path}")

        if in_path.suffix.lower() == ".zip":
            store_root = self.store_dir.resolve()
            with zipfile.ZipFile(in_path, "r") as zf:
                raw_json = zf.read("cognee_portable_graph.json").decode("utf-8")
                bundle = json.loads(raw_json)
                if mode == "replace":
                    for member in zf.namelist():
                        norm_member = member.replace("\\", "/")
                        if (
                            norm_member.startswith(("cognee_system/", "cognee_data/"))
                            and not norm_member.endswith("/")
                        ):
                            dest = (store_root / norm_member).resolve()
                            if not dest.is_relative_to(store_root):
                                raise ValueError(
                                    f"Unsafe path traversal entry detected in portable bundle: {member!r}"
                                )
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(member) as src, dest.open("wb") as dst:
                                shutil.copyfileobj(src, dst)
                            if norm_member.startswith("cognee_system/"):
                                self._has_persisted_cognee_graph = True
        else:
            bundle = json.loads(in_path.read_text(encoding="utf-8"))

        schema_version = str(bundle.get("schema_version", "unknown"))
        is_legacy_v1 = schema_version.startswith("ares-cognee-portable-1.") or schema_version == "unknown"
        if is_legacy_v1 and bundle.get("defender_repair_knowledge"):
            self.migrated_legacy_schema = schema_version
            logger.warning(
                "Migrating legacy memory bundle '%s' (schema=%s): backfilling `verified_repair_diffs` "
                "and clearing legacy `verified_repairs` to prevent pre-v3 clean-code leakage. "
                "Pass `--fresh-memory` for a clean Turn-0 baseline.",
                in_path,
                schema_version,
            )

        normalized_def_know: dict[str, dict[str, Any]] = {}
        for dtype, raw_dk in bundle.get("defender_repair_knowledge", {}).items():
            dk = dict(raw_dk) if isinstance(raw_dk, dict) else {}
            dk.setdefault("defect_type", dtype)
            dk.setdefault("encounters", 0)
            dk.setdefault("repairs_succeeded", 0)
            dk.setdefault("classifications_succeeded", 0)
            dk.setdefault("known_code_clues", [])
            dk.setdefault("verified_repair_diffs", [])
            dk["verified_repairs"] = {} if is_legacy_v1 else dict(dk.get("verified_repairs", {}))
            normalized_def_know[dtype] = dk

        if mode == "replace":
            self.nodes = bundle.get("nodes", {})
            self.edges = bundle.get("edges", [])
            self.attacker_strategy_stats = bundle.get("attacker_strategy_stats", {})
            self.defender_repair_knowledge = normalized_def_know
            self.attacker_in_context = bundle.get("attacker_in_context", [])
            self.defender_in_context = bundle.get("defender_in_context", [])
            self.episodes = bundle.get("episodes", [])
        else:
            for nid, nval in bundle.get("nodes", {}).items():
                self.nodes[nid] = nval
            existing_edge_ids = {e["edge_id"] for e in self.edges}
            for edge in bundle.get("edges", []):
                if edge["edge_id"] not in existing_edge_ids:
                    self.edges.append(edge)
                    existing_edge_ids.add(edge["edge_id"])
            for skey, sstat in bundle.get("attacker_strategy_stats", {}).items():
                if skey not in self.attacker_strategy_stats:
                    self.attacker_strategy_stats[skey] = dict(sstat)
                else:
                    cur = self.attacker_strategy_stats[skey]
                    cur["uses"] = cur.get("uses", 0) + sstat.get("uses", 0)
                    cur["valid_attacks"] = cur.get("valid_attacks", 0) + sstat.get("valid_attacks", 0)
                    cur["attacker_wins"] = cur.get("attacker_wins", 0) + sstat.get("attacker_wins", 0)
                    cur["caught_by_defender"] = cur.get("caught_by_defender", 0) + sstat.get("caught_by_defender", 0)
                    cur["over_broken"] = cur.get("over_broken", 0) + sstat.get("over_broken", 0)
            for dtype, dknow in normalized_def_know.items():
                if dtype not in self.defender_repair_knowledge:
                    self.defender_repair_knowledge[dtype] = dknow
                else:
                    cur_d = self.defender_repair_knowledge[dtype]
                    cur_d["encounters"] = cur_d.get("encounters", 0) + dknow.get("encounters", 0)
                    cur_d["repairs_succeeded"] = cur_d.get("repairs_succeeded", 0) + dknow.get("repairs_succeeded", 0)
                    cur_d["classifications_succeeded"] = cur_d.get("classifications_succeeded", 0) + dknow.get("classifications_succeeded", 0)
                    cur_clues = cur_d.setdefault("known_code_clues", [])
                    for clue in dknow.get("known_code_clues", []):
                        if clue not in cur_clues:
                            cur_clues.append(clue)
                    cur_d["known_code_clues"] = cur_clues[-8:]
                    cur_diffs = cur_d.setdefault("verified_repair_diffs", [])
                    for diff in dknow.get("verified_repair_diffs", []):
                        if diff not in cur_diffs:
                            cur_diffs.append(diff)
                    cur_d["verified_repair_diffs"] = cur_diffs[-8:]
                    cur_d.setdefault("verified_repairs", {}).update(dknow.get("verified_repairs", {}))
            existing_ep_ids = {ep["episode_id"] for ep in self.episodes}
            newly_merged_eps: list[dict[str, Any]] = []
            for ep in bundle.get("episodes", []):
                if ep["episode_id"] not in existing_ep_ids:
                    self.episodes.append(ep)
                    existing_ep_ids.add(ep["episode_id"])
                    newly_merged_eps.append(ep)
            for ic in bundle.get("attacker_in_context", []):
                if ic not in self.attacker_in_context:
                    self.attacker_in_context.append(ic)
            self.attacker_in_context = self.attacker_in_context[-self.in_context_window :]
            for ic in bundle.get("defender_in_context", []):
                if ic not in self.defender_in_context:
                    self.defender_in_context.append(ic)
            self.defender_in_context = self.defender_in_context[-self.in_context_window :]

            # Re-ingest newly merged lessons into real Cognee if active
            if self.active_backend == "cognee" and newly_merged_eps:
                staged_items: list[tuple[str, str]] = []
                for ep in newly_merged_eps:
                    if ep.get("attacker_lesson"):
                        staged_items.append((RealCogneeEngine.ATTACKER_DATASET, ep["attacker_lesson"]))
                    if ep.get("defender_lesson"):
                        staged_items.append((RealCogneeEngine.DEFENDER_DATASET, ep["defender_lesson"]))
                if staged_items and self.cognee_engine.add_lessons(staged_items):
                    self._pending_cognee_turns += len(newly_merged_eps)
                    self.flush_cognee()

        # Keep SQLite 100% synchronized with imported/merged in-memory state
        self._sync_sqlite_from_memory()

        return {
            "mode": mode,
            "active_backend": self.active_backend,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_episodes": len(self.episodes),
            "learned_defect_types": list(self.defender_repair_knowledge.keys()),
            "tracked_attack_strategies": len(self.attacker_strategy_stats),
        }

    @classmethod
    def inspect_bundle(cls, source_path: str | Path) -> dict[str, Any]:
        """Inspect any exported `.json` or `.zip` memory bundle and return its detailed metadata."""
        in_path = Path(source_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Portable memory bundle not found: {in_path}")

        if in_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(in_path, "r") as zf:
                bundle = json.loads(zf.read("cognee_portable_graph.json").decode("utf-8"))
        else:
            bundle = json.loads(in_path.read_text(encoding="utf-8"))

        def_know = bundle.get("defender_repair_knowledge", {})
        defect_breakdown = {
            k: {
                "encounters": v.get("encounters", 0),
                "repairs_succeeded": v.get("repairs_succeeded", 0),
                "verified_functions": list(v.get("verified_repairs", {}).keys()),
            }
            for k, v in def_know.items()
        }
        return {
            "file": str(in_path),
            "schema_version": bundle.get("schema_version", "unknown"),
            "active_backend": bundle.get("active_backend", "local_graph"),
            "exported_at": bundle.get("exported_at", "unknown"),
            "exported_role": bundle.get("exported_role", "both"),
            "summary": bundle.get("summary", {}),
            "defender_defect_breakdown": defect_breakdown,
        }


# Honest alias for clarity when used in local_graph or hybrid Cognee mode
ARESMemoryManager = CogneeMemoryManager
