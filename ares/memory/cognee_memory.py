"""Cognee-backed Portable Knowledge Graph Memory Manager for ARES.

Provides a unified, plug-and-play memory layer that:
1. Interfaces with the `cognee` library (`cognee.add`, `cognee.cognify`, `cognee.search`)
   when installed and configured with an embedding/LLM provider.
2. Maintains a synchronized, 100% portable Cognee Knowledge Graph (Nodes, Edges,
   Strategy Weights, and Repair Rules) persisted to both JSON (`cognee_portable_graph.json`),
   SQLite (`cognee_portable.sqlite3`), and optional `.zip` portable archives.
3. Supports on-demand export (`json` or `zip`, filtered by role) and on-demand import
   with both `replace` and `merge` modes so agent memory can be combined across datasets.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import re
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ares.models import (
    AttackerOutput,
    CodeProblem,
    DefenderOutput,
    RefereeVerdict,
)


def _tokenize(text: str) -> set[str]:
    """Extract normalized alphanumeric and operator tokens from text/code."""
    words = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|<=|>=|==|!=", text.lower()))
    return {w for w in words if len(w) > 1}


class CogneeMemoryManager:
    """Portable Knowledge Graph & Episodic Memory powered by Cognee architecture."""

    def __init__(
        self,
        store_dir: str | Path = "memory_store",
        bundle_filename: str = "cognee_portable_graph.json",
        enable_native_cognee: bool = True,
        in_context_window: int = 6,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_path = self.store_dir / bundle_filename
        self.sqlite_path = self.store_dir / "cognee_portable.sqlite3"
        self.in_context_window = in_context_window

        self.cognee_available = (
            enable_native_cognee
            and importlib.util.find_spec("cognee") is not None
        )

        # Knowledge Graph structures (Cognee Entity-Relation schema)
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []

        # Learned statistics for portable transfer across models/sessions/datasets
        self.attacker_strategy_stats: dict[str, dict[str, Any]] = {}
        self.defender_repair_knowledge: dict[str, dict[str, Any]] = {}

        # Rolling in-context memory buffers for each agent
        self.attacker_in_context: list[str] = []
        self.defender_in_context: list[str] = []

        self._init_sqlite()
        if self.bundle_path.exists():
            self.import_portable_bundle(self.bundle_path, mode="replace")

    def _init_sqlite(self) -> None:
        """Initialize portable SQLite backing store for knowledge graph nodes & episodes."""
        with sqlite3.connect(self.sqlite_path) as conn:
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

    def _upsert_node(
        self, node_id: str, node_type: str, label: str, properties: dict[str, Any]
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
            "properties": properties,
            "updated_at": now,
        }
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO kg_nodes (node_id, node_type, label, properties_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (node_id, node_type, label, json.dumps(properties), now),
            )
            conn.commit()

    def _add_edge(
        self,
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
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO kg_edges (edge_id, source_id, target_id, relation, weight, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (edge_id, source_id, target_id, relation, weight, json.dumps(meta)),
            )
            conn.commit()

    def record_turn_memory(
        self,
        turn_id: int,
        problem: CodeProblem,
        attacker_output: AttackerOutput,
        defender_output: DefenderOutput,
        verdict: RefereeVerdict,
    ) -> list[str]:
        """Ingest a self-play interaction into the Cognee Knowledge Graph and in-context memory."""
        now = datetime.now(timezone.utc).isoformat()
        episode_id = f"ep_turn_{turn_id:04d}_{problem.problem_id}"

        prob_node_id = f"problem:{problem.problem_id}"
        defect_node_id = f"defect:{attacker_output.injected_defect_type}"
        strat_key = f"{problem.problem_id}::{attacker_output.attack_strategy}"
        strat_node_id = f"strategy:{hashlib.md5(strat_key.encode()).hexdigest()[:10]}"
        repair_node_id = f"repair:{attacker_output.injected_defect_type}"

        self._upsert_node(
            prob_node_id,
            "CodeProblem",
            problem.title,
            {
                "problem_id": problem.problem_id,
                "function_name": problem.function_name,
                "mlcq_category": problem.mlcq_category,
            },
        )
        self._upsert_node(
            defect_node_id,
            "DefectCategory",
            attacker_output.injected_defect_type,
            {
                "defect_type": attacker_output.injected_defect_type,
                "targeted_edge_case": attacker_output.targeted_edge_case,
            },
        )

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
            },
        )
        s_stats["uses"] += 1
        s_stats["broken_test_count"] = verdict.attacker_suite_result.failed_count
        s_stats["last_reward"] = verdict.attacker_reward
        if verdict.attack_validity == "valid_subtle_attack":
            s_stats["valid_attacks"] += 1
            if verdict.winner == "attacker":
                s_stats["attacker_wins"] += 1
            else:
                s_stats["caught_by_defender"] += 1
        elif verdict.attack_validity == "invalid_over_broken_attack":
            s_stats["over_broken"] += 1
        self.attacker_strategy_stats[strat_key] = s_stats

        self._upsert_node(
            strat_node_id,
            "AttackStrategy",
            attacker_output.attack_strategy,
            s_stats,
        )

        d_know = self.defender_repair_knowledge.get(
            attacker_output.injected_defect_type,
            {
                "defect_type": attacker_output.injected_defect_type,
                "encounters": 0,
                "repairs_succeeded": 0,
                "classifications_succeeded": 0,
                "known_code_clues": [],
                "verified_repairs": {},
            },
        )
        d_know["encounters"] += 1
        if verdict.defense_repaired_all_tests:
            d_know["repairs_succeeded"] += 1
            d_know["verified_repairs"][problem.function_name] = defender_output.repaired_code
        else:
            d_know["verified_repairs"][problem.function_name] = problem.clean_code

        if verdict.classification_correct:
            d_know["classifications_succeeded"] += 1

        clue_summary = (
            f"When {verdict.attacker_suite_result.failed_test_ids} fail with "
            f"'{attacker_output.targeted_edge_case}', classify as "
            f"'{attacker_output.injected_defect_type}' ({attacker_output.explanation})"
        )
        if clue_summary not in d_know["known_code_clues"]:
            d_know["known_code_clues"].append(clue_summary)
            d_know["known_code_clues"] = d_know["known_code_clues"][-8:]

        self.defender_repair_knowledge[attacker_output.injected_defect_type] = d_know

        self._upsert_node(
            repair_node_id,
            "DefenderRepairKnowledge",
            f"Repair rule for {attacker_output.injected_defect_type}",
            {
                "defect_type": d_know["defect_type"],
                "encounters": d_know["encounters"],
                "repairs_succeeded": d_know["repairs_succeeded"],
                "known_code_clues": d_know["known_code_clues"],
            },
        )

        self._add_edge(
            strat_node_id,
            prob_node_id,
            "APPLIED_TO_PROBLEM",
            weight=verdict.attacker_reward,
            metadata={"turn_id": turn_id, "validity": verdict.attack_validity},
        )
        self._add_edge(
            strat_node_id,
            defect_node_id,
            "INJECTS_DEFECT",
            weight=1.0,
            metadata={"broken_tests": verdict.attacker_suite_result.failed_test_ids},
        )
        self._add_edge(
            repair_node_id,
            defect_node_id,
            "MITIGATES_DEFECT",
            weight=verdict.defender_reward,
            metadata={"repaired_all": verdict.defense_repaired_all_tests},
        )

        episode_payload = {
            "episode_id": episode_id,
            "turn_id": turn_id,
            "problem_id": problem.problem_id,
            "function_name": problem.function_name,
            "mlcq_category": problem.mlcq_category,
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
            "attacker_feedback": verdict.attacker_feedback,
            "defender_feedback": verdict.defender_feedback,
            "created_at": now,
        }
        self.episodes.append(episode_payload)

        with sqlite3.connect(self.sqlite_path) as conn:
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
                    attacker_output.injected_defect_type,
                    verdict.attack_validity,
                    verdict.winner,
                    verdict.attacker_reward,
                    verdict.defender_reward,
                    json.dumps(episode_payload),
                    now,
                ),
            )
            conn.commit()

        self.attacker_in_context.append(
            f"Turn {turn_id} ({problem.problem_id}): Strategy='{attacker_output.attack_strategy}' "
            f"[{attacker_output.injected_defect_type}] -> broke "
            f"{verdict.attacker_suite_result.failed_count}/{verdict.attacker_suite_result.total_tests} "
            f"tests, validity={verdict.attack_validity}, winner={verdict.winner}, "
            f"reward={verdict.attacker_reward}. Feedback: {verdict.attacker_feedback}"
        )
        self.attacker_in_context = self.attacker_in_context[-self.in_context_window :]

        self.defender_in_context.append(
            f"Turn {turn_id} ({problem.problem_id}): Attacker injected '{attacker_output.injected_defect_type}' "
            f"breaking {verdict.attacker_suite_result.failed_test_ids}. "
            f"Your prediction='{defender_output.predicted_defect_type}', "
            f"repaired_all={verdict.defense_repaired_all_tests}, reward={verdict.defender_reward}. "
            f"Lesson: {attacker_output.explanation}"
        )
        self.defender_in_context = self.defender_in_context[-self.in_context_window :]

        if self.cognee_available:
            self._try_native_cognee_ingest(episode_payload)

        self.export_portable_bundle(self.bundle_path)

        return [episode_id, prob_node_id, defect_node_id, strat_node_id, repair_node_id]

    def _try_native_cognee_ingest(self, episode_payload: dict[str, Any]) -> None:
        """Best-effort call to native `cognee.add` if installed and configured with an API key."""
        try:
            import os
            import cognee  # type: ignore

            if not os.environ.get("LLM_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
                return
            text_doc = json.dumps(episode_payload, ensure_ascii=False)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(cognee.add(text_doc, "ares_self_play_memory"))
            finally:
                loop.close()
        except Exception:
            pass

    def retrieve_attacker_memories(
        self, problem: CodeProblem, top_k: int = 4
    ) -> list[str]:
        """Retrieve relevant portable Cognee graph memories + in-context history for Attacker."""
        memories: list[str] = []

        for _, stats in self.attacker_strategy_stats.items():
            if stats["problem_id"] == problem.problem_id:
                memories.append(
                    f"[Cognee Graph - StrategyStat] On {problem.problem_id}, strategy "
                    f"'{stats['strategy_name']}' ({stats['defect_type']}) broke "
                    f"{stats['broken_test_count']} tests: valid={stats['valid_attacks']}/{stats['uses']}, "
                    f"attacker_wins={stats['attacker_wins']}, caught_by_defender={stats['caught_by_defender']}, "
                    f"over_broken={stats['over_broken']}."
                )

        query_tokens = _tokenize(problem.clean_code + " " + problem.mlcq_category)
        scored_eps: list[tuple[float, dict[str, Any]]] = []
        for ep in self.episodes:
            ep_tokens = _tokenize(
                f"{ep['function_name']} {ep['mlcq_category']} {ep['injected_defect_type']} {ep['targeted_edge_case']}"
            )
            overlap = len(query_tokens & ep_tokens) / max(1, len(query_tokens | ep_tokens))
            scored_eps.append((overlap, ep))

        scored_eps.sort(key=lambda x: x[0], reverse=True)
        for _, ep in scored_eps[:top_k]:
            memories.append(
                f"[Cognee Episode {ep['episode_id']}] Function `{ep['function_name']}`: "
                f"defect='{ep['injected_defect_type']}' broke {ep['broken_tests']} -> "
                f"validity={ep['attack_validity']}, winner={ep['winner']}."
            )

        for ic in self.attacker_in_context[-3:]:
            if ic not in memories:
                memories.append(f"[In-Context] {ic}")

        return memories[: top_k + 3]

    def retrieve_defender_memories(
        self,
        problem: CodeProblem,
        mutated_code: str,
        failed_test_ids: list[str] | None = None,
        top_k: int = 4,
    ) -> list[str]:
        """Retrieve relevant portable Cognee graph memories + repair clues for Defender."""
        memories: list[str] = []

        code_tokens = _tokenize(mutated_code + " " + problem.description)
        for defect_type, d_know in self.defender_repair_knowledge.items():
            clues = "; ".join(d_know.get("known_code_clues", [])[-2:])
            memories.append(
                f"[Cognee RepairRule - {defect_type}] Encounters={d_know['encounters']}, "
                f"RepairsSucceeded={d_know['repairs_succeeded']}. Clues: {clues}"
            )

        scored_eps: list[tuple[float, dict[str, Any]]] = []
        for ep in self.episodes:
            ep_tokens = _tokenize(
                f"{ep['function_name']} {ep['injected_defect_type']} {ep['targeted_edge_case']} {ep['attacker_feedback']}"
            )
            score = len(code_tokens & ep_tokens) / max(1, len(code_tokens | ep_tokens))
            if ep["problem_id"] == problem.problem_id:
                score += 0.5
            scored_eps.append((score, ep))

        scored_eps.sort(key=lambda x: x[0], reverse=True)
        for _, ep in scored_eps[:top_k]:
            memories.append(
                f"[Cognee Episode {ep['episode_id']}] On `{ep['function_name']}`, "
                f"failing {ep['broken_tests']} was caused by '{ep['injected_defect_type']}' "
                f"({ep['targeted_edge_case']})."
            )

        for ic in self.defender_in_context[-3:]:
            if ic not in memories:
                memories.append(f"[In-Context] {ic}")

        return memories[: top_k + 4]

    def get_experience_level(self) -> int:
        """Return total number of self-play episodes or learned knowledge entries in portable memory."""
        return max(
            len(self.episodes),
            sum(d.get("encounters", 0) for d in self.defender_repair_knowledge.values()),
        )

    def has_defender_learned_defect(self, defect_type: str) -> bool:
        """Check whether the Defender has accumulated knowledge for `defect_type` in Cognee memory."""
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
            "schema_version": "ares-cognee-portable-2.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "exported_role": role,
            "summary": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "total_episodes": len(self.episodes),
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
        """Export Cognee Knowledge Graph & Episodic Memory to `.json` or `.zip` archive on demand."""
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
                    shutil.copy2(self.sqlite_path, tmp_root / "cognee_portable.sqlite3")

                with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for item in tmp_root.iterdir():
                        zf.write(item, arcname=item.name)
            return out_path

        out_path.write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out_path

    def import_portable_bundle(
        self,
        source_path: str | Path,
        mode: Literal["replace", "merge"] = "replace",
    ) -> dict[str, Any]:
        """Import a portable Cognee Knowledge Graph bundle (`.json` or `.zip`) in `replace` or `merge` mode."""
        in_path = Path(source_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Portable memory bundle not found: {in_path}")

        if in_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(in_path, "r") as zf:
                raw_json = zf.read("cognee_portable_graph.json").decode("utf-8")
                bundle = json.loads(raw_json)
        else:
            bundle = json.loads(in_path.read_text(encoding="utf-8"))

        if mode == "replace":
            self.nodes = bundle.get("nodes", {})
            self.edges = bundle.get("edges", [])
            self.attacker_strategy_stats = bundle.get("attacker_strategy_stats", {})
            self.defender_repair_knowledge = bundle.get("defender_repair_knowledge", {})
            self.attacker_in_context = bundle.get("attacker_in_context", [])
            self.defender_in_context = bundle.get("defender_in_context", [])
            self.episodes = bundle.get("episodes", [])
        else:  # mode == "merge"
            # Merge nodes
            for nid, nval in bundle.get("nodes", {}).items():
                self.nodes[nid] = nval
            # Merge edges without duplicates
            existing_edge_ids = {e["edge_id"] for e in self.edges}
            for edge in bundle.get("edges", []):
                if edge["edge_id"] not in existing_edge_ids:
                    self.edges.append(edge)
                    existing_edge_ids.add(edge["edge_id"])
            # Merge attacker strategy stats
            for skey, sstat in bundle.get("attacker_strategy_stats", {}).items():
                if skey not in self.attacker_strategy_stats:
                    self.attacker_strategy_stats[skey] = sstat
                else:
                    cur = self.attacker_strategy_stats[skey]
                    cur["uses"] += sstat.get("uses", 0)
                    cur["valid_attacks"] += sstat.get("valid_attacks", 0)
                    cur["attacker_wins"] += sstat.get("attacker_wins", 0)
                    cur["caught_by_defender"] += sstat.get("caught_by_defender", 0)
                    cur["over_broken"] += sstat.get("over_broken", 0)
            # Merge defender repair knowledge
            for dtype, dknow in bundle.get("defender_repair_knowledge", {}).items():
                if dtype not in self.defender_repair_knowledge:
                    self.defender_repair_knowledge[dtype] = dknow
                else:
                    cur_d = self.defender_repair_knowledge[dtype]
                    cur_d["encounters"] += dknow.get("encounters", 0)
                    cur_d["repairs_succeeded"] += dknow.get("repairs_succeeded", 0)
                    cur_d["classifications_succeeded"] += dknow.get(
                        "classifications_succeeded", 0
                    )
                    for clue in dknow.get("known_code_clues", []):
                        if clue not in cur_d["known_code_clues"]:
                            cur_d["known_code_clues"].append(clue)
                    cur_d.setdefault("verified_repairs", {}).update(
                        dknow.get("verified_repairs", {})
                    )
            # Merge episodes
            existing_ep_ids = {ep["episode_id"] for ep in self.episodes}
            for ep in bundle.get("episodes", []):
                if ep["episode_id"] not in existing_ep_ids:
                    self.episodes.append(ep)
                    existing_ep_ids.add(ep["episode_id"])
            # Merge rolling in-context buffers
            for ic in bundle.get("attacker_in_context", []):
                if ic not in self.attacker_in_context:
                    self.attacker_in_context.append(ic)
            self.attacker_in_context = self.attacker_in_context[-self.in_context_window :]
            for ic in bundle.get("defender_in_context", []):
                if ic not in self.defender_in_context:
                    self.defender_in_context.append(ic)
            self.defender_in_context = self.defender_in_context[-self.in_context_window :]

        return {
            "mode": mode,
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
            "exported_at": bundle.get("exported_at", "unknown"),
            "exported_role": bundle.get("exported_role", "both"),
            "summary": bundle.get("summary", {}),
            "defender_defect_breakdown": defect_breakdown,
        }
