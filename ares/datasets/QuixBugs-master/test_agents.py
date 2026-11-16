"""Unit and Integration Tests for QuixBugs 4-Agent Memory Framework."""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
workspace_dir = Path(__file__).resolve().parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from quixbugs_agents.dataset_loader import QuixBugsDatasetLoader
from quixbugs_agents.execution_sandbox import PythonSandbox
from quixbugs_agents.memory import (
    Mem0Manager, CogneeManager, ZepManager, QdrantManager
)
from quixbugs_agents.agents import (
    LangGraphMem0Agent,
    LangGraphCogneeAgent,
    LangGraphZepAgent,
    LangGraphQdrantAgent
)


class TestQuixBugsDatasetLoader(unittest.TestCase):
    def test_list_algorithms(self):
        algos = QuixBugsDatasetLoader.list_algorithms()
        self.assertGreater(len(algos), 20)
        self.assertIn("bitcount", algos)
        self.assertIn("breadth_first_search", algos)

    def test_load_sample(self):
        buggy = QuixBugsDatasetLoader.load_sample("bitcount", variant="buggy")
        self.assertIsNotNone(buggy)
        self.assertTrue(buggy["ground_truth_is_buggy"])
        self.assertIn("^= n - 1", buggy["code"])

        correct = QuixBugsDatasetLoader.load_sample("bitcount", variant="correct")
        self.assertIsNotNone(correct)
        self.assertFalse(correct["ground_truth_is_buggy"])
        self.assertIn("&= n - 1", correct["code"])

    def test_load_dataset_balanced(self):
        samples = QuixBugsDatasetLoader.load_dataset(limit=4, balance=True)
        self.assertEqual(len(samples), 4)
        buggy_cnt = sum(1 for s in samples if s["ground_truth_is_buggy"])
        clean_cnt = sum(1 for s in samples if not s["ground_truth_is_buggy"])
        self.assertEqual(buggy_cnt, 2)
        self.assertEqual(clean_cnt, 2)

    def test_load_cleanvul_dataset(self):
        samples = QuixBugsDatasetLoader.load_dataset(limit=4, balance=True, dataset="cleanvul")
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]["dataset"], "cleanvul")

    def test_load_agent_green_dataset(self):
        samples = QuixBugsDatasetLoader.load_dataset(limit=4, balance=True, dataset="agent_green")
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]["dataset"], "agent_green")


class TestExecutionSandbox(unittest.TestCase):
    def test_correct_code_execution(self):
        correct_sample = QuixBugsDatasetLoader.load_sample("bitcount", variant="correct")
        res = PythonSandbox.run_quixbugs_tests(
            code=correct_sample["code"],
            algorithm="bitcount",
            test_cases=correct_sample["test_cases"]
        )
        self.assertTrue(res["run"])
        self.assertTrue(res["passed"])

    def test_buggy_code_execution_timeout(self):
        buggy_sample = QuixBugsDatasetLoader.load_sample("bitcount", variant="buggy")
        # bitcount buggy has infinite loop for non-zero inputs, sandbox should timeout safely
        res = PythonSandbox.run_quixbugs_tests(
            code=buggy_sample["code"],
            algorithm="bitcount",
            test_cases=buggy_sample["test_cases"][:1],
            timeout_sec=1
        )
        self.assertTrue(res["run"])
        self.assertFalse(res["passed"])


class TestMemoryEngines(unittest.TestCase):
    def test_mem0_engine(self):
        mem0 = Mem0Manager.get_instance()
        entry = mem0.add("Test bug insight on breadth_first_search", metadata={"algorithm": "breadth_first_search"})
        self.assertIsNotNone(entry["id"])
        hits = mem0.search("breadth_first_search loop", limit=2)
        self.assertGreater(len(hits), 0)

    def test_cognee_engine(self):
        cognee = CogneeManager.get_instance()
        code_buggy = "def bitcount(n):\n    while n:\n        n ^= n - 1"
        graph_res = cognee.parse_code_to_graph(code_buggy, "bitcount")
        self.assertTrue(graph_res["success"])
        self.assertGreater(len(graph_res["violations"]), 0)

    def test_zep_engine(self):
        zep = ZepManager.get_instance()
        fact = zep.add_session_fact("bitcount", "bitwise", "Infinite loop confirmed on XOR update")
        self.assertIn("fact_id", fact)
        facts = zep.get_relevant_facts("bitcount")
        self.assertGreater(len(facts), 0)

    def test_qdrant_engine(self):
        qdrant = QdrantManager.get_instance()
        qdrant.upsert("p_test_1", "def bitcount(n): while n: n ^= n - 1", {"algorithm": "bitcount", "is_buggy": True})
        results = qdrant.search("bitcount while n", limit=2)
        self.assertGreater(len(results), 0)


class TestFourAgentsExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = QuixBugsDatasetLoader.load_sample("bitcount", variant="buggy")

    def test_langgraph_mem0_agent(self):
        agent = LangGraphMem0Agent(provider="mock", max_iterations=2)
        res = agent.evaluate_sample(self.sample)
        self.assertEqual(res["framework"], "langgraph_mem0")
        self.assertIsNotNone(res["final_matches_gt"])

    def test_langgraph_cognee_agent(self):
        agent = LangGraphCogneeAgent(provider="mock", max_iterations=2)
        res = agent.evaluate_sample(self.sample)
        self.assertEqual(res["framework"], "langgraph_cognee")
        self.assertIsNotNone(res["final_matches_gt"])

    def test_langgraph_zep_agent(self):
        agent = LangGraphZepAgent(provider="mock", max_iterations=2)
        res = agent.evaluate_sample(self.sample)
        self.assertEqual(res["framework"], "langgraph_zep")
        self.assertIsNotNone(res["final_matches_gt"])

    def test_langgraph_qdrant_agent(self):
        agent = LangGraphQdrantAgent(provider="mock", max_iterations=2)
        res = agent.evaluate_sample(self.sample)
        self.assertEqual(res["framework"], "langgraph_qdrant")
        self.assertIsNotNone(res["final_matches_gt"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
