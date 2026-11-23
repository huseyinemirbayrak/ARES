"""Datasets package for ARES."""

from ares.datasets.adapter import PlugAndPlayDatasetAdapter
from ares.datasets.bug_collector import GeneratedBugDatasetStore
from ares.datasets.external_suites import (
    load_agent_green_suite,
    load_all_external_suites,
    load_cleanvul_suite,
    load_quixbugs_suite,
)
from ares.datasets.loader import (
    DEFAULT_EXTERNAL_EXAM_DATASET,
    DEFAULT_SELF_PLAY_DATASET,
    load_problems,
)

__all__ = [
    "PlugAndPlayDatasetAdapter",
    "GeneratedBugDatasetStore",
    "load_problems",
    "load_quixbugs_suite",
    "load_agent_green_suite",
    "load_cleanvul_suite",
    "load_all_external_suites",
    "DEFAULT_SELF_PLAY_DATASET",
    "DEFAULT_EXTERNAL_EXAM_DATASET",
]
