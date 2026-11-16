"""Dataset loader for MLCQ Technical Debt Self-Play, External Exam Benchmarks, and Plug-and-Play Custom Datasets."""

from __future__ import annotations

from pathlib import Path
from ares.datasets.adapter import PlugAndPlayDatasetAdapter
from ares.models import CodeProblem


DEFAULT_SELF_PLAY_DATASET = Path(__file__).parent / "mlcq_technical_debt_suite.json"
DEFAULT_EXTERNAL_EXAM_DATASET = Path(__file__).parent / "external_exam_benchmark.json"


def load_problems(path: str | Path | None = None, exam: bool = False) -> list[CodeProblem]:
    """Load and normalize a list of CodeProblem instances from any supported file format (.json, .jsonl, .yaml, .csv)."""
    target = (
        Path(path)
        if path
        else (DEFAULT_EXTERNAL_EXAM_DATASET if exam else DEFAULT_SELF_PLAY_DATASET)
    )
    return PlugAndPlayDatasetAdapter.load_any_dataset(target)
