"""Evaluation package for ARES."""

from ares.evaluation.exam_evaluator import (
    ExternalExamEvaluator,
    build_pre_post_comparison,
    render_pre_post_markdown,
)

__all__ = [
    "ExternalExamEvaluator",
    "build_pre_post_comparison",
    "render_pre_post_markdown",
]

