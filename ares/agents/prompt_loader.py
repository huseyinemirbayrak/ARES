"""Loader and validator for 5-part structured agent prompt parameter files (YAML/JSON)."""

from __future__ import annotations

from pathlib import Path
import yaml
from ares.models import StructuredPromptConfig


DEFAULT_ATTACKER_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "prompts" / "attacker_prompt.yaml"
)
DEFAULT_DEFENDER_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "prompts" / "defender_prompt.yaml"
)

REQUIRED_SECTIONS = ("task", "recipe", "output", "validation", "examples")


def load_prompt_config(path: str | Path) -> StructuredPromptConfig:
    """Load a YAML/JSON parameter file and validate all 5 prompt sections."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt configuration file not found: {file_path}")

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    sections = raw.get("prompt_sections", {})
    missing = [sec for sec in REQUIRED_SECTIONS if sec not in sections or not sections[sec]]
    if missing:
        raise ValueError(
            f"Prompt config {file_path} is missing required sections: {missing}"
        )

    return StructuredPromptConfig(
        agent_role=raw.get("agent_role", "attacker"),
        version=str(raw.get("version", "1.0.0")),
        description=raw.get("description", ""),
        task=sections["task"],
        recipe=sections["recipe"],
        output=sections["output"],
        validation=sections["validation"],
        examples=sections["examples"],
    )

