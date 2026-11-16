"""Agents package for ARES."""

from ares.agents.attacker import AttackerAgent
from ares.agents.defender import DefenderAgent
from ares.agents.llm_client import LocalLLMClient
from ares.agents.prompt_loader import load_prompt_config

__all__ = ["AttackerAgent", "DefenderAgent", "LocalLLMClient", "load_prompt_config"]

