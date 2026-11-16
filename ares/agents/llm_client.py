"""Local LLM Client supporting Ollama, OpenAI-compatible local servers (LM Studio/vLLM), and Adaptive Local Mode."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
import httpx

logger = logging.getLogger(__name__)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from an LLM response (handling markdown code fences)."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]
    return json.loads(cleaned)


class LocalLLMClient:
    """Client for local LLMs (Ollama / LM Studio / vLLM) with health-check caching."""

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        mode: str = "auto",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.mode = mode.lower()
        self.timeout_seconds = timeout_seconds
        self._server_reachable: bool | None = None
        self.last_error: str | None = None

    def is_live_llm_available(self) -> bool:
        """Check once if the local LLM endpoint is reachable and has models."""
        if self.mode == "adaptive_local":
            return False
        if self._server_reachable is not None:
            return self._server_reachable

        try:
            if self.provider == "ollama":
                resp = httpx.get(f"{self.base_url}/api/tags", timeout=1.5)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    self._server_reachable = len(models) > 0
                    # If configured model isn't present, pick first available local model
                    names = [m.get("name", "") for m in models]
                    if self._server_reachable and self.model_name not in names and names:
                        self.model_name = names[0]
                    return self._server_reachable
            else:
                resp = httpx.get(f"{self.base_url}/v1/models", timeout=1.5)
                self._server_reachable = resp.status_code == 200
                return self._server_reachable
        except Exception:
            self._server_reachable = False
        return False

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        """Call local LLM and parse JSON response; returns None if offline or unparseable."""
        if not self.is_live_llm_available():
            return None

        try:
            if self.provider == "ollama":
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": self.temperature},
                }
                resp = httpx.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "")
                return extract_json_object(content)
            else:
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.temperature,
                }
                resp = httpx.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()
                content = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return extract_json_object(content)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "LocalLLMClient (%s/%s) call failed, falling back to adaptive_local: %s",
                self.provider,
                self.model_name,
                self.last_error,
            )
            return None

