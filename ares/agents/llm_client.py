"""Local LLM Client supporting Ollama, OpenAI-compatible local servers (LM Studio/vLLM), and Adaptive Local Mode."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
import httpx

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_HINTS = (
    "embed",
    "nomic",
    "bge-",
    "mxbai",
    "minilm",
    "e5-",
    "rerank",
)


def _is_embedding_model(name: str) -> bool:
    """Return True if an Ollama model name looks like an embedding-only model."""
    lowered = name.lower()
    return any(hint in lowered for hint in _EMBEDDING_MODEL_HINTS)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Robustly extract a JSON object from an LLM response, handling multi-fence outputs and nested braces."""
    cleaned = raw_text.strip()
    if not cleaned:
        raise json.JSONDecodeError("Empty LLM response", raw_text, 0)

    # 1. Fast path: entire response is valid JSON
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Check individual fenced code blocks (prefer last block / blocks with ARES payload keys)
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    candidates: list[dict[str, Any]] = []
    for block in reversed(fenced_blocks):
        try:
            obj = json.loads(block.strip())
            if isinstance(obj, dict):
                if "mutated_code" in obj or "repaired_code" in obj:
                    return obj
                candidates.append(obj)
        except json.JSONDecodeError:
            continue

    # 3. Balanced JSONDecoder scan across all '{' positions in raw text
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned, match.start())
            if isinstance(obj, dict):
                if "mutated_code" in obj or "repaired_code" in obj:
                    return obj
                candidates.append(obj)
        except json.JSONDecodeError:
            continue

    if candidates:
        return candidates[0]

    raise json.JSONDecodeError("No valid JSON object found in LLM response", cleaned, 0)


def infer_llm_provider(
    llm_mode: str,
    base_url: str,
    api_key: str | None = None,
) -> str:
    """Infer whether an endpoint speaks Ollama native API or OpenAI-compatible Chat Completions."""
    mode = llm_mode.strip().lower()
    if mode == "openai_compatible":
        return "openai_compatible"
    if mode == "ollama":
        return "ollama"
    if mode == "adaptive_local":
        return "ollama"

    # mode == "auto": detect OpenAI-compatible URLs / API keys automatically
    url_lower = base_url.strip().rstrip("/").lower()
    if api_key:
        return "openai_compatible"
    if url_lower.endswith("/v1") or "/v1/" in url_lower:
        return "openai_compatible"
    if url_lower.startswith("https://"):
        return "openai_compatible"
    # Non-Ollama local ports (e.g. LM Studio :1234, vLLM :8000, llama.cpp :8080)
    if ":11434" not in url_lower and "ollama" not in url_lower:
        return "openai_compatible"
    return "ollama"


class LocalLLMClient:
    """Client for local/remote LLMs (Ollama / LM Studio / vLLM / OpenAI-compatible) with safe model matching."""

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        mode: str = "auto",
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        num_ctx: int = 8192,
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.mode = mode.lower()
        self.timeout_seconds = timeout_seconds
        self.num_ctx = max(2048, int(num_ctx))
        self.api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self._server_reachable: bool | None = None
        self.last_error: str | None = None

    def _openai_url(self, suffix: str) -> str:
        """Build an OpenAI-compatible endpoint URL without duplicating `/v1`."""
        clean_suffix = suffix.lstrip("/")
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/{clean_suffix}"
        return f"{self.base_url}/v1/{clean_suffix}"

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def is_live_llm_available(self) -> bool:
        """Check once if the local LLM endpoint is reachable and has a usable chat model."""
        if self.mode == "adaptive_local":
            return False
        if self._server_reachable is not None:
            return self._server_reachable

        try:
            if self.provider == "ollama":
                resp = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    names = [m.get("name", "") for m in models if m.get("name")]
                    chat_models = [n for n in names if not _is_embedding_model(n)]
                    if not chat_models:
                        self.last_error = (
                            f"Ollama at {self.base_url} has no chat models (found only: {names})."
                        )
                        logger.warning("%s", self.last_error)
                        self._server_reachable = False
                        return False

                    if self.model_name in chat_models:
                        self._server_reachable = True
                        return True
                    # Match `<model>:latest` or tag prefix cleanly
                    for candidate in chat_models:
                        if candidate == f"{self.model_name}:latest" or candidate.split(":")[0] == self.model_name:
                            self.model_name = candidate
                            self._server_reachable = True
                            return True

                    # Fall back ONLY to a verified non-embedding chat model and log a clear warning
                    chosen = chat_models[0]
                    logger.warning(
                        "Configured Ollama model '%s' not found; falling back to chat model '%s' (available: %s).",
                        self.model_name,
                        chosen,
                        chat_models,
                    )
                    self.model_name = chosen
                    self._server_reachable = True
                    return True
                # In auto mode, fall through to probe OpenAI-compatible `/v1/models` if Ollama `/api/tags` wasn't 200
                if self.mode == "auto":
                    alt = httpx.get(
                        self._openai_url("models"),
                        headers=self._auth_headers(),
                        timeout=2.0,
                    )
                    if alt.status_code == 200:
                        self.provider = "openai_compatible"
                        self._server_reachable = True
                        return True
            else:
                resp = httpx.get(
                    self._openai_url("models"),
                    headers=self._auth_headers(),
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    self._server_reachable = True
                    return True
                # Some hosted OpenAI-compatible gateways omit GET /models; allow if key set
                if self.api_key and resp.status_code in (404, 405):
                    self._server_reachable = True
                    return True
                self.last_error = f"OpenAI-compatible endpoint returned HTTP {resp.status_code}"
                self._server_reachable = False
                return False
        except Exception as exc:
            self.last_error = f"LLM endpoint {self.base_url} unreachable: {exc}"
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
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": self.num_ctx,
                    },
                }
                resp = httpx.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "")
                self.last_error = None
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
                    self._openai_url("chat/completions"),
                    headers=self._auth_headers(),
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
                self.last_error = None
                return extract_json_object(content)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "LocalLLMClient (%s/%s) call failed: %s",
                self.provider,
                self.model_name,
                self.last_error,
            )
            return None
