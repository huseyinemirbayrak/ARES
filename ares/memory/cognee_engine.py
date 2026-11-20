"""Genuine Cognee SDK Engine wired for local Ollama or OpenAI-compatible LLM & Embeddings.

Executes real `cognee.add`, `cognee.cognify`, `cognee.search`, and `cognee.prune`
against isolated local `cognee_system` / `cognee_data` directories inside `store_dir`.
"""

from __future__ import annotations

import asyncio
import atexit
import importlib.util
import inspect
import logging
import os
import threading
import weakref
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_HF_TOKENIZERS: dict[str, str] = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


class RealCogneeEngine:
    """Thin, version-resilient adapter over the official `cognee` Python SDK."""

    ATTACKER_DATASET = "ares_attacker_memory"
    DEFENDER_DATASET = "ares_defender_memory"

    def __init__(
        self,
        store_dir: str | Path,
        llm_provider: str = "ollama",
        llm_model: str = "qwen2.5-coder:7b",
        llm_endpoint: str = "http://localhost:11434",
        llm_api_key: str = "ollama",
        embedding_provider: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        embedding_endpoint: str | None = None,
        embedding_dimensions: int = 768,
        huggingface_tokenizer: str | None = None,
        search_type: str = "CHUNKS",
    ) -> None:
        self.store_dir = Path(store_dir).resolve()
        self.system_root = self.store_dir / "cognee_system"
        self.data_root = self.store_dir / "cognee_data"
        raw_llm_provider = llm_provider.lower()
        # Cognee expects "openai" (with custom LLM_ENDPOINT) for OpenAI-compatible servers
        self.llm_provider = "openai" if raw_llm_provider == "openai_compatible" else raw_llm_provider
        self.llm_model = llm_model
        # LiteLLM native Ollama provider expects bare host URL (no trailing `/v1`)
        cleaned_endpoint = llm_endpoint.rstrip("/")
        if self.llm_provider == "ollama" and cleaned_endpoint.endswith("/v1"):
            cleaned_endpoint = cleaned_endpoint[:-3]
        self.llm_endpoint = cleaned_endpoint
        if llm_api_key and llm_api_key != "ollama":
            self.llm_api_key = llm_api_key
        elif self.llm_provider == "ollama":
            # Never leak host OPENAI_API_KEY into local Ollama configuration
            self.llm_api_key = "ollama"
        else:
            env_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
            self.llm_api_key = env_key if env_key and env_key != "ollama" else "ollama"
        raw_embed_provider = embedding_provider.lower()
        self.embedding_provider = (
            "openai" if raw_embed_provider == "openai_compatible" else raw_embed_provider
        )
        self.embedding_model = embedding_model
        if embedding_endpoint:
            self.embedding_endpoint = embedding_endpoint.rstrip("/")
        elif self.embedding_provider == "ollama":
            ollama_base = (
                self.llm_endpoint
                if self.llm_provider == "ollama"
                else "http://localhost:11434"
            )
            self.embedding_endpoint = f"{ollama_base}/api/embed"
        else:
            self.embedding_endpoint = self.llm_endpoint
        self.embedding_dimensions = embedding_dimensions
        embed_base = self.embedding_model.split(":")[0].lower()
        self.huggingface_tokenizer = (
            huggingface_tokenizer
            or os.environ.get("HUGGINGFACE_TOKENIZER")
            or _DEFAULT_HF_TOKENIZERS.get(embed_base, "")
        )
        self.search_type = search_type.upper()

        self._configured = False
        self._cognee_mod: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self.last_error: str | None = None
        self.errors: list[str] = []

        self_ref = weakref.ref(self)

        def _cleanup_loop() -> None:
            inst = self_ref()
            if inst is not None:
                inst.close()

        atexit.register(_cleanup_loop)

    @staticmethod
    def is_package_installed() -> bool:
        """Return True iff the real `cognee` package is installed in the current environment."""
        return importlib.util.find_spec("cognee") is not None

    def _openai_models_url(self) -> str:
        """Build `/v1/models` URL without duplicating `/v1`."""
        if self.llm_endpoint.endswith("/v1"):
            return f"{self.llm_endpoint}/models"
        return f"{self.llm_endpoint}/v1/models"

    def check_readiness(self) -> tuple[bool, str]:
        """Verify BOTH `cognee` package installation AND live local/remote LLM + embedding reachability."""
        if not self.is_package_installed():
            return (
                False,
                "Package `cognee` is not installed. Install via `uv sync --extra cognee` or `pip install -e \".[cognee]\"`.",
            )

        if self.llm_provider == "ollama":
            try:
                resp = httpx.get(f"{self.llm_endpoint}/api/tags", timeout=2.0)
                if resp.status_code != 200:
                    return False, f"Ollama endpoint {self.llm_endpoint} returned HTTP {resp.status_code}."
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                if not models:
                    return False, f"Ollama at {self.llm_endpoint} has no pulled models."
                if self.embedding_provider == "ollama":
                    embed_base = self.embedding_model.split(":")[0]
                    has_embed = any(embed_base in m for m in models)
                    if not has_embed:
                        return (
                            False,
                            f"Ollama is running, but embedding model '{self.embedding_model}' is not pulled. "
                            f"Run: `ollama pull {self.embedding_model}`",
                        )
                return True, f"Cognee ready via Ollama ({self.llm_model} + {self.embedding_model})"
            except Exception as exc:
                return False, f"Ollama endpoint {self.llm_endpoint} unreachable: {exc}"

        if self.llm_provider in ("openai", "openai_compatible"):
            has_key = bool(self.llm_api_key and self.llm_api_key != "ollama")
            # If hybrid mode uses local Ollama embeddings with a remote/OpenAI-compatible LLM, verify Ollama too
            if self.embedding_provider == "ollama":
                ollama_base = self.embedding_endpoint.replace("/api/embed", "").rstrip("/")
                try:
                    er = httpx.get(f"{ollama_base}/api/tags", timeout=2.0)
                    if er.status_code != 200:
                        return False, f"Ollama embedding endpoint {ollama_base} returned HTTP {er.status_code}."
                    em_models = [m.get("name", "") for m in er.json().get("models", [])]
                    embed_base = self.embedding_model.split(":")[0]
                    if not any(embed_base in m for m in em_models):
                        return (
                            False,
                            f"Ollama embedding model '{self.embedding_model}' not pulled at {ollama_base}. "
                            f"Run: `ollama pull {self.embedding_model}`",
                        )
                except Exception as exc:
                    return False, f"Ollama embedding endpoint {ollama_base} unreachable: {exc}"

            if has_key:
                return True, f"Cognee ready via {self.llm_provider} ({self.llm_model} + {self.embedding_model})"

            headers = {"Authorization": f"Bearer {self.llm_api_key}"} if has_key else {}
            try:
                resp = httpx.get(self._openai_models_url(), headers=headers, timeout=2.0)
                if resp.status_code == 200:
                    return True, f"Cognee ready via {self.llm_endpoint} ({self.llm_model})"
                return (
                    False,
                    f"OpenAI-compatible endpoint {self._openai_models_url()} returned HTTP {resp.status_code}.",
                )
            except Exception as exc:
                return False, f"OpenAI-compatible endpoint {self.llm_endpoint} unreachable: {exc}"

        return True, f"Cognee ready via provider '{self.llm_provider}'"

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Create (or reuse) a single persistent background event loop for Cognee async SDK calls."""
        with self._loop_lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop
            loop = asyncio.new_event_loop()
            ready_evt = threading.Event()

            def _runner() -> None:
                asyncio.set_event_loop(loop)
                ready_evt.set()
                loop.run_forever()

            thread = threading.Thread(
                target=_runner, name="ares-cognee-loop", daemon=True
            )
            thread.start()
            ready_evt.wait(timeout=5.0)
            self._loop = loop
            self._loop_thread = thread
            return loop

    def _run_async(self, coro: Any) -> Any:
        """Execute a Cognee coroutine on the persistent event loop from synchronous LangGraph nodes."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def close(self) -> None:
        """Gracefully stop the background Cognee event loop thread."""
        with self._loop_lock:
            loop = self._loop
            thread = self._loop_thread
            self._loop = None
            self._loop_thread = None
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
            try:
                loop.close()
            except RuntimeError:
                pass

    def _ensure_configured(self) -> Any:
        """Apply env vars and Cognee programmatic config before importing/running SDK methods."""
        if self._configured and self._cognee_mod is not None:
            return self._cognee_mod

        self.system_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

        # Set environment variables BEFORE importing cognee so internal singletons read them
        os.environ["TELEMETRY_DISABLED"] = "true"
        os.environ["SYSTEM_ROOT_DIRECTORY"] = str(self.system_root)
        os.environ["DATA_ROOT_DIRECTORY"] = str(self.data_root)
        os.environ["LLM_PROVIDER"] = self.llm_provider
        os.environ["LLM_MODEL"] = self.llm_model
        os.environ["LLM_ENDPOINT"] = self.llm_endpoint
        os.environ["LLM_API_KEY"] = self.llm_api_key
        os.environ["EMBEDDING_PROVIDER"] = self.embedding_provider
        os.environ["EMBEDDING_MODEL"] = self.embedding_model
        os.environ["EMBEDDING_ENDPOINT"] = self.embedding_endpoint
        os.environ["EMBEDDING_DIMENSIONS"] = str(self.embedding_dimensions)
        if self.huggingface_tokenizer:
            os.environ["HUGGINGFACE_TOKENIZER"] = self.huggingface_tokenizer

        import cognee  # type: ignore

        cfg = getattr(cognee, "config", None)
        if cfg is not None:
            if hasattr(cfg, "system_root_directory"):
                cfg.system_root_directory(str(self.system_root))
            if hasattr(cfg, "data_root_directory"):
                cfg.data_root_directory(str(self.data_root))
            if hasattr(cfg, "set_llm_config"):
                try:
                    cfg.set_llm_config(
                        {
                            "llm_provider": self.llm_provider,
                            "llm_model": self.llm_model,
                            "llm_endpoint": self.llm_endpoint,
                            "llm_api_key": self.llm_api_key,
                        }
                    )
                except Exception as exc:
                    logger.debug("cognee.config.set_llm_config skipped: %s", exc)
            if hasattr(cfg, "set_embedding_config"):
                try:
                    cfg.set_embedding_config(
                        {
                            "embedding_provider": self.embedding_provider,
                            "embedding_model": self.embedding_model,
                            "embedding_endpoint": self.embedding_endpoint,
                            "embedding_dimensions": self.embedding_dimensions,
                        }
                    )
                except Exception as exc:
                    logger.debug("cognee.config.set_embedding_config skipped: %s", exc)

        self._cognee_mod = cognee
        self._configured = True
        return cognee

    def add_lessons(self, items: list[tuple[str, str]]) -> bool:
        """Stage one or more `(dataset_name, text)` lessons via `cognee.add(...)`."""
        if not items:
            return True
        try:
            cognee = self._ensure_configured()

            async def _do_add() -> None:
                add_sig = inspect.signature(cognee.add)
                use_kw = "dataset_name" in add_sig.parameters
                for dataset_name, lesson_text in items:
                    if use_kw:
                        await cognee.add(lesson_text, dataset_name=dataset_name)
                    else:
                        await cognee.add(lesson_text, dataset_name)

            self._run_async(_do_add())
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = f"Cognee add failed: {type(exc).__name__}: {exc}"
            self.errors.append(self.last_error)
            logger.warning("%s", self.last_error)
            return False

    def cognify_datasets(self, datasets: list[str]) -> bool:
        """Run a single batched `cognee.cognify(...)` pass across all target datasets."""
        if not datasets:
            return True
        try:
            cognee = self._ensure_configured()

            async def _do_cognify() -> None:
                cog_sig = inspect.signature(cognee.cognify)
                if "datasets" in cog_sig.parameters:
                    await cognee.cognify(datasets=datasets)
                elif len(cog_sig.parameters) >= 1:
                    await cognee.cognify(datasets)
                else:
                    await cognee.cognify()

            self._run_async(_do_cognify())
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = f"Cognee cognify failed: {type(exc).__name__}: {exc}"
            self.errors.append(self.last_error)
            logger.warning("%s", self.last_error)
            return False

    def ingest_lesson(self, dataset_name: str, lesson_text: str) -> bool:
        """Run real `cognee.add(...)` followed by `cognee.cognify(...)` on a single dataset."""
        if not self.add_lessons([(dataset_name, lesson_text)]):
            return False
        return self.cognify_datasets([dataset_name])

    def search_lessons(
        self, dataset_name: str, query_text: str, top_k: int = 4
    ) -> list[str]:
        """Run real `cognee.search(...)` (CHUNKS / GRAPH_COMPLETION / SUMMARIES) and format results."""
        try:
            cognee = self._ensure_configured()

            search_type_enum = getattr(cognee, "SearchType", None)
            if search_type_enum is None:
                try:
                    from cognee.api.v1.search import SearchType as _ST  # type: ignore

                    search_type_enum = _ST
                except Exception:
                    search_type_enum = None

            resolved_type: Any = self.search_type
            if search_type_enum is not None and hasattr(search_type_enum, self.search_type):
                resolved_type = getattr(search_type_enum, self.search_type)

            async def _do_search() -> Any:
                sig = inspect.signature(cognee.search)
                kwargs: dict[str, Any] = {}
                if "query_type" in sig.parameters:
                    kwargs["query_type"] = resolved_type
                elif "search_type" in sig.parameters:
                    kwargs["search_type"] = resolved_type
                if "query_text" in sig.parameters:
                    kwargs["query_text"] = query_text
                if "datasets" in sig.parameters:
                    kwargs["datasets"] = [dataset_name]
                if "top_k" in sig.parameters:
                    kwargs["top_k"] = top_k

                if "query_text" in kwargs:
                    return await cognee.search(**kwargs)
                return await cognee.search(query_text, **kwargs)

            raw_results = self._run_async(_do_search())
            formatted: list[str] = []
            for item in (raw_results or [])[:top_k]:
                text = self._extract_result_text(item)
                if text:
                    formatted.append(f"[Real Cognee {self.search_type}] {text}")
            self.last_error = None
            return formatted
        except Exception as exc:
            self.last_error = f"Cognee search failed: {type(exc).__name__}: {exc}"
            self.errors.append(self.last_error)
            logger.warning("%s", self.last_error)
            return []

    @staticmethod
    def _extract_result_text(item: Any) -> str:
        """Extract clean text from diverse Cognee search result objects."""
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("text", "chunk_text", "summary", "answer", "content", "payload"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return str(item).strip()
        for attr in ("text", "chunk_text", "summary", "content"):
            val = getattr(item, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return str(item).strip()

    def prune(self) -> bool:
        """Reset real Cognee data and metadata stores cleanly."""
        try:
            cognee = self._ensure_configured()

            async def _do_prune() -> None:
                if hasattr(cognee, "prune"):
                    if hasattr(cognee.prune, "prune_data"):
                        await cognee.prune.prune_data()
                    if hasattr(cognee.prune, "prune_system"):
                        await cognee.prune.prune_system(metadata=True)

            self._run_async(_do_prune())
            return True
        except Exception as exc:
            self.last_error = f"Cognee prune failed: {type(exc).__name__}: {exc}"
            logger.warning("%s", self.last_error)
            return False
