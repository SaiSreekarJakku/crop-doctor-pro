"""Provider-agnostic LLM layer.

The reference project has no LLM at all. We add one, but we refuse to marry a
single vendor. A provider is just something that can turn a (system, user)
prompt into text. Adapters exist for OpenAI (ChatGPT), Anthropic (Claude) and
Ollama (local/open models); a ``deterministic`` no-LLM path is always available
so the system runs with zero keys and zero network.

Selection order:
1. explicit ``provider`` argument / ``--provider`` flag,
2. ``LLM_PROVIDER`` env var,
3. auto-detect whichever vendor key is present in the environment,
4. fall back to ``deterministic``.

Whatever the choice, the LLM only *rephrases* guidance that already exists in
the knowledge base, and the faithfulness layer verifies the output. The provider
is swappable; the safety guarantees are not.
"""
from __future__ import annotations

import os
from typing import List, Optional, Protocol


class LLMProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str: ...


class DeterministicProvider:
    """Not a real LLM — signals the composer to use template assembly."""

    name = "deterministic"

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        raise NotImplementedError("deterministic provider does not call an LLM")


def _registry():
    # Imported here to keep optional SDKs lazy.
    from .openai_adapter import OpenAIProvider
    from .anthropic_adapter import AnthropicProvider
    from .ollama_adapter import OllamaProvider

    return {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "deterministic": DeterministicProvider,
        "none": DeterministicProvider,
    }


def available_providers() -> List[str]:
    return list(_registry().keys())


def _auto_detect() -> Optional[str]:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_MODEL"):
        return "ollama"
    return None


def get_provider(name: Optional[str] = None, *, model: Optional[str] = None) -> LLMProvider:
    """Resolve a provider, gracefully degrading to deterministic.

    Never raises for a missing key/SDK: an unavailable provider falls back to the
    deterministic composer so the pipeline always produces grounded output.
    """
    choice = (name or os.getenv("LLM_PROVIDER") or _auto_detect() or "deterministic").lower()
    reg = _registry()
    cls = reg.get(choice)
    if cls is None:
        return DeterministicProvider()
    try:
        provider = cls(model=model) if choice in ("openai", "anthropic", "ollama") else cls()
    except Exception:
        return DeterministicProvider()
    if not provider.available():
        return DeterministicProvider()
    return provider
