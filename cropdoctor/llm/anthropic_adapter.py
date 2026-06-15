"""Anthropic (Claude) adapter. SDK imported lazily; absent SDK => unavailable."""
from __future__ import annotations

import os
from typing import Optional


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._client = None

    def available(self) -> bool:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return True

    def _ensure(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        client = self._ensure()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)
