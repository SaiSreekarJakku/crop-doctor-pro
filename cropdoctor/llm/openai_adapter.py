"""OpenAI (ChatGPT) adapter. SDK imported lazily; absent SDK => unavailable."""
from __future__ import annotations

import os
from typing import Optional


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None

    def available(self) -> bool:
        if not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return True

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        client = self._ensure()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
