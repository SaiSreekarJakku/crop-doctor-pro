"""Ollama adapter for local / open-source models (Llama, Mistral, Qwen, ...).

Talks to a local Ollama server over HTTP using only the stdlib, so no extra SDK
is required. Set ``OLLAMA_MODEL`` (e.g. ``llama3.1``) and optionally
``OLLAMA_HOST`` (default ``http://localhost:11434``).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read().decode("utf-8"))
        return body.get("message", {}).get("content", "")
