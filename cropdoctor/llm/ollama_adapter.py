"""Ollama adapter — local models AND Ollama Cloud (e.g. gpt-oss:120b-cloud).

Talks to an Ollama-compatible ``/api/chat`` endpoint over HTTP using only the
stdlib (no extra SDK). Works in three modes via env vars:

* **Local daemon** (default): ``OLLAMA_HOST=http://localhost:11434``,
  ``OLLAMA_MODEL=llama3.1``.
* **Local daemon proxying a cloud model**: same host, but signed in
  (``ollama signin``) with ``OLLAMA_MODEL=gpt-oss:120b-cloud``.
* **Ollama Cloud API directly** (no local install): ``OLLAMA_HOST=https://ollama.com``,
  ``OLLAMA_API_KEY=<key from ollama.com/settings/keys>``,
  ``OLLAMA_MODEL=gpt-oss:120b``.

When ``OLLAMA_API_KEY`` is set we send ``Authorization: Bearer <key>`` so the
same code path serves both local and cloud.
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
        self.api_key = os.getenv("OLLAMA_API_KEY")

    def _headers(self, extra: Optional[dict] = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            headers.update(extra)
        return headers

    def available(self) -> bool:
        # Cloud API: presence of a key + cloud host is enough; avoid a slow probe.
        if self.api_key and self.host.startswith("https://"):
            return True
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", headers=self._headers())
            with urllib.request.urlopen(req, timeout=3) as r:
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
            f"{self.host}/api/chat", data=data, headers=self._headers()
        )
        # Cloud 120B models can take a while on first token; allow generous timeout.
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read().decode("utf-8"))
        return body.get("message", {}).get("content", "")
