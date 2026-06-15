"""Faithfulness / grounding verification.

This is the safety backstop that makes a generative layer acceptable in an
agronomy context: *no actionable advice may appear in the output unless it is
traceable to the knowledge base*. The LLM is allowed to rephrase, reorder and
summarize, but it may not introduce a new chemical, a new dosage, or a new
treatment the KB never authorized.

We verify two things:
  1. Every action item (immediate step / treatment / prevention) is grounded in
     a KB phrase by token-overlap.
  2. The free-text summary does not mention a treatment agent (a named fungicide,
     pesticide class, etc.) that the KB entry does not contain.

If verification fails, the caller discards the LLM output and falls back to the
deterministic, KB-verbatim composer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..kb.loader import KBEntry

_STOP = {
    "a", "an", "the", "to", "of", "and", "or", "for", "with", "in", "on", "at",
    "per", "by", "from", "as", "is", "are", "do", "not", "your", "you", "it",
    "that", "this", "than", "into", "out", "up", "well", "where", "possible",
    "label", "according", "apply", "use", "remove", "avoid",
}

# Treatment agents we care about not being hallucinated into the summary.
_TREATMENT_LEXICON = [
    "copper", "chlorothalonil", "mancozeb", "captan", "sulfur", "myclobutanil",
    "azoxystrobin", "propiconazole", "neem", "bacillus", "insecticidal soap",
    "horticultural oil", "fungicide", "pesticide", "miticide", "imidacloprid",
    "streptomycin", "fixed copper",
]


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def _containment(a: set, b: set) -> float:
    """Fraction of `a`'s tokens also present in `b` (asymmetric overlap)."""
    if not a:
        return 1.0
    return len(a & b) / len(a)


def _phrase_grounded(phrase: str, kb_phrases: List[str], threshold: float) -> bool:
    pt = _tokens(phrase)
    if not pt:
        return True
    return any(_containment(pt, _tokens(k)) >= threshold for k in kb_phrases)


def verify_guidance(
    summary: str,
    immediate_steps: List[str],
    treatment_options: List[str],
    prevention: List[str],
    entry: KBEntry,
    *,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    kb_phrases = entry.all_treatment_phrases + [entry.summary]
    kb_blob = " ".join(kb_phrases).lower()

    ungrounded: List[str] = []
    for label, items in (
        ("immediate_steps", immediate_steps),
        ("treatment_options", treatment_options),
        ("prevention", prevention),
    ):
        for item in items:
            if not _phrase_grounded(item, kb_phrases, threshold):
                ungrounded.append(f"{label}: {item!r}")

    # Hallucinated treatment agents in the free-text summary.
    hallucinated_agents: List[str] = []
    for agent in _TREATMENT_LEXICON:
        if agent in summary.lower() and agent not in kb_blob:
            hallucinated_agents.append(agent)

    faithful = not ungrounded and not hallucinated_agents
    return {
        "faithful": faithful,
        "ungrounded_items": ungrounded,
        "hallucinated_agents": hallucinated_agents,
        "threshold": threshold,
        "sources": [entry.source_id],
    }
