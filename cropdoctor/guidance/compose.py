"""Assemble treatment guidance from a KB entry, optionally via an LLM.

The contract: guidance content originates in the knowledge base. The LLM (if one
is configured) may only rephrase and tailor the *summary* and re-present the KB's
action lists. We then run the faithfulness check; if it fails for any reason
(bad JSON, ungrounded item, hallucinated chemical, network error) we silently
fall back to the deterministic, KB-verbatim composer. The output is always safe.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ..faithfulness.verify import verify_guidance
from ..kb.loader import KBEntry
from ..llm.provider import DeterministicProvider, LLMProvider, get_provider
from ..types import Guidance

_SYSTEM = (
    "You are an agronomy guidance assistant. You will be given a JSON knowledge-"
    "base entry for a specific, already-diagnosed crop disease. Your ONLY job is "
    "to re-present that entry as clear advice for a grower. Strict rules:\n"
    "1. Do NOT introduce any treatment, chemical, product, or step that is not "
    "present in the provided entry. No new fungicides, no dosages.\n"
    "2. Keep the SAME number of items in each list as the entry, and keep each "
    "item a close paraphrase of the corresponding entry item — reuse its key "
    "nouns (leaves, mulch, copper, fungicide, spacing, etc.). Do not merge, "
    "split, add, or drop items. You have the most freedom in the summary.\n"
    "3. Reply with ONLY a JSON object: {\"summary\": str, \"immediate_steps\": "
    "[str], \"treatment_options\": [str], \"prevention\": [str]}."
)


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.7:
        return "moderate"
    return "borderline"


def _hedge(band: str) -> str:
    return {
        "high": "",
        "moderate": "This diagnosis is reasonably but not fully certain; "
                    "confirm symptoms before applying chemical treatments. ",
        "borderline": "Confidence is borderline — treat this as a tentative "
                      "suggestion and verify with an expert before acting. ",
    }[band]


def _deterministic(entry: KBEntry, confidence: float, provider_name: str,
                   faithfulness: dict) -> Guidance:
    band = _confidence_band(confidence)
    summary = _hedge(band) + re.sub(r"\s+", " ", entry.summary).strip()
    return Guidance(
        summary=summary,
        immediate_steps=list(entry.immediate_steps),
        treatment_options=list(entry.treatment_options),
        prevention=list(entry.prevention),
        sources=[entry.source_id],
        provider=provider_name,
        faithfulness=faithfulness or {"faithful": True, "mode": "kb_verbatim",
                                      "sources": [entry.source_id]},
    )


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    # Tolerate code fences / surrounding prose.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def compose_guidance(
    entry: KBEntry,
    confidence: float,
    provider: Optional[LLMProvider] = None,
) -> Guidance:
    provider = provider or get_provider()

    # Deterministic path: no LLM call at all.
    if isinstance(provider, DeterministicProvider):
        return _deterministic(entry, confidence, "deterministic", {})

    band = _confidence_band(confidence)
    user = json.dumps(
        {
            "confidence": round(confidence, 3),
            "confidence_band": band,
            "entry": {
                "crop": entry.crop,
                "disease": entry.disease,
                "pathogen": entry.pathogen,
                "summary": entry.summary,
                "immediate_steps": entry.immediate_steps,
                "treatment_options": entry.treatment_options,
                "prevention": entry.prevention,
            },
        },
        indent=2,
    )

    try:
        raw = provider.complete(_SYSTEM, user, temperature=0.1)
    except Exception:
        return _deterministic(entry, confidence, f"{provider.name}->fallback:error", {})

    parsed = _parse_json(raw)
    if not parsed:
        return _deterministic(entry, confidence,
                              f"{provider.name}->fallback:parse_error", {})

    summary = str(parsed.get("summary", "")).strip()
    immediate = [str(x) for x in parsed.get("immediate_steps", [])]
    treatment = [str(x) for x in parsed.get("treatment_options", [])]
    prevention = [str(x) for x in parsed.get("prevention", [])]

    fcheck = verify_guidance(summary, immediate, treatment, prevention, entry)
    if not fcheck["faithful"] or not summary:
        # The generative output drifted from the KB — discard it.
        return _deterministic(
            entry, confidence, f"{provider.name}->fallback:unfaithful", fcheck
        )

    return Guidance(
        summary=summary,
        immediate_steps=immediate,
        treatment_options=treatment,
        prevention=prevention,
        sources=[entry.source_id],
        provider=provider.name,
        faithfulness=fcheck,
    )
