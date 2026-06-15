"""Knowledge-base loader.

The KB is the single source of truth for treatment guidance: one entry per
disease class, authored by hand and reviewed, stored as YAML. Every YAML file
under ``data/`` is a mapping of ``kb_key -> entry``; this loader merges them and
validates that each entry has the required fields.

Crucially, guidance can *only* come from here — the LLM layer rephrases these
fields, it never invents treatments. The faithfulness layer enforces that.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

_REQUIRED = ["crop", "disease", "summary", "immediate_steps",
             "treatment_options", "prevention", "source_id"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class KBEntry:
    kb_key: str
    crop: str
    disease: str
    summary: str
    immediate_steps: List[str]
    treatment_options: List[str]
    prevention: List[str]
    source_id: str
    symptoms: List[str] = field(default_factory=list)
    pathogen: str = ""
    healthy: bool = False

    @property
    def all_treatment_phrases(self) -> List[str]:
        """Every advice phrase that must be traceable to this entry."""
        return list(self.immediate_steps) + list(self.treatment_options) + list(self.prevention)


class KnowledgeBase:
    def __init__(self, entries: Dict[str, KBEntry]):
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def get(self, key: str) -> Optional[KBEntry]:
        return self._entries.get(key)

    def keys(self) -> List[str]:
        return sorted(self._entries.keys())

    def crops(self) -> List[str]:
        return sorted({e.crop for e in self._entries.values()})


def _str_list(kb_key: str, field_name: str, value) -> List[str]:
    """Validate a YAML list is a flat list of strings.

    Catches a sneaky YAML failure mode: ``- Foo: bar`` parses as a dict, not a
    string, which would otherwise leak ``{'Foo': 'bar'}`` into guidance output.
    """
    if not isinstance(value, list):
        raise ValueError(f"KB entry '{kb_key}' field '{field_name}' must be a list")
    out = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"KB entry '{kb_key}' field '{field_name}' has a non-string item "
                f"{item!r} — check for an unquoted ':' in the YAML"
            )
        out.append(item)
    return out


def _validate(kb_key: str, raw: dict) -> KBEntry:
    missing = [f for f in _REQUIRED if f not in raw]
    if missing:
        raise ValueError(f"KB entry '{kb_key}' missing required fields: {missing}")
    return KBEntry(
        kb_key=kb_key,
        crop=str(raw["crop"]).lower(),
        disease=str(raw["disease"]),
        summary=str(raw["summary"]).strip(),
        immediate_steps=_str_list(kb_key, "immediate_steps", raw["immediate_steps"]),
        treatment_options=_str_list(kb_key, "treatment_options", raw["treatment_options"]),
        prevention=_str_list(kb_key, "prevention", raw["prevention"]),
        source_id=str(raw["source_id"]),
        symptoms=_str_list(kb_key, "symptoms", raw.get("symptoms", [])),
        pathogen=str(raw.get("pathogen", "")),
        healthy=bool(raw.get("healthy", False)),
    )


def load_kb(data_dir: str = _DATA_DIR) -> KnowledgeBase:
    entries: Dict[str, KBEntry] = {}
    if not os.path.isdir(data_dir):
        return KnowledgeBase(entries)
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        with open(os.path.join(data_dir, fname), "r") as fh:
            doc = yaml.safe_load(fh) or {}
        for kb_key, raw in doc.items():
            if kb_key in entries:
                raise ValueError(f"Duplicate KB key '{kb_key}' in {fname}")
            entries[kb_key] = _validate(kb_key, raw)
    return KnowledgeBase(entries)
