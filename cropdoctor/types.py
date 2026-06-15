"""Shared, framework-agnostic data types used across the pipeline.

Keeping these in one place means the vision layer, the abstention gate, the LLM
guidance layer and the API/CLI all speak the same vocabulary without importing
each other's heavy dependencies (torch, transformers, gradio, ...).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ClassProb:
    """A single class with its probability and the KB key it maps to."""
    label: str          # raw model label, e.g. "Tomato___Early_blight"
    prob: float
    crop: str           # e.g. "tomato"
    disease: str        # human disease name, e.g. "Early Blight"
    kb_key: str         # e.g. "tomato_early_blight" (or "tomato_healthy")
    healthy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisionResult:
    """Output of the vision layer: a calibrated probability distribution."""
    top: ClassProb
    ranked: List[ClassProb]
    entropy: float                 # normalized predictive entropy in [0, 1]
    temperature: float = 1.0       # calibration temperature applied
    backend: str = "stub"          # which classifier produced this
    quality: Dict[str, Any] = field(default_factory=dict)  # image quality metrics

    def top_k(self, k: int = 3) -> List[ClassProb]:
        return self.ranked[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top": self.top.to_dict(),
            "top_3": [c.to_dict() for c in self.top_k(3)],
            "entropy": self.entropy,
            "temperature": self.temperature,
            "backend": self.backend,
            "quality": self.quality,
        }


@dataclass
class GateDecision:
    """Result of the confidence gate. `abstain=True` is a *success* state."""
    abstain: bool
    reasons: List[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Guidance:
    """Treatment guidance assembled (only) from the knowledge base."""
    summary: str
    immediate_steps: List[str]
    treatment_options: List[str]
    prevention: List[str]
    sources: List[str]
    provider: str = "deterministic"   # which LLM provider assembled it
    faithfulness: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DISCLAIMER = (
    "Demonstration tool; confirm with a local agricultural extension service "
    "or a qualified agronomist before applying any treatment."
)


@dataclass
class Diagnosis:
    """The full, serializable response of the pipeline."""
    crop: Optional[str]
    prediction: Optional[Dict[str, Any]]
    abstained: bool
    gate: Dict[str, Any]
    guidance: Optional[Dict[str, Any]]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
