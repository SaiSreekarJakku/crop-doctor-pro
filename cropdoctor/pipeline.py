"""End-to-end orchestration: image -> vision -> gate -> KB -> guidance.

This is the public entry point used by both the CLI and the web API. It composes
the layers but contains no ML or vendor code itself, so it stays trivially
testable with the stub classifier.
"""
from __future__ import annotations

from typing import Optional

from .gate.gate import GateConfig, evaluate
from .guidance.compose import compose_guidance
from .kb.loader import KnowledgeBase, load_kb
from .llm.provider import LLMProvider, get_provider
from .types import Diagnosis, VisionResult
from .vision.classifier import Classifier


class CropDoctor:
    def __init__(
        self,
        classifier: Classifier,
        kb: Optional[KnowledgeBase] = None,
        gate_config: Optional[GateConfig] = None,
        provider: Optional[LLMProvider] = None,
    ):
        self.classifier = classifier
        self.kb = kb or load_kb()
        self.gate_config = gate_config or GateConfig()
        self.provider = provider or get_provider()

    def diagnose(self, image, user_crop: Optional[str] = None,
                 user_region: Optional[str] = None) -> Diagnosis:
        vision = self.classifier.classify(image)
        return self._finish(vision, user_crop, user_region)

    def diagnose_vision(self, vision: VisionResult, user_crop: Optional[str] = None,
                        user_region: Optional[str] = None) -> Diagnosis:
        """Diagnose from an already-computed VisionResult (used in tests)."""
        return self._finish(vision, user_crop, user_region)

    def _finish(self, vision: VisionResult, user_crop, user_region) -> Diagnosis:
        gate = evaluate(vision, self.gate_config, user_crop=user_crop)
        top = vision.top

        prediction = {
            "disease": top.disease,
            "confidence": round(top.prob, 4),
            "crop": top.crop,
            "top_3": [
                {"disease": c.disease, "crop": c.crop, "prob": round(c.prob, 4)}
                for c in vision.top_k(3)
            ],
            "backend": vision.backend,
            "temperature": vision.temperature,
            "entropy": round(vision.entropy, 4),
        }

        # Abstain: return prediction context but NO treatment guidance.
        if gate.abstain:
            return Diagnosis(
                crop=user_crop or top.crop,
                prediction=prediction,
                abstained=True,
                gate=gate.to_dict(),
                guidance=None,
            )

        entry = self.kb.get(top.kb_key)
        if entry is None:
            # Confident class but no curated KB entry: abstain rather than invent.
            gate.abstain = True
            gate.reasons.append(
                f"no_kb_entry: '{top.kb_key}' is not in the curated knowledge base"
            )
            return Diagnosis(
                crop=user_crop or top.crop,
                prediction=prediction,
                abstained=True,
                gate=gate.to_dict(),
                guidance=None,
            )

        guidance = compose_guidance(entry, top.prob, provider=self.provider)
        return Diagnosis(
            crop=top.crop,
            prediction=prediction,
            abstained=False,
            gate=gate.to_dict(),
            guidance=guidance.to_dict(),
        )
