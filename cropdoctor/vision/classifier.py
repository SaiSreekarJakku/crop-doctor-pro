"""The vision layer: turn an actual leaf image into a calibrated distribution.

This is the part the reference project left as a stub. Here we provide:

* ``Classifier`` — a tiny protocol (the same plug-in contract).
* ``StubClassifier`` — deterministic, no-ML mock for tests / offline demos.
* ``HFClassifier`` — a REAL model: a pretrained PlantVillage image classifier
  pulled from the Hugging Face Hub, run on the CPU, with temperature scaling.

Swapping a self-trained checkpoint in is just a different ``model_id`` (a local
path works too, see ``train/train.py``); everything downstream is unchanged.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

import numpy as np
from PIL import Image

from ..labels import parse_label
from ..types import ClassProb, VisionResult
from .calibration import normalized_entropy, softmax
from .quality import assess

DEFAULT_MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"


def _open_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _result_from_logits(
    logits: Sequence[float],
    labels: Sequence[str],
    temperature: float,
    backend: str,
    quality: dict,
) -> VisionResult:
    probs = softmax(logits, temperature=temperature)
    order = np.argsort(probs)[::-1]
    ranked: List[ClassProb] = []
    for i in order:
        crop, disease, kb_key, healthy = parse_label(labels[i])
        ranked.append(
            ClassProb(
                label=labels[i],
                prob=float(probs[i]),
                crop=crop,
                disease=disease,
                kb_key=kb_key,
                healthy=healthy,
            )
        )
    return VisionResult(
        top=ranked[0],
        ranked=ranked,
        entropy=normalized_entropy(probs),
        temperature=temperature,
        backend=backend,
        quality=quality,
    )


class Classifier(Protocol):
    def classify(self, image) -> VisionResult: ...


class StubClassifier:
    """Deterministic mock: returns a fixed distribution, ignores the image.

    Unlike the reference project, the stub is only for tests/offline demos — the
    real path is ``HFClassifier``. Construct from raw ``(label, logit)`` pairs.
    """

    def __init__(self, logits_by_label: Sequence[tuple], temperature: float = 1.0):
        self._labels = [l for l, _ in logits_by_label]
        self._logits = [float(v) for _, v in logits_by_label]
        self._temperature = temperature

    def classify(self, image=None) -> VisionResult:
        quality = {}
        if image is not None:
            try:
                quality = assess(_open_image(image))
            except Exception:
                quality = {}
        return _result_from_logits(
            self._logits, self._labels, self._temperature, "stub", quality
        )


class HFClassifier:
    """Real classifier backed by a Hugging Face image-classification model."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.device = device
        self._model = None
        self._processor = None
        self._labels: Optional[List[str]] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imported lazily so the package (CLI, KB, gate, tests) works without
        # torch/transformers installed.
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        id2label = self._model.config.id2label
        self._labels = [id2label[i] for i in range(len(id2label))]

    @property
    def labels(self) -> List[str]:
        self._ensure_loaded()
        return list(self._labels or [])

    def classify(self, image) -> VisionResult:
        import torch

        self._ensure_loaded()
        pil = _open_image(image)
        quality = assess(pil)
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0].cpu().numpy().tolist()
        return _result_from_logits(
            logits, self._labels, self.temperature, f"hf:{self.model_id}", quality
        )


def load_classifier(
    backend: str = "hf",
    model_id: str = DEFAULT_MODEL_ID,
    temperature: float = 1.0,
) -> Classifier:
    """Factory used by the pipeline/CLI/API."""
    if backend == "stub":
        # A harmless default distribution for smoke tests.
        return StubClassifier(
            [("Tomato___Early_blight", 3.0), ("Tomato___Late_blight", 0.5),
             ("Tomato___healthy", 0.1)],
            temperature=temperature,
        )
    return HFClassifier(model_id=model_id, temperature=temperature)
