"""Deterministic tests for the core pipeline — no model download required.

These mirror the reference project's testability goal but exercise the REAL
calibration, gate, KB, faithfulness and guidance logic via the stub classifier.
"""
import math

import numpy as np
import pytest

from cropdoctor.gate.gate import GateConfig, evaluate
from cropdoctor.guidance.compose import compose_guidance
from cropdoctor.kb.loader import load_kb
from cropdoctor.labels import parse_label
from cropdoctor.llm.provider import DeterministicProvider, get_provider
from cropdoctor.pipeline import CropDoctor
from cropdoctor.types import VisionResult
from cropdoctor.vision.calibration import (fit_temperature, normalized_entropy,
                                           softmax)
from cropdoctor.vision.classifier import StubClassifier
from cropdoctor.faithfulness.verify import verify_guidance


# ---------- labels ----------
def test_parse_label_plantvillage():
    assert parse_label("Tomato___Early_blight") == (
        "tomato", "Early Blight", "tomato_early_blight", False)
    crop, disease, key, healthy = parse_label("Potato___healthy")
    assert crop == "potato" and healthy and key == "potato_healthy"
    crop, _, _, _ = parse_label("Corn_(maize)___Common_rust_")
    assert crop == "corn"
    crop, _, _, _ = parse_label("Pepper,_bell___Bacterial_spot")
    assert crop == "pepper"


# ---------- calibration ----------
def test_softmax_and_entropy():
    p = softmax([2.0, 1.0, 0.1])
    assert abs(p.sum() - 1.0) < 1e-9 and p[0] > p[1] > p[2]
    assert normalized_entropy([1.0, 0, 0]) == 0.0
    assert abs(normalized_entropy([0.25, 0.25, 0.25, 0.25]) - 1.0) < 1e-9


def test_temperature_softens_confidence():
    logits = [4.0, 1.0, 0.0]
    hot = softmax(logits, temperature=3.0)
    cold = softmax(logits, temperature=1.0)
    assert hot[0] < cold[0]  # higher temperature => less peaky


def test_fit_temperature_runs():
    rng = np.random.default_rng(0)
    logits = (rng.normal(size=(200, 5)) * 3).tolist()
    labels = [int(np.argmax(row)) for row in logits]
    T = fit_temperature(logits, labels)
    assert 0.05 < T < 20


# ---------- gate ----------
def _vr(probs_labels, entropy=0.2, quality=None):
    clf = StubClassifier([(l, math.log(p + 1e-9)) for l, p in probs_labels])
    vr = clf.classify(None)
    vr.entropy = entropy
    if quality is not None:
        vr.quality = quality
    return vr


def test_gate_confident_passes():
    vr = _vr([("Tomato___Early_blight", 0.9), ("Tomato___Late_blight", 0.05),
              ("Tomato___healthy", 0.05)])
    assert evaluate(vr, GateConfig()).abstain is False


def test_gate_abstains_on_near_tie():
    vr = _vr([("Tomato___Early_blight", 0.50), ("Tomato___Late_blight", 0.45),
              ("Tomato___healthy", 0.05)])
    d = evaluate(vr, GateConfig())
    assert d.abstain and any("near_tie" in r or "low_confidence" in r for r in d.reasons)


def test_gate_crop_mismatch():
    vr = _vr([("Grape___Black_rot", 0.92), ("Grape___healthy", 0.05),
              ("Tomato___Early_blight", 0.03)])
    d = evaluate(vr, GateConfig(), user_crop="tomato")
    assert d.abstain and any("crop_mismatch" in r for r in d.reasons)


def test_gate_quality_abstain():
    vr = _vr([("Tomato___Early_blight", 0.95), ("Tomato___healthy", 0.05)],
             quality={"blurry": True, "looks_like_leaf": True})
    assert evaluate(vr, GateConfig()).abstain


# ---------- KB ----------
def test_kb_loads_and_validates():
    kb = load_kb()
    assert len(kb) >= 15
    e = kb.get("tomato_early_blight")
    assert e and e.crop == "tomato" and e.treatment_options
    assert "tomato" in kb.crops() and "apple" in kb.crops()


def test_kb_entries_are_flat_string_lists():
    # Guards against the `- Foo: bar` YAML-becomes-dict trap.
    kb = load_kb()
    for key in kb.keys():
        e = kb.get(key)
        for field_items in (e.immediate_steps, e.treatment_options, e.prevention,
                            e.symptoms):
            assert all(isinstance(x, str) for x in field_items), key


# ---------- faithfulness ----------
def test_faithfulness_flags_hallucinated_agent():
    kb = load_kb()
    e = kb.get("tomato_early_blight")
    bad = verify_guidance(
        "Spray imidacloprid weekly.", ["Remove affected lower leaves"],
        ["Apply imidacloprid"], ["Mulch"], e)
    assert bad["faithful"] is False
    good = verify_guidance(
        e.summary, e.immediate_steps, e.treatment_options, e.prevention, e)
    assert good["faithful"] is True


def test_faithfulness_allows_paraphrase():
    # Genuine reword of KB content (as a real LLM produces) must pass.
    kb = load_kb()
    e = kb.get("tomato_early_blight")
    paraphrased = verify_guidance(
        "Early blight, caused by Alternaria solani, shows bullseye lesions on "
        "older lower leaves and is favored by warm humid weather.",
        ["Cut off and discard the infected lower leaves; do not add them to compost.",
         "Stop watering from overhead and water at the soil line early in the day.",
         "Increase airflow by staking plants and trimming the lower foliage."],
        ["Apply a copper-based fungicide following the label directions.",
         "For organic growers, use a Bacillus subtilis bio-fungicide as directed."],
        ["Lay mulch around the plant base to keep soil splashes off the leaves.",
         "Space plants for good air movement and select resistant varieties."],
        e)
    assert paraphrased["faithful"] is True, paraphrased


# ---------- guidance ----------
def test_deterministic_guidance_is_kb_verbatim():
    kb = load_kb()
    e = kb.get("tomato_late_blight")
    g = compose_guidance(e, 0.9, provider=DeterministicProvider())
    assert g.provider == "deterministic"
    assert g.treatment_options == e.treatment_options
    assert g.sources == [e.source_id]


def test_provider_defaults_to_deterministic_without_keys(monkeypatch):
    for k in ["LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
              "OLLAMA_HOST", "OLLAMA_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    assert get_provider().name == "deterministic"


# ---------- pipeline (end-to-end with stub) ----------
def test_pipeline_confident():
    clf = StubClassifier([("Tomato___Early_blight", 3.0),
                          ("Tomato___Late_blight", 0.2), ("Tomato___healthy", 0.1)])
    diag = CropDoctor(clf, provider=DeterministicProvider()).diagnose(None).to_dict()
    assert diag["abstained"] is False
    assert "Early Blight" in diag["prediction"]["disease"]
    assert diag["guidance"]["sources"] == ["KB-tomato-earlyblight"]
    assert diag["disclaimer"]


def test_pipeline_abstains_low_confidence():
    clf = StubClassifier([("Tomato___Early_blight", 1.0),
                          ("Tomato___Late_blight", 0.95), ("Tomato___healthy", 0.9)])
    diag = CropDoctor(clf, provider=DeterministicProvider()).diagnose(None).to_dict()
    assert diag["abstained"] is True and diag["guidance"] is None
    assert diag["gate"]["reasons"]


def test_pipeline_abstains_when_no_kb_entry():
    # Confident class that has no curated KB entry -> abstain, do not invent.
    clf = StubClassifier([("Soybean___healthy", 5.0), ("Tomato___healthy", 0.1)])
    diag = CropDoctor(clf, provider=DeterministicProvider()).diagnose(None).to_dict()
    # soybean_healthy isn't in KB -> abstain with no_kb_entry reason
    assert diag["abstained"] is True
    assert any("no_kb_entry" in r for r in diag["gate"]["reasons"])
