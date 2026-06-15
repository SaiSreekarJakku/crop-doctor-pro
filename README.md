# 🌿 Crop Doctor Pro

Take a photo of a crop leaf → a **real vision model** classifies the disease →
the system **abstains and recommends an expert** when it isn't confident →
otherwise it produces treatment guidance assembled **only** from a curated,
source-cited agronomy knowledge base.

> The vision model makes the diagnosis; the LLM never decides the disease. The
> LLM only *assembles* treatment guidance from the KB entry for the predicted
> class, and every output is checked for faithfulness. The system abstains when
> confidence is low.

This is a from-scratch, **working** implementation of that brief — built to be
better than [a reference project](https://github.com/Zumitify/crop-disease-identifier)
that has the same goal but **stubs out the entire vision model**.

---

## Why this is better than the reference project

The reference project (`Zumitify/crop-disease-identifier`) is well-structured but
its classifier is a **stub** — `StubClassifier` ignores the image and returns
probabilities **you type in by hand** (`--pred tomato_earlyblight=0.88`). The
whole hard part — turning a leaf photo into a prediction — is missing. It is also
CLI-only with template guidance and no calibration, OOD handling, LLM, or eval.

| Capability | Reference (`Zumitify`) | **Crop Doctor Pro** |
|---|---|---|
| Vision model | ❌ Stub; you type the probabilities | ✅ **Real** model; takes an actual `leaf.jpg` |
| Input | Hand-entered class probs | ✅ Image file / uploaded photo |
| Confidence calibration | Described, not implemented | ✅ Temperature scaling (+ fitter & ECE eval) |
| Abstention signals | Confidence + near-tie | ✅ Confidence, near-tie, **entropy**, **image quality / OOD**, **crop mismatch** |
| Out-of-distribution input | — | ✅ "not a leaf / blurry / too dark" abstention |
| Guidance | Templates only | ✅ **Provider-agnostic LLM** (OpenAI / Anthropic / Ollama) **or** deterministic, your choice |
| Faithfulness / grounding | Source attribution | ✅ Grounding check that rejects hallucinated treatments → auto-fallback |
| Knowledge base | YAML | ✅ YAML, 27 entries / 6 crops, validated on load |
| Interface | CLI only | ✅ CLI **and** web UI (upload a photo) |
| Train your own | Described | ✅ Runnable PlantVillage fine-tune + calibration pipeline |
| Evaluation | — | ✅ Accuracy + **ECE** + **risk–coverage** harness |
| Tests | pytest | ✅ 16 pytest tests, deterministic, no download |

The architecture keeps the reference's good ideas (plug-in classifier protocol,
KB as single source of truth, abstention-as-success, mandatory disclaimer) and
adds the parts it was missing.

---

## Architecture

```
 image ──▶ Vision layer ──▶ Confidence gate ──▶ KB lookup ──▶ Guidance ──▶ Faithfulness ──▶ output
           (real model,      (abstain?)          (curated)     (LLM or       (grounded?
            calibrated)                                         template)      else fallback)
```

* **`cropdoctor/vision/`** — `HFClassifier` (real Hugging Face model, CPU),
  temperature-scaling calibration, image-quality/leaf-likeness checks, and a
  `StubClassifier` for tests. Plug in any checkpoint via `model_id`.
* **`cropdoctor/gate/`** — combines top-1 confidence, top-1/top-2 margin,
  predictive entropy, image quality (blur/dark/not-a-leaf), and crop mismatch.
  **Abstention is a success state.**
* **`cropdoctor/kb/`** — curated YAML, one entry per disease class, validated on
  load (including a guard against the `- Foo: bar` YAML-becomes-dict trap).
* **`cropdoctor/llm/`** — provider-agnostic layer: `openai`, `anthropic`,
  `ollama`, and a `deterministic` (no-LLM) path. Auto-detects keys; always falls
  back gracefully so it runs offline with zero config.
* **`cropdoctor/guidance/`** — builds the prompt from the KB entry, calls the
  chosen provider, then verifies the result.
* **`cropdoctor/faithfulness/`** — guarantees no treatment appears that isn't
  grounded in the KB; on any drift it discards the LLM output and uses the
  KB-verbatim composer.

---

## Quick start

```bash
cd crop-doctor-pro
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, transformers, gradio, ...

# 1) Diagnose a real leaf photo (downloads a ~14 MB model on first run)
python -m cropdoctor diagnose examples/tomato_early_blight.jpg --crop tomato

# 2) Raw JSON (machine-readable)
python -m cropdoctor diagnose examples/apple_scab.jpg --json

# 3) Launch the web UI — upload a photo in the browser
python -m cropdoctor serve            # http://127.0.0.1:7860

# 4) Inspect the knowledge base / providers
python -m cropdoctor info

# 5) Run the tests (no download needed)
pytest -q
```

> **Python version:** developed and tested on Python 3.9. The web UI pins
> `gradio==4.44.1` (Gradio 5 requires Python ≥3.10); a documented monkeypatch in
> `webapp.py` works around a gradio_client schema bug on 3.9. Everything else
> works on 3.9–3.12.

---

## Choosing your LLM (you are not locked to one vendor)

By design, **any** LLM can assemble the guidance — or none.

```bash
# OpenAI / ChatGPT
export OPENAI_API_KEY=sk-...           # optional: OPENAI_MODEL=gpt-4o-mini
python -m cropdoctor diagnose leaf.jpg --provider openai

# Anthropic / Claude
export ANTHROPIC_API_KEY=...           # optional: ANTHROPIC_MODEL=claude-sonnet-4-6
python -m cropdoctor diagnose leaf.jpg --provider anthropic

# Local / open models via Ollama (no API key, fully offline)
export OLLAMA_MODEL=llama3.1
python -m cropdoctor diagnose leaf.jpg --provider ollama

# No LLM at all — deterministic, KB-verbatim (the default if no key is found)
python -m cropdoctor diagnose leaf.jpg --provider none
```

Selection order: `--provider` flag → `LLM_PROVIDER` env → auto-detected key →
`deterministic`. **Whatever you pick, the LLM may only rephrase KB content**, and
the faithfulness check rejects anything it invents (it then silently falls back
to the deterministic composer). The provider is swappable; the safety is not.

---

## Output shape

```json
{
  "crop": "tomato",
  "prediction": {
    "disease": "Early Blight",
    "confidence": 0.8394,
    "top_3": [{"disease": "Early Blight", "crop": "tomato", "prob": 0.8394}, ...],
    "backend": "hf:linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "entropy": 0.202
  },
  "abstained": false,
  "gate": { "abstain": false, "reasons": [], "signals": { "margin": 0.78, ... } },
  "guidance": {
    "summary": "...",
    "immediate_steps": ["Remove and destroy affected lower leaves; do not compost them", ...],
    "treatment_options": ["Apply a copper-based fungicide according to the product label", ...],
    "prevention": ["Mulch around the base to prevent soil splash onto leaves", ...],
    "sources": ["KB-tomato-earlyblight"],
    "provider": "deterministic",
    "faithfulness": { "faithful": true, "sources": ["KB-tomato-earlyblight"] }
  },
  "disclaimer": "Demonstration tool; confirm with a local agricultural extension ..."
}
```

When the gate abstains, `guidance` is `null` and `gate.reasons` explains why
(e.g. `near_tie`, `not_a_leaf`, `crop_mismatch`).

---

## Train your own model (the path the reference only described)

The default model is a pretrained PlantVillage MobileNetV2 so the demo works
immediately. To fine-tune your own checkpoint:

```bash
pip install "transformers[torch]" datasets accelerate scikit-learn

# data/plantvillage/ laid out as ImageFolder: Crop___Disease/*.jpg
python train/train.py --data-dir data/plantvillage --out models/plantvillage-mnv2 \
    --epochs 3 --freeze-backbone           # head-only tuning runs on CPU

# Evaluate: accuracy + ECE (calibration) + risk-coverage curve
python train/evaluate.py --data-dir data/plantvillage_test \
    --model-id models/plantvillage-mnv2 --temperature 1.7

# Use it — folder names become labels, so KB keys line up automatically
python -m cropdoctor diagnose leaf.jpg --model-id models/plantvillage-mnv2 --temperature 1.7
```

`train.py` also fits and saves a calibration temperature; pass it via
`--temperature`.

---

## Knowledge base

27 curated entries across **tomato, potato, apple, grape, corn, pepper** (plus
healthy classes), in `cropdoctor/kb/data/*.yaml`. Each entry has a summary,
symptoms, immediate steps, treatment options, prevention, and a `source_id`.
Add a crop by dropping in a YAML file keyed by `crop_disease` — no code changes.

Confident predictions whose class has **no** KB entry **abstain** rather than
invent advice.

---

## Responsible-use note

This is a demonstration system, not an agronomy decision tool. Models trained on
clean PlantVillage images perform worse on real field photos. Guidance is
informational, not a prescription. **Always confirm with a local agricultural
extension service or a qualified agronomist before applying any treatment.**
