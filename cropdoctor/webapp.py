"""Gradio web UI — upload a leaf photo, get a grounded diagnosis or an abstention.

The reference project is CLI-only. This gives non-technical users (the actual
audience: growers, extension workers) a browser interface. Gradio is imported
lazily so the rest of the package does not depend on it.
"""
from __future__ import annotations

from typing import Optional

from .gate.gate import GateConfig
from .llm.provider import get_provider
from .pipeline import CropDoctor
from .vision.classifier import DEFAULT_MODEL_ID, load_classifier

_DOCTOR_CACHE = {}


def _get_doctor(provider_name: str, min_confidence: float, temperature: float) -> CropDoctor:
    key = (provider_name, round(min_confidence, 3), round(temperature, 3))
    if key not in _DOCTOR_CACHE:
        _DOCTOR_CACHE[key] = CropDoctor(
            classifier=load_classifier("hf", DEFAULT_MODEL_ID, temperature=temperature),
            gate_config=GateConfig(min_confidence=min_confidence),
            provider=get_provider(None if provider_name == "auto" else provider_name),
        )
    return _DOCTOR_CACHE[key]


def _render_markdown(d: dict) -> str:
    pred = d.get("prediction") or {}
    lines = [f"### {'⚠️ Abstained' if d['abstained'] else '✅ Diagnosis'}",
             f"**Crop:** {d.get('crop')}  ",
             f"**Top prediction:** {pred.get('disease')} "
             f"(confidence **{pred.get('confidence')}**, entropy {pred.get('entropy')})  ",
             f"**Model:** `{pred.get('backend')}`  "]
    lines.append("\n**Top-3:**")
    for c in pred.get("top_3", []):
        lines.append(f"- {c['disease']} [{c['crop']}] — {c['prob']}")

    if d["abstained"]:
        lines.append("\n**Why we held back:**")
        for r in d["gate"]["reasons"]:
            lines.append(f"- {r}")
        lines.append("\n> Consult a local agricultural extension service or a "
                     "qualified agronomist for a confident diagnosis.")
    else:
        g = d["guidance"]
        lines.append(f"\n#### Guidance  \n*(assembled by `{g['provider']}`, "
                     f"sources: {', '.join(g['sources'])})*")
        lines.append(f"\n{g['summary']}")
        for title, k in [("Immediate steps", "immediate_steps"),
                         ("Treatment options", "treatment_options"),
                         ("Prevention", "prevention")]:
            lines.append(f"\n**{title}:**")
            for item in g[k]:
                lines.append(f"- {item}")
        if g.get("faithfulness"):
            lines.append(f"\n`faithfulness: {g['faithfulness'].get('faithful', True)}`")
    lines.append(f"\n---\n_{d['disclaimer']}_")
    return "\n".join(lines)


def _patch_gradio_client_schema_bug():
    """Work around a gradio_client 1.3.0 bug on Python 3.9.

    Gradio 5 fixes it but needs Python >=3.10; on 3.9 we're pinned to gradio
    4.44 / gradio_client 1.3.0, whose ``_json_schema_to_python_type`` crashes on
    boolean JSON schemas (``additionalProperties: true``) with
    ``TypeError: argument of type 'bool' is not iterable``. We make the schema
    walkers tolerate booleans. Harmless if the installed version isn't affected.
    """
    try:
        import gradio_client.utils as gcu
    except Exception:
        return

    _orig_json = gcu._json_schema_to_python_type
    _orig_get = gcu.get_type

    def safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_get(schema)

    def safe_json(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json(schema, defs)

    gcu.get_type = safe_get_type
    gcu._json_schema_to_python_type = safe_json


def build_ui():
    _patch_gradio_client_schema_bug()
    import gradio as gr

    import json as _json

    def run(image, crop, region, provider_name, min_confidence, temperature):
        if image is None:
            return "Please upload a leaf image."
        doctor = _get_doctor(provider_name, min_confidence, temperature)
        diag = doctor.diagnose(
            image, user_crop=crop or None, user_region=region or None
        ).to_dict()
        md = _render_markdown(diag)
        md += ("\n\n<details><summary>Raw JSON response</summary>\n\n```json\n"
               + _json.dumps(diag, indent=2) + "\n```\n</details>")
        return md

    with gr.Blocks(title="Crop Doctor Pro") as demo:
        gr.Markdown(
            "# 🌿 Crop Doctor Pro\n"
            "Upload a leaf photo. A **real** vision model predicts the disease, a "
            "calibrated gate **abstains** when unsure, and guidance is assembled "
            "**only** from a curated, source-cited knowledge base."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Leaf photo")
                crop = gr.Textbox(label="Crop (optional)", placeholder="tomato")
                region = gr.Textbox(label="Region (optional)", placeholder="e.g. Karnataka")
                provider_name = gr.Dropdown(
                    ["auto", "deterministic", "openai", "anthropic", "ollama"],
                    value="auto", label="Guidance LLM provider")
                min_confidence = gr.Slider(0.3, 0.95, value=0.65, step=0.05,
                                           label="Abstain below confidence")
                temperature = gr.Slider(1.0, 3.0, value=1.0, step=0.1,
                                        label="Calibration temperature")
                btn = gr.Button("Diagnose", variant="primary")
            with gr.Column(scale=1):
                out_md = gr.Markdown(label="Result")
        btn.click(run, [image, crop, region, provider_name, min_confidence, temperature],
                  [out_md])
    return demo


if __name__ == "__main__":
    build_ui().launch()
