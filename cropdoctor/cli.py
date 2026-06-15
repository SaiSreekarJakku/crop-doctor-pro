"""Command-line interface.

Unlike the reference project's CLI (which asks YOU to type the probabilities),
this one takes a real image path and runs a real model.

    python -m cropdoctor diagnose leaf.jpg --crop tomato
    python -m cropdoctor diagnose leaf.jpg --provider openai
    python -m cropdoctor info
"""
from __future__ import annotations

import json
import os
from typing import Optional

import typer


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Existing env vars win."""
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


_load_dotenv()

from .gate.gate import GateConfig
from .kb.loader import load_kb
from .llm.provider import available_providers, get_provider
from .pipeline import CropDoctor
from .vision.classifier import DEFAULT_MODEL_ID, load_classifier

app = typer.Typer(add_completion=False, help="Crop Doctor Pro — grounded leaf-disease diagnosis.")


def _emit(diag, pretty: bool):
    data = diag.to_dict()
    if pretty:
        _print_pretty(data)
    else:
        typer.echo(json.dumps(data, indent=2))


def _print_pretty(d: dict):
    pred = d.get("prediction") or {}
    typer.secho(f"\nCrop:       {d.get('crop')}", fg="cyan")
    typer.secho(f"Prediction: {pred.get('disease')}  "
                f"(confidence {pred.get('confidence')}, backend {pred.get('backend')})")
    typer.echo("Top-3:")
    for c in pred.get("top_3", []):
        typer.echo(f"   - {c['disease']} [{c['crop']}]  {c['prob']}")
    if d["abstained"]:
        typer.secho("\n⚠  ABSTAINED — not confident enough to advise.", fg="yellow", bold=True)
        for r in d["gate"]["reasons"]:
            typer.echo(f"   • {r}")
        typer.secho("\nRecommendation: consult a local agricultural extension or agronomist.",
                    fg="yellow")
    else:
        g = d["guidance"]
        typer.secho(f"\nGuidance (provider: {g['provider']}, sources: {g['sources']}):",
                    fg="green", bold=True)
        typer.echo(f"\n{g['summary']}\n")
        for title, key in [("Immediate steps", "immediate_steps"),
                           ("Treatment options", "treatment_options"),
                           ("Prevention", "prevention")]:
            typer.secho(title + ":", bold=True)
            for item in g[key]:
                typer.echo(f"   • {item}")
    typer.secho(f"\n{d['disclaimer']}", fg="bright_black")


@app.command()
def diagnose(
    image: str = typer.Argument(..., help="Path to a leaf image."),
    crop: Optional[str] = typer.Option(None, help="Optional crop type (enables mismatch check)."),
    region: Optional[str] = typer.Option(None, help="Optional region (recorded for context)."),
    provider: Optional[str] = typer.Option(None, help="LLM provider: openai|anthropic|ollama|none."),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, help="Vision model id or local path."),
    backend: str = typer.Option("hf", help="Vision backend: hf|stub."),
    temperature: float = typer.Option(1.0, help="Calibration temperature (>=1 softens confidence)."),
    min_confidence: float = typer.Option(0.65, help="Abstain below this calibrated top-1."),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON instead of pretty text."),
):
    """Diagnose a leaf image and produce grounded guidance (or abstain)."""
    clf = load_classifier(backend=backend, model_id=model_id, temperature=temperature)
    doctor = CropDoctor(
        classifier=clf,
        gate_config=GateConfig(min_confidence=min_confidence),
        provider=get_provider(provider),
    )
    diag = doctor.diagnose(image, user_crop=crop, user_region=region)
    _emit(diag, pretty=not json_out)


@app.command()
def info():
    """Show the loaded knowledge base, providers, and default model."""
    kb = load_kb()
    active = get_provider().name
    typer.secho("Crop Doctor Pro", fg="cyan", bold=True)
    typer.echo(f"Default vision model : {DEFAULT_MODEL_ID}")
    typer.echo(f"LLM providers        : {', '.join(available_providers())}")
    typer.echo(f"Active provider      : {active}")
    typer.echo(f"KB entries           : {len(kb)}")
    typer.echo(f"KB crops             : {', '.join(kb.crops())}")
    typer.echo("KB keys:")
    for k in kb.keys():
        typer.echo(f"   - {k}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 7860):
    """Launch the modern web UI (FastAPI + custom frontend)."""
    from .server import serve as _serve

    typer.secho(f"Crop Doctor Pro → http://{host}:{port}", fg="green", bold=True)
    _serve(host=host, port=port)


@app.command(name="serve-gradio")
def serve_gradio(host: str = "127.0.0.1", port: int = 7861):
    """Launch the legacy Gradio UI (fallback)."""
    from .webapp import build_ui

    build_ui().launch(server_name=host, server_port=port)


if __name__ == "__main__":
    app()
