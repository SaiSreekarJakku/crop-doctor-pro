"""Modern web app: FastAPI backend + a hand-built single-page frontend.

This replaces the stock Gradio UI with a designed interface (drag-and-drop
upload, animated confidence bars, abstention banner, provider badge, dark
theme). No build step and no CDN — the frontend is plain HTML/CSS/JS served as
static files, so it works offline.

Run it with:  python -m cropdoctor serve   (then open http://127.0.0.1:7860)

Note: we intentionally do NOT use ``from __future__ import annotations`` here —
FastAPI/pydantic must see the real ``UploadFile`` type, not a string forward
reference, to build the multipart request model.
"""
import io
import os

from .gate.gate import GateConfig
from .kb.loader import load_kb
from .llm.provider import available_providers, get_provider
from .pipeline import CropDoctor
from .vision.classifier import DEFAULT_MODEL_ID, load_classifier

_WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
_DOCTORS = {}


def _doctor(provider_name, min_confidence, temperature):
    key = (provider_name, round(min_confidence, 3), round(temperature, 3))
    if key not in _DOCTORS:
        _DOCTORS[key] = CropDoctor(
            classifier=load_classifier("hf", DEFAULT_MODEL_ID, temperature=temperature),
            gate_config=GateConfig(min_confidence=min_confidence),
            provider=get_provider(None if provider_name in ("auto", "") else provider_name),
        )
    return _DOCTORS[key]


def create_app():
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from PIL import Image

    app = FastAPI(title="Crop Doctor Pro", docs_url="/api/docs")
    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_WEB_DIR, "index.html"))

    @app.get("/api/info")
    def info():
        kb = load_kb()
        return {
            "model": DEFAULT_MODEL_ID,
            "active_provider": get_provider().name,
            "providers": available_providers(),
            "crops": kb.crops(),
            "kb_entries": len(kb),
        }

    @app.post("/api/diagnose")
    async def diagnose(
        image: UploadFile = File(...),
        crop: str = Form(""),
        region: str = Form(""),
        provider: str = Form("auto"),
        min_confidence: float = Form(0.65),
        temperature: float = Form(1.0),
    ):
        raw = await image.read()
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return JSONResponse({"error": "Could not read image file."}, status_code=400)
        doctor = _doctor(provider, min_confidence, temperature)
        diag = doctor.diagnose(pil, user_crop=crop or None, user_region=region or None)
        return diag.to_dict()

    return app


def serve(host="127.0.0.1", port=7860):
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
