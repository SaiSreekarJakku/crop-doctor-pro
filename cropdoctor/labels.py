"""Map raw model class labels to (crop, disease, kb_key, healthy).

The default model uses the PlantVillage 38-class label set, where labels look
like ``"Tomato___Early_blight"`` or ``"Corn_(maize)___Common_rust_"``. This
module normalizes any such label into the pieces the rest of the system needs,
so the KB and guidance layers never have to parse raw model strings.

The parser is deliberately generic: it works for the PlantVillage scheme but
degrades gracefully for arbitrary ``Crop___Disease`` labels from a custom
fine-tuned model, so swapping in your own checkpoint Just Works.
"""
from __future__ import annotations

import re
from typing import Tuple

# Crop display-name cleanups for the noisier PlantVillage tokens.
# Keys are in *slugified* form (parens/commas already collapsed to "_"),
# because the fixup is applied after _slug().
_CROP_FIXUPS = {
    "corn_maize": "corn",
    "pepper_bell": "pepper",
    "cherry_including_sour": "cherry",
}

_HEALTHY_TOKENS = {"healthy", "background_without_leaves", "background"}

# Some Hub checkpoints ship free-form English labels instead of the canonical
# PlantVillage ``Crop___Disease`` scheme. Map those to canonical labels so the
# KB keys line up. The generic parser below handles everything else (including
# your own fine-tuned checkpoints from train/train.py, which keep folder names).
LABEL_ALIASES = {
    "Apple Scab": "Apple___Apple_scab",
    "Apple with Black Rot": "Apple___Black_rot",
    "Cedar Apple Rust": "Apple___Cedar_apple_rust",
    "Healthy Apple": "Apple___healthy",
    "Healthy Blueberry Plant": "Blueberry___healthy",
    "Cherry with Powdery Mildew": "Cherry_(including_sour)___Powdery_mildew",
    "Healthy Cherry Plant": "Cherry_(including_sour)___healthy",
    "Corn (Maize) with Cercospora and Gray Leaf Spot":
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn (Maize) with Common Rust": "Corn_(maize)___Common_rust_",
    "Corn (Maize) with Northern Leaf Blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Healthy Corn (Maize) Plant": "Corn_(maize)___healthy",
    "Grape with Black Rot": "Grape___Black_rot",
    "Grape with Esca (Black Measles)": "Grape___Esca_(Black_Measles)",
    "Grape with Isariopsis Leaf Spot": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Healthy Grape Plant": "Grape___healthy",
    "Orange with Citrus Greening": "Orange___Haunglongbing_(Citrus_greening)",
    "Peach with Bacterial Spot": "Peach___Bacterial_spot",
    "Healthy Peach Plant": "Peach___healthy",
    "Bell Pepper with Bacterial Spot": "Pepper,_bell___Bacterial_spot",
    "Healthy Bell Pepper Plant": "Pepper,_bell___healthy",
    "Potato with Early Blight": "Potato___Early_blight",
    "Potato with Late Blight": "Potato___Late_blight",
    "Healthy Potato Plant": "Potato___healthy",
    "Healthy Raspberry Plant": "Raspberry___healthy",
    "Healthy Soybean Plant": "Soybean___healthy",
    "Squash with Powdery Mildew": "Squash___Powdery_mildew",
    "Strawberry with Leaf Scorch": "Strawberry___Leaf_scorch",
    "Healthy Strawberry Plant": "Strawberry___healthy",
    "Tomato with Bacterial Spot": "Tomato___Bacterial_spot",
    "Tomato with Early Blight": "Tomato___Early_blight",
    "Tomato with Late Blight": "Tomato___Late_blight",
    "Tomato with Leaf Mold": "Tomato___Leaf_Mold",
    "Tomato with Septoria Leaf Spot": "Tomato___Septoria_leaf_spot",
    "Tomato with Spider Mites or Two-spotted Spider Mite":
        "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato with Target Spot": "Tomato___Target_Spot",
    "Tomato Yellow Leaf Curl Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato Mosaic Virus": "Tomato___Tomato_mosaic_virus",
    "Healthy Tomato Plant": "Tomato___healthy",
}


def _slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _titleize(slug: str) -> str:
    words = [w for w in slug.split("_") if w]
    # Keep short scientific-ish tokens lowercase-friendly but readable.
    return " ".join(w.capitalize() for w in words)


def parse_label(label: str) -> Tuple[str, str, str, bool]:
    """Return ``(crop, disease, kb_key, healthy)`` for a raw model label.

    Examples
    --------
    >>> parse_label("Tomato___Early_blight")
    ('tomato', 'Early Blight', 'tomato_early_blight', False)
    >>> parse_label("Potato___healthy")
    ('potato', 'Healthy', 'potato_healthy', True)
    """
    raw = LABEL_ALIASES.get(label.strip(), label.strip())
    if "___" in raw:
        crop_part, disease_part = raw.split("___", 1)
    elif "__" in raw:
        crop_part, disease_part = raw.split("__", 1)
    else:
        # No crop delimiter; treat whole thing as disease, crop unknown.
        crop_part, disease_part = "unknown", raw

    crop_slug = _slug(crop_part)
    crop_slug = _CROP_FIXUPS.get(crop_slug, crop_slug)

    disease_slug = _slug(disease_part)
    healthy = disease_slug in _HEALTHY_TOKENS

    crop = crop_slug
    disease = "Healthy" if healthy else _titleize(disease_slug)
    kb_key = f"{crop_slug}_{disease_slug}"
    return crop, disease, kb_key, healthy
