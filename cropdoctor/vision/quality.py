"""Cheap, dependency-light image-quality and leaf-likeness checks.

These run before/alongside classification and feed the abstention gate. A model
trained on clean PlantVillage leaves will confidently mislabel a blurry photo,
a screenshot, or a picture of a cat — so we surface simple signals the gate can
use to abstain on obviously out-of-distribution inputs. None of this needs the
heavy model; it's pure PIL + numpy.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — a standard, fast blur metric."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    g = gray.astype(np.float64)
    # 'valid' convolution via slicing to avoid a scipy dependency.
    out = (
        k[0, 1] * g[:-2, 1:-1] + k[1, 0] * g[1:-1, :-2] + k[1, 1] * g[1:-1, 1:-1]
        + k[1, 2] * g[1:-1, 2:] + k[2, 1] * g[2:, 1:-1]
    )
    return float(out.var())


def assess(image: Image.Image) -> Dict[str, Any]:
    """Return quality signals plus a coarse ``looks_like_leaf`` heuristic."""
    img = image.convert("RGB")
    small = img.resize((256, 256))
    arr = np.asarray(small, dtype=np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    sharpness = _laplacian_variance(gray)  # higher = sharper

    # Vegetation heuristic: excess-green index fraction. Real leaf crops have a
    # substantial green-dominant area even when diseased/browned.
    exg = 2 * g - r - b
    green_fraction = float((exg > 12).mean())

    # Diseased leaves can be quite brown, so keep this permissive — it's a soft
    # signal, the gate decides thresholds.
    looks_like_leaf = green_fraction > 0.08

    return {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "sharpness": round(sharpness, 1),
        "green_fraction": round(green_fraction, 3),
        "looks_like_leaf": bool(looks_like_leaf),
        "blurry": bool(sharpness < 80.0),
        "too_dark": bool(brightness < 0.12),
        "too_bright": bool(brightness > 0.95),
        "low_contrast": bool(contrast < 0.05),
    }
