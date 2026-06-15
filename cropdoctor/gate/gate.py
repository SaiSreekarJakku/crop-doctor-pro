"""The confidence gate: decide whether it is responsible to give advice.

Abstention is a *success* state. We would much rather say "I'm not sure, see an
expert" than confidently hand a farmer the wrong treatment. The gate combines
several independent signals so it catches failure modes a single top-1 threshold
would miss:

* **Low confidence** — calibrated top-1 probability below ``min_confidence``.
* **Near-tie** — the margin between top-1 and top-2 is too small to trust.
* **High entropy** — the whole distribution is spread out (model hedging).
* **Image quality / OOD** — blurry, too dark, or "doesn't look like a leaf".
* **Crop mismatch** — user said "tomato" but the model is sure it's a grape leaf.

If *any* trip, we abstain and explain why.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import GateDecision, VisionResult


@dataclass
class GateConfig:
    min_confidence: float = 0.65   # calibrated top-1 must clear this
    min_margin: float = 0.15       # top1 - top2 must clear this
    max_entropy: float = 0.55      # normalized entropy ceiling
    require_leaf: bool = True       # abstain if image doesn't look like a leaf
    block_on_blur: bool = True


def evaluate(
    vision: VisionResult,
    config: Optional[GateConfig] = None,
    user_crop: Optional[str] = None,
) -> GateDecision:
    cfg = config or GateConfig()
    reasons = []

    top = vision.top
    top2_prob = vision.ranked[1].prob if len(vision.ranked) > 1 else 0.0
    margin = top.prob - top2_prob

    signals = {
        "top_confidence": round(top.prob, 4),
        "margin": round(margin, 4),
        "entropy": round(vision.entropy, 4),
        "top_label": top.label,
        "second_label": vision.ranked[1].label if len(vision.ranked) > 1 else None,
    }

    if top.prob < cfg.min_confidence:
        reasons.append(
            f"low_confidence: top probability {top.prob:.2f} < {cfg.min_confidence:.2f}"
        )
    if margin < cfg.min_margin:
        reasons.append(
            f"near_tie: top-1/top-2 margin {margin:.2f} < {cfg.min_margin:.2f}"
        )
    if vision.entropy > cfg.max_entropy:
        reasons.append(
            f"high_entropy: {vision.entropy:.2f} > {cfg.max_entropy:.2f} (model is hedging)"
        )

    q = vision.quality or {}
    if cfg.block_on_blur and q.get("blurry"):
        reasons.append("blurry_image: photo is too out of focus to trust")
    if q.get("too_dark"):
        reasons.append("too_dark: image is underexposed")
    if q.get("too_bright"):
        reasons.append("too_bright / washed out: image is overexposed")
    if cfg.require_leaf and q.get("looks_like_leaf") is False:
        reasons.append("not_a_leaf: image does not appear to contain leaf tissue")

    # Crop mismatch: only meaningful when the model is otherwise confident.
    if user_crop:
        uc = user_crop.strip().lower()
        if uc and top.crop and uc != top.crop and not top.healthy:
            reasons.append(
                f"crop_mismatch: you specified '{uc}' but the model predicts a "
                f"'{top.crop}' leaf"
            )
            signals["user_crop"] = uc

    return GateDecision(abstain=bool(reasons), reasons=reasons, signals=signals)
