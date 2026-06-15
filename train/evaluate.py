"""Evaluate a checkpoint: accuracy, calibration (ECE), and risk-coverage.

A competition-grade abstention system should be judged on more than top-1
accuracy. This harness reports:

* **Top-1 accuracy** on a held-out ImageFolder test set.
* **ECE** (Expected Calibration Error) — how well confidence matches accuracy.
  Lower is better; this is what temperature scaling improves.
* **Risk-coverage** — as we raise the confidence threshold the system abstains
  more (lower coverage) but is right more often on what it does answer (higher
  selective accuracy). This quantifies the core value proposition.

Usage
-----
    python train/evaluate.py --data-dir data/plantvillage_test \
        --model-id models/plantvillage-mnv2 --temperature 1.7
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def expected_calibration_error(confidences, correct, n_bins=10):
    import numpy as np
    conf = np.asarray(confidences)
    corr = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = corr[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="ImageFolder test root.")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="Cap images (0 = all).")
    args = ap.parse_args()

    import numpy as np
    from cropdoctor.vision.classifier import HFClassifier

    clf = HFClassifier(model_id=args.model_id, temperature=args.temperature)

    confidences, correct, abstain_conf = [], [], []
    paths = []
    for cls in sorted(os.listdir(args.data_dir)):
        d = os.path.join(args.data_dir, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append((os.path.join(d, f), cls))
    if args.limit:
        paths = paths[: args.limit]
    print(f"Evaluating {len(paths)} images ...")

    for i, (path, true_label) in enumerate(paths):
        vr = clf.classify(path)
        pred = vr.top.label
        is_correct = int(pred == true_label)
        confidences.append(vr.top.prob)
        correct.append(is_correct)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)}")

    acc = float(np.mean(correct)) if correct else 0.0
    ece = expected_calibration_error(confidences, correct)
    print(f"\nTop-1 accuracy : {acc:.4f}")
    print(f"ECE            : {ece:.4f}  (lower = better calibrated)")

    print("\nRisk-Coverage (abstain below threshold):")
    print(f"  {'thresh':>7} {'coverage':>9} {'sel.acc':>8}")
    conf = np.asarray(confidences)
    corr = np.asarray(correct, dtype=np.float64)
    for t in [0.0, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]:
        covered = conf >= t
        coverage = float(covered.mean())
        sel_acc = float(corr[covered].mean()) if covered.sum() else float("nan")
        print(f"  {t:>7.2f} {coverage:>9.3f} {sel_acc:>8.3f}")


if __name__ == "__main__":
    main()
