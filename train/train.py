"""Fine-tune a leaf-disease classifier on PlantVillage, then calibrate it.

This is the path the reference project only *described*. Run it to train your own
checkpoint that ``HFClassifier(model_id=<output_dir>)`` can load directly.

Usage
-----
1. Get the PlantVillage dataset laid out as ImageFolder:

       data/plantvillage/
         Tomato___Early_blight/ *.jpg
         Tomato___Late_blight/  *.jpg
         ...

   (e.g. the public "PlantVillage" / "New Plant Diseases Dataset" on Kaggle.)

2. Install the training extras:

       pip install "transformers[torch]" datasets accelerate scikit-learn

3. Train (CPU works for a few epochs of head-tuning; GPU strongly preferred):

       python train/train.py --data-dir data/plantvillage --out models/plantvillage-mnv2 \
           --epochs 3 --freeze-backbone

The script keeps the PlantVillage ``Crop___Disease`` folder names as labels, so
``cropdoctor.labels.parse_label`` and the KB keys line up automatically.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="ImageFolder root (Crop___Disease/*).")
    ap.add_argument("--out", default="models/plantvillage-mnv2")
    ap.add_argument("--base", default="google/mobilenet_v2_1.0_224",
                    help="Backbone to fine-tune.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Train only the classifier head (fast, CPU-friendly).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import (AutoImageProcessor,
                              AutoModelForImageClassification, Trainer,
                              TrainingArguments)

    # ImageFolder dataset; folder names become ClassLabel names == our labels.
    ds = load_dataset("imagefolder", data_dir=args.data_dir, split="train")
    ds = ds.train_test_split(test_size=args.val_split, seed=args.seed)
    label_names = ds["train"].features["label"].names
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in id2label.items()}
    print(f"Classes ({len(label_names)}): {label_names[:5]} ...")

    processor = AutoImageProcessor.from_pretrained(args.base)
    size = processor.size.get("shortest_edge", 224) if hasattr(processor, "size") else 224

    def transform(batch):
        imgs = [im.convert("RGB") for im in batch["image"]]
        enc = processor(images=imgs, return_tensors="pt")
        batch["pixel_values"] = enc["pixel_values"]
        return batch

    ds = ds.with_transform(transform)

    model = AutoModelForImageClassification.from_pretrained(
        args.base, num_labels=len(label_names), id2label=id2label,
        label2id=label2id, ignore_mismatched_sizes=True,
    )
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    def collate(examples):
        return {
            "pixel_values": torch.stack([e["pixel_values"][0] if e["pixel_values"].ndim == 4
                                         else e["pixel_values"] for e in examples]),
            "labels": torch.tensor([e["label"] for e in examples]),
        }

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": float((preds == labels).mean())}

    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "_trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        remove_unused_columns=False,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=ds["test"], data_collator=collate,
                      compute_metrics=metrics)
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)

    # --- Temperature calibration on the validation split ---
    print("Fitting calibration temperature ...")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cropdoctor.vision.calibration import fit_temperature

    logits_all, labels_all = [], []
    model.eval()
    loader = torch.utils.data.DataLoader(ds["test"], batch_size=args.batch_size,
                                         collate_fn=collate)
    with torch.no_grad():
        for batch in loader:
            out = model(pixel_values=batch["pixel_values"]).logits
            logits_all.extend(out.cpu().numpy().tolist())
            labels_all.extend(batch["labels"].cpu().numpy().tolist())
    T = fit_temperature(logits_all, labels_all)
    with open(os.path.join(args.out, "calibration.json"), "w") as fh:
        json.dump({"temperature": T}, fh, indent=2)
    print(f"Saved model + calibration (T={T:.3f}) to {args.out}")
    print(f"Use it:  python -m cropdoctor diagnose leaf.jpg "
          f"--model-id {args.out} --temperature {T:.3f}")


if __name__ == "__main__":
    main()
