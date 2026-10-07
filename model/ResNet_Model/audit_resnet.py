"""Audit classification, embeddings and cosine retrieval on unseen images."""

import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms

from config import (
    DATASET_NAME, DATASET_SPLIT, DEVICE, IMAGE_SIZE, MODEL_PATH,
    NORMALIZE_MEAN, NORMALIZE_STD, RANDOM_STATE, RESIZE_SIZE,
    SELECTED_CLASSES,
)
from model import FashionResNet


ROOT = Path(__file__).resolve().parent
REPORT_PATH = Path(os.getenv("RESNET_AUDIT_PATH", str(ROOT / "resnet_audit.json")))
SAMPLES_PER_CLASS = int(os.getenv("RESNET_AUDIT_PER_CLASS", "100"))


def main():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = list(checkpoint.get("classes", SELECTED_CLASSES))
    model = FashionResNet(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE).eval()
    backbone = torch.nn.Sequential(*list(model.model.children())[:-1])
    transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    filtered = dataset.filter(lambda row: row["articleType"] in classes)

    training = checkpoint.get("training") or {}
    recorded_indices = training.get("dataset_indices")
    if recorded_indices is not None:
        training_indices = set(recorded_indices)
        evaluation_policy = "complement of checkpoint dataset_indices"
    else:
        # Legacy checkpoint provenance fallback.
        legacy_rng = random.Random(RANDOM_STATE)
        legacy_count = min(2000, len(filtered))
        training_indices = set(legacy_rng.sample(range(len(filtered)), legacy_count))
        evaluation_policy = "complement of reconstructed legacy 2000-image sample"
    pools = {name: [] for name in classes}
    for index in range(len(filtered)):
        if index not in training_indices:
            pools[filtered[index]["articleType"]].append(index)
    rng = random.Random(RANDOM_STATE + 1)
    selected = []
    for name in classes:
        selected.extend(rng.sample(pools[name], min(SAMPLES_PER_CLASS, len(pools[name]))))
    rng.shuffle(selected)

    tensors = torch.stack([transform(filtered[index]["image"].convert("RGB"))
                           for index in selected])
    labels = [filtered[index]["articleType"] for index in selected]
    label_ids = np.asarray([classes.index(label) for label in labels])
    feature_parts, logit_parts = [], []
    batch_size = 32
    with torch.inference_mode():
        for offset in range(0, len(tensors), batch_size):
            batch = tensors[offset:offset + batch_size].to(DEVICE)
            flipped = torch.flip(batch, dims=[3])
            views = torch.cat([batch, flipped], dim=0)
            features = backbone(views).flatten(1)
            logits = model.model.fc(features)
            count = len(batch)
            logits = (logits[:count] + logits[count:]) / 2
            features = torch.nn.functional.normalize(features[:count], dim=1)
            feature_parts.append(features.cpu())
            logit_parts.append(logits.cpu())
    features = torch.cat(feature_parts).numpy()
    logits = torch.cat(logit_parts)
    probabilities = logits.softmax(1).numpy()
    predictions = probabilities.argmax(1)

    cosine = features @ features.T
    cosine_reference = torch.nn.functional.cosine_similarity(
        torch.from_numpy(features[:1]), torch.from_numpy(features), dim=1
    ).numpy()
    cosine_error = float(np.max(np.abs(cosine[0] - cosine_reference)))
    diagonal = np.diag(cosine).copy()
    np.fill_diagonal(cosine, -np.inf)
    ranking = np.argsort(-cosine, axis=1, kind="stable")
    retrieval_top1 = float(np.mean([
        label_ids[row] == label_ids[order[0]] for row, order in enumerate(ranking)
    ]))
    retrieval_top5 = float(np.mean([
        label_ids[row] in set(label_ids[order[:5]]) for row, order in enumerate(ranking)
    ]))

    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, predicted in zip(label_ids, predictions):
        confusion[actual, predicted] += 1
    per_class = {}
    for index, name in enumerate(classes):
        total = int(confusion[index].sum())
        per_class[name] = {
            "samples": total,
            "accuracy": float(confusion[index, index] / total) if total else None,
        }
    macro_accuracy = float(np.mean([row["accuracy"] for row in per_class.values()
                                    if row["accuracy"] is not None]))
    report = {
        "evaluation_set": {
            "policy": evaluation_policy,
            "samples": len(selected),
            "class_counts": dict(Counter(labels)),
        },
        "embedding": {
            "shape": list(features.shape),
            "finite": bool(np.isfinite(features).all()),
            "norm_min": float(np.linalg.norm(features, axis=1).min()),
            "norm_max": float(np.linalg.norm(features, axis=1).max()),
        },
        "cosine": {
            "self_similarity_min": float(diagonal.min()),
            "self_similarity_max": float(diagonal.max()),
            "implementation_max_error": cosine_error,
            "retrieval_top1": retrieval_top1,
            "retrieval_top5": retrieval_top5,
        },
        "classification": {
            "accuracy": float(np.mean(predictions == label_ids)),
            "macro_accuracy": macro_accuracy,
            "average_confidence": float(probabilities.max(1).mean()),
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
        },
        "classes": classes,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
