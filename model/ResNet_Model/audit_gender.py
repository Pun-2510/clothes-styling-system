"""Audit the gender head on images excluded from its 3,000-image sample."""

import json
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    DATASET_NAME, DATASET_SPLIT, DEVICE, IMAGE_SIZE, MODEL_PATH,
    NORMALIZE_MEAN, NORMALIZE_STD, RANDOM_STATE, RESIZE_SIZE, SELECTED_CLASSES,
)
from gender import GENDER_CLASSES, GenderHead
from model import FashionResNet
from train_gender import normalize_gender


ROOT = Path(__file__).resolve().parent
GENDER_PATH = ROOT.parent / "Web_Test" / "models" / "resnet_gender.pth"


class Rows(Dataset):
    def __init__(self, dataset, rows, transform):
        self.dataset, self.rows, self.transform = dataset, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source_index, label, category = self.rows[index]
        image = self.dataset[source_index]["image"].convert("RGB")
        return self.transform(image), label, category


def main():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    pools = {name: [] for name in GENDER_CLASSES}
    for index, row in enumerate(dataset):
        gender = normalize_gender(row["gender"])
        if gender and row["articleType"] in SELECTED_CLASSES:
            pools[gender].append(index)
    rng = random.Random(RANDOM_STATE)
    excluded = {}
    for gender in GENDER_CLASSES:
        excluded[gender] = set(rng.sample(pools[gender], min(1500, len(pools[gender]))))
    audit_rng = random.Random(RANDOM_STATE + 17)
    rows = []
    for gender in GENDER_CLASSES:
        candidates = [index for index in pools[gender] if index not in excluded[gender]]
        for index in audit_rng.sample(candidates, min(300, len(candidates))):
            rows.append((index, GENDER_CLASSES.index(gender), dataset[index]["articleType"]))

    transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    fashion_checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    fashion = FashionResNet(len(fashion_checkpoint["classes"]), pretrained=False)
    fashion.load_state_dict(fashion_checkpoint["model_state_dict"])
    backbone = torch.nn.Sequential(*list(fashion.model.children())[:-1]).to(DEVICE).eval()
    gender_checkpoint = torch.load(GENDER_PATH, map_location="cpu", weights_only=True)
    head = GenderHead(num_classes=2).to(DEVICE).eval()
    head.load_state_dict(gender_checkpoint["model_state_dict"])

    confusion = torch.zeros(2, 2, dtype=torch.long)
    category_correct, category_total = {}, {}
    with torch.inference_mode():
        for images, labels, categories in DataLoader(Rows(dataset, rows, transform), batch_size=32):
            embeddings = backbone(images.to(DEVICE)).flatten(1)
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            predictions = head(embeddings).argmax(1).cpu()
            for actual, predicted, category in zip(labels, predictions, categories):
                confusion[int(actual), int(predicted)] += 1
                category_total[category] = category_total.get(category, 0) + 1
                category_correct[category] = category_correct.get(category, 0) + int(actual == predicted)
    report = {
        "samples": len(rows),
        "accuracy": float(confusion.diag().sum() / confusion.sum()),
        "confusion_matrix": confusion.tolist(),
        "per_gender": {
            name: float(confusion[index, index] / confusion[index].sum())
            for index, name in enumerate(GENDER_CLASSES)
        },
        "per_category": {
            category: category_correct[category] / total
            for category, total in sorted(category_total.items())
        },
        "policy": "balanced complement of gender training sample",
    }
    path = ROOT / "gender_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
