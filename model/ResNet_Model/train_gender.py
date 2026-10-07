"""Train a balanced Men/Women head on frozen fashion ResNet embeddings."""

import os
import random
from copy import deepcopy
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    DATASET_NAME, DATASET_SPLIT, DEVICE, IMAGE_SIZE, MODEL_PATH,
    NORMALIZE_MEAN, NORMALIZE_STD, RANDOM_STATE, RESIZE_SIZE, SELECTED_CLASSES,
)
from gender import GENDER_CLASSES, GenderHead
from model import FashionResNet


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "Web_Test" / "models" / "resnet_gender.pth"
SAMPLES_PER_GENDER = 1500


def normalize_gender(value):
    if value in {"Men", "Boys"}:
        return "Men"
    if value in {"Women", "Girls"}:
        return "Women"
    return None


class Rows(Dataset):
    def __init__(self, dataset, rows, transform):
        self.dataset, self.rows, self.transform = dataset, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source_index, label = self.rows[index]
        image = self.dataset[source_index]["image"].convert("RGB")
        return self.transform(image), label


def main():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    pools = {name: [] for name in GENDER_CLASSES}
    for index, row in enumerate(dataset):
        gender = normalize_gender(row["gender"])
        if gender and row["articleType"] in SELECTED_CLASSES:
            pools[gender].append(index)
    rng = random.Random(RANDOM_STATE)
    selected = []
    for gender in GENDER_CLASSES:
        indices = rng.sample(pools[gender], min(SAMPLES_PER_GENDER, len(pools[gender])))
        selected.extend((index, GENDER_CLASSES.index(gender)) for index in indices)
    labels = [label for _, label in selected]
    train_rows, val_rows = train_test_split(
        selected, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    backbone_model = FashionResNet(len(checkpoint["classes"]), pretrained=False)
    backbone_model.load_state_dict(checkpoint["model_state_dict"])
    backbone = torch.nn.Sequential(*list(backbone_model.model.children())[:-1]).to(DEVICE).eval()

    def encode(rows):
        loader = DataLoader(Rows(dataset, rows, transform), batch_size=32, shuffle=False)
        features, targets = [], []
        with torch.inference_mode():
            for images, labels_batch in loader:
                value = backbone(images.to(DEVICE)).flatten(1)
                value = torch.nn.functional.normalize(value, dim=1)
                features.append(value.cpu())
                targets.append(labels_batch)
        return torch.cat(features), torch.cat(targets)

    print(f"Encoding {len(train_rows)} train and {len(val_rows)} validation images...")
    train_x, train_y = encode(train_rows)
    val_x, val_y = encode(val_rows)
    head = GenderHead().to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best_state, best_accuracy, stale = None, 0.0, 0
    for epoch in range(50):
        head.train()
        order = torch.randperm(len(train_x))
        for offset in range(0, len(order), 64):
            indices = order[offset:offset + 64]
            optimizer.zero_grad()
            loss = criterion(head(train_x[indices].to(DEVICE)), train_y[indices].to(DEVICE))
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            predictions = head(val_x.to(DEVICE)).argmax(1).cpu()
        accuracy = float((predictions == val_y).float().mean())
        print(f"Epoch {epoch + 1}: validation accuracy {accuracy * 100:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy, best_state, stale = accuracy, deepcopy(head.state_dict()), 0
        else:
            stale += 1
            if stale >= 6:
                break
    torch.save({
        "model_state_dict": best_state,
        "classes": GENDER_CLASSES,
        "validation_accuracy": best_accuracy,
        "samples": {"train": len(train_rows), "validation": len(val_rows)},
        "backbone_checkpoint": str(MODEL_PATH),
    }, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}; best accuracy {best_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
