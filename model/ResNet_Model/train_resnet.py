"""Train ResNet18 with base fashion classes plus local custom classes."""

import json
import random
from collections import Counter

import torch
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    BATCH_SIZE, BEST_MODEL_PATH, CONTRASTIVE_TEMPERATURE, CONTRASTIVE_WEIGHT,
    CUSTOM_CLASSES, CUSTOM_DATA_DIR,
    DATASET_NAME, DATASET_SPLIT, DEVICE, EARLY_STOPPING_PATIENCE, EPOCHS,
    GRADIENT_CLIP_NORM, IMAGE_SIZE, LABEL_SMOOTHING, LEARNING_RATE,
    LR_FACTOR, LR_PATIENCE, MAX_IMAGES, MIN_CUSTOM_IMAGES_PER_CLASS,
    MIN_LEARNING_RATE, MODEL_PATH,
    NORMALIZE_MEAN, NORMALIZE_STD, RANDOM_STATE, RESIZE_SIZE,
    SELECTED_CLASSES, TEST_SIZE, WEIGHT_DECAY,
)
from model import FashionResNet
from train import train_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_rows():
    custom_paths = {}
    for category in CUSTOM_CLASSES:
        directory = CUSTOM_DATA_DIR / category
        paths = sorted(
            path for path in directory.glob("**/*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if directory.is_dir() else []
        if len(paths) < MIN_CUSTOM_IMAGES_PER_CLASS:
            raise ValueError(
                f"Class {category!r} requires at least "
                f"{MIN_CUSTOM_IMAGES_PER_CLASS} images in {directory}; "
                f"found {len(paths)}. ResNet cannot learn an unseen class."
            )
        custom_paths[category] = paths

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    base = dataset.filter(lambda row: row["articleType"] in SELECTED_CLASSES)
    rng = random.Random(RANDOM_STATE)
    base_indices = list(range(len(base)))
    if len(base_indices) > MAX_IMAGES:
        base_indices = rng.sample(base_indices, MAX_IMAGES)
    rows = [{
        "image": base[index]["image"].convert("RGB"),
        "category": str(base[index]["articleType"]),
    } for index in base_indices]

    custom_counts = {category: len(paths)
                     for category, paths in custom_paths.items()}
    for category, paths in custom_paths.items():
        rows.extend({"image": str(path.resolve()), "category": category}
                    for path in paths)
    return rows, custom_counts, base_indices


class FashionDataset(Dataset):
    def __init__(self, rows, indices, class_to_idx, transform):
        self.rows = rows
        self.indices = indices
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        row = self.rows[self.indices[position]]
        value = row["image"]
        if isinstance(value, Image.Image):
            image = value.copy().convert("RGB")
        else:
            with Image.open(value) as source:
                image = source.convert("RGB")
        return self.transform(image), self.class_to_idx[row["category"]]


def main():
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    rows, custom_counts, base_indices = load_rows()
    classes = list(SELECTED_CLASSES) + [
        category for category in CUSTOM_CLASSES if category not in SELECTED_CLASSES
    ]
    class_to_idx = {category: index for index, category in enumerate(classes)}
    labels = [class_to_idx[row["category"]] for row in rows]
    indices = list(range(len(rows)))
    train_indices, val_indices = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    train_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    train_counts = Counter(labels[index] for index in train_indices)
    train_weights = [
        1.0 / train_counts[labels[index]] for index in train_indices
    ]
    train_loader = DataLoader(
        FashionDataset(rows, train_indices, class_to_idx, train_transform),
        batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        ),
    )
    val_loader = DataLoader(
        FashionDataset(rows, val_indices, class_to_idx, val_transform),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    print(f"Device: {DEVICE}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train: {len(train_indices)}; validation: {len(val_indices)}")
    print(f"Custom images: {custom_counts}")

    model = FashionResNet(num_classes=len(classes), pretrained=True)
    warm_started = False
    if MODEL_PATH.is_file():
        existing = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if list(existing.get("classes", [])) == classes:
            model.load_state_dict(existing["model_state_dict"])
            warm_started = True
            print(f"Warm start: {MODEL_PATH}")
    model = model.to(DEVICE)
    model, train_accuracy, val_accuracy, best_accuracy = train_model(
        model, train_loader, val_loader, DEVICE, EPOCHS, LEARNING_RATE,
        WEIGHT_DECAY, BEST_MODEL_PATH,
        label_smoothing=LABEL_SMOOTHING,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        lr_patience=LR_PATIENCE,
        lr_factor=LR_FACTOR,
        min_learning_rate=MIN_LEARNING_RATE,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
        contrastive_weight=CONTRASTIVE_WEIGHT,
        contrastive_temperature=CONTRASTIVE_TEMPERATURE,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "best_validation_accuracy": best_accuracy,
        "training": {
            "initialization": "ImageNet pretrained ResNet18",
            "warm_started_from_existing_checkpoint": warm_started,
            "selection_metric": "macro validation accuracy",
            "epochs": EPOCHS,
            "label_smoothing": LABEL_SMOOTHING,
            "contrastive_weight": CONTRASTIVE_WEIGHT,
            "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "custom_counts": custom_counts,
            "random_state": RANDOM_STATE,
            "dataset_indices": base_indices,
        },
    }
    torch.save(checkpoint, MODEL_PATH)
    MODEL_PATH.with_suffix(".json").write_text(
        json.dumps({key: value for key, value in checkpoint.items()
                    if key != "model_state_dict"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved best checkpoint: {MODEL_PATH}")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
