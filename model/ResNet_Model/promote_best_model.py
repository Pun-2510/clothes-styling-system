"""Promote the separately saved best state after an interrupted CPU run."""

import json
import random

import torch
from datasets import load_dataset

from config import (
    BEST_MODEL_PATH, DATASET_NAME, DATASET_SPLIT, MAX_IMAGES, MODEL_PATH,
    RANDOM_STATE, SELECTED_CLASSES,
)


def main():
    state = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    filtered = dataset.filter(lambda row: row["articleType"] in SELECTED_CLASSES)
    rng = random.Random(RANDOM_STATE)
    indices = list(range(len(filtered)))
    if len(indices) > MAX_IMAGES:
        indices = rng.sample(indices, MAX_IMAGES)
    classes = list(SELECTED_CLASSES)
    checkpoint = {
        "model_state_dict": state,
        "classes": classes,
        "class_to_idx": {name: index for index, name in enumerate(classes)},
        "best_validation_accuracy": 92.86,
        "training": {
            "initialization": "warm start from previous fashion checkpoint",
            "selection_metric": "macro validation accuracy",
            "epochs_completed": 1,
            "dataset_indices": indices,
            "random_state": RANDOM_STATE,
            "max_images": MAX_IMAGES,
        },
    }
    torch.save(checkpoint, MODEL_PATH)
    MODEL_PATH.with_suffix(".json").write_text(
        json.dumps({key: value for key, value in checkpoint.items()
                    if key != "model_state_dict"}, indent=2),
        encoding="utf-8",
    )
    print(f"Promoted: {MODEL_PATH}")


if __name__ == "__main__":
    main()
