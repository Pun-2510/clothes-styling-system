"""Fine-tune CLIP image/text pairs and select a checkpoint on validation only."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.clip_data import file_sha256, load_experiment_data, prepare_splits
from src.clip_runtime import encode_products, load_clip, model_identity
from src.config import CLIP_MODEL, PROCESSED_CSV, RANDOM_SEED
from src.evaluation import evaluate_retrieval


class ProductPairs(Dataset):
    def __init__(self, products):
        self.rows = products.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.image_path) as image:
            return image.convert("RGB"), row.product_text, str(row.product_id)


class PairCollator:
    def __init__(self, processor, max_length):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, rows):
        images, texts, ids = zip(*rows)
        inputs = self.processor(images=list(images), text=list(texts), padding=True,
                                truncation=True, max_length=self.max_length,
                                return_tensors="pt")
        # Multiple photos of one product or identical prompts are positives too.
        positives = torch.tensor([
            [left_id == right_id or left_text == right_text
             for right_id, right_text in zip(ids, texts)]
            for left_id, left_text in zip(ids, texts)
        ], dtype=torch.bool)
        return inputs, positives


def contrastive_loss(logits, positives):
    """Symmetric CLIP cross entropy, supporting multiple positives per row."""
    targets = positives.to(dtype=logits.dtype)
    image_targets = targets / targets.sum(dim=1, keepdim=True)
    text_targets = targets.T / targets.T.sum(dim=1, keepdim=True)
    return -(image_targets * logits.log_softmax(dim=1)).sum(dim=1).mean() / 2 - (
        text_targets * logits.T.log_softmax(dim=1)
    ).sum(dim=1).mean() / 2


def configure_trainable(model, mode):
    if mode not in {"full", "projections"}:
        raise ValueError("Trainable mode must be full or projections.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(mode == "full" or name.startswith(
            ("visual_projection.", "text_projection.", "logit_scale")
        ))
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_epoch(model, loader, optimizer, parameters, device):
    model.train()
    total_loss, count = 0.0, 0
    for inputs, positives in tqdm(loader, desc="Fine-tuning"):
        if len(positives) < 2:
            continue  # A singleton has no contrastive negatives.
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs.to(device), return_loss=False)
        loss = contrastive_loss(output.logits_per_image, positives.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; reduce learning rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))
        total_loss += loss.item() * len(positives)
        count += len(positives)
    if not count:
        raise ValueError("No training batches with at least two products.")
    return total_loss / count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROCESSED_CSV)
    parser.add_argument("--model", default=CLIP_MODEL, help="Original pretrained model ID or local directory")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/clip_finetune"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--trainable", choices=("full", "projections"), default="full")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-only", action="store_true", help="Validate images and save fixed splits without loading CLIP")
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.epochs < 1 or args.batch_size < 2 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        raise ValueError("Need epochs >= 1, batch size >= 2, positive LR and nonnegative decay.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    run_dir = args.run_dir
    metadata_path = run_dir / "experiment.json"
    if (run_dir / "training.json").exists() or (run_dir / "best").exists():
        raise FileExistsError("This run already contains training; choose a new --run-dir.")
    if metadata_path.exists():
        products, metadata = load_experiment_data(run_dir)
        if (metadata["source_csv_sha256"] != file_sha256(args.csv)
                or metadata["seed"] != args.seed or metadata["val_fraction"] != args.val_fraction
                or metadata["test_fraction"] != args.test_fraction):
            raise ValueError("Split settings or source CSV changed; use a new --run-dir.")
    else:
        products, metadata = prepare_splits(args.csv, args.seed, args.val_fraction, args.test_fraction)
        run_dir.mkdir(parents=True, exist_ok=True)
        products.to_csv(run_dir / "dataset.csv", index=False)
        metadata.update(source_csv=str(args.csv.resolve()), source_csv_sha256=file_sha256(args.csv),
                        dataset_sha256=file_sha256(run_dir / "dataset.csv"))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    if args.prepare_only:
        return

    device = torch.device(args.device)
    model, processor = load_clip(args.model, device)
    parameters = configure_trainable(model, args.trainable)
    print(f"Device: {device}; trainable parameters: {sum(p.numel() for p in parameters):,}", flush=True)
    train = products.loc[products.split == "train"]
    validation = products.loc[products.split == "validation"]
    loader = DataLoader(ProductPairs(train), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0,
                        collate_fn=PairCollator(processor, model.config.text_config.max_position_embeddings))
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    training = {"base_model": str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model,
                "base_model_identity": model_identity(args.model),
                "dataset_sha256": metadata["dataset_sha256"], "trainable": args.trainable,
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "batch_size": args.batch_size, "epochs": args.epochs, "seed": args.seed,
                "selection_metric": "mean validation text_to_image/image_to_text recall@1",
                "history": [], "status": "running"}
    training_path = run_dir / "training.json"
    training_path.write_text(json.dumps(training, indent=2), encoding="utf-8")
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, parameters, device)
        images, texts = encode_products(model, processor, validation, device, args.batch_size)
        metrics = evaluate_retrieval(images, texts, validation)
        score = (metrics["text_to_image"]["recall@1"] + metrics["image_to_text"]["recall@1"]) / 2
        training["history"].append({"epoch": epoch, "train_loss": loss, "validation": metrics})
        if score > best_score:
            best_score = score
            model.save_pretrained(run_dir / "best")
            processor.save_pretrained(run_dir / "best")
            training.update(best_epoch=epoch, best_validation_score=score)
        training_path.write_text(json.dumps(training, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}: loss={loss:.6f}, validation mean Recall@1={score:.4f}", flush=True)
    training["status"] = "complete"
    training["checkpoint_identity"] = model_identity(run_dir / "best")
    training_path.write_text(json.dumps(training, indent=2), encoding="utf-8")
    print(f"Checkpoint: {run_dir / 'best'}\nCompare: python -m src.compare_clip --run-dir {run_dir}")


if __name__ == "__main__":
    main()
