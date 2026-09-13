"""Compare pretrained and fine-tuned CLIP on the exact same held-out test set."""

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch

from src.clip_data import load_experiment_data
from src.clip_runtime import encode_products, load_clip, model_identity
from src.evaluation import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/clip_finetune"))
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or any(k <= 0 for k in args.ks):
        parser.error("batch-size and ks must be positive")
    products, metadata = load_experiment_data(args.run_dir)
    training = json.loads((args.run_dir / "training.json").read_text(encoding="utf-8"))
    if training["dataset_sha256"] != metadata["dataset_sha256"] or training["status"] != "complete":
        raise ValueError("Need completed training on this exact dataset split.")
    if (training["base_model_identity"] != model_identity(training["base_model"])
            or training["checkpoint_identity"] != model_identity(args.run_dir / "best")):
        raise ValueError("Baseline or fine-tuned checkpoint changed since training.")
    test = products.loc[products.split == "test"].reset_index(drop=True)
    device = torch.device(args.device)
    report = {"split": "test", "dataset_sha256": metadata["dataset_sha256"],
              "gallery": "test rows only; same gallery and queries for both models",
              "ks": sorted(set(args.ks)), "best_epoch": training["best_epoch"],
              "selection_metric": training["selection_metric"],
              "category_relevance": "same ground-truth category, excluding the query product; proxy only",
              "pair_relevance": "same product_id", "models": {}, "metrics": {}}
    for label, source in (("pretrained", training["base_model"]),
                          ("finetuned", str(args.run_dir / "best"))):
        print(f"Evaluating {label}: {source}", flush=True)
        report["models"][label] = model_identity(source)
        model, processor = load_clip(source, device)
        images, texts = encode_products(model, processor, test, device, args.batch_size)
        report["metrics"][label] = evaluate_retrieval(images, texts, test, args.ks)
        del model, processor, images, texts
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = []
    for task, baseline in report["metrics"]["pretrained"].items():
        tuned = report["metrics"]["finetuned"][task]
        for k in report["ks"]:
            metric = f"recall@{k}"
            before, after = baseline[metric], tuned[metric]
            rows.append({"task": task, "metric": metric, "pretrained": before,
                         "finetuned": after,
                         "delta_percentage_points": None if before is None else 100 * (after - before),
                         "evaluated_queries": baseline["evaluated_queries"],
                         "skipped_queries": baseline["skipped_queries"],
                         "gallery_size": baseline["gallery_size"]})
    report["comparison"] = rows
    (args.run_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.run_dir / "comparison.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved comparison.json and comparison.csv in {args.run_dir}")


if __name__ == "__main__":
    main()
