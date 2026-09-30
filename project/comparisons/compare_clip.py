"""Compare pretrained and fine-tuned CLIP on the exact same held-out test set."""

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from src.clip_data import load_experiment_data, build_product_text, file_sha256, resolve_image_path
from src.clip_runtime import (encode_products, load_clip, model_identity,
                              same_model_identity, validate_embedding_metadata)
from src.evaluation import evaluate_retrieval


def catalog_test_rows(test, catalog):
    """Intersect with the original held-out split, never create a new test split."""
    keys = [(str(row.product_id), file_sha256(resolve_image_path(row.image_path)))
            for row in catalog.itertuples()]
    if len(keys) != len(set(keys)):
        raise ValueError("Catalog contains duplicate product/image pairs.")
    positions = {key: index for index, key in enumerate(keys)}
    selected, indices = [], []
    for index, row in test.iterrows():
        position = positions.get((str(row.product_id), row.image_sha256))
        if position is None:
            continue
        candidate = catalog.iloc[position]
        if (build_product_text(candidate) != build_product_text(row)
                or str(candidate.category) != str(row.category)):
            raise ValueError(f"Catalog text/category differs from the training run: {row.product_id}")
        selected.append(index)
        indices.append(position)
    if len(selected) < 2:
        raise ValueError("Need at least two original held-out test products in the selected catalog. "
                         "Increase --num-products, relax filters, or train a new experiment.")
    return test.loc[selected].reset_index(drop=True), indices


def load_cached_pair(directory, identity, csv_path, catalog_size, indices):
    arrays = []
    for name in ("image_embeddings.npy", "text_embeddings.npy"):
        path = directory / name
        validate_embedding_metadata(path, identity, csv_path)
        features = np.load(path, allow_pickle=False)
        if features.ndim != 2 or len(features) != catalog_size:
            raise ValueError(f"Embedding dimensions do not match the catalog: {path}")
        arrays.append(features[indices])
    return tuple(arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/clip_finetune"))
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Report destination under comparisons/results/.",
    )
    parser.add_argument("--catalog-csv", type=Path, help="Catalog used for cached embeddings")
    parser.add_argument("--cache-root", type=Path, help="Contains pretrained/ and finetuned/ embeddings")
    args = parser.parse_args()
    if bool(args.catalog_csv) != bool(args.cache_root):
        parser.error("--catalog-csv and --cache-root must be supplied together")
    if args.batch_size < 1 or any(k <= 0 for k in args.ks):
        parser.error("batch-size and ks must be positive")
    products, metadata = load_experiment_data(args.run_dir)
    training = json.loads((args.run_dir / "training.json").read_text(encoding="utf-8"))
    if training["dataset_sha256"] != metadata["dataset_sha256"] or training["status"] != "complete":
        raise ValueError("Need completed training on this exact dataset split.")
    if (not same_model_identity(
            training["base_model_identity"], model_identity(training["base_model"])
        ) or not same_model_identity(
            training["checkpoint_identity"], model_identity(args.run_dir / "best")
        )):
        raise ValueError("Baseline or fine-tuned checkpoint changed since training.")
    test = products.loc[products.split == "test"].reset_index(drop=True)
    original_test_size = len(test)
    catalog, indices = None, None
    if args.catalog_csv:
        catalog = pd.read_csv(args.catalog_csv, dtype={"product_id": str})
        test, indices = catalog_test_rows(test, catalog)
    device = torch.device(args.device)
    report = {"split": "test", "dataset_sha256": metadata["dataset_sha256"],
              "gallery": "test rows only; same gallery and queries for both models",
              "ks": sorted(set(args.ks)), "best_epoch": training["best_epoch"],
              "selection_metric": training["selection_metric"],
              "metric_aggregation": "macro average over eligible queries",
              "precision_definition": "relevant retrieved / actual returned in top-k",
              "recall_definition": "relevant retrieved / all relevant in gallery",
              "f1_definition": "per-query harmonic mean of precision and recall",
              "category_relevance": "same ground-truth category, excluding the query product; proxy only",
              "pair_relevance": "same product_id", "models": {}, "metrics": {}}
    report.update(original_test_size=original_test_size, evaluated_test_size=len(test),
                  omitted_test_products=original_test_size - len(test),
                  catalog_size=len(catalog) if catalog is not None else None,
                  catalog_sha256=file_sha256(args.catalog_csv) if args.catalog_csv else None,
                  evaluation_protocol="original held-out test intersected with catalog" if catalog is not None
                  else "original held-out test", training_run=str(args.run_dir.resolve()))
    for label, source in (("pretrained", training["base_model"]),
                          ("finetuned", str(args.run_dir / "best"))):
        print(f"Evaluating {label}: {source}", flush=True)
        report["models"][label] = model_identity(source)
        if args.cache_root:
            images, texts = load_cached_pair(args.cache_root / label, report["models"][label],
                                            args.catalog_csv, len(catalog), indices)
        else:
            model, processor = load_clip(source, device)
            images, texts = encode_products(model, processor, test, device, args.batch_size)
            del model, processor
        report["metrics"][label] = evaluate_retrieval(images, texts, test, args.ks)
        del images, texts
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = []
    for task, baseline in report["metrics"]["pretrained"].items():
        tuned = report["metrics"]["finetuned"][task]
        for metric_name in ("precision", "recall", "f1"):
            for k in report["ks"]:
                metric = f"{metric_name}@{k}"
                before, after = baseline[metric], tuned[metric]
                rows.append({"task": task, "metric": metric, "pretrained": before,
                             "finetuned": after,
                             "delta_percentage_points": None if before is None else 100 * (after - before),
                             "evaluated_queries": baseline["evaluated_queries"],
                             "skipped_queries": baseline["skipped_queries"],
                             "gallery_size": baseline["gallery_size"]})
    report["comparison"] = rows
    category_rows = []
    for task, baseline in report["metrics"]["pretrained"].items():
        tuned = report["metrics"]["finetuned"][task]
        baseline_categories = baseline.get("per_category", {})
        tuned_categories = tuned.get("per_category", {})
        for category in sorted(set(baseline_categories) | set(tuned_categories)):
            before_row = baseline_categories.get(category, {})
            after_row = tuned_categories.get(category, {})
            for metric_name in ("precision", "recall", "f1"):
                for k in report["ks"]:
                    metric = f"{metric_name}@{k}"
                    before = before_row.get(metric)
                    after = after_row.get(metric)
                    category_rows.append(
                        {
                            "task": task,
                            "category": category,
                            "metric": metric,
                            "pretrained": before,
                            "finetuned": after,
                            "delta_percentage_points": (
                                None
                                if before is None or after is None
                                else 100 * (after - before)
                            ),
                            "query_count": before_row.get("query_count", 0),
                            "evaluated_queries": before_row.get("evaluated_queries", 0),
                            "skipped_queries": before_row.get("skipped_queries", 0),
                        }
                    )
    report["category_comparison"] = category_rows
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "comparison.csv", index=False)
    pd.DataFrame(category_rows).to_csv(
        output_dir / "comparison_by_category.csv", index=False
    )
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "Saved comparison.json, comparison.csv and "
        f"comparison_by_category.csv in {output_dir}"
    )


if __name__ == "__main__":
    main()
