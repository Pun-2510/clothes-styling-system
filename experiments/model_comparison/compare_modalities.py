"""Compare text-only, image-only and combined fashion retrieval systems.

The benchmark uses the exact held-out test split saved by CLIP fine-tuning.
SBERT and both CLIP text encoders receive product names/descriptions without
the category label, ResNet18 receives images, and the combined baseline fuses
the SBERT and ResNet18 rankings with RRF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .encoders import SBERT_MODEL_ID, encode_clip_text, encode_resnet18, encode_sbert
from .evaluation import evaluate_category_rankings, rank_by_cosine
from .fusion import reciprocal_rank_fusion


REPOSITORY = Path(__file__).resolve().parents[2]
PROJECT = REPOSITORY / "project"
DEFAULT_RUN = PROJECT / "runs" / "clip_finetune_3epochs"
DEFAULT_RESNET_CHECKPOINT = REPOSITORY / "model" / "Web_Test" / "models" / "resnet_outfit.pth"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_CACHE = Path(__file__).resolve().parent / "cache"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--resnet-source",
        choices=("project-trained", "imagenet"),
        default="project-trained",
        help="Use the six-category project checkpoint or standard ImageNet weights.",
    )
    parser.add_argument("--resnet-checkpoint", type=Path, default=DEFAULT_RESNET_CHECKPOINT)
    parser.add_argument("--sbert-model", default=SBERT_MODEL_ID)
    parser.add_argument(
        "--clip-base-model",
        help="Defaults to the base model recorded in the selected training run.",
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        help="Defaults to <run-dir>/best.",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without loading models.")
    return parser.parse_args()


def build_evaluation_text(row):
    """Build text without category labels to prevent target-label leakage."""
    parts = []
    for label, column in (("Product", "product_name"), ("Description", "description")):
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            parts.append(f"{label}: {str(value).strip()}")
    return ". ".join(parts) or "Fashion product"


def _hash_payload(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_signature(products, model_identity):
    rows = [
        [str(row.product_id), str(row.image_sha256), str(row.evaluation_text)]
        for row in products.itertuples()
    ]
    return _hash_payload({"model": model_identity, "rows": rows})


def _cached_embeddings(cache_dir, name, signature, producer, use_cache):
    array_path = cache_dir / f"{name}.npy"
    metadata_path = cache_dir / f"{name}.json"
    if use_cache and array_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            embeddings = np.load(array_path, allow_pickle=False)
            if embeddings.ndim == 2 and len(embeddings) == metadata.get("rows"):
                print(f"Using cached {name} embeddings: {array_path}", flush=True)
                return embeddings, metadata["model"]

    started = time.perf_counter()
    embeddings, model_metadata = producer()
    elapsed = time.perf_counter() - started
    model_metadata = {**model_metadata, "encoding_seconds": elapsed}
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(array_path, embeddings, allow_pickle=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "rows": len(embeddings),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model": model_metadata,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return embeddings, model_metadata


def _prepare_output(requested):
    output = requested or DEFAULT_RESULTS / datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _metric_rows(system, input_type, metrics, ks):
    rows = []
    for k in ks:
        rows.append(
            {
                "system": system,
                "input": input_type,
                "task": "same-category product retrieval",
                "k": k,
                "precision": metrics[f"precision@{k}"],
                "recall": metrics[f"recall@{k}"],
                "f1": metrics[f"f1@{k}"],
                "category_macro_precision": metrics["category_macro"][f"precision@{k}"],
                "category_macro_recall": metrics["category_macro"][f"recall@{k}"],
                "category_macro_f1": metrics["category_macro"][f"f1@{k}"],
                "evaluated_queries": metrics["evaluated_queries"],
                "skipped_queries": metrics["skipped_queries"],
                "gallery_size": metrics["gallery_size"],
            }
        )
    return rows


def _category_rows(system, metrics, ks):
    rows = []
    for category, values in metrics["per_category"].items():
        for k in ks:
            rows.append(
                {
                    "system": system,
                    "category": category,
                    "k": k,
                    "precision": values[f"precision@{k}"],
                    "recall": values[f"recall@{k}"],
                    "f1": values[f"f1@{k}"],
                    "query_count": values["query_count"],
                    "evaluated_queries": values["evaluated_queries"],
                    "skipped_queries": values["skipped_queries"],
                }
            )
    return rows


def _summary(rows, products, settings):
    lines = [
        "# Modality comparison",
        "",
        f"The benchmark used {len(products)} held-out products and {products.category.nunique()} categories.",
        "The query product was excluded, and category labels were not included in any text-encoder input.",
        "",
        "| System | Input | K | Precision | Recall | F1 | Evaluated | Skipped |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['input']} | {row['k']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['evaluated_queries']} | {row['skipped_queries']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This table measures same-category retrieval, which is a proxy for relevance rather than human judgement of visual style.",
        "The combined system uses rank fusion because SBERT and ResNet18 embeddings are not in a shared vector space.",
        f"RRF uses k = {settings['rrf_k']}.",
        "The project-trained ResNet18 checkpoint was trained on six categories, so its result must not be presented as a 142-category classifier score.",
        "The fine-tuned CLIP checkpoint was trained with category-bearing prompts; its category-free text result is therefore an inference ablation, not a category-free retraining run.",
    ]
    return "\n".join(lines) + "\n"


def self_test():
    from .fusion import reciprocal_rank_fusion

    ids = np.array(["a", "b", "c", "d"])
    categories = np.array(["shoe", "shoe", "shoe", "hat"])
    rankings = [np.array([1, 2, 3]), np.array([0, 2, 3]), np.array([1, 0, 3]), np.array([0, 1, 2])]
    result = evaluate_category_rankings(rankings, ids, categories, (1, 2))
    assert result["evaluated_queries"] == 3
    assert result["skipped_queries"] == 1
    assert result["precision@1"] == 1.0
    assert result["recall@2"] == 1.0
    fused = reciprocal_rank_fusion([rankings, rankings], len(ids), 60)
    assert all(np.array_equal(left, right) for left, right in zip(fused, rankings))
    sample = pd.Series({"product_name": "Red dress", "description": "Short cotton dress", "category": "Dresses"})
    assert "Dresses" not in build_evaluation_text(sample)
    print("PASS: category metrics, self exclusion assumptions, RRF, and label-free text")


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    if args.batch_size < 1 or args.rrf_k < 1 or any(k < 1 for k in args.ks):
        raise ValueError("batch-size, rrf-k and every K value must be positive.")

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    sys.path.insert(0, str(PROJECT))
    from src.clip_data import file_sha256, load_experiment_data
    from src.clip_runtime import model_identity, same_model_identity

    products, experiment = load_experiment_data(args.run_dir)
    training_path = args.run_dir / "training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("status") != "complete":
        raise ValueError("The selected CLIP training run is not complete.")
    if training.get("dataset_sha256") != experiment.get("dataset_sha256"):
        raise ValueError("training.json and experiment.json refer to different datasets.")
    clip_base_model = args.clip_base_model or training["base_model"]
    clip_checkpoint = args.clip_checkpoint or args.run_dir / "best"
    if not clip_checkpoint.is_dir():
        raise FileNotFoundError(f"Fine-tuned CLIP checkpoint not found: {clip_checkpoint}")
    base_identity = model_identity(clip_base_model)
    checkpoint_identity = model_identity(clip_checkpoint)
    if not same_model_identity(training["base_model_identity"], base_identity):
        raise ValueError("The pretrained CLIP identity differs from the training record.")
    if not same_model_identity(training["checkpoint_identity"], checkpoint_identity):
        raise ValueError("The fine-tuned CLIP checkpoint differs from the training record.")
    products = products.loc[products.split == "test"].copy().reset_index(drop=True)
    if len(products) < 2:
        raise ValueError("The experiment needs at least two held-out test products.")
    products["evaluation_text"] = products.apply(build_evaluation_text, axis=1)
    ks = sorted(set(args.ks))

    settings = {
        "run_dir": str(args.run_dir.resolve()),
        "dataset_sha256": experiment["dataset_sha256"],
        "training_json_sha256": file_sha256(training_path),
        "split": "test",
        "rows": len(products),
        "categories": int(products.category.nunique()),
        "ks": ks,
        "rrf_k": args.rrf_k,
        "batch_size": args.batch_size,
        "device": args.device,
        "resnet_source": args.resnet_source,
        "resnet_checkpoint": str(args.resnet_checkpoint.resolve()),
        "sbert_model": args.sbert_model,
        "clip_base_model": str(clip_base_model),
        "clip_checkpoint": str(clip_checkpoint.resolve()),
        "text_template": "Product: {product_name}. Description: {description}",
        "category_in_text": False,
        "relevance": "same ground-truth category, excluding the query product",
        "aggregation": "per-query macro average; category macro also reported",
    }
    if args.dry_run:
        print(json.dumps(settings, indent=2, ensure_ascii=False))
        print("DRY RUN: dataset and configuration are valid; models were not loaded.")
        return

    output = _prepare_output(args.output_dir)
    evaluation_columns = [
        "product_id", "image_reference", "product_name", "description", "category",
        "image_path", "image_sha256", "evaluation_text",
    ]
    products[[column for column in evaluation_columns if column in products]].to_csv(
        output / "evaluation_catalog.csv", index=False
    )

    checkpoint_identity = (
        file_sha256(args.resnet_checkpoint)
        if args.resnet_source == "project-trained" and args.resnet_checkpoint.is_file()
        else "torchvision.ResNet18_Weights.DEFAULT"
    )
    resnet_signature = _cache_signature(
        products, {"architecture": "ResNet18", "source": args.resnet_source, "identity": checkpoint_identity}
    )
    sbert_signature = _cache_signature(
        products, {"architecture": "Sentence-BERT", "source": args.sbert_model, "category_in_text": False}
    )
    clip_pretrained_signature = _cache_signature(
        products,
        {"architecture": "CLIP text encoder", "identity": base_identity, "category_in_text": False},
    )
    clip_finetuned_signature = _cache_signature(
        products,
        {"architecture": "CLIP text encoder", "identity": checkpoint_identity, "category_in_text": False},
    )
    use_cache = not args.no_cache

    print(f"Encoding {len(products)} test images with ResNet18...", flush=True)
    image_embeddings, resnet_metadata = _cached_embeddings(
        args.cache_dir,
        "resnet18_test",
        resnet_signature,
        lambda: encode_resnet18(
            products.image_path.tolist(), args.device, args.batch_size,
            args.resnet_source, args.resnet_checkpoint,
        ),
        use_cache,
    )
    print(f"Encoding {len(products)} test descriptions with SBERT...", flush=True)
    text_embeddings, sbert_metadata = _cached_embeddings(
        args.cache_dir,
        "sbert_test",
        sbert_signature,
        lambda: encode_sbert(
            products.evaluation_text.tolist(), args.device, args.batch_size, args.sbert_model
        ),
        use_cache,
    )
    print(f"Encoding {len(products)} category-free descriptions with pretrained CLIP...", flush=True)
    clip_pretrained_text, clip_pretrained_metadata = _cached_embeddings(
        args.cache_dir,
        "clip_pretrained_category_free_text_test",
        clip_pretrained_signature,
        lambda: encode_clip_text(
            products.evaluation_text.tolist(), clip_base_model, args.device, args.batch_size
        ),
        use_cache,
    )
    print(f"Encoding {len(products)} category-free descriptions with fine-tuned CLIP...", flush=True)
    clip_finetuned_text, clip_finetuned_metadata = _cached_embeddings(
        args.cache_dir,
        "clip_finetuned_category_free_text_test",
        clip_finetuned_signature,
        lambda: encode_clip_text(
            products.evaluation_text.tolist(), clip_checkpoint, args.device, args.batch_size
        ),
        use_cache,
    )

    ids = products.product_id.astype(str).to_numpy()
    categories = products.category.fillna("").astype(str).to_numpy()
    image_rankings = rank_by_cosine(image_embeddings, ids)
    text_rankings = rank_by_cosine(text_embeddings, ids)
    clip_pretrained_text_rankings = rank_by_cosine(clip_pretrained_text, ids)
    clip_finetuned_text_rankings = rank_by_cosine(clip_finetuned_text, ids)
    combined_rankings = reciprocal_rank_fusion(
        [text_rankings, image_rankings], len(products), args.rrf_k
    )
    systems = {
        "sbert_text_only": ("text", text_rankings),
        "clip_pretrained_text_only": ("text", clip_pretrained_text_rankings),
        "clip_finetuned_text_only": ("text", clip_finetuned_text_rankings),
        "resnet18_image_only": ("image", image_rankings),
        "sbert_resnet18_rrf": ("image + text", combined_rankings),
    }
    metrics = {
        name: evaluate_category_rankings(rankings, ids, categories, ks)
        for name, (_, rankings) in systems.items()
    }
    rows = [
        row
        for name, (input_type, _) in systems.items()
        for row in _metric_rows(name, input_type, metrics[name], ks)
    ]
    category_rows = [
        row
        for name in systems
        for row in _category_rows(name, metrics[name], ks)
    ]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "settings": settings,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
        },
        "models": {
            "sbert": sbert_metadata,
            "clip_pretrained_text": clip_pretrained_metadata,
            "clip_finetuned_text": clip_finetuned_metadata,
            "resnet18": resnet_metadata,
        },
        "metrics": metrics,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "experiment.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(category_rows).to_csv(output / "comparison_by_category.csv", index=False)
    (output / "summary.md").write_text(
        _summary(rows, products, settings), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved modality comparison to: {output}")


if __name__ == "__main__":
    main()
