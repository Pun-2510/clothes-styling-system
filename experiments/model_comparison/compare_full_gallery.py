"""Run unified baselines, CLIP alignment and serving comparisons on 5,000 products."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_modalities import (
    DEFAULT_RESNET_CHECKPOINT,
    DEFAULT_RUN,
    PROJECT,
    REPOSITORY,
    _cache_signature,
    _cached_embeddings,
    _category_rows,
    _metric_rows,
    _prepare_output,
    build_evaluation_text,
)
from .encoders import (
    SBERT_MODEL_ID,
    encode_clip_images,
    encode_clip_text,
    encode_resnet18,
    encode_sbert,
)
from .evaluation import (
    evaluate_query_gallery_category_rankings,
    evaluate_query_gallery_identity_rankings,
    rank_queries_by_cosine,
)
from .fusion import (
    clip_query_gallery_score_fusion,
    clip_text_query_gallery_score_fusion,
    reciprocal_rank_fusion,
)


DEFAULT_CATALOG = PROJECT / "data" / "processed" / "products.csv"
DEFAULT_CACHE = Path(__file__).resolve().parent / "cache" / "full_gallery_5000"
DEFAULT_FINETUNED_IMAGES = PROJECT / "embeddings" / "image_embeddings.npy"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--catalog-csv", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--expected-gallery-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--clip-image-weight", type=float, default=0.5)
    parser.add_argument("--clip-text-image-weight", type=float, default=0.5)
    parser.add_argument(
        "--resnet-source",
        choices=("project-trained", "imagenet"),
        default="project-trained",
    )
    parser.add_argument("--resnet-checkpoint", type=Path, default=DEFAULT_RESNET_CHECKPOINT)
    parser.add_argument("--sbert-model", default=SBERT_MODEL_ID)
    parser.add_argument("--clip-base-model")
    parser.add_argument("--clip-checkpoint", type=Path)
    parser.add_argument(
        "--finetuned-image-embeddings",
        type=Path,
        default=DEFAULT_FINETUNED_IMAGES,
        help="Verified full-catalog image embeddings to reuse; encode when absent.",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_arguments(args):
    if args.batch_size < 1 or args.rrf_k < 1 or args.expected_gallery_size < 2:
        raise ValueError("Batch size, RRF k and expected gallery size must be positive.")
    if any(k < 1 for k in args.ks):
        raise ValueError("Every K value must be positive.")
    if not 0.0 <= args.clip_image_weight <= 1.0:
        raise ValueError("clip-image-weight must be between 0 and 1.")
    if not 0.0 <= args.clip_text_image_weight <= 1.0:
        raise ValueError("clip-text-image-weight must be between 0 and 1.")


def _load_and_validate_data(args, file_sha256):
    sys.path.insert(0, str(PROJECT))
    from src.clip_data import load_experiment_data

    products, experiment = load_experiment_data(args.run_dir)
    training_path = args.run_dir / "training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("status") != "complete":
        raise ValueError("The selected CLIP training run is not complete.")
    if training.get("dataset_sha256") != experiment.get("dataset_sha256"):
        raise ValueError("training.json and experiment.json refer to different datasets.")

    gallery = pd.read_csv(args.catalog_csv, dtype={"product_id": str})
    required = {
        "product_id", "image_reference", "product_name", "description",
        "category", "image_path", "image_sha256",
    }
    if not required.issubset(gallery.columns):
        raise ValueError(f"Catalog must contain: {sorted(required)}")
    if len(gallery) != args.expected_gallery_size:
        raise ValueError(
            f"Expected {args.expected_gallery_size} gallery products, found {len(gallery)}."
        )
    if gallery.product_id.duplicated().any():
        raise ValueError("Full-gallery evaluation requires unique product IDs.")
    mandatory = [
        "product_id", "image_reference", "product_name", "category",
        "image_path", "image_sha256",
    ]
    if gallery[mandatory].isna().any().any():
        raise ValueError("Catalog identity, product name, category and image fields cannot be missing.")

    gallery["evaluation_text"] = gallery.apply(build_evaluation_text, axis=1)
    saved = products.set_index(products.product_id.astype(str), drop=False)
    catalog = gallery.set_index(gallery.product_id.astype(str), drop=False)
    if set(saved.index) != set(catalog.index):
        raise ValueError("The full gallery is not the exact 5,000-product training catalog.")
    for product_id, row in saved.iterrows():
        candidate = catalog.loc[product_id]
        if (
            str(row.image_sha256) != str(candidate.image_sha256)
            or str(row.category) != str(candidate.category)
            or build_evaluation_text(row) != str(candidate.evaluation_text)
        ):
            raise ValueError(f"Catalog content changed for product: {product_id}")

    query_split = getattr(args, "query_split", "test")
    if query_split not in {"validation", "test"}:
        raise ValueError("query_split must be either validation or test.")
    queries = (
        products.loc[products.split == query_split]
        .copy()
        .reset_index(drop=True)
    )
    if queries.empty:
        raise ValueError(f"No products were assigned to the {query_split} split.")
    queries["evaluation_text"] = queries.apply(build_evaluation_text, axis=1)
    gallery_positions = {value: index for index, value in enumerate(gallery.product_id)}
    query_positions = np.asarray(
        [gallery_positions[str(product_id)] for product_id in queries.product_id],
        dtype=np.int64,
    )
    return products, queries, gallery, query_positions, experiment, training, training_path


def _load_verified_finetuned_images(path, identity, catalog_csv, rows):
    from src.clip_runtime import validate_embedding_metadata

    validate_embedding_metadata(path, identity, catalog_csv)
    embeddings = np.load(path, allow_pickle=False)
    if embeddings.ndim != 2 or len(embeddings) != rows or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid full-catalog embeddings: {path}")
    return embeddings.astype(np.float32, copy=False), {
        "architecture": "CLIP vision encoder",
        "source": str(path.resolve()),
        "reused_verified_embeddings": True,
        "embedding_dimension": int(embeddings.shape[1]),
    }


def _annotated_rows(system_specs, metrics, ks, task):
    rows = []
    for name, specification in system_specs.items():
        for metric_row in _metric_rows(
            name,
            specification["input"],
            metrics[name],
            ks,
            task=task,
        ):
            rows.append(
                {
                    "comparison_group": specification["comparison_group"],
                    "role": specification["role"],
                    "deployed": specification["deployed"],
                    "formula": specification["formula"],
                    **metric_row,
                }
            )
    return rows


def _annotated_category_rows(system_specs, metrics, ks, task):
    rows = []
    for name, specification in system_specs.items():
        for metric_row in _category_rows(name, metrics[name], ks):
            rows.append(
                {
                    "comparison_group": specification["comparison_group"],
                    "role": specification["role"],
                    "deployed": specification["deployed"],
                    "formula": specification["formula"],
                    "task": task,
                    **metric_row,
                }
            )
    return rows


def _markdown_table(title, rows):
    lines = [
        f"## {title}",
        "",
        "| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['role']} | {row['input']} | "
            f"`{row['formula']}` | {row['k']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['ndcg']:.4f} |"
        )
    return lines


def _summary(controlled_rows, alignment_rows, serving_rows, query_count, gallery_count):
    lines = [
        "# Unified 5,000-product comparison",
        "",
        f"The benchmark used {query_count} held-out test queries against {gallery_count} catalog products.",
        "All tables use the same query rows, catalog rows, K values and model checkpoints.",
        "",
    ]
    lines += _markdown_table("A. Controlled modality baselines", controlled_rows)
    lines += [""]
    lines += _markdown_table("B. Pretrained CLIP versus fine-tuned CLIP", alignment_rows)
    lines += [""]
    lines += _markdown_table("C. Website serving ablation", serving_rows)
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Controlled and serving tables exclude the query product and use same-category relevance.",
        "The CLIP alignment table keeps the paired product and uses exact product-ID relevance.",
        "Category labels are used only after unrestricted ranking; they are omitted from encoder text.",
        "Serving rows do not apply category filtering, translation or attribute overrides.",
        "Results must not be mixed with the 473 x 473 test-only protocol without naming the gallery size.",
    ]
    return "\n".join(lines) + "\n"


def self_test():
    query_ids = np.array(["a", "d"])
    gallery_ids = np.array(["a", "b", "c", "d"])
    query_categories = np.array(["shirt", "hat"])
    gallery_categories = np.array(["shirt", "shirt", "shoe", "hat"])
    gallery = np.eye(4, dtype=np.float32)
    queries = gallery[[0, 3]]
    rankings = rank_queries_by_cosine(queries, gallery, query_ids, gallery_ids)
    assert all(query_id not in gallery_ids[ranking] for query_id, ranking in zip(query_ids, rankings))
    metrics = evaluate_query_gallery_category_rankings(
        rankings, query_ids, query_categories, gallery_ids, gallery_categories, (1, 2)
    )
    assert metrics["evaluated_queries"] == 1
    assert metrics["skipped_queries"] == 1
    assert metrics["ndcg@1"] == 1.0
    fused = clip_query_gallery_score_fusion(
        queries, queries, gallery, gallery, query_ids, gallery_ids
    )
    assert len(fused) == len(query_ids)
    text_fused = clip_text_query_gallery_score_fusion(
        queries, gallery, gallery, query_ids, gallery_ids
    )
    assert len(text_fused) == len(query_ids)
    paired_rankings = rank_queries_by_cosine(
        queries,
        gallery,
        query_ids,
        gallery_ids,
        exclude_same_product=False,
    )
    paired_metrics = evaluate_query_gallery_identity_rankings(
        paired_rankings, query_ids, query_categories, gallery_ids, (1, 2)
    )
    assert paired_metrics["recall@1"] == 1.0
    assert paired_metrics["ndcg@1"] == 1.0
    print(
        "PASS: shared full-gallery protocol, category/identity metrics, "
        "text serving fusion and multimodal CLIP fusion"
    )


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    _validate_arguments(args)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    sys.path.insert(0, str(PROJECT))
    from src.clip_data import file_sha256
    from src.clip_runtime import model_identity, same_model_identity

    _, queries, gallery, query_positions, experiment, training, training_path = (
        _load_and_validate_data(args, file_sha256)
    )
    clip_base_model = args.clip_base_model or training["base_model"]
    clip_checkpoint = args.clip_checkpoint or args.run_dir / "best"
    base_identity = model_identity(clip_base_model)
    clip_checkpoint_identity = model_identity(clip_checkpoint)
    if not same_model_identity(training["base_model_identity"], base_identity):
        raise ValueError("The pretrained CLIP identity differs from the training record.")
    if not same_model_identity(training["checkpoint_identity"], clip_checkpoint_identity):
        raise ValueError("The fine-tuned CLIP checkpoint differs from the training record.")

    ks = sorted(set(args.ks))
    settings = {
        "protocol": "held-out test queries against full catalog gallery",
        "run_dir": str(args.run_dir.resolve()),
        "catalog_csv": str(args.catalog_csv.resolve()),
        "dataset_sha256": experiment["dataset_sha256"],
        "training_json_sha256": file_sha256(training_path),
        "catalog_sha256": file_sha256(args.catalog_csv),
        "query_split": "test",
        "query_count": len(queries),
        "gallery_size": len(gallery),
        "ks": ks,
        "rrf_k": args.rrf_k,
        "clip_image_weight": args.clip_image_weight,
        "clip_text_image_weight": args.clip_text_image_weight,
        "batch_size": args.batch_size,
        "device": args.device,
        "resnet_source": args.resnet_source,
        "resnet_checkpoint": str(args.resnet_checkpoint.resolve()),
        "sbert_model": args.sbert_model,
        "clip_base_model": str(clip_base_model),
        "clip_checkpoint": str(clip_checkpoint.resolve()),
        "category_in_text": False,
        "shared_context": {
            "query_rows": "held-out test split",
            "gallery_rows": "the same complete 5,000-product catalog",
            "ranking": "unrestricted before relevance labels are applied",
        },
        "category_retrieval": {
            "self_match": "all gallery rows with the query product_id excluded",
            "relevance": "same ground-truth category after ranking",
        },
        "paired_cross_modal_retrieval": {
            "self_match": "paired product retained as the exact target",
            "relevance": "same product_id across image and text modalities",
        },
    }
    if args.dry_run:
        print(json.dumps(settings, indent=2, ensure_ascii=False))
        print("DRY RUN: 473-query x 5,000-gallery configuration is valid; models were not loaded.")
        return

    use_cache = not args.no_cache
    resnet_identity = (
        file_sha256(args.resnet_checkpoint)
        if args.resnet_source == "project-trained" and args.resnet_checkpoint.is_file()
        else "torchvision.ResNet18_Weights.DEFAULT"
    )
    signatures = {
        "resnet": _cache_signature(gallery, {
            "architecture": "ResNet18", "source": args.resnet_source, "identity": resnet_identity,
        }),
        "sbert": _cache_signature(gallery, {
            "architecture": "Sentence-BERT", "source": args.sbert_model, "category_in_text": False,
        }),
        "clip_pretrained_text": _cache_signature(gallery, {
            "architecture": "CLIP text encoder", "identity": base_identity, "category_in_text": False,
        }),
        "clip_finetuned_text": _cache_signature(gallery, {
            "architecture": "CLIP text encoder", "identity": clip_checkpoint_identity, "category_in_text": False,
        }),
        "clip_pretrained_image": _cache_signature(gallery, {
            "architecture": "CLIP vision encoder", "identity": base_identity,
        }),
        "clip_finetuned_image": _cache_signature(gallery, {
            "architecture": "CLIP vision encoder", "identity": clip_checkpoint_identity,
        }),
    }

    print(f"Encoding {len(gallery)} gallery images with ResNet18...", flush=True)
    resnet_images, resnet_metadata = _cached_embeddings(
        args.cache_dir, "resnet18_gallery", signatures["resnet"],
        lambda: encode_resnet18(
            gallery.image_path.tolist(), args.device, args.batch_size,
            args.resnet_source, args.resnet_checkpoint,
        ), use_cache,
    )
    print(f"Encoding {len(gallery)} category-free descriptions with SBERT...", flush=True)
    sbert_text, sbert_metadata = _cached_embeddings(
        args.cache_dir, "sbert_gallery", signatures["sbert"],
        lambda: encode_sbert(
            gallery.evaluation_text.tolist(), args.device, args.batch_size, args.sbert_model
        ), use_cache,
    )
    print(f"Encoding {len(gallery)} category-free descriptions with pretrained CLIP...", flush=True)
    clip_pretrained_text, clip_pretrained_text_metadata = _cached_embeddings(
        args.cache_dir, "clip_pretrained_text_gallery", signatures["clip_pretrained_text"],
        lambda: encode_clip_text(
            gallery.evaluation_text.tolist(), clip_base_model, args.device, args.batch_size
        ), use_cache,
    )
    print(f"Encoding {len(gallery)} category-free descriptions with fine-tuned CLIP...", flush=True)
    clip_finetuned_text, clip_finetuned_text_metadata = _cached_embeddings(
        args.cache_dir, "clip_finetuned_text_gallery", signatures["clip_finetuned_text"],
        lambda: encode_clip_text(
            gallery.evaluation_text.tolist(), clip_checkpoint, args.device, args.batch_size
        ), use_cache,
    )
    print(f"Encoding {len(gallery)} images with pretrained CLIP...", flush=True)
    clip_pretrained_images, clip_pretrained_image_metadata = _cached_embeddings(
        args.cache_dir, "clip_pretrained_image_gallery", signatures["clip_pretrained_image"],
        lambda: encode_clip_images(
            gallery.image_path.tolist(), clip_base_model, args.device, args.batch_size
        ), use_cache,
    )

    if args.finetuned_image_embeddings and args.finetuned_image_embeddings.is_file():
        print(f"Using verified fine-tuned CLIP images: {args.finetuned_image_embeddings}", flush=True)
        clip_finetuned_images, clip_finetuned_image_metadata = _load_verified_finetuned_images(
            args.finetuned_image_embeddings,
            clip_checkpoint_identity,
            args.catalog_csv,
            len(gallery),
        )
    else:
        print(f"Encoding {len(gallery)} images with fine-tuned CLIP...", flush=True)
        clip_finetuned_images, clip_finetuned_image_metadata = _cached_embeddings(
            args.cache_dir, "clip_finetuned_image_gallery", signatures["clip_finetuned_image"],
            lambda: encode_clip_images(
                gallery.image_path.tolist(), clip_checkpoint, args.device, args.batch_size
            ), use_cache,
        )

    query_ids = queries.product_id.astype(str).to_numpy()
    query_categories = queries.category.fillna("").astype(str).to_numpy()
    gallery_ids = gallery.product_id.astype(str).to_numpy()
    gallery_categories = gallery.category.fillna("").astype(str).to_numpy()

    def query_rows(matrix):
        return matrix[query_positions]

    controlled_rankings = {
        "sbert_text_only": rank_queries_by_cosine(
            query_rows(sbert_text), sbert_text, query_ids, gallery_ids
        ),
        "clip_pretrained_text_only": rank_queries_by_cosine(
            query_rows(clip_pretrained_text), clip_pretrained_text, query_ids, gallery_ids
        ),
        "clip_finetuned_text_only": rank_queries_by_cosine(
            query_rows(clip_finetuned_text), clip_finetuned_text, query_ids, gallery_ids
        ),
        "resnet18_image_only": rank_queries_by_cosine(
            query_rows(resnet_images), resnet_images, query_ids, gallery_ids
        ),
        "clip_pretrained_image_only": rank_queries_by_cosine(
            query_rows(clip_pretrained_images), clip_pretrained_images, query_ids, gallery_ids
        ),
        "clip_finetuned_image_only": rank_queries_by_cosine(
            query_rows(clip_finetuned_images), clip_finetuned_images, query_ids, gallery_ids
        ),
    }
    controlled_rankings["sbert_resnet18_rrf"] = reciprocal_rank_fusion(
        [controlled_rankings["sbert_text_only"], controlled_rankings["resnet18_image_only"]],
        len(gallery), args.rrf_k,
    )

    serving_rankings = {
        "clip_finetuned_image_serving": controlled_rankings["clip_finetuned_image_only"],
        "clip_finetuned_text_to_image_component": rank_queries_by_cosine(
            query_rows(clip_finetuned_text), clip_finetuned_images, query_ids, gallery_ids
        ),
        "clip_finetuned_text_to_text_component": controlled_rankings["clip_finetuned_text_only"],
    }
    serving_rankings["clip_finetuned_text_serving_raw"] = (
        clip_text_query_gallery_score_fusion(
            query_rows(clip_finetuned_text),
            clip_finetuned_images,
            clip_finetuned_text,
            query_ids,
            gallery_ids,
            args.clip_text_image_weight,
        )
    )
    serving_rankings["clip_finetuned_multimodal_raw"] = clip_query_gallery_score_fusion(
        query_rows(clip_finetuned_images), query_rows(clip_finetuned_text),
        clip_finetuned_images, clip_finetuned_text, query_ids, gallery_ids,
        args.clip_image_weight, args.clip_text_image_weight,
    )

    alignment_rankings = {
        "clip_pretrained_text_to_image": rank_queries_by_cosine(
            query_rows(clip_pretrained_text),
            clip_pretrained_images,
            query_ids,
            gallery_ids,
            exclude_same_product=False,
        ),
        "clip_finetuned_text_to_image": rank_queries_by_cosine(
            query_rows(clip_finetuned_text),
            clip_finetuned_images,
            query_ids,
            gallery_ids,
            exclude_same_product=False,
        ),
        "clip_pretrained_image_to_text": rank_queries_by_cosine(
            query_rows(clip_pretrained_images),
            clip_pretrained_text,
            query_ids,
            gallery_ids,
            exclude_same_product=False,
        ),
        "clip_finetuned_image_to_text": rank_queries_by_cosine(
            query_rows(clip_finetuned_images),
            clip_finetuned_text,
            query_ids,
            gallery_ids,
            exclude_same_product=False,
        ),
    }

    controlled_specs = {
        "sbert_text_only": {
            "comparison_group": "controlled_baseline", "role": "text baseline",
            "deployed": False, "input": "text", "formula": "S_TT (SBERT)",
        },
        "clip_pretrained_text_only": {
            "comparison_group": "controlled_baseline", "role": "text baseline",
            "deployed": False, "input": "text", "formula": "S_TT (pretrained CLIP)",
        },
        "clip_finetuned_text_only": {
            "comparison_group": "controlled_baseline", "role": "text baseline",
            "deployed": False, "input": "text", "formula": "S_TT (fine-tuned CLIP)",
        },
        "resnet18_image_only": {
            "comparison_group": "controlled_baseline", "role": "image baseline",
            "deployed": False, "input": "image", "formula": "S_II (ResNet18)",
        },
        "clip_pretrained_image_only": {
            "comparison_group": "controlled_baseline", "role": "image baseline",
            "deployed": False, "input": "image", "formula": "S_II (pretrained CLIP)",
        },
        "clip_finetuned_image_only": {
            "comparison_group": "controlled_baseline", "role": "image baseline",
            "deployed": False, "input": "image", "formula": "S_II (fine-tuned CLIP)",
        },
        "sbert_resnet18_rrf": {
            "comparison_group": "controlled_baseline", "role": "independent-model fusion",
            "deployed": False, "input": "image + text", "formula": "RRF(S_TT, S_II)",
        },
    }
    serving_specs = {
        "clip_finetuned_image_serving": {
            "comparison_group": "serving_ablation", "role": "image serving",
            "deployed": True, "input": "image", "formula": "S_II",
        },
        "clip_finetuned_text_to_image_component": {
            "comparison_group": "serving_ablation", "role": "text component",
            "deployed": False, "input": "text", "formula": "S_TI",
        },
        "clip_finetuned_text_to_text_component": {
            "comparison_group": "serving_ablation", "role": "text component",
            "deployed": False, "input": "text", "formula": "S_TT",
        },
        "clip_finetuned_text_serving_raw": {
            "comparison_group": "serving_ablation", "role": "text serving",
            "deployed": True, "input": "text",
            "formula": "w_t*S_TI + (1-w_t)*S_TT",
        },
        "clip_finetuned_multimodal_raw": {
            "comparison_group": "serving_ablation", "role": "multimodal serving",
            "deployed": True, "input": "image + text",
            "formula": "w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)",
        },
    }
    alignment_specs = {
        "clip_pretrained_text_to_image": {
            "comparison_group": "clip_finetuning", "role": "pretrained CLIP",
            "deployed": False, "input": "text -> image", "formula": "S_TI",
        },
        "clip_finetuned_text_to_image": {
            "comparison_group": "clip_finetuning", "role": "fine-tuned CLIP",
            "deployed": False, "input": "text -> image", "formula": "S_TI",
        },
        "clip_pretrained_image_to_text": {
            "comparison_group": "clip_finetuning", "role": "pretrained CLIP",
            "deployed": False, "input": "image -> text", "formula": "S_IT",
        },
        "clip_finetuned_image_to_text": {
            "comparison_group": "clip_finetuning", "role": "fine-tuned CLIP",
            "deployed": False, "input": "image -> text", "formula": "S_IT",
        },
    }

    controlled_metrics = {
        name: evaluate_query_gallery_category_rankings(
            values, query_ids, query_categories, gallery_ids, gallery_categories, ks
        )
        for name, values in controlled_rankings.items()
    }
    serving_metrics = {
        name: evaluate_query_gallery_category_rankings(
            values, query_ids, query_categories, gallery_ids, gallery_categories, ks
        )
        for name, values in serving_rankings.items()
    }
    alignment_metrics = {
        name: evaluate_query_gallery_identity_rankings(
            values, query_ids, query_categories, gallery_ids, ks
        )
        for name, values in alignment_rankings.items()
    }
    controlled_rows = _annotated_rows(
        controlled_specs, controlled_metrics, ks, "same-category product retrieval"
    )
    serving_rows = _annotated_rows(
        serving_specs, serving_metrics, ks, "same-category product retrieval"
    )
    alignment_rows = _annotated_rows(
        alignment_specs, alignment_metrics, ks, "exact paired-product retrieval"
    )
    controlled_category_rows = _annotated_category_rows(
        controlled_specs, controlled_metrics, ks, "same-category product retrieval"
    )
    serving_category_rows = _annotated_category_rows(
        serving_specs, serving_metrics, ks, "same-category product retrieval"
    )
    alignment_category_rows = _annotated_category_rows(
        alignment_specs, alignment_metrics, ks, "exact paired-product retrieval"
    )

    output = _prepare_output(args.output_dir)
    query_columns = [
        "product_id", "image_reference", "product_name", "description",
        "category", "image_path", "image_sha256", "evaluation_text",
    ]
    gallery_columns = query_columns
    queries[query_columns].to_csv(output / "evaluation_queries.csv", index=False)
    gallery[gallery_columns].to_csv(output / "evaluation_gallery.csv", index=False)
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
            "resnet18": resnet_metadata,
            "clip_pretrained_text": clip_pretrained_text_metadata,
            "clip_finetuned_text": clip_finetuned_text_metadata,
            "clip_pretrained_image": clip_pretrained_image_metadata,
            "clip_finetuned_image": clip_finetuned_image_metadata,
        },
        "metrics": {
            "controlled_baselines": controlled_metrics,
            "clip_finetuning": alignment_metrics,
            "serving_ablation": serving_metrics,
        },
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "experiment.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    grouped_outputs = {
        "controlled_baselines": (controlled_rows, controlled_category_rows),
        "clip_finetuning": (alignment_rows, alignment_category_rows),
        "serving_ablation": (serving_rows, serving_category_rows),
    }
    for folder_name, (group_rows, group_category_rows) in grouped_outputs.items():
        folder = output / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(group_rows).to_csv(folder / "comparison.csv", index=False)
        pd.DataFrame(group_category_rows).to_csv(
            folder / "comparison_by_category.csv", index=False
        )

    all_rows = controlled_rows + alignment_rows + serving_rows
    all_category_rows = (
        controlled_category_rows + alignment_category_rows + serving_category_rows
    )
    pd.DataFrame(all_rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(all_category_rows).to_csv(
        output / "comparison_by_category.csv", index=False
    )
    (output / "summary.md").write_text(
        _summary(
            controlled_rows,
            alignment_rows,
            serving_rows,
            len(queries),
            len(gallery),
        ),
        encoding="utf-8",
    )
    for group_name, group_rows in (
        ("CONTROLLED BASELINES", controlled_rows),
        ("CLIP PRETRAINED VS FINE-TUNED", alignment_rows),
        ("WEBSITE SERVING ABLATION", serving_rows),
    ):
        print(f"\n=== {group_name} ===")
        print(
            pd.DataFrame(group_rows).to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )
    print(f"Saved unified 473-query x 5,000-gallery comparisons to: {output}")


if __name__ == "__main__":
    main()
